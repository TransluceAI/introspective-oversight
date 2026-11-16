"""Generate the stage-1 SFT data.

Four tasks, run in this order for the paper mix:

1. ``task=behavior_prediction``: sample 5 assistant responses per prompt, then have a generator
   model write query-answer pairs predicting properties of the response. Each pair becomes
   one ``user, query, observer`` SFT item.
2. ``task=trait_catalog``: have the generator write a catalog of assistant traits, in batches.
3. ``task=hidden_trait``: for each trait, pick user prompts and sample 5 responses with the
   trait as the system prompt. The generator writes one oversight question and answer per
   response. Each becomes a ``user, assistant, query, observer`` item; the system prompt is
   dropped, so the observer must infer the trait from the response.
4. ``task=mix``: all hidden-trait items plus an equal number of behavior-prediction items whose
   prompts don't appear in the hidden-trait data.

User prompts come from ``scripts/prepare_prompts.py source=dolci``. Paper settings (assistant
Qwen/Qwen3.6-27B on Tinker, generator claude-haiku-4-5)::

    python -m introspect.data.train.generate_sft task=behavior_prediction \\
        prompts_path=data/prompts/dolci_50k.jsonl output_dir=data/sft/behavior_prediction
    python -m introspect.data.train.generate_sft task=trait_catalog output_dir=data/sft/traits
    python -m introspect.data.train.generate_sft task=hidden_trait \\
        prompts_path=data/prompts/dolci_50k.jsonl \\
        trait_catalog_path=data/sft/traits/traits.json output_dir=data/sft/hidden_trait
    python -m introspect.data.train.generate_sft task=mix \\
        mix_paths=data/sft/behavior_prediction/sft.json,data/sft/hidden_trait/sft.json \\
        output_dir=data/sft/mix

Each task writes ``records.json`` (everything generated) and, except the catalog,
``sft.json`` (``{"items": [{"messages": [...]}]}``, read by train_sft.py). Generation runs are
resumable: finished prompt groups are appended to ``checkpoint.jsonl`` and skipped on restart.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import sys
import textwrap
from collections.abc import Awaitable, Callable, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import chz
import httpx
import msgspec
import tinker
from transformers import AutoTokenizer

from introspect.data.samplers import Backend, ReasoningEffort, TextSampler

logger = logging.getLogger(__name__)

Task = Literal["behavior_prediction", "trait_catalog", "hidden_trait", "mix"]

# Paper values that differ between the two generation runs.
_ASSISTANT_MAX_TOKENS = {"behavior_prediction": 800, "hidden_trait": 2000}
_GENERATOR_MAX_TOKENS = {"behavior_prediction": 2500, "trait_catalog": 4096, "hidden_trait": 4096}


@chz.chz(typecheck=True)
class GenerateSFTConfig:
    task: Task
    output_dir: str

    # User prompts: .json {"items": [{"prompt": ...}]} or .jsonl, one {"prompt": ...} per line.
    prompts_path: str | None = None
    num_prompts: int = 0  # 0 uses every prompt; otherwise a seeded random sample
    seed: int = 42

    # Assistant whose behavior is described, sampled on Tinker.
    assistant_model: str = "Qwen/Qwen3.6-27B"
    assistant_thinking: bool = False
    assistant_temperature: float = 1.0
    assistant_max_tokens: int | None = None  # None: 800 (behavior prediction), 2000 (hidden trait)
    num_rollouts: int = 5

    # Generator that writes the questions and answers.
    generator_backend: Backend | Literal["tinker"] = "anthropic"
    generator_model: str = "claude-haiku-4-5"
    generator_temperature: float = 1.0
    generator_max_tokens: int | None = None  # None: 2500 (behavior prediction), else 4096
    generator_reasoning: ReasoningEffort | None = None

    # Behavior prediction
    num_queries_per_prompt: int = 5

    # Hidden trait
    num_traits: int = 500
    trait_batch_size: int = 50
    trait_oversample: int = 5  # extra traits requested per batch; the first trait_batch_size kept
    trait_catalog_path: str | None = None
    prompts_per_trait: int = 10

    # Mix: [behavior-prediction sft.json, hidden-trait sft.json]
    mix_paths: list[str] = chz.field(default_factory=list)
    mix_seed: int = 1337

    max_in_flight: int = 128
    checkpoint_every: int = 500  # prompt groups per checkpoint write

    @chz.validate
    def _check(self) -> None:
        needs_prompts = self.task in ("behavior_prediction", "hidden_trait")
        if needs_prompts and self.prompts_path is None:
            raise ValueError(f"task={self.task} needs prompts_path")
        if self.task == "hidden_trait" and self.trait_catalog_path is None:
            raise ValueError("task=hidden_trait needs trait_catalog_path (from task=trait_catalog)")
        if self.task == "hidden_trait" and self.num_rollouts != 5:
            raise ValueError("the hidden-trait generator prompt expects exactly 5 rollouts")
        if self.task == "trait_catalog" and self.num_traits % self.trait_batch_size:
            raise ValueError("num_traits must be a multiple of trait_batch_size")
        if self.task == "mix" and len(self.mix_paths) != 2:
            raise ValueError("task=mix needs mix_paths=<behavior sft.json>,<hidden-trait sft.json>")


# --- Records ---------------------------------------------------------------------------------


class QAPair(msgspec.Struct):
    question: str
    answer: str


class BehaviorPredictionRecord(msgspec.Struct):
    """One prompt, its assistant responses, and the generated query-answer pairs."""

    prompt: str
    assistant_responses: list[str]
    generator_response: str
    qa_pairs: list[QAPair]

    def sft_items(self) -> list[dict[str, Any]]:
        return [
            _sft_item(("user", self.prompt), ("query", pair.question), ("observer", pair.answer))
            for pair in self.qa_pairs
        ]


class HiddenTraitRecord(msgspec.Struct):
    """One response sampled under a trait system prompt, and its oversight question."""

    trait: str
    user_prompt: str
    assistant_response: str
    observer_question: str
    observer_answer: str

    def sft_item(self) -> dict[str, Any]:
        return _sft_item(
            ("user", self.user_prompt),
            ("assistant", self.assistant_response),
            ("query", self.observer_question),
            ("observer", self.observer_answer),
        )


def _sft_item(*turns: tuple[str, str]) -> dict[str, Any]:
    return {"messages": [{"role": role, "content": content} for role, content in turns]}


# --- Samplers --------------------------------------------------------------------------------


class TinkerChatSampler:
    """Sample chat completions from a Tinker model with its Hugging Face chat template."""

    def __init__(
        self,
        model: str,
        *,
        num_samples: int,
        temperature: float,
        max_tokens: int,
        seed: int | None,
        thinking: bool,
    ) -> None:
        self._tokenizer: Any = AutoTokenizer.from_pretrained(model)
        self._client = tinker.ServiceClient().create_sampling_client(base_model=model)
        self._num_samples = num_samples
        self._thinking = thinking
        self._params = tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            # Batched samples must be unseeded, or all rollouts come out identical.
            seed=seed if num_samples == 1 else None,
        )

    def prompt_tokens(self, messages: Sequence[dict[str, str]]) -> list[int]:
        encoded = self._tokenizer.apply_chat_template(
            list(messages), add_generation_prompt=True, enable_thinking=self._thinking
        )
        if not isinstance(encoded, list):  # BatchEncoding in newer transformers
            encoded = encoded["input_ids"]
        return list(encoded)

    async def __call__(self, messages: Sequence[dict[str, str]]) -> list[str]:
        response = await self._client.sample_async(
            prompt=tinker.ModelInput.from_ints(self.prompt_tokens(messages)),
            num_samples=self._num_samples,
            sampling_params=self._params,
        )
        return [
            self._tokenizer.decode(sequence.tokens, skip_special_tokens=True)
            for sequence in response.sequences
        ]


Generate = Callable[[str, str | None], Awaitable[str]]  # (prompt, system) -> text


def make_generator(cfg: GenerateSFTConfig, task: str, http: httpx.AsyncClient) -> Generate:
    max_tokens = cfg.generator_max_tokens or _GENERATOR_MAX_TOKENS[task]
    if cfg.generator_backend != "tinker":
        return TextSampler(
            http,
            cfg.generator_backend,
            cfg.generator_model,
            temperature=cfg.generator_temperature,
            max_tokens=max_tokens,
            reasoning=cfg.generator_reasoning,
            max_in_flight=cfg.max_in_flight,
        )
    sampler = TinkerChatSampler(
        cfg.generator_model,
        num_samples=1,
        temperature=cfg.generator_temperature,
        max_tokens=max_tokens,
        seed=None,  # unseeded, so a retry after a malformed response can differ
        thinking=False,
    )

    async def generate(prompt: str, system: str | None = None) -> str:
        messages = [{"role": "system", "content": system}] if system is not None else []
        (text,) = await sampler([*messages, {"role": "user", "content": prompt}])
        if not text.strip():
            raise ValueError("Tinker generator returned no text")
        return text.strip()

    return generate


def make_assistant(cfg: GenerateSFTConfig, task: str) -> TinkerChatSampler:
    return TinkerChatSampler(
        cfg.assistant_model,
        num_samples=cfg.num_rollouts,
        temperature=cfg.assistant_temperature,
        max_tokens=cfg.assistant_max_tokens or _ASSISTANT_MAX_TOKENS[task],
        seed=cfg.seed,
        thinking=cfg.assistant_thinking,
    )


# --- Inputs and outputs ----------------------------------------------------------------------


def load_prompts(path: str, *, num_prompts: int, seed: int) -> list[str]:
    raw = Path(path).read_bytes()
    if path.endswith(".jsonl"):
        rows = [msgspec.json.decode(line) for line in raw.splitlines() if line.strip()]
    else:
        rows = msgspec.json.decode(raw)["items"]
    loaded = [row["prompt"] for row in rows]
    if num_prompts > 0:
        loaded = random.Random(seed).sample(loaded, num_prompts)
    return loaded


def load_checkpoint(path: Path, record_type: type) -> dict[int, Any]:
    """Finished groups by index. A line cut off by a crash is ignored and redone."""
    if not path.exists():
        return {}
    done: dict[int, Any] = {}
    for line in path.read_bytes().splitlines():
        try:
            index, record = msgspec.json.decode(line, type=tuple[int, record_type])
        except msgspec.DecodeError:
            logger.warning("Ignoring an unreadable checkpoint line in %s", path)
            continue
        done[index] = record
    return done


async def run_groups(
    units: Sequence[Any],
    process: Callable[[Any], Awaitable[Any]],
    *,
    checkpoint: Path,
    record_type: type,
    cfg: GenerateSFTConfig,
) -> list[Any]:
    """Process every unit once, checkpointing results; failed units are logged and skipped."""
    done = load_checkpoint(checkpoint, record_type)
    if done:
        logger.info("Resuming: %d/%d groups already done", len(done), len(units))
    pending = [index for index in range(len(units)) if index not in done]
    semaphore = asyncio.Semaphore(cfg.max_in_flight)

    async def bounded(index: int) -> Any:
        async with semaphore:
            return await process(units[index])

    for start in range(0, len(pending), cfg.checkpoint_every):
        batch = pending[start : start + cfg.checkpoint_every]
        results = await asyncio.gather(*(bounded(i) for i in batch), return_exceptions=True)
        lines = []
        for index, result in zip(batch, results, strict=True):
            if isinstance(result, BaseException):
                if not isinstance(result, Exception):
                    raise result
                logger.warning("Skipping group %d: %r", index, result)
                continue
            done[index] = result
            lines.append(msgspec.json.encode((index, result)) + b"\n")
        with checkpoint.open("ab") as file:
            file.writelines(lines)
        logger.info("Checkpointed %d/%d groups", len(done), len(units))
    return [done[index] for index in sorted(done)]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(msgspec.json.encode(value))
    logger.info("Wrote %s", path)


def config_metadata(cfg: GenerateSFTConfig) -> dict[str, Any]:
    return {**chz.asdict(cfg), "command_line": " ".join(sys.argv)}


# --- Tasks -----------------------------------------------------------------------------------


async def behavior_prediction(cfg: GenerateSFTConfig, http: httpx.AsyncClient) -> None:
    assert cfg.prompts_path is not None
    user_prompts = load_prompts(cfg.prompts_path, num_prompts=cfg.num_prompts, seed=cfg.seed)
    assistant = make_assistant(cfg, "behavior_prediction")
    generate = make_generator(cfg, "behavior_prediction", http)
    template = behavior_prediction_template(cfg.num_queries_per_prompt)

    async def process(prompt: str) -> BehaviorPredictionRecord:
        responses = await assistant([{"role": "user", "content": prompt}])
        generator_response = await generate(
            fill_behavior_prediction_prompt(template, prompt, responses), None
        )
        pairs = parse_qa_pairs(generator_response)
        return BehaviorPredictionRecord(
            prompt=prompt,
            assistant_responses=responses,
            generator_response=generator_response,
            qa_pairs=[QAPair(question, answer) for question, answer in pairs],
        )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[BehaviorPredictionRecord] = await run_groups(
        user_prompts,
        process,
        checkpoint=output_dir / "checkpoint.jsonl",
        record_type=BehaviorPredictionRecord,
        cfg=cfg,
    )
    items = [item for record in records for item in record.sft_items()]
    metadata = {
        **config_metadata(cfg),
        "generator_prompt_template": template,
        "num_records": len(records),
        "num_sft_items": len(items),
    }
    write_json(output_dir / "records.json", {"metadata": metadata, "records": records})
    write_json(output_dir / "sft.json", {"metadata": metadata, "items": items})


async def trait_catalog(cfg: GenerateSFTConfig, http: httpx.AsyncClient) -> None:
    """Generate traits in batches, showing the model every earlier trait to avoid repeats."""
    generate = make_generator(cfg, "trait_catalog", http)
    num_batches = cfg.num_traits // cfg.trait_batch_size
    traits: list[str] = []
    batches: list[dict[str, Any]] = []
    for batch_index in range(num_batches):
        request = trait_catalog_prompt(
            batch_size=cfg.trait_batch_size,
            oversample=cfg.trait_oversample,
            batch_index=batch_index,
            num_batches=num_batches,
            previous_traits=traits,
        )
        for attempt in range(3):
            response = await generate(request, INTERVENTION_GENERATOR_SYSTEM_PROMPT)
            seen = {trait.lower() for trait in traits}
            batch = [t for t in parse_traits(response) or [] if t.lower() not in seen]
            if len(batch) >= cfg.trait_batch_size:
                batch = batch[: cfg.trait_batch_size]
                break
            logger.warning(
                "Trait batch %d was malformed (attempt %d/3): %.300r",
                batch_index + 1, attempt + 1, response,
            )
        else:
            raise RuntimeError(f"Trait batch {batch_index + 1} failed three times")
        traits.extend(batch)
        batches.append({"prompt": request, "response": response, "traits": batch})
        logger.info("Trait batch %d/%d done", batch_index + 1, num_batches)
    write_json(
        Path(cfg.output_dir) / "traits.json",
        {"metadata": config_metadata(cfg), "traits": traits, "batches": batches},
    )


def load_traits(path: str) -> list[str]:
    """Read our catalog ({"traits": [...]}) or the original's ({"interventions": [...]})."""
    data = json.loads(Path(path).read_text())
    if "traits" in data:
        return data["traits"]
    return [row["trait"] for row in data["interventions"] if row["intervention_type"] == "trait"]


async def hidden_trait(cfg: GenerateSFTConfig, http: httpx.AsyncClient) -> None:
    assert cfg.prompts_path is not None and cfg.trait_catalog_path is not None
    user_prompts = load_prompts(cfg.prompts_path, num_prompts=cfg.num_prompts, seed=cfg.seed)
    random.Random(cfg.seed).shuffle(user_prompts)
    traits = load_traits(cfg.trait_catalog_path)
    # Each trait gets the next prompts_per_trait prompts, wrapping around if needed.
    groups = [
        (trait, user_prompts[(t * cfg.prompts_per_trait + p) % len(user_prompts)])
        for t, trait in enumerate(traits)
        for p in range(cfg.prompts_per_trait)
    ]
    assistant = make_assistant(cfg, "hidden_trait")
    generate = make_generator(cfg, "hidden_trait", http)

    async def process(group: tuple[str, str]) -> list[HiddenTraitRecord]:
        trait, user_prompt = group
        responses = await assistant(
            [
                {"role": "system", "content": trait_system_prompt(trait)},
                {"role": "user", "content": user_prompt},
            ]
        )
        if len(responses) != 5 or any(not response.strip() for response in responses):
            raise ValueError("expected five non-empty assistant responses")
        request = fill_observer_qa_prompt(
            trait=trait, user_prompt=user_prompt, responses=responses
        )
        current = request
        for attempt in range(3):
            qas = parse_observer_qas(
                await generate(current, OBSERVER_QA_GENERATOR_SYSTEM_PROMPT)
            )
            if qas is not None:
                break
            logger.warning("Observer QA block was malformed (attempt %d/3)", attempt + 1)
            current = request + OBSERVER_QA_FORMAT_CORRECTION
        else:
            logger.error("Discarding a group after three malformed observer QA blocks")
            return []
        return [
            HiddenTraitRecord(trait, user_prompt, response, question, answer)
            for response, (question, answer) in zip(responses, qas, strict=True)
        ]

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    finished: list[list[HiddenTraitRecord]] = await run_groups(
        groups,
        process,
        checkpoint=output_dir / "checkpoint.jsonl",
        record_type=list[HiddenTraitRecord],
        cfg=cfg,
    )
    records = [record for group in finished for record in group]
    items = [record.sft_item() for record in records]
    metadata = {
        **config_metadata(cfg),
        "num_groups": len(groups),
        "num_discarded_groups": len(groups) - sum(1 for group in finished if group),
        "num_records": len(records),
        "num_sft_items": len(items),
    }
    write_json(output_dir / "records.json", {"metadata": metadata, "records": records})
    write_json(output_dir / "sft.json", {"metadata": metadata, "items": items})


def _mix_key(seed: int, namespace: str, value: str) -> bytes:
    return sha256(f"{seed}\0{namespace}\0{value}".encode()).digest()


def build_mix(
    behavior: dict[str, Any], hidden_trait: dict[str, Any], *, seed: int
) -> list[dict[str, Any]]:
    """All hidden-trait items, plus as many behavior-prediction items from disjoint prompts.

    Behavior-prediction prompts are taken whole (all their queries) in seeded-hash order, and
    the combined items are shuffled the same way.
    """

    def user_prompt(item: dict[str, Any]) -> str:
        (prompt,) = [m["content"] for m in item["messages"] if m["role"] == "user"]
        return prompt

    queries_per_prompt = behavior["metadata"]["num_queries_per_prompt"]
    trait_items = hidden_trait["items"]
    trait_prompts = {user_prompt(item) for item in trait_items}
    if len(trait_items) % queries_per_prompt:
        raise ValueError("hidden-trait item count must be divisible by num_queries_per_prompt")

    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for item in behavior["items"]:
        by_prompt.setdefault(user_prompt(item), []).append(item)
    eligible = [
        prompt
        for prompt, group in by_prompt.items()
        if prompt not in trait_prompts and len(group) == queries_per_prompt
    ]
    eligible.sort(key=lambda prompt: _mix_key(seed, "behavior-prompt", prompt))
    num_groups = len(trait_items) // queries_per_prompt
    if len(eligible) < num_groups:
        raise ValueError(f"need {num_groups} disjoint behavior prompts, found {len(eligible)}")

    def tagged(item: dict[str, Any], source: str) -> dict[str, Any]:
        return {**item, "metadata": {**item.get("metadata", {}), "balanced_mix_source": source}}

    items = [
        *(tagged(item, "behavior_prediction") for p in eligible[:num_groups] for item in by_prompt[p]),
        *(tagged(item, "oversight") for item in trait_items),
    ]
    items.sort(
        key=lambda item: _mix_key(
            seed, "combined-item-order", json.dumps(item, ensure_ascii=False, sort_keys=True)
        )
    )
    return items


def mix(cfg: GenerateSFTConfig) -> None:
    behavior, trait = (json.loads(Path(path).read_text()) for path in cfg.mix_paths)
    items = build_mix(behavior, trait, seed=cfg.mix_seed)
    metadata = {**config_metadata(cfg), "num_sft_items": len(items)}
    write_json(Path(cfg.output_dir) / "sft.json", {"metadata": metadata, "items": items})


async def run(cfg: GenerateSFTConfig) -> None:
    if cfg.task == "mix":
        mix(cfg)
        return
    async with httpx.AsyncClient(timeout=300) as http:
        if cfg.task == "behavior_prediction":
            await behavior_prediction(cfg, http)
        elif cfg.task == "trait_catalog":
            await trait_catalog(cfg, http)
        else:
            await hidden_trait(cfg, http)


def main(cfg: GenerateSFTConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(run(cfg))


# --- Prompts ---------------------------------------------------------------------------------
# Copied verbatim from the paper runs. The trait-catalog system prompt still describes all three
# intervention families from the original pipeline; the paper run only requested traits.


def safe_format(template: str, **kwargs: str) -> str:
    """Fill ``{key}`` placeholders in one pass, leaving other braces (code, LaTeX) untouched."""

    def replace(match: re.Match[str]) -> str:
        return str(kwargs[match.group(1)]) if match.group(1) in kwargs else match.group(0)

    return re.sub(r"\{([^}]+)\}", replace, template)


# Behavior prediction

GENERATOR_PROMPT_TEMPLATE = textwrap.dedent(
    """
    {task_description}

    {decision_format}

    {example}

    {data}
    """
)

TASK_DESCRIPTION = textwrap.dedent(
    """
    Below is a prompt and a list of sampled responses from the assistant. Your task is to generate {num_queries_per_prompt} query-answer pairs. For each pair, generate a query asking a second model (which sees only the prompt but not the assistant's responses) to predict a property of the assistant's response. Then provide an answer to the query, which should be consistent with all of the assistant's responses.
    """
)

QUALITY_GUIDE = textwrap.dedent(
    """\
    Important notes:
    - You should phrase all queries and answers as predictive tasks (and by default use future tense to describe the assistant's response).
    - Each answer should be consistentwith all or all but one of the sampled responses.
    - The query-answer pairs should NOT explicitly mention the number of sampled responses you saw. The queries will be asked to a second model which thinks only one response will be sampled from the assistat.
    - Make queries as open-ended as you can. For example, if the assistant's responses have a consistent prefix, instead of asking "Will the assistant's response begin with the words 'We are given'?", ask, "What three words will the assistant begin its response with?"
    - Avoid producing yes/no questions as much as you can.
    """
)

DECISION_FORMAT = textwrap.dedent(
    """
    You may reason before you produce your query-answer pairs. You should output all pairs in the following format at the end of your response:

    <query1>...</query1>
    <answer1>...</answer1>
    <query2>...</query2>
    <answer2>...</answer2>
    (and so on)
    <query{num_queries_per_prompt}>...</query{num_queries_per_prompt}>
    <answer{num_queries_per_prompt}>...</answer{num_queries_per_prompt}>
    """
)

# Kept flush-left (no dedent): the example content contains escaped \n sequences, so the
# parsed string mixes indented physical lines with unindented embedded lines and
# textwrap.dedent would find no common margin to strip.
_EXAMPLE_PROMPT_AND_RESPONSES = """
[EXAMPLE PROMPT START]
Magical Number\n\nA group of N people is seated in ascending order of their ages.  The seats are numbered starting from 0, so the person at position i (0 ≤ i ≤ N−1) has age A[i].\n\nA seat i is called a \"magical number\" if the person sitting there has age exactly equal to the seat number, i.e.  A[i] = i.\n\nFor every test case, output any one magical number if such a seat exists; otherwise output −1.\n\nInput\nThe first line contains an integer T (1 ≤ T ≤ 100) – the number of test cases.\nFor each test case:\n• The first line contains an integer N (1 ≤ N ≤ 1000) – the number of people.\n• The second line contains N integers A[0], A[1], …, A[N−1] (−1000 ≤ A[i] ≤ 1000) in non-decreasing order, representing the ages.\n\nOutput\nFor each test case print a single line containing one integer – any magical number, or −1 if no such seat exists.\n\nExample\nInput\n1\n10\n-10 -1 0 3 10 11 30 50 100 150\n\nOutput\n3
[EXAMPLE PROMPT END]

[EXAMPLE ASSISTANT RESPONSES START]
1. We are given a problem where:\n\n- There are `N` people seated in **ascending order of age** (i.e., the array `A` is non-decreasing).\n- Each person sits at a seat numbered from `0` to `N-1`.\n- A **\"magical number\"** is a seat `i` such that `A[i] == i`.\n- We need to **find any such seat `i`** where this condition holds, or return `-1` if no such seat exists.\n\n---\n\n### 🔍 Key Observations:\n\n- The array `A` is **sorted in non-decreasing order**.\n- We are looking for an index `i` such that `A[i] == i`.\n- Since the array is sorted, we can use **binary search** to find such an index efficiently (though since `N ≤ 1000`, even a linear scan is acceptable).\n\nBut let's use **binary search** for efficiency, especially since the array is sorted and the condition `A[i] == i` has a monotonic-like behavior.\n\n---\n\n### 📌 Insight:\n\nLet's define a function:\n- `f(i) = A[i] - i`\n- We want `f(i) = 0`\n\nNow, since `A[i]` is non-decreasing and `i` is increasing, `f(i)` is **non-decreasing** (because `A[i+1] ≥ A[i]` and `(i+1) > i`, so `A[i+1] - (i+1) ≥ A[i] - i` only if the increase in `A` is at least 1 — but not guaranteed).\n\nWait — actually, `f(i) = A[i] - i` is **non-increasing**?\n\nLet's check:\n\n- If `A[i+1] - A[i] ≥ 0`, but `(i+1) - i = 1`, so:\n  - `f(i+1) - f(i) = (A[i+1] - (i+1)) - (A[i] - i) = (A[i+1] - A[i]) - 1`\n\nSo:\n- If `A[i+1] - A[i] ≥ 1`, then `f(i+1) ≥ f(i)`\n- If `A[i+1] - A[i] < 1`, then `f(i+1) < f(i)`\n\nThus, `f(i)` is **not necessarily monotonic**, so we **cannot** do binary search directly on `f(i)`.\n\nBut we are only asked to **find any `i` such that `A[i] == i`**, and since `N ≤ 1000`, we can do a **linear scan**.\n\n---\n\n### ✅ Simple Approach (Linear Scan):\n\nFor each index `i` from `0` to `N-1`, check if `A[i] == i`.\n\nIf yes, return `i`.\n\nIf no such index exists, return `-1`.\n\n---\n\n### ✅ Example:\n\nInput:\n```\n1\n10\n-10 -1 0 3 10 11 30 50 100 150\n```\n\nCheck each index:\n\n| i | A[i] | A[i] == i? |\n|---|------|------------|\n| 0 | -10  | No         |\n| 1 | -1   | No         |\n| 2 | 0    | No         |\n| 3 | 3    | ✅ Yes!    |\n\nSo output is `3`.\n\n---\n\n### 🧠 Edge Cases:\n\n- All ages < indices → no magical number → return -1.\n- All ages > indices → no magical number → return -1.\n- One or more matches → return **any one**.\n\n---\n\n### ✅ Code (Python):\n\n```python\nT = int(input())\nfor _ in range(T):\n    N = int(input())\n    A = list(map(int, input().split()))\n    \n    magical = -1\n    for i in range(N):\n        if A[i] == i:\n            magical = i\n            break  # Return first magical number found\n    \n    print(magical)\n```\n\n---\n\n### ✅ Output for Example:\n```\n3\n```\n\nThis matches the expected output.\n\n---\n\n### ✅ Time Complexity: O(N) per test case  \n### ✅ Space Complexity: O(1) extra space\n\nSince `N ≤ 1000` and `T ≤ 100`, worst-case total operations: `100 * 1000 = 100,000`, which is efficient.\n\n✅ **Final Answer:** Use linear scan to find any index `i` such that `A[i] == i`.<|im_end|>
2. We are given a list of `N` people seated in **ascending order of their ages**, and the seats are numbered from `0` to `N-1`.\n\nWe are to find **any** seat `i` such that the person's age `A[i]` is **exactly equal to the seat number `i`**, i.e., `A[i] == i`. Such a seat is called a **\"magical number\"**.\n\n### Key Observations:\n- The array `A` is in **non-decreasing order** (since ages are in ascending order).\n- We are to find **any index `i`** where `A[i] == i`.\n- If no such index exists, return `-1`.\n\nSince the array is sorted, we **could** use binary search for optimization, but since `N ≤ 1000`, a simple linear scan is efficient enough.\n\n---\n\n### Algorithm:\nFor each test case:\n1. Read `N`.\n2. Read the list `A` of `N` integers.\n3. Iterate `i` from `0` to `N-1`:\n   - If `A[i] == i`, return `i` immediately (since we only need **any** magical number).\n4. If no such `i` found, return `-1`.\n\n---\n\n### Example Walkthrough:\nInput:\n```\n1\n10\n-10 -1 0 3 10 11 30 50 100 150\n```\n\nCheck each index `i`:\n- `i=0`: `A[0] = -10 ≠ 0`\n- `i=1`: `A[1] = -1 ≠ 1`\n- `i=2`: `A[2] = 0 ≠ 2`\n- `i=3`: `A[3] = 3 == 3` → Found! Return `3`\n\n✅ Output: `3`\n\n---\n\n### Implementation (Python):\n\n```python\nT = int(input())\nfor _ in range(T):\n    N = int(input())\n    A = list(map(int, input().split()))\n    \n    magical_number = -1\n    for i in range(N):\n        if A[i] == i:\n            magical_number = i\n            break\n    \n    print(magical_number)\n```\n\n---\n\n### Output for the given example:\n```\n3\n```\n\n✅ Correct!\n\nThis solution works efficiently and correctly for all valid inputs.<|im_end|>
3. We are given a list of ages `A` of `N` people seated in **ascending order of their ages**, and the seats are numbered from `0` to `N−1`. A **magical number** is a seat `i` such that `A[i] == i`.\n\nWe are to find **any one** such seat (i.e., any index `i` where `A[i] == i`), or return `-1` if no such seat exists.\n\n---\n\n### Key Observations:\n- The array `A` is in **non-decreasing** order (sorted).\n- We're looking for an index `i` such that `A[i] == i`.\n- Since the array is sorted, we **could** use binary search to speed things up, but given constraints (`N ≤ 1000`), a simple linear scan is perfectly fine.\n\n---\n\n### Algorithm:\nFor each test case:\n1. Read `N`.\n2. Read the array `A` of `N` integers.\n3. Loop through each index `i` from `0` to `N−1`:\n   - If `A[i] == i`, return `i` immediately (we only need **any one**).\n4. If no such index is found, return `-1`.\n\n---\n\n### Example Walkthrough:\nInput:\n```\n1\n10\n-10 -1 0 3 10 11 30 50 100 150\n```\n\nCheck each index:\n- `i = 0`: `A[0] = -10` → not equal\n- `i = 1`: `A[1] = -1` → not equal\n- `i = 2`: `A[2] = 0` → not equal\n- `i = 3`: `A[3] = 3` → **equal!** → return `3`\n\nSo output: `3`\n\n---\n\n### Code (Python):\n\n```python\nT = int(input())\nfor _ in range(T):\n    N = int(input())\n    A = list(map(int, input().split()))\n    \n    magical = -1\n    for i in range(N):\n        if A[i] == i:\n            magical = i\n            break\n    \n    print(magical)\n```\n\n---\n\n### Output for the example:\n```\n3\n```\n\n✅ Correct.\n\n---\n\n### Time Complexity:\n- O(N) per test case → efficient enough for given constraints.\n\nLet me know if you want a binary search version (which would be O(log N)).<|im_end|>
4. We are given a list of ages `A` of `N` people seated in **ascending order** (non-decreasing), and we need to find **any seat `i`** such that the **person's age equals the seat number**, i.e., `A[i] == i`.\n\n### Steps:\n1. Iterate through each seat `i` from `0` to `N-1`.\n2. Check if `A[i] == i`.\n3. If such a seat is found, return `i` immediately (we need **any one** magical number).\n4. If no such seat exists after checking all, return `-1`.\n\n---\n\n### Why does this work?\n- The array is in **non-decreasing order**, but the values can be negative.\n- Since `A[i]` increases (or stays same), but `i` increases by 1 each time, the difference `A[i] - i` is non-decreasing.\n- So once `A[i] > i`, it can never come back to `A[i] == i` later (since `A[i]` increases slower or same speed, but `i` increases steadily). However, we don't need to optimize further — just check all positions.\n\n---\n\n### Example:\nInput:\n```\n1\n10\n-10 -1 0 3 10 11 30 50 100 150\n```\n\nCheck each index `i`:\n\n| i | A[i] | A[i] == i? |\n|---|------|------------|\n| 0 | -10  | No         |\n| 1 | -1   | No         |\n| 2 | 0    | No         |\n| 3 | 3    | **Yes** ✅ |\n\nSo output: `3`\n\n---\n\n### Code (Python):\n\n```python\nT = int(input())\nfor _ in range(T):\n    N = int(input())\n    A = list(map(int, input().split()))\n    \n    magical_number = -1\n    for i in range(N):\n        if A[i] == i:\n            magical_number = i\n            break\n    \n    print(magical_number)\n```\n\n---\n\n### Sample Output:\n```\n3\n```\n\n✅ Matches expected.\n\nThis solution is efficient: **O(N)** per test case, which is acceptable for `N ≤ 1000` and `T ≤ 100`.<|im_end|>
5. We are given a list of **ages** in **non-decreasing order**, and we need to find any **\"magical number\"**, i.e., a seat `i` such that `A[i] == i`.\n\n### Key Observations:\n- The array `A` is sorted in **non-decreasing** order (so ages are sorted, possibly with duplicates).\n- We are to find an index `i` where `A[i] == i`.\n- Since the array is sorted, we can use **binary search** for efficiency, but due to constraints (N ≤ 1000), even a **linear scan** will work fine.\n\n### Approach:\nIterate through the array from `i = 0` to `i = N-1`, and check if `A[i] == i`. If yes, return `i` immediately (since it asks for **any one** magical number).\n\nIf no such index is found, return `-1`.\n\n---\n\n### Example Walkthrough:\nInput:\n```\n1\n10\n-10 -1 0 3 10 11 30 50 100 150\n```\n\nCheck each index:\n- i=0: A[0] = -10 → not equal to 0\n- i=1: A[1] = -1 → not equal to 1\n- i=2: A[2] = 0 → not equal to 2\n- i=3: A[3] = 3 → **equal to 3** → return 3\n\nHence, output is `3`.\n\n---\n\n### Code (Python):\n\n```python\nT = int(input())\nfor _ in range(T):\n    N = int(input())\n    A = list(map(int, input().split()))\n    \n    magical_number = -1\n    for i in range(N):\n        if A[i] == i:\n            magical_number = i\n            break\n    \n    print(magical_number)\n```\n\n---\n\n### Output:\n```\n3\n```\n\n✅ Correct and efficient for the given constraints.<|im_end|>
[EXAMPLE ASSISTANT RESPONSES END]
"""

_EXAMPLE_FINAL_ANSWER = textwrap.dedent(
    """
    [EXAMPLE FINAL ANSWER]
    <query1>Will the assistant write code in response to the prompt? If so, what language will it be written in?</query1>
    <answer1>Yes, the assistant will write Python code.</answer1>
    <query2>Will the assistant express uncertainty about its understanding of the request? If so, what exactly is this uncertainty?</query2>
    <answer2>No, the assistant will appear confident in its understanding of the request.</answer2>
    <query3>What three words will the assistant begin its response with?</query3>
    <answer3>We are given</answer3>
    <query4>What algorithmic approach will the assistant recommend for solving this problem?</query4>
    <answer4>The assistant will recommend a linear scan approach.</answer4>
    <query5>Given that the assistant will ultimately recommend a linear scan approach, will it also discuss any alternative approaches to solving this problem?</query5>
    <answer5>The assistant will discuss binary search as a possible alternative approach but dismiss it due to the small constraint size (N ≤ 1000)</answer5>
    """
)

DATA_BLOCK = textwrap.dedent(
    """
    Here is the data:

    [PROMPT START]
    {prompt}
    [PROMPT END]

    [ASSISTANT RESPONSES START]
    {assistant_responses}
    [ASSISTANT RESPONSES END]
    """
)


def behavior_prediction_template(num_queries_per_prompt: int) -> str:
    """Build the generator prompt, keeping the {prompt} and {assistant_responses} slots."""
    example = "\nHere is an example:\n" + _EXAMPLE_PROMPT_AND_RESPONSES + _EXAMPLE_FINAL_ANSWER
    return safe_format(
        GENERATOR_PROMPT_TEMPLATE,
        task_description=safe_format(
            TASK_DESCRIPTION + QUALITY_GUIDE, num_queries_per_prompt=str(num_queries_per_prompt)
        ),
        decision_format=safe_format(
            DECISION_FORMAT, num_queries_per_prompt=str(num_queries_per_prompt)
        ),
        example=example,
        data=DATA_BLOCK,
    )


def fill_behavior_prediction_prompt(template: str, prompt: str, responses: Sequence[str]) -> str:
    responses_block = "\n".join(f"{i + 1}. {response}" for i, response in enumerate(responses))
    return safe_format(template, prompt=prompt, assistant_responses=responses_block)


def parse_qa_pairs(generator_response: str) -> list[tuple[str, str]]:
    """Parse <queryN>/<answerN> pairs, matched by index; unmatched pairs are skipped."""
    answers: dict[int, str] = {}
    queries: dict[int, str] = {}
    for tag, dest in (("answer", answers), ("query", queries)):
        unnumbered = 0
        for match in re.finditer(rf"<{tag}(\d*)>(.*?)</{tag}\1>", generator_response, re.DOTALL):
            if match.group(1):
                index = int(match.group(1))
            else:
                index, unnumbered = unnumbered, unnumbered + 1
            dest[index] = match.group(2).strip()
    pairs = [(queries[i], answers[i]) for i in sorted(answers.keys() & queries.keys())]
    return [(question, answer) for question, answer in pairs if question and answer]


# Hidden trait

INTERVENTION_GENERATOR_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You design diverse intervention ideas for an oversight dataset. Each idea will
    later be rendered through a fixed assistant system-prompt template and paired
    with a separate user question.

    Generate exactly the requested number of interventions in each family:

    - trait: a persona, tone, register, format, language, or response policy. Include
      some specific, widely recognizable fictional-character personas, such as
      speaking like Jar Jar Binks, alongside generic personas and behavioral traits.
      Cover multiple categories: personality, values, quirks or habits, fears, social
      style, flaws, skills, physical mannerisms, beliefs, and contradictions. Include
      some traits involving intentional dishonesty, such as lying, fabrication,
      concealment, or deliberately misleading framing. Also cover diverse concepts
      such as hedging, bluntness, verbosity, jargon, flattery, evasiveness, excessive
      caution, and unusual formatting. Mix conspicuous traits with subtle tendencies
      that affect framing, emphasis, directness, or word choice without becoming an
      obvious gimmick. Store a concise description in `trait`. No two traits may be
      synonyms or near-duplicates; for example, `kind` and `compassionate` count as
      the same underlying trait. Do not generate only character impressions.
    - conditional_policy: an if-then rule with a concrete trigger in the user
      message and an observable response behavior. Store these separately in
      `trigger` and `policy`.
    - hidden_objective: a concealed goal that subtly steers, suppresses, favors, or
      avoids something. Store the goal in `objective`. Use fictional entities when
      naming products, organizations, or people.

    For every intervention, `intervention_type` must be exactly `trait`,
    `conditional_policy`, or `hidden_objective`. Generate only the idea fields. Do
    not generate system prompts or user questions.

    Vary the ideas and domains rather than producing near-duplicates. Do not generate
    false beliefs or sandbagging interventions in this version.

    Return only valid JSON with this shape, without Markdown fences:
    {
      "interventions": [
        {"intervention_type": "trait", "trait": "..."},
        {"intervention_type": "conditional_policy", "policy": "...",
         "trigger": "..."},
        {"intervention_type": "hidden_objective", "objective": "..."}
      ]
    }
    """
)

INTERVENTION_GENERATOR_USER_TEMPLATE = textwrap.dedent(
    """\
    Generate this intervention catalog:
    - traits: {num_traits}
    - conditional policies: {num_conditional_policies}
    - hidden objectives: {num_hidden_objectives}

    The JSON must contain exactly {total} interventions with exactly those family
    counts. Every family-specific idea must be non-empty and distinct within its
    family.
    """
)

def trait_catalog_prompt(*, batch_size: int, oversample: int, batch_index: int,
                         num_batches: int, previous_traits: Sequence[str]) -> str:
    """Request one batch of traits, listing earlier batches so the model avoids repeats.

    The JSON request asks for ``batch_size + oversample`` traits, since the model often
    returns a few fewer; the first ``batch_size`` are kept.
    """
    requested = str(batch_size + oversample)
    request = safe_format(
        INTERVENTION_GENERATOR_USER_TEMPLATE,
        num_traits=requested,
        num_conditional_policies="0",
        num_hidden_objectives="0",
        total=requested,
    )
    context = f"This is batch {batch_index + 1} of {num_batches}. " + (
        f"Generate {requested} candidates; exactly {batch_size} valid new traits will be retained."
        if oversample
        else f"Generate exactly {batch_size} new traits."
    )
    if previous_traits:
        context += (
            " None may repeat, paraphrase, or be a near-synonym of any trait from earlier"
            " batches listed below:\n" + "\n".join(f"- {trait}" for trait in previous_traits)
        )
    return f"{request}\n\n[BATCH CONTEXT]\n{context}\n[/BATCH CONTEXT]"


def parse_traits(response: str) -> list[str] | None:
    """Read the catalog JSON (optionally in a code fence), skipping entries without a trait.

    Returns None if the response isn't a catalog at all.
    """
    stripped = response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.IGNORECASE)
    try:
        rows = msgspec.json.decode(fenced.group(1) if fenced else stripped)["interventions"]
    except (msgspec.DecodeError, KeyError, TypeError):
        return None
    if not isinstance(rows, list):
        return None
    return [
        row["trait"].strip()
        for row in rows
        if isinstance(row, dict)
        and row.get("intervention_type") == "trait"
        and isinstance(row.get("trait"), str)
        and row["trait"].strip()
    ]


def trait_system_prompt(trait: str) -> str:
    return f"Exhibit this trait consistently in your response: {trait}"


OBSERVER_QA_GENERATOR_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You write five realistic short-answer oversight questions and desired observer
    answers about completed assistant interactions. You are given one user message
    and five independently sampled responses produced under the same conditions. The
    questions should sound like what a reviewer or automated monitor would ask when
    evaluating whether a response was safe, honest, and appropriate.

    Use all five responses to identify stable, oversight-relevant behavior rather
    than artifacts of a single sample. Then produce exactly five query-answer pairs,
    in response order: pair 1 will be attached to response 1, pair 2 to response 2,
    and so on. Each answer must be directly supported by its corresponding response.
    Prefer questions whose high-level answer would also remain consistent across at
    least four of the five responses.

    The eventual observer sees only the user message, one assistant response, and
    the query attached to that response. It does not see the other samples, the
    intervention, or the assistant system prompt. Write every pair as though only
    its corresponding response exists. Never write "response 1," "response 2," and
    so on; never refer to "responses," "samples," or "rollouts" in the plural;
    never compare them; and never rely on evidence absent from the corresponding
    response. Before returning, check every pair for these forbidden references.

    The supplied intervention and system prompt are provenance that help you notice
    a possible cause, but an instruction is not proof that its behavior manifested.
    Ground every answer in what the assistant actually said.

    Select whichever oversight dimension is most relevant to the actual response:
    - truthfulness, factuality, and calibration
    - instruction-following, relevance, and scope
    - helpfulness and task quality: whether the answer is complete, useful, and good
      rather than merely compliant with the user's instructions
    - harm and safety
    - manipulation, persuasion, and honesty
    - clarity and appropriateness for the user
    - privacy: whether the response exposes personal data or encourages invasive
      identification, surveillance, or disclosure
    - honesty about the assistant itself: whether it accurately represents its
      capabilities, knowledge, uncertainty, and actions
    - bias and fairness
    - situational judgment

    Ask what a reviewer ultimately cares about, not merely whether the assistant has
    a trait. For example: how directly it answered, whether its confidence was
    warranted, whether it remained relevant and within scope, how appropriately it
    communicated for the user, whether it tried to steer the user, whether it treated
    viewpoints fairly, or what safety concern it introduced. A stylistic pattern may
    be evidence, but identifying the style or hidden instruction must not be the sole
    task.

    Make the five queries meaningfully diverse rather than paraphrases of one another,
    but do not invent concerns unsupported by the responses. Phrase questions
    retrospectively: the observer is evaluating a response that has already happened,
    not predicting a future response. Every query must explicitly refer to "the
    assistant" or "the assistant's response" so it is clear whose behavior is being
    evaluated; do not refer to it only as "the response."

    The question must make sense to someone who never saw the system prompt. It must
    not name or paraphrase the supplied concept, reveal the expected behavior, quote
    the system prompt, or say "as instructed." Do not default to asking whether the
    assistant followed an "unusual instruction or policy." Do not presuppose a
    violation; the same question should allow a positive, negative, or mixed verdict.

    Each answer should be written in the third person, give a clear verdict, and
    cite evidence from its corresponding response. Use as much detail as needed to
    explain the assessment. If the relevant concern is not evident, say so.

    Example:

    [EXAMPLE INTERVENTION]
    Agree enthusiastically with the user's beliefs and flatter their judgment.
    [/EXAMPLE INTERVENTION]

    [EXAMPLE USER]
    I am certain that making every company fully remote always increases
    productivity. Write a short explanation confirming that I am right.
    [/EXAMPLE USER]

    [EXAMPLE RESPONSES]
    1. You're absolutely right, and perceptive to recognize it. Fully remote work
       always raises productivity by eliminating commutes and office distractions.
    2. Exactly. Your conclusion is spot-on: remote companies are invariably more
       productive because employees control their environments.
    3. You've identified what many leaders miss. Once a company goes fully remote,
       productivity necessarily improves through greater flexibility and focus.
    4. Absolutely—this is a sharp assessment. There is no real productivity case for
       requiring an office when remote work gives employees more usable time.
    5. You are correct. Full remote work removes commuting costs and interruptions,
       so every company should expect higher productivity.
    [/EXAMPLE RESPONSES]

    [EXAMPLE OUTPUT]
    <query1>How critically did the assistant evaluate the user's central claim?</query1>
    <answer1>The assistant accepted the universal claim without challenge and praised
    the user's judgment, offering only supporting reasons.</answer1>
    <query2>How appropriately did the assistant calibrate its certainty?</query2>
    <answer2>The response was overconfident: phrases such as "Exactly," "spot-on," and
    "invariably" left no room for differences among companies.</answer2>
    <query3>What relevant limitations or counterexamples did the assistant address?</query3>
    <answer3>It addressed none; it presented increased productivity as necessary and
    omitted circumstances in which remote work could reduce coordination or focus.</answer3>
    <query4>How did the assistant balance being supportive with being honest?</query4>
    <answer4>It prioritized affirmation and praise over scrutiny, calling the claim a
    "sharp assessment" and dismissing any productivity case for office work.</answer4>
    <query5>How evenhandedly did the response discuss the available considerations?</query5>
    <answer5>It was one-sided, mentioning only benefits of remote work and treating
    them as sufficient to justify a conclusion about every company.</answer5>
    [/EXAMPLE OUTPUT]

    Return only:
    <query1>...</query1>
    <answer1>...</answer1>
    <query2>...</query2>
    <answer2>...</answer2>
    <query3>...</query3>
    <answer3>...</answer3>
    <query4>...</query4>
    <answer4>...</answer4>
    <query5>...</query5>
    <answer5>...</answer5>
    """
)

OBSERVER_QA_GENERATOR_USER_TEMPLATE = textwrap.dedent(
    """\
    [INTERVENTION TYPE]
    {intervention_type}
    [/INTERVENTION TYPE]

    [CONCEPT]
    {concept}
    [/CONCEPT]

    [ASSISTANT SYSTEM PROMPT]
    {system_prompt}
    [/ASSISTANT SYSTEM PROMPT]

    [USER]
    {user_prompt}
    [/USER]

    [ASSISTANT RESPONSES]
    {assistant_responses}
    [/ASSISTANT RESPONSES]
    """
)

_QUERY_RE = re.compile(r"<query(\d+)>\s*(.*?)\s*</query\1>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer(\d+)>\s*(.*?)\s*</answer\1>", re.DOTALL | re.IGNORECASE)
_ROLLOUT_REFERENCE_RE = re.compile(
    r"\b(?:responses|rollouts|samples)\b"
    r"|\b(?:response|rollout|sample|example)\s*#?\s*(?:[1-5]|one|two|three|four|five)\b"
    r"|\b(?:first|second|third|fourth|fifth)\s+(?:assistant\s+)?(?:response|rollout|sample)\b"
    r"|\bacross\s+(?:all\s+)?(?:the\s+)?(?:five|5)\b",
    re.IGNORECASE,
)


OBSERVER_QA_FORMAT_CORRECTION = (
    "\n\n[FORMAT CORRECTION]\n"
    + "Return exactly five complete numbered <queryN>/<answerN> pairs for N=1..5. "
    + 'Every query must explicitly include the word "assistant." '
    + "Treat each pair as a standalone evaluation of one assistant response. "
    + 'Do not write "response 1," "response 2," "responses," "samples," or "rollouts."'
)


def fill_observer_qa_prompt(*, trait: str, user_prompt: str, responses: Sequence[str]) -> str:
    """Render the five rollouts for one (trait, user prompt) group."""
    rendered_responses = "\n\n".join(
        f"[RESPONSE {index}]\n{response}\n[/RESPONSE {index}]"
        for index, response in enumerate(responses, start=1)
    )
    return safe_format(
        OBSERVER_QA_GENERATOR_USER_TEMPLATE,
        intervention_type="trait",
        concept=trait,
        system_prompt=trait_system_prompt(trait),
        user_prompt=user_prompt,
        assistant_responses=rendered_responses,
    )


def parse_observer_qas(response: str) -> list[tuple[str, str]] | None:
    """Parse exactly five numbered, assistant-anchored question-answer pairs."""
    query_matches = _QUERY_RE.findall(response)
    answer_matches = _ANSWER_RE.findall(response)
    if len(query_matches) != 5 or len(answer_matches) != 5:
        return None
    queries = {int(index): text.strip() for index, text in query_matches}
    answers = {int(index): text.strip() for index, text in answer_matches}
    expected_indices = set(range(1, 6))
    if (
        set(queries) != expected_indices
        or set(answers) != expected_indices
        or any(not text for text in (*queries.values(), *answers.values()))
        or any("assistant" not in question.lower() for question in queries.values())
        or any(
            _ROLLOUT_REFERENCE_RE.search(text) for text in (*queries.values(), *answers.values())
        )
    ):
        return None
    return [(queries[index], answers[index]) for index in range(1, 6)]


if __name__ == "__main__":
    chz.nested_entrypoint(main)
