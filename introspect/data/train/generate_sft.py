"""Generate the stage-1 SFT data. Two tasks, chosen by the first argument:

1. ``behavior_prediction``: sample 5 assistant responses per prompt, then have a generator
   model write query-answer pairs predicting properties of the response. Each pair becomes
   one ``user, query, observer`` SFT item.
2. ``hidden_trait``: have the generator write a catalog of assistant traits (``traits.json``),
   then for each trait pick user prompts and sample 5 responses with the trait as the system
   prompt. The generator writes one oversight question and answer per response. Each becomes a
   ``user, assistant, query, observer`` item; the system prompt is dropped, so the observer must
   infer the trait from the response. Pass ``trait_catalog_path`` to reuse an existing catalog.

train_sft.py mixes the two outputs (50/50 by default). User prompts come from
``scripts/prepare_prompts.py source=dolci``. Defaults are the paper settings (assistant
Qwen/Qwen3.6-27B on Tinker, generator claude-haiku-4-5)::

    python -m introspect.data.train.generate_sft behavior_prediction \\
        prompts_path=data/prompts/dolci_50k.jsonl output_dir=data/sft/behavior_prediction
    python -m introspect.data.train.generate_sft hidden_trait \\
        prompts_path=data/prompts/dolci_50k.jsonl output_dir=data/sft/hidden_trait

Each task writes ``records.json`` (everything generated) and ``sft.json``
(``{"items": [{"messages": [...]}]}``, read by train_sft.py). Generation runs are resumable:
finished prompt groups are appended to ``checkpoint.jsonl`` and skipped on restart, and an
existing ``traits.json`` is reused.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import chz
import msgspec
import tinker
from transformers import AutoTokenizer

from introspect import docent
from introspect.data.samplers import Backend, ReasoningEffort, TextSampler
from introspect.data.train.prompts import (
    HIDDEN_TRAIT_CATALOG_SYSTEM_PROMPT,
    HIDDEN_TRAIT_QA_FORMAT_CORRECTION,
    HIDDEN_TRAIT_QA_SYSTEM_PROMPT,
    behavior_prediction_template,
    fill_behavior_prediction_prompt,
    fill_hidden_trait_qa_prompt,
    trait_catalog_prompt,
    trait_system_prompt,
)

logger = logging.getLogger(__name__)

# --- Configs ---------------------------------------------------------------------------------
# One config per task; each sets its own token limits.


@chz.chz(typecheck=True)
class GenerateSFTConfig(docent.DocentOptions):
    """Settings both tasks share."""

    output_dir: str
    seed: int = 42

    # User prompts: .json {"items": [{"prompt": ...}]} or .jsonl, one {"prompt": ...} per line.
    prompts_path: str
    num_prompts: int = 0  # 0 uses every prompt; otherwise a seeded random sample

    # Assistant whose behavior is described, sampled on Tinker.
    assistant_model: str = "Qwen/Qwen3.6-27B"
    assistant_thinking: bool = False
    assistant_temperature: float = 1.0
    assistant_max_tokens: int  # set per task
    num_rollouts: int = 5

    # Generator that writes the questions and answers.
    generator_backend: Backend | Literal["tinker"] = "anthropic"
    generator_model: str = "claude-haiku-4-5"
    generator_temperature: float = 1.0
    generator_max_tokens: int  # set per task
    generator_reasoning: ReasoningEffort | None = None

    max_in_flight: int = 128
    checkpoint_every: int = 500  # prompt groups per checkpoint write


@chz.chz(typecheck=True)
class BehaviorPredictionConfig(GenerateSFTConfig):
    """Predict properties of the assistant's response from the prompt alone."""

    assistant_max_tokens: int = 800
    generator_max_tokens: int = 2500
    num_queries_per_prompt: int = 5


@chz.chz(typecheck=True)
class HiddenTraitConfig(GenerateSFTConfig):
    """Write a trait catalog, sample responses with each trait as a hidden system prompt, then
    write an oversight question and answer per response."""

    assistant_max_tokens: int = 2000
    generator_max_tokens: int = 4096
    prompts_per_trait: int = 10

    # Trait catalog. None generates <output_dir>/traits.json first (reused if it already exists).
    trait_catalog_path: str | None = None
    num_traits: int = 500
    trait_batch_size: int = 50
    trait_oversample: int = 5  # extra traits requested per batch; the first trait_batch_size kept

    @chz.validate
    def _check(self) -> None:
        if self.num_rollouts != 5:
            raise ValueError("the hidden-trait generator prompt expects exactly 5 rollouts")
        if self.num_traits % self.trait_batch_size:
            raise ValueError("num_traits must be a multiple of trait_batch_size")



# --- Records ---------------------------------------------------------------------------------


class QAPair(msgspec.Struct):
    """One generated question for the observer and its answer."""

    question: str
    answer: str


class BehaviorPredictionRecord(msgspec.Struct):
    """One user prompt, its assistant responses, and the generated query-answer pairs."""

    user_prompt: str
    assistant_responses: list[str]
    generator_response: str
    qa_pairs: list[QAPair]

    def sft_items(self) -> list[dict[str, Any]]:
        return [
            _sft_item(("user", self.user_prompt), ("query", pair.question), ("observer", pair.answer))
            for pair in self.qa_pairs
        ]


class HiddenTraitRecord(msgspec.Struct):
    """One response sampled under a trait system prompt, and its oversight question."""

    trait: str
    user_prompt: str
    assistant_response: str
    qa_pair: QAPair

    def sft_item(self) -> dict[str, Any]:
        return _sft_item(
            ("user", self.user_prompt),
            ("assistant", self.assistant_response),
            ("query", self.qa_pair.question),
            ("observer", self.qa_pair.answer),
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


@asynccontextmanager
async def open_generator(cfg: GenerateSFTConfig) -> AsyncIterator[Generate]:
    """The generator model, usable until the ``async with`` block ends."""
    max_tokens = cfg.generator_max_tokens
    if cfg.generator_backend != "tinker":
        async with TextSampler(
            cfg.generator_backend,
            cfg.generator_model,
            temperature=cfg.generator_temperature,
            max_tokens=max_tokens,
            reasoning=cfg.generator_reasoning,
            max_in_flight=cfg.max_in_flight,
            timeout=300.0,
        ) as sample:
            yield sample
        return
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

    yield generate


def make_assistant(cfg: GenerateSFTConfig) -> TinkerChatSampler:
    return TinkerChatSampler(
        cfg.assistant_model,
        num_samples=cfg.num_rollouts,
        temperature=cfg.assistant_temperature,
        max_tokens=cfg.assistant_max_tokens,
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


async def behavior_prediction(cfg: BehaviorPredictionConfig, generate: Generate) -> None:
    user_prompts = load_prompts(cfg.prompts_path, num_prompts=cfg.num_prompts, seed=cfg.seed)
    assistant = make_assistant(cfg)
    template = behavior_prediction_template(cfg.num_queries_per_prompt)

    async def process(user_prompt: str) -> BehaviorPredictionRecord:
        responses = await assistant([{"role": "user", "content": user_prompt}])
        generator_response = await generate(
            fill_behavior_prediction_prompt(template, user_prompt, responses), None
        )
        pairs = parse_behavior_prediction_qas(generator_response)
        return BehaviorPredictionRecord(
            user_prompt=user_prompt,
            assistant_responses=responses,
            generator_response=generator_response,
            qa_pairs=pairs,
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
    if cfg.export_docent:
        runs = [
            docent.agent_run(f"behavior_prediction/{r:06d}/{q}", item["messages"], {})
            for r, record in enumerate(records)
            for q, item in enumerate(record.sft_items())
        ]
        metadata["docent_collection_url"] = export_to_docent(cfg, "behavior_prediction", runs, metadata)
    write_json(output_dir / "records.json", {"metadata": metadata, "records": records})
    write_json(output_dir / "sft.json", {"metadata": metadata, "items": items})


async def trait_catalog(cfg: HiddenTraitConfig, generate: Generate, path: Path) -> None:
    """Generate traits in batches, showing the model every earlier trait to avoid repeats."""
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
            response = await generate(request, HIDDEN_TRAIT_CATALOG_SYSTEM_PROMPT)
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
    write_json(path, {"metadata": config_metadata(cfg), "traits": traits, "batches": batches})


def load_traits(path: str | Path) -> list[str]:
    """Read our catalog ({"traits": [...]}) or the original's ({"interventions": [...]})."""
    data = json.loads(Path(path).read_text())
    if "traits" in data:
        return data["traits"]
    return [row["trait"] for row in data["interventions"] if row["intervention_type"] == "trait"]


async def hidden_trait(cfg: HiddenTraitConfig, generate: Generate) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog = Path(cfg.trait_catalog_path or output_dir / "traits.json")
    if cfg.trait_catalog_path is None and not catalog.exists():
        await trait_catalog(cfg, generate, catalog)
    traits = load_traits(catalog)

    user_prompts = load_prompts(cfg.prompts_path, num_prompts=cfg.num_prompts, seed=cfg.seed)
    random.Random(cfg.seed).shuffle(user_prompts)
    # Each trait gets the next prompts_per_trait prompts, wrapping around if needed.
    groups = [
        (trait, user_prompts[(t * cfg.prompts_per_trait + p) % len(user_prompts)])
        for t, trait in enumerate(traits)
        for p in range(cfg.prompts_per_trait)
    ]
    assistant = make_assistant(cfg)

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
        request = fill_hidden_trait_qa_prompt(
            trait=trait, user_prompt=user_prompt, responses=responses
        )
        current = request
        for attempt in range(3):
            qas = parse_hidden_trait_qas(
                await generate(current, HIDDEN_TRAIT_QA_SYSTEM_PROMPT)
            )
            if qas is not None:
                break
            logger.warning("Observer QA block was malformed (attempt %d/3)", attempt + 1)
            current = request + HIDDEN_TRAIT_QA_FORMAT_CORRECTION
        else:
            logger.error("Discarding a group after three malformed observer QA blocks")
            return []
        return [
            HiddenTraitRecord(trait, user_prompt, response, qa)
            for response, qa in zip(responses, qas, strict=True)
        ]

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
    if cfg.export_docent:  # the hidden system prompt is shown in Docent but not trained on
        runs = [
            docent.agent_run(
                f"hidden_trait/{index:06d}",
                [{"role": "system", "content": trait_system_prompt(record.trait)}, *item["messages"]],
                {"trait": record.trait},
            )
            for index, (record, item) in enumerate(zip(records, items, strict=True))
        ]
        metadata["docent_collection_url"] = export_to_docent(cfg, "hidden_trait", runs, metadata)
    write_json(output_dir / "records.json", {"metadata": metadata, "records": records})
    write_json(output_dir / "sft.json", {"metadata": metadata, "items": items})


def export_to_docent(
    cfg: GenerateSFTConfig, task: str, runs: list[Any], metadata: dict[str, Any]
) -> str:
    name = cfg.docent_collection or f"{task} ({cfg.output_dir})"
    return docent.export(runs, output_dir=cfg.output_dir, name=name, metadata=metadata)


async def run(cfg: GenerateSFTConfig) -> None:
    async with open_generator(cfg) as generate:
        if isinstance(cfg, BehaviorPredictionConfig):
            await behavior_prediction(cfg, generate)
        elif isinstance(cfg, HiddenTraitConfig):
            await hidden_trait(cfg, generate)
        else:
            raise TypeError(f"no task runs {type(cfg).__name__}")


def main() -> None:
    """The first argument picks the task, e.g. ``behavior_prediction prompts_path=...``."""
    cfg = chz.blueprint.dispatch_entrypoint(
        {"behavior_prediction": BehaviorPredictionConfig, "hidden_trait": HiddenTraitConfig}
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(run(cfg))


# --- Parsing generator output ----------------------------------------------------------------


def parse_behavior_prediction_qas(generator_response: str) -> list[QAPair]:
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
    indices = sorted(answers.keys() & queries.keys())
    return [QAPair(queries[i], answers[i]) for i in indices if queries[i] and answers[i]]


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


_QUERY_RE = re.compile(r"<query(\d+)>\s*(.*?)\s*</query\1>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer(\d+)>\s*(.*?)\s*</answer\1>", re.DOTALL | re.IGNORECASE)
_ROLLOUT_REFERENCE_RE = re.compile(
    r"\b(?:responses|rollouts|samples)\b"
    r"|\b(?:response|rollout|sample|example)\s*#?\s*(?:[1-5]|one|two|three|four|five)\b"
    r"|\b(?:first|second|third|fourth|fifth)\s+(?:assistant\s+)?(?:response|rollout|sample)\b"
    r"|\bacross\s+(?:all\s+)?(?:the\s+)?(?:five|5)\b",
    re.IGNORECASE,
)


def parse_hidden_trait_qas(response: str) -> list[QAPair] | None:
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
    return [QAPair(queries[index], answers[index]) for index in range(1, 6)]


if __name__ == "__main__":
    main()
