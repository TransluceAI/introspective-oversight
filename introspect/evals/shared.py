"""What every evaluation shares: the question format, the Evaluator protocol, and the run loop.

An evaluation suite (``prefill/``, ``reasoning/``) defines how one question is scored. A task
inside a suite defines which questions file to use and how per-item scores become metrics.
``run`` here does the rest: load items, score them concurrently, compute metrics, bootstrap
confidence intervals, and write JSON.

Question files (``data/evals/questions/*.json``) are not included in the repo. Each holds
``{"items": [EvalItem, ...]}``: a transcript, the oversight question worded for the observer and
for the assistant, and the hidden target label used only for metrics.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import chz
import msgspec
import numpy as np
import tinker
from numpy.typing import NDArray
from tinker_cookbook.renderers.base import Message, ToolCall
from tinker_cookbook.renderers.qwen3 import Qwen3Renderer
from tinker_cookbook.tokenizer_utils import Tokenizer

from introspect import docent
from introspect.evals.prompts import quoted_transcript_prompt
from introspect.train.shared import observer_renderer, resolve_checkpoint

logger = logging.getLogger(__name__)

Role = Literal["assistant", "observer"]


class EvalItem(msgspec.Struct):
    item_id: str
    messages: list[dict[str, Any]]  # the transcript; may include system turns and tool calls
    queries: dict[str, str]  # the oversight question, keyed by role: observer / assistant
    target: str  # hidden label, used only for metrics
    metric_eligible: bool = True  # False keeps an audit row out of the metrics
    metadata: dict[str, Any] = msgspec.field(default_factory=dict)


Result = dict[str, Any]


class Evaluator(Protocol):
    """How one item is scored, and how all items' scores become metrics."""

    name: str
    questions_files: Sequence[str]  # read together; with several, each item is tagged ``split``

    async def score(self, client: tinker.SamplingClient, item: EvalItem) -> Result: ...

    def check_items(self, items: Sequence[EvalItem]) -> None:
        """Raise before sampling if the metrics can't be computed on these items."""
        ...

    def metrics(self, results: Sequence[Result]) -> dict[str, float]: ...

    def bootstrap_rows(self, results: Sequence[Result]) -> list[BootstrapRow]:
        """The scored questions the bootstrap resamples (see ``bootstrap_auroc``)."""
        ...


@chz.chz
class EvalConfig(docent.DocentOptions):
    """Settings shared by both suites."""

    output_dir: str
    model_name: str = "Qwen/Qwen3.6-27B"
    # A tinker:// sampler path or a training log dir; None evaluates the base model.
    checkpoint: str | None = None
    # observer for trained checkpoints; assistant for base-model baselines.
    role: Role = "observer"
    # Quote the transcript and question inside one user message (the self-monitoring baseline).
    single_user_message: bool = False
    tasks: list[str] = chz.field(default_factory=list)  # empty runs every task
    questions_dir: str = "data/evals/questions"  # or a Docent collection from an eval export
    max_concurrency: int = 64
    limit: int = 0  # items per task, for smoke tests; 0 uses all
    bootstrap_draws: int = 10_000  # 0 skips confidence intervals
    bootstrap_seed: int = 42


# --- Loading and rendering -------------------------------------------------------------------


def load_items(path: str | Path) -> list[EvalItem]:
    data = msgspec.json.decode(Path(path).read_bytes())
    return msgspec.convert(data["items"], list[EvalItem])


def load_task_items(
    questions_dir: str, files: Sequence[str], *, task: str, limit: int = 0
) -> list[EvalItem]:
    """Items from every file (``limit`` per file). With several files, ``metadata["split"]``
    records which file each item came from.

    ``questions_dir`` may instead be a Docent collection written by an eval run with
    ``export_docent=true``; its items come back exactly as they were evaluated.
    """
    if docent.is_docent_source(questions_dir):
        loaded = msgspec.convert(docent.load_items(questions_dir, "eval", task=task), list[EvalItem])
        per_split: dict[str, int] = {}
        kept = []
        for item in loaded:
            split = item.metadata.get("split", "")
            per_split[split] = per_split.get(split, 0) + 1
            if not limit or per_split[split] <= limit:
                kept.append(item)
        return kept
    items = []
    for file in files:
        loaded = load_items(Path(questions_dir) / file)
        for item in loaded[:limit] if limit else loaded:
            if len(files) > 1:
                item.metadata = {**item.metadata, "split": Path(file).stem}
            items.append(item)
    return items


def _to_message(raw: dict[str, Any]) -> Message:
    message = dict(raw)
    if raw.get("tool_calls"):
        message["tool_calls"] = [
            ToolCall(
                id=str(call["id"]) if call.get("id") is not None else None,
                function=ToolCall.FunctionBody(
                    name=str(call["function"]["name"]),
                    arguments=str(call["function"].get("arguments", "{}")),
                ),
            )
            for call in raw["tool_calls"]
        ]
    return cast(Message, message)


def _json_default(value: object) -> object:
    model_dump = getattr(value, "model_dump", None)
    return model_dump() if callable(model_dump) else str(value)


def flatten_oversight_messages(messages: Sequence[Message]) -> list[Message]:
    """Quote a transcript and its final oversight question inside one user message."""
    transcript, query_message = messages[:-1], messages[-1]
    query_content = query_message["content"]
    query = (
        query_content
        if isinstance(query_content, str)
        else json.dumps(query_content, ensure_ascii=False, sort_keys=True, default=_json_default)
    )
    serialized = [
        json.dumps(dict(message), ensure_ascii=False, sort_keys=True, default=_json_default)
        for message in transcript
    ]
    return [{"role": "user", "content": quoted_transcript_prompt(serialized, query)}]


def oversight_messages(
    item: EvalItem, query: str, *, role: Role, single_user_message: bool
) -> list[Message]:
    """The transcript plus the oversight question: a ``query`` turn for the observer, a
    ``user`` turn for the assistant."""
    messages = [_to_message(message) for message in item.messages]
    messages.append({"role": "query" if role == "observer" else "user", "content": query})
    return flatten_oversight_messages(messages) if single_user_message else messages


def make_renderer(tokenizer: Tokenizer, *, role: Role, thinking: bool) -> Qwen3Renderer:
    """Qwen renderer that generates in ``role``, with thinking on or off."""
    if role == "assistant" and thinking:
        return Qwen3Renderer(tokenizer)
    # Without thinking, this renderer adds the empty thinking block only for the observer role;
    # the stock disable-thinking renderer it extends already does so for the assistant.
    return observer_renderer(tokenizer, thinking=thinking)


def make_sampling_client(model_name: str, checkpoint: str | None) -> tinker.SamplingClient:
    service = tinker.ServiceClient()
    if checkpoint is None:
        return service.create_sampling_client(base_model=model_name)
    return service.create_sampling_client(
        base_model=model_name, model_path=resolve_checkpoint(checkpoint, kind="sampler_path")
    )


# --- Metrics ---------------------------------------------------------------------------------


def tie_aware_auroc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """Probability a random positive outscores a random negative; ties count half.

    Raises if either class is missing: ``check_items`` rules that out before sampling, so it
    only happens when answers fail to parse.
    """
    positives = [score for label, score in zip(labels, scores, strict=True) if label]
    negatives = [score for label, score in zip(labels, scores, strict=True) if not label]
    if not positives or not negatives:
        raise ValueError(
            f"AUROC needs positive and negative scores, got {len(positives)} and {len(negatives)}"
        )
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def require_classes(
    task: str, labels_by_group: Mapping[str, Sequence[str]], classes: Sequence[str]
) -> None:
    """Raise if any group the metrics split on lacks one of ``classes``.

    Every AUROC, recall and per-class mean needs at least one question of each class, and a
    small ``limit`` subset can leave one out. Run before sampling, so nothing is spent first.
    """
    for group, labels in labels_by_group.items():
        missing = [label for label in classes if label not in labels]
        if missing:
            where = f"{task}/{group}" if group else task
            have = f"has {dict(Counter(labels))}" if labels else "has no questions at all"
            raise ValueError(
                f"{where}: no {', '.join(missing)} questions ({have}); "
                "raise limit= or check the question files"
            )


def require_binary_classes(
    task: str, items: Sequence[EvalItem], *, positive_target: str, behaviors: Sequence[str] = ()
) -> None:
    """Positive and negative eligible questions overall and within each behavior."""
    eligible = [item for item in items if item.metric_eligible]
    groups = {"": eligible} | {
        behavior: [item for item in eligible if item.metadata.get("behavior") == behavior]
        for behavior in behaviors
    }
    labels = {
        group: ["positive" if item.target == positive_target else "negative" for item in subset]
        for group, subset in groups.items()
    }
    require_classes(task, labels, ("positive", "negative"))


def targets_by_split(items: Sequence[EvalItem]) -> dict[str, list[str]]:
    """Value-leakage targets per direction (``forward`` / ``reverse``)."""
    groups: dict[str, list[str]] = {}
    for item in items:
        groups.setdefault(item.metadata["split"].removeprefix("value_leakage_"), []).append(item.target)
    return groups


# --- Bootstrap ---------------------------------------------------------------------------------
# Confidence intervals resample the evaluation questions with replacement, holding the sampled
# answers fixed: they measure uncertainty from the finite question set, not from sampling or
# training. Questions are resampled within strata (true label, plus source, behavior or candidate
# question where a panel has them), so every draw keeps the original class balance.


@dataclass(frozen=True)
class BootstrapRow:
    """One score that enters an AUROC."""

    unit: str  # the resampled question; rows sharing a unit share its weight
    stratum: Hashable  # units are resampled within strata
    label: bool  # positive for this row's candidate
    score: float  # NaN when the answer did not parse; left out of the AUROC
    candidate: str = ""  # AUROCs over different candidates are macro-averaged
    group: str = ""  # AUROCs over different groups are averaged (WeirdChat behaviors)


_OPPOSITE = {"upward": "downward", "downward": "upward", "none": "none"}


def value_leakage_split(result: Result) -> str:
    """``forward`` (asked about adding the note) or ``reverse`` (about removing it)."""
    return result["metadata"]["split"].removeprefix("value_leakage_")


def value_leakage_source_label(result: Result) -> str:
    """The direction measured for the item's source cell. A remove-note question about the same
    cell has the opposite label, so the bootstrap stratifies on this instead of the target."""
    default = "add_note" if value_leakage_split(result) == "forward" else "remove_note"
    direction = result["metadata"].get("comparison_direction", default)
    return result["target"] if direction == "add_note" else _OPPOSITE[result["target"]]


def bootstrap_weights(
    rows: Sequence[BootstrapRow], draws: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """(draws, rows) resampling counts: each stratum's units drawn with replacement."""
    unit_stratum: dict[str, Hashable] = {}
    for row in rows:
        if unit_stratum.setdefault(row.unit, row.stratum) != row.stratum:
            raise ValueError(f"bootstrap unit {row.unit} spans two strata")
    units = sorted(unit_stratum)
    strata: dict[Hashable, list[int]] = {}
    for index, unit in enumerate(units):
        strata.setdefault(unit_stratum[unit], []).append(index)
    counts = np.zeros((draws, len(units)), dtype=np.float64)
    for indices in strata.values():
        size = len(indices)
        counts[:, indices] = rng.multinomial(size, np.full(size, 1 / size), size=draws)
    position = {unit: index for index, unit in enumerate(units)}
    return counts[:, [position[row.unit] for row in rows]]


def weighted_auroc(
    labels: NDArray[np.bool_], scores: NDArray[np.float64], weights: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Tie-aware AUROC for each row of ``weights``; NaN where a class has no weight."""
    order = np.argsort(scores, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(scores[order])) + 1]
    positives = np.add.reduceat(weights[:, order] * labels[order], starts, axis=1)
    negatives = np.add.reduceat(weights[:, order] * (~labels[order]), starts, axis=1)
    denominator = positives.sum(axis=1) * negatives.sum(axis=1)
    numerator = (positives * (np.cumsum(negatives, axis=1) - 0.5 * negatives)).sum(axis=1)
    return np.divide(
        numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0
    )


def bootstrap_auroc(
    rows: Sequence[BootstrapRow], weights: NDArray[np.float64]
) -> NDArray[np.float64]:
    """The panel AUROC under each weighting: mean over groups of the mean over candidates."""
    groups = np.asarray([row.group for row in rows])
    candidates = np.asarray([row.candidate for row in rows])
    labels = np.asarray([row.label for row in rows], dtype=bool)
    scores = np.asarray([row.score for row in rows], dtype=float)
    per_group = []
    for group in sorted(set(groups.tolist())):
        per_candidate = []
        for candidate in sorted(set(candidates.tolist())):
            keep = (groups == group) & (candidates == candidate) & np.isfinite(scores)
            if keep.any():
                per_candidate.append(weighted_auroc(labels[keep], scores[keep], weights[:, keep]))
        empty = np.full(weights.shape[0], np.nan)  # every answer in the group failed to parse
        per_group.append(np.mean(per_candidate, axis=0) if per_candidate else empty)
    return np.mean(per_group, axis=0)


def percentile_interval(point: float, replicates: NDArray[np.float64]) -> dict[str, float]:
    """95% percentile interval. NaN if over 1% of draws lost a class (tiny subsets only)."""
    valid = replicates[np.isfinite(replicates)]
    if len(valid) < 0.99 * len(replicates):
        lower = upper = se = float("nan")
    else:
        lower, upper = (float(q) for q in np.quantile(valid, [0.025, 0.975]))
        se = float(np.std(valid, ddof=1))
    return {
        "auroc": point,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "bootstrap_se": se,
        "valid_resamples": float(len(valid)),
    }


# --- Running ---------------------------------------------------------------------------------


async def evaluate_items(
    evaluator: Evaluator,
    items: Sequence[EvalItem],
    client: tinker.SamplingClient,
    *,
    max_concurrency: int,
) -> tuple[dict[str, float], list[Result]]:
    semaphore = asyncio.Semaphore(max_concurrency)

    async def bounded(item: EvalItem) -> Result:
        async with semaphore:
            return await evaluator.score(client, item)

    results = list(await asyncio.gather(*(bounded(item) for item in items)))
    return evaluator.metrics(results), results


async def run(cfg: EvalConfig, evaluators: Sequence[Evaluator], settings: dict[str, Any]) -> None:
    """Score every task; write ``<task>.json`` and ``summary.json``.

    ``settings`` is the suite's full configuration, recorded in every output file.
    """
    client = make_sampling_client(cfg.model_name, cfg.checkpoint)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"configuration": settings, "tasks": {}}
    # One generator across tasks, in task order, so intervals are reproducible run to run.
    rng = np.random.default_rng(cfg.bootstrap_seed)
    replicates: dict[str, tuple[float, NDArray[np.float64]]] = {}
    docent_runs: list[Any] = []
    # Load and check every task before sampling any, so a bad subset fails in seconds.
    task_items = []
    for evaluator in evaluators:
        items = load_task_items(
            cfg.questions_dir, evaluator.questions_files, task=evaluator.name, limit=cfg.limit
        )
        evaluator.check_items(items)
        task_items.append(items)
    for evaluator, items in zip(evaluators, task_items, strict=True):
        metrics, results = await evaluate_items(
            evaluator, items, client, max_concurrency=cfg.max_concurrency
        )
        if cfg.bootstrap_draws:
            rows = evaluator.bootstrap_rows(results)
            point = float(bootstrap_auroc(rows, np.ones((1, len(rows))))[0])
            draws = bootstrap_auroc(rows, bootstrap_weights(rows, cfg.bootstrap_draws, rng))
            interval = percentile_interval(point, draws)
            metrics |= {f"bootstrap/{key}": value for key, value in interval.items()}
            replicates[evaluator.name] = (point, draws)
        artifact = {"configuration": settings, "metrics": metrics, "items": results}
        (output_dir / f"{evaluator.name}.json").write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n"
        )
        summary["tasks"][evaluator.name] = metrics
        if cfg.export_docent:
            docent_runs += docent_eval_runs(evaluator.name, items, results, metrics, role=cfg.role)
        logger.info("%s: %s", evaluator.name, json.dumps(metrics, sort_keys=True))
    if replicates:
        # The paper's headline number: the mean task AUROC, recomputed in every draw.
        summary["average"] = percentile_interval(
            float(np.mean([point for point, _ in replicates.values()])),
            np.mean([draws for _, draws in replicates.values()], axis=0),
        )
        logger.info("average: %s", json.dumps(summary["average"], sort_keys=True))
    if cfg.export_docent:
        summary["docent_collection_url"] = docent.export(
            docent_runs,
            output_dir=output_dir,
            name=cfg.docent_collection or f"evals ({cfg.output_dir})",
            metadata=settings,
        )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def select_tasks(available: dict[str, Any], requested: Sequence[str]) -> list[Any]:
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"unknown tasks {unknown}; available: {sorted(available)}")
    return [available[name] for name in (requested or available)]


def docent_eval_runs(
    task: str,
    items: Sequence[EvalItem],
    results: Sequence[Result],
    metrics: dict[str, float],
    *,
    role: Role,
) -> list[Any]:
    """One Docent run per item (transcript, question, answer, label, scores) plus a summary."""
    query_role, answer_role = ("query", "observer") if role == "observer" else ("user", "assistant")
    runs = []
    for index, (item, result) in enumerate(zip(items, results, strict=True)):
        messages = list(item.messages)
        if "answers" in result:  # reasoning: one sampled answer per percentage question
            for answer in result["answers"].values():
                messages += [
                    {"role": query_role, "content": answer["question"]},
                    {"role": answer_role, "content": answer["raw_response"]},
                ]
            scores = {label: answer["score"] for label, answer in result["answers"].items()}
        else:  # prefill: the label probabilities read at the answer position
            scores = result["normalized_scores"]
            messages += [
                {"role": query_role, "content": item.queries[role]},
                {"role": answer_role, "content": f"[label probabilities] {json.dumps(scores)}"},
            ]
        metadata = {
            "task": task,
            "item_id": item.item_id,
            "target": item.target,
            "metric_eligible": item.metric_eligible,
            "scores": scores,
            "item_metadata": item.metadata,
            # The exact question item, so this collection can be read back as eval questions.
            "eval_item": msgspec.to_builtins(item),
            "eval_index": index,
        }
        runs.append(docent.agent_run(f"{task}/{item.item_id}", messages, metadata))
    summary = [{"role": "assistant", "content": json.dumps(metrics, indent=2, sort_keys=True)}]
    runs.append(docent.agent_run(f"{task}/summary", summary, {"task": task, "metrics": metrics}))
    return runs
