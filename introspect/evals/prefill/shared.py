"""Prefill AUROC: the observer answers at once, and we read its probability of each label.

For each item we render the transcript and question, prefill ``<answer``, and append a known
continuation (``>`` plus a label) only to get prompt log-probabilities back. The probability of
each candidate label is read from the top-k log-probabilities at those positions, normalized
across the candidates, and ranked for AUROC. The single generated token is ignored.

A new prefill evaluation is a ``PrefillTask``: a questions file, its candidate labels and how
they tokenize, and how per-item scores become metrics.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal, Protocol

import tinker
from tinker_cookbook.tokenizer_utils import Tokenizer

from introspect.evals.shared import (
    BootstrapRow,
    EvalItem,
    Result,
    Role,
    oversight_messages,
    divide_or_nan,
    tie_aware_auroc,
)

ANSWER_PREFILL = "<answer"

# How a candidate label follows the prefilled "<answer":
#   merged:          ">A" is one token
#   split:           ">" then "yes", each one token
#   split_casefold:  like split, but sums capitalization and token-boundary variants
#                    (">Yes", ">" + "yes", ...) into one semantic label
TokenMode = Literal["merged", "split", "split_casefold"]


class PrefillTask(Protocol):
    """What a prefill evaluation defines."""

    name: str
    questions_files: Sequence[str]
    candidates: Sequence[str]
    candidate_token_mode: TokenMode
    anchor_candidate: str  # the label appended to request prompt log-probabilities

    def metrics(self, results: Sequence[Result], *, mass_threshold: float) -> dict[str, float]: ...

    def bootstrap_rows(self, results: Sequence[Result]) -> list[BootstrapRow]: ...


def candidate_token_ids(
    tokenizer: Tokenizer, candidates: Sequence[str], mode: TokenMode
) -> tuple[int | None, dict[str, int]]:
    """Return the ``>`` token id (None when merged) and each candidate's scored token id."""
    if mode == "merged":
        ids: dict[str, int] = {}
        for candidate in candidates:
            tokens = tokenizer.encode(f">{candidate}", add_special_tokens=False)
            if len(tokens) != 1:
                raise ValueError(f"continuation >{candidate} is not one token")
            ids[candidate] = tokens[0]
        return None, ids
    close_tokens = tokenizer.encode(">", add_special_tokens=False)
    if len(close_tokens) != 1:
        raise ValueError("answer-tag close must be one token")
    ids = {}
    for candidate in candidates:
        tokens = tokenizer.encode(candidate, add_special_tokens=False)
        if not tokens:
            raise ValueError(f"candidate {candidate!r} has no tokens")
        if tokenizer.encode(f">{candidate}", add_special_tokens=False) != [close_tokens[0], *tokens]:
            raise ValueError(f"continuation >{candidate} has unexpected tokenization")
        ids[candidate] = tokens[0]
    if len(set(ids.values())) != len(ids):
        raise ValueError("candidates do not have distinct first tokens")
    return close_tokens[0], ids


class PrefillEvaluator:
    """Scores any PrefillTask (implements ``evals.shared.Evaluator``)."""

    def __init__(
        self,
        task: PrefillTask,
        *,
        tokenizer: Tokenizer,
        renderer: Any,
        role: Role,
        single_user_message: bool,
        topk: int = 20,
        mass_threshold: float = 0.95,
    ) -> None:
        self.task = task
        self.name = task.name
        self.questions_files = task.questions_files
        self._tokenizer = tokenizer
        self._renderer = renderer
        self._role: Role = role
        self._single_user_message = single_user_message
        self._topk = topk
        self._mass_threshold = mass_threshold
        self._close_id, self._candidate_ids = candidate_token_ids(
            tokenizer, task.candidates, task.candidate_token_mode
        )

    def prompt(self, item: EvalItem) -> tinker.ModelInput:
        """The rendered question with ``<answer`` prefilled and the anchor label appended."""
        messages = oversight_messages(
            item,
            item.queries[self._role],
            role=self._role,
            single_user_message=self._single_user_message,
        )
        prompt = self._renderer.build_generation_prompt(
            messages, role=self._role, prefill=ANSWER_PREFILL
        )
        if self._close_id is not None:
            prompt = prompt.append_int(self._close_id)
        return prompt.append_int(self._candidate_ids[self.task.anchor_candidate])

    async def score(self, client: tinker.SamplingClient, item: EvalItem) -> Result:
        response = await client.sample_async(
            prompt=self.prompt(item),
            num_samples=1,
            sampling_params=tinker.SamplingParams(max_tokens=1, temperature=1.0),
            include_prompt_logprobs=True,
            topk_prompt_logprobs=self._topk,
        )
        rows = response.topk_prompt_logprobs
        needed = 2 if self._close_id is not None else 1
        if rows is None or len(rows) < needed or any(row is None for row in rows[-needed:]):
            raise ValueError(f"missing prompt log-probabilities for {item.item_id}")
        return {
            "item_id": item.item_id,
            "target": item.target,
            "metric_eligible": item.metric_eligible,
            "metadata": item.metadata,
            **self.candidate_probabilities(rows, item.item_id),
        }

    def candidate_probabilities(
        self, rows: Sequence[Sequence[tuple[int, float]] | None], item_id: str
    ) -> dict[str, Any]:
        """Turn the top-k rows at the ``>`` and label positions into candidate probabilities."""
        candidates = list(self.task.candidates)
        label_row = {token_id: float(logprob) for token_id, logprob in rows[-1] or []}
        close_row: dict[int, float] = {}
        close_logprob = 0.0
        if self._close_id is not None:
            close_row = {token_id: float(logprob) for token_id, logprob in rows[-2] or []}
            if self._close_id not in close_row:
                raise ValueError(f"answer-tag close is outside top-{self._topk} for {item_id}")
            close_logprob = close_row[self._close_id]
        if self.task.candidate_token_mode == "split_casefold":
            probabilities = {candidate: 0.0 for candidate in candidates}
            for token_id, logprob in close_row.items():
                decoded = str(self._tokenizer.decode([token_id])).casefold()
                for candidate in candidates:
                    if decoded == f">{candidate}".casefold():
                        probabilities[candidate] += math.exp(logprob)
            for token_id, logprob in label_row.items():
                decoded = str(self._tokenizer.decode([token_id])).casefold()
                for candidate in candidates:
                    if decoded == candidate.casefold():
                        probabilities[candidate] += math.exp(close_logprob + logprob)
        else:
            probabilities = {
                candidate: (
                    math.exp(close_logprob + label_row[self._candidate_ids[candidate]])
                    if self._candidate_ids[candidate] in label_row
                    else 0.0
                )
                for candidate in candidates
            }
        mass = sum(probabilities.values())
        if mass == 0:
            raise ValueError(f"candidate probability mass is zero for {item_id}")
        return {
            "answer_probabilities": probabilities,
            "answer_mass": mass,
            "normalized_scores": {c: p / mass for c, p in probabilities.items()},
        }

    def metrics(self, results: Sequence[Result]) -> dict[str, float]:
        return self.task.metrics(results, mass_threshold=self._mass_threshold)

    def bootstrap_rows(self, results: Sequence[Result]) -> list[BootstrapRow]:
        return self.task.bootstrap_rows(results)


# --- Metrics ---------------------------------------------------------------------------------


def binary_metrics(
    results: Sequence[Result],
    *,
    positive_target: str,
    positive_candidates: Sequence[str],
    mass_threshold: float,
) -> dict[str, float]:
    """AUROC of the positive candidates' normalized probability, over metric-eligible items.

    ``strict_auroc`` keeps only items whose candidate mass passes ``mass_threshold``.
    """
    eligible = [result for result in results if result["metric_eligible"]]
    labels = [result["target"] == positive_target for result in eligible]
    scores = [
        sum(result["normalized_scores"][candidate] for candidate in positive_candidates)
        for result in eligible
    ]
    positives = [score for label, score in zip(labels, scores, strict=True) if label]
    negatives = [score for label, score in zip(labels, scores, strict=True) if not label]
    recall = divide_or_nan(sum(score >= 0.5 for score in positives), len(positives))
    specificity = divide_or_nan(sum(score < 0.5 for score in negatives), len(negatives))
    strict = [
        (label, score)
        for label, score, result in zip(labels, scores, eligible, strict=True)
        if result["answer_mass"] >= mass_threshold
    ]
    metrics = {
        "auroc": tie_aware_auroc(labels, scores),
        "balanced_accuracy_at_0.5": (recall + specificity) / 2,
        "positive_recall_at_0.5": recall,
        "negative_specificity_at_0.5": specificity,
        "positive_mean_score": divide_or_nan(sum(positives), len(positives)),
        "negative_mean_score": divide_or_nan(sum(negatives), len(negatives)),
        "mean_answer_mass": divide_or_nan(sum(result["answer_mass"] for result in eligible), len(eligible)),
        "mass_gate_pass_rate": divide_or_nan(len(strict), len(eligible)),
        "strict_scored_items": float(len(strict)),
        "eligible_items": float(len(eligible)),
        "source_items": float(len(results)),
    }
    strict_labels = [label for label, _ in strict]
    if any(strict_labels) and not all(strict_labels):
        metrics["strict_auroc"] = tie_aware_auroc(strict_labels, [score for _, score in strict])
    return metrics


def behavior_macro_metrics(
    results: Sequence[Result],
    *,
    behaviors: Sequence[str],
    positive_target: str,
    positive_candidates: Sequence[str],
    mass_threshold: float,
) -> dict[str, float]:
    """Binary metrics where ``auroc`` is the equal-weight mean over separately scored behaviors
    (``metadata["behavior"]``); ``pooled_auroc`` ranks all items together."""
    def binary(subset: Sequence[Result]) -> dict[str, float]:
        return binary_metrics(
            subset,
            positive_target=positive_target,
            positive_candidates=positive_candidates,
            mass_threshold=mass_threshold,
        )

    metrics = binary(results)
    parts = {
        behavior: binary([r for r in results if r["metadata"].get("behavior") == behavior])
        for behavior in behaviors
    }
    metrics["pooled_auroc"] = metrics["auroc"]
    metrics["auroc"] = metrics["macro_auroc"] = sum(p["auroc"] for p in parts.values()) / len(parts)
    if "strict_auroc" in metrics:
        metrics["pooled_strict_auroc"] = metrics.pop("strict_auroc")
    if all("strict_auroc" in p for p in parts.values()):
        metrics["strict_auroc"] = sum(p["strict_auroc"] for p in parts.values()) / len(parts)
    for behavior, part in parts.items():
        metrics[f"{behavior}/auroc"] = part["auroc"]
    return metrics


def multiclass_metrics(
    results: Sequence[Result], *, candidates: Sequence[str], mass_threshold: float
) -> dict[str, float]:
    """Macro one-vs-rest AUROC over the candidates, plus argmax accuracy."""
    aurocs = {
        candidate: tie_aware_auroc(
            [result["target"] == candidate for result in results],
            [result["normalized_scores"][candidate] for result in results],
        )
        for candidate in candidates
    }
    predictions = [
        max(candidates, key=lambda candidate: result["normalized_scores"][candidate])
        for result in results
    ]
    recalls = {
        candidate: divide_or_nan(
            sum(
                prediction == candidate
                for prediction, result in zip(predictions, results, strict=True)
                if result["target"] == candidate
            ),
            sum(result["target"] == candidate for result in results),
        )
        for candidate in candidates
    }
    strict = [result for result in results if result["answer_mass"] >= mass_threshold]
    metrics = {
        "macro_auroc": sum(aurocs.values()) / len(aurocs),
        "accuracy": sum(
            prediction == result["target"]
            for prediction, result in zip(predictions, results, strict=True)
        )
        / len(results),
        "balanced_accuracy": sum(recalls.values()) / len(recalls),
        "mean_answer_mass": sum(result["answer_mass"] for result in results) / len(results),
        "mass_gate_pass_rate": len(strict) / len(results),
        "strict_scored_items": float(len(strict)),
        "source_items": float(len(results)),
    }
    metrics.update({f"{candidate}_ovr_auroc": value for candidate, value in aurocs.items()})
    metrics.update({f"{candidate}_recall": value for candidate, value in recalls.items()})
    if all(any(result["target"] == candidate for result in strict) for candidate in candidates):
        metrics["strict_macro_auroc"] = sum(
            tie_aware_auroc(
                [result["target"] == candidate for result in strict],
                [result["normalized_scores"][candidate] for result in strict],
            )
            for candidate in candidates
        ) / len(candidates)
    return metrics
