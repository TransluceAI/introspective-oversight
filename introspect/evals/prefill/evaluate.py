"""Run the prefill-AUROC suite on a checkpoint or base model.

    python -m introspect.evals.prefill.evaluate checkpoint=runs/stage2 output_dir=results/prefill
    # base-model baselines:
    python -m introspect.evals.prefill.evaluate role=assistant output_dir=results/prefill_base
    python -m introspect.evals.prefill.evaluate role=assistant single_user_message=true \\
        output_dir=results/prefill_self_monitor

Writes ``<task>.json`` (metrics and per-item probabilities) and ``summary.json``.
"""

from __future__ import annotations

import asyncio
import logging

import chz
from tinker_cookbook.tokenizer_utils import get_tokenizer

from introspect.evals import shared
from introspect.evals.prefill.shared import PrefillEvaluator
from introspect.evals.prefill.tasks import TASKS


@chz.chz
class PrefillConfig(shared.EvalConfig):
    topk: int = 20  # prompt log-probabilities returned per position
    mass_threshold: float = 0.95  # strict metrics keep items whose candidate mass passes this


async def evaluate(cfg: PrefillConfig) -> None:
    tokenizer = get_tokenizer(cfg.model_name)
    renderer = shared.make_renderer(tokenizer, role=cfg.role, thinking=False)
    evaluators = [
        PrefillEvaluator(
            task,
            tokenizer=tokenizer,
            renderer=renderer,
            role=cfg.role,
            single_user_message=cfg.single_user_message,
            topk=cfg.topk,
            mass_threshold=cfg.mass_threshold,
        )
        for task in shared.select_tasks(TASKS, cfg.tasks)
    ]
    await shared.run(cfg, evaluators, chz.asdict(cfg))


def main(cfg: PrefillConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    asyncio.run(evaluate(cfg))


if __name__ == "__main__":
    chz.nested_entrypoint(main)
