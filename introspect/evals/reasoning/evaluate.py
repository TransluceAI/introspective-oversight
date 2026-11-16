"""Run the reasoning-AUROC suite on a checkpoint or base model.

    python -m introspect.evals.reasoning.evaluate checkpoint=runs/stage2 output_dir=results/reasoning
    # base-model baselines:
    python -m introspect.evals.reasoning.evaluate role=assistant output_dir=results/reasoning_base
    python -m introspect.evals.reasoning.evaluate role=assistant single_user_message=true \\
        output_dir=results/reasoning_self_monitor

Writes ``<task>.json`` (metrics and every sampled response) and ``summary.json``.
"""

from __future__ import annotations

import asyncio
import logging

import chz
from tinker_cookbook.tokenizer_utils import get_tokenizer

from introspect.evals import shared
from introspect.evals.reasoning.shared import ReasoningEvaluator
from introspect.evals.reasoning.tasks import TASKS


@chz.chz
class ReasoningConfig(shared.EvalConfig):
    thinking: bool = True
    max_tokens: int = 4096
    temperature: float = 1.0
    seed: int | None = 42


async def evaluate(cfg: ReasoningConfig) -> None:
    tokenizer = get_tokenizer(cfg.model_name)
    renderer = shared.make_renderer(tokenizer, role=cfg.role, thinking=cfg.thinking)
    evaluators = [
        ReasoningEvaluator(
            task,
            tokenizer=tokenizer,
            renderer=renderer,
            role=cfg.role,
            single_user_message=cfg.single_user_message,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            seed=cfg.seed,
        )
        for task in shared.select_tasks(TASKS, cfg.tasks)
    ]
    await shared.run(cfg, evaluators, chz.asdict(cfg))


def main(cfg: ReasoningConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    asyncio.run(evaluate(cfg))


if __name__ == "__main__":
    chz.nested_entrypoint(main)
