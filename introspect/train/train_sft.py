"""Stage 1: SFT on observer answers, optionally interleaved with OPD.

Each SFT item is a short conversation ending in an ``observer`` turn; only that turn is trained.

    {"items": [{"messages": [{"role": "user", "content": ...},
                             {"role": "assistant", "content": ...},   # oversight items only
                             {"role": "query", "content": ...},
                             {"role": "observer", "content": ...}]}]}

Pass the two ``generate_sft.py`` outputs and they are mixed here: ``hidden_trait_fraction`` of
the items come from hidden trait (default half), the rest from behavior prediction. Any other
SFT files go in ``train_paths`` and are used as-is.

With ``use_opd=true`` each step first takes one on-policy distillation step toward the base
model on plain assistant prompts, so the observer training doesn't degrade the assistant.

    python -m introspect.train.train_sft \
        behavior_prediction_path=data/sft/behavior_prediction/sft.json \
        hidden_trait_path=data/sft/hidden_trait/sft.json \
        opd_prompt_paths=data/prompts/tulu3_opd.jsonl log_path=runs/stage1_sft
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import chz
import msgspec
import numpy as np
import tinker
import torch
from tinker_cookbook.renderers.base import Message
from tinker_cookbook.renderers.qwen3 import Qwen3Renderer
from tinker_cookbook.supervised.common import compute_mean_nll, datum_from_model_input_weights
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer
from tinker_cookbook.utils import ml_log

from introspect import docent
from introspect.train.shared import (
    ChatMessage,
    NON_THINKING_PREFILL,
    OBSERVER_ROLE,
    THINKING_PREFILL,
    build_opd_dataset,
    create_training_client,
    load_opd_prompts,
    opd_step,
    save_checkpoint,
    split_by_key,
)

logger = logging.getLogger(__name__)


@chz.chz
class SFTConfig:
    # Data and checkpoints
    log_path: str
    behavior_prediction_path: str | None = None  # sft.json from generate_sft behavior_prediction
    hidden_trait_path: str | None = None  # sft.json from generate_sft hidden_trait
    hidden_trait_fraction: float = 0.5  # share of hidden-trait items when both are given
    train_paths: list[str] = chz.field(default_factory=list)  # other SFT files, used as-is
    init_checkpoint: str | None = None  # tinker:// URI or a previous log dir; None = base model

    # SFT
    model_name: str = "Qwen/Qwen3.6-27B"
    lora_rank: int = 32
    learning_rate: float = 1e-5
    batch_size: int = 128
    max_length: int = 8192
    num_epochs: int = 1
    test_num_prompts: int = 1000  # prompts held out for test NLL
    seed: int = 1337

    # OPD
    use_opd: bool = True
    opd_prompt_paths: list[str] = chz.field(default_factory=list)
    opd_held_out_prompts: int = 500  # kept out of OPD training, as in the paper run
    opd_lr_multiplier: float = 1.0
    opd_groups_per_batch: int = 4
    opd_max_prompt_tokens: int = 1024
    opd_max_tokens: int = 256
    opd_kl_coef: float = 1.0

    # Logging
    save_every: int = 25
    eval_every: int = 50
    wandb_project: str | None = None

    @chz.validate
    def _check(self) -> None:
        if not (self.behavior_prediction_path or self.hidden_trait_path or self.train_paths):
            raise ValueError("give behavior_prediction_path, hidden_trait_path, or train_paths")
        if not 0 <= self.hidden_trait_fraction <= 1:
            raise ValueError("hidden_trait_fraction must be in [0, 1]")
        if self.use_opd and not self.opd_prompt_paths:
            raise ValueError("use_opd=true requires opd_prompt_paths")


# --- Data ------------------------------------------------------------------------------------


class SFTItem(msgspec.Struct):
    messages: list[ChatMessage]

    def __post_init__(self) -> None:
        if not self.messages or self.messages[-1].role != OBSERVER_ROLE:
            raise ValueError("SFT item must end with the observer turn to train")

    def user_prompt(self) -> str:
        return next(message.content for message in self.messages if message.role == "user")


def load_items(paths: list[str], *, snapshot_dir: str | None = None) -> list[SFTItem]:
    """Read ``{"items": [...]}`` files (other top-level keys are ignored) or Docent collections.

    A Docent collection (a link or ``docent://<id>``) is saved to ``snapshot_dir`` as read.
    """
    items = []
    for path in paths:
        if docent.is_docent_source(path):
            rows = docent.load_items(path, "sft", snapshot_dir=snapshot_dir)
        else:
            rows = msgspec.json.decode(Path(path).read_bytes())["items"]
        items += msgspec.convert(rows, list[SFTItem])
    if not items:
        raise ValueError("SFT files contain no items")
    return items


def mix_sources(
    behavior: list[SFTItem], hidden_trait: list[SFTItem], *, hidden_trait_fraction: float, seed: int
) -> list[SFTItem]:
    """The largest mix with ``hidden_trait_fraction`` of its items from hidden trait.

    Behavior-prediction items whose prompt also appears in the hidden-trait data are dropped
    first. Whichever source runs out first is used whole; the other is sampled down.
    """
    hidden_prompts = {item.user_prompt() for item in hidden_trait}
    behavior = [item for item in behavior if item.user_prompt() not in hidden_prompts]
    if hidden_trait_fraction == 0:
        return behavior
    if hidden_trait_fraction == 1:
        return hidden_trait
    total = min(
        len(hidden_trait) / hidden_trait_fraction, len(behavior) / (1 - hidden_trait_fraction)
    )
    num_hidden = min(len(hidden_trait), round(total * hidden_trait_fraction))
    num_behavior = min(len(behavior), round(total * (1 - hidden_trait_fraction)))
    rng = random.Random(seed)
    return rng.sample(hidden_trait, num_hidden) + rng.sample(behavior, num_behavior)


def load_training_items(cfg: SFTConfig) -> list[SFTItem]:
    """``train_paths`` as-is, plus the two generated sources mixed at ``hidden_trait_fraction``."""
    def load(paths: list[str]) -> list[SFTItem]:
        return load_items(paths, snapshot_dir=cfg.log_path)

    behavior = load([cfg.behavior_prediction_path]) if cfg.behavior_prediction_path else []
    hidden = load([cfg.hidden_trait_path]) if cfg.hidden_trait_path else []
    if behavior and hidden:
        mixed = mix_sources(
            behavior, hidden, hidden_trait_fraction=cfg.hidden_trait_fraction, seed=cfg.seed
        )
        logger.info(
            "Mixed %d items (%.0f%% hidden trait) from %d behavior-prediction and %d hidden-trait",
            len(mixed), 100 * cfg.hidden_trait_fraction, len(behavior), len(hidden),
        )
    else:
        mixed = behavior + hidden
    return (load(cfg.train_paths) if cfg.train_paths else []) + mixed


@dataclass(frozen=True)
class ObserverTarget:
    """An observer response split into optional reasoning and the final answer."""

    reasoning: str | None
    answer: str

    @classmethod
    def parse(cls, content: str) -> ObserverTarget:
        if content.startswith("<think>"):
            content = content.removeprefix("<think>").lstrip("\n")
        if "</think>" not in content:
            return cls(reasoning=None, answer=content)
        reasoning, _, answer = content.partition("</think>")
        if not answer.strip():
            raise ValueError("Observer reasoning needs a final answer after </think>")
        return cls(reasoning=reasoning.lstrip("\n") or None, answer=answer.lstrip("\n"))


def to_datum(item: SFTItem, *, tokenizer: Tokenizer, renderer: Qwen3Renderer, max_length: int) -> tinker.Datum:
    """Mask the conversation and the thinking prefill; train the observer completion + EOS.

    Non-thinking: MASK <think>\\n\\n</think>\\n\\n | LOSS answer <|im_end|>
    Thinking:     MASK <think>\\n              | LOSS reasoning</think>\\n\\n answer <|im_end|>
    """
    target = ObserverTarget.parse(item.messages[-1].content)
    if target.reasoning is None:
        prefill, completion = NON_THINKING_PREFILL, target.answer
    else:
        prefill, completion = THINKING_PREFILL, f"{target.reasoning}</think>\n\n{target.answer}"
    context = [cast(Message, {"role": m.role, "content": m.content}) for m in item.messages[:-1]]
    prompt = renderer.build_generation_prompt(context, role=OBSERVER_ROLE).append(
        tinker.EncodedTextChunk(tokens=tokenizer.encode(prefill, add_special_tokens=False))
    )
    completion_tokens = tokenizer.encode(
        completion, add_special_tokens=False, split_special_tokens=True
    ) + tokenizer.encode("<|im_end|>", add_special_tokens=False)
    sequence = prompt.append(tinker.EncodedTextChunk(tokens=completion_tokens))
    if sequence.length > max_length:
        raise ValueError(f"SFT item needs {sequence.length} tokens, over max_length={max_length}")
    weights = torch.cat([torch.zeros(prompt.length), torch.ones(len(completion_tokens))])
    # The cookbook shifts inputs, targets, and weights together; don't mask a second time.
    return datum_from_model_input_weights(sequence, weights, reduction="mean")


def epoch_order(num_items: int, *, seed: int, epoch: int) -> np.ndarray:
    """Item order for one epoch, shuffled the same way as the paper run.

    That run shuffled once with HF ``Dataset.shuffle(seed)`` and then reshuffled each epoch with
    the cookbook's ``set_epoch(seed=epoch)``. Both are numpy ``default_rng`` permutations.
    """
    base = np.random.default_rng(seed).permutation(num_items)
    return base[np.random.default_rng(epoch).permutation(num_items)]


def split_items(
    items: list[SFTItem], *, num_held_out_prompts: int, seed: int
) -> tuple[list[SFTItem], list[SFTItem]]:
    """Split by user prompt so no held-out prompt appears in training."""
    if num_held_out_prompts == 0:
        return items, []
    train_prompts, _ = split_by_key(
        [item.user_prompt() for item in items], num_held_out=num_held_out_prompts, seed=seed
    )
    train = [item for item in items if item.user_prompt() in train_prompts]
    held_out = [item for item in items if item.user_prompt() not in train_prompts]
    return train, held_out


# --- Steps -----------------------------------------------------------------------------------


def _nll_metrics(
    result: tinker.ForwardBackwardOutput, data: list[tinker.Datum], prefix: str
) -> dict[str, float]:
    logprobs = [output["logprobs"] for output in result.loss_fn_outputs]
    weights = [datum.loss_fn_inputs["weights"] for datum in data]
    return {f"{prefix}/nll": compute_mean_nll(logprobs, weights)}


async def sft_step(
    training_client: tinker.TrainingClient, data: list[tinker.Datum], learning_rate: float
) -> dict[str, float]:
    forward_backward = await training_client.forward_backward_async(data, loss_fn="cross_entropy")
    optim = await training_client.optim_step_async(tinker.AdamParams(learning_rate=learning_rate))
    result = await forward_backward.result_async()
    await optim.result_async()
    return _nll_metrics(result, data, "sft/train") | {"sft/train/num_sequences": float(len(data))}


async def held_out_nll(
    training_client: tinker.TrainingClient, data: list[tinker.Datum], batch_size: int
) -> dict[str, float]:
    """Mean per-example NLL on the held-out prompts, without updating weights."""
    total = 0.0
    for start in range(0, len(data), batch_size):
        chunk = data[start : start + batch_size]
        future = await training_client.forward_async(chunk, loss_fn="cross_entropy")
        total += _nll_metrics(await future.result_async(), chunk, "x")["x/nll"] * len(chunk)
    return {"sft/test/nll": total / len(data)}


# --- Loop ------------------------------------------------------------------------------------


async def train(cfg: SFTConfig) -> None:
    os.makedirs(cfg.log_path, exist_ok=True)
    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path, wandb_project=cfg.wandb_project, config=cfg
    )
    tokenizer = get_tokenizer(cfg.model_name)
    renderer = Qwen3Renderer(tokenizer)  # observer prefill is added by to_datum

    items = load_training_items(cfg)
    train_items, test_items = split_items(
        items, num_held_out_prompts=cfg.test_num_prompts, seed=cfg.seed
    )
    logger.info("SFT: %d train items, %d test items", len(train_items), len(test_items))

    def datum(item: SFTItem) -> tinker.Datum:
        return to_datum(item, tokenizer=tokenizer, renderer=renderer, max_length=cfg.max_length)

    test_data = [datum(item) for item in test_items]
    steps_per_epoch = len(train_items) // cfg.batch_size
    if steps_per_epoch == 0:
        raise ValueError(
            f"{len(train_items)} train items is less than one batch; lower batch_size "
            f"(now {cfg.batch_size}) or test_num_prompts (now {cfg.test_num_prompts})"
        )
    num_steps = steps_per_epoch * cfg.num_epochs

    def batch(step: int) -> list[tinker.Datum]:
        epoch, index = divmod(step, steps_per_epoch)
        order = epoch_order(len(train_items), seed=cfg.seed, epoch=epoch)
        selected = order[index * cfg.batch_size : (index + 1) * cfg.batch_size]
        return [datum(train_items[int(i)]) for i in selected]

    service_client = tinker.ServiceClient()
    training_client, start_step = await create_training_client(
        service_client,
        model_name=cfg.model_name,
        lora_rank=cfg.lora_rank,
        log_path=cfg.log_path,
        init_checkpoint=cfg.init_checkpoint,
    )
    opd_dataset = opd_teacher = None
    if cfg.use_opd:
        opd_dataset = build_opd_dataset(
            load_opd_prompts(cfg.opd_prompt_paths),
            tokenizer=tokenizer,
            groups_per_batch=cfg.opd_groups_per_batch,
            max_prompt_tokens=cfg.opd_max_prompt_tokens,
            num_held_out=cfg.opd_held_out_prompts,
            seed=cfg.seed,
        )
        # The teacher is the frozen base model underneath the LoRA.
        opd_teacher = service_client.create_sampling_client(base_model=cfg.model_name)

    for step in range(start_step, num_steps):
        metrics: dict[str, Any] = {"progress": step / num_steps}
        if step > 0 and step % cfg.save_every == 0:
            await save_checkpoint(
                training_client, log_path=cfg.log_path, name=f"{step:06d}", next_step=step
            )
        if step % cfg.eval_every == 0 and test_data:
            metrics |= await held_out_nll(training_client, test_data, cfg.batch_size)
        if opd_dataset is not None and opd_teacher is not None:
            student = await training_client.save_weights_and_get_sampling_client_async(
                f"opd_{step:06d}"
            )
            metrics |= await opd_step(
                training_client,
                student,
                opd_teacher,
                opd_dataset,
                step=step,
                learning_rate=cfg.learning_rate * cfg.opd_lr_multiplier,
                max_tokens=cfg.opd_max_tokens,
                kl_coef=cfg.opd_kl_coef,
            )
        metrics |= await sft_step(training_client, batch(step), cfg.learning_rate)
        ml_logger.log_metrics(metrics, step=step)

    await save_checkpoint(training_client, log_path=cfg.log_path, name="final", next_step=num_steps)
    if test_data:
        ml_logger.log_metrics(
            await held_out_nll(training_client, test_data, cfg.batch_size), step=num_steps
        )
    ml_logger.close()


def main(cfg: SFTConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(train(cfg))


if __name__ == "__main__":
    chz.nested_entrypoint(main)
