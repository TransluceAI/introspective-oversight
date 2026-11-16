"""Pieces shared by train_sft.py and train_rl.py: observer renderers, checkpoints, and OPD.

The trained model answers in a custom ``observer`` role. Stock Qwen renderers only frame the
``assistant`` role, so the renderers here add Qwen's thinking / non-thinking marker to the
observer header themselves.

OPD (on-policy distillation) keeps ordinary assistant behavior intact while training the
observer: the student samples assistant responses to plain prompts and is pushed toward the
frozen base model with a per-token reverse-KL advantage.
"""

from __future__ import annotations

import asyncio
import logging
import random
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import msgspec
import tinker
import torch
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.distillation.datasets import PromptOnlyDataset
from tinker_cookbook.renderers.base import (
    Message,
    ParseTermination,
    RenderContext,
    RenderedMessage,
    TextPart,
)
from tinker_cookbook.renderers.qwen3 import Qwen3DisableThinkingRenderer, Qwen3Renderer
from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages
from tinker_cookbook.rl.metrics import discounted_future_sum_vectorized
from tinker_cookbook.rl.rollouts import do_group_rollout_and_filter_constant_reward
from tinker_cookbook.rl.train import train_step
from tinker_cookbook.rl.types import RLDataset, TrajectoryGroup
from tinker_cookbook.tokenizer_utils import Tokenizer

logger = logging.getLogger(__name__)

OBSERVER_ROLE = "observer"
THINKING_PREFILL = "<think>\n"
NON_THINKING_PREFILL = "<think>\n\n</think>\n\n"


class ChatMessage(msgspec.Struct):
    role: str  # user | assistant | query | observer
    content: str


# --- Observer-role renderers ---------------------------------------------------------------


def _with_header_prefill(
    rendered: RenderedMessage, tokenizer: Tokenizer, prefill: str
) -> RenderedMessage:
    header_tokens = list(rendered.header.tokens) if rendered.header else []
    prefill_tokens = tokenizer.encode(prefill, add_special_tokens=False)
    return RenderedMessage(
        header=tinker.EncodedTextChunk(tokens=header_tokens + prefill_tokens),
        output=rendered.output,
        stop_overlap=rendered.stop_overlap,
    )


class QwenNonThinkingObserverRenderer(Qwen3DisableThinkingRenderer):
    """Generate the observer role with Qwen's empty thinking block already closed."""

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        rendered = super().render_message(message, ctx)
        if message["role"] == OBSERVER_ROLE and ctx.is_last:
            rendered = _with_header_prefill(rendered, self.tokenizer, NON_THINKING_PREFILL)
        return rendered


class QwenThinkingObserverRenderer(Qwen3Renderer):
    """Generate the observer role with an open thinking block, and parse it back out."""

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        if message["role"] == OBSERVER_ROLE and ctx.is_last:
            # The header owns the opening marker, so strip it from a stored target.
            content = message["content"]
            if isinstance(content, str) and content.startswith("<think>"):
                message = {**message, "content": content[len("<think>") :].lstrip("\n")}
            elif isinstance(content, list) and content and content[0]["type"] == "thinking":
                text = content[0]["thinking"].lstrip("\n") + "</think>"
                message = {**message, "content": [TextPart(type="text", text=text), *content[1:]]}
        rendered = super().render_message(message, ctx)
        if message["role"] == OBSERVER_ROLE and ctx.is_last:
            rendered = _with_header_prefill(rendered, self.tokenizer, THINKING_PREFILL)
        return rendered

    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
        """Restore the masked prefill so Qwen parses reasoning into structured content."""
        prefill_tokens = self.tokenizer.encode(THINKING_PREFILL, add_special_tokens=False)
        return super().parse_response([*prefill_tokens, *response])


def observer_renderer(tokenizer: Tokenizer, *, thinking: bool) -> Qwen3Renderer:
    """Return the Qwen renderer that generates in the observer role."""
    if thinking:
        return QwenThinkingObserverRenderer(tokenizer)
    return QwenNonThinkingObserverRenderer(tokenizer)


# --- Checkpoints -----------------------------------------------------------------------------


def resolve_checkpoint(path: str, *, kind: str = "state_path") -> str:
    """Resolve a ``tinker://`` URI as-is, or a previous run's log dir to its last checkpoint.

    ``kind`` is ``state_path`` (weights, for training) or ``sampler_path`` (for sampling).
    """
    if path.startswith("tinker://"):
        return path
    if not Path(path, "checkpoints.jsonl").exists():
        raise ValueError(f"{path} is neither a tinker:// URI nor a log dir with checkpoints.jsonl")
    record = checkpoint_utils.get_last_checkpoint(path, required_key=kind)
    if record is None:
        raise ValueError(f"No checkpoint with {kind} in {path}")
    resolved = getattr(record, kind)
    assert isinstance(resolved, str)
    return resolved


async def create_training_client(
    service_client: tinker.ServiceClient,
    *,
    model_name: str,
    lora_rank: int,
    log_path: str,
    init_checkpoint: str | None,
) -> tuple[tinker.TrainingClient, int]:
    """Resume this run if it has checkpoints, else start from init_checkpoint or the base model.

    Returns the client and the step to start from.
    """
    resume = checkpoint_utils.get_last_checkpoint(log_path)
    if resume is not None:
        assert resume.state_path is not None
        logger.info("Resuming from %s", resume.state_path)
        client = await service_client.create_training_client_from_state_with_optimizer_async(
            resume.state_path
        )
        return client, int(resume.extra["step"])
    if init_checkpoint:
        state_path = resolve_checkpoint(init_checkpoint)
        logger.info("Initializing weights from %s", state_path)
        client = await service_client.create_training_client_from_state_async(state_path)
        return client, 0
    client = await service_client.create_lora_training_client_async(
        base_model=model_name, rank=lora_rank
    )
    return client, 0


async def save_checkpoint(
    training_client: tinker.TrainingClient, *, log_path: str, name: str, next_step: int
) -> None:
    """Save weights + optimizer state and a sampler snapshot, recording where to resume."""
    await checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name=name,
        log_path=log_path,
        loop_state={"step": next_step},
        kind="both",
    )


def split_by_key(
    keys: list[str], *, num_held_out: int, seed: int
) -> tuple[set[str], set[str]]:
    """Deterministically split unique keys into (train, held-out) by a seeded hash."""
    unique = sorted(set(keys), key=lambda key: sha256(f"{seed}\0{key}".encode()).digest())
    if num_held_out >= len(unique):
        raise ValueError(f"num_held_out={num_held_out} must be below {len(unique)} unique keys")
    return set(unique[num_held_out:]), set(unique[:num_held_out])


# --- OPD -------------------------------------------------------------------------------------


class _OPDRow(msgspec.Struct):
    """One prompt row. Accepts ``text``, ``prompt``, or chat ``messages``."""

    text: str | None = None
    prompt: str | None = None
    messages: list[dict[str, Any]] | None = None

    def user_prompt(self) -> str:
        for value in (self.text, self.prompt):
            if value and value.strip():
                return value.strip()
        for message in self.messages or []:
            if message.get("role") == "user" and str(message.get("content", "")).strip():
                return str(message["content"]).strip()
        raise ValueError("OPD row has no text, prompt, or user message")


def load_opd_prompts(paths: list[str]) -> list[str]:
    """Read JSONL prompt files."""
    decoder = msgspec.json.Decoder(_OPDRow)
    prompts = [
        decoder.decode(line).user_prompt()
        for path in paths
        for line in Path(path).read_bytes().splitlines()
        if line.strip()
    ]
    if not prompts:
        raise ValueError("OPD prompt files contain no prompts")
    return prompts


def opd_train_prompts(prompts: list[str], *, num_held_out: int, seed: int) -> list[str]:
    """Drop ``num_held_out`` unique prompts, then shuffle the rest."""
    if num_held_out:
        train_set, _ = split_by_key(prompts, num_held_out=num_held_out, seed=seed)
        prompts = [prompt for prompt in prompts if prompt in train_set]
    else:
        prompts = list(prompts)
    random.Random(seed).shuffle(prompts)
    return prompts


def build_opd_dataset(
    prompts: list[str],
    *,
    tokenizer: Tokenizer,
    groups_per_batch: int,
    max_prompt_tokens: int,
    num_held_out: int,
    seed: int,
) -> RLDataset:
    """Plain user -> assistant prompts; OPD never uses the observer role.

    ``num_held_out`` unique prompts are removed first (the paper run reserved them for a
    held-out KL eval), then the rest are shuffled.
    """
    return PromptOnlyDataset(
        prompts=opd_train_prompts(prompts, num_held_out=num_held_out, seed=seed),
        batch_size=groups_per_batch,
        group_size=1,
        renderer=Qwen3DisableThinkingRenderer(tokenizer),
        tokenizer=tokenizer,
        max_prompt_tokens=max_prompt_tokens,
        convo_prefix=None,
        dataset_name="opd",
    )


async def opd_step(
    training_client: tinker.TrainingClient,
    student: tinker.SamplingClient,
    teacher: tinker.SamplingClient,
    dataset: RLDataset,
    *,
    step: int,
    learning_rate: float,
    max_tokens: int,
    kl_coef: float,
    kl_discount: float = 0.0,
) -> dict[str, float]:
    """Sample one prompt batch from the student and train on -kl_coef * (log p_s - log p_t)."""
    builders = dataset.get_batch(step % len(dataset))
    groups = await asyncio.gather(
        *(
            do_group_rollout_and_filter_constant_reward(
                student,
                builder,
                max_tokens=max_tokens,
                temperature=1.0,
                # Prompt-only envs have zero task reward; keep them, the KL is the signal.
                do_remove_constant_reward_groups=False,
            )
            for builder in builders
        )
    )
    trajectory_groups = [cast(TrajectoryGroup, group) for group in groups if group is not None]
    data, _ = assemble_training_data(trajectory_groups, compute_advantages(trajectory_groups))
    if not data:
        return {"opd/skipped_empty_batch": 1.0}

    sequences = [
        datum.model_input.append_int(cast(int, datum.loss_fn_inputs["target_tokens"].data[-1]))
        for datum in data
    ]
    teacher_logprobs = await asyncio.gather(
        *(teacher.compute_logprobs_async(sequence) for sequence in sequences)
    )
    kl_sum, num_tokens = 0.0, 0.0
    for datum, sequence_logprobs in zip(data, teacher_logprobs, strict=True):
        student_logprobs = datum.loss_fn_inputs["logprobs"].to_torch()
        mask = datum.loss_fn_inputs["mask"].to_torch().float()
        teacher_targets = torch.tensor(sequence_logprobs[1:], dtype=student_logprobs.dtype)
        reverse_kl = (student_logprobs - teacher_targets) * mask
        advantage = -kl_coef * reverse_kl
        if kl_discount > 0:
            advantage = discounted_future_sum_vectorized(advantage, kl_discount)
        datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(
            datum.loss_fn_inputs["advantages"].to_torch() + advantage
        )
        kl_sum += float(reverse_kl.sum())
        num_tokens += float(mask.sum())

    await train_step(
        data_D=data,
        training_client=training_client,
        learning_rate=learning_rate,
        num_substeps=1,
        loss_fn="importance_sampling",
    )
    return {"opd/teacher_kl": kl_sum / max(num_tokens, 1.0), "opd/num_tokens": num_tokens}
