"""Download the public user-prompt sets the pipeline samples from, as JSONL ``{"user_prompt": ...}`` rows.

    # OPD user prompts for train_sft.py (and train_rl.py use_opd): all 896,050 single-turn Tulu 3 ones
    python scripts/prepare_prompts.py source=tulu3 output=data/prompts/tulu3_opd.jsonl
    # User prompts for generate_sft.py: a seeded sample of single-turn Dolci prompts
    python scripts/prepare_prompts.py source=dolci num_user_prompts=50000 output=data/prompts/dolci_50k.jsonl

Both keep conversations that are exactly one user turn and one assistant reply, and keep only the
user prompt. Tulu 3 is pinned to the revision the paper used, and its prompts keep their dataset
order, which reproduces the paper's OPD prompt file exactly. The paper's 50k Dolci sample was
drawn by an unrecorded procedure, so this draws a fresh seeded one from Dolci pinned to its
2026-02-03 revision, the latest as of mid-August 2026.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, cast

import chz
from datasets import load_dataset


# hash for prompt version. 
SOURCES = {
    "tulu3": ("allenai/tulu-3-sft-mixture", "b14afda60f1bbebe55d5d2fa1e4df5042f97f8be"),
    "dolci": ("allenai/Dolci-Instruct-SFT-7B", "bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221"),
}


@chz.chz
class PrepareConfig:
    source: Literal["tulu3", "dolci"]
    output: str
    num_user_prompts: int = 0  # 0 keeps every user prompt; otherwise a seeded random sample
    seed: int = 42

# we only include single-turn prompts for SFT + OPD training.
def single_turn_user_prompts(rows: Iterable[dict[str, Any]]) -> list[str]:
    """User prompts of one-turn conversations, in dataset order."""
    user_prompts = []
    for row in rows:
        messages = row["messages"]
        if [m["role"] for m in messages] == ["user", "assistant"] and messages[0]["content"].strip():
            user_prompts.append(messages[0]["content"].strip())
    return user_prompts


def main(cfg: PrepareConfig) -> None:
    name, revision = SOURCES[cfg.source]
    dataset = load_dataset(name, revision=revision, split="train").select_columns(["messages"])
    user_prompts = single_turn_user_prompts(cast(Iterable[dict[str, Any]], dataset))  # row dicts
    if cfg.num_user_prompts:
        user_prompts = random.Random(cfg.seed).sample(user_prompts, cfg.num_user_prompts)
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as file:
        for user_prompt in user_prompts:
            file.write(json.dumps({"user_prompt": user_prompt}, ensure_ascii=False) + "\n")
    print(f"wrote {len(user_prompts)} {cfg.source} user prompts to {output}")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
