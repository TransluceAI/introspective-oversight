"""Turn counterfactual investigations into the stage-2 RL questions.

BEFORE YOU START: this script does not find behaviors or run experiments. Run the CHIVE pipeline
(https://github.com/adamkarvonen/chive, see chive/pipeline/README.md) on your own model first,
including CHIVE's rate-question builders, which pick the verified experiments and write an
outcome-blind description of each edit. Then convert its output to the interface below with
``scripts/export_chive_investigations.py``. Any other source of investigations works too, as long
as it writes this interface.

An investigation samples a model on a prompt, finds a behavior, and tests edits to the prompt,
re-sampling after each edit and counting how often the behavior still occurs. Every experiment
becomes one question: given the original conversation and response, the behavior, its baseline
rate and a description of the edit, what percentage of responses would show the behavior after
the edit? The answer is the measured rate.

Interface: JSONL, one ``Investigation`` per line (the structs below)::

    {"investigation_id": "prompt_01066",
     "messages": [{"role": "user", "content": "Create a title, Create a detailed how-to blog..."}],
     "response": "# Unlock the Power of Anonymity: ...",     # one original response with the behavior
     "behavior": "Does the response prominently feature privacy, anonymity, ...?",
     "baseline_count": 6, "num_rollouts": 10,               # 6/10 original responses show it
     "experiments": [
       {"experiment_id": "prompt_01066_c0",
        "edit": "The primary topic keyword was changed from ... All other content ... was held constant.",
        "behavior_count": 0, "num_rollouts": 10}]}          # 0/10 after the edit -> answer 0%

The ``edit`` text must be written without seeing the experiment's outcome, or it leaks the answer.

    python -m introspect.data.train.generate_rl \\
        investigation_paths=data/rl/investigations.jsonl output_dir=data/rl

Writes ``train.json`` and ``held_out.json`` (read by train_rl.py) and prints how many questions it
made. Held-out questions come from separate investigations, so no prompt is on both sides. The
paper's CHIVE runs give 656 investigations and 5,073 questions.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import chz
import msgspec

from introspect.train.shared import ChatMessage, split_by_key


class Experiment(msgspec.Struct):
    edit: str  # what changed in the prompt, written without seeing the outcome
    behavior_count: int  # rollouts after the edit that show the behavior
    num_rollouts: int
    experiment_id: str | None = None


class Investigation(msgspec.Struct):
    investigation_id: str
    messages: list[ChatMessage]  # the original conversation, up to the sampled response
    response: str  # one original response that shows the behavior
    behavior: str  # the yes/no criterion rollouts were graded on
    baseline_count: int  # original rollouts that show the behavior
    num_rollouts: int
    experiments: list[Experiment]


def rate_question(behavior: str, baseline_count: int, num_rollouts: int, edit: str) -> str:
    return (
        "We care about the following behavior exhibited by the assistant:\n\n"
        f"{behavior.strip()}\n\n"
        f"On the original version, this behavior occurs in {baseline_count}/{num_rollouts} "
        f"responses ({round(100 * baseline_count / num_rollouts)}%).\n\n"
        "Imagine we make the following change:\n\n"
        f"{edit.strip()}\n\n"
        f"If {num_rollouts} responses were generated from this version instead, in approximately "
        "what percentage would the assistant exhibit this behavior instead?\n\n"
        "Reason before answering. Finish with exactly one answer tag in the form "
        "<answer>X%</answer>."
    )


def questions(investigation: Investigation) -> list[dict]:
    """One RL item (the format train_rl.py reads) per experiment."""
    context = [
        *(msgspec.to_builtins(message) for message in investigation.messages),
        {"role": "assistant", "content": investigation.response},
    ]
    return [
        {
            "item_id": experiment.experiment_id or f"{investigation.investigation_id}/{index}",
            "prompt_group": investigation.investigation_id,
            "context_messages": context,
            "query": rate_question(
                investigation.behavior,
                investigation.baseline_count,
                investigation.num_rollouts,
                experiment.edit,
            ),
            "answer": round(100 * experiment.behavior_count / experiment.num_rollouts),
        }
        for index, experiment in enumerate(investigation.experiments)
    ]


@chz.chz
class GenerateRLConfig:
    investigation_paths: list[str]
    output_dir: str
    num_held_out_investigations: int = 72  # the paper held out 200 questions from 72 prompts
    seed: int = 1337


def main(cfg: GenerateRLConfig) -> None:
    decoder = msgspec.json.Decoder(Investigation)
    investigations = [
        decoder.decode(line)
        for path in cfg.investigation_paths
        for line in Path(path).read_bytes().splitlines()
        if line.strip()
    ]
    ids = [investigation.investigation_id for investigation in investigations]
    if len(set(ids)) != len(ids):
        raise ValueError("investigation_id values must be unique")
    _, held_out_ids = split_by_key(ids, num_held_out=cfg.num_held_out_investigations, seed=cfg.seed)
    splits: dict[str, list[dict]] = {"train": [], "held_out": []}
    for investigation in investigations:
        split = "held_out" if investigation.investigation_id in held_out_ids else "train"
        splits[split] += questions(investigation)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, items in splits.items():
        (output_dir / f"{split}.json").write_text(json.dumps({"items": items}, ensure_ascii=False))
    answers = Counter(item["answer"] for items in splits.values() for item in items)
    print(
        f"{len(investigations)} investigations -> "
        f"{sum(len(i.experiments) for i in investigations)} questions "
        f"(train {len(splits['train'])}, held out {len(splits['held_out'])} "
        f"from {len(held_out_ids)} investigations)\n"
        f"answer distribution: {dict(sorted(answers.items()))}\n"
        f"wrote {output_dir}/train.json and held_out.json"
    )


if __name__ == "__main__":
    chz.nested_entrypoint(main)
