"""Export CHIVE counterfactual experiments as investigations for generate_rl.py.

Run CHIVE first (https://github.com/adamkarvonen/chive; see chive/pipeline/README.md), including
its rate-question builders, which select the verified experiments and write an outcome-blind
description of each edit. Each builder writes a ``questions.jsonl`` with one row per
experiment; this script groups those rows back into one ``Investigation`` per prompt
(see introspect/data/train/generate_rl.py).

The paper used these two CHIVE outputs, 5,073 experiments from 656 investigations:

    python scripts/export_chive_investigations.py output=data/rl/investigations.jsonl \\
        paths=$CHIVE/data/runs/counterfactual_rate_questions_balanced_combined_4012/questions.jsonl,$CHIVE/data/runs/counterfactual_rate_questions_high_verification_unassessed_1061/questions.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

import chz


@chz.chz
class ExportConfig:
    paths: list[str]  # CHIVE rate-question files (questions.jsonl)
    output: str


def main(cfg: ExportConfig) -> None:
    investigations: dict[str, dict] = {}
    for path in cfg.paths:
        # Split bytes, not str: str.splitlines also breaks on separators inside JSON strings.
        for line in Path(path).read_bytes().splitlines():
            row = json.loads(line)
            investigation = {
                "investigation_id": row["prompt_id"],
                "messages": row["original_messages"],
                "response": row["original_chive_response"],
                "behavior": row["behavior_criterion"],
                "baseline_count": row["baseline_count"],
                # CHIVE's rate questions assume 10 rollouts. One paper experiment
                # (prompt_02784_unassessed_exp05) had only 9 graded and was still scored out of 10.
                "num_rollouts": 10,
            }
            existing = investigations.setdefault(row["prompt_id"], {**investigation, "experiments": []})
            if any(existing[key] != value for key, value in investigation.items()):
                raise ValueError(f"rows for {row['prompt_id']} disagree about the original prompt")
            existing["experiments"].append(
                {
                    "experiment_id": row["claim_id"],
                    "edit": row["intervention"],
                    "behavior_count": row["counterfactual_count"],
                    "num_rollouts": 10,
                }
            )
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as file:
        for investigation in investigations.values():
            file.write(json.dumps(investigation, ensure_ascii=False) + "\n")
    num_experiments = sum(len(i["experiments"]) for i in investigations.values())
    print(f"wrote {len(investigations)} investigations ({num_experiments} experiments) to {output}")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
