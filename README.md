# Training Models for Introspective Oversight

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Observer--Qwen3.6--27B-yellow)](https://huggingface.co/Transluce/Observer-Qwen3.6-27B)

This repository contains the code for the paper **"Training Models for Introspective Oversight"**.

We train a model to answer oversight questions about its own behavior in a separate **observer** role: given a conversation, the observer is asked a **query** (for example, "did the assistant lie?") and answers it. Training has two stages, and this repo lets you run both on your own model, then evaluate the result.

## Quick Start

### Prerequisites

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- A [Tinker](https://thinkingmachines.ai/tinker/) API key. All training and sampling runs on Tinker, so you do not need a local GPU.
- An Anthropic API key (generates the stage-1 training data) and a Gemini API key (the stage-2 answer-extraction judge). Both models are configurable; OpenRouter also works.

### Installation

```bash
git clone https://github.com/TransluceAI/introspective-oversight.git
cd introspective-oversight
uv sync
```

### Environment Setup

```bash
export TINKER_API_KEY=...
export ANTHROPIC_API_KEY=...   # generate_sft.py (generator_backend=anthropic)
export GEMINI_API_KEY=...      # train_rl.py (judge_backend=gemini)
```

## Pipeline Overview

| Stage | What it trains | Data | Script |
|---|---|---|---|
| 1. SFT | Observer answers questions about the assistant's responses | Generated from public prompts | `introspect/train/train_sft.py` |
| 2. RL | Observer predicts how often a behavior occurs after an edit to the prompt | Counterfactual experiments from [CHIVE](https://github.com/adamkarvonen/chive) | `introspect/train/train_rl.py` |

Defaults reproduce the paper run: `Qwen/Qwen3.6-27B` with a rank-32 LoRA. Every script takes `key=value` arguments; pass `--help` to list them.

**Pre-trained model:** the final stage-2 RL checkpoint from the paper is on Hugging Face at [Transluce/Observer-Qwen3.6-27B](https://huggingface.co/Transluce/Observer-Qwen3.6-27B). See the model card for how to load it.

---

## Stage 1: SFT

The observer learns to describe the assistant's behavior. Each step is interleaved with one on-policy distillation (OPD) step toward the base model, so that training the observer does not degrade the assistant.

### Data

Download the prompts, then generate the four SFT tasks in order:

```bash
# Prompts: Tulu 3 for OPD, a 50k Dolci sample for SFT generation
uv run python scripts/prepare_prompts.py source=tulu3 output=data/prompts/tulu3_opd.jsonl
uv run python scripts/prepare_prompts.py source=dolci num_prompts=50000 output=data/prompts/dolci_50k.jsonl

# 1. Predict properties of the assistant's response from the prompt alone
uv run python -m introspect.data.train.generate_sft task=behavior_prediction \
    prompts_path=data/prompts/dolci_50k.jsonl output_dir=data/sft/behavior_prediction
# 2-3. Write a catalog of traits, then sample responses with each trait as a hidden system prompt
uv run python -m introspect.data.train.generate_sft task=trait_catalog output_dir=data/sft/traits
uv run python -m introspect.data.train.generate_sft task=hidden_trait \
    prompts_path=data/prompts/dolci_50k.jsonl \
    trait_catalog_path=data/sft/traits/traits.json output_dir=data/sft/hidden_trait
# 4. Mix the two in equal parts
uv run python -m introspect.data.train.generate_sft task=mix \
    mix_paths=data/sft/behavior_prediction/sft.json,data/sft/hidden_trait/sft.json \
    output_dir=data/sft/mix
```

Generation is resumable: rerun the same command to continue an interrupted run.

To train on your own data instead, write a file of conversations that end in an `observer` turn. Only that turn is trained:

```json
{"items": [{"messages": [
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."},
  {"role": "query", "content": "Did the assistant ...?"},
  {"role": "observer", "content": "..."}
]}]}
```

### Training

```bash
uv run python -m introspect.train.train_sft \
    train_paths=data/sft/mix/sft.json \
    opd_prompt_paths=data/prompts/tulu3_opd.jsonl \
    log_path=runs/stage1_sft
```

Useful options: `model_name`, `lora_rank`, `learning_rate`, `batch_size`, `num_epochs`, `use_opd=false`, `wandb_project`. Checkpoints are written to `log_path`, and rerunning the command resumes from the last one.

---

## Stage 2: RL

The observer sees a conversation, a behavior, and a description of an edit to the prompt, and predicts what percentage of responses would show the behavior after the edit. It reasons, then answers `<answer>X%</answer>`. The reward is the Brier score against the measured rate.

### Data

This stage needs counterfactual experiments on your own model. Run the [CHIVE pipeline](https://github.com/adamkarvonen/chive) first, including its rate-question builders, then convert its output:

```bash
uv run python scripts/export_chive_investigations.py \
    paths=/PATH/TO/CHIVE/questions.jsonl output=data/rl/investigations.jsonl
uv run python -m introspect.data.train.generate_rl \
    investigation_paths=data/rl/investigations.jsonl output_dir=data/rl
```

This writes `data/rl/train.json` and `data/rl/held_out.json`, split so that no prompt is on both sides. Any other source of experiments works if it writes the investigation format documented at the top of `introspect/data/train/generate_rl.py`. The edit descriptions must be written without seeing the outcome, or they leak the answer.

### Training

```bash
uv run python -m introspect.train.train_rl \
    train_paths=data/rl/train.json held_out_path=data/rl/held_out.json \
    init_checkpoint=runs/stage1_sft log_path=runs/stage2_rl
```

`init_checkpoint` is the stage-1 log directory or a `tinker://` URI. The paper's extra optimizations each have a `use_*` flag (`use_kl_penalty`, `use_group_normalization`, `use_easy_group_filter`, `use_judge_fallback`, `use_format_bonus`), all on by default.

---

## Evaluation

Two suites measure AUROC on five oversight tasks (sycophancy, harm pressure, reward hacking, WeirdChat, value leakage). **Prefill** reads the observer's probability of each label directly; **reasoning** lets it think and state a percentage.

```bash
uv run python -m introspect.evals.prefill.evaluate checkpoint=runs/stage2_rl output_dir=results/prefill
uv run python -m introspect.evals.reasoning.evaluate checkpoint=runs/stage2_rl output_dir=results/reasoning
```

**Baselines** on the untrained base model:

```bash
# Ask the assistant in a follow-up turn
uv run python -m introspect.evals.prefill.evaluate role=assistant output_dir=results/prefill_base
# Quote the transcript and question inside one user message
uv run python -m introspect.evals.prefill.evaluate role=assistant single_user_message=true \
    output_dir=results/prefill_self_monitor
```

Each run writes `<task>.json` and `summary.json`, with bootstrap 95% intervals. Use `tasks=sycophancy,weirdchat` to run a subset and `limit=20` for a smoke test. Questions are read from `data/evals/questions/`.

To plot runs side by side:

```bash
uv run python scripts/generate_plots.py title="Prefill AUROC" output=results/plots/prefill \
    runs="Final Observer=results/prefill,Assistant Follow-Up=results/prefill_base"
```

## Layout

```
introspect/
  data/        samplers for API models, and the SFT / RL data generators
  train/       train_sft.py (stage 1), train_rl.py (stage 2), shared observer renderers and OPD
  evals/       prefill/ and reasoning/ suites; shared.py holds the run loop
scripts/       prompt download, CHIVE export, plotting
```

## Citation

<!-- TODO: add BibTeX once the paper is public -->
