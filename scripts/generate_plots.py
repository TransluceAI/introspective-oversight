"""Plot AUROC per task, one bar per evaluated model, from eval output directories.

Each run is ``Label=<output_dir>``, where the directory was written by
``introspect.evals.prefill.evaluate`` or ``introspect.evals.reasoning.evaluate``::

    python scripts/generate_plots.py title="Prefill AUROC" output=results/plots/prefill \\
        runs="Final Observer=results/prefill,Assistant Follow-Up=results/prefill_base"

Error bars are the bootstrap 95% intervals saved by the evals. Anything that can't be drawn
(a missing run, task or interval, an undefined AUROC, a partial average) is listed in a note
under the plot, so a figure never silently drops a bar. Also writes ``<output>.md``, a table of
every value and the same notes.
"""

from __future__ import annotations

import json
import math
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

TASK_LABELS = {
    "sycophancy": "Sycophancy",
    "harm_pressure": "Harm pressure",
    "reward_hacking": "Reward hacking",
    "weirdchat": "WeirdChat",
    "value_leakage": "Value leakage",
}
# Categorical slots in fixed order (validated for color-vision deficiency on a light surface).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


@chz.chz
class PlotConfig:
    runs: list[str]  # "Label=results_dir", one bar per run in each task group
    output: str = "results/plots/auroc"  # writes <output>.png, .pdf and .md
    title: str = "AUROC"
    tasks: list[str] = chz.field(default_factory=lambda: list(TASK_LABELS))
    show_average: bool = True


@dataclass
class Bar:
    value: float | None = None  # None: no bar
    lower: float | None = None  # None: no error bar
    upper: float | None = None


@dataclass
class Run:
    label: str
    path: Path
    bars: dict[str, Bar] = field(default_factory=dict)


Note = tuple[str, str, str]  # (run label, task label or "", what is missing)


def merge_notes(notes: list[Note]) -> list[str]:
    """One line per (run, reason), listing every task it applies to."""
    merged: dict[tuple[str, str], list[str]] = {}
    for label, task, message in notes:
        merged.setdefault((label, message), []).append(task)
    lines = []
    for (label, message), tasks in merged.items():
        named = [task for task in tasks if task]
        lines.append(f"{label} / {', '.join(named)}: {message}" if named else f"{label}: {message}")
    return lines


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def load_run(label: str, path: Path, tasks: list[str], notes: list[Note]) -> Run:
    """Read one run's summary.json into bars, noting everything that is missing."""
    run = Run(label, path)
    summary_path = path / "summary.json"
    if not summary_path.exists():
        notes.append((label, "", f"no summary.json in {path}; every bar is missing."))
        return run
    summary = json.loads(summary_path.read_text())
    configuration = summary.get("configuration", {})
    if configuration.get("limit"):
        notes.append((
            label,
            "",
            f"evaluated only the first {configuration['limit']} items per question file "
            "(limit); not comparable to full results.",
        ))
    for task in tasks:
        name = TASK_LABELS.get(task, task)
        metrics = summary.get("tasks", {}).get(task)
        if metrics is None:
            notes.append((label, name, "no results; bar missing."))
            run.bars[task] = Bar()
            continue
        value = metrics.get("auroc")
        if not _finite(value):
            notes.append((label, name, "AUROC undefined (a class had no scored items); bar missing."))
            run.bars[task] = Bar()
            continue
        bar = Bar(value=float(value))
        lower, upper = metrics.get("bootstrap/ci95_lower"), metrics.get("bootstrap/ci95_upper")
        if lower is None or upper is None:
            notes.append((label, name, "no bootstrap data; bars shown without error bars."))
        elif not (_finite(lower) and _finite(upper)):
            notes.append((
                label,
                name,
                "bootstrap interval undefined (too many resamples lost a class); "
                "bars shown without error bars.",
            ))
        else:
            bar.lower, bar.upper = float(lower), float(upper)
        format_rate = metrics.get("format_valid_rate")
        if _finite(format_rate) and format_rate < 0.95:
            notes.append((
                label,
                name,
                f"only {format_rate:.0%} of answers parsed; unparsed answers are left out of "
                "the AUROC.",
            ))
        run.bars[task] = bar
    run.bars["average"] = average_bar(run, summary, tasks, notes)
    return run


def average_bar(run: Run, summary: dict[str, Any], tasks: list[str], notes: list[Note]) -> Bar:
    """The mean over the plotted tasks, with the saved interval when it covers the same tasks."""
    missing = [TASK_LABELS.get(t, t) for t in tasks if run.bars[t].value is None]
    if missing:
        notes.append((run.label, "Average", f"not shown; missing {', '.join(missing)}."))
        return Bar()
    value = sum(run.bars[t].value or 0.0 for t in tasks) / len(tasks)
    saved = summary.get("average")
    evaluated = set(summary.get("tasks", {}))
    if saved is None:
        notes.append((run.label, "Average", "no bootstrap data; bars shown without error bars."))
        return Bar(value)
    if evaluated != set(tasks) or not math.isclose(saved["auroc"], value, abs_tol=1e-9):
        notes.append((
            run.label,
            "Average",
            "the saved interval covers a different set of tasks than this plot; bar shown "
            "without an error bar.",
        ))
        return Bar(value)
    if not (_finite(saved["ci95_lower"]) and _finite(saved["ci95_upper"])):
        notes.append((run.label, "Average", "bootstrap interval undefined; no error bar."))
        return Bar(value)
    return Bar(value, saved["ci95_lower"], saved["ci95_upper"])


def plot(cfg: PlotConfig, runs: list[Run], notes: list[str]) -> Figure:
    groups = [*cfg.tasks, *(["average"] if cfg.show_average else [])]
    labels = [TASK_LABELS.get(g, "Average" if g == "average" else g) for g in groups]
    wrapped = [
        line for note in notes for line in textwrap.wrap(f"• {note}", 125, subsequent_indent="   ")
    ]
    # Fixed margins in inches: title above, tick labels and legend below, then the notes.
    notes_height = 0.18 * (len(wrapped) + 1) if wrapped else 0.0
    legend_height = 0.35 if len(runs) > 1 else 0.0
    plot_height, top_margin, bottom_margin = 3.2, 0.45, 0.45 + legend_height
    height = top_margin + plot_height + bottom_margin + notes_height
    width = max(7.0, 1.4 * len(groups) + 0.35 * len(runs))
    left_margin, right_margin = 1.05, 0.2
    fig = plt.figure(figsize=(width, height))
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes((
        left_margin / width,
        (notes_height + bottom_margin) / height,
        1 - (left_margin + right_margin) / width,
        plot_height / height,
    ))
    ax.set_facecolor(SURFACE)

    bar_width = 0.8 / max(len(runs), 1)
    for index, run in enumerate(runs):
        color = COLORS[index % len(COLORS)]
        for position, group in enumerate(groups):
            bar = run.bars.get(group, Bar())
            x = position - 0.4 + bar_width * (index + 0.5)
            if bar.value is None:  # mark the gap so it points at the notes
                ax.text(x, 0.02, "no data", rotation=90, color=MUTED, fontsize=7,
                        ha="center", va="bottom")
                continue
            ax.bar(x, bar.value, width=bar_width, color=color, edgecolor=SURFACE, linewidth=1.5)
            if bar.lower is not None and bar.upper is not None:
                ax.errorbar(
                    x, bar.value, yerr=[[bar.value - bar.lower], [bar.upper - bar.value]],
                    fmt="none", ecolor=INK, elinewidth=1.0, capsize=2.5,
                )
    ax.axhline(0.5, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=0)
    ax.set_yticks([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0], ["0", "0.2", "0.4", "0.5 (chance)", "0.6", "0.8", "1.0"])
    ax.set_xticks(range(len(groups)), labels, color=INK)
    if cfg.show_average:
        ax.axvline(len(groups) - 1.5, color=GRID, linewidth=1.0)
    ax.set_xlim(-0.55, len(groups) - 0.45)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUROC", color=INK)
    ax.set_title(cfg.title, color=INK, loc="left", fontsize=12)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    if len(runs) > 1:  # every run, in bar order, including runs with no bars at all
        handles = [
            Patch(
                color=COLORS[index % len(COLORS)],
                label=run.label
                + ("" if any(b.value is not None for b in run.bars.values()) else " (no data)"),
            )
            for index, run in enumerate(runs)
        ]
        ax.legend(handles=handles, frameon=False, ncols=min(len(runs), 4), loc="upper left",
                  bbox_to_anchor=(0, -0.1), labelcolor=INK)
    if wrapped:
        fig.text(0.02, notes_height / height, "Notes", color=INK,
                 fontsize=8.5, fontweight="bold", va="top")
        fig.text(0.02, (notes_height - 0.2) / height, "\n".join(wrapped),
                 color=MUTED, fontsize=8, va="top", linespacing=1.35)
    return fig


def table(cfg: PlotConfig, runs: list[Run], notes: list[str]) -> str:
    groups = [*cfg.tasks, *(["average"] if cfg.show_average else [])]
    header = ["Run", *(TASK_LABELS.get(g, "Average" if g == "average" else g) for g in groups)]
    lines = [f"# {cfg.title}", "", "| " + " | ".join(header) + " |",
             "|" + "---|" + "---:|" * len(groups)]
    for run in runs:
        cells = []
        for group in groups:
            bar = run.bars.get(group, Bar())
            if bar.value is None:
                cells.append("missing")
            elif bar.lower is None or bar.upper is None:
                cells.append(f"{bar.value:.3f}")
            else:
                cells.append(f"{bar.value:.3f} [{bar.lower:.3f}, {bar.upper:.3f}]")
        lines.append(f"| {run.label} | " + " | ".join(cells) + " |")
    lines += ["", "Brackets are bootstrap 95% intervals."]
    if notes:
        lines += ["", "## Notes", "", *(f"- {note}" for note in notes)]
    return "\n".join(lines) + "\n"


def main(cfg: PlotConfig) -> None:
    collected: list[Note] = []
    runs = []
    for spec in cfg.runs:
        label, separator, path = spec.partition("=")
        if not separator:
            raise ValueError(f"run {spec!r} must look like Label=results_dir")
        runs.append(load_run(label.strip(), Path(path.strip()), cfg.tasks, collected))
    notes = merge_notes(collected)
    if len(runs) > len(COLORS):
        raise ValueError(f"at most {len(COLORS)} runs fit in one plot")
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plot(cfg, runs, notes)
    for suffix in (".png", ".pdf"):
        fig.savefig(output.with_suffix(suffix), dpi=200, facecolor=SURFACE)
    output.with_suffix(".md").write_text(table(cfg, runs, notes))
    print(f"Wrote {output}.png/.pdf/.md" + (f" with {len(notes)} note(s)" if notes else ""))


if __name__ == "__main__":
    chz.nested_entrypoint(main)
