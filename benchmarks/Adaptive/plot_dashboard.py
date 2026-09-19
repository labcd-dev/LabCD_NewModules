"""Plot the finalized Adaptive benchmark: success, tokens, and cost.

Run adaptive_benchmark.py and finalize_results.py first. Success is grouped by
the artifacts' discrete complexity levels; token and cost bars are model means.
Mock mode plots only mock results and labels its synthetic token cost.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter
import numpy as np

ADAPTIVE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ADAPTIVE_DIR / "json_reports" / "results"
DEFAULT_OUTPUT = ADAPTIVE_DIR / "plots" / "adaptive_dashboard.png"

# ------------------------------ PLOT SETTINGS ------------------------------
USE_MOCK_RESULTS = False  # False plots every available folder below.
MODEL_FOLDERS_TO_PLOT = [
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-4.1-nano",
    "groq-gpt-oss-120b",
    "gpt-4o-mini",
]
SHOW_SUCCESS_VALUES = True  # Automatically hidden when grouped bars are dense.
# ---------------------------------------------------------------------------

COLORS = ["#42B7E9", "#FF9F43", "#58D6B0", "#B084F5", "#F26B8A",
          "#FFD166", "#7DCE82", "#A9B7C6", "#EF8354", "#6C91BF"]
TEXT = "#F0F6FC"
MUTED = "#A5AFBA"


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def selected_result_files() -> list[Path]:
    folders = ["mock"] if USE_MOCK_RESULTS else MODEL_FOLDERS_TO_PLOT
    if not folders or len(folders) != len(set(folders)):
        raise ValueError("Select one or more unique result folders.")
    paths = []
    for folder in folders:
        if (not isinstance(folder, str) or not folder or folder in (".", "..")
                or any(char in folder for char in '/\\:<>"|?*')):
            raise ValueError(f"Unsafe model folder name: {folder!r}")
        path = RESULTS_DIR / folder / "batch_tuning_results_brief.json"
        if path.is_file():
            paths.append(path)
        else:
            print(f"Skipping model without finalized results: {folder}")
    if not paths:
        raise FileNotFoundError(
            "No selected Adaptive brief reports exist. Run adaptive_benchmark.py "
            "and finalize_results.py first."
        )
    return paths


def load_dashboard_rows(files: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in files:
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise TypeError(
                f"{path} is a single-run walkthrough report, not a finalized "
                "benchmark array. Run finalize_results.py on a batch report "
                "(or run adaptive_benchmark.py first)."
            )
        for record in records:
            complexity = _number(record.get("total_complexity"))
            if complexity is not None and (not complexity.is_integer() or not 1 <= complexity <= 10):
                raise ValueError(f"{path}: total_complexity must be an integer from 1 to 10.")
            success = _number(record.get("success_rate"))
            if success is not None and not 0 <= success <= 1:
                raise ValueError(f"{path}: success_rate must be a fraction from 0 to 1.")
            dynamics = record.get("dynamics") or {}
            verification = record.get("verification") or {}
            tokens = record.get("token_counts") or {}
            total_tokens = (tokens.get("total") or {}).get("total_tokens")
            mse = (verification.get("tracking_mse") or {}).get("full")
            rmse = (
                [math.sqrt(float(value)) for value in mse]
                if isinstance(mse, list) and all(_number(value) is not None and value >= 0 for value in mse)
                else None
            )
            rows.append({
                "model": path.parent.name,
                "plant": record.get("plant") or dynamics.get("system_name") or "Unknown",
                "test_id": record.get("test_id"),
                "initial_condition": dynamics.get("initial_condition"),
                "references": dynamics.get("references"),
                "rmse_by_output": rmse,
                "complexity": int(complexity) if complexity is not None else None,
                "success_rate": success,
                "total_tokens": _number(total_tokens),
                "cost_usd": _number(tokens.get("cost_usd")),
                "cost_simulated": bool(tokens.get("simulated")),
            })
    if not rows:
        raise ValueError("The selected brief reports contain no benchmark runs.")
    return rows


def _model_name(folder: str) -> str:
    return {
        "mock": "Mock agents",
        "groq-gpt-oss-120b": "Groq GPT-OSS 120B",
        "gpt-4o-mini": "GPT-4o Mini",
        "gpt-4.1-nano": "GPT-4.1 Nano",
        "gpt-5.4-mini": "GPT-5.4 Mini",
        "gpt-5.4-nano": "GPT-5.4 Nano",
    }.get(folder, folder.replace("_", " "))


def _format_values(values: Any) -> str:
    if not isinstance(values, list):
        return "N/A"
    return "[" + ", ".join(
        f"{value:.4g}" if isinstance(value, (int, float)) else str(value)
        for value in values
    ) + "]"


def _details(model: str, level: int, records: list[dict[str, Any]]) -> str:
    mean = np.mean([row["success_rate"] for row in records]) * 100
    lines = [f"Model: {_model_name(model)}", f"Complexity: {level}",
             f"Mean success: {mean:.1f}% across {len(records)} run(s)"]
    for row in records[:3]:
        refs = row.get("references") or []
        targets = ", ".join(
            f"{ref.get('output')}={ref.get('expr')}" for ref in refs if isinstance(ref, dict)
        ) or "N/A"
        lines.extend([
            "",
            f"Plant: {row['plant']} ({row.get('test_id') or 'test'})",
            f"Initial state: {_format_values(row['initial_condition'])}",
            f"Output references: {targets}",
            f"Full-window RMSE: {_format_values(row['rmse_by_output'])}",
        ])
    if len(records) > 3:
        lines.append(f"\n+ {len(records) - 3} other run(s) in this mean")
    return "\n".join(lines)


def _style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor("#161B22")
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#8B949E", linewidth=0.8, alpha=0.20)
    axis.tick_params(colors="#D0D7DE", labelsize=9.5, length=0, pad=7)
    for spine in axis.spines.values():
        spine.set_color("#484F58")


def _draw_success(
    axis: plt.Axes, grouped: list[tuple[str, list[dict[str, Any]]]],
) -> dict[Any, tuple[str, int, list[dict[str, Any]]]]:
    levels = sorted({row["complexity"] for _, entries in grouped for row in entries
                     if row["complexity"] is not None})
    axis.set_title("Success Rate vs Complexity", fontsize=14, weight="bold", pad=13)
    axis.set_xlabel("Manual task complexity (1 = easy, 10 = hard)", labelpad=11)
    axis.set_ylabel("Mean success rate", labelpad=10)
    axis.set_ylim(0, 106)
    axis.set_yticks(np.arange(0, 101, 20))
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    if not levels:
        axis.text(0.5, 0.5, "No assigned complexity levels", transform=axis.transAxes,
                  ha="center", va="center", color=MUTED)
        axis.set_xticks([])
        return {}

    positions = np.arange(len(levels), dtype=float)
    width = min(0.72, 0.82 / len(grouped))
    show_labels = SHOW_SUCCESS_VALUES and len(grouped) * len(levels) <= 12 and width >= 0.16
    selectable = {}
    for model_index, (model, entries) in enumerate(grouped):
        offset = (model_index - (len(grouped) - 1) / 2) * width
        for x, level in zip(positions, levels):
            matches = [row for row in entries if row["complexity"] == level
                       and row["success_rate"] is not None]
            if not matches:
                continue  # Missing model/level data is never plotted as zero.
            value = float(np.mean([row["success_rate"] for row in matches])) * 100
            bar = axis.bar(x + offset, value, width=width * 0.88,
                           color=COLORS[model_index % len(COLORS)],
                           edgecolor=TEXT, linewidth=0.7, alpha=0.92,
                           zorder=3, picker=True)[0]
            selectable[bar] = (model, level, matches)
            if show_labels:
                inside = value > 82
                axis.annotate(f"{value:.0f}%", xy=(x + offset, value),
                              xytext=(0, -5 if inside else 5), textcoords="offset points",
                              ha="center", va="top" if inside else "bottom",
                              fontsize=8, weight="bold",
                              color="#0D1117" if inside else TEXT)
    axis.set_xticks(positions, [str(level) for level in levels])
    axis.set_xlim(-0.6, len(levels) - 0.4)
    return selectable


def _draw_mean(
    axis: plt.Axes, grouped: list[tuple[str, list[dict[str, Any]]]],
    field: str, formatter,
) -> None:
    means = []
    for _, entries in grouped:
        values = [row[field] for row in entries if row[field] is not None]
        means.append(float(np.mean(values)) if values else None)
    maximum = max((mean for mean in means if mean is not None), default=0)
    axis.set_ylim(0, maximum * 1.25 if maximum > 0 else 1)
    for index, mean in enumerate(means):
        if mean is None:
            axis.text(index, axis.get_ylim()[1] * 0.03, "N/A", ha="center",
                      va="bottom", color=MUTED, fontsize=9)
            continue
        axis.bar(index, mean, width=0.62, color=COLORS[index % len(COLORS)],
                 edgecolor=TEXT, linewidth=0.8, alpha=0.92, zorder=3)
        axis.text(index, mean + axis.get_ylim()[1] * 0.025, formatter(mean),
                  ha="center", va="bottom", color=TEXT, fontsize=9, weight="bold")
    names = [_model_name(model) for model, _ in grouped]
    axis.set_xticks(range(len(names)), names)
    if len(names) > 3:
        plt.setp(axis.get_xticklabels(), rotation=24, ha="right", rotation_mode="anchor")
    axis.set_xlim(-0.65, max(len(names) - 0.35, 0.65))


def create_dashboard(
    rows: list[dict[str, Any]], output: Path = DEFAULT_OUTPUT, show_plot: bool = True,
) -> Path:
    if not rows:
        raise ValueError("No finalized Adaptive benchmark rows to plot.")
    grouped = [(model, [row for row in rows if row["model"] == model])
               for model in dict.fromkeys(row["model"] for row in rows)]
    plt.style.use("dark_background")
    figure, axes = plt.subplots(1, 3, figsize=(23, 8),
                                gridspec_kw={"width_ratios": [1.7, 1, 1]})
    figure.patch.set_facecolor("#0D1117")
    figure.subplots_adjust(left=0.06, right=0.98, bottom=0.23, top=0.76, wspace=0.33)
    for axis in axes:
        _style_axis(axis)

    selectable = _draw_success(axes[0], grouped)
    annotation = axes[0].annotate(
        "", xy=(0, 0), xytext=(18, 18), textcoords="offset points",
        fontsize=9, color=TEXT, ha="left", va="bottom",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#21262D",
              "edgecolor": "#58A6FF", "alpha": 0.97},
        arrowprops={"arrowstyle": "->", "color": "#58A6FF"},
        zorder=10, annotation_clip=False,
    )
    annotation.set_visible(False)

    def on_pick(event) -> None:
        selected = selectable.get(event.artist)
        if selected is None:
            return
        model, level, matching = selected
        bar = event.artist
        center = bar.get_x() + bar.get_width() / 2
        annotation.xy = (center, bar.get_height())
        right = center > sum(axes[0].get_xlim()) / 2
        high = bar.get_height() > 70
        annotation.set_position((-18 if right else 18, -18 if high else 18))
        annotation.set_ha("right" if right else "left")
        annotation.set_va("top" if high else "bottom")
        annotation.set_text(_details(model, level, matching))
        annotation.set_visible(True)
        figure.canvas.draw_idle()

    figure.canvas.mpl_connect("pick_event", on_pick)

    _draw_mean(axes[1], grouped, "total_tokens", lambda value: f"{value:,.0f}")
    axes[1].set_title("Mean Token Usage", fontsize=14, weight="bold", pad=13)
    axes[1].set_xlabel("LLM model", labelpad=10)
    axes[1].set_ylabel("Mean total tokens per run", labelpad=10)
    axes[1].yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    axes[1].yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value / 1000:g}k" if value >= 1000 else f"{value:g}"))

    _draw_mean(axes[2], grouped, "cost_usd", lambda value: f"${value:,.4f}")
    axes[2].set_title("Mean Cost per Run", fontsize=14, weight="bold", pad=13)
    axes[2].set_xlabel("LLM model", labelpad=10)
    axes[2].set_ylabel("Mean estimated cost (USD)", labelpad=10)
    axes[2].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[2].yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"${value:.4f}" if abs(value) < 0.1 else f"${value:.2f}"))

    figure.legend(
        handles=[Patch(facecolor=COLORS[index % len(COLORS)], edgecolor=TEXT,
                       label=_model_name(model)) for index, (model, _) in enumerate(grouped)],
        loc="upper center", ncol=min(len(grouped), 6), frameon=False, fontsize=10,
        bbox_to_anchor=(0.5, 0.855), columnspacing=1.5,
    )
    figure.suptitle("ADAPTIVE TUNING BENCHMARK", x=0.055, y=0.97,
                    ha="left", fontsize=21, weight="bold", color=TEXT)
    figure.text(0.055, 0.91,
                f"AI-agent controller tuning · {len(rows)} runs · {len(grouped)} model(s)",
                color=MUTED, fontsize=11, ha="left")
    has_mock = any(row["cost_simulated"] for row in rows)
    has_real = any(not row["cost_simulated"] for row in rows)
    if has_mock and has_real:
        footer = "Mock costs are synthetic; real costs are local estimates. "
    elif has_mock:
        footer = "Mock token costs are synthetic; no API was billed. "
    else:
        footer = "Real-model costs are local price-table estimates. "
    figure.text(0.055, 0.04,
                footer + "Click a success bar to inspect its tests; N/A means unavailable data.",
                color=MUTED, fontsize=9, ha="left")

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, facecolor=figure.get_facecolor(),
                   bbox_inches="tight", pad_inches=0.18)
    if show_plot:
        plt.show()
    else:
        plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the Adaptive benchmark dashboard.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-show", action="store_true",
                        help="Save without opening an interactive plot window.")
    args = parser.parse_args()
    files = selected_result_files()
    rows = load_dashboard_rows(files)
    output = create_dashboard(rows, output=args.output, show_plot=not args.no_show)
    print(f"Loaded {len(rows)} runs from {len(files)} model report(s).")
    print(f"Dashboard written to: {output}")


if __name__ == "__main__":
    main()
