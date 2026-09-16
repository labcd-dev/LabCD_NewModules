"""Create an MPC dashboard using manually assigned discrete complexity levels.

The dashboard contains:

1. Grouped success-rate bars: one group per complexity and one bar per LLM.
2. Mean total token usage for each selected LLM model.
3. Mean estimated cost for each selected LLM model.

In real-model mode, every folder listed in ``MODEL_FOLDERS_TO_PLOT`` is read.
In mock mode, that array is ignored and only the mock result is plotted.
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

MPC_DIR = Path(__file__).resolve().parent
RESULTS_DIR = MPC_DIR / "json_reports" / "results"
PLOTS_DIR = MPC_DIR / "plots"
DEFAULT_OUTPUT_FILE = PLOTS_DIR / "mpc_dashboard.png"

# ------------------------- DASHBOARD SETTINGS -------------------------
# True: ignore MODEL_FOLDERS_TO_PLOT and plot only results/mock.
# False: plot every result folder named in MODEL_FOLDERS_TO_PLOT.
USE_MOCK_RESULTS = False
MODEL_FOLDERS_TO_PLOT = [
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-4o-mini",
    "gpt-4.1-nano",
    # "groq-gpt-oss-120b",
]
# True displays mean success percentages, positioned to avoid label overlap.
SHOW_SUCCESS_VALUES = True
# ---------------------------------------------------------------

COLORS = [
    "#42B7E9",
    "#FF9F43",
    "#58D6B0",
    "#B084F5",
    "#F26B8A",
    "#FFD166",
    "#7DCE82",
    "#A9B7C6",
    "#EF8354",
    "#6C91BF",
]


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _model_label(record: dict[str, Any], source_path: Path) -> str:
    """Read the stable model label from the result folder or legacy data."""
    if source_path.parent.parent == RESULTS_DIR:
        return source_path.parent.name

    agent_mode = record.get("agent_mode")
    if isinstance(agent_mode, str) and agent_mode:
        return agent_mode

    agent = record.get("agent") or {}
    if isinstance(agent, dict) and agent.get("folder_name"):
        return str(agent["folder_name"])

    token_usage = (record.get("tuning") or {}).get("token_usage") or {}
    per_model = token_usage.get("per_model") or {}
    if isinstance(per_model, dict) and per_model:
        return " + ".join(sorted(str(name) for name in per_model))
    return source_path.stem


def _total_complexity(record: dict[str, Any]) -> float | None:
    return _number(record.get("total_complexity"))


def _success_rate(record: dict[str, Any]) -> float | None:
    return _number(record.get("success_rate"))


def load_dashboard_rows(result_files: list[Path]) -> list[dict[str, Any]]:
    """Load compact finalized results, without reading complexity components."""
    rows = []
    for result_file in result_files:
        records = json.loads(result_file.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise TypeError(f"{result_file} must contain a JSON array of runs.")
        for record in records:
            usage = (record.get("tuning") or {}).get("token_usage") or {}
            metrics = (record.get("verification") or {}).get("metrics") or {}
            complexity = _total_complexity(record)
            if complexity is not None and (not complexity.is_integer() or not 1 <= complexity <= 10):
                raise ValueError(f"{result_file}: complexity must be an integer from 1 to 10.")
            rows.append({
                "model": _model_label(record, result_file),
                "dynamic_model": (record.get("model") or {}).get("name", "Unknown"),
                "initial_state": record.get("initial_condition"),
                "target": record.get("target"),
                "rmse": _number(metrics.get("rmse")),
                "complexity": complexity,
                "success_rate": _success_rate(record),
                "total_tokens": _number(usage.get("total_tokens")),
                "cost_usd": _number(usage.get("cost_usd")),
            })
    return rows


def _format_vector(value: Any) -> str:
    if not isinstance(value, (list, tuple)):
        return "N/A"
    values = []
    for item in value:
        number = _number(item)
        values.append(f"{number:.4g}" if number is not None else str(item))
    return "[" + ", ".join(values) + "]"


def _point_details(row: dict[str, Any]) -> str:
    rmse = _number(row.get("rmse"))
    complexity = _number(row.get("complexity"))
    return "\n".join([
        f"Dynamic model: {row.get('dynamic_model') or 'Unknown'}",
        f"Initial state: {_format_vector(row.get('initial_state'))}",
        f"Target: {_format_vector(row.get('target'))}",
        f"RMSE: {rmse:.6g}" if rmse is not None else "RMSE: N/A",
        f"Total complexity: {complexity:g}" if complexity is not None else "Total complexity: not assigned",
    ])


def _group_by_model(
    rows: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    # Preserve the order selected in MODEL_FOLDERS_TO_PLOT.
    models = list(dict.fromkeys(str(row["model"]) for row in rows))
    return [(model, [row for row in rows if row["model"] == model]) for model in models]


def _display_model_name(model: str) -> str:
    known_names = {
        "mock": "Mock agents",
        "gpt-5.4-mini": "GPT-5.4 Mini",
        "gpt-5.4-nano": "GPT-5.4 Nano",
        "gpt-4o-mini": "GPT-4o Mini",
        "gpt-4.1-nano": "GPT-4.1 Nano",
        "groq-gpt-oss-120b": "Groq · GPT-OSS 120B",
    }
    return known_names.get(model, model.replace("_", " "))


def default_result_files() -> list[Path]:
    """Resolve the mock report or all reports selected in the model array."""
    selected_folders = ["mock"] if USE_MOCK_RESULTS else MODEL_FOLDERS_TO_PLOT
    if not selected_folders:
        raise ValueError(
            "MODEL_FOLDERS_TO_PLOT must contain at least one folder when "
            "USE_MOCK_RESULTS=False."
        )

    result_files = [
        RESULTS_DIR / folder_name / "batch_tuning_results_full.json"
        for folder_name in selected_folders
    ]
    missing = [path for path in result_files if not path.is_file()]
    for path in missing:
        print(f"Skipping model without finalized results: {path.parent.name}")
    existing = [path for path in result_files if path.is_file()]
    if not existing:
        raise FileNotFoundError(
            "No selected finalized results exist. Run mpc_benchmark.py and "
            "finalize_results.py first, then check the dashboard model settings."
        )
    return existing


def _finite_values(model_rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in model_rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number):
                values.append(number)
    return values


def _style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor("#161B22")
    axis.set_axisbelow(True)
    axis.grid(True, axis="y", color="#8B949E", alpha=0.18, linewidth=0.8)
    axis.tick_params(colors="#D0D7DE", labelsize=9.5, length=0, pad=7)
    for spine in axis.spines.values():
        spine.set_color("#484F58")
        spine.set_linewidth(0.9)


def _draw_success_bars(
    axis: plt.Axes,
    grouped_rows: list[tuple[str, list[dict[str, Any]]]],
) -> dict[Any, tuple[str, int, list[dict[str, Any]]]]:
    """Draw side-by-side model means without treating missing results as zero."""
    levels = sorted({
        int(row["complexity"])
        for _model, model_rows in grouped_rows
        for row in model_rows if row.get("complexity") is not None
    })
    axis.set_title("Success Rate vs Complexity", fontsize=14, weight="bold", pad=13)
    axis.set_xlabel("Manual task complexity (1 = easy, 10 = hard)", labelpad=11)
    axis.set_ylabel("Mean success rate", labelpad=10)
    # Headroom keeps labels on short bars clear without adding ticks above 100%.
    axis.set_ylim(0.0, 108.0)
    axis.set_yticks(np.arange(0.0, 101.0, 20.0))
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))
    selectable = {}
    if not levels:
        axis.set_xticks([])
        axis.text(
            0.5, 0.5,
            "No assigned complexity levels yet.\nFill artifact total_complexity and finalize again.",
            transform=axis.transAxes, ha="center", va="center",
            color="#8B949E", fontsize=10,
        )
        return selectable

    positions = np.arange(len(levels), dtype=float)
    width = 0.82 / len(grouped_rows)
    # Decide from available display width, not just the number of bars.
    bar_pixels = axis.get_window_extent().width / (len(levels) + 0.2) * width * 0.9
    label_pixels = 44.0 * axis.figure.dpi / 100.0
    crowded = bar_pixels < label_pixels
    for model_index, (model, model_rows) in enumerate(grouped_rows):
        offset = (model_index - (len(grouped_rows) - 1) / 2) * width
        for position, level in zip(positions, levels):
            matching = [
                row for row in model_rows
                if row.get("complexity") == level and row.get("success_rate") is not None
            ]
            x_position = position + offset
            if not matching:
                axis.text(
                    x_position, 2.0, "N/A", ha="center", va="bottom",
                    color="#8B949E", fontsize=8, rotation=90 if crowded else 0,
                )
                continue
            mean_success = float(np.mean([row["success_rate"] for row in matching])) * 100.0
            bar = axis.bar(
                x_position, mean_success, width=width * 0.9,
                color=COLORS[model_index % len(COLORS)],
                edgecolor="#F0F6FC", linewidth=0.7, alpha=0.92, zorder=3,
                picker=True,
            )[0]
            selectable[bar] = (model, level, matching)
            if SHOW_SUCCESS_VALUES:
                inside = mean_success >= (25.0 if crowded else 85.0)
                axis.annotate(
                    f"{mean_success:.1f}%", xy=(x_position, mean_success),
                    xytext=(0, -6 if inside else 5), textcoords="offset points",
                    ha="center", va="top" if inside else "bottom",
                    fontsize=8.5, weight="bold",
                    color="#0D1117" if inside else "#F0F6FC",
                    rotation=90 if crowded else 0,
                )

    axis.set_xticks(positions)
    axis.set_xticklabels([str(level) for level in levels])
    axis.set_xlim(-0.6, len(levels) - 0.4)
    return selectable


def _bar_details(model: str, level: int, rows: list[dict[str, Any]]) -> str:
    mean_success = float(np.mean([row["success_rate"] for row in rows])) * 100.0
    details = [
        f"LLM: {_display_model_name(model)}",
        f"Complexity level: {level}",
        f"Mean success: {mean_success:.1f}% ({len(rows)} verified run(s))",
    ]
    for row in rows[:3]:
        details.extend(["", _point_details(row)])
    if len(rows) > 3:
        details.append(f"\n+ {len(rows) - 3} additional run(s) included in this mean.")
    return "\n".join(details)


def _draw_mean_bars(
    axis: plt.Axes,
    grouped_rows: list[tuple[str, list[dict[str, Any]]]],
    key: str,
    value_formatter,
) -> None:
    positions = np.arange(len(grouped_rows), dtype=float)
    means: list[float | None] = []

    for _model, model_rows in grouped_rows:
        values = _finite_values(model_rows, key)
        means.append(float(np.mean(values)) if values else None)

    valid_positions = [
        position for position, mean in zip(positions, means) if mean is not None
    ]
    valid_means = [mean for mean in means if mean is not None]
    valid_colors = [
        COLORS[index % len(COLORS)]
        for index, mean in enumerate(means)
        if mean is not None
    ]

    if valid_means:
        axis.bar(
            valid_positions,
            valid_means,
            width=0.62,
            color=valid_colors,
            edgecolor="#F0F6FC",
            linewidth=0.9,
            alpha=0.92,
            zorder=3,
        )
        upper_limit = max(valid_means) * 1.22
        if upper_limit <= 0.0:
            upper_limit = 1.0
        axis.set_ylim(0.0, upper_limit)
    else:
        axis.set_ylim(0.0, 1.0)

    label_height = axis.get_ylim()[1] * 0.025
    for position, mean in zip(positions, means):
        if mean is None:
            axis.text(
                position,
                label_height,
                "N/A",
                ha="center",
                va="bottom",
                color="#8B949E",
                fontsize=9,
            )
            continue
        axis.text(
            position,
            mean + label_height,
            value_formatter(mean),
            ha="center",
            va="bottom",
            color="#F0F6FC",
            fontsize=9,
            weight="bold",
        )

    axis.set_xticks(positions)
    axis.set_xticklabels(
        [
            _display_model_name(model)
            for model, _rows in grouped_rows
        ],
        rotation=0 if len(grouped_rows) <= 3 else 22,
        ha="center" if len(grouped_rows) <= 3 else "right",
        rotation_mode="anchor",
    )
    axis.set_xlim(-0.65, max(len(grouped_rows) - 0.35, 0.65))


def create_dashboard(
    rows: list[dict[str, Any]],
    output_file: Path = DEFAULT_OUTPUT_FILE,
    show_plot: bool = True,
) -> Path:
    if not rows:
        raise ValueError("No dashboard records were found.")

    grouped_rows = _group_by_model(rows)
    plt.style.use("dark_background")
    # The grouped chart needs more room than the single-bar-per-model panels.
    figure, axes = plt.subplots(
        1, 3, figsize=(23, 8), gridspec_kw={"width_ratios": [1.7, 1, 1]},
    )
    figure.patch.set_facecolor("#0D1117")
    figure.subplots_adjust(
        left=0.06,
        right=0.98,
        bottom=0.23,
        top=0.76,
        wspace=0.32,
    )

    for axis in axes:
        _style_axis(axis)

    # Panel 1: comparable LLM bars within each discrete complexity group.
    selectable_bars = _draw_success_bars(axes[0], grouped_rows)
    annotation = axes[0].annotate(
        "", xy=(0.0, 0.0), xytext=(18, 18), textcoords="offset points",
        ha="left", va="bottom", fontsize=9, color="#F0F6FC",
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#21262D",
              "edgecolor": "#58A6FF", "alpha": 0.97},
        arrowprops={"arrowstyle": "->", "color": "#58A6FF", "linewidth": 1.0},
        zorder=10, annotation_clip=False,
    )
    annotation.set_visible(False)

    def show_selected_bar(event) -> None:
        selected = selectable_bars.get(event.artist)
        if selected is None:
            return
        model, level, matching = selected
        bar = event.artist
        center = bar.get_x() + bar.get_width() / 2
        high_bar = bar.get_height() > 70.0
        right_side = center > sum(axes[0].get_xlim()) / 2
        annotation.xy = (center, bar.get_height())
        annotation.set_position((-18 if right_side else 18, -18 if high_bar else 18))
        annotation.set_ha("right" if right_side else "left")
        annotation.set_va("top" if high_bar else "bottom")
        annotation.set_text(_bar_details(model, level, matching))
        annotation.set_visible(True)
        figure.canvas.draw_idle()

    figure.canvas.mpl_connect("pick_event", show_selected_bar)

    # Panels 2 and 3: one aggregate bar per selected model.
    _draw_mean_bars(
        axes[1],
        grouped_rows,
        key="total_tokens",
        value_formatter=lambda value: f"{value:,.0f}",
    )
    axes[1].set_title("Mean Token Usage", fontsize=14, weight="bold", pad=13)
    axes[1].set_xlabel("LLM model", labelpad=10)
    axes[1].set_ylabel("Mean total tokens per run", labelpad=10)
    axes[1].yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    axes[1].yaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"{value / 1000:.0f}k" if value else "0")
    )

    _draw_mean_bars(
        axes[2],
        grouped_rows,
        key="cost_usd",
        value_formatter=lambda value: f"${value:,.4f}",
    )
    axes[2].set_title("Mean Cost per Run", fontsize=14, weight="bold", pad=13)
    axes[2].set_xlabel("LLM model", labelpad=10)
    axes[2].set_ylabel("Mean estimated cost (USD)", labelpad=10)
    axes[2].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[2].yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _position: (
                f"${value:.4f}" if abs(value) < 0.01 else f"${value:.3f}"
            )
        )
    )

    legend_handles = [
        Patch(
            facecolor=COLORS[index % len(COLORS)],
            edgecolor="#F0F6FC",
            label=_display_model_name(model),
        )
        for index, (model, _model_rows) in enumerate(grouped_rows)
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=min(len(grouped_rows), 6),
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.855),
        handletextpad=0.55,
        columnspacing=1.7,
    )

    figure.suptitle(
        "MPC TUNING BENCHMARK",
        x=0.055,
        y=0.97,
        ha="left",
        fontsize=21,
        weight="bold",
        color="#F0F6FC",
    )
    figure.text(
        0.055,
        0.91,
        f"AI-agent controller tuning · {len(rows)} benchmark runs · "
        f"{len(grouped_rows)} model{'s' if len(grouped_rows) != 1 else ''}",
        ha="left",
        fontsize=11,
        color="#8B949E",
    )
    figure.text(
        0.985,
        0.91,
        "Grouped bars compare LLM means at the same complexity level.",
        ha="right",
        fontsize=9,
        color="#8B949E",
    )
    figure.text(
        0.055, 0.04,
        "Click a success bar to inspect its runs. N/A indicates missing verification scores.",
        ha="left", fontsize=9, color="#8B949E",
    )

    output_file = output_file.resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_file,
        dpi=240,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.18,
    )
    if show_plot:
        plt.show()
    else:
        plt.close(figure)
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the MPC benchmark dashboard."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the dashboard without opening an interactive plot window.",
    )
    args = parser.parse_args()

    result_files = default_result_files()

    rows = load_dashboard_rows(result_files)
    output = create_dashboard(
        rows,
        output_file=args.output,
        show_plot=not args.no_show,
    )
    print(f"Loaded {len(rows)} runs from {len(result_files)} result files.")
    print(f"Aggregated {len(_group_by_model(rows))} models.")
    print(f"Dashboard written to: {output}")


if __name__ == "__main__":
    main()
