"""Create the revised MPC benchmark dashboard.

The dashboard contains:

1. Success score versus task complexity, with one marker per run.
2. Mean total token usage for each selected LLM model.
3. Mean estimated cost for each selected LLM model.

Result-file selection and JSON compatibility are shared with
``plot_dashboard.py`` so both dashboards read the same benchmark data.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator, PercentFormatter
import numpy as np

from plot_dashboard import (
    PLOTS_DIR,
    RESULTS_DIR,
    _display_model_name,
    _group_by_model,
    default_result_files,
    load_dashboard_rows,
)


DEFAULT_OUTPUT_FILE = PLOTS_DIR / "mpc_dashboard_revised.png"

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
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]


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
    axis.set_box_aspect(1.0)
    axis.set_axisbelow(True)
    axis.grid(True, axis="y", color="#8B949E", alpha=0.20, linewidth=0.8)
    axis.tick_params(colors="#D0D7DE", labelsize=10, length=0, pad=6)
    for spine in axis.spines.values():
        spine.set_color("#484F58")
        spine.set_linewidth(0.9)


def _draw_mean_bars(
    axis: plt.Axes,
    grouped_rows: list[tuple[str, list[dict[str, Any]]]],
    key: str,
    value_formatter,
) -> None:
    positions = np.arange(len(grouped_rows), dtype=float)
    means: list[float | None] = []
    counts: list[int] = []

    for _model, model_rows in grouped_rows:
        values = _finite_values(model_rows, key)
        means.append(float(np.mean(values)) if values else None)
        counts.append(len(values))

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
            fontsize=10,
            weight="bold",
        )

    axis.set_xticks(positions)
    axis.set_xticklabels(
        [
            f"{_display_model_name(model)}\n(n={count})"
            for (model, _rows), count in zip(grouped_rows, counts)
        ],
        rotation=0 if len(grouped_rows) <= 3 else 15,
        ha="center" if len(grouped_rows) <= 3 else "right",
    )
    axis.set_xlim(-0.65, max(len(grouped_rows) - 0.35, 0.65))


def create_revised_dashboard(
    rows: list[dict[str, Any]],
    output_file: Path = DEFAULT_OUTPUT_FILE,
    show_plot: bool = True,
) -> Path:
    if not rows:
        raise ValueError("No dashboard records were found.")

    grouped_rows = _group_by_model(rows)
    plt.style.use("dark_background")
    figure, axes = plt.subplots(1, 3, figsize=(18, 7.0))
    figure.patch.set_facecolor("#0D1117")
    figure.subplots_adjust(
        left=0.055,
        right=0.985,
        bottom=0.145,
        top=0.735,
        wspace=0.28,
    )

    for axis in axes:
        _style_axis(axis)

    # Panel 1: retain every verification run and connect each model's points
    # in ascending complexity order.
    axes[0].grid(True, axis="both", color="#8B949E", alpha=0.20, linewidth=0.8)
    for index, (model, model_rows) in enumerate(grouped_rows):
        points = sorted(
            (
                (float(row["complexity"]), float(row["success_rate"]) * 100.0)
                for row in model_rows
                if row.get("complexity") is not None
                and row.get("success_rate") is not None
            ),
            key=lambda point: point[0],
        )
        if not points:
            continue

        color = COLORS[index % len(COLORS)]
        if len(points) > 1:
            axes[0].plot(
                [point[0] for point in points],
                [point[1] for point in points],
                color=color,
                linewidth=2.0,
                alpha=0.58,
                zorder=2,
            )
        axes[0].scatter(
            [point[0] for point in points],
            [point[1] for point in points],
            color=color,
            marker=MARKERS[index % len(MARKERS)],
            s=72,
            edgecolor="#F0F6FC",
            linewidth=1.0,
            zorder=3,
        )

    axes[0].set_title("Performance by Complexity", fontsize=14, weight="bold", pad=13)
    axes[0].set_xlabel("Task complexity  (1 = easy, 10 = hard)", labelpad=10)
    axes[0].set_ylabel("Success score", labelpad=10)
    axes[0].set_xlim(1.0, 10.0)
    axes[0].set_xticks(np.arange(1, 11, 1))
    axes[0].set_ylim(0.0, 100.0)
    axes[0].set_yticks(np.arange(0.0, 101.0, 20.0))
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))

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
    axes[1].yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
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
    axes[2].yaxis.set_major_locator(MultipleLocator(0.001))
    axes[2].yaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"${value:.3f}")
    )

    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color=COLORS[index % len(COLORS)],
            marker=MARKERS[index % len(MARKERS)],
            linewidth=2.0,
            markersize=7,
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
        bbox_to_anchor=(0.5, 0.845),
        handletextpad=0.55,
        columnspacing=1.7,
    )

    figure.suptitle(
        "MPC TUNING BENCHMARK",
        x=0.055,
        y=0.965,
        ha="left",
        fontsize=21,
        weight="bold",
        color="#F0F6FC",
    )
    figure.text(
        0.055,
        0.905,
        f"AI-agent controller tuning · {len(rows)} benchmark runs · "
        f"{len(grouped_rows)} model{'s' if len(grouped_rows) != 1 else ''}",
        ha="left",
        fontsize=11,
        color="#8B949E",
    )
    figure.text(
        0.985,
        0.905,
        "Bars show the arithmetic mean across valid runs; n is the included run count.",
        ha="right",
        fontsize=9,
        color="#8B949E",
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
        description="Create the revised MPC benchmark dashboard."
    )
    parser.add_argument(
        "results_json",
        nargs="*",
        type=Path,
        help=(
            "Enriched *_full.json files. If omitted, the mock/model-folder "
            "settings in plot_dashboard.py are used."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the dashboard without opening an interactive plot window.",
    )
    args = parser.parse_args()

    result_files = [path.resolve() for path in args.results_json]
    if not result_files:
        result_files = default_result_files()
    if not result_files:
        raise FileNotFoundError(f"No benchmark result files found in {RESULTS_DIR}.")

    rows = load_dashboard_rows(result_files)
    output = create_revised_dashboard(
        rows,
        output_file=args.output,
        show_plot=not args.no_show,
    )
    print(f"Loaded {len(rows)} runs from {len(result_files)} result files.")
    print(f"Aggregated {len(_group_by_model(rows))} models.")
    print(f"Revised dashboard written to: {output}")


if __name__ == "__main__":
    main()
