"""Create a presentation-ready dashboard for enriched MPC benchmark results.

Reads ``batch_tuning_results_full.json`` (mock folder by default) and draws
three comparison plots, each marker being a single benchmark run:

  1. success rate  vs  total complexity
  2. total tokens  vs  total complexity
  3. success rate  vs  cost (USD)

Series are colored by LLM model. The legacy pre-migration record layout is also
supported so old result files continue to work.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator, PercentFormatter
import numpy as np


MPC_DIR = Path(__file__).resolve().parent
RESULTS_DIR = MPC_DIR / "json_reports" / "results"
PLOTS_DIR = MPC_DIR / "plots"
DEFAULT_OUTPUT_FILE = PLOTS_DIR / "mpc_scatter_dashboard.png"

# ------------------------- USER INPUTS -------------------------
# True: plot json_reports/results/mock/batch_tuning_results_full.json.
# False: plot every model folder named in DASHBOARD_LLM_MODEL_FOLDERS.
USE_MOCK_RESULTS = False
DASHBOARD_LLM_MODEL_FOLDERS = [
    "gpt-4o-mini",
    "groq-gpt-oss-120b",
]
# ---------------------------------------------------------------

def _model_label(record: dict[str, Any], source_path: Path) -> str:
    # With the new folder layout, the directory name is the stable dashboard
    # label selected by the user in the benchmark configuration.
    if source_path.parent.parent == RESULTS_DIR:
        return source_path.parent.name

    # agent_mode is "mock" or the LLM model name.
    agent_mode = record.get("agent_mode")
    if isinstance(agent_mode, str) and agent_mode:
        return agent_mode

    # Pre-migration records nested the label under "agent".
    agent = record.get("agent") or {}
    if agent.get("folder_name"):
        return str(agent["folder_name"])
    if isinstance(agent_mode, dict) and agent_mode.get("folder_name"):
        return str(agent_mode["folder_name"])

    token_usage = (record.get("tuning") or {}).get("token_usage") or {}
    per_model = token_usage.get("per_model") or {}
    if isinstance(per_model, dict) and per_model:
        return " + ".join(sorted(str(name) for name in per_model))

    return source_path.stem


def _total_complexity(record: dict[str, Any]) -> float | None:
    """The 1-10 total complexity, from the current or pre-migration layout."""
    current = record.get("total_complexity") or {}
    score = _number(current.get("score_1_to_10"))
    if score is not None:
        return score
    legacy = (record.get("complexity") or {}).get("total") or {}
    return _number(legacy.get("score_1_to_10"))


def _success_rate(record: dict[str, Any]) -> float | None:
    """The success rate, from the current top level or inside verification."""
    score = _number(record.get("success_rate"))
    if score is not None:
        return score
    verification = record.get("verification") or {}
    return _number(verification.get("success_rate"))


def _test_label(record: dict[str, Any]) -> str | None:
    test_id = record.get("test_id")
    if isinstance(test_id, str) and test_id:
        return test_id
    legacy_scenario = record.get("scenario") or {}
    if isinstance(legacy_scenario, dict) and "level" in legacy_scenario:
        level = legacy_scenario.get("level")
        if level is not None:
            return f"level {level}"
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def load_dashboard_rows(result_files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result_file in result_files:
        records = json.loads(result_file.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise TypeError(f"{result_file} must contain a JSON array of runs.")

        for record in records:
            complexity = _total_complexity(record)
            success_rate = _success_rate(record)
            token_usage = (record.get("tuning") or {}).get("token_usage") or {}
            total_tokens = _number(token_usage.get("total_tokens"))
            cost_usd = _number(token_usage.get("cost_usd"))

            # Every dot needs the x and y it is plotted on; missing x/y pairs
            # are skipped per panel rather than failing the whole dashboard.
            rows.append(
                {
                    "model": _model_label(record, result_file),
                    "test": _test_label(record),
                    "complexity": complexity,
                    "success_rate": success_rate,
                    "total_tokens": total_tokens,
                    "cost_usd": cost_usd,
                    "source": result_file.name,
                }
            )
    return rows


def _group_by_model(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    models = sorted({str(row["model"]) for row in rows})
    return [(model, [row for row in rows if row["model"] == model]) for model in models]


def _display_model_name(model: str) -> str:
    """Return compact, customer-facing labels for known result folders."""
    known_names = {
        "mock": "Mock agents",
        "gpt-4o-mini": "GPT-4o Mini",
        "groq-gpt-oss-120b": "Groq · GPT-OSS 120B",
    }
    return known_names.get(model, model.replace("_", " "))


def create_dashboard(
    rows: list[dict[str, Any]],
    output_file: Path = DEFAULT_OUTPUT_FILE,
    show_plot: bool = True,
) -> Path:
    if not rows:
        raise ValueError("No dashboard records were found.")

    models = sorted({str(row["model"]) for row in rows})
    professional_colors = [
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
    colors = [
        professional_colors[index % len(professional_colors)]
        for index in range(max(len(models), 1))
    ]
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]

    plt.style.use("dark_background")
    figure, axes = plt.subplots(1, 3, figsize=(18, 7.0))
    figure.patch.set_facecolor("#0D1117")
    figure.subplots_adjust(
        left=0.055,
        right=0.985,
        bottom=0.125,
        top=0.735,
        wspace=0.28,
    )

    for axis in axes:
        axis.set_facecolor("#161B22")
        axis.set_box_aspect(1.0)
        axis.set_axisbelow(True)
        axis.grid(True, color="#8B949E", alpha=0.20, linewidth=0.8)
        axis.tick_params(colors="#D0D7DE", labelsize=10, length=0, pad=6)
        for spine in axis.spines.values():
            spine.set_color("#484F58")
            spine.set_linewidth(0.9)

    def plot_model_points(
        axis,
        x_getter,
        y_getter,
        *,
        y_scale: float = 1.0,
        connect_by_complexity: bool = False,
    ) -> None:
        """Draw one marker per run, using one consistent style per LLM."""

        for index, (model, model_rows) in enumerate(_group_by_model(rows)):
            points: list[tuple[float, float]] = []
            for row in model_rows:
                x_value = x_getter(row)
                y_value = y_getter(row)
                if x_value is None or y_value is None:
                    continue
                points.append((float(x_value), float(y_value) * y_scale))
            if not points:
                continue

            if connect_by_complexity:
                points.sort(key=lambda point: point[0])
                if len(points) > 1:
                    axis.plot(
                        [point[0] for point in points],
                        [point[1] for point in points],
                        color=colors[index],
                        linewidth=2.0,
                        alpha=0.55,
                        zorder=2,
                    )

            axis.scatter(
                [point[0] for point in points],
                [point[1] for point in points],
                color=colors[index],
                marker=markers[index % len(markers)],
                s=72,
                edgecolor="#F0F6FC",
                linewidth=1.0,
                alpha=1.0,
                zorder=3,
            )

    # 1. Success rate vs total complexity.
    plot_model_points(
        axes[0],
        x_getter=lambda row: row["complexity"],
        y_getter=lambda row: row["success_rate"],
        y_scale=100.0,
        connect_by_complexity=True,
    )
    axes[0].set_title("Performance by Complexity", fontsize=14, weight="bold", pad=13)
    axes[0].set_xlabel("Task complexity  (1 = easy, 10 = hard)", labelpad=10)
    axes[0].set_ylabel("Success score", labelpad=10)

    # 2. Total tokens vs total complexity.
    plot_model_points(
        axes[1],
        x_getter=lambda row: row["complexity"],
        y_getter=lambda row: row["total_tokens"],
        connect_by_complexity=True,
    )
    axes[1].set_title("Token Usage by Complexity", fontsize=14, weight="bold", pad=13)
    axes[1].set_xlabel("Task complexity  (1 = easy, 10 = hard)", labelpad=10)
    axes[1].set_ylabel("Total tokens per run", labelpad=10)

    # 3. Success rate vs cost.
    plot_model_points(
        axes[2],
        x_getter=lambda row: row["cost_usd"],
        y_getter=lambda row: row["success_rate"],
        y_scale=100.0,
    )
    axes[2].set_title("Performance vs Cost", fontsize=14, weight="bold", pad=13)
    axes[2].set_xlabel("Estimated cost per run (USD)", labelpad=10)
    axes[2].set_ylabel("Success score", labelpad=10)

    # Consistent scales make the panels directly comparable and prevent
    # auto-zoom from exaggerating small differences.
    for axis in axes[:2]:
        axis.set_xlim(1.0, 10.0)
        axis.set_xticks(np.arange(1, 11, 1))

    for axis in (axes[0], axes[2]):
        axis.set_ylim(0.0, 100.0)
        axis.set_yticks(np.arange(0.0, 101.0, 20.0))
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))

    token_values = [
        float(row["total_tokens"])
        for row in rows
        if row["total_tokens"] is not None
    ]
    if token_values:
        axes[1].set_ylim(0.0, max(token_values) * 1.10)
    axes[1].yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    axes[1].yaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"{value / 1000:.0f}k" if value else "0")
    )

    cost_values = [
        float(row["cost_usd"])
        for row in rows
        if row["cost_usd"] is not None
    ]
    if cost_values:
        cost_upper_limit = math.ceil(max(cost_values) * 1.10 * 1000.0) / 1000.0
        axes[2].set_xlim(0.0, cost_upper_limit)
    axes[2].xaxis.set_major_locator(MultipleLocator(0.001))
    axes[2].xaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"${value:.3f}")
    )

    # A single shared legend: one entry per plant/model.
    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color=colors[index],
            marker=markers[index % len(markers)],
            linestyle="None",
            markersize=8,
            label=_display_model_name(model),
        )
        for index, model in enumerate(models)
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=min(len(models), 6),
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.845),
        handletextpad=0.55,
        columnspacing=1.7,
    )

    n_points = len(rows)
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
        f"AI-agent controller tuning · {n_points} benchmark runs · "
        f"{len(models)} model{'s' if len(models) != 1 else ''}",
        ha="left",
        fontsize=11,
        color="#8B949E",
    )
    figure.text(
        0.985,
        0.045,
        "Each marker represents one plant/scenario test.",
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


def default_result_files() -> list[Path]:
    selected_folders = ["mock"] if USE_MOCK_RESULTS else DASHBOARD_LLM_MODEL_FOLDERS
    if not selected_folders:
        raise ValueError(
            "DASHBOARD_LLM_MODEL_FOLDERS must contain at least one folder "
            "when USE_MOCK_RESULTS=False."
        )

    result_files = [
        RESULTS_DIR / folder_name / "batch_tuning_results_full.json"
        for folder_name in selected_folders
    ]
    missing = [path for path in result_files if not path.is_file()]
    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            f"Selected dashboard result files do not exist:\n{missing_text}"
        )
    return result_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the MPC benchmark scatter dashboard (one point per run)."
    )
    parser.add_argument(
        "results_json",
        nargs="*",
        type=Path,
        help="Enriched *_full.json files. If omitted, the mock/list settings in this file are used.",
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
    output = create_dashboard(rows, args.output, show_plot=not args.no_show)
    success_rows = sum(row["success_rate"] is not None for row in rows)
    token_rows = sum(row["total_tokens"] is not None for row in rows)
    cost_rows = sum(row["cost_usd"] is not None for row in rows)
    print(f"Loaded {len(rows)} runs from {len(result_files)} result files.")
    print(
        f"Panel points -> success rate: {success_rows}, "
        f"tokens: {token_rows}, cost: {cost_rows}."
    )
    print(f"Dashboard written to: {output}")


if __name__ == "__main__":
    main()
