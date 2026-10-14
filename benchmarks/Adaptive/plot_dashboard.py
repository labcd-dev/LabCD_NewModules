"""Plot the finalized Adaptive benchmark: success, tokens, and cost.

Run adaptive_benchmark.py and finalize_results.py first. Two dashboards compare
Success Rate 1 (normalized MSE) and Rate 2 (fixed-target pass fraction).
Success has one LLM-bar
group per plant/test, ordered and spaced by discrete complexity.
Token and cost bars are model means.
Mock mode plots only mock results and labels its synthetic token cost.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
import sys
import textwrap
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter
import numpy as np

try:
    from .finalize_results import report_run_seed, result_directories
except ImportError:  # Direct script execution.
    from finalize_results import report_run_seed, result_directories

ADAPTIVE_DIR = Path(__file__).resolve().parent
REPO_ROOT = ADAPTIVE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = ADAPTIVE_DIR / "json_reports" / "results"
RESULT_FILENAME = "batch_tuning_results.json"  # Historical batch fallback only.
DEFAULT_OUTPUT = ADAPTIVE_DIR / "plots" / "dashboard" / "adaptive_dashboard.png"
SCORE_FIELDS = ("success_rate_1", "success_rate_2")

# ------------------------------ PLOT SETTINGS ------------------------------
USE_MOCK_RESULTS = False  # False plots every available folder below.
MODEL_FOLDERS_TO_PLOT = [
    "groq-gpt-oss-120b",
    "gpt-4o-mini",
    "gpt-4.1-nano",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
]
SHOW_SUCCESS_VALUES = True  # Automatically hidden when grouped bars are dense.
SEEDS_TO_PLOT: list[int] | None = None  # None: mean over all saved seeds.
CASE_GAP = 0.22  # Space between case groups at the same complexity.
COMPLEXITY_GAP = 0.80  # Larger space between different complexity levels.
PLANT_LABEL_ROTATION = 45  # Match MPC: readable angled names for dense groups.
# ---------------------------------------------------------------------------

COLORS = ["#42B7E9", "#FF9F43", "#58D6B0", "#B084F5", "#F26B8A",
          "#FFD166", "#7DCE82", "#A9B7C6", "#EF8354", "#6C91BF"]
TEXT = "#F0F6FC"
MUTED = "#A5AFBA"
_CONFIGURED_SEEDS = object()


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def selected_result_files(folders: list[str] | None = None, *,
                          seeds: Any = _CONFIGURED_SEEDS) -> list[Path]:
    folders = (["mock"] if USE_MOCK_RESULTS else MODEL_FOLDERS_TO_PLOT) if folders is None else folders
    if not folders or len(folders) != len(set(folders)):
        raise ValueError("Select one or more unique result folders.")
    paths = []
    for folder in folders:
        if (not isinstance(folder, str) or not folder or folder in (".", "..")
                or any(char in folder for char in '/\\:<>"|?*')):
            raise ValueError(f"Unsafe model folder name: {folder!r}")
        model_dir = RESULTS_DIR / folder
        filename = RESULT_FILENAME
        pattern = "*_results.json"
        selected_seeds = SEEDS_TO_PLOT if seeds is _CONFIGURED_SEEDS else seeds
        for directory in result_directories(model_dir, selected_seeds):
            label = directory.relative_to(RESULTS_DIR).as_posix()
            plant_reports = sorted(path for path in directory.glob(pattern)
                                   if path.is_file() and path.name != filename)
            path = directory / filename
            if plant_reports:
                print(f"Using {label}: {len(plant_reports)} per-artifact report(s)")
                paths.extend(plant_reports)
            elif path.is_file():
                print(f"Using {label}: {path.name}")
                paths.append(path)
            else:
                print(f"Skipping model without selected reports: {label}")
    if not paths:
        raise FileNotFoundError(
            "No selected Adaptive reports exist. Run adaptive_benchmark.py "
            "and finalize_results.py first."
        )
    return paths


def _model_label(record: dict[str, Any], report: Path) -> str:
    if (report.parent.name.startswith("seed") and report.parent.name[4:].isdigit()):
        return report.parent.parent.name
    return report.parent.name


def load_dashboard_rows(files: list[Path]) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for path in files:
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise TypeError(
                f"{path} is a single-run walkthrough report, not a finalized "
                "benchmark array. Run finalize_results.py on a batch report "
                "(or run adaptive_benchmark.py first)."
            )
        for record in records:
            if not isinstance(record, dict):
                raise TypeError(f"{path}: every run must be a JSON object.")
            seed = report_run_seed(record, path)
            complexity = _number(record.get("total_complexity"))
            if complexity is not None and (not complexity.is_integer() or not 1 <= complexity <= 10):
                raise ValueError(f"{path}: total_complexity must be an integer from 1 to 10.")
            scores = {
                "success_rate_1": _number(record.get("success_rate_1", record.get("success_rate"))),
                "success_rate_2": _number(record.get("success_rate_2")),
            }
            for field, success in scores.items():
                if success is not None and not 0 <= success <= 1:
                    raise ValueError(f"{path}: {field} must be a fraction from 0 to 1.")
            dynamics = ((record.get("applied_system_spec") or {}).get("dynamics")
                        or record.get("dynamics") or {})
            verification = record.get("verification") or {}
            tokens = ((record.get("tuning") or {}).get("token_usage")
                      or record.get("token_counts") or {})
            test = record.get("test") or {}
            total_tokens = (tokens.get("total") or {}).get("total_tokens")
            mse = (verification.get("tracking_mse") or {}).get("full")
            rmse = (
                [math.sqrt(float(value)) for value in mse]
                if isinstance(mse, list) and all(_number(value) is not None and value >= 0 for value in mse)
                else None
            )
            row = {
                "model": _model_label(record, path),
                "seed": seed,
                "artifact": record.get("artifact"),
                "plant": record.get("plant") or dynamics.get("system_name") or "Unknown",
                "test_id": record.get("test_id"),
                "initial_condition": dynamics.get("x0", dynamics.get("initial_condition",
                                                    test.get("initial_condition"))),
                "references": dynamics.get("references", test.get("targets")),
                "uncertainty": dynamics.get("uncertainty"),
                "disturbance": dynamics.get("disturbance"),
                "rmse_by_output": rmse,
                "complexity": int(complexity) if complexity is not None else None,
                **scores,
                "success_rate": scores["success_rate_1"],  # Legacy in-memory callers only.
                "success_rate_2_details": record.get("success_rate_2_details"),
                "total_tokens": _number(total_tokens),
                "cost_usd": _number(tokens.get("cost_usd")),
                "cost_simulated": bool(tokens.get("simulated")),
            }
            identity = (row["model"], _case_key(row), seed)
            if seed is not None and identity in seen:
                raise ValueError(f"{path}: duplicate model/case/seed result: {identity}.")
            seen.add(identity)
            rows.append(row)
    if not rows:
        raise ValueError("The selected reports contain no benchmark runs.")
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


def _score_value(row: dict[str, Any], field: str) -> float | None:
    if field not in SCORE_FIELDS:
        raise ValueError(f"Unsupported success score: {field}")
    return row.get(field, row.get("success_rate") if field == "success_rate_1" else None)


def _details(model: str, level: int, records: list[dict[str, Any]],
             score_field: str = "success_rate_1") -> str:
    mean = np.mean([_score_value(row, score_field) for row in records]) * 100
    lines = [f"Model: {_model_name(model)}", f"Complexity: {level}",
             f"Mean {score_field.replace('_', ' ')}: {mean:.1f}% across {len(records)} run(s)"]
    seeds = sorted({row["seed"] for row in records if row.get("seed") is not None})
    if seeds:
        lines.append("Seeds: " + ", ".join(map(str, seeds)))
    for row in records[:3]:
        refs = row.get("references") or []
        targets = ", ".join(
            f"{ref.get('output')}={ref.get('expr')}" for ref in refs if isinstance(ref, dict)
        ) or "N/A"
        lines.extend([
            "",
            f"Plant: {row['plant']} ({row.get('test_id') or 'test'})",
            f"Seed: {row.get('seed') if row.get('seed') is not None else 'legacy / unspecified'}",
            f"Initial state: {_format_values(row['initial_condition'])}",
            f"Output references: {targets}",
            f"Full-window RMSE: {_format_values(row['rmse_by_output'])}",
        ])
        fixed = row.get("success_rate_2_details")
        if score_field == "success_rate_2" and isinstance(fixed, dict):
            lines.append(f"Fixed targets passed: {fixed.get('passed_targets')}/{fixed.get('total_targets')}")
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


def _case_key(row: dict[str, Any]) -> tuple[str, str]:
    """Keep different plants/tests separate, even at the same complexity."""
    plant = str(row.get("artifact") or row["plant"])
    test = row.get("test_id")
    if not test:
        # Older reports without IDs still need to distinguish their conditions.
        test = json.dumps([row.get("initial_condition"), row.get("references")], sort_keys=True)
    return plant, str(test)


def _plant_name(name: str) -> str:
    known = {
        "AUVDepthDynamics": "AUV", "BallBeamDynamics": "Ball & Beam",
        "CartPolePendulum": "Cart-Pole", "DCMotorDynamics": "DC Motor",
        "InvertedPendulumDynamics": "Inverted Pendulum",
        "QuanserHelicopterDynamics": "Quanser Helicopter",
        "TwoLinkManipulatorDynamics": "Two-Link Manipulator",
        "AircraftPitchDynamics": "Aircraft Pitch", "CSTRDynamics": "CSTR",
    }
    return known.get(name, re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ",
                                name.removesuffix("Dynamics")).replace("_", " "))


def _success_cases(grouped: list[tuple[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    """Collect cases, rejecting contradictory saved rankings across LLMs."""
    cases = {}
    for _, entries in grouped:
        for row in entries:
            level = row["complexity"]
            if level is None:
                continue
            key = _case_key(row)
            if key in cases and cases[key]["level"] != level:
                raise ValueError(f"Conflicting saved complexities for {key}. "
                                 "Run finalize_results.py for all selected models first.")
            conditions = (row.get("initial_condition"), row.get("references"),
                          row.get("uncertainty"), row.get("disturbance"))
            if key in cases and cases[key]["conditions"] != conditions:
                raise ValueError(f"Different test conditions saved for {key}; "
                                 "rerun matching cases before comparing seeds/models.")
            cases.setdefault(key, {"key": key, "level": level, "plant": row["plant"],
                                   "test_id": row.get("test_id"), "conditions": conditions})
    ordered = sorted(cases.values(), key=lambda case: (
        case["level"], _plant_name(case["plant"]).casefold(), case["key"]))
    if not (0 <= CASE_GAP < COMPLEXITY_GAP):
        raise ValueError("Require 0 <= CASE_GAP < COMPLEXITY_GAP.")
    position = 0.0
    for index, case in enumerate(ordered):
        if index:
            gap = CASE_GAP if case["level"] == ordered[index - 1]["level"] else COMPLEXITY_GAP
            position += 0.82 + gap
        case["position"] = position
    return ordered


def mean_case_success(entries: list[dict[str, Any]], case: dict[str, Any],
                      score_field: str = "success_rate_1") -> tuple[float | None, list]:
    """Average saved success scores for one LLM/plant/test over selected seeds."""
    matches = [row for row in entries if _case_key(row) == case["key"]
               and row["complexity"] == case["level"] and _score_value(row, score_field) is not None]
    return (float(np.mean([_score_value(row, score_field) for row in matches])) if matches else None, matches)


def _draw_success(
    axis: plt.Axes, grouped: list[tuple[str, list[dict[str, Any]]]],
    score_field: str = "success_rate_1",
) -> dict[Any, tuple[str, int, list[dict[str, Any]]]]:
    if score_field == "success_rate_2":
        signatures = {}
        for _, entries in grouped:
            for row in entries:
                details = row.get("success_rate_2_details")
                if _score_value(row, score_field) is None or not isinstance(details, dict):
                    continue
                key = _case_key(row)
                signature = (details.get("mse_scope"), details.get("overshoot_scope"),
                             tuple(details.get("outputs") or []),
                             tuple(sorted((name, check.get("limit")) for name, check
                                          in (details.get("checks") or {}).items())))
                if key in signatures and signatures[key] != signature:
                    raise ValueError(f"Different fixed-target definitions for {key}; "
                                     "finalize all selected models/seeds again.")
                signatures[key] = signature
    cases = _success_cases(grouped)
    axis.set_title(f"{score_field.replace('_', ' ').title()} vs Complexity",
                   fontsize=14, weight="bold", pad=13)
    axis.set_xlabel("Dynamic-model cases grouped by manual complexity", labelpad=80)
    axis.set_ylabel("Mean success rate across seeds", labelpad=10)
    axis.set_ylim(0, 106)
    axis.set_yticks(np.arange(0, 101, 20))
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    if not cases:
        axis.text(0.5, 0.5, "No assigned complexity levels", transform=axis.transAxes,
                  ha="center", va="center", color=MUTED)
        axis.set_xticks([])
        return {}

    positions = [case["position"] for case in cases]
    width = min(0.72, 0.82 / len(grouped))
    show_labels = SHOW_SUCCESS_VALUES and len(grouped) * len(cases) <= 12 and width >= 0.16
    selectable = {}
    for model_index, (model, entries) in enumerate(grouped):
        offset = (model_index - (len(grouped) - 1) / 2) * width
        for case in cases:
            x, level = case["position"], case["level"]
            mean_success, matches = mean_case_success(entries, case, score_field)
            if mean_success is None:
                axis.text(x + offset, 2, "N/A", ha="center", va="bottom",
                          color=MUTED, fontsize=8, rotation=90 if not show_labels else 0)
                continue  # Missing model/case data is never plotted as zero.
            value = mean_success * 100
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
    labels = []
    for case in cases:
        label = textwrap.fill(_plant_name(case["plant"]), width=12,
                              break_long_words=False, break_on_hyphens=False)
        if sum(other["key"][0] == case["key"][0] for other in cases) > 1:
            label += f"\n{case['test_id'] or 'case'}"
        labels.append(label)
    axis.set_xticks(positions, labels)
    axis.tick_params(axis="x", labelsize=10.5, pad=10)
    plt.setp(axis.get_xticklabels(), rotation=PLANT_LABEL_ROTATION if len(cases) > 4 else 0,
             ha="right" if len(cases) > 4 else "center", rotation_mode="anchor",
             linespacing=1.3)
    # A second label tier shows the complexity shared by neighboring cases.
    levels = sorted({case["level"] for case in cases})
    # Place the complexity tier BELOW the rendered names, not at a guessed height.
    axis.figure.canvas.draw()
    renderer = axis.figure.canvas.get_renderer()
    label_bottom = min(label.get_window_extent(renderer).y0 for label in axis.get_xticklabels())
    tier_y = axis.transAxes.inverted().transform((axis.bbox.x0, label_bottom - 16))[1]
    axis.xaxis.set_label_coords(0.5, tier_y - 0.115)
    for index, level in enumerate(levels):
        members = [case for case in cases if case["level"] == level]
        left, right = members[0]["position"] - 0.46, members[-1]["position"] + 0.46
        axis.axvspan(left, right, color=TEXT, alpha=0.035 if index % 2 == 0 else 0.015,
                     zorder=0)
        axis.plot([left, left, right, right],
                  [tier_y + 0.015, tier_y, tier_y, tier_y + 0.015],
                  transform=axis.get_xaxis_transform(), color=MUTED,
                  linewidth=0.9, clip_on=False)
        axis.text((left + right) / 2, tier_y - 0.015, f"Complexity {level}",
                  transform=axis.get_xaxis_transform(), ha="center", va="top",
                  fontsize=9, weight="bold", color=TEXT, clip_on=False)
    axis.set_xlim(positions[0] - 0.60, positions[-1] + 0.60)
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
    *, score_field: str = "success_rate_1", keep_open: bool = False,
) -> Path:
    if not rows:
        raise ValueError("No finalized Adaptive benchmark rows to plot.")
    if score_field not in SCORE_FIELDS:
        raise ValueError(f"Unsupported success score: {score_field}")
    grouped = [(model, [row for row in rows if row["model"] == model])
               for model in dict.fromkeys(row["model"] for row in rows)]
    plt.style.use("dark_background")
    cases = _success_cases(grouped)
    success_width = max(11, len(cases) * 1.7)
    figure, axes = plt.subplots(1, 3, figsize=(success_width + 12, 9),
                                gridspec_kw={"width_ratios": [success_width, 5, 5]})
    figure.patch.set_facecolor("#0D1117")
    figure.subplots_adjust(left=0.055, right=0.98, bottom=0.34, top=0.76, wspace=0.28)
    for axis in axes:
        _style_axis(axis)

    selectable = _draw_success(axes[0], grouped, score_field)
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
        annotation.set_text(_details(model, level, matching, score_field))
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
    figure.suptitle("ADAPTIVE TUNING BENCHMARK — " + score_field.replace("_", " ").upper(), x=0.055, y=0.97,
                    ha="left", fontsize=21, weight="bold", color=TEXT)
    figure.text(0.055, 0.91,
                f"AI-agent controller tuning | {len(rows)} runs | {len(grouped)} model(s)",
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
                footer + "Success bars average available seeds per LLM/case; click for seed details. N/A = unavailable.",
                color=MUTED, fontsize=9, ha="left")

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, facecolor=figure.get_facecolor(),
                   bbox_inches="tight", pad_inches=0.18)
    if show_plot:
        plt.show()
    elif not keep_open:
        plt.close(figure)
    return output


def create_dashboards(rows: list[dict[str, Any]], output: Path = DEFAULT_OUTPUT,
                      show_plot: bool = True) -> list[Path]:
    """Create/save both figures before opening windows; average scores per seed."""
    original_figures = set(plt.get_fignums())
    try:
        outputs = [create_dashboard(
            rows, output.with_name(f"{output.stem}_{field}{output.suffix}"),
            show_plot=False, score_field=field, keep_open=show_plot,
        ) for field in SCORE_FIELDS]
        if show_plot:
            plt.show()
        return outputs
    finally:
        for number in set(plt.get_fignums()) - original_figures:
            plt.close(number)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the Adaptive benchmark dashboard.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Base image filename; _success_rate_1 and _success_rate_2 are appended.")
    parser.add_argument("--no-show", action="store_true",
                        help="Save without opening an interactive plot window.")
    parser.add_argument("--models", nargs="+", help="Model result folders to plot.")
    parser.add_argument("--seeds", nargs="+", type=int, help="Seeds to average (default: plot settings).")
    args = parser.parse_args()
    folders = ["mock"] if USE_MOCK_RESULTS else args.models
    files = selected_result_files(folders, seeds=SEEDS_TO_PLOT if args.seeds is None else args.seeds)
    rows = load_dashboard_rows(files)
    outputs = create_dashboards(rows, output=args.output, show_plot=not args.no_show)
    print(f"Loaded {len(rows)} runs from {len(files)} model report(s).")
    for output in outputs:
        print(f"Dashboard written to: {output}")


if __name__ == "__main__":
    main()
