"""Plot saved final-controller simulation trajectories, without tuning/API calls.

One state figure and one applied-input figure per saved test. Targets are drawn
only for declared controlled outputs; other states have no invented targets.
Defined bounds are drawn from the matching current artifact, not inferred
from trajectory data. Historical runs may predate those limits.
Output: plots/<LLM folder>/seed<value>/<artifact stem>/<test_id>_<plot type>.png.
Legacy unseeded reports retain their original output layout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

try:
    from . import plot_dashboard
except ImportError:  # Direct execution: python benchmarks/Adaptive/plot_trajectories.py
    import plot_dashboard

ADAPTIVE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = ADAPTIVE_DIR / "artifacts"
PLOTS_DIR = ADAPTIVE_DIR / "plots"
ARRAY_NAMES = ("states", "inputs", "reference", "times", "outputs")

# ------------------------------ PLOT SETTINGS ------------------------------
USE_MOCK_RESULTS = False  # True ignores the real-model list and plots only mock.
MODEL_FOLDERS_TO_PLOT = [
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-4o-mini",
    "gpt-4.1-nano",
    "groq-gpt-oss-120b",
]
ARTIFACT_FILES: list[str] | None = None  # None plots every saved artifact.
TEST_IDS: list[str] | None = None  # None plots every saved test.
SEEDS_TO_PLOT: list[int] | None = None  # None plots every saved seed independently.
SHOW_PLOTS = False  # Save only by default; True also opens the figures.
# ---------------------------------------------------------------------------


def safe_name(value: Any) -> str:
    """Require one safe directory/filename component before writing plots."""
    if (not isinstance(value, str) or not value or value in (".", "..")
            or value.endswith((" ", "."))
            or any(char in value for char in '/\\:<>"|?*')
            or any(ord(char) < 32 for char in value)):
        raise ValueError(f"Unsafe plot name: {value!r}")
    return value


def load_trajectories(record: dict[str, Any], report: Path) -> dict[str, np.ndarray]:
    """Load the linked numeric NPZ, or arrays embedded in a legacy JSON report."""
    series = record.get("series") or ((record.get("core_outputs") or {}).get("result") or {}).get("series") or {}
    channels = series.get("channels") or {}
    link = series.get("trajectory_file")
    if link:
        archive_path = Path(link)
        if archive_path.is_absolute():
            raise ValueError("trajectory_file must be relative to the results JSON.")
        archive_path = (report.parent / archive_path).resolve()
        if not archive_path.is_relative_to(report.parent.resolve()):
            raise ValueError("trajectory_file must stay inside the model results folder.")
        with np.load(archive_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in ARRAY_NAMES}
    elif all(isinstance(channels.get(key), dict) and "data" in channels[key]
             for key in ("t", "x", "y", "ref", "u")):
        arrays = {name: np.asarray(channels[key]["data"], dtype=float)
                  for key, name in {"t": "times", "x": "states", "y": "outputs",
                                    "ref": "reference", "u": "inputs"}.items()}
    else:
        raise ValueError("No saved verification trajectories; no simulation will be rerun.")

    if any(array.dtype.kind not in "biuf" for array in arrays.values()):
        raise ValueError("Trajectory arrays must be numeric.")
    times, states, reference, inputs = (arrays[name] for name in
                                        ("times", "states", "reference", "inputs"))
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("Trajectory times must be finite and strictly increasing.")
    if states.ndim != 2 or states.shape[0] != len(times) or states.shape[1] == 0:
        raise ValueError("States must have one row per timestamp and one column per state.")
    outputs = arrays["outputs"]
    if (outputs.ndim != 2 or outputs.shape[0] != len(times) or outputs.shape[1] == 0
            or reference.shape != outputs.shape):
        raise ValueError("Output references must match controlled outputs, not all internal states.")
    if (inputs.ndim != 2 or inputs.shape[1] == 0
            or len(inputs) not in (len(times), len(times) - 1)):
        raise ValueError("Inputs must have one row per timestamp or per control interval.")
    return arrays


def matching_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Read the matching plant's current metadata without executing any code."""
    artifact_name = safe_name(record.get("artifact"))
    artifact_path = ARTIFACTS_DIR / artifact_name
    if artifact_path.is_file():
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        # Do not attach names from a different plant to historical results.
        if artifact.get("system_name") == record.get("plant"):
            metadata = artifact.get("metadata") or {}
            return metadata if isinstance(metadata, dict) else {}
    return {}


def channel_names(record: dict[str, Any], arrays: dict[str, np.ndarray],
                  metadata: dict[str, Any] | None = None) -> tuple[list[str], list[str]]:
    """Read channel labels only; never execute an artifact's Python code."""
    metadata = matching_metadata(record) if metadata is None else metadata
    channels = (record.get("series") or {}).get("channels") or {}
    spec = (record.get("applied_system_spec") or {}).get("dynamics") or {}

    def names(key: str, count: int, prefix: str) -> list[str]:
        channel = "x" if key == "states" else "u"
        values = (channels.get(channel) or {}).get("names") or spec.get(key) or metadata.get(key)
        if isinstance(values, list) and len(values) == count and all(isinstance(v, str) for v in values):
            return values
        return [f"{prefix} {index + 1}" for index in range(count)]

    return (names("states", arrays["states"].shape[1], "State"),
            names("inputs", arrays["inputs"].shape[1], "Input"))


def controlled_output_names(record: dict[str, Any], count: int,
                            state_names: list[str], metadata: dict[str, Any]) -> list[str]:
    """Map output references to states by saved names, never by column number."""
    channels = (record.get("series") or {}).get("channels") or {}
    spec = (record.get("applied_system_spec") or {}).get("dynamics") or {}
    names = (channels.get("y") or {}).get("names") or spec.get("outputs") or metadata.get("outputs")
    if (not isinstance(names, list) or len(names) != count
            or any(not isinstance(name, str) or name not in state_names for name in names)
            or len(names) != len(set(names))):
        raise ValueError("Saved controlled output names must identify their state/reference columns.")
    ref_names = (channels.get("ref") or {}).get("names")
    if ref_names is not None and ref_names != names:
        raise ValueError("Saved output and reference channel order must match.")
    return names


def channel_bounds(metadata: dict[str, Any], key: str, count: int) -> np.ndarray | None:
    """Validate [lower, upper] limits against the saved channel dimensions."""
    limits = metadata.get(key)
    if limits is None:
        return None
    names = metadata.get("states" if key == "state_bounds" else "inputs")
    if not isinstance(names, list) or len(names) != count:
        raise ValueError(f"Current artifact {key} channel count does not match saved trajectories.")
    if (not isinstance(limits, list) or len(limits) != 2
            or any(not isinstance(vector, list) or len(vector) != count for vector in limits)):
        raise ValueError(f"metadata.{key} must be [lower, upper], each with {count} values.")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           for vector in limits for value in vector):
        raise ValueError(f"metadata.{key} must contain numeric limits.")
    values = np.asarray(limits, dtype=float)
    if not np.isfinite(values).all() or np.any(values[0] > values[1]):
        raise ValueError(f"metadata.{key} limits must be finite and lower <= upper.")
    return values


def _draw_bounds(axis: plt.Axes, bounds: np.ndarray | None, index: int) -> None:
    if bounds is None:
        return
    lower, upper = bounds[:, index]
    if lower == upper:
        axis.axhline(lower, color="#DC2626", linestyle="-.", linewidth=1.3,
                     label=f"Fixed bound ({lower:g})")
        return
    for value, label in ((lower, "Lower bound"), (upper, "Upper bound")):
        axis.axhline(value, color="#DC2626", linestyle="-.", linewidth=1.3,
                     label=f"{label} ({value:g})")


def _figure(names: list[str], title: str) -> tuple[plt.Figure, np.ndarray]:
    figure, axes = plt.subplots(len(names), 1, sharex=True, squeeze=False,
                                figsize=(11.5, 1.4 + 2.25 * len(names)),
                                layout="constrained")
    figure.suptitle(title, fontsize=14, weight="bold")
    for axis, name in zip(axes[:, 0], names):
        axis.set_title(name.replace("_", " "), loc="left", fontsize=11)
        axis.set_ylabel("Value")  # Units are not guessed from channel names.
        axis.grid(True, color="#CBD5E1", alpha=0.55, linewidth=0.7)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=9)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=8))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        for edge in ("top", "right"):
            axis.spines[edge].set_visible(False)
    axes[-1, 0].set_xlabel("Time (s)")
    return figure, axes[:, 0]


def plot_record(record: dict[str, Any], report: Path, output_root: Path = PLOTS_DIR,
                show_plot: bool = False) -> list[Path]:
    """Plot actual saved references and applied inputs, retaining every sample."""
    plot_dashboard.report_run_seed(record, report)
    arrays = load_trajectories(record, report)
    metadata = matching_metadata(record)
    states_names, input_names = channel_names(record, arrays, metadata)
    output_names = controlled_output_names(record, arrays["reference"].shape[1], states_names, metadata)
    state_bounds = channel_bounds(metadata, "state_bounds", len(states_names))
    input_bounds = channel_bounds(metadata, "input_bounds", len(input_names))
    artifact = Path(safe_name(record.get("artifact"))).stem
    seed_folder = (safe_name(report.parent.name) if report.parent.name.startswith("seed")
                   and report.parent.name[4:].isdigit() else None)
    model = safe_name(report.parent.parent.name if seed_folder else report.parent.name)
    test_id = safe_name(record.get("test_id") or "test")
    output_dir = output_root / model
    if seed_folder:
        output_dir /= seed_folder
    output_dir /= safe_name(artifact)
    output_dir.mkdir(parents=True, exist_ok=True)
    system_name = record.get("plant") or artifact
    title = f"{system_name} | {model} | {test_id}\nFinal-controller simulation"
    if seed_folder:
        title += f" | {seed_folder}"
    if state_bounds is not None or input_bounds is not None:
        title += " | Bound lines: current artifact limits"
    times = arrays["times"]
    figures = []
    outputs = []
    try:
        with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                             "figure.facecolor": "white", "axes.facecolor": "white",
                             "text.color": "#18212F", "axes.labelcolor": "#18212F",
                             "axes.edgecolor": "#94A3B8", "xtick.color": "#334155",
                             "ytick.color": "#334155"}):
            figure, axes = _figure(states_names, "States and targets vs time\n" + title)
            figures.append(figure)
            for index, axis in enumerate(axes):
                axis.plot(times, arrays["states"][:, index], color="#2563EB",
                          linewidth=1.6, label="Actual state")
                state_name = states_names[index]
                if state_name in output_names:
                    ref_index = output_names.index(state_name)
                    axis.plot(times, arrays["reference"][:, ref_index], color="#D97706",
                              linestyle="--", linewidth=1.6, label="Target / reference")
                else:
                    axis.text(0.99, 0.95, "Not a selected tracking output", ha="right", va="top",
                              fontsize=8, color="#64748B", transform=axis.transAxes)
                _draw_bounds(axis, state_bounds, index)
                axis.legend(loc="best", fontsize=9, framealpha=0.9,
                            ncol=2 if state_bounds is not None else 1)
            outputs.append(output_dir / f"{test_id}_states_vs_time.png")

            figure, axes = _figure(input_names, "Applied control inputs vs time\n" + title)
            figures.append(figure)
            for index, axis in enumerate(axes):
                values = arrays["inputs"][:, index]
                if len(values) == len(times) - 1 and len(values):
                    # u[k] is applied on [times[k], times[k+1]), not the ending timestamp.
                    axis.stairs(values, times, baseline=None, color="#059669",
                                linewidth=1.5, label="Applied input")
                elif len(values):
                    axis.step(times, values, where="post", color="#059669",
                              linewidth=1.5, label="Applied input")
                else:
                    axis.text(0.5, 0.5, "No control steps recorded", ha="center",
                              va="center", transform=axis.transAxes)
                axis.set_xlim(times[0], times[-1] if len(times) > 1 else times[0] + 1)
                _draw_bounds(axis, input_bounds, index)
                if len(values) or input_bounds is not None:
                    axis.legend(loc="best", fontsize=9, framealpha=0.9)
            outputs.append(output_dir / f"{test_id}_inputs_vs_time.png")

            for figure, output in zip(figures, outputs):
                figure.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
            if show_plot:
                plt.show()
    finally:
        for figure in figures:
            plt.close(figure)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="Plot only mock results.")
    parser.add_argument("--models", nargs="+", help="LLM results folders to plot.")
    parser.add_argument("--artifacts", nargs="+", help="Artifact filenames to plot.")
    parser.add_argument("--tests", nargs="+", help="Test IDs to plot.")
    parser.add_argument("--seeds", nargs="+", type=int, help="Seed values to plot.")
    parser.add_argument("--show", action="store_true", help="Also open plot windows.")
    args = parser.parse_args()
    folders = ["mock"] if USE_MOCK_RESULTS or args.mock else (args.models or MODEL_FOLDERS_TO_PLOT)
    artifacts = args.artifacts if args.artifacts is not None else ARTIFACT_FILES
    test_ids = args.tests if args.tests is not None else TEST_IDS
    seeds = args.seeds if args.seeds is not None else SEEDS_TO_PLOT
    files = plot_dashboard.selected_result_files(folders, seeds=seeds)
    plotted = skipped = 0
    for report in files:
        records = json.loads(report.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise TypeError(f"{report} must contain a JSON array of benchmark runs.")
        for record in records:
            if artifacts is not None and record.get("artifact") not in artifacts:
                continue
            if test_ids is not None and record.get("test_id") not in test_ids:
                continue
            try:
                outputs = plot_record(record, report, show_plot=SHOW_PLOTS or args.show)
            except (OSError, ValueError, KeyError, TypeError) as error:
                skipped += 1
                print(f"Skipping {report} / {record.get('test_id')}: {error}")
                continue
            plotted += 1
            for output in outputs:
                print(f"Saved: {output}")
    print(f"Plotted {plotted} test(s); skipped {skipped} unavailable/invalid trajectory set(s).")
    if not plotted:
        raise SystemExit("No selected saved trajectories could be plotted.")


if __name__ == "__main__":
    main()
