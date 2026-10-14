"""Saved Adaptive trajectory plotting, reference mapping, and file selection."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.Adaptive import adaptive_benchmark as benchmark
from benchmarks.Adaptive import plot_dashboard, plot_trajectories as plotter


class TrajectoryPlotTests(unittest.TestCase):
    def setUp(self):
        self.series = {"n_points": 3, "channels": {
            "t": {"label": "time", "unit": "s", "data": [0.0, 0.5, 1.0]},
            "x": {"names": ["position", "velocity", "angle"], "data": [[1, 0, 2], [2, 1, 3], [3, 0, 4]]},
            "y": {"names": ["angle", "position"], "data": [[2, 1], [3, 2], [4, 3]]},
            "ref": {"names": ["angle", "position"], "data": [[8, 3], [8, 3], [8, 3]]},
            "u": {"names": ["force"], "data": [[1], [0.5], [0]]}}}
        self.record = {"artifact": "Example.json", "plant": "ExamplePlant", "test_id": "test_1", "series": self.series}

    def test_legacy_inline_and_external_trajectories_are_equal(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(benchmark, "RESULTS_DIR", Path(directory)):
            path = benchmark.save_artifact_results(Path("Example.json"), {"folder_name": "mock"}, [self.record])
            saved = json.loads(path.read_text())[0]
            legacy = plotter.load_trajectories(self.record, path)
            linked = plotter.load_trajectories(saved, path)
            for name in plotter.ARRAY_NAMES:
                np.testing.assert_array_equal(legacy[name], linked[name])
            self.assertEqual(linked["states"].shape, (3, 3))
            self.assertEqual(linked["reference"].shape, (3, 2))

    def test_plots_follow_model_plant_folder_and_only_selected_states_have_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "mock" / "Example_results.json"
            root = Path(directory) / "plots"
            figures = []
            original = plotter._figure
            def capture(*args):
                figure, axes = original(*args)
                figures.append((figure, axes))
                return figure, axes
            with patch.object(plotter, "_figure", side_effect=capture):
                paths = plotter.plot_record(self.record, report, root)
            self.assertEqual([path.name for path in paths], ["test_1_states_vs_time.png", "test_1_inputs_vs_time.png"])
            self.assertTrue(all(path.is_file() and path.parent == root / "mock" / "Example" for path in paths))
            axes = figures[0][1]
            np.testing.assert_array_equal(axes[0].lines[1].get_ydata(), [3, 3, 3])
            self.assertEqual(len(axes[1].lines), 1)  # No fabricated velocity reference.
            np.testing.assert_array_equal(axes[2].lines[1].get_ydata(), [8, 8, 8])
            self.assertFalse(plt.get_fignums())

    def test_missing_arrays_invalid_shapes_and_unsafe_paths_are_rejected(self):
        report = Path.cwd() / "mock" / "Example_results.json"
        with self.assertRaisesRegex(ValueError, "No saved"):
            plotter.load_trajectories({}, report)
        with self.assertRaisesRegex(ValueError, "inside"):
            plotter.load_trajectories({"series": {"trajectory_file": "../outside.npz"}}, report)
        record = copy.deepcopy(self.record)
        record["series"]["channels"]["ref"]["data"] = [[1], [2], [3]]
        with self.assertRaisesRegex(ValueError, "controlled outputs"):
            plotter.load_trajectories(record, report)
        record = copy.deepcopy(self.record)
        record["series"]["channels"]["t"]["data"] = [0, 0, 1]
        with self.assertRaisesRegex(ValueError, "increasing"):
            plotter.load_trajectories(record, report)
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            plotter.safe_name("../escape")

    def test_reference_names_must_match_output_order(self):
        record = copy.deepcopy(self.record)
        record["series"]["channels"]["ref"]["names"] = ["position", "angle"]
        with self.assertRaisesRegex(ValueError, "order"):
            plotter.controlled_output_names(record, 2, ["position", "velocity", "angle"], {})

    def test_raw_and_brief_report_discovery_and_dashboard_folder(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(plot_dashboard, "RESULTS_DIR", Path(directory)):
            folder = Path(directory) / "mock"
            folder.mkdir()
            for name in ("AUV_results.json", "AUV_results_brief.json", "batch_tuning_results.json", "batch_tuning_results_brief.json"):
                (folder / name).write_text("[]", encoding="utf-8")
            self.assertEqual([p.name for p in plot_dashboard.selected_result_files(["mock"])], ["AUV_results.json"])
            self.assertEqual(plot_dashboard.DEFAULT_OUTPUT.parent, plot_dashboard.ADAPTIVE_DIR / "plots" / "dashboard")


if __name__ == "__main__":
    unittest.main()
