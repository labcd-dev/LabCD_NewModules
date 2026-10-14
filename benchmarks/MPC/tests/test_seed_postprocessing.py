"""Offline finalization/discovery/averaging checks for model/seed/artifact reports."""

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.MPC import finalize_results as finalize
from benchmarks.MPC import plot_dashboard as dashboard
from benchmarks.MPC import plot_trajectories as trajectories


class SeedPostprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for module in (finalize, dashboard):
            patcher = patch.object(module, "RESULTS_DIR", self.root)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.test = {"test_id": "test_1", "initial_condition": [1.0],
                     "target": [0.0], "scenario_level": 1, "total_complexity": 2}
        self.artifacts = {name + ".json": {"system_name": name,
                          "metadata": {"outputs": ["position"]},
                          "tests": {"combinations": [self.test]}}
                          for name in ("PlantA", "PlantB")}

    def save(self, model, seed, plant="PlantA", score=None):
        report = self.root / model / f"seed{seed}" / f"{plant}_results.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "artifact": plant + ".json", "model": {"name": plant},
            "agent_mode": model, "seeds": {"run": seed},
            **copy.deepcopy(self.test), "success_rate": score,
            "tuning": {"token_usage": {"total_tokens": 20, "cost_usd": 0.01}},
            "verification": {
                "outputs": ["position"], "tracking_mse": {"full": [0.01]},
                "task_scale": [1.0], "task_scale_basis": "mse",
                "raw_result": {"trajectory_file": "trajectories/test.npz"},
            },
        }
        report.write_text(json.dumps([record]), encoding="utf-8")
        return report

    def test_finalize_selects_each_seed_and_updates_only_selected_reports(self):
        paths = [self.save("model-a", seed) for seed in (42, 123, 2024)]
        old = self.root / "model-a" / "PlantA_results.json"
        old.write_text("legacy untouched", encoding="utf-8")
        (self.root / "benchmark_progress.json").write_text("{}", encoding="utf-8")
        with patch.object(finalize, "USE_MOCK_RESULTS", False), \
                patch.object(finalize, "SEEDS_TO_UPDATE", None):
            selected = finalize.select_result_paths([], model_folders=["model-a"], seeds=[42, 2024])
        self.assertEqual(selected, [paths[0], paths[2]])
        for path in selected:
            original = json.loads(path.read_text(encoding="utf-8"))[0]
            with redirect_stdout(io.StringIO()):
                finalize.update_results(path, self.artifacts)
            updated = json.loads(path.read_text(encoding="utf-8"))[0]
            self.assertIsNotNone(updated["success_rate_1"])
            for key in ("seeds", "verification", "tuning"):
                self.assertEqual(updated[key], original[key])
        self.assertIsNone(json.loads(paths[1].read_text(encoding="utf-8"))[0]["success_rate"])
        self.assertEqual(old.read_text(encoding="utf-8"), "legacy untouched")
        self.assertEqual(list(self.root.rglob("*_full.json")), [])

    def test_bar_mean_is_per_model_case_across_seeds_not_across_plants(self):
        paths = [self.save("model-a", 42, score=0.9),
                 self.save("model-a", 123, score=0.6),
                 self.save("model-a", 2024, score=0.0),
                 self.save("model-a", 42, "PlantB", score=0.2),
                 self.save("model-b", 42, score=1.0)]
        rows = dashboard.load_dashboard_rows(paths)
        grouped = [(name, [row for row in rows if row["model"] == name])
                   for name in ("model-a", "model-b")]
        cases = dashboard._success_cases(grouped)
        self.assertEqual(len(cases), 2)  # Both plants have complexity 2.
        case_a = next(case for case in cases if case["plant"] == "PlantA")
        mean, matches = dashboard.mean_case_success(grouped[0][1], case_a)
        self.assertAlmostEqual(mean, 0.5)
        self.assertEqual([row["seed"] for row in matches], [42, 123, 2024])
        figure, axis = plt.subplots()
        try:
            bars = dashboard._draw_success(axis, grouped)
            bar = next(bar for bar, (model, _, entries) in bars.items()
                       if model == "model-a" and entries[0]["plant"] == "PlantA")
            self.assertAlmostEqual(bar.get_height(), 50.0)
        finally:
            plt.close(figure)

    def test_missing_seed_score_excluded_but_real_zero_included(self):
        paths = [self.save("model-a", 42, score=1.0),
                 self.save("model-a", 123, score=None),
                 self.save("model-a", 2024, score=0.0)]
        rows = dashboard.load_dashboard_rows(paths)
        case = dashboard._success_cases([("model-a", rows)])[0]
        mean, matches = dashboard.mean_case_success(rows, case)
        self.assertEqual(mean, 0.5)
        self.assertEqual(len(matches), 2)
        self.assertIn("Seeds: 42, 2024", dashboard._details("model-a", 2, matches))

    def test_mismatch_seed_rejected_without_overwriting_report(self):
        path = self.save("model-a", 42)
        records = json.loads(path.read_text(encoding="utf-8"))
        records[0]["seeds"]["run"] = 123
        path.write_text(json.dumps(records), encoding="utf-8")
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "disagrees"):
            finalize.update_results(path, self.artifacts)
        self.assertEqual(path.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            dashboard.load_dashboard_rows([path])
        with self.assertRaisesRegex(ValueError, "disagrees"):
            trajectories.plot_record(records[0], path)

    def test_duplicate_seed_and_changed_conditions_not_silently_averaged(self):
        path = self.save("model-a", 42, score=1.0)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            dashboard.load_dashboard_rows([path, path])
        other = self.save("model-a", 123, score=0.5)
        rows = dashboard.load_dashboard_rows([path, other])
        rows[1]["initial_condition"] = [2.0]
        with self.assertRaisesRegex(ValueError, "Different test conditions"):
            dashboard._success_cases([("model-a", rows)])

    def test_trajectory_discovery_all_seeds_independent_of_dashboard_filter(self):
        paths = [self.save("model-a", seed) for seed in (42, 123, 2024)]
        with patch.object(dashboard, "SEEDS_TO_PLOT", [42]), redirect_stdout(io.StringIO()):
            self.assertEqual(dashboard.selected_result_files(["model-a"]), [paths[0]])
            self.assertEqual(dashboard.selected_result_files(["model-a"], seeds=None),
                             [paths[0], paths[1], paths[2]])

    def test_seed_trajectory_archive_and_plot_output_are_isolated(self):
        outputs = []
        for seed in (42, 123):
            path = self.save("model-a", seed)
            archive = path.parent / "trajectories" / "test.npz"
            archive.parent.mkdir()
            np.savez_compressed(archive, states=np.array([[seed], [0.0]]),
                                inputs=np.array([[0.0]]), reference=np.zeros((2, 1)),
                                times=np.array([0.0, 1.0]))
            record = json.loads(path.read_text(encoding="utf-8"))[0]
            arrays = trajectories.load_trajectories(record, path)
            self.assertEqual(arrays["states"][0, 0], seed)
            with patch.object(trajectories, "matching_metadata",
                              return_value={"states": ["position"], "inputs": ["force"]}):
                plots = trajectories.plot_record(record, path, output_root=self.root / "plots")
            self.assertTrue(all(p.parent == self.root / "plots" / "model-a" / f"seed{seed}" /
                                "PlantA" and p.is_file() for p in plots))
            outputs.extend(plots)
        self.assertEqual(len(set(outputs)), 4)


if __name__ == "__main__":
    unittest.main()
