"""Both Adaptive scores, real overshoot semantics and offline finalization."""

from contextlib import redirect_stdout
import copy
import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.Adaptive import adaptive_benchmark as benchmark
from benchmarks.Adaptive import finalize_results as finalize
from benchmarks.Adaptive import plot_dashboard as dashboard
from benchmarks.Adaptive.success_rate.success_rate_1 import calculate_success_rate_1
from benchmarks.Adaptive.success_rate.success_rate_2 import evaluate_fixed_targets, true_overshoot


class SuccessRateTests(unittest.TestCase):
    def setUp(self):
        self.outputs = ["position"]
        self.targets = {"mse_max": 0.1, "settling_time_max_s": 2.0,
                        "overshoot_max_pct": 5.0, "control_effort_max": 0.1}
        self.verification = {
            "outputs": self.outputs, "tracking_mse": {"full": [0.039]},
            "task_scale": [1.0], "numerically_healthy": True,
            "success_checks": {"finite": True, "bounded": True, "mse_target": True},
            "overshoot_frac": [1.0], "settling_time": 3.07, "settling_time_reached": True,
            "control_rms": [0.10017],
        }
        self.record = {
            "verification": self.verification, "applied_system_spec": {"dynamics": {
                "outputs": self.outputs}},
            "series": {"n_points": 3, "downsampled": False, "channels": {
                "t": {"data": [0.0, 1.0, 2.0]},
                "y": {"names": self.outputs, "data": [[0.0], [0.5], [1.0]]},
                "ref": {"names": self.outputs, "data": [[1.0], [1.0], [1.0]]},
            }},
        }

    def evaluate(self, limits=None):
        return evaluate_fixed_targets(self.verification, self.targets if limits is None else limits,
                                      self.outputs, record=self.record)

    def test_rate_1_is_unchanged(self):
        self.assertEqual(calculate_success_rate_1(self.verification, self.outputs),
                         round(math.exp(-10 * 0.039), 6))
        self.verification["numerically_healthy"] = False
        self.assertEqual(calculate_success_rate_1(self.verification, self.outputs), 0.0)

    def test_initial_error_is_not_overshoot_and_three_of_four_pass(self):
        result = self.evaluate()
        self.assertEqual(result["success_rate_2"], 0.75)
        self.assertEqual(result["passed_targets"], 3)
        self.assertEqual(result["checks"]["overshoot_max_pct"]["measured_value"], 0.0)
        self.assertTrue(result["checks"]["overshoot_max_pct"]["passed"])
        self.assertAlmostEqual(result["checks"]["control_effort_max"]["measured_value"], 0.10017**2)
        self.assertFalse(result["checks"]["settling_time_max_s"]["passed"])
        self.assertEqual(self.verification["overshoot_frac"], [1.0])  # Original metrics intact.

    def test_true_overshoot_for_zero_reference_both_directions_and_boundaries(self):
        channels = self.record["series"]["channels"]
        channels["y"]["data"] = [[1.0], [0.0], [-0.1]]
        channels["ref"]["data"] = [[0.0], [0.0], [0.0]]
        measured = true_overshoot(self.record, self.outputs)
        self.assertEqual(measured["measured_value"], 10.0)
        self.assertEqual(self.evaluate({"overshoot_max_pct": 10.0})["success_rate_2"], 1.0)
        channels["y"]["data"] = [[-1.0], [0.0], [0.2]]
        self.assertEqual(true_overshoot(self.record, self.outputs)["measured_value"], 20.0)

    def test_mimo_overshoot_uses_worst_eligible_controlled_output_by_name(self):
        channels = self.record["series"]["channels"]
        channels["y"] = {"names": ["b", "a"], "data": [[1, 2], [0, 0], [-0.2, -0.8]]}
        channels["ref"] = {"names": ["b", "a"], "data": [[0, 0], [0, 0], [0, 0]]}
        measured = true_overshoot(self.record, ["a", "b"])
        self.assertEqual(measured["per_output_percent"], {"a": 40.0, "b": 20.0})
        self.assertEqual(measured["measured_value"], 40.0)
        self.assertEqual(true_overshoot(self.record, ["b"])["measured_value"], 20.0)

    def test_mean_squared_effort_is_mean_of_squared_channel_rms(self):
        self.verification["control_rms"] = [1.0, 3.0]
        result = self.evaluate({"control_effort_max": 4.5})
        self.assertEqual(result["checks"]["control_effort_max"]["measured_value"], 5.0)
        self.assertEqual(result["success_rate_2"], 0.0)  # Squaring mean RMS gives wrong 4.
        self.assertEqual(self.evaluate({"control_effort_max": 5.0})["success_rate_2"], 1.0)

    def test_missing_downsampled_and_undefined_overshoot_are_not_false_passes(self):
        self.record["series"]["downsampled"] = True
        self.assertIsNone(self.evaluate()["success_rate_2"])
        self.record["series"]["channels"]["y"]["data"][-1] = [1.2]
        result = self.evaluate()
        self.assertEqual(result["success_rate_2"], 0.5)  # Observed overshoot proves failure.
        self.assertFalse(result["checks"]["overshoot_max_pct"]["exact"])
        self.record["series"]["downsampled"] = False
        self.record["series"]["channels"]["ref"]["data"] = [[0], [0.5], [1]]
        self.assertIsNone(self.evaluate()["success_rate_2"])  # Moving reference: undefined.
        self.record["series"] = None
        self.assertIsNone(self.evaluate()["success_rate_2"])
        self.assertEqual(self.evaluate({"mse_max": 0.1})["success_rate_2"], 1.0)
        self.assertIsNone(self.evaluate({})["success_rate_2"])

    def test_numeric_failures_invalid_limits_and_unreached_settling(self):
        self.verification["settling_time"] = 1.0
        self.verification["settling_time_reached"] = False
        self.assertEqual(self.evaluate()["success_rate_2"], 0.75)
        self.verification["numerically_healthy"] = False
        self.assertEqual(self.evaluate()["success_rate_2"], 0.0)
        for limits in ([], {"unknown": 1}, {"mse_max": -1}, {"mse_max": True},
                       {"mse_max": float("nan")}):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                self.evaluate(limits)

    def test_archive_security_and_invalid_arrays_are_unavailable_not_executed(self):
        self.record["series"]["trajectory_file"] = "../outside.npz"
        self.assertIn("must stay inside", true_overshoot(
            self.record, self.outputs, Path("seed42/Plant_results.json"))["reason"])
        self.record["series"].pop("trajectory_file")
        self.record["series"]["channels"]["y"]["data"][-1] = [float("nan")]
        self.assertIsNone(self.evaluate()["success_rate_2"])

    def test_finalization_reads_own_seed_archive_updates_json_preserves_raw_metrics(self):
        artifact = finalize.load_artifacts()["AUV.json"]
        test = artifact["tests"]["combinations"][0]
        record = copy.deepcopy(self.record)
        record.update(artifact="AUV.json", plant=artifact["system_name"], test_id=test["test_id"],
                      test=copy.deepcopy(test), seeds={"run": 42}, tuning={}, pipeline_completed=True)
        record["applied_system_spec"] = {"system_name": artifact["system_name"],
                                        "dynamics": {"outputs": ["depth"]}}
        record["verification"]["outputs"] = ["depth"]
        series = record["series"]
        series["trajectory_file"] = "trajectories/AUV.npz"
        series["channels"] = {"y": {"names": ["depth"]}, "ref": {"names": ["depth"]}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model" / "seed42" / "AUV_results.json"
            archive = path.parent / series["trajectory_file"]
            archive.parent.mkdir(parents=True)
            np.savez_compressed(archive, outputs=[[0.0], [0.5], [1.0]], reference=[[1.0]]*3,
                                times=[0.0, 1.0, 2.0])
            path.write_text(json.dumps([record]), encoding="utf-8")
            full_before, archive_before = path.read_bytes(), archive.read_bytes()
            brief_path = finalize.update_results(path, {"AUV.json": artifact})
            brief = json.loads(brief_path.read_text())[0]
            self.assertEqual(brief["success_rate_2"], 0.75)
            self.assertNotIn("success_rate", brief)
            self.assertEqual(brief["verification"]["overshoot_frac"], [1.0])
            self.assertEqual(brief["seeds"], record["seeds"])
            self.assertNotEqual(path.read_bytes(), full_before)
            self.assertEqual(brief_path, path)
            self.assertEqual(brief["series"], record["series"])
            self.assertFalse(path.with_name("AUV_results_brief.json").exists())
            self.assertEqual(archive.read_bytes(), archive_before)
            first_brief = brief_path.read_bytes()
            finalize.update_results(path, {"AUV.json": artifact})
            self.assertEqual(brief_path.read_bytes(), first_brief)

    def test_independent_seed_means_and_two_dashboard_figures(self):
        rows = [{"model": "model-a", "seed": seed, "artifact": "AUV.json", "plant": "AUV",
                 "test_id": "test_1", "complexity": 1, "initial_condition": [0],
                 "references": [{"output": "depth", "expr": "1"}], "uncertainty": [],
                 "disturbance": [], "success_rate_1": first, "success_rate_2": second,
                 "rmse_by_output": [0.1], "total_tokens": 10, "cost_usd": 0.0, "cost_simulated": False}
                for seed, first, second in ((42, 0.9, 0.75), (123, 0.6, 0.0), (2024, 0.0, None))]
        case = dashboard._success_cases([("model-a", rows)])[0]
        self.assertEqual(dashboard.mean_case_success(rows, case, "success_rate_1")[0], 0.5)
        self.assertEqual(dashboard.mean_case_success(rows, case, "success_rate_2")[0], 0.375)
        with tempfile.TemporaryDirectory() as directory:
            outputs = dashboard.create_dashboards(rows, Path(directory) / "dashboard.png", show_plot=False)
            self.assertEqual([p.name for p in outputs],
                             ["dashboard_success_rate_1.png", "dashboard_success_rate_2.png"])
            self.assertTrue(all(p.stat().st_size > 1000 for p in outputs))
        self.assertFalse(plt.get_fignums())

    def test_export_limit_keeps_full_samples_and_restores_environment(self):
        settings = {"simulation_time": 50.0, "dt": 0.01}
        key = "LABCD_ADAPTIVE_SERIES_MAX_POINTS"
        with patch.dict(os.environ, {key: "100"}):
            with self.assertRaises(RuntimeError):
                with benchmark.full_series_export(settings):
                    self.assertGreaterEqual(int(os.environ[key]), 5001)
                    raise RuntimeError("test restoration")
            self.assertEqual(os.environ[key], "100")


if __name__ == "__main__":
    unittest.main()
