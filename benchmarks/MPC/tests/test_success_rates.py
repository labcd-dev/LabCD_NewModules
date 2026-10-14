"""Both scores and seed-averaged plots, using synthetic reports only."""

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.MPC import finalize_results as finalize
from benchmarks.MPC import plot_dashboard as dashboard
from benchmarks.MPC.success_rate.success_rate_1 import calculate_success_rate_1
from benchmarks.MPC.success_rate.success_rate_2 import calculate_success_rate_2, evaluate_fixed_targets


class SuccessRateTests(unittest.TestCase):
    def setUp(self):
        self.outputs = ["angle"]
        self.targets = {"mse_max": 0.01, "settling_time_max_s": 3.0,
                        "overshoot_max_pct": 10.0, "control_effort_max": 0.25}
        self.verification = {
            "outputs": ["angle", "other"], "tracking_mse": {"full": [0.01, 1000.0]},
            "task_scale": [1.0, 1.0], "task_scale_basis": "mse",
            "raw_result": {"unstable": False, "trajectory_file": "trajectories/test.npz",
                           "metrics": {"mse": 500.0, "settling_time": 3.0, "settled": True,
                                       "overshoot": 0.1, "overshoot_meaningful": True,
                                       "control_effort": 0.25}},
        }

    def test_rate_1_keeps_existing_formula_and_legacy_rms_conversion(self):
        self.assertEqual(calculate_success_rate_1(self.verification, self.outputs),
                         round(math.exp(-0.1), 6))
        legacy = copy.deepcopy(self.verification)
        legacy["task_scale"] = [2.0, 1.0]
        legacy.pop("task_scale_basis")
        self.assertEqual(calculate_success_rate_1(legacy, self.outputs),
                         round(math.exp(-10 * 0.01 / 4), 6))
        legacy["raw_result"]["unstable"] = True
        self.assertEqual(calculate_success_rate_1(legacy, self.outputs), 0.0)

    def test_boundaries_and_controlled_output_mse_ignore_aggregate_and_other_states(self):
        result = evaluate_fixed_targets(self.verification, self.targets, self.outputs)
        self.assertEqual(result["success_rate_2"], 1.0)
        self.assertEqual(result["passed_targets"], 4)
        self.assertTrue(result["all_targets_passed"])
        self.assertEqual(result["checks"]["mse_max"]["measured_value"], 0.01)
        self.assertEqual(result["checks"]["overshoot_max_pct"]["measured_value"], 10.0)
        self.assertEqual(result["checks"]["control_effort_max"]["measured_value"], 0.25)

    def test_three_passes_of_four_is_075_not_full_task_pass(self):
        self.verification["raw_result"]["metrics"]["control_effort"] = 0.3
        result = evaluate_fixed_targets(self.verification, self.targets, self.outputs)
        self.assertEqual(result["success_rate_2"], 0.75)
        self.assertEqual(result["failed_targets"], 1)
        self.assertFalse(result["all_targets_passed"])

    def test_unsettled_fallback_time_never_passes(self):
        metrics = self.verification["raw_result"]["metrics"]
        metrics.update(settled=False, settling_time=1.0)
        result = evaluate_fixed_targets(self.verification, self.targets, self.outputs)
        self.assertEqual(result["success_rate_2"], 0.75)
        self.assertFalse(result["checks"]["settling_time_max_s"]["passed"])

    def test_multiple_outputs_use_mean_mse_not_max_or_normalized_mse(self):
        self.verification["tracking_mse"]["full"] = [0.01, 0.03]
        self.assertEqual(calculate_success_rate_2(
            self.verification, {"mse_max": 0.02}, ["other", "angle"]), 1.0)
        details = evaluate_fixed_targets(self.verification, {"mse_max": 0.02}, ["other", "angle"])
        self.assertEqual(details["checks"]["mse_max"]["per_output_mse"], {"other": 0.03, "angle": 0.01})

    def test_unavailable_check_is_null_not_inflated_score_and_disabled_checks_omitted(self):
        self.verification["raw_result"]["metrics"]["overshoot_meaningful"] = False
        result = evaluate_fixed_targets(self.verification, self.targets, self.outputs)
        self.assertIsNone(result["success_rate_2"])
        self.assertEqual(result["passed_targets"], 3)
        self.assertEqual(result["total_targets"], 4)
        self.assertEqual(result["unavailable_targets"], ["overshoot_max_pct"])
        limits = {**self.targets, "overshoot_max_pct": None}
        self.assertEqual(calculate_success_rate_2(self.verification, limits, self.outputs), 1.0)
        self.assertIsNone(calculate_success_rate_2(self.verification, {}, self.outputs))
        self.assertIsNone(calculate_success_rate_2(None, self.targets, self.outputs))

    def test_malformed_limits_raise_and_nonfinite_metrics_do_not_pass(self):
        for limits in ([], {"unknown_target": 1}, {"mse_max": True}, {"mse_max": -1},
                       {"mse_max": float("nan")}, {"mse_max": float("inf")}):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                evaluate_fixed_targets(self.verification, limits, self.outputs)
        for value in (True, -1, None, float("inf"), float("nan")):
            with self.subTest(value=value):
                self.verification["tracking_mse"]["full"][0] = value
                self.assertIsNone(calculate_success_rate_2(
                    self.verification, {"mse_max": 0.01}, self.outputs))

    def test_numerical_failure_cannot_earn_target_credit(self):
        self.verification["raw_result"]["unstable"] = True
        details = evaluate_fixed_targets(self.verification, self.targets, self.outputs)
        self.assertEqual(details["success_rate_2"], 0.0)
        self.assertEqual(details["failed_targets"], 4)
        self.assertTrue(details["numerically_failed"])

    def test_finalization_saves_both_scores_idempotently_without_touching_metrics(self):
        artifact = {"system_name": "Plant", "metadata": {"outputs": self.outputs},
                    "simulation_parameters": {"fixed_targets": self.targets},
                    "tests": {"combinations": [{"test_id": "test_1", "initial_condition": [1.0],
                             "target": [0.0], "scenario_level": 1, "total_complexity": 2}]}}
        record = {"artifact": "Plant.json", "model": {"name": "Plant"},
                  "test_id": "test_1", "initial_condition": [1.0], "target": [0.0],
                  "scenario_level": 1, "seeds": {"run": 42}, "success_rate": 0.5,
                  "verification": copy.deepcopy(self.verification)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model" / "seed42" / "Plant_results.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps([record]), encoding="utf-8")
            finalize.update_results(path, {"Plant.json": artifact})
            updated = json.loads(path.read_text(encoding="utf-8"))[0]
            self.assertEqual(updated["success_rate_1"], round(math.exp(-0.1), 6))
            self.assertEqual(updated["success_rate_2"], 1.0)
            self.assertNotIn("success_rate", updated)
            self.assertEqual(updated["verification"], record["verification"])
            self.assertEqual(updated["seeds"], record["seeds"])
            first_write = path.read_bytes()
            finalize.update_results(path, {"Plant.json": artifact})
            self.assertEqual(path.read_bytes(), first_write)
            self.assertEqual(list(path.parent.glob("*_full.json")), [])

    def test_each_score_has_its_own_seed_mean_and_two_saved_figures(self):
        rows = [{"model": "model-a", "seed": seed, "artifact": "Plant.json", "plant": "Plant",
                 "test_id": "test_1", "complexity": 2, "initial_condition": [1.0],
                 "target": [0.0], "scenario_level": 1, "success_rate_1": first,
                 "success_rate_2": second, "total_tokens": 10, "cost_usd": 0.0,
                 "cost_simulated": False, "rmse": 0.1}
                for seed, first, second in ((42, 0.9, 0.75), (123, 0.6, 0.25), (2024, 0.0, 0.5))]
        case = dashboard._success_cases([("model-a", rows)])[0]
        for field in dashboard.SCORE_FIELDS:
            self.assertAlmostEqual(dashboard.mean_case_success(rows, case, field)[0], 0.5)
            figure, axis = plt.subplots()
            try:
                bars = dashboard._draw_success(axis, [("model-a", rows)], field)
                self.assertAlmostEqual(next(iter(bars)).get_height(), 50.0)
                self.assertIn(field.replace("_", " ").title(), axis.get_title())
            finally:
                plt.close(figure)
        rows[2]["success_rate_2"] = None
        self.assertAlmostEqual(dashboard.mean_case_success(rows, case, "success_rate_2")[0], 0.5)
        with tempfile.TemporaryDirectory() as directory:
            paths = dashboard.create_dashboards(rows, Path(directory) / "dashboard.png", show_plot=False)
            self.assertEqual([path.name for path in paths],
                             ["dashboard_success_rate_1.png", "dashboard_success_rate_2.png"])
            self.assertTrue(all(path.stat().st_size > 1000 for path in paths))
        self.assertFalse(plt.get_fignums())

    def test_conflicting_target_definitions_are_rejected_before_averaging(self):
        rows = [{"model": "model-a", "seed": seed, "artifact": "Plant.json", "plant": "Plant",
                 "test_id": "test_1", "complexity": 2, "success_rate_2": 1.0,
                 "success_rate_2_details": {
                     "mse_scope": "controlled_outputs", "outputs": ["angle"],
                     "checks": {"mse_max": {"limit": limit}}}}
                for seed, limit in ((42, 0.01), (123, 0.1))]
        figure, axis = plt.subplots()
        try:
            with self.assertRaisesRegex(ValueError, "Different fixed-target definitions"):
                dashboard._draw_success(axis, [("model-a", rows)], "success_rate_2")
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
