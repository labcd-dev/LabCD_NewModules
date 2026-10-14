"""MPC-style per-case dashboard grouping and shared artifact rankings."""

import json
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.Adaptive import plot_dashboard as dashboard
from benchmarks.MPC import plot_dashboard as mpc_dashboard


class DashboardTests(unittest.TestCase):
    def rows(self):
        cases = [("AUV", "AUVDepthDynamics", 1, 0.2),
                 ("DCMotor", "DCMotorDynamics", 1, 0.9),
                 ("CartPole", "CartPolePendulum", 2, 0.7)]
        return [{"model": model, "artifact": name + ".json", "plant": plant,
                 "test_id": "test_1", "complexity": level, "success_rate": success,
                 "initial_condition": [0], "references": [{"output": "y", "expr": "1"}],
                 "rmse_by_output": [0.1], "total_tokens": 1000, "cost_usd": 0.002,
                 "cost_simulated": model == "mock"}
                for model in ("mock", "gpt-4.1-nano") for name, plant, level, success in cases]

    def grouped(self, rows):
        return [(model, [row for row in rows if row["model"] == model])
                for model in dict.fromkeys(row["model"] for row in rows)]

    def test_case_positions_match_mpc_and_separate_plants_at_equal_complexity(self):
        grouped = self.grouped(self.rows())
        cases = dashboard._success_cases(grouped)
        self.assertEqual(len(cases), 3)
        # Conditions differ between modules (output references vs full-state
        # targets); only the displayed case layout should be identical.
        displayed = lambda items: [{key: value for key, value in case.items() if key != "conditions"}
                                   for case in items]
        self.assertEqual(displayed(cases), displayed(mpc_dashboard._success_cases(grouped)))
        same_gap = cases[1]["position"] - cases[0]["position"]
        level_gap = cases[2]["position"] - cases[1]["position"]
        self.assertAlmostEqual(same_gap, 0.82 + dashboard.CASE_GAP)
        self.assertAlmostEqual(level_gap, 0.82 + dashboard.COMPLEXITY_GAP)
        self.assertGreater(level_gap, same_gap)

    def test_one_bar_per_case_model_without_averaging_different_plants(self):
        figure, axis = plt.subplots()
        try:
            selectable = dashboard._draw_success(axis, self.grouped(self.rows()))
            self.assertEqual(len(selectable), 6)
            self.assertTrue(all(len(matches) == 1 for _, _, matches in selectable.values()))
            heights = sorted(round(bar.get_height(), 6) for bar in selectable)
            self.assertEqual(heights, [20, 20, 70, 70, 90, 90])
            self.assertEqual(len(axis.get_xticklabels()), 3)
            self.assertTrue(any("AUV" in label.get_text() for label in axis.get_xticklabels()))
        finally:
            plt.close(figure)

    def test_conflicting_rankings_are_rejected_and_missing_success_is_not_zero(self):
        rows = self.rows()
        rows[3]["complexity"] = 3
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            dashboard._success_cases(self.grouped(rows))
        rows = self.rows()
        rows[0]["success_rate"] = None
        figure, axis = plt.subplots()
        try:
            selectable = dashboard._draw_success(axis, self.grouped(rows))
            self.assertEqual(len(selectable), 5)
            self.assertTrue(any(text.get_text() == "N/A" for text in axis.texts))
        finally:
            plt.close(figure)

    def test_load_accepted_result_keeps_identity_and_saves_three_panel_dashboard(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "mock"
            folder.mkdir()
            path = folder / "AUV_results.json"
            path.write_text(json.dumps([{"artifact": "AUV.json", "plant": "AUVDepthDynamics",
                "test_id": "test_1", "total_complexity": 1, "success_rate": 0.7,
                "applied_system_spec": {"dynamics": {"x0": [0], "references": [{"output": "depth", "expr": "1"}]}},
                "verification": {"tracking_mse": {"full": [0.01]}},
                "tuning": {"token_usage": {"total": {"total_tokens": 200}, "cost_usd": 0, "simulated": True}}}]), encoding="utf-8")
            rows = dashboard.load_dashboard_rows([path])
            self.assertEqual(rows[0]["artifact"], "AUV.json")
            self.assertEqual(rows[0]["rmse_by_output"], [0.1])
            self.assertEqual(rows[0]["total_tokens"], 200)
            output = Path(directory) / "plots" / "dashboard" / "adaptive_dashboard.png"
            result = dashboard.create_dashboard(rows, output, show_plot=False)
            self.assertEqual(result, output.resolve())
            self.assertGreater(output.stat().st_size, 1000)
            self.assertFalse(plt.get_fignums())

    def test_all_active_artifact_complexities_match_mpc_by_test_id(self):
        adaptive_dir = dashboard.ADAPTIVE_DIR / "artifacts"
        mpc_dir = mpc_dashboard.MPC_DIR / "artifacts"
        expected = {"AUV.json": 1, "DCMotor.json": 1, "CartPole.json": 2,
                    "InvPendulum.json": 2, "QuanserHelicopter.json": 3, "TwoLinkManipulator.json": 3}
        for path in adaptive_dir.glob("*.json"):
            adaptive = json.loads(path.read_text(encoding="utf-8"))
            mpc = json.loads((mpc_dir / path.name).read_text(encoding="utf-8"))
            self.assertEqual(adaptive["system_name"], mpc["system_name"])
            source = {test["test_id"]: test["total_complexity"] for test in mpc["tests"]["combinations"]}
            for test in adaptive["tests"]["combinations"]:
                self.assertEqual(test["total_complexity"], source[test["test_id"]])
                self.assertEqual(test["total_complexity"], expected[path.name])


if __name__ == "__main__":
    unittest.main()
