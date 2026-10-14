"""Output-MSE scoring and per-artifact finalization; no simulations/API calls."""

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.Adaptive import finalize_results as finalizer
from benchmarks.MPC import finalize_results as mpc_finalizer


class FinalizeResultsTests(unittest.TestCase):
    def test_mse_score_matches_current_mpc_and_averages_ratios_per_output(self):
        verification = {"outputs": ["angle", "position"],
                        "tracking_mse": {"full": [0.04, 0.09]}, "task_scale": [2.0, 3.0]}
        expected = round(math.exp(-10 * (0.04 / 4 + 0.09 / 9) / 2), 6)
        self.assertEqual(finalizer.calculate_success_rate(verification, ["position", "angle"]), expected)
        self.assertEqual(mpc_finalizer.calculate_success_rate(verification, ["position", "angle"]), expected)
        self.assertEqual(finalizer.output_task_scales_mse(verification), [4.0, 9.0])
        self.assertEqual(verification["task_scale"], [2.0, 3.0])

    def test_only_selected_outputs_and_no_repeated_squaring(self):
        verification = {"outputs": ["selected", "other"], "task_scale": [2.0, 4.0],
                        "tracking_mse": {"full": [0.04, 1000]}}
        self.assertEqual(finalizer.calculate_success_rate(verification, ["selected"]), round(math.exp(-0.1), 6))
        verification["task_scale_mse"] = [4.0, 16.0]
        self.assertEqual(finalizer.calculate_success_rate(verification, ["selected"]), round(math.exp(-0.1), 6))
        verification.pop("task_scale_mse")
        verification.update(task_scale=[4.0, 16.0], task_scale_basis="mse")
        self.assertEqual(finalizer.calculate_success_rate(verification, ["selected"]), round(math.exp(-0.1), 6))

    def test_health_zero_scales_and_invalid_metrics(self):
        verification = {"tracking_mse": {"full": [0.0]}, "task_scale": [0.0]}
        self.assertEqual(finalizer.calculate_success_rate(verification), 1.0)
        verification["tracking_mse"]["full"] = [0.1]
        self.assertEqual(finalizer.calculate_success_rate(verification), 0.0)
        for value in (None, True, -1, float("nan"), float("inf")):
            with self.subTest(value=value):
                self.assertIsNone(finalizer.calculate_success_rate({"tracking_mse": {"full": [value]}, "task_scale": [1]}))
        self.assertIsNone(finalizer.calculate_success_rate({"tracking_mse": {"full": [0.1]}, "task_scale": []}))
        self.assertEqual(finalizer.calculate_success_rate({"numerically_healthy": False}), 0.0)
        self.assertEqual(finalizer.calculate_success_rate({"success_checks": {"bounded": False}}), 0.0)
        # Missing the steady-state target is not a numerical failure.
        self.assertGreater(finalizer.calculate_success_rate({"success": False,
                           "tracking_mse": {"full": [0.01]}, "task_scale": [1]}), 0)

    def test_per_artifact_updates_in_place_and_adds_named_mse_scales(self):
        artifacts = finalizer.load_artifacts()
        artifact = artifacts["AUV.json"]
        test = artifact["tests"]["combinations"][0]
        record = {"artifact": "AUV.json", "plant": artifact["system_name"],
                  "test_id": test["test_id"], "test": copy.deepcopy(test),
                  "applied_system_spec": {"system_name": artifact["system_name"],
                                          "dynamics": {"outputs": ["depth"]}},
                  "verification": {"tracking_mse": {"full": [0.04]}, "task_scale": [2.0]},
                  "tuning": {}, "pipeline_completed": True}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "AUV_results.json"
            path.write_text(json.dumps([record]), encoding="utf-8")
            original = path.read_bytes()
            output = finalizer.update_results(path, artifacts)
            self.assertEqual(output, path)
            brief = json.loads(output.read_text())[0]
            self.assertEqual(brief["verification"]["outputs"], ["depth"])
            self.assertEqual(brief["verification"]["task_scale"], [2.0])
            self.assertEqual(brief["verification"]["task_scale_mse"], [4.0])
            self.assertEqual(brief["success_rate_1"], round(math.exp(-0.1), 6))
            self.assertNotEqual(path.read_bytes(), original)
            self.assertFalse(path.with_name("AUV_results_brief.json").exists())
            self.assertIn("applied_system_spec", brief)
            self.assertNotIn("dynamics", brief)
            self.assertEqual(finalizer.calculate_success_rate(brief["verification"]), brief["success_rate_1"])
            record["test"]["initial_condition"] = [99] * 3
            path.write_text(json.dumps([record]), encoding="utf-8")
            previous_brief = output.read_bytes()
            with self.assertRaisesRegex(ValueError, "conditions changed"):
                finalizer.update_results(path, artifacts)
            self.assertEqual(output.read_bytes(), previous_brief)

    def test_selection_prefers_per_artifact_and_mock_ignores_paid_models(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(finalizer, "RESULTS_DIR", Path(directory)):
            folder = Path(directory) / "mock"
            folder.mkdir()
            for name in ("AUV_results.json", "DCMotor_results.json", "AUV_results_brief.json", finalizer.INPUT_FILENAME):
                (folder / name).write_text("[]", encoding="utf-8")
            with patch.object(finalizer, "USE_MOCK_RESULTS", True):
                paths = finalizer.select_result_paths([Path("paid_model.json")])
            self.assertEqual([path.name for path in paths], ["AUV_results.json", "DCMotor_results.json"])

    def test_in_place_update_preserves_accepted_messages_usage_and_series(self):
        artifacts = finalizer.load_artifacts()
        artifact = artifacts["AUV.json"]
        test = artifact["tests"]["combinations"][0]
        record = {"artifact": "AUV.json", "plant": artifact["system_name"],
                  "test_id": test["test_id"], "test": copy.deepcopy(test),
                  "seeds": {"run": 42}, "summary": "keep the full report",
                  "tuning": {
                      "best_params": {"K": 2},
                      "agent_messages": [{"type": "ai", "content": "keep this message",
                                          "response_metadata": {"provider_detail": [1, 2]}}],
                      "token_usage": {"simulated": True, "total": {"total_tokens": 12}, "cost_usd": 0.0},
                  },
                  "verification": {"tracking_mse": {"full": [0.01]}, "task_scale": [1.0]},
                  "series": None, "core_outputs": {"unaccepted": "must not persist"}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model" / "seed42" / "AUV_results.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps([record]), encoding="utf-8")
            finalizer.update_results(path, artifacts)
            updated = json.loads(path.read_text(encoding="utf-8"))[0]
            self.assertEqual(updated["tuning"], record["tuning"])
            self.assertEqual(updated["test"], record["test"])
            self.assertEqual(updated["summary"], record["summary"])
            self.assertNotIn("core_outputs", updated)
            self.assertNotIn("token_counts", updated)
            self.assertNotIn("dynamics", updated)
            first = path.read_bytes()
            finalizer.update_results(path, artifacts)
            self.assertEqual(path.read_bytes(), first)
            self.assertEqual(list(path.parent.glob("*_brief.json")), [])

    def test_obsolete_brief_path_is_rejected_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "AUV_results_brief.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "canonical"):
                finalizer.update_results(path, {})
            self.assertEqual(path.read_text(encoding="utf-8"), "[]")


if __name__ == "__main__":
    unittest.main()
