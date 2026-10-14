"""Check the explanatory guide against actual saved Adaptive output shapes."""

import json
import unittest

from benchmarks.Adaptive import adaptive_benchmark as benchmark


class ResultsTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template_path = benchmark.ADAPTIVE_DIR / "json_reports" / "results_template.json"
        cls.template = json.loads(cls.template_path.read_text(encoding="utf-8"))

    def test_one_explanatory_record_with_only_nonempty_description_leaves(self):
        self.assertIsInstance(self.template, list)
        self.assertEqual(len(self.template), 1)

        def check(value, path):
            if isinstance(value, dict):
                self.assertTrue(value, path)
                for key, child in value.items():
                    check(child, f"{path}.{key}")
            elif isinstance(value, list):
                self.assertTrue(value, path)
                for child in value:
                    check(child, path + "[]")
            else:
                self.assertIsInstance(value, str, path)
                self.assertTrue(value.strip(), path)

        check(self.template, "template")

    def test_every_saved_mock_field_has_a_description(self):
        reports = sorted((benchmark.RESULTS_DIR / "mock").glob("*_results.json"))
        reports = [path for path in reports if path.name != "batch_tuning_results.json"]
        if not reports:
            self.skipTest("No saved per-artifact mock outputs available in this checkout.")

        def cover(actual, guide, path):
            # A description can cover a plant/provider-dependent arbitrary value.
            if isinstance(guide, str):
                return
            if isinstance(actual, dict):
                self.assertIsInstance(guide, dict, path)
                wildcard = next((key for key in guide if key.startswith("<") and key.endswith(">")), None)
                for key, value in actual.items():
                    self.assertTrue(key in guide or wildcard is not None, f"Missing {path}.{key}")
                    cover(value, guide[key] if key in guide else guide[wildcard], f"{path}.{key}")
            elif isinstance(actual, list):
                self.assertIsInstance(guide, list, path)
                for index, value in enumerate(actual):
                    cover(value, guide[min(index, len(guide) - 1)], f"{path}[{index}]")

        for path in reports:
            for record in json.loads(path.read_text(encoding="utf-8")):
                with self.subTest(report=path.name, test=record.get("test_id")):
                    cover(record, self.template[0], path.name)

    def test_trial_final_and_optional_field_explanations_are_distinct(self):
        record = self.template[0]
        self.assertIn("round 0", record["tuning"]["history"][0]["round"])
        self.assertIn("Best candidate", record["tuning"]["best_metrics"]["steady_rms"][0])
        self.assertIn("Final simulation", record["verification"]["steady_rms"][0])
        self.assertIn("safe initial fallback", record["verification"]["success"])
        self.assertIn("NOT exp()", record["verification"]["tracking_pct"]["full"][0])
        self.assertIn("RMS, not MSE", record["verification"]["task_scale"][0])
        self.assertIn("NOT the exact real model", record["agent_mode"])
        self.assertNotIn("success_rate", record)
        self.assertIn("does NOT enforce", record["tuning_inputs"]["fixed_targets"]["mse_max"])
        self.assertIn("zero placeholders", record["verification"]["tracking_mse"]["full"][0])
        self.assertIn("Optional", record["core_outputs"]["events"][0]["tool_called"])
        self.assertIn("Failure-only", record["error"]["type"])
        self.assertIn("per INTERNAL STATE", record["verification"]["estimator_residual_rms"][0])

    def test_accepted_template_keeps_series_examples_and_explains_external_storage(self):
        path = benchmark.ADAPTIVE_DIR / "json_reports" / "accepted_results_template.json"
        record = json.loads(path.read_text(encoding="utf-8"))[0]
        self.assertNotIn("core_outputs", record)
        self.assertNotIn("history", record["tuning"])
        series = record["series"]
        self.assertIn("trajectories", series["trajectory_file"])
        self.assertIn("separately", series["_description"])
        for channel, array_key in {"t": "times", "x": "states", "y": "outputs",
                                   "ref": "reference", "u": "inputs"}.items():
            details = series["channels"][channel]
            self.assertIn(array_key, details["array_key"])
            self.assertIn("Documentation only", str(details["data"]))
            self.assertIn("not inline", str(details["data"]))

    def test_benchmark_loads_the_accepted_template_as_saving_structure(self):
        path = benchmark.ADAPTIVE_DIR / "json_reports" / "accepted_results_template.json"
        expected = json.loads(path.read_text(encoding="utf-8"))[0]
        self.assertEqual(benchmark.load_accepted_template(), expected)


if __name__ == "__main__":
    unittest.main()
