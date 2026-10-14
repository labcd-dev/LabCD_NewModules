"""Progress/checkpoint tests using temporary reports and stubbed runs only."""

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.MPC import mpc_benchmark as benchmark


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        for name, value in (("RESULTS_DIR", self.root), ("RUN_SEEDS", [42]),
                            ("USE_MOCK_AGENTS", False)):
            patcher = patch.object(benchmark, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.model = {"folder_name": "test-model", "model": "test-model"}
        self.path = Path("Plant.json")
        self.tests = [{"test_id": name, "initial_condition": [1.0],
                       "target": [0.0], "scenario_level": 1}
                      for name in ("test_1", "test_2")]
        self.artifact = {"system_name": "Plant", "tests": {"combinations": self.tests}}

    def record(self, test, error=None):
        result = {
            "artifact": self.path.name, "model": {"name": "Plant"},
            "agent_mode": "test-model", "seeds": {"run": 42}, **test,
            "tuning": {"completed": True, "best_params": {"Np": 10}},
            "verification": {"raw_result": {"metrics": {"Unstable": True}}},
        }
        if error:
            result["error"] = {"type": "RunError", "message": error}
        return result

    def snapshot(self):
        return json.loads((self.root / benchmark.PROGRESS_FILENAME).read_text(encoding="utf-8"))

    def plan(self):
        return benchmark.build_progress_plan([(self.path, self.artifact)], [self.model])

    def test_completion_is_pipeline_finish_not_controller_performance(self):
        record = self.record(self.tests[0])
        self.assertEqual(benchmark.completion_status(record), ("completed", None))
        for field in ("tuning", "verification"):
            malformed = {**record, field: 5}
            self.assertEqual(benchmark.completion_status(malformed)[0], "failed")
        record["tuning"]["completed"] = False
        self.assertEqual(benchmark.completion_status(record)[0], "failed")

    def test_saved_inventory_and_selection_matching(self):
        good = self.record(self.tests[0])
        different_seed = {**good, "seeds": {"run": 123}}
        different_target = {**good, "target": [2.0]}
        malformed = {**good, "model": 5}
        failed = self.record(self.tests[1], "simulation failed")
        benchmark.save_artifact_results(self.path, {**self.model, "run_seed": 42},
                                        [different_seed, different_target, malformed, good, failed])
        plan = self.plan()
        self.assertEqual([item["status"] for item in plan], ["completed", "failed"])
        benchmark.save_progress(plan, "ready")
        report = self.snapshot()
        self.assertEqual(report["summary"]["done_out_of_total"], "1/2")
        self.assertEqual(report["summary"]["progress_percent"], 50.0)
        self.assertEqual(report["remaining_combinations"], report["failed_combinations"])
        self.assertEqual(report["completed_combinations"][0]["result_file"],
                         "test-model/seed42/Plant_results.json")

    def test_unreadable_report_is_pending_not_completed(self):
        output = benchmark.result_directory({**self.model, "run_seed": 42}) / "Plant_results.json"
        benchmark.write_json_checkpoint(output, {"not": "a list"})
        plan = self.plan()
        self.assertTrue(all(item["status"] == "pending" for item in plan))
        self.assertTrue(all("saved_report_issue" in item for item in plan))

    def test_running_and_empty_counts(self):
        plan = self.plan()
        plan[0]["status"] = "running"
        benchmark.save_progress(plan, "running")
        report = self.snapshot()
        self.assertEqual(report["summary"]["done_out_of_total"], "0/2")
        self.assertEqual(report["summary"]["running"], 1)
        self.assertEqual(report["current_combination"]["test_id"], "test_1")
        benchmark.save_progress([], "ready")
        self.assertEqual(self.snapshot()["summary"]["progress_percent"], 0.0)

    def run_main(self, side_effect):
        with patch.object(benchmark, "RUN_BENCHMARK", True), \
                patch.object(benchmark, "selected_artifacts", return_value=[(self.path, self.artifact)]), \
                patch.object(benchmark, "selected_models", return_value=[self.model]), \
                patch.object(benchmark, "configure_model"), \
                patch.object(benchmark, "run_one", side_effect=side_effect), \
                redirect_stdout(io.StringIO()):
            benchmark.main()

    def test_main_keeps_failed_saved_run_remaining(self):
        self.run_main([self.record(self.tests[0]), self.record(self.tests[1], "failed")])
        report = self.snapshot()
        self.assertEqual(report["status"], "finished")
        self.assertEqual(report["summary"]["done_out_of_total"], "1/2")
        self.assertEqual(report["summary"]["failed"], 1)

    def test_interrupted_run_preserves_previous_completion(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_main([self.record(self.tests[0]), KeyboardInterrupt()])
        report = self.snapshot()
        self.assertEqual(report["status"], "interrupted")
        self.assertEqual(report["summary"]["done_out_of_total"], "1/2")
        self.assertEqual(report["remaining_combinations"][0]["status"], "pending")
        self.assertIsNone(report["current_combination"])
        self.assertEqual(report["last_error"]["type"], "KeyboardInterrupt")

    def test_unsaved_run_does_not_count_as_done(self):
        with patch.object(benchmark, "save_artifact_results", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                self.run_main([self.record(self.tests[0])])
        self.assertEqual(self.snapshot()["summary"]["done"], 0)
        self.assertEqual(self.snapshot()["status"], "interrupted")

    def test_atomic_checkpoint_preserves_old_report_on_replace_failure(self):
        output = self.root / "checkpoint.json"
        benchmark.write_json_checkpoint(output, {"old": True})
        with patch.object(Path, "replace", side_effect=OSError("cannot replace")):
            with self.assertRaises(OSError):
                benchmark.write_json_checkpoint(output, {"old": False})
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"old": True})
        self.assertEqual(list(self.root.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
