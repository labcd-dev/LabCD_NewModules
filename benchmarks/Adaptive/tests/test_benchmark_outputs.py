"""Full-output checkpoint tests; no real LLM/API calls."""

from contextlib import nullcontext, redirect_stderr, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
import numpy as np
import sympy as sp

from benchmarks.Adaptive import adaptive_benchmark as benchmark


class BenchmarkOutputTests(unittest.TestCase):
    def setUp(self):
        self.path = benchmark.ARTIFACTS_DIR / "InvPendulum.json"
        self.artifact = json.loads(self.path.read_text(encoding="utf-8"))
        self.test = self.artifact["tests"]["combinations"][0]
        self.config = {"folder_name": "mock", "provider": "mock", "model": "mock-adaptive"}

    def test_message_metadata_and_symbolic_values_are_not_discarded(self):
        message = AIMessage(content="proposal", id="answer-id",
            tool_calls=[{"name": "design", "args": {"gain": 2}, "id": "call-1", "type": "tool_call"}],
            response_metadata={"model_name": "test-model", "finish_reason": "tool_calls"},
            usage_metadata={"input_tokens": 12, "output_tokens": 3, "total_tokens": 15})
        saved = benchmark.json_safe(message)
        self.assertEqual(saved["tool_calls"][0]["args"], {"gain": 2})
        self.assertEqual(saved["usage_metadata"]["total_tokens"], 15)
        self.assertEqual(saved["response_metadata"]["model_name"], "test-model")
        self.assertEqual(saved["id"], "answer-id")
        self.assertEqual(benchmark.json_safe(sp.Symbol("x") + 1), "x + 1")
        self.assertEqual(benchmark.json_safe(sp.Matrix([[1, 2]])), [["1", "2"]])
        self.assertEqual(benchmark.json_safe(np.array([1, np.nan])), [1, None])

    def test_complete_core_returns_series_events_and_console_are_saved(self):
        message = AIMessage(content="full report", response_metadata={"extra": 1})
        result = {"messages": [message], "final_metrics": {"success": True},
                  "series": {"channels": {"x": {"data": np.ones((3000, 2))}}},
                  "abstract": "abstract", "future_field": {"keep": [1, 2]}}
        usage = {"total": {"total_tokens": 99}, "future_usage": "keep"}
        history = [{"round": 0, "future_history": "keep"}]
        best = {"tuning": {"c_gains": [2, 4]}, "round": 0,
                "metrics": {"success": True},
                "components": {"control_law": "u = -x", "stability": "proof"},
                "future_best": "keep"}

        def pipeline(**kwargs):
            event = {"kind": "stage_done", "args": {"values": [1]}}
            kwargs["on_event"](event)
            event["args"]["values"][0] = 99
            print("core stdout")
            print("core stderr", file=__import__("sys").stderr)
            return result, usage, history, best

        with (patch.object(benchmark, "USE_MOCK_AGENTS", True),
              patch.object(benchmark, "build_system_spec", return_value={"dynamics": {}}),
              patch("benchmarks.Adaptive.mock.tuner_agent.run_full_pipeline", side_effect=pipeline),
              patch("benchmarks.Adaptive.fast_report.fast_two_link_report", return_value=nullcontext()),
              redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
            saved = benchmark.run_one(self.path, self.artifact, self.test, self.config, 42)
        core = saved["core_outputs"]
        self.assertEqual(core["result"], benchmark.json_safe(result))
        self.assertEqual(core["usage"], usage)
        self.assertEqual(core["tuning_history"], history)
        self.assertEqual(core["best"], best)
        self.assertEqual(core["events"][0]["args"]["values"], [1])
        self.assertEqual(core["console_output"], {"stdout": "core stdout\n", "stderr": "core stderr\n"})
        self.assertEqual(len(saved["series"]["channels"]["x"]["data"]), 3000)
        self.assertTrue(saved["pipeline_completed"])
        json.dumps(saved, allow_nan=False)

    def test_saved_records_follow_accepted_fields_not_complete_internal_returns(self):
        record = {
            "artifact": "InvPendulum.json", "test_id": "test_1", "plant": "Pendulum",
            "pipeline_completed": True, "verification": {"success": True, "unknown_metric": 99},
            "applied_system_spec": {"dynamics": {"parameters": {"mass": 1, "gravity": 9.81}}},
            "tuning": {"best_params": {"c_gains": [2, 4]}, "best_metrics": {"success": True},
                       "best_round": 0, "history": [{"round": 0}],
                       "tuning_history": [{"round": 0}],
                       "agent_messages": [{"content": "report", "response_metadata": {"model": "mock"}}],
                       "token_usage": {"by_agent": {"Design Agent": {"model": "mock"}}}},
            "core_outputs": {"result": {"series": self.make_series(3)}, "events": ["unused"]},
            "series": self.make_series(3), "unknown_top_level": "not accepted",
        }
        with tempfile.TemporaryDirectory() as directory, patch.object(benchmark, "RESULTS_DIR", Path(directory)):
            path = benchmark.save_artifact_results(self.path, self.config, [record])
            saved = json.loads(path.read_text())[0]
            self.assertNotIn("core_outputs", saved)
            self.assertNotIn("unknown_top_level", saved)
            self.assertNotIn("unknown_metric", saved["verification"])
            self.assertNotIn("history", saved["tuning"])
            self.assertNotIn("tuning_history", saved["tuning"])
            self.assertEqual(saved["tuning"]["best_params"], {"c_gains": [2, 4]})
            self.assertEqual(saved["tuning"]["agent_messages"][0]["response_metadata"], {"model": "mock"})
            self.assertEqual(saved["applied_system_spec"]["dynamics"]["parameters"], {"mass": 1, "gravity": 9.81})
            self.assertNotIn("error", saved)
            self.assertNotIn("_description", saved["series"])
            self.assertTrue((path.parent / saved["series"]["trajectory_file"]).is_file())
            for details in saved["series"]["channels"].values():
                self.assertNotIn("data", details)
            # Persistence filtering does not alter the core's in-memory returns.
            self.assertIn("core_outputs", record)
            self.assertIn("history", record["tuning"])

    def test_accepted_projection_preserves_optional_null_empty_and_repeated_values(self):
        structure = benchmark.load_accepted_template()
        record = {"verification": None, "series": None, "tuning": None,
                  "test": {"targets": [{"output": "first", "expr": "0"},
                                       {"output": "second", "expr": "1"}], "disturbance": []},
                  "error": {"type": "RuntimeError", "message": "failure", "traceback": "trace"},
                  "core_outputs": {"events": [1]}, "_description": "do not save"}
        saved = benchmark.accepted_result_record(record, structure)
        self.assertIsNone(saved["verification"])
        self.assertIsNone(saved["series"])
        self.assertIsNone(saved["tuning"])
        self.assertEqual(saved["test"], record["test"])
        self.assertEqual(saved["error"], record["error"])
        self.assertNotIn("core_outputs", saved)
        self.assertNotIn("_description", saved)

    def test_missing_or_invalid_template_stops_before_any_run(self):
        with tempfile.TemporaryDirectory() as directory:
            template = Path(directory) / "accepted_results_template.json"
            with (patch.object(benchmark, "RUN_BENCHMARK", True),
                  patch.object(benchmark, "ACCEPTED_TEMPLATE_PATH", template),
                  patch.object(benchmark, "selected_artifacts") as select,
                  patch.object(benchmark, "run_one") as run):
                with self.assertRaises(FileNotFoundError):
                    benchmark.main()
                select.assert_not_called()
                run.assert_not_called()
                template.write_text("not valid JSON", encoding="utf-8")
                with self.assertRaises(json.JSONDecodeError):
                    benchmark.main()
                select.assert_not_called()
                run.assert_not_called()

    def test_pipeline_exception_preserves_progress_and_traceback(self):
        def pipeline(**kwargs):
            kwargs["on_event"]({"kind": "stage_start", "stage": "build"})
            print("before failure")
            raise RuntimeError("simulation failed")

        with (patch.object(benchmark, "USE_MOCK_AGENTS", True),
              patch.object(benchmark, "build_system_spec", return_value={}),
              patch("benchmarks.Adaptive.mock.tuner_agent.run_full_pipeline", side_effect=pipeline),
              redirect_stdout(io.StringIO())):
            saved = benchmark.run_one(self.path, self.artifact, self.test, self.config, 42)
        self.assertFalse(saved["pipeline_completed"])
        self.assertEqual(saved["core_outputs"]["events"][0]["stage"], "build")
        self.assertIn("before failure", saved["core_outputs"]["console_output"]["stdout"])
        self.assertIn("RuntimeError: simulation failed", saved["error"]["traceback"])

    def test_report_paths_and_unrelated_files_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(benchmark, "RESULTS_DIR", Path(directory)):
            old = benchmark.save_artifact_results(Path("AUV.json"), self.config, [{"plant": "AUV"}])
            untouched = old.read_bytes()
            path = benchmark.save_artifact_results(self.path, self.config, [{"test_id": "test_1"}])
            path = benchmark.save_artifact_results(self.path, self.config, [{"test_id": "test_2"}])
            self.assertEqual(path, Path(directory) / "mock" / "InvPendulum_results.json")
            self.assertEqual(json.loads(path.read_text()), [{"test_id": "test_2"}])
            self.assertEqual(old.read_bytes(), untouched)
            self.assertFalse(list(Path(directory).rglob("*.tmp")))
            self.assertFalse((path.parent / "trajectories").exists())
            with self.assertRaises(ValueError):
                benchmark.save_artifact_results(self.path, {"folder_name": "../escape"}, [])

    def test_main_checkpoints_each_artifact_per_model_and_restores_environment(self):
        artifacts = [(Path("AUV.json"), {"system_name": "AUV", "tests": {"combinations": [
            {"test_id": "a"}, {"test_id": "b"}]}}),
            (Path("DCMotor.json"), {"system_name": "Motor", "tests": {"combinations": [{"test_id": "c"}]}})]
        models = [{"folder_name": name, "model": name, "provider": "openai"}
                  for name in ("model-a", "model-b")]

        def run(path, artifact, test, config, seed):
            return {"test_id": test["test_id"], "pipeline_completed": True}

        with (tempfile.TemporaryDirectory() as directory,
              patch.object(benchmark, "RESULTS_DIR", Path(directory)),
              patch.object(benchmark, "RUN_BENCHMARK", True),
              patch.object(benchmark, "RUN_SEEDS", [42, 123, 2024]),
              patch.object(benchmark, "selected_artifacts", return_value=artifacts),
              patch.object(benchmark, "selected_models", return_value=models),
              patch.object(benchmark, "role_environment", return_value={"ADAPTIVE_TEST_ROUTING": "inside"}),
              patch.dict(os.environ, {"ADAPTIVE_TEST_ROUTING": "original"}),
              patch.object(benchmark, "run_one", side_effect=run),
              patch.object(benchmark, "save_artifact_results", wraps=benchmark.save_artifact_results) as save,
              redirect_stdout(io.StringIO())):
            benchmark.main()
            self.assertEqual(save.call_count, 18)
            self.assertEqual(os.environ["ADAPTIVE_TEST_ROUTING"], "original")
            for model in models:
                for seed in benchmark.RUN_SEEDS:
                    folder = Path(directory) / model["folder_name"] / f"seed{seed}"
                    self.assertEqual(len(json.loads((folder / "AUV_results.json").read_text())), 2)
                    self.assertEqual(len(json.loads((folder / "DCMotor_results.json").read_text())), 1)
                    self.assertFalse((folder / "batch_tuning_results.json").exists())

    def test_real_controller_offline_mock_smoke_saves_external_series(self):
        artifact = copy.deepcopy(self.artifact)
        artifact["simulation_parameters"].update(simulation_time=0.2, dt=0.01, tuning_rounds=1)
        test = artifact["tests"]["combinations"][0]
        test["initial_condition"] = [0.05, 0.0]
        test["targets"] = [{"output": "theta", "expr": "0.0"}]
        with (patch.object(benchmark, "USE_MOCK_AGENTS", True),
              patch.object(benchmark, "ENABLE_TUNING", True),
              patch.dict(os.environ, {"LABCD_ADAPTIVE_SHOW_PLOTS": "0", "MPLBACKEND": "Agg"}),
              redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
            saved = benchmark.run_one(self.path, artifact, test, self.config, 42)
        self.assertNotIn("error", saved)
        self.assertTrue(saved["pipeline_completed"])
        self.assertTrue(saved["core_outputs"]["events"])
        self.assertEqual(saved["series"], saved["core_outputs"]["result"]["series"])
        self.assertTrue(saved["series"]["channels"]["t"]["data"])
        self.assertIn("control_law", saved["core_outputs"]["best"]["components"])
        with tempfile.TemporaryDirectory() as directory, patch.object(benchmark, "RESULTS_DIR", Path(directory)):
            path = benchmark.save_artifact_results(self.path, self.config, [saved])
            report = json.loads(path.read_text())[0]
            self.assertEqual(report["verification"], saved["verification"])
            self.assertNotIn("core_outputs", report)
            self.assertNotIn("history", report["tuning"])
            self.assertNotIn("data", report["series"]["channels"]["x"])
            with np.load(path.parent / report["series"]["trajectory_file"], allow_pickle=False) as arrays:
                self.assertEqual(set(arrays.files), {"times", "states", "outputs", "reference", "inputs"})
                np.testing.assert_array_equal(arrays["states"], saved["series"]["channels"]["x"]["data"])
                self.assertEqual(len(arrays["times"]), saved["series"]["n_points"])

    def make_series(self, count=3001):
        return {"version": 1, "dt": 0.01, "t_end": (count - 1) * 0.01,
                "n_points": count, "downsampled": False, "max_points": 4000,
                "channels": {
                    "t": {"label": "time", "unit": "s", "data": np.arange(count) * 0.01},
                    "x": {"label": "states", "names": ["theta", "theta_dot"],
                          "data": np.arange(count * 2, dtype=float).reshape(count, 2)},
                    "y": {"label": "outputs", "names": ["theta"], "data": np.ones((count, 1))},
                    "ref": {"label": "references", "names": ["theta"], "data": np.zeros((count, 1))},
                    "u": {"label": "inputs", "names": ["torque"], "data": np.full((count, 1), 2.5)},
                }}

    def test_external_series_roundtrip_all_samples_and_metadata_without_mutating_returns(self):
        series = self.make_series()
        series["channels"]["x"]["data"][0, 0] = np.nan
        series = benchmark.json_safe(series)
        records = [{"test_id": "test_1", "series": series,
                    "verification": {"success": False},
                    "core_outputs": {"result": {"series": series, "future_field": "keep"}}}]
        original = copy.deepcopy(records)
        with tempfile.TemporaryDirectory() as directory, patch.object(benchmark, "RESULTS_DIR", Path(directory)):
            path = benchmark.save_artifact_results(self.path, self.config, records)
            saved = json.loads(path.read_text())[0]
            self.assertNotIn("core_outputs", saved)
            self.assertEqual(saved["verification"], {"success": False})
            self.assertEqual(saved["series"]["trajectory_file"], "trajectories/InvPendulum_test_1_verification.npz")
            for channel in series["channels"]:
                metadata = saved["series"]["channels"][channel]
                self.assertNotIn("data", metadata)
                self.assertEqual(metadata["label"], series["channels"][channel]["label"])
            with np.load(path.parent / saved["series"]["trajectory_file"], allow_pickle=False) as archive:
                self.assertEqual(len(archive["times"]), 3001)
                for channel, metadata in saved["series"]["channels"].items():
                    np.testing.assert_array_equal(archive[metadata["array_key"]],
                                                  np.asarray(series["channels"][channel]["data"], dtype=float))
            self.assertEqual(records, original)
            self.assertFalse(list(Path(directory).rglob("*.tmp")))
            self.assertEqual(len(list(Path(directory).rglob("*.npz"))), 1)

    def test_trajectory_paths_isolate_tests_and_models(self):
        records = [{"test_id": name, "series": self.make_series(3)} for name in ("test_1", "test_2")]
        with tempfile.TemporaryDirectory() as directory, patch.object(benchmark, "RESULTS_DIR", Path(directory)):
            for config in (self.config, {"folder_name": "another-model"}):
                path = benchmark.save_artifact_results(self.path, config, records)
                saved = json.loads(path.read_text())
                self.assertNotEqual(saved[0]["series"]["trajectory_file"], saved[1]["series"]["trajectory_file"])
                self.assertTrue(all((path.parent / item["series"]["trajectory_file"]).is_file() for item in saved))
            self.assertEqual(len(list(Path(directory).rglob("*.npz"))), 4)

    def test_invalid_trajectory_does_not_replace_existing_json_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(benchmark, "RESULTS_DIR", Path(directory)):
            path = benchmark.save_artifact_results(self.path, self.config, [{"test_id": "previous"}])
            original = path.read_bytes()
            for test_id, series in (("../escape", self.make_series(3)),
                                    ("test_1", {"channels": {"x": {"data": [[1]]}}})):
                with self.assertRaises(ValueError):
                    benchmark.save_artifact_results(self.path, self.config, [{"test_id": test_id, "series": series}])
                self.assertEqual(path.read_bytes(), original)
            series = self.make_series(3)
            series["channels"]["u"]["data"] = [[1], [2]]
            with self.assertRaisesRegex(ValueError, "align"):
                benchmark.save_artifact_results(self.path, self.config, [{"test_id": "test_1", "series": series}])
            series = self.make_series(3)
            series["channels"]["x"]["data"] = [["bad", "data"]] * 3
            with self.assertRaisesRegex(ValueError, "numeric"):
                benchmark.save_artifact_results(self.path, self.config, [{"test_id": "test_1", "series": series}])
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse(list(Path(directory).rglob("*.npz")))


if __name__ == "__main__":
    unittest.main()
