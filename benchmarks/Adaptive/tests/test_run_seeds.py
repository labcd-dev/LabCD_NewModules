"""Seed repetition, offline postprocessing and routing checks; no API calls."""

from contextlib import redirect_stdout
import copy
import io
import json
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
from benchmarks.Adaptive import plot_trajectories as trajectories


class AdaptiveSeedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for module in (benchmark, finalize, dashboard):
            patcher = patch.object(module, "RESULTS_DIR", self.root)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.path = benchmark.ARTIFACTS_DIR / "InvPendulum.json"
        self.artifact = json.loads(self.path.read_text(encoding="utf-8"))
        self.test = self.artifact["tests"]["combinations"][0]
        self.model = {"folder_name": "model-a", "provider": "openai", "model": "model-a"}

    def record(self, seed, model=None):
        config = model or self.model
        return {
            "artifact": self.path.name, "plant": self.artifact["system_name"],
            "test_id": self.test["test_id"], "test": copy.deepcopy(self.test),
            "agent_mode": config["provider"], "llm_model": config["model"],
            "seeds": {"run": seed, "llm": seed, "mock": None, "estimator": 0, "simulation": None},
            "total_complexity": self.test["total_complexity"],
            "applied_system_spec": {"system_name": self.artifact["system_name"], "dynamics": {
                "states": ["theta", "theta_dot"], "inputs": ["torque"], "outputs": ["theta"],
                "x0": self.test["initial_condition"], "references": self.test["targets"],
                "uncertainty": self.test["uncertainty"], "disturbance": self.test["disturbance"],
            }},
            "verification": {"tracking_mse": {"full": [0.01]}, "task_scale": [1.0],
                             "numerically_healthy": True},
            "tuning": {"best_params": {"c_gains": [2, 4]},
                       "token_usage": {"total": {"total_tokens": 10}, "simulated": True, "cost_usd": 0}},
            "pipeline_completed": True, "series": None,
        }

    def series(self, seed):
        return {"n_points": 2, "channels": {
            "t": {"data": [0.0, 1.0]},
            "x": {"names": ["theta", "theta_dot"], "data": [[seed, 0], [0, 0]]},
            "y": {"names": ["theta"], "data": [[seed], [0]]},
            "ref": {"names": ["theta"], "data": [[0], [0]]},
            "u": {"names": ["torque"], "data": [[seed], [0]]},
        }}

    def save(self, seed, model=None):
        model = model or self.model
        record = self.record(seed, model)
        record["series"] = self.series(seed)
        return benchmark.save_artifact_results(self.path, {**model, "run_seed": seed}, [record])

    def test_seed_validation_and_directory_safety(self):
        for seeds in ([], [42, 42], [True], [-1], [2**32], [1.5], "42", [[42]]):
            with patch.object(benchmark, "RUN_SEEDS", seeds), self.assertRaises(ValueError):
                benchmark.validate_settings()
        self.assertEqual(benchmark.result_directory({**self.model, "run_seed": 42}),
                         self.root / "model-a" / "seed42")
        for config in ({"folder_name": "../escape"}, {**self.model, "run_seed": "../escape"}):
            with self.assertRaises(ValueError):
                benchmark.result_directory(config)

    def test_initial_controller_only_is_completed_even_without_best_params(self):
        record = self.record(42)
        record["tuning"]["best_params"] = None
        record["verification"]["success"] = False
        self.assertEqual(benchmark.completion_status(record), ("completed", None))

    def test_real_llm_factory_gets_seed_for_every_role_and_restores_on_error(self):
        from backend_core.AgentAdaptive.agents import llm_factory
        original = llm_factory.build_llm
        with patch.object(benchmark, "USE_MOCK_AGENTS", False), \
                patch("labcd_agents.LLMFactory.create", return_value=object()) as create:
            for seed in (42, 123, 2024):
                with self.assertRaises(RuntimeError):
                    with benchmark.seeded_llm_routing(seed):
                        for role in ("design", "tuner", "reporter", "clarifier"):
                            llm_factory.build_llm(role)
                            self.assertEqual(create.call_args.kwargs["seed"], seed)
                            self.assertEqual(create.call_args.kwargs["temperature"],
                                             llm_factory._ROLES[role]["temperature"])
                        raise RuntimeError("force restoration")
                self.assertIs(llm_factory.build_llm, original)
        self.assertEqual(create.call_count, 12)

    def test_main_runs_each_model_seed_and_saves_progress(self):
        models = [self.model, {**self.model, "folder_name": "model-b", "model": "model-b"}]
        calls = []

        def run(path, artifact, test, config, seed):
            calls.append((config["folder_name"], config["run_seed"], seed))
            record = self.record(seed, config)
            record["series"] = self.series(seed)
            return record

        old = self.root / "model-a" / "InvPendulum_results.json"
        old.parent.mkdir()
        old.write_text("old report", encoding="utf-8")
        with patch.object(benchmark, "RUN_SEEDS", [42, 123, 2024]), \
                patch.object(benchmark, "USE_MOCK_AGENTS", False), \
                patch.object(benchmark, "selected_artifacts", return_value=[(self.path, self.artifact)]), \
                patch.object(benchmark, "selected_models", return_value=models), \
                patch.object(benchmark, "role_environment", return_value={"TEST_ROUTE": "inside"}), \
                patch.dict(os.environ, {"TEST_ROUTE": "outside"}), \
                patch.object(benchmark, "run_one", side_effect=run), redirect_stdout(io.StringIO()):
            benchmark.main()
            self.assertEqual(os.environ["TEST_ROUTE"], "outside")
        self.assertEqual(len(calls), 6)
        self.assertTrue(all(run_seed == seed for _, run_seed, seed in calls))
        self.assertEqual(old.read_text(encoding="utf-8"), "old report")
        progress = json.loads((self.root / benchmark.PROGRESS_FILENAME).read_text())
        self.assertEqual(progress["summary"]["done_out_of_total"], "6/6")
        with patch.object(benchmark, "USE_MOCK_AGENTS", False):
            plan = benchmark.build_progress_plan([(self.path, self.artifact)], models)
        self.assertTrue(all(item["status"] == "completed" for item in plan))

    def test_finalization_preserves_seed_full_file_and_npz(self):
        path = self.save(42)
        source = path.read_bytes()
        raw = json.loads(source)[0]
        archive = path.parent / raw["series"]["trajectory_file"]
        arrays_before = archive.read_bytes()
        output = finalize.update_results(path, {self.path.name: self.artifact})
        brief = json.loads(output.read_text())[0]
        self.assertEqual(output.parent, self.root / "model-a" / "seed42")
        self.assertEqual(output, path)
        self.assertEqual(brief["seeds"], raw["seeds"])
        self.assertIsNotNone(brief["success_rate_1"])
        self.assertNotEqual(path.read_bytes(), source)
        self.assertEqual(brief["tuning"]["best_params"], raw["tuning"]["best_params"])
        self.assertEqual(brief["series"], raw["series"])
        self.assertFalse(path.with_name("InvPendulum_results_brief.json").exists())
        self.assertEqual(archive.read_bytes(), arrays_before)

    def test_seed_discovery_prefers_new_reports_and_keeps_plot_filters_independent(self):
        paths = [self.save(seed) for seed in (42, 123, 2024)]
        for path in paths:
            finalize.update_results(path, {self.path.name: self.artifact})
        old = self.root / "model-a" / "InvPendulum_results.json"
        old.write_text("[]")
        with patch.object(finalize, "USE_MOCK_RESULTS", False), \
                patch.object(finalize, "SEEDS_TO_UPDATE", None), \
                patch.object(dashboard, "SEEDS_TO_PLOT", [42]), redirect_stdout(io.StringIO()):
            self.assertEqual(finalize.select_result_paths([], model_folders=["model-a"]), paths)
            self.assertEqual(finalize.select_result_paths([], model_folders=["model-a"], seeds=[123]), [paths[1]])
            self.assertEqual(dashboard.selected_result_files(["model-a"]), [paths[0]])
            self.assertEqual(dashboard.selected_result_files(["model-a"], seeds=None), paths)
            self.assertEqual(len(dashboard.selected_result_files(["model-a"], seeds=None)), 3)

    def test_success_bar_mean_includes_zero_excludes_missing_and_rejects_duplicates(self):
        paths = [self.save(seed) for seed in (42, 123, 2024)]
        briefs = [finalize.update_results(path, {self.path.name: self.artifact}) for path in paths]
        for path, score in zip(briefs, (0.9, 0.6, 0.0)):
            records = json.loads(path.read_text())
            records[0]["success_rate_1"] = score
            path.write_text(json.dumps(records))
        rows = dashboard.load_dashboard_rows(briefs)
        self.assertTrue(all(row["model"] == "model-a" for row in rows))
        case = dashboard._success_cases([("model-a", rows)])[0]
        mean, matched = dashboard.mean_case_success(rows, case)
        self.assertAlmostEqual(mean, 0.5)
        self.assertEqual([row["seed"] for row in matched], [42, 123, 2024])
        figure, axis = plt.subplots()
        try:
            bars = dashboard._draw_success(axis, [("model-a", rows)])
            self.assertAlmostEqual(next(iter(bars)).get_height(), 50.0)
        finally:
            plt.close(figure)
        rows[1]["success_rate_1"] = None
        self.assertAlmostEqual(dashboard.mean_case_success(rows, case)[0], 0.45)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            dashboard.load_dashboard_rows([briefs[0], briefs[0]])
        rows[1]["initial_condition"] = [99, 99]
        with self.assertRaisesRegex(ValueError, "Different test conditions"):
            dashboard._success_cases([("model-a", rows)])

    def test_trajectories_and_plots_are_isolated_by_seed(self):
        for seed in (42, 123):
            path = self.save(seed)
            record = json.loads(path.read_text())[0]
            arrays = trajectories.load_trajectories(record, path)
            self.assertEqual(arrays["states"][0, 0], seed)
            outputs = trajectories.plot_record(record, path, output_root=self.root / "plots")
            self.assertEqual(len(outputs), 2)
            self.assertTrue(all(output.is_file() and output.parent == self.root / "plots" /
                                "model-a" / f"seed{seed}" / "InvPendulum" for output in outputs))
            record["seeds"]["run"] = 2024
            with self.assertRaisesRegex(ValueError, "disagrees"):
                trajectories.plot_record(record, path)

    def test_interrupt_and_failed_attempt_remain_pending_or_failed(self):
        good = self.record(42)
        failed = self.record(123)
        failed["error"] = {"type": "RunError", "message": "test failure"}
        with patch.object(benchmark, "RUN_SEEDS", [42, 123, 2024]), \
                patch.object(benchmark, "selected_artifacts", return_value=[(self.path, self.artifact)]), \
                patch.object(benchmark, "selected_models", return_value=[self.model]), \
                patch.object(benchmark, "role_environment", return_value={}), \
                patch.object(benchmark, "run_one", side_effect=[good, failed, KeyboardInterrupt()]), \
                redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
            benchmark.main()
        progress = json.loads((self.root / benchmark.PROGRESS_FILENAME).read_text())
        self.assertEqual(progress["status"], "interrupted")
        self.assertEqual(progress["summary"]["done_out_of_total"], "1/3")
        self.assertEqual(progress["summary"]["failed"], 1)
        self.assertEqual(progress["summary"]["pending"], 1)


if __name__ == "__main__":
    unittest.main()
