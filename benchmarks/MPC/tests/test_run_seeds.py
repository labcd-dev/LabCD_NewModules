"""Seed repetition and result-layout tests; no network or paid LLM calls."""

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
import numpy as np

from benchmarks.MPC import mpc_benchmark as benchmark
from benchmarks.MPC import finalize_results as finalize
from benchmarks.MPC import plot_dashboard as dashboard
from benchmarks.MPC import plot_trajectories as trajectories


class RunSeedTests(unittest.TestCase):
    def test_seed_validation_before_execution(self):
        for seeds in ([], [1, 1], [True], [-1], [1.5], [2**32], "123", [[1]]):
            with self.subTest(seeds=seeds), patch.object(benchmark, "RUN_SEEDS", seeds):
                with self.assertRaises(ValueError):
                    benchmark.validate_settings()
        with patch.object(benchmark, "RUN_SEEDS", [1, 2, 3]):
            benchmark.validate_settings()

    def test_directory_validation_and_legacy_compatibility(self):
        base = {"folder_name": "gpt-4.1-nano"}
        self.assertEqual(benchmark.result_directory(base), benchmark.RESULTS_DIR / base["folder_name"])
        self.assertEqual(benchmark.result_directory({**base, "run_seed": 2}),
                         benchmark.RESULTS_DIR / base["folder_name"] / "seed2")
        for config in ({"folder_name": "../escape", "run_seed": 1},
                       {**base, "run_seed": "../escape"}, {**base, "run_seed": True}):
            with self.assertRaises(ValueError):
                benchmark.result_directory(config)

    def test_model_factory_captures_each_repetition_seed(self):
        config = {"model": "test-model", "provider": "openai",
                  "api_key_env_var": "OPENAI_API_KEY"}
        with patch.object(benchmark, "USE_MOCK_AGENTS", False), \
                patch("dotenv.load_dotenv"), patch("labcd_agents.require_api_key", return_value="dummy"), \
                patch("labcd_agents.LLMFactory.create") as create, \
                patch.object(benchmark, "configure_llm") as configure:
            for seed in (1, 2, 3):
                benchmark.configure_model(config, seed)
                configure.call_args.kwargs["factory"]()
            self.assertEqual([call.kwargs["seed"] for call in create.call_args_list], [1, 2, 3])

    def test_main_checkpoints_every_model_seed_test_and_isolates_npz(self):
        artifact_path = Path("Plant.json")
        artifact = {"system_name": "Plant", "tests": {"combinations": [
            {"test_id": "test_1"}, {"test_id": "test_2"}]}}
        models = [{"folder_name": name, "model": name} for name in ("model-a", "model-b")]
        calls = []

        def run(path, plant, test, config, seed):
            calls.append((config["folder_name"], config["run_seed"], test["test_id"], seed))
            arrays = {"states": np.full((3, 1), seed, dtype=float),
                      "reference": np.zeros((3, 1)), "inputs": np.zeros((2, 1)),
                      "times": np.arange(3, dtype=float)}
            link = benchmark.save_verification_trajectories(path, test["test_id"], config, arrays)
            return {"artifact": path.name, "test_id": test["test_id"], "seeds": {"run": seed},
                    "tuning": {"completed": True, "best_params": {"Np": 10}},
                    "verification": {"raw_result": {"metrics": {}, "trajectory_file": link}}}

        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(benchmark, "RESULTS_DIR", Path(temporary)), \
                patch.object(benchmark, "RUN_BENCHMARK", True), \
                patch.object(benchmark, "RUN_SEEDS", [1, 2, 3]), \
                patch.object(benchmark, "selected_artifacts", return_value=[(artifact_path, artifact)]), \
                patch.object(benchmark, "selected_models", return_value=models), \
                patch.object(benchmark, "configure_model") as configure, \
                patch.object(benchmark, "run_one", side_effect=run), redirect_stdout(io.StringIO()):
            old = Path(temporary) / "model-a" / "Plant_results.json"
            old.parent.mkdir()
            old.write_text("old report", encoding="utf-8")
            benchmark.main()
            self.assertEqual(len(calls), 12)
            self.assertEqual(configure.call_count, 6)
            self.assertTrue(all(run_seed == seed for _, run_seed, _, seed in calls))
            for model in models:
                for seed in (1, 2, 3):
                    report = Path(temporary) / model["folder_name"] / f"seed{seed}" / "Plant_results.json"
                    records = json.loads(report.read_text(encoding="utf-8"))
                    self.assertEqual(len(records), 2)
                    self.assertEqual([row["seeds"]["run"] for row in records], [seed, seed])
                    for record in records:
                        link = record["verification"]["raw_result"]["trajectory_file"]
                        with np.load(report.parent / link, allow_pickle=False) as archive:
                            np.testing.assert_array_equal(archive["states"], np.full((3, 1), seed))
            self.assertEqual(old.read_text(encoding="utf-8"), "old report")
            progress = json.loads((Path(temporary) / benchmark.PROGRESS_FILENAME).read_text(encoding="utf-8"))
            self.assertEqual(progress["summary"]["done_out_of_total"], "12/12")
            self.assertEqual(progress["remaining_combinations"], [])
            self.assertEqual(progress["status"], "finished")

    def test_discovery_prefers_seed_runs_and_supports_seed_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "gpt-4.1-nano"
            for directory in (model, model / "seed1", model / "seed2", model / "seed3"):
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "Plant_results.json").write_text("[]", encoding="utf-8")
            with patch.object(finalize, "RESULTS_DIR", root), \
                    patch.object(finalize, "USE_MOCK_RESULTS", False), \
                    patch.object(finalize, "MODEL_FOLDERS_TO_UPDATE", ["gpt-4.1-nano"]), \
                    patch.object(finalize, "SEEDS_TO_UPDATE", None), \
                    patch.object(dashboard, "RESULTS_DIR", root), \
                    patch.object(dashboard, "SEEDS_TO_PLOT", None), redirect_stdout(io.StringIO()):
                paths = finalize.select_result_paths([])
                self.assertEqual([p.parent.name for p in paths], ["seed1", "seed2", "seed3"])
                self.assertEqual(dashboard.selected_result_files(["gpt-4.1-nano"]), paths)
                self.assertEqual(dashboard._model_label({}, paths[0]), "gpt-4.1-nano")
                self.assertEqual([p.parent.name for p in dashboard.selected_result_files(
                    ["gpt-4.1-nano"], seeds=[2])], ["seed2"])
                with patch.object(finalize, "SEEDS_TO_UPDATE", [3]):
                    self.assertEqual([p.parent.name for p in finalize.select_result_paths([])], ["seed3"])
            for seeds in ([], [1, 1], [True], [-1]):
                with self.assertRaises(ValueError):
                    finalize.result_directories(model, seeds)

    def test_legacy_only_directory_is_still_readable(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "mock"
            model.mkdir()
            (model / "Plant_results.json").write_text("[]", encoding="utf-8")
            self.assertEqual(finalize.result_directories(model, None), [model])
            self.assertEqual(finalize.result_directories(model, [2]), [model / "seed2"])

    def test_real_controller_with_mock_agents_records_seed_and_plots_separately(self):
        path = benchmark.ARTIFACTS_DIR / "InvPendulum.json"
        artifact = copy.deepcopy(benchmark.load_artifact(path))
        artifact["simulation_parameters"].update(simulation_time=1.0, dt_mpc=0.02)
        test = artifact["tests"]["combinations"][0]
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(benchmark, "RESULTS_DIR", Path(temporary) / "results"), \
                patch.object(benchmark, "USE_MOCK_AGENTS", True), \
                patch.object(benchmark, "MAX_ITERATIONS", 1), \
                patch.object(benchmark, "MIN_EXPLORE_ITERATIONS", 0):
            config = {"folder_name": "mock", "model": "mock-random", "run_seed": 2}
            record = benchmark.run_one(path, artifact, test, config, 2)
            self.assertNotIn("error", record)
            self.assertEqual(record["seeds"], {"run": 2, "llm": None, "simulation": 2, "mock": 2})
            report = benchmark.save_artifact_results(path, config, [record])
            arrays = trajectories.load_trajectories(record, report)
            self.assertGreater(len(arrays["times"]), 1)
            plots = trajectories.plot_record(record, report, output_root=Path(temporary) / "plots")
            self.assertEqual(len(plots), 2)
            self.assertTrue(all(p.is_file() and p.parent == Path(temporary) / "plots" /
                                "mock" / "seed2" / path.stem for p in plots))
            updated = finalize.finalize_record(record, {path.name: artifact})
            self.assertEqual(updated["seeds"], record["seeds"])
            self.assertIsNotNone(updated["success_rate_1"])


if __name__ == "__main__":
    unittest.main()
