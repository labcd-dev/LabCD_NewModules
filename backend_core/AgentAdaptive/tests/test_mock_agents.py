"""Offline regression checks for mock routing and real Adaptive simulation."""

import contextlib
import copy
import io
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from backend_core.AgentAdaptive.agents import llm_factory, tuner_agent, tuner_agent_main
from backend_core.AgentAdaptive.agents import run_full_pipeline as default_pipeline
from backend_core.AgentAdaptive.agents.designer_agent import MockDesigner
from backend_core.AgentAdaptive.agents.mock_random import mock_session, current_session
from backend_core.plant_compiler import PlantCompiler


class MockAdaptiveTests(unittest.TestCase):
    def spec(self):
        root = Path(__file__).resolve().parents[3]
        artifact = json.loads((root / "benchmarks/Adaptive/artifacts/single_pole.json").read_text(encoding="utf-8"))
        test = artifact["tests"]["combinations"][0]
        spec = PlantCompiler().generate_adaptive_spec(artifact, {
            "initial_state": test["initial_condition"],
            "total_simulation_time": 1.0, "solver_sample_time": 0.005,
        })
        spec["dynamics"]["references"] = [{"output": "theta", "expr": "0"}]
        return spec

    def run_mock(self, seed):
        with patch.object(llm_factory.LLMFactory, "create", side_effect=AssertionError("API factory called")), \
             patch.dict(os.environ, {"LABCD_ADAPTIVE_SHOW_PLOTS": "0"}), \
             contextlib.redirect_stdout(io.StringIO()):
            return tuner_agent.run_full_pipeline(
                description="", system_spec=self.spec(), enable_tuning=True,
                max_tuning_rounds=2, tuning_objectives={"control_effort": 3},
                mock_seed=seed,
            )

    def test_real_simulation_and_reproducible_mock_messages(self):
        first = self.run_mock(42)
        repeated = self.run_mock(42)
        changed = self.run_mock(43)
        result, usage, history, best = first
        self.assertIsNotNone(result.get("final_metrics"), result.get("messages"))
        self.assertTrue(result["series"]["channels"]["t"]["data"])
        self.assertEqual([entry["round"] for entry in history], [0, 1, 2])
        self.assertTrue(history[1]["changed"])
        self.assertIsNotNone(best["tuning"])
        self.assertEqual(result["mock_agent_messages"], repeated[0]["mock_agent_messages"])
        self.assertNotEqual(history[1]["tuning"], changed[2][1]["tuning"])
        self.assertEqual(result["final_metrics"], repeated[0]["final_metrics"])
        self.assertEqual({m["agent"] for m in result["mock_agent_messages"]}, {"design", "tuner", "reporter"})
        total = usage["total"]
        self.assertEqual(total["input_tokens"] + total["output_tokens"], total["total_tokens"])
        self.assertTrue(usage["simulated"])
        self.assertEqual(usage["actual_cost_usd"], 0.0)
        self.assertAlmostEqual(usage["cost_usd"], total["total_tokens"] * 0.001)
        self.assertEqual(usage["clarifier"]["total_tokens"], 0)
        self.assertIsNone(current_session())

    def test_real_agents_remain_separate_and_mock_context_resets(self):
        sentinel = object()
        self.assertIs(default_pipeline, tuner_agent_main.run_full_pipeline)
        self.assertEqual(tuner_agent_main.run_extraction.__module__,
                         "backend_core.AgentAdaptive.agents.designer_agent_main")
        self.assertEqual(tuner_agent_main.write_abstract.__module__,
                         "backend_core.AgentAdaptive.agents.report_writer_main")
        with mock_session(7):
            with patch.object(llm_factory.LLMFactory, "create", return_value=sentinel) as create:
                with patch.dict(os.environ, {"LLM_PROVIDER": "groq", "OPENAI_MODEL": "example-model"}):
                    self.assertIs(llm_factory.build_llm("design"), sentinel)
                    self.assertEqual(create.call_count, 1)
            self.assertIsNotNone(current_session())
        with self.assertRaisesRegex(RuntimeError, "failure"):
            with mock_session(7):
                raise RuntimeError("failure")
        self.assertIsNone(current_session())

    def test_mock_tuner_honors_parameter_fence_and_square_designer(self):
        spec = copy.deepcopy(self.spec())
        spec["dynamics"]["outputs"] = ["omega"]
        with mock_session(42) as session:
            designer = MockDesigner(session, "design", {"spec": spec})
            self.assertEqual(json.loads(designer.invoke([]).content)["method"], "smc")
            tuner = tuner_agent.MockTuner(session, "tuner", {
                "method": "smc", "current_tuning": {"K": 1.5, "Lam": 5.0, "surface_lambda": 2.0},
                "allowed_params": {"K"},
            })
            proposal = json.loads(tuner.invoke([]).content)
            self.assertEqual(set(proposal), {"reasoning", "diagnosed_symptom", "K"})
            self.assertTrue(0.85 * 1.5 <= proposal["K"] <= 1.20 * 1.5)


if __name__ == "__main__":
    unittest.main()
