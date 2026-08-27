"""
================================================================================
run_agents.py
================================================================================
Runs the full LLM-driven MPC tuning loop (Scenarist -> Actor -> Evaluator ->
Terminator -> Critic/Juror) using Groq (via `langchain-groq`), following the
same .env / proxy pattern used in the original notebook.

Setup:
    pip install -r requirements.txt   # includes python-dotenv, langchain-groq
    echo "GROQ_API_KEY=your-groq-api-key" >> .env

Run:
    python run_agents.py

Before running for real: replace the placeholder prompt templates in
backend_core/AgentMPC/agents/{actor,critic,scenarist,terminator,juror}.py with your
actual domain-tuned prompts (each file has a clearly marked TODO block).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow `python backend_core/AgentMPC/run_agents.py` from repo root or anywhere
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from dotenv import load_dotenv
from langchain_groq import ChatGroq

from backend_core.AgentMPC.agents.llm_base import configure_llm
from backend_core.AgentMPC.dynamics.loader import DynamicLoader
from backend_core.AgentMPC.graph.workflow import build_mpc_tuning_graph, initial_state
from backend_core.AgentMPC.mpc.config import Config
from backend_core.AgentMPC.utils.logging_utils import configure_logging, get_logger

log = get_logger(__name__)


def main() -> None:
    configure_logging()

    # ------------------------------------------------------------------
    # 1) Load environment variables (same pattern as the original notebook).
    # ------------------------------------------------------------------
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    log.info("Groq API key loaded: %s", groq_api_key is not None)
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY not found. Add it to your .env file or export it before running.")

    # Proxy settings (uncomment if you're behind a proxy, same as the
    # original notebook -- left off by default here).
    # os.environ["http_proxy"] = "http://127.0.0.1:10808"
    # os.environ["https_proxy"] = "http://127.0.0.1:10808"
    # os.environ["HTTP_PROXY"] = "http://127.0.0.1:10808"
    # os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10808"

    # ------------------------------------------------------------------
    # 2) Configure the LLM. `configure_llm` takes a zero-argument factory
    #    (called fresh for each agent node) rather than a single shared
    #    instance -- this makes it trivial to give different agents
    #    different models/temperatures later if you want to (e.g. a
    #    cheaper/faster model for the Terminator's routine calls, a
    #    stronger one for the Actor).
    # ------------------------------------------------------------------
    configure_llm(
        lambda: ChatGroq(
            model="openai/gpt-oss-120b",   # swap for whichever Groq-hosted model you want -- note:
                                             # llama-3.3-70b-versatile was retired by Groq on Aug 16, 2026
            api_key=groq_api_key,
            temperature=0.3,
            max_retries=2,
        )
    )

    # ------------------------------------------------------------------
    # 3) Load the dynamics plugin (swap this path for your own system,
    #    e.g. Overactuated_Quadcopter.py, once it's rewritten against the
    #    BaseDynamics contract in dynamics/base.py).
    # ------------------------------------------------------------------
    _plugin = Path(__file__).resolve().parent / "dynamics" / "plugins" / "example_pendulum.py"
    plugin = DynamicLoader.load_from_path(str(_plugin))
    dynamics = plugin.create_dynamics()
    log.info("Loaded dynamics: %s", plugin.summary())

    # ------------------------------------------------------------------
    # 4) MPC configuration for the closed-loop evaluations the Evaluator
    #    agent will run every iteration.
    # ------------------------------------------------------------------
    cfg = Config()
    cfg.mpc.prediction_horizon = 12
    cfg.mpc.control_horizon = 4
    cfg.data.dt_mpc = 0.02
    cfg.data.simulation_time = 3.0

    # ------------------------------------------------------------------
    # 5) Build and run the tuning graph.
    # ------------------------------------------------------------------
    graph = build_mpc_tuning_graph(dynamics, cfg)
    state = initial_state(dynamics, system_name="cart_pole_pendulum", max_iterations=15)

    final_state = graph.invoke(state)

    print("\n" + "=" * 70)
    print("TUNING RUN COMPLETE")
    print("=" * 70)
    print("Best params:", final_state.get("best_params"))
    print("Best MSE:", final_state.get("best_mse"))
    print("Iterations used:", final_state.get("iteration"))
    print("Termination reason:", final_state.get("termination_reason"))


if __name__ == "__main__":
    main()
