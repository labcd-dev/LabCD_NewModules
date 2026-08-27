"""
================================================================================
agents/scenarist.py
================================================================================
Scenarist node: picks/designs the test scenario (Level I/II/III difficulty,
initial state, target) for the upcoming evaluation round.
Port your prompt text from the original notebook (cell 15) into
SCENARIST_PROMPT_TEMPLATE below.
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from ..utils.logging_utils import get_logger
from .formatting import round_floats
from .llm_base import get_llm, invoke_with_retry, merge_last_output

log = get_logger(__name__)


class Scenario(BaseModel):
    level: str = Field(description="'I' (easy), 'II' (moderate), or 'III' (aggressive/edge-case).")
    initial_state: List[float]
    target_state: List[float]
    rationale: str


# TODO: paste the full prompt text from the original notebook (cell 15) here.
SCENARIST_PROMPT_TEMPLATE = """
You are the Scenarist for MPC tuning of "{system_name}" ({n_states} states: {state_names}).
Default initial state: {default_initial_state}
Default target: {default_target}
Current iteration: {iteration}

Propose a test scenario (difficulty level I/II/III, initial_state, target_state)
appropriate for this stage of tuning.
""".strip()

scenarist_prompt = PromptTemplate(
    input_variables=["system_name", "n_states", "state_names", "default_initial_state", "default_target", "iteration"],
    template=SCENARIST_PROMPT_TEMPLATE,
)


def scenarist_node(state: Dict[str, Any], *, default_initial_state, default_target) -> Dict[str, Any]:
    llm = get_llm().with_structured_output(Scenario)
    prompt_text = scenarist_prompt.format(
        system_name=state.get("system_name", "unknown"),
        n_states=state["n_states"],
        state_names=state.get("state_names", []),
        default_initial_state=round_floats(list(default_initial_state)),
        default_target=round_floats(list(default_target)),
        iteration=state.get("iteration", 0),
    )

    try:
        scenario: Scenario = invoke_with_retry(llm, prompt_text, max_retries=1, node_name="Scenarist")
    except Exception as e:  # noqa: BLE001
        # Same reasoning as agents/actor.py -- fall back to the plugin's own
        # nominal defaults (equivalent to scenario Level I) rather than
        # crashing the whole run over a single failed scenario proposal.
        log.error("[Scenarist] LLM call failed after retry, using the plugin's nominal defaults: %s", e)
        history: List[str] = state.get("history", []) + [
            f"[Scenarist] FAILED ({e}); using nominal default initial state/target (Level I)."
        ]
        return {
            **state,
            "scenario_level": "I",
            "initial_state": list(default_initial_state),
            "target_state": list(default_target),
            "history": history,
            "last_outputs": merge_last_output(state, "scenarist", f"FAILED ({e}); using nominal default initial state/target (Level I)."),
        }

    log.info("[Scenarist] level=%s", scenario.level)
    history: List[str] = state.get("history", []) + [f"[Scenarist] level {scenario.level}: {scenario.rationale[:150]}"]

    return {
        **state,
        "scenario_level": scenario.level,
        "initial_state": scenario.initial_state,
        "target_state": scenario.target_state,
        "history": history,
        "last_outputs": merge_last_output(state, "scenarist", f"Level {scenario.level}: {scenario.rationale}"),
    }
