"""
================================================================================
agents/scenarist.py
================================================================================
Scenarist node: picks/designs the test scenario (Level I/II/III difficulty,
initial state, target) for the upcoming evaluation round.
Prompt text lives in ../prompts/scenarist.yaml.
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from ..utils.logging_utils import get_logger
from .formatting import round_floats
from .llm_base import get_llm, invoke_with_retry, merge_last_output
from .prompt_library import get_prompt

log = get_logger(__name__)


class Scenario(BaseModel):
    level: str = Field(description="'I' (easy), 'II' (moderate), or 'III' (aggressive/edge-case).")
    initial_state: List[float]
    target_state: List[float]
    rationale: str


# Prompt text lives in ../prompts/scenarist.yaml.
SCENARIST_PROMPT_TEMPLATE = get_prompt("scenarist")

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
            f"[Scenarist] level=I (fallback)\n"
            f"initial_state={round_floats(list(default_initial_state))}\n"
            f"target_state={round_floats(list(default_target))}\n\n"
            f"FAILED ({e}); using the plugin's nominal default initial state/target."
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
    # initial_state/target_state are numeric arrays the Scenarist itself
    # proposed -- previously invisible in the Agent Reasoning panel, which
    # only ever showed the rationale prose, truncated to 150 characters.
    history: List[str] = state.get("history", []) + [
        f"[Scenarist] level={scenario.level}\n"
        f"initial_state={round_floats(scenario.initial_state)}\n"
        f"target_state={round_floats(scenario.target_state)}\n\n"
        f"{scenario.rationale}"
    ]

    return {
        **state,
        "scenario_level": scenario.level,
        "initial_state": scenario.initial_state,
        "target_state": scenario.target_state,
        "history": history,
        "last_outputs": merge_last_output(state, "scenarist", f"Level {scenario.level}: {scenario.rationale}"),
    }
