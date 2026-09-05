"""
================================================================================
agents/config_advisor_agent.py
================================================================================
Config Advisor Agent: the human-in-the-loop consultation step. Two related
capabilities, deliberately kept in one module because they share the same
system context:

  1. ``chat(...)`` -- free-form conversation about the uploaded system
     ("should I even use MPC here?", "what should I watch out for?").
     Plain text, no schema.

  2. ``suggest_config(...)`` -- returns a STRUCTURED set of suggested
     starting values (state/input constraint bounds + the general tuning
     settings: simulation time, settling tolerance, iteration budget,
     exploration behavior), each with a one-line rationale, so the UI can
     offer "use these" as a single click while still letting the user
     override every field by hand.

Why these two are different shapes on purpose: the chat is meant to read
like a real back-and-forth, so forcing it into a schema would work against
it; the suggestions have to land in specific numeric UI fields, so they
need a schema. Both are grounded in the SAME system context builder
(``build_system_context``) so the advice in the chat and the numbers in the
suggestions can't drift apart.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..utils.logging_utils import get_logger
from .llm_base import TokenUsageTracker, get_llm, invoke_with_retry
from .prompt_library import get_prompt

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared system context
# ---------------------------------------------------------------------------

def build_system_context(summary: Dict[str, Any], setup_notes: Optional[List[str]] = None,
                          derivative_pairs: Optional[list] = None) -> str:
    """Builds a compact description of the loaded system from whatever the
    deterministic Setup Agent analyses already found (see
    agents/dynamics_validator.py) -- so both the chat and the structured
    suggestions start from genuinely informed context (an already-detected
    fast/slow timescale, a suggested dt, the derivative structure) rather
    than just the bare state/input names."""
    lines = [
        f"- Class name: {summary.get('dynamics_class', 'Unknown')}",
        f"- States ({summary.get('n_states', 0)}): {', '.join(summary.get('state_names', [])) or 'unknown'}",
        f"- Inputs ({summary.get('n_inputs', 0)}): {', '.join(summary.get('input_names', [])) or 'unknown'}",
        f"- Physical parameters: {summary.get('params', {}) or 'none declared'}",
    ]
    if summary.get("input_bounds"):
        lines.append(f"- Input bounds declared by the plugin: {summary['input_bounds']}")
    if summary.get("state_bounds"):
        lines.append(f"- State bounds declared by the plugin: {summary['state_bounds']}")
    if derivative_pairs:
        state_names = summary.get("state_names", [])
        pairs_desc = ", ".join(
            f"{state_names[j]} = d({state_names[i]})/dt" for i, j in derivative_pairs
            if i < len(state_names) and j < len(state_names)
        )
        if pairs_desc:
            lines.append(f"- Detected derivative structure: {pairs_desc}")
    if setup_notes:
        lines.append("- Setup Agent's own analysis notes:")
        for note in setup_notes[:6]:
            lines.append(f"  - {note}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. Free-form chat
# ---------------------------------------------------------------------------

_CHAT_SYSTEM_PROMPT = get_prompt("config_advisor_agent", "chat_system_prompt")


def chat(
    user_message: str,
    conversation_history: List[Dict[str, str]],
    summary: Dict[str, Any],
    setup_notes: Optional[List[str]] = None,
    derivative_pairs: Optional[list] = None,
    tracker: Optional[TokenUsageTracker] = None,
) -> str:
    """One turn of the advisory conversation. ``conversation_history`` is
    the list of {"role": "user"|"assistant", "content": ...} dicts BEFORE
    this turn. Returns the assistant's reply text. Raises whatever the
    underlying LLM call raises -- callers should catch and surface it."""
    system_prompt = _CHAT_SYSTEM_PROMPT.format(
        system_context=build_system_context(summary, setup_notes, derivative_pairs)
    )
    messages = [("system", system_prompt)]
    for turn in conversation_history:
        messages.append(("user" if turn["role"] == "user" else "assistant", turn["content"]))
    messages.append(("user", user_message))

    llm = get_llm()
    invoke_kwargs = {"config": {"callbacks": [tracker]}} if tracker is not None else {}
    response = llm.invoke(messages, **invoke_kwargs)
    return response.content if hasattr(response, "content") else str(response)


# ---------------------------------------------------------------------------
# 2. Structured config suggestions
# ---------------------------------------------------------------------------

class BoundSuggestion(BaseModel):
    name: str = Field(description="The exact state or input name this bound applies to, copied from the list given.")
    lower: Optional[float] = Field(default=None, description="Suggested lower bound, or null for unbounded on this side.")
    upper: Optional[float] = Field(default=None, description="Suggested upper bound, or null for unbounded on this side.")
    rationale: str = Field(description="One short line on why -- physical limit, safety margin, actuator range, etc.")


class GeneralSettingsSuggestion(BaseModel):
    simulation_time: float = Field(
        description="Seconds to simulate each candidate. Must be between 2 and 20. Should comfortably "
                    "cover several times the system's slowest meaningful time constant so settling is visible.")
    settling_tolerance_pct: int = Field(
        description="What counts as 'settled', as a percent of initial error. Must be between 1 and 20. "
                    "Lower is stricter.")
    max_iterations: int = Field(
        description="How many candidate parameter sets to try. Must be between 3 and 30.")
    min_explore_iterations: int = Field(
        description="Iterations of forced exploration before fine-tuning is allowed. Must be between 0 and 15, "
                    "and less than max_iterations.")
    exploration_intensity: int = Field(
        description="How bold parameter changes are while exploring, as a percent. Must be between 1 and 100. "
                    "50 is normal.")
    rationale: str = Field(
        description="2-3 sentences explaining these choices with respect to THIS system's timescales and "
                    "difficulty -- not generic advice.")


class ConfigSuggestion(BaseModel):
    summary: str = Field(description="2-3 sentences: what kind of system this is and what that implies for tuning it.")
    input_bounds: List[BoundSuggestion] = Field(
        description="One entry per input, in the same order as the input list given. Use null for a side "
                    "that genuinely shouldn't be bounded.")
    state_bounds: List[BoundSuggestion] = Field(
        description="One entry per state, in the same order as the state list given. Most states are often "
                    "genuinely unbounded -- use null/null freely rather than inventing limits.")
    general_settings: GeneralSettingsSuggestion
    warnings: List[str] = Field(
        default_factory=list,
        description="Anything the user should know before running -- an unstable equilibrium, a very fast "
                    "timescale, an under-actuated system, etc. Empty list if nothing notable.")


_SUGGEST_SYSTEM_PROMPT = get_prompt("config_advisor_agent", "suggest_system_prompt")


def suggest_config(
    summary: Dict[str, Any],
    setup_notes: Optional[List[str]] = None,
    derivative_pairs: Optional[list] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    tracker: Optional[TokenUsageTracker] = None,
) -> ConfigSuggestion:
    """Returns structured suggested starting values. If
    ``conversation_history`` is given, whatever the user said in the chat is
    folded in as additional context -- so if they mentioned e.g. "my
    actuator saturates at 5 N" mid-conversation, that actually influences
    the suggested bounds rather than being forgotten the moment they click
    the suggest button.

    Raises on failure -- callers should catch and fall back to their own
    defaults rather than blocking the user from proceeding."""
    system_prompt = _SUGGEST_SYSTEM_PROMPT.format(
        system_context=build_system_context(summary, setup_notes, derivative_pairs)
    )
    if conversation_history:
        convo = "\n".join(f"{t['role']}: {t['content']}" for t in conversation_history[-8:])
        system_prompt += (
            f"\n\nThe user also discussed this system with you already -- take anything "
            f"relevant here (stated physical limits, priorities, concerns) into account:\n{convo}"
        )

    llm = get_llm().with_structured_output(ConfigSuggestion)
    return invoke_with_retry(llm, system_prompt, max_retries=1,
                              node_name="ConfigAdvisor", tracker=tracker)


def clamp_general_settings(s: GeneralSettingsSuggestion) -> GeneralSettingsSuggestion:
    """Defense in depth: the prompt states every valid range and the retry
    feeds validation errors back, but an LLM can still return an
    out-of-range number that happens to satisfy the schema's type. These
    values go straight into st.slider(...) calls, which raise if the value
    is outside min/max -- so clamp here rather than letting a bad
    suggestion crash the page."""
    return GeneralSettingsSuggestion(
        simulation_time=float(min(max(s.simulation_time, 2.0), 20.0)),
        settling_tolerance_pct=int(min(max(s.settling_tolerance_pct, 1), 20)),
        max_iterations=int(min(max(s.max_iterations, 3), 30)),
        min_explore_iterations=int(min(max(s.min_explore_iterations, 0), 15)),
        exploration_intensity=int(min(max(s.exploration_intensity, 1), 100)),
        rationale=s.rationale,
    )
