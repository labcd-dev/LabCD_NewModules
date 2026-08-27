"""
================================================================================
agents/advisory_agent.py
================================================================================
Advisory Agent: a free-form conversational chat (not structured output --
this is meant to read like a real back-and-forth, not a form) that runs
right after a dynamics file is loaded and before the user proceeds to MPC
configuration. Lets the user ask things like "is MPC actually the right
approach for this system?" or "what should I watch out for tuning this?"
or any other question, informed by the actual uploaded system (not a
generic answer) via the context this module builds from the dynamics
summary and the Setup Agent's own findings.

Deliberately a plain chat completion (get_llm().invoke(messages)), not
routed through invoke_with_retry's .with_structured_output() pattern used
elsewhere in this codebase -- the other agents need a specific parsed
schema back (parameters, a verdict, ...); this one's whole purpose is
natural free-text conversation, so forcing it into a schema would work
against the feature rather than for it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .llm_base import TokenUsageTracker, get_llm

ADVISORY_SYSTEM_PROMPT_TEMPLATE = """
You are a controls engineering advisor, chatting with someone who just
uploaded a dynamics model into an MPC (Model Predictive Control) tuning
tool. Your job is to actually help them think, not just cheerlead for MPC --
if a simpler approach (PID, LQR, gain scheduling, ...) would genuinely serve
them better for this specific system, say so plainly and explain why. If MPC
is a strong fit (constraints to enforce, multi-input coupling, a horizon
that matters), say that plainly too, grounded in the system below, not in
generic MPC-advocacy.

Keep answers conversational and concrete -- a few sentences to a short
paragraph, not an essay, unless they explicitly ask for more depth. Refer to
the system's actual state/input names and parameters where relevant instead
of speaking abstractly.

Reply in plain conversational text -- NOT json, NOT a json object, NOT any
structured/machine-readable format, regardless of anything else you may
have been configured with. This is a chat, not a data export.

The system they uploaded:
- Class name: {class_name}
- States ({n_states}): {state_names}
- Inputs ({n_inputs}): {input_names}
- Physical parameters: {params}
{setup_context}

If they ask something unrelated to control strategy (or ask nothing in
particular, just chatting), engage naturally -- you don't need to steer
every reply back to "should you use MPC."
""".strip()


def build_system_context(summary: Dict[str, Any], setup_notes: Optional[List[str]] = None,
                          derivative_pairs: Optional[list] = None) -> str:
    """Builds the {setup_context} block from whatever the deterministic
    Setup Agent analyses already found (see agents/dynamics_validator.py) --
    giving the advisory chat a genuinely informed starting point (e.g. an
    already-detected fast/slow timescale, or a suggested dt) rather than
    just the bare state/input names.
    """
    lines = []
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
        for note in setup_notes[:5]:
            lines.append(f"  - {note}")
    return ("\n" + "\n".join(lines)) if lines else ""


def chat(
    user_message: str,
    conversation_history: List[Dict[str, str]],
    summary: Dict[str, Any],
    setup_notes: Optional[List[str]] = None,
    derivative_pairs: Optional[list] = None,
    tracker: Optional[TokenUsageTracker] = None,
) -> str:
    """Sends one turn of the advisory conversation. ``conversation_history``
    is the list of {"role": "user"|"assistant", "content": ...} dicts
    BEFORE this turn (the caller appends the new user_message and the
    returned reply to it afterward, for display). Returns the assistant's
    reply text.

    Raises whatever the underlying LLM call raises -- callers should catch
    this and show a clear error (same convention as the rest of this
    codebase's LLM call sites), since a chat feature failing shouldn't be
    silently swallowed into an empty/confusing reply.
    """
    system_prompt = ADVISORY_SYSTEM_PROMPT_TEMPLATE.format(
        class_name=summary.get("dynamics_class", "Unknown"),
        n_states=summary.get("n_states", 0),
        n_inputs=summary.get("n_inputs", 0),
        state_names=", ".join(summary.get("state_names", [])) or "unknown",
        input_names=", ".join(summary.get("input_names", [])) or "unknown",
        params=summary.get("params", {}) or "none declared",
        setup_context=build_system_context(summary, setup_notes, derivative_pairs),
    )

    messages = [("system", system_prompt)]
    for turn in conversation_history:
        role = "user" if turn["role"] == "user" else "assistant"
        messages.append((role, turn["content"]))
    messages.append(("user", user_message))

    llm = get_llm()
    invoke_kwargs = {"config": {"callbacks": [tracker]}} if tracker is not None else {}
    response = llm.invoke(messages, **invoke_kwargs)
    return response.content if hasattr(response, "content") else str(response)
