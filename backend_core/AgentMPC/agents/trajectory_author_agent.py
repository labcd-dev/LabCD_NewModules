"""
================================================================================
agents/trajectory_author_agent.py
================================================================================
Writes a custom reference-trajectory .py file from a plain-language request.

The UI already accepted a hand-written trajectory file and, if it didn't match
the standard, repaired it (agents/trajectory_validator.py). That assumed the
user had a file to begin with. This module covers the other case -- knowing
what you want the reference to do ("theta1 sinusoidal, omega1 its cosine,
amplitude 0.2, frequency 0.4") without wanting to write NumPy to express it.

The generated file is not trusted on the model's word: it goes straight through
the same deterministic validator the upload path uses, and through the same
LLM repair loop if that first check fails. So the output of this agent is
either a file that actually loads, or a clear failure -- never an unchecked
snippet handed to the simulator.

Why the derivative pairs matter here specifically: if a state is the time
derivative of another (agents/dynamics_validator.py:detect_derivative_pairs
finds these, and the Setup Agent panel lets the user correct them), then a
sinusoidal position reference has exactly one physically consistent velocity
reference -- amplitude scaled by omega, not the same amplitude. Getting that
wrong doesn't crash anything; it quietly gives the controller an impossible
target and shows up later as tracking error that looks like bad tuning. The
pairs are therefore fed into the prompt rather than left for the model to
guess at.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..utils.logging_utils import get_logger
from .prompt_library import get_prompt

log = get_logger(__name__)

SYSTEM_PROMPT_TEMPLATE = get_prompt("trajectory_author_agent", "system_prompt")
USER_PROMPT_TEMPLATE = get_prompt("trajectory_author_agent", "user_prompt_template")
REVISION_PROMPT_TEMPLATE = get_prompt("trajectory_author_agent", "revision_prompt_template")


@dataclass
class AuthoredTrajectory:
    """Result of one authoring attempt.

    ``valid`` is the verdict of the deterministic loader, not of the model.
    ``was_repaired`` records that the first draft failed that check and the
    existing repair loop had to fix it -- worth surfacing, since it means the
    explanation describes the draft rather than the final file.
    """

    valid: bool
    code: str = ""
    explanation: str = ""
    was_repaired: bool = False
    error: Optional[str] = None
    history: List[str] = field(default_factory=list)


def _derivative_context(state_names: List[str], derivative_pairs: Optional[list]) -> str:
    if not derivative_pairs:
        return ("- Derivative pairs: none known for this system. Treat every state as "
                "independent unless the user says otherwise.")
    lines = ["- Derivative pairs (the second state IS d/dt of the first):"]
    for i, j in derivative_pairs:
        if i < len(state_names) and j < len(state_names):
            lines.append(f"  - {state_names[j]} = d({state_names[i]})/dt "
                         f"(index {j} is the derivative of index {i})")
    return "\n".join(lines)


_ROLE_ALIASES = {"assistant": "ai", "user": "user", "human": "user", "ai": "ai"}


def author_trajectory(
    request: str,
    summary: Dict[str, Any],
    derivative_pairs: Optional[list] = None,
    tracker: Optional[Any] = None,
    max_fix_attempts: int = 2,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    previous_code: Optional[str] = None,
) -> AuthoredTrajectory:
    """Turn ``request`` into a validated trajectory file.

    ``conversation_history`` / ``previous_code`` turn this from a one-shot
    generator into a conversation: pass the turns so far plus the file that
    is currently on screen, and ``request`` is treated as a REVISION of that
    file ("make the amplitude smaller", "also give theta2 a pulse") instead
    of a fresh brief. Without them it behaves exactly as before. The model
    is handed its own previous file rather than only the description of it,
    because "change the amplitude" is only answerable against the actual
    code -- and the alternative (regenerating from scratch each time) loses
    every detail the earlier turns already got right.

    ``conversation_history`` holds only the turns BEFORE this one, each
    ``{"role": "user"|"assistant", "content": ...}``.

    Raises whatever the underlying LLM call raises -- the caller shows that
    directly, matching how every other LLM call site in this package behaves.
    """
    from pydantic import BaseModel, Field

    from .llm_base import get_llm
    from .trajectory_validator import (
        TRAJECTORY_STANDARD,
        validate_and_fix_trajectory,
        validate_trajectory_source,
    )

    class TrajectoryDraft(BaseModel):
        explanation: str = Field(
            description="Plain-language summary of what the reference does, state by state."
        )
        python_code: str = Field(
            description="The complete .py file defining create_trajectory(...). No markdown fences."
        )

    state_names = list(summary.get("state_names") or [])
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        standard=TRAJECTORY_STANDARD,
        n_states=summary.get("n_states", len(state_names)),
        state_names=", ".join(state_names) or "unknown",
        n_inputs=summary.get("n_inputs", 0),
        input_names=", ".join(summary.get("input_names") or []) or "unknown",
        derivative_context=_derivative_context(state_names, derivative_pairs),
    )

    messages: List[Any] = [("system", system_prompt)]
    for turn in conversation_history or []:
        role = _ROLE_ALIASES.get(str(turn.get("role", "user")).lower(), "user")
        messages.append((role, turn.get("content", "")))
    is_revision = bool(previous_code and (conversation_history or previous_code))
    if is_revision:
        messages.append(("user", REVISION_PROMPT_TEMPLATE.format(
            current_code=previous_code, request=request)))
    else:
        messages.append(("user", USER_PROMPT_TEMPLATE.format(request=request)))

    llm = get_llm().with_structured_output(TrajectoryDraft)
    invoke_kwargs = {"config": {"callbacks": [tracker]}} if tracker is not None else {}
    draft: TrajectoryDraft = llm.invoke(messages, **invoke_kwargs)

    code = _strip_fences(draft.python_code)
    history = ["Revision written by the trajectory agent." if is_revision
               else "Draft written by the trajectory agent."]

    outcome = validate_trajectory_source(code)
    if outcome.valid:
        log.info("[TrajectoryAuthor] draft validated on the first attempt")
        return AuthoredTrajectory(valid=True, code=code, explanation=draft.explanation,
                                  history=history + ["Validated as written."])

    # Reuse the upload path's repair loop rather than a second bespoke one --
    # it already knows how to feed the standard and the loader's error back to
    # the model, and it re-validates whatever comes out.
    log.warning("[TrajectoryAuthor] draft failed validation (%s) -- repairing", outcome.error)
    history.append(f"First draft failed validation: {outcome.error}")
    fix = validate_and_fix_trajectory(code, max_attempts=max_fix_attempts)
    if not fix.valid:
        return AuthoredTrajectory(
            valid=False, code=code, explanation=draft.explanation, was_repaired=True,
            error=f"{outcome.error} (still failing after repair: {fix.still_broken_error})",
            history=history + ["Repair attempts did not produce a loadable file."],
        )
    return AuthoredTrajectory(
        valid=True, code=fix.final_code, explanation=draft.explanation, was_repaired=True,
        history=history + [f"Repaired automatically: {fix.explanation}"],
    )


def _strip_fences(code: str) -> str:
    """Drop a markdown fence if the model added one anyway.

    The prompt asks for bare code, but a fence is the single most common way a
    structured-output field still comes back wrapped, and it turns a valid file
    into a SyntaxError on the very first line.
    """
    text = (code or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    lines = lines[1:]                      # opening ``` (with or without a language tag)
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()
