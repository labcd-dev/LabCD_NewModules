"""
================================================================================
agents/trajectory_validator.py
================================================================================
Same design as agents/dynamics_validator.py, applied to custom reference-
trajectory files instead of dynamics files: validate deterministically first
(free, no LLM), and only ask the LLM to repair the file if that fails --
always re-verifying any LLM-produced fix with the same deterministic check
before trusting it. See dynamics_validator.py's module docstring for the
full reasoning; it's identical here.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..dynamics.trajectory_loader import TrajectoryLoader, TrajectoryPluginError
from ..utils.logging_utils import get_logger
from .prompt_library import get_prompt

log = get_logger(__name__)


# Re-exported at module level (not just used as a prompt): the Streamlit UI
# imports this name and renders it as the user-facing trajectory contract.
TRAJECTORY_STANDARD = get_prompt("trajectory_validator", "standard")


@dataclass
class ValidationOutcome:
    valid: bool
    error: Optional[str] = None


@dataclass
class FixOutcome:
    valid: bool
    used_llm_fix: bool
    final_code: str
    original_error: Optional[str] = None
    explanation: Optional[str] = None
    still_broken_error: Optional[str] = None
    attempts: int = 0
    history: list = field(default_factory=list)


def validate_trajectory_source(source_code: str) -> ValidationOutcome:
    """Deterministic check -- no LLM involved."""
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(source_code)
            temp_path = f.name
        TrajectoryLoader.load_from_path(temp_path)
        return ValidationOutcome(valid=True)
    except TrajectoryPluginError as e:
        return ValidationOutcome(valid=False, error=str(e))
    except Exception as e:  # noqa: BLE001
        return ValidationOutcome(valid=False, error=f"{type(e).__name__}: {e}")
    finally:
        if temp_path and Path(temp_path).exists():
            Path(temp_path).unlink()


def _fix_prompt(source_code: str, error_message: str) -> str:
    return get_prompt("trajectory_validator", "fix_prompt").format(
        standard=TRAJECTORY_STANDARD,
        error_message=error_message,
        source_code=source_code,
    )


def fix_trajectory_with_llm(source_code: str, error_message: str):
    from pydantic import BaseModel, Field

    from .llm_base import get_llm

    class FixProposal(BaseModel):
        explanation: str = Field(description="Plain-language summary of what was wrong and what was changed.")
        fixed_code: str = Field(description="The complete, corrected .py file content.")

    llm = get_llm().with_structured_output(FixProposal)
    return llm.invoke(_fix_prompt(source_code, error_message))


def validate_and_fix_trajectory(source_code: str, max_attempts: int = 2) -> FixOutcome:
    outcome = validate_trajectory_source(source_code)
    if outcome.valid:
        return FixOutcome(valid=True, used_llm_fix=False, final_code=source_code, attempts=0)

    original_error = outcome.error
    current_code = source_code
    current_error = outcome.error
    history = []

    for attempt in range(1, max_attempts + 1):
        log.info("Trajectory validation failed (attempt %d/%d): %s", attempt, max_attempts, current_error)
        try:
            proposal = fix_trajectory_with_llm(current_code, current_error)
        except Exception as e:  # noqa: BLE001
            log.warning("LLM fix attempt %d failed to even run: %s", attempt, e)
            history.append({"attempt": attempt, "error": f"LLM call failed: {e}"})
            break

        recheck = validate_trajectory_source(proposal.fixed_code)
        if recheck.valid:
            return FixOutcome(
                valid=True, used_llm_fix=True, final_code=proposal.fixed_code,
                original_error=original_error, explanation=proposal.explanation,
                attempts=attempt, history=history,
            )

        history.append({"attempt": attempt, "error": recheck.error})
        current_code = proposal.fixed_code
        current_error = recheck.error

    return FixOutcome(
        valid=False, used_llm_fix=True, final_code=current_code,
        original_error=original_error, still_broken_error=current_error,
        attempts=max_attempts, history=history,
    )
