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

log = get_logger(__name__)


TRAJECTORY_STANDARD = """
# AgentMPC Custom Trajectory File Standard

A custom trajectory file is a single .py file defining ONE reference
trajectory generator. It must define exactly one function:

## `create_trajectory(dt_mpc, simulation_time, n_states, state_names) -> np.ndarray`

| Argument          | Type        | Meaning |
|--------------------|-------------|---------|
| `dt_mpc`             | float        | timestep between samples, in seconds |
| `simulation_time`      | float         | total duration to cover, in seconds |
| `n_states`               | int            | length of the state vector (from the loaded dynamics plugin) |
| `state_names`              | list[str]        | names of each state, in order (e.g. `["cart_pos", "cart_vel", "pole_angle", "pole_ang_vel"]`) |

**Returns:** a NumPy array of shape `(n_steps, n_states)` where
`n_steps >= simulation_time / dt_mpc`, giving the desired value of every
state at every timestep.

## Physical consistency (important)

If your state vector pairs a position-like quantity with its own
velocity/derivative (a very common pattern -- e.g. `cart_pos`/`cart_vel`),
the velocity column should be the actual time-derivative of the position
column, not an independently made-up signal. For example, for a sinusoidal
position reference `amplitude * sin(omega * t)`, the matching velocity
reference is `amplitude * omega * cos(omega * t)` -- NOT another sine with
an arbitrary phase shift. Getting this wrong doesn't break the simulation,
but it gives the controller a physically-inconsistent target to chase,
which shows up as needless tracking error that has nothing to do with how
good the MPC tuning actually is.

## Example

```python
def create_trajectory(dt_mpc, simulation_time, n_states, state_names):
    n_steps = int(simulation_time / dt_mpc) + 1
    t = np.linspace(0, simulation_time, n_steps)
    ref = np.zeros((n_steps, n_states))

    amplitude, freq = 0.8, 0.3
    omega = 2 * np.pi * freq
    ref[:, 0] = amplitude * np.sin(omega * t)          # position-like
    if n_states > 1:
        ref[:, 1] = amplitude * omega * np.cos(omega * t)   # matching velocity-like (d/dt of the line above)

    return ref
```

## What you do NOT need to do

  - No need to `import numpy as np` -- `np` is injected into the file's
    namespace automatically, same as dynamics plugins.
  - No need to handle `n_steps` exceeding what you return by a little --
    the loader only requires *at least* `simulation_time / dt_mpc` steps.
""".strip()


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
    return f"""
You are fixing a Python "custom trajectory" file so it conforms to the
following standard:

{TRAJECTORY_STANDARD}

The file below FAILED validation with this exact error:

    {error_message}

Here is the current file content:

```python
{source_code}
```

Produce a COMPLETE, corrected version of this file that:
  1. Fixes the validation error above.
  2. Preserves the original intended trajectory shape/behavior as faithfully
     as possible -- only fix structural/API issues, don't change what
     trajectory it's meant to describe.
  3. Maintains the position/velocity derivative consistency described in the
     standard, if the file pairs states that way.
  4. Is a complete, standalone, directly-usable .py file (not a diff/patch).

Return the full corrected file content and a short explanation of what was
wrong and what you changed.
""".strip()


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
