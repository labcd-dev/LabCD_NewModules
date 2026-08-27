"""
================================================================================
agents/schemas.py
================================================================================
Structured-output schema the Actor LLM must return. Kept identical in spirit
to the original notebook's ``MPCParameters`` model.

pydantic is an optional dependency of this sub-package (only agents/ needs
it) -- dynamics/ and mpc/ have zero dependency on it, which is why it's
imported lazily here instead of at the top of the whole package.
"""

from __future__ import annotations

from typing import List, Optional

try:
    from pydantic import BaseModel, Field, model_validator
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "pydantic is required for backend_core.AgentMPC.agents (pip install pydantic). "
        "The dynamics/ and mpc/ sub-packages work without it."
    ) from e


class MPCParameters(BaseModel):
    """MPC hyperparameters proposed by the Actor agent."""

    reasoning: str = Field(description="Reasoning for the proposed change, grounded in the Critic's feedback.")
    strategy: str = Field(description="'explore' (large changes) or 'exploit' (fine-tune near the best known params).")

    Np: int = Field(ge=1, le=60, description="Prediction horizon.")
    Nc: int = Field(ge=1, le=60, description="Control horizon, must be <= Np.")
    Q: List[float] = Field(description="State weights, one per state, all positive.")
    R: List[float] = Field(description="Input weights, one per input, all positive.")
    P: Optional[List[float]] = Field(
        default=None,
        description="Terminal weights, same size as Q. This now genuinely changes the controller's "
        "behaviour (see mpc/controller.py) -- unlike in the original version, tuning it is not a no-op.",
    )
    dt: Optional[float] = Field(
        default=None, gt=0.0, le=2.0,
        description="MPC sample time in seconds. Optional -- omit to leave it unchanged from the current "
        "value. Only propose a new one if you have a specific reason (e.g. the system's response looks "
        "under-sampled or unnecessarily fine); most iterations should leave this alone and focus on Q/R/Np/Nc.",
    )

    @model_validator(mode="after")
    def _clamp_horizons(self) -> "MPCParameters":
        if self.Nc > self.Np:
            self.Nc = self.Np
        return self

    def to_dict(self) -> dict:
        return {"Np": self.Np, "Nc": self.Nc, "Q": self.Q, "R": self.R, "P": self.P, "dt": self.dt}
