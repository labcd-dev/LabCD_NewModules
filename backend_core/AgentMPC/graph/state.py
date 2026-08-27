"""
================================================================================
graph/state.py
================================================================================
Shared state schema passed between LangGraph nodes (Scenarist, Actor,
Evaluator, Critic, Terminator, Juror).

Kept as a single, explicit TypedDict so every node's input/output contract is
visible in one place, instead of being implicitly defined by whatever keys
each node function happened to read/write in the original notebook.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class MPCGraphState(TypedDict, total=False):
    # --- system / run context -------------------------------------------------
    system_name: str
    n_states: int
    n_inputs: int
    state_names: List[str]
    input_names: List[str]

    # --- scenario (set by Scenarist) -------------------------------------------
    scenario_level: str          # "I" | "II" | "III"
    initial_state: List[float]
    target_state: List[float]

    # --- current proposal (set by Actor) ----------------------------------------
    current_params: Dict[str, Any]      # {"Np":..., "Nc":..., "Q":[...], "R":[...], "P":[...]}
    strategy: str                       # "explore" | "exploit"
    actor_reasoning: str

    # --- evaluation results (set by Evaluator) ----------------------------------
    current_mse: float
    current_overshoot: float
    current_settling: float
    current_effort: float
    current_per_state_mse: Dict[str, float]     # e.g. {"pole_angle": 0.15, "cart_pos": 0.001, ...}
    current_per_state_overshoot: Dict[str, float]
    current_per_state_iae: Dict[str, float]      # integral |error| dt, per state -- meaningful for any trajectory type
    current_per_state_ise: Dict[str, float]       # integral error^2 dt, per state
    current_is_regulation: bool                     # False for a moving (sin/pulse) reference -- see agents/metrics.py
    current_oscillation_count: int
    current_unstable: bool
    avg_solve_time: float
    eval_error: Optional[str]
    eval_traceback: Optional[str]
    # UI-facing fields (app.py). These MUST be declared here -- LangGraph builds
    # its state channels strictly from this TypedDict's fields, so any key a
    # node returns that isn't declared here gets silently dropped when the
    # state is merged/streamed. That was the actual root cause of the
    # Streamlit UI seeing `eval_error=None` but `metrics={}`: evaluator_node
    # was always constructing and returning `metrics`/`simulation_data`/
    # `success`/`exploration_strategy` correctly, but LangGraph discarded them
    # before app.py ever saw them, because they weren't part of the schema.
    metrics: Dict[str, Any]
    simulation_data: Dict[str, Any]
    success: bool
    unstable: bool
    exploration_strategy: str

    # --- running best-so-far (maintained across iterations, ranked by the
    #     composite scalar_cost -- see agents/metrics.py -- not raw MSE) --------
    best_mse: float
    best_cost: float
    best_params: Dict[str, Any]
    best_overshoot: float
    best_settling: float
    best_effort: float

    # --- history (for trend-aware prompts / plotting) ---------------------------
    mse_history: List[float]
    overshoot_history: List[float]
    settling_history: List[float]
    effort_history: List[float]
    params_history: List[Dict[str, Any]]

    # --- critic / terminator / juror outputs -------------------------------------
    critic_feedback: str
    should_continue: bool
    termination_reason: str
    juror_verdict: str

    # --- bookkeeping --------------------------------------------------------
    iteration: int
    max_iterations: int
    history: List[str]     # free-text log entries (mirrors old add_to_history)
    ui_scenario_level: int   # 1/2/3, set by the Streamlit UI's dropdown (bypasses the LLM Scenarist)
    user_guidance: str        # free-text steering from the user (e.g. "only minimize control effort");
                                # empty string means "no extra guidance, use the default balanced objective"
    cost_weights: Optional[Dict[str, float]]   # from app.py's "Optimization Focus" selector -- see
                                                  # agents/metrics.py:OPTIMIZATION_FOCUS_PRESETS. None = balanced default.
    min_explore_iterations: int   # Critic can't recommend 'exploit' before this many iterations have run
    dt_mpc: float           # only SET in state when the Juror actually changes it (see agents/juror.py) --
                               # the authoritative value always lives in cfg.data.dt_mpc; this mirror exists
                               # purely so the UI/history can show when and to what it was changed.
    dt_tuned_by_juror: bool  # True once the Juror has changed dt_mpc at least once this run
    token_tracker: Any  # optional TokenUsageTracker (see agents/llm_base.py) -- accumulates LLM token usage
                          # across every agent call in this run when set; None disables tracking entirely
    last_outputs: Dict[str, str]   # {"actor": "...", "critic": "...", ...} -- the exact prompt text each
                                     # agent was last given, merged in by each node (see agents/*.py). Used
                                     # by app.py's live flow diagram to show a hover tooltip per node.
    exploration_intensity: int     # 1-100, from app.py's "Exploration Intensity" slider -- how bold the
                                     # Actor is during 'explore' (see agents/actor.py). 50 = default behavior.
    _next: str    # internal: Terminator's routing decision ("critic"/"juror"/"end"), read by
                   # graph/workflow.py's conditional edge. Also would be silently dropped if
                   # left undeclared -- see the comment on the UI-facing fields above.
