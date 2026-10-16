"""
================================================================================
frontend_streamlit/reference_ui.py
================================================================================
Three guards around what a user may ask an AgentMPC run to do -- shared by both
AgentMPC frontends so the two behave identically:

  1. the REFERENCE GATE: on an under-actuated plant, more states given their own
     trajectory than the inputs can steer independently is refused with a
     pop-up that says why (agents/reference_authority.py:reference_authority);
  2. the FEASIBLE REFERENCE option (experimental, opt-in): one primary state and
     where it should go, and AgentMPC computes what every other state must do
     for it to get there, backstepping-style (FeasibleReference);
  3. the DISTURBANCE LIMIT: the Level-4 disturbance slider stops at the largest
     push this plant can still be held against (agents/disturbance_limits.py).

Only native Streamlit components are used, so each app's own theme styles them.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import streamlit as st

from backend_core.AgentMPC.agents.disturbance_limits import disturbance_limits
from backend_core.AgentMPC.agents.reference_authority import (
    DEFAULT_APPROACH_TIME, FeasibleReference, moving_states, reference_authority,
)

#: The Level-4 slider's own upper end, in percent, when the plant allows more.
DISTURBANCE_SLIDER_MAX = 25.0


# =============================================================================
# 1. the reference gate
# =============================================================================
@st.dialog("This reference is not allowed", width="large")
def _reference_refused(auth: Dict[str, Any]) -> None:
    st.error(auth["title"])
    for para in auth["message"].split("\n\n"):
        st.markdown(para)
    st.latex(r"G_S(s) = C_S\,(sI - A)^{-1} B, \qquad "
             rf"\operatorname{{rank}} G_S = {auth['rank']} \;<\; |S| = {len(auth['requested'])}")
    if st.button("OK, I will change the reference", type="primary", key="reference_refused_ok"):
        st.rerun()


def reference_gate(dyn, reference, key: str) -> Optional[str]:
    """Check ``reference`` against the plant's actuation. Returns None when
    it is allowed; otherwise opens the pop-up (once per distinct request),
    shows an inline error and returns the error text, which the caller uses
    to block the launch."""
    if reference is None:
        return None
    ref = np.asarray(reference, dtype=float)
    moving = tuple(moving_states(ref))
    cache = st.session_state.setdefault("_reference_gate_cache", {})
    ck = (id(dyn), moving)
    if ck not in cache:
        try:
            cache[ck] = reference_authority(dyn, moving=moving)
        except Exception:  # noqa: BLE001 -- a failed check must not block a run
            cache[ck] = None
    auth = cache[ck]
    seen_key = f"{key}_refused_for"
    if auth is None or auth["allowed"]:
        st.session_state.pop(seen_key, None)
        return None
    request = tuple(auth["requested"])
    if st.session_state.get(seen_key) != request:
        st.session_state[seen_key] = request
        _reference_refused(auth)
    short = (f"Not allowed: {', '.join(auth['requested'])} cannot all follow their own trajectories -- "
             f"this plant can steer only {auth['rank']} of them independently. Keep a trajectory on "
             f"at most {auth['rank']} (e.g. {', '.join(auth['keep_options'])}), or use the feasible "
             f"reference option.")
    st.error(short)
    return short


# =============================================================================
# 2. the feasible (backstepping-like) reference
# =============================================================================
def feasible_reference_controls(dyn, key: str) -> Optional[FeasibleReference]:
    """The opt-in controls. Returns the configured FeasibleReference, or
    None when the option is off."""
    on = st.checkbox(
        "Feasible reference for an under-actuated plant (backstepping-like, experimental)",
        value=False, key=f"{key}_feasible_on",
        help="Choose ONE state you care about and where it should go. AgentMPC computes the "
             "trajectory every other state must follow for it to get there while the plant stays "
             "stable -- their 'virtual references', as in backstepping -- by simulating the model "
             "under a stabilising LQR. The MPC then tracks a trajectory the plant can actually do.")
    if not on:
        return None
    names = list(dyn.state_names)
    target = np.asarray(dyn.config.default_target, dtype=float)
    c1, c2 = st.columns(2)
    with c1:
        primary_name = st.selectbox("Primary state", names, key=f"{key}_feasible_primary",
                                    help="The state whose value you care about.")
    p = names.index(primary_name)
    with c2:
        shape_label = st.radio("Its reference", ["Step to a value", "Sinusoid"], horizontal=True,
                               key=f"{key}_feasible_shape")
    shape = "step" if shape_label.startswith("Step") else "sin"
    goal, amp, freq = None, 0.5, 0.2
    c3, c4, c5 = st.columns(3)
    if shape == "step":
        with c3:
            goal = st.number_input(f"Desired {primary_name}", value=float(target[p]), format="%.4f",
                                   key=f"{key}_feasible_goal")
    else:
        with c3:
            amp = st.number_input("Amplitude", min_value=0.0, value=0.5, format="%.4f",
                                  key=f"{key}_feasible_amp")
        with c4:
            freq = st.number_input("Frequency (Hz)", min_value=0.001, value=0.2, format="%.3f",
                                   key=f"{key}_feasible_freq")
    with c5:
        approach = st.number_input(
            "Approach time (s)", min_value=0.05, max_value=120.0, value=float(DEFAULT_APPROACH_TIME),
            step=0.5, key=f"{key}_feasible_approach",
            help="How quickly the primary state's command moves to the new value (about 98% of the "
                 "way in this time). Raised automatically if the trajectory would diverge or leave "
                 "the bounds.")
    st.latex(r"f(x_{eq}, u_{eq}) = 0,\ \ x_{eq,p} = r(t) \qquad "
             r"u_k = \operatorname{sat}\!\big(u_{eq} - K\,\delta x_k - G B^{\top} s_{k+1}\big),\ \ "
             r"s_k = (A - BK)^{\top} s_{k+1} - Q\,\delta x_{eq}(r_k)"
             r"\qquad x_{ref}(t):\ \dot{x} = f(x, u)")
    st.caption("The steady state the primary state asks for is solved from the model; the other states' "
               "references are the trajectory they take, on the full nonlinear model, under a "
               "stabilising LQ tracker with preview (feedback, plus a feedforward that anticipates the "
               "known command) -- so the whole reference is something the plant can actually do, and "
               "it ends exactly where you asked. The reference-type cards above are not used.")
    return FeasibleReference(dyn, p, goal=goal, shape=shape, amplitude=amp, frequency=freq,
                             approach_time=approach)


def feasible_reference_preview(fr: FeasibleReference, duration: float, dt: float) -> Dict[str, Any]:
    """Generate the reference for the preview and report on it. Returns the
    generator's result dict (``reference`` is None when there is none)."""
    try:
        res = fr.generate(dt, duration)
    except Exception as exc:  # noqa: BLE001
        res = {"ok": False, "reference": None, "message": f"Could not compute the reference: {exc}"}
    if res.get("reference") is None or not np.all(np.isfinite(res["reference"])):
        st.error(res["message"])
        res["reference"] = None
    elif res["ok"]:
        (st.success if res.get("settled", True) else st.info)(res["message"])
    else:
        st.warning(res["message"])
    return res


# =============================================================================
# 3. the disturbance limit
# =============================================================================
def _limits(dyn, effectiveness, per_state: bool = False) -> Dict[str, Any]:
    lam = np.atleast_1d(np.asarray(1.0 if effectiveness is None else effectiveness, dtype=float))
    acc = getattr(dyn.config, "acceptable_deviation", None)
    ck = (id(dyn), tuple(np.round(lam, 4)), None if acc is None else tuple(np.round(acc, 9)), per_state)
    cache = st.session_state.setdefault("_disturbance_limit_cache", {})
    if ck not in cache:
        try:
            with st.spinner("Working out the largest disturbance this plant can be held against..."):
                cache[ck] = disturbance_limits(dyn, effectiveness=lam, per_state=per_state)
        except Exception as exc:  # noqa: BLE001
            cache[ck] = {"ok": None, "error": str(exc)}
    return cache[ck]


def disturbance_slider(dyn, effectiveness, key: str, label: str, default_pct: float,
                       help_text: str = "") -> float:
    """The Level-4 disturbance strength slider, capped at this plant's limit.
    Returns the chosen FRACTION (0.05 = 5%)."""
    lim = _limits(dyn, effectiveness)
    if lim.get("ok") is None:
        st.caption(f"Disturbance limit could not be computed ({lim.get('error')}); the slider is "
                   f"not capped.")
        cap = DISTURBANCE_SLIDER_MAX
    else:
        cap = min(DISTURBANCE_SLIDER_MAX, 100.0 * float(lim["max_fraction"] or 0.0))
    min_lam = lim.get("min_effectiveness")
    lam_arr = np.atleast_1d(np.asarray(1.0 if effectiveness is None else effectiveness, dtype=float))
    if min_lam is not None and np.any(lam_arr < min_lam - 1e-9):
        st.warning(f"Actuator effectiveness below λ = {min_lam:.2f} cannot even hold this plant at its "
                   f"target, before any disturbance -- the run would fail whatever the tuning.")
    if lim.get("ok") is False or cap <= 0.0:
        st.error("No external disturbance is admissible here. " + lim.get("message", ""))
        st.session_state[key] = 0.0
        st.slider(label, 0.0, 1.0, 0.0, disabled=True, key=f"{key}_locked")
        return 0.0
    step = 0.5 if cap >= 5.0 else max(round(cap / 20.0, 3), 0.001)
    cap = round(float(np.floor(cap / step) * step) if cap >= step else float(cap), 6)
    # Seed and clamp through session state only (passing value= as well makes
    # Streamlit warn): a value kept from another plant may exceed this cap.
    if st.session_state.get(key) is None:
        st.session_state[key] = min(float(default_pct), cap)
    elif st.session_state[key] > cap:
        st.session_state[key] = cap
    value = st.slider(label, min_value=0.0, max_value=cap, step=step, key=key, help=help_text)
    if cap < DISTURBANCE_SLIDER_MAX:
        st.caption(f"Capped at {cap:.3g}% for this plant. {lim.get('message', '')}")
    if st.button("Show the limit per state", key=f"{key}_per_state"):
        st.session_state[f"{key}_show_per_state"] = True
    if st.session_state.get(f"{key}_show_per_state"):
        full = _limits(dyn, effectiveness, per_state=True)
        per = full.get("per_state_fraction")
        if per:
            import pandas as pd

            st.dataframe(pd.DataFrame({
                "State": list(dyn.state_names),
                "Largest push on this state alone (% of range/s)": [
                    ">= 100" if v >= 1.0 else f"{100 * v:.3g}" for v in per],
            }), hide_index=True, width="stretch")
    return value / 100.0
