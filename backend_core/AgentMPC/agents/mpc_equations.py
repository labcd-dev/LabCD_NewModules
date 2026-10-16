"""
================================================================================
agents/mpc_equations.py
================================================================================
The MPC control law, as LaTeX, for whichever configuration a run will use.

One source of truth for both Streamlit frontends. Each renders these steps in
its own visual idiom, but the mathematics comes from here -- so the two cannot
drift apart, and a change to the controller (mpc/controller.py) needs updating
in exactly one place to be shown correctly everywhere.

Pure data, no Streamlit import: returns plain dicts, which also keeps it
usable from a report or an API response.

Terms the disturbance observer adds or alters are wrapped in \\textcolor so a
reader can see at a glance what the toggle changes -- and, just as usefully,
what it does not: only the affine offset moves, so q and the state-constraint
rows change and H stays the same.
"""

from __future__ import annotations

from typing import Dict, List

#: Colour for the terms the disturbance observer introduces or changes.
OBSERVER_COLOUR = "#e8590c"


def _hi(expr: str, on: bool) -> str:
    return r"\textcolor{" + OBSERVER_COLOUR + "}{" + expr + "}" if on else expr


def mpc_algorithm_latex(observer: bool = False, alpha: float = 0.3,
                        terminal_dare: bool = True) -> List[Dict[str, object]]:
    """The control law, one step per entry, in the order the controller runs.

    Each entry is ``{"title", "latex": [str, ...], "caption", "changed"}``.
    ``changed`` marks a step that exists or differs only because the observer
    is on, so a renderer can emphasise it.
    """
    c_sym = _hi(r"\tilde{c}_k", True) if observer else r"c_k"
    sc_sym = _hi(r"\tilde{S}_c", True) if observer else r"S_c"

    steps: List[Dict[str, object]] = [
        {
            "title": "Plant and model",
            "latex": [
                r"\text{plant:}\quad \dot{x} = f_{\text{true}}\big(x,\ \lambda \odot u + d_u\big) + d_x"
                r"\qquad\qquad \text{model:}\quad \dot{x} = f_{\text{nom}}(x,\ u)",
            ],
            "caption": ("The controller only ever predicts with the nominal model. Actuator "
                        "effectiveness lambda, the disturbances d_u and d_x, and any parameter "
                        "perturbation exist only in the plant."),
            "changed": False,
        },
        {
            "title": "Linearise and discretise at the current point (x_k, u_{k-1})",
            "latex": [
                r"A = I + \Delta t\,\frac{\partial f_{\text{nom}}}{\partial x},\qquad "
                r"B = \Delta t\,\frac{\partial f_{\text{nom}}}{\partial u},\qquad "
                r"c_k = \Delta t\,\big(f_{\text{nom}}(x_k,u_{k-1}) - A_c\,x_k\big)",
            ],
            "caption": ("Forward Euler at the sample time. c_k is the linearisation remainder of "
                        "the nominal model -- not a disturbance estimate."),
            "changed": False,
        },
    ]

    if observer:
        steps.append({
            "title": "Disturbance observer -- estimate what the model got wrong last step",
            "latex": [
                _hi(r"\hat{x}_{k|k-1} = A\,x_{k-1} + B\,u_{k-1} + c_{k-1} + \hat{d}_{k-1}", True),
                _hi(r"\varepsilon_k = x_k - \hat{x}_{k|k-1},\qquad "
                    r"\hat{d}_k = \operatorname{sat}\!\big(\hat{d}_{k-1} + "
                    + f"{alpha:.2f}" + r"\,\varepsilon_k\big)", True),
                _hi(r"\tilde{c}_k = c_k + \hat{d}_k", True),
            ],
            "caption": (f"alpha = {alpha:.2f}. d_hat is held constant across the horizon (a "
                        "constant-disturbance model) and saturated at dt x each state's declared "
                        "range, so an unreachable target cannot wind it up. The controller is "
                        "still never told the disturbance -- it infers it from what it measures."),
            "changed": True,
        })

    steps += [
        {
            "title": "Predict over the horizon N_p, with moves Delta U over N_c",
            "latex": [
                r"X = S_x\,x_k + " + sc_sym + r" + S_u\,\Delta U,\qquad "
                + sc_sym + r"[i] = \sum_{j=0}^{i} A^{j}\," + c_sym,
            ],
            "caption": "Condensed prediction: every future state is affine in the moves.",
            "changed": observer,
        },
        {
            "title": "Solve the QP",
            "latex": [
                r"\min_{\Delta U}\ \tfrac12\,\Delta U^{\!\top} H\,\Delta U + q^{\top}\Delta U",
                # The trailing space matters: these two raw strings are
                # concatenated, and without it KaTeX reads '\qquadq' as one
                # unknown command and prints it in red.
                r"H = S_u^{\top}\bar{Q}\,S_u + \bar{R}\qquad "
                r"q = S_u^{\top}\bar{Q}\,\big(S_x x_k + " + sc_sym + r" - X_{\text{ref}}\big)",
                r"L\,\Delta U \in \big[u_{\min} - u_{k-1},\ u_{\max} - u_{k-1}\big]",
                r"S_u\,\Delta U \in \big[x_{\min} - (S_x x_k + " + sc_sym + r"),\ "
                r"x_{\max} - (S_x x_k + " + sc_sym + r")\big]",
            ] + ([r"P = A^{\top}PA - A^{\top}PB\,\big(R + B^{\top}PB\big)^{-1}B^{\top}PA + Q"
                  r"\quad\text{(DARE, every step)}"] if terminal_dare else
                 [r"P = P_{\text{given}}\quad(= Q\ \text{if none is given})"]),
            "caption": ("Q-bar is diag(Q, ..., Q, P) with the terminal weight P on the last block; "
                        + ("P solves the discrete Riccati equation of the current linearisation -- the "
                           "infinite-horizon cost-to-go, which keeps the loop nominally stable even with "
                           "a short horizon. " if terminal_dare else "P is the given terminal weight. ")
                        + "R-bar is I_Nc (x) R. L maps the moves to the cumulative input, so the "
                        "input bound holds at every step of the horizon, not only the first."),
            "changed": observer,
        },
        {
            "title": "Apply the first move (receding horizon)",
            "latex": [r"u_k = u_{k-1} + \Delta u_0^{*}"],
            "caption": "Only the first move is applied; the whole problem is re-solved next step.",
            "changed": False,
        },
    ]
    return steps


def observer_summary(observer: bool) -> str:
    """One sentence on what the toggle does to steady-state behaviour."""
    if observer:
        return ("Offset-free: once d_hat converges the model is exact, and delta-u = 0 then "
                "requires x = x_ref. H is unchanged, so the QP's sparsity pattern -- and the "
                "solver's warm start -- survive.")
    return ("Without the observer, a constant disturbance leaves a permanent state offset: the "
            "incremental form finds the input that cancels it, but can only HOLD that input "
            "while an error persists to generate it. No Q/R/Np/Nc removes it.")
