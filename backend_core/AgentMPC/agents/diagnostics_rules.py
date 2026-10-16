"""
================================================================================
agents/diagnostics_rules.py
================================================================================
Every rule the Diagnostics assistant applies, written out as formulas.

Rendered by both frontends in the Diagnostics tab ("How the assistant
decides") so a user can see exactly WHY something was raised, not just THAT it
was. Pure data, no Streamlit import.

The thresholds are not restated here -- they are imported from the modules
that apply them (mpc_suitability, diagnostics_agent, metrics). Change a
threshold there and this page changes with it; the explanation can never
describe a rule the code no longer follows.

Shape: a list of groups, each ``{"group", "intro", "rules": [...]}``, each rule
``{"title", "question", "latex": [str], "decision", "severity"}``. ``severity``
is the strongest level the rule can raise: "error", "warning", "info", or
"basis" for definitions the other rules build on.
"""

from __future__ import annotations

from typing import Any, Dict, List

from . import mpc_suitability as S
from .diagnostics_agent import (
    FREQUENT_INSTABILITY_FRACTION, FREQUENT_INSTABILITY_MIN, SOLVER_STRUGGLE_RATIO,
    _ERROR_CATEGORIES,
)


def _pct(x: float) -> str:
    return f"{x * 100:g}%"


def diagnostics_rules() -> List[Dict[str, Any]]:
    acc = S.EULER_ACCURACY_FACTOR
    return [
        # ------------------------------------------------------------------
        {"group": "Plant suitability",
         "intro": "Before anything is tuned: can this MPC control this plant at all? Evaluated at the "
                  "declared target and the equilibrium input -- the operating point the controller "
                  "spends its time near.",
         "rules": [
             {"title": "Linearisation at the operating point", "severity": "basis",
              "question": "The model every check below reasons about.",
              "latex": [r"A_c = \left.\frac{\partial f}{\partial x}\right|_{(x^{*},\,u_{eq})},\qquad "
                        r"B_c = \left.\frac{\partial f}{\partial u}\right|_{(x^{*},\,u_{eq})},\qquad "
                        r"\lambda_1,\dots,\lambda_n = \operatorname{eig}(A_c)"],
              "decision": "Each eigenvalue is a mode: its real part says whether it decays or grows, "
                          "its imaginary part whether it oscillates."},

             {"title": "The dynamics can be evaluated", "severity": "error",
              "question": "Does f(x, u) return a usable number where the run starts and ends?",
              "latex": [r"f(x_0,\ u_{eq}) \in \mathbb{R}^n \quad\text{and}\quad "
                        r"f(x^{*},\ u_{eq}) \in \mathbb{R}^n \quad (\text{finite})"],
              "decision": "Any NaN/Inf, or an exception -> error, and no other check runs: "
                          "linearisation, prediction and the QP all start from here."},

             {"title": "Stabilisability (PBH test)", "severity": "error",
              "question": "Can ANY controller bring every unstable mode to rest?",
              "latex": [r"\forall\,\lambda_i:\ \operatorname{Re}\lambda_i \ge 0 \;\Rightarrow\; "
                        r"\operatorname{rank}\begin{bmatrix}\lambda_i I - A_c & B_c\end{bmatrix} = n"],
              "decision": "Rank below n means that mode is unreachable from every input. If it is "
                          "also unstable or marginal it grows unchecked -> error. This is the only "
                          "'not suitable' that no setting can fix."},

             {"title": "Controllability (Kalman rank)", "severity": "warning",
              "question": "Can the plant be steered to ANY state?",
              "latex": [r"\operatorname{rank}\begin{bmatrix} B_c & A_cB_c & A_c^2B_c & \cdots & "
                        r"A_c^{\,n-1}B_c\end{bmatrix} = n"],
              "decision": "Rank below n while stabilisable -> warning: the uncontrollable modes "
                          "settle on their own but cannot be steered anywhere."},

             {"title": "Open-loop instability", "severity": "info",
              "question": "Would it run away without control, and how fast?",
              "latex": [r"t_{\text{double}} = \frac{\ln 2}{\max_i \operatorname{Re}\lambda_i},"
                        r"\qquad \operatorname{Re}\lambda_i > 0"],
              "decision": "Info only -- MPC handles it -- but dt must be well below the doubling time "
                          "and Np x dt a few times longer."},

             {"title": "Input authority at the target", "severity": "error",
              "question": "Can the actuators hold the target at all?",
              "latex": [r"u_{\min} \le u_{eq} \le u_{\max}"],
              "decision": "Outside the bounds -> error: holding the target needs more input than the "
                          "actuators have."},

             {"title": "Forward Euler: a stable mode predicted unstable", "severity": "error",
              "question": "Does the controller's own discretisation turn a stable mode unstable?",
              "latex": [r"z_i = 1 + dt\,\lambda_i,\qquad |z_i| < 1 \iff dt < "
                        r"\frac{-2\operatorname{Re}\lambda_i}{|\lambda_i|^2}"],
              "decision": "For a mode with Re < 0 and |z| >= 1 -> error, with that bound as the "
                          "largest usable dt. The controller would be planning against a system "
                          "that does not exist."},

             {"title": "Forward Euler: an undamped oscillation inflated", "severity": "error",
              "question": "How much does Euler inflate a mode with Re = 0 over a horizon?",
              "latex": [r"|1 + j\,dt\,\omega| = \sqrt{1 + (dt\,\omega)^2} > 1 \quad \forall\, dt",
                        rf"g = |1 + dt\,\lambda|^{{{S.EULER_GROWTH_HORIZON}}}"],
              "decision": f"Always inflated -- the question is how much. g > {S.EULER_GROWTH_ERROR:g} "
                          f"-> error, g > {S.EULER_GROWTH_WARN:g} -> warning, over "
                          f"{S.EULER_GROWTH_HORIZON} steps."},

             {"title": "Forward Euler: accuracy", "severity": "warning",
              "question": "Is dt fine enough for the fastest mode even when it is stable?",
              "latex": [rf"|\lambda|_{{\max}}\,dt > {S.EULER_COARSE_MULTIPLE * acc:g}"
                        rf"\qquad dt_{{\text{{recommended}}}} = \frac{{{acc:g}}}{{|\lambda|_{{\max}}}}"],
              "decision": f"Above the bound -> warning. The recommended dt keeps the fastest mode's "
                          f"per-step Euler error to about one percent."},

             {"title": "Stiffness", "severity": "warning",
              "question": "Are very fast and very slow modes present at once?",
              "latex": [rf"\frac{{\max_i|\lambda_i|}}{{\min_i|\lambda_i|}} > {S.STIFFNESS_RATIO:g}"
                        r"\qquad (\lambda_i \ne 0)"],
              "decision": "-> warning: the fast mode needs a small dt, the slow one a long horizon, "
                          "so Np grows large and every QP gets expensive."},

             {"title": "Nonlinearity over the operating envelope", "severity": "warning",
              "question": "How far off is the linear model around the target?",
              "latex": [r"f_{\text{lin}}(x) = f(x^{*}) + A_c\,(x - x^{*})",
                        r"e(x) = \frac{\lVert f(x) - f_{\text{lin}}(x)\rVert}"
                        r"{\lVert f(x) - f(x^{*})\rVert},\qquad "
                        rf"x \sim x^{{*}} \pm {S.NONLINEAR_SPREAD:g}\cdot\text{{range}}"],
              "decision": f"Median of e over {S.NONLINEAR_SAMPLES} random points: "
                          f"> {_pct(S.NONLINEAR_WARN)} -> warning (keep Np x dt moderate), "
                          f"> {_pct(S.NONLINEAR_INFO)} -> info. The denominator measures the error "
                          f"against the change that actually happened, not against a fixed scale."},

             {"title": "Scaling", "severity": "info",
              "question": "Do the states differ wildly in size?",
              "latex": [rf"\frac{{\max_i \text{{range}}_i}}{{\min_i \text{{range}}_i}} > {S.SCALE_RATIO:g}"],
              "decision": "-> info: start from the Setup Agent's role-based weights rather than "
                          "uniform ones."},

             {"title": "Verdict", "severity": "basis",
              "question": "How the findings combine into one answer.",
              "latex": [r"\text{verdict} = \begin{cases}\text{not suitable} & \exists\ \text{error}\\"
                        r"\text{suitable, with caveats} & \exists\ \text{warning}\\"
                        r"\text{suitable} & \text{otherwise}\end{cases}"],
              "decision": "Thresholds are calibrated so that an ordinary, sensible setup raises "
                          "nothing -- an assistant that warns about everything gets ignored."},
         ]},

        # ------------------------------------------------------------------
        {"group": "Actuation and target",
         "intro": "How many state values can be held at once, and is the target one of them? "
                  "Counting inputs against states is the wrong test -- a double integrator has two "
                  "states and one input and holds a position perfectly well.",
         "rules": [
             {"title": "Classification", "severity": "info",
              "question": "Under-, fully, or over-actuated?",
              "latex": [r"\operatorname{rank}B_c = n \Rightarrow \text{fully actuated},\qquad "
                        r"n_u > \operatorname{rank}B_c \Rightarrow \text{over-actuated}"],
              "decision": "Otherwise under-actuated."},

             {"title": "Independently holdable states", "severity": "basis",
              "question": "How many state values can be dialled in at the same time?",
              "latex": [r"f(x^{*},u^{*}) = 0 \;\Rightarrow\; A_c\,x^{*} \in \operatorname{range}(B_c)",
                        r"N:\ \text{left null space of } B_c \quad(\text{from the SVD})",
                        r"\text{holdable} = n - \operatorname{rank}\!\left(N^{\top}A_c\right)"],
              "decision": "The reachable equilibria form the subspace {x : N^T A_c x = 0}; its "
                          "dimension is the number of states that can be targeted independently."},

             {"title": "Is the target a resting point?", "severity": "error",
              "question": "Is there an input that holds the plant exactly at the target?",
              "latex": [r"\Delta u^{*} = \arg\min_{\Delta u}\ \big\lVert f(x^{*},u_{eq}) + "
                        r"B_c\,\Delta u\big\rVert,\qquad r = f(x^{*},u_{eq}) + B_c\,\Delta u^{*}"],
              "decision": "r != 0 -> error, naming the states the residual lands on. The run would "
                          "converge to the nearest reachable point and report the gap as bad tuning."},

             {"title": "Derivative pairs", "severity": "warning",
              "question": "Is a velocity given a nonzero constant target?",
              "latex": [r"x_j = \dot x_i \;\text{(detected pair)},\quad x_j^{*} \ne 0 \;\Rightarrow\;"
                        r"\text{not a steady state}"],
              "decision": "-> warning: if the velocity holds, the position never stops moving."},
         ]},

        # ------------------------------------------------------------------
        {"group": "Reference",
         "intro": "When a reference is chosen: can the plant actually MOVE that way? The general form "
                  "of the target check -- a constant reference reduces to it.",
         "rules": [
             {"title": "Dynamic feasibility, sample by sample", "severity": "error",
              "question": "At each instant, is there an input producing the motion the reference asks for?",
              "latex": [r"\dot r_k \approx \frac{r_{k+1} - r_k}{dt}",
                        r"u \leftarrow u + \arg\min_{\Delta u}\big\lVert B(r_k,u)\,\Delta u + "
                        r"f(r_k,u) - \dot r_k\big\rVert \quad(\times 2,\ \text{from } u_{eq})",
                        rf"\frac{{\lVert f(r_k,u) - \dot r_k\rVert}}{{\text{{scale}}}} > "
                        rf"{S.REFERENCE_RESIDUAL_TOL:g} \;\Rightarrow\; \text{{infeasible sample}}"],
              "decision": "A CONSTANT reference with infeasible samples -> error (a setup mistake). "
                          "A MOVING one -> warning (a trade-off: on an under-actuated plant moving one "
                          "state disturbs the coupled ones), naming which states absorb the error. "
                          "Gauss-Newton converges in one step for plants affine in u."},

             {"title": "Velocity is the derivative of its position", "severity": "warning",
              "question": "Do a position reference and its velocity reference agree?",
              "latex": [rf"\frac{{\max_k\big|\dot r^{{\,\text{{pos}}}}_k - r^{{\,\text{{vel}}}}_k\big|}}"
                        rf"{{\text{{scale}}}} > {S.DERIVATIVE_MISMATCH_TOL:g}"],
              "decision": "-> warning. The classic mistake: a sinusoid on the position with the "
                          "velocity held at zero."},

             {"title": "Actuator authority along the way", "severity": "warning",
              "question": "Does following the reference need more input than the bounds allow?",
              "latex": [r"\exists\,k:\ u_k \notin [u_{\min},\ u_{\max}]"],
              "decision": "-> warning: the controller will saturate there and fall behind."},

             {"title": "Finite", "severity": "error",
              "question": "Any NaN/Inf in the reference?",
              "latex": [r"r_k \in \mathbb{R}^n\ \ \forall k"],
              "decision": "-> error: every proposal would fail at step 0, since the reference does not "
                          "depend on Q, R, Np or Nc."},
         ]},

        # ------------------------------------------------------------------
        {"group": "During and after the run",
         "intro": "Raised from what the run actually did.",
         "rules": [
             {"title": "QP data is finite", "severity": "error",
              "question": "Did a NaN reach the solver?",
              "latex": [r"P,\ q,\ A \in \mathbb{R},\qquad l,\ u \in \mathbb{R}\cup\{\pm\infty\}"],
              "decision": "Checked before every solve, and the offending vector is named. A NaN in q "
                          "alone otherwise reaches OSQP and is mislabelled 'maximum iterations reached'."},

             {"title": "Frequent instability", "severity": "warning",
              "question": "Did many iterations diverge?",
              "latex": [rf"\frac{{n_{{\text{{unstable}}}}}}{{n_{{\text{{iter}}}}}} > "
                        rf"{FREQUENT_INSTABILITY_FRACTION:g}\quad\text{{and}}\quad "
                        rf"n_{{\text{{unstable}}}} \ge {FREQUENT_INSTABILITY_MIN}"],
              "decision": "-> finding passed to the Diagnostics Agent."},

             {"title": "Solver struggling", "severity": "warning",
              "question": "Did the QP solver often fail to converge cleanly?",
              "latex": [rf"\frac{{n_{{\text{{inaccurate}}}} + n_{{\text{{cap}}}} + n_{{\text{{failed}}}}}}"
                        rf"{{n_{{\text{{iter}}}}}} > {SOLVER_STRUGGLE_RATIO:g}"],
              "decision": "KNOWN FLAW: the numerator counts individual QP solves (hundreds per "
                          "iteration) while the denominator counts iterations, so a single bad solve "
                          "in a short run already crosses the threshold. The fix is to divide by the "
                          "total number of solves -- pending your go-ahead."},

             {"title": "Keyword scan of the logs", "severity": "warning",
              "question": "Do the agents' messages mention a known failure?",
              "latex": [r"\exists\ \text{keyword} \in \text{log message}"],
              "decision": "Categories: " + "; ".join(
                  f"{c['title']} ({', '.join(c['keywords'][:3])}...)"
                  for c in _ERROR_CATEGORIES.values() if c.get("keywords"))},
         ]},

        # ------------------------------------------------------------------
        {"group": "Result verdicts",
         "intro": "How a finished run is judged.",
         "rules": [
             {"title": "Stable: the error stays bounded", "severity": "basis",
              "question": "Is the last part of the run worse than anything before it?",
              "latex": [r"\text{Stable} \iff \neg\,\text{stopped early}\ \land\ "
                        r"\text{RMS}_{4} \le 1.05\ \max\big(\text{RMS}_{1},\ \text{RMS}_{2},\ "
                        r"\text{RMS}_{3}\big)\qquad(\text{quarters of the run})"],
              "decision": "Compared with the WORST earlier quarter, not the first, so a regulator that "
                          "starts on its setpoint, or a periodic tracking error, is not mislabelled "
                          "unstable. A run stopped early (Status UNSTABLE) is never Stable: its "
                          "quarters cover only the truncated window."},

             {"title": "Best iteration", "severity": "basis",
              "question": "Which iteration is reported?",
              "latex": [r"\text{best} = \arg\min_{\text{stable rows}} J"
                        r"\quad(\text{all rows if none is stable})"],
              "decision": "unstable (the simulation blew up) is excluded outright; is_stable is "
                          "preferred."},

             {"title": "Run verdict", "severity": "basis",
              "question": "Did the run find a usable controller?",
              "latex": [r"\text{FAIL} \iff \text{no iterations} \ \lor\ \text{nothing usable}\ "
                        r"\lor\ \nexists\ \text{stable iteration}"],
              "decision": "Judges the whole run, and says so when the lowest-cost iteration is not "
                          "the one reported."},
         ]},
    ]
