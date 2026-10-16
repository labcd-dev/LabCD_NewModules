"""
Run with: pytest backend_core/AgentMPC/tests/test_stability_and_verdict.py -v

Regression guard for "the PDF says FAIL for a system that was stabilised".

Three separate defects produced that, and they compound:

  1. The "best iteration" was ranked on composite cost alone, which never
     looks at `is_stable`. Note that `unstable` and `is_stable` are different
     fields: `unstable` means the simulation actually blew up, `is_stable` is
     the metric-side judgement that the error stayed bounded. A run could
     therefore complete perfectly and still fail the stability check.

  2. The verdict then read `is_stable` off whichever row the ranking had
     nominated, so a run with seven stable controllers was branded FAIL
     because the single cheapest one missed a lenient threshold. Worse, this
     got likelier the BETTER the search did: cost has no stability term, so
     the winner tends to be the most aggressive tuning.

  3. The stability check itself compared the last quarter of the run against
     the FIRST quarter, which assumes every run starts with a large transient.
     A regulator already sitting on its setpoint has no transient, so its
     first quarter is the quietest part of the run by construction and almost
     anything later exceeds it -- and on a moving reference the error is
     periodic, so the comparison measures phase rather than stability.
"""

import numpy as np
import pytest

from backend_core.AgentMPC.agents.actor import MAX_NC_FRACTION_OF_NP, _repair_dimensions
from backend_core.AgentMPC.agents.metrics import compute_metrics
from backend_core.AgentMPC.agents.report_pdf import _verdict
from backend_core.AgentMPC.agents.schemas import MPCParameters

DT = 0.02
N = 400
T = np.arange(N) * DT


def _is_stable(signal, is_regulation=True):
    """Run one synthetic error trajectory through the real metric code."""
    states = np.column_stack([signal, np.zeros_like(signal)])
    reference = np.zeros_like(states)
    inputs = np.zeros((len(signal) - 1, 1))
    return compute_metrics(states, inputs, reference, dt=DT, settling_tolerance=0.05,
                            is_regulation=is_regulation).is_stable


# --------------------------------------------------------------- criterion --
@pytest.mark.parametrize("name, signal", [
    ("decaying transient", 0.5 * np.exp(-1.5 * T)),
    ("already at setpoint, noise only",
     np.random.default_rng(3).normal(0, 0.004, N)),
    ("bounded oscillation of constant amplitude", 0.1 * np.sin(2 * np.pi * 0.7 * T)),
])
def test_bounded_errors_count_as_stable(name, signal):
    """The near-setpoint case is the one that regressed: with no transient to
    inflate it, the first quarter is the quietest window in the run."""
    assert _is_stable(signal) is True, name


def test_a_periodic_tracking_error_is_stable():
    """A moving reference never lets the error decay. That is not instability."""
    assert _is_stable(0.2 * np.sin(2 * np.pi * 0.5 * T), is_regulation=False) is True


@pytest.mark.parametrize("name, signal", [
    ("exponential growth", 0.01 * np.exp(1.2 * T)),
    ("linear growth", 0.01 + 0.4 * T / T[-1]),
    ("oscillation with growing amplitude",
     0.02 * (1 + 6 * T / T[-1]) * np.sin(2 * np.pi * 0.7 * T)),
    ("settles, then walks away late",
     np.concatenate([0.4 * np.exp(-2.0 * T[:300]), 0.4 * np.linspace(0.01, 1.2, N - 300)])),
])
def test_growing_errors_are_still_rejected(name, signal):
    """The looser criterion must not become a rubber stamp."""
    assert _is_stable(signal) is False, name


# ----------------------------------------------------------------- verdict --
def _row(iteration, cost, is_stable, mse=0.01):
    return {"iteration": iteration, "ok": True, "unstable": False, "is_stable": is_stable,
            "mse": mse, "settling": 2.0, "cost": cost}


def _select_best(rows):
    """The ranking both frontends use, post-fix: prefer a stable iteration."""
    cand = [r for r in rows if r["ok"] and not r.get("unstable") and r.get("cost") is not None]
    if not cand:
        return None
    return min([r for r in cand if r.get("is_stable")] or cand, key=lambda r: r["cost"])


def test_a_run_that_found_a_stable_controller_passes():
    """Seven stable iterations must not be reported as a failed run just
    because the cheapest one missed the stability check."""
    rows = [_row(1, 5.10, True), _row(2, 5.05, True), _row(3, 4.90, False)]
    best = _select_best(rows)

    assert best["iteration"] == 2, "should pick the cheapest STABLE row, not the cheapest row"
    ok, label = _verdict(rows, best)
    assert ok is True
    assert "PASS" in label


def test_the_verdict_says_when_a_cheaper_unstable_row_was_passed_over():
    """The parameters printed elsewhere in the report are the reported row's,
    so the reader has to be told it is not the lowest-cost one."""
    rows = [_row(1, 5.10, True), _row(2, 5.05, True), _row(3, 4.90, False)]
    _, label = _verdict(rows, _select_best(rows))

    assert "lowest-cost iteration (3)" in label
    assert "iteration 2 is reported instead" in label


def test_fail_is_kept_when_nothing_was_stable():
    rows = [_row(1, 5.10, False), _row(2, 4.90, False)]
    ok, label = _verdict(rows, _select_best(rows))

    assert ok is False
    assert "no iteration produced a stable controller" in label


def test_no_iterations_and_no_usable_row_still_fail():
    assert _verdict([], None)[0] is False
    assert _verdict([_row(1, 5.0, True)], None)[0] is False


# ------------------------------------------------------------------ Nc cap --
@pytest.mark.parametrize("np_horizon, proposed_nc, expected", [
    (12, 12, 3),    # 20% of 12 = 2.4 -> 3
    (12, 3, 3),     # already at the cap, untouched
    (12, 2, 2),     # below the cap, untouched
    (20, 20, 4),
    (30, 6, 6),     # exactly 20% of 30, untouched
    (3, 3, 1),      # 20% of 3 = 0.6 -> 1
    (1, 1, 1),      # never clamped below 1
])
def test_nc_is_capped_at_20_percent_of_np(np_horizon, proposed_nc, expected):
    """An over-long control horizon buys very little -- the last moves hardly
    shift the prediction -- while every one of the hundreds of QP solves per
    iteration gets more expensive, since the QP carries Nc x n_inputs
    decision variables. The prompt states the rule; this enforces it, because
    a model can always ignore an instruction."""
    proposal = MPCParameters(Np=np_horizon, Nc=proposed_nc, Q=[1.0] * 4, R=[0.1],
                              P=[1.0] * 4, strategy="explore", reasoning="test")

    assert _repair_dimensions(proposal, 4, 1)["Nc"] == expected


def test_the_cap_matches_what_the_prompt_promises():
    from backend_core.AgentMPC.agents.prompt_library import get_prompt

    assert MAX_NC_FRACTION_OF_NP == 0.2
    assert "20% of Np" in get_prompt("actor")


def test_a_run_stopped_early_is_never_reported_stable():
    """Status UNSTABLE and Stable Yes on the same row was a contradiction.

    The run below climbs towards z = 1 and is cut off by the plugin's termination
    check at z = 0.6. Up to that point its error only shrank, so the quarter-RMS test
    on the truncated window read it as bounded -- and the results table
    showed the iteration as both UNSTABLE and Stable.
    """
    from pathlib import Path

    from backend_core.AgentMPC.agents.evaluator import run_closed_loop
    from backend_core.AgentMPC.dynamics.loader import DynamicLoader
    from backend_core.AgentMPC.mpc.config import Config

    plugin = Path(__file__).resolve().parents[1] / "dynamics" / "plugins" / "example_quadrotor_vertical.py"
    dyn = DynamicLoader.load_from_path(str(plugin)).create_dynamics()
    # The plugin's termination check, not a declared state bound: the MPC
    # would enforce a declared bound as a constraint and never reach it.
    dyn.check_termination = lambda x: bool(x[0] > 0.6)
    cfg = Config()
    cfg.mpc.prediction_horizon, cfg.mpc.control_horizon = 10, 2
    cfg.data.dt_mpc, cfg.data.simulation_time = 0.05, 6.0
    res = run_closed_loop(dyn, cfg, {"Np": 10, "Nc": 2, "Q": [10.0, 1.0, 1.0, 1.0], "R": [0.1] * dyn.n_inputs})

    assert res["unstable"] and "state bounds" in res["unstable_reason"]
    assert res["metrics"].is_stable is False
    assert res["metrics"].settled is False


def test_stop_reasons_get_a_short_table_label():
    from backend_core.AgentMPC.agents.formatting import short_stop_reason

    assert short_stop_reason("controller failed at step 12: QP data is not finite") == "solver failed"
    assert short_stop_reason("state became non-finite (NaN/Inf)") == "NaN state"
    assert short_stop_reason("state magnitude exceeded 1e+06 (diverged)") == "diverged"
    assert short_stop_reason("hit the plugin's declared state bounds") == "hit state bound"
    assert short_stop_reason(None) == "stopped early"
