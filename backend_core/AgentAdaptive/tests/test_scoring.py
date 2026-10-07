"""
Run from the repo root with:
    python -m pytest backend_core/AgentAdaptive/tests/test_scoring.py -v
"""

import numpy as np
import pytest

from backend_core.AgentAdaptive.tools.scoring import compute_success_verdict, compute_tracking_score

T = np.linspace(0, 10, 1001)


def test_a_stay_put_output_with_a_tiny_error_is_not_zero_percent():
    # hover-like output: starts on a constant reference and ends 2e-5 away from it
    ref = np.ones_like(T)
    y = ref + 2e-5 * np.sin(T)
    score = compute_tracking_score(T, y, ref)
    assert score["task_trivial"] == [True]
    assert score["tracking_pct"]["steady"][0] == pytest.approx(100.0, abs=0.01)
    assert score["tracking_pct_headline"] > 99.9


def test_a_perfect_stay_put_is_exactly_one_hundred():
    ref = np.zeros_like(T)
    assert compute_tracking_score(T, ref, ref)["tracking_pct_headline"] == 100.0


def test_the_headline_no_longer_disagrees_with_the_success_verdict():
    # the quadrotor case: three channels track well, the fourth sits still with a 2e-5 error
    ref = np.stack([0.5 * np.sin(T), 0.5 * np.cos(T), np.ones_like(T)], axis=1)
    y = ref + np.array([3e-3, 3e-3, 2e-5])[None, :] * np.sin(5 * T)[:, None]
    y[0] = ref[0]
    u = np.zeros((len(T), 1))
    verdict = compute_success_verdict(T, y, ref, u, y, 0.01)
    score = compute_tracking_score(T, y, ref)
    assert verdict["success"] is True
    assert score["tracking_pct_headline"] > 98.0


def test_an_output_that_really_drifts_from_a_stay_put_task_still_scores_low():
    ref = np.zeros_like(T)
    y = 0.5 * np.minimum(T / 5.0, 1.0)
    score = compute_tracking_score(T, y, ref)
    assert score["task_trivial"] == [True] and score["tracking_pct"]["steady"][0] < 60.0


def test_a_moving_task_is_scored_exactly_as_before():
    ref = np.sin(T)
    y = ref + 0.05
    score = compute_tracking_score(T, y, ref)
    assert score["task_trivial"] == [False]
    scale = score["task_scale"][0]
    want = 100.0 * (1.0 - np.sqrt(np.mean((y[-200:] - ref[-200:]) ** 2)) / scale)
    assert score["tracking_pct"]["steady"][0] == pytest.approx(want)


def _verdict_with_u(u):
    ref = np.sin(T)
    return compute_success_verdict(T, ref, ref, u, np.zeros((len(T), 1)), dt=0.01)


def test_a_lone_control_spike_fails_the_bounded_check():
    # the cart-pole case: the output tracks fine but u jumps to ~1e6 at isolated instants
    u = 10.0 * np.sin(T)
    u[400] = -1.4e6
    verdict = _verdict_with_u(u)
    assert verdict["checks"]["bounded"] is False
    assert verdict["success"] is False
    assert verdict["reason"].startswith("bounded")


def test_a_normal_control_signal_stays_bounded():
    assert _verdict_with_u(10.0 * np.sin(T))["checks"]["bounded"] is True


def test_a_small_blip_on_a_near_zero_control_is_not_a_spike():
    # the floor of 1 on the scale: u ~ 0 with a 50-unit blip is not a 5000x event
    u = np.zeros_like(T)
    u[300] = 50.0
    assert _verdict_with_u(u)["checks"]["bounded"] is True


def test_a_big_kick_at_the_start_is_not_a_spike():
    # step-like reference: u legitimately jumps at t~0 and then settles
    u = 10.0 * np.sin(T)
    u[5] = 5e3
    assert _verdict_with_u(u)["checks"]["bounded"] is True
