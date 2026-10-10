"""
Run from the repo root with:
    python -m pytest backend_core/AgentAdaptive/tests/test_input_limits.py -v
"""

import contextlib
import io

import numpy as np
import pytest

from backend_core.AgentAdaptive.agents import tuner_agent as ta
from backend_core.AgentAdaptive.controller import simulation
from backend_core.AgentAdaptive.controller.structure_build import _build_structure_from_spec
from backend_core.AgentAdaptive.tools import diagnosis_chain as dc
from backend_core.AgentAdaptive.tools import diagnostics_evidence as de
from backend_core.AgentAdaptive.tools import scoring, system_spec


def _spec(limits=None, inputs=("u",)):
    dyn = {"states": ["x1", "x2"], "inputs": list(inputs), "outputs": ["x1"],
           "state_equations": ["x2", "-x1 - 0.5*x2 + u"], "x0": [0.0, 0.0],
           "references": [{"output": "x1", "expr": "sin(t)"}], "sim_time": 5.0,
           "solver_step": 0.01}
    if limits is not None:
        dyn["input_limits"] = limits
    return {"system_name": "P", "dynamics": dyn}


def test_spec_keeps_limits_and_drops_empty_ones():
    spec = system_spec.normalize_spec(_spec([
        {"input": "u", "min": "-3", "max": 3},
        {"input": "v", "min": None, "max": None}, "junk"]))
    assert spec["dynamics"]["input_limits"] == [{"input": "u", "min": -3.0, "max": 3.0}]


def test_one_sided_limit_is_kept():
    dyn = system_spec.normalize_spec(_spec([{"input": "u", "max": 2}]))["dynamics"]
    assert system_spec.limits_per_input(dyn) == [[None, 2.0]]


def test_limit_problems_are_named():
    assert system_spec.input_limit_problems(_spec([{"input": "u", "min": 1, "max": 1}]))
    assert system_spec.input_limit_problems(_spec([{"input": "w", "min": -1, "max": 1}]))
    assert not system_spec.input_limit_problems(_spec([{"input": "u", "min": -1, "max": 1}]))


def test_merge_sim_knobs_sets_limits_and_old_specs_have_none():
    spec = system_spec.merge_sim_knobs(_spec(), input_limits=[{"input": "u", "min": -1, "max": 1}])
    assert spec["dynamics"]["input_limits"] == [{"input": "u", "min": -1.0, "max": 1.0}]
    assert system_spec.normalize_spec(_spec())["dynamics"]["input_limits"] == []


def test_structure_lines_limits_up_with_inputs_and_is_none_without():
    struct, notes = _build_structure_from_spec(system_spec.normalize_spec(
        _spec([{"input": "u", "min": -2, "max": 2}])))
    assert struct["input_limits"] == [[-2.0, 2.0]] and not notes
    assert _build_structure_from_spec(system_spec.normalize_spec(_spec()))[0]["input_limits"] is None


def test_a_limit_on_a_missing_input_is_reported_not_applied():
    struct, notes = _build_structure_from_spec(system_spec.normalize_spec(
        _spec([{"input": "w", "min": -2, "max": 2}])))
    assert struct["input_limits"] is None and "w" in notes[0]


def test_limit_arrays():
    assert simulation.limit_arrays(None, 1) is None
    assert simulation.limit_arrays([None], 1) is None
    lo, hi = simulation.limit_arrays([[None, 2.0], [-1.0, 1.0]], 2)
    assert lo[0] == -np.inf and hi[0] == 2.0 and (lo[1], hi[1]) == (-1.0, 1.0)


def _args(method, limits, t_end=8.0, dt=0.002):
    return dict(
        states=["x1", "x2"], dynamics=["x2", "-x1 - 0.5*x2 + u"], inputs=["u"], outputs=["x1"],
        x0=[0.0, 0.0], refs=[{"expr": "1.5*sin(t)"}], has_delta=True, has_disturbance=True,
        delta_exprs=["0", "0.8*sin(x1)"], dist_exprs=["0", "0.5*sin(2*t)"],
        method=method, dt=dt, t_end=t_end, reasoning="", fail_tol=0.05, input_limits=limits)


def _run(method, limits):
    with contextlib.redirect_stdout(io.StringIO()):
        return ta._rerun_design_directly(_args(method, limits), for_tuning=True)


@pytest.mark.parametrize("method", ["smc", "backstepping"])
def test_the_plant_never_gets_more_than_the_limit_and_it_still_tracks(method):
    # the estimators must see the applied u: when they saw the raw command, this exact run
    # tracked 0% (saturated 99.9% of the time)
    _, free = _run(method, None)
    _, m = _run(method, [[-3.0, 3.0]])
    assert free["control_max"][0] > 3.0 and "input_saturation" not in free
    assert m["control_max"][0] <= 3.0 + 1e-9
    assert m["success"] and m["tracking_pct_headline"] > 90.0
    sat = m["input_saturation"]
    assert 0.0 < sat["any_input_fraction"] < 0.2 and sat["first_saturated_at_s"] is not None
    assert sat["peak_command"][0] > 3.0


@pytest.mark.parametrize("method", ["smc", "backstepping"])
def test_an_unreachable_reference_is_reported_as_failed_and_clipped(method):
    _, m = _run(method, [[-1.0, 1.0]])
    assert not m["success"] and m["input_saturation"]["any_input_fraction"] > 0.5


def test_a_limit_nobody_hits_changes_nothing():
    _, free = _run("smc", None)
    _, loose = _run("smc", [[-1e6, 1e6]])
    assert loose["steady_rms_frac"] == pytest.approx(free["steady_rms_frac"])
    assert loose["input_saturation"]["any_input_fraction"] == 0.0


def test_gamma_stays_zero_without_clipping_and_moves_with_it():
    from backend_core.AgentAdaptive.controller import backstepping_design
    import sympy as sp
    x1, x2, u = sp.symbols("x1 x2 u", real=True)
    system = {"states": [x1, x2], "inputs": [u], "outputs": [x1], "f": [x2, -x1 - 0.5 * x2 + u]}
    _, _, adaptive, _, _ = backstepping_design.design_backstepping(system, True)
    adaptive.reset([0.0, 0.0])
    cmd = adaptive.compute([0.5, 0.0], [[0.0, 0.0, 0.0]])
    adaptive.compute_derivs(np.zeros(2))
    adaptive.step(0.01)
    assert adaptive.gamma == 0.0
    cmd = adaptive.compute([0.5, 0.0], [[0.0, 0.0, 0.0]])
    adaptive.apply_input(cmd, [cmd[0] / 10.0])
    adaptive.compute_derivs(np.zeros(2))
    adaptive.step(0.01)
    assert adaptive.gamma != 0.0


def test_saturation_metric_counts_clipped_time():
    t = np.linspace(0, 1, 11)
    cmd = np.array([0, 0, 5, 5, 0, 0, 0, 0, 0, 0, 0], dtype=float).reshape(-1, 1)
    sat = scoring.compute_input_saturation(t, cmd, (np.array([-2.0]), np.array([2.0])))
    assert sat["any_input_fraction"] == pytest.approx(2 / 11)
    assert sat["first_saturated_at_s"] == pytest.approx(0.2) and sat["limits"] == [[-2.0, 2.0]]


def test_the_report_mentions_the_clipping_only_when_there_was_a_limit():
    base = {"numerically_healthy": True, "transient_rms": [0.1], "steady_rms": [0.1],
            "steady_rms_frac": [0.01], "overshoot_frac": [0.1], "settling_time_reached": False,
            "settling_time": None, "control_rms": [1.0], "control_max": [3.0],
            "control_rate_max": [1.0], "chattering_tv_per_sec": [1.0]}
    assert "input limits" not in scoring.format_metrics_report(base, "SMC", False, False, False)
    base["input_saturation"] = {"any_input_fraction": 0.25, "fraction_of_time": [0.25],
                                "peak_command": [9.0]}
    assert "clipped 25.0%" in scoring.format_metrics_report(base, "SMC", False, False, False)


def test_evidence_carries_limits_and_the_requested_command():
    t = np.linspace(0, 1, 5)
    col = np.zeros((5, 1))
    ctx = {"method": "smc", "states": ["x"], "inputs": ["u"], "outputs": ["x"], "t": t, "y": col,
           "ref": col, "u": col, "x": col, "x0": [0.0], "system_text": "", "control_law_text": "",
           "refs": [], "dt": 0.25, "t_end": 1.0}
    assert "input_limits" not in de.build_evidence(ctx, {})[0]
    ctx.update(input_limits=[[-2.0, 2.0]], u_cmd=np.arange(5.0).reshape(-1, 1),
               input_saturation={"any_input_fraction": 0.4})
    ev, charts = de.build_evidence(ctx, {})
    assert ev["input_limits"]["limits"] == {"u": [-2.0, 2.0]}
    assert ev["input_limits"]["saturation"]["any_input_fraction"] == 0.4
    assert charts["input_command:u"]["lines"]["u (requested)"][-1] == 4.0
    assert "input_command:u" in ev["available_charts"]


def _chain_spec(limits):
    return {"system_name": "P", "dynamics": {
        "states": ["x"], "inputs": ["u"], "outputs": ["x"], "state_equations": ["-x + u"],
        "x0": [0.1], "references": [{"output": "x", "expr": "sin(t)"}], "solver_step": 0.001,
        "sim_time": 10.0, "parameters": {}, "input_limits": limits}}


def test_changing_the_limits_between_runs_is_listed_as_a_change():
    first = dc.make_attempt([], _chain_spec([]), {"success": False})
    now = _chain_spec([{"input": "u", "min": -2.0, "max": 2.0}])
    changes = dc.changes_for_evidence([first], now)
    assert [c["what"] for c in changes] == ["input limit for u"]
    assert dc.changes_for_evidence([first], _chain_spec([])) == []


def test_an_attempt_stored_before_limits_existed_does_not_break_the_diff():
    first = dc.make_attempt([], _chain_spec([]), {"success": False})
    del first["snapshot"]["input_limits"]
    assert dc.changes_for_evidence([first], _chain_spec([])) == []


def test_pipeline_end_to_end_a_reference_the_actuator_cannot_follow(monkeypatch):
    # fake extraction + fake model. cart-pole with 1*sin(5t) needs a peak force near 67, so a
    # +-30 limit can't follow it: the run fails, is clipped, and the diagnoser is told so
    import json
    import os
    os.environ["LABCD_ADAPTIVE_SHOW_PLOTS"] = "0"
    from backend_core.AgentAdaptive.agents import llm_factory
    from backend_core.AgentAdaptive.agents.agent_io import _SyntheticMessage, _empty_usage
    from backend_core.AgentAdaptive.controller.structure_build import _verify_structure

    def fake_extraction(spec, on_event=None, clarification_record=None):
        st = _verify_structure(_build_structure_from_spec(spec)[0])
        args = dict(st, method="backstepping", reasoning="t", notes_limitations=None)
        use = {"agent": _empty_usage(), "total": _empty_usage(), "agent_turns": [], "timeline": []}
        return {"messages": [_SyntheticMessage("d")]}, use, {"ok": True, "report": "", "args": args,
                                                              "why": "t"}

    seen = []

    class Reply:
        content = json.dumps({"headline": "h", "explanation": ["e"], "suggestions": []})
        usage_metadata = {}

    class LLM:
        def invoke(self, msgs):
            seen.append(json.loads(msgs[1]["content"]))
            return Reply()

    monkeypatch.setattr(ta, "run_extraction", fake_extraction)
    monkeypatch.setattr(ta, "write_abstract", lambda *a, **k: ("abs", ta._empty_usage()))
    monkeypatch.setattr(llm_factory, "build_llm", lambda role, **k: LLM())
    spec = system_spec.normalize_defaults({"status": "complete", "system_name": "CP", "dynamics": {
        "states": ["x1", "x2"], "inputs": ["u"], "outputs": ["x1"],
        "state_equations": ["x2", "(-(u + 0.06*x2**2*sin(x1))*cos(x1) + 11.2815*sin(x1))/"
                                  "(0.06*sin(x1)**2 + 0.4)"],
        "x0": [0.2, 0.0], "references": [{"output": "x1", "expr": "1*sin(5*t)"}],
        "solver_step": 0.001, "sim_time": 10.0,
        "input_limits": [{"input": "u", "min": -30, "max": 30}]}})
    result, *_ = ta.run_full_pipeline("", system_spec=spec)
    metrics = result["final_metrics"]
    assert not metrics["success"] and metrics["control_max"][0] <= 30.0 + 1e-9
    assert metrics["input_saturation"]["any_input_fraction"] > 0.4
    assert seen and seen[0]["input_limits"]["limits"] == {"u": [-30.0, 30.0]}
    assert "input_command:u" in seen[0]["available_charts"]
    assert result["attempt"]["snapshot"]["input_limits"] == {"u": [-30.0, 30.0]}


def test_a_suggestion_to_change_the_limits_is_dropped_but_a_reference_one_stays():
    from backend_core.AgentAdaptive.agents import diagnoser_agent as da
    report = {"suggestions": [
        {"title": "Revisit Input Limits", "lever": "system", "options": []},
        {"title": "Revisit input limit", "lever": None, "options": []},
        {"title": "Lower the amplitude to fit the input limits", "lever": "reference", "options": []}]}
    kept = da._prune_for_evidence(dict(report), {"input_limits": {"limits": {"u": [-50, 50]}}})
    assert [s["lever"] for s in kept["suggestions"]] == ["reference"]
    # the model never proposes limits, whether or not this run had one
    assert [s["lever"] for s in da._prune_for_evidence(dict(report), {})["suggestions"]] == ["reference"]


def test_the_report_lists_the_limits_only_when_there_are_some():
    from backend_core.AgentAdaptive.tools import reporter
    comps = {"system": "s", "control_law": "c", "stability": "p"}
    args = {"inputs": ["u1", "u2"], "input_limits": [[-2.0, 2.0], None]}
    text = reporter.render_final_report("SMC", args, comps, "why", None)
    assert "| input_limits | u1 in [-2, 2], u2 in free |" in text
    assert "input_limits" not in reporter.render_final_report("SMC", {"inputs": ["u"]}, comps, "w", None)
