"""
Run from the repo root with:
    python -m pytest backend_core/AgentAdaptive/tests/test_failure_mode.py -v
"""

import json
import os

import pytest

from backend_core.AgentAdaptive.agents import diagnoser_agent as da
from backend_core.AgentAdaptive.agents import tuner_agent as ta
from backend_core.AgentAdaptive.controller.structure_build import (
    _build_structure_from_spec, _verify_structure)
from backend_core.AgentAdaptive.tools import diagnosis_chain as dc
from backend_core.AgentAdaptive.tools import failure_mode as fm
from backend_core.AgentAdaptive.tools import system_spec


def _ev(frac=0.5, free=None, spikes=None, warnings=None, limits=None, first_sat=0.8):
    ev = {"input_limits": {"limits": limits or {"u": [-10.0, 10.0]},
                           "saturation": {"any_input_fraction": frac, "first_saturated_at_s": first_sat},
                           "without_limits": free}}
    if spikes:
        ev["control_spikes"] = spikes
    if warnings:
        ev["pre_run_warnings"] = warnings
    return ev


_FREE_PASS = {"passes": True, "tracking_pct": 99.6, "failed_check": None, "peak_by_input": {"u": 13.9}}


def _chain_spec(limits):
    return {"system_name": "P", "dynamics": {
        "states": ["x"], "inputs": ["u"], "outputs": ["x"], "state_equations": ["-x + u"],
        "x0": [0.1], "references": [{"output": "x", "expr": "sin(t)"}], "solver_step": 0.001,
        "sim_time": 10.0, "parameters": {}, "input_limits": limits}}


def test_limit_too_small_when_the_same_design_passes_without_it():
    spikes = {"first_t": 3.9, "largest": [{"states_then": {"x1": -1.571}}]}
    out = fm.classify(_ev(free=_FREE_PASS, spikes=spikes))
    assert out["mode"] == "limit_too_small"
    text = " ".join(out["chain"])
    assert "WITHOUT the limits passes" in text and "13.9" in text
    assert "consequence of the clipping" in text and "x1=-1.571" in text


def test_a_spike_before_any_clipping_is_not_called_a_consequence():
    spikes = {"first_t": 0.3, "largest": [{"states_then": {"x1": 1.5}}]}
    out = fm.classify(_ev(free=_FREE_PASS, spikes=spikes))
    assert "consequence" not in " ".join(out["chain"])


def test_other_modes():
    fail = {"passes": False, "tracking_pct": 6.0, "failed_check": "bounded", "peak_by_input": {"u": 9e5}}
    assert fm.classify(_ev(free=fail, warnings=["crosses 1.57"]))["mode"] == "fails_even_without_limit"
    assert fm.classify(_ev(frac=0.0))["mode"] == "not_limit_related"
    assert fm.classify(_ev(free=None))["mode"] == "clipped"
    assert fm.classify({"pre_run_warnings": ["w"]})["mode"] == "singularity_or_spikes"
    assert fm.classify({})["mode"] == "tracking_error_only"


def test_the_limit_card_has_a_computed_number_and_only_when_it_can_help():
    ev = _ev(free=_FREE_PASS)
    ev["failure_mode"] = fm.classify(ev)
    card = fm.limit_card(ev)
    assert card["lever"] == "input_limits" and card["target"] == "u"
    assert card["options"][0]["value"] == "-18,18"            # 1.25 x 13.9, up to 2 figures
    far = _ev(free=dict(_FREE_PASS, peak_by_input={"u": 900.0}))
    far["failure_mode"] = fm.classify(far)
    # the design passes without a limit, so even a big need is a measured fact, not a guess
    assert fm.limit_card(far)["options"][0]["value"] == "-1200,1200"
    other = _ev(frac=0.0)
    other["failure_mode"] = fm.classify(other)
    assert fm.limit_card(other) is None


def test_the_card_keeps_a_free_side_free():
    ev = _ev(free=_FREE_PASS, limits={"u": [None, 10.0]})
    ev["failure_mode"] = fm.classify(ev)
    assert fm.limit_card(ev)["options"][0]["value"] == ",18"


def test_the_python_card_replaces_whatever_the_model_said_about_limits():
    ev = _ev(free=_FREE_PASS)
    ev["failure_mode"] = fm.classify(ev)
    report = {"explanation": ["e"], "suggestions": [
        {"title": "Restore the broader actuator range", "lever": "system", "options": []},
        {"title": "Lower the amplitude", "lever": "reference", "options": []}]}
    out = da._prune_for_evidence(report, ev)
    assert [s["lever"] for s in out["suggestions"]] == ["input_limits", "reference"]


def test_the_change_bullet_is_computed_from_the_real_outcomes():
    prev = [{"run": 4, "outcome": {"success": False, "tracking_pct": 0.0},
             "changes_since_previous_run": []}]
    changes = [{"lever": "input_limits", "what": "input limit for u", "target": "u",
                "from": [-30.0, 30.0], "to": [-20.0, 20.0], "followed": None}]
    both_fail = dc.describe_changes(prev, changes, {"success": False, "tracking_pct_headline": 0.0})
    assert "failed, tracking 0.0%" in both_fail and "[-30, 30] to [-20, 20]" in both_fail
    assert "about the same" in both_fail
    assert "worse" in dc.describe_changes(
        [{"run": 1, "outcome": {"success": True, "tracking_pct": 99.0}}], [],
        {"success": False, "tracking_pct_headline": 40.0})
    assert "better" in dc.describe_changes(prev, [], {"success": True, "tracking_pct_headline": 99.0})
    removed = dc.describe_changes(prev, [{"lever": "input_limits", "what": "input limit for u",
                                          "target": "u", "from": [None, 30.0], "to": None,
                                          "followed": None}],
                                  {"success": False, "tracking_pct_headline": 0.0})
    assert "was removed (it was [-inf, 30])" in removed
    assert dc.describe_changes([], [], {}) is None


def test_following_the_limit_card_is_recognised():
    first = dc.make_attempt([], _chain_spec([{"input": "u", "min": -10.0, "max": 10.0}]),
                            {"success": False},
                            report={"suggestions": [{"title": "Raise the u limit to about 18",
                                                     "lever": "input_limits", "target": "u",
                                                     "options": [{"value": "-18,18"}]}]})
    now = _chain_spec([{"input": "u", "min": -18.0, "max": 18.0}])
    change = dc.changes_for_evidence([first], now)[0]
    assert change["followed"] == {"title": "Raise the u limit to about 18", "exact_option": True}


def test_pipeline_end_to_end_the_limit_is_the_cause_and_the_card_is_offered(monkeypatch):
    # cart-pole 0.5*sin(5t) needs a peak of ~13.9; a +-10 limit makes it fail and the pole fall
    os.environ["LABCD_ADAPTIVE_SHOW_PLOTS"] = "0"
    from backend_core.AgentAdaptive.agents import llm_factory
    from backend_core.AgentAdaptive.agents.agent_io import _SyntheticMessage, _empty_usage

    def fake_extraction(spec, on_event=None, clarification_record=None):
        st = _verify_structure(_build_structure_from_spec(spec)[0])
        args = dict(st, method="backstepping", reasoning="t", notes_limitations=None)
        use = {"agent": _empty_usage(), "total": _empty_usage(), "agent_turns": [], "timeline": []}
        return {"messages": [_SyntheticMessage("d")]}, use, {"ok": True, "report": "", "args": args,
                                                              "why": "t"}

    class Reply:
        content = json.dumps({"headline": "h", "explanation": ["e"], "suggestions": [
            {"title": "Revisit the actuator range", "lever": "system"},
            {"title": "Start closer to upright", "lever": "initial_condition",
             "options": [{"value": "0,0"}]}]})
        usage_metadata = {}

    class LLM:
        def invoke(self, msgs):
            return Reply()

    monkeypatch.setattr(ta, "run_extraction", fake_extraction)
    monkeypatch.setattr(ta, "write_abstract", lambda *a, **k: ("abs", ta._empty_usage()))
    monkeypatch.setattr(llm_factory, "build_llm", lambda role, **k: LLM())
    spec = system_spec.normalize_defaults({"status": "complete", "system_name": "CP", "dynamics": {
        "states": ["x1", "x2"], "inputs": ["u"], "outputs": ["x1"],
        "state_equations": ["x2", "(-(u + 0.06*x2**2*sin(x1))*cos(x1) + 11.2815*sin(x1))/"
                                  "(0.06*sin(x1)**2 + 0.4)"],
        "x0": [0.2, 0.0], "references": [{"output": "x1", "expr": "0.5*sin(5*t)"}],
        "solver_step": 0.001, "sim_time": 10.0,
        "input_limits": [{"input": "u", "min": -10, "max": 10}]}})
    result, *_ = ta.run_full_pipeline("", system_spec=spec)
    diag = result["diagnosis"]
    ev, report = diag["evidence"], diag["report"]
    assert ev["failure_mode"]["mode"] == "limit_too_small"
    assert ev["input_limits"]["without_limits"]["passes"] is True
    assert ev["input_limits"]["without_limits"]["peak_by_input"]["u"] == pytest.approx(13.9, abs=0.2)
    assert ev["control_spikes"] is not None                    # measured on the requested command
    assert [s["lever"] for s in report["suggestions"]] == ["input_limits"]   # IC card dropped too
    assert report["suggestions"][0]["options"][0]["value"] == "-18,18"
