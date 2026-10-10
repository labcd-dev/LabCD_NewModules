"""
Run from the repo root with:
    python -m pytest backend_core/AgentAdaptive/tests/test_diagnosis_chain.py -v
"""

import pytest

from backend_core.AgentAdaptive.tools import diagnosis_chain as dc


def _spec(eq="-x + u", ref="sin(t)", x0=(0.1,), dt=0.001, params=None, name="Plant"):
    return {"system_name": name, "dynamics": {
        "states": ["x"], "inputs": ["u"], "outputs": ["x"], "state_equations": [eq],
        "x0": list(x0), "references": [{"output": "x", "expr": ref}],
        "solver_step": dt, "sim_time": 10.0, "parameters": params or {"k": 1.0}}}


def _metrics(ok=False, reason="mse_target: too big"):
    return {"success": ok, "success_reason": reason, "tracking_pct_headline": 40.0,
            "steady_rms_frac": [0.3]}


def test_same_structure_shares_a_chain_whatever_the_name_or_numbers():
    reg = {}
    fp1, c1 = dc.chain_for(reg, _spec(name="A"))
    fp2, c2 = dc.chain_for(reg, _spec(name="B", ref="2*sin(t)", x0=(0.5,), dt=0.01, params={"k": 3.0}))
    assert fp1 == fp2 and c1 is c2


def test_same_name_different_equations_are_separate_chains():
    reg = {}
    _, c1 = dc.chain_for(reg, _spec(eq="-x + u"))
    _, c2 = dc.chain_for(reg, _spec(eq="-2*x + u"))
    assert c1["id"] != c2["id"]


def test_whitespace_in_equations_does_not_split_a_chain():
    assert dc.fingerprint(_spec(eq="-x + u")) == dc.fingerprint(_spec(eq="-x+u"))


def test_first_attempt_has_no_changes():
    a = dc.make_attempt([], _spec(), _metrics())
    assert a["n"] == 1 and a["changes"] is None


def test_changes_are_found_from_the_inputs_not_from_clicks():
    first = dc.make_attempt([], _spec(), _metrics())
    second = dc.make_attempt([first], _spec(ref="0.5*sin(t)", dt=0.01), _metrics())
    levers = {c["lever"] for c in second["changes"]}
    assert levers == {"reference", "step_time"}
    assert second["n"] == 2


def test_a_followed_suggestion_is_tagged_with_the_exact_option():
    report = {"headline": "h", "note": "n", "suggestions": [
        {"title": "Lower amplitude", "lever": "reference", "target": "x",
         "options": [{"value": "0.5*sin(t)"}]}]}
    first = dc.make_attempt([], _spec(), _metrics(), report=report)
    second = dc.make_attempt([first], _spec(ref="0.5 * sin(t)"), _metrics())
    assert second["changes"][0]["followed"] == {"title": "Lower amplitude", "exact_option": True}


def test_own_value_on_a_suggested_lever_is_followed_but_not_exact():
    report = {"headline": "h", "suggestions": [
        {"title": "Lower amplitude", "lever": "reference", "target": "x",
         "options": [{"value": "0.5*sin(t)"}]}]}
    first = dc.make_attempt([], _spec(), _metrics(), report=report)
    second = dc.make_attempt([first], _spec(ref="0.3*sin(t)"), _metrics())
    assert second["changes"][0]["followed"]["exact_option"] is False


def test_a_change_nobody_suggested_is_not_tagged():
    first = dc.make_attempt([], _spec(), _metrics(), report={"headline": "h", "suggestions": []})
    second = dc.make_attempt([first], _spec(x0=(0.9,)), _metrics())
    assert second["changes"][0]["followed"] is None


def test_the_note_is_capped_and_an_error_report_is_ignored():
    long = {"headline": "h", "note": "x" * 500, "suggestions": []}
    assert len(dc.make_attempt([], _spec(), _metrics(), report=long)["note"]) == dc.NOTE_CHARS
    assert dc.make_attempt([], _spec(), _metrics(), report={"error": "boom", "note": "hi"})["note"] is None


def test_an_edited_plant_sees_the_last_run_of_the_one_it_came_from():
    reg = {}
    fp_old, old = dc.chain_for(reg, _spec(eq="-x + u"))
    dc.record(old, dc.make_attempt([], _spec(eq="-x + u"), _metrics()))
    new_spec = _spec(eq="-3*x + u")
    fp_new, new = dc.chain_for(reg, new_spec, parent_fp=fp_old)
    assert fp_new != fp_old
    parent = dc.parent_attempts(reg, new)
    rows = dc.history_for_evidence(new["attempts"], parent)
    assert rows[0]["earlier_plant_version"] is True
    assert [c["what"] for c in dc.changes_for_evidence(new["attempts"], new_spec, parent)] == ["state equations"]


def test_only_the_last_few_runs_go_to_the_model():
    attempts = []
    for _ in range(7):
        attempts.append(dc.make_attempt(attempts, _spec(), _metrics()))
    assert [r["run"] for r in dc.history_for_evidence(attempts)] == [4, 5, 6, 7]


def test_registry_is_capped():
    reg = {}
    for i in range(dc.MAX_CHAINS + 3):
        dc.chain_for(reg, _spec(eq="-%d*x + u" % (i + 1)))
    assert len(reg) == dc.MAX_CHAINS


def test_evidence_carries_previous_runs_only_when_there_are_some():
    import numpy as np
    from backend_core.AgentAdaptive.tools import diagnostics_evidence as de
    t = np.linspace(0, 1, 5)
    col = np.zeros((5, 1))
    ctx = {"method": "smc", "states": ["x"], "inputs": ["u"], "outputs": ["x"], "t": t, "y": col,
           "ref": col, "u": col, "x": col, "x0": [0.0], "system_text": "", "control_law_text": "",
           "refs": [], "dt": 0.25, "t_end": 1.0}
    bare, _ = de.build_evidence(ctx, {})
    assert "previous_runs" not in bare
    first = dc.make_attempt([], _spec(), _metrics())
    rows = dc.history_for_evidence([first])
    withh, _ = de.build_evidence(ctx, {}, previous_runs=rows, changes_since_last_run=[])
    assert withh["previous_runs"][0]["run"] == 1 and withh["changes_since_last_run"] == []


def test_the_diagnoser_keeps_a_capped_note():
    from backend_core.AgentAdaptive.agents import diagnoser_agent as da
    rep = da._coerce_report({"headline": "h", "explanation": ["e"], "note": "y" * 400})
    assert len(rep["note"]) == dc.NOTE_CHARS
    assert da._coerce_report({"headline": "h", "explanation": ["e"]})["note"] is None


def test_evidence_reports_control_spikes_with_the_state_at_that_instant():
    import numpy as np
    from backend_core.AgentAdaptive.tools import diagnostics_evidence as de
    t = np.linspace(0, 10, 1001)
    u = (10 * np.sin(t)).reshape(-1, 1)
    u[600] = 5e5
    x = np.column_stack([np.sin(t), np.cos(t)])
    ctx = {"method": "smc", "states": ["a", "b"], "inputs": ["u"], "outputs": ["a"], "t": t,
           "y": x[:, :1], "ref": x[:, :1], "u": u, "x": x, "x0": [0.0, 1.0], "system_text": "",
           "control_law_text": "", "refs": [], "dt": 0.01, "t_end": 10.0}
    ev, _ = de.build_evidence(ctx, {})
    spikes = ev["control_spikes"]
    assert spikes["spike_events"] == 1 and spikes["largest"][0]["t"] == 6.0
    assert set(spikes["state_range_over_all_spikes"]) == {"a", "b"}
    assert spikes["largest"][0]["states_then"]["a"] == pytest.approx(np.sin(6.0), abs=1e-3)
    ctx["u"] = (10 * np.sin(t)).reshape(-1, 1)
    assert de.build_evidence(ctx, {})[0]["control_spikes"] is None


def test_evidence_gives_the_real_span_of_the_reference():
    import numpy as np
    from backend_core.AgentAdaptive.tools import diagnostics_evidence as de
    t = np.linspace(0, 10, 11)
    ref = (10 * np.sin(t)).reshape(-1, 1)
    z = np.zeros((11, 1))
    ctx = {"method": "smc", "states": ["a"], "inputs": ["u"], "outputs": ["a"], "t": t, "y": z, "ref": ref,
           "u": z, "x": z, "x0": [0.0], "system_text": "", "control_law_text": "", "refs": [],
           "dt": 1.0, "t_end": 10.0}
    span = de.build_evidence(ctx, {})[0]["reference_span"]["a"]
    assert span["max"] == pytest.approx(10 * np.sin(t).max(), rel=1e-3) and span["min"] < -9


def _spike_evidence(x0):
    return {"plant": {"states": ["x1", "x2"], "x0": x0},
            "control_spikes": {"largest": [{"t": 2.3, "states_then": {"x1": -7.85, "x2": 30.0}}]}}


def _report():
    return {"headline": "h", "explanation": ["e"], "chart_series": [], "note": None, "suggestions": [
        {"title": "ref", "lever": "reference", "target": "x1", "detail": "", "options": []},
        {"title": "x0", "lever": "initial_condition", "target": None, "detail": "", "options": []}]}


def test_an_initial_condition_fix_is_dropped_when_x0_is_not_the_bad_point():
    from backend_core.AgentAdaptive.agents import diagnoser_agent as da
    out = da._prune_for_evidence(_report(), _spike_evidence([0.2, 0.0]))
    assert [s["lever"] for s in out["suggestions"]] == ["reference"]


def test_an_initial_condition_fix_stays_when_x0_is_at_the_bad_point():
    from backend_core.AgentAdaptive.agents import diagnoser_agent as da
    out = da._prune_for_evidence(_report(), _spike_evidence([-7.85, 30.0]))
    assert len(out["suggestions"]) == 2


def test_an_initial_condition_fix_needs_early_trouble_or_a_bad_x0():
    from backend_core.AgentAdaptive.agents import diagnoser_agent as da
    base = {"plant": {"states": ["x1"], "x0": [0.0]}, "simulation": {"t_end": 10.0}}
    late = da._prune_for_evidence(_report(), dict(base, first_trouble_s=3.0))
    none = da._prune_for_evidence(_report(), dict(base))
    early = da._prune_for_evidence(_report(), dict(base, first_trouble_s=0.2))
    assert [len(r["suggestions"]) for r in (late, none, early)] == [1, 1, 2]


def test_pipeline_warns_before_the_run_and_hands_the_warning_to_the_diagnoser(monkeypatch):
    # fake extraction + fake model: the clearance warning must reach the event trace, the
    # result and the diagnoser's evidence for a reference that crosses cos(x1)=0
    import json
    import os
    os.environ["LABCD_ADAPTIVE_SHOW_PLOTS"] = "0"
    from backend_core.AgentAdaptive.agents import tuner_agent, llm_factory
    from backend_core.AgentAdaptive.agents.agent_io import _SyntheticMessage, _empty_usage
    from backend_core.AgentAdaptive.controller.structure_build import _build_structure_from_spec, _verify_structure
    from backend_core.AgentAdaptive.tools import system_spec

    def fake_extraction(spec, on_event=None, clarification_record=None):
        st = _verify_structure(_build_structure_from_spec(spec)[0])
        args = dict(st, method="backstepping", reasoning="t", notes_limitations=None)
        use = {"agent": _empty_usage(), "total": _empty_usage(), "agent_turns": [], "timeline": []}
        return {"messages": [_SyntheticMessage("d")]}, use, {"ok": True, "report": "", "args": args, "why": "t"}

    seen = []

    class R:
        content = json.dumps({"headline": "h", "explanation": ["e"], "suggestions": []})
        usage_metadata = {}

    class LLM:
        def invoke(self, msgs):
            seen.append(json.loads(msgs[1]["content"]))
            return R()

    monkeypatch.setattr(tuner_agent, "run_extraction", fake_extraction)
    monkeypatch.setattr(tuner_agent, "write_abstract", lambda *a, **k: ("abs", _empty_usage()))
    monkeypatch.setattr(llm_factory, "build_llm", lambda role, **k: LLM())
    spec = system_spec.normalize_defaults({"status": "complete", "system_name": "CP", "dynamics": {
        "states": ["x1", "x2"], "inputs": ["u"], "outputs": ["x1"],
        "state_equations": ["x2", "(-(u + 0.06*x2**2*sin(x1))*cos(x1) + 11.2815*sin(x1))/(0.06*sin(x1)**2 + 0.4)"],
        "x0": [0.2, 0.0], "references": [{"output": "x1", "expr": "10*sin(5*t)"}],
        "solver_step": 0.001, "sim_time": 10.0}})
    events = []
    result, *_ = tuner_agent.run_full_pipeline("", system_spec=spec, on_event=events.append)
    assert len(result["pre_run_warnings"]) == 1 and "1.57" in result["pre_run_warnings"][0]
    assert any("Warning before the run" in str(e.get("text", "")) for e in events)
    assert seen and seen[0]["pre_run_warnings"] == result["pre_run_warnings"]
    lo, hi = seen[0]["reference_windows"]["x1"]
    assert lo == pytest.approx(-1.5708, abs=0.01) and hi == pytest.approx(1.5708, abs=0.01)
