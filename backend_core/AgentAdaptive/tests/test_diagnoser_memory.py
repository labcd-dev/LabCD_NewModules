"""
Run from the repo root with:
    python -m pytest backend_core/AgentAdaptive/tests/test_diagnoser_memory.py -v
"""

from backend_core.AgentAdaptive.agents import diagnoser_agent as da
from backend_core.AgentAdaptive.tools import diagnosis_chain as dc
from backend_core.AgentAdaptive.tools import failure_mode as fm


def _spec(ref="0.5*sin(5*t)", x0=(0.2, 0.0), dt=0.001, limits=None):
    return {"system_name": "P", "dynamics": {
        "states": ["x1", "x2"], "inputs": ["u"], "outputs": ["x1"], "state_equations": ["x2", "-x1 + u"],
        "x0": list(x0), "references": [{"output": "x1", "expr": ref}], "solver_step": dt,
        "sim_time": 10.0, "parameters": {}, "input_limits": limits or []}}


def _fail(pct=40.0):
    return {"success": False, "success_reason": "mse_target: x", "tracking_pct_headline": pct}


def _card(lever, target, values, title="card"):
    return {"title": title, "detail": "", "lever": lever, "target": target,
            "options": [{"label": v, "value": v} for v in values]}


def _attempt(spec, cards=None, prev=None, pct=40.0):
    report = {"suggestions": [{"title": c["title"], "lever": c["lever"], "target": c["target"],
                               "options": c["options"], "computed": c.get("computed", False)}
                              for c in (cards or [])]}
    return dc.make_attempt(prev or [], spec, _fail(pct), report=report)


def test_a_value_that_already_failed_is_not_offered_again():
    first = _attempt(_spec(ref="0.4*sin(5*t)"))
    memory = dc.suggestion_memory([first], _spec(ref="0.5*sin(5*t)"))
    report = {"explanation": ["e"], "suggestions": [
        _card("reference", "x1", ["0.4 * sin(5*t)", "0.3*sin(5*t)", "0.5*sin(5*t)"])]}
    out = da._prune_for_evidence(report, {}, memory)
    assert [o["value"] for o in out["suggestions"][0]["options"]] == ["0.3*sin(5*t)"]


def test_a_card_with_nothing_new_left_is_removed():
    memory = dc.suggestion_memory([_attempt(_spec(ref="0.4*sin(5*t)"))], _spec(ref="0.3*sin(5*t)"))
    report = {"explanation": ["e"], "suggestions": [_card("reference", "x1", ["0.4*sin(5*t)", "0.3*sin(5*t)"])]}
    assert da._prune_for_evidence(report, {}, memory)["suggestions"] == []


def test_a_card_shown_twice_and_never_followed_is_dropped_the_third_time():
    card = _card("reference", "x1", ["0.2*sin(5*t)", "0.1*sin(5*t)"], title="Smaller swing")
    a1 = _attempt(_spec(ref="0.5*sin(5*t)"), [card])
    a2 = _attempt(_spec(ref="0.6*sin(5*t)"), [card], prev=[a1])
    memory = dc.suggestion_memory([a1, a2], _spec(ref="0.7*sin(5*t)"), changes_now=[])
    report = {"explanation": ["e"], "suggestions": [dict(card)]}
    assert da._prune_for_evidence(report, {}, memory)["suggestions"] == []


def test_a_card_that_was_followed_is_not_called_ignored():
    card = _card("reference", "x1", ["0.2*sin(5*t)", "0.1*sin(5*t)"], title="Smaller swing")
    a1 = _attempt(_spec(ref="0.5*sin(5*t)"), [card])
    a2 = _attempt(_spec(ref="0.2*sin(5*t)"), [card], prev=[a1])         # the user took 0.2
    assert a2["changes"][0]["followed"]["exact_option"] is True
    memory = dc.suggestion_memory([a1, a2], _spec(ref="0.7*sin(5*t)"), changes_now=[])
    assert not memory["ignored"]


def test_the_memory_reaches_the_model_in_words():
    memory = dc.suggestion_memory([_attempt(_spec(ref="0.4*sin(5*t)", limits=[
        {"input": "u", "min": -10.0, "max": 10.0}]))], _spec())
    said = dc.memory_for_evidence(memory)["values_already_run_and_failed"]
    assert "0.4*sin(5*t)" in said["reference of x1"] and "0.5*sin(5*t)" in said["reference of x1"]
    assert said["input limit of u"] == ["-10,10"]
    assert said["solver step dt"] == ["0.001"]


def _step_report():
    return {"explanation": ["e"], "suggestions": [_card("step_time", None, ["0.01", "0.0001"])]}


def test_a_step_for_a_non_numerical_failure_is_dropped_and_a_bigger_one_never_kept():
    healthy = {"simulation": {"dt": 0.001}, "metrics": {"numerically_healthy": True}}
    assert da._prune_for_evidence(_step_report(), healthy)["suggestions"] == []
    blown = {"simulation": {"dt": 0.001}, "metrics": {"numerically_healthy": False}}
    out = da._prune_for_evidence(_step_report(), blown)["suggestions"]
    assert [o["value"] for o in out[0]["options"]] == ["0.0001"]


def test_an_explained_instability_is_not_a_numerical_one():
    explained = {"simulation": {"dt": 0.001}, "metrics": {"numerically_healthy": True},
                 "failure_window": {"t": []}, "control_spikes": {"first_t": 1.0}}
    assert da._prune_for_evidence(_step_report(), explained)["suggestions"] == []
    unexplained = {"simulation": {"dt": 0.001}, "metrics": {"numerically_healthy": True},
                   "failure_window": {"t": []}}
    assert da._prune_for_evidence(_step_report(), unexplained)["suggestions"]


def test_a_model_sentence_about_earlier_runs_is_removed_and_the_computed_one_leads():
    first = _attempt(_spec())
    ev = {"previous_runs": dc.history_for_evidence([first]),
          "changes_since_last_run": [], "metrics": _fail(40.0)}
    report = {"explanation": ["Previous runs with lower amplitudes did better.", "The error stays at 40%."],
              "suggestions": []}
    out = da._prune_for_evidence(report, ev)["explanation"]
    assert len(out) == 2 and out[0].startswith("Since run 1:") and "Previous" not in " ".join(out)


def test_the_run_number_is_an_integer_and_only_exact_values_are_credited():
    prev = [{"run": 3.0, "outcome": {"success": True, "tracking_pct": 99.0}}]
    ch = [{"lever": "reference", "what": "reference for x1", "from": "a", "to": "b",
           "followed": {"title": "Smaller", "exact_option": False}}]
    now = {"success": False, "tracking_pct_headline": 10.0}
    text = dc.describe_changes(prev, ch, now)
    assert text.startswith("Since run 3:") and "suggest" not in text
    ch[0]["followed"]["exact_option"] = True
    assert 'suggested under "Smaller"' in dc.describe_changes(prev, ch, now)


def test_ruled_out_lists_what_was_checked_when_there_is_no_computed_cause():
    out = fm.classify({"metrics": {"numerically_healthy": True}, "failure_window": None,
                       "input_limits": {"limits": {"u": [-100, 100]},
                                        "saturation": {"any_input_fraction": 0.0}}})
    assert out["mode"] == "not_limit_related"
    assert len(out["ruled_out"]) == 4 and "input limits" in out["ruled_out"][0]
    assert "ruled_out" not in fm.classify({"pre_run_warnings": ["w"], "metrics": {}})


def test_a_limit_that_makes_a_failing_design_much_worse_has_its_own_mode():
    ev = {"metrics": {"tracking_pct_headline": 0.0},
          "input_limits": {"limits": {"u": [-10, 10]},
                           "saturation": {"any_input_fraction": 0.6, "first_saturated_at_s": 0.9},
                           "without_limits": {"passes": False, "tracking_pct": 75.2,
                                              "failed_check": "mse_target", "peak_by_input": {"u": 12.0}}}}
    out = fm.classify(ev)
    assert out["mode"] == "limit_adds_to_failure" and "much worse" in " ".join(out["chain"])
    ev["metrics"]["tracking_pct_headline"] = 70.0
    assert fm.classify(ev)["mode"] == "fails_even_without_limit"


def test_a_free_run_that_failed_on_spikes_is_not_compared_on_tracking():
    ev = {"metrics": {"tracking_pct_headline": 6.2},
          "input_limits": {"limits": {"u": [-30, 30]},
                           "saturation": {"any_input_fraction": 0.9, "first_saturated_at_s": 0.0},
                           "without_limits": {"passes": False, "tracking_pct": 99.6,
                                              "failed_check": "bounded", "peak_by_input": {"u": 9e5}}}}
    assert fm.classify(ev)["mode"] == "fails_even_without_limit"


def test_clipping_from_the_first_instant_is_not_an_initial_condition_problem():
    import numpy as np
    from backend_core.AgentAdaptive.tools import diagnostics_evidence as de
    t = np.linspace(0, 10, 101)
    col = np.zeros((101, 1))
    ctx = {"method": "smc", "states": ["x"], "inputs": ["u"], "outputs": ["x"], "t": t, "y": col,
           "ref": col, "u": col, "x": col, "x0": [0.0], "system_text": "", "control_law_text": "",
           "refs": [], "dt": 0.1, "t_end": 10.0, "input_limits": [[-1.0, 1.0]],
           "u_cmd": np.full((101, 1), 5.0),
           "input_saturation": {"any_input_fraction": 1.0, "first_saturated_at_s": 0.0,
                                "peak_command": [5.0]}}
    ev, _ = de.build_evidence(ctx, {})
    assert ev["first_trouble_s"] is None
    out = da._prune_for_evidence({"explanation": ["e"], "suggestions": [
        _card("initial_condition", None, ["0.1,0"])]}, ev)
    assert out["suggestions"] == []


def test_a_limit_card_from_the_model_goes_even_when_this_run_had_no_limit():
    report = {"explanation": ["e"], "suggestions": [
        _card("system", None, [], title="Increase the input limit for u"),
        _card("reference", "x1", ["0.3*sin(5*t)"], title="Smaller swing")]}
    out = da._prune_for_evidence(report, {})
    assert [s["lever"] for s in out["suggestions"]] == ["reference"]


def test_previously_is_a_history_claim():
    report = {"explanation": ["The limit was previously raised to 30.", "x1 tracks 75%."],
              "suggestions": []}
    assert da._prune_for_evidence(report, {})["explanation"] == ["x1 tracks 75%."]


def test_the_causal_chain_and_the_ruled_out_list_become_computed_bullets():
    ev = {"failure_mode": {"mode": "not_limit_related", "chain": ["The limits were never reached."],
                           "ruled_out": ["control-signal spikes (none)", "numerical blow-up (none)"]},
          "previous_runs": dc.history_for_evidence([_attempt(_spec())]),
          "changes_since_last_run": [], "metrics": _fail(40.0)}
    out = da._prune_for_evidence({"explanation": ["The error stays at 40%."], "suggestions": []}, ev)
    text = out["explanation"]
    assert text[0].startswith("Since run 1:") and text[1] == "The limits were never reached."
    assert text[2] == "Checked and not the cause: control-signal spikes (none); numerical blow-up (none)."
    assert text[3] == "The error stays at 40%."


def test_a_model_bullet_that_restates_a_computed_fact_is_dropped():
    ev = {"failure_mode": {"mode": "limit_too_small", "chain": [
        "The input is limited to u in [-10, 10]; the plant clipped the command 67% of the run."]}}
    report = {"explanation": ["The input is limited to u in [-10, 10]; the plant clipped the command 67% of the run.",
                              "The input is limited to u in [-10, 10], and the plant clipped the command 67% of the run.",
                              "x1 swings to -1.57 rad in the chart."], "suggestions": []}
    out = da._prune_for_evidence(report, ev)["explanation"]
    assert out == [ev["failure_mode"]["chain"][0], "x1 swings to -1.57 rad in the chart."]


def test_a_reference_option_outside_the_computed_window_is_removed():
    ev = {"reference_windows": {"x1": [-1.5708, 1.5708]}, "simulation": {"t_end": 10.0}}
    report = {"explanation": ["e"], "suggestions": [
        _card("reference", "x1", ["8*sin(5*t)", "1.2*sin(5*t)", "1.5*sin(2*t)"])]}
    out = da._prune_for_evidence(report, ev)["suggestions"]
    assert [o["value"] for o in out[0]["options"]] == ["1.2*sin(5*t)", "1.5*sin(2*t)"]
    only_bad = {"explanation": ["e"], "suggestions": [_card("reference", "x1", ["8*sin(5*t)"])]}
    assert da._prune_for_evidence(only_bad, ev)["suggestions"] == []


def test_an_option_that_cannot_be_evaluated_is_kept_not_guessed_at():
    ev = {"reference_windows": {"x1": [-1.5708, 1.5708]}, "simulation": {"t_end": 10.0}}
    report = {"explanation": ["e"], "suggestions": [_card("reference", "x1", ["not a formula ((("])]}
    assert da._prune_for_evidence(report, ev)["suggestions"]


def _window_evidence(expr="10*sin(5*t)", lo=-10, hi=10):
    return {"reference_windows": {"x1": [-1.5708, 1.5708]}, "reference_span": {"x1": {"min": lo, "max": hi}},
            "reference": [{"output": "x1", "expr": expr}], "simulation": {"t_end": 10.0}}


def test_the_reference_card_is_the_same_reference_scaled_into_the_window():
    card = fm.reference_card(_window_evidence())
    assert card["lever"] == "reference" and card["target"] == "x1" and card["computed"] is True
    assert [o["value"] for o in card["options"]] == ["1.257*sin(5*t)", "0.7854*sin(5*t)"]
    assert "(-1.571, 1.571)" in card["title"]


def test_the_scaled_reference_really_fits_even_with_an_offset():
    card = fm.reference_card(_window_evidence("3*sin(5*t) + 1", -2, 4))
    assert [o["value"] for o in card["options"]] == ["0.7854*sin(5*t) + 0.2618"]


def test_no_reference_card_when_the_reference_is_already_inside_or_nothing_is_known():
    assert fm.reference_card(_window_evidence("1*sin(5*t)", -1, 1)) is None
    assert fm.reference_card({}) is None
    assert fm.reference_card(dict(_window_evidence(), reference=[])) is None


def test_the_computed_reference_card_replaces_the_models_own_reference_cards():
    report = {"explanation": ["e"], "suggestions": [
        _card("reference", "x1", ["8*sin(5*t)", "0.2*sin(5*t)"], title="Smaller swing"),
        _card("initial_condition", None, ["0,0"], title="Start elsewhere")]}
    ev = dict(_window_evidence(), first_trouble_s=0.1)
    out = da._prune_for_evidence(report, ev)["suggestions"]
    assert [s["title"] for s in out] == ["Keep x1 inside (-1.571, 1.571)", "Start elsewhere"]


def test_an_ignored_lever_drops_the_models_cards_even_with_different_options():
    a1 = _attempt(_spec(ref="0.5*sin(5*t)"), [_card("reference", "x1", ["0.4*sin(5*t)", "0.3*sin(5*t)"])])
    a2 = _attempt(_spec(ref="0.6*sin(5*t)"), [_card("reference", "x1", ["0.4*sin(5*t)", "0.3*sin(5*t)",
                                                                       "0.2*sin(5*t)"])], prev=[a1])
    memory = dc.suggestion_memory([a1, a2], _spec(ref="0.7*sin(5*t)"), changes_now=[])
    report = {"explanation": ["e"], "suggestions": [_card("reference", "x1", ["0.15*sin(5*t)"])]}
    assert da._prune_for_evidence(report, {}, memory)["suggestions"] == []


def test_a_computed_card_is_never_dropped_as_ignored():
    a1 = _attempt(_spec(), [_card("reference", "x1", ["1*sin(5*t)"])])
    a2 = _attempt(_spec(ref="0.6*sin(5*t)"), [_card("reference", "x1", ["1*sin(5*t)"])], prev=[a1])
    memory = dc.suggestion_memory([a1, a2], _spec(ref="10*sin(5*t)"), changes_now=[])
    out = da._prune_for_evidence({"explanation": ["e"], "suggestions": []},
                                 dict(_window_evidence(), first_trouble_s=0.1), memory)
    assert [s["title"] for s in out["suggestions"]] == ["Keep x1 inside (-1.571, 1.571)"]


def _computed(values=("1.2*sin(5*t)",)):
    return dict(_card("reference", "x1", list(values), title="Keep x1 inside"), computed=True)


def test_a_computed_suggestion_is_shown_three_times_then_rests():
    a1 = _attempt(_spec(ref="10*sin(5*t)"), [_computed()])
    a2 = _attempt(_spec(ref="9*sin(5*t)"), [_computed()], prev=[a1])
    two = dc.suggestion_memory([a1, a2], _spec(ref="8*sin(5*t)"), changes_now=[])
    assert not two["tired"]                                   # still shown the third time
    a3 = _attempt(_spec(ref="8*sin(5*t)"), [_computed()], prev=[a1, a2])
    three = dc.suggestion_memory([a1, a2, a3], _spec(ref="7*sin(5*t)"), changes_now=[])
    assert three["tired"] == {("reference", "x1")}
    out = da._prune_for_evidence({"explanation": [], "suggestions": [_computed(("1.3*sin(5*t)",))]}, {}, three)
    assert out["suggestions"] == []


def test_taking_the_computed_value_keeps_it_from_resting():
    a1 = _attempt(_spec(ref="10*sin(5*t)"), [_computed()])
    a2 = _attempt(_spec(ref="1.2*sin(5*t)"), [_computed()], prev=[a1])      # took the offered value
    a3 = _attempt(_spec(ref="9*sin(5*t)"), [_computed()], prev=[a1, a2])
    memory = dc.suggestion_memory([a1, a2, a3], _spec(ref="8*sin(5*t)"), changes_now=[])
    assert not memory["tired"]
