"""
Run from the repo root with:
    python -m pytest backend_core/AgentAdaptive/tests/test_conversion_check.py -v
"""

import copy

import numpy as np

from backend_core.AgentAdaptive.tools.conversion_check import apply_simulation_check, simulation_check
from backend_core.AgentAdaptive.tools.reporter import _render_conversion_section


def series_of(columns, names):
    data = np.array(columns, dtype=float).T
    rows = [[None if not np.isfinite(v) else float(v) for v in row] for row in data]
    return {"channels": {"x": {"names": names, "data": rows}}}


def test_bounded_run_passes_and_reports_untracked_outputs():
    t = np.linspace(0, 10, 400)
    check = simulation_check(series_of([np.sin(t), 0.5 * np.cos(t)], ["x1", "x2"]), ["x2"])
    assert check["available"] and check["ok"] and check["diverging_states"] == []
    assert abs(check["untracked_outputs"]["x2"]["max"] - 0.5) < 1e-3


def test_state_that_blows_up_late_is_flagged():
    t = np.linspace(0, 10, 400)
    grow = np.where(t > 7.5, np.exp(t - 7.5), 0.1)
    check = simulation_check(series_of([np.sin(t), grow], ["x1", "x2"]))
    assert check["ok"] is False and check["diverging_states"] == ["x2"]


def test_non_finite_values_count_as_diverging():
    col = np.ones(100)
    col[60:] = np.nan
    check = simulation_check(series_of([col], ["x1"]))
    assert check["diverging_states"] == ["x1"]


def test_missing_series_is_reported_not_guessed():
    assert simulation_check(None) == {"available": False}
    assert simulation_check({"channels": {}}) == {"available": False}


def _record(label="A"):
    return {"id": "T2", "title": "track a subset of the outputs", "label": label,
            "label_reasons": [], "task_changed": True, "untracked_outputs": ["x1"],
            "changes": ["The plant now tracks x2 only."], "original": {"states": ["x1", "x2"]},
            "details": {}, "why": "The plant does not fit.", "verification": {"summary": "Fits.", "warnings": []},
            "selection": {"option_id": "p1", "source": "llm", "chosen_by": "user", "rationale": "x2 is clean",
                          "risks": "x1 is dropped", "vetoed": False, "veto_reason": ""},
            "explanation": {"summary": "Tracks x2.", "recommendation": "Use p1."}, "simulation_check": None}


def test_blowup_downgrades_the_label_and_leaves_the_input_alone():
    rec = _record()
    before = copy.deepcopy(rec)
    t = np.linspace(0, 10, 400)
    out = apply_simulation_check(rec, series_of([np.sin(t), np.where(t > 7.5, np.exp(t - 7.5), 0.1)], ["x1", "x2"]))
    assert out["label"] == "C" and "x2" in out["label_reasons"][0]
    assert out["simulation_check"]["ok"] is False
    assert rec == before


def test_healthy_run_keeps_label_a():
    t = np.linspace(0, 10, 400)
    out = apply_simulation_check(_record(), series_of([np.sin(t), np.cos(t)], ["x1", "x2"]))
    assert out["label"] == "A" and out["simulation_check"]["ok"] is True


def test_section_is_empty_without_a_conversion():
    assert _render_conversion_section(None) == ""
    assert _render_conversion_section({}) == ""


def test_section_states_the_facts_the_user_needs():
    t = np.linspace(0, 10, 400)
    rec = apply_simulation_check(_record(), series_of([np.sin(t), np.cos(t)], ["x1", "x2"]))
    rec["details"] = {"input_map": {"u1": "0.83333333333333*v1"}, "input_map_display": {"u1": "0.8333*v1"}}
    text = _render_conversion_section(rec)
    assert text.startswith("\n## Plant Conversion")
    assert "The plant does not fit." in text and "The plant now tracks x2 only." in text
    assert "**Safety label:** A:" in text
    assert "changes the task" in text and "x1" in text
    assert "u1 = 0.8333*v1" in text and "0.83333333" not in text
    assert "all states stayed bounded" in text and "| x1 |" in text
    assert "you chose it" in text and "the model" in text
    assert "Tracks x2." in text and "Use p1." not in text
    assert "not independently verified" not in text and "x1 is dropped" not in text


def test_section_shows_the_relative_degrees_python_found():
    rec = _record(label="C")
    rec["verification"].update(relative_degrees=[4, 4], hidden_dimension=0)
    text = _render_conversion_section(rec)
    assert "Relative degrees of the outputs: [4, 4]; no hidden states." in text


def test_section_lists_label_reasons_and_auto_selection_from_rules():
    rec = _record(label="C")
    rec["label_reasons"] = ["2 hidden state(s) are only checked locally."]
    rec["selection"].update(chosen_by="auto-selected", source="rules", vetoed=True, veto_reason="drops x1")
    text = _render_conversion_section(rec)
    assert "**Safety label:** C:" in text and "- 2 hidden state(s) are only checked locally." in text
    assert "selected automatically" in text and "fixed rules (no model was used)" in text
    assert "The automatic reviewer raised a concern (an opinion, not a check): drops x1" in text


def test_section_explains_an_extension_and_a_missing_reference():
    rec = _record(label="C")
    rec["details"] = {"rate_inputs": ["v_dot"], "extended_inputs": ["v"], "initial_inputs": {"v": 1.0},
                      "assumed_initial_inputs": ["v"], "missing_references": ["x2"]}
    text = _render_conversion_section(rec)
    assert "**New control inputs:** v_dot (rate of change of v)" in text
    assert "integrator" in text and "- v starts at 1 (assumed)" in text
    assert "**A reference is still needed for:** x2." in text
