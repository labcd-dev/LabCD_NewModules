"""
Run from the repo root with:
    python -m pytest backend_core/AgentAdaptive/tests/test_compat_flow.py -v
"""

import json

import pytest

from test_compat_agent import (  # noqa: F401  (models is a fixture)
    FakeLLM, THREE_IN_TWO_OUT, UNSTABLE_PICK, critic, models, plan, spec_of, strategist,
)
from backend_core.AgentAdaptive.agents import compat_agent, tuner_agent
from backend_core.AgentAdaptive.agents.agent_io import _SyntheticMessage, _empty_usage
from backend_core.AgentAdaptive.controller.structure_build import _build_structure_from_spec, _verify_structure
from backend_core.AgentAdaptive.tools import model_pricing


def proposal_for(models, plant=THREE_IN_TWO_OUT, plans=None, verdicts=None):
    plans = plans or [plan(params={"weights": [1, 1, 1]})]
    models(compat=[strategist(*plans)], critic=[critic(**(verdicts or {p["id"]: None for p in plans}))])
    spec = spec_of(plant)
    return spec, compat_agent.propose_conversions(spec)


def explainer_reply(**over):
    reply = {"headline": "Your plant has more inputs than outputs",
             "why_not_fit": ["3 inputs but only 2 outputs"],
             "options": [{"id": "p1", "summary": "Drive the 3 inputs with 2 commands."}],
             "recommendation": "Use p1 because it keeps every output."}
    reply.update(over)
    return reply


def test_explainer_merges_the_model_text_over_the_verified_template(models):
    spec, prop = proposal_for(models)
    llms = models(compat_explainer=[explainer_reply(options=[
        {"id": "p1", "summary": "Drive the 3 inputs with 2 commands."},
        {"id": "ghost", "summary": "should be ignored"}])])
    out = compat_agent.explain(prop, spec)
    assert out["source"] == "llm" and out["error"] is None
    assert out["headline"] == "Your plant has more inputs than outputs"
    assert out["options"] == {"p1": "Drive the 3 inputs with 2 commands."}
    assert out["usage"]["total_tokens"] == 150
    evidence = json.loads(llms["compat_explainer"].calls[0][1]["content"])
    assert evidence["options"][0]["label"] == "C" and evidence["recommended"] == "p1"
    assert evidence["plant"]["inputs"] == ["u1", "u2", "u3"]


def test_explainer_fills_gaps_from_the_template(models):
    spec, prop = proposal_for(models)
    models(compat_explainer=[{"headline": "", "options": [], "recommendation": ""}])
    out = compat_agent.explain(prop, spec)
    assert out["source"] == "llm"
    assert "3 input(s) and 2 output(s)" in out["why_not_fit"][0]
    assert out["options"]["p1"] == prop["options"][0]["changes"][0]
    assert out["recommendation"].startswith("Recommended: p1")


@pytest.mark.parametrize("reply", [RuntimeError("down"), "not json"])
def test_explainer_falls_back_to_the_template_and_says_why(models, reply):
    spec, prop = proposal_for(models)
    models(compat_explainer=[reply])
    out = compat_agent.explain(prop, spec)
    assert out["source"] == "template" and out["error"]
    assert out["options"]["p1"] == prop["options"][0]["changes"][0]


def test_explainer_is_not_called_when_nothing_can_be_converted(models):
    from test_plant_analysis import UNFIXABLE
    spec = spec_of(UNFIXABLE)
    prop = compat_agent.propose_conversions(spec)
    out = compat_agent.explain(prop, spec)
    assert out["source"] == "template" and out["error"] is None
    assert "cannot be designed" in out["headline"] and out["options"] == {}


def test_vetoed_everything_is_explained_with_the_reason(models):
    spec, prop = proposal_for(models, UNSTABLE_PICK, [plan("p1", "T2", {"outputs": ["x1"]})], {"p1": "user wants x2"})
    out = compat_agent._template_explanation(prop)
    assert "vetoed every one" in out["recommendation"]
    assert "user wants x2" in out["options"]["p1"]


def test_followup_passes_question_history_and_evidence(models):
    spec, prop = proposal_for(models)
    llms = models(compat_explainer=[{"reply": "Option p1 keeps both outputs."}])
    history = [{"role": "user", "text": "why 3 inputs?"}, {"role": "assistant", "text": "Because."}]
    reply, usage = compat_agent.answer_followup(prop, spec, "Does it keep y2?", history=history)
    assert reply == "Option p1 keeps both outputs." and usage["total_tokens"] == 150
    content = llms["compat_explainer"].calls[0][1]["content"]
    assert "Does it keep y2?" in content and "User: why 3 inputs?" in content and "You: Because." in content
    assert '"label": "C"' in content


@pytest.mark.parametrize("reply", [RuntimeError("down"), {"reply": ""}, "garbage"])
def test_followup_failure_is_a_plain_apology(models, reply):
    spec, prop = proposal_for(models)
    models(compat_explainer=[reply])
    text, _ = compat_agent.answer_followup(prop, spec, "hi")
    assert text.startswith("I couldn't answer that")


def test_replan_puts_the_feedback_in_front_of_the_model(models):
    spec, prop = proposal_for(models)
    llms = models(compat=[strategist(plan("p1", "T1", {"weights": [1, 1, 50]}))], critic=[critic(p1=None)])
    new = compat_agent.replan(spec, prop, "u3 is a tiny motor, spare it", clarification_record="earlier chat")
    assert new["status"] == "needs_choice" and new["options"][0]["params"]["weights"][2] == 50.0
    evidence = json.loads(llms["compat"].calls[0][1]["content"])
    text = evidence["user_conversation"]
    assert text.startswith("User feedback on the previous options: u3 is a tiny motor")
    assert "p1 (" in text and text.endswith("earlier chat")


def test_conversion_record_carries_everything_the_report_needs(models):
    spec, prop = proposal_for(models)
    expl = {"headline": "h", "why_not_fit": ["w"], "options": {"p1": "does a thing"}, "recommendation": "r"}
    applied = compat_agent.apply_option(spec, prop["options"][0])
    rec = compat_agent.build_conversion_record(prop, prop["options"][0], applied, "auto-selected", expl)
    json.dumps(rec)
    assert rec["id"] == "T1" and rec["label"] == "C" and rec["simulation_check"] is None
    assert rec["why"] == prop["analysis"]["summary"]
    assert rec["selection"] == {"option_id": "p1", "source": "llm", "chosen_by": "auto-selected",
                                "rationale": "because", "risks": "some risk", "vetoed": False, "veto_reason": ""}
    assert rec["explanation"]["summary"] == "does a thing"
    assert rec["verification"]["summary"].startswith("The plant fits")
    assert rec["original"]["inputs"] == ["u1", "u2", "u3"]


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.setenv("LABCD_ADAPTIVE_SHOW_PLOTS", "0")
    monkeypatch.setenv("MPLBACKEND", "Agg")

    def fake_extraction(spec, on_event=None, clarification_record=None):
        structure = _verify_structure(_build_structure_from_spec(spec)[0])
        args = dict(structure, method="smc", reasoning="test", notes_limitations=None)
        usage = {"agent": _empty_usage(), "total": _empty_usage(), "agent_turns": [], "timeline": []}
        return ({"messages": [_SyntheticMessage("design")]}, usage,
                {"ok": True, "report": "", "args": args, "why": "test"})

    monkeypatch.setattr(tuner_agent, "run_extraction", fake_extraction)
    monkeypatch.setattr(tuner_agent, "write_abstract", lambda *a, **k: ("abstract", _empty_usage()))
    monkeypatch.setattr(tuner_agent, "_diagnose_if_failed", lambda *a, **k: (None, _empty_usage()))


def converted_spec(models):
    spec, prop = proposal_for(models)
    applied = compat_agent.apply_option(spec, prop["options"][0])
    dyn = applied["spec"]["dynamics"]
    dyn.update(references=[{"output": "x1", "expr": "0.1*sin(t)"}, {"output": "x2", "expr": "0.1"}],
               x0=[0.1, 0.0, 0.0, 0.0], sim_time=2.0, solver_step=0.01)
    return spec, prop, applied


def test_run_full_pipeline_reports_the_conversion_and_counts_its_tokens(models, pipeline):
    spec, prop, applied = converted_spec(models)
    record = compat_agent.build_conversion_record(prop, prop["options"][0], applied, "user")
    result, usage, _, _ = tuner_agent.run_full_pipeline(
        "", system_spec=applied["spec"], conversion_record=record, compat_usage=prop["usage_by_role"])

    text = result["messages"][-1].content
    assert "## Plant Conversion" in text and "split the commands across the inputs" in text
    conv = result["conversion"]
    assert conv["simulation_check"]["available"] is True
    assert set(conv["simulation_check"]["states"]) == {"x1", "x2", "x3", "x4"}
    assert usage["compat"]["total_tokens"] == 150 and usage["compat_critic"]["total_tokens"] == 150
    assert usage["total"]["total_tokens"] >= 300
    assert usage["timeline"][0]["actor"] == "compat"
    rows, _ = model_pricing.run_cost_rows(usage)
    assert {"Compatibility Agent", "Compatibility Reviewer"} <= {r["label"] for r in rows}


def test_run_full_pipeline_without_a_conversion_is_unchanged(models, pipeline):
    _, _, applied = converted_spec(models)
    result, usage, _, _ = tuner_agent.run_full_pipeline("", system_spec=applied["spec"])
    assert "Plant Conversion" not in result["messages"][-1].content
    assert result["conversion"] is None
    assert "compat" not in usage


def test_conversion_record_survives_a_failed_extraction(models, monkeypatch, pipeline):
    _, prop, applied = converted_spec(models)
    record = compat_agent.build_conversion_record(prop, prop["options"][0], applied, "user")

    def failing(spec, on_event=None, clarification_record=None):
        usage = {"agent": _empty_usage(), "total": _empty_usage(), "agent_turns": [], "timeline": []}
        return {"messages": [_SyntheticMessage("could not design")]}, usage, {"ok": False, "report": "", "args": {}}

    monkeypatch.setattr(tuner_agent, "run_extraction", failing)
    result, _, _, _ = tuner_agent.run_full_pipeline("", system_spec=applied["spec"], conversion_record=record)
    assert result["conversion"] == record


def test_models_see_which_inputs_share_an_output_and_the_real_split(models):
    llms = models(compat=[strategist(plan(params={"weights": [1, 1, 5]}))], critic=[critic(p1=None)],
                  compat_explainer=[explainer_reply()])
    spec = spec_of(THREE_IN_TWO_OUT)
    prop = compat_agent.propose_conversions(spec)
    menu = json.loads(llms["compat"].calls[0][1]["content"])["menu"][0]
    assert menu["influence"]["matrix"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]

    opt = prop["options"][0]
    assert opt["input_map"]["u1"] == "1.0*v1"
    assert float(opt["input_map"]["u3"].split("*")[0]) == pytest.approx(1 / 6)

    assert opt["input_map_display"] == {"u1": "1.0*v1", "u2": "0.8333*v2", "u3": "0.1667*v2"}
    seen_by_critic = json.loads(llms["compat_critic"].calls[0][1]["content"])["options"][0]
    assert seen_by_critic["input_map"] == opt["input_map_display"]
    compat_agent.explain(prop, spec)
    seen_by_explainer = json.loads(llms["compat_explainer"].calls[0][1]["content"])["options"][0]
    assert seen_by_explainer["input_map"] == opt["input_map_display"]


def test_prompts_tell_the_models_to_use_the_influence_table_and_the_map():
    assert "menu.influence" in compat_agent.STRATEGIST_PROMPT
    assert "input_map" in compat_agent.EXPLAINER_PROMPT and "input_map" in compat_agent.CRITIC_PROMPT
