"""
Run from the repo root with:
    python -m pytest backend_core/AgentAdaptive/tests/test_prose_check.py -v
"""

import json

import pytest

from test_compat_agent import models, plan, spec_of, strategist, critic  # noqa: F401  (models is a fixture)
from test_compat_flow import explainer_reply, proposal_for  # noqa: F401
from test_plant_analysis import UNICYCLE
from backend_core.AgentAdaptive.agents import compat_agent
from backend_core.AgentAdaptive.tools.prose_check import check

EVIDENCE = json.dumps({"inputs": ["v", "om"], "outputs": ["x", "y"], "p1": "v_dot", "n": 32, "w": 0.5, "need": "1 input and 1 output", "has": "2 inputs and 2 outputs"})
DIMS = {"n_inputs": 2, "n_outputs": 2}


def test_clean_text_passes():
    assert check("Option p1 turns v into a state driven by v_dot; 32 points were checked.", EVIDENCE, DIMS) == []


def test_invented_numbers_and_names_are_caught():
    assert "the number 40" in check("It failed at 40 points.", EVIDENCE, DIMS)[0]
    assert "x7" in check("State x7 is unused.", EVIDENCE, DIMS)[0]
    assert "'q'" in check("The input 'q' is extended.", EVIDENCE, DIMS)[0]


def test_numbers_can_be_limited_to_one_fact():
    assert check("There are 32 samples.", EVIDENCE, DIMS, number_source="0.5 only") != []
    assert check("There are 32 samples.", EVIDENCE, DIMS, numbers=False) == []


def test_wrong_input_output_comparisons_are_caught():
    bad = check("This addresses the issue of having more inputs than outputs.", EVIDENCE, DIMS)
    assert "more inputs than outputs" in bad[0]
    assert check("2 inputs but 2 outputs, so it cannot work.", EVIDENCE, DIMS) != []
    assert check("The plant has 2 inputs and 2 outputs; backstepping needs 1 input and 1 output.", EVIDENCE, DIMS) == []
    assert check("The plant has more inputs than outputs.", EVIDENCE, {"n_inputs": 3, "n_outputs": 2}) == []


def test_stability_promises_are_caught_unless_negated():
    assert check("This keeps the plant stable and is proven safe.", EVIDENCE, DIMS) != []
    assert check("Nothing here is guaranteed.", EVIDENCE, DIMS) == []
    assert check("There is no guarantee for the untracked output.", EVIDENCE, DIMS) == []
    assert check("This guarantees stability.", EVIDENCE, DIMS) != []


def test_persian_digits_count_as_numbers():
    assert check("۴۰ نقطه", EVIDENCE, DIMS) != []
    assert check("۳۲ نقطه", EVIDENCE, DIMS) == []


def unicycle_proposal(models):
    spec = spec_of(UNICYCLE)
    models(compat=[strategist(plan("p1", "T4", {"inputs": ["v"]}))], critic=[critic(p1=None)])
    return spec, compat_agent.propose_conversions(spec)


def test_explainer_rewrites_every_fact_and_keeps_the_good_ones(models):
    spec, prop = unicycle_proposal(models)
    facts = compat_agent._facts(prop["analysis"])
    assert len(facts) >= 1
    models(compat_explainer=[{"headline": "Your plant needs one extra step",
                              "why_not_fit": [{"id": "f1", "text": "The tracked outputs cannot be commanded yet."}],
                              "options": [{"id": "p1", "summary": "Make v a state and command v_dot."}],
                              "recommendation": "Use p1."}])
    out = compat_agent.explain(prop, spec)
    assert out["source"] == "llm" and out["replaced"] == []
    assert out["why_not_fit"][0] == "The tracked outputs cannot be commanded yet."
    assert out["why_not_fit"][1:] == facts[1:]
    assert len(out["why_not_fit"]) == len(facts)


def test_a_bullet_with_a_wrong_claim_is_replaced_by_the_fact_after_one_repair(models):
    spec, prop = unicycle_proposal(models)
    facts = compat_agent._facts(prop["analysis"])
    wrong = explainer_reply(headline="Fine", options=[], recommendation="Use p1.")
    wrong["why_not_fit"] = [{"id": "f1", "text": "It has more inputs than outputs."}]
    llms = models(compat_explainer=[wrong, wrong])
    out = compat_agent.explain(prop, spec)
    assert out["why_not_fit"][0] == facts[0]
    assert out["replaced"] and out["replaced"][0]["field"] == "why_not_fit f1"
    assert len(llms["compat_explainer"].calls) == 2
    retry = llms["compat_explainer"].calls[1][-1]["content"]
    assert "rejected_sentences" in retry and "more inputs than outputs" in retry
    assert out["usage"]["total_tokens"] == 300


def test_the_repair_round_can_fix_it(models):
    spec, prop = unicycle_proposal(models)
    wrong = explainer_reply(headline="Fine", options=[], recommendation="Use p1.")
    wrong["why_not_fit"] = [{"id": "f1", "text": "It has more inputs than outputs."}]
    right = explainer_reply(headline="Fine", options=[], recommendation="Use p1.")
    right["why_not_fit"] = [{"id": "f1", "text": "The decoupling matrix is singular."}]
    models(compat_explainer=[wrong, right])
    out = compat_agent.explain(prop, spec)
    assert out["why_not_fit"][0] == "The decoupling matrix is singular." and out["replaced"] == []


def test_followup_is_checked_and_replaced_when_it_keeps_failing(models):
    spec, prop = unicycle_proposal(models)
    bad = {"reply": "Option p1 makes the plant stable, guaranteed."}
    llms = models(compat_explainer=[bad, bad])
    text, usage = compat_agent.answer_followup(prop, spec, "is it safe?")
    assert text == compat_agent._CHAT_FALLBACK and len(llms["compat_explainer"].calls) == 2
    assert usage["total_tokens"] == 300


def test_followup_repairs_once_and_allows_general_numbers(models):
    spec, prop = unicycle_proposal(models)
    models(compat_explainer=[{"reply": "It is stable and proven."},
                             {"reply": "A second-order system needs 2 states; p1 adds one for v."}])
    text, _ = compat_agent.answer_followup(prop, spec, "tell me about integrators")
    assert text.startswith("A second-order system")


def test_a_plan_text_that_contradicts_the_facts_is_replaced_and_noted(models):
    bad = plan("p1", "T4", {"inputs": ["v"]}, rationale="Fixes the case of more inputs than outputs.",
               risks="Starting value of 'q' is assumed.")
    models(compat=[strategist(bad)], critic=[critic(p1=None)])
    out = compat_agent.propose_conversions(spec_of(UNICYCLE))
    opt = out["options"][0]
    assert opt["rationale"] == "The smallest set of inputs that restores an invertible matrix."
    assert opt["risks"] == compat_agent._RULE_TEXT["T4"][1]
    assert len([n for n in out["notes"] if "replaced by fixed text" in n]) == 2
