"""
Run from the repo root with:
    python -m pytest backend_core/AgentAdaptive/tests/test_compat_agent.py -v
"""

import json

import pytest

from test_plant_analysis import (
    CART_POLE, THREE_IN_TWO_OUT, UNCONTROLLABLE, UNFIXABLE, UNICYCLE, UNSTABLE_ZERO, dyn,
)
from backend_core.AgentAdaptive.agents import compat_agent, llm_factory
from backend_core.AgentAdaptive.controller.plant_conversion import apply_input_squaring

UNSTABLE_PICK = dyn(["x1", "x2"], ["u", "x1 + x2"], ["u"], ["x1", "x2"])


def spec_of(plant):
    d = dict(plant, references=[], uncertainty=[], parameters={})
    return {"status": "complete", "system_name": "t", "dynamics": d}


class FakeResponse:
    def __init__(self, content):
        self.content = content
        self.usage_metadata = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}


class FakeLLM:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def invoke(self, messages):
        self.calls.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return FakeResponse(reply if isinstance(reply, str) else json.dumps(reply))


@pytest.fixture
def models(monkeypatch):
    llms = {}

    def install(compat=None, critic=None, compat_explainer=None):
        if compat is not None:
            llms["compat"] = FakeLLM(compat)
        if critic is not None:
            llms["compat_critic"] = FakeLLM(critic)
        if compat_explainer is not None:
            llms["compat_explainer"] = FakeLLM(compat_explainer)
        return llms

    def build(role, **kwargs):
        if role not in llms:
            raise AssertionError("no model was expected for role %r" % role)
        return llms[role]

    monkeypatch.setattr(llm_factory, "build_llm", build)
    return install


def plan(pid="p1", ptype="T1", params=None, rationale="because", risks="some risk"):
    return {"id": pid, "type": ptype, "params": params if params is not None else {},
            "rationale": rationale, "risks": risks}


def strategist(*plans, recommended=None):
    return {"plans": list(plans), "recommended": recommended or plans[0]["id"]}


def critic(**verdicts):
    return {"verdicts": [{"id": k, "verdict": "veto" if v else "ok", "reason": v or ""}
                         for k, v in verdicts.items()]}


def test_fitting_plant_never_reaches_a_model(models):
    out = compat_agent.propose_conversions(spec_of(CART_POLE))
    assert out["status"] == "fits" and out["llm_calls"] == 0 and out["options"] == []


def test_plant_no_conversion_can_fix_is_rejected_without_a_model(models):
    out = compat_agent.propose_conversions(spec_of(UNFIXABLE))
    assert out["status"] == "rejected" and out["llm_calls"] == 0
    assert "Looked for a fix (T3)" in out["message"]


def test_strategist_plan_is_verified_labelled_and_reviewed(models):
    llms = models(compat=[strategist(plan(params={"weights": [1, 1, 100]}))], critic=[critic(p1=None)])
    events = []
    out = compat_agent.propose_conversions(spec_of(THREE_IN_TWO_OUT), clarification_record="user wants u3 spared",
                                           on_event=events.append)
    assert out["status"] == "needs_choice" and out["source"] == "llm"
    assert out["recommended"] == "p1" and out["llm_calls"] == 2
    opt = out["options"][0]
    assert opt["type"] == "T1" and opt["params"]["weights"] == [1.0, 1.0, 100.0]
    assert opt["label"] == "C" and opt["task_changed"] is False and opt["vetoed"] is False
    assert out["usage"]["total_tokens"] == 300
    assert set(out["usage_by_role"]) == {"compat", "compat_critic"}
    assert out["critic"] == {"status": "ok", "vetoes": 0, "dropped": []}
    assert [e["kind"] for e in events] == ["note"] * len(events)
    assert any("does not fit" in e["text"] for e in events)

    system_msg, user_msg = llms["compat"].calls[0]
    assert system_msg["role"] == "system" and "T1" in system_msg["content"]
    evidence = json.loads(user_msg["content"])
    assert [m["type"] for m in evidence["menu"]] == ["T1"]
    assert evidence["user_conversation"] == "user wants u3 spared"
    assert evidence["plant"]["inputs"] == ["u1", "u2", "u3"]
    json.dumps(out)


def test_python_rejects_a_plan_the_menu_does_not_allow_and_the_model_repairs_it(models):
    bad = strategist(plan(ptype="T1", params={"weights": [1, 1]}))
    good = strategist(plan(params={"weights": [1, 1, 1]}))
    llms = models(compat=[bad, good], critic=[critic(p1=None)])
    out = compat_agent.propose_conversions(spec_of(THREE_IN_TWO_OUT))
    assert out["status"] == "needs_choice" and out["source"] == "llm"
    assert out["llm_calls"] == 3
    repair = llms["compat"].calls[1]
    assert repair[-2]["role"] == "assistant"
    assert "weights needs 3" in repair[-1]["content"]


def test_unreadable_reply_counts_as_a_failed_round(models):
    llms = models(compat=["not json at all", strategist(plan(params={"weights": [1, 1, 1]}))],
                  critic=[critic(p1=None)])
    out = compat_agent.propose_conversions(spec_of(THREE_IN_TWO_OUT))
    assert out["source"] == "llm" and out["llm_calls"] == 3
    assert "could not be read as JSON" in llms["compat"].calls[1][-1]["content"]


def test_unsupported_plan_types_are_dropped_with_a_note(models):
    reply = strategist(plan("a", "T9"), plan("b", "T1", {"weights": [1, 1, 1]}))
    models(compat=[reply], critic=[critic(b=None)])
    out = compat_agent.propose_conversions(spec_of(THREE_IN_TWO_OUT))
    assert [o["id"] for o in out["options"]] == ["b"]
    assert any("'T9'" in n for n in out["notes"])


def test_fixed_rules_take_over_after_the_repair_rounds_run_out(models):
    bad = strategist(plan(params={"weights": [1]}))
    llms = models(compat=[bad, bad, bad])
    out = compat_agent.propose_conversions(spec_of(THREE_IN_TWO_OUT))
    assert out["status"] == "needs_choice" and out["source"] == "rules"
    assert out["llm_calls"] == 3 and out["critic"] is None
    assert out["options"][0]["id"] == "r1" and out["options"][0]["type"] == "T1"
    assert "fixed rules" in out["message"] and any("fixed rules" in n for n in out["notes"])
    assert len(llms["compat"].calls) == 3


def test_model_outage_falls_back_to_rules_and_says_so(models):
    models(compat=[RuntimeError("boom")])
    out = compat_agent.propose_conversions(spec_of(THREE_IN_TWO_OUT))
    assert out["source"] == "rules" and "boom" in out["llm_error"]
    assert out["status"] == "needs_choice" and out["llm_calls"] == 1


def test_wrong_conversion_for_the_plant_is_repaired_to_the_right_one(models):
    wrong = strategist(plan(ptype="T1", params={"weights": [1]}))
    right = strategist(plan(ptype="T2", params={"outputs": ["x2"]}))
    llms = models(compat=[wrong, right], critic=[critic(p1=None)])
    out = compat_agent.propose_conversions(spec_of(UNSTABLE_PICK))
    assert out["options"][0]["type"] == "T2" and out["options"][0]["task_changed"] is True
    assert out["options"][0]["untracked_outputs"] == ["x1"]
    assert "more inputs than outputs" in llms["compat"].calls[1][-1]["content"]


def test_critic_veto_flags_the_option_and_moves_the_recommendation(models):
    reply = strategist(plan("p1", "T2", {"outputs": ["x1"]}), plan("p2", "T2", {"outputs": ["x2"]}),
                       recommended="p1")
    models(compat=[reply], critic=[critic(p1="The user asked to track x1.", p2=None)])
    out = compat_agent.propose_conversions(spec_of(UNSTABLE_PICK))
    assert out["recommended"] == "p2"
    assert [o["id"] for o in out["options"]] == ["p2", "p1"]
    vetoed = out["options"][1]
    assert vetoed["vetoed"] and vetoed["veto_reason"] == "The user asked to track x1."
    assert out["critic"] == {"status": "ok", "vetoes": 1, "dropped": []}


def test_everything_vetoed_leaves_no_recommendation(models):
    models(compat=[strategist(plan("p1", "T2", {"outputs": ["x2"]}))], critic=[critic(p1="no")])
    out = compat_agent.propose_conversions(spec_of(UNSTABLE_PICK))
    assert out["recommended"] is None and out["options"][0]["vetoed"] is True
    assert "vetoed by the reviewer" in out["message"]


def test_critic_outage_leaves_the_options_untouched(models):
    models(compat=[strategist(plan(params={"weights": [1, 1, 1]}))], critic=[RuntimeError("down")])
    out = compat_agent.propose_conversions(spec_of(THREE_IN_TWO_OUT))
    assert out["critic"]["status"] == "unavailable" and "down" in out["critic"]["error"]
    assert out["recommended"] == "p1" and out["options"][0]["vetoed"] is False


def test_veto_without_a_reason_is_ignored(models):
    reply = {"verdicts": [{"id": "p1", "verdict": "veto", "reason": ""}]}
    models(compat=[strategist(plan(params={"weights": [1, 1, 1]}))], critic=[reply])
    out = compat_agent.propose_conversions(spec_of(THREE_IN_TWO_OUT))
    assert out["options"][0]["vetoed"] is False


def _opt(oid, label="A", task=False, vetoed=False):
    return {"id": oid, "label": label, "task_changed": task, "vetoed": vetoed}


def test_ordering_prefers_safe_options_over_the_models_task_changing_pick():
    ranked, best = compat_agent._order([_opt("a", task=False), _opt("b", task=True)], "b")
    assert best == "a" and [o["id"] for o in ranked] == ["a", "b"]


def test_ordering_honors_the_models_pick_when_it_is_safe():
    ranked, best = compat_agent._order([_opt("a", label="A"), _opt("b", label="C")], "b")
    assert best == "b"


def test_ordering_puts_vetoed_last_and_never_recommends_them():
    ranked, best = compat_agent._order([_opt("a", vetoed=True), _opt("b", label="C")], "a")
    assert [o["id"] for o in ranked] == ["b", "a"] and best == "b"


def test_chosen_option_can_be_applied_later(models):
    models(compat=[strategist(plan(params={"weights": [1, 1, 100]}))], critic=[critic(p1=None)])
    spec = spec_of(THREE_IN_TWO_OUT)
    option = compat_agent.propose_conversions(spec)["options"][0]
    direct = apply_input_squaring(spec, weights=[1, 1, 100])
    applied = compat_agent.apply_option(spec, option)
    assert applied["spec"] == direct["spec"]
    with pytest.raises(ValueError, match="unknown conversion"):
        compat_agent.apply_option(spec, {"type": "T7"})


def test_roles_and_prompts_are_registered():
    assert llm_factory._config("compat")["max_tokens"] == 2200
    assert llm_factory._config("compat_critic")["temperature"] == 0
    assert {"compat", "compat_critic"} <= set(llm_factory.resolve_models())
    assert all(t in compat_agent.STRATEGIST_PROMPT for t in ("T0", "T1", "T2"))
    assert "veto" in compat_agent.CRITIC_PROMPT


def test_uncontrollable_output_gets_a_reselection_plan_that_asks_for_a_reference(models):
    llms = models(compat=[strategist(plan("p1", "T3", {"outputs": ["i_L"]}, "i_L is reachable", "v_out is dropped"))],
                  critic=[critic(p1=None)])
    spec = spec_of(UNCONTROLLABLE)
    out = compat_agent.propose_conversions(spec)
    menu = json.loads(llms["compat"].calls[0][1]["content"])["menu"]
    assert [m["type"] for m in menu] == ["T3"] and menu[0]["choices"][0]["outputs"] == ["i_L"]
    opt = out["options"][0]
    assert opt["type"] == "T3" and opt["needs_references"] == ["i_L"] and opt["task_changed"] is True
    applied = compat_agent.apply_option(spec, opt, references={"i_L": "0.5"})
    assert applied["spec"]["dynamics"]["references"] == [{"output": "i_L", "expr": "0.5"}]


def test_singular_decoupling_matrix_gets_an_extension_plan(models):
    llms = models(compat=[strategist(plan("p1", "T4", {"inputs": ["v"], "initial_inputs": {"v": 2}}))],
                  critic=[critic(p1=None)])
    out = compat_agent.propose_conversions(spec_of(UNICYCLE))
    menu = json.loads(llms["compat"].calls[0][1]["content"])["menu"]
    assert [m["type"] for m in menu] == ["T4"] and menu[0]["choices"][0]["inputs"] == ["v"]
    opt = out["options"][0]
    assert opt["type"] == "T4" and opt["params"] == {"inputs": ["v"], "initial_inputs": {"v": 2.0}}
    assert opt["assumed_initial_inputs"] == [] and opt["label"] == "C"


def test_fixed_rules_cover_reselection_and_extension(models):
    models(compat=[RuntimeError("down")])
    out = compat_agent.propose_conversions(spec_of(UNCONTROLLABLE))
    assert out["source"] == "rules" and out["options"][0]["id"] == "r3"
    assert out["options"][0]["params"] == {"outputs": ["i_L"]}
    models(compat=[RuntimeError("down")])
    out = compat_agent.propose_conversions(spec_of(UNICYCLE))
    assert out["options"][0]["id"] == "r4" and out["options"][0]["params"]["inputs"] == ["v"]
    assert out["options"][0]["assumed_initial_inputs"] == ["v"]


def test_unstable_hidden_dynamics_are_only_offered_when_asked(models):
    spec = spec_of(UNSTABLE_ZERO)
    default = compat_agent.propose_conversions(spec)
    assert default["status"] == "fits" and default["llm_calls"] == 0 and default["plant_fits"] is True

    models(compat=[strategist(plan("p1", "T3", {"outputs": ["x2"]}))], critic=[critic(p1=None)])
    out = compat_agent.propose_conversions(spec, offer_for_warnings=True)
    assert out["status"] == "needs_choice" and out["plant_fits"] is True
    assert out["options"][0]["type"] == "T3" and out["options"][0]["label"] == "A"
    assert "already fits" in out["message"]


EXP2 = dyn(["x1", "x2"], ["-x1 + exp(x2)", "u"], ["u"], ["x1"], [0.0, 0.0])


def test_chain_coordinates_are_offered_only_when_backstepping_is_wanted(models):
    spec = spec_of(EXP2)
    default = compat_agent.propose_conversions(spec)
    assert default["status"] == "fits" and default["llm_calls"] == 0

    models(compat=[strategist(plan("p1", "T5"))], critic=[critic(p1=None)])
    out = compat_agent.propose_conversions(spec, want_backstepping=True)
    assert out["status"] == "needs_choice" and out["plant_fits"] is True
    opt = out["options"][0]
    assert opt["type"] == "T5" and opt["task_changed"] is False
    applied = compat_agent.apply_option(spec, opt)
    assert applied["spec"]["dynamics"]["states"] == ["x1", "z2"]


def test_fixed_rules_cover_chain_coordinates(models):
    models(compat=[RuntimeError("down")])
    out = compat_agent.propose_conversions(spec_of(EXP2), want_backstepping=True)
    assert out["source"] == "rules" and out["options"][0]["id"] == "r5"


def test_wanting_backstepping_on_a_plant_that_cannot_get_it_is_not_a_rejection(models):
    # relative degree 2 < 3 states: no hint, so no model call and the plant still fits SMC
    from test_plant_analysis import NON_CHAIN
    out = compat_agent.propose_conversions(spec_of(NON_CHAIN), want_backstepping=True)
    assert out["status"] == "fits" and out["llm_calls"] == 0


def test_term_shift_is_experimental_and_only_offered_on_request(models):
    from test_plant_conversion import SMALL_JUMP
    spec = spec_of(SMALL_JUMP)
    assert compat_agent.propose_conversions(spec)["status"] == "fits"
    llms = models(compat=[strategist(plan("p1", "T6"))], critic=[critic(p1=None)])
    out = compat_agent.propose_conversions(spec, want_backstepping=True)
    menu = json.loads(llms["compat"].calls[0][1]["content"])["menu"]
    assert [m["type"] for m in menu] == ["T6"] and menu[0]["experimental"] is True
    assert out["options"][0]["type"] == "T6" and out["options"][0]["label"] == "C"


def test_optional_conversions_are_found_without_any_model(models):
    # no model is installed: asking for one would raise
    kinds = [(h["kind"], h["experimental"]) for h in compat_agent.optional_conversions(spec_of(EXP2))]
    assert kinds == [("backstepping", False)]
    from test_plant_conversion import SMALL_JUMP
    kinds = [(h["kind"], h["experimental"]) for h in compat_agent.optional_conversions(spec_of(SMALL_JUMP))]
    assert kinds == [("backstepping", True)]
    hints = compat_agent.optional_conversions(spec_of(UNSTABLE_ZERO))
    assert [h["kind"] for h in hints] == ["hidden_dynamics"] and "task" in hints[0]["text"]


def test_optional_conversions_are_empty_for_plants_that_need_nothing_or_do_not_fit(models):
    from test_plant_analysis import MAGLEV
    assert compat_agent.optional_conversions(spec_of(MAGLEV)) == []
    assert compat_agent.optional_conversions(spec_of(UNICYCLE)) == []


def test_menu_and_plan_carry_the_number_of_integrators(models):
    from test_plant_conversion import PVTOL
    spec = spec_of(PVTOL)
    spec["dynamics"]["x0"] = [0.0, 0.0, 0.0, 0.0, 0.1, 0.0]
    llms = models(compat=[strategist(plan("p1", "T4", {"inputs": ["T"], "orders": {"T": 2}, "initial_inputs": {"T": 9.81}}))],
                  critic=[critic(p1=None)])
    out = compat_agent.propose_conversions(spec)
    menu = json.loads(llms["compat"].calls[0][1]["content"])["menu"]
    assert menu[0]["type"] == "T4" and menu[0]["choices"][0]["orders"] == {"T": 2}
    opt = out["options"][0]
    assert opt["params"] == {"inputs": ["T"], "initial_inputs": {"T": 9.81}, "orders": {"T": 2}}
    applied = compat_agent.apply_option(spec, opt)
    assert applied["spec"]["dynamics"]["inputs"] == ["T_ddot", "tau"]


def test_fixed_rules_use_the_cheapest_number_of_integrators(models):
    from test_plant_conversion import PVTOL
    models(compat=[RuntimeError("down")])
    out = compat_agent.propose_conversions(spec_of(PVTOL))
    assert out["source"] == "rules" and out["options"][0]["params"]["orders"] == {"T": 2}


def test_when_every_plan_is_vetoed_the_strategist_gets_one_revision_round(models):
    spec = spec_of(THREE_IN_TWO_OUT)
    llms = models(compat=[strategist(plan("p1", "T1", {"weights": [1, 1, 1]})),
                          strategist(plan("p1", "T1", {"weights": [1, 1, 10]}))],
                  critic=[critic(p1="u3 is a small trim motor"), critic(p1=None)])
    out = compat_agent.propose_conversions(spec, clarification_record="u3 is a small trim motor")
    assert out["options"][0]["params"]["weights"] == [1.0, 1.0, 10.0] and not out["options"][0]["vetoed"]
    assert out["recommended"] == "p1" and out["llm_calls"] == 4
    assert any("revised after the reviewer's objection" in n for n in out["notes"])
    retry = json.loads(llms["compat"].calls[1][-1]["content"])
    assert retry["reviewer_objections"] == [{"id": "p1", "reason": "u3 is a small trim motor"}]


def test_a_revision_that_is_vetoed_again_keeps_the_original_options(models):
    spec = spec_of(THREE_IN_TWO_OUT)
    models(compat=[strategist(plan("p1", "T1", {"weights": [1, 1, 1]})),
                   strategist(plan("p1", "T1", {"weights": [1, 1, 2]}))],
           critic=[critic(p1="no"), critic(p1="still no")])
    out = compat_agent.propose_conversions(spec)
    assert out["options"][0]["params"]["weights"] == [1.0, 1.0, 1.0] and out["options"][0]["vetoed"]
    assert out["recommended"] is None and any("vetoed it too" in n for n in out["notes"])


def test_a_failed_revision_call_does_not_break_the_result(models):
    spec = spec_of(THREE_IN_TWO_OUT)
    models(compat=[strategist(plan("p1", "T1", {"weights": [1, 1, 1]})), RuntimeError("down")],
           critic=[critic(p1="no")])
    out = compat_agent.propose_conversions(spec)
    assert out["status"] == "needs_choice" and out["options"][0]["vetoed"]
    assert any("revision after the reviewer's objection failed" in n for n in out["notes"])


def test_t1_reports_how_much_each_input_carries_of_each_command():
    from backend_core.AgentAdaptive.controller.plant_conversion import apply_input_squaring
    from test_plant_analysis import THREE_IN_TWO_OUT
    out = apply_input_squaring(spec_of(THREE_IN_TWO_OUT), weights=[1, 1, 10])
    shares = out["record"]["details"]["effort_share"]
    assert shares["v1"] == {"u1": 100.0} and shares["v2"] == {"u2": 90.9, "u3": 9.1}
    assert "v2: u2 90.9%, u3 9.1%" in out["record"]["changes"][0]
    opt = compat_agent._option(plan("p1", "T1"), out)
    assert opt["effort_share"] == shares and opt["limits"] == ["Actuator limits are not handled."]


def test_a_veto_that_names_things_the_facts_do_not_contain_is_discarded(models):
    spec = spec_of(THREE_IN_TWO_OUT)
    models(compat=[strategist(plan("p1", "T1", {"weights": [1, 1, 10]}))],
           critic=[critic(p1="u3 gets 70 percent of v7, far too much")])
    out = compat_agent.propose_conversions(spec)
    assert out["options"][0]["vetoed"] is False and out["recommended"] == "p1"
    assert out["critic"]["vetoes"] == 0 and out["critic"]["dropped"][0]["id"] == "p1"


def test_the_critic_and_the_explainer_see_the_facts_not_the_strategists_reasoning(models):
    spec = spec_of(THREE_IN_TWO_OUT)
    llms = models(compat=[strategist(plan("p1", "T1", {"weights": [1, 1, 10]}, rationale="MY-OWN-RATIONALE",
                                          risks="MY-OWN-RISKS"))], critic=[critic(p1=None)])
    out = compat_agent.propose_conversions(spec)
    critic_view = llms["compat_critic"].calls[0][1]["content"]
    assert "effort_share" in critic_view and "Actuator limits" in critic_view
    llms = models(compat_explainer=[{"headline": "", "why_not_fit": [], "options": [], "recommendation": ""}])
    compat_agent.explain(out, spec)
    seen = llms["compat_explainer"].calls[0][1]["content"]
    assert "effort_share" in seen and "MY-OWN-RATIONALE" not in seen and "MY-OWN-RISKS" not in seen
    assert "MY-OWN-RATIONALE" in out["options"][0]["rationale"]
