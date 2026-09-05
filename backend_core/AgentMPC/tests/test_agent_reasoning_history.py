"""
Run with: pytest backend_core/AgentMPC/tests/test_agent_reasoning_history.py -v

Regression guard for the "Agent Reasoning panel doesn't show what the agent
actually decided" report: every node's ``state["history"]`` entry -- the
string the Streamlit UI's Agent Reasoning panel renders verbatim -- used to
carry only prose, truncated (200 chars for Actor, 150 for Juror/Scenarist)
and with no numbers at all: an Actor entry never showed the Np/Nc/Q/R/P it
had just proposed, a Scenarist entry never showed the initial_state/
target_state it had just designed, and a Critic entry never showed the
strategy label or suggested_multipliers -- only the free-text explanation.

``history`` is never read back into any prompt (grep the package: every
other node only ever *appends* to it), so enriching it here cannot change
tuning behaviour or prompt size -- confirmed by asserting each node's other
returned fields (current_params, critic_feedback, ...) are untouched by
these tests.

The LLM itself is stubbed (get_llm().with_structured_output(...).invoke(...)
returns a fixed Pydantic instance), so no API key is needed.
"""

from unittest.mock import MagicMock

import pytest

from backend_core.AgentMPC.agents import actor, critic, juror, scenarist, terminator


def _stub_llm(monkeypatch, module, response):
    """Patches module.get_llm so `.with_structured_output(x).invoke(...)`
    returns `response` regardless of the prompt text."""
    structured = MagicMock()
    structured.invoke.return_value = response
    client = MagicMock()
    client.with_structured_output.return_value = structured
    monkeypatch.setattr(module, "get_llm", lambda: client)
    return structured


BASE_STATE = {
    "system_name": "InvertedPendulum", "n_states": 4, "n_inputs": 1,
    "state_names": ["x", "x_dot", "theta", "theta_dot"], "input_names": ["F"],
    "current_params": {"Np": 10, "Nc": 4, "Q": [1, 1, 10, 1], "R": [0.1], "P": [1, 1, 10, 1], "dt": 0.02},
    "current_dt": 0.02, "critic_feedback": "(first iteration)", "strategy": "explore",
    "best_mse": 0.05, "current_mse": 0.08, "current_per_state_mse": {}, "current_per_state_ise": {},
    "current_unstable": False, "exploration_intensity": 50, "iteration": 3, "max_iterations": 10,
    "history": [], "mse_history": [0.2, 0.1, 0.08], "min_explore_iterations": 1,
}


# --------------------------------------------------------------------------
# Actor
# --------------------------------------------------------------------------

def test_actor_history_shows_full_reasoning_and_numeric_params(monkeypatch):
    proposal = actor.MPCParameters(
        reasoning="A" * 400,  # longer than the old 200-char cutoff
        strategy="exploit", Np=12, Nc=4, Q=[10.0, 1.0, 100.0, 1.0], R=[0.3], P=[10.0, 1.0, 100.0, 1.0], dt=None,
    )
    _stub_llm(monkeypatch, actor, proposal)
    result = actor.actor_node(dict(BASE_STATE), cfg=None)
    entry = result["history"][-1]

    assert entry.startswith("[Actor] strategy=exploit")
    assert "Np=12" in entry and "Nc=4" in entry
    assert "Q=[10.0, 1.0, 100.0, 1.0]" in entry
    assert "R=[0.3]" in entry
    assert "A" * 400 in entry, "reasoning must not be truncated"
    # the field OTHER agents read (Actor doesn't feed its own reasoning back
    # anywhere, but current_params must stay the clean dict, not the display text)
    assert result["current_params"]["Np"] == 12


def test_actor_failure_path_shows_the_kept_parameters(monkeypatch):
    structured = _stub_llm(monkeypatch, actor, None)
    structured.invoke.side_effect = RuntimeError("simulated provider error")
    result = actor.actor_node(dict(BASE_STATE), cfg=None)
    entry = result["history"][-1]
    assert "FAILED" in entry
    assert "Np=10" in entry  # kept the previous params, and shows them
    assert result["strategy"] == "exploit"


# --------------------------------------------------------------------------
# Critic
# --------------------------------------------------------------------------

def test_critic_history_shows_strategy_label_and_multipliers(monkeypatch):
    feedback = critic.CriticFeedback(
        feedback="B" * 300, strategy_recommendation="aggressive_explore",
        suggested_multipliers={"Q_2": 1.5, "R_0": 0.8},
    )
    _stub_llm(monkeypatch, critic, feedback)
    state = {**BASE_STATE, "iteration": 5, "min_explore_iterations": 1}
    result = critic.critic_node(state)
    entry = result["history"][-1]

    assert "strategy=aggressive_explore" in entry
    assert "Suggested multipliers" in entry and "Q_2" in entry
    assert "B" * 300 in entry
    # critic_feedback (read by the Actor's own prompt) must stay clean prose,
    # not the enriched multi-line display text
    assert result["critic_feedback"] == feedback.feedback


def test_critic_history_notes_when_a_guard_overrides_the_llms_recommendation(monkeypatch):
    feedback = critic.CriticFeedback(feedback="looks fine", strategy_recommendation="exploit")
    _stub_llm(monkeypatch, critic, feedback)
    # min_explore_iterations=10 with iteration=1 forces the "still exploring" guard
    state = {**BASE_STATE, "iteration": 1, "min_explore_iterations": 10}
    result = critic.critic_node(state)
    entry = result["history"][-1]
    assert "strategy=explore" in entry
    assert "LLM recommended: exploit" in entry


# --------------------------------------------------------------------------
# Terminator
# --------------------------------------------------------------------------

def test_terminator_history_shows_the_numbers_it_reasoned_over(monkeypatch):
    decision = terminator.TerminationDecision(decision="juror", reason="C" * 300)
    _stub_llm(monkeypatch, terminator, decision)
    # round_floats defaults to 2 decimal places -- values chosen to round cleanly
    state = {**BASE_STATE, "iteration": 4, "max_iterations": 10, "current_mse": 0.03, "best_mse": 0.02}
    result = terminator.should_continue(state)
    entry = result["history"][-1]

    assert "decision=juror" in entry
    assert "iteration=4/10" in entry
    assert "current_mse=0.03" in entry
    assert "best_mse=0.02" in entry
    assert "C" * 300 in entry


def test_terminator_numeric_guard_path_names_the_actual_numbers(monkeypatch):
    state = {**BASE_STATE, "iteration": 10, "max_iterations": 10}
    result = terminator.should_continue(state)  # no LLM call on this path at all
    entry = result["history"][-1]
    assert "decision=juror" in entry
    assert "iteration=10/10" in entry


# --------------------------------------------------------------------------
# Juror
# --------------------------------------------------------------------------

def test_juror_history_is_not_truncated(monkeypatch):
    verdict = juror.JurorVerdict(verdict="retry_with_wider_search", explanation="D" * 250)
    _stub_llm(monkeypatch, juror, verdict)
    state = {**BASE_STATE, "iteration": 3, "max_iterations": 10, "best_params": {"Np": 8}}
    result = juror.juror_node(state)
    entry = result["history"][-1]

    assert "verdict=retry_with_wider_search" in entry
    assert "iteration=3/10" in entry
    assert "D" * 250 in entry, "explanation must not be cut to 150 characters"


def test_juror_budget_exhausted_override_is_visible_untruncated(monkeypatch):
    verdict = juror.JurorVerdict(verdict="retry_with_wider_search", explanation="E" * 250)
    _stub_llm(monkeypatch, juror, verdict)
    state = {**BASE_STATE, "iteration": 10, "max_iterations": 10, "best_params": {"Np": 8}}
    result = juror.juror_node(state)
    entry = result["history"][-1]
    assert "verdict=accept_and_end" in entry  # overridden -- budget exhausted
    assert "budget exhausted" in entry
    assert "E" * 250 in entry


# --------------------------------------------------------------------------
# Scenarist
# --------------------------------------------------------------------------

def test_scenarist_history_shows_the_numeric_state_vectors(monkeypatch):
    scenario = scenarist.Scenario(
        level="II", initial_state=[0.3, 0.0, -0.2, 0.1], target_state=[0.0, 0.0, 0.0, 0.0],
        rationale="F" * 250,
    )
    _stub_llm(monkeypatch, scenarist, scenario)
    state = {**BASE_STATE, "default_initial_state": [0.0, 0.0, 0.0, 0.0], "default_target": [0.0, 0.0, 0.0, 0.0]}
    result = scenarist.scenarist_node(state, default_initial_state=[0.0, 0.0, 0.0, 0.0],
                                      default_target=[0.0, 0.0, 0.0, 0.0])
    entry = result["history"][-1]

    assert "level=II" in entry
    assert "initial_state=[0.3, 0.0, -0.2, 0.1]" in entry
    assert "target_state=[0.0, 0.0, 0.0, 0.0]" in entry
    assert "F" * 250 in entry, "rationale must not be cut to 150 characters"


def test_scenarist_failure_path_shows_the_nominal_defaults(monkeypatch):
    structured = _stub_llm(monkeypatch, scenarist, None)
    structured.invoke.side_effect = RuntimeError("simulated provider error")
    result = scenarist.scenarist_node(dict(BASE_STATE), default_initial_state=[1.0, 0.0, 0.0, 0.0],
                                      default_target=[0.0, 0.0, 0.0, 0.0])
    entry = result["history"][-1]
    assert "level=I" in entry
    assert "initial_state=[1.0, 0.0, 0.0, 0.0]" in entry


# --------------------------------------------------------------------------
# history is display-only: never fed back into a prompt
# --------------------------------------------------------------------------

def test_history_is_never_read_back_into_a_prompt():
    """If some future change starts reading state["history"] back into a
    prompt, the enriched multi-line entries this module adds would bloat
    every subsequent prompt. Checks every ``*_prompt.format(...)`` call site
    in each file: none may pass a keyword argument literally named
    "history" (a plain ``history = history + [...]`` assignment, which every
    node does, is a Name/BinOp -- not a call keyword -- so it does not
    trigger this)."""
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "agents"
    for name in ("actor.py", "critic.py", "terminator.py", "juror.py", "scenarist.py"):
        src = (root / name).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "format"):
                continue
            bad = [kw.arg for kw in node.keywords if kw.arg == "history"]
            assert not bad, f"{name}:{node.lineno} passes history= into a .format(...) call"
