"""
Run with: pytest backend_core/AgentMPC/tests/test_trajectory_author.py -v

Covers the trajectory-authoring agent (agents/trajectory_author_agent.py),
which turns a plain-language request into a custom reference-trajectory file.

The LLM itself is stubbed, so no API key is needed. What is exercised is
everything around it -- the part that decides whether a generated file is
usable: the deterministic validation of whatever the model returns, the
fallback to the existing repair loop when the first draft doesn't load, and
the derivative-pair context that keeps a position/velocity reference
physically consistent.
"""

import numpy as np
import pytest

from backend_core.AgentMPC.agents import trajectory_author_agent as author
from backend_core.AgentMPC.agents.trajectory_author_agent import (
    AuthoredTrajectory,
    _derivative_context,
    _strip_fences,
    author_trajectory,
)

SUMMARY = {
    "n_states": 4, "n_inputs": 1,
    "state_names": ["theta1", "omega1", "theta2", "omega2"],
    "input_names": ["tau"],
}

GOOD_FILE = '''import numpy as np


def create_trajectory(dt_mpc, simulation_time, n_states, state_names):
    n_steps = int(simulation_time / dt_mpc) + 1
    t = np.linspace(0, simulation_time, n_steps)
    ref = np.zeros((n_steps, n_states))
    amplitude, freq = 0.2, 0.4
    omega = 2 * np.pi * freq
    ref[:, 0] = amplitude * np.sin(omega * t)              # theta1
    if n_states > 1:
        ref[:, 1] = amplitude * omega * np.cos(omega * t)  # omega1 = d(theta1)/dt
    return ref
'''

BROKEN_FILE = "def create_trajectory(:\n    pass\n"


class _Draft:
    def __init__(self, code, explanation="theta1 follows a sine; omega1 is its derivative."):
        self.python_code = code
        self.explanation = explanation


class _StubLLM:
    """Stands in for get_llm().with_structured_output(...)."""

    def __init__(self, draft):
        self._draft = draft
        self.system_prompt = None
        self.messages = None

    def with_structured_output(self, _schema):
        return self

    def invoke(self, messages, **_kwargs):
        self.messages = messages
        self.system_prompt = dict(messages)["system"] if isinstance(messages, dict) else messages[0][1]
        return self._draft


@pytest.fixture
def stub_llm(monkeypatch):
    """Patches get_llm at its import site inside trajectory_author_agent."""

    def install(draft):
        stub = _StubLLM(draft)
        import backend_core.AgentMPC.agents.llm_base as llm_base
        monkeypatch.setattr(llm_base, "get_llm", lambda *a, **k: stub)
        return stub

    return install


# --------------------------------------------------------------------------
# Fence stripping -- the most common way a good file arrives unusable
# --------------------------------------------------------------------------

def test_strip_fences_removes_a_tagged_markdown_block():
    assert _strip_fences("```python\nimport numpy as np\n```") == "import numpy as np"


def test_strip_fences_removes_an_untagged_block():
    assert _strip_fences("```\nx = 1\n```") == "x = 1"


def test_strip_fences_leaves_bare_code_alone():
    assert _strip_fences("import numpy as np\nx = 1") == "import numpy as np\nx = 1"


def test_strip_fences_tolerates_empty_input():
    assert _strip_fences("") == ""
    assert _strip_fences(None) == ""


# --------------------------------------------------------------------------
# Derivative context -- what keeps a velocity reference physically consistent
# --------------------------------------------------------------------------

def test_derivative_context_names_both_states_and_their_indices():
    text = _derivative_context(SUMMARY["state_names"], [(0, 1), (2, 3)])
    assert "omega1 = d(theta1)/dt" in text
    assert "omega2 = d(theta2)/dt" in text
    assert "index 1 is the derivative of index 0" in text


def test_derivative_context_says_so_when_there_are_none():
    text = _derivative_context(SUMMARY["state_names"], [])
    assert "none known" in text


def test_derivative_context_ignores_pairs_pointing_outside_the_state_vector():
    """A stale pair from a previously-loaded plugin must not crash the prompt."""
    text = _derivative_context(["a", "b"], [(0, 1), (5, 9)])
    assert "b = d(a)/dt" in text
    assert "index 9" not in text


# --------------------------------------------------------------------------
# End to end, with the model stubbed
# --------------------------------------------------------------------------

def test_a_valid_draft_is_accepted_and_actually_loads(stub_llm):
    stub_llm(_Draft(GOOD_FILE))
    result = author_trajectory("theta1 sinusoidal, omega1 its cosine", SUMMARY,
                               derivative_pairs=[(0, 1)])
    assert result.valid
    assert not result.was_repaired
    assert "create_trajectory" in result.code

    # the file is genuinely runnable, not just syntactically plausible
    namespace = {}
    exec(compile(result.code, "<authored>", "exec"), namespace)
    ref = namespace["create_trajectory"](0.05, 2.0, 4, SUMMARY["state_names"])
    assert ref.shape[1] == 4
    assert ref.shape[0] >= int(2.0 / 0.05)
    assert np.any(ref[:, 0] != 0)


def test_a_fenced_draft_still_loads(stub_llm):
    """A fence would otherwise be a SyntaxError on line one."""
    stub_llm(_Draft("```python\n" + GOOD_FILE + "```"))
    assert author_trajectory("anything", SUMMARY, derivative_pairs=[(0, 1)]).valid


def test_the_prompt_carries_the_system_and_its_derivative_pairs(stub_llm):
    stub = stub_llm(_Draft(GOOD_FILE))
    author_trajectory("theta1 sinusoidal", SUMMARY, derivative_pairs=[(0, 1)])
    assert "theta1, omega1, theta2, omega2" in stub.system_prompt
    assert "omega1 = d(theta1)/dt" in stub.system_prompt
    # the standard itself must be in the prompt, not just referenced
    assert "create_trajectory(dt_mpc, simulation_time, n_states, state_names)" in stub.system_prompt


def test_an_unloadable_draft_is_not_returned_as_valid(stub_llm, monkeypatch):
    """The verdict comes from the loader, never from the model's confidence."""
    stub_llm(_Draft(BROKEN_FILE))

    # make the repair loop fail too, so this test covers the give-up path only
    import backend_core.AgentMPC.agents.trajectory_validator as tv
    monkeypatch.setattr(tv, "validate_and_fix_trajectory",
                        lambda *a, **k: tv.FixOutcome(valid=False, used_llm_fix=True,
                                                      final_code=BROKEN_FILE,
                                                      still_broken_error="still broken", attempts=2))
    result = author_trajectory("something impossible", SUMMARY)
    assert not result.valid
    assert result.error and "still broken" in result.error


def test_a_broken_draft_that_the_repair_loop_fixes_is_flagged_as_repaired(stub_llm, monkeypatch):
    stub_llm(_Draft(BROKEN_FILE))
    import backend_core.AgentMPC.agents.trajectory_validator as tv
    monkeypatch.setattr(tv, "validate_and_fix_trajectory",
                        lambda *a, **k: tv.FixOutcome(valid=True, used_llm_fix=True,
                                                      final_code=GOOD_FILE,
                                                      explanation="fixed the signature", attempts=1))
    result = author_trajectory("theta1 sinusoidal", SUMMARY)
    assert result.valid
    assert result.was_repaired
    assert result.code == GOOD_FILE


def test_result_type_defaults_are_safe():
    """The UI reads .code/.explanation even on the failure path."""
    empty = AuthoredTrajectory(valid=False)
    assert empty.code == "" and empty.explanation == "" and empty.history == []


# --------------------------------------------------------------------------
# Conversation: revising the file that was already written
# --------------------------------------------------------------------------

def test_a_first_request_sends_no_history_and_no_previous_code(stub_llm):
    stub = stub_llm(_Draft(GOOD_FILE))
    author_trajectory("theta1 sinusoidal", SUMMARY, derivative_pairs=[(0, 1)])
    roles = [role for role, _ in stub.messages]
    assert roles == ["system", "user"]
    assert "Write the trajectory file for this request" in stub.messages[-1][1]


def test_a_revision_feeds_back_the_current_file_and_the_thread(stub_llm):
    """"Make the amplitude smaller" is only answerable against the actual
    file -- so the previous code goes in, and the earlier turns go in with
    it, otherwise every revision would be a blind regeneration."""
    stub = stub_llm(_Draft(GOOD_FILE, explanation="halved the amplitude"))
    history = [
        {"role": "user", "content": "theta1 sinusoidal, amplitude 0.2"},
        {"role": "assistant", "content": "theta1 follows a 0.2 sine; omega1 is its derivative."},
    ]
    author_trajectory("make the amplitude 0.1", SUMMARY, derivative_pairs=[(0, 1)],
                      conversation_history=history, previous_code=GOOD_FILE)

    roles = [role for role, _ in stub.messages]
    assert roles == ["system", "user", "ai", "user"]
    joined = "\n".join(content for _, content in stub.messages)
    assert "theta1 sinusoidal, amplitude 0.2" in joined      # the earlier turn
    assert "amplitude, freq = 0.2, 0.4" in joined            # the file being revised
    assert "make the amplitude 0.1" in joined                # the new instruction
    assert "COMPLETE updated file" in stub.messages[-1][1]   # revision, not fresh brief


def test_a_revision_still_carries_the_derivative_pairs(stub_llm):
    """The structural context has to survive every turn, not just the first --
    a revision that forgets the pairing silently produces a velocity
    reference that isn't the derivative of its position."""
    stub = stub_llm(_Draft(GOOD_FILE))
    author_trajectory("make it slower", SUMMARY, derivative_pairs=[(0, 1), (2, 3)],
                      conversation_history=[{"role": "user", "content": "first ask"}],
                      previous_code=GOOD_FILE)
    assert "omega1 = d(theta1)/dt" in stub.system_prompt
    assert "omega2 = d(theta2)/dt" in stub.system_prompt


def test_assistant_role_is_normalized_for_langchain(stub_llm):
    """LangChain's message parser wants "ai"; the UI stores "assistant"."""
    stub = stub_llm(_Draft(GOOD_FILE))
    author_trajectory("tweak it", SUMMARY,
                      conversation_history=[{"role": "assistant", "content": "prior answer"}],
                      previous_code=GOOD_FILE)
    assert ("ai", "prior answer") in stub.messages


def test_a_revision_is_validated_exactly_like_a_first_draft(stub_llm, monkeypatch):
    """A bad revision must not overwrite a good file unchecked."""
    stub_llm(_Draft(BROKEN_FILE))
    import backend_core.AgentMPC.agents.trajectory_validator as tv
    monkeypatch.setattr(tv, "validate_and_fix_trajectory",
                        lambda *a, **k: tv.FixOutcome(valid=False, used_llm_fix=True,
                                                      final_code=BROKEN_FILE,
                                                      still_broken_error="still broken", attempts=2))
    result = author_trajectory("break it", SUMMARY, conversation_history=[], previous_code=GOOD_FILE)
    assert not result.valid


def test_history_labels_a_revision_as_such(stub_llm):
    stub_llm(_Draft(GOOD_FILE))
    fresh = author_trajectory("first", SUMMARY)
    revised = author_trajectory("second", SUMMARY, previous_code=GOOD_FILE)
    assert "Draft written" in fresh.history[0]
    assert "Revision written" in revised.history[0]
