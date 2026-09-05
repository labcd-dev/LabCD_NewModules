"""
Run with: pytest backend_core/AgentMPC/tests/test_workflow_recursion.py -v

Regression guard for the GraphRecursionError that ended tuning runs at
iteration six.

The tuning graph contains cycles (critic -> actor, juror -> actor), so LangGraph
applies a superstep budget. That budget was left at LangGraph's default of 25
while one iteration costs four supersteps, which capped every run at six
iterations -- below the UI's default of ten and far below its maximum of thirty.
The run then died with GraphRecursionError *before* the ``max_iterations`` stop
could fire, and the caller discarded six iterations of valid results.

These tests use stub nodes that reproduce the real deterministic stop logic
(agents/terminator.py's numeric guard and agents/juror.py's budget override)
against the real graph topology and the real budget formula. No LLM, no API key.
"""

import pytest

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph

from backend_core.AgentMPC.graph import workflow as wf
from backend_core.AgentMPC.graph.state import MPCGraphState

# The full range of the UI's "Max Iterations" slider (see agent_mpc_app.py).
SLIDER_MIN, SLIDER_DEFAULT, SLIDER_MAX = 3, 10, 30


def _stub_nodes(trace):
    """Nodes that mirror the deterministic parts of the real agents.

    Only the LLM calls are dropped: the Terminator's ``iteration >=
    max_iterations`` numeric guard and the Juror's budget override are the
    stops that actually have to fire, so they are reproduced faithfully.
    """

    def make(name):
        def node(state):
            trace.append(name)
            iteration = state.get("iteration", 0)
            max_iterations = state.get("max_iterations", 20)
            if name == "actor":
                return {**state, "iteration": iteration + 1}
            if name == "terminator":
                # Never ends a run itself; routes to the Juror once the budget
                # is spent, and otherwise keeps tuning via the Critic. Always
                # choosing "critic" here is the worst realistic case: an LLM
                # that never decides on its own that the run is finished.
                nxt = "juror" if iteration >= max_iterations else "critic"
                return {**state, "should_continue": True, "_next": nxt}
            if name == "juror":
                # Budget exhausted -> forced accept_and_end.
                return {**state, "should_continue": not (iteration >= max_iterations)}
            return dict(state)

        return node

    return make


def _build(max_iterations, trace, recursion_limit=None):
    """The real topology from workflow._register_common_nodes, with stub nodes.

    ``recursion_limit=None`` uses the module's own formula -- i.e. what the
    shipped builders configure.
    """
    make = _stub_nodes(trace)
    graph = StateGraph(MPCGraphState)
    for name in ("actor", "evaluator", "terminator", "critic", "juror"):
        graph.add_node(name, make(name))
    graph.add_edge("actor", "evaluator")
    graph.add_edge("evaluator", "terminator")
    graph.add_conditional_edges(
        "terminator", wf._route_after_terminator,
        {"critic": "critic", "juror": "juror", "end": END},
    )
    graph.add_edge("critic", "actor")
    graph.add_conditional_edges("juror", wf._route_after_juror, {"actor": "actor", "end": END})
    graph.set_entry_point("actor")
    limit = wf.recursion_limit_for(max_iterations) if recursion_limit is None else recursion_limit
    return graph.compile().with_config({"recursion_limit": limit})


def _run(max_iterations, recursion_limit=None):
    trace = []
    compiled = _build(max_iterations, trace, recursion_limit)
    state = {"iteration": 0, "max_iterations": max_iterations, "should_continue": True}
    for _ in compiled.stream(state):
        pass
    return trace


# --------------------------------------------------------------------------
# The budget formula
# --------------------------------------------------------------------------

def test_budget_always_exceeds_what_a_full_run_costs():
    """The policy fuse (max_iterations) must fire before the safety fuse."""
    for max_iterations in range(1, 51):
        cost = max_iterations * wf.SUPERSTEPS_PER_ITERATION
        assert wf.recursion_limit_for(max_iterations) > cost


def test_budget_scales_with_the_iteration_cap():
    """A fixed constant was rejected precisely because it does not scale: 100
    would still be too small for a 30-iteration run, which costs 120."""
    assert wf.recursion_limit_for(30) > wf.recursion_limit_for(10)
    assert wf.recursion_limit_for(30) > 30 * wf.SUPERSTEPS_PER_ITERATION


def test_budget_stays_bounded_so_a_runaway_loop_is_still_caught():
    """The safety fuse must remain a fuse -- not effectively infinite."""
    assert wf.recursion_limit_for(10) < 200


def test_zero_or_negative_iteration_caps_do_not_produce_an_unusable_budget():
    for bad in (0, -1, -100):
        assert wf.recursion_limit_for(bad) >= wf.SUPERSTEPS_PER_ITERATION


# --------------------------------------------------------------------------
# End-to-end through the real topology
# --------------------------------------------------------------------------

@pytest.mark.parametrize("max_iterations", [SLIDER_MIN, 5, 6, 7, SLIDER_DEFAULT, 15, 20, SLIDER_MAX])
def test_every_slider_position_runs_to_its_iteration_cap(max_iterations):
    """Before the fix, everything at 7 and above died at iteration six."""
    trace = _run(max_iterations)
    assert trace.count("evaluator") == max_iterations
    assert trace[-1] == "juror"  # the Juror is the only path to END


def test_the_old_default_budget_reproduces_the_reported_failure():
    """Pins the regression itself: with LangGraph's default of 25, a run
    configured for the UI's default of 10 iterations crashes at six."""
    trace = []
    compiled = _build(SLIDER_DEFAULT, trace, recursion_limit=25)
    with pytest.raises(GraphRecursionError):
        for _ in compiled.stream(
            {"iteration": 0, "max_iterations": SLIDER_DEFAULT, "should_continue": True}
        ):
            pass
    assert trace.count("evaluator") == 6


def test_one_iteration_costs_the_documented_number_of_supersteps():
    """SUPERSTEPS_PER_ITERATION is a hand-maintained mirror of the cycle wired
    in _register_common_nodes; this fails if a node is added to that loop
    without updating the constant."""
    trace = _run(5)
    assert len(trace) // trace.count("evaluator") == wf.SUPERSTEPS_PER_ITERATION
    assert trace[:4] == ["actor", "evaluator", "terminator", "critic"]


# --------------------------------------------------------------------------
# The builders wire the budget in
# --------------------------------------------------------------------------

def test_builders_accept_and_apply_max_iterations():
    """Both builders must expose max_iterations -- a caller that omits it gets
    the default budget, which is what left the UI crashing above 20."""
    import inspect

    for builder in (wf.build_ui_tuning_graph, wf.build_mpc_tuning_graph):
        params = inspect.signature(builder).parameters
        assert "max_iterations" in params, builder.__name__
