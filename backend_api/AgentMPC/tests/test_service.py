"""Service adapter tests (mocked graph, no live LLM)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from backend_api.AgentMPC.job_store import InMemoryJobStore
from backend_api.AgentMPC.schemas import MPCDynamicsInput, MPCJobCreateRequest, MPCJobOptions
from backend_api.AgentMPC.service import (
    cancel_job,
    get_job,
    get_results,
    list_jobs,
    submit_job,
    _resolve_plugin_path,
)


def test_resolve_default_plugin():
    path = _resolve_plugin_path(None)
    assert path.endswith("example_pendulum.py")


def test_resolve_plugin_id():
    path = _resolve_plugin_path(MPCDynamicsInput(plugin_id="example_pendulum"))
    assert path.endswith("example_pendulum.py")


def test_submit_runs_graph_mocked():
    """Drive submit → worker with core graph/loader patched at import sites."""
    store = InMemoryJobStore()
    fake_final = {
        "best_params": {"Np": 12, "Nc": 4},
        "best_mse": 0.01,
        "iteration": 3,
        "termination_reason": "max_iterations",
        "mse_history": [0.1, 0.05, 0.01],
        "params_history": [{"Np": 10}, {"Np": 12}],
        "history": [],
        "current_params": {"Np": 12},
    }

    mock_dynamics = MagicMock()
    mock_dynamics.n_states = 2
    mock_dynamics.n_inputs = 1
    mock_dynamics.state_names = ["th", "w"]
    mock_dynamics.input_names = ["u"]

    mock_plugin = MagicMock()
    mock_plugin.create_dynamics.return_value = mock_dynamics
    mock_plugin.source_name = "example_pendulum.py"

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = fake_final

    mock_cfg = MagicMock()
    mock_cfg.mpc = MagicMock()
    mock_cfg.data = MagicMock()

    # Stub langgraph-dependent modules before the worker imports them.
    import sys
    import types

    fake_loader_mod = types.ModuleType("backend_core.AgentMPC.dynamics.loader")
    fake_loader_mod.DynamicLoader = MagicMock()
    fake_loader_mod.DynamicLoader.load_from_path = MagicMock(return_value=mock_plugin)

    fake_wf = types.ModuleType("backend_core.AgentMPC.graph.workflow")
    fake_wf.build_ui_tuning_graph = MagicMock(return_value=mock_graph)
    fake_wf.build_mpc_tuning_graph = MagicMock(return_value=mock_graph)
    fake_wf.initial_state = MagicMock(return_value={"iteration": 0})

    fake_cfg_mod = types.ModuleType("backend_core.AgentMPC.mpc.config")
    fake_cfg_mod.Config = MagicMock(return_value=mock_cfg)

    with patch.dict(
        sys.modules,
        {
            "backend_core.AgentMPC.dynamics.loader": fake_loader_mod,
            "backend_core.AgentMPC.graph.workflow": fake_wf,
            "backend_core.AgentMPC.mpc.config": fake_cfg_mod,
        },
    ):
        with patch(
            "backend_api.AgentMPC.service._resolve_plugin_path",
            return_value="/fake/example_pendulum.py",
        ):
            with patch(
                "backend_api.AgentMPC.service._start_tuning_async",
                side_effect=lambda job_id, s: __import__(
                    "backend_api.AgentMPC.service", fromlist=["_run_tuning_thread"]
                )._run_tuning_thread(job_id, s),
            ):
                resp = submit_job(
                    MPCJobCreateRequest(
                        dynamics=MPCDynamicsInput(plugin_id="example_pendulum"),
                        options=MPCJobOptions(max_iterations=5, use_ui_graph=True),
                    ),
                    store=store,
                )

    assert resp.job_id
    status = get_job(resp.job_id, store=store)
    assert status.status == "completed"
    assert status.stage == "done"

    results = get_results(resp.job_id, store=store)
    assert results.best_params == {"Np": 12, "Nc": 4}
    assert results.best_mse == 0.01
    assert results.iteration == 3
    assert results.termination_reason == "max_iterations"

    mock_graph.invoke.assert_called_once()


def test_cancel_job():
    store = InMemoryJobStore()
    rec = store.create(dynamics_ref={}, options={}, system_name="x")
    store.update(rec.job_id, status="running", stage="actor")
    out = cancel_job(rec.job_id, store=store)
    assert out.status == "cancelled"
    assert store.is_cancel_requested(rec.job_id)


def test_list_jobs():
    store = InMemoryJobStore()
    store.create(dynamics_ref={}, options={}, user_id=7, system_name="a")
    store.create(dynamics_ref={}, options={}, user_id=8, system_name="b")
    assert len(list_jobs(store=store)) == 2
    assert len(list_jobs(user_id=7, store=store)) == 1
