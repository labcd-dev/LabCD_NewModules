"""Tests for simulation series export + results wiring."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend_core.AgentAdaptive.tools.series_export import (
    SERIES_KEY,
    attach_series,
    build_series,
    extract_series,
)
from backend_api.AgentAdaptive.app import app
from backend_api.AgentAdaptive.job_store import InMemoryJobStore


def test_build_series_shapes_and_names():
    t = np.linspace(0, 1, 11)
    y = np.column_stack([t, t * 2])
    ref = np.ones_like(y)
    u = np.zeros((11, 1))
    x = y.copy()
    series = build_series(
        t,
        y,
        ref,
        u,
        x,
        dt=0.1,
        t_end=1.0,
        output_names=["a", "b"],
        input_names=["u"],
        state_names=["a", "b"],
        max_points=2000,
    )
    assert series["version"] == 1
    assert series["n_points"] == 11
    assert series["downsampled"] is False
    assert series["channels"]["t"]["data"][0] == 0.0
    assert series["channels"]["y"]["names"] == ["a", "b"]
    assert len(series["channels"]["y"]["data"]) == 11
    assert len(series["channels"]["y"]["data"][0]) == 2
    assert series["channels"]["u"]["names"] == ["u"]


def test_build_series_downsample():
    n = 5000
    t = np.linspace(0, 10, n)
    y = np.zeros((n, 1))
    series = build_series(
        t, y, y, y, y, dt=0.002, t_end=10.0, max_points=100
    )
    assert series["downsampled"] is True
    assert series["n_points"] <= 100
    assert series["n_points"] == len(series["channels"]["t"]["data"])


def test_attach_extract():
    components = {"intro": "x"}
    t = np.array([0.0, 1.0])
    y = np.array([[0.0], [1.0]])
    attach_series(
        components, t, y, y, y, y, dt=1.0, t_end=1.0, outputs=["x"], include=True
    )
    assert SERIES_KEY in components
    assert extract_series(components)["n_points"] == 2


def test_results_include_series(client_store=None):
    store = InMemoryJobStore()
    series = build_series(
        np.array([0.0, 0.1]),
        np.array([[0.0], [0.1]]),
        np.array([[1.0], [1.0]]),
        np.array([[0.0], [0.5]]),
        np.array([[0.0], [0.1]]),
        dt=0.1,
        t_end=0.1,
        output_names=["x"],
        input_names=["u"],
        state_names=["x"],
    )
    fake_result = {
        "messages": [SimpleNamespace(content="report")],
        "abstract": "abs",
        "final_metrics": {"success": True},
        "series": series,
    }

    with patch("backend_api.AgentAdaptive.router.default_job_store", store):
        with patch("backend_api.AgentAdaptive.service.default_job_store", store):
            with patch(
                "backend_api.AgentAdaptive.service.run_full_pipeline",
                return_value=(fake_result, {}, [], None),
            ):
                with patch(
                    "backend_api.AgentAdaptive.service._start_pipeline_async",
                    side_effect=lambda job_id, s: __import__(
                        "backend_api.AgentAdaptive.service",
                        fromlist=["_run_pipeline_thread"],
                    )._run_pipeline_thread(job_id, s),
                ):
                    http = TestClient(app)
                    r = http.post(
                        "/api/adaptive/jobs",
                        json={
                            "system_spec": {
                                "status": "complete",
                                "system_name": "s",
                                "dynamics": {
                                    "states": ["x"],
                                    "state_meanings": ["x"],
                                    "inputs": ["u"],
                                    "outputs": ["x"],
                                    "state_equations": ["u"],
                                    "parameters": {},
                                    "system_type": "SISO",
                                    "assumptions": [],
                                    "x0": [0.0],
                                    "references": [],
                                    "uncertainty": [],
                                    "disturbance": [],
                                    "sim_time": 1.0,
                                    "solver_step": 0.1,
                                },
                            },
                            "options": {"skip_clarify": True},
                        },
                    )
                    assert r.status_code == 201, r.text
                    job_id = r.json()["job_id"]
                    results = http.get(f"/api/adaptive/jobs/{job_id}/results")
                    assert results.status_code == 200
                    body = results.json()
                    assert body["series"] is not None
                    assert body["series"]["n_points"] == 2
                    assert "y" in body["series"]["channels"]
