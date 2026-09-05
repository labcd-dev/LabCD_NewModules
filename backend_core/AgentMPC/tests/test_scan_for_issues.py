"""
Run with: pytest backend_core/AgentMPC/tests/test_scan_for_issues.py -v

Regression guard for the "Diagnostics tab says 'No issues detected' even
though iterations failed" bug: scan_for_issues() was checking a failed
row's error text under the key "eval_error", but agent_mpc_app.py's
row-building code (both the "eval_error present" branch and the "empty
metrics" branch) stores it under the row-local key "error" -- no row has
ever had an "eval_error" key, so the dynamics_crash category was dead for
every real run, no matter how many iterations failed.

Found live: a real tuning run with 6 failed iterations (Np exceeding the
reference trajectory length after the Actor proposed a large dt_mpc) still
showed "No issues detected by the automatic scan" in the Diagnostics tab.
"""

from backend_core.AgentMPC.agents.diagnostics_agent import scan_for_issues

OK_ROW = {
    "iteration": 1, "ok": True, "unstable": False, "error": None,
    "mse": 0.02, "solver_diagnostics": {"solved": 100, "solved_inaccurate": 0, "other": 0},
}


def _failed_row(iteration: int, error: str) -> dict:
    """Matches the ACTUAL shape agent_mpc_app.py's row-building code
    produces for a failed iteration -- "error", not "eval_error"."""
    return {
        "iteration": iteration, "ok": False, "unstable": False, "error": error,
        "traceback": None, "mse": None,
    }


def test_failed_iterations_are_detected_as_dynamics_crash():
    rows = [_failed_row(1, "No simulation steps to run: prediction horizon Np=18 is >= ...")]
    findings = scan_for_issues(logs=[], results_data=rows, last_outputs={})
    assert "dynamics_crash" in findings
    assert findings["dynamics_crash"]["count"] == 1
    assert findings["dynamics_crash"]["iterations"] == [1]
    assert "prediction horizon" in findings["dynamics_crash"]["examples"][0]


def test_a_clean_run_reports_no_dynamics_crash():
    findings = scan_for_issues(logs=[], results_data=[OK_ROW, OK_ROW], last_outputs={})
    assert "dynamics_crash" not in findings


def test_multiple_failed_iterations_are_all_counted():
    rows = [_failed_row(1, "boom one"), OK_ROW, _failed_row(3, "boom two"), _failed_row(4, "boom three")]
    findings = scan_for_issues(logs=[], results_data=rows, last_outputs={})
    assert findings["dynamics_crash"]["count"] == 3
    assert findings["dynamics_crash"]["iterations"] == [1, 3, 4]


def test_an_ok_row_with_a_stray_error_field_is_not_double_counted():
    """ok=True rows always carry error=None (see the row-building code) --
    guard against a future regression where a truthy `error` on an OK row
    would get miscounted as a crash."""
    row = {**OK_ROW, "error": "should be ignored, ok=True"}
    findings = scan_for_issues(logs=[], results_data=[row], last_outputs={})
    assert "dynamics_crash" not in findings


def test_frequent_instability_still_detected():
    rows = [{"iteration": i, "ok": True, "unstable": True, "error": None} for i in range(1, 4)]
    rows += [{"iteration": 4, "ok": True, "unstable": False, "error": None}]
    findings = scan_for_issues(logs=[], results_data=rows, last_outputs={})
    assert findings["frequent_instability"]["count"] == 3


def test_solver_struggles_still_detected():
    rows = [
        {"iteration": i, "ok": True, "unstable": False, "error": None,
         "solver_diagnostics": {"solved": 1, "solved_inaccurate": 3, "other": 0}}
        for i in range(1, 4)
    ]
    findings = scan_for_issues(logs=[], results_data=rows, last_outputs={})
    assert "solver_struggles" in findings


def test_text_pattern_categories_still_scan_logs():
    logs = [{"time": "10:00:00", "message": "Error: rate limit exceeded, please retry"}]
    findings = scan_for_issues(logs=logs, results_data=[], last_outputs={})
    assert "rate_limit" in findings


def test_empty_run_reports_nothing():
    assert scan_for_issues(logs=[], results_data=[], last_outputs={}) == {}
