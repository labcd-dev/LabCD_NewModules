"""
Run with: pytest backend_core/AgentMPC/tests/test_report_pdf.py -v

Covers the PDF report composition, which delegates layout to ``labcd_pdfmaker``
and keeps only MPC-specific knowhow here. No API key and no LaTeX install
needed: the default backend is reportlab.

What these guard, specifically:
  * the two call-site shapes used by the Streamlit app (one all-keyword, one
    positional) keep working -- the signature is public API in practice;
  * conditional sections stay contiguously numbered, since a run with no
    failures and no tracker has fewer sections than one with both;
  * "this number does not apply here" stays distinguishable from "this
    iteration produced nothing at all" -- the part of the old hand-built
    tables that was easiest to lose;
  * the app's dark-themed charts come out light on paper, and the caller's
    Figure object is handed back untouched;
  * a run where nothing succeeded still produces a report rather than raising.
"""

import io
import os
import re
import tempfile

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from labcd_pdfmaker import Backend  # noqa: E402

from backend_core.AgentMPC.agents.report_pdf import (  # noqa: E402
    _bounds_rows,
    _executive_summary,
    _failure_rows,
    _figure_png,
    _history_rows,
    _improvement_diff,
    _metric_rows,
    _param_rows,
    _per_state_rows,
    _physical_param_rows,
    _plant_rows,
    _progression_rows,
    _usage_rows,
    _verdict,
    build_pdf_report,
)

DARK = "#0a0e1a"


class _Analysis:
    """Stands in for report_agent.ReportAnalysis (all seven text fields)."""

    system_analysis = "System analysis text."
    search_process_analysis = "Search process text."
    controller_analysis = "Controller analysis text."
    results_analysis = "Results analysis text."
    theoretical_context = "Theoretical context text."
    recommendations = "Recommendations text."
    conclusion = "Conclusion text."


SUMMARY = {
    "source_file": "example_pendulum.py", "dynamics_class": "InvertedPendulum",
    "n_states": 4, "n_inputs": 1,
    "state_names": ["x", "x_dot", "theta", "theta_dot"], "input_names": ["F"],
    "params": {"m_c": 1.0, "m_p": 0.1, "l": 0.5, "g": 9.81},
    "input_bounds": ([-10.0], [10.0]),
    "state_bounds": ([-2.4, -5.0], [2.4, 5.0]),
}

BEST_ROW = {
    "iteration": 4, "np": 12, "nc": 4, "Q_formatted": "[10, 1]", "R_formatted": "[0.1]",
    "P_formatted": "[10, 1]", "dt_mpc": 0.05, "mse": 0.0121, "overshoot": 0.034,
    "overshoot_meaningful": True, "settling": 2.4, "effort": 18.7, "iae": 4.2, "ise": 0.51,
    "oscillation_count": 2, "is_stable": True, "strategy": "exploit",
    "per_state_mse": {"x": 0.004, "theta": 0.0038},
    "per_state_overshoot": {"x": 0.02, "theta": 0.034},
}

RESULTS = [
    {"iteration": 1, "ok": True, "unstable": False, "np": 8, "nc": 3, "mse": 0.084,
     "strategy": "explore", "overshoot": 0.21, "settling": 5.1, "effort": 24.1,
     "is_stable": True, "dt_mpc": 0.05},
    {"iteration": 2, "ok": True, "unstable": True, "np": 20, "nc": 9, "mse": 3.9,
     "strategy": "explore", "overshoot": 1.7, "settling": float("inf"), "effort": 91.0,
     "is_stable": False, "dt_mpc": 0.05, "unstable_reason": "State norm exceeded 1e3"},
    {"iteration": 3, "ok": False, "unstable": False, "np": 30, "nc": 30,
     "error": "OSQP reported primal infeasibility."},
    {**BEST_ROW, "ok": True, "unstable": False},
    {"iteration": 5, "ok": True, "unstable": False, "np": 14, "nc": 5, "mse": 0.014,
     "strategy": "exploit", "overshoot": None, "overshoot_meaningful": False,
     "settling": 2.9, "effort": 19.9, "is_stable": True, "dt_mpc": 0.04},
]


class _Tracker:
    def snapshot(self):
        return {
            "total_tokens": 48210,
            "per_model": {"gpt-4o-mini": {"prompt": 31020, "completion": 9110, "total": 40130}},
            "cost_usd": 0.0142,
            "unpriced_models": [],
        }


@pytest.fixture
def dark_figure():
    """A figure styled the way the app styles its charts, for the dark UI."""
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.plot([1, 2, 3], [3, 2, 1], color="#4d9fff", label="MSE")
    ax.set_xlabel("iteration")
    ax.legend()
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)
    ax.tick_params(colors="#6a7a9a")
    ax.xaxis.label.set_color("#6a7a9a")
    yield fig
    plt.close(fig)


@pytest.fixture
def out_path():
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


def _pdf_text(path):
    pypdf = pytest.importorskip("pypdf")
    reader = pypdf.PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _section_titles(text):
    """Section headings as they appear, deduplicated -- the table of contents
    repeats every one of them."""
    seen = []
    for line in text.split("\n"):
        s = line.strip()
        if re.match(r"^\d+\. [A-Z]", s) and len(s) < 60 and s not in seen:
            seen.append(s)
    return seen


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def test_dark_chart_is_rendered_light_for_paper(dark_figure):
    """The app's charts are near-black for its dark UI. Printed as-is they are
    a dark rectangle with unreadable labels, so the report re-renders them."""
    pil_image = pytest.importorskip("PIL.Image")

    plain = io.BytesIO()
    dark_figure.savefig(plain, format="png", dpi=100, facecolor="white")

    def white_fraction(data):
        im = pil_image.open(io.BytesIO(data)).convert("RGB")
        pixels = list(im.convert("L").tobytes())
        return sum(1 for v in pixels if v > 240) / len(pixels)

    assert white_fraction(plain.getvalue()) < 0.5   # dark plot area survives
    assert white_fraction(_figure_png(dark_figure)) > 0.85


def test_figure_colours_are_restored_for_the_caller(dark_figure):
    """The docstring promises the caller keeps ownership of the figure."""
    before = (dark_figure.patch.get_facecolor(), dark_figure.get_axes()[0].patch.get_facecolor())
    _figure_png(dark_figure)
    after = (dark_figure.patch.get_facecolor(), dark_figure.get_axes()[0].patch.get_facecolor())
    assert before == after


# --------------------------------------------------------------------------
# Row formatting
# --------------------------------------------------------------------------

def test_plant_rows_read_the_summary_the_old_report_ignored():
    values = dict(_plant_rows(SUMMARY))
    assert values["Dynamics class"] == "InvertedPendulum"
    assert values["State variables"] == "x, x_dot, theta, theta_dot"


def test_physical_parameters_are_listed():
    assert dict(_physical_param_rows(SUMMARY))["g"] == "9.81"


def test_bounds_pair_each_limit_with_its_variable_name():
    rows = _bounds_rows(SUMMARY)
    assert ["Input", "F", "-10", "10"] in rows
    assert ["State", "x", "-2.4", "2.4"] in rows


def test_bounds_are_absent_when_the_plugin_declares_none():
    assert _bounds_rows({"dynamics_class": "X"}) == []


def test_param_rows_fall_back_when_best_row_is_empty():
    values = dict(_param_rows({}))
    assert values["Prediction horizon (Np)"] == "n/a"
    assert values["Sample time (dt_mpc)"] == "n/a"


def test_metric_rows_spell_out_a_run_that_never_settled():
    rows = dict(tuple(r) for r in _metric_rows({**BEST_ROW, "settling": float("inf")}))
    assert rows["Settling time"] == "not settled in window"


def test_metric_rows_mark_overshoot_not_applicable_for_a_moving_reference():
    rows = dict(tuple(r) for r in _metric_rows({**BEST_ROW, "overshoot_meaningful": False}))
    assert rows["Overshoot"] == "N/A"


def test_per_state_breakdown_lists_every_state():
    rows = _per_state_rows(BEST_ROW)
    assert [r[0] for r in rows] == ["x", "theta"]


def test_per_state_breakdown_absent_when_the_evaluator_reported_none():
    assert _per_state_rows({"mse": 1.0}) == []


def test_progression_marks_only_genuine_improvements():
    rows = _progression_rows(RESULTS)
    # iterations 1, 2, 4, 5 succeeded; 1 and 4 set a new best
    assert [r[0] for r in rows] == ["1", "2", "4", "5"]
    assert [r[0] for r in rows if r[4] == "yes"] == ["1", "4"]


def test_improvement_diff_compares_first_success_with_best():
    changed = _improvement_diff(RESULTS, BEST_ROW)
    assert changed["Iteration"] == ("1", "4")
    assert changed["Np / Nc"] == ("8 / 3", "12 / 4")


def test_improvement_diff_empty_when_the_first_success_is_the_best():
    assert _improvement_diff([{**BEST_ROW, "ok": True}], BEST_ROW) == {}


def test_history_distinguishes_failed_from_not_applicable():
    """A FAILED iteration produced no numbers at all ("--"); a successful one
    whose overshoot doesn't apply shows "N/A". Collapsing the two would make a
    crashed run look like a merely unmeasurable one."""
    rows = {r[0]: r for r in _history_rows(RESULTS)}

    failed = rows["3"]
    assert failed[1] == "FAILED"
    assert failed[4:8] == ["--", "--", "--", "--"]

    not_applicable = rows["5"]
    assert not_applicable[1] == "OK"
    assert not_applicable[5] == "N/A"
    assert not_applicable[4] == "0.014"

    unstable = rows["2"]
    assert unstable[1] == "UNSTABLE"
    assert unstable[6] == "N/A"


def test_failure_rows_carry_the_recorded_reason():
    rows = {r[0]: r for r in _failure_rows(RESULTS)}
    assert "primal infeasibility" in rows["3"][3]
    assert "State norm exceeded" in rows["2"][3]
    assert "1" not in rows  # a clean iteration is not a failure


def test_usage_rows_tolerate_a_missing_tracker():
    assert _usage_rows(None) == []


def test_usage_rows_report_per_model_totals():
    assert _usage_rows(_Tracker())[0][:4] == ["gpt-4o-mini", "31020", "9110", "40130"]


# --------------------------------------------------------------------------
# Verdict / summary
# --------------------------------------------------------------------------

def test_verdict_fails_when_the_best_controller_is_unstable():
    ok, label = _verdict(RESULTS, {**BEST_ROW, "is_stable": False})
    assert not ok and "FAIL" in label


def test_verdict_flags_a_stable_run_that_never_settled():
    ok, label = _verdict(RESULTS, {**BEST_ROW, "settling": float("inf")})
    assert ok and "caveat" in label


def test_verdict_fails_when_nothing_succeeded():
    ok, _ = _verdict([], {})
    assert not ok


def test_executive_summary_counts_each_outcome():
    text = _executive_summary("Pendulum", RESULTS, BEST_ROW, stopped_by_user=True)
    assert "stopped early by the user" in text
    assert "iteration **4**" in text


# --------------------------------------------------------------------------
# End-to-end
# --------------------------------------------------------------------------

def test_builds_a_full_report_with_every_optional_section(out_path, dark_figure):
    build_pdf_report(
        out_path,
        system_name="InvertedPendulum",
        dynamics_summary=SUMMARY,
        results_data=RESULTS,
        best_row=BEST_ROW,
        analysis=_Analysis,
        convergence_fig=dark_figure,
        simulation_fig=dark_figure,
        stopped_by_user=False,
        tracker=_Tracker(),
    )
    text = _pdf_text(out_path)
    assert _section_titles(text) == [
        "1. System Under Control",
        "2. System Analysis",
        "3. How the Search Unfolded",
        "4. Controller Analysis",
        "5. Results Analysis",
        "6. Theoretical Context",
        "7. Convergence",
        "8. Best-Iteration Simulation",
        "9. Iteration History",
        "10. Failure Diagnostics",
        "11. Token Usage and Cost",
        "12. Recommendations",
        "13. Conclusion",
    ]
    assert "RUN VERDICT" in text
    assert "Physical parameters" in text
    assert "Per-state error breakdown" in text


def test_section_numbers_stay_contiguous_without_failures_or_tracker(out_path):
    """Failure diagnostics and token usage are conditional; dropping them must
    not leave a gap in the numbering."""
    clean = [r for r in RESULTS if r.get("ok") and not r.get("unstable")]
    build_pdf_report(out_path, "System", SUMMARY, clean, BEST_ROW, _Analysis)
    titles = _section_titles(_pdf_text(out_path))
    assert [t.split(".")[0] for t in titles] == [str(i) for i in range(1, len(titles) + 1)]
    assert not any("Failure Diagnostics" in t for t in titles)
    assert not any("Token Usage" in t for t in titles)


def test_positional_call_site_still_works(out_path):
    """agent_mpc_ui_components.py calls this positionally -- the first six
    parameters are effectively public API."""
    build_pdf_report(out_path, "System", SUMMARY, RESULTS, BEST_ROW, _Analysis)
    assert os.path.getsize(out_path) > 0


def test_survives_a_run_with_no_successful_iteration(out_path):
    build_pdf_report(
        out_path, "Empty", {}, [], None, _Analysis,
        convergence_fig=None, simulation_fig=None, stopped_by_user=True,
    )
    text = _pdf_text(out_path)
    assert "No successful iterations to chart." in text
    assert "No successful iteration to plot." in text
    assert "stopped by user" in text


def test_toc_can_be_switched_off(out_path):
    build_pdf_report(out_path, "System", SUMMARY, RESULTS, BEST_ROW, _Analysis, include_toc=False)
    assert "Contents" not in _pdf_text(out_path)


def test_defaults_to_reportlab_so_no_system_latex_is_required():
    """AgentMPC runs on the end user's own machine, so the report must not
    depend on a TeX install being present."""
    import inspect

    default = inspect.signature(build_pdf_report).parameters["backend"].default
    assert default is Backend.REPORTLAB


def test_no_math_warning_on_the_default_backend(out_path, recwarn):
    """The MPC cost function is only emitted on a backend that can typeset it,
    so the default path must not trip labcd_pdfmaker's math warning."""
    build_pdf_report(out_path, "System", SUMMARY, RESULTS, BEST_ROW, _Analysis)
    assert not [w for w in recwarn if "math" in str(w.message).lower()]
