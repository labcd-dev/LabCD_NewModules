"""
================================================================================
agents/report_pdf.py
================================================================================
Builds the final PDF report from: the Report Agent's analysis text (see
report_agent.py), the loaded plugin's ``dynamics_summary``, the run's
results_data/best_row, and two matplotlib figures (convergence chart,
best-iteration simulation plot) that app.py already knows how to generate.

Layout and rendering are delegated to ``labcd_pdfmaker`` (packages/), the
shared LabCD report builder that AgentAdaptive already moved onto. This
module only *composes* the report -- which sections exist, in what order, and
how an MPC run's numbers get formatted into them. Report-specific knowhow
(what a tuning iteration means, how Q/R/P are displayed, when a metric is
"not meaningful") stays here; page setup, fonts, table styling and figure
scaling live in one place shared across modules.

WHAT THE REPORT COVERS
--------------------------------------------------------------------------------
The run produces far more data than the narrative sections alone can carry, so
the report pairs each piece of the Report Agent's prose with the actual numbers
behind it:

  * the plant itself -- class, source file, state/input names, physical
    parameters and actuator/state limits, all read from ``dynamics_summary``,
    which the earlier version of this report accepted and then ignored;
  * how the search progressed -- best-so-far MSE per iteration and which
    strategy the Actor was following at each step;
  * where the error actually lives -- per-state MSE and overshoot, rather than
    only the aggregate that hides which state is misbehaving;
  * what tuning bought -- first successful iteration against the best one;
  * what went wrong -- failed and unstable iterations with their reasons, so a
    disappointing run is diagnosable from the report alone;
  * what it cost -- token usage per model, when a tracker is supplied.

Sections whose data is absent are skipped rather than printed empty, and
section numbers are assigned at build time so the numbering stays contiguous.

FIGURES
--------------------------------------------------------------------------------
The app's charts are styled for its dark UI (near-black backgrounds, pale grey
labels). Dropped onto white paper as-is, they read as a dark rectangle with
barely-visible text, so ``_figure_png`` re-renders them light: white canvas,
dark text, grey gridlines. It restores every colour it touched afterwards, so
a caller that still wants to display the same Figure object gets it back
unchanged.

BACKEND
--------------------------------------------------------------------------------
``Backend.REPORTLAB`` is the default, deliberately. reportlab is pure Python
with no external system dependency, and this code runs on whatever machine the
Streamlit app is running on -- typically the end user's own computer, not a
controlled sandbox -- so it can't assume anything beyond requirements.txt is
installed. Callers with a LaTeX install can pass ``backend=Backend.AUTO`` (or
``Backend.XELATEX``) for real math typesetting; the MPC cost function is only
included when a backend that can actually typeset it is in use.

Font: reportlab's built-in "Times-Roman" / "Times-Bold" / "Times-Italic" are
three of the 14 standard PDF fonts baked into the PDF specification itself,
rendered correctly by every PDF viewer without needing an actual Times New
Roman .ttf file present on the system -- this is what makes it possible to
honor the "Times New Roman throughout" requirement reliably across platforms.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from labcd_pdfmaker import Backend, ReportBuilder, xelatex_available

# Shown in place of a figure that doesn't exist, e.g. when no iteration ever
# completed successfully and there is nothing to chart.
_NO_CONVERGENCE = "No successful iterations to chart."
_NO_SIMULATION = "No successful iteration to plot."

_HISTORY_NOTE = (
    "*Every iteration attempted during this run. UNSTABLE rows are highlighted in red and "
    "FAILED rows in amber. The full data (including Q/R/P and every metric) is available "
    "via the app's CSV export.*"
)

# The cost function the controller is actually minimising. Only emitted on a
# backend that can typeset it -- see the module docstring.
_MPC_COST_LATEX = (
    r"J = \sum_{k=0}^{N_p-1} \left( x_k^\top Q\, x_k + u_k^\top R\, u_k \right) "
    r"+ x_{N_p}^\top P\, x_{N_p}"
)

# Matplotlib colours for re-rendering the app's dark-themed charts onto paper.
_PAPER_FG = "#12213f"      # axis labels, titles, tick labels
_PAPER_GRID = "#c7cfdd"    # gridlines and spines
_PAPER_BG = "white"


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------

def _fmt(value: Any, spec: str = ".4g", *, suffix: str = "", fallback: str = "n/a") -> str:
    """Format a number that may legitimately be absent."""
    if value is None:
        return fallback
    try:
        return f"{value:{spec}}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_settling(value: Any, *, absent: str) -> str:
    """Settling time is ``inf`` when the run never settled inside the window,
    which is a real result rather than missing data.

    ``absent`` is what to print for that case, and differs by table: the
    best-iteration panel has room to spell it out, while the per-iteration
    history table is nine columns wide and only has room for a marker.
    """
    if value is None or value == float("inf"):
        return absent
    return f"{value:.3g}s"


def _fmt_overshoot(row: Dict[str, Any]) -> str:
    """Overshoot is meaningless for a moving reference (the target itself
    moves, so the step-response definition doesn't apply). The evaluator flags
    that via ``overshoot_meaningful`` rather than emitting a fake number."""
    if row.get("overshoot") is None or not row.get("overshoot_meaningful", True):
        return "N/A"
    return _fmt(row.get("overshoot"))


def _fmt_seq(values: Any, spec: str = ".4g") -> str:
    if not isinstance(values, (list, tuple)):
        return str(values)
    return "[" + ", ".join(_fmt(v, spec) for v in values) + "]"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _figure_png(fig) -> bytes:
    """Render ``fig`` to PNG bytes on a white background.

    The app builds its charts for a dark UI, which is unreadable on paper.
    Every colour changed here is restored before returning, so the caller's
    Figure object is left exactly as it was found -- this function reads the
    figure, it doesn't take ownership of it.
    """
    saved: List[Tuple[Any, str, Any]] = []

    def override(obj, setter_name, getter_name, value):
        getter = getattr(obj, getter_name, None)
        setter = getattr(obj, setter_name, None)
        if getter is None or setter is None:
            return
        saved.append((obj, setter_name, getter()))
        setter(value)

    override(fig.patch, "set_facecolor", "get_facecolor", _PAPER_BG)
    override(fig.patch, "set_edgecolor", "get_edgecolor", _PAPER_BG)

    for ax in fig.get_axes():
        override(ax.patch, "set_facecolor", "get_facecolor", _PAPER_BG)
        for spine in ax.spines.values():
            override(spine, "set_edgecolor", "get_edgecolor", _PAPER_GRID)
        for text in (ax.title, ax.xaxis.label, ax.yaxis.label):
            override(text, "set_color", "get_color", _PAPER_FG)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            override(label, "set_color", "get_color", _PAPER_FG)
        ax.tick_params(colors=_PAPER_FG)
        for gridline in ax.get_xgridlines() + ax.get_ygridlines():
            override(gridline, "set_color", "get_color", _PAPER_GRID)
        legend = ax.get_legend()
        if legend is not None:
            override(legend.get_frame(), "set_facecolor", "get_facecolor", _PAPER_BG)
            override(legend.get_frame(), "set_edgecolor", "get_edgecolor", _PAPER_GRID)
            for text in legend.get_texts():
                override(text, "set_color", "get_color", _PAPER_FG)

    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=_PAPER_BG)
    finally:
        for obj, setter_name, old in reversed(saved):
            getattr(obj, setter_name)(old)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Table builders -- one per section that shows numbers
# ---------------------------------------------------------------------------

def _plant_rows(summary: Dict[str, Any]) -> List[Tuple[str, str]]:
    rows = [
        ("Dynamics class", str(summary.get("dynamics_class", "n/a"))),
        ("Source file", str(summary.get("source_file", "n/a"))),
        ("States", str(summary.get("n_states", "n/a"))),
        ("Inputs", str(summary.get("n_inputs", "n/a"))),
    ]
    state_names = summary.get("state_names") or []
    input_names = summary.get("input_names") or []
    if state_names:
        rows.append(("State variables", ", ".join(str(s) for s in state_names)))
    if input_names:
        rows.append(("Input variables", ", ".join(str(s) for s in input_names)))
    return rows


def _physical_param_rows(summary: Dict[str, Any]) -> List[Tuple[str, str]]:
    params = summary.get("params") or {}
    if not isinstance(params, dict):
        return []
    return [(str(k), _fmt(v)) for k, v in params.items()]


def _bounds_rows(summary: Dict[str, Any]) -> List[List[str]]:
    """Actuator and state limits, which constrain what any controller can do
    and therefore belong next to the tuning result rather than buried in the
    plugin file."""
    rows: List[List[str]] = []
    for label, key, names_key in (
        ("Input", "input_bounds", "input_names"),
        ("State", "state_bounds", "state_names"),
    ):
        bounds = summary.get(key)
        if not bounds:
            continue
        try:
            lower, upper = bounds
        except (TypeError, ValueError):
            continue
        names = summary.get(names_key) or []
        for i, (lo, hi) in enumerate(zip(lower, upper)):
            name = str(names[i]) if i < len(names) else f"{label.lower()}[{i}]"
            rows.append([label, name, _fmt(lo), _fmt(hi)])
    return rows


def _param_rows(best_row: Dict[str, Any]) -> List[Tuple[str, str]]:
    return [
        ("Prediction horizon (Np)", str(best_row.get("np", "n/a"))),
        ("Control horizon (Nc)", str(best_row.get("nc", "n/a"))),
        ("State weights (Q)", best_row.get("Q_formatted", "n/a")),
        ("Input weights (R)", best_row.get("R_formatted", "n/a")),
        ("Terminal weights (P)", best_row.get("P_formatted", "n/a")),
        ("Sample time (dt_mpc)", _fmt(best_row.get("dt_mpc"), suffix="s")),
    ]


def _metric_rows(best_row: Dict[str, Any]) -> List[List[str]]:
    return [
        ["MSE", _fmt(best_row.get("mse"))],
        ["Overshoot", _fmt_overshoot(best_row)],
        ["Settling time", _fmt_settling(best_row.get("settling"), absent="not settled in window")],
        ["Control effort (RMS)", _fmt(best_row.get("effort"))],
        ["Integral absolute error (IAE)", _fmt(best_row.get("iae"))],
        ["Integral squared error (ISE)", _fmt(best_row.get("ise"))],
        ["Oscillation count", str(best_row.get("oscillation_count", "n/a"))],
        ["Stable", "Yes" if best_row.get("is_stable") else "No"],
    ]


def _per_state_rows(best_row: Dict[str, Any]) -> List[List[str]]:
    """Aggregate MSE hides *which* state is misbehaving; this is the breakdown
    the Critic and Actor already reason about, surfaced for the reader too."""
    per_mse = best_row.get("per_state_mse") or {}
    per_os = best_row.get("per_state_overshoot") or {}
    if not isinstance(per_mse, dict) or not per_mse:
        return []
    overshoot_applies = best_row.get("overshoot_meaningful", True)
    rows = []
    for name, mse in per_mse.items():
        overshoot = per_os.get(name) if isinstance(per_os, dict) else None
        rows.append([
            str(name),
            _fmt(mse),
            _fmt(overshoot) if overshoot is not None and overshoot_applies else "N/A",
        ])
    return rows


def _progression_rows(results_data: Sequence[Dict[str, Any]]) -> List[List[str]]:
    """Best-MSE-so-far after each iteration, with the strategy that produced
    it -- this is what "the search converged" actually looks like as numbers."""
    rows = []
    best_so_far = None
    for r in results_data:
        if not r.get("ok") or r.get("mse") is None:
            continue
        mse = r["mse"]
        improved = best_so_far is None or mse < best_so_far
        if improved:
            best_so_far = mse
        rows.append([
            str(r.get("iteration", "")),
            str(r.get("strategy", "n/a")),
            _fmt(mse),
            _fmt(best_so_far),
            "yes" if improved else "",
        ])
    return rows


def _improvement_diff(
    results_data: Sequence[Dict[str, Any]], best_row: Dict[str, Any]
) -> Dict[str, Tuple[str, str]]:
    """First successful iteration vs. the best one -- what the tuning loop
    actually bought, as a before/after table."""
    first = next((r for r in results_data if r.get("ok")), None)
    if not first or not best_row or first.get("iteration") == best_row.get("iteration"):
        return {}
    return {
        "Iteration": (str(first.get("iteration", "?")), str(best_row.get("iteration", "?"))),
        "MSE": (_fmt(first.get("mse")), _fmt(best_row.get("mse"))),
        "Overshoot": (_fmt_overshoot(first), _fmt_overshoot(best_row)),
        "Settling time": (
            _fmt_settling(first.get("settling"), absent="not settled"),
            _fmt_settling(best_row.get("settling"), absent="not settled"),
        ),
        "Control effort (RMS)": (_fmt(first.get("effort")), _fmt(best_row.get("effort"))),
        "Np / Nc": (
            f"{first.get('np', '?')} / {first.get('nc', '?')}",
            f"{best_row.get('np', '?')} / {best_row.get('nc', '?')}",
        ),
    }


def _history_rows(results_data: Sequence[Dict[str, Any]]) -> List[List[str]]:
    """One row per attempted iteration.

    A failed iteration has no metrics at all ("--"), while a successful one
    whose overshoot/settling simply don't apply shows "N/A" -- those are
    different outcomes and the table keeps them distinguishable.
    """
    rows = []
    for r in results_data:
        ok = bool(r.get("ok"))
        status = "UNSTABLE" if r.get("unstable") else ("OK" if ok else "FAILED")
        if not ok:
            mse = overshoot = settling = stable = "--"
        else:
            mse = _fmt(r.get("mse"), fallback="--")
            overshoot = _fmt_overshoot(r)
            settling = _fmt_settling(r.get("settling"), absent="N/A")
            stable = "Yes" if r.get("is_stable") else "No"
        rows.append([
            str(r.get("iteration", "")),
            status,
            str(r.get("np", "")),
            str(r.get("nc", "")),
            mse,
            overshoot,
            settling,
            stable,
            _fmt(r.get("dt_mpc"), fallback="--"),
        ])
    return rows


def _history_row_style(row) -> Optional[str]:
    """UNSTABLE rows red, FAILED rows amber -- the same two highlight colours
    this module used to apply through reportlab TableStyle directly."""
    status = row[1]
    if status == "UNSTABLE":
        return "bad"
    if status == "FAILED":
        return "warn"
    return None


def _failure_rows(results_data: Sequence[Dict[str, Any]]) -> List[List[str]]:
    """Failed and unstable iterations with their reasons, so a disappointing
    run can be diagnosed from the report without reopening the app."""
    rows = []
    for r in results_data:
        if r.get("ok") and not r.get("unstable"):
            continue
        if r.get("ok"):
            reason = r.get("unstable_reason") or "Diverged (see convergence chart)."
            kind = "UNSTABLE"
        else:
            reason = r.get("error") or "No error message recorded."
            kind = "FAILED"
        rows.append([
            str(r.get("iteration", "")),
            kind,
            f"{r.get('np', '?')} / {r.get('nc', '?')}",
            str(reason).strip().replace("\n", " ")[:300],
        ])
    return rows


def _usage_rows(tracker) -> List[List[str]]:
    """Per-model token usage from the run's TokenUsageTracker, if one was
    passed. Models the price table doesn't recognise are reported as unpriced
    rather than silently counted as free."""
    snapshot = getattr(tracker, "snapshot", None)
    if not callable(snapshot):
        return []
    try:
        data = snapshot()
    except Exception:  # noqa: BLE001 -- usage reporting must never break a report
        return []
    per_model = data.get("per_model") or {}
    if not per_model:
        return []
    unpriced = set(data.get("unpriced_models") or [])
    rows = []
    for model, usage in per_model.items():
        rows.append([
            str(model),
            str(usage.get("prompt", 0)),
            str(usage.get("completion", 0)),
            str(usage.get("total", 0)),
            "not priced" if model in unpriced else "",
        ])
    return rows


# ---------------------------------------------------------------------------
# Verdict + executive summary
# ---------------------------------------------------------------------------

def _verdict(results_data: Sequence[Dict[str, Any]], best_row: Dict[str, Any]) -> Tuple[bool, str]:
    if not results_data:
        return False, "RUN VERDICT: NO ITERATIONS COMPLETED"
    if not best_row:
        return False, "RUN VERDICT: FAIL - no iteration produced a usable controller"
    if not best_row.get("is_stable"):
        return False, "RUN VERDICT: FAIL - the best iteration is not stable"
    settling = best_row.get("settling")
    if settling is None or settling == float("inf"):
        return True, "RUN VERDICT: PASS (with caveat) - stable, but never settled inside the window"
    return True, "RUN VERDICT: PASS - stable controller found"


def _executive_summary(
    system_name: str,
    results_data: Sequence[Dict[str, Any]],
    best_row: Dict[str, Any],
    stopped_by_user: bool,
) -> str:
    n_ok = sum(1 for r in results_data if r.get("ok"))
    n_unstable = sum(1 for r in results_data if r.get("unstable"))
    n_failed = sum(1 for r in results_data if not r.get("ok"))
    ending = "stopped early by the user" if stopped_by_user else "run to completion"

    lines = [
        f"Automated MPC tuning for **{system_name}**, {ending}. "
        f"{len(results_data)} iteration(s) were attempted: {n_ok} evaluated successfully, "
        f"{n_unstable} diverged, {n_failed} failed to simulate."
    ]
    if best_row:
        lines.append(
            f"The best controller was found at iteration **{best_row.get('iteration', '?')}**, "
            f"with Np={best_row.get('np', '?')}, Nc={best_row.get('nc', '?')} and "
            f"MSE={_fmt(best_row.get('mse'))}. Settling time was "
            f"{_fmt_settling(best_row.get('settling'), absent='never reached inside the window')} "
            f"at a control effort (RMS) of {_fmt(best_row.get('effort'))}."
        )
    else:
        lines.append(
            "No iteration produced a usable controller, so there is no final configuration "
            "to report. The failure diagnostics section lists what went wrong in each attempt."
        )
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Section numbering
# ---------------------------------------------------------------------------

class _Sections:
    """Hands out contiguous section numbers at build time.

    Sections are conditional -- a run with no failures has no diagnostics
    section, a run without a tracker has no cost section -- so the numbers
    can't be hardcoded in the strings without leaving gaps.
    """

    def __init__(self) -> None:
        self._n = 0

    def __call__(self, title: str) -> str:
        self._n += 1
        return f"{self._n}. {title}"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_pdf_report(
    path: str,
    system_name: str,
    dynamics_summary: Dict[str, Any],
    results_data: List[Dict[str, Any]],
    best_row: Optional[Dict[str, Any]],
    analysis,
    convergence_fig=None,
    simulation_fig=None,
    stopped_by_user: bool = False,
    *,
    tracker=None,
    backend: Backend = Backend.REPORTLAB,
    include_toc: bool = True,
) -> None:
    """Writes the complete PDF report to ``path``.

    Callers should plt.close() the figures themselves after this returns --
    this function reads them but doesn't take ownership, and restores any
    styling it changes while rendering.

    ``tracker`` is the run's TokenUsageTracker (agents/llm_base.py). When
    supplied, the report gains a token-usage and cost section; when omitted,
    that section is simply absent.
    """
    best = best_row or {}
    summary = dynamics_summary or {}
    results = results_data or []
    n_ok = sum(1 for r in results if r.get("ok"))
    section = _Sections()

    # Math is only worth emitting on a backend that can typeset it; on
    # reportlab it would render as literal monospace and trip a UserWarning.
    math_ok = backend == Backend.XELATEX or (
        backend == Backend.AUTO and xelatex_available()
    )

    builder = ReportBuilder(
        title="MPC Auto-Tuning Report",
        subtitle=system_name,
        backend=backend,
        meta_lines=[
            f"{len(results)} iteration(s) run · {n_ok} successful · "
            f"{'stopped by user' if stopped_by_user else 'completed automatically'}",
        ],
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

    ok, label = _verdict(results, best)
    builder.add_status_badge(ok, label)
    builder.add_abstract(_executive_summary(system_name, results, best, stopped_by_user))

    if include_toc:
        builder.add_table_of_contents()

    # -- The plant ---------------------------------------------------------
    builder.add_section(
        section("System Under Control"),
        "The plant this controller was tuned against, as declared by the loaded "
        "dynamics plugin.",
    )
    builder.add_key_value_table(_plant_rows(summary), title="Plant")

    physical = _physical_param_rows(summary)
    if physical:
        builder.add_key_value_table(physical, title="Physical parameters")

    bounds = _bounds_rows(summary)
    if bounds:
        builder.add_data_table(
            ["Kind", "Variable", "Lower", "Upper"], bounds,
            title="Actuator and state limits", markdown_cells=False,
        )

    builder.add_section(section("System Analysis"), analysis.system_analysis)
    builder.add_page_break()

    # -- The search --------------------------------------------------------
    builder.add_section(section("How the Search Unfolded"), analysis.search_process_analysis)
    progression = _progression_rows(results)
    if progression:
        builder.add_data_table(
            ["Iter", "Strategy", "MSE", "Best so far", "New best"], progression,
            title="Search progression", markdown_cells=False,
        )

    # -- The controller ----------------------------------------------------
    builder.add_section(section("Controller Analysis"), analysis.controller_analysis)
    if math_ok:
        builder.add_markdown(
            "The tuned weights enter the finite-horizon cost the controller minimises "
            "at every step:"
        )
        builder.add_math_display(_MPC_COST_LATEX)
    builder.add_key_value_table(_param_rows(best), title="Final controller configuration")

    # -- The results -------------------------------------------------------
    builder.add_section(section("Results Analysis"), analysis.results_analysis)
    builder.add_data_table(
        ["Metric", "Value"], _metric_rows(best),
        title="Best-iteration metrics", markdown_cells=False,
    )

    per_state = _per_state_rows(best)
    if per_state:
        builder.add_markdown(
            "Aggregate MSE hides which state is responsible for the error. The breakdown "
            "below is the same per-state signal the Critic uses to decide *which* weight "
            "to change."
        )
        builder.add_data_table(
            ["State", "MSE", "Overshoot"], per_state,
            title="Per-state error breakdown", markdown_cells=False,
        )

    changed = _improvement_diff(results, best)
    if changed:
        builder.add_subsection("What tuning changed")
        builder.add_markdown(
            "The first successful iteration compared with the best one -- the net effect "
            "of the tuning loop."
        )
        builder.add_diff_table(changed)

    builder.add_page_break()
    builder.add_section(section("Theoretical Context"), analysis.theoretical_context)

    # -- Charts ------------------------------------------------------------
    builder.add_section(section("Convergence"))
    if convergence_fig is None:
        builder.add_markdown(_NO_CONVERGENCE)
    else:
        builder.add_figure(
            _figure_png(convergence_fig),
            caption="Metric convergence across all tuning iterations.",
        )

    builder.add_section(section("Best-Iteration Simulation"))
    if simulation_fig is None:
        builder.add_markdown(_NO_SIMULATION)
    else:
        builder.add_figure(
            _figure_png(simulation_fig),
            caption="State and input trajectories for the best-performing iteration.",
        )
    builder.add_page_break()

    # -- The full record ---------------------------------------------------
    builder.add_section(section("Iteration History"), _HISTORY_NOTE)
    builder.add_data_table(
        ["Iter", "Status", "Np", "Nc", "MSE", "Overshoot", "Settling", "Stable", "dt (s)"],
        _history_rows(results),
        row_style=_history_row_style,
        markdown_cells=False,
    )

    failures = _failure_rows(results)
    if failures:
        builder.add_section(
            section("Failure Diagnostics"),
            "Iterations that diverged or failed to simulate, with the reason recorded at "
            "the time. A run that ended badly is diagnosable from this table alone.",
        )
        builder.add_data_table(
            ["Iter", "Kind", "Np / Nc", "Reason"], failures, markdown_cells=False,
        )

    usage = _usage_rows(tracker)
    if usage:
        builder.add_section(
            section("Token Usage and Cost"),
            "LLM usage for this run, per model.",
        )
        builder.add_data_table(
            ["Model", "Input tokens", "Output tokens", "Total", "Note"], usage,
            markdown_cells=False,
        )

    # -- Closing -----------------------------------------------------------
    builder.add_section(section("Recommendations"), analysis.recommendations)
    builder.add_section(section("Conclusion"), analysis.conclusion)

    builder.build(path=path)
