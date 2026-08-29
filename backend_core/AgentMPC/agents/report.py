"""Composes the AgentMPC PDF report on top of the shared labcd_pdfmaker
package (see packages/labcd_pdfmaker). This module owns AgentMPC's report
structure (System Analysis, Search Process, Controller Analysis, ...) and
its two data tables (final controller configuration, iteration history);
it only calls the generic ReportBuilder API to lay them out. It replaces
the old, hand-rolled reportlab pipeline that used to live directly in
this file (``report_pdf.py``).

Uses Backend.REPORTLAB explicitly: this report is plain prose plus
numeric tables, so it never needs real math typesetting, and reportlab
is the backend that needs nothing beyond what's already in
requirements.txt -- important since this typically runs on the end
user's own machine via Streamlit, not a controlled sandbox.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from labcd_pdfmaker import Backend, ReportBuilder


def _param_table_rows(best_row: Dict[str, Any]):
    return [
        ("Prediction horizon (Np)", str(best_row.get("np", "n/a"))),
        ("Control horizon (Nc)", str(best_row.get("nc", "n/a"))),
        ("State weights (Q)", best_row.get("Q_formatted", "n/a")),
        ("Input weights (R)", best_row.get("R_formatted", "n/a")),
        ("Terminal weights (P)", best_row.get("P_formatted", "n/a")),
        ("Sample time (dt_mpc)",
         f"{best_row['dt_mpc']:.4g}s" if best_row.get("dt_mpc") is not None else "n/a"),
    ]


def _metrics_table_rows(best_row: Dict[str, Any]):
    settling = best_row.get("settling")
    settling_str = f"{settling:.3g}s" if settling not in (None, float("inf")) else "not settled in window"
    return [
        ("MSE", f"{best_row.get('mse'):.4g}" if best_row.get("mse") is not None else "n/a"),
        ("Overshoot", f"{best_row.get('overshoot'):.4g}" if best_row.get("overshoot") is not None
                       and best_row.get("overshoot_meaningful", True) else "N/A"),
        ("Settling time", settling_str),
        ("Control effort", f"{best_row.get('effort'):.4g}" if best_row.get("effort") is not None else "n/a"),
        ("Oscillation count", str(best_row.get("oscillation_count", "n/a"))),
        ("Stable", "Yes" if best_row.get("is_stable") else "No"),
    ]


def _history_table(results_data: List[Dict[str, Any]]):
    header = ["Iter", "Status", "Np", "Nc", "MSE", "Overshoot", "Settling", "Stable", "dt (s)"]
    rows = []
    for r in results_data:
        status = "UNSTABLE" if r.get("unstable") else ("OK" if r.get("ok") else "FAILED")
        mse = f"{r['mse']:.4g}" if r.get("ok") and r.get("mse") is not None else "--"
        overshoot = (f"{r['overshoot']:.4g}" if r.get("ok") and r.get("overshoot") is not None
                     and r.get("overshoot_meaningful", True) else ("N/A" if r.get("ok") else "--"))
        settling = (f"{r['settling']:.3g}s" if r.get("ok") and r.get("settling") not in (None, float("inf")) else
                    ("N/A" if r.get("ok") else "--"))
        stable = ("Yes" if r.get("ok") and r.get("is_stable") else ("No" if r.get("ok") else "--"))
        dt_val = f"{r['dt_mpc']:.4g}" if r.get("dt_mpc") is not None else "--"
        rows.append([str(r.get("iteration", "")), status, str(r.get("np", "")), str(r.get("nc", "")),
                     mse, overshoot, settling, stable, dt_val])
    return header, rows


def _history_row_style(row):
    if row[1] == "UNSTABLE":
        return "bad"
    if row[1] == "FAILED":
        return "warn"
    return None


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
) -> None:
    """Writes the complete PDF report to ``path``. Callers should
    plt.close() the figures themselves after this returns -- this
    function reads them but doesn't take ownership."""
    n_ok = sum(1 for r in results_data if r.get("ok"))
    meta = (
        f"{len(results_data)} iteration(s) run &middot; {n_ok} successful &middot; "
        f"{'stopped by user' if stopped_by_user else 'completed automatically'}"
    )

    rb = ReportBuilder(
        title="MPC Auto-Tuning Report",
        subtitle=system_name,
        backend=Backend.REPORTLAB,
        meta_lines=[meta, datetime.now().strftime("%Y-%m-%d %H:%M")],
    )
    rb.add_page_break()

    rb.add_section("1. System Analysis", analysis.system_analysis)

    rb.add_section("2. How the Search Unfolded", analysis.search_process_analysis)
    rb.add_page_break()

    rb.add_section("3. Controller Analysis", analysis.controller_analysis)
    rb.add_key_value_table(_param_table_rows(best_row or {}), title="Final controller configuration",
                            markdown_cells=False)

    rb.add_section("4. Results Analysis", analysis.results_analysis)
    rb.add_key_value_table(_metrics_table_rows(best_row or {}), title="Best-iteration metrics",
                            markdown_cells=False)
    rb.add_page_break()

    rb.add_section("5. Theoretical Context", analysis.theoretical_context)

    rb.add_section("6. Convergence")
    if convergence_fig is not None:
        rb.add_figure(convergence_fig, caption="Metric convergence across all tuning iterations.")
    else:
        rb.add_markdown("No successful iterations to chart.")

    rb.add_section("7. Best-Iteration Simulation")
    if simulation_fig is not None:
        rb.add_figure(simulation_fig, caption="State and input trajectories for the best-performing iteration.")
    else:
        rb.add_markdown("No successful iteration to plot.")
    rb.add_page_break()

    rb.add_section("8. Iteration History")
    rb.add_markdown(
        "Every iteration attempted during this run. UNSTABLE/FAILED rows are highlighted. "
        "The full data (including Q/R/P and every metric) is available via the app's CSV export.")
    header, rows = _history_table(results_data)
    rb.add_data_table(header, rows, row_style=_history_row_style, markdown_cells=False)

    rb.add_section("9. Recommendations", analysis.recommendations)

    rb.add_section("10. Conclusion", analysis.conclusion)

    rb.build(path=path)
