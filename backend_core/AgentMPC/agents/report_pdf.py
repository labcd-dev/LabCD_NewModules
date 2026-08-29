"""
================================================================================
agents/report_pdf.py
================================================================================
Builds the final PDF report from: the Report Agent's analysis text (see
report_agent.py), the run's results_data/best_row, and two matplotlib
figures (convergence chart, best-iteration simulation plot) that app.py
already knows how to generate.

Uses reportlab (pure Python, pip-installable, no external system dependency
like LibreOffice or a headless browser) specifically because this runs on
whatever machine the Streamlit app itself is running on -- typically the
end user's own computer, not a controlled sandbox -- so it can't assume
anything beyond what's in requirements.txt is installed.

Font: reportlab's built-in "Times-Roman" / "Times-Bold" / "Times-Italic"
are three of the 14 standard PDF fonts baked into the PDF specification
itself, rendered correctly by every PDF viewer without needing an actual
Times New Roman .ttf file to be present on the system -- this is what
makes it possible to honor the "Times New Roman throughout" requirement
reliably across platforms.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)


def _styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("ReportTitle", parent=base["Title"], fontName="Times-Bold",
                                 fontSize=26, spaceAfter=6, textColor=colors.HexColor("#12213f")),
        "subtitle": ParagraphStyle("ReportSubtitle", parent=base["Normal"], fontName="Times-Italic",
                                    fontSize=13, textColor=colors.HexColor("#4a5a7a"), spaceAfter=4),
        "meta": ParagraphStyle("ReportMeta", parent=base["Normal"], fontName="Times-Roman",
                                fontSize=10, textColor=colors.HexColor("#6a7a9a")),
        "h1": ParagraphStyle("H1", parent=base["Heading1"], fontName="Times-Bold", fontSize=16,
                              spaceBefore=18, spaceAfter=8, textColor=colors.HexColor("#12213f"),
                              borderColor=colors.HexColor("#4d9fff"), borderWidth=0, borderPadding=0),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Times-Bold", fontSize=12.5,
                              spaceBefore=10, spaceAfter=5, textColor=colors.HexColor("#2f5597")),
        "body": ParagraphStyle("Body", parent=base["Normal"], fontName="Times-Roman", fontSize=10.5,
                                leading=15, spaceAfter=8, alignment=4),  # 4 = justified
        "caption": ParagraphStyle("Caption", parent=base["Normal"], fontName="Times-Italic", fontSize=9,
                                   textColor=colors.HexColor("#6a7a9a"), alignment=1, spaceAfter=14),
        "note": ParagraphStyle("Note", parent=base["Normal"], fontName="Times-Italic", fontSize=9,
                                textColor=colors.HexColor("#8a8a8a"), spaceAfter=6),
    }


def _param_table(best_row: Dict[str, Any], styles) -> Table:
    rows = [
        ["Parameter", "Value"],
        ["Prediction horizon (Np)", str(best_row.get("np", "n/a"))],
        ["Control horizon (Nc)", str(best_row.get("nc", "n/a"))],
        ["State weights (Q)", best_row.get("Q_formatted", "n/a")],
        ["Input weights (R)", best_row.get("R_formatted", "n/a")],
        ["Terminal weights (P)", best_row.get("P_formatted", "n/a")],
        ["Sample time (dt_mpc)", f"{best_row['dt_mpc']:.4g}s" if best_row.get("dt_mpc") is not None else "n/a"],
    ]
    t = Table(rows, colWidths=[2.4 * inch, 4.1 * inch])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f5597")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5fa")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c7cfdd")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def _metrics_table(best_row: Dict[str, Any], styles) -> Table:
    settling = best_row.get("settling")
    settling_str = f"{settling:.3g}s" if settling not in (None, float("inf")) else "not settled in window"
    rows = [
        ["Metric", "Value"],
        ["MSE", f"{best_row.get('mse'):.4g}" if best_row.get("mse") is not None else "n/a"],
        ["Overshoot", f"{best_row.get('overshoot'):.4g}" if best_row.get("overshoot") is not None
                       and best_row.get("overshoot_meaningful", True) else "N/A"],
        ["Settling time", settling_str],
        ["Control effort", f"{best_row.get('effort'):.4g}" if best_row.get("effort") is not None else "n/a"],
        ["Oscillation count", str(best_row.get("oscillation_count", "n/a"))],
        ["Stable", "Yes" if best_row.get("is_stable") else "No"],
    ]
    t = Table(rows, colWidths=[2.4 * inch, 4.1 * inch])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f5597")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5fa")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c7cfdd")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def _history_table(results_data: List[Dict[str, Any]], styles) -> Table:
    header = ["Iter", "Status", "Np", "Nc", "MSE", "Overshoot", "Settling", "Stable", "dt (s)"]
    rows = [header]
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

    t = Table(rows, colWidths=[0.5 * inch, 0.75 * inch, 0.45 * inch, 0.45 * inch,
                                 0.8 * inch, 0.8 * inch, 0.75 * inch, 0.55 * inch, 0.65 * inch], repeatRows=1)
    style = [
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f5597")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5fa")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c7cfdd")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for i, r in enumerate(results_data, start=1):
        if r.get("unstable"):
            style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fbe4e4")))
        elif not r.get("ok"):
            style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f5f0e0")))
    t.setStyle(TableStyle(style))
    return t


def _fig_to_image_flowable(fig, max_width_in: float = 6.7) -> Optional[Image]:
    """Renders a matplotlib figure to PNG in memory and wraps it as a
    reportlab Image flowable, scaled to fit the page width. Returns None if
    fig is None (e.g. no successful iterations to plot)."""
    if fig is None:
        return None
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    img = Image(buf)
    aspect = img.imageHeight / img.imageWidth
    img.drawWidth = max_width_in * inch
    img.drawHeight = max_width_in * inch * aspect
    return img


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
    """Writes the complete PDF report to ``path``. Callers should plt.close()
    the figures themselves after this returns -- this function reads them
    but doesn't take ownership."""
    styles = _styles()
    doc = SimpleDocTemplate(path, pagesize=letter,
                             topMargin=0.85 * inch, bottomMargin=0.85 * inch,
                             leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                             title=f"MPC Tuning Report -- {system_name}", author="Agent-MPC Report Agent")
    story: List[Any] = []

    story.append(Spacer(1, 1.4 * inch))
    story.append(Paragraph("MPC Auto-Tuning Report", styles["title"]))
    story.append(Paragraph(system_name, styles["subtitle"]))
    story.append(Spacer(1, 0.3 * inch))
    n_ok = sum(1 for r in results_data if r.get("ok"))
    story.append(Paragraph(
        f"{len(results_data)} iteration(s) run &middot; {n_ok} successful &middot; "
        f"{'stopped by user' if stopped_by_user else 'completed automatically'}",
        styles["meta"]))
    story.append(Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M"), styles["meta"]))
    story.append(PageBreak())

    story.append(Paragraph("1. System Analysis", styles["h1"]))
    story.append(Paragraph(analysis.system_analysis, styles["body"]))

    story.append(Paragraph("2. How the Search Unfolded", styles["h1"]))
    story.append(Paragraph(analysis.search_process_analysis, styles["body"]))
    story.append(PageBreak())

    story.append(Paragraph("3. Controller Analysis", styles["h1"]))
    story.append(Paragraph(analysis.controller_analysis, styles["body"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Final controller configuration", styles["h2"]))
    story.append(_param_table(best_row or {}, styles))
    story.append(Spacer(1, 10))

    story.append(Paragraph("4. Results Analysis", styles["h1"]))
    story.append(Paragraph(analysis.results_analysis, styles["body"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Best-iteration metrics", styles["h2"]))
    story.append(_metrics_table(best_row or {}, styles))
    story.append(PageBreak())

    story.append(Paragraph("5. Theoretical Context", styles["h1"]))
    story.append(Paragraph(analysis.theoretical_context, styles["body"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("6. Convergence", styles["h1"]))
    conv_img = _fig_to_image_flowable(convergence_fig)
    if conv_img is not None:
        story.append(conv_img)
        story.append(Paragraph("Metric convergence across all tuning iterations.", styles["caption"]))
    else:
        story.append(Paragraph("No successful iterations to chart.", styles["body"]))

    story.append(Paragraph("7. Best-Iteration Simulation", styles["h1"]))
    sim_img = _fig_to_image_flowable(simulation_fig)
    if sim_img is not None:
        story.append(sim_img)
        story.append(Paragraph("State and input trajectories for the best-performing iteration.", styles["caption"]))
    else:
        story.append(Paragraph("No successful iteration to plot.", styles["body"]))
    story.append(PageBreak())

    story.append(Paragraph("8. Iteration History", styles["h1"]))
    story.append(Paragraph(
        "Every iteration attempted during this run. UNSTABLE/FAILED rows are highlighted. "
        "The full data (including Q/R/P and every metric) is available via the app's CSV export.",
        styles["note"]))
    story.append(_history_table(results_data, styles))
    story.append(Spacer(1, 14))

    story.append(Paragraph("9. Recommendations", styles["h1"]))
    story.append(Paragraph(analysis.recommendations, styles["body"]))

    story.append(Paragraph("10. Conclusion", styles["h1"]))
    story.append(Paragraph(analysis.conclusion, styles["body"]))

    doc.build(story)
