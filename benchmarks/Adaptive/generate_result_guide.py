"""Generate a readable PDF guide for one Adaptive result JSON file."""

from __future__ import annotations

import argparse
import html
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)


HERE = Path(__file__).resolve().parent


def _fmt(value: Any, digits: int = 8) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        if value == 0:
            return "0"
        if abs(value) < 1e-4 or abs(value) >= 1e6:
            return f"{value:.6e}"
        return f"{value:.{digits}g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt(item, digits) for item in value) + "]"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value)


def _text(value: Any) -> str:
    return html.escape(_fmt(value)).replace("\n", "<br/>")


def _page(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.line(18 * mm, 16 * mm, A4[0] - 18 * mm, 16 * mm)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(18 * mm, 10 * mm, "LabCD Adaptive result report")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


class Guide:
    def __init__(self, result: dict[str, Any], source: Path, destination: Path):
        self.result = result
        self.source = source
        self.destination = destination
        self.styles = getSampleStyleSheet()
        self.styles.add(ParagraphStyle(
            "GuideTitle", parent=self.styles["Title"], fontName="Helvetica-Bold",
            fontSize=22, leading=27, textColor=colors.HexColor("#0F172A"),
            alignment=TA_CENTER, spaceAfter=10,
        ))
        self.styles.add(ParagraphStyle(
            "GuideSub", parent=self.styles["Normal"], fontSize=10, leading=14,
            textColor=colors.HexColor("#475569"), alignment=TA_CENTER, spaceAfter=16,
        ))
        self.styles.add(ParagraphStyle(
            "H1x", parent=self.styles["Heading1"], fontName="Helvetica-Bold",
            fontSize=16, leading=20, textColor=colors.HexColor("#0F4C81"),
            spaceBefore=11, spaceAfter=7,
        ))
        self.styles.add(ParagraphStyle(
            "H2x", parent=self.styles["Heading2"], fontName="Helvetica-Bold",
            fontSize=12, leading=15, textColor=colors.HexColor("#1E3A5F"),
            spaceBefore=8, spaceAfter=5,
        ))
        self.styles.add(ParagraphStyle(
            "Bodyx", parent=self.styles["BodyText"], fontSize=9.2, leading=13.2,
            textColor=colors.HexColor("#1E293B"), spaceAfter=5, wordWrap="CJK",
        ))
        self.styles.add(ParagraphStyle(
            "Smallx", parent=self.styles["BodyText"], fontSize=7.8, leading=10.5,
            textColor=colors.HexColor("#334155"), spaceAfter=3, wordWrap="CJK",
        ))
        self.styles.add(ParagraphStyle(
            "Callout", parent=self.styles["BodyText"], fontSize=9, leading=13,
            textColor=colors.HexColor("#713F12"), backColor=colors.HexColor("#FEF3C7"),
            borderColor=colors.HexColor("#F59E0B"), borderWidth=0.6,
            borderPadding=7, spaceBefore=5, spaceAfter=7,
        ))
        self.story = []

    def p(self, text: str, style: str = "Bodyx"):
        self.story.append(Paragraph(text, self.styles[style]))

    def h1(self, text: str):
        self.p(html.escape(text), "H1x")

    def h2(self, text: str):
        self.p(html.escape(text), "H2x")

    def table(self, headers, rows, widths=None, repeat=1, small=False):
        style_name = "Smallx" if small else "Bodyx"
        data = [[Paragraph(f"<b>{html.escape(str(h))}</b>", self.styles[style_name])
                 for h in headers]]
        for row in rows:
            data.append([Paragraph(_text(cell), self.styles[style_name]) for cell in row])
        table = Table(data, colWidths=widths, repeatRows=repeat, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCEAF7")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0F2942")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#94A3B8")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#F8FAFC")]),
        ]))
        self.story.extend([table, Spacer(1, 4 * mm)])

    def metric_rows(self, metrics: dict[str, Any]):
        checks = metrics.get("success_checks") or {}
        return [
            ("numerically_healthy", metrics.get("numerically_healthy"),
             "All states, outputs and controls stayed finite."),
            ("success", metrics.get("success"),
             "True only when finite, bounded and mse_target all pass."),
            ("success_checks.finite", checks.get("finite"),
             "No NaN or infinity occurred."),
            ("success_checks.bounded", checks.get("bounded"),
             "Output stayed below 1000 times the effective task scale and |u| stayed below 1e12."),
            ("success_checks.mse_target", checks.get("mse_target"),
             "Every output's steady RMS fraction was at or below success_target_frac."),
            ("success_reason", metrics.get("success_reason") or "empty (passed)",
             "The first failed hard check; empty means all checks passed."),
            ("success_target_frac", metrics.get("success_target_frac"),
             "Allowed steady RMS error as a fraction of task scale; 0.02 means 2%."),
            ("tracking_pct.full", (metrics.get("tracking_pct") or {}).get("full"),
             "Tracking percentage over the complete simulation."),
            ("tracking_pct.steady", (metrics.get("tracking_pct") or {}).get("steady"),
             "Tracking percentage over the final 20%."),
            ("tracking_pct.transient", (metrics.get("tracking_pct") or {}).get("transient"),
             "Tracking percentage over the first 20%."),
            ("tracking_mse.full", (metrics.get("tracking_mse") or {}).get("full"),
             "Mean squared tracking error over the full simulation."),
            ("tracking_mse.steady", (metrics.get("tracking_mse") or {}).get("steady"),
             "Mean squared tracking error over the final 20%."),
            ("tracking_mse.transient", (metrics.get("tracking_mse") or {}).get("transient"),
             "Mean squared tracking error over the first 20%."),
            ("task_scale", metrics.get("task_scale"),
             "RMS error of an output frozen at its initial value. Here it equals the initial 0.05 rad error."),
            ("task_trivial", metrics.get("task_trivial"),
             "True only when the initial output is already effectively at its reference."),
            ("tracking_pct_headline", metrics.get("tracking_pct_headline"),
             "Worst steady tracking percentage across controlled outputs."),
            ("tracking_pct_mean", metrics.get("tracking_pct_mean"),
             "Mean steady tracking percentage across controlled outputs."),
            ("transient_rms", metrics.get("transient_rms"),
             "RMS tracking error over the first 20%; radians for this pole."),
            ("steady_rms", metrics.get("steady_rms"),
             "RMS tracking error over the final 20%; radians for this pole."),
            ("steady_rms_frac", metrics.get("steady_rms_frac"),
             "steady_rms divided by effective task scale; used by mse_target."),
            ("overshoot_frac", metrics.get("overshoot_frac"),
             "Maximum absolute error divided by reference amplitude with a 1e-9 floor."),
            ("settling_time", metrics.get("settling_time"),
             "First time after which error always remains inside the 2% reference-amplitude band."),
            ("settling_time_reached", metrics.get("settling_time_reached"),
             "Whether that settling condition was reached within the horizon."),
            ("control_rms", metrics.get("control_rms"),
             "RMS control torque over the simulation; the selected tuning objective."),
            ("control_max", metrics.get("control_max"),
             "Largest absolute control torque."),
            ("control_rate_max", metrics.get("control_rate_max"),
             "Largest absolute finite-difference rate |du/dt|."),
            ("chattering_tv_per_sec", metrics.get("chattering_tv_per_sec"),
             "Total variation of the control per second; mainly an SMC diagnostic."),
        ]

    def finish(self, title: str, subject: str):
        frame = Frame(17 * mm, 19 * mm, A4[0] - 34 * mm, A4[1] - 34 * mm,
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        doc = BaseDocTemplate(
            str(self.destination), pagesize=A4,
            leftMargin=17 * mm, rightMargin=17 * mm,
            topMargin=17 * mm, bottomMargin=19 * mm,
            title=title,
            author="LabCD benchmark tooling",
            subject=subject,
        )
        doc.addPageTemplates([PageTemplate(id="guide", frames=[frame], onPage=_page)])
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        doc.build(self.story)

    def build(self):
        r = self.result
        tuning = r.get("tuning") or {}
        verification = r.get("verification") or {}
        history = tuning.get("history") or []
        messages = tuning.get("mock_agent_messages") or []
        usage = tuning.get("token_usage") or {}

        self.p("Adaptive Controller Result Guide", "GuideTitle")
        self.p(
            f"Plant: <b>{html.escape(str(r.get('plant', 'unknown')))}</b> &nbsp; | &nbsp; "
            f"Test: <b>{html.escape(str(r.get('test_id', 'unknown')))}</b> &nbsp; | &nbsp; "
            f"Mode: <b>{html.escape(str(r.get('agent_mode', 'unknown')))}</b>",
            "GuideSub",
        )
        self.table(
            ["Item", "Recorded value"],
            [
                ("Source JSON", self.source.name),
                ("Artifact", r.get("artifact")),
                ("Pipeline completed", r.get("pipeline_completed")),
                ("Verification success", verification.get("success")),
                ("Best tuning round", tuning.get("best_round")),
                ("Generated", datetime.now().astimezone().isoformat(timespec="seconds")),
            ],
            widths=[52 * mm, 123 * mm],
        )
        self.p(
            "This guide separates four kinds of information: agent responses, tuning-round "
            "results, final numerical verification, and notebook packaging. The agents propose "
            "or describe decisions. Python performs the controller design, simulations, metric "
            "calculations, best-round selection and JSON/PDF writing."
        )

        self.h1("1. Run flow and responsibility")
        self.table(
            ["Stage", "What happens", "Source"],
            [
                ("Designer", "Chooses Backstepping or SMC from the plant structure.", "Mock/LLM agent"),
                ("Round 0", "Runs the initial controller parameters.", "Python simulation"),
                ("Tuning rounds", "Agent proposes gains; every proposal is simulated and scored.", "Agent + Python"),
                ("Best selection", "Chooses the lowest score for the selected objectives.", "Python"),
                ("Verification", "Runs the selected best parameters again on the final path.", "Python simulation"),
                ("Reporter", "Creates a short abstract from the final outcome.", "Mock/LLM agent"),
                ("JSON/PDF", "Organizes and saves all fields.", "Notebook / this generator"),
            ],
            widths=[28 * mm, 104 * mm, 43 * mm],
        )

        self.h1("2. Applied test")
        dyn = ((r.get("applied_system_spec") or {}).get("dynamics") or {})
        test_rows = [
            ("States", dyn.get("states")),
            ("Inputs", dyn.get("inputs")),
            ("Outputs", dyn.get("outputs")),
            ("Initial state", dyn.get("x0")),
            ("References", dyn.get("references")),
            ("Uncertainty", dyn.get("uncertainty")),
            ("Disturbance", dyn.get("disturbance")),
            ("Simulation time", dyn.get("sim_time")),
            ("Solver step", dyn.get("solver_step")),
        ]
        self.table(["Field", "Value"], test_rows, widths=[52 * mm, 123 * mm])

        self.h1("3. Agent messages")
        self.p(
            "The mock run records one Designer call, one Tuner call for each tuning round, "
            "and one Reporter call. Responses are seeded and repeatable. The numerical metrics "
            "that follow are calculated from simulations."
        )
        for message in messages:
            role = str(message.get("agent", "unknown")).title()
            call = message.get("call")
            self.h2(f"{role} call {call}")
            response = message.get("response") or {}
            self.table(
                ["Response field", "Value"],
                [(key, value) for key, value in response.items()],
                widths=[48 * mm, 127 * mm], small=True,
            )
            msg_usage = message.get("usage") or {}
            self.p(
                "Synthetic usage: input {0}, output {1}, total {2}.".format(
                    _fmt(msg_usage.get("input_tokens")),
                    _fmt(msg_usage.get("output_tokens")),
                    _fmt(msg_usage.get("total_tokens")),
                ),
                "Smallx",
            )

        self.h1("4. Best parameters and best-round metrics")
        self.p(
            "best_params contains the controller parameters from the tuning round with the "
            "lowest internal objective score. This run selected only control_effort, so the "
            "ranking is based on control_rms relative to round 0. The objective weight is 3; "
            "with one objective, changing that single weight does not change the ranking."
        )
        best_params = tuning.get("best_params") or {}
        self.table(
            ["Parameter", "Best value"],
            [(key, value) for key, value in best_params.items()],
            widths=[62 * mm, 113 * mm],
        )
        self.p(
            f"Selected round: <b>{_text(tuning.get('best_round'))}</b>. "
            "best_metrics is the simulation result recorded when this candidate was tested "
            "inside the tuning loop."
        )
        best_metrics = tuning.get("best_metrics") or {}
        self.table(
            ["Metric", "Best-round value", "Meaning"],
            self.metric_rows(best_metrics),
            widths=[40 * mm, 38 * mm, 97 * mm], small=True,
        )

        self.h1("5. Exact metric formulas")
        formulas = [
            ("Tracking error", "e[k] = y[k] - reference[k]"),
            ("Window MSE", "MSE = mean(e[k]^2) within full, first 20%, or last 20%"),
            ("Window RMS", "RMS = sqrt(MSE)"),
            ("Task scale", "S = RMS(y[0] - reference[k]); here S = 0.05 rad"),
            ("Tracking percentage", "100 * max(0, 1 - RMS / S)"),
            ("Steady RMS fraction", "steady_rms / effective_task_scale"),
            ("Success target", "steady_rms_frac <= success_target_frac for every output"),
            ("Control RMS", "sqrt(mean(u[k]^2))"),
            ("Control maximum", "max(abs(u[k]))"),
            ("Control rate maximum", "max(abs((u[k+1] - u[k]) / dt))"),
            ("Control variation", "sum(abs(u[k+1] - u[k])) / simulation_duration"),
        ]
        self.table(["Quantity", "Formula"], formulas, widths=[52 * mm, 123 * mm])
        self.p(
            "The JSON key mse_target is historical naming: the implemented hard check uses "
            "steady normalized RMS, not raw MSE.",
            "Callout",
        )
        self.p(
            "Zero-reference warning: overshoot_frac divides maximum absolute error by max reference "
            "amplitude, with a 1e-9 floor. For this zero reference, 0.05 / 1e-9 = 50,000,000. "
            "That large value is a normalization artifact, not physical overshoot. The settling "
            "band also uses 2% of the floored reference amplitude, only 2e-11 rad here, so the "
            "reported 7.86 s is much stricter than a practical task-scale settling band.",
            "Callout",
        )

        self.h1("6. Tuning history")
        self.p(
            "History is the initial design plus the result after each tuning proposal. Round 0 "
            "uses default gains. Each later row records the gains actually simulated, what changed, "
            "the simulation verdict and the selected objective value."
        )
        history_rows = []
        for entry in history:
            obj = entry.get("objective_values") or []
            history_rows.append((
                entry.get("round"),
                (entry.get("tuning") or {}).get("c_gains"),
                entry.get("changed") or {},
                entry.get("success"),
                entry.get("met_target"),
                entry.get("tracking_pct_headline"),
                obj,
            ))
        self.table(
            ["Round", "c_gains", "Changed", "Pass", "Target met", "Steady tracking %", "Objective"],
            history_rows,
            widths=[11 * mm, 27 * mm, 42 * mm, 13 * mm, 18 * mm, 27 * mm, 37 * mm],
            small=True,
        )
        self.p(
            "success and met_target are different. success is the numerical/verification verdict. "
            "met_target controls tuner early stopping. Since this run selected only control_effort "
            "and no tracking objective, the implementation deliberately keeps met_target false and "
            "runs all configured tuning rounds, even when the tracking success check passes."
        )
        for entry in history:
            self.h2(f"Round {entry.get('round')}")
            self.p(f"<b>Reasoning:</b> {_text(entry.get('reasoning'))}")
            self.p(f"<b>Parameters:</b> {_text(entry.get('tuning'))}", "Smallx")
            self.p(f"<b>Changed:</b> {_text(entry.get('changed') or 'none')}", "Smallx")
            if entry.get("report"):
                safe_report = html.escape(str(entry["report"]))
                self.story.append(Preformatted(
                    safe_report,
                    ParagraphStyle("Code", parent=self.styles["Code"], fontName="Courier",
                                   fontSize=6.7, leading=8.4, textColor=colors.HexColor("#334155")),
                    maxLineLength=105,
                ))

        self.h1("7. Token usage")
        total = usage.get("total") or {}
        self.table(
            ["Field", "Value", "Meaning"],
            [
                ("input_tokens", total.get("input_tokens"), "Total synthetic prompt tokens."),
                ("output_tokens", total.get("output_tokens"), "Total synthetic response tokens."),
                ("total_tokens", total.get("total_tokens"), "Input plus output tokens."),
                ("cached_input_tokens", total.get("cached_input_tokens"), "Synthetic cached subset; zero here."),
                ("simulated", usage.get("simulated"), "True means these are test values, not tokenizer measurements."),
                ("mock_seed", usage.get("mock_seed"), "Seed controlling repeatable mock output."),
                ("price_usd_per_token", usage.get("price_usd_per_token"), "Synthetic benchmark price."),
                ("cost_usd", usage.get("cost_usd"), "Synthetic total_tokens times synthetic price."),
                ("actual_cost_usd", usage.get("actual_cost_usd"), "Actual API charge; zero for mock mode."),
            ],
            widths=[48 * mm, 34 * mm, 93 * mm],
        )
        timeline_rows = [
            (item.get("actor"), item.get("round"), item.get("label"),
             item.get("input_tokens"), item.get("output_tokens"), item.get("tokens"),
             item.get("detail"))
            for item in (usage.get("timeline") or [])
        ]
        self.table(
            ["Agent", "Round", "Label", "Input", "Output", "Total", "Detail"],
            timeline_rows,
            widths=[19 * mm, 13 * mm, 40 * mm, 18 * mm, 18 * mm, 18 * mm, 49 * mm],
            small=True,
        )

        self.h1("8. Final verification")
        self.p(
            "After choosing best_params, the pipeline runs those parameters once more through the "
            "final simulation path. verification contains that final result. best_metrics and "
            "verification therefore refer to two executions of the same selected parameters. In "
            "this deterministic run their numerical values match. Verification additionally includes "
            "mse_target_from_tuner=true, which records that the tuner target was used."
        )
        self.table(
            ["Metric", "Verification value", "Meaning"],
            self.metric_rows(verification) + [
                ("mse_target_from_tuner", verification.get("mse_target_from_tuner"),
                 "The final success threshold came from the tuning configuration."),
            ],
            widths=[40 * mm, 38 * mm, 97 * mm], small=True,
        )
        self.p(
            "pipeline_completed=true means method selection, controller construction and final "
            "simulation completed and time-series data existed. verification.success=true is the "
            "separate statement that the final run passed finite, bounded and steady-error checks."
        )

        self.h1("9. Summary field")
        self.p(
            "The summary is built by Python from the selected method, symbolic controller, stability "
            "proof, final parameters and verification metrics. The Reporter does not create this full "
            "section. Its short abstract is stored in the reporter entry inside mock_agent_messages; "
            "the current compact notebook JSON does not copy that abstract to a root-level field."
        )

        self.finish(
            title=f"Adaptive result guide - {r.get('plant', '')}",
            subject="Adaptive controller tuning result fields and interpretation",
        )


class BriefGuide(Guide):
    """Compact run report containing only the main engineering results."""

    def build(self):
        r = self.result
        tuning = r.get("tuning") or {}
        verification = r.get("verification") or {}
        usage = tuning.get("token_usage") or {}
        total_usage = usage.get("total") or {}
        spec = r.get("applied_system_spec") or {}
        dyn = spec.get("dynamics") or {}

        self.p("Adaptive Controller Brief Report", "GuideTitle")
        self.p(
            f"Plant: <b>{html.escape(str(r.get('plant', 'unknown')))}</b> &nbsp; | &nbsp; "
            f"Test: <b>{html.escape(str(r.get('test_id', 'unknown')))}</b> &nbsp; | &nbsp; "
            f"Mode: <b>{html.escape(str(r.get('agent_mode', 'unknown')))}</b>",
            "GuideSub",
        )
        self.table(
            ["Run field", "Value"],
            [
                ("Source JSON", self.source.name),
                ("Artifact", r.get("artifact")),
                ("System name", spec.get("system_name") or r.get("plant")),
                ("Pipeline completed", r.get("pipeline_completed")),
                ("Verification success", verification.get("success")),
                ("Best tuning round", tuning.get("best_round")),
                ("Generated", datetime.now().astimezone().isoformat(timespec="seconds")),
            ],
            widths=[54 * mm, 121 * mm],
        )

        self.h1("Dynamics and test")
        self.table(
            ["Field", "Value"],
            [
                ("States", dyn.get("states")),
                ("State meanings", dyn.get("state_meanings")),
                ("Inputs", dyn.get("inputs")),
                ("Outputs", dyn.get("outputs")),
                ("State equations", dyn.get("state_equations")),
                ("Physical parameters", dyn.get("parameters")),
                ("Initial state", dyn.get("x0")),
                ("References / targets", dyn.get("references")),
                ("Uncertainty", dyn.get("uncertainty")),
                ("Disturbance", dyn.get("disturbance")),
                ("Simulation time", dyn.get("sim_time")),
                ("Solver step", dyn.get("solver_step")),
            ],
            widths=[54 * mm, 121 * mm],
            small=True,
        )

        self.h1("Best controller parameters")
        self.table(
            ["Parameter", "Best value"],
            [(key, value) for key, value in (tuning.get("best_params") or {}).items()],
            widths=[64 * mm, 111 * mm],
        )

        self.h1("Final verification")
        self.p(
            "These values come from the final simulation using the selected best parameters. "
            "They are not agent-generated text."
        )
        self.table(
            ["Metric", "Verification value"],
            [(name, value) for name, value, _meaning in self.metric_rows(verification)] + [
                ("mse_target_from_tuner", verification.get("mse_target_from_tuner")),
            ],
            widths=[72 * mm, 103 * mm],
            small=True,
        )

        self.h1("Token counts")
        role_rows = []
        for key, label in (
            ("agent", "Designer"),
            ("tuner", "Tuner"),
            ("reporter", "Reporter"),
            ("clarifier", "Clarifier"),
            ("total", "Total"),
        ):
            role_usage = usage.get(key) or {}
            role_rows.append((
                label,
                role_usage.get("input_tokens"),
                role_usage.get("output_tokens"),
                role_usage.get("total_tokens"),
                role_usage.get("cached_input_tokens"),
            ))
        self.table(
            ["Role", "Input", "Output", "Total", "Cached input"],
            role_rows,
            widths=[47 * mm, 32 * mm, 32 * mm, 32 * mm, 32 * mm],
            small=True,
        )
        self.table(
            ["Usage field", "Value"],
            [
                ("Model mapping", usage.get("models")),
                ("Simulated token usage", usage.get("simulated")),
                ("Mock seed", usage.get("mock_seed")),
                ("Recorded total tokens", total_usage.get("total_tokens")),
                ("Price per token (USD)", usage.get("price_usd_per_token")),
                ("Recorded cost (USD)", usage.get("cost_usd")),
                ("Actual API cost (USD)", usage.get("actual_cost_usd")),
            ],
            widths=[64 * mm, 111 * mm],
            small=True,
        )

        self.finish(
            title=f"Adaptive brief report - {r.get('plant', '')}",
            subject="Compact Adaptive controller tuning and verification report",
        )


def generate_guide(
    result_json: Path | str,
    output: Path | str | None = None,
) -> Path:
    """Generate the single permanent guide for understanding full results."""
    source = Path(result_json).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    result = json.loads(source.read_text(encoding="utf-8"))
    destination = (
        Path(output).resolve()
        if output is not None
        else HERE / "adaptive_results_guide.pdf"
    )
    Guide(result, source, destination).build()
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", type=Path, help="Adaptive result JSON to explain")
    parser.add_argument("--output", type=Path, help="Guide PDF path")
    args = parser.parse_args()
    destination = generate_guide(
        result_json=args.result_json,
        output=args.output,
    )
    print(f"Result guide PDF: {destination}")


if __name__ == "__main__":
    main()
