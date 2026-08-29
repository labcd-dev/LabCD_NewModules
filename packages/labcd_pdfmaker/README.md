# labcd_pdfmaker

One PDF-report builder for every LabCD backend module, with two interchangeable
backends:

| Backend | Dependency | Good for |
|---|---|---|
| `Backend.REPORTLAB` | `reportlab` (pure Python, already a hard dependency) | No math, needs to run on an end-user machine with nothing extra installed |
| `Backend.XELATEX` | a system `xelatex` binary (TeX Live / MiKTeX) | Real inline/display math typesetting |
| `Backend.AUTO` (default) | either | Picks xelatex only if the report actually uses math **and** xelatex is on `PATH`; falls back to reportlab otherwise |

Replaces AgentAdaptive's old hand-rolled Markdown->LaTeX pipeline
(`backend_core/AgentAdaptive/tools/pdf_report.py`). Designed to also fit
AgentMPC's reportlab-based report (`backend_core/AgentMPC/agents/report_pdf.py`),
which is unchanged for now and can migrate later.

## Install

```bash
pip install -e "packages/labcd_pdfmaker"
```

The xelatex backend needs a system LaTeX distribution (see the repo root
`packages.txt` for the Debian package list); nothing extra to `pip install`.

## Usage

```python
from labcd_pdfmaker import ReportBuilder, Backend

builder = ReportBuilder(
    title="MPC Auto-Tuning Report",
    subtitle="InvertedPendulum",
    backend=Backend.AUTO,          # or Backend.REPORTLAB / Backend.XELATEX
    meta_lines=["12 iterations run · 9 successful"],
)

builder.add_table_of_contents()
builder.add_abstract("Short plain-English summary of the run.")

builder.add_section("System Analysis", "The plant is underactuated with...")

builder.add_status_badge(ok=True, label="RUN VERDICT: PASS")

builder.add_key_value_table(
    [("Prediction horizon (Np)", "12"), ("Control horizon (Nc)", "4")],
    title="Final controller configuration",
)

builder.add_data_table(
    ["Iter", "Status", "MSE", "Stable"],
    [[1, "OK", "0.012", "Yes"], [2, "UNSTABLE", "--", "No"]],
    row_style=lambda row: "bad" if row[1] == "UNSTABLE" else None,
    title="Iteration history",
)

builder.add_figure(convergence_png_bytes, caption="Metric convergence across iterations.")
builder.add_math_display(r"\dot{x} = Ax + Bu")   # only typeset for real on XELATEX
builder.add_verbatim(console_log_text)

pdf_bytes = builder.build()          # or builder.build(path="report.pdf")
```

## Content model

`ReportBuilder` accumulates backend-agnostic blocks (`labcd_pdfmaker.blocks`):
title page, abstract, table of contents, headings, Markdown text (headings,
bold/italic/inline code, bullet lists, block quotes, tables, inline/display
math), key-value tables, multi-column data tables (with optional per-row
`row_style` -> `"bad"` / `"warn"` colouring), before/after diff tables, status
badges, figures (raw PNG bytes or a matplotlib `Figure`), and verbatim/log
blocks. Each backend (`labcd_pdfmaker.backends.reportlab_backend` /
`xelatex_backend`) renders that same block list in its own way.

**Math on the reportlab backend degrades gracefully**: `$...$` spans render as
literal monospace text instead of typeset math, and `ReportBuilder.build()`
emits a `UserWarning` when this happens. There is no silent quality loss --
either you get real math (xelatex) or you get a clearly-marked plain-text
fallback (reportlab), never a report that quietly pretends math isn't there.

## Migration notes

- **AgentAdaptive** (`backend_core/AgentAdaptive/tools/report.py`) composes a
  `ReportBuilder` with `Backend.AUTO` and its existing domain sections
  (run verdict, clarifications, tuning history, token usage) built from
  `add_status_badge` / `add_data_table` / `add_diff_table` / `add_verbatim`.
  This replaces the module's old `tools/pdf_report.py`.
- **AgentMPC** still uses its own `agents/report_pdf.py` (reportlab,
  unchanged) -- not migrated yet. Its report structure (System Analysis,
  Search Process, Controller Analysis, ...) would map onto `add_section` /
  `add_key_value_table` / `add_data_table` / `add_figure` the same way
  AgentAdaptive's did, whenever that migration happens.

Domain-specific knowledge (what a "tuning round" or a "run verdict" means,
how to format a Q/R weight, pricing/cost math) intentionally stays in each
module's own report-composition code, not in this package -- `labcd_pdfmaker`
only knows how to lay out generic report content.
