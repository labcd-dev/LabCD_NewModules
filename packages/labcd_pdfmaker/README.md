# labcd_pdfmaker

One PDF builder for LabCD modules, two backends:

| Backend | Needs | Good for |
|---|---|---|
| `Backend.REPORTLAB` | just `reportlab` (already a hard dep) | no-math reports, runs anywhere |
| `Backend.XELATEX` | a system `xelatex` (TeX Live / MiKTeX) | real math typesetting |
| `Backend.AUTO` (default) | either | xelatex if the report has math *and* it's on `PATH`, else reportlab |

Replaces AgentAdaptive's old `tools/pdf_report.py`. AgentMPC still runs its
own `agents/report_pdf.py` (reportlab) -- not migrated yet, but the package
was built to fit its report shape too, for whenever that happens.

## Install

```bash
pip install -e "packages/labcd_pdfmaker"
```

The xelatex backend needs a system LaTeX install (see repo-root
`packages.txt`) -- nothing extra to `pip install`.

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

`ReportBuilder` just accumulates a flat list of blocks (`labcd_pdfmaker.blocks`)
-- title page, abstract, TOC, headings, Markdown text, key-value/data/diff
tables, status badges, figures, verbatim blocks -- and each backend
(`reportlab_backend` / `xelatex_backend`) renders that same list its own way.

On reportlab, `$...$` math spans render as plain monospace text instead of
real typesetting, and `build()` raises a `UserWarning` when that happens --
no silent degradation.

## Migration notes

- **AgentAdaptive** (`tools/report.py`) uses `Backend.AUTO`. Replaces the
  old `tools/pdf_report.py`.
- **AgentMPC** hasn't moved over yet -- still on `agents/report_pdf.py`.
  Its sections (System Analysis, Controller Analysis, ...) would map onto
  `add_section` / `add_key_value_table` / `add_data_table` the same way.

Report-specific knowhow (what a "tuning round" means, Q/R formatting, cost
math, ...) stays in each module's own composition code -- this package only
lays out generic content.
