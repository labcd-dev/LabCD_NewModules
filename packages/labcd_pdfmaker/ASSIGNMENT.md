# Assignment: Unified LabCD PDF Maker (`labcd_pdfmaker`)

## Goal

Create a **single, shared Python package** (`packages/labcd_pdfmaker`) that replaces the two independent PDF-generation implementations currently living inside:

| Module | Current file | Backend |
|--------|--------------|---------|
| **AgentMPC** | `backend_core/AgentMPC/agents/report_pdf.py` | reportlab (pure Python) |
| **AgentAdaptive** | `backend_core/AgentAdaptive/tools/pdf_report.py` | XeLaTeX (`xelatex`) |

The new package must satisfy **all** requirements of both modules (and be extensible for future LabCD agents) so that MPC, Adaptive, and any future module can produce professional PDF reports through one common API.

---

## Why this exists

Today the two report generators diverge in critical ways:

- **AgentMPC** needs a dependency-light, pure-Python solution (reportlab) because it often runs on the end-user’s machine via Streamlit. It produces structured sections, styled tables, and embedded matplotlib figures. It has **no** real math typesetting.
- **AgentAdaptive** needs high-quality mathematical typesetting (inline `$...$` and display `$$...$$`), Markdown → LaTeX conversion, longtables, booktabs styling, coloured status text, and compilation via `xelatex`. It returns PDF **bytes**.

A unified package must support both worlds (and hybrid usage) without forcing every consumer to pull in a full TeX distribution when it is not needed.

---

## Functional requirements

The package must provide a clean public API that can produce a complete PDF report from a structured description of content. At minimum it must support:

### 1. Document structure
- Title, subtitle, author, date
- Hierarchical headings (H1 / H2 / section / subsection)
- Page breaks / clear-page
- Optional table of contents
- Optional abstract / front-matter section
- Consistent visual identity (LabCD colour palette, typography)

### 2. Text & Markdown
- Plain paragraphs (justified or left-aligned)
- Full Markdown subset currently used by AgentAdaptive:
  - Headings (`##`, `###`)
  - Bold / italic / inline code
  - Bullet lists
  - Block quotes
  - Markdown tables
- Safe escaping of special characters

### 3. Mathematics (critical for AgentAdaptive)
- Inline math: `$...$`
- Display math: `$$...$$` (or `\[ ... \]`)
- Must render correctly when the LaTeX backend is active
- Graceful degradation (or clear error) when only the pure-Python backend is available

### 4. Tables
- Simple key-value tables (parameter tables, metrics tables) — matching the styled look of AgentMPC (`Times` fonts, header background `#2f5597`, alternating row colours)
- Multi-column data tables (iteration history, tuning history, cost tables)
- Support for:
  - Header row styling
  - Conditional row colouring (e.g. UNSTABLE / FAILED rows)
  - Column width control / auto-scaling
  - Long tables that can span pages (when using LaTeX backend)

### 5. Figures / images
- Accept matplotlib `Figure` objects **or** raw PNG bytes
- Automatic scaling to page width while preserving aspect ratio
- Captions under each figure
- Multiple figures in a dedicated “Plots” section

### 6. Special content blocks
- Status badges / coloured labels (`PASS` / `FAIL`, green / red)
- Verbatim / console-log blocks (with line wrapping so long lines do not overflow)
- Diff-style “before → after” parameter tables
- Token-usage / cost tables

### 7. Output modes
The package must support **at least two backends**:

| Backend | When to use | Output | External dependency |
|---------|-------------|--------|---------------------|
| **reportlab** | AgentMPC, lightweight deployments, no TeX installed | write to file path **or** return bytes | none beyond `reportlab` |
| **xelatex** | AgentAdaptive, any report that needs real math | return bytes (preferred) or write to path | `xelatex` must be available |

A single high-level entry point should choose the backend (explicitly or via auto-detection) and produce equivalent visual results as far as the backend allows.

### 8. API shape (suggested)

Something along these lines (exact design is left to the implementer, but the spirit should be preserved):

```python
from labcd_pdfmaker import ReportBuilder, Backend

builder = ReportBuilder(
    title="MPC Auto-Tuning Report",
    subtitle=system_name,
    backend=Backend.REPORTLAB,   # or Backend.XELATEX / Backend.AUTO
)

builder.add_title_page(...)
builder.add_section("System Analysis", markdown_or_text)
builder.add_table(headers, rows, style="metrics")
builder.add_figure(fig_or_png_bytes, caption="...")
builder.add_math_display(r"\dot{x} = Ax + Bu")
...

pdf_bytes = builder.build()          # or builder.build(path="report.pdf")
```

Lower-level helpers for the common LabCD tables (parameter table, metrics table, history table, cost table, etc.) are highly desirable so callers do not re-implement styling.

---

## Non-functional requirements

- **Zero hard dependency on TeX** for the reportlab path. AgentMPC must continue to work on machines that only have the packages listed in the project’s `requirements.txt`.
- Pure-Python code preferred; any subprocess usage must be isolated behind the LaTeX backend.
- Deterministic, reproducible output given the same inputs.
- Clear, actionable error messages when the chosen backend cannot satisfy a requested feature (e.g. math requested on reportlab-only install).
- Package lives under `packages/labcd_pdfmaker` and is importable as `labcd_pdfmaker`.
- Type hints, docstrings, and a short usage example in the package README.
- Unit tests that cover both backends (LaTeX tests may be skipped when `xelatex` is absent).

---

## Reference implementations (read these carefully)

### AgentMPC – `backend_core/AgentMPC/agents/report_pdf.py`
- Uses reportlab `SimpleDocTemplate`, custom `ParagraphStyle`s (Times family).
- Three table helpers: `_param_table`, `_metrics_table`, `_history_table` with specific colours and conditional row backgrounds.
- Matplotlib figures converted in-memory via `BytesIO` → reportlab `Image`.
- Hard-coded section structure driven by a report-agent analysis object.

### AgentAdaptive – `backend_core/AgentAdaptive/tools/pdf_report.py`
- Full Markdown → LaTeX pipeline (`markdown_to_latex_body`) with math preservation.
- Longtable + booktabs styling.
- Rich section renderers: abstract, run scores / tracking metrics, clarifications, tuning history (with diffs), token cost, figures, appendix log.
- Compiles with `xelatex` inside a temporary directory and returns PDF bytes.
- Requires `amsmath`, `graphicx`, `longtable`, `booktabs`, `xcolor`, `fancyvrb`, `hyperref`.

The new package should absorb the *capabilities* of both files, not merely copy their internal structure.

---

## Deliverables

1. Package skeleton under `packages/labcd_pdfmaker/`:
   - `__init__.py` exposing the public API
   - Core builder / document model
   - Backend implementations (reportlab + xelatex)
   - Shared styling constants (colours, fonts, page geometry)
   - Helpers for the common LabCD table types
2. `README.md` with installation notes, backend requirements, and a minimal end-to-end example.
3. Basic test suite (pytest) demonstrating both backends.
4. Migration notes (short) describing how AgentMPC and AgentAdaptive should switch from their private modules to `labcd_pdfmaker`.

---

## Out of scope (for this assignment)

- Changing the content / wording of the existing reports.
- Replacing the Streamlit UI or the report-agent logic itself.
- Supporting engines other than reportlab and xelatex (e.g. WeasyPrint, Playwright) — these may be considered later.

---

## Success criteria

- Both existing modules can be migrated to the new package with **no loss of visual quality or features**.
- A future third agent can produce a PDF report by constructing a `ReportBuilder` without touching reportlab or LaTeX internals.
- The pure-Python path remains installable and usable without a TeX distribution.
- Math-heavy Adaptive reports continue to render correctly via the xelatex backend.

---

*This assignment is the single source of truth for the design of `labcd_pdfmaker`. Implementers should treat the two current PDF modules as living specifications of required behaviour.*
