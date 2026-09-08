import datetime
import io
import os
import re
import shutil
import subprocess
import tempfile

from . import model_pricing


def latex_available() -> bool:
    return shutil.which("xelatex") is not None

_MATH_SPAN_RE = re.compile(r"(\$[^$]*\$)")
_BACKSLASH_PLACEHOLDER = "\x00BACKSLASH\x00"


def _escape_latex_text(segment: str) -> str:
    segment = segment.replace("\\", _BACKSLASH_PLACEHOLDER)
    for ch, esc in (
        ("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"),
        ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ):
        segment = segment.replace(ch, esc)
    return segment.replace(_BACKSLASH_PLACEHOLDER, r"\textbackslash{}")


def _md_inline_to_tex(text: str) -> str:
    parts = _MATH_SPAN_RE.split(text)
    out = []
    for part in parts:
        if part.startswith("$") and part.endswith("$") and len(part) > 1:
            out.append(part)  # math span, passed through untouched
            continue
        part = re.sub(r"`(.+?)`", lambda m: "\x01" + m.group(1) + "\x02", part)
        part = re.sub(r"\*\*(.+?)\*\*", lambda m: "\x03" + m.group(1) + "\x04", part)
        part = re.sub(r"\*(.+?)\*", lambda m: "\x05" + m.group(1) + "\x06", part)
        part = _escape_latex_text(part)
        part = part.replace("\x01", r"\texttt{").replace("\x02", "}")
        part = part.replace("\x03", r"\textbf{").replace("\x04", "}")
        part = part.replace("\x05", r"\textit{").replace("\x06", "}")
        out.append(part)
    return "".join(out)


def _tex_text(text: str) -> str:
    return _escape_latex_text(text or "")


def _parse_markdown_table(lines, start_idx):
    rows = []
    i = start_idx
    while i < len(lines) and lines[i].strip().startswith("|"):
        row = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        rows.append(row)
        i += 1
    if len(rows) >= 2 and re.match(r"^:?-+:?$", rows[1][0].replace(" ", "")):
        rows.pop(1)

    ncols = max(len(r) for r in rows)
    tex = "\n".join(_longtable_tex(rows, ncols, lambda c: _md_inline_to_tex(c)))
    return tex, i


def _longtable_tex(rows, ncols, cell_fn):
    # booktabs style, first row is always the header. col width scales
    # with ncols so a wide table doesn't run off the page.
    col_width = min(0.32, 0.90 / max(ncols, 1))
    colspec = ("L{%.3f\\textwidth}" % col_width) * ncols
    out = ["\\begingroup\\small",
           "\\renewcommand{\\arraystretch}{1.5}",
           "\\setlength{\\tabcolsep}{10pt}",
           "\\begin{longtable}{%s}" % colspec,
           "\\toprule"]
    for r_idx, row in enumerate(rows):
        cells = [cell_fn(c) for c in row] + [""] * (ncols - len(row))
        if r_idx == 0:
            cells = ["\\textbf{%s}" % c for c in cells]
        out.append(" & ".join(cells) + " \\\\")
        if r_idx == 0:
            out.append("\\midrule")
            out.append("\\endhead")
    out.append("\\bottomrule")
    out.append("\\end{longtable}")
    out.append("\\endgroup")
    return out


def markdown_to_latex_body(markdown_text: str) -> str:
    raw_lines = markdown_text.split("\n")
    lines = []
    is_quote = []
    for raw in raw_lines:
        s = raw.strip()
        if s.startswith(">"):
            is_quote.append(True)
            content = s[1:]
            if content.startswith(" "):
                content = content[1:]
            lines.append(content)
        else:
            is_quote.append(False)
            lines.append(raw)

    out = []
    i = 0
    in_bullets = False
    in_quote = False

    def close_bullets():
        nonlocal in_bullets
        if in_bullets:
            out.append("\\end{itemize}")
            in_bullets = False

    def close_quote():
        nonlocal in_quote
        if in_quote:
            out.append("\\end{quote}")
            in_quote = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if is_quote[i] and not in_quote:
            close_bullets()
            out.append("\\begin{quote}")
            in_quote = True
        elif not is_quote[i] and in_quote:
            close_bullets()
            close_quote()

        if stripped.startswith("$$"):
            inner = stripped[2:]
            trailing = ""
            if "$$" in inner:
                close_idx = inner.index("$$")
                body = inner[:close_idx]
                trailing = inner[close_idx + 2:].strip()
                i += 1
            else:
                buf = [inner]
                j = i + 1
                while j < len(lines) and "$$" not in lines[j]:
                    buf.append(lines[j])
                    j += 1
                if j < len(lines):
                    close_idx = lines[j].index("$$")
                    buf.append(lines[j][:close_idx])
                    trailing = lines[j][close_idx + 2:].strip()
                    i = j + 1
                else:
                    i = j
                body = " ".join(buf)
            close_bullets()
            out.append("\\[" + body + "\\]")
            if trailing:
                out.append(_md_inline_to_tex(trailing) + "\\par")
            continue

        if stripped.startswith("## "):
            close_bullets()
            out.append("\\section{%s}" % _md_inline_to_tex(stripped[3:]))
            i += 1
            continue
        if stripped.startswith("### "):
            close_bullets()
            out.append("\\subsection{%s}" % _md_inline_to_tex(stripped[4:]))
            i += 1
            continue

        if stripped.startswith("|"):
            close_bullets()
            table_tex, i = _parse_markdown_table(lines, i)
            out.append(table_tex)
            continue

        if stripped.startswith("- ") or stripped.startswith("* "):
            if not in_bullets:
                out.append("\\begin{itemize}\\setlength\\itemsep{2pt}")
                in_bullets = True
            out.append("\\item " + _md_inline_to_tex(stripped[2:]))
            i += 1
            continue

        close_bullets()
        if not stripped:
            out.append("")
        else:
            out.append(_md_inline_to_tex(stripped) + "\\par")
        i += 1

    close_bullets()
    close_quote()
    return "\n".join(out)

def _diff_table_tex(changed: dict) -> str:
    if not changed:
        return ""
    rows = [["Parameter", "Before", "After"]]
    for field, (old, new) in changed.items():
        rows.append([str(field), str(old), str(new)])
    return "\n".join(_longtable_tex(rows, 3, lambda c: _md_inline_to_tex(c)))


_VERBATIM_GUARD_RE = re.compile(r"\\end\{Verbatim\}")
_WRAP_WIDTH = 100


def _wrap_long_lines(text: str, width: int = _WRAP_WIDTH) -> str:
    # plain texlive's fancyvrb has no breaklines key, so a long log line
    # can run off the page and abort xelatex. Wrap it here instead.
    out_lines = []
    for line in text.split("\n"):
        while len(line) > width:
            out_lines.append(line[:width])
            line = line[width:]
        out_lines.append(line)
    return "\n".join(out_lines)


def _verbatim_block(text: str, fontsize: str = r"\small") -> str:
    if not text:
        return ""
    safe = _VERBATIM_GUARD_RE.sub(lambda m: "\\end\\string{Verbatim\\string}", text)
    safe = _wrap_long_lines(safe)
    return (
        "\\begingroup" + fontsize + "\n"
        "\\begin{Verbatim}\n"
        + safe + "\n"
        "\\end{Verbatim}\n"
        "\\endgroup"
    )


def _status_tex(ok: bool, label: str) -> str:
    color = "ForestGreen" if ok else "BrickRed"
    return r"\textcolor{%s}{\textbf{%s}}" % (color, _tex_text(label))


def _pct_cell(value) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if f != f:  # NaN
        return "n/a"
    return "%.1f\\%%" % f


def _render_abstract_section(abstract_markdown) -> str:
    if not abstract_markdown or not abstract_markdown.strip():
        return ""
    return "\\section*{Abstract}\n" + markdown_to_latex_body(abstract_markdown.strip())


def _render_run_scores_section(final_metrics) -> str:
    if not isinstance(final_metrics, dict):
        return ""
    has_success = "success" in final_metrics
    has_tracking = "tracking_pct_headline" in final_metrics
    if not has_success and not has_tracking:
        return ""

    parts = [r"\section{Run Verdict and Reference Tracking}"]

    if has_success:
        success = bool(final_metrics.get("success"))
        if success:
            parts.append(_status_tex(True, "RUN VERDICT: PASS") + r"\par")
            target_frac = final_metrics.get("success_target_frac")
            # leave "%" unescaped here: it goes through _tex_text right
            # after, escaping it twice would double-escape the backslash
            target_str = ("%.1f%%" % (100.0 * target_frac)
                          if isinstance(target_frac, (int, float)) else "the target")
            parts.append(_tex_text(
                "The closed loop stayed finite and bounded, and every output's "
                "steady-state MSE reached the user's target (%s of the task "
                "scale)." % target_str) + r"\par")
        else:
            reason = str(final_metrics.get("success_reason")
                         or "one of the run checks failed")
            parts.append(_status_tex(False, "RUN VERDICT: FAIL: " + reason) + r"\par")
        checks = final_metrics.get("success_checks") or {}
        if isinstance(checks, dict) and checks:
            rows = [["Check", "Result"]]
            for name, ok in checks.items():
                rows.append([str(name), "pass" if ok else "FAIL"])
            parts.append("\n".join(_longtable_tex(rows, 2, lambda c: _tex_text(c))))

    if has_tracking:
        headline = final_metrics.get("tracking_pct_headline")
        mean = final_metrics.get("tracking_pct_mean")
        parts.append(r"\textbf{Reference tracking: %s} (worst output, steady state)"
                      % _pct_cell(headline))
        if mean is not None:
            parts.append(r"~--~%s mean over outputs." % _pct_cell(mean))
        parts.append(r"\par")
        parts.append(_tex_text(
            "Scored against the RMS error a null controller (one that left "
            "the output frozen at its initial value) would have produced "
            "over the same reference. 100% means the output sits exactly on "
            "the reference; 0% means it tracked no better than not acting at "
            "all. This normalizer, rather than the peak reference magnitude, "
            "is what keeps the score honest for a reference that oscillates "
            "by a little around a large offset, and defined at all for pure "
            "regulation to zero.") + r"\par")

        pct = final_metrics.get("tracking_pct")
        pct = pct if isinstance(pct, dict) else {}
        full = list(pct.get("full") or [])
        steady = list(pct.get("steady") or [])
        transient = list(pct.get("transient") or [])
        trivial = list(final_metrics.get("task_trivial") or [])
        n = max(len(full), len(steady), len(transient))
        if n:
            header = ["Output", "Full run", "Steady state (last 20\\%)",
                      "Transient (first 20\\%)"]
            show_trivial = any(trivial)
            if show_trivial:
                # only added when it happened, since a "stay put" task's
                # unlabelled 100% would otherwise look like a real tracking win
                header.append("Task")
            rows = [header]
            for i in range(n):
                row = ["y%d" % (i + 1),
                       _pct_cell(full[i] if i < len(full) else None),
                       _pct_cell(steady[i] if i < len(steady) else None),
                       _pct_cell(transient[i] if i < len(transient) else None)]
                if show_trivial:
                    row.append("stay put" if (i < len(trivial) and trivial[i])
                               else "tracking")
                rows.append(row)
            # cells are pre-escaped above, so pass through untouched instead
            # of through _tex_text (which would double-escape the % signs)
            parts.append("\n".join(_longtable_tex(rows, len(header), lambda c: c)))

    parts.append(_tex_text(
        "The verdict answers “does this design work at all”; the tuning "
        "target reported in the Parameter Tuning section answers the separate "
        "and stricter question of whether it is good enough. A run can pass "
        "here and still miss that target.") + r"\par")
    return "\n".join(parts)


def _render_clarification_section(clarification_record) -> str:
    if not clarification_record:
        return ""

    rows = [["Question", "Answer", "Source"]]
    for entry in clarification_record:
        if not isinstance(entry, dict):
            continue
        question = str(entry.get("question") or entry.get("id") or "")
        answer = str(entry.get("answer_label") or entry.get("answer_value")
                     or entry.get("default_label") or "")
        # 3 sources, not 2 (text read straight from the user's sentence isn't
        # an "assumed default"): it needs its own bucket, cited in this cell
        source = str(entry.get("source")
                     or ("user" if entry.get("answered") else "default"))
        if source == "description":
            evidence = str(entry.get("evidence") or "").strip()
            source = ("read from your description: “%s”" % evidence
                      if evidence else "read from your description")
        elif source == "user":
            source = "user"
        else:
            source = "assumed default"
        rows.append([question, answer, source])

    if len(rows) == 1:
        return ""

    parts = [r"\section{Clarifications}"]
    parts.append(
        "Before extraction, the Clarifier Agent asked about the parts of the "
        "system description that could be read more than one way. The answers "
        "below were treated as authoritative by every later stage. Rows marked "
        "\\textit{assumed default} were not answered by the user. The stated "
        "default was applied instead, so those are the assumptions the design "
        "depends on. Rows marked \\textit{read from your description} were "
        "never put to the user at all: the description already settled them, "
        "and the sentence it was read from is quoted so the reading can be "
        "checked.\\par"
    )
    # this is _tex_text, not _md_inline_to_tex. Plain user text means an
    # asterisk should print literally, not turn into markdown emphasis
    parts.append("\n".join(_longtable_tex(rows, 3, lambda c: _tex_text(c))))
    return "\n".join(parts)


def _render_tuning_section(tuning_log, tuning_best) -> str:
    if not tuning_log:
        return ""
    parts = [r"\section{Parameter Tuning (Tuner Agent)}"]
    parts.append(
        "The Tuner Agent never sees or changes the system description, states, "
        "dynamics, or has\\_delta/has\\_disturbance; it only reads the "
        "simulation metrics and proposes new tuning parameters, using its "
        "own model/API call, separate from the Design Agent.\\par"
    )

    if len(tuning_log) == 1 and tuning_log[0]["round"] == 0:
        r0 = tuning_log[0].get("reasoning") or ""
        if r0 and r0 != "(initial design, before tuning)":
            parts.append(_md_inline_to_tex(r0) + r"\par")

    if tuning_best is not None:
        best_round = tuning_best["round"]
        best_entry = next((e for e in tuning_log if e["round"] == best_round), None)
        if best_round == 0:
            parts.append("Tuning made no improvement: the Design Agent's original design "
                          "(round 0) is still the best result.\\par")
        else:
            met = bool(best_entry and best_entry.get("met_target"))
            parts.append(_status_tex(met,
                "Target met." if met else
                "Best result reached after %d round(s); target not fully met, "
                "best attempt kept." % best_round) + r"\par")
            if best_entry and best_entry.get("reasoning"):
                parts.append(r"\textbf{Tuner Agent's reasoning for this final result:} "
                              + _md_inline_to_tex(best_entry["reasoning"]) + r"\par")

        parts.append(r"\subsection*{Final tuning parameter values}")
        rows = [["Parameter", "Value"]] + [
            [str(k), str(v)] for k, v in tuning_best["tuning"].items()
        ]
        parts.append("\n".join(_longtable_tex(rows, 2, lambda c: _md_inline_to_tex(c))))

        if best_entry and best_entry.get("report"):
            parts.append(_verbatim_block(best_entry["report"]))

    if len(tuning_log) > 1:
        parts.append(r"\subsection*{Full tuning round history}")
        for entry in tuning_log:
            label = "Round %d" % entry["round"]
            label += " (Design Agent's original design)" if entry["round"] == 0 else " (Tuner Agent's proposal)"
            if entry.get("met_target"):
                label += ", target met"
            parts.append(r"\textbf{%s}\par" % _tex_text(label))

            reasoning = entry.get("reasoning")
            if reasoning and reasoning != "(initial design, before tuning)":
                parts.append(_md_inline_to_tex(reasoning) + r"\par")

            changed = entry.get("changed") or {}
            if entry["round"] > 0:
                if changed:
                    parts.append(_diff_table_tex(changed))
                else:
                    parts.append(r"\textit{No tuning parameters were changed this round.}\par")

            if entry.get("report"):
                parts.append(_verbatim_block(entry["report"], fontsize=r"\footnotesize"))
            parts.append(r"\vspace{4pt}")

    return "\n".join(parts)


def _render_usage_section(usage) -> str:
    if not usage:
        return ""
    cost_rows, cost_total = model_pricing.run_cost_rows(usage)
    if not cost_rows:
        return ""

    # 5 cols, not 6 (a cached-tokens column would overflow, so it rides in the
    # input cell). No "$" in the cost column either, since that's LaTeX math mode.
    rows = [["Agent", "Model", "Input tokens", "Output tokens", "Cost (USD)"]]
    for r in cost_rows:
        cached = r["cached_input_tokens"]
        input_cell = format(r["input_tokens"], ",")
        if cached:
            input_cell += " (%s cached)" % format(cached, ",")
        rows.append([r["label"], r["model"], input_cell,
                     format(r["output_tokens"], ","),
                     model_pricing.format_cost_plain(r["cost"])])
    rows.append(["Total", "", "",
                 format((usage.get("total") or {}).get("total_tokens", 0), ","),
                 model_pricing.format_cost_plain(cost_total)])

    parts = [r"\section{Token Usage and Cost}"]
    parts.append("\n".join(_longtable_tex(rows, 5, lambda c: _md_inline_to_tex(c))))
    parts.append(_tex_text(
        "Cached input is the portion served from the provider's prompt cache "
        "and billed at a reduced rate; it is already counted inside the input "
        "column. Figures are an estimate based on the tokens this run "
        "reported."))
    return "\n".join(parts)


def _normalize_figure(fig, index):
    if isinstance(fig, (bytes, bytearray)):
        return bytes(fig), ""
    return fig["png"], fig.get("title") or ""


def _render_figures_section(figures, tmpdir) -> str:
    if not figures:
        return ""
    parts = [r"\clearpage\section{Plots}"]
    for i, fig in enumerate(figures):
        png_bytes, title = _normalize_figure(fig, i)
        fname = "plot_%d.png" % i
        with open(os.path.join(tmpdir, fname), "wb") as f:
            f.write(png_bytes)
        caption = ("Figure %d. %s" % (i + 1, title)) if title else ("Figure %d." % (i + 1))
        parts.append(r"\begin{center}")
        parts.append(r"\includegraphics[width=0.92\textwidth]{%s}\par" % fname)
        parts.append(r"\vspace{2pt}\textbf{%s}" % _tex_text(caption))
        parts.append(r"\end{center}\vspace{10pt}")
    return "\n".join(parts)


_PREAMBLE = r"""
\documentclass[10pt]{article}
\usepackage[margin=0.9in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{longtable}
\usepackage{array}
\usepackage{booktabs}
\usepackage{parskip}
\usepackage[dvipsnames]{xcolor}
\usepackage{fancyvrb}
\usepackage[hidelinks,bookmarksopen]{hyperref}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\setlength{\parindent}{0pt}
\setcounter{secnumdepth}{2}
\setcounter{tocdepth}{2}
\begin{document}
"""

_POSTAMBLE = r"\end{document}"


def build_pdf_report(summary_markdown: str, figures,
                      usage=None, log_text=None, tuning_log=None,
                      tuning_best=None, clarification_record=None,
                      final_metrics=None, abstract_markdown=None) -> bytes:
    # builds the .tex and compiles it with xelatex - raises RuntimeError (with the
    # compiler log attached) instead of silently handing back a broken PDF.

    if not latex_available():
        raise RuntimeError(
            "xelatex is not installed on this system. Install a LaTeX "
            "distribution that includes XeLaTeX (the 'texlive-xetex' "
            "package on Debian/Ubuntu, or MiKTeX/TeX Live on Windows/macOS)."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        body = []
        body.append(r"\title{Agentic Nonlinear Control Designer: Report}")
        body.append(r"\author{}")
        body.append(r"\date{%s}" % _tex_text(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
        body.append(r"\maketitle")

        # abstract goes ahead of the table of contents, same spot it'd take
        # in any engineering report. Omitted entirely if none was produced
        abstract_tex = _render_abstract_section(abstract_markdown)
        if abstract_tex:
            body.append(abstract_tex)

        body.append(r"\tableofcontents")
        body.append(r"\clearpage")

        # run verdict goes right after the contents, ahead of the design
        # summary, since whether it worked at all is the reader's entry point
        scores_tex = _render_run_scores_section(final_metrics)
        if scores_tex:
            body.append(scores_tex)

        body.append(r"\section{Design Summary}")
        body.append(markdown_to_latex_body(summary_markdown or ""))

        # clarifications sit right after the summary, since they're what the
        # summary was derived from
        clarification_tex = _render_clarification_section(clarification_record)
        if clarification_tex:
            body.append(clarification_tex)

        tuning_tex = _render_tuning_section(tuning_log, tuning_best)
        if tuning_tex:
            body.append(r"\clearpage")
            body.append(tuning_tex)

        usage_tex = _render_usage_section(usage)
        if usage_tex:
            body.append(usage_tex)

        body.append(_render_figures_section(figures, tmpdir))

        if log_text:
            body.append(r"\clearpage\section{Appendix: Full Console Log}")
            body.append(_verbatim_block(log_text, fontsize=r"\scriptsize"))

        tex_source = _PREAMBLE + "\n".join(body) + "\n" + _POSTAMBLE

        tex_path = os.path.join(tmpdir, "report.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex_source)
        log_output = ""
        returncode = 1
        for _ in range(3):
            proc = subprocess.run(
                ["xelatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
                cwd=tmpdir, capture_output=True, timeout=180,
                encoding="utf-8", errors="replace",
            )
            log_output = proc.stdout + proc.stderr
            returncode = proc.returncode

        pdf_path = os.path.join(tmpdir, "report.pdf")
        if returncode != 0 or not os.path.exists(pdf_path):
            raise RuntimeError(
                "PDF compilation failed (xelatex exit code %d). The PDF, if "
                "any, would be missing content after the error point. "
                "xelatex log (last 4000 chars):\n\n%s"
                % (returncode, log_output[-4000:])
            )

        with open(pdf_path, "rb") as f:
            return f.read()
