from __future__ import annotations

# real math typesetting via xelatex -- ported from AgentAdaptive's old
# tools/pdf_report.py, generalized to walk the shared block/markdown model.

import os
import re
import shutil
import subprocess
import tempfile
from typing import List, Optional, Sequence

from .. import blocks as b
from ..markdown import (
    MDBlank, MDBullet, MDDisplayMath, MDHeading, MDParagraph, MDTable, parse_markdown,
)
from ..styles import Palette
from ..utils import wrap_long_lines
from .base import PdfBackend


def xelatex_available() -> bool:
    return shutil.which("xelatex") is not None


_MATH_SPAN_RE = re.compile(r"(\$[^$]*\$)")
_BACKSLASH_PLACEHOLDER = "\x00BACKSLASH\x00"

_ROW_COLOR_HEX = {
    "bad": Palette.ROW_BAD.lstrip("#").upper(),
    "warn": Palette.ROW_WARN.lstrip("#").upper(),
}


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


def _longtable_tex(rows: Sequence[Sequence[str]], ncols: int, cell_fn, row_colors=None) -> str:
    # booktabs style, first row is always the header. col width scales with
    # ncols so a wide table doesn't run off the page.
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
        color = row_colors[r_idx - 1] if (row_colors and r_idx > 0) else None
        if color:
            out.append("\\rowcolor[HTML]{%s}" % _ROW_COLOR_HEX.get(color, color))
        out.append(" & ".join(cells) + " \\\\")
        if r_idx == 0:
            out.append("\\midrule")
            out.append("\\endhead")
    out.append("\\bottomrule")
    out.append("\\end{longtable}")
    out.append("\\endgroup")
    return "\n".join(out)


def _md_nodes_to_tex(nodes) -> str:
    out: List[str] = []
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

    for node in nodes:
        is_quote_para = isinstance(node, MDParagraph) and node.quote
        if is_quote_para and not in_quote:
            close_bullets()
            out.append("\\begin{quote}")
            in_quote = True
        elif not is_quote_para and in_quote:
            close_bullets()
            close_quote()

        if isinstance(node, MDDisplayMath):
            close_bullets()
            out.append("\\[" + node.body + "\\]")
            if node.trailing:
                out.append(_md_inline_to_tex(node.trailing) + "\\par")
        elif isinstance(node, MDHeading):
            close_bullets()
            cmd = "subsection" if node.level == 3 else "subsection" if node.level == 2 else "section"
            out.append("\\%s{%s}" % (cmd, _md_inline_to_tex(node.text)))
        elif isinstance(node, MDTable):
            close_bullets()
            if node.rows:
                ncols = max(len(r) for r in node.rows)
                out.append(_longtable_tex(node.rows, ncols, _md_inline_to_tex))
        elif isinstance(node, MDBullet):
            if not in_bullets:
                out.append("\\begin{itemize}\\setlength\\itemsep{2pt}")
                in_bullets = True
            out.append("\\item " + _md_inline_to_tex(node.text))
        elif isinstance(node, MDBlank):
            close_bullets()
            out.append("")
        elif isinstance(node, MDParagraph):
            close_bullets()
            out.append(_md_inline_to_tex(node.text) + "\\par")

    close_bullets()
    close_quote()
    return "\n".join(out)


def _status_tex(ok: bool, label: str) -> str:
    color = Palette.SUCCESS if ok else Palette.FAIL
    return r"\textcolor{%s}{\textbf{%s}}" % (color, _tex_text(label))


_VERBATIM_GUARD_RE = re.compile(r"\\end\{Verbatim\}")


def _verbatim_block(text: str, fontsize: str) -> str:
    if not text:
        return ""
    size_cmd = {"small": r"\small", "footnotesize": r"\footnotesize",
                "scriptsize": r"\scriptsize"}.get(fontsize, r"\small")
    safe = _VERBATIM_GUARD_RE.sub(lambda m: "\\end\\string{Verbatim\\string}", text)
    safe = wrap_long_lines(safe)
    return (
        "\\begingroup" + size_cmd + "\n"
        "\\begin{Verbatim}\n" + safe + "\n"
        "\\end{Verbatim}\n"
        "\\endgroup"
    )


_PREAMBLE = r"""
\documentclass[10pt]{article}
\usepackage[margin=0.9in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{longtable}
\usepackage{array}
\usepackage{booktabs}
\usepackage{parskip}
\usepackage[dvipsnames,table]{xcolor}
\usepackage{fancyvrb}
\usepackage[hidelinks,bookmarksopen]{hyperref}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\setlength{\parindent}{0pt}
\setcounter{secnumdepth}{2}
\setcounter{tocdepth}{2}
\begin{document}
"""
_POSTAMBLE = r"\end{document}"


class XeLatexBackend(PdfBackend):
    def render(self, block_list: List[b.Block], *, path: Optional[str] = None) -> Optional[bytes]:
        if not xelatex_available():
            raise RuntimeError(
                "xelatex is not installed on this system. Install a LaTeX "
                "distribution that includes XeLaTeX (the 'texlive-xetex' "
                "package on Debian/Ubuntu, or MiKTeX/TeX Live on Windows/macOS)."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            body = [self._render_block(block, tmpdir) for block in block_list]
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

            if path:
                shutil.copyfile(pdf_path, path)
                return None
            with open(pdf_path, "rb") as f:
                return f.read()

    def _render_block(self, block: b.Block, tmpdir: str) -> str:
        if isinstance(block, b.TitlePage):
            parts = [r"\title{%s}" % _tex_text(block.title), r"\author{}"]
            date = block.date or ""
            parts.append(r"\date{%s}" % _tex_text(date))
            parts.append(r"\maketitle")
            if block.subtitle:
                parts.append(r"\begin{center}\textit{%s}\end{center}" % _tex_text(block.subtitle))
            for line in block.meta_lines:
                parts.append(r"\begin{center}\small %s\end{center}" % _tex_text(line))
            return "\n".join(parts)

        if isinstance(block, b.Abstract):
            if not block.markdown.strip():
                return ""
            return "\\section*{Abstract}\n" + _md_nodes_to_tex(parse_markdown(block.markdown.strip()))

        if isinstance(block, b.TableOfContents):
            return r"\tableofcontents\clearpage"

        if isinstance(block, b.Heading):
            cmd = "section" if block.level <= 1 else "subsection"
            return "\\%s{%s}" % (cmd, _tex_text(block.text))

        if isinstance(block, b.Markdown):
            return _md_nodes_to_tex(parse_markdown(block.text))

        if isinstance(block, b.KeyValueTable):
            cell_fn = _md_inline_to_tex if block.markdown_cells else _tex_text
            rows = [["Parameter", "Value"]] + [[str(k), str(v)] for k, v in block.rows]
            parts = []
            if block.title:
                parts.append(r"\subsection*{%s}" % _tex_text(block.title))
            parts.append(_longtable_tex(rows, 2, cell_fn))
            return "\n".join(parts)

        if isinstance(block, b.DataTable):
            cell_fn = _md_inline_to_tex if block.markdown_cells else _tex_text
            rows = [block.headers] + [[str(c) for c in r] for r in block.rows]
            row_colors = None
            if block.row_style:
                row_colors = [block.row_style(r) for r in block.rows]
            parts = []
            if block.title:
                parts.append(r"\subsection*{%s}" % _tex_text(block.title))
            parts.append(_longtable_tex(rows, len(block.headers), cell_fn, row_colors=row_colors))
            return "\n".join(parts)

        if isinstance(block, b.DiffTable):
            rows = [["Parameter", "Before", "After"]]
            for field_name, (old, new) in block.changed.items():
                rows.append([str(field_name), str(old), str(new)])
            parts = []
            if block.title:
                parts.append(r"\subsection*{%s}" % _tex_text(block.title))
            parts.append(_longtable_tex(rows, 3, _md_inline_to_tex))
            return "\n".join(parts)

        if isinstance(block, b.StatusBadge):
            return _status_tex(block.ok, block.label) + r"\par"

        if isinstance(block, b.Figure):
            png_bytes = _figure_to_png_bytes(block.image)
            idx = len([f for f in os.listdir(tmpdir) if f.startswith("plot_")])
            fname = "plot_%d.png" % idx
            with open(os.path.join(tmpdir, fname), "wb") as f:
                f.write(png_bytes)
            caption = ("Figure %d. %s" % (idx + 1, block.caption)) if block.caption else ("Figure %d." % (idx + 1))
            return (
                r"\begin{center}"
                r"\includegraphics[width=0.92\textwidth]{%s}\par" % fname +
                r"\vspace{2pt}\textbf{%s}" % _tex_text(caption) +
                r"\end{center}\vspace{10pt}"
            )

        if isinstance(block, b.Verbatim):
            return _verbatim_block(block.text, block.fontsize)

        if isinstance(block, b.PageBreak):
            return r"\clearpage"

        raise TypeError("Unknown block type: %r" % (block,))


def _figure_to_png_bytes(image) -> bytes:
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)
    # duck-typed matplotlib Figure
    import io
    buf = io.BytesIO()
    image.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    return buf.getvalue()
