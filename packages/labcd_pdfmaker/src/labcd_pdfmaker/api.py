from __future__ import annotations

import shutil
import warnings
from enum import Enum
from typing import Any, Callable, List, Optional, Sequence, Tuple

from . import blocks as b
from .backends.base import PdfBackend


class Backend(Enum):
    REPORTLAB = "reportlab"
    XELATEX = "xelatex"
    AUTO = "auto"


def xelatex_available() -> bool:
    return shutil.which("xelatex") is not None


class ReportBuilder:
    """Accumulates report content, then renders it with one of the two
    backends. Content methods return ``self`` so calls can be chained.

    ``backend=Backend.AUTO`` (the default) picks xelatex when the report
    contains LaTeX math (``$...$`` / ``$$...$$``) *and* xelatex is on
    PATH, and reportlab otherwise -- so a report with no math never pays
    for a TeX distribution it doesn't need, while a math-heavy one still
    gets real typesetting when the toolchain is available.
    """

    def __init__(self, title: str, subtitle: str = "", *, backend: Backend = Backend.AUTO,
                 meta_lines: Optional[Sequence[str]] = None, date: Optional[str] = None):
        self._backend_choice = backend
        self._needs_math = False
        self._blocks: List[b.Block] = [
            b.TitlePage(title=title, subtitle=subtitle, meta_lines=tuple(meta_lines or ()), date=date)
        ]

    def add_abstract(self, markdown: str) -> "ReportBuilder":
        if markdown and markdown.strip():
            self._note_math(markdown)
            self._blocks.append(b.Abstract(markdown))
        return self

    def add_table_of_contents(self) -> "ReportBuilder":
        self._blocks.append(b.TableOfContents())
        return self

    def add_section(self, title: str, markdown: Optional[str] = None) -> "ReportBuilder":
        self._blocks.append(b.Heading(title, level=1))
        if markdown:
            self.add_markdown(markdown)
        return self

    def add_subsection(self, title: str, markdown: Optional[str] = None) -> "ReportBuilder":
        self._blocks.append(b.Heading(title, level=2))
        if markdown:
            self.add_markdown(markdown)
        return self

    def add_markdown(self, text: Optional[str]) -> "ReportBuilder":
        if not text:
            return self
        self._note_math(text)
        self._blocks.append(b.Markdown(text))
        return self

    def add_math_display(self, latex: str) -> "ReportBuilder":
        self._needs_math = True
        self._blocks.append(b.Markdown("$$%s$$" % latex))
        return self

    def add_key_value_table(self, rows: Sequence[Tuple[str, str]], *,
                             title: Optional[str] = None, markdown_cells: bool = True) -> "ReportBuilder":
        self._blocks.append(b.KeyValueTable(rows=list(rows), title=title, markdown_cells=markdown_cells))
        return self

    def add_data_table(self, headers: Sequence[str], rows: Sequence[Sequence[Any]], *,
                        title: Optional[str] = None, markdown_cells: bool = True,
                        row_style: Optional[Callable[[Sequence[Any]], Optional[str]]] = None
                        ) -> "ReportBuilder":
        self._blocks.append(b.DataTable(
            headers=list(headers), rows=[list(r) for r in rows],
            title=title, row_style=row_style, markdown_cells=markdown_cells,
        ))
        return self

    def add_diff_table(self, changed: dict, *, title: Optional[str] = None) -> "ReportBuilder":
        if changed:
            self._blocks.append(b.DiffTable(changed=changed, title=title))
        return self

    def add_status_badge(self, ok: bool, label: str) -> "ReportBuilder":
        self._blocks.append(b.StatusBadge(ok=ok, label=label))
        return self

    def add_figure(self, image: Any, caption: Optional[str] = None) -> "ReportBuilder":
        self._blocks.append(b.Figure(image=image, caption=caption))
        return self

    def add_verbatim(self, text: Optional[str], *, fontsize: str = "small") -> "ReportBuilder":
        if text:
            self._blocks.append(b.Verbatim(text=text, fontsize=fontsize))
        return self

    def add_page_break(self) -> "ReportBuilder":
        self._blocks.append(b.PageBreak())
        return self

    def _note_math(self, text: str) -> None:
        if "$" in text:
            self._needs_math = True

    def _resolve_backend(self) -> Backend:
        if self._backend_choice == Backend.AUTO:
            return Backend.XELATEX if (self._needs_math and xelatex_available()) else Backend.REPORTLAB
        return self._backend_choice

    def _make_backend(self, resolved: Backend) -> PdfBackend:
        if resolved == Backend.XELATEX:
            if not xelatex_available():
                raise RuntimeError(
                    "xelatex is not installed on this system. Install a LaTeX "
                    "distribution that includes XeLaTeX (the 'texlive-xetex' "
                    "package on Debian/Ubuntu, or MiKTeX/TeX Live on Windows/macOS)."
                )
            from .backends.xelatex_backend import XeLatexBackend
            return XeLatexBackend()
        from .backends.reportlab_backend import ReportLabBackend
        return ReportLabBackend()

    def build(self, path: Optional[str] = None) -> Optional[bytes]:
        """Renders the accumulated content. Returns PDF bytes if ``path``
        is ``None``, otherwise writes to ``path`` and returns ``None``."""
        resolved = self._resolve_backend()
        if resolved == Backend.REPORTLAB and self._needs_math:
            warnings.warn(
                "This report contains LaTeX math ($...$ / $$...$$) but is being "
                "rendered with the reportlab backend, which has no math "
                "typesetting -- math spans will show as literal monospace text. "
                "Use Backend.XELATEX (or leave backend=Backend.AUTO with xelatex "
                "installed) for real math rendering.",
                stacklevel=2,
            )
        backend = self._make_backend(resolved)
        return backend.render(self._blocks, path=path)
