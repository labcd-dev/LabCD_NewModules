"""Pure-Python backend built on reportlab.

Ported from AgentMPC's original ``agents/report_pdf.py`` (Times-family
styling, styled 2-col/multi-col tables with conditional row colouring,
embedded matplotlib figures) and extended with a Markdown renderer and an
optional table of contents so it can also serve reports that used to be
LaTeX-only, minus real math typesetting (see the module-level note on
``MDDisplayMath`` below).

No system dependency beyond the ``reportlab`` pip package -- this is the
backend every LabCD module can rely on being available, since it (unlike
xelatex) doesn't depend on what's installed on the machine running the
Streamlit app.
"""

from __future__ import annotations

import io
import re
from typing import Any, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image, ListFlowable, ListItem, PageBreak, Paragraph, Preformatted,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents as RLTableOfContents

from .. import blocks as b
from ..markdown import (
    MDBlank, MDBullet, MDDisplayMath, MDHeading, MDParagraph, MDTable, parse_markdown,
)
from ..styles import FAIL_HEX, Palette, SUCCESS_HEX
from ..utils import wrap_long_lines
from .base import PdfBackend

USABLE_WIDTH_IN = 6.7  # letter width minus 2 * 0.9in margins


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("ReportTitle", parent=base["Title"], fontName="Times-Bold",
                                 fontSize=26, spaceAfter=6, textColor=colors.HexColor(Palette.NAVY)),
        "subtitle": ParagraphStyle("ReportSubtitle", parent=base["Normal"], fontName="Times-Italic",
                                    fontSize=13, textColor=colors.HexColor(Palette.SUBTITLE), spaceAfter=4),
        "meta": ParagraphStyle("ReportMeta", parent=base["Normal"], fontName="Times-Roman",
                                fontSize=10, textColor=colors.HexColor(Palette.META)),
        "h1": ParagraphStyle("H1", parent=base["Heading1"], fontName="Times-Bold", fontSize=16,
                              spaceBefore=18, spaceAfter=8, textColor=colors.HexColor(Palette.NAVY)),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Times-Bold", fontSize=12.5,
                              spaceBefore=10, spaceAfter=5, textColor=colors.HexColor(Palette.ACCENT_BLUE)),
        "h3": ParagraphStyle("H3", parent=base["Heading3"], fontName="Times-Bold", fontSize=11,
                              spaceBefore=8, spaceAfter=4, textColor=colors.HexColor(Palette.ACCENT_BLUE)),
        "body": ParagraphStyle("Body", parent=base["Normal"], fontName="Times-Roman", fontSize=10.5,
                                leading=15, spaceAfter=8, alignment=4),  # 4 = justified
        "quote": ParagraphStyle("Quote", parent=base["Normal"], fontName="Times-Italic", fontSize=10.5,
                                 leading=15, spaceAfter=8, leftIndent=18, textColor=colors.HexColor("#3a3a3a")),
        "caption": ParagraphStyle("Caption", parent=base["Normal"], fontName="Times-Italic", fontSize=9,
                                   textColor=colors.HexColor(Palette.META), alignment=1, spaceAfter=14),
        "note": ParagraphStyle("Note", parent=base["Normal"], fontName="Times-Italic", fontSize=9,
                                textColor=colors.HexColor(Palette.NOTE), spaceAfter=6),
        "math": ParagraphStyle("MathFallback", parent=base["Normal"], fontName="Courier", fontSize=10,
                                alignment=1, spaceAfter=8, textColor=colors.HexColor("#333333")),
        "verbatim": ParagraphStyle("Verbatim", parent=base["Normal"], fontName="Courier", fontSize=8,
                                    leading=10),
    }


_XML_ESCAPE = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}
_MATH_SPAN_RE = re.compile(r"(\$[^$]*\$)")


def _xml_escape(text: str) -> str:
    for ch, esc in _XML_ESCAPE.items():
        text = text.replace(ch, esc)
    return text


def _inline_to_rl_markup(text: str) -> str:
    """Converts the shared inline Markdown subset to reportlab's minimal
    paragraph XML. Math spans have no real typesetting here (that's the
    xelatex backend's job) -- they degrade gracefully to literal monospace
    text so a math-heavy report is still readable, just not typeset."""
    parts = _MATH_SPAN_RE.split(text)
    out = []
    for part in parts:
        if part.startswith("$") and part.endswith("$") and len(part) > 1:
            out.append('<font face="Courier">%s</font>' % _xml_escape(part[1:-1]))
            continue
        part = re.sub(r"`(.+?)`", lambda m: "\x01" + m.group(1) + "\x02", part)
        part = re.sub(r"\*\*(.+?)\*\*", lambda m: "\x03" + m.group(1) + "\x04", part)
        part = re.sub(r"\*(.+?)\*", lambda m: "\x05" + m.group(1) + "\x06", part)
        part = _xml_escape(part)
        part = part.replace("\x01", '<font face="Courier">').replace("\x02", "</font>")
        part = part.replace("\x03", "<b>").replace("\x04", "</b>")
        part = part.replace("\x05", "<i>").replace("\x06", "</i>")
        out.append(part)
    return "".join(out)


def _table_style(ncols: int, row_colors=None) -> TableStyle:
    style = [
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5 if ncols <= 5 else (8.5 if ncols <= 9 else 7.5)),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(Palette.ACCENT_BLUE)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(Palette.ROW_ALT)]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor(Palette.BORDER)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]
    if row_colors:
        color_hex = {"bad": Palette.ROW_BAD, "warn": Palette.ROW_WARN}
        for i, tag in enumerate(row_colors, start=1):
            if tag in color_hex:
                style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor(color_hex[tag])))
    return TableStyle(style)


def _cell_paragraph(text: str, markdown_cells: bool, style: ParagraphStyle) -> Paragraph:
    markup = _inline_to_rl_markup(str(text)) if markdown_cells else _xml_escape(str(text))
    return Paragraph(markup, style)


def _build_table(rows: List[List[Any]], col_widths, markdown_cells: bool, row_colors=None) -> Table:
    styles = _styles()
    header_style = ParagraphStyle("CellHeader", parent=styles["body"], fontName="Times-Bold",
                                   fontSize=9.5, textColor=colors.white, alignment=0, spaceAfter=0)
    cell_style = ParagraphStyle("Cell", parent=styles["body"], fontSize=9.5, alignment=0, spaceAfter=0)
    ncols = len(rows[0])
    data = [[_cell_paragraph(c, False, header_style) for c in rows[0]]]
    for row in rows[1:]:
        data.append([_cell_paragraph(c, markdown_cells, cell_style) for c in row])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(_table_style(ncols, row_colors))
    return t


def _fig_to_image_flowable(image, max_width_in: float = USABLE_WIDTH_IN) -> Image:
    if isinstance(image, (bytes, bytearray)):
        buf = io.BytesIO(bytes(image))
    else:
        buf = io.BytesIO()
        image.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
        buf.seek(0)
    img = Image(buf)
    aspect = img.imageHeight / img.imageWidth
    img.drawWidth = max_width_in * inch
    img.drawHeight = max_width_in * inch * aspect
    return img


class _TocDocTemplate(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            name = getattr(flowable.style, "name", "")
            if name == "H1":
                self.notify("TOCEntry", (0, flowable.getPlainText(), self.page))
            elif name == "H2":
                self.notify("TOCEntry", (1, flowable.getPlainText(), self.page))


def _markdown_flowables(text: str, styles) -> List[Any]:
    nodes = parse_markdown(text)
    out: List[Any] = []
    bullet_buf: List[str] = []

    def flush_bullets():
        if bullet_buf:
            items = [ListItem(Paragraph(_inline_to_rl_markup(t), styles["body"]), leftIndent=14)
                     for t in bullet_buf]
            out.append(ListFlowable(items, bulletType="bullet", start="circle"))
            bullet_buf.clear()

    for node in nodes:
        if isinstance(node, MDBullet):
            bullet_buf.append(node.text)
            continue
        flush_bullets()
        if isinstance(node, MDHeading):
            style = styles["h2"] if node.level == 2 else styles["h3"]
            out.append(Paragraph(_inline_to_rl_markup(node.text), style))
        elif isinstance(node, MDTable):
            if node.rows:
                ncols = max(len(r) for r in node.rows)
                col_width = (USABLE_WIDTH_IN / ncols) * inch
                padded = [r + [""] * (ncols - len(r)) for r in node.rows]
                out.append(_build_table(padded, [col_width] * ncols, markdown_cells=True))
                out.append(Spacer(1, 8))
        elif isinstance(node, MDDisplayMath):
            out.append(Paragraph(_xml_escape(node.body.strip()), styles["math"]))
            if node.trailing:
                out.append(Paragraph(_inline_to_rl_markup(node.trailing), styles["body"]))
        elif isinstance(node, MDParagraph):
            style = styles["quote"] if node.quote else styles["body"]
            out.append(Paragraph(_inline_to_rl_markup(node.text), style))
        elif isinstance(node, MDBlank):
            out.append(Spacer(1, 4))
    flush_bullets()
    return out


class ReportLabBackend(PdfBackend):
    def render(self, block_list: List[b.Block], *, path: Optional[str] = None) -> Optional[bytes]:
        styles = _styles()
        has_toc = any(isinstance(bl, b.TableOfContents) for bl in block_list)
        title_block = next((bl for bl in block_list if isinstance(bl, b.TitlePage)), None)

        target = path if path is not None else io.BytesIO()
        doc_cls = _TocDocTemplate if has_toc else SimpleDocTemplate
        doc = doc_cls(
            target, pagesize=letter,
            topMargin=0.85 * inch, bottomMargin=0.85 * inch,
            leftMargin=0.9 * inch, rightMargin=0.9 * inch,
            title=title_block.title if title_block else "Report",
            author="LabCD",
        )

        story: List[Any] = []
        for block in block_list:
            story.extend(self._render_block(block, styles))

        if has_toc:
            doc.multiBuild(story)
        else:
            doc.build(story)

        if path is not None:
            return None
        return target.getvalue()

    def _render_block(self, block: b.Block, styles) -> List[Any]:
        if isinstance(block, b.TitlePage):
            out = [Spacer(1, 1.4 * inch), Paragraph(block.title, styles["title"])]
            if block.subtitle:
                out.append(Paragraph(block.subtitle, styles["subtitle"]))
            out.append(Spacer(1, 0.3 * inch))
            for line in block.meta_lines:
                out.append(Paragraph(line, styles["meta"]))
            if block.date:
                out.append(Paragraph(block.date, styles["meta"]))
            return out

        if isinstance(block, b.Abstract):
            if not block.markdown.strip():
                return []
            return [Paragraph("Abstract", styles["h1"])] + _markdown_flowables(block.markdown.strip(), styles)

        if isinstance(block, b.TableOfContents):
            toc = RLTableOfContents()
            toc.levelStyles = [
                ParagraphStyle(fontName="Times-Bold", fontSize=11, name="TOCHeading1",
                                leftIndent=20, firstLineIndent=-20, spaceBefore=6, leading=14),
                ParagraphStyle(fontName="Times-Roman", fontSize=10, name="TOCHeading2",
                                leftIndent=40, firstLineIndent=-20, spaceBefore=2, leading=12),
            ]
            return [Paragraph("Contents", styles["h1"]), toc, PageBreak()]

        if isinstance(block, b.Heading):
            style = styles["h1"] if block.level <= 1 else styles["h2"]
            return [Paragraph(block.text, style)]

        if isinstance(block, b.Markdown):
            return _markdown_flowables(block.text, styles)

        if isinstance(block, b.KeyValueTable):
            out = []
            if block.title:
                out.append(Paragraph(block.title, styles["h2"]))
            rows = [["Parameter", "Value"]] + [[str(k), str(v)] for k, v in block.rows]
            col_widths = [2.4 * inch, 4.1 * inch]
            out.append(_build_table(rows, col_widths, block.markdown_cells))
            out.append(Spacer(1, 10))
            return out

        if isinstance(block, b.DataTable):
            out = []
            if block.title:
                out.append(Paragraph(block.title, styles["h2"]))
            rows = [block.headers] + [[str(c) for c in r] for r in block.rows]
            ncols = len(block.headers)
            col_width = (USABLE_WIDTH_IN / ncols) * inch
            row_colors = [block.row_style(r) for r in block.rows] if block.row_style else None
            out.append(_build_table(rows, [col_width] * ncols, block.markdown_cells, row_colors))
            out.append(Spacer(1, 10))
            return out

        if isinstance(block, b.DiffTable):
            out = []
            if block.title:
                out.append(Paragraph(block.title, styles["h2"]))
            rows = [["Parameter", "Before", "After"]]
            for field_name, (old, new) in block.changed.items():
                rows.append([str(field_name), str(old), str(new)])
            col_widths = [2.2 * inch, 2.25 * inch, 2.25 * inch]
            out.append(_build_table(rows, col_widths, True))
            out.append(Spacer(1, 10))
            return out

        if isinstance(block, b.StatusBadge):
            color_hex = SUCCESS_HEX if block.ok else FAIL_HEX
            style = ParagraphStyle("StatusBadge", parent=styles["body"], fontName="Times-Bold",
                                    textColor=colors.HexColor(color_hex), alignment=0)
            return [Paragraph(_xml_escape(block.label), style)]

        if isinstance(block, b.Figure):
            img = _fig_to_image_flowable(block.image)
            out = [img]
            if block.caption:
                out.append(Paragraph(_xml_escape(block.caption), styles["caption"]))
            return out

        if isinstance(block, b.Verbatim):
            wrapped = wrap_long_lines(block.text)
            size = {"small": 8, "footnotesize": 7, "scriptsize": 6}.get(block.fontsize, 8)
            style = ParagraphStyle("VerbatimSized", parent=styles["verbatim"], fontSize=size, leading=size + 2)
            return [Preformatted(wrapped, style)]

        if isinstance(block, b.PageBreak):
            return [PageBreak()]

        raise TypeError("Unknown block type: %r" % (block,))
