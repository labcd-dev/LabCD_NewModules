from __future__ import annotations

# only tokenizes structure (headings/quotes/bullets/tables/display math) --
# inline bold/italic/code and escaping are each backend's own job.

import re
from dataclasses import dataclass, field
from typing import List, Union


@dataclass
class MDHeading:
    level: int  # 2 ("## ") or 3 ("### ")
    text: str


@dataclass
class MDParagraph:
    text: str
    quote: bool = False


@dataclass
class MDBullet:
    text: str


@dataclass
class MDTable:
    rows: List[List[str]] = field(default_factory=list)  # rows[0] is the header


@dataclass
class MDDisplayMath:
    body: str
    trailing: str = ""


@dataclass
class MDBlank:
    pass


MDNode = Union[MDHeading, MDParagraph, MDBullet, MDTable, MDDisplayMath, MDBlank]


def _parse_table(lines: List[str], start_idx: int):
    rows: List[List[str]] = []
    i = start_idx
    while i < len(lines) and lines[i].strip().startswith("|"):
        row = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        rows.append(row)
        i += 1
    # drop the "---|---|---" separator row Markdown tables use under the header
    if len(rows) >= 2 and re.match(r"^:?-+:?$", rows[1][0].replace(" ", "")):
        rows.pop(1)
    return MDTable(rows=rows), i


def parse_markdown(markdown_text: str) -> List[MDNode]:
    raw_lines = (markdown_text or "").split("\n")
    lines: List[str] = []
    is_quote: List[bool] = []
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

    nodes: List[MDNode] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

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
            nodes.append(MDDisplayMath(body=body, trailing=trailing))
            continue

        if stripped.startswith("### "):
            nodes.append(MDHeading(level=3, text=stripped[4:]))
            i += 1
            continue
        if stripped.startswith("## "):
            nodes.append(MDHeading(level=2, text=stripped[3:]))
            i += 1
            continue

        if stripped.startswith("|"):
            table, i = _parse_table(lines, i)
            nodes.append(table)
            continue

        if stripped.startswith("- ") or stripped.startswith("* "):
            nodes.append(MDBullet(text=stripped[2:]))
            i += 1
            continue

        if not stripped:
            nodes.append(MDBlank())
        else:
            nodes.append(MDParagraph(text=stripped, quote=is_quote[i]))
        i += 1

    return nodes
