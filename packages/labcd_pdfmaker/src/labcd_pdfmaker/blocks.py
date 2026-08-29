from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

# A figure is either raw PNG bytes or a matplotlib Figure (duck-typed --
# this package does not hard-depend on matplotlib).
ImageLike = Union[bytes, bytearray, Any]

RowStyle = Callable[[Sequence[Any]], Optional[str]]  # row -> None | "bad" | "warn"


@dataclass
class TitlePage:
    title: str
    subtitle: str = ""
    meta_lines: Sequence[str] = field(default_factory=tuple)
    date: Optional[str] = None


@dataclass
class Abstract:
    markdown: str


@dataclass
class TableOfContents:
    pass


@dataclass
class Heading:
    text: str
    level: int = 1  # 1 = section, 2 = subsection


@dataclass
class Markdown:
    text: str


@dataclass
class KeyValueTable:
    rows: List[Tuple[str, str]]
    title: Optional[str] = None
    markdown_cells: bool = True


@dataclass
class DataTable:
    headers: List[str]
    rows: List[List[Any]]
    title: Optional[str] = None
    row_style: Optional[RowStyle] = None
    # False for tables holding literal user text (e.g. free-form answers)
    # that should never be reinterpreted as bold/italic/code markdown.
    markdown_cells: bool = True


@dataclass
class DiffTable:
    changed: dict
    title: Optional[str] = None


@dataclass
class StatusBadge:
    ok: bool
    label: str


@dataclass
class Figure:
    image: ImageLike
    caption: Optional[str] = None


@dataclass
class Verbatim:
    text: str
    fontsize: str = "small"  # "small" | "footnotesize" | "scriptsize"


@dataclass
class PageBreak:
    pass


Block = Union[
    TitlePage, Abstract, TableOfContents, Heading, Markdown, KeyValueTable,
    DataTable, DiffTable, StatusBadge, Figure, Verbatim, PageBreak,
]
