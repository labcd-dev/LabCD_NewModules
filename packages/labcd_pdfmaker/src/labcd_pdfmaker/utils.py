from __future__ import annotations


def wrap_long_lines(text: str, width: int = 100) -> str:
    """Hard-wraps lines longer than ``width``.

    Both backends render verbatim/console-log blocks in a fixed-width font
    with no automatic line breaking (plain LaTeX ``fancyvrb`` has no
    ``breaklines`` key without extra packages; reportlab's ``Preformatted``
    doesn't wrap either), so a single very long line can run off the page
    -- or, for xelatex, abort compilation outright. Wrapping here keeps
    both backends' verbatim rendering simple.
    """
    out_lines = []
    for line in (text or "").split("\n"):
        while len(line) > width:
            out_lines.append(line[:width])
            line = line[width:]
        out_lines.append(line)
    return "\n".join(out_lines)
