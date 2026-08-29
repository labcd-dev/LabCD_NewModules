from __future__ import annotations


def wrap_long_lines(text: str, width: int = 100) -> str:
    # neither backend's verbatim block wraps on its own -- a long log line
    # just runs off the page, or aborts xelatex outright.
    out_lines = []
    for line in (text or "").split("\n"):
        while len(line) > width:
            out_lines.append(line[:width])
            line = line[width:]
        out_lines.append(line)
    return "\n".join(out_lines)
