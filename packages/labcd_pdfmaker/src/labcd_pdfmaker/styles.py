from __future__ import annotations

# colours pulled straight from the two old report generators (AgentMPC's
# reportlab styling, AgentAdaptive's status colours) -- same look, one palette.


class Palette:
    NAVY = "#12213f"          # titles
    ACCENT_BLUE = "#2f5597"   # section headings, table header background
    SUBTITLE = "#4a5a7a"      # subtitle text
    META = "#6a7a9a"          # meta / caption text
    BORDER = "#c7cfdd"        # table gridlines
    ROW_ALT = "#f2f5fa"       # alternating table row background
    ROW_BAD = "#fbe4e4"       # e.g. UNSTABLE/FAILED row highlight
    ROW_WARN = "#f5f0e0"      # e.g. degraded-but-not-failed row highlight
    NOTE = "#8a8a8a"          # footnote-style text

    SUCCESS = "ForestGreen"   # PASS badges (named colour understood by both
    FAIL = "BrickRed"         # reportlab's HexColor lookup and LaTeX xcolor)


SUCCESS_HEX = "#1e7d34"
FAIL_HEX = "#9b1c1c"

FONT_TITLE = "Times-Bold"
FONT_BODY = "Times-Roman"
FONT_ITALIC = "Times-Italic"

PAGE_MARGIN_IN = 0.9
LONGTABLE_MAX_COL_FRAC = 0.32
