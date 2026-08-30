from labcd_pdfmaker.markdown import (
    MDBullet, MDDisplayMath, MDHeading, MDParagraph, MDTable, parse_markdown,
)


def test_headings():
    nodes = parse_markdown("## Section\n### Subsection")
    assert nodes[0] == MDHeading(level=2, text="Section")
    assert nodes[1] == MDHeading(level=3, text="Subsection")


def test_bullets():
    nodes = parse_markdown("- one\n- two")
    assert nodes == [MDBullet(text="one"), MDBullet(text="two")]


def test_quote():
    nodes = parse_markdown("> quoted line")
    assert nodes == [MDParagraph(text="quoted line", quote=True)]


def test_table():
    nodes = parse_markdown("| A | B |\n|---|---|\n| 1 | 2 |")
    assert len(nodes) == 1
    table = nodes[0]
    assert isinstance(table, MDTable)
    assert table.rows == [["A", "B"], ["1", "2"]]


def test_display_math_single_line():
    nodes = parse_markdown("$$x = 1$$ trailing text")
    assert nodes == [MDDisplayMath(body="x = 1", trailing="trailing text")]


def test_display_math_multiline():
    nodes = parse_markdown("$$\nx = 1\ny = 2\n$$")
    assert isinstance(nodes[0], MDDisplayMath)
    assert "x = 1" in nodes[0].body and "y = 2" in nodes[0].body
