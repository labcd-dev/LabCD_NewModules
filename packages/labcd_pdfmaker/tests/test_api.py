import warnings

import pytest

from labcd_pdfmaker import Backend, ReportBuilder
from labcd_pdfmaker.api import xelatex_available


def _kitchen_sink(builder: ReportBuilder) -> ReportBuilder:
    builder.add_table_of_contents()
    builder.add_abstract("This run **succeeded**.")
    builder.add_section("System Analysis", "The plant has 2 states and 1 input.")
    builder.add_status_badge(ok=True, label="RUN VERDICT: PASS")
    builder.add_key_value_table([("Np", "12"), ("Nc", "4")], title="Config")
    builder.add_data_table(
        ["Iter", "Status", "MSE"],
        [[1, "OK", "0.01"], [2, "UNSTABLE", "--"]],
        row_style=lambda row: "bad" if row[1] == "UNSTABLE" else None,
        title="History",
    )
    builder.add_diff_table({"kp": (1.0, 2.0)}, title="Changes")
    builder.add_verbatim("line one\nline two")
    builder.add_page_break()
    builder.add_section("Conclusion", "- point one\n- point two\n\n> a quoted remark")
    return builder


def test_reportlab_backend_builds_pdf_bytes():
    builder = ReportBuilder("Test Report", "subtitle", backend=Backend.REPORTLAB)
    _kitchen_sink(builder)
    pdf_bytes = builder.build()
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")


def test_reportlab_backend_writes_to_path(tmp_path):
    builder = ReportBuilder("Test Report", backend=Backend.REPORTLAB)
    _kitchen_sink(builder)
    out = tmp_path / "report.pdf"
    result = builder.build(path=str(out))
    assert result is None
    assert out.read_bytes().startswith(b"%PDF")


def test_math_on_reportlab_warns_and_degrades():
    builder = ReportBuilder("Math Report", backend=Backend.REPORTLAB)
    builder.add_markdown("The law is $\\dot{x} = Ax$.")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pdf_bytes = builder.build()
    assert any("math" in str(w.message).lower() for w in caught)
    assert pdf_bytes.startswith(b"%PDF")


def test_auto_backend_prefers_reportlab_without_math():
    builder = ReportBuilder("Plain Report", backend=Backend.AUTO)
    builder.add_markdown("No math here at all.")
    assert builder._resolve_backend() == Backend.REPORTLAB


@pytest.mark.skipif(not xelatex_available(), reason="xelatex not installed")
def test_xelatex_backend_builds_pdf_bytes():
    builder = ReportBuilder("Test Report", "subtitle", backend=Backend.XELATEX)
    _kitchen_sink(builder)
    builder.add_math_display(r"\dot{x} = Ax + Bu")
    pdf_bytes = builder.build()
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")


def test_xelatex_backend_missing_raises_clear_error(monkeypatch):
    monkeypatch.setattr("labcd_pdfmaker.api.xelatex_available", lambda: False)
    builder = ReportBuilder("Test Report", backend=Backend.XELATEX)
    builder.add_markdown("hello")
    with pytest.raises(RuntimeError, match="xelatex is not installed"):
        builder.build()
