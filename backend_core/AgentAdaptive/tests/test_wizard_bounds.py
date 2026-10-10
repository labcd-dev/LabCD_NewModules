"""
Run from the repo root with:
    python -m pytest backend_core/AgentAdaptive/tests/test_wizard_bounds.py -v
"""

import os

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

from backend_core.AgentAdaptive.tools import system_spec  # noqa: E402

APP = os.path.join(os.path.dirname(__file__), "..", "..", "..", "frontend_streamlit", "streamlit_app.py")
EQ = ["x2", "(-(u + 0.06*x2**2*sin(x1))*cos(x1) + 11.2815*sin(x1))/(0.06*sin(x1)**2 + 0.4)"]


def _app():
    os.environ["LABCD_ADAPTIVE_SHOW_PLOTS"] = "0"
    plant = system_spec.normalize_defaults({"status": "complete", "system_name": "Cart-Pole", "dynamics": {
        "states": ["x1", "x2"], "inputs": ["u"], "outputs": ["x1"], "state_equations": EQ,
        "state_meanings": ["angle", "rate"], "system_type": "SISO"}})
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["plant_spec"] = plant
    at.run()
    return at


def _fields(at):
    return {t.key: t for t in at.text_input if t.key}


def _confirm(at):
    [b for b in at.button if "Confirm parameters" in (b.label or "")][0].click().run()


def test_a_reference_outside_the_bound_is_refused_at_confirm_and_the_card_fixes_it():
    at = _app()
    f = _fields(at)
    f["ref_x1"].set_value("0.5*sin(5*t)")
    f["bnd_lo_x1"].set_value("-0.3")
    f["bnd_hi_x1"].set_value("0.3")
    _confirm(at)
    assert not at.exception
    assert at.session_state["sim_knobs_spec"] is None
    assert "not strictly inside its bound [-0.3, 0.3]" in at.error[0].value
    cards = [b for b in at.button if (b.key or "").startswith("adv_")]
    assert [b.label.replace("\\", "") for b in cards] == ["0.24*sin(5*t)", "0.15*sin(5*t)"]   # asterisks are escaped
    cards[0].click().run()
    assert not at.exception and _fields(at)["ref_x1"].value == "0.24*sin(5*t)"
    _confirm(at)
    assert not at.exception and not at.error
    assert at.session_state["sim_knobs_spec"] is not None
    assert any("x1 gets an error limit of 0.06" in i.value for i in at.info)


def test_a_start_outside_the_bound_offers_a_new_start():
    at = _app()
    f = _fields(at)
    f["ref_x1"].set_value("0")
    f["bnd_lo_x1"].set_value("-0.25")
    f["bnd_hi_x1"].set_value("0.25")
    [t for t in at.text_input if "Initial condition" in (t.label or "")][0].set_value("0.3, 0")
    _confirm(at)
    assert at.session_state["sim_knobs_spec"] is None
    assert [b.label.replace("\\", "") for b in at.button if (b.key or "").startswith("adv_")] == ["0,0"]


def test_both_constraints_show_the_note_about_what_is_guaranteed():
    at = _app()
    f = _fields(at)
    f["ref_x1"].set_value("0.15*sin(1.2*t)")
    f["bnd_lo_x1"].set_value("-0.4")
    f["bnd_hi_x1"].set_value("0.4")
    f["lim_lo_u"].set_value("-5")
    f["lim_hi_u"].set_value("5")
    _confirm(at)
    assert at.session_state["sim_knobs_spec"] is not None
    assert any("only guaranteed while the actuator is not clipping" in i.value for i in at.info)
