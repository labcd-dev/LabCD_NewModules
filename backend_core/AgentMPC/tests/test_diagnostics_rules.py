"""
Run with: pytest backend_core/AgentMPC/tests/test_diagnostics_rules.py -v

The "How the assistant decides" panel: every Diagnostics rule as a formula.

Two properties matter. The formulas must render -- the live equations panel
already shipped one LaTeX string that KaTeX printed as a red error -- and the
numbers on the page must be the numbers the code applies, so a threshold
changed in mpc_suitability.py shows up here without anyone editing the text.
"""

import re

import pytest

from backend_core.AgentMPC.agents import mpc_suitability as S
from backend_core.AgentMPC.agents.diagnostics_agent import (
    FREQUENT_INSTABILITY_FRACTION, SOLVER_STRUGGLE_RATIO,
)
from backend_core.AgentMPC.agents.diagnostics_rules import diagnostics_rules

GROUPS = diagnostics_rules()
RULES = [r for g in GROUPS for r in g["rules"]]
EXPRESSIONS = [e for r in RULES for e in r["latex"]]


def test_every_rule_is_complete():
    for rule in RULES:
        assert rule["title"] and rule["question"] and rule["decision"], rule
        assert rule["latex"], rule["title"]
        assert rule["severity"] in {"error", "warning", "info", "basis"}, rule["title"]


@pytest.mark.parametrize("expr", EXPRESSIONS)
def test_braces_balance(expr):
    """An unbalanced brace is the most common way a LaTeX string fails to
    render; catch it here instead of as red text on the page."""
    depth = 0
    for ch in re.sub(r"\\[{}]", "", expr):
        depth += ch == "{"
        depth -= ch == "}"
        assert depth >= 0, expr
    assert depth == 0, expr


@pytest.mark.parametrize("expr", EXPRESSIONS)
def test_no_spacing_command_runs_into_the_next_symbol(expr):
    assert not re.search(r"\\q?quad[A-Za-z]", expr), expr


@pytest.mark.parametrize("expr", EXPRESSIONS)
def test_no_python_format_braces_leaked(expr):
    """An f-string with a missed doubled brace leaves '{S.' or '{acc' in the
    output instead of the number."""
    assert "{S." not in expr and "{acc" not in expr, expr


def _page():
    return " ".join(" ".join(r["latex"]) + " " + r["decision"] for r in RULES)


def test_the_page_shows_the_thresholds_the_code_applies():
    page = _page()
    for value in (S.EULER_GROWTH_ERROR, S.EULER_GROWTH_WARN, S.EULER_GROWTH_HORIZON,
                  S.STIFFNESS_RATIO, S.REFERENCE_RESIDUAL_TOL, S.DERIVATIVE_MISMATCH_TOL,
                  FREQUENT_INSTABILITY_FRACTION, SOLVER_STRUGGLE_RATIO):
        assert f"{value:g}" in page, value


def test_a_changed_threshold_changes_the_page(monkeypatch):
    monkeypatch.setattr(S, "STIFFNESS_RATIO", 12345.0)
    from backend_core.AgentMPC.agents import diagnostics_rules as R

    assert "12345" in " ".join(e for g in R.diagnostics_rules() for r in g["rules"]
                               for e in r["latex"])


def test_the_known_solver_ratio_flaw_is_disclosed():
    """Until the units bug is fixed, the page must not present that rule as
    sound."""
    rule = next(r for r in RULES if r["title"] == "Solver struggling")
    assert "KNOWN FLAW" in rule["decision"]
