"""Fast, display-only control-law formatting for a complex benchmark plant.

The two-link manipulator's numerical SMC law is quick to derive, but expanding
and trig-simplifying its symbolic expression can take an unbounded amount of
time. This context manager changes only its report formatting during one
sequential benchmark run; the controller and simulation are untouched.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import sympy as sp

from backend_core.AgentAdaptive.controller import runs, structure_build


def format_two_link_control_law(u_symbolic, yd_symbols, refs) -> str:
    """Show exact unsimplified formulas without expensive expand/trigsimp."""
    try:
        replacements = {}
        for index, symbols in enumerate(yd_symbols):
            derivatives = structure_build._ref_symbolic_derivatives(
                refs[index]["expr"], len(symbols) - 1, structure_build._REF_T
            )
            replacements.update(zip(symbols, derivatives))

        readable_symbols = structure_build._pretty_ref_symbols(yd_symbols)
        symbolic = [
            f"$$u_{{{index + 1}}} = {sp.latex(law.subs(readable_symbols))}$$"
            for index, law in enumerate(u_symbolic)
        ]
        with_reference = [
            f"$$u_{{{index + 1}}}(x,t) = {sp.latex(law.subs(replacements))}$$"
            for index, law in enumerate(u_symbolic)
        ]
        return (
            "Exact SMC control law (left unsimplified for fast display):\n\n"
            + "\n\n".join(symbolic)
            + "\n\nWith this test's reference substituted:\n\n"
            + "\n\n".join(with_reference)
        )
    except Exception as exc:
        return (
            "The numerical SMC control law was computed, but its display could "
            f"not be formatted ({type(exc).__name__}: {exc})."
        )


@contextmanager
def fast_two_link_report(enabled: bool) -> Iterator[None]:
    """Temporarily swap the formatting alias used by ``runs._run_smc``."""
    if not enabled:
        yield
        return

    previous = runs._format_u_with_reference_smc
    runs._format_u_with_reference_smc = format_two_link_control_law
    try:
        yield
    finally:
        runs._format_u_with_reference_smc = previous
