"""Export the displayed final control laws from current Adaptive result JSONs.

Only the law with the test's actual reference substituted is exported.
LaTeX is converted to readable math. Repeated expressions in long laws are
named inside control_law; their definitions are retained in that same list.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import sympy as sp
from sympy.printing.precedence import precedence
from sympy.printing.str import StrPrinter
from sympy.parsing.sympy_parser import (
    convert_xor, implicit_multiplication_application, parse_expr, rationalize,
    standard_transformations,
)


REPORTS_DIR = Path(__file__).resolve().parent / "json_reports"
RESULTS_DIR = REPORTS_DIR / "results"
OUTPUT_PATH = REPORTS_DIR / "control_laws.json"
SECTION = re.compile(r"^## Control Law[^\S\n]*\n(.*?)(?=^## |\Z)", re.M | re.S)
EQUATION = re.compile(r"\$\$(.*?)\$\$", re.S)
LEFT_SIDE = re.compile(r"u(?:_\{(\d+)\})?(\(x,t\))?")


class ReadablePrinter(StrPrinter):
    """Print exact terminating decimals instead of large integer fractions."""

    def _print_Integer(self, expr):
        digits = str(abs(int(expr)))
        if len(digits) <= 16:
            return str(expr)
        significant = digits.rstrip("0")
        mantissa = significant[0]
        if len(significant) > 1:
            mantissa += "." + significant[1:]
        return ("-" if expr < 0 else "") + mantissa + "e" + str(len(digits) - 1)

    def _print_Rational(self, expr):
        denominator = int(expr.q)
        twos = fives = 0
        while denominator % 2 == 0:
            denominator //= 2
            twos += 1
        while denominator % 5 == 0:
            denominator //= 5
            fives += 1
        if denominator != 1:
            return super()._print_Rational(expr)
        places = max(twos, fives)
        digits = str(abs(int(expr.p)) * 2**(places - twos) * 5**(places - fives))
        digits = digits.zfill(places + 1)
        if places:
            digits = (digits[:-places] + "." + digits[-places:]).rstrip("0").rstrip(".")
        return ("-" if expr < 0 else "") + digits

    def _print_Mul(self, expr):
        coefficient, rest = expr.as_coeff_Mul()
        if isinstance(coefficient, sp.Rational) and coefficient.q != 1:
            displayed = self._print_Rational(coefficient)
            if "/" not in displayed:
                return displayed + "*" + self.parenthesize(rest, precedence(expr))
        return super()._print_Mul(expr)


def _group(text: str, position: int) -> tuple[str, int]:
    while position < len(text) and text[position].isspace():
        position += 1
    if position >= len(text) or text[position] != "{":
        raise ValueError("Expected a braced LaTeX argument.")
    depth = 1
    end = position + 1
    while end < len(text) and depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    if depth:
        raise ValueError("Unclosed LaTeX argument.")
    return text[position + 1:end - 1], end


def _plain_latex(text: str) -> str:
    """Translate the fraction/trig notation used by the saved core reports."""
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = text.replace(r"\operatorname{sat}", "sat")
    command = re.compile(r"\\(frac|sin|cos|tan)")
    while match := command.search(text):
        name = match[1]
        position = match.end()
        power = None
        if position < len(text) and text[position] == "^":
            power, position = _group(text, position + 1)
        arg, end = _group(text, position)
        if name == "frac":
            denominator, end = _group(text, end)
            replacement = f"(({_plain_latex(arg)})/({_plain_latex(denominator)}))"
        else:
            replacement = f"{name}({_plain_latex(arg)})"
            if power is not None:
                replacement = f"({replacement})^({_plain_latex(power)})"
        text = text[:match.start()] + replacement + text[end:]
    text = text.replace(r"\cdot", "*").replace("{", "(").replace("}", ")")
    if "\\" in text:
        raise ValueError(f"Unsupported LaTeX notation: {text}")
    return text


@lru_cache(maxsize=None)
def clean_laws(equations: tuple[str, ...], states: tuple[str, ...],
               inputs: tuple[str, ...]) -> tuple[str, ...]:
    """Keep exact printed coefficients, factor repeats, and preserve input order."""
    bindings = {name: sp.Symbol(name) for name in (*states, "t")}
    bindings.update({"sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
                     "sat": sp.Function("sat")})
    substitutions = sorted(((sp.latex(sp.Symbol(name)), name) for name in states),
                           key=lambda pair: len(pair[0]), reverse=True)
    expressions = []
    for equation in equations:
        rhs = equation.partition("=")[2].strip()
        for latex_name, plain_name in substitutions:
            rhs = rhs.replace(latex_name, plain_name)
        plain = _plain_latex(rhs)
        transformations = standard_transformations + (
            convert_xor, implicit_multiplication_application, rationalize,
        )
        expr = parse_expr(plain, local_dict=bindings, transformations=transformations)
        if not expr.free_symbols <= {bindings[name] for name in (*states, "t")}:
            raise ValueError(f"Unrecognized state symbols in: {equation}")
        expressions.append(expr)

    def display(expr: sp.Expr) -> str:
        return ReadablePrinter().doprint(expr).replace("**", "^")

    direct = [f"{name} = {display(expr)}" for name, expr in zip(inputs, expressions)]
    # Common subexpressions shorten the huge expanded manipulator formulas.
    definitions, reduced = sp.cse(expressions, symbols=sp.numbered_symbols("c", start=1))
    lines = [f"{symbol} = {display(expr)}" for symbol, expr in definitions]
    lines += [f"{name} = {display(expr)}" for name, expr in zip(inputs, reduced)]
    if sum(map(len, lines)) >= 0.8 * sum(map(len, direct)):
        lines = direct
    if any(expr.has(bindings["sat"]) for expr in expressions):
        lines.append("sat(z) = min(1, max(-1, z))")
    return tuple(lines)


def extract_record(record: dict, path: Path) -> dict:
    """Require a complete, consistently labeled law for every declared input."""
    seed_match = re.fullmatch(r"seed(\d+)", path.parent.name)
    if seed_match is None:
        raise ValueError(f"Invalid seed folder: {path}")
    seed = int(seed_match[1])
    if record.get("seeds", {}).get("run") != seed:
        raise ValueError(f"Saved run seed disagrees with its folder: {path}")
    model_folder = path.parent.parent.name
    model = record.get("llm_model")
    if model_folder == "mock":
        if model is not None:
            raise ValueError(f"Mock folder contains a real model: {path}")
        model = "mock"
    elif not isinstance(model, str) or not model:
        raise ValueError(f"Missing LLM model: {path}")

    spec = record.get("applied_system_spec", {})
    plant = record.get("plant")
    if not isinstance(plant, str) or plant != spec.get("system_name"):
        raise ValueError(f"Inconsistent dynamic model name: {path}")
    inputs = spec.get("dynamics", {}).get("inputs")
    if not isinstance(inputs, list) or not inputs or len(set(inputs)) != len(inputs):
        raise ValueError(f"Missing or repeated input names: {path}")
    summary = record.get("summary")
    if not isinstance(summary, str):
        raise ValueError(f"Missing final report: {path}")
    sections = list(SECTION.finditer(summary))
    if len(sections) != 1:
        raise ValueError(f"Expected one final Control Law section: {path}")
    laws = {"symbolic": {}, "reference_substituted": {}}
    for match in EQUATION.finditer(sections[0][1]):
        equation = match[1].strip()
        left, separator, _ = equation.partition("=")
        left_match = LEFT_SIDE.fullmatch(left.strip())
        if not separator or left_match is None:
            raise ValueError(f"Unrecognized control equation left side {left!r}: {path}")
        index_text, substituted = left_match.groups()
        if index_text is None and len(inputs) != 1:
            raise ValueError(f"Unnumbered control equation in a multi-input run: {path}")
        index = int(index_text) - 1 if index_text is not None else 0
        if not 0 <= index < len(inputs):
            raise ValueError(f"Control equation refers to an unknown input: {path}")
        variant = "reference_substituted" if substituted else "symbolic"
        if index in laws[variant]:
            raise ValueError(f"Duplicate control equation for input {index + 1}: {path}")
        laws[variant][index] = {"input": inputs[index], "equation_latex": equation}
    expected_indexes = set(range(len(inputs)))
    for variant, equations in laws.items():
        if set(equations) != expected_indexes:
            raise ValueError(f"Incomplete {variant} control law: {path}")

    return {
        "llm_model": model,
        "seed": seed,
        "dynamic_model": plant,
        "control_law": list(clean_laws(
            tuple(laws["reference_substituted"][index]["equation_latex"]
                  for index in range(len(inputs))),
            tuple(spec["dynamics"]["states"]), tuple(inputs),
        )),
    }


def main() -> None:
    paths = sorted(
        path for path in RESULTS_DIR.glob("*/seed*/*_results.json")
        if path.is_file() and re.fullmatch(r"seed\d+", path.parent.name)
    )
    if not paths:
        raise FileNotFoundError(f"No current seeded Adaptive results in {RESULTS_DIR}")
    records = []
    identities = set()
    for path in paths:
        runs = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(runs, list) or not runs:
            raise ValueError(f"Expected a nonempty array of run records: {path}")
        for run in runs:
            if not isinstance(run, dict):
                raise ValueError(f"Run is not a JSON object: {path}")
            extracted = extract_record(run, path)
            identity = tuple(extracted[key] for key in
                             ("llm_model", "seed", "dynamic_model"))
            if identity in identities:
                raise ValueError(f"Duplicate run identity: {identity}")
            identities.add(identity)
            records.append(extracted)
    records.sort(key=lambda r: (r["llm_model"], r["seed"], r["dynamic_model"]))
    OUTPUT_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=False,
                                      allow_nan=False) + "\n", encoding="utf-8")
    print(f"Exported {len(records)} run control laws from {len(paths)} source files.")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
