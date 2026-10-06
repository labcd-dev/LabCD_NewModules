"""Checks model-written sentences against the facts the model was given: invented numbers
and names, wrong input/output comparisons, stability promises. It can't judge whether a
sentence is wise, only that it doesn't contradict what Python verified.
"""

import re

_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?!\w)")
# x1, z2, p1, r4, v_dot, i_L: names with a digit or an underscore
_NAME = re.compile(r"(?<!\w)(?:[A-Za-z]+\d[A-Za-z0-9]*|[A-Za-z]+_[A-Za-z0-9_]+)(?!\w)")
_QUOTED = re.compile(r"[`'\"]([A-Za-z_][A-Za-z0-9_]*)[`'\"]")
_COUNTS = re.compile(r"(\d+)\s+(?:control\s+)?inputs?\s+but\s+(?:only\s+)?(\d+)\s+(?:control\s+)?outputs?", re.I)
_PROMISE = re.compile(r"\bguarantee[sd]?\b|\bproven\b|\bprovably\b|\b(?:is|will be|stays) stable\b", re.I)
_NEGATION = re.compile(r"\b(?:no|not|never|nothing|without|cannot)\b|n't", re.I)


def _norm(number):
    return format(float(number), "g")


def _numbers(text):
    return {_norm(n) for n in _NUMBER.findall(text.translate(_DIGITS))}


def _has_word(token, text):
    return re.search(r"(?<!\w)%s(?!\w)" % re.escape(token), text) is not None


def check(text, evidence_text, dims=None, number_source=None, numbers=True):
    # returns a list of problems; an empty list means nothing contradicts the facts.
    # number_source narrows where numbers may come from (default: the whole evidence);
    # numbers=False skips the number check (free chat may quote general numbers)
    problems = []
    text = text or ""
    allowed = _numbers(number_source if number_source is not None else evidence_text)
    for n in sorted(_numbers(text) - allowed if numbers else ()):
        problems.append("the number %s is not in the facts you were given." % n)
    seen = set(_NAME.findall(text)) | set(_QUOTED.findall(text))
    for name in sorted(seen):
        if not _has_word(name, evidence_text):
            problems.append("the name '%s' is not in the facts you were given." % name)
    if dims:
        n_in, n_out = dims.get("n_inputs"), dims.get("n_outputs")
        low = text.lower()
        claims = (("more inputs than outputs", n_in > n_out), ("more outputs than inputs", n_out > n_in),
                  ("fewer inputs than outputs", n_in < n_out), ("fewer outputs than inputs", n_out < n_in),
                  ("as many inputs as outputs", n_in == n_out))
        for phrase, true in claims:
            if phrase in low and not true:
                problems.append("the plant has %d input(s) and %d output(s), so \"%s\" is wrong." % (n_in, n_out, phrase))
        for m in _COUNTS.finditer(text.translate(_DIGITS)):
            a, b = int(m.group(1)), int(m.group(2))
            # "X inputs but Y outputs" has to match the plant and the two must differ
            # ("and" is left alone: "needs 1 input and 1 output" is a requirement, not a count)
            if (a, b) != (n_in, n_out) or a == b:
                problems.append("\"%s\" does not match the plant (%d input(s), %d output(s))." % (m.group(0), n_in, n_out))
    for m in _PROMISE.finditer(text):
        if not _NEGATION.search(text[max(0, m.start() - 40):m.start()]):
            problems.append("do not promise or imply stability (\"%s\")." % m.group(0))
    return problems
