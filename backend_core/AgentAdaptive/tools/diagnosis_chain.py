# the diagnoser's memory across runs of the same plant. python diffs what the user changed,
# the model only adds a short note. a chain = one plant, found by a structural fingerprint
# (not its name), so edited numbers keep the chain and same-name plants don't merge

import hashlib
import json
import re
import uuid

import numpy as np

# how many earlier runs go into the evidence, and the cap on the model's own note
MAX_ATTEMPTS_SHOWN = 4
NOTE_CHARS = 120
# chains kept per session, oldest dropped first
MAX_CHAINS = 10
MAX_ATTEMPTS_KEPT = 8


def _squash(text):
    return re.sub(r"\s+", "", str(text or ""))


def fingerprint(spec):
    # structure only (names + equation text before parameters are baked in), so a
    # changed gain/reference/x0/dt keeps the chain and changed equations start one
    dyn = (spec or {}).get("dynamics") or {}
    parts = {k: [_squash(v) for v in (dyn.get(k) or [])]
             for k in ("states", "inputs", "outputs", "state_equations")}
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def chain_for(registry, spec, parent_fp=None):
    # registry: {fingerprint: {"id", "parent", "attempts"}}. creates the chain on first sight.
    # parent_fp is the plant this one was edited from (a "system" suggestion), if known
    fp = fingerprint(spec)
    chain = registry.get(fp)
    if chain is None:
        chain = {"id": str(uuid.uuid4()), "parent": parent_fp if parent_fp != fp else None,
                 "attempts": []}
        registry[fp] = chain
        for old in list(registry)[:-MAX_CHAINS]:
            del registry[old]
    return fp, chain


def snapshot(spec):
    # the things a user can change between runs, normalised so they compare cleanly
    dyn = (spec or {}).get("dynamics") or {}
    refs = {r.get("output"): _squash(r.get("expr")) for r in (dyn.get("references") or [])
            if isinstance(r, dict)}
    return {
        "references": refs,
        "x0": [float(v) for v in (dyn.get("x0") or [])],
        "dt": dyn.get("solver_step"),
        "sim_time": dyn.get("sim_time"),
        "parameters": dict(dyn.get("parameters") or {}),
        "equations": [_squash(e) for e in (dyn.get("state_equations") or [])],
    }


def diff(prev, cur):
    # list of {"lever", "what", "from", "to"}; lever uses the same names as the suggestions
    changes = []
    for out in sorted(set(prev["references"]) | set(cur["references"]), key=str):
        a, b = prev["references"].get(out), cur["references"].get(out)
        if a != b:
            changes.append({"lever": "reference", "what": "reference for %s" % out,
                            "target": out, "from": a, "to": b})
    if prev["x0"] != cur["x0"]:
        changes.append({"lever": "initial_condition", "what": "initial condition x0",
                        "from": prev["x0"], "to": cur["x0"]})
    if prev["dt"] != cur["dt"]:
        changes.append({"lever": "step_time", "what": "solver step time dt",
                        "from": prev["dt"], "to": cur["dt"]})
    if prev["sim_time"] != cur["sim_time"]:
        changes.append({"lever": None, "what": "simulation time",
                        "from": prev["sim_time"], "to": cur["sim_time"]})
    for name in sorted(set(prev["parameters"]) | set(cur["parameters"])):
        a, b = prev["parameters"].get(name), cur["parameters"].get(name)
        if a != b:
            changes.append({"lever": "system", "what": "plant parameter %s" % name,
                            "from": a, "to": b})
    if prev["equations"] != cur["equations"]:
        changes.append({"lever": "system", "what": "state equations", "from": None, "to": None})
    return changes


def _same_value(lever, a, b):
    if lever == "initial_condition":
        try:
            xs = [float(p) for p in str(b).split(",")]
            return list(a) == xs
        except (TypeError, ValueError):
            return False
    if lever == "step_time":
        try:
            return float(a) == float(b)
        except (TypeError, ValueError):
            return False
    return _squash(a) == _squash(b)


def _match_suggestions(changes, suggestions):
    # tags each change with the earlier suggestion it followed: the exact value offered,
    # the same lever with the user's own value, or nothing (the user went their own way)
    for ch in changes:
        ch["followed"] = None
        for s in suggestions or []:
            if s.get("lever") != ch["lever"]:
                continue
            if ch.get("target") and s.get("target") and s["target"] != ch["target"]:
                continue
            exact = any(_same_value(ch["lever"], ch["to"], v) for v in s.get("options") or [])
            ch["followed"] = {"title": s["title"], "exact_option": exact}
            if exact:
                break
    return changes


def outcome(metrics, context=None):
    # how the run ended, as plain facts. unstable_at_s is the first non-finite/huge row, in seconds
    metrics = metrics or {}
    reason = str(metrics.get("success_reason") or "")
    out = {"success": bool(metrics.get("success")),
           "failed_check": reason.split(":")[0] if reason else None,
           "tracking_pct": metrics.get("tracking_pct_headline"),
           "steady_rms_frac": metrics.get("steady_rms_frac"),
           "unstable_at_s": None}
    if context:
        try:
            from backend_core.AgentAdaptive.tools.diagnostics_evidence import first_instability_index
            idx = first_instability_index(context["x"], x0=context.get("x0"))
            if idx is not None:
                out["unstable_at_s"] = float(np.asarray(context["t"], dtype=float)[idx])
        except Exception:  # noqa: BLE001 -- a missing fact beats a broken run
            pass
    return out


def make_attempt(previous, spec, metrics, context=None, report=None, parent=None):
    # one record per finished run. `previous` is this chain's attempts so far; `parent` is the
    # earlier plant's attempts, used only to diff against when this chain has no run yet
    snap = snapshot(spec)
    changes = changes_for_evidence(previous, spec, parent)
    report = report if isinstance(report, dict) and not report.get("error") else {}
    note = str(report.get("note") or "").strip()[:NOTE_CHARS] or None
    suggestions = [{"title": s["title"], "lever": s.get("lever"), "target": s.get("target"),
                    "options": [o["value"] for o in s.get("options") or []]}
                   for s in report.get("suggestions") or []]
    return {"n": (previous[-1]["n"] + 1) if previous else 1, "snapshot": snap, "changes": changes,
            "outcome": outcome(metrics, context), "headline": report.get("headline"),
            "note": note, "suggestions": suggestions}


def record(chain, attempt):
    chain["attempts"].append(attempt)
    del chain["attempts"][:-MAX_ATTEMPTS_KEPT]


def parent_attempts(registry, chain):
    parent = registry.get(chain.get("parent"))
    return list(parent["attempts"]) if parent else []


def history_for_evidence(attempts, parent=None):
    # what the model gets: the last few runs of this plant (plus the last run of the plant it
    # was edited from, marked as such), without the raw snapshots, which are only for diffing
    rows = []
    if parent:
        rows.append(_view(parent[-1], earlier_plant_version=True))
    rows += [_view(a) for a in attempts[-MAX_ATTEMPTS_SHOWN:]]
    return rows


def _view(attempt, earlier_plant_version=False):
    view = {"run": attempt["n"], "outcome": attempt["outcome"], "headline": attempt["headline"],
            "note": attempt["note"], "changes_since_previous_run": attempt["changes"],
            "suggestions_offered": [{k: s[k] for k in ("title", "lever", "target", "options")}
                                    for s in attempt["suggestions"]]}
    if earlier_plant_version:
        view["earlier_plant_version"] = True
    return view


def changes_for_evidence(attempts, spec, parent=None):
    # the user's edits between the last recorded run and the one being diagnosed now
    base = attempts or parent
    if not base:
        return None
    return _match_suggestions(diff(base[-1]["snapshot"], snapshot(spec)), base[-1].get("suggestions"))
