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
        "input_limits": {l["input"]: [l["min"], l["max"]]
                         for l in (dyn.get("input_limits") or []) if isinstance(l, dict)},
        "output_bounds": {b["output"]: [b["min"], b["max"]]
                          for b in (dyn.get("output_bounds") or []) if isinstance(b, dict)},
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
    # .get: a snapshot stored before input limits existed has no such key
    old_lim, new_lim = prev.get("input_limits") or {}, cur.get("input_limits") or {}
    for name in sorted(set(old_lim) | set(new_lim)):
        if old_lim.get(name) != new_lim.get(name):
            changes.append({"lever": "input_limits", "what": "input limit for %s" % name,
                            "target": name, "from": old_lim.get(name), "to": new_lim.get(name)})
    old_b, new_b = prev.get("output_bounds") or {}, cur.get("output_bounds") or {}
    for name in sorted(set(old_b) | set(new_b)):
        if old_b.get(name) != new_b.get(name):
            changes.append({"lever": "output_bounds", "what": "output bound for %s" % name,
                            "target": name, "from": old_b.get(name), "to": new_b.get(name)})
    for name in sorted(set(prev["parameters"]) | set(cur["parameters"])):
        a, b = prev["parameters"].get(name), cur["parameters"].get(name)
        if a != b:
            changes.append({"lever": "system", "what": "plant parameter %s" % name,
                            "from": a, "to": b})
    if prev["equations"] != cur["equations"]:
        changes.append({"lever": "system", "what": "state equations", "from": None, "to": None})
    return changes


_RANGE_LEVERS = ("input_limits", "output_bounds")


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
    if lever in _RANGE_LEVERS:
        # option value is "lo,hi" with a blank for a free side; a is [lo, hi] (None = free)
        try:
            lo, hi = [(float(p) if p.strip() else None) for p in str(b).split(",")]
            return list(a or []) == [lo, hi]
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
                    "computed": bool(s.get("computed")),
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


def _limit_words(lim):
    if not lim:
        return "no limit"
    lo, hi = lim
    return "[%s, %s]" % ("-inf" if lo is None else "%g" % lo, "+inf" if hi is None else "%g" % hi)


def _change_words(ch):
    what, a, b = ch["what"], ch.get("from"), ch.get("to")
    if ch["lever"] in _RANGE_LEVERS:
        if not b:
            text = "%s was removed (it was %s)" % (what, _limit_words(a))
        elif not a:
            text = "%s was added: %s" % (what, _limit_words(b))
        else:
            text = "%s changed from %s to %s" % (what, _limit_words(a), _limit_words(b))
    elif a is None and b is None:
        text = "%s was edited" % what
    else:
        text = "%s changed from %s to %s" % (what, a, b)
    follow = ch.get("followed")
    if follow and follow.get("exact_option"):
        # only a value that really was the suggested one is credited to the suggestion
        text += ' (the value suggested under "%s")' % follow["title"]
    return text


def describe_changes(previous_runs, changes, now):
    # the bullet "what changed since the last run and what happened", written here from computed
    # facts so the model cannot misread an outcome. now = this run's metrics (success, tracking)
    if not previous_runs:
        return None
    last = previous_runs[-1]
    before = last.get("outcome") or {}

    def result(success, pct):
        return "%s, tracking %.1f%%" % ("passed" if success else "failed", pct or 0.0)

    cur_pct = (now or {}).get("tracking_pct_headline")
    cur_ok = bool((now or {}).get("success"))
    prev_pct = before.get("tracking_pct")
    if before.get("success") == cur_ok and abs((cur_pct or 0.0) - (prev_pct or 0.0)) < 5.0:
        trend = "about the same"
    elif (cur_ok and not before.get("success")) or (cur_ok == before.get("success")
                                                  and (cur_pct or 0.0) > (prev_pct or 0.0)):
        trend = "better"
    else:
        trend = "worse"
    label = "run %d%s" % (int(last["run"]), " (an earlier version of the plant)"
                          if last.get("earlier_plant_version") else "")
    what = ("; ".join(_change_words(c) for c in changes) if changes
            else "nothing you can edit here was different")
    return ("Since %s: %s. That run %s; this run %s, which is %s."
            % (label, what, result(before.get("success"), prev_pct), result(cur_ok, cur_pct), trend))


def _norm_value(lever, value):
    # one spelling per value so "0.4*sin(5*t)" == "0.4 * sin(5*t)", "0.2, 0" == "0.2,0.0"
    try:
        if lever == "initial_condition":
            return ",".join("%g" % float(p) for p in str(value).split(","))
        if lever == "step_time":
            return "%g" % float(value)
        if lever in _RANGE_LEVERS:
            lo, hi = [(float(p) if p.strip() else None) for p in str(value).split(",")]
            return "%s,%s" % ("" if lo is None else "%g" % lo, "" if hi is None else "%g" % hi)
    except (TypeError, ValueError):
        pass
    return _squash(value)


def _card_key(s):
    return (s.get("lever"), s.get("target"),
            # options are plain values in a stored attempt, {"label", "value"} in a fresh card
            tuple(sorted(_norm_value(s.get("lever"), v["value"] if isinstance(v, dict) else v)
                         for v in s.get("options") or [])))


def suggestion_memory(attempts, spec, changes_now=None):
    # what the diagnoser must not say again, from every kept run (not only the few shown to the
    # model): `tried` = values of the editable inputs that were run and failed (this run counts),
    # `ignored` = (lever, target) offered at least twice and never taken exactly
    tried = {}

    def add(lever, target, value):
        tried.setdefault((lever, target), set()).add(_norm_value(lever, value))

    failed = [a["snapshot"] for a in attempts if not (a.get("outcome") or {}).get("success")]
    for snap in failed + [snapshot(spec)]:
        for out, expr in (snap.get("references") or {}).items():
            add("reference", out, expr)
        if snap.get("x0"):
            add("initial_condition", None, ",".join(str(v) for v in snap["x0"]))
        if snap.get("dt") is not None:
            add("step_time", None, snap["dt"])
        for name, lim in (snap.get("input_limits") or {}).items():
            add("input_limits", name, ",".join("" if v is None else str(v) for v in lim))
        for name, lim in (snap.get("output_bounds") or {}).items():
            add("output_bounds", name, ",".join("" if v is None else str(v) for v in lim))

    offers, computed = {}, {}
    for i, att in enumerate(attempts):
        later = attempts[i + 1]["changes"] if i + 1 < len(attempts) else (changes_now or [])
        # only taking the offered value counts: changing the same lever by hand is not following
        followed = {c["followed"]["title"] for c in later or []
                    if c.get("followed") and c["followed"].get("exact_option")}
        for sug in att.get("suggestions") or []:
            if not sug.get("lever"):
                continue
            key = (sug["lever"], sug.get("target"))
            rec = offers.setdefault(key, [0, 0])
            rec[0] += 1
            rec[1] += 1 if sug["title"] in followed else 0
            if sug.get("computed"):
                crec = computed.setdefault(key, [0, 0])
                crec[0] += 1
                crec[1] += 1 if sug["title"] in followed else 0
    ignored = {key for key, (n_off, n_fol) in offers.items() if n_off >= 2 and n_fol == 0}
    # a computed suggestion (a fact that stays true) is shown up to three times, then rests
    tired = {key for key, (n_off, n_fol) in computed.items() if n_off >= 3 and n_fol == 0}
    return {"tried": tried, "ignored": ignored, "tired": tired}


def memory_for_evidence(memory):
    # the same memory in words, so the model sees why a value will not be accepted
    names = {"reference": "reference", "initial_condition": "initial condition x0",
             "step_time": "solver step dt", "input_limits": "input limit",
             "output_bounds": "output bound"}
    out = {}
    for (lever, target), values in sorted(memory["tried"].items(), key=str):
        label = names.get(lever, lever) + ((" of %s" % target) if target else "")
        out[label] = sorted(values)
    return {"values_already_run_and_failed": out}
