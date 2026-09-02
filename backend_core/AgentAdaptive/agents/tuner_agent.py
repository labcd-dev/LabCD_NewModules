import io
import os
import json
import time
import traceback
import contextlib

from . import llm_factory
from backend_core.AgentAdaptive.tools import model_pricing
from backend_core.AgentAdaptive.tools import tuning_objectives as tuning_objectives_mod
from backend_core.AgentAdaptive.tools import system_spec as system_spec_mod
from .prompt_loader import load_prompt

from backend_core.AgentAdaptive.controller.runs import _run_smc, _run_backstepping
from backend_core.AgentAdaptive.tools.scoring import format_metrics_report, _fmt_list
from backend_core.AgentAdaptive.tools.reporter import render_final_report, _render_clarification_section
from backend_core.AgentAdaptive.tools.progress import _emit, _remap_note_stage
from backend_core.AgentAdaptive.tools.series_export import extract_series
from .designer_agent import run_extraction
from .report_writer import write_abstract
from .agent_io import (
    _extract_json_payload, _empty_usage, _sum_usage,
    _sum_usage_from_messages, resolved_models, _SyntheticMessage,
)

TUNER_SYSTEM_PROMPT = load_prompt("tuner_agent_prompt.yaml")

# Climb agents -> AgentAdaptive -> backend_core -> repo root for the debug log.
TUNER_DEBUG_LOG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "tuner_debug.log")


def _debug(title, **fields):
    try:
        with open(TUNER_DEBUG_LOG, "a", encoding="utf-8") as fh:
            fh.write("\n%s\n=== %s  %s ===\n"
                     % ("-" * 70, title, time.strftime("%Y-%m-%d %H:%M:%S")))
            for key, value in fields.items():
                text = value if isinstance(value, str) else repr(value)
                fh.write("--- %s ---\n%s\n" % (key, text))
    except Exception:
        pass

DEFAULT_OPENAI_MODEL = llm_factory.DEFAULT_OPENAI_MODEL

SMC_TUNING_DEFAULTS = dict(
    surface_lambda=2, K=1.5, Lam=5, phi_layer=0.05, Gamma=25, kappa=5,
    kappa_s=None, k2=1, k3=1, k4=1, sigma_W=0.1, N=25, width=1.5,
    rbf_spread=1.0, rbf_normalize="meanstd",
)
BACKSTEPPING_TUNING_DEFAULTS = dict(
    c_gains=None, Gamma=25, kappa=5, k2=1, k3=1, k4=1, sigma_W=0.1,
    tau=0.05, N=25, width=1.5, rbf_spread=1.0, rbf_normalize="meanstd",
    use_filtered_error=False, lambda_I=0.5,
)

TUNING_SYMPTOMS = (
    "numerical_divergence", "steady_state_error", "slow_transient",
    "high_overshoot", "control_effort_too_high", "chattering", "estimator_lag",
)

_SYMPTOM_PARAM_FAMILY = {
    "numerical_divergence": {"K", "Lam", "c_gains", "Gamma", "kappa", "surface_lambda"},
    "steady_state_error": {"K", "Lam", "c_gains", "Gamma", "kappa", "kappa_s", "k2", "k3", "k4"},
    "slow_transient": {"K", "Lam", "c_gains"},
    "high_overshoot": {"K", "Lam", "c_gains"},
    "control_effort_too_high": {"K", "Lam", "c_gains"},
    "chattering": {"phi_layer", "K"},
    "estimator_lag": {"Gamma", "sigma_W"},
}

# capping changes per round so we can actually tell which change did what.
# otherwise you'd have five things moving at once, with no idea which one fixed (or broke) it
MAX_TUNING_PARAMS_PER_ROUND = 3

# don't let one bad LLM proposal 100x a gain in one shot: cap moves to roughly
# double/half per round (like gain scheduling), it can still get there, just over several rounds
TUNING_RATE_LIMIT_MAX_FACTOR = 2.0   # new_val <= old_val * this
TUNING_RATE_LIMIT_MIN_FACTOR = 0.5   # new_val >= old_val * this

# N is an int (rbf count), not a continuous gain, but it's still in here since
# jumping it too far in one round can wreck the estimator just like any other gain
_RATE_LIMITED_PARAMS = {
    "surface_lambda", "K", "Lam", "phi_layer", "kappa_s", "c_gains",
    "tau", "lambda_I", "Gamma", "kappa", "k2", "k3", "k4", "sigma_W",
    "N", "width", "rbf_spread",
}


def _tuning_diff(prev_tuning, new_tuning):
    diff = {}
    for key, new_val in new_tuning.items():
        old_val = prev_tuning.get(key)
        if old_val != new_val:
            diff[key] = (old_val, new_val)
    return diff


def _format_tuning_diff(diff):
    if not diff:
        return "(no change from previous round)"
    return "; ".join("%s: %s -> %s" % (k, old, new) for k, (old, new) in diff.items())


def _target_met(metrics, target_rms_frac, selection=None):
    if not metrics.get("numerically_healthy", True):
        return False
    # if the user picked priorities and none of them are about tracking, hitting the RMS
    # target doesn't count as "done" (we'd otherwise stop before touching what they asked for)
    if selection and not any(k in selection for k in
                             ("steady_state_error", "tracking_mse")):
        return False
    frac = metrics.get("steady_rms_frac")
    if not frac:
        return False
    return all(f <= target_rms_frac for f in frac)


def _scope_warning(symptom, changed, selection=None):
    if selection and symptom not in tuning_objectives_mod.allowed_symptoms(selection):
        return True
    family = _SYMPTOM_PARAM_FAMILY.get(symptom)
    if not family or not changed:
        return False
    return any(k not in family for k in changed)


def _format_tuning_ledger(rows, limit=4):
    if not rows:
        return "(no rounds yet)"
    lines = []
    for r in rows[-limit:]:
        reasoning = r.get("reasoning") or ""
        if len(reasoning) > 300:
            reasoning = reasoning[:300] + "..."
        headline = r.get("tracking_pct_headline")
        tracking_str = ("%.1f%%" % headline) if isinstance(headline, (int, float)) else "n/a"
        if "success" in r:
            failed_check = (r.get("success_reason") or "?").split(":")[0]
            verdict_str = "PASS" if r["success"] else ("FAIL (%s)" % failed_check)
        else:
            verdict_str = "n/a"
        lines.append(
            "round %d: changed=%s | symptom=%s%s | verdict=%s | tracking=%s | "
            "steady_rms_frac=%s | healthy=%s | met_target=%s | reasoning: %s"
            % (r["round"], _format_tuning_diff(r["changed"]), r["symptom"],
               " (SCOPE WARNING: touched params outside this symptom's family)"
               if r["scope_warning"] else "",
               verdict_str, tracking_str,
               r["steady_rms_frac"], r["numerically_healthy"], r["met_target"],
               reasoning)
        )
    return "\n".join(lines)


_TUNER_FLOAT_PARAMS = ("surface_lambda", "K", "Lam", "phi_layer", "kappa_s",
                       "tau", "lambda_I", "Gamma", "kappa", "k2", "k3", "k4",
                       "sigma_W", "width", "rbf_spread")
_TUNER_INT_PARAMS = ("N",)
_TUNER_BOOL_PARAMS = ("use_filtered_error",)
_TUNER_STR_PARAMS = ("rbf_normalize",)
_TUNER_LIST_PARAMS = ("c_gains",)
_TUNER_PARAM_NAMES = (_TUNER_FLOAT_PARAMS + _TUNER_INT_PARAMS
                      + _TUNER_BOOL_PARAMS + _TUNER_STR_PARAMS + _TUNER_LIST_PARAMS)


def _coerce_tuning_payload(payload):
    if not isinstance(payload, dict):
        raise TypeError("the reply was not a JSON object")
    missing = [k for k in ("reasoning", "diagnosed_symptom") if k not in payload]
    if missing:
        raise KeyError("missing required field(s): %s" % ", ".join(missing))
    reasoning = str(payload["reasoning"])
    diagnosed_symptom = str(payload["diagnosed_symptom"])
    overrides = {}
    for k in _TUNER_FLOAT_PARAMS:
        v = payload.get(k)
        if v is not None:
            overrides[k] = float(v)
    for k in _TUNER_INT_PARAMS:
        v = payload.get(k)
        if v is not None:
            overrides[k] = int(v)
    for k in _TUNER_BOOL_PARAMS:
        v = payload.get(k)
        if v is not None:
            overrides[k] = bool(v)
    for k in _TUNER_STR_PARAMS:
        v = payload.get(k)
        if v is not None:
            overrides[k] = str(v)
    for k in _TUNER_LIST_PARAMS:
        v = payload.get(k)
        if v is not None:
            overrides[k] = [float(x) for x in v]
    return reasoning, diagnosed_symptom, overrides


def _apply_tuning_proposal(reasoning, diagnosed_symptom, overrides, fixed_args,
                           current_tuning, session, structure_cache=None,
                           allowed_params=None):
    method = fixed_args.get("method", "smc")
    method_label = "SMC" if method == "smc" else "Backstepping"
    _param_names = _TUNER_PARAM_NAMES
    # this has to run before the count-cap below since an out-of-scope param must never
    # take one of the 3 allowed slots. drop it first, then count-cap what's left
    fence_note = ""
    if allowed_params is not None:
        out_of_scope = [k for k in _param_names
                        if k in overrides and k not in allowed_params]
        if out_of_scope:
            overrides = {k: v for k, v in overrides.items()
                        if k not in out_of_scope}
            if not overrides:
                # nothing survived the fence, and re-simulating now would just repeat last round's
                # numbers, so bail out here instead of burning a real simulation for nothing
                return (
                    "NO CHANGE APPLIED: every parameter you proposed (%s) is "
                    "out of scope for this run, so nothing was changed and no "
                    "simulation was run. Propose again choosing only from: %s."
                    % (", ".join(out_of_scope), ", ".join(sorted(allowed_params)))
                )
            fence_note = (
                "NOTE: %d parameter(s) were ignored this round because they "
                "are out of scope for this run: %s. Only these may be "
                "changed: %s."
                % (len(out_of_scope), ", ".join(out_of_scope),
                   ", ".join(sorted(allowed_params)))
            )
    drop_note = ""
    if len(overrides) > MAX_TUNING_PARAMS_PER_ROUND:
        # ranking: in-scope params win over the diagnosed symptom's family, which wins over
        # declaration order. that ordering is what keeps this deterministic instead of picking whatever dict order gives us
        family = _SYMPTOM_PARAM_FAMILY.get(diagnosed_symptom, set())
        _decl_order = {k: i for i, k in enumerate(_param_names)}
        keep = sorted(
            (k for k in _param_names if k in overrides),
            key=lambda k: (0 if (allowed_params is None or k in allowed_params) else 1,
                          0 if k in family else 1,
                          _decl_order[k]))
        keep = keep[:MAX_TUNING_PARAMS_PER_ROUND]
        dropped_names = [k for k in _param_names if k in overrides and k not in keep]
        overrides = {k: overrides[k] for k in keep}
        drop_note = (
            "NOTE: %d parameter(s) were ignored this round (limit is %d "
            "changed parameters per round): %s. Consider changing them in a "
            "later round."
            % (len(dropped_names), MAX_TUNING_PARAMS_PER_ROUND, ", ".join(dropped_names))
        )
    # clamps whatever's left after the cap above to within 0.5x-2x of its value at the
    # start of this round; c_gains is a list, so each element gets clamped on its own
    clamped_notes = []
    for param in list(overrides.keys()):
        if param not in _RATE_LIMITED_PARAMS:
            continue
        new_val = overrides[param]
        old_val = current_tuning.get(param)
        if param == "c_gains":
            if not isinstance(old_val, (list, tuple)) or len(old_val) != len(new_val):
                continue
            clamped_list = []
            changed = False
            for old_g, new_g in zip(old_val, new_val):
                if old_g is None or old_g == 0:
                    clamped_list.append(new_g)
                    continue
                lo = old_g * TUNING_RATE_LIMIT_MIN_FACTOR
                hi = old_g * TUNING_RATE_LIMIT_MAX_FACTOR
                c = max(lo, min(hi, new_g))
                if c != new_g:
                    changed = True
                clamped_list.append(c)
            if changed:
                clamped_notes.append(
                    "%s: requested %s, clamped to %s (each element bounded to "
                    "%sx/%sx of previous value %s)"
                    % (param, new_val, clamped_list, TUNING_RATE_LIMIT_MAX_FACTOR,
                       TUNING_RATE_LIMIT_MIN_FACTOR, old_val)
                )
                overrides[param] = clamped_list
            continue
        if old_val is None or old_val == 0:
            continue
        lo = old_val * TUNING_RATE_LIMIT_MIN_FACTOR
        hi = old_val * TUNING_RATE_LIMIT_MAX_FACTOR
        clamped = max(lo, min(hi, new_val))
        if param == "N":
            clamped = int(round(clamped))
        if clamped != new_val:
            clamped_notes.append(
                "%s: requested %s, clamped to %s (max change per round is "
                "%sx/%sx of previous value %s)"
                % (param, new_val, clamped, TUNING_RATE_LIMIT_MAX_FACTOR,
                   TUNING_RATE_LIMIT_MIN_FACTOR, old_val)
            )
            overrides[param] = clamped
    current_tuning.update(overrides)
    try:
        if method == "smc":
            components, metrics = _run_smc(
                fixed_args["states"], fixed_args["dynamics"], fixed_args["inputs"],
                fixed_args["outputs"], fixed_args["x0"], fixed_args["refs"],
                fixed_args["has_delta"], fixed_args["has_disturbance"],
                fixed_args.get("delta_exprs"), fixed_args.get("dist_exprs"),
                current_tuning["surface_lambda"], current_tuning["K"], current_tuning["Lam"],
                current_tuning["phi_layer"], current_tuning["Gamma"], current_tuning["kappa"],
                current_tuning["kappa_s"], current_tuning["k2"], current_tuning["k3"],
                current_tuning["k4"], current_tuning["sigma_W"], current_tuning["N"],
                current_tuning["width"], current_tuning["rbf_spread"], current_tuning["rbf_normalize"],
                fixed_args["dt"], fixed_args["t_end"], reasoning, for_tuning=True,
                structure_cache=structure_cache, fail_tol=fixed_args.get("fail_tol", 0.02))
        else:
            components, metrics = _run_backstepping(
                fixed_args["states"], fixed_args["dynamics"], fixed_args["inputs"],
                fixed_args["outputs"], fixed_args["x0"], fixed_args["refs"],
                fixed_args["has_delta"], fixed_args["has_disturbance"],
                fixed_args.get("delta_exprs"), fixed_args.get("dist_exprs"),
                current_tuning["c_gains"], current_tuning["Gamma"], current_tuning["kappa"],
                current_tuning["k2"], current_tuning["k3"], current_tuning["k4"],
                current_tuning["sigma_W"], current_tuning["tau"], current_tuning["N"],
                current_tuning["width"], current_tuning["rbf_spread"], current_tuning["rbf_normalize"],
                current_tuning["use_filtered_error"], current_tuning["lambda_I"],
                fixed_args.get("filtered_error_output_index", 0),
                fixed_args["dt"], fixed_args["t_end"], reasoning, for_tuning=True,
                structure_cache=structure_cache, fail_tol=fixed_args.get("fail_tol", 0.02))
    except Exception as e:
        return "DESIGN FAILED with these parameters (%s: %s). Try less aggressive values." % (type(e).__name__, e)
    explicit_unc = bool(fixed_args.get("delta_exprs")) or bool(fixed_args.get("dist_exprs"))
    report = format_metrics_report(metrics, method_label, fixed_args["has_delta"],
                                    fixed_args["has_disturbance"], explicit_unc)
    if fence_note:
        report = report + "\n\n" + fence_note
    if drop_note:
        report = report + "\n\n" + drop_note
    if clamped_notes:
        report = report + "\n\nNOTE: rate-limit margin applied to %d parameter(s):\n" \
            % len(clamped_notes) + "\n".join(clamped_notes)
    session["history"].append({"tuning": dict(current_tuning), "metrics": metrics,
                               "reasoning": reasoning, "symptom": diagnosed_symptom,
                               "components": components})
    return report


def run_tuner_round(fixed_args, current_tuning, session, prompt, structure_cache=None,
                    allowed_params=None, round_num=None):
    llm = llm_factory.build_llm("tuner")
    messages = [{"role": "system", "content": TUNER_SYSTEM_PROMPT},
               {"role": "user", "content": prompt}]
    _debug("REQUEST round %s" % round_num,
          current_tuning_before=dict(current_tuning),
          messages=json.dumps(messages, indent=2, ensure_ascii=False))
    resp = llm.invoke(messages)
    _debug("REPLY round %s" % round_num,
          content=resp.content, usage_metadata=getattr(resp, "usage_metadata", None))
    payload, parse_err = _extract_json_payload(resp.content)
    if payload is None:
        print("note: tuner reply could not be read as JSON (%s: %s). Round produced nothing"
             % (type(parse_err).__name__, parse_err))
        _debug("PARSE FAILED round %s" % round_num,
              error="%s: %s" % (type(parse_err).__name__, parse_err))
        return {"messages": [resp]}
    try:
        reasoning, diagnosed_symptom, overrides = _coerce_tuning_payload(payload)
    except Exception as e:
        print("note: tuner reply was missing required field(s) (%s: %s); round produced nothing"
             % (type(e).__name__, e))
        _debug("PARSE FAILED round %s" % round_num, error="%s: %s" % (type(e).__name__, e))
        return {"messages": [resp]}
    _debug("PARSED round %s" % round_num, reasoning=reasoning,
          diagnosed_symptom=diagnosed_symptom, overrides=overrides)
    report = _apply_tuning_proposal(
        reasoning, diagnosed_symptom, overrides, fixed_args, current_tuning,
        session, structure_cache=structure_cache, allowed_params=allowed_params)
    _debug("RESULT round %s" % round_num,
          current_tuning_after=dict(current_tuning), report=report)
    if report.startswith("DESIGN FAILED") or report.startswith("NO CHANGE APPLIED"):
        # _SyntheticMessage carries no usage_metadata by design, so the usage-summing code
        # still attributes this whole round's token cost to `resp` alone, not double-counted
        return {"messages": [resp, _SyntheticMessage(report)]}
    return {"messages": [resp]}


def run_tuning_loop(fixed_args, initial_components, initial_metrics,
                     target_rms_frac=0.02, max_rounds=4,
                     on_event=None, should_stop=None, structure_cache=None,
                     objectives=None):
    method = fixed_args.get("method", "smc")
    method_label = "SMC" if method == "smc" else "Backstepping"
    defaults = SMC_TUNING_DEFAULTS if method == "smc" else BACKSTEPPING_TUNING_DEFAULTS
    current_tuning = dict(defaults)
    for k in defaults:
        if k in fixed_args and fixed_args[k] is not None:
            current_tuning[k] = fixed_args[k]

    if method == "backstepping" and current_tuning.get("c_gains") is None:
        # without this, c_gains stays None, and the tuner literally can't see it to tune.
        # here's why it's forced: a real run once burned 23 rounds cranking estimator gains while c_gains sat untouched
        current_tuning["c_gains"] = [2.0 + 2.0 * i for i in range(len(fixed_args["states"]))]

    selection = tuning_objectives_mod.normalize_selection(objectives)
    if selection:
        has_estimator = bool(fixed_args.get("has_delta")
                             or fixed_args.get("has_disturbance"))
        measurable = set(tuning_objectives_mod.applicable_objectives(
            method, has_estimator=has_estimator))
        selection = tuning_objectives_mod.normalize_selection(
            {k: w for k, w in selection.items() if k in measurable})
    priorities_block = tuning_objectives_mod.format_priorities_block(selection)

    _tracking_ticked = (not selection
                        or any(k in selection for k in
                               ("steady_state_error", "tracking_mse")))
    # don't tell the model to chase the RMS target if the user's priorities have nothing
    # to do with tracking: that's a mixed signal it has no way to resolve on its own
    if _tracking_ticked:
        target_sentence = (
            "Target: steady-state RMS should be <= %.3f (as a fraction of "
            "reference amplitude) for every output." % target_rms_frac)
    else:
        target_sentence = (
            "Target: improve the objectives listed in USER PRIORITIES above. "
            "The steady-state RMS target does NOT apply this run, so do not "
            "spend a round chasing it.")

    explicit_unc = bool(fixed_args.get("delta_exprs")) or bool(fixed_args.get("dist_exprs"))
    initial_met = _target_met(initial_metrics, target_rms_frac, selection)
    tuning_log = [{
        "round": 0,
        "reasoning": ("(initial design already meets the target: no tuning needed)"
                      if initial_met else "(initial design, before tuning)"),
        "report": format_metrics_report(initial_metrics, method_label, fixed_args["has_delta"],
                                         fixed_args["has_disturbance"], explicit_unc),
        "met_target": initial_met,
        "tuning": dict(current_tuning),
        "changed": {},
        "success": initial_metrics.get("success", True) if initial_metrics else False,
        "tracking_pct_headline": (initial_metrics.get("tracking_pct_headline")
                                   if initial_metrics else None),
        "objective_values": tuning_objectives_mod.objective_rows(
            selection, initial_metrics, fixed_args),
        "objectives": dict(selection),
    }]
    best = {"components": initial_components, "metrics": initial_metrics,
            "tuning": dict(current_tuning), "round": 0}
    usage_total = _empty_usage()
    ledger_rows = []
    timeline = []

    if tuning_log[0]["met_target"] or max_rounds < 1:
        _emit(on_event, kind="stage_done", stage="tuning", round=0,
              met_target=tuning_log[0]["met_target"], skipped=True,
              reasoning=tuning_log[0]["reasoning"], changed=tuning_log[0]["changed"])
        timeline.append({
            "actor": "tuner", "round": 0, "label": "Tuning: not needed",
            "tokens": 0, "input_tokens": 0, "output_tokens": 0,
            "detail": tuning_log[0]["reasoning"],
        })
        return tuning_log, best, usage_total, timeline

    session = {"history": []}
    allowed_params = tuning_objectives_mod.allowed_params(selection)

    def _score(m):
        # gate on "bounded" only, not full success. gating on full success stuck `best`
        # on round 0 forever, since every round scored inf until target was actually hit.
        if not m.get("numerically_healthy", True):
            return float("inf")
        if not m.get("success_checks", {}).get("bounded", True):
            return float("inf")
        if selection:
            weighted = tuning_objectives_mod.weighted_score(
                m, selection, baseline_metrics=initial_metrics,
                fixed_args=fixed_args)
            if weighted is not None:
                return weighted
        return sum(m.get("steady_rms_frac", [float("inf")]))

    for round_num in range(1, max_rounds + 1):
        if should_stop is not None and should_stop():
            _emit(on_event, kind="cancelled", stage="tuning", round=round_num,
                  reasoning="(stopped before this round started)", changed={})
            break
        _emit(on_event, kind="stage_start", stage="tuning",
              round=round_num, max_rounds=max_rounds,
              current_tuning=dict(current_tuning))
        _emit(on_event, kind="note", stage="tuning", round=round_num,
              text="Tuner Agent is proposing new tuning parameters for round %d..." % round_num)
        prompt = (
            "Round %d of at most %d. Current tuning parameters: %s\n\n"
            "Metrics from the last simulation:\n%s\n\n"
            "Your last rounds (most recent last):\n%s\n\n"
            "%s Reply with one JSON object proposing new tuning parameters, "
            "as described in your system prompt."
            % (round_num, max_rounds, current_tuning, tuning_log[-1]["report"],
               _format_tuning_ledger(ledger_rows), target_sentence)
        )
        if priorities_block:
            prompt = priorities_block + "\n\n" + prompt
        n_before = len(session["history"])
        result = run_tuner_round(fixed_args, current_tuning, session, prompt,
                                 structure_cache=structure_cache,
                                 allowed_params=allowed_params, round_num=round_num)
        round_usage = _sum_usage_from_messages(result["messages"])
        usage_total = _sum_usage(usage_total, round_usage)

        if len(session["history"]) == n_before:
            tuning_log.append({
                "round": round_num, "reasoning": "(no usable proposal this round)",
                "report": result["messages"][-1].content, "met_target": False,
                "tuning": dict(current_tuning), "changed": {},
                "success": False, "tracking_pct_headline": None,
            })
            timeline.append({
                "actor": "tuner", "round": round_num, "label": "Tuning, round %d" % round_num,
                "tokens": round_usage["total_tokens"], "input_tokens": round_usage["input_tokens"],
                "output_tokens": round_usage["output_tokens"], "detail": "proposal not parsed",
            })
            _emit(on_event, kind="note", stage="tuning", round=round_num,
                  text="Tuner Agent's proposal could not be parsed this round.")
            _emit(on_event, kind="stage_done", stage="tuning", round=round_num,
                  met_target=False, tool_called=False,
                  reasoning="(no usable proposal this round)", changed={})
            continue

        entry = session["history"][-1]
        met = _target_met(entry["metrics"], target_rms_frac, selection)
        report = format_metrics_report(entry["metrics"], method_label, fixed_args["has_delta"],
                                        fixed_args["has_disturbance"], explicit_unc)
        changed = _tuning_diff(tuning_log[-1]["tuning"], entry["tuning"])
        objective_values = tuning_objectives_mod.objective_rows(
            selection, entry["metrics"], fixed_args)
        tuning_log.append({
            "round": round_num, "reasoning": entry["reasoning"], "report": report,
            "met_target": met, "tuning": dict(entry["tuning"]), "changed": changed,
            "success": entry["metrics"].get("success", True),
            "tracking_pct_headline": entry["metrics"].get("tracking_pct_headline"),
            "objective_values": objective_values,
        })
        ledger_rows.append({
            "round": round_num, "changed": changed, "symptom": entry.get("symptom", "?"),
            "scope_warning": _scope_warning(entry.get("symptom"), changed, selection),
            "steady_rms_frac": (_fmt_list(entry["metrics"].get("steady_rms_frac"))
                                 if entry["metrics"].get("numerically_healthy", True) else "N/A"),
            "numerically_healthy": entry["metrics"].get("numerically_healthy", True),
            "success": entry["metrics"].get("success", True),
            "success_reason": entry["metrics"].get("success_reason", ""),
            "tracking_pct_headline": entry["metrics"].get("tracking_pct_headline"),
            "met_target": met,
            "reasoning": entry["reasoning"],
        })
        if _score(entry["metrics"]) < _score(best["metrics"]):
            best = {"components": entry["components"], "metrics": entry["metrics"],
                    "tuning": dict(entry["tuning"]), "round": round_num}
        _emit(on_event, kind="note", stage="tuning", round=round_num,
              text="Changed %s: %s. Target %s."
              % (_format_tuning_diff(changed), entry["reasoning"],
                 "met" if met else "not yet met"))
        timeline.append({
            "actor": "tuner", "round": round_num, "label": "Tuning, round %d" % round_num,
            "tokens": round_usage["total_tokens"], "input_tokens": round_usage["input_tokens"],
            "output_tokens": round_usage["output_tokens"],
            "detail": "target met" if met else "target not yet met",
        })
        _emit(on_event, kind="stage_done", stage="tuning", round=round_num,
              met_target=met, tool_called=True,
              reasoning=entry["reasoning"], changed=changed,
              objective_values=objective_values)
        if met:
            break

    return tuning_log, best, usage_total, timeline


def _rerun_design_directly(args, for_tuning=False, structure_cache=None, on_event=None):
    method = args.get("method", "smc")
    d = SMC_TUNING_DEFAULTS if method == "smc" else BACKSTEPPING_TUNING_DEFAULTS
    g = lambda k: args.get(k, d[k])
    if method == "smc":
        return _run_smc(
            args["states"], args["dynamics"], args["inputs"], args["outputs"],
            args["x0"], args["refs"], args["has_delta"], args["has_disturbance"],
            args.get("delta_exprs"), args.get("dist_exprs"),
            g("surface_lambda"), g("K"), g("Lam"), g("phi_layer"), g("Gamma"),
            g("kappa"), g("kappa_s"), g("k2"), g("k3"), g("k4"), g("sigma_W"),
            g("N"), g("width"), g("rbf_spread"), g("rbf_normalize"),
            args.get("dt", 0.001), args.get("t_end", 8.0), args.get("reasoning", ""),
            on_event=on_event, for_tuning=for_tuning, structure_cache=structure_cache,
            fail_tol=args.get("fail_tol", 0.02))
    return _run_backstepping(
        args["states"], args["dynamics"], args["inputs"], args["outputs"],
        args["x0"], args["refs"], args["has_delta"], args["has_disturbance"],
        args.get("delta_exprs"), args.get("dist_exprs"),
        g("c_gains"), g("Gamma"), g("kappa"), g("k2"), g("k3"), g("k4"),
        g("sigma_W"), g("tau"), g("N"), g("width"), g("rbf_spread"), g("rbf_normalize"),
        g("use_filtered_error"), g("lambda_I"), args.get("filtered_error_output_index", 0),
        args.get("dt", 0.001), args.get("t_end", 8.0), args.get("reasoning", ""),
        on_event=on_event, for_tuning=for_tuning, structure_cache=structure_cache,
        fail_tol=args.get("fail_tol", 0.02))


def run_full_pipeline(description, enable_tuning=False, target_rms_frac=0.02,
                       max_tuning_rounds=4, on_event=None, should_stop=None,
                       clarification_record=None, sim_overrides=None,
                       clarifier_usage=None, tuning_objectives=None,
                       system_spec=None):
    if not system_spec:
        result = {"messages": [_SyntheticMessage(
            "EXTRACTION FAILED. No confirmed system spec was given. This "
            "pipeline reads the plant JSON + sim-knobs form + Clarifier "
            "output produced by the Streamlit wizard; it does not read a "
            "free-text description.")]}
        usage = {"agent": _empty_usage(), "total": _empty_usage(),
                 "agent_turns": [], "timeline": [], "tuner": _empty_usage(),
                 "clarifier": _empty_usage(), "reporter": _empty_usage(),
                 "models": resolved_models()}
        return result, usage, [], None

    substituted_spec = system_spec_mod.substitute_parameters(system_spec)
    if sim_overrides is None:
        sim_overrides = system_spec_mod.sim_overrides_from_spec(system_spec)

    result, usage, last = run_extraction(
        substituted_spec, on_event=on_event, clarification_record=clarification_record)
    tuning_log, tuning_best = [], None
    tuning_usage = _empty_usage()
    tuning_timeline = []
    reporter_usage = _empty_usage()

    do_build = bool(last and last["ok"])

    if do_build and should_stop is not None and should_stop():
        _emit(on_event, kind="cancelled", stage="build", round=0,
              reasoning="(stopped before the final build could start)", changed={})
        do_build = False

    if not do_build and enable_tuning:
        reason = "Tuning skipped: no successful design was found to tune."
        tuning_log = [{"round": 0, "reasoning": reason, "report": "",
                       "met_target": False, "tuning": {}, "changed": {},
                       "success": False, "tracking_pct_headline": None}]

    if do_build:
        args = dict(last["args"])
        # fail_tol/dt/t_end need to land on `args` itself, not just the fixed_args copy below.
        # the deferred build reads straight off `args`, so a copy-only set would skew round 0's target.
        args["fail_tol"] = float(target_rms_frac) if enable_tuning else 0.02
        if sim_overrides:
            if sim_overrides.get("t_end") is not None:
                args["t_end"] = float(sim_overrides["t_end"])
            if sim_overrides.get("dt") is not None:
                args["dt"] = float(sim_overrides["dt"])
        structure_cache = {}
        final_metrics = None
        try:
            _emit(on_event, kind="stage_start", stage="build", for_tuning=enable_tuning)
            with contextlib.redirect_stdout(io.StringIO()):
                build_components, build_metrics = _rerun_design_directly(
                    args, for_tuning=enable_tuning, structure_cache=structure_cache,
                    on_event=_remap_note_stage(on_event, "design", "build"))
            build_explicit_unc = bool(args.get("delta_exprs")) or bool(args.get("dist_exprs"))
            build_method_label = "SMC" if args.get("method", "smc") == "smc" else "Backstepping"
            build_report_text = format_metrics_report(
                build_metrics, build_method_label, args.get("has_delta"),
                args.get("has_disturbance"), build_explicit_unc) if build_metrics is not None else None
            _emit(on_event, kind="stage_done", stage="build", for_tuning=enable_tuning,
                  report=build_report_text)
            fixed_args = dict(args)
            fixed_args.setdefault("dt", 0.001)
            fixed_args.setdefault("t_end", 8.0)

            final_components, final_args_for_report, final_metrics = build_components, fixed_args, build_metrics

            if enable_tuning:
                tuning_log, tuning_best, tuning_usage, tuning_timeline = run_tuning_loop(
                    fixed_args, build_components, build_metrics,
                    target_rms_frac=target_rms_frac, max_rounds=max_tuning_rounds,
                    on_event=on_event, should_stop=should_stop, structure_cache=structure_cache,
                    objectives=tuning_objectives)

                if tuning_best is not None:
                    final_args = dict(fixed_args)
                    final_args.update(tuning_best["tuning"])
                    # tuning_best's params never ran the real (for_tuning=False) path, so this can
                    # still blow up (fall back to fixed_args, which is proven safe, rather than lose it all)
                    try:
                        with contextlib.redirect_stdout(io.StringIO()):
                            final_components, final_metrics = _rerun_design_directly(
                                final_args, structure_cache=structure_cache,
                                on_event=_remap_note_stage(on_event, "design", "build"))
                        final_args_for_report = final_args
                    except Exception:
                        with contextlib.redirect_stdout(io.StringIO()):
                            final_components, final_metrics = _rerun_design_directly(
                                fixed_args, structure_cache=structure_cache,
                                on_event=_remap_note_stage(on_event, "design", "build"))
                        final_args_for_report = fixed_args

            method_label = "SMC" if fixed_args.get("method", "smc") == "smc" else "Backstepping"
            _tuning_defaults = SMC_TUNING_DEFAULTS if fixed_args.get("method", "smc") == "smc" else BACKSTEPPING_TUNING_DEFAULTS
            tuning_values = {k: final_args_for_report[k] for k in _tuning_defaults
                              if k in final_args_for_report}
            explicit_unc = bool(final_args_for_report.get("delta_exprs")) or bool(final_args_for_report.get("dist_exprs"))
            metrics_report_text = (
                format_metrics_report(final_metrics, method_label,
                                      final_args_for_report.get("has_delta"),
                                      final_args_for_report.get("has_disturbance"),
                                      explicit_unc)
                if final_metrics is not None else None
            )
            final_report_text = render_final_report(
                method_label, final_args_for_report, final_components,
                last.get("why", ""), final_args_for_report.get("notes_limitations"),
                tuning_values=tuning_values, metrics_report_text=metrics_report_text,
            ) + _render_clarification_section(clarification_record)
            result = dict(result)
            result["messages"] = list(result["messages"])
            result["messages"][-1] = _SyntheticMessage(final_report_text)
            if final_metrics is not None:
                final_metrics["mse_target_from_tuner"] = enable_tuning
            result["final_metrics"] = final_metrics
            result["series"] = extract_series(final_components)

            agents_used = ["Design Agent"]
            if clarifier_usage:
                agents_used.insert(0, "Clarifier Agent")
            if enable_tuning:
                agents_used.append("Tuner Agent")
            outcome_bits = []
            if isinstance(final_metrics, dict) and "success" in final_metrics:
                outcome_bits.append(
                    "run verdict PASS" if final_metrics.get("success")
                    else "run verdict FAIL (%s)" % (final_metrics.get("success_reason")
                                                    or "a run check failed"))
            if isinstance(final_metrics, dict) and final_metrics.get("tracking_pct_headline") is not None:
                outcome_bits.append("reference tracking %.1f%% (worst output, steady state)"
                                    % final_metrics["tracking_pct_headline"])
            if enable_tuning and tuning_best is not None:
                best_entry = next((e for e in tuning_log if e["round"] == tuning_best["round"]), None)
                outcome_bits.append(
                    "tuning ran %d round(s), target %s"
                    % (max((e["round"] for e in tuning_log), default=0),
                       "met" if (best_entry and best_entry.get("met_target")) else "not fully met"))
            outcome_text = "; ".join(outcome_bits) or "the run completed."
            abstract_text, reporter_usage = write_abstract(
                substituted_spec.get("system_name"), method_label, agents_used,
                outcome_text, why=last.get("why", ""))
            result["abstract"] = abstract_text
        except Exception as e:
            fail_text = ("Design passed review but failed during the final "
                          "build/simulation: %s: %s" % (type(e).__name__, e))
            # the swallowed message alone doesn't say which float() call saw the
            # complex value -- print the real traceback so it lands in the run's
            # full log instead of vanishing.
            print("build/simulation failed:\n" + traceback.format_exc())
            if enable_tuning:
                tuning_log = [{"round": 0, "reasoning": "Tuning loop could not start: " + fail_text,
                               "report": "", "met_target": False, "tuning": {}, "changed": {},
                               "success": False, "tracking_pct_headline": None}]
            result = dict(result)
            result["messages"] = list(result["messages"])
            result["messages"][-1] = _SyntheticMessage(
                fail_text + _render_clarification_section(clarification_record))
            if final_metrics is not None:
                final_metrics["mse_target_from_tuner"] = enable_tuning
            result["final_metrics"] = final_metrics

    usage["tuner"] = tuning_usage
    usage["total"] = _sum_usage(usage["total"], tuning_usage)
    usage["timeline"] = usage.get("timeline", []) + tuning_timeline
    usage["reporter"] = reporter_usage
    usage["total"] = _sum_usage(usage["total"], reporter_usage)
    if clarifier_usage:
        usage["clarifier"] = clarifier_usage
        usage["total"] = _sum_usage(usage["total"], clarifier_usage)
        usage["timeline"] = [{
            "actor": "clarify", "round": 0, "label": "Clarification",
            "tokens": clarifier_usage.get("total_tokens", 0),
            "input_tokens": clarifier_usage.get("input_tokens", 0),
            "output_tokens": clarifier_usage.get("output_tokens", 0),
            "detail": "%d question(s)" % len(clarification_record or []),
        }] + usage["timeline"]
    else:
        usage["clarifier"] = _empty_usage()

    usage["models"] = resolved_models()
    cost_report = model_pricing.format_run_cost_report(usage)
    if cost_report:
        print(cost_report)
    return result, usage, tuning_log, tuning_best
