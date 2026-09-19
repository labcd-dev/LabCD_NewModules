"""Mock Tuner and offline pipeline. Real agents live in the *_main.py files.

Only mock agent functions are called here. Numerical helpers are reused from
main so controller simulation, scoring, and proposal limits remain the same.
"""

import contextlib
import io
import math

from backend_core.AgentAdaptive.tools import tuning_objectives as tuning_objectives_mod
from backend_core.AgentAdaptive.tools import system_spec as system_spec_mod
from backend_core.AgentAdaptive.tools.scoring import format_metrics_report, _fmt_list
from backend_core.AgentAdaptive.tools.reporter import render_final_report, _render_clarification_section
from backend_core.AgentAdaptive.tools.progress import _emit, _remap_note_stage
from backend_core.AgentAdaptive.tools.series_export import extract_series
from .designer_agent import run_extraction
from .report_writer import write_abstract
from .agent_io import _empty_usage, _sum_usage, _sum_usage_from_messages, _SyntheticMessage
from .mock_random import (
    _MockAgent, MockSession, current_session, mock_session, resolved_models,
    MOCK_COST_USD_PER_TOKEN,
)
from .tuner_agent_main import (
    SMC_TUNING_DEFAULTS, BACKSTEPPING_TUNING_DEFAULTS,
    _apply_tuning_proposal, _coerce_tuning_payload, _rerun_design_directly,
    _target_met, _scope_warning, _format_tuning_ledger,
    _tuning_diff, _format_tuning_diff,
)


class MockTuner(_MockAgent):
    def payload(self):
        current = self.context["current_tuning"]
        allowed = self.context.get("allowed_params")
        candidates = (["c_gains"] if self.context["method"] == "backstepping"
                      else ["K", "Lam", "surface_lambda"])
        candidates = [key for key in candidates if current.get(key) is not None
                      and (allowed is None or key in allowed)]
        proposal = {
            "reasoning": "MOCK Tuner: seeded random gain adjustment for exercising the "
                         "tuning loop. Improvements are decided by real simulation metrics.",
            "diagnosed_symptom": "slow_transient",
        }
        for key in self.rng.sample(candidates, min(len(candidates), 2)):
            factor = self.rng.uniform(0.85, 1.20)
            value = current[key]
            if isinstance(value, (list, tuple)):
                proposal[key] = [round(float(v) * factor, 8) for v in value]
            elif math.isfinite(float(value)) and float(value) > 0:
                proposal[key] = round(float(value) * factor, 8)
        return proposal


def run_tuner_round(fixed_args, current_tuning, session, prompt, structure_cache=None,
                    allowed_params=None, round_num=None):
    mocks = current_session() or MockSession(42)
    response = MockTuner(mocks, "tuner", {
        "current_tuning": current_tuning, "method": fixed_args.get("method", "smc"),
        "allowed_params": allowed_params,
    }).invoke([])
    payload = mocks.messages[-1]["response"]
    reasoning, symptom, overrides = _coerce_tuning_payload(payload)
    report = _apply_tuning_proposal(
        reasoning, symptom, overrides, fixed_args, current_tuning, session,
        structure_cache=structure_cache, allowed_params=allowed_params,
    )
    if report.startswith("DESIGN FAILED") or report.startswith("NO CHANGE APPLIED"):
        return {"messages": [response, _SyntheticMessage(report)]}
    return {"messages": [response]}


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


def run_full_pipeline(description, enable_tuning=False, target_rms_frac=0.02,
                       max_tuning_rounds=4, on_event=None, should_stop=None,
                       clarification_record=None, sim_overrides=None,
                       clarifier_usage=None, tuning_objectives=None,
                       system_spec=None, *, mock_seed=42):
    """Run mock agents and real simulations with no LLM API calls."""
    with mock_session(seed=mock_seed) as session:
        result, usage, tuning_log, best = _run_full_pipeline(
            description=description, enable_tuning=enable_tuning,
            target_rms_frac=target_rms_frac, max_tuning_rounds=max_tuning_rounds,
            on_event=on_event, should_stop=should_stop,
            clarification_record=clarification_record, sim_overrides=sim_overrides,
            clarifier_usage=None, tuning_objectives=tuning_objectives,
            system_spec=system_spec,
        )
        result["agent_mode"] = "mock"
        result["mock_agent_messages"] = session.messages
        usage.update({
            "simulated": True, "mock_seed": mock_seed,
            "price_usd_per_token": MOCK_COST_USD_PER_TOKEN,
            "cost_usd": round(usage["total"]["total_tokens"] * MOCK_COST_USD_PER_TOKEN, 6),
            "actual_cost_usd": 0.0,
        })
        return result, usage, tuning_log, best


def _run_full_pipeline(description, enable_tuning=False, target_rms_frac=0.02,
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
    return result, usage, tuning_log, tuning_best
