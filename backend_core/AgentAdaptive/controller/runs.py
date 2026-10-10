import contextlib
import io

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from . import estimators, smc_design, backstepping_design, simulation, plotting
from . import output_bounds as ob
from backend_core.AgentAdaptive.content.stability_proofs import (
    smc_stability_proof as _smc_stability_proof,
    backstepping_stability_proof as _backstepping_stability_proof,
    input_limit_proof as _input_limit_proof,
    output_bound_proof as _output_bound_proof,
)

from .structure_build import (
    _parse_system, coerce_float_list, _print_system_echo,
    _build_delta_func, _build_dist_func,
    _format_u_with_reference_smc, _format_u_with_reference_backstepping,
)
from backend_core.AgentAdaptive.tools.scoring import compute_simulation_metrics, bound_arrays
from backend_core.AgentAdaptive.tools.progress import _emit
from backend_core.AgentAdaptive.tools.series_export import (
    attach_series, maybe_show_plots, should_create_plots,
)

DIAGNOSTICS_CONTEXT_KEY = "_diagnostics_context"


def _sat_kwargs(extras):
    return {"u_cmd": extras.get("u_cmd"), "limits": extras.get("input_limits"),
            "bounds": extras.get("output_bounds")}


def _barrier_widths(output_bounds, refs, outputs, t_end, state_syms, output_syms, x0_vals):
    # kb per output (None = no output bound). raises OutputBoundError if the barrier can't start
    if not output_bounds or not any(b is not None for b in output_bounds):
        return None
    y0 = [float(sp.lambdify(list(state_syms), h, "numpy")(*x0_vals)) for h in output_syms]
    return ob.run_half_widths(output_bounds, [r["expr"] for r in refs], list(outputs), t_end, y0)


def simplified_law_report(job):
    # the control law with the reference substituted AND trigonometrically simplified, built on request
    # (the simplification takes seconds, so the run itself never does it). job comes from the run
    state_syms, input_syms, output_syms, dyn_exprs, _ = _parse_system(
        job["states"], job["dynamics"], job["inputs"], job["outputs"])
    sys_dict = {"states": state_syms, "inputs": input_syms, "outputs": output_syms, "f": dyn_exprs}
    with contextlib.redirect_stdout(io.StringIO()):
        if job["method"] == "smc":
            _, _, _, u_symbolic, yd_symbols = smc_design.design_smc(
                sys_dict, False, surface_lambda=job["surface_lambda"], K=job["K"], Lam=job["Lam"],
                phi_layer=job["phi_layer"])
            return _format_u_with_reference_smc(u_symbolic, yd_symbols, job["refs"], simplify=True)
        _, _, _, u_law, yd = backstepping_design.design_backstepping(
            sys_dict, False, c_gains=job["c_gains"])
        return _format_u_with_reference_backstepping(u_law, yd, job["refs"][0], simplify=True)


def _uncertainty_funcs(structure_cache, states, inputs, delta_exprs, dist_exprs, has_delta,
                       has_disturbance, symbol_map, state_syms, input_syms):
    # the true uncertainty and disturbance functions and the "does Delta depend on u" probe depend only on
    # the plant, so tuning rounds (same structure_cache) build them once
    key = ("uncertainty", tuple(states), tuple(inputs), tuple(delta_exprs or ()), tuple(dist_exprs or ()),
           bool(has_delta), bool(has_disturbance))
    hit = structure_cache.get(key) if structure_cache is not None else None
    if hit is None:
        delta = _build_delta_func(delta_exprs, symbol_map, state_syms, input_syms) if has_delta else None
        dist = _build_dist_func(dist_exprs) if has_disturbance else None
        dep = (estimators.delta_depends_on_u(delta, len(state_syms), len(input_syms))
               if (has_delta or has_disturbance) else False)
        hit = (delta, dist, dep)
        if structure_cache is not None:
            structure_cache[key] = hit
    return hit


def _bounds_context(metrics, output_bounds, kb):
    # what the diagnoser needs about the output bounds. empty when none was given
    if kb is None:
        return {}
    return {"output_bounds": output_bounds, "output_bound_stats": (metrics or {}).get("output_bounds"),
            "barrier_kb": [None if not np.isfinite(v) else float(v) for v in kb]}


def _run_context(metrics, input_limits, output_bounds, kb, extras, free_run=None):
    return {**_limits_context(metrics, input_limits, extras, free_run),
            **_bounds_context(metrics, output_bounds, kb)}


def _limits_context(metrics, input_limits, extras, free_run=None):
    # what the diagnoser needs to talk about clipped inputs. empty when no limit was given
    if extras.get("input_limits") is None:
        return {}
    return {"input_limits": input_limits, "input_saturation": (metrics or {}).get("input_saturation"),
            "u_cmd": extras.get("u_cmd"), "free_run": free_run}


def facts_from_metrics(m, inputs):
    # the same facts as _free_run_facts, read off the metrics of a run that was already made
    # (no limit, no bound). None when that run has no control peak to read (it diverged)
    peak = (m or {}).get("control_max")
    if peak is None or "tracking_pct_headline" not in m:
        return None
    return {"passes": bool(m["success"]), "tracking_pct": float(m["tracking_pct_headline"]),
            "failed_check": (m.get("success_reason") or "").split(":")[0] or None,
            "peak_by_input": {name: float(p) for name, p in zip(inputs, peak)}}


def _free_run_facts(simulate_free, inputs, dt, fail_tol, bounds=None):
    # same design with the limits removed: the one fact that says whether the limit itself broke
    # the run. a failed counterfactual is just "unknown", never an error for the real run
    try:
        t, y, ref, u, x, _ = simulate_free()
        m = compute_simulation_metrics(t, y, ref, u, x, None, dt, fail_tol=fail_tol, bounds=bounds)
        peak = np.nan_to_num(np.abs(np.asarray(u, dtype=float).reshape(len(t), -1)), nan=np.inf).max(axis=0)
        return {"passes": bool(m["success"]), "tracking_pct": float(m["tracking_pct_headline"]),
                "failed_check": (m.get("success_reason") or "").split(":")[0] or None,
                "peak_by_input": {name: float(p) for name, p in zip(inputs, peak)}}
    except Exception:  # noqa: BLE001
        return None


def _attach_diagnostics_context(components, method, system_section, u_report,
                                states, inputs, outputs, refs, x0_vals, tuning,
                                dt, t_end, t, y, ref, u, x_states, alog, for_tuning, extra=None):
    # just stashing raw stuff here, not building evidence yet -- that only
    # happens if the run actually fails. skip during tuning rounds, same as attach_series
    if for_tuning:
        return components
    n = min(len(outputs), len(refs))
    paired_refs = [{"output": outputs[i], "expr": (refs[i].get("expr") or "")} for i in range(n)]
    components[DIAGNOSTICS_CONTEXT_KEY] = {
        "method": method,
        "system_text": system_section,
        "control_law_text": u_report,
        "states": list(states), "inputs": list(inputs), "outputs": list(outputs),
        "refs": paired_refs, "x0": list(x0_vals), "tuning": dict(tuning),
        "dt": dt, "t_end": t_end,
        "t": t, "y": y, "ref": ref, "u": u, "x": x_states, "alog": alog,
    }
    components[DIAGNOSTICS_CONTEXT_KEY].update(extra or {})
    return components


def _run_smc(states, dynamics, inputs, outputs, x0, refs,
             has_delta, has_disturbance, delta_exprs, dist_exprs,
             surface_lambda, K, Lam, phi_layer,
             Gamma, kappa, kappa_s, k2, k3, k4, sigma_W, N, width,
             rbf_spread, rbf_normalize, dt, t_end, reasoning, on_event=None,
             for_tuning=False, structure_cache=None, skip_simulation=False,
             fail_tol=0.02, input_limits=None, output_bounds=None, free_run_known=None):
    # skip_simulation derives the law only, no sim/plot. for_tuning skips plots
    # and extra comparison sims since only `metrics` matters. structure_cache reuses derivation across rounds when gains alone change.

    plt.close("all")
    extras = {}

    state_syms, input_syms, output_syms, dyn_exprs, symbol_map = _parse_system(
        states, dynamics, inputs, outputs)
    x0_vals = coerce_float_list(x0, len(state_syms), "x0")
    system_section = _print_system_echo(states, dynamics, inputs, outputs, 1,
                                          has_delta, has_disturbance, reasoning,
                                          state_syms, dyn_exprs, delta_exprs, dist_exprs)

    sys_dict = {"states": state_syms, "inputs": input_syms,
                "outputs": output_syms, "f": dyn_exprs}

    kb = _barrier_widths(output_bounds, refs, outputs, t_end, state_syms, output_syms, x0_vals)
    extras["output_bounds"] = bound_arrays(output_bounds, len(outputs)) if kb is not None else None

    true_delta_func, true_dist_func, delta_u_dep = _uncertainty_funcs(
        structure_cache, states, inputs, delta_exprs, dist_exprs, has_delta, has_disturbance,
        symbol_map, state_syms, input_syms)

    use_ue = has_delta or has_disturbance

    # has_delta/has_disturbance just says "estimate it". explicit_uncertainty
    # asks a different question: did we ALSO get a formula, real ground truth to compare against?
    explicit_uncertainty = bool(delta_exprs) or bool(dist_exprs)
    if use_ue:
        print("system declares:  has_delta=%s  has_disturbance=%s  explicit_uncertainty=%s"
              % (has_delta, has_disturbance, explicit_uncertainty))
    if use_ue:
        print("Delta u-dependence probe: %s" % delta_u_dep)

    def build(dist_obs_flag=None):
        dob_on = has_disturbance if dist_obs_flag is None else dist_obs_flag
        return smc_design.design_smc(
            sys_dict, use_ue,
            surface_lambda=surface_lambda, K=K, Lam=Lam, phi_layer=phi_layer,
            Gamma=Gamma, kappa=kappa, kappa_s=kappa_s, k2=k2, k3=k3, k4=k4,
            sigma_W=sigma_W, N=N, width=width,
            rbf_spread=rbf_spread, rbf_normalize=rbf_normalize,
            delta_u_dependent=delta_u_dep,
            estimate_delta=has_delta, estimate_disturbance=dob_on,
            structure_cache=structure_cache, output_bounds_kb=kb)

    controller, ref_orders, adaptive, u_symbolic, yd_symbols = build()

    def free_facts(metrics):
        # only for a failed, limited, final run (tuning rounds skip it)
        if for_tuning or extras.get("input_limits") is None or metrics.get("success", True):
            return None
        if free_run_known is not None:
            return free_run_known
        ctrl, ro, ad = build()[:3]
        return _free_run_facts(lambda: simulation.simulate(
            sys_dict, ctrl, ro, refs, x0_vals, dt=dt, t_end=t_end, adaptive=ad,
            true_delta_func=true_delta_func, true_dist_func=true_dist_func,
            structure_cache=structure_cache), inputs, dt, fail_tol, extras.get("output_bounds"))
    # the displayed law is only read in the final report, never during a tuning round
    u_report = "" if for_tuning else _format_u_with_reference_smc(u_symbolic, yd_symbols, refs)
    stability_proof = _smc_stability_proof(has_delta, has_disturbance)
    if simulation.limit_arrays(input_limits, len(input_syms)) is not None:
        stability_proof += "\n\n" + _input_limit_proof("smc")
    if kb is not None:
        stability_proof += "\n\n" + _output_bound_proof("smc")
    tuning_snapshot = dict(
        surface_lambda=surface_lambda, K=K, Lam=Lam, phi_layer=phi_layer,
        Gamma=Gamma, kappa=kappa, kappa_s=kappa_s, k2=k2, k3=k3, k4=k4,
        sigma_W=sigma_W, N=N, width=width, rbf_spread=rbf_spread, rbf_normalize=rbf_normalize)

    if skip_simulation:
        components = {"intro": "(structure validated, numeric simulation deferred)",
                      "system": system_section, "control_law": u_report,
                      "stability": stability_proof, "rms_str": None}
        return components, None

    _emit(on_event, kind="note", stage="design", text="Control law derived: running the simulation...")

    if not use_ue:
        t, y, ref, u, x_states, _ = simulation.simulate(
            sys_dict, controller, ref_orders, refs, x0_vals, dt=dt, t_end=t_end,
            true_delta_func=true_delta_func, true_dist_func=true_dist_func,
            structure_cache=structure_cache, input_limits=input_limits, extras=extras)
        if not for_tuning and should_create_plots():
            _emit(on_event, kind="note", stage="design", text="Simulation complete: generating plots...")
            plotting.plot(t, y, ref, u)
            plotting.plot_states(t, x_states, state_syms)
            maybe_show_plots()
        metrics = compute_simulation_metrics(t, y, ref, u, x_states, None, dt, fail_tol=fail_tol, **_sat_kwargs(extras))
        components = {"intro": "SMC controller designed (no uncertainty estimation).",
                      "system": system_section, "control_law": u_report,
                      "stability": stability_proof, "rms_str": None}
        attach_series(components, t, y, ref, u, x_states, dt=dt, t_end=t_end,
                      outputs=outputs, inputs=inputs, states=states,
                      include=not for_tuning)
        _attach_diagnostics_context(components, "smc", system_section, u_report,
                                    states, inputs, outputs, refs, x0_vals, tuning_snapshot,
                                    dt, t_end, t, y, ref, u, x_states, None, for_tuning,
                                    extra=_run_context(metrics, input_limits, output_bounds, kb, extras,
                                                   free_facts(metrics)))
        return components, metrics

    if not explicit_uncertainty:
        # there's no ground truth here, so the off/on comparison run and the
        # true-vs-estimated plots get skipped, they'd just compare against a fake zero
        t, y_on, ref, u_on, x_on, alog = simulation.simulate(
            sys_dict, controller, ref_orders, refs, x0_vals, dt=dt, t_end=t_end,
            adaptive=adaptive, true_delta_func=true_delta_func, true_dist_func=true_dist_func,
            structure_cache=structure_cache, input_limits=input_limits, extras=extras)
        if not for_tuning and should_create_plots():
            _emit(on_event, kind="note", stage="design", text="Simulation complete: generating plots...")
            plotting.plot(t, y_on, ref, u_on)
            plotting.plot_states(t, x_on, state_syms)
            plotting.plot_combined_uncertainty(t, alog, show_true=False)
            maybe_show_plots()
        window = min(2000, y_on.shape[0])
        rms = np.sqrt(((y_on[-window:] - ref[-window:]) ** 2).mean(axis=0))
        rms_str = ", ".join("%.4f" % r for r in rms)
        metrics = compute_simulation_metrics(t, y_on, ref, u_on, x_on, alog, dt, fail_tol=fail_tol, **_sat_kwargs(extras))
        components = {
            "intro": ("SMC controller designed with has_delta=%s, has_disturbance=%s "
                      "(qualitative only, since no explicit true expression was given, "
                      "estimation was applied but no with/without-estimator comparison "
                      "was run)." % (has_delta, has_disturbance)),
            "system": system_section, "control_law": u_report,
            "stability": stability_proof, "rms_str": rms_str,
        }
        attach_series(components, t, y_on, ref, u_on, x_on, dt=dt, t_end=t_end,
                      outputs=outputs, inputs=inputs, states=states,
                      include=not for_tuning)
        _attach_diagnostics_context(components, "smc", system_section, u_report,
                                    states, inputs, outputs, refs, x0_vals, tuning_snapshot,
                                    dt, t_end, t, y_on, ref, u_on, x_on, alog, for_tuning,
                                    extra=_run_context(metrics, input_limits, output_bounds, kb, extras,
                                                   free_facts(metrics)))
        return components, metrics

    # the "off" run and disturbance-observer comparison only feed comparison
    # plots/prints. metrics always come from "on", so skip both during tuning
    if not for_tuning and should_create_plots():
        t, y_off, ref, u_off, x_off, _ = simulation.simulate(
            sys_dict, controller, ref_orders, refs, x0_vals, dt=dt, t_end=t_end,
            adaptive=None, true_delta_func=true_delta_func, true_dist_func=true_dist_func,
            structure_cache=structure_cache, input_limits=input_limits)
    t, y_on, ref, u_on, x_on, alog = simulation.simulate(
        sys_dict, controller, ref_orders, refs, x0_vals, dt=dt, t_end=t_end,
        adaptive=adaptive, true_delta_func=true_delta_func, true_dist_func=true_dist_func,
        structure_cache=structure_cache, input_limits=input_limits, extras=extras)

    if not for_tuning and should_create_plots():
        _emit(on_event, kind="note", stage="design", text="Simulation complete: generating plots...")
        plotting.plot(t, y_on, ref, u_on)
        plotting.plot_states(t, x_on, state_syms)
        plotting.plot_tracking_compare(t, y_off, y_on, ref)
        plotting.plot_combined_uncertainty(t, alog, show_true=True)

    window = min(2000, y_on.shape[0])
    rms = np.sqrt(((y_on[-window:] - ref[-window:]) ** 2).mean(axis=0))
    rms_str = ", ".join("%.4f" % r for r in rms)
    metrics = compute_simulation_metrics(t, y_on, ref, u_on, x_on, alog, dt, fail_tol=fail_tol, **_sat_kwargs(extras))

    if not for_tuning and has_disturbance and "D_hat" in alog and should_create_plots():
        _, ro_dob, adaptive_dob_off = build(dist_obs_flag=False)[:3]
        t_dob, y_dob_off, ref_dob, u_dob_off, x_dob_off, _ = simulation.simulate(
            sys_dict, controller, ro_dob, refs, x0_vals, dt=dt, t_end=t_end,
            adaptive=adaptive_dob_off, true_delta_func=true_delta_func,
            true_dist_func=true_dist_func, structure_cache=structure_cache, input_limits=input_limits)
        plotting.plot_dist_obs_compare(t_dob, y_dob_off, y_on, ref)
        rms_off = np.sqrt(((y_dob_off[-window:] - ref_dob[-window:]) ** 2).mean(axis=0))
        print("Disturbance Observer OFF vs ON RMS:", rms_off, "vs", rms)

    if not for_tuning:
        maybe_show_plots()
    components = {
        "intro": "SMC controller designed with has_delta=%s, has_disturbance=%s." % (has_delta, has_disturbance),
        "system": system_section, "control_law": u_report,
        "stability": stability_proof, "rms_str": rms_str,
    }
    attach_series(components, t, y_on, ref, u_on, x_on, dt=dt, t_end=t_end,
                  outputs=outputs, inputs=inputs, states=states,
                  include=not for_tuning)
    _attach_diagnostics_context(components, "smc", system_section, u_report,
                                states, inputs, outputs, refs, x0_vals, tuning_snapshot,
                                dt, t_end, t, y_on, ref, u_on, x_on, alog, for_tuning,
                                extra=_run_context(metrics, input_limits, output_bounds, kb, extras,
                                                   free_facts(metrics)))
    return components, metrics


def _run_backstepping(states, dynamics, inputs, outputs, x0, refs,
                       has_delta, has_disturbance, delta_exprs, dist_exprs,
                       c_gains, Gamma, kappa, k2, k3, k4, sigma_W, tau, N, width,
                       rbf_spread, rbf_normalize, use_filtered_error, lambda_I,
                       filtered_error_output_index, dt, t_end, reasoning, on_event=None,
                       for_tuning=False, structure_cache=None, skip_simulation=False,
                       fail_tol=0.02, input_limits=None, output_bounds=None, free_run_known=None):
    # same skip_simulation/for_tuning/structure_cache deal as _run_smc above.
    # only difference: structure_cache keys on c_gains here, not surface_lambda

    plt.close("all")
    extras = {}

    state_syms, input_syms, output_syms, dyn_exprs, symbol_map = _parse_system(
        states, dynamics, inputs, outputs)
    x0_vals = coerce_float_list(x0, len(state_syms), "x0")
    system_section = _print_system_echo(states, dynamics, inputs, outputs, 2,
                                          has_delta, has_disturbance, reasoning,
                                          state_syms, dyn_exprs, delta_exprs, dist_exprs)

    sys_dict = {"states": state_syms, "inputs": input_syms,
                "outputs": output_syms, "f": dyn_exprs}

    kb = _barrier_widths(output_bounds, refs, outputs, t_end, state_syms, output_syms, x0_vals)
    extras["output_bounds"] = bound_arrays(output_bounds, len(outputs)) if kb is not None else None

    true_delta_func, true_dist_func, delta_u_dep = _uncertainty_funcs(
        structure_cache, states, inputs, delta_exprs, dist_exprs, has_delta, has_disturbance,
        symbol_map, state_syms, input_syms)

    use_ue = has_delta or has_disturbance

    explicit_uncertainty = bool(delta_exprs) or bool(dist_exprs)
    if use_ue:
        print("system declares:  has_delta=%s  has_disturbance=%s  explicit_uncertainty=%s"
              % (has_delta, has_disturbance, explicit_uncertainty))

    def build(filtered_error_flag):
        return backstepping_design.design_backstepping(
            sys_dict, use_ue,
            c_gains=c_gains, Gamma=Gamma, kappa=kappa, k2=k2, k3=k3, k4=k4,
            sigma_W=sigma_W, tau=tau, N=N, width=width,
            rbf_spread=rbf_spread, rbf_normalize=rbf_normalize,
            delta_u_dependent=delta_u_dep,
            use_filtered_error=filtered_error_flag, lambda_I=lambda_I,
            estimate_delta=has_delta, estimate_disturbance=has_disturbance,
            structure_cache=structure_cache,
            output_bound=(None if kb is None else float(kb[0])))

    controller, ref_orders, adaptive, u_law, yd = build(use_filtered_error)

    def free_facts(metrics):
        if for_tuning or extras.get("input_limits") is None or metrics.get("success", True):
            return None
        if free_run_known is not None:
            return free_run_known
        ctrl, ro, ad = build(use_filtered_error)[:3]
        return _free_run_facts(lambda: simulation.simulate(
            sys_dict, ctrl, ro, refs, x0_vals, dt=dt, t_end=t_end, adaptive=ad,
            true_delta_func=true_delta_func, true_dist_func=true_dist_func,
            structure_cache=structure_cache), inputs, dt, fail_tol, extras.get("output_bounds"))
    u_report = "" if for_tuning else _format_u_with_reference_backstepping(
        u_law, yd, refs[0], barrier_kb=None if kb is None else float(kb[0]))
    stability_proof = _backstepping_stability_proof(has_delta, has_disturbance)
    if simulation.limit_arrays(input_limits, len(input_syms)) is not None:
        stability_proof += "\n\n" + _input_limit_proof("backstepping")
    if kb is not None:
        stability_proof += "\n\n" + _output_bound_proof("backstepping")
    tuning_snapshot = dict(
        c_gains=c_gains, Gamma=Gamma, kappa=kappa, k2=k2, k3=k3, k4=k4, sigma_W=sigma_W,
        tau=tau, N=N, width=width, rbf_spread=rbf_spread, rbf_normalize=rbf_normalize,
        use_filtered_error=use_filtered_error, lambda_I=lambda_I)

    if skip_simulation:
        components = {"intro": "(structure validated, numeric simulation deferred)",
                      "system": system_section, "control_law": u_report,
                      "stability": stability_proof, "rms_str": None}
        return components, None

    _emit(on_event, kind="note", stage="design", text="Control law derived: running the simulation...")

    if not use_ue:
        t, y, ref, u, x_states, _ = simulation.simulate(
            sys_dict, controller, ref_orders, refs, x0_vals, dt=dt, t_end=t_end,
            true_delta_func=true_delta_func, true_dist_func=true_dist_func,
            structure_cache=structure_cache, input_limits=input_limits, extras=extras)
        if not for_tuning and should_create_plots():
            _emit(on_event, kind="note", stage="design", text="Simulation complete: generating plots...")
            plotting.plot(t, y, ref, u)
            plotting.plot_states(t, x_states, state_syms)
            maybe_show_plots()
        metrics = compute_simulation_metrics(t, y, ref, u, x_states, None, dt, fail_tol=fail_tol, **_sat_kwargs(extras))
        components = {"intro": "Backstepping controller designed (no uncertainty estimation).",
                      "system": system_section, "control_law": u_report,
                      "stability": stability_proof, "rms_str": None}
        attach_series(components, t, y, ref, u, x_states, dt=dt, t_end=t_end,
                      outputs=outputs, inputs=inputs, states=states,
                      include=not for_tuning)
        _attach_diagnostics_context(components, "backstepping", system_section, u_report,
                                    states, inputs, outputs, refs, x0_vals, tuning_snapshot,
                                    dt, t_end, t, y, ref, u, x_states, None, for_tuning,
                                    extra=_run_context(metrics, input_limits, output_bounds, kb, extras,
                                                   free_facts(metrics)))
        return components, metrics

    if not explicit_uncertainty:
        t, y_on, ref, u_on, x_on, alog = simulation.simulate(
            sys_dict, controller, ref_orders, refs, x0_vals, dt=dt, t_end=t_end,
            adaptive=adaptive, true_delta_func=true_delta_func, true_dist_func=true_dist_func,
            structure_cache=structure_cache, input_limits=input_limits, extras=extras)
        if not for_tuning and should_create_plots():
            _emit(on_event, kind="note", stage="design", text="Simulation complete: generating plots...")
            plotting.plot(t, y_on, ref, u_on)
            plotting.plot_states(t, x_on, state_syms)
            plotting.plot_combined_uncertainty(t, alog, show_true=False)
            plotting.plot_command_filter(t, alog)
            maybe_show_plots()
        window = min(2000, y_on.shape[0])
        rms = np.sqrt(((y_on[-window:] - ref[-window:]) ** 2).mean(axis=0))
        rms_str = ", ".join("%.4f" % r for r in rms)
        metrics = compute_simulation_metrics(t, y_on, ref, u_on, x_on, alog, dt, fail_tol=fail_tol, **_sat_kwargs(extras))
        components = {
            "intro": ("Backstepping controller designed with has_delta=%s, has_disturbance=%s "
                      "(qualitative only, since no explicit true expression was given, "
                      "estimation was applied but no with/without-estimator comparison "
                      "was run)." % (has_delta, has_disturbance)),
            "system": system_section, "control_law": u_report,
            "stability": stability_proof, "rms_str": rms_str,
        }
        attach_series(components, t, y_on, ref, u_on, x_on, dt=dt, t_end=t_end,
                      outputs=outputs, inputs=inputs, states=states,
                      include=not for_tuning)
        _attach_diagnostics_context(components, "backstepping", system_section, u_report,
                                    states, inputs, outputs, refs, x0_vals, tuning_snapshot,
                                    dt, t_end, t, y_on, ref, u_on, x_on, alog, for_tuning,
                                    extra=_run_context(metrics, input_limits, output_bounds, kb, extras,
                                                   free_facts(metrics)))
        return components, metrics

    if not for_tuning and should_create_plots():
        t, y_off, ref, u_off, x_off, _ = simulation.simulate(
            sys_dict, controller, ref_orders, refs, x0_vals, dt=dt, t_end=t_end,
            adaptive=None, true_delta_func=true_delta_func, true_dist_func=true_dist_func,
            structure_cache=structure_cache, input_limits=input_limits)
    t, y_on, ref, u_on, x_on, alog = simulation.simulate(
        sys_dict, controller, ref_orders, refs, x0_vals, dt=dt, t_end=t_end,
        adaptive=adaptive, true_delta_func=true_delta_func, true_dist_func=true_dist_func,
        structure_cache=structure_cache, input_limits=input_limits, extras=extras)

    if not for_tuning and should_create_plots():
        _emit(on_event, kind="note", stage="design", text="Simulation complete: generating plots...")
        plotting.plot(t, y_on, ref, u_on)
        plotting.plot_states(t, x_on, state_syms)
        plotting.plot_tracking_compare(t, y_off, y_on, ref)
        plotting.plot_combined_uncertainty(t, alog, show_true=True)
        plotting.plot_command_filter(t, alog)

    window = min(2000, y_on.shape[0])
    rms = np.sqrt(((y_on[-window:] - ref[-window:]) ** 2).mean(axis=0))
    rms_str = ", ".join("%.4f" % r for r in rms)
    metrics = compute_simulation_metrics(t, y_on, ref, u_on, x_on, alog, dt, fail_tol=fail_tol, **_sat_kwargs(extras))

    if not for_tuning and use_filtered_error and should_create_plots():
        _, _, adaptive_filt_off = build(False)[:3]
        t_f, y_filt_off, ref_f, u_filt_off, x_filt_off, _ = simulation.simulate(
            sys_dict, controller, ref_orders, refs, x0_vals, dt=dt, t_end=t_end,
            adaptive=adaptive_filt_off, true_delta_func=true_delta_func,
            true_dist_func=true_dist_func, structure_cache=structure_cache, input_limits=input_limits)
        out_index = filtered_error_output_index
        err_off = np.abs(y_filt_off[:, out_index] - ref_f[:, out_index])
        err_on = np.abs(y_on[:, out_index] - ref[:, out_index])
        plotting.plot_filtered_error_compare(t_f, err_off, err_on, out_index)

    if not for_tuning:
        maybe_show_plots()
    components = {
        "intro": "Backstepping controller designed with has_delta=%s, has_disturbance=%s." % (has_delta, has_disturbance),
        "system": system_section, "control_law": u_report,
        "stability": stability_proof, "rms_str": rms_str,
    }
    attach_series(components, t, y_on, ref, u_on, x_on, dt=dt, t_end=t_end,
                  outputs=outputs, inputs=inputs, states=states,
                  include=not for_tuning)
    _attach_diagnostics_context(components, "backstepping", system_section, u_report,
                                states, inputs, outputs, refs, x0_vals, tuning_snapshot,
                                dt, t_end, t, y_on, ref, u_on, x_on, alog, for_tuning,
                                extra=_run_context(metrics, input_limits, output_bounds, kb, extras,
                                                   free_facts(metrics)))
    return components, metrics
