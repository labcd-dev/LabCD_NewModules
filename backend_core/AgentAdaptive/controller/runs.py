import numpy as np
import matplotlib.pyplot as plt

from . import estimators, smc_design, backstepping_design, simulation, plotting
from backend_core.AgentAdaptive.content.stability_proofs import (
    smc_stability_proof as _smc_stability_proof,
    backstepping_stability_proof as _backstepping_stability_proof,
)

from .structure_build import (
    _parse_system, coerce_float_list, _print_system_echo,
    _build_delta_func, _build_dist_func,
    _format_u_with_reference_smc, _format_u_with_reference_backstepping,
)
from backend_core.AgentAdaptive.tools.scoring import compute_simulation_metrics
from backend_core.AgentAdaptive.tools.progress import _emit


def _run_smc(states, dynamics, inputs, outputs, x0, refs,
             has_delta, has_disturbance, delta_exprs, dist_exprs,
             surface_lambda, K, Lam, phi_layer,
             Gamma, kappa, kappa_s, k2, k3, k4, sigma_W, N, width,
             rbf_spread, rbf_normalize, dt, t_end, reasoning, on_event=None,
             for_tuning=False, structure_cache=None, skip_simulation=False,
             fail_tol=0.02):
    # skip_simulation derives the law only, no sim/plot. for_tuning skips plots
    # and extra comparison sims since only `metrics` matters. structure_cache reuses derivation across rounds when gains alone change.

    plt.close("all")

    state_syms, input_syms, output_syms, dyn_exprs, symbol_map = _parse_system(
        states, dynamics, inputs, outputs)
    x0_vals = coerce_float_list(x0, len(state_syms), "x0")
    system_section = _print_system_echo(states, dynamics, inputs, outputs, 1,
                                          has_delta, has_disturbance, reasoning,
                                          state_syms, dyn_exprs, delta_exprs, dist_exprs)

    sys_dict = {"states": state_syms, "inputs": input_syms,
                "outputs": output_syms, "f": dyn_exprs}

    true_delta_func = _build_delta_func(delta_exprs, symbol_map, state_syms, input_syms) \
        if has_delta else None
    true_dist_func = _build_dist_func(dist_exprs) if has_disturbance else None

    use_ue = has_delta or has_disturbance

    # has_delta/has_disturbance just says "estimate it". explicit_uncertainty
    # asks a different question: did we ALSO get a formula, real ground truth to compare against?
    explicit_uncertainty = bool(delta_exprs) or bool(dist_exprs)
    if use_ue:
        print("system declares:  has_delta=%s  has_disturbance=%s  explicit_uncertainty=%s"
              % (has_delta, has_disturbance, explicit_uncertainty))
    delta_u_dep = estimators.delta_depends_on_u(true_delta_func, len(state_syms), len(input_syms)) \
        if use_ue else False
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
            structure_cache=structure_cache)

    controller, ref_orders, adaptive, u_symbolic, yd_symbols = build()
    u_report = _format_u_with_reference_smc(u_symbolic, yd_symbols, refs)
    stability_proof = _smc_stability_proof(has_delta, has_disturbance)

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
            structure_cache=structure_cache)
        if not for_tuning:
            _emit(on_event, kind="note", stage="design", text="Simulation complete: generating plots...")
            plotting.plot(t, y, ref, u)
            plotting.plot_states(t, x_states, state_syms)
            plt.show()
        metrics = compute_simulation_metrics(t, y, ref, u, x_states, None, dt, fail_tol=fail_tol)
        components = {"intro": "SMC controller designed (no uncertainty estimation).",
                      "system": system_section, "control_law": u_report,
                      "stability": stability_proof, "rms_str": None}
        return components, metrics

    if not explicit_uncertainty:
        # there's no ground truth here, so the off/on comparison run and the
        # true-vs-estimated plots get skipped, they'd just compare against a fake zero
        t, y_on, ref, u_on, x_on, alog = simulation.simulate(
            sys_dict, controller, ref_orders, refs, x0_vals, dt=dt, t_end=t_end,
            adaptive=adaptive, true_delta_func=true_delta_func, true_dist_func=true_dist_func,
            structure_cache=structure_cache)
        if not for_tuning:
            _emit(on_event, kind="note", stage="design", text="Simulation complete: generating plots...")
            plotting.plot(t, y_on, ref, u_on)
            plotting.plot_states(t, x_on, state_syms)
            plotting.plot_combined_uncertainty(t, alog, show_true=False)
            plt.show()
        window = min(2000, y_on.shape[0])
        rms = np.sqrt(((y_on[-window:] - ref[-window:]) ** 2).mean(axis=0))
        rms_str = ", ".join("%.4f" % r for r in rms)
        metrics = compute_simulation_metrics(t, y_on, ref, u_on, x_on, alog, dt, fail_tol=fail_tol)
        components = {
            "intro": ("SMC controller designed with has_delta=%s, has_disturbance=%s "
                      "(qualitative only, since no explicit true expression was given, "
                      "estimation was applied but no with/without-estimator comparison "
                      "was run)." % (has_delta, has_disturbance)),
            "system": system_section, "control_law": u_report,
            "stability": stability_proof, "rms_str": rms_str,
        }
        return components, metrics

    # the "off" run and disturbance-observer comparison only feed comparison
    # plots/prints. metrics always come from "on", so skip both during tuning
    if not for_tuning:
        t, y_off, ref, u_off, x_off, _ = simulation.simulate(
            sys_dict, controller, ref_orders, refs, x0_vals, dt=dt, t_end=t_end,
            adaptive=None, true_delta_func=true_delta_func, true_dist_func=true_dist_func,
            structure_cache=structure_cache)
    t, y_on, ref, u_on, x_on, alog = simulation.simulate(
        sys_dict, controller, ref_orders, refs, x0_vals, dt=dt, t_end=t_end,
        adaptive=adaptive, true_delta_func=true_delta_func, true_dist_func=true_dist_func,
        structure_cache=structure_cache)

    if not for_tuning:
        _emit(on_event, kind="note", stage="design", text="Simulation complete: generating plots...")
        plotting.plot(t, y_on, ref, u_on)
        plotting.plot_states(t, x_on, state_syms)
        plotting.plot_tracking_compare(t, y_off, y_on, ref)
        plotting.plot_combined_uncertainty(t, alog, show_true=True)

    window = min(2000, y_on.shape[0])
    rms = np.sqrt(((y_on[-window:] - ref[-window:]) ** 2).mean(axis=0))
    rms_str = ", ".join("%.4f" % r for r in rms)
    metrics = compute_simulation_metrics(t, y_on, ref, u_on, x_on, alog, dt, fail_tol=fail_tol)

    if not for_tuning and has_disturbance and "D_hat" in alog:
        _, ro_dob, adaptive_dob_off = build(dist_obs_flag=False)[:3]
        t_dob, y_dob_off, ref_dob, u_dob_off, x_dob_off, _ = simulation.simulate(
            sys_dict, controller, ro_dob, refs, x0_vals, dt=dt, t_end=t_end,
            adaptive=adaptive_dob_off, true_delta_func=true_delta_func,
            true_dist_func=true_dist_func, structure_cache=structure_cache)
        plotting.plot_dist_obs_compare(t_dob, y_dob_off, y_on, ref)
        rms_off = np.sqrt(((y_dob_off[-window:] - ref_dob[-window:]) ** 2).mean(axis=0))
        print("Disturbance Observer OFF vs ON RMS:", rms_off, "vs", rms)

    if not for_tuning:
        plt.show()
    components = {
        "intro": "SMC controller designed with has_delta=%s, has_disturbance=%s." % (has_delta, has_disturbance),
        "system": system_section, "control_law": u_report,
        "stability": stability_proof, "rms_str": rms_str,
    }
    return components, metrics


def _run_backstepping(states, dynamics, inputs, outputs, x0, refs,
                       has_delta, has_disturbance, delta_exprs, dist_exprs,
                       c_gains, Gamma, kappa, k2, k3, k4, sigma_W, tau, N, width,
                       rbf_spread, rbf_normalize, use_filtered_error, lambda_I,
                       filtered_error_output_index, dt, t_end, reasoning, on_event=None,
                       for_tuning=False, structure_cache=None, skip_simulation=False,
                       fail_tol=0.02):
    # same skip_simulation/for_tuning/structure_cache deal as _run_smc above.
    # only difference: structure_cache keys on c_gains here, not surface_lambda

    plt.close("all")

    state_syms, input_syms, output_syms, dyn_exprs, symbol_map = _parse_system(
        states, dynamics, inputs, outputs)
    x0_vals = coerce_float_list(x0, len(state_syms), "x0")
    system_section = _print_system_echo(states, dynamics, inputs, outputs, 2,
                                          has_delta, has_disturbance, reasoning,
                                          state_syms, dyn_exprs, delta_exprs, dist_exprs)

    sys_dict = {"states": state_syms, "inputs": input_syms,
                "outputs": output_syms, "f": dyn_exprs}

    true_delta_func = _build_delta_func(delta_exprs, symbol_map, state_syms, input_syms) \
        if has_delta else None
    true_dist_func = _build_dist_func(dist_exprs) if has_disturbance else None

    use_ue = has_delta or has_disturbance

    explicit_uncertainty = bool(delta_exprs) or bool(dist_exprs)
    if use_ue:
        print("system declares:  has_delta=%s  has_disturbance=%s  explicit_uncertainty=%s"
              % (has_delta, has_disturbance, explicit_uncertainty))
    delta_u_dep = estimators.delta_depends_on_u(true_delta_func, len(state_syms), len(input_syms)) \
        if use_ue else False

    def build(filtered_error_flag):
        return backstepping_design.design_backstepping(
            sys_dict, use_ue,
            c_gains=c_gains, Gamma=Gamma, kappa=kappa, k2=k2, k3=k3, k4=k4,
            sigma_W=sigma_W, tau=tau, N=N, width=width,
            rbf_spread=rbf_spread, rbf_normalize=rbf_normalize,
            delta_u_dependent=delta_u_dep,
            use_filtered_error=filtered_error_flag, lambda_I=lambda_I,
            estimate_delta=has_delta, estimate_disturbance=has_disturbance,
            structure_cache=structure_cache)

    controller, ref_orders, adaptive, u_law, yd = build(use_filtered_error)
    u_report = _format_u_with_reference_backstepping(u_law, yd, refs[0])
    stability_proof = _backstepping_stability_proof(has_delta, has_disturbance)

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
            structure_cache=structure_cache)
        if not for_tuning:
            _emit(on_event, kind="note", stage="design", text="Simulation complete: generating plots...")
            plotting.plot(t, y, ref, u)
            plotting.plot_states(t, x_states, state_syms)
            plt.show()
        metrics = compute_simulation_metrics(t, y, ref, u, x_states, None, dt, fail_tol=fail_tol)
        components = {"intro": "Backstepping controller designed (no uncertainty estimation).",
                      "system": system_section, "control_law": u_report,
                      "stability": stability_proof, "rms_str": None}
        return components, metrics

    if not explicit_uncertainty:
        t, y_on, ref, u_on, x_on, alog = simulation.simulate(
            sys_dict, controller, ref_orders, refs, x0_vals, dt=dt, t_end=t_end,
            adaptive=adaptive, true_delta_func=true_delta_func, true_dist_func=true_dist_func,
            structure_cache=structure_cache)
        if not for_tuning:
            _emit(on_event, kind="note", stage="design", text="Simulation complete: generating plots...")
            plotting.plot(t, y_on, ref, u_on)
            plotting.plot_states(t, x_on, state_syms)
            plotting.plot_combined_uncertainty(t, alog, show_true=False)
            plotting.plot_command_filter(t, alog)
            plt.show()
        window = min(2000, y_on.shape[0])
        rms = np.sqrt(((y_on[-window:] - ref[-window:]) ** 2).mean(axis=0))
        rms_str = ", ".join("%.4f" % r for r in rms)
        metrics = compute_simulation_metrics(t, y_on, ref, u_on, x_on, alog, dt, fail_tol=fail_tol)
        components = {
            "intro": ("Backstepping controller designed with has_delta=%s, has_disturbance=%s "
                      "(qualitative only, since no explicit true expression was given, "
                      "estimation was applied but no with/without-estimator comparison "
                      "was run)." % (has_delta, has_disturbance)),
            "system": system_section, "control_law": u_report,
            "stability": stability_proof, "rms_str": rms_str,
        }
        return components, metrics

    if not for_tuning:
        t, y_off, ref, u_off, x_off, _ = simulation.simulate(
            sys_dict, controller, ref_orders, refs, x0_vals, dt=dt, t_end=t_end,
            adaptive=None, true_delta_func=true_delta_func, true_dist_func=true_dist_func,
            structure_cache=structure_cache)
    t, y_on, ref, u_on, x_on, alog = simulation.simulate(
        sys_dict, controller, ref_orders, refs, x0_vals, dt=dt, t_end=t_end,
        adaptive=adaptive, true_delta_func=true_delta_func, true_dist_func=true_dist_func,
        structure_cache=structure_cache)

    if not for_tuning:
        _emit(on_event, kind="note", stage="design", text="Simulation complete: generating plots...")
        plotting.plot(t, y_on, ref, u_on)
        plotting.plot_states(t, x_on, state_syms)
        plotting.plot_tracking_compare(t, y_off, y_on, ref)
        plotting.plot_combined_uncertainty(t, alog, show_true=True)
        plotting.plot_command_filter(t, alog)

    window = min(2000, y_on.shape[0])
    rms = np.sqrt(((y_on[-window:] - ref[-window:]) ** 2).mean(axis=0))
    rms_str = ", ".join("%.4f" % r for r in rms)
    metrics = compute_simulation_metrics(t, y_on, ref, u_on, x_on, alog, dt, fail_tol=fail_tol)

    if not for_tuning and use_filtered_error:
        _, _, adaptive_filt_off = build(False)[:3]
        t_f, y_filt_off, ref_f, u_filt_off, x_filt_off, _ = simulation.simulate(
            sys_dict, controller, ref_orders, refs, x0_vals, dt=dt, t_end=t_end,
            adaptive=adaptive_filt_off, true_delta_func=true_delta_func,
            true_dist_func=true_dist_func, structure_cache=structure_cache)
        out_index = filtered_error_output_index
        err_off = np.abs(y_filt_off[:, out_index] - ref_f[:, out_index])
        err_on = np.abs(y_on[:, out_index] - ref[:, out_index])
        plotting.plot_filtered_error_compare(t_f, err_off, err_on, out_index)

    if not for_tuning:
        plt.show()
    components = {
        "intro": "Backstepping controller designed with has_delta=%s, has_disturbance=%s." % (has_delta, has_disturbance),
        "system": system_section, "control_law": u_report,
        "stability": stability_proof, "rms_str": rms_str,
    }
    return components, metrics
