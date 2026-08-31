import numpy as np
import matplotlib.pyplot as plt

from backend_core.AgentAdaptive.tools.series_export import should_show_plots


def _finish_figure() -> None:
    """Interactive show when allowed. Leave figures open for Streamlit PDF capture."""
    if should_show_plots():
        plt.show(block=False)
        plt.pause(0.1)



def plot(t, y, ref, u):
    p = y.shape[1]
    m = u.shape[1]

    plt.figure(figsize=(9, 6))

    plt.subplot(2, 1, 1)
    for i in range(p):
        plt.plot(t, ref[:, i], "--", linewidth=3, color="black", label="reference %d" % (i + 1))
        plt.plot(t, y[:, i],          linewidth=1.5, label="output %d" % (i + 1))
    plt.title("desired input")
    plt.ylabel("outputs")
    plt.legend(fontsize=8)
    plt.grid(True)

    plt.subplot(2, 1, 2)
    for j in range(m):
        plt.plot(t, u[:, j], label="input %d" % (j + 1))
    plt.ylabel("control input")
    plt.xlabel("time (s)")
    plt.legend(fontsize=8)
    plt.grid(True)

    plt.tight_layout()
    _finish_figure()

def plot_states(t, x_states, state_syms):
    n = x_states.shape[1]

    plt.figure(figsize=(9, 6))
    for i in range(n):
        plt.plot(t, x_states[:, i], label=str(state_syms[i]))
    plt.title("all states")
    plt.xlabel("time (s)")
    plt.ylabel("states")
    plt.legend(fontsize=8)
    plt.grid(True)

    plt.tight_layout()
    _finish_figure()


def plot_tracking_compare(t, y_off, y_on, ref):
    p = ref.shape[1]

    plt.figure(figsize=(9, 6))
    for i in range(p):
        plt.plot(t, ref[:, i], "--", linewidth=3, color="black",
                 label="reference %d" % (i + 1))
        plt.plot(t, y_off[:, i], ":", linewidth=1.5,
                 label="output %d  (estimator OFF)" % (i + 1))
        plt.plot(t, y_on[:, i], "-", linewidth=1.5,
                 label="output %d  (estimator ON)" % (i + 1))
    plt.title("tracking: uncertainty estimator OFF vs ON (same injected uncertainty)")
    plt.xlabel("time (s)")
    plt.ylabel("outputs")
    plt.legend(fontsize=8)
    plt.grid(True)

    plt.tight_layout()
    _finish_figure()


def plot_combined_uncertainty(t, alog, show_true=True):
    # alog's shape depends on the controller: SMC's g_hat comes pre-summed,
    # backstepping keeps Delta_hat/D_hat separate. detect which one and sum if needed.
    if alog is None:
        return

    if "g_hat" in alog:
        combined_hat = alog["g_hat"]
        combined_true = alog.get("g_true")
    elif "Delta_hat" in alog and "D_hat" in alog:
        combined_hat = alog["Delta_hat"] + alog["D_hat"]
        combined_true = (alog["Delta_true"] + alog["D_true"]) \
            if ("Delta_true" in alog and "D_true" in alog) else None
    else:
        return

    n = combined_hat.shape[1]
    plt.figure(figsize=(9, 2.6 * n))
    for i in range(n):
        plt.subplot(n, 1, i + 1)
        if show_true and combined_true is not None:
            plt.plot(t, combined_true[:, i], "--", color="black",
                      label="Delta+d true (state %d)" % (i + 1))
        plt.plot(t, combined_hat[:, i], label="Delta+d hat (state %d)" % (i + 1))
        plt.ylabel("state %d" % (i + 1))
        plt.legend(fontsize=8)
        plt.grid(True)
    plt.xlabel("time (s)")
    plt.suptitle("combined uncertainty estimate:  Delta(x,u) + d(t),  per state")

    plt.tight_layout()
    _finish_figure()


def plot_uncertainty(t, alog, show_delta=True, show_disturbance=True):
    if not (show_delta or show_disturbance):
        return
    Delta_hat = alog["Delta_hat"]
    D_hat = alog["D_hat"]
    Delta_true = alog["Delta_true"]
    D_true = alog["D_true"]
    n = Delta_hat.shape[1]

    plt.figure(figsize=(9, 3 * n))
    for i in range(n):
        plt.subplot(n, 1, i + 1)
        if show_delta:
            plt.plot(t, Delta_true[:, i], "--", color="black",
                     label="Delta_%d true" % (i + 1))
            plt.plot(t, Delta_hat[:, i], label="Delta_%d hat (NN)" % (i + 1))
        if show_disturbance:
            plt.plot(t, D_true[:, i], "--", color="red",
                     label="d_%d true" % (i + 1))
            plt.plot(t, D_hat[:, i], label="D_%d hat (observer)" % (i + 1))
        plt.ylabel("component %d" % (i + 1))
        plt.legend(fontsize=8)
        plt.grid(True)
    plt.xlabel("time (s)")
    parts = []
    if show_delta:
        parts.append("Delta (NN)")
    if show_disturbance:
        parts.append("d (observer)")
    plt.suptitle("true vs estimated uncertainty:  " + "  and  ".join(parts))

    plt.tight_layout()
    _finish_figure()


def plot_lumped_uncertainty(t, alog):
    if "Xi_hat" not in alog:
        return
    Xi_hat = alog["Xi_hat"]
    Xi_true = alog["Xi_true"]
    m = Xi_hat.shape[1]

    plt.figure(figsize=(9, 3 * m))
    for i in range(m):
        plt.subplot(m, 1, i + 1)
        plt.plot(t, Xi_true[:, i], "--", color="black",
                 label="Xi_%d true  (= [J_s(Delta+d)]_%d)" % (i + 1, i + 1))
        plt.plot(t, Xi_hat[:, i], label="Xi_%d hat  (RBFNN)" % (i + 1))
        plt.ylabel("output %d" % (i + 1))
        plt.legend(fontsize=8)
        plt.grid(True)
    plt.xlabel("time (s)")
    plt.suptitle("generalized lumped uncertainty:  true Xi  vs  RBFNN estimate")

    plt.tight_layout()
    _finish_figure()


def plot_state_uncertainty_smc(t, alog):
    if "Delta_hat" not in alog:
        return
    delta_hat = alog["Delta_hat"]
    delta_true = alog["Delta_true"]
    n = delta_hat.shape[1]

    plt.figure(figsize=(9, 2.6 * n))
    for i in range(n):
        plt.subplot(n, 1, i + 1)
        plt.plot(t, delta_true[:, i], "--", color="black",
                 label="Delta_%d true" % (i + 1))
        plt.plot(t, delta_hat[:, i], label="Delta_%d hat (identifier)" % (i + 1))
        plt.ylabel("state %d" % (i + 1))
        plt.legend(fontsize=8)
        plt.grid(True)
    plt.xlabel("time (s)")
    plt.suptitle("per-channel Delta  (state-space identifier)")

    plt.tight_layout()
    _finish_figure()


def plot_dist_obs_compare(t, y_off, y_on, ref):
    p = ref.shape[1]
    plt.figure(figsize=(9, 3 * p))
    for i in range(p):
        plt.subplot(p, 1, i + 1)
        plt.plot(t, ref[:, i], "--", linewidth=3, color="black",
                 label="reference %d" % (i + 1))
        plt.plot(t, y_off[:, i], ":", linewidth=1.6,
                 label="output %d  (Disturbance Observer OFF)" % (i + 1))
        plt.plot(t, y_on[:, i], "-", linewidth=1.6,
                 label="output %d  (Disturbance Observer ON)" % (i + 1))
        plt.ylabel("output %d" % (i + 1))
        plt.legend(fontsize=8)
        plt.grid(True)
    plt.xlabel("time (s)")
    plt.suptitle("tracking: Disturbance Observer OFF vs ON  (identical injected d(t))")
    plt.tight_layout()
    _finish_figure()


def plot_dist_estimate(t, alog):
    if "D_hat" not in alog:
        return
    D_hat_arr = alog["D_hat"]
    D_true_arr = alog["D_true"]
    p = D_hat_arr.shape[1]
    # only bother plotting channels that actually had a disturbance injected
    active = [i for i in range(p) if np.any(np.abs(D_true_arr[:, i]) > 1e-9)]
    if not active:
        return
    plt.figure(figsize=(9, 2.6 * len(active)))
    for row, i in enumerate(active):
        plt.subplot(len(active), 1, row + 1)
        plt.plot(t, D_true_arr[:, i], "--", color="black",
                 label="[J_s d(t)]_%d  injected (matched projection)" % (i + 1))
        plt.plot(t, D_hat_arr[:, i], label="D_hat_%d (surface observer, eq 101')" % (i + 1))
        plt.ylabel("surface %d" % (i + 1))
        plt.legend(fontsize=8)
        plt.grid(True)
    plt.xlabel("time (s)")
    plt.suptitle("Disturbance Observer:  J_s d(t)  vs  estimate D_hat  (surface space)")
    plt.tight_layout()
    _finish_figure()


def plot_combined_uncertainty_smc(t, alog):
    # lifts D_hat into state space (pseudo-inverse, diagnostic only).
    # then it gets added to Delta_hat and compared against the true combined mismatch
    if "g_hat" not in alog:
        return
    g_hat_arr = alog["g_hat"]
    g_true_arr = alog["g_true"]
    n = g_hat_arr.shape[1]

    plt.figure(figsize=(9, 2.6 * n))
    for i in range(n):
        plt.subplot(n, 1, i + 1)
        plt.plot(t, g_true_arr[:, i], "--", color="black",
                 label="Delta_%d + d_%d(t)  true (combined)" % (i + 1, i + 1))
        plt.plot(t, g_hat_arr[:, i],
                 label="Delta_hat_%d + [Js^+ D_hat]_%d  (combined estimate)" % (i + 1, i + 1))
        plt.ylabel("state %d" % (i + 1))
        plt.legend(fontsize=8)
        plt.grid(True)
    plt.xlabel("time (s)")
    plt.suptitle("combined uncertainty estimate:  Delta_hat + Js^+ D_hat  vs  Delta_true + d(t)")

    plt.tight_layout()
    _finish_figure()


def plot_command_filter(t, alog):
    if "xc" not in alog:
        return
    xc = alog["xc"]
    xdf = alog["xdf"]
    n = xc.shape[1]

    plt.figure(figsize=(9, 6))
    for i in range(1, n):
        if np.all(np.isnan(xc[:, i])):
            continue
        plt.plot(t, xc[:, i], ":", label="x_%d,c  (raw command)" % (i + 1))
        plt.plot(t, xdf[:, i], "-", label="x_%d,d  (filtered)" % (i + 1))
    plt.title("command filter: raw virtual control vs filtered (lag compensated)")
    plt.xlabel("time (s)")
    plt.ylabel("virtual control")
    plt.legend(fontsize=8)
    plt.grid(True)

    plt.tight_layout()
    _finish_figure()


def plot_filtered_error_compare(t, err_off, err_on, out_index=0):
    plt.figure(figsize=(9, 4))
    plt.plot(t, err_off, ":", linewidth=1.5,
             label="|y-ref|  (use_filtered_error = OFF)")
    plt.plot(t, err_on, "-", linewidth=1.5,
             label="|y-ref|  (use_filtered_error = ON)")
    plt.title("filtered error  s = e + lambda_I*integral(e)  removes the steady-state offset")
    plt.xlabel("time (s)")
    plt.ylabel("tracking error |y%d - ref%d|" % (out_index + 1, out_index + 1))
    plt.legend(fontsize=8)
    plt.grid(True)

    plt.tight_layout()
    _finish_figure()
