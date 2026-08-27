import inspect
import numpy as np
import sympy as sp


def eval_uncertainty(func, x, u, t, n):
    # arg count tells us the shape: d(t), Delta(x,u), or Delta(x,u,t).
    # that way callers can pass whatever form makes sense for them
    if func is None:
        return np.zeros(n)
    try:
        n_args = len(inspect.signature(func).parameters)
    except (ValueError, TypeError):
        n_args = 3
    if n_args == 1:
        value = func(t)
    elif n_args == 2:
        value = func(x, u)
    else:
        value = func(x, u, t)
    return np.asarray(value, dtype=float).reshape(n)


def delta_depends_on_u(func, n, m, n_probes=8, seed=0, tol=1e-7, h=1e-6):
    # finite-differences w.r.t. u at random points to see if uncertainty
    # reacts to control; if it does, the estimator needs u (frozen one step behind).
    if func is None:
        return False

    try:
        if len(inspect.signature(func).parameters) == 1:
            return False
    except (ValueError, TypeError):
        pass

    rng = np.random.default_rng(seed)
    for _ in range(n_probes):
        x = rng.uniform(-2.0, 2.0, size=n)
        t = rng.uniform(0.0, 10.0)
        u0 = rng.uniform(-2.0, 2.0, size=m)
        try:
            for j in range(m):
                u_plus = u0.copy(); u_plus[j] += h
                u_minus = u0.copy(); u_minus[j] -= h
                dDelta_duj = (eval_uncertainty(func, x, u_plus, t, n)
                              - eval_uncertainty(func, x, u_minus, t, n)) / (2.0 * h)

                if np.max(np.abs(dDelta_duj)) > tol:
                    return True

        except Exception:
            return True
    return False


def make_rbf(dim,
             N,
             width,
             seed=0,
             spread=1.0,
             normalize="adaptive",
             eps=1e-6,
             min_scale=1.0):

    rng = np.random.default_rng(seed)

    centers = rng.uniform(-spread, spread, size=(N, dim))

    scale = np.ones(dim) * min_scale

    def _normalize(z):
        nonlocal scale

        if normalize is None:
            return z

        # just an online running max-abs scale, not real mean/std or tanh.
        # any non-None value here gets this same behavior anyway
        scale = np.maximum(scale, np.abs(z))

        return z / (scale + eps)

    def mu(x_vec, u_vec=()):

        xv = np.asarray(x_vec, dtype=float).reshape(-1)
        uv = np.asarray(u_vec, dtype=float).reshape(-1)

        z = np.concatenate([xv, uv]).reshape(dim)

        z = _normalize(z)

        d2 = np.sum((centers - z)**2, axis=1)

        return np.exp(-d2 / (2.0 * width**2))

    return mu


def round_floats(expr, n=4):
    # display only. sp.Float(x, n)'s n means sig figs not decimal places,
    # so round via a formatted string instead to get real decimal rounding.
    return expr.xreplace({a: sp.Float(("%." + str(n) + "f") % float(a))
                           for a in expr.atoms(sp.Float)})


def saturate(value):
    # boundary-layer trick: swap the discontinuous sign() for a clamped ramp
    # so the control doesn't chatter right at the sliding surface
    if value > 1.0:
        return 1.0
    if value < -1.0:
        return -1.0
    return value


class AdaptiveSMC:
    # runs two estimators side by side: an NN learning Delta(x,u), which via
    # C(x) also covers unmatched uncertainty, plus a matched-channel disturbance observer.

    def __init__(self, n_states, n_outputs, surface_func, drift_func,
                 reference_func, decoupling_inv_func, surface_jacobian_func,
                 nominal_dynamics_func, compensation_func, basis_func,
                 K, Lam, phi_layer, Gamma, sigma_W, n_centers,
                 kappa=5.0, kappa_s=5.0, k2=1.0, k3=1.0, k4=1.0,
                 delta_u_dependent=False,
                 estimate_delta=True, estimate_disturbance=True):

        self.n = n_states
        self.p = n_outputs

        self.K = K
        self.Lam = Lam
        self.phi_layer = phi_layer

        self.surface_func = surface_func
        self.drift_func = drift_func
        self.reference_func = reference_func
        self.decoupling_inv_func = decoupling_inv_func
        self.surface_jacobian_func = surface_jacobian_func
        self.nominal_dynamics_func = nominal_dynamics_func
        self.compensation_func = compensation_func

        self.basis_func = basis_func
        self.Gamma = Gamma
        self.sigma_W = sigma_W
        self.kappa = kappa
        self.estimate_delta = estimate_delta
        self.W_Delta = np.zeros((n_centers, self.n))
        self.x_hat = None

        self.kappa_s = kappa_s
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.estimate_disturbance = estimate_disturbance
        self.D_hat = np.zeros(self.p)
        self.s_hat = None

        self.delta_u_dependent = delta_u_dependent
        self.u_prev = None
        self._cache = None

    def reset(self, x0):
        self.u_prev = None
        self.x_hat = np.array(x0, dtype=float).reshape(self.n)
        self.D_hat = np.zeros(self.p)
        self.s_hat = None

    def _evaluate_basis(self, x):
        if self.delta_u_dependent:
            u_feat = self.u_prev.copy() if self.u_prev is not None \
                else np.zeros(self.p)
            return np.asarray(self.basis_func(x, u_feat), dtype=float)
        return np.asarray(self.basis_func(x), dtype=float)

    def compute(self, x_values, ref_tables):
        n, p = self.n, self.p
        x = np.asarray(x_values, dtype=float)

        yd_now = []
        for i in range(p):
            for value in ref_tables[i]:
                yd_now.append(value)

        s = np.array(self.surface_func(*x, *yd_now), dtype=float).reshape(p)
        beta = np.array(self.drift_func(*x), dtype=float).reshape(p)
        v = np.array(self.reference_func(*yd_now), dtype=float).reshape(p)
        Dinv = np.array(self.decoupling_inv_func(*x), dtype=float).reshape(p, p)
        Js = np.array(self.surface_jacobian_func(*x, *yd_now), dtype=float).reshape(p, n)

        if self.s_hat is None:
            self.s_hat = s.copy()

        phi = self._evaluate_basis(x)
        Delta_hat = self.W_Delta.T @ phi

        # push the (n-dim) Delta_hat estimate into the surface through C(x),
        # letting it cancel unmatched uncertainty too, not just the matched part
        C = np.array(self.compensation_func(*x), dtype=float).reshape(p, n)
        sigma = s + C @ Delta_hat

        Js_Delta_hat = Js @ Delta_hat
        Xi_hat = Js_Delta_hat + self.D_hat

        reaching_term = np.array(
            [self.K * saturate(sigma[i] / self.phi_layer) + self.Lam * sigma[i]
             for i in range(p)])
        u = np.asarray(Dinv.dot(v - beta - Xi_hat - reaching_term), dtype=float)
        self.u_prev = u.copy()

        self._cache = dict(phi=phi, s=s, sigma=sigma, Js=Js, x=x, u=u, v=v,
                           beta=beta, Js_Delta_hat=Js_Delta_hat.copy(),
                           Delta_hat=Delta_hat.copy(), D_hat=self.D_hat.copy(),
                           Xi_hat=Xi_hat.copy())
        return list(u)

    def compute_derivs(self, x_dot_real):
        cache = self._cache
        x_dot_real = np.asarray(x_dot_real, dtype=float).reshape(self.n)
        f_nom = np.array(self.nominal_dynamics_func(*cache["x"], *cache["u"]),
                         dtype=float).reshape(self.n)
        x_tilde = cache["x"] - self.x_hat

        cache["x_hat_dot"] = f_nom + cache["Delta_hat"] + self.kappa * x_tilde
        if self.estimate_delta:
            cache["W_Delta_dot"] = self.Gamma * (
                np.outer(cache["phi"], x_tilde) - self.sigma_W * self.W_Delta)
        else:
            cache["W_Delta_dot"] = np.zeros_like(self.W_Delta)

        s_dot_real = cache["Js"] @ x_dot_real - cache["v"]
        s_dot_nominal = cache["Js"] @ f_nom - cache["v"]
        e_D = self.s_hat - cache["s"]

        cache["s_hat_dot"] = (s_dot_nominal + cache["Js_Delta_hat"] + self.D_hat
                              - self.kappa_s * e_D)

        drive = self.k2 * cache["s"] - self.k3 * e_D

        if self.estimate_disturbance:
            cache["D_hat_dot"] = drive - self.k4 * (
                cache["s_hat_dot"] - s_dot_real + self.kappa_s * e_D)
        else:
            cache["D_hat_dot"] = np.zeros(self.p)

        # for diagnostics only; none of this feeds back into the update laws above
        total_mismatch = x_dot_real - f_nom
        cache["Xi_true"] = cache["Js"] @ total_mismatch
        cache["Delta_true"] = total_mismatch

    def step(self, dt):
        cache = self._cache
        self.W_Delta = self.W_Delta + dt * cache["W_Delta_dot"]
        self.x_hat = self.x_hat + dt * cache["x_hat_dot"]
        self.s_hat = self.s_hat + dt * cache["s_hat_dot"]
        self.D_hat = self.D_hat + dt * cache["D_hat_dot"]

    def log(self):
        cache = self._cache
        return dict(Xi_hat=cache["Xi_hat"], Xi_true=cache["Xi_true"],
                    Delta_hat=cache["Delta_hat"], Delta_true=cache["Delta_true"],
                    D_hat=cache["D_hat"], Js=cache["Js"])


class AdaptiveBackstepping:
    # command-filtered backstepping. each step has its own NN+observer over
    # x[0..i], with a filter standing in for the virtual control's derivative

    def __init__(self, n, f_part_funcs, g_funcs, c, rbf_list, tau,
                 Gamma, kappa, k2, k3, k4, sigma_W, N,
                 delta_u_dependent=True, use_filtered_error=False,
                 lambda_I=0.5, estimate_delta=True, estimate_disturbance=True):

        self.n = n

        self.f_part_funcs = f_part_funcs
        self.g_funcs = g_funcs
        self.c = c

        self.rbf_list = rbf_list

        self.tau = tau
        self.Gamma = Gamma
        self.kappa = kappa
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.sigma_W = sigma_W
        self.N = N

        self.W_hat = [np.zeros(N) for _ in range(n)]
        self.D_hat = np.zeros(n)
        self.x_hat = np.zeros(n)

        self.xdf = [None] * n
        self.xdf_dot = np.zeros(n)

        self.xi = np.zeros(n)
        self.first = True

        self.delta_u_dependent = delta_u_dependent
        self.cmd_prev = [None] * n

        self.estimate_delta = estimate_delta
        self.estimate_disturbance = estimate_disturbance

        self.use_filtered_error = use_filtered_error
        self.lambda_I = lambda_I
        self.eps_int = np.zeros(n)

        self._cache = None

    def reset(self, x0):
        self.x_hat = np.array(x0, dtype=float)
        self.cmd_prev = [None] * self.n
        self.eps_int = np.zeros(self.n)

    def _eval_delta_hat_i(self, i, x_slice, cmd):
        if self.delta_u_dependent:
            mu_i = self.rbf_list[i](x_slice, [cmd])
        else:
            mu_i = self.rbf_list[i](x_slice)
        return mu_i, self.W_hat[i] @ mu_i

    def compute(self, x_values, ref_tables):
        n = self.n
        x = np.asarray(x_values, dtype=float)
        ref = ref_tables[0]

        z = np.zeros(n)
        eps = np.zeros(n)
        xc = np.zeros(n)
        caches = []
        u_value = 0.0

        z[0] = x[0] - ref[0]

        for i in range(n):
            if i == 0:
                des_dot = ref[1]
            else:
                des_dot = self.xdf_dot[i]

            # the last step has no filter to compensate against.
            # its compensated error is just the raw error
            if i <= n - 2:
                eps[i] = z[i] - self.xi[i]
            else:
                eps[i] = z[i]

            fi = float(self.f_part_funcs[i](*x))
            gi = float(self.g_funcs[i](*x))

            if i == 0:
                cross = 0.0
            else:
                cross = self.g_funcs[i - 1](*x) * eps[i - 1]

            x_slice = x[:i + 1]
            if self.delta_u_dependent:
                # network also sees its own control, frozen at the PREVIOUS
                # command, which avoids a fixed-point solve at every step.
                cmd_prev = self.cmd_prev[i] if self.cmd_prev[i] is not None else 0.0
                mu_i, Delta_hat_i = self._eval_delta_hat_i(i, x_slice, cmd_prev)
                command = (-self.c[i] * eps[i] - cross - fi - Delta_hat_i
                           - self.D_hat[i] + des_dot) / gi
                self.cmd_prev[i] = command
            else:
                mu_i, Delta_hat_i = self._eval_delta_hat_i(i, x_slice, 0.0)
                command = (-self.c[i] * eps[i] - cross - fi - Delta_hat_i
                           - self.D_hat[i] + des_dot) / gi
                self.cmd_prev[i] = command

            if i < n - 1:
                xc[i + 1] = command
                if self.first:
                    self.xdf[i + 1] = command

                self.xdf_dot[i + 1] = (xc[i + 1] - self.xdf[i + 1]) / self.tau

                z[i + 1] = x[i + 1] - self.xdf[i + 1]
                next_val = x[i + 1]
            else:
                u_value = command
                next_val = command

            x_i_model_dot = fi + gi * next_val
            x_hat_i_dot = (x_i_model_dot + Delta_hat_i + self.D_hat[i]
                           + self.kappa * (x[i] - self.x_hat[i]))

            if self.use_filtered_error:
                s_filt_i = eps[i] + self.lambda_I * self.eps_int[i]
            else:
                s_filt_i = eps[i]

            caches.append(dict(i=i, mu=mu_i, eps=eps[i], s_filt=s_filt_i,
                               e_D=self.x_hat[i] - x[i],
                               x_hat_dot=x_hat_i_dot,
                               gi=gi,
                               xdf=(self.xdf[i + 1] if i < n - 1 else 0.0),
                               xc=(xc[i + 1] if i < n - 1 else 0.0),
                               Delta_hat=Delta_hat_i))

        self._cache = dict(caches=caches, z=z, eps=eps, xc=xc, u=u_value)
        return [u_value]

    def compute_derivs(self, x_dot_real):
        n = self.n
        caches = self._cache["caches"]

        x_hat_dot_vec = np.zeros(n)
        W_hat_dot = [None] * n
        D_hat_dot = np.zeros(n)
        xi_dot = np.zeros(n)

        for c in caches:
            i = c["i"]
            mu_i, e_D = c["mu"], c["e_D"]

            err_i = c["s_filt"]
            x_hat_dot_vec[i] = c["x_hat_dot"]

            drive = self.k2 * err_i - self.k3 * e_D

            W_hat_dot[i] = self.Gamma * (mu_i * drive - self.sigma_W * self.W_hat[i])

            D_hat_dot[i] = drive - self.k4 * (c["x_hat_dot"] - x_dot_real[i]
                                              + self.kappa * e_D)

            if i < n - 1:
                xi_dot[i] = -self.c[i] * self.xi[i] + c["gi"] * (c["xdf"] - c["xc"])

        if not self.estimate_delta:
            W_hat_dot = [np.zeros_like(w) for w in self.W_hat]
        if not self.estimate_disturbance:
            D_hat_dot = np.zeros(n)

        self._cache["x_hat_dot_vec"] = x_hat_dot_vec
        self._cache["W_hat_dot"] = W_hat_dot
        self._cache["D_hat_dot"] = D_hat_dot
        self._cache["xi_dot"] = xi_dot

    def step(self, dt):
        n = self.n
        c = self._cache

        self.x_hat = self.x_hat + dt * c["x_hat_dot_vec"]

        for i in range(n):
            self.W_hat[i] = self.W_hat[i] + dt * c["W_hat_dot"][i]
        self.D_hat = self.D_hat + dt * c["D_hat_dot"]

        for i in range(1, n):
            self.xdf[i] = self.xdf[i] + dt * self.xdf_dot[i]
        self.xi = self.xi + dt * c["xi_dot"]

        if self.use_filtered_error:
            for cc in c["caches"]:
                j = cc["i"]
                self.eps_int[j] = self.eps_int[j] + dt * cc["eps"]
        self.first = False

    def log(self):
        n = self.n
        caches = self._cache["caches"]
        Delta_hat = np.zeros(n)
        xc = np.full(n, np.nan)
        xdf = np.full(n, np.nan)
        for c in caches:
            i = c["i"]
            Delta_hat[i] = c["Delta_hat"]
            if i < n - 1:
                xc[i + 1] = c["xc"]
                xdf[i + 1] = c["xdf"]
        return dict(Delta_hat=Delta_hat, D_hat=self.D_hat.copy(), xc=xc, xdf=xdf)
