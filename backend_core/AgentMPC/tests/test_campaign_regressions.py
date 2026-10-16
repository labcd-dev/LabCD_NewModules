"""
Regression tests for the bugs found by the overactuated-quadcopter test
campaign (backend_core/AgentMPC/reports/quadcopter_campaign). One section per
bug, numbered as in the campaign report; each test reproduces the failure the
campaign observed and fails on the code as it was before the fix.

Run with: pytest backend_core/AgentMPC/tests/test_campaign_regressions.py -v
"""
import numpy as np

from backend_core.AgentMPC.agents.metrics import compute_metrics


def _regulation(error_col, dt=0.01, n_states=1):
    """States/reference pair whose error in state 0 is ``error_col``."""
    T = len(error_col)
    states = np.zeros((T, n_states))
    states[:, 0] = error_col
    return states, np.zeros((T, n_states)), np.zeros((T, 1))


# ---------------------------------------------------------------------------
# Bug 1 -- the oscillation counter had no deadband. After a controller had
# converged, floating-point chatter of order 1e-8 around zero flipped the
# error's sign every few samples and every flip was counted: the user's
# quadcopter controller scored "103 oscillations", which alone made up 90% of
# its cost and ranked it below a controller that never converged.
# ---------------------------------------------------------------------------

def test_numerical_chatter_after_convergence_is_not_oscillation():
    t = np.arange(0, 5, 0.01)
    e = 0.348 * np.exp(-6 * t)
    rng = np.random.default_rng(0)
    e[t > 1.0] = 1e-8 * rng.choice([-1.0, 1.0], size=(t > 1.0).sum())
    states, ref, u = _regulation(e)
    m = compute_metrics(states, u, ref, dt=0.01)
    assert m.oscillation_count == 0


def test_a_genuine_ringing_response_is_still_counted():
    t = np.arange(0, 5, 0.01)
    e = 0.348 * np.exp(-0.8 * t) * np.cos(2 * np.pi * 1.0 * t)
    states, ref, u = _regulation(e)
    m = compute_metrics(states, u, ref, dt=0.01)
    # Swings that leave a 2% band around zero: the first several periods.
    assert 6 <= m.oscillation_count <= 12


# ---------------------------------------------------------------------------
# Bug 2 -- MSE, the settling band and the cost mixed units. A quadcopter
# state vector holds metres, m/s, radians and rad/s; summing their raw squared
# errors let one fast state (a body rate swinging through 2 rad/s for a
# fraction of a second) outweigh an attitude that never reached its target.
# The campaign's Setup-Agent controller left 0.22 rad on roll and still had
# the lower MSE and the lower cost.
# ---------------------------------------------------------------------------

def _two_controllers(dt=0.01, T=5.0):
    """state 0: an angle starting 0.348 rad off; state 1: its rate (rad/s)."""
    t = np.arange(0, T, dt)
    # A -- converges fully; pays for it with a brief, large rate transient.
    a = np.zeros((len(t), 2))
    a[:, 0] = 0.348 * np.exp(-15 * t)
    a[:, 1] = -0.348 * 15 * np.exp(-15 * t)
    # B -- sluggish: settles at 0.22 rad and stays there.
    b = np.zeros((len(t), 2))
    b[:, 0] = 0.22 + 0.128 * np.exp(-3 * t)
    b[:, 1] = -0.128 * 3 * np.exp(-3 * t)
    return a, b, np.zeros((len(t), 2)), np.zeros((len(t), 1))


def test_normalised_mse_ranks_the_converging_controller_first():
    """Raw MSE: A 0.106 vs B 0.029 -- the controller that never arrives looks
    better. Each state's error measured against its own initial gap does not."""
    from backend_core.AgentMPC.agents.metrics import OPTIMIZATION_FOCUS_PRESETS, scalar_cost

    a, b, ref, u = _two_controllers()
    ma, mb = (compute_metrics(x, u, ref, dt=0.01) for x in (a, b))
    assert ma.mse > mb.mse            # raw MSE is still reported as-is ...
    assert ma.nmse < mb.nmse          # ... but the ranking uses the normalised one
    for focus, w in OPTIMIZATION_FOCUS_PRESETS.items():
        assert scalar_cost(ma, w) < scalar_cost(mb, w), focus


def test_settling_is_judged_per_state_not_on_a_mixed_unit_norm():
    """Position starts 5 m off and arrives; roll starts 0.348 rad off and
    stops at 0.22 rad. On the combined norm 0.22 is under 5% of 5.01, so the
    run used to be reported as settled."""
    t = np.arange(0, 5, 0.01)
    x = np.zeros((len(t), 2))
    x[:, 0] = 5.0 * np.exp(-4 * t)
    x[:, 1] = 0.22 + 0.128 * np.exp(-3 * t)
    m = compute_metrics(x, np.zeros((len(t), 1)), np.zeros((len(t), 2)), dt=0.01)
    assert not m.settled


def test_a_state_drifting_away_from_a_held_target_blocks_settling():
    """Tilted hover: the attitude settles, but position (which started on
    target, so it carries no initial gap to normalise by) drifts away."""
    t = np.arange(0, 5, 0.01)
    x = np.zeros((len(t), 2))
    x[:, 0] = 0.2 * np.exp(-5 * t)   # tracked: settles
    x[:, 1] = 0.3 * t                # started on target, drifts off
    m = compute_metrics(x, np.zeros((len(t), 1)), np.zeros((len(t), 2)), dt=0.01)
    assert not m.settled
    assert not m.is_stable


def test_sensor_noise_on_states_that_start_on_target_does_not_block_settling():
    rng = np.random.default_rng(1)
    t = np.arange(0, 5, 0.01)
    x = np.zeros((len(t), 3))
    x[:, 0] = 1.0 * np.exp(-4 * t) + 0.005 * rng.standard_normal(len(t))
    x[:, 1:] = 0.01 * rng.standard_normal((len(t), 2))
    m = compute_metrics(x, np.zeros((len(t), 1)), np.zeros((len(t), 3)), dt=0.01)
    assert m.settled and m.is_stable


# ---------------------------------------------------------------------------
# Bug 3 -- the evaluated window was T/dt - Np steps. A longer prediction
# horizon shortened the run being judged, and an unsettled run's "settling
# time" is the window length, so raising Np lowered the cost by itself: 97%
# of the agents' "improvement" in campaign run D1 (Np 30 -> 60) was this.
# ---------------------------------------------------------------------------

def _vertical_quadrotor():
    from pathlib import Path
    from backend_core.AgentMPC.dynamics.loader import DynamicLoader

    plugin = Path(__file__).resolve().parents[1] / "dynamics" / "plugins" / "example_quadrotor_vertical.py"
    return DynamicLoader.load_from_path(str(plugin)).create_dynamics()


def _run(Np, T=2.0, dt=0.05, **cfg_data):
    from backend_core.AgentMPC.agents.evaluator import run_closed_loop
    from backend_core.AgentMPC.mpc.config import Config

    dyn = _vertical_quadrotor()
    cfg = Config()
    cfg.mpc.prediction_horizon, cfg.mpc.control_horizon = Np, 2
    cfg.data.dt_mpc, cfg.data.simulation_time = dt, T
    for k, v in cfg_data.items():
        setattr(cfg.data, k, v)
    return run_closed_loop(dyn, cfg, {"Np": Np, "Nc": 2, "Q": [1.0, 0.1, 0.1, 0.1],
                                      "R": [10.0] * dyn.n_inputs})


def test_the_evaluated_window_does_not_depend_on_the_prediction_horizon():
    short, long_ = _run(Np=5), _run(Np=30)
    assert len(short["states"]) == len(long_["states"]) == int(2.0 / 0.05) + 1


def test_an_unsettled_run_is_not_cheaper_just_because_np_is_longer():
    from backend_core.AgentMPC.agents.metrics import scalar_cost

    short, long_ = _run(Np=5, T=1.0), _run(Np=15, T=1.0)
    assert not short["metrics"].settled and not long_["metrics"].settled
    assert short["metrics"].settling_time == long_["metrics"].settling_time


# ---------------------------------------------------------------------------
# Bug 16 -- the reference's time axis was linspace(0, T, T/dt): sample k sat at
# k*T/(n-1), not k*dt, so the reference ran 0.2% slow at T=5 s, dt=0.01 (and
# by a different amount for every T), out of step with the simulation clock.
# ---------------------------------------------------------------------------

def test_reference_samples_sit_exactly_on_the_control_clock():
    dyn = _vertical_quadrotor()
    dt, T, f = 0.01, 5.0, 0.5
    ref = dyn.config.desired_trajectory(dt, T, per_state_modes=["sin", "reg", "reg", "reg"],
                                        amplitude=1.0, frequency=f)
    k = np.arange(len(ref))
    np.testing.assert_allclose(ref[:, 0], np.sin(2 * np.pi * f * k * dt), atol=1e-12)


def test_an_unsettled_iteration_reports_no_settling_time():
    """The results table showed '4.81s' under Settling for runs that never
    settled: the evaluator handed the window length over as if it were one."""
    from backend_core.AgentMPC.agents.evaluator import evaluator_node
    from backend_core.AgentMPC.mpc.config import Config

    dyn = _vertical_quadrotor()
    cfg = Config()
    cfg.data.dt_mpc, cfg.data.simulation_time = 0.05, 1.0
    params = {"Np": 5, "Nc": 2, "Q": [1.0, 0.1, 0.1, 0.1], "R": [10.0] * dyn.n_inputs}
    out = evaluator_node({"current_params": params, "iteration": 0, "history": []}, dynamics=dyn, cfg=cfg)
    assert out["metrics"]["Settled"] is False
    assert out["metrics"]["Settling_Time"] == float("inf")
    assert "not settled" in out["history"][-1]


# ---------------------------------------------------------------------------
# Bug 4 -- (a) with very uneven weights (R ~ 1e-5 against Q ~ 1e6) the QP's
# Hessian reaches cond ~ 1e14 and OSQP's ADMM stops far from the optimum while
# reporting success: on a QP captured from campaign run D3 its first move was
# 8x too small. (b) Near a singularity of the plant's own equations (Euler
# angles at pitch = -90 deg) the linearised prediction explodes over the
# horizon and OSQP called the garbage "problem non convex", which reads like a
# tuning problem. Both QPs are saved in tests/data.
# ---------------------------------------------------------------------------

def _qp(name):
    from pathlib import Path
    d = np.load(Path(__file__).parent / "data" / name)
    P = (d["P"] + d["P"].T) / 2
    return P, d["q"], d["A"], d["l"], d["u"]


def test_an_ill_conditioned_qp_is_solved_to_its_true_optimum():
    import scipy.sparse as sp
    from backend_core.AgentMPC.mpc.solver import QPSolver

    P, q, A, l, u = _qp("qp_inaccurate_quadcopter.npz")
    assert np.linalg.cond(P) > 1e13
    x = QPSolver().solve(sp.csc_matrix(P), q, sp.csc_matrix(A), l, u)
    slack = np.minimum(A @ x - l, u - A @ x)
    assert slack.min() > -1e-9                       # feasible
    # No constraint is active at this QP's optimum, so optimality means the
    # gradient vanishes -- an independent check of the answer.
    assert np.all(slack > 1e-6)
    assert np.linalg.norm(P @ x + q) <= 1e-6 * np.linalg.norm(q)


def test_a_prediction_that_explodes_at_a_singularity_is_named_not_called_non_convex():
    import pytest
    from backend_core.AgentMPC.dynamics.base import BaseDynamics, SystemConfig
    from backend_core.AgentMPC.mpc.config import Config
    from backend_core.AgentMPC.mpc.controller import GenericMPC

    class EulerLike(BaseDynamics):
        """phi' = sec(theta)^2 * phi, theta' = u -- singular at theta = -pi/2,
        like Euler-angle kinematics at gimbal lock."""
        def dynamics(self, x, u):
            return np.array([x[0] / np.cos(x[1]) ** 2, u[0]])

    dyn = EulerLike(SystemConfig(n_states=2, n_inputs=1, params={}, state_names=["phi", "theta"],
                                 input_names=["u"], default_initial_state=np.zeros(2),
                                 default_target=np.zeros(2),
                                 input_bounds=(np.array([-1.0]), np.array([1.0]))))
    cfg = Config()
    cfg.mpc.prediction_horizon, cfg.mpc.control_horizon = 20, 4
    cfg.data.dt_mpc = 0.01
    cfg.mpc.set_from_dict({"Np": 20, "Nc": 4, "Q": [1.0, 1.0], "R": [1.0], "P": [1.0, 1.0]})
    mpc = GenericMPC(dyn, cfg)
    x = np.array([0.1, -1.569])
    with pytest.raises(Exception) as exc:
        mpc.control(x, np.zeros(1), np.zeros((20, 2)))
    msg = str(exc.value)
    assert "non convex" not in msg
    assert "explode" in msg and "phi" in msg


# ---------------------------------------------------------------------------
# Bug 5 -- with no position/velocity pairs known, the global "sin"/"pulse"
# modes guessed them from index adjacency, (0,1), (2,3), ... For the
# quadcopter that paired x with y, z with u, phi with theta, psi with p:
# every one of the 12 states got a meaningless reference, and with the user's
# weights the run chased it into gimbal lock and failed.
# ---------------------------------------------------------------------------

def _no_pairs_plant(n=4):
    from backend_core.AgentMPC.dynamics.base import SystemConfig
    return SystemConfig(n_states=n, n_inputs=1, params={}, state_names=[f"s{i}" for i in range(n)],
                        input_names=["u"], default_initial_state=np.zeros(n), default_target=np.zeros(n))


def test_global_sin_without_known_pairs_is_refused_not_guessed():
    import pytest
    cfg = _no_pairs_plant()
    for mode in ("sin", "pulse"):
        with pytest.raises(ValueError, match="per-state"):
            cfg.desired_trajectory(0.01, 2.0, mode=mode)


def test_global_sin_still_uses_detected_pairs():
    cfg = _no_pairs_plant()
    cfg.derivative_pairs = [(0, 2)]
    ref = cfg.desired_trajectory(0.01, 2.0, mode="sin", amplitude=1.0, frequency=0.5)
    moving = [i for i in range(4) if np.ptp(ref[:, i]) > 0]
    assert moving == [0, 2]


# ---------------------------------------------------------------------------
# Bug 6 -- per-state "cos" was documented as "exactly d/dt[sin]" but is
# A*cos(wt), not A*w*cos(wt): setting z = sin and w = cos asked for a vertical
# speed pi times too small at 0.5 Hz, and the reference contradicted itself.
# On the quadcopter that run's MSE was 17x the consistent reference's.
# Derivative modes now exist for exactly this.
# ---------------------------------------------------------------------------

def test_per_state_derivative_modes_are_the_derivatives():
    dyn = _vertical_quadrotor()
    dt, T, A, f = 0.001, 3.0, 0.5, 0.5
    for pos, vel in (("sin", "d_sin"), ("cos", "d_cos"), ("pulse", "d_pulse")):
        ref = dyn.config.desired_trajectory(dt, T, per_state_modes=[pos, vel, "reg", "reg"],
                                            amplitude=A, frequency=f)
        numeric = np.gradient(ref[:, 0], dt)
        np.testing.assert_allclose(ref[2:-2, 1], numeric[2:-2], atol=2e-3 * max(1.0, np.abs(numeric).max()),
                                   err_msg=f"{vel} is not d/dt of {pos}")


# ---------------------------------------------------------------------------
# Bug 7 -- the Diagnostics scan matched keywords as bare substrings, so the
# HTTP codes "401"/"429" were found inside metric values: campaign runs D1 and
# D3 were flagged "authentication error" and "rate limited" because of
# MSE=0.0074750733401940... and MSE=0.0378093217242908...
# ---------------------------------------------------------------------------

def test_status_codes_inside_numbers_are_not_api_errors():
    from backend_core.AgentMPC.agents.diagnostics_agent import scan_for_issues

    logs = [{"node": "METRIC", "message": "MSE=0.007475073340194079 unstable=False"},
            {"node": "METRIC", "message": "MSE=0.03780932172429084 unstable=False"}]
    found = scan_for_issues(logs, [])
    assert "auth_error" not in found and "rate_limit" not in found


def test_real_api_errors_are_still_recognised():
    from backend_core.AgentMPC.agents.diagnostics_agent import scan_for_issues

    logs = [{"node": "ERROR", "message": "Error code: 401 - {'error': 'Invalid API Key'}"},
            {"node": "ERROR", "message": "Error code: 429 - Rate limit reached for requests"}]
    found = scan_for_issues(logs, [])
    assert "auth_error" in found and "rate_limit" in found


def test_a_keyword_ending_a_sentence_still_matches():
    from backend_core.AgentMPC.agents.diagnostics_agent import scan_for_issues

    found = scan_for_issues([{"node": "ERROR", "message": "Request failed: unauthorized."}], [])
    assert "auth_error" in found


# ---------------------------------------------------------------------------
# Bug 8 -- the exported standalone script did not reproduce the run it was
# exported from: it solved with a different QP method (SLSQP), let the
# controller see the perturbed plant's true parameters, and dropped the
# observer, actuator faults, disturbances, noise and custom references. On
# the quadcopter with the user's weights its MSE was 0.341 against the app's
# 0.065.
# ---------------------------------------------------------------------------

def test_the_exported_script_reproduces_the_run_it_came_from(tmp_path):
    import re
    import subprocess
    import sys
    from pathlib import Path

    from backend_core.AgentMPC.agents.evaluator import run_closed_loop
    from backend_core.AgentMPC.agents.export_script import export_scenario_from_cfg, generate_standalone_script
    from backend_core.AgentMPC.dynamics.loader import DynamicLoader
    from backend_core.AgentMPC.mpc.config import Config

    plugin = Path(__file__).resolve().parents[1] / "dynamics" / "plugins" / "example_quadrotor_vertical.py"
    loaded = DynamicLoader.load_from_path(str(plugin))
    dyn = loaded.create_dynamics()
    cfg = Config()
    cfg.mpc.prediction_horizon, cfg.mpc.control_horizon = 10, 2
    cfg.data.dt_mpc, cfg.data.simulation_time = 0.05, 3.0
    cfg.data.nominal_dynamics = loaded.create_dynamics()        # what the MPC believes
    dyn.params["mass"] *= 1.15                                   # what is really there
    cfg.mpc.use_disturbance_observer, cfg.mpc.observer_gain = True, 0.4
    cfg.data.actuator_effectiveness = np.array([0.8, 1.0, 0.6])
    cfg.data.state_disturbance = lambda t: np.array([0.0, 0.3 * np.sin(2 * t), 0.0, 0.05])
    cfg.data.noise_std = np.array([0.002, 0.002, 0.001, 0.001])
    params = {"Np": 10, "Nc": 2, "Q": [10.0, 1.0, 1.0, 1.0], "R": [0.1, 1.0, 1.0], "P": [10.0, 1.0, 1.0, 1.0]}
    app = run_closed_loop(dyn, cfg, params)
    assert "error" not in app

    text = generate_standalone_script(plugin.read_text(encoding="utf-8"), "QuadrotorVerticalDynamics", params,
                                      0.05, 3.0, "vertical quadrotor",
                                      **export_scenario_from_cfg(cfg, dyn))
    script = tmp_path / "export.py"
    script.write_text(text, encoding="utf-8")
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=tmp_path,
                         timeout=600, env={**__import__("os").environ, "MPLBACKEND": "Agg"})
    assert out.returncode == 0, out.stderr[-2000:]
    mse = float(re.search(r"MSE .*?: ([0-9.eE+-]+)", out.stdout).group(1))
    steps = int(re.search(r"Steps simulated: (\d+)", out.stdout).group(1))
    assert steps == len(app["states"]) - 1
    assert mse == pytest_approx(app["metrics"].mse)


def pytest_approx(v, rel=1e-6):
    import pytest
    return pytest.approx(v, rel=rel)


# ---------------------------------------------------------------------------
# Bug 9 -- every number in the agents' prompts and the report analysis was
# rounded to 2 decimal places. Campaign run D1's MSE improved 35%
# (0.0112 -> 0.0073) and the Report Agent, shown "0.01 ... 0.01", wrote
# that MSE "remained at 0.01 throughout the tuning process".
# ---------------------------------------------------------------------------

def test_rounding_keeps_three_significant_figures():
    from backend_core.AgentMPC.agents.formatting import fmt_num, round_floats, smart_round

    assert round_floats([0.011190266584208021, 0.007349958380174677]) == [0.0112, 0.00735]
    assert smart_round(123.456) == 123.46          # large values: 2 decimals, as before
    assert fmt_num(0.011190266584208021) == "0.0112"
    assert fmt_num(0.007349958380174677) == "0.00735"
    assert fmt_num(2.5) == "2.50"


# ---------------------------------------------------------------------------
# Bug 10 -- the robustness scenarios were not realistic on the quadcopter:
# Level 3 raised gravity g by 19.5% (a constant of nature, not an uncertain
# plant parameter), spent selections on parameters that are exactly zero
# (Ixz, Iyz), and only ever increased values; Level 4's default disturbance on
# a plant without declared state bounds was 0.0005 per second on every state
# -- 5% of a 0.01 "span" that was really a noise sigma.
# ---------------------------------------------------------------------------

def _quad_like_plant():
    from backend_core.AgentMPC.dynamics.base import BaseDynamics, SystemConfig

    class P(BaseDynamics):
        def dynamics(self, x, u):
            return np.zeros(3)

    params = {"m": 2.0, "g": 9.81, "Ixx": 0.02, "Ixy": 0.0, "Ixz": 0.0, "Iyy": 0.02, "Izz": 0.037,
              "Kt": 2.83e-6, "Kd": 4.04e-8, "L": 0.175, "e": 0.02}
    return P(SystemConfig(n_states=3, n_inputs=1, params=params, state_names=["x", "u", "phi"],
                          input_names=["U"], default_initial_state=np.array([0.1, 0.0, 0.348]),
                          default_target=np.zeros(3)))


def test_robust_level_leaves_constants_and_zero_parameters_alone_and_moves_both_ways():
    from backend_core.AgentMPC.agents.scenario_presets import perturb_physical_parameters

    signs = set()
    for seed in range(20):
        dyn = _quad_like_plant()
        changed = perturb_physical_parameters(dyn, rng=np.random.default_rng(seed))
        assert "g" not in changed
        assert not ({"Ixy", "Ixz"} & set(changed))
        assert changed
        signs |= {np.sign(new - old) for old, new in changed.values()}
    assert signs == {-1.0, 1.0}


def test_default_disturbance_is_scaled_to_the_states_not_to_a_noise_sigma():
    from backend_core.AgentMPC.agents.scenario_presets import apply_scenario_level
    from backend_core.AgentMPC.mpc.config import Config

    dyn, cfg = _quad_like_plant(), Config()
    apply_scenario_level(dyn, cfg, 4)
    d = np.asarray(cfg.data.state_disturbance)
    # 5% of each state's own scale per second: the initial gap where there is
    # one (0.1 m, 0.348 rad), a unit scale where there is none.
    np.testing.assert_allclose(d, 0.05 * np.array([0.1, 1.0, 0.348]))


def test_default_noise_follows_each_states_own_scale():
    """Not one absolute 0.01 for metres, m/s and radians alike."""
    from backend_core.AgentMPC.agents.scenario_presets import apply_scenario_level
    from backend_core.AgentMPC.mpc.config import Config

    dyn, cfg = _quad_like_plant(), Config()
    apply_scenario_level(dyn, cfg, 2)
    np.testing.assert_allclose(cfg.data.noise_std, 0.01 * np.array([0.1, 1.0, 0.348]))


# ---------------------------------------------------------------------------
# Bug 11 -- numeric_tuner's default search space (Q in 1e-2..1e3, R in
# 1e-3..10) cannot express a Q/R ratio above 1e6, while the weights that make
# the quadcopter converge sit at ~1e4 (grouped search) to ~1e11 (the user's
# hand tuning); and searching all 20 weights independently found nothing that
# converged in 48 tries. A grouped search -- one weight per physical group of
# states, plus R, Latin-hypercube in log space -- found converging weights.
# ---------------------------------------------------------------------------

def test_default_search_space_reaches_the_ratios_that_work():
    from backend_core.AgentMPC.agents.numeric_tuner import SearchBounds

    b = SearchBounds()
    assert 10 ** (b.q_log_range[1] - b.r_log_range[0]) >= 1e11   # the user's hand-tuned ratio
    assert 10 ** (b.q_log_range[0] - b.r_log_range[1]) <= 1e-2


def test_grouped_search_shares_one_weight_per_group_and_covers_the_range():
    from backend_core.AgentMPC.agents.numeric_tuner import SearchBounds, grouped_search

    seen = []

    def objective(params):
        seen.append(params)
        q = np.log10(params["Q"])
        return float((q[0] - 4) ** 2 + (q[2] - 1) ** 2 + (np.log10(params["R"][0]) + 2) ** 2)

    groups = [[0, 1], [2, 3]]
    res = grouped_search(objective, n_states=4, n_inputs=2, groups=groups, n_trials=30,
                         bounds=SearchBounds(), Np=10, Nc=2, seed=0)
    for p in seen:
        assert p["Q"][0] == p["Q"][1] and p["Q"][2] == p["Q"][3]
        assert p["R"][0] == p["R"][1] and (p["Np"], p["Nc"]) == (10, 2)
    q0 = np.log10([p["Q"][0] for p in seen[:20]])          # the space-filling stage
    assert q0.min() < -1 and q0.max() > 5
    assert res.best_cost < 0.5                              # and it homes in on the optimum


# ---------------------------------------------------------------------------
# Bug 12 -- Setup's starting weights came from Bryson's rule applied to the
# plant's OPEN-LOOP excursion over a 2 s probe at 25% of the input range. A
# state that rotates fast (yaw, roll) swept through large angles in that time
# and got an almost-zero weight: the quadcopter seed was phi 0.08, theta 10,
# psi 0.006 -- one attitude angle weighted 1700x another -- and never
# converged; the agents, seeded with it, never repaired it.
# ---------------------------------------------------------------------------

def _rigid_body_2d():
    """x, y, psi with their rates; yaw responds 50x faster than translation."""
    from backend_core.AgentMPC.dynamics.base import BaseDynamics, SystemConfig

    class Body(BaseDynamics):
        def dynamics(self, x, u):
            return np.array([x[3], x[4], x[5], u[0], u[1], 50.0 * u[2]])

    return Body(SystemConfig(n_states=6, n_inputs=3, params={}, state_names=["x", "y", "psi", "vx", "vy", "r"],
                             input_names=["Fx", "Fy", "Mz"], default_initial_state=np.zeros(6),
                             default_target=np.zeros(6),
                             input_bounds=(np.full(3, -1.0), np.full(3, 1.0))))


def test_seed_weights_states_of_the_same_role_alike():
    from backend_core.AgentMPC.agents.dynamics_validator import estimate_initial_qr

    Q, R, note, _ = estimate_initial_qr(_rigid_body_2d())
    x, y, psi, vx, vy, r = Q
    assert x == y == psi                 # positions and the angle: one role
    assert vx == vy == r                 # their rates: another
    assert vx < x                        # rates matter less than where the body is


def test_seed_does_not_mistake_a_gravity_coupling_for_a_rate():
    """Ball and beam: r_dot' = (5/7) g theta. theta is an angle to hold, not
    the velocity of anything, even though r_dot's row has only that entry."""
    from backend_core.AgentMPC.agents.dynamics_validator import estimate_initial_qr
    from backend_core.AgentMPC.dynamics.base import BaseDynamics, SystemConfig

    class BallBeam(BaseDynamics):
        def dynamics(self, x, u):
            return np.array([x[1], 7.0 * x[2], x[3], u[0]])

    dyn = BallBeam(SystemConfig(n_states=4, n_inputs=1, params={}, state_names=["r", "r_dot", "theta", "theta_dot"],
                                input_names=["tau"], default_initial_state=np.zeros(4), default_target=np.zeros(4),
                                input_bounds=(np.array([-1.0]), np.array([1.0]))))
    Q, *_ = estimate_initial_qr(dyn)
    assert Q[2] == Q[0] and Q[1] < Q[0] and Q[3] < Q[2]


# ---------------------------------------------------------------------------
# Bug 13 -- (a) Setup suggested dt = 0.00375 s for the quadcopter, from the
# open-loop rise time of its directly actuated states (always near-instant),
# while 0.01-0.02 s worked perfectly: runs were 2.7-5x slower for nothing.
# (b) the suitability facts reported recommended_dt_max = 7600 s for its
# integrator chains: 0.1 / (an eigenvalue of 1.3e-5 that is rounding noise).
# ---------------------------------------------------------------------------

def _rigid_body_with_rounding_noise():
    dyn = _rigid_body_2d()
    base = dyn.dynamics
    dyn.dynamics = lambda x, u: base(x, u) + 1.3e-5 * np.array([0, 0, 0, x[1], -x[0], 0])
    return dyn


def test_no_recommended_dt_from_a_rounding_noise_eigenvalue():
    from backend_core.AgentMPC.agents.mpc_suitability import assess_mpc_suitability

    facts = assess_mpc_suitability(_rigid_body_with_rounding_noise(), dt=0.02)["facts"]
    assert facts.get("recommended_dt_max") is None


def test_suggested_dt_comes_from_the_closed_loop_not_the_actuator_reaction():
    from backend_core.AgentMPC.agents.dynamics_validator import estimate_dt
    from backend_core.AgentMPC.agents.mpc_suitability import assess_mpc_suitability

    dyn = _rigid_body_with_rounding_noise()
    dt = estimate_dt(dyn)
    assert dt >= 0.005
    codes = {f["code"] for f in assess_mpc_suitability(dyn, dt=dt)["findings"]}
    assert not {c for c in codes if c.startswith("euler")}


def test_suggested_dt_keeps_euler_stable_on_lightly_damped_modes():
    """A suggested dt must pass the suitability check made from the same
    plant: a lightly damped stable mode needs dt < -2 Re(l) / |l|^2."""
    from backend_core.AgentMPC.agents.dynamics_validator import estimate_dt
    from backend_core.AgentMPC.agents.mpc_suitability import assess_mpc_suitability
    from backend_core.AgentMPC.dynamics.base import BaseDynamics, SystemConfig

    class LightlyDamped(BaseDynamics):
        def dynamics(self, x, u):       # wn = 20 rad/s, zeta = 0.01 -> Euler limit 0.001 s
            return np.array([x[1], -400.0 * x[0] - 0.4 * x[1] + u[0]])

    dyn = LightlyDamped(SystemConfig(n_states=2, n_inputs=1, params={}, state_names=["q", "q_dot"],
                                     input_names=["u"], default_initial_state=np.zeros(2),
                                     default_target=np.zeros(2), input_bounds=(np.array([-1.0]), np.array([1.0]))))
    dt = estimate_dt(dyn)
    codes = {f["code"] for f in assess_mpc_suitability(dyn, dt=dt)["findings"]}
    assert "euler_unstable_at_dt" not in codes, dt


# ---------------------------------------------------------------------------
# Bug 14 -- the "reference cannot be followed exactly" warning always blamed
# under-actuation ("On an under-actuated plant this is the norm..."). On the
# over-actuated quadcopter the real cause was a velocity reference that was
# not the derivative of its position reference, and the message pointed the
# wrong way.
# ---------------------------------------------------------------------------

def test_infeasible_reference_message_names_the_actual_cause():
    from backend_core.AgentMPC.agents.mpc_suitability import assess_reference

    dyn = _rigid_body_2d()                     # 3 independent inputs, 6 states
    t = np.arange(0, 4, 0.02)
    ref = np.zeros((len(t), 6))
    ref[:, 0] = 0.5 * np.sin(2 * np.pi * 0.5 * t)  # x moves, vx stays 0: kinematically impossible
    msgs = [f["message"] + f["suggestion"] for f in assess_reference(dyn, ref, 0.02)
            if f["code"] == "reference_infeasible"]
    assert msgs and "under-actuated" not in msgs[0] and "derivative" in msgs[0]

    ref = np.zeros((len(t), 6))                # four positions/rates moving independently > rank 3
    for i in (0, 1, 2, 3):
        ref[:, i] = 0.3 * np.sin(2 * np.pi * (0.3 + 0.2 * i) * t)
    msgs = [f["message"] for f in assess_reference(dyn, ref, 0.02) if f["code"] == "reference_infeasible"]
    assert msgs and "independent" in msgs[0]


# ---------------------------------------------------------------------------
# Bug 15 -- actor.py's comment said Nc is capped at 40% of Np while the code
# (and the prompt) use 20%. Guard: the rule the Actor is TOLD must be the rule
# the code ENFORCES, worked example included.
# ---------------------------------------------------------------------------

def test_the_nc_cap_in_the_prompt_is_the_one_enforced():
    import math
    import re
    from pathlib import Path

    from backend_core.AgentMPC.agents.actor import MAX_NC_FRACTION_OF_NP

    prompt = (Path(__file__).resolve().parents[1] / "prompts" / "actor.yaml").read_text(encoding="utf-8")
    pct, np_ex, nc_ex = re.search(r"Nc must not exceed (\d+)% of Np \(rounded up, so Np=(\d+) allows Nc up to (\d+)\)",
                                  prompt).groups()
    assert int(pct) / 100 == MAX_NC_FRACTION_OF_NP
    assert math.ceil(MAX_NC_FRACTION_OF_NP * int(np_ex)) == int(nc_ex)
    src = (Path(__file__).resolve().parents[1] / "agents" / "actor.py").read_text(encoding="utf-8")
    assert "40% of the prediction horizon" not in src


# ---------------------------------------------------------------------------
# Bug 12, follow-up -- the before/after comparison on all eight test-suite
# plants found two regressions of the first role-based seed: (a) units taken
# from declared bounds put a rate (narrow bounds) ABOVE the angle it is the
# rate of on the double inverted pendulum; (b) R was picked from an
# infinite-horizon LQR on u, while the controller is a short-horizon MPC that
# penalises du -- on an unstable plant the "fast enough" LQR R let the MPC
# diverge (double pendulum: final error 16x its start).
# ---------------------------------------------------------------------------

def _unstable_pendulum(rate_bound=0.2):
    from backend_core.AgentMPC.dynamics.base import BaseDynamics, SystemConfig

    class Pend(BaseDynamics):
        def dynamics(self, x, u):
            return np.array([x[1], 20.0 * np.sin(x[0]) + 5.0 * u[0]])

    return Pend(SystemConfig(n_states=2, n_inputs=1, params={}, state_names=["theta", "omega"],
                             input_names=["tau"], default_initial_state=np.array([0.2, 0.0]),
                             default_target=np.zeros(2),
                             state_bounds=(np.array([-3.14, -rate_bound]), np.array([3.14, rate_bound])),
                             input_bounds=(np.array([-10.0]), np.array([10.0]))))


def test_a_rate_never_outweighs_the_state_it_is_the_rate_of():
    from backend_core.AgentMPC.agents.dynamics_validator import estimate_initial_qr

    Q, *_ = estimate_initial_qr(_unstable_pendulum())
    assert Q[1] < Q[0]


def test_the_seed_stabilises_the_mpc_it_is_meant_for_on_an_unstable_plant():
    from backend_core.AgentMPC.agents.dynamics_validator import estimate_dt, estimate_initial_qr
    from backend_core.AgentMPC.agents.evaluator import run_closed_loop
    from backend_core.AgentMPC.mpc.config import Config

    dyn = _unstable_pendulum(rate_bound=10.0)
    Q, R, *_ = estimate_initial_qr(dyn)
    cfg = Config()
    cfg.mpc.prediction_horizon, cfg.mpc.control_horizon = 20, 4
    cfg.data.dt_mpc, cfg.data.simulation_time = estimate_dt(dyn), 3.0
    res = run_closed_loop(dyn, cfg, {"Np": 20, "Nc": 4, "Q": Q, "R": R, "P": Q})
    assert "error" not in res and not res["unstable"], res.get("unstable_reason") or res.get("error")
    assert res["metrics"].settled


def test_suggested_dt_is_not_inflated_when_the_seed_misses_its_time_constant():
    """Planar VTOL: no R lets a 20-step MPC reach the 0.3 s target, the
    'best achievable' time constant was ~2.8 s, and dt = tau/15 = 0.19 s made
    the run diverge. A missed target is no basis for a dt."""
    from backend_core.AgentMPC.agents import dynamics_validator as V

    dyn = _rigid_body_2d()
    real = V._role_seed
    try:
        V._role_seed = lambda *a, **k: {**real(*a, **k), "tau": 2.8, "tau_met": False}
        dt = V.estimate_dt(dyn)
    finally:
        V._role_seed = real
    assert dt < 2.8 / 15


# ---------------------------------------------------------------------------
# Initial-parameter methods (approved after the eight-plant test report):
# Bryson's rule for Q/R, the Riccati (DARE) terminal weight, and a horizon
# rule for Np/Nc.
# ---------------------------------------------------------------------------

def test_bryson_weights_follow_the_acceptable_deviations():
    """Q_ii = 1/a_i^2: a state allowed twice the deviation weighs a quarter;
    a rate with no value of its own may close its position's gap in RATE_TIME."""
    from backend_core.AgentMPC.agents.dynamics_validator import RATE_TIME, estimate_initial_qr

    dyn = _rigid_body_2d()                                    # x, y, psi, vx, vy, r
    Q, *_ = estimate_initial_qr(dyn, acceptable=[0.1, 0.2, 0.05, np.nan, np.nan, np.nan])
    Q = np.asarray(Q)
    assert Q[0] / Q[1] == pytest_approx(4.0)                  # (0.2 / 0.1)^2
    assert Q[2] / Q[0] == pytest_approx(4.0)                  # (0.1 / 0.05)^2
    assert Q[3] / Q[0] == pytest_approx(RATE_TIME ** 2)       # vx may be 0.1 / RATE_TIME
    assert Q[5] / Q[2] == pytest_approx(RATE_TIME ** 2)


def test_a_plugin_can_declare_its_acceptable_deviations():
    from backend_core.AgentMPC.agents.dynamics_validator import estimate_initial_qr

    dyn = _rigid_body_2d()
    dyn.config.acceptable_deviation = np.array([0.1, 0.2, 0.05, np.nan, np.nan, np.nan])
    Q, *_ = estimate_initial_qr(dyn)
    assert Q[0] / Q[1] == pytest_approx(4.0)


def test_riccati_terminal_weight_stabilises_where_p_equal_q_does_not():
    """Unstable pendulum, Np = 60, Nc = 12: with P = Q the end of a long
    horizon is weighted no more than any other step and the loop does not
    settle; the DARE terminal weight is the infinite-horizon cost-to-go."""
    from backend_core.AgentMPC.agents.evaluator import run_closed_loop
    from backend_core.AgentMPC.mpc.config import Config

    settled = {}
    for mode in ("given", "dare"):
        dyn = _unstable_pendulum(rate_bound=10.0)
        cfg = Config()
        cfg.mpc.prediction_horizon, cfg.mpc.control_horizon = 60, 12
        cfg.data.dt_mpc, cfg.data.simulation_time = 0.02, 3.0
        cfg.mpc.terminal_weight_mode = mode
        res = run_closed_loop(dyn, cfg, {"Np": 60, "Nc": 12, "Q": [10.0, 1.0], "R": [0.1], "P": [10.0, 1.0]})
        settled[mode] = res["metrics"].settled
    assert settled == {"given": False, "dare": True}


def test_dare_is_the_default_terminal_weight():
    from backend_core.AgentMPC.mpc.config import MPCConfig
    assert MPCConfig().terminal_weight_mode == "dare"


def test_horizon_covers_the_settling_time_but_not_an_unstable_blow_up():
    from backend_core.AgentMPC.agents.dynamics_validator import (HORIZON_GROWTH_LIMIT, SEED_TIME_CONSTANT,
                                                                 suggest_horizon)

    stable = suggest_horizon(np.zeros((2, 2)), 0.02)          # integrators: 4 time constants
    assert stable["horizon_s"] == pytest_approx(4 * SEED_TIME_CONSTANT, rel=0.05)
    assert stable["Nc"] == int(np.ceil(0.2 * stable["Np"]))
    unstable = suggest_horizon(np.array([[0.0, 1.0], [32.0, 0.0]]), 0.005)   # lambda = 5.66 1/s
    assert unstable["horizon_s"] <= np.log(HORIZON_GROWTH_LIMIT) / np.sqrt(32.0) + 0.005


def test_setup_suggests_np_and_nc():
    from pathlib import Path
    from backend_core.AgentMPC.agents.dynamics_validator import analyze_and_setup

    src = (Path(__file__).resolve().parents[1] / "dynamics" / "plugins" / "example_quadrotor_vertical.py").read_text(encoding="utf-8")
    s = analyze_and_setup(src)
    assert s.suggested_Np and s.suggested_Nc and 10 <= s.suggested_Np <= 60
    assert s.suggested_Nc == int(np.ceil(0.2 * s.suggested_Np))


def test_the_actor_is_told_who_sets_the_terminal_weight():
    from backend_core.AgentMPC.agents.actor import _dt_policy_block
    from backend_core.AgentMPC.mpc.config import Config

    cfg = Config()
    assert "Riccati" in _dt_policy_block(cfg) and "ignored" in _dt_policy_block(cfg)
    cfg.mpc.terminal_weight_mode = "given"
    assert "yours to set" in _dt_policy_block(cfg)


def test_the_export_keeps_the_terminal_weight_mode():
    from backend_core.AgentMPC.agents.export_script import export_scenario_from_cfg
    from backend_core.AgentMPC.mpc.config import Config

    cfg = Config()
    cfg.mpc.terminal_weight_mode = "given"
    assert export_scenario_from_cfg(cfg, _vertical_quadrotor())["terminal_weight_mode"] == "given"
