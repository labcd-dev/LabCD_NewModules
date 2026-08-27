# Plant-Model Agent — Interaction Examples

This document defines the expected behavior of `PlantModelAgent` across a
wide spectrum of user personas, input styles, and edge cases. It is both a
product specification and a regression test plan.

---

## Core Invariants (must hold for every case)

| # | Invariant |
|---|-----------|
| 1 | **Every model response is exactly one JSON object** — no markdown fences, no prose outside JSON. |
| 2 | **No hardcoded UI strings** — all display text originates from the model's `reply`, `system_name`, or `python_code` fields. |
| 3 | **`status: complete` is never premature** — only after explicit user acceptance, max drafts, or a pasted finished model. |
| 4 | **`python_code` is always runnable** — `def dynamics(t, x, u):`, imports only `numpy` (and `scipy` if truly needed), no placeholders, no `pass`, no `TODO`. |
| 5 | **Assumptions are labelled in code comments** — never silent magic numbers. |
| 6 | **One focused question per `continue` turn** — no checklist bombardment. |

---

## Personas & Cases

### Case I — Advanced user (paste full equations)

**Profile:** Knows exactly what they want. Pastes parameters and state equations.

**Turn 1 — Human**
\`\`\`
Here is my system:
\`\`\`python
m, Iz, W, B, zb, zg = 50, 5, 500, 510, -0.1, 0
Zw, Zq, Zd, Mw, Mq, Md = -20, -5, -10, -3, -8, -5
z, w, theta, q = x[0], x[1], x[2], x[3]
xdot = [w*np.cos(theta) - np.sin(theta),
        (Zw*w + Zq*q + Zd*u + (W-B)*np.sin(theta)) / m,
        q,
        (Mw*w + Mq*q + Md*u + (zg*W - zb*B)*np.sin(theta)) / Iz]
\`\`\`
\`\`\`

**Expected AI:** `status: draft` — full `dynamics(t, x, u)` with the same
parameters, state unpacking commented, and the "finish to accept" cue.
`final_result` is **null** (not accepted yet).

**Turn 2 — Human**
```
done
```

**Expected AI:** `status: complete` (or agent promotes latest draft).
`final_result` populated. UI shows success panel with download button.

---

### Case II — Beginner user (guided discovery + revisions)

**Profile:** Has a physical idea but no equations. Needs the agent to derive
dynamics from a description.

**Turn 1 — Human**
```
Hi how you doing?
```

**Expected AI:** `status: continue`. Short greeting + **one** concrete
question about the plant type. No code. `final_result` null.

**Turn 2 — Human**
```
it's a motor turning a beam. the beam can rotate, and a ball on the beam
rolls over it under gravity. we're over the moon, not earth
```

**Expected AI:** `status: draft`. Ball-and-beam with `g = 1.62` (lunar).
States: `theta` (beam angle), `omega` (angular rate), `r` (ball position),
`v` (ball velocity). Input: motor torque or voltage. All parameters labelled
as assumed. `final_result` null.

**Turn 3 — Human**
```
let's forget about the moon, make it earth
```

**Expected AI:** `status: draft`. Revised draft with `g = 9.81`. Same
structure otherwise. Still null `final_result`.

**Turn 4 — Human**
```
finish
```

**Expected AI:** `status: complete` (or promotion). `final_result` set.
UI locks the model name (e.g. "Ball-and-beam (earth g)").

---

### Case III — Raw-text describer (unstructured paragraph)

**Profile:** Dumps a wall of text with mixed relevant and irrelevant details.

**Turn 1 — Human**
```
So I'm working on this project for my controls class and it's like a drone
but underwater kind of? It has four thrusters and we need to model how it
moves up and down. The prof said we should use Newton-Euler. There's also
some buoyancy stuff and the mass is around 30 kg. We care mostly about
heave and pitch for now. The thrusters take voltage commands. Oh and we
did a pool test last week but I don't have those numbers yet.
```

**Expected AI:** `status: continue`. Acknowledge the system (underwater
vehicle / AUV), confirm the relevant degrees of freedom (heave, pitch),
and ask **one** high-value question — e.g. "Do you have approximate
thruster coefficients, or should I use textbook assumptions?" Do NOT ask
for pool test details or the prof's name.

**Turn 2 — Human**
```
just use textbook assumptions for everything
```

**Expected AI:** `status: draft`. Heave-pitch AUV model with labelled
assumptions for mass, added mass, thruster coefficients, buoyancy, and
drag. States: `z`, `w`, `theta`, `q`. Input: `u[0]` (port voltage),
`u[1]` (starboard voltage) or a single aggregated thrust if SISO
preferred. Mention assumptions in `reply`.

---

### Case IV — Sequential info giver (drip-feed)

**Profile:** Reveals one fact per turn. The agent must accumulate context
and draft as soon as possible.

**Turn 1 — Human**
```
I need a model of a DC motor
```

**Expected AI:** `status: continue`. "Armature-controlled or field-controlled?
And is the load inertial or negligible?"

**Turn 2 — Human**
```
armature controlled, with an inertial load
```

**Expected AI:** `status: continue`. "What are the approximate values for
armature resistance R, inductance L, torque constant Kt, back-EMF constant
Kb, rotor inertia J, and friction b? Or should I use typical values?"

**Turn 3 — Human**
```
use typical values
```

**Expected AI:** `status: draft`. DC motor with inertial load.
States: `theta`, `omega`, `i_a` (or `i`). Parameters labelled as typical.
Input: armature voltage. Ask what to change or to finish.

**Turn 4 — Human**
```
add a flexible shaft between motor and load
```

**Expected AI:** `status: draft`. Two-mass flexible shaft model.
States: `theta_m`, `omega_m`, `theta_l`, `omega_l`, plus `i_a`.
Spring `k` and damper `d` labelled as assumed.

**Turn 5 — Human**
```
finish
```

**Expected AI:** `status: complete`. `final_result` populated.

---

### Case V — Uncertain / exploratory user

**Profile:** Not sure what system they need. Might ask "what should I model?"

**Turn 1 — Human**
```
I need a plant model for my MPC class but I'm not sure what to pick
```

**Expected AI:** `status: continue`. Do NOT pick for them without context.
Ask: "What physical domain interests you — mechanical, thermal, chemical,
electrical, or aerospace? And do you prefer a system with 2–4 states for
simplicity, or a more realistic higher-order model?"

**Turn 2 — Human**
```
maybe something thermal, simple is better
```

**Expected AI:** `status: continue`. "A classic choice is a heated tank or
a room HVAC zone. Which sounds closer — a fluid tank with inlet/outlet
temperatures, or a room with a heater and ambient heat loss?"

**Turn 3 — Human**
```
heated tank sounds good
```

**Expected AI:** `status: draft`. Single-state or two-state thermal tank.
State: `T` (fluid temperature). Input: `u[0]` heater power or inlet
temperature. Parameters: volume, flow rate, specific heat, density, heat
transfer coefficient — all labelled as assumed.

---

### Case VI — Domain expert with implicit / wrong assumptions

**Profile:** Knows their field but states something physically questionable.
The agent should politely correct via `continue`, not silently comply.

**Turn 1 — Human**
```
I want a quadrotor model with 12 states but ignore gravity because we're
in space
```

**Expected AI:** `status: continue`. "If this is a free-floating space
robot, gravity can indeed be ignored, but quadrotor propulsion (rotor
thrust) requires an atmosphere or reaction wheels. Are you modelling:
(a) a space robot with reaction wheels, or (b) a drone in a vacuum
chamber where gravity is cancelled by a test rig?"

**Turn 2 — Human**
```
space robot with reaction wheels
```

**Expected AI:** `status: draft`. Rename system appropriately
(e.g. "Spacecraft attitude with reaction wheels"). States: attitude
(quaternion or Euler angles), angular rates, wheel speeds. No gravity.
No aerodynamic drag. Parameters labelled.

---

### Case VII — Minimalist / one-word user

**Profile:** Gives terse responses. Agent must not get stuck in infinite
`continue` loops.

**Turn 1 — Human**
```
cstr
```

**Expected AI:** `status: continue`. "Continuous stirred-tank reactor.
Isothermal or exothermic? And do you want concentration and temperature
as states, or just concentration?"

**Turn 2 — Human**
```
exothermic, both
```

**Expected AI:** `status: draft`. Exothermic CSTR with cooling jacket.
States: `C_A`, `T`, `T_c` (or `T_j`). Input: coolant flow or temperature.
Arrhenius rate law. Parameters labelled.

**Turn 3 — Human**
```
ok
```

**Expected AI:** `status: continue` is acceptable here if the agent asks
"Should I adjust any parameters, or is this ready to finalise?" However,
if the agent interprets "ok" as acceptance after a draft was shown, it
may also promote to `complete`. Either is valid as long as `final_result`
is only set on clear acceptance or max drafts.

---

### Case VIII — Revision-heavy user (churn)

**Profile:** Accepts a draft, then keeps changing requirements. Draft
counter must increment correctly.

**Turn 1 — Human**
```
model a spring-mass-damper
```

**Expected AI:** `status: draft`. SMD with states `x`, `v`.

**Turn 2 — Human**
```
make it nonlinear with a cubic spring
```

**Expected AI:** `status: draft`. `F_spring = k1*x + k3*x**3`. Draft
counter = 1.

**Turn 3 — Human**
```
add dry friction too
```

**Expected AI:** `status: draft`. Coulomb + viscous friction.
Draft counter = 2.

**Turn 4 — Human**
```
actually remove the cubic spring, keep linear
```

**Expected AI:** `status: draft`. Linear spring, dry friction retained.
Draft counter = 3.

**Turn 5 — Human**
```
finish
```

**Expected AI:** `status: complete`. `final_result` set. Draft counter
stops incrementing.

---

### Case IX — Meta-question / off-topic

**Profile:** Asks about the agent itself or unrelated topics.

| Input | Expected AI |
|-------|-------------|
| `who made you?` | `status: continue`. Short factual answer + immediate pivot back: "What system are you modelling?" |
| `what is MPC?` | `status: continue`. One-sentence definition + pivot: "Do you have a specific plant in mind for your controller?" |
| `write me a poem` | `status: continue`. Polite refusal + pivot to plant modelling. |
| `what's 2+2?` | `status: continue`. Answer briefly, then pivot. |

**Invariant:** Never emit `python_code` for meta questions. Never get
derailed for more than one turn.

---

### Case X — Safety / physically impossible request

**Profile:** Asks for dynamics that violate physics or are unsafe.

**Turn 1 — Human**
```
model a perpetual motion machine with positive energy output
```

**Expected AI:** `status: continue`. Refuse politely: "I can't model a
perpetual motion machine because it violates conservation of energy.
Would you like a conventional pendulum, a flywheel energy storage
system, or something else instead?"

**Turn 1 — Human**
```
model a car that accelerates without any input force
```

**Expected AI:** `status: continue`. "A car coasting with zero input
force will decelerate due to drag and rolling resistance. Should I model
coast-down dynamics, or do you mean a car with a motor input?"

---

### Case XI — MIMO / high-dimensional user

**Profile:** Wants a multi-input, multi-output system with many states.

**Turn 1 — Human**
```
I need a 6-DOF rigid body for a submarine with 4 thrusters
```

**Expected AI:** `status: continue`. "Full 6-DOF is 12+ states. Do you
need all 6-DOF, or can we start with a reduced model (e.g. surge, heave,
pitch) to keep the MPC tractable?"

**Turn 2 — Human**
```
full 6-DOF is fine
```

**Expected AI:** `status: draft`. 6-DOF Newton-Euler submarine.
States: `x, y, z, phi, theta, psi, u, v, w, p, q, r` (12 states).
Inputs: `u[0..3]` for 4 thrusters (or aggregated into body-frame forces/
moments). Hydrodynamic parameters (added mass, drag, buoyancy) labelled
as assumed. Keep code clean with comments per state block.

**Check:** `python_code` must be complete and runnable, not truncated.
If the model is too long for one response, it is acceptable to emit a
compact version with a note: "This is a compact 6-DOF model; say which
hydrodynamic coefficients you have measured and I will refine."

---

### Case XII — Discrete-time request

**Profile:** Explicitly wants a discrete-time model.

**Turn 1 — Human**
```
I need a discrete-time model of a double integrator with sampling time 0.1s
```

**Expected AI:** `status: draft`. Discrete dynamics using matrix
exponential or Euler approximation, clearly commented:
`# Discrete-time dynamics, Ts = 0.1 s`. States: `x[0]`, `x[1]`.
Return `x_next` instead of `dx`. Note in `reply` that this is a
zero-order-hold discretisation.

---

### Case XIII — Structure known, parameters unknown

**Profile:** Knows the physics and state space structure but has no numbers.

**Turn 1 — Human**
```
I want the Van der Pol oscillator but I don't know mu or the natural
frequency
```

**Expected AI:** `status: draft`. Van der Pol dynamics with `mu` and
`omega_n` as symbolic-like labelled parameters (e.g. `mu = 1.0  # —
tune to desired limit cycle amplitude`). Do NOT refuse to draft.
Do NOT ask "what is mu?" — draft with a sensible default and label it.

---

### Case XIV — Code-review user (pastes broken code)

**Profile:** Has existing code that is malformed or incomplete.

**Turn 1 — Human**
```
here's my attempt:
def dynamics(t,x,u):
    return [x[1], -9.81*sin(x[0])]
```
*(Missing imports, inconsistent return type, no input usage.)*

**Expected AI:** `status: draft` (or `complete` if they ask to clean it).
Fix the code: add `import numpy as np`, ensure `np.array` return,
comment states, add `u` parameter even if unused (`# u[0]: unused`),
label `g = 9.81`. Reply notes what was fixed.

---

### Case XV — Partial equations (some states given, others missing)

**Profile:** Gives some equations but not a complete dynamics vector.

**Turn 1 — Human**
```
my system has states x, y, theta. I know dx/dt = v*cos(theta) and
dy/dt = v*sin(theta) but I don't know dtheta/dt
```

**Expected AI:** `status: continue`. "You have a unicycle or car-like
model. Is `v` a state or an input? And does the heading rate come from
steering angle `phi` (bicycle model) or directly from an angular
velocity input `omega`?"

**Turn 2 — Human**
```
v is an input, and dtheta/dt = omega where omega is also an input
```

**Expected AI:** `status: draft`. Unicycle: `x, y, theta`. Inputs:
`u[0] = v`, `u[1] = omega`. Complete dynamics. Label any wheelbase or
other parameters if introduced.

---

### Case XVI — Multi-plant confusion (user mixes two systems)

**Turn 1 — Human**
```
I want a model of a quadcopter carrying a slung load on a pendulum
```

**Expected AI:** `status: continue`. "That is a coupled UAV-pendulum
system. Should I model the quadcopter as a point mass with thrust
vectoring, or as full 6-DOF? And is the pendulum attachment point at
the quadcopter CoM or offset?"

**Turn 2 — Human**
```
point mass is fine, attachment at CoM
```

**Expected AI:** `status: draft`. 5-state or 8-state model:
quad position (3), velocity (3), load angle (2), load angular rate (2).
Simplified to planar if appropriate, or full 3D with note. Parameters
labelled.

---

## Edge Cases & Stress Tests

| # | Scenario | Expected Behavior |
|---|----------|-------------------|
| 1 | User says `finish` before any draft exists | `status: continue`. "I haven't shown you a draft yet. What system are you modelling?" |
| 2 | User accepts draft with typo (`finsh`) | `_FINISH_RE` should NOT match. Treat as `continue` or ask for clarification. |
| 3 | User accepts draft with `looks good` after `continue` (no draft shown) | `_FINISH_RE` matches but `_latest_draft` is `None` → agent must not crash; treat as no-op `continue`. |
| 4 | Model emits invalid JSON | Agent appends `_REPAIR_NOTE`, re-invokes LLM once. If still invalid, shows raw text. |
| 5 | Model emits `complete` too early (before min turns) | Agent appends `_FORCE_DRAFT_NOTE`, re-invokes. If model insists on `complete`, accept to avoid infinite loop. |
| 6 | Max drafts reached | Agent auto-accepts latest draft. `final_result` set. UI shows success. |
| 7 | User resets conversation mid-draft | `reset_conversation_state()` clears `_draft_count` and `_latest_draft`. Fresh start. |
| 8 | Empty user message | Ignore or treat as `(empty)` — no crash. |
| 9 | User asks for same system twice in a row | If `final_result` is already set, chat is disabled (UI). If reset, normal flow. |
| 10 | Model `reply` is empty string in `continue` | Agent forces `_FORCE_DRAFT_NOTE` once. If still empty, shows raw text or "(empty model response)". |


---

## Other useful probes

| Input | What to check |
|-------|----------------|
| `who made you?` | continue only; no code |
| `DC motor` | continue or quick draft with labelled R, L, K, J, b assumptions |
| After a draft: `finish` / `done` / `looks good` / `ship it` | must set `final_result` |
