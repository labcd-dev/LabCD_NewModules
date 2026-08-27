def smc_stability_proof(has_delta, has_disturbance):
    if not (has_delta or has_disturbance):
        return (
"""**Stability proof -- Sliding Mode Control, ideal case (no active uncertainty/disturbance estimation)**

**What is being proved.** With no uncertainty and no disturbance, the closed-loop tracking error converges to zero -- in fact exponentially fast -- for any positive choice of the reaching-law gains, with no condition on those gains beyond simple positivity.

**Setup -- the plant, the sliding surface, and the control law.**

The plant is a square system (number of control inputs $u\\in\\mathbb R^p$ equals number of outputs $p$), and the *sliding surface* $s\\in\\mathbb R^p$ is a fixed linear combination of the tracking error and its derivatives (one such combination per output, built from that output's own relative degree), constructed so that forcing $s=0$ makes the remaining tracking error decay to zero on its own, with no further control action needed. Differentiating $s$ along the true plant gives the *surface dynamics*
$$\\dot s=\\beta(x)+B_s(x)u-v(y_d)+\\Xi(x,t)$$
where:
- $\\beta(x)$ is the **drift**: the part of $\\dot s$ that would remain even if the control input were zero.
- $B_s(x)$ is the **decoupling matrix**: how strongly each control input moves the surface. It is assumed invertible (needed below to solve for $u$).
- $v(y_d)$ is the **reference feedforward**: the part of $\\dot s$ contributed by the desired trajectory $y_d$ and its own derivatives -- known exactly, since the reference is chosen by the user, not measured from the plant.
- $\\Xi(x,t)$ is the **surface-level image of every unmodeled effect**: everything in the true plant not captured by the nominal model used to build $\\beta,B_s$.

**Step 1 -- the assumption that makes this the "ideal" case.**

*Assumption 1.* The plant matches the nominal model exactly: there is no unmodeled uncertainty ($\\Delta(x)\\equiv0$) and no external disturbance ($d(t)\\equiv0$), so $\\Xi(x,t)\\equiv0$ identically -- not merely small, but exactly zero for all $x,t$.

**Step 2 -- the reaching law, and the resulting closed-loop surface dynamics.**

Take the control law
$$u=D(x)^{-1}\\Big(v-\\beta-K\\,\\mathrm{sat}(s/\\Phi)-\\Lambda s\\Big),\\qquad D(x):=B_s(x)$$
where:
- $\\Lambda>0$ is the **reaching gain**: how fast the surface is pulled back toward zero.
- $K\\ge0$ is the **switching gain**: an extra push directed straight toward the surface.
- $\\Phi>0$ is the **boundary-layer width**: a small tolerance band, needed because an ideal discontinuous $\\mathrm{sign}(s)$ switching law would chatter infinitely fast right at $s=0$ in any real (non-idealized) implementation.
- $\\mathrm{sat}(\\cdot)$ is the **saturation function**, $\\mathrm{sat}(z)=z$ for $|z|\\le1$ and $\\mathrm{sat}(z)=\\mathrm{sign}(z)$ for $|z|>1$ -- so $\\mathrm{sat}(s_i/\\Phi)$ behaves exactly like $\\mathrm{sign}(s_i)$ once $|s_i|$ exceeds $\\Phi$, and like a plain linear gain $s_i/\\Phi$ inside the band: a smoothed version of "push as hard as possible toward the surface," capped so it does not chatter inside that band.

Substituting this control law into the surface dynamics of the Setup: the $+v-\\beta$ inside the control law cancels the $-\\beta$ and $-v$ already present; $B_s(x)\\cdot D(x)^{-1}=I$ (the identity matrix) since $D(x):=B_s(x)$, so the decoupling matrix cancels exactly; and, by Assumption 1, $\\Xi\\equiv0$ drops out entirely. What remains is **exactly**
$$\\boxed{\\ \\dot s=-\\Lambda s-K\\,\\mathrm{sat}(s/\\Phi)\\ }$$ (S1)
with no leftover term of any kind -- because, unlike every other system configuration this tool can design for, Assumption 1 leaves nothing unmodeled to produce one.

**Step 3 -- the Lyapunov function, and why it suffices to examine it channel by channel.**

Take
$$V=\\tfrac12s^\\top s=\\tfrac12\\sum_{i=1}^ps_i^2$$
the **Lyapunov function**: the squared "distance" of the current surface value from zero, summed over every output channel. By construction $V\\ge0$ always, and $V=0$ exactly when $s=0$ (every output exactly on its own sliding surface) -- these two properties (non-negative, zero only at the target) are what make $V$ usable as a measure of "how far from stable" the system currently is.

Differentiating channel by channel using (S1) applied componentwise, $\\dot s_i=-\\Lambda s_i-K\\,\\mathrm{sat}(s_i/\\Phi)$:
$$\\frac{d}{dt}\\Big(\\tfrac12s_i^2\\Big)=s_i\\dot s_i=-\\Lambda s_i^2-Ks_i\\,\\mathrm{sat}(s_i/\\Phi)$$
The term $-\\Lambda s_i^2$ is manifestly $\\le0$. The only term needing a closer look is $-Ks_i\\,\\mathrm{sat}(s_i/\\Phi)$; the claim is that it, too, is never positive. Check both regimes of the saturation function in turn:
- *Outside the boundary layer* ($|s_i|>\\Phi$): $\\mathrm{sat}(s_i/\\Phi)=\\mathrm{sign}(s_i)$, so $s_i\\,\\mathrm{sat}(s_i/\\Phi)=s_i\\,\\mathrm{sign}(s_i)=|s_i|\\ge0$.
- *Inside the boundary layer* ($|s_i|\\le\\Phi$): $\\mathrm{sat}(s_i/\\Phi)=s_i/\\Phi$, so $s_i\\,\\mathrm{sat}(s_i/\\Phi)=s_i^2/\\Phi\\ge0$.

So $s_i\\,\\mathrm{sat}(s_i/\\Phi)\\ge0$ in *both* regimes, with equality only at $s_i=0$; hence $-Ks_i\\,\\mathrm{sat}(s_i/\\Phi)\\le0$ always. Dropping this already-non-positive term (a valid weakening: removing something $\\le0$ from the right-hand side of an upper bound can only make the bound looser, never wrong) gives, for every channel and both regimes with no exception,
$$\\frac{d}{dt}\\Big(\\tfrac12s_i^2\\Big)\\le-\\Lambda s_i^2$$

**Step 4 -- summing across channels.**

Summing this bound over $i=1,\\dots,p$:
$$\\boxed{\\ \\dot V\\le-\\Lambda\\sum_{i=1}^ps_i^2=-2\\Lambda V\\ }$$ (S2)

**Step 5 -- solving this differential inequality: the Comparison Lemma.**

> **Comparison Lemma.** If a differentiable scalar function $V(t)\\ge0$ satisfies $\\dot V\\le-\\rho V+b$ for constants $\\rho>0$ and $b\\ge0$, then
> $$V(t)\\le V(0)e^{-\\rho t}+\\frac b\\rho\\big(1-e^{-\\rho t}\\big)\\qquad\\text{for all }t\\ge0$$
>
> *Proof.* Rewrite the hypothesis as $\\dot V+\\rho V\\le b$ and multiply both sides by the integrating factor $e^{\\rho t}>0$ (multiplying an inequality by a positive quantity preserves its direction):
> $$e^{\\rho t}\\dot V+\\rho e^{\\rho t}V\\le be^{\\rho t}$$
> By the product rule, the left-hand side is exactly $\\dfrac{d}{dt}\\big(e^{\\rho t}V\\big)$. Integrating both sides from $0$ to $t$:
> $$e^{\\rho t}V(t)-V(0)\\le\\frac b\\rho\\big(e^{\\rho t}-1\\big)$$
> Dividing through by $e^{\\rho t}>0$ gives the claim. $\\blacksquare$
>
> When $b=0$ this reduces to plain exponential decay, $V(t)\\le V(0)e^{-\\rho t}$. When $b>0$, $V(t)$ is eventually trapped arbitrarily close to, and never permanently above, the ball $V\\le b/\\rho$: it converges *to*, and stays confined *within*, that ball, rather than converging exactly to zero -- $b$ is what keeps the system from settling exactly at the origin, and $\\rho$ is how fast it gets pulled toward that ball.

Inequality (S2) is exactly this, with $\\rho=2\\Lambda$ and $b=0$ -- there is no residual constant at all here, because Assumption 1 removed every possible source of one. The Comparison Lemma therefore gives $V(t)\\le V(0)e^{-2\\Lambda t}$, and since $V=\\tfrac12\\|s\\|^2$,
$$\\|s(t)\\|\\le\\|s(0)\\|\\,e^{-\\Lambda t}$$

**Step 6 -- conclusion, in plain terms.**

The "distance" $V$ from the sliding surface shrinks at a guaranteed exponential rate $2\\Lambda$ *everywhere*, not merely once the surface is approached from outside the boundary layer: inside the layer the switching term does not vanish but instead becomes extra linear damping, since there $\\dot s=-\\Lambda s-\\tfrac K\\Phi s=-\\big(\\Lambda+\\tfrac K\\Phi\\big)s$ from (S1) with $\\mathrm{sat}(s/\\Phi)=s/\\Phi$. Since $s\\to0$ exponentially, and the surface was constructed (in the Setup) precisely so that $s=0$ forces the remaining tracking error to decay to zero on its own, the actual tracking error also converges to zero.

**Conclusion: global exponential stability**, for any reaching gain $\\Lambda>0$ and any switching gain $K\\ge0$ -- no gain condition beyond simple positivity, and no residual ball of any size, because Assumption 1 leaves nothing unmodeled for any residual to come from."""
        )

    return _smc_composite_proof(has_delta, has_disturbance)


# only hit when at least one of delta/disturbance is on. the both-off
# (pure ideal) case never reaches this, smc_stability_proof returns early for it
def _smc_composite_proof(has_delta, has_disturbance):
    if not has_delta:
        return (
"""**Stability proof -- Sliding Mode Control, disturbance observer only (no state-dependent uncertainty)**

**What is being proved.** With a bounded external disturbance $d(t)$ but no state-dependent model uncertainty, the tracking error, the disturbance-observer's internal predictor error, and the disturbance-observer's estimation error all converge to, and remain inside, a ball around zero -- a property called **semi-global Uniformly Ultimately Bounded (UUB)** stability. "Semi-global" means the result holds starting from any initial condition inside as large an operating region $\\Omega$ as one likes, by picking the gains appropriately for that region -- not literally from every conceivable starting point in the universe. "Ultimately bounded" means the errors are not driven exactly to zero (there is a genuinely non-zero disturbance to contend with), but are driven into, and thereafter kept inside, a ball whose radius is computed explicitly below.

**Setup -- the plant, the surface, and why $\\sigma=s$ exactly here.**

As in the ideal case, the sliding surface $s\\in\\mathbb R^p$ obeys
$$\\dot s=\\beta(x)+B_s(x)u-v(y_d)+\\Xi(x,t),\\qquad \\Xi(x,t):=J_s(x)\\,G(x,t)$$
where $\\beta,B_s,v$ have the same meaning as the ideal case (drift, decoupling matrix, reference feedforward), $J_s(x):=\\partial s/\\partial x$ is the **surface Jacobian** (how a small change in the state moves the surface), and $G(x,t)$ is the true, total mismatch between the plant and its nominal model. Here $G(x,t)=d(t)$ purely (an **Assumption**, stated next) -- no state-dependent part -- so $\\Xi(x,t)=J_s(x)\\,d(t)$: the disturbance's image at the surface level.

The controller can, in general, inject an *unmatched-compensation* correction $C(x)\\hat\\Delta$ into the surface it actually drives, $\\sigma:=s+C(x)\\hat\\Delta$, where $\\hat\\Delta$ is a state-dependent-uncertainty estimate (see the case where that estimate is active). Here there is no such estimate to inject ($\\hat\\Delta\\equiv0$ throughout, since nothing is trained to produce it), so $\\sigma=s$ **exactly** -- an identity used freely below.

**Step 1 -- the assumption.**

*Assumption 1.* The external disturbance $d(t)$ is bounded, and its surface-level image $J_s(x)d(t)$ does not change faster than a bound $\\bar w'$, i.e. $\\big\\|\\frac{d}{dt}\\big(J_sd\\big)\\big\\|\\le\\bar w'$. This bound is a *closed-loop* quantity, not a plant property alone: the control input $u$ depends on the disturbance estimate $\\hat D$ (defined in Step 2), and $u$ feeds back into the plant, so $\\bar w'$ implicitly depends on how the gains are chosen -- in particular a larger reaching gain $\\Lambda$ (defined below) amplifies $u$, which amplifies how fast $J_sd$ can effectively be forced to move. A fixed a priori bound is still sufficient for everything below; it is simply not a fact about the disturbance in isolation.

**Step 2 -- the disturbance observer: predictor, estimate, and estimation error.**

Introduce:
- $\\hat s$ -- an internally computed, predicted copy of the surface, run by the controller in parallel with the real one, defined by the update rule
  $$\\dot{\\hat s}=\\beta+B_su-v+\\hat D-\\kappa_se_D,\\qquad e_D:=\\hat s-s$$
  where $e_D$ is the **predictor error** (how far the internal prediction has drifted from the real, measured surface) and $\\kappa_s>0$ is the **predictor gain** (how fast the internal prediction is pulled back toward the real value).
- $\\hat D$ -- the **disturbance-observer estimate**: the controller's current best guess of the surface-level disturbance image $J_s(x)d(t)$.
- $\\tilde D:=\\hat D-J_sd$ -- the **disturbance estimation error**: how far the current guess is from the truth (unmeasurable directly, since $d(t)$ itself is not measured -- this is exactly what the observer is trying to drive to zero).

> **Lemma 1 (Predictor error dynamics).** $\\dot e_D=-\\kappa_se_D+\\tilde D$.
>
> *Proof.* By definition, $\\dot e_D=\\dot{\\hat s}-\\dot s$. Substituting the predictor's own definition for $\\dot{\\hat s}$ and the true surface dynamics of the Setup for $\\dot s$ (with $\\Xi=J_sd$ here):
> $$\\dot e_D=\\big(\\beta+B_su-v+\\hat D-\\kappa_se_D\\big)-\\big(\\beta+B_su-v+J_sd\\big)=\\hat D-J_sd-\\kappa_se_D=\\tilde D-\\kappa_se_D$$
> since the $\\beta+B_su-v$ terms are identical in both expressions and cancel exactly, and $\\hat D-J_sd=\\tilde D$ by the definition of $\\tilde D$ just above. $\\blacksquare$

This is the sense in which $e_D$ is "the training signal" for $\\hat D$: it is driven purely by the (unmeasurable) disturbance estimation error $\\tilde D$, decaying toward zero on its own at rate $\\kappa_s$ whenever $\\tilde D=0$.

**Step 3 -- the update law for $\\hat D$, and why it needs no derivative of any measured signal.**

Define the **drive** signal $\\mathrm{dr}:=k_2s-k_3e_D$ (a combination of the surface itself and the predictor error, with gains $k_2,k_3>0$), and update $\\hat D$ by
$$\\hat D:=\\zeta-k_4e_D,\\qquad \\dot\\zeta:=\\mathrm{dr}-k_4\\kappa_se_D$$
where $\\zeta$ is an auxiliary internal state (not itself a physical quantity) and $k_4>0$ is the **observer gain**.

> **Lemma 2 (Derivative-free observer identity).** With the update law just defined, $\\dot{\\hat D}=\\mathrm{dr}-k_4\\tilde D$ -- equivalently, $\\dot{\\tilde D}=\\mathrm{dr}-k_4\\tilde D-\\tfrac{d}{dt}(J_sd)$, using no time-derivative of any measured plant signal.
>
> *Proof.* Differentiating the definition of $\\hat D$: $\\dot{\\hat D}=\\dot\\zeta-k_4\\dot e_D$. Substitute $\\dot\\zeta=\\mathrm{dr}-k_4\\kappa_se_D$ and, from Lemma 1, $\\dot e_D=-\\kappa_se_D+\\tilde D$:
> $$\\dot{\\hat D}=\\big(\\mathrm{dr}-k_4\\kappa_se_D\\big)-k_4\\big(-\\kappa_se_D+\\tilde D\\big)=\\mathrm{dr}-k_4\\kappa_se_D+k_4\\kappa_se_D-k_4\\tilde D=\\mathrm{dr}-k_4\\tilde D$$
> the $\\mp k_4\\kappa_se_D$ terms cancelling exactly. Since $\\tilde D=\\hat D-J_sd$, $\\dot{\\tilde D}=\\dot{\\hat D}-\\tfrac{d}{dt}(J_sd)=\\mathrm{dr}-k_4\\tilde D-\\tfrac{d}{dt}(J_sd)$. Both formulas use only $e_D=\\hat s-s$ (available, since $\\hat s$ is computed internally and $s$ is measured) and $\\mathrm{dr}$ (built from $s,e_D$) -- never $\\dot s$ or $\\dot{\\hat s}$ directly. $\\blacksquare$
>
> *Why this matters.* The "obvious" way to write a disturbance observer that feeds back on its own error is $\\dot{\\hat D}=\\mathrm{dr}-k_4\\big(\\dot{\\hat s}-\\dot s+\\kappa_se_D\\big)$, using the identity $\\dot{\\hat s}-\\dot s+\\kappa_se_D=\\dot e_D+\\kappa_se_D=\\tilde D$ (from Lemma 1) to place proportional feedback on the otherwise-unmeasurable $\\tilde D$. Written that way, it appears to need $\\dot s$, which is not directly measured (only $s$ is). Lemma 2 shows this is avoidable: absorbing the $-k_4e_D$ term into the auxiliary state $\\zeta$ produces an *algebraically identical* update law that needs no derivative at all -- this is exactly the identity implemented in `AdaptiveSMC.compute_derivs` as `D_hat_dot = drive - k4*(s_hat_dot - s_dot_real + kappa_s*e_D)`, which computes $\\mathrm{dr}-k_4\\tilde D$ using only already-available quantities.

**Step 4 -- the control law, and the resulting closed-loop surface dynamics.**

Take
$$u=D(x)^{-1}\\Big(v-\\beta-\\hat D-\\Lambda s-K\\,\\mathrm{sat}(s/\\Phi)\\Big),\\qquad D(x):=B_s(x)$$
-- the same reaching law as the ideal case, but now also subtracting the current disturbance estimate $\\hat D$ before it can push the surface off zero (with $\\Lambda,K,\\Phi$ the same reaching gain, switching gain, and boundary-layer width as before). Substituting into the true surface dynamics of the Setup exactly as in Step 2 of the ideal case (the $v-\\beta$ and $B_sD(x)^{-1}$ terms cancel identically), but now $\\Xi=J_sd=\\hat D+\\tilde D-\\tilde D=\\hat D-(-\\tilde D)$... more directly, $\\Xi-\\hat D=J_sd-\\hat D=-\\tilde D$ by the definition of $\\tilde D$, so what remains is **exactly**
$$\\boxed{\\ \\dot s=-\\Lambda s-K\\,\\mathrm{sat}(s/\\Phi)-\\tilde D\\ }$$ (S3)
-- identical to the ideal case's (S1), except for one new term, $-\\tilde D$: the *only* channel through which the (unknown) disturbance still affects the surface, once the estimate $\\hat D$ has been subtracted off.

**Step 5 -- the Lyapunov function, and differentiating it block by block.**

Take
$$V=\\underbrace{\\tfrac{k_2}2\\|s\\|^2}_{\\text{surface block}}+\\underbrace{\\tfrac{k_3}2\\|e_D\\|^2}_{\\text{predictor block}}+\\underbrace{\\tfrac12\\|\\tilde D\\|^2}_{\\text{observer block}}$$
The weights $k_2$ on the surface block and $k_3$ on the predictor block are not free choices made for convenience -- they are *forced* to be exactly the same $k_2,k_3$ that appear in the drive $\\mathrm{dr}=k_2s-k_3e_D$, because that is what makes the cross terms below cancel exactly (shown next); any other weighting would leave an uncancelled term.

*Surface block.* Using (S3), $k_2s^\\top\\dot s=k_2s^\\top\\big(-\\Lambda s-K\\,\\mathrm{sat}(s/\\Phi)-\\tilde D\\big)=-k_2\\Lambda\\|s\\|^2-k_2Ks^\\top\\mathrm{sat}(s/\\Phi)-k_2s^\\top\\tilde D$.

*Predictor block.* Using Lemma 1, $k_3e_D^\\top\\dot e_D=k_3e_D^\\top\\big(-\\kappa_se_D+\\tilde D\\big)=-k_3\\kappa_s\\|e_D\\|^2+k_3e_D^\\top\\tilde D$.

*Observer block.* Using Lemma 2, $\\tilde D^\\top\\dot{\\tilde D}=\\tilde D^\\top\\Big(\\mathrm{dr}-k_4\\tilde D-\\tfrac{d}{dt}(J_sd)\\Big)=\\mathrm{dr}^\\top\\tilde D-k_4\\|\\tilde D\\|^2-\\tilde D^\\top\\tfrac{d}{dt}(J_sd)$. Substituting $\\mathrm{dr}=k_2s-k_3e_D$: $\\mathrm{dr}^\\top\\tilde D=k_2s^\\top\\tilde D-k_3e_D^\\top\\tilde D$.

**Step 6 -- the exact cancellation.**

Adding the three blocks together, the cross terms are, collecting them explicitly:
$$\\underbrace{-k_2s^\\top\\tilde D}_{\\text{surface block}}\\ +\\ \\underbrace{k_3e_D^\\top\\tilde D}_{\\text{predictor block}}\\ +\\ \\underbrace{k_2s^\\top\\tilde D-k_3e_D^\\top\\tilde D}_{\\text{observer block}}$$
The $-k_2s^\\top\\tilde D$ term cancels the $+k_2s^\\top\\tilde D$ term from the observer block; the $+k_3e_D^\\top\\tilde D$ term cancels the $-k_3e_D^\\top\\tilde D$ term from the observer block. **Every cross term is gone, identically** -- this exact cancellation, not an approximation, is precisely why the drive $\\mathrm{dr}$ must be shared between the predictor's own error and the surface, weighted to match $k_2,k_3$. What remains, summing the three blocks:
$$\\dot V=-k_2\\Lambda\\|s\\|^2-k_2Ks^\\top\\mathrm{sat}(s/\\Phi)-k_3\\kappa_s\\|e_D\\|^2-k_4\\|\\tilde D\\|^2-\\tilde D^\\top\\tfrac{d}{dt}(J_sd)$$ (S4)

**Step 7 -- closing the two remaining terms with Young's inequality.**

Two terms in (S4) are not already manifestly $\\le0$: the switching term $-k_2Ks^\\top\\mathrm{sat}(s/\\Phi)$, and the disturbance-rate term $-\\tilde D^\\top\\tfrac{d}{dt}(J_sd)$.

*The switching term.* Since $\\sigma=s$ exactly here (Setup), the same **Boundary-layer lemma** used throughout this proof family applies directly to $s$ itself:

> **Boundary-layer lemma.** For every component $i$, $s_i\\,\\mathrm{sat}(s_i/\\Phi)\\ge|s_i|-\\Phi/2$.
>
> *Proof.* If $|s_i|>\\Phi$: $\\mathrm{sat}(s_i/\\Phi)=\\mathrm{sign}(s_i)$, so the claim reads $|s_i|\\ge|s_i|-\\Phi/2$, true since $\\Phi>0$. If $|s_i|\\le\\Phi$: $\\mathrm{sat}(s_i/\\Phi)=s_i/\\Phi$, so the claim reads $s_i^2/\\Phi\\ge|s_i|-\\Phi/2$, i.e. $s_i^2-\\Phi|s_i|+\\Phi^2/2\\ge0$; viewed as a quadratic in $|s_i|\\ge0$, its discriminant is $\\Phi^2-4(1)(\\Phi^2/2)=-\\Phi^2<0$ and its leading coefficient is positive, so it is strictly positive everywhere on the real line, in particular $\\ge0$. $\\blacksquare$

Summing over the $p$ components, $s^\\top\\mathrm{sat}(s/\\Phi)\\ge\\|s\\|_1-p\\Phi/2$, so $-k_2Ks^\\top\\mathrm{sat}(s/\\Phi)\\le-k_2K\\|s\\|_1+\\tfrac{k_2Kp\\Phi}2\\le\\tfrac{k_2Kp\\Phi}2$ (dropping the remaining $-k_2K\\|s\\|_1\\le0$ term, a valid weakening).

*The disturbance-rate term.* Here Young's inequality is needed:

> **Young's inequality.** For real numbers (or vectors, via Cauchy-Schwarz) $a,b$ and any $c>0$: $ab\\le\\dfrac{a^2}{2c}+\\dfrac{cb^2}2$.
>
> *Proof.* $\\Big(\\dfrac a{\\sqrt c}-\\sqrt c\\,b\\Big)^2\\ge0$ (any real square is non-negative) expands to $\\dfrac{a^2}c-2ab+cb^2\\ge0$, i.e. $2ab\\le\\dfrac{a^2}c+cb^2$; dividing by $2$ gives the claim. $\\blacksquare$ The free parameter $c$ controls how a sign-indefinite cross term like $-ab$ is split between two different parts of a bound: larger $c$ shrinks the $a^2$ share at the cost of enlarging the $b^2$ share, or vice versa for smaller $c$.

Apply this with $a=\\|\\tilde D\\|$, $b=\\big\\|\\tfrac{d}{dt}(J_sd)\\big\\|\\le\\bar w'$ (Assumption 1) and a free parameter $c=1/\\theta$ for some chosen $\\theta\\in(0,2)$:
$$-\\tilde D^\\top\\tfrac{d}{dt}(J_sd)\\le\\|\\tilde D\\|\\Big\\|\\tfrac{d}{dt}(J_sd)\\Big\\|\\le\\frac\\theta2\\|\\tilde D\\|^2+\\frac1{2\\theta}\\bar w'^2$$

**Step 8 -- collecting the final inequality.**

Substituting both bounds into (S4):
$$\\dot V\\le-k_2\\Lambda\\|s\\|^2-k_3\\kappa_s\\|e_D\\|^2-k_4\\Big(1-\\tfrac\\theta2\\Big)\\|\\tilde D\\|^2+\\frac{\\bar w'^2}{2\\theta}+\\frac{k_2Kp\\Phi}2$$ (S5)
This is negative-definite in every block (i.e. every block's coefficient is a strictly negative multiple of that block, for $\\|s\\|,\\|e_D\\|,\\|\\tilde D\\|\\ne0$) for **any** positive $\\Lambda,\\kappa_s,k_4$ and any $\\theta\\in(0,2)$ -- there is no leakage-vs-coupling trade-off to solve for here, because with no network active there is nothing for the disturbance observer to drift against (contrast this with the cases where a state-dependent-uncertainty estimator is also active, below).

Writing each negative term as (a rate) $\\times$ (that block's own weight in $V$): $-k_2\\Lambda\\|s\\|^2=-2\\Lambda\\cdot\\big(\\tfrac{k_2}2\\|s\\|^2\\big)$, $-k_3\\kappa_s\\|e_D\\|^2=-2\\kappa_s\\cdot\\big(\\tfrac{k_3}2\\|e_D\\|^2\\big)$, $-k_4(1-\\theta/2)\\|\\tilde D\\|^2=-k_4(2-\\theta)\\cdot\\big(\\tfrac12\\|\\tilde D\\|^2\\big)$, and taking $\\rho$ as the smallest of these three rates and $b$ as the sum of the two residual constants,
$$\\rho:=\\min\\Big\\{2\\Lambda,\\ 2\\kappa_s,\\ k_4(2-\\theta)\\Big\\},\\qquad b:=\\frac{\\bar w'^2}{2\\theta}+\\frac{k_2Kp\\Phi}2$$
(S5) is exactly $\\dot V\\le-\\rho V+b$, the hypothesis of the **Comparison Lemma** stated and proved in the ideal-case proof of this same proof family:

> **Comparison Lemma.** If $\\dot V\\le-\\rho V+b$ with $\\rho>0,b\\ge0$, then $V(t)\\le V(0)e^{-\\rho t}+\\tfrac b\\rho\\big(1-e^{-\\rho t}\\big)$, so $V(t)$ converges to, and remains confined within, the ball $V\\le b/\\rho$.

Since $V\\ge\\tfrac{k_2}2\\|s\\|^2$, this also gives $\\limsup_{t\\to\\infty}\\|s\\|\\le\\sqrt{2b/(k_2\\rho)}$.

**Step 9 -- conclusion, in plain terms.**

$V$ decays at a guaranteed rate $\\rho$ toward, and then stays inside, a ball of size set by $b$ -- $s$, $e_D$, and $\\tilde D$ all converge to, and remain confined within, a ball around zero, exactly as the Comparison Lemma describes. This ball shrinks as the disturbance $J_sd$ varies more slowly (smaller $\\bar w'$) and as the reaching gain $\\Lambda$ is kept moderate (recall from Assumption 1 that a large $\\Lambda$ amplifies $u$, which amplifies how fast $J_sd$ effectively needs to be tracked, so keeping $\\Lambda$ moderate is what keeps $\\bar w'$ itself small in practice, not just what appears explicitly in $\\rho,b$). Setting $K=0$ removes the $\\tfrac{k_2Kp\\Phi}2$ term from $b$ entirely and gives the smallest guaranteed ball; $K>0$ is only useful as extra margin while the observer has not yet converged.

**Conclusion: semi-global Uniformly Ultimately Bounded (UUB) stability** of $s,e_D,\\tilde D$. Finally, note that the fixed-step (Euler) integration used by the simulator (`dt`) must stay below the step-size bound implied by this system's own Lipschitz constants for this continuous-time guarantee to carry over to the discrete-time simulation; this is a standard requirement of any Euler-integrated closed loop and is not itself part of the Lyapunov argument above."""
        )

    if not has_disturbance:
        return (
"""**Stability proof -- Sliding Mode Control, state-space neural-network identifier only (no external disturbance)**

**What is being proved.** With a state-dependent model uncertainty $\\Delta(x)$ but no external disturbance, the surface $s$, the internal state-predictor error $\\tilde x$, and the network's weight error $\\tilde W_\\Delta$ all converge to, and remain inside, a ball around zero -- semi-global Uniformly Ultimately Bounded (UUB) stability. Unlike the disturbance-observer-only case, the residual ball here never shrinks to exactly zero even with a very well-trained network, because (as shown in Step 1) the network's own leftover fitting error is never estimated by anything.

**Setup -- the plant, the surface, and the compensated surface $\\sigma$.**

The true plant is $\\dot x=f(x,u)+\\Delta(x)$ (no disturbance term here -- see Step 1), where $f(x,u)$ is the known nominal model and $\\Delta(x)$ is the unknown, state-dependent uncertainty. As in the other cases, differentiating the sliding surface $s\\in\\mathbb R^p$ along the true plant gives
$$\\dot s=\\beta(x)+B_s(x)u-v(y_d)+\\Xi(x,t),\\qquad \\Xi(x,t):=J_s(x)\\,\\Delta(x)$$
with $\\beta,B_s,v$ as before (drift, decoupling matrix, reference feedforward) and $J_s(x):=\\partial s/\\partial x$ the surface Jacobian.

Because $\\Delta(x)$ may enter the plant *before* the point where the control input acts (for outputs with relative degree $r_i\\ge2$), a network estimate built only at the surface level cannot fully cancel it from the actual tracking error, even though it could drive $s$ itself to a small ball (this point is developed further in the plain-language conclusion at the end). To address this, the controller estimates $\\Delta$ directly in the *state space* and injects the estimate upstream, ahead of the surface construction, via a **compensated surface**
$$\\sigma:=s+C(x)\\hat\\Delta$$
where $\\hat\\Delta$ is the state-space network estimate (defined in Step 2) and $C(x)$ is the fixed **unmatched-compensation matrix** (`compensation_func` in the code, built from the same derivatives used to construct $s$ itself) that maps a state-space correction into the surface's own coordinates. It is $\\sigma$, not $s$, that the reaching law below actually drives to zero.

**Step 1 -- the assumption.**

*Assumption 1.* The true uncertainty $\\Delta(x)$ can be approximated by a network with a *fixed* set of $N$ radial basis functions $\\phi:\\mathbb R^n\\to\\mathbb R^N$ (centers and widths chosen once, not adapted) and *some* constant "ideal" weight matrix $W^*\\in\\mathbb R^{N\\times n}$, to within a bounded leftover error: there exists $\\bar w_x\\ge0$, computable on the region $\\Omega$ the system actually operates in, such that
$$\\big\\|\\Delta(x)-W^{*\\top}\\phi(x)\\big\\|\\le\\bar w_x\\qquad\\text{for all }x\\in\\Omega$$
A network with a large enough, well-placed basis can make $\\bar w_x$ arbitrarily small, but with a *fixed, finite* basis it can never be driven to exactly zero -- this is why $\\bar w_x$ appears in the final residual below no matter how well the network is trained. Define the **fitting residual**
$$w_x(x):=\\Delta(x)-W^{*\\top}\\phi(x),\\qquad \\|w_x\\|\\le\\bar w_x$$
Also assume $0<\\phi_j(x)\\le1$ for each of the $N$ basis functions (standard for a normalized RBF basis), so $\\|\\phi(x)\\|\\le\\sqrt N=:\\bar\\mu$, and $\\|W^*\\|_F\\le\\bar W$ (Frobenius norm) for some fixed bound $\\bar W$.

**Step 2 -- the state-space network, its predictor, and the resulting error dynamics.**

Introduce:
- $\\hat\\Delta:=W_\\Delta^\\top\\phi(x)$ -- the network's *current* estimate of $\\Delta(x)$, using its *current* (adapting) weight matrix $W_\\Delta$, evaluated through the same fixed basis $\\phi$ from Assumption 1.
- $\\tilde W_\\Delta:=W_\\Delta-W^*$ -- the **weight error**: how far the current weights are from the (unknown, fixed) ideal weights $W^*$ of Assumption 1. Smaller $\\tilde W_\\Delta$ means a better estimate.
- $\\hat x$ -- an internally simulated copy of the *full state*, run by the controller in parallel with the real plant, updated by
  $$\\dot{\\hat x}:=f(x,u)+\\hat\\Delta+\\kappa\\tilde x,\\qquad \\tilde x:=x-\\hat x$$
  where $\\tilde x$ is the **state-prediction error** (how far the internal copy has drifted from the real, measured state) and $\\kappa>0$ is the **predictor gain**. This is a *different, separate* predictor from any surface-level one: it lives in the full state space precisely so that $C(x)\\hat\\Delta$ (Setup) can correct uncertainty regardless of where in the relative-degree chain it enters, which a surface-level-only estimate cannot do.

Combine $\\hat\\Delta-\\Delta=W_\\Delta^\\top\\phi-\\big(W^{*\\top}\\phi+w_x\\big)=\\tilde W_\\Delta^\\top\\phi-w_x$ (using Assumption 1's definition of $w_x$); define
$$\\psi_x:=\\tilde W_\\Delta^\\top\\phi-w_x=\\hat\\Delta-\\Delta$$
the **total state-space estimation error** -- the gap between what the network currently outputs and the true uncertainty, made up of a weight-error part and an unremovable fitting-residual part.

> **Lemma 1 (State-predictor error dynamics).** $\\dot{\\tilde x}=-\\kappa\\tilde x-\\psi_x$.
>
> *Proof.* By definition, $\\dot{\\tilde x}=\\dot x-\\dot{\\hat x}$. The true plant gives $\\dot x=f(x,u)+\\Delta(x)$; the predictor's own definition gives $\\dot{\\hat x}=f(x,u)+\\hat\\Delta+\\kappa\\tilde x$. Subtracting:
> $$\\dot{\\tilde x}=\\big(f(x,u)+\\Delta\\big)-\\big(f(x,u)+\\hat\\Delta+\\kappa\\tilde x\\big)=\\Delta-\\hat\\Delta-\\kappa\\tilde x=-\\psi_x-\\kappa\\tilde x$$
> since $f(x,u)$ cancels exactly and $\\Delta-\\hat\\Delta=-\\psi_x$ by the definition of $\\psi_x$ above. $\\blacksquare$

**Step 3 -- the network's adaptation law, and why its weight in $V$ is forced to exactly $\\tfrac12$.**

The network weights are updated by
$$\\dot W_\\Delta:=\\Gamma\\Big(\\phi\\,\\tilde x^\\top-\\sigma_WW_\\Delta\\Big)$$
where $\\Gamma>0$ is the **learning rate** and $\\sigma_W>0$ is the **leakage gain** (a small "forgetting" term, standard practice known as $\\sigma$-modification, that stops the weights from drifting to infinity when $\\Delta$ cannot be matched exactly). Note this law is driven *directly* by $\\tilde x$ -- not by a combination of gains chosen independently, the way the disturbance observer's drive is (contrast the disturbance-observer case) -- and this fixes how large a role $\\tilde x$'s own Lyapunov weight must play, shown next.

> **Lemma 2 (Exact weight-block identity).** With $\\tilde W_\\Delta:=W_\\Delta-W^*$ ($W^*$ constant, so $\\dot{\\tilde W}_\\Delta=\\dot W_\\Delta$),
> $$\\frac1\\Gamma\\,\\mathrm{tr}\\big(\\tilde W_\\Delta^\\top\\dot W_\\Delta\\big)=\\tilde x^\\top\\big(\\tilde W_\\Delta^\\top\\phi\\big)-\\sigma_W\\,\\mathrm{tr}\\big(\\tilde W_\\Delta^\\top W_\\Delta\\big)$$
>
> *Proof.* Substituting the adaptation law, $\\frac1\\Gamma\\dot W_\\Delta=\\phi\\,\\tilde x^\\top-\\sigma_WW_\\Delta$, so $\\frac1\\Gamma\\mathrm{tr}\\big(\\tilde W_\\Delta^\\top\\dot W_\\Delta\\big)=\\mathrm{tr}\\big(\\tilde W_\\Delta^\\top\\phi\\,\\tilde x^\\top\\big)-\\sigma_W\\,\\mathrm{tr}\\big(\\tilde W_\\Delta^\\top W_\\Delta\\big)$. For the first term, use the **cyclic property of the trace** -- for any matrices $A,B$ of compatible size, $\\mathrm{tr}(AB)=\\mathrm{tr}(BA)$, which holds because both sides equal $\\sum_{i,j}A_{ij}B_{ji}$ by direct computation of the diagonal entries of $AB$ (or $BA$) and summing. Applying it with $A=\\tilde W_\\Delta^\\top\\phi$ (an $n\\times1$ column) and $B=\\tilde x^\\top$ (a $1\\times n$ row): $\\mathrm{tr}\\big(\\tilde W_\\Delta^\\top\\phi\\,\\tilde x^\\top\\big)=\\mathrm{tr}\\big(\\tilde x^\\top\\tilde W_\\Delta^\\top\\phi\\big)=\\tilde x^\\top\\tilde W_\\Delta^\\top\\phi$ (a $1\\times1$ scalar, whose trace is itself). $\\blacksquare$

**Step 4 -- the closed-loop surface dynamics.**

Take the control law
$$u=D(x)^{-1}\\Big(v-\\beta-\\hat\\Delta_s-K\\,\\mathrm{sat}(\\sigma/\\Phi)-\\Lambda\\sigma\\Big),\\qquad D(x):=B_s(x),\\qquad \\hat\\Delta_s:=J_s(x)\\hat\\Delta$$
-- the reaching law of the earlier cases, but now driving the *compensated* surface $\\sigma$ (Setup) rather than $s$ directly, and subtracting $\\hat\\Delta_s$, the network's estimate *projected onto the surface* via $J_s$. Substituting into the surface dynamics of the Setup (the $v-\\beta$ and $B_sD(x)^{-1}$ terms cancel exactly as before) and using $\\Xi-\\hat\\Delta_s=J_s\\Delta-J_s\\hat\\Delta=-J_s\\psi_x$ (Step 2's definition of $\\psi_x$):
$$\\dot s=-\\Lambda\\sigma-K\\,\\mathrm{sat}(\\sigma/\\Phi)-J_s\\psi_x$$
Since $\\sigma=s+C\\hat\\Delta$ (Setup), $-\\Lambda\\sigma=-\\Lambda s-\\Lambda C\\hat\\Delta$, so this is **exactly**
$$\\boxed{\\ \\dot s=-\\Lambda s-J_s\\psi_x-\\Lambda C\\hat\\Delta-K\\,\\mathrm{sat}(\\sigma/\\Phi)\\ }$$ (B1)
Two of these four terms are new compared to the ideal case's (S1): $-J_s\\psi_x$ (the network's estimation error, projected onto the surface through the surface Jacobian) and $-\\Lambda C\\hat\\Delta$ (an algebraic side effect of driving the *compensated* surface $\\sigma$ rather than $s$ itself).

**Step 5 -- the Lyapunov function.**

Take
$$V=\\underbrace{\\tfrac{k_2}2\\|s\\|^2}_{\\text{surface block}}+\\underbrace{\\tfrac12\\|\\tilde x\\|^2}_{\\text{state-predictor block}}+\\underbrace{\\tfrac1{2\\Gamma}\\|\\tilde W_\\Delta\\|_F^2}_{\\text{weight block}}$$
where $k_2>0$ is a free weight on the surface block, but the weight $\\tfrac12$ on $\\|\\tilde x\\|^2$ is **not** free -- it is the unique value for which the cross term between the state-predictor block and the weight block cancels exactly, shown next (contrast the surface/predictor blocks of the disturbance-observer case, whose weights $k_2,k_3$ are likewise forced, but by matching the drive's own gains rather than by this unit-coefficient requirement).

*State-predictor block.* Using Lemma 1, $\\tilde x^\\top\\dot{\\tilde x}=\\tilde x^\\top\\big(-\\kappa\\tilde x-\\psi_x\\big)=-\\kappa\\|\\tilde x\\|^2-\\tilde x^\\top\\psi_x$.

*Weight block.* Using Lemma 2 and $\\tilde W_\\Delta^\\top\\phi=\\psi_x+w_x$ (rearranging Step 2's definition $\\psi_x=\\tilde W_\\Delta^\\top\\phi-w_x$):
$$\\frac1\\Gamma\\mathrm{tr}\\big(\\tilde W_\\Delta^\\top\\dot W_\\Delta\\big)=\\tilde x^\\top\\big(\\psi_x+w_x\\big)-\\sigma_W\\,\\mathrm{tr}\\big(\\tilde W_\\Delta^\\top W_\\Delta\\big)=\\tilde x^\\top\\psi_x+\\tilde x^\\top w_x-\\sigma_W\\,\\mathrm{tr}\\big(\\tilde W_\\Delta^\\top W_\\Delta\\big)$$

**Step 6 -- the exact cancellation, and what is left over.**

Adding the state-predictor block and the weight block: the $-\\tilde x^\\top\\psi_x$ term (predictor block) and $+\\tilde x^\\top\\psi_x$ term (weight block) **cancel identically**. What remains from these two blocks:
$$-\\kappa\\|\\tilde x\\|^2+\\tilde x^\\top w_x-\\sigma_W\\,\\mathrm{tr}\\big(\\tilde W_\\Delta^\\top W_\\Delta\\big)$$
Adding the surface block (from (B1), $k_2s^\\top\\dot s=-k_2\\Lambda\\|s\\|^2-k_2s^\\top J_s\\psi_x-k_2\\Lambda s^\\top C\\hat\\Delta-k_2Ks^\\top\\mathrm{sat}(\\sigma/\\Phi)$), the full sum is
$$\\dot V=-k_2\\Lambda\\|s\\|^2-\\kappa\\|\\tilde x\\|^2-\\sigma_W\\,\\mathrm{tr}\\big(\\tilde W_\\Delta^\\top W_\\Delta\\big)-k_2s^\\top J_s\\psi_x+\\tilde x^\\top w_x-k_2\\Lambda s^\\top C\\hat\\Delta-k_2Ks^\\top\\mathrm{sat}(\\sigma/\\Phi)$$ (B2)
Unlike the disturbance-observer case, **the term $-k_2s^\\top J_s\\psi_x$ has no partner left to cancel it against**: $\\psi_x$ already used its one "cancellation opportunity" against the state-predictor block above (that is a hard structural fact -- $\\psi_x$ is generated once, by the state-space identifier, and there is no second copy of it available to also cancel its appearance in the surface block). This term, along with $\\tilde x^\\top w_x$, $-k_2\\Lambda s^\\top C\\hat\\Delta$, and $-k_2Ks^\\top\\mathrm{sat}(\\sigma/\\Phi)$, must instead be closed with Young's inequality.

**Step 7 -- closing the leftover terms.**

> **Young's inequality.** For real numbers (or vectors, via Cauchy-Schwarz) $a,b$ and any $c>0$: $ab\\le\\dfrac{a^2}{2c}+\\dfrac{cb^2}2$.
>
> *Proof.* $\\Big(\\dfrac a{\\sqrt c}-\\sqrt c\\,b\\Big)^2\\ge0$ expands to $\\dfrac{a^2}c-2ab+cb^2\\ge0$, i.e. $2ab\\le\\dfrac{a^2}c+cb^2$; dividing by $2$ gives the claim. $\\blacksquare$ The free parameter $c$ (written $\\theta_1,\\theta_2,\\dots$ below, one per application) decides how a cross term is split between two blocks of the bound.

Let $\\bar J_s:=\\sup_\\Omega\\|J_s(x)\\|$ and $\\bar C:=\\sup_\\Omega\\|C(x)\\|$ (fixed bounds on the operating region $\\Omega$, computable from the plant and the reference).

*Leakage term.* $-k_2s^\\top J_s\\psi_x=-k_2s^\\top J_s\\tilde W_\\Delta^\\top\\phi+k_2s^\\top J_sw_x$. Bound each piece:
$$-k_2s^\\top J_s\\tilde W_\\Delta^\\top\\phi\\le k_2\\bar J_s\\bar\\mu\\,\\|s\\|\\,\\|\\tilde W_\\Delta\\|_F\\overset{\\text{Young, }c=\\theta_1}{\\le}\\frac{\\theta_1}2\\|s\\|^2+\\frac{k_2^2\\bar J_s^2\\bar\\mu^2}{2\\theta_1}\\|\\tilde W_\\Delta\\|_F^2$$
$$k_2s^\\top J_sw_x\\le k_2\\bar J_s\\bar w_x\\,\\|s\\|\\overset{\\text{Young, }c=\\theta_2}{\\le}\\frac{\\theta_2}2\\|s\\|^2+\\frac{k_2^2\\bar J_s^2\\bar w_x^2}{2\\theta_2}$$
(using $\\|\\phi\\|\\le\\bar\\mu$ from Assumption 1, and $\\|w_x\\|\\le\\bar w_x$).

*State-residual term.* $\\tilde x^\\top w_x\\le\\|\\tilde x\\|\\,\\bar w_x\\overset{\\text{Young, }c=\\theta_4}{\\le}\\dfrac{\\theta_4}2\\|\\tilde x\\|^2+\\dfrac{\\bar w_x^2}{2\\theta_4}$.

*Unmatched-injection term.* Since $\\hat\\Delta=\\tilde W_\\Delta^\\top\\phi+W^{*\\top}\\phi$, $\\|\\hat\\Delta\\|\\le\\bar\\mu\\big(\\|\\tilde W_\\Delta\\|_F+\\bar W\\big)$; using $(\\|\\tilde W_\\Delta\\|_F+\\bar W)^2\\le2\\|\\tilde W_\\Delta\\|_F^2+2\\bar W^2$ (a standard consequence of $(a+b)^2\\le2a^2+2b^2$, itself Young's inequality with $c=1$):
$$-k_2\\Lambda s^\\top C\\hat\\Delta\\le k_2\\Lambda\\bar C\\bar\\mu\\,\\|s\\|\\big(\\|\\tilde W_\\Delta\\|_F+\\bar W\\big)\\overset{\\text{Young, }c=\\theta_2}{\\le}\\frac{\\theta_2}2\\|s\\|^2+\\frac{(k_2\\Lambda\\bar C\\bar\\mu)^2}{\\theta_2}\\|\\tilde W_\\Delta\\|_F^2+\\frac{(k_2\\Lambda\\bar C\\bar\\mu\\bar W)^2}{\\theta_2}$$
(reusing the free parameter $\\theta_2$ from the leakage term above, since both bounds load the same $\\|s\\|^2$ and $\\|\\tilde W_\\Delta\\|_F^2$ blocks).

*Switching term.* Because $\\sigma\\ne s$ here (Setup), the tight **Boundary-layer lemma** used in the disturbance-observer-only case (which bounds $s_i\\,\\mathrm{sat}(s_i/\\Phi)$ against *its own argument* $s_i$) does not apply directly to $s^\\top\\mathrm{sat}(\\sigma/\\Phi)$, whose two factors involve *different* quantities. Instead use the elementary facts $|\\mathrm{sat}(z)|\\le1$ for every $z$ (the saturation function never exceeds $\\pm1$, by its own definition) and the Cauchy-Schwarz inequality $|a^\\top b|\\le\\|a\\|\\|b\\|$:
$$-k_2Ks^\\top\\mathrm{sat}(\\sigma/\\Phi)\\le k_2K\\|s\\|\\,\\|\\mathrm{sat}(\\sigma/\\Phi)\\|\\le k_2K\\sqrt p\\,\\|s\\|\\overset{\\text{Young, }c=\\theta_3}{\\le}\\frac{\\theta_3}2\\|s\\|^2+\\frac{k_2^2K^2p}{2\\theta_3}$$
(using $\\|\\mathrm{sat}(\\sigma/\\Phi)\\|\\le\\sqrt p$, since each of its $p$ components has magnitude at most $1$). This is a looser bound than the tight boundary-layer lemma gives when $\\sigma=s$; **consequently $K=0$ is the recommended default whenever the estimator is trusted**, since it removes this term's contribution to the residual entirely, at no cost (see the plain-language conclusion below).

*Leakage.* As in every other composite case, $-\\sigma_W\\,\\mathrm{tr}\\big(\\tilde W_\\Delta^\\top W_\\Delta\\big)$ is bounded using $W_\\Delta=\\tilde W_\\Delta+W^*$:
$$-\\sigma_W\\,\\mathrm{tr}\\big(\\tilde W_\\Delta^\\top W_\\Delta\\big)=-\\sigma_W\\|\\tilde W_\\Delta\\|_F^2-\\sigma_W\\,\\mathrm{tr}\\big(\\tilde W_\\Delta^\\top W^*\\big)\\le-\\frac{\\sigma_W}2\\|\\tilde W_\\Delta\\|_F^2+\\frac{\\sigma_W}2\\|W^*\\|_F^2$$
(the cross term bounded via $\\mathrm{tr}(\\tilde W_\\Delta^\\top W^*)\\ge-\\|\\tilde W_\\Delta\\|_F\\|W^*\\|_F$, itself Young's inequality with $a=\\|\\tilde W_\\Delta\\|_F,b=\\|W^*\\|_F,c=1$).

**Step 8 -- collecting the final inequality.**

Substituting every bound from Step 7 into (B2):
$$\\dot V\\le-q_s\\|s\\|^2-q_x\\|\\tilde x\\|^2-q_W\\|\\tilde W_\\Delta\\|_F^2+b$$ (B3)
$$q_s=k_2\\Lambda-\\frac{\\theta_1+\\theta_2+\\theta_3}2,\\qquad q_x=\\kappa-\\theta_4,\\qquad q_W=\\frac{\\sigma_W}2-\\frac{k_2^2\\bar J_s^2\\bar\\mu^2}{2\\theta_1}-\\frac{(k_2\\Lambda\\bar C\\bar\\mu)^2}{\\theta_2}$$
$$b=\\frac{k_2^2\\bar J_s^2\\bar w_x^2}{2\\theta_2}+\\frac{(k_2\\Lambda\\bar C\\bar\\mu\\bar W)^2}{\\theta_2}+\\frac{k_2^2K^2p}{2\\theta_3}+\\frac{\\bar w_x^2}{2\\theta_4}+\\frac{\\sigma_W}2\\|W^*\\|_F^2$$
(S5)-style, this is negative-definite in every block -- $q_s,q_x,q_W>0$ -- for **any** choice of free parameters $\\theta_1,\\theta_2,\\theta_3,\\theta_4>0$ satisfying $2k_2\\Lambda>\\theta_1+\\theta_2+\\theta_3$, $\\kappa>\\theta_4$, and $\\sigma_W$ large enough to dominate the two coupling loads inside $q_W$. This last condition is the **leakage-vs-coupling trade-off**: the leakage gain $\\sigma_W$ (which keeps the weights from drifting) must be large enough to outweigh *two* coupling paths at once -- the surface-Jacobian leakage $J_s\\psi_x$ and the unmatched-injection term $C\\hat\\Delta$ -- rather than just one, as in a case with no unmatched-compensation mechanism. Since $\\bar\\mu$ enters both coupling terms quadratically, narrowing the RBF basis (fewer, more localized nodes, hence a smaller $\\bar\\mu=\\sqrt N$) is the cheapest way to relax this condition.

Write each negative term in (B3) as (a rate) $\\times$ (its own block's weight in $V$): $-q_s\\|s\\|^2=-\\tfrac{2q_s}{k_2}\\big(\\tfrac{k_2}2\\|s\\|^2\\big)$, $-q_x\\|\\tilde x\\|^2=-2q_x\\big(\\tfrac12\\|\\tilde x\\|^2\\big)$, $-q_W\\|\\tilde W_\\Delta\\|_F^2=-2\\Gamma q_W\\big(\\tfrac1{2\\Gamma}\\|\\tilde W_\\Delta\\|_F^2\\big)$, and set
$$\\rho:=\\min\\Big\\{\\frac{2q_s}{k_2},\\ 2q_x,\\ 2\\Gamma q_W\\Big\\}$$
so that (B3) is exactly $\\dot V\\le-\\rho V+b$.

**Step 9 -- the Comparison Lemma, and the final bound.**

> **Comparison Lemma.** If $\\dot V\\le-\\rho V+b$ with $\\rho>0,b\\ge0$, then $V(t)\\le V(0)e^{-\\rho t}+\\tfrac b\\rho\\big(1-e^{-\\rho t}\\big)$, so $V(t)$ converges to, and remains confined within, the ball $V\\le b/\\rho$. (*Proof:* multiply $\\dot V+\\rho V\\le b$ by the integrating factor $e^{\\rho t}$, recognize the left side as $\\frac{d}{dt}(e^{\\rho t}V)$, and integrate from $0$ to $t$.)

Since $V\\ge\\tfrac{k_2}2\\|s\\|^2$, this gives $\\limsup_{t\\to\\infty}\\|s\\|\\le\\sqrt{2b/(k_2\\rho)}$, and analogous bounds for $\\|\\tilde x\\|,\\|\\tilde W_\\Delta\\|_F$ from their own weights in $V$.

**Step 10 -- conclusion, in plain terms.**

Two differences from the disturbance-observer-only case are worth stating explicitly. First, the residual $b$ here does **not** vanish even with a perfectly-tuned network, because the fitting residual $\\bar w_x$ (Assumption 1) is never driven to zero by any observer -- there is no state-space equivalent of the disturbance observer's $\\hat D$ chasing it down; it is only ever bounded, not eliminated. Second, because the switching term had to be closed with the looser Cauchy-Schwarz bound rather than the tight boundary-layer lemma (Step 7), **$K=0$ gives the smallest guaranteed residual** and is the recommended default once the network is trusted to have converged; $K>0$ only provides extra margin while it has not.

**Conclusion: semi-global Uniformly Ultimately Bounded (UUB) stability** of $s,\\tilde x,\\tilde W_\\Delta$ -- they converge to, and remain confined within, a ball around zero, per the Comparison Lemma, shrinking as the network's basis is chosen to fit $\\Delta(x)$ more tightly (smaller $\\bar w_x,\\bar\\mu$) and as the gains satisfy the margin conditions of Step 8 with room to spare. As always, the fixed-step (Euler) integration used by the simulator (`dt`) must stay below the step-size bound implied by this system's own Lipschitz constants for this continuous-time guarantee to carry over to the discrete-time simulation."""
        )
    return (
"""**Stability proof -- Sliding Mode Control, both estimators active (state-space NN identifier and surface-level disturbance observer)**

**What is being proved.** With both a state-dependent model uncertainty $\\Delta(x)$ and an external disturbance $d(t)$, all five error quantities this controller tracks -- the surface $s$, the disturbance-predictor error $e_D$, the disturbance-estimation error $\\tilde D$, the state-predictor error $\\tilde x$, and the network's weight error $\\tilde W_\\Delta$ -- converge to, and remain inside, a ball around zero (semi-global UUB). This case combines the mechanisms of the two single-estimator cases; rather than repeat every definition, this proof states what changes and re-derives the two places where the two mechanisms interact.

**Setup -- combining both mechanisms.**

The true plant is $\\dot x=f(x,u)+\\Delta(x)+d(t)$. As in the disturbance-observer-only case, the sliding surface obeys $\\dot s=\\beta+B_su-v+\\Xi$ with $\\Xi=J_s(\\Delta+d)$; as in the state-space-identifier-only case, the controller injects a compensated surface $\\sigma:=s+C(x)\\hat\\Delta$, with $\\hat\\Delta=W_\\Delta^\\top\\phi(x)$ the state-space network estimate. In addition to that state-space predictor $\\hat x$, the controller now also runs a *second, separate* surface-level predictor $\\hat s$ feeding a disturbance observer $\\hat D$, exactly as in the disturbance-observer-only case.

**Step 1 -- the assumptions (both are needed simultaneously).**

*Assumption 1.* $\\Delta(x)$ can be approximated by a fixed-basis network to within a bounded fitting residual: $\\|\\Delta(x)-W^{*\\top}\\phi(x)\\|\\le\\bar w_x$ on $\\Omega$, with $\\|\\phi\\|\\le\\bar\\mu=\\sqrt N$ and $\\|W^*\\|_F\\le\\bar W$ (same $W^*,\\phi,\\bar w_x,\\bar\\mu,\\bar W$ as the state-space-identifier-only case).

*Assumption 2.* The external disturbance is bounded, $\\|d(t)\\|\\le\\bar d$, and its surface-level image does not change faster than a (closed-loop) bound $\\bar w'$: $\\big\\|\\frac{d}{dt}(J_sd)\\big\\|\\le\\bar w'$ (same $\\bar d,\\bar w'$ as the disturbance-observer-only case, plus the new plain magnitude bound $\\bar d$, needed because -- see Step 2 -- the raw disturbance now also perturbs the state predictor directly).

**Step 2 -- the state predictor now also sees the raw disturbance.**

The state predictor is defined exactly as in the state-space-identifier-only case, $\\dot{\\hat x}:=f(x,u)+\\hat\\Delta+\\kappa\\tilde x$ -- note it does **not** include the disturbance estimate $\\hat D$ at all (by construction: `AdaptiveSMC`'s `x_hat_dot` never adds `D_hat`, since $\\hat D$ is reserved for the surface-level channel only). With $\\tilde x:=x-\\hat x$ and the true dynamics now including $d(t)$:
$$\\dot{\\tilde x}=\\dot x-\\dot{\\hat x}=\\big(f(x,u)+\\Delta+d\\big)-\\big(f(x,u)+\\hat\\Delta+\\kappa\\tilde x\\big)=-\\kappa\\tilde x-\\psi_x+d$$
using $\\Delta-\\hat\\Delta=-\\psi_x$ exactly as in Lemma 1 of the state-space-identifier-only case ($\\psi_x:=\\tilde W_\\Delta^\\top\\phi-w_x$, the total state-space estimation error). The only change from that case is the extra $+d$ term: the state predictor is now perturbed not only by the network's own estimation error, but also directly by the (unestimated, at this level) external disturbance.

**Step 3 -- the surface-level predictor now also leaks in the network's error.**

The surface-level predictor is defined as in the disturbance-observer-only case, but its nominal model now includes the network's own surface-level projection $J_s\\hat\\Delta$ (since that part of the mismatch is already being explained by the state-space identifier):
$$\\dot{\\hat s}:=\\beta+B_su-v+J_s\\hat\\Delta+\\hat D-\\kappa_se_D,\\qquad e_D:=\\hat s-s$$
Using the true surface dynamics $\\dot s=\\beta+B_su-v+J_s(\\Delta+d)$ and $\\tilde D:=\\hat D-J_sd$:
$$\\dot e_D=\\dot{\\hat s}-\\dot s=J_s\\hat\\Delta+\\hat D-J_s(\\Delta+d)-\\kappa_se_D=-J_s(\\Delta-\\hat\\Delta)+\\big(\\hat D-J_sd\\big)-\\kappa_se_D=J_s\\psi_x+\\tilde D-\\kappa_se_D$$
i.e. **$\\dot e_D=-\\kappa_se_D+J_s\\psi_x+\\tilde D$** -- identical in form to Lemma 1 of the disturbance-observer-only case ($\\dot e_D=-\\kappa_se_D+\\tilde D$), but with one extra forcing term, $J_s\\psi_x$: the state-space network's own estimation error, leaking into the surface-level predictor through $J_s\\hat\\Delta$.

With the drive $\\mathrm{dr}:=k_2s-k_3e_D$ and the same derivative-free update law as the disturbance-observer-only case, $\\hat D:=\\zeta-k_4e_D,\\ \\dot\\zeta:=\\mathrm{dr}-k_4\\kappa_se_D$, the identical algebra of that case's Lemma 2 (only the substitution of $\\dot e_D$ changes) gives
$$\\dot{\\hat D}=\\dot\\zeta-k_4\\dot e_D=\\big(\\mathrm{dr}-k_4\\kappa_se_D\\big)-k_4\\big(-\\kappa_se_D+J_s\\psi_x+\\tilde D\\big)=\\mathrm{dr}-k_4\\big(J_s\\psi_x+\\tilde D\\big)$$
so **$\\dot{\\tilde D}=\\mathrm{dr}-k_4\\big(J_s\\psi_x+\\tilde D\\big)-\\tfrac{d}{dt}(J_sd)$** -- again identical in form to before, with the same extra $J_s\\psi_x$ term.

**Step 4 -- the closed-loop surface dynamics.**

Exactly as (B1) in the state-space-identifier-only case, but now also carrying $-\\tilde D$ (the disturbance-observer's own error, since the control law also subtracts $\\hat D$ before it can act):
$$\\boxed{\\ \\dot s=-\\Lambda s-J_s\\psi_x-\\tilde D-\\Lambda C\\hat\\Delta-K\\,\\mathrm{sat}(\\sigma/\\Phi)\\ }$$ (B4)

**Step 5 -- the Lyapunov function and its five blocks.**

$$V=\\underbrace{\\tfrac{k_2}2\\|s\\|^2}_{\\text{surface}}+\\underbrace{\\tfrac{k_3}2\\|e_D\\|^2}_{\\text{disturbance predictor}}+\\underbrace{\\tfrac12\\|\\tilde D\\|^2}_{\\text{disturbance observer}}+\\underbrace{\\tfrac12\\|\\tilde x\\|^2}_{\\text{state predictor}}+\\underbrace{\\tfrac1{2\\Gamma}\\|\\tilde W_\\Delta\\|_F^2}_{\\text{network weights}}$$

Differentiating each block using Steps 2--4 (the same computations as the two single-estimator cases, just with the extra leakage terms carried along):

*Surface block:* $k_2s^\\top\\dot s=-k_2\\Lambda\\|s\\|^2-k_2s^\\top J_s\\psi_x-k_2s^\\top\\tilde D-k_2\\Lambda s^\\top C\\hat\\Delta-k_2Ks^\\top\\mathrm{sat}(\\sigma/\\Phi)$.

*Disturbance-predictor block:* $k_3e_D^\\top\\dot e_D=-k_3\\kappa_s\\|e_D\\|^2+k_3e_D^\\top J_s\\psi_x+k_3e_D^\\top\\tilde D$.

*Disturbance-observer block:* $\\tilde D^\\top\\dot{\\tilde D}=\\mathrm{dr}^\\top\\tilde D-k_4\\tilde D^\\top J_s\\psi_x-k_4\\|\\tilde D\\|^2-\\tilde D^\\top\\tfrac{d}{dt}(J_sd)$, and $\\mathrm{dr}^\\top\\tilde D=(k_2s-k_3e_D)^\\top\\tilde D=k_2s^\\top\\tilde D-k_3e_D^\\top\\tilde D$.

*State-predictor block:* $\\tilde x^\\top\\dot{\\tilde x}=-\\kappa\\|\\tilde x\\|^2-\\tilde x^\\top\\psi_x+\\tilde x^\\top d$.

*Weight block:* using the same Lemma (cyclic-trace identity) as the state-space-identifier-only case, $\\tfrac1\\Gamma\\mathrm{tr}(\\tilde W_\\Delta^\\top\\dot W_\\Delta)=\\tilde x^\\top\\psi_x+\\tilde x^\\top w_x-\\sigma_W\\,\\mathrm{tr}(\\tilde W_\\Delta^\\top W_\\Delta)$.

**Step 6 -- the two exact cancellations.**

*First cancellation (surface / disturbance-predictor / disturbance-observer).* Collecting the $\\tilde D$-cross terms from the three blocks above: $-k_2s^\\top\\tilde D$ (surface) $+\\,k_3e_D^\\top\\tilde D$ (disturbance predictor) $+\\,k_2s^\\top\\tilde D-k_3e_D^\\top\\tilde D$ (disturbance observer) $=0$ -- exactly the same cancellation as the disturbance-observer-only case, **unaffected** by the presence of the network (its leakage terms are separate and handled next).

*Second cancellation (state predictor / network weights).* Collecting the $\\psi_x$-cross terms from those two blocks: $-\\tilde x^\\top\\psi_x+\\tilde x^\\top\\psi_x=0$ -- exactly the same cancellation as the state-space-identifier-only case, unaffected by the disturbance observer.

What is left, after both cancellations, is
$$\\dot V=-k_2\\Lambda\\|s\\|^2-k_3\\kappa_s\\|e_D\\|^2-k_4\\|\\tilde D\\|^2-\\kappa\\|\\tilde x\\|^2-\\sigma_W\\,\\mathrm{tr}\\big(\\tilde W_\\Delta^\\top W_\\Delta\\big)$$
$$+\\ \\underbrace{\\big(-k_2s+k_3e_D-k_4\\tilde D\\big)^\\top J_s\\psi_x}_{\\text{new: leaks into all three of }s,e_D,\\tilde D}\\ +\\ \\tilde x^\\top d\\ +\\ \\tilde x^\\top w_x\\ -\\ k_2\\Lambda s^\\top C\\hat\\Delta\\ -\\ k_2Ks^\\top\\mathrm{sat}(\\sigma/\\Phi)\\ -\\ \\tilde D^\\top\\tfrac{d}{dt}(J_sd)$$ (B5)
The state-space network's leakage term $J_s\\psi_x$ now reaches **three** blocks ($s,e_D,\\tilde D$) instead of just $s$ (as in the identifier-only case), since it also perturbs the surface-level predictor (Step 3); this is the one genuinely new structural feature of having both estimators active at once.

**Step 7 -- closing every leftover term with Young's inequality.**

> **Young's inequality.** For real numbers (or vectors, via Cauchy-Schwarz) $a,b$ and any $c>0$: $ab\\le\\dfrac{a^2}{2c}+\\dfrac{cb^2}2$, since $\\Big(\\dfrac a{\\sqrt c}-\\sqrt c\\,b\\Big)^2\\ge0$ expands to exactly this after rearranging and dividing by $2$.

*Leakage term (three pieces, one shared free parameter $\\theta_1$).* Write $\\Sigma:=-k_2s+k_3e_D-k_4\\tilde D$, so the leakage term is $\\Sigma^\\top J_s\\psi_x=\\Sigma^\\top J_s\\tilde W_\\Delta^\\top\\phi-\\Sigma^\\top J_sw_x$. Bounding each of the three components of $\\Sigma$ against $\\tilde W_\\Delta$ separately, with $\\bar J_s:=\\sup_\\Omega\\|J_s(x)\\|$ as in the identifier-only case:
$$-k_2s^\\top J_s\\tilde W_\\Delta^\\top\\phi\\le\\frac{\\theta_1}2\\|s\\|^2+\\frac{k_2^2\\bar J_s^2\\bar\\mu^2}{2\\theta_1}\\|\\tilde W_\\Delta\\|_F^2,\\quad k_3e_D^\\top J_s\\tilde W_\\Delta^\\top\\phi\\le\\frac{\\theta_1}2\\|e_D\\|^2+\\frac{k_3^2\\bar J_s^2\\bar\\mu^2}{2\\theta_1}\\|\\tilde W_\\Delta\\|_F^2$$
$$-k_4\\tilde D^\\top J_s\\tilde W_\\Delta^\\top\\phi\\le\\frac{\\theta_1}2\\|\\tilde D\\|^2+\\frac{k_4^2\\bar J_s^2\\bar\\mu^2}{2\\theta_1}\\|\\tilde W_\\Delta\\|_F^2$$
and, identically in structure, the residual-fitting-error piece $\\mp(\\cdot)^\\top J_sw_x$ for each of $s,e_D,\\tilde D$ contributes $\\tfrac{k_i^2\\bar J_s^2\\bar w_x^2}{2\\theta_1}$ (constant) to the residual for $i=2,3,4$ respectively, reusing $\\theta_1$ again.

*State-residual terms (state predictor, two pieces, shared $\\theta_4$).* $\\tilde x^\\top w_x\\le\\tfrac{\\theta_4}2\\|\\tilde x\\|^2+\\tfrac{\\bar w_x^2}{2\\theta_4}$ and, using Assumption 2's $\\|d\\|\\le\\bar d$, $\\tilde x^\\top d\\le\\tfrac{\\theta_4}2\\|\\tilde x\\|^2+\\tfrac{\\bar d^2}{2\\theta_4}$.

*Unmatched-injection and switching terms (surface block, exactly as the identifier-only case).* $-k_2\\Lambda s^\\top C\\hat\\Delta\\le\\tfrac{\\theta_2}2\\|s\\|^2+\\tfrac{(k_2\\Lambda\\bar C\\bar\\mu)^2}{\\theta_2}\\|\\tilde W_\\Delta\\|_F^2+\\tfrac{(k_2\\Lambda\\bar C\\bar\\mu\\bar W)^2}{\\theta_2}$, and $-k_2Ks^\\top\\mathrm{sat}(\\sigma/\\Phi)\\le\\tfrac{\\theta_3}2\\|s\\|^2+\\tfrac{k_2^2K^2p}{2\\theta_3}$ (both derived exactly as in Step 7 of that case, using $\\bar C:=\\sup_\\Omega\\|C(x)\\|$).

*Disturbance-rate term (disturbance-observer block, exactly as the disturbance-observer-only case).* $-\\tilde D^\\top\\tfrac{d}{dt}(J_sd)\\le\\tfrac{\\theta_5}2\\|\\tilde D\\|^2+\\tfrac{\\bar w'^2}{2\\theta_5}$ for a free parameter $\\theta_5\\in(0,2)$.

*Leakage in the weight block.* Exactly as the identifier-only case, $-\\sigma_W\\,\\mathrm{tr}\\big(\\tilde W_\\Delta^\\top W_\\Delta\\big)\\le-\\tfrac{\\sigma_W}2\\|\\tilde W_\\Delta\\|_F^2+\\tfrac{\\sigma_W}2\\|W^*\\|_F^2$.

**Step 8 -- collecting the final inequality.**

Substituting every bound above into (B5):
$$\\dot V\\le-q_s\\|s\\|^2-q_e\\|e_D\\|^2-q_D\\|\\tilde D\\|^2-q_x\\|\\tilde x\\|^2-q_W\\|\\tilde W_\\Delta\\|_F^2+b$$ (B6)
$$q_s=k_2\\Lambda-\\tfrac{\\theta_1+\\theta_2+\\theta_3}2,\\quad q_e=k_3\\kappa_s-\\tfrac{\\theta_1}2,\\quad q_D=k_4\\Big(1-\\tfrac{\\theta_5}2\\Big)-\\tfrac{\\theta_1}2,\\quad q_x=\\kappa-\\theta_4$$
$$q_W=\\frac{\\sigma_W}2-\\frac{(k_2^2+k_3^2+k_4^2)\\bar J_s^2\\bar\\mu^2}{2\\theta_1}-\\frac{(k_2\\Lambda\\bar C\\bar\\mu)^2}{\\theta_2}$$
$$b=\\frac{(k_2^2+k_3^2+k_4^2)\\bar J_s^2\\bar w_x^2}{2\\theta_1}+\\frac{(k_2\\Lambda\\bar C\\bar\\mu\\bar W)^2}{\\theta_2}+\\frac{k_2^2K^2p}{2\\theta_3}+\\frac{\\bar w_x^2+\\bar d^2}{2\\theta_4}+\\frac{\\bar w'^2}{2\\theta_5}+\\frac{\\sigma_W}2\\|W^*\\|_F^2$$
positive-definite in every block ($q_s,q_e,q_D,q_x,q_W>0$) for free parameters $\\theta_1,\\dots,\\theta_4>0$, $\\theta_5\\in(0,2)$ satisfying the analogous margin conditions of the two single-estimator cases. The important qualitative fact is that **$q_W$ must now dominate two coupling sources simultaneously**: the network-vs-surface leakage (arriving via *three* channels, $s,e_D,\\tilde D$, since the state-space network's error leaks into all of them -- versus just $s$ alone in the identifier-only case) and the unmatched-injection coupling from $C(x)$ (as in the identifier-only case). So with both estimators active, $\\sigma_W$ needs more headroom than in either single-estimator case alone, and $k_4$ keeps its usual two-sided window: a floor from $q_D>0$ (the observer must be fast enough relative to its own leakage load), a ceiling from $q_W>0$ (too large a $k_4$ overloads the leakage margin).

Writing each term as (rate) $\\times$ (block weight) exactly as in the single-estimator cases:
$$\\rho:=\\min\\Big\\{\\frac{2q_s}{k_2},\\ \\frac{2q_e}{k_3},\\ 2q_D,\\ 2q_x,\\ 2\\Gamma q_W\\Big\\}$$
so (B6) is $\\dot V\\le-\\rho V+b$.

**Step 9 -- the Comparison Lemma and the final bound.**

> **Comparison Lemma.** If $\\dot V\\le-\\rho V+b$ with $\\rho>0,b\\ge0$, then $V(t)\\le V(0)e^{-\\rho t}+\\tfrac b\\rho\\big(1-e^{-\\rho t}\\big)$, so $V(t)$ converges to, and remains confined within, the ball $V\\le b/\\rho$. (*Proof:* multiply $\\dot V+\\rho V\\le b$ by $e^{\\rho t}$, recognize the left side as $\\frac{d}{dt}(e^{\\rho t}V)$, integrate from $0$ to $t$, divide by $e^{\\rho t}$.)

Since $V\\ge\\tfrac{k_2}2\\|s\\|^2$, this gives $\\limsup_{t\\to\\infty}\\|s\\|\\le\\sqrt{2b/(k_2\\rho)}$.

**Step 10 -- conclusion, in plain terms.**

As in the state-space-identifier-only case, $\\sigma\\ne s$ here (the compensated surface differs from the raw one), so the switching term was closed with the looser Cauchy-Schwarz bound rather than the tight boundary-layer lemma -- **$K=0$ remains the recommended default** once both estimators are trusted to have converged. The residual $b$ collects contributions from every unremovable source in the system at once: the network's fitting residual $\\bar w_x$, the disturbance's own size $\\bar d$ and rate $\\bar w'$, the switching margin, and the leakage constant $\\|W^*\\|_F$ -- none of which can be driven to zero, only bounded, which is why this (like every composite case except the ideal one) converges to a ball around zero rather than to zero exactly.

**Conclusion: semi-global Uniformly Ultimately Bounded (UUB) stability** of $s,e_D,\\tilde D,\\tilde x,\\tilde W_\\Delta$ -- all five converge to, and remain confined within, a ball around zero, per the Comparison Lemma, shrinking as the network's basis fits $\\Delta$ more tightly, the disturbance varies more slowly, and the gains satisfy the margin conditions of Step 8 with room to spare. As always, the fixed-step (Euler) integration used by the simulator (`dt`) must stay below the step-size bound implied by this system's own Lipschitz constants for this continuous-time guarantee to carry over to the discrete-time simulation."""
    )


def backstepping_stability_proof(has_delta, has_disturbance):
    if not (has_delta or has_disturbance):
        return (
"""**Stability proof -- Command-Filtered Backstepping, ideal case (no active uncertainty/disturbance estimation)**

**What is being proved.** With no uncertainty and no disturbance, the closed-loop tracking error converges to zero -- in fact exponentially fast -- for any positive choice of the per-step feedback gains, with no condition on those gains beyond simple positivity.

**Setup -- the plant and the backstepping recursion.**

The plant is a strict-feedback chain, $\\dot x_i=f_i(\\bar x_i)+g_i(\\bar x_i)x_{i+1}$ for $i=1,\\dots,n-1$ and $\\dot x_n=f_n(x)+g_n(x)u$, output $y=x_1$, where $\\bar x_i:=[x_1,\\dots,x_i]^\\top$ (each state's own dynamics depend only on itself and the states before it, plus the *next* state or, at the last step, the control input). Define, step by step:
- $z_1:=x_1-y_d$ -- the **tracking error**: how far the output is from the desired trajectory $y_d$.
- $\\alpha_i$ -- the **virtual control** at step $i$: the value the recursion *wants* $x_{i+1}$ to take so that step $i$'s own error shrinks. The real control input is $u:=\\alpha_n$, the virtual control of the last step.
- $z_i:=x_i-\\alpha_{i-1}$ for $i\\ge2$ -- the error at step $i$: how far the actual state $x_i$ is from the *target* set by the previous step's virtual control.
- $c_i>0$ -- the **feedback gain** at step $i$: how hard that step pushes its own error toward zero.

The virtual controls are built recursively, using the *exact* time derivative of the previous step's virtual control (obtained by symbolic differentiation, `time_derivative()` in the code -- not a filter, since there is no uncertainty here to make that derivative expensive to track any other way):
$$\\alpha_i=\\frac1{g_i}\\Big(-c_iz_i-g_{i-1}z_{i-1}-f_i+\\dot\\alpha_{i-1}\\Big),\\qquad g_0z_0:=0,\\quad\\dot\\alpha_0:=\\dot y_d$$
The term $-g_{i-1}z_{i-1}$ (present from step $2$ onward) is not incidental -- it exists specifically to make a cross term between consecutive steps cancel exactly, shown in Step 2.

**Step 1 -- the assumption that makes this the "ideal" case.**

*Assumption 1.* The plant matches the nominal model exactly -- no $\\Delta_i$, no $d_i(t)$ in any state equation.

*Assumption 2.* Each state equation is exactly linear in the next state (or, at the last step, in $u$) -- this is what makes the division by $g_i$ in the recursion above well posed, and what makes the exact derivative $\\dot\\alpha_{i-1}$ well defined via the chain rule.

**Step 2 -- the exact error dynamics, and why the recursion's design produces an exact telescoping cancellation.**

> **Lemma (Exact compensated-error identity).** With the recursion of the Setup, for every step $i=1,\\dots,n$ (with the convention $g_0z_0:=0$ for $i=1$, and $z_{n+1}:=0$ for $i=n$),
> $$\\dot z_i=-c_iz_i+g_iz_{i+1}-g_{i-1}z_{i-1}$$
>
> *Proof.* Write $x_{i+1}=z_{i+1}+\\alpha_i$ (by the definition of $z_{i+1}$, or $x_{i+1}=u=\\alpha_n$ directly at the last step). Then
> $$\\dot z_i=\\dot x_i-\\dot\\alpha_{i-1}=\\big(f_i+g_ix_{i+1}\\big)-\\dot\\alpha_{i-1}=f_i+g_i\\big(z_{i+1}+\\alpha_i\\big)-\\dot\\alpha_{i-1}=f_i+g_iz_{i+1}+g_i\\alpha_i-\\dot\\alpha_{i-1}$$
> Substituting the recursion's own definition of $\\alpha_i$, rearranged as $g_i\\alpha_i=-c_iz_i-g_{i-1}z_{i-1}-f_i+\\dot\\alpha_{i-1}$:
> $$\\dot z_i=f_i+g_iz_{i+1}+\\big(-c_iz_i-g_{i-1}z_{i-1}-f_i+\\dot\\alpha_{i-1}\\big)-\\dot\\alpha_{i-1}=-c_iz_i+g_iz_{i+1}-g_{i-1}z_{i-1}$$
> since $f_i$ and $\\dot\\alpha_{i-1}$ each cancel exactly. $\\blacksquare$

**Step 3 -- the Lyapunov function, differentiated using the Lemma.**

Take $V=\\tfrac12\\sum_{i=1}^nz_i^2$ -- the total squared tracking error, summed across every step. Differentiating using the Lemma:
$$\\dot V=\\sum_{i=1}^nz_i\\dot z_i=\\sum_{i=1}^nz_i\\big(-c_iz_i+g_iz_{i+1}-g_{i-1}z_{i-1}\\big)=-\\sum_{i=1}^nc_iz_i^2+\\underbrace{\\sum_{i=1}^ng_iz_iz_{i+1}-\\sum_{i=1}^ng_{i-1}z_{i-1}z_i}_{\\text{claim: this bracket is exactly }0}$$

**Step 4 -- the exact cancellation of the bracket.**

Re-index the second sum with $j:=i-1$ (as $i$ ranges over $1,\\dots,n$, $j$ ranges over $0,\\dots,n-1$, and the $j=0$ term is $g_0z_0z_1=0$ by convention): $\\sum_{i=1}^ng_{i-1}z_{i-1}z_i=\\sum_{j=0}^{n-1}g_jz_jz_{j+1}=\\sum_{j=1}^{n-1}g_jz_jz_{j+1}$. This is **termwise identical** to the first sum $\\sum_{i=1}^ng_iz_iz_{i+1}=\\sum_{i=1}^{n-1}g_iz_iz_{i+1}$ (the $i=n$ term vanishes since $z_{n+1}:=0$) -- both run over the same index range with the same summand $g_jz_jz_{j+1}$. So the bracket is exactly $0$, and
$$\\boxed{\\ \\dot V=-\\sum_{i=1}^nc_iz_i^2\\ }$$ (BS1)
This telescoping cancellation is precisely why the recursion carries the $-g_{i-1}z_{i-1}$ term in $\\alpha_i$: without it, this cross-step coupling term would not cancel and would need to be bounded instead of eliminated.

**Step 5 -- solving this differential inequality: the Comparison Lemma.**

> **Comparison Lemma.** If a differentiable scalar function $V(t)\\ge0$ satisfies $\\dot V\\le-\\rho V+b$ for constants $\\rho>0,b\\ge0$, then $V(t)\\le V(0)e^{-\\rho t}+\\tfrac b\\rho(1-e^{-\\rho t})$.
>
> *Proof.* Multiply $\\dot V+\\rho V\\le b$ by the integrating factor $e^{\\rho t}>0$: $\\frac{d}{dt}(e^{\\rho t}V)=e^{\\rho t}(\\dot V+\\rho V)\\le be^{\\rho t}$. Integrate from $0$ to $t$ and divide by $e^{\\rho t}$. $\\blacksquare$ When $b=0$ this is plain exponential decay.

Since $c_iz_i^2\\ge\\big(\\min_jc_j\\big)z_i^2$ for every $i$, (BS1) gives $\\dot V\\le-\\big(\\min_ic_i\\big)\\sum_iz_i^2=-2\\big(\\min_ic_i\\big)V$ -- exactly the Comparison Lemma's hypothesis with $\\rho=2\\min_ic_i$ and $b=0$ (no residual constant at all, since Assumption 1 removed every possible source of one). Hence $V(t)\\le V(0)e^{-2(\\min_ic_i)t}$.

**Step 6 -- conclusion, in plain terms.**

Each step's error-cancelling term ($-g_{i-1}z_{i-1}$ inside $\\alpha_i$) is built specifically so that, once every step's contribution is added up, everything except a purely negative sum cancels out exactly -- so the total squared tracking error always strictly shrinks, at a guaranteed rate set by the slowest step.

**Conclusion: global exponential stability**, in fact $\\sum_iz_i(t)^2\\le\\big(\\sum_iz_i(0)^2\\big)e^{-2(\\min_ic_i)t}$, for any $c_i>0$ -- no gain condition beyond simple positivity, and no residual ball, because Assumption 1 leaves nothing unmodeled for any residual to come from."""
        )

    return _backstepping_composite_proof(has_delta, has_disturbance)


# same deal as _smc_composite_proof - the rest is one big string with inline
# ternaries on has_delta covering "disturbance only" and "both active".
def _backstepping_composite_proof(has_delta, has_disturbance):
    if has_delta and not has_disturbance:
        return (
"""**Stability proof -- Command-Filtered Backstepping, per-step neural-network identifier only (no external disturbance)**

**What is being proved.** With a state-dependent model uncertainty $\\Delta_i$ at every step but no external disturbance, the compensated tracking error, the state-predictor error, and the network weight error -- all at every step -- converge to, and remain inside, a ball around zero (semi-global UUB).

**Setup -- the command filter and the compensated error.**

The plant is the same strict-feedback chain as the ideal case, $\\dot x_i=f_i(\\bar x_i)+g_i(\\bar x_i)x_{i+1}+\\Delta_i(\\bar x_i)$ for $i<n$ and $\\dot x_n=f_n+g_nu+\\Delta_n$ (no $d_i(t)$ term anywhere -- see Step 1). Unlike the ideal case, computing each virtual control's exact derivative $\\dot\\alpha_{i-1}$ symbolically would now also have to differentiate the (adapting, hence unpredictable in closed form) network estimate $\\hat\\Delta_{i-1}$ at every step -- an ever-growing symbolic expression. Instead, each raw virtual command $x_{i+1,c}$ is passed through a first-order **command filter** that supplies its own derivative for free, at the cost of a small, explicitly compensated lag:
- $x_{i+1,d}$ -- the filtered version of the raw command $x_{i+1,c}$, with $\\dot x_{i+1,d}$ available directly as an internal filter state (no differentiation needed).
- $\\chi_i:=x_{i+1,d}-x_{i+1,c}$ -- the **filter tracking error**: how far the filtered command still lags the raw one.
- $z_i:=x_i-x_{i,d}$ (with $x_{1,d}:=y_d$) -- the raw error at step $i$, exactly as the ideal case.
- $\\xi_i$ -- the **compensating signal**, defined by $\\dot\\xi_i:=-c_i\\xi_i+g_i\\chi_i+g_i\\xi_{i+1}$, $\\xi_i(0)=0$, $\\xi_n\\equiv0$ -- correcting for the filter lag *before* it can enter the stability certificate.
- $\\varepsilon_i:=z_i-\\xi_i$ -- the **compensated tracking error** actually used to build the control law below.

**Step 1 -- the assumption.**

*Assumption 1.* Each step's uncertainty $\\Delta_i(\\bar x_i)$ can be approximated by a network with a fixed basis $\\mu_i:\\mathbb R^i\\to\\mathbb R^N$ and some ideal weight $W_i^*\\in\\mathbb R^N$, to within a bounded leftover error: there exists $\\bar w_i\\ge0$ such that
$$\\big\\|\\Delta_i(\\bar x_i)-W_i^{*\\top}\\mu_i(\\bar x_i)\\big\\|\\le\\bar w_i\\qquad\\text{on the region the system actually operates in}$$
Define the **fitting residual** $w_i(\\bar x_i):=\\Delta_i(\\bar x_i)-W_i^{*\\top}\\mu_i(\\bar x_i)$, $\\|w_i\\|\\le\\bar w_i$. As in the SMC case, a larger, better-placed basis shrinks $\\bar w_i$ but a fixed, finite one can never drive it to exactly zero.

*Assumption 2.* Each equation is exactly linear in the next state (needed for the recursion to be well posed, as in the ideal case), and there is no external disturbance: $d_i(t)\\equiv0$ for every $i$.

**Step 2 -- the per-step network, its predictor, and the resulting error dynamics.**

At each step, introduce:
- $\\hat\\Delta_i:=W_i^\\top\\mu_i(\\bar x_i)$ -- the network's current estimate, using its current (adapting) weight $W_i$.
- $\\tilde W_i:=W_i-W_i^*$ -- the **weight error**.
- $\\hat x_i$ -- an internally simulated copy of state $i$, updated by $\\dot{\\hat x}_i:=f_i+g_ix_{i+1}+\\hat\\Delta_i+\\kappa(x_i-\\hat x_i)$; $e_i:=\\hat x_i-x_i$ -- the **state-predictor error** (note: $\\kappa$, the predictor gain, is shared across all steps).

Combine $\\hat\\Delta_i-\\Delta_i=\\tilde W_i^\\top\\mu_i-w_i$ (using Assumption 1); define $\\psi_i:=\\tilde W_i^\\top\\mu_i-w_i=\\hat\\Delta_i-\\Delta_i$, the **total per-step estimation error**.

> **Lemma 1 (Per-step predictor error dynamics).** $\\dot e_i=-\\kappa e_i+\\psi_i$.
>
> *Proof.* $\\dot e_i=\\dot{\\hat x}_i-\\dot x_i=\\big(f_i+g_ix_{i+1}+\\hat\\Delta_i-\\kappa e_i\\big)-\\big(f_i+g_ix_{i+1}+\\Delta_i\\big)=\\hat\\Delta_i-\\Delta_i-\\kappa e_i=\\psi_i-\\kappa e_i$, since $f_i+g_ix_{i+1}$ cancels exactly and $\\hat\\Delta_i-\\Delta_i=\\psi_i$ by definition. (Note the sign of the feedback term above, $-\\kappa(x_i-\\hat x_i)=-\\kappa\\cdot(-e_i)=\\kappa e_i$ inside $\\dot{\\hat x}_i$, is written to make $-\\kappa e_i$ appear correctly here; equivalently $\\dot{\\hat x}_i=f_i+g_ix_{i+1}+\\hat\\Delta_i-\\kappa e_i$.) $\\blacksquare$

**Step 3 -- the virtual command, the exact compensated-error identity, and the network's adaptation law.**

The virtual command at step $i$ is
$$x_{i+1,c}=\\frac1{g_i}\\Big(-c_i\\varepsilon_i-g_{i-1}\\varepsilon_{i-1}-f_i-\\hat\\Delta_i+\\dot x_{i,d}\\Big),\\qquad g_0\\varepsilon_0:=0,\\quad\\dot x_{1,d}:=\\dot y_d$$
-- the same recursive structure as the ideal case, but built from the *compensated* error $\\varepsilon_i$ (Setup) rather than the raw $z_i$, using the *estimated* uncertainty $\\hat\\Delta_i$, and using $\\dot x_{i,d}$ supplied directly by the filter rather than an exact symbolic derivative.

> **Lemma 2 (Exact compensated-error identity, with filter and estimation).** $\\dot\\varepsilon_i=-c_i\\varepsilon_i+g_i\\varepsilon_{i+1}-g_{i-1}\\varepsilon_{i-1}-\\psi_i$ for every $i$ (with $\\varepsilon_{n+1}:=0$).
>
> *Proof.* Write $x_{i+1}=z_{i+1}+x_{i+1,d}=z_{i+1}+\\chi_i+x_{i+1,c}$ (Setup's definition of $\\chi_i$). Then
> $$\\dot z_i=f_i+g_ix_{i+1}+\\Delta_i-\\dot x_{i,d}=f_i+g_iz_{i+1}+g_i\\chi_i+g_ix_{i+1,c}+\\Delta_i-\\dot x_{i,d}$$
> Substituting $g_ix_{i+1,c}$ from the virtual-command definition above (rearranged: $g_ix_{i+1,c}=-c_i\\varepsilon_i-g_{i-1}\\varepsilon_{i-1}-f_i-\\hat\\Delta_i+\\dot x_{i,d}$), the $f_i$ and $\\dot x_{i,d}$ terms cancel exactly:
> $$\\dot z_i=-c_i\\varepsilon_i+g_iz_{i+1}+g_i\\chi_i-g_{i-1}\\varepsilon_{i-1}+\\big(\\Delta_i-\\hat\\Delta_i\\big)=-c_i\\varepsilon_i+g_iz_{i+1}+g_i\\chi_i-g_{i-1}\\varepsilon_{i-1}-\\psi_i$$
> using $\\Delta_i-\\hat\\Delta_i=-\\psi_i$. Subtracting $\\dot\\xi_i=-c_i\\xi_i+g_i\\chi_i+g_i\\xi_{i+1}$ (Setup's definition):
> $$\\dot\\varepsilon_i=\\dot z_i-\\dot\\xi_i=-c_i(z_i-\\xi_i)+g_i(z_{i+1}-\\xi_{i+1})-g_{i-1}\\varepsilon_{i-1}-\\psi_i=-c_i\\varepsilon_i+g_i\\varepsilon_{i+1}-g_{i-1}\\varepsilon_{i-1}-\\psi_i$$
> since $z_i-\\xi_i=\\varepsilon_i$ and $z_{i+1}-\\xi_{i+1}=\\varepsilon_{i+1}$ by definition, and the $g_i\\chi_i$ terms cancel exactly between the two lines. At the last step ($i=n$) there is no filter, so $\\xi_n\\equiv0$, $\\varepsilon_n=z_n$, and the identity reduces to $\\dot z_n=-c_nz_n-g_{n-1}\\varepsilon_{n-1}-\\psi_n$, consistent with $\\varepsilon_{n+1}:=0$. $\\blacksquare$
>
> This identity holds regardless of whether a disturbance observer exists anywhere in the loop: its derivation only used the definition of $w_i$ (hence $\\psi_i$), never any observer dynamics.

The network weights adapt by
$$\\dot W_i:=\\Gamma\\Big(\\mu_i\\,\\mathrm{dr}_i-\\sigma_WW_i\\Big),\\qquad \\mathrm{dr}_i:=k_2\\varepsilon_i-k_3e_i$$
with $\\Gamma>0$ the learning rate, $\\sigma_W>0$ the leakage gain (Step 6 below explains why the drive $\\mathrm{dr}_i$ -- not just $e_i$ alone, as in the SMC case's state-space identifier -- is used here: this network lives in the *same* per-step space as the compensated error $\\varepsilon_i$, so, unlike the SMC state-space network, it *can* share a drive with $\\varepsilon_i$).

**Step 4 -- the Lyapunov function.**

$$V=\\sum_{i=1}^n\\Big[\\underbrace{\\tfrac{k_2}2\\varepsilon_i^2}_{\\text{tracking}}+\\underbrace{\\tfrac{k_3}2e_i^2}_{\\text{predictor}}+\\underbrace{\\tfrac1{2\\Gamma}\\|\\tilde W_i\\|^2}_{\\text{weights}}\\Big]$$
with $k_2,k_3>0$ forced to match the drive's own gains, exactly as in the disturbance-observer cases of the SMC proof family, for the same reason: that is what makes the cross terms below cancel.

*Tracking block.* Using Lemma 2 and summing over $i$: $k_2\\sum_i\\varepsilon_i\\dot\\varepsilon_i=-k_2\\sum_ic_i\\varepsilon_i^2+k_2\\Big[\\sum_ig_i\\varepsilon_i\\varepsilon_{i+1}-\\sum_ig_{i-1}\\varepsilon_{i-1}\\varepsilon_i\\Big]-k_2\\sum_i\\varepsilon_i\\psi_i$. The bracketed term vanishes by exactly the same re-indexing argument as (BS1)/Step 4 of the ideal case (the second sum with $j=i-1$ is termwise identical to the first) -- this is precisely why the virtual command still carries the $-g_{i-1}\\varepsilon_{i-1}$ term.

*Predictor block.* Using Lemma 1: $k_3\\sum_ie_i\\dot e_i=-k_3\\kappa\\sum_ie_i^2+k_3\\sum_ie_i\\psi_i$.

*Weight block.* By the same cyclic-trace argument as the SMC state-space identifier's Lemma 2 (here in scalar-per-step form, so simply $\\tfrac1\\Gamma W_i\\dot W_i$ summed): $\\tfrac1\\Gamma\\sum_i\\tilde W_i\\dot W_i=\\sum_i\\mathrm{dr}_i\\big(\\tilde W_i^\\top\\mu_i\\big)-\\sigma_W\\sum_i\\tilde W_iW_i=\\sum_i\\big(k_2\\varepsilon_i-k_3e_i\\big)\\big(\\tilde W_i^\\top\\mu_i\\big)-\\sigma_W\\sum_i\\tilde W_iW_i$.

**Step 5 -- the exact cancellation, and what is left over.**

Using $\\tilde W_i^\\top\\mu_i=\\psi_i+w_i$ (rearranging Step 2's $\\psi_i=\\tilde W_i^\\top\\mu_i-w_i$), the weight block's first term is $\\sum_i(k_2\\varepsilon_i-k_3e_i)(\\psi_i+w_i)=k_2\\sum_i\\varepsilon_i\\psi_i-k_3\\sum_ie_i\\psi_i+\\sum_i(k_2\\varepsilon_i-k_3e_i)w_i$. Collecting the $\\psi_i$-cross terms from all three blocks:
$$\\underbrace{-k_2\\sum_i\\varepsilon_i\\psi_i}_{\\text{tracking}}+\\underbrace{k_3\\sum_ie_i\\psi_i}_{\\text{predictor}}+\\underbrace{k_2\\sum_i\\varepsilon_i\\psi_i-k_3\\sum_ie_i\\psi_i}_{\\text{weights}}=0$$
**Every $\\psi_i$-cross term cancels identically**, *even though no disturbance observer exists anywhere in this design*, because the cancellation only ever needed the drive $\\mathrm{dr}_i=k_2\\varepsilon_i-k_3e_i$ to be shared correctly between the tracking-error dynamics (Lemma 2) and the predictor dynamics (Lemma 1) -- not for any observer to be present. What is left over is the one genuinely new term per step, $\\sum_i(k_2\\varepsilon_i-k_3e_i)w_i$: since the fitting residual $w_i$ (Assumption 1) is never estimated by anything (there is no per-step disturbance observer here to chase it), it does not cancel, only bound. Collecting everything:
$$\\dot V=-k_2\\sum_ic_i\\varepsilon_i^2-k_3\\kappa\\sum_ie_i^2-\\sigma_W\\sum_i\\tilde W_iW_i+\\sum_i\\big(k_2\\varepsilon_i-k_3e_i\\big)w_i$$ (BS2)

**Step 6 -- closing the two remaining terms with Young's inequality.**

> **Young's inequality.** For real numbers $a,b$ and any $c>0$: $ab\\le\\dfrac{a^2}{2c}+\\dfrac{cb^2}2$, since $\\Big(\\dfrac a{\\sqrt c}-\\sqrt c\\,b\\Big)^2\\ge0$ expands to exactly this after rearranging and dividing by $2$.

*Leakage.* Exactly as in every other case of this proof family, with $W_i=\\tilde W_i+W_i^*$: $-\\sigma_W\\tilde W_iW_i=-\\sigma_W\\tilde W_i^2-\\sigma_W\\tilde W_iW_i^*\\le-\\tfrac{\\sigma_W}2\\tilde W_i^2+\\tfrac{\\sigma_W}2(W_i^*)^2$ (using $-\\tilde W_iW_i^*\\le\\tfrac12\\tilde W_i^2+\\tfrac12(W_i^*)^2$, Young's inequality with $a=\\tilde W_i,b=W_i^*,c=1$).

*Residual forcing.* For each step, using free parameters $\\theta_a,\\theta_b>0$ (one shared choice across all steps, for simplicity): $k_2\\varepsilon_iw_i\\le\\tfrac{\\theta_a}2\\varepsilon_i^2+\\tfrac{k_2^2}{2\\theta_a}\\bar w_i^2$ and $-k_3e_iw_i\\le\\tfrac{\\theta_b}2e_i^2+\\tfrac{k_3^2}{2\\theta_b}\\bar w_i^2$ (using $|w_i|\\le\\bar w_i$, Assumption 1).

**Step 7 -- collecting the final inequality.**

Substituting into (BS2):
$$\\dot V\\le-\\sum_i\\Big(k_2c_i-\\frac{\\theta_a}2\\Big)\\varepsilon_i^2-\\sum_i\\Big(k_3\\kappa-\\frac{\\theta_b}2\\Big)e_i^2-\\frac{\\sigma_W}2\\sum_i\\tilde W_i^2+b,\\qquad b=\\Big(\\frac{k_2^2}{2\\theta_a}+\\frac{k_3^2}{2\\theta_b}\\Big)\\sum_i\\bar w_i^2+\\frac{\\sigma_W}2\\sum_i(W_i^*)^2$$ (BS3)
positive-definite in every block for **any** $c_i,\\kappa,\\sigma_W>0$ (any positive $\\theta_a,\\theta_b$ work, just trading margin against residual size) -- notably, **no leakage-vs-coupling condition on any observer gain is needed at all here**, since with no disturbance observer running there is nothing for the network to drift against; this is strictly simpler than every SMC case with $\\Delta$ active, whose network reaches the surface through $J_s$ and does carry such a condition.

Writing each term as (rate) $\\times$ (block weight): $\\rho:=\\min\\Big\\{2\\big(\\min_ic_i-\\tfrac{\\theta_a}{2k_2}\\big),\\ 2\\big(\\kappa-\\tfrac{\\theta_b}{2k_3}\\big),\\ \\Gamma\\sigma_W\\Big\\}$ (choosing $\\theta_a,\\theta_b$ small enough relative to $k_2\\min_ic_i,k_3\\kappa$ that these are positive), so (BS3) reads $\\dot V\\le-\\rho V+b$.

**Step 8 -- the Comparison Lemma and the final bound.**

> **Comparison Lemma.** If $\\dot V\\le-\\rho V+b$ with $\\rho>0,b\\ge0$, then $V(t)\\le V(0)e^{-\\rho t}+\\tfrac b\\rho(1-e^{-\\rho t})$, so $V(t)$ converges to, and remains confined within, the ball $V\\le b/\\rho$. (*Proof:* multiply $\\dot V+\\rho V\\le b$ by $e^{\\rho t}$, recognize the left side as $\\frac{d}{dt}(e^{\\rho t}V)$, integrate.)

**Step 9 -- boundedness of the physical state, and conclusion.**

Since $z_1=\\varepsilon_1+\\xi_1$ and $x_{i+1}=z_{i+1}+x_{i+1,d}$ with $|x_{i+1,d}|\\le X_{i+1}$ (the command filter's own design-time magnitude limit), boundedness of every $\\varepsilon_i$ (from the Comparison Lemma above) propagates recursively to boundedness of every physical state $x_i$.

**Conclusion: semi-global Uniformly Ultimately Bounded (UUB) stability** of $\\varepsilon_i,e_i,\\tilde W_i$ at every step -- they converge to, and remain confined within, a ball around zero, shrinking as each step's network fits its own $\\Delta_i$ more tightly. The command-filter time constant `tau` must stay an order of magnitude above the integration step `dt` to avoid a numerically stiff filter, and the fixed-step (Euler) integration itself must stay below the step-size bound implied by this system's own Lipschitz constants, exactly as in every other case of this proof family."""
        )
    return (
"""**Stability proof -- Command-Filtered Backstepping, """
+ ("both estimators active (per-step neural-network identifier and disturbance observer)"
   if has_delta else
   "per-step disturbance observer only (no state-dependent uncertainty)")
+ """**

**What is being proved.** """
+ ("With both a state-dependent uncertainty $\\Delta_i$ and an external disturbance $d_i(t)$ at every step, the compensated tracking error, the predictor error, the network weight error, and the disturbance-estimation error -- all at every step -- converge to, and remain inside, a ball around zero (semi-global UUB)."
   if has_delta else
   "With a bounded external disturbance $d_i(t)$ at every step but no state-dependent uncertainty, the compensated tracking error, the predictor error, and the disturbance-estimation error -- all at every step -- converge to, and remain inside, a ball around zero (semi-global UUB).")
+ """

**Setup -- the command filter and the shared per-step predictor.**

The plant is the strict-feedback chain $\\dot x_i=f_i(\\bar x_i)+g_i(\\bar x_i)x_{i+1}+\\Theta_i(\\bar x_i,t)$ for $i<n$ and $\\dot x_n=f_n+g_nu+\\Theta_n$, where $\\Theta_i$ is everything the nominal model $f_i,g_i$ misses at step $i$. """
+ ("$\\Theta_i=\\Delta_i(\\bar x_i)+d_i(t)$, both a state-dependent part and an external disturbance."
   if has_delta else
   "$\\Theta_i=d_i(t)$ purely (an **Assumption**, stated next) -- no state-dependent part.")
+ """ As in the ideal case, $z_i:=x_i-x_{i,d}$ is the raw error, $\\chi_i:=x_{i+1,d}-x_{i+1,c}$ the filter tracking error, $\\xi_i$ the compensating signal ($\\dot\\xi_i:=-c_i\\xi_i+g_i\\chi_i+g_i\\xi_{i+1}$, $\\xi_i(0)=0$, $\\xi_n\\equiv0$), and $\\varepsilon_i:=z_i-\\xi_i$ the compensated tracking error.

Unlike the ideal and delta-only cases, this design uses **one shared predictor per step** to train *both* estimators at once:
$$\\dot{\\hat x}_i:=f_i+g_ix_{i+1}+\\hat\\Delta_i+\\hat w_i-\\kappa e_i,\\qquad e_i:=\\hat x_i-x_i$$
where $\\hat\\Delta_i""" + (":=W_i^\\top\\mu_i(\\bar x_i)$ (defined below, Step 2)" if has_delta else ":=0$ (frozen -- see Step 1)")
+ """ and $\\hat w_i$ (defined in Step 3) are the network estimate and disturbance-observer estimate respectively, and $e_i$ is the shared **predictor error**.

**Step 1 -- the assumption(s), the ideal weight, and the residual $w_i$.**

"""
+ ("""*Assumption 1.* Each step's state-dependent part $\\Delta_i(\\bar x_i)$ can be approximated by a fixed-basis network to within a bounded fitting error (as in the delta-only case).

""" if has_delta else "")
+ """*Assumption """ + ("2" if has_delta else "1") + """.* Fix, once and for all, a per-step regressor $\\mu_i:\\mathbb R^i\\to\\mathbb R^N$ and define the **ideal weight** $W_i^*$ as any constant vector achieving the best uniform fit of $\\Theta_i$ on the region the system actually operates in, and the **residual**
$$w_i(\\bar x_i,t):=\\Theta_i(\\bar x_i,t)-W_i^{*\\top}\\mu_i(\\bar x_i)$$
"""
+ ("Here, since $\\Delta_i$ is what the network is built to capture, $w_i$ is chiefly the *disturbance* $d_i(t)$ plus whatever small part of $\\Delta_i$ the fixed basis cannot represent -- it is bounded, $\\|w_i\\|\\le\\bar w_i$, but **not assumed small**: it genuinely contains the disturbance in full."
   if has_delta else
   "Since there is no state-dependent part here ($\\Theta_i=d_i(t)$, Setup), the natural choice is $W_i^*\\equiv0$ (a fixed regressor in the state cannot usefully represent a signal that depends only on time), so $w_i=d_i(t)$ exactly -- the residual *is* the disturbance.")
+ """ For $i<n$, since $\\bar x_i$'s own dynamics involve only $x_1,\\dots,x_{i+1}$ and $\\Theta_1,\\dots,\\Theta_i$ -- **never the control input $u$** -- $w_i$'s rate of change is bounded by a genuine, fixed plant constant $\\bar w_i'$: $\\|\\dot w_i\\|\\le\\bar w_i'$. Only the *last* channel is different (Step 6): $u=x_{n+1,c}$ depends on every estimate, including this step's own $\\hat w_n$, so bounding $\\dot w_n$ is a closed-loop matter, not a plant property alone.

**Step 2 -- """ + ("the network estimate and its role in the shared drive." if has_delta else "why the network block is entirely absent.") + """**

"""
+ ("""Define $\\hat\\Delta_i:=W_i^\\top\\mu_i(\\bar x_i)$, $\\tilde W_i:=W_i-W_i^*$ (the weight error), updated by $\\dot W_i:=\\Gamma\\big(\\mu_i\\,\\mathrm{dr}_i-\\sigma_WW_i\\big)$ with the **same** drive $\\mathrm{dr}_i$ used for the disturbance observer below (Step 3) -- this sharing is what makes the exact cancellation of Step 5 possible."""
   if has_delta else
   """Since `estimate_delta=False` here, $W_i\\equiv0$ for all time (its update law is gated off and it starts at $0$), so $\\hat\\Delta_i\\equiv0$ identically and $\\tilde W_i\\equiv0$ -- not approximately small, exactly zero, since $W_i^*=0$ too (Step 1). The weight block of $V$ (Step 4) is therefore entirely absent in this case, not merely negligible.""")
+ """

**Step 3 -- the disturbance observer, and the shared drive.**

Define the **drive** $\\mathrm{dr}_i:=k_2\\varepsilon_i-k_3e_i$ and update
$$\\hat w_i:=\\zeta_i-k_4e_i,\\qquad \\dot\\zeta_i:=\\mathrm{dr}_i-k_4\\kappa e_i$$
Define the **total per-step estimation error** $\\psi_i:=\\tilde W_i^\\top\\mu_i+\\tilde w_i$, where $\\tilde w_i:=\\hat w_i-w_i$ (so $\\psi_i=\\hat\\Delta_i+\\hat w_i-\\Theta_i$, the gap between everything currently estimated and the truth).

> **Lemma 1 (Predictor error dynamics).** $\\dot e_i=-\\kappa e_i+\\psi_i$.
>
> *Proof.* $\\dot e_i=\\dot{\\hat x}_i-\\dot x_i=\\big(f_i+g_ix_{i+1}+\\hat\\Delta_i+\\hat w_i-\\kappa e_i\\big)-\\big(f_i+g_ix_{i+1}+\\Theta_i\\big)=\\big(\\hat\\Delta_i+\\hat w_i-\\Theta_i\\big)-\\kappa e_i=\\psi_i-\\kappa e_i$, using the Setup's predictor definition and the true dynamics, with $f_i+g_ix_{i+1}$ cancelling exactly, and $\\hat\\Delta_i+\\hat w_i-\\Theta_i=\\big(W_i^\\top\\mu_i-W_i^{*\\top}\\mu_i\\big)+\\big(\\hat w_i-w_i\\big)=\\tilde W_i^\\top\\mu_i+\\tilde w_i=\\psi_i$ by Step 1's definition of $w_i$ and Step 3's definition of $\\tilde w_i$. $\\blacksquare$

> **Lemma 2 (Derivative-free observer identity).** $\\dot{\\hat w}_i=\\mathrm{dr}_i-k_4\\psi_i$, equivalently $\\dot{\\tilde w}_i=\\mathrm{dr}_i-k_4\\psi_i-\\dot w_i$, using no derivative of any measured signal.
>
> *Proof.* $\\dot{\\hat w}_i=\\dot\\zeta_i-k_4\\dot e_i=\\big(\\mathrm{dr}_i-k_4\\kappa e_i\\big)-k_4\\big(-\\kappa e_i+\\psi_i\\big)=\\mathrm{dr}_i-k_4\\kappa e_i+k_4\\kappa e_i-k_4\\psi_i=\\mathrm{dr}_i-k_4\\psi_i$ (Lemma 1 substituted for $\\dot e_i$, the $\\mp k_4\\kappa e_i$ terms cancelling exactly). Since $\\tilde w_i=\\hat w_i-w_i$, $\\dot{\\tilde w}_i=\\dot{\\hat w}_i-\\dot w_i=\\mathrm{dr}_i-k_4\\psi_i-\\dot w_i$. This is exactly the identity implemented in `AdaptiveBackstepping.compute_derivs` as `D_hat_dot = drive - k4*(x_hat_dot - x_dot_real + kappa*e_D)`, using the identity $\\dot{\\hat x}_i-\\dot x_i+\\kappa e_i=\\dot e_i+\\kappa e_i=\\psi_i$ (Lemma 1) to place feedback on $\\psi_i$ without ever needing $\\dot x_i$ directly. $\\blacksquare$

**Step 4 -- the virtual command and the exact compensated-error identity.**

$$x_{i+1,c}=\\frac1{g_i}\\Big(-c_i\\varepsilon_i-g_{i-1}\\varepsilon_{i-1}-f_i-\\hat\\Delta_i-\\hat w_i+\\dot x_{i,d}\\Big)$$

> **Lemma 3 (Exact compensated-error identity).** $\\dot\\varepsilon_i=-c_i\\varepsilon_i+g_i\\varepsilon_{i+1}-g_{i-1}\\varepsilon_{i-1}-\\psi_i$ for every $i$ (with $\\varepsilon_{n+1}:=0$).
>
> *Proof.* Identical in structure to the delta-only case's Lemma 2: writing $x_{i+1}=z_{i+1}+\\chi_i+x_{i+1,c}$ and substituting the virtual-command definition into $\\dot z_i=f_i+g_ix_{i+1}+\\Theta_i-\\dot x_{i,d}$, the $f_i,\\dot x_{i,d}$ terms cancel and $\\Theta_i-\\hat\\Delta_i-\\hat w_i=-\\psi_i$ (Step 3's definition of $\\psi_i$) gives $\\dot z_i=-c_i\\varepsilon_i+g_iz_{i+1}+g_i\\chi_i-g_{i-1}\\varepsilon_{i-1}-\\psi_i$; subtracting $\\dot\\xi_i$ gives the claim. $\\blacksquare$

**Step 5 -- the Lyapunov function, and the exact cancellation of every cross term.**

$$V=\\sum_{i=1}^n\\Big[\\tfrac{k_2}2\\varepsilon_i^2+\\tfrac{k_3}2e_i^2"""
+ ("+\\tfrac1{2\\Gamma}\\|\\tilde W_i\\|^2" if has_delta else "")
+ """+\\tfrac12\\tilde w_i^2\\Big]$$

Differentiating block by block using Lemmas 1--3 (and, """ + ("for the weight block, the same cyclic-trace argument as the delta-only case's Step 4" if has_delta else "the weight block being absent, Step 2") + """):

*Tracking block:* $k_2\\sum_i\\varepsilon_i\\dot\\varepsilon_i=-k_2\\sum_ic_i\\varepsilon_i^2+0-k_2\\sum_i\\varepsilon_i\\psi_i$ (the bracketed cross-step term vanishing exactly as in the ideal case, Step 4 there).

*Predictor block:* $k_3\\sum_ie_i\\dot e_i=-k_3\\kappa\\sum_ie_i^2+k_3\\sum_ie_i\\psi_i$.

"""
+ ("""*Weight block:* $\\tfrac1\\Gamma\\sum_i\\tilde W_i^\\top\\dot W_i=\\sum_i\\mathrm{dr}_i\\big(\\tilde W_i^\\top\\mu_i\\big)-\\sigma_W\\sum_i\\mathrm{tr}(\\tilde W_i^\\top W_i)$.

""" if has_delta else "")
+ """*Observer block:* $\\sum_i\\tilde w_i\\dot{\\tilde w}_i=\\sum_i\\mathrm{dr}_i\\tilde w_i-k_4\\sum_i\\tilde w_i\\psi_i-\\sum_i\\tilde w_i\\dot w_i$.

Adding """ + ("the weight and observer blocks" if has_delta else "the observer block, using $\\tilde W_i^\\top\\mu_i\\equiv0$ since $\\tilde W_i\\equiv0$") + """: $\\sum_i\\mathrm{dr}_i\\big(""" + ("\\tilde W_i^\\top\\mu_i+\\tilde w_i" if has_delta else "\\tilde w_i") + """\\big)=\\sum_i\\mathrm{dr}_i\\psi_i=\\sum_i\\big(k_2\\varepsilon_i-k_3e_i\\big)\\psi_i=k_2\\sum_i\\varepsilon_i\\psi_i-k_3\\sum_ie_i\\psi_i$ (using $\\psi_i=\\tilde W_i^\\top\\mu_i+\\tilde w_i$ from Step 3). Collecting the $\\psi_i$-cross terms from **every** block:
$$\\underbrace{-k_2\\sum_i\\varepsilon_i\\psi_i}_{\\text{tracking}}+\\underbrace{k_3\\sum_ie_i\\psi_i}_{\\text{predictor}}+\\underbrace{k_2\\sum_i\\varepsilon_i\\psi_i-k_3\\sum_ie_i\\psi_i}_{\\text{weight + observer}}=0$$
**Every cross term cancels identically** -- this is precisely why the drive $\\mathrm{dr}_i$ must be shared between """ + ("both update laws" if has_delta else "the observer's update law and the predictor") + """, weighted to match $k_2,k_3$ exactly. What is left:
$$\\dot V=-k_2\\sum_ic_i\\varepsilon_i^2-k_3\\kappa\\sum_ie_i^2"""
+ ("-\\sigma_W\\sum_i\\mathrm{tr}(\\tilde W_i^\\top W_i)" if has_delta else "")
+ """-k_4\\sum_i\\tilde w_i^2-\\sum_i\\tilde w_i\\dot w_i$$ (BS4)

**Step 6 -- closing """ + ("the leakage and the disturbance-rate terms." if has_delta else "the disturbance-rate term.") + """**
"""
+ ("""
> **Young's inequality.** For real numbers $a,b$ and any $c>0$: $ab\\le\\dfrac{a^2}{2c}+\\dfrac{cb^2}2$, from $\\big(\\tfrac a{\\sqrt c}-\\sqrt c\\,b\\big)^2\\ge0$.

*Leakage.* As in every other composite case, with $W_i=\\tilde W_i+W_i^*$: $-\\sigma_W\\,\\mathrm{tr}(\\tilde W_i^\\top W_i)\\le-\\tfrac{\\sigma_W}2\\|\\tilde W_i\\|^2+\\tfrac{\\sigma_W}2\\|W_i^*\\|^2$."""
   if has_delta else """
> **Young's inequality.** For real numbers $a,b$ and any $c>0$: $ab\\le\\dfrac{a^2}{2c}+\\dfrac{cb^2}2$, from $\\big(\\tfrac a{\\sqrt c}-\\sqrt c\\,b\\big)^2\\ge0$.""")
+ """

*Disturbance-rate terms, channel by channel.* For $i<n$, Step 1 gives a genuine plant constant $\\bar w_i'$ (no closed-loop dependence), so directly: $-\\tilde w_i\\dot w_i\\le\\tfrac12\\tilde w_i^2+\\tfrac1{2}\\bar w_i'^2$ (Young with $c=1$). For $i=n$, $\\dot w_n$ depends on $u=x_{n+1,c}$, which depends on every estimate at step $n$ (Step 1's warning); write $\\|\\dot w_n\\|\\le\\bar w_0+\\bar w_1\\|\\tilde W_n\\|+\\bar w_2\\|\\tilde w_n\\|+\\bar w_3\\big(|\\varepsilon_n|+|\\varepsilon_{n-1}|\\big)$ for constants $\\bar w_0,\\bar w_1,\\bar w_2,\\bar w_3\\ge0$ computable from $g_{\\min},\\bar g,c_n$ and the plant's own Lipschitz constants (this is the same style of decomposition used for the SMC disturbance-observer's $\\bar w'$, made explicit here since it now interacts with a per-step network too), and close $-\\tilde w_n\\dot w_n$ with one further Young split per term, exactly as in Step 7 of the SMC composite cases, at the cost of small additional loads on the $\\varepsilon_n,\\varepsilon_{n-1}$""" + (",\\tilde W_n" if has_delta else "") + """ margins below.

**Step 7 -- collecting the final inequality.**

$$\\dot V\\le-q_c\\sum_i\\varepsilon_i^2-\\kappa\\sum_ie_i^2"""
+ ("-q_W\\sum_i\\|\\tilde W_i\\|^2" if has_delta else "")
+ """-q_D\\sum_i\\tilde w_i^2+b_b$$ (BS5)
with, in the notation of the source document's Theorem 2 (Part 2), for a free parameter $\\theta\\in(0,2)$ and Young parameters $\\varepsilon_0,\\varepsilon_1,\\varepsilon_3>0$ absorbing the extra channel-$n$ loads of Step 6:
$$q_c=\\min_ic_i-\\frac{\\omega_3^2}{k_2\\varepsilon_3},\\qquad q_D=k_4\\Big(1-\\frac\\theta2\\Big)-\\omega_2-\\frac{\\varepsilon_0+\\varepsilon_1+\\varepsilon_3}2"""
+ (",\\qquad q_W=\\frac12\\Big(\\sigma_W-\\frac{k_4\\bar\\mu^2}\\theta\\Big)-\\frac{\\omega_1^2}{2\\varepsilon_1}" if has_delta else "")
+ """$$
where $\\omega_1,\\omega_2,\\omega_3$ (built from $\\bar w_0,\\dots,\\bar w_3$ of Step 6) come **only from the last step** -- every earlier step's disturbance rate is the plain fixed bound $\\bar w_i'$, not a closed-loop one, since only $x_n$'s channel sees $u$ at all. This is genuinely simpler than the SMC case, where *every* channel sees $u$ through the decoupling matrix $B_s$. Positive-definiteness ($q_c,q_D"""
+ (",q_W" if has_delta else "")
+ """>0$) holds, with margin against the small-gain loads $\\omega_2,\\omega_3$, for $c_i,\\kappa,k_4>0$"""
+ (" and $\\sigma_W>\\tfrac{k_4\\bar\\mu^2}\\theta+\\tfrac{\\omega_1^2}{\\varepsilon_1}$ (the leakage-vs-coupling condition, exactly as in the SMC shared-predictor case)."
   if has_delta else " -- no leakage condition is needed here at all, since with no network active there is nothing for the disturbance observer to drift against.")
+ """

Writing each term as (rate) $\\times$ (block weight):
$$\\rho_b:=\\min\\Big\\{\\frac{2q_c}{k_2},\\ 2\\kappa,"""
+ ("\\ 2\\Gamma q_W," if has_delta else "")
+ """\\ 2q_D\\Big\\}$$
so (BS5) reads $\\dot V\\le-\\rho_bV+b_b$.

**Step 8 -- the Comparison Lemma, boundedness of the state, and conclusion.**

> **Comparison Lemma.** If $\\dot V\\le-\\rho V+b$ with $\\rho>0,b\\ge0$, then $V(t)\\le V(0)e^{-\\rho t}+\\tfrac b\\rho(1-e^{-\\rho t})$, so $V(t)$ converges to, and remains confined within, the ball $V\\le b/\\rho$. (*Proof:* multiply $\\dot V+\\rho V\\le b$ by $e^{\\rho t}$, recognize the left side as $\\frac{d}{dt}(e^{\\rho t}V)$, integrate from $0$ to $t$.)

As in the delta-only and ideal cases, $z_1=\\varepsilon_1+\\xi_1$ and $x_{i+1}=z_{i+1}+x_{i+1,d}$ with $|x_{i+1,d}|\\le X_{i+1}$ (the filter's own design-time limit) propagate boundedness of every $\\varepsilon_i$ recursively to boundedness of every physical state $x_i$.

One structural point worth stating explicitly, a consequence of the **exact compensator** ($\\xi_i$, carrying the $+g_i\\xi_{i+1}$ term): the filter time constant `tau` and the command rate $\\dot x_{i,c}$ never appear anywhere in $\\rho_b,b_b$ above -- the filter lag has been removed from the certificate altogether, rather than bounded inside it; `tau` reappears only in the final tracking-error bound (proportional to $\\bar\\chi=\\mathcal O(\\tau)$), where it is harmless, and only needs to stay an order of magnitude above the integration step `dt`.

**Conclusion: semi-global Uniformly Ultimately Bounded (UUB) stability** of $\\varepsilon_i,e_i"""
+ (",\\tilde W_i" if has_delta else "")
+ """,\\tilde w_i$ at every step -- all converge to, and remain confined within, a ball around zero, shrinking as """
+ ("each step's network fits its own uncertainty more tightly, " if has_delta else "")
+ """the disturbance varies more slowly and the backstepping gains $c_i$ are kept moderate (they enter $\\omega_3$, exactly as the SMC surface gain $\\Lambda$ does). The fixed-step (Euler) integration used by the simulator (`dt`) must stay below the step-size bound implied by this system's own Lipschitz constants for this continuous-time guarantee to carry over to the discrete-time simulation."""
    )
