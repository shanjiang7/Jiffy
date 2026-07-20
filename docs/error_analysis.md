# Error Analysis: A-Posteriori Self-Convergence Estimation

*Draft section for the paper / AD appendix. Notation follows the JIFFY paper;
all statements are for the linearized heat problem the solver integrates.*

## 1. Setup and exact error decomposition

Let the laser path be partitioned into components $C_1, \dots, C_P$ (contiguous
supersegment ranges assigned to ranks). Because the governing equation is
linear in the temperature deviation from ambient, the serial solution
superposes per-component source contributions: writing $s_i(t)$ for the field
produced by component $C_i$'s source alone,

$$
u^{\mathrm{serial}}(t) \;=\; \sum_{i \le j} s_i(t),
\qquad t \in \mathcal{W}_j ,
$$

where $\mathcal{W}_j$ is component $j$'s time window. The parallel scheme
computes $s_j$ on $\mathcal{W}_j$ exactly (the rank's base solve starts from
ambient, so its terminal state **is** $s_i(T_i^{\mathrm{end}})$, uncontaminated
by upstream heat), and reconstructs the upstream terms by *corrections*:
$\delta_{ij}(t)$, the source-off evolution of $s_i(T_i^{\mathrm{end}})$
evaluated on (a sub-window of) $\mathcal{W}_j$. Each correction is an exact
evolution of an exact initial state; the only defect is at discretization
level (§5).

Decompose every correction into **atoms** $a$ = (pair $(i,j)$, one
snapshot-window slice), with field contribution $\delta_a$. A parallel run
retains an atom set $S$ determined by the dependency DAG (which pairs are
connected) and the correction horizons (how deep into $\mathcal{W}_j$ each
correction extends). Then, exactly,

$$
e_S \;:=\; u^{\mathrm{serial}} - u_S
\;=\; \sum_{a \notin S} \delta_a \;+\; \varepsilon_h ,
\tag{1}
$$

with $\varepsilon_h$ the discretization-level replay defect. *The parallel
error is literally the sum of the neglected corrections.*

## 2. The refinement ladder and the telescoping identity

The self-convergence procedure builds a nested family
$S_0 \subset S_1 \subset \cdots \subset S_K$, where $S_0$ is the production
retention and rung $k$ admits the next *tier* of atoms
$T_k := \{\, \delta_a : a \in S_k \setminus S_{k-1} \,\}$. Two admission
channels are defined:

- **(A) threshold descent** — pairs first connected when the retention
  threshold is tightened to $\varepsilon/\gamma^k$ (full-window corrections;
  audit mode only, see §4);
- **(B) horizon advance** — every connected pair's correction window is
  extended by a fixed increment of $N$ supersegments per iteration
  (default $N = 4$), simulating only the incremental window.

Each rung superposes its tier onto the current iterate, so the measured
inter-iteration shift is

$$
d_k \;:=\; u_{k+1} - u_k \;=\; \textstyle\sum_{a \in T_{k+1}} \delta_a ,
$$

and summing (1) over the ladder gives the **telescoping identity**

$$
e_0 \;=\; d_0 + d_1 + d_2 + \cdots \;+\; \varepsilon_h .
\tag{2}
$$

This is an identity, not an approximation: the true error field of the
production run equals the sum of all present and future measured shifts. The
estimator's headline is the **cumulative** $\lVert u_k - u_0 \rVert$ (per
snapshot, relative $L_2$, max over the source-on snapshot set — the same
population as the accuracy metric), which converges to the true production
error as the ladder exhausts the tiers; the per-rung shifts
$\lVert d_k \rVert$ serve as the convergence/stopping signal.

## 3. Tier decay and the two-sided bound

**Monotone convergence (positivity).** Every neglected correction is the
evolution of a non-negative temperature deviation (the source only adds
heat, and source-off evolution preserves non-negativity). Hence $u_k \uparrow
u^{\mathrm{serial}}$ pointwise: the cumulative estimate
$\lVert u_k - u_0 \rVert$ increases monotonically to the true production
error, and the true error $\lVert e_k \rVert$ decreases monotonically —
the two curves pinch. This is unconditional; the rate is governed by the
following assumption.

**Assumption (A1) — geometric tier decay.** There is $\rho < 1$ with
$\lVert T_{m+1} \rVert \le \rho\, \lVert T_m \rVert$ per snapshot.

*Justification.* Channel (B): tier $m$'s window lies one supersegment duration
$\Delta t_{\mathrm{SS}}$ further from the cut; the neglected source field
decays monotonically in elapsed time (diffusion plus boundary drainage), so
successive windows shrink by the tail-decay factor over $\Delta t_{\mathrm{SS}}$.
Channel (A): tier $m$'s pairs have peak influence in the band
$(\varepsilon/\gamma^m, \varepsilon/\gamma^{m-1}]$; by the calibration map
(near log–log linear with slope $\approx 1$; Table I) per-pair magnitudes fall
by $\approx 1/\gamma$ per tier while the pair count per band grows much more
slowly than $\gamma$ (the influence radius grows only weakly as the threshold
falls).

Under (A1), the triangle inequality applied to (2) gives, per snapshot and
hence for the max over snapshots,

$$
\underbrace{\frac{1 - 2\rho}{1 - \rho}}_{\text{(meaningful for } \rho < 1/2)}
\,\lVert d_0 \rVert
\;\;\le\;\; \lVert e_0 \rVert
\;\;\le\;\; \frac{\lVert d_0 \rVert}{1 - \rho} .
\tag{3}
$$

The first shift is therefore a two-sided estimate of the true rel-$L_2$ with a
bracket controlled by $\rho$ — and $\rho$ is *measurable from the ladder
itself*: $\hat\rho_k = \lVert d_k \rVert / \lVert d_{k-1} \rVert$, yielding
the Richardson-style corrected estimate

$$
\hat E_0 \;=\; \frac{\lVert d_0 \rVert}{1 - \hat\rho}
\qquad\text{and the stopping rule}\qquad
\lVert d_k \rVert \le (1-\hat\rho)\,\tau
\;\Rightarrow\; \lVert e_k \rVert \lesssim \tau ,
\tag{4}
$$

giving self-terminating error *control* at any rank count: iterate until the
shift falls below $(1-\hat\rho)\tau$, and the current iterate meets the target
$\tau$ (down to the floor of §5). No serial reference appears anywhere.

## 4. The horizon-only specialization

**Assumption (A2) — pair-coverage completeness.** Every pair with
non-negligible influence is connected in $S_0$'s DAG; equivalently, the
channel-(A) tiers are subdominant.

Under (A2) the ladder may drop channel (A) entirely ("horizon-only" mode):
each rung is a pure $+N$-supersegment window extension, requiring **no DAG
rebuilds and no influence-lookup tables** — the estimator's planning cost is
zero.
(A2) is not an article of faith: it is exactly what the calibrated DAG
construction provides (every pair with pointwise influence $\ge \varepsilon$
at its elapsed time is connected), and it is *checkable* by running the full
ladder once per configuration and reading the per-channel attribution. For
the validated chord-lookup DAG we measured the channel-(A) contribution at
$\sim 10^{-13}$ (numerical noise) on every rung of the straight, hybrid
spiral–raster, and Bull paths at 31 cuts — including the rung that detected
the hybrid's $3.9\times10^{-4}$ tolerance exceedance, which was pure
channel (B). Configurations with unvalidated DAGs should run the full mode as
an audit.

## 5. The correction floor

The identity (1) carries the defect $\varepsilon_h$: corrections replay the
bridge path's moving-domain trajectory from the source's terminal state,
which differs from the serial march at discretization level (interpolation
and window-shift asymmetries). $\varepsilon_h$ does not shrink as atoms are
added, so it bounds both the achievable refinement and the estimator's
resolution: shifts eventually collapse below the floor while the true error
plateaus at it. Measured floors: $5.8\times10^{-9}$ (straight line, 2 ranks),
$3.4\times10^{-8}$ (hybrid, 32 ranks) — two to four orders below both
tolerance targets, so the estimator governs the entire regime of interest.

## 6. Empirical validation

Validated against serial ground truth (available to us; never used by the
estimator). Primary exhibit: the Bull path at 32 ranks / 31 cuts, tolerance
target $10^{-4}$ — an operating point where the plain parallel configuration
*exceeds* its target — with the horizon ladder at $N=4$ supersegments per
iteration:

| iterate | true max rel-$L_2$ (vs serial) | shift $\lVert u_k - u_{k-1}\rVert$ |
|---|---|---|
| $u_0$ (production) | $1.0767\times10^{-4}$ (over target) | — |
| $u_1$ | $3.6498\times10^{-5}$ | $1.0767\times10^{-4}$ |
| $u_2$ | $1.6113\times10^{-5}$ | $3.6497\times10^{-5}$ |
| $u_3$ | $9.9789\times10^{-6}$ | $1.6113\times10^{-5}$ |
| $u_4$ | $3.3293\times10^{-6}$ | $9.9778\times10^{-6}$ |
| $u_5$ | $8.9734\times10^{-7}$ | $3.3284\times10^{-6}$ |
| $u_6$ | $1.8131\times10^{-7}$ | $8.9678\times10^{-7}$ |

Headline: converged cumulative estimate $1.0767\times10^{-4}$ vs true
production error $1.0767\times10^{-4}$ (ratio 1.000). Three observations:

1. **The shift sequence decays monotonically** — the self-convergence
   signature — mirroring the monotone decay of the true error (§3,
   positivity).
2. **Every shift equals the true error of the previous iterate to 3–5
   significant digits** (compare the shift in row $u_k$ with the truth in row
   $u_{k-1}$): the ladder is a per-iteration error meter, not only a
   convergence indicator.
3. **The ladder repairs the exceedance it detects**: one iteration brings the
   run under its $10^{-4}$ target; six reach $1.8\times10^{-7}$.

Corroboration on other geometries: the straight-line anchor (2 ranks) gives a
digit-exact single-tier estimate ($4.2800\times10^{-5}$ vs true
$4.2800\times10^{-5}$), and the hybrid spiral–raster path at the same 31-cut
operating point yields the identical headline ratio 1.000 (estimated
$3.9433\times10^{-4}$ vs true $3.9430\times10^{-4}$); its iteration profile
reflects its different revisit geometry (a compact spiral-centre plateau
terminating in a dead-end) and is omitted here for brevity.

*(Tables generated by `dev/run_selfcheck_{hybrid,bull}_sbatch.sh`.)*
