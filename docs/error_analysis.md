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
  threshold is tightened to $\varepsilon/\gamma^k$ (full-window corrections);
- **(B) horizon advance** — for every already-connected pair, the correction
  window is extended to
  $\max(\text{DAG}_k\ \text{horizon},\ \text{previous} + 1\ \text{supersegment})$,
  simulating only the incremental window.

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
estimator reports $\lVert d_0 \rVert$ (per snapshot, relative $L_2$, reduced
to max/RMS over the source-on snapshot set — the same population as the
accuracy metric).

## 3. Tier decay and the two-sided bound

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
each rung is a pure $+1$-supersegment extension, requiring **no DAG rebuilds
and no influence-lookup tables** — the estimator's planning cost is zero.
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

The assumptions and bounds above were validated against serial ground truth
(available to us, not required by the estimator):

- **Straight line (2 ranks, 1 cut):** estimate $4.2800\times10^{-5}$ vs true
  $4.2800\times10^{-5}$ — digit-exact ($\hat\rho \approx 10^{-8}$: the first
  tier *is* the error on a chain).
- **Bull, 31 cuts (tol $10^{-4}$):** estimate tracks truth to 3–4 significant
  digits at every iterate while the ladder reduces the true error from a
  *failing* $1.08\times10^{-4}$ to $6.8\times10^{-8}$ over six iterations
  (measured $\hat\rho \approx 0.3$–$0.7$; bracket (3) satisfied at every
  rung).
- **Hybrid, 31 cuts (tol $10^{-4}$):** the production estimate
  $3.943\times10^{-4}$ matches the true $3.943\times10^{-4}$ to four digits —
  a correct no-serial-reference detection of a tolerance violation — and the
  ladder repairs it to the floor within two iterations.

*(Tables generated by `dev/run_selfcheck_{hybrid,bull}_sbatch.sh`; smoke
anchor by the 2-rank straight-line run in the commit history.)*
