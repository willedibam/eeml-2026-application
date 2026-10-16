# Experimental log — SPI vocabulary as an interpretable descriptor basis

Status as of 2026-07-28. Numbers here are measured unless marked otherwise.
Purpose: record what is established, what failed, and why, so conclusions can
be audited without re-deriving them.

---

## 1. Claim, as it has evolved

**Original (EEML abstract).** A learned weight vector `w` over ~125 named
pairwise statistics "identifies which statistics are task-relevant", yielding
an interpretable signature of the coupling that drives a task.

**Current, supported form.** *A linear probe over the vocabulary can be added
at no accuracy cost, and it recovers the generating mechanism at MODULE level.*

Three measurements forced the revision:

- `fixed-spi` (no `w` at all) ties `spi-mpnn` everywhere, including at M=20
  where `top_d=5` retains 26% of edges rather than 56% (1.0000 vs 0.9987,
  n=700). So `w` is **not load-bearing for performance**.
- `w` correlates only **rho = 0.344** (p=3e-9) with each SPI's independently
  measured standalone discriminative power. Its top-10 sit at median utility
  rank 54/282. So `w` is not a univariate relevance ranker.
- Individual SPIs are **not stably selected** (max stability-selection
  frequency 0.73; none reach 0.80). Only modules are.

The probe is therefore a **diagnostic overlay**, not a performance component.
That is weaker than the abstract states, and it is what the evidence supports.

**Framing.** Everything synthetic is *instrument calibration*. R0 shows the
readout is correct when the answer is known; a second regime tests whether it
tracks a change in mechanism. Neither is a discovery. TUH is the only planned
component whose answer is unknown.

---

## 2. Established results

### 2.1 Module-level enrichment (R0-297, K=284, lambda_g=0.01, n=700)

Share of |w| landing on a labelled SPI set, against a permutation null
(GSEA-style, Subramanian et al. 2005). Directedness is MEASURED from the data
(`||A - A^T|| / ||A||`), not assigned by hand.

| set | n | % of \|w\| | enrichment | z |
|---|---|---|---|---|
| M01 (phase-slope, group-delay) | 6 | 29.1% | **13.77x** | 10.4 |
| M05 (parametric GC / TE) | 20 | 36.3% | 5.15x | 6.2 |
| M06 (directed-spectral, largest) | 42 | 8.5% | 0.58x | -1.0 |
| directed | 81 | 68.2% | 2.39x | 4.8 |
| directed & linear | 60 | 65.0% | 3.07x | 5.9 |
| directed & nonlinear | 21 | 3.3% | 0.44x | -0.9 |

Two-sided: nonlinear statistics are actively **depleted** on a linear VAR, and
M06 is depleted despite being the largest directed module.

### 2.2 The readout is not a point pick

Across a **50x lambda range** (0.001–0.05), directed enrichment stays
1.46–2.39x (all z>3), `dir&linear` stays enriched and `dir&nonlinear` never
does, and M01/M05 are significantly enriched at every value. F1 stays
0.89–0.99. This matters because a lasso path is a family of solutions indexed
by lambda; reporting one point as "the signature" is the error.

### 2.3 Which grouping resolution to report

Variance in log|w| explained by group membership, calibrated against permuted
labels (holds group count and size profile fixed):

| scheme | k | eta^2 | z | after removing `axes` |
|---|---|---|---|---|
| families (hand-assigned) | 7 | 0.135 | 9.0 | 0.089 |
| **modules (Cliff et al.)** | 15 | **0.575** | **25.4** | **0.574 (z=27.3)** |
| axes (directed x nonlinear) | 4 | 0.197 | 22.2 | — |

Modules dominate and are near-orthogonal to the axes (AMI ~0.4 between all
pairs). Families add almost nothing beyond the axes and additionally misfile
lagged correlations as "other". **Report modules; keep axes as a coarse
summary; retire families.**

### 2.4 Stability selection (45/50 converged, val F1 0.925)

No individual SPI reaches 0.80 selection frequency (max 0.73). But at module
level, **26 of 284 SPIs (M05 + M01) carry 60.4% of all selections**, while the
two largest modules (M14 71 SPIs, M06 42 SPIs) carry 8.7% and M04/M07/M11 are
never selected. Members are interchangeable; the module is not.

### 2.5 The vocabulary is necessary; structure is necessary

- `node-only` on R0: **0.352** (chance 0.333) — per-channel statistics carry
  nothing.
- Flat logistic regression on permutation-invariant SPI summaries (8 stats x
  282 SPIs): **0.58**, vs GNN 0.97–0.99.
  *Caveat: those summaries aggregate away the matrix structure, so this shows
  structure matters, not that message passing specifically does. A
  structure-aware non-GNN baseline is untested.*

---

### 2.7 TUH (focal vs generalized seizure): NEGATIVE, and cleanly so

The real-data bet did not pay off. Full run, patient-disjoint, 5 seeds, 46/52
patients, 1427 windows (macro F1; chance 0.5):

| n | 20 | 50 | 100 | 200 | 350 |
|---|---|---|---|---|---|
| `node-only` | 0.520 | 0.568 | 0.516 | 0.516 | 0.531 |
| `shuffled` | 0.607 | 0.574 | 0.585 | 0.593 | 0.565 |
| **`spi-mpnn`** | 0.593 | 0.542 | 0.570 | 0.584 | **0.547** |
| `latent-directed` | 0.579 | 0.584 | 0.586 | 0.597 | 0.560 |
| `fixed-spi` | 0.579 | 0.578 | 0.578 | 0.584 | 0.567 |

**Nothing learns.** Every model sits in 0.52-0.61 and none improves with a 17x
increase in training data. For contrast the same model goes 0.67 -> 0.99 on R0
and 0.40 -> 0.67 on R1b over the same range.

**`shuffled` matches or beats `spi-mpnn` at 4 of 5 sample sizes.** So the small
amount of information present is in the marginal DISTRIBUTION of SPI values, not
in which pair carries which value. Pair correspondence -- the thing the method
is about -- contributes nothing here.

**The pre-registered prediction is falsified.** The hypothesis was that focal
seizures (directed propagation from a source) versus generalized (symmetric
bilateral synchrony) would load the signature onto directed measures. Measured:
`directed` **0.88x, z = -2.6** -- significantly DEPLETED. `directed & nonlinear`
0.75x, z = -2.4. The weight instead concentrates on M14 (1.32x, z=5.9) and MXX
(1.29x, z=4.7), and the top six SPIs are ALL precision-matrix estimators
(`prec_GraphicalLassoCV`, `prec_ShrunkCovariance`, `prec_OAS`, ...), i.e. the
most generic contemporaneous linear descriptors in the vocabulary. That is what
a model with no coupling signal falls back on.

**What this does and does not establish.** It is a scope result, not a method
failure: `CLAUDE.md` already states the method works only when classes differ in
HOW channels couple. Focal vs generalized apparently does not, at this patient
count and window length. Confounds that are NOT ruled out and should be stated
rather than used as excuses:
- **8 s windows (T=1024) across 22 channels = 231 pairs.** Estimator variance
  per SPI may swamp any class difference at this length.
- **98 patients, ~8 in the test split.** The estimate is patient-dominated;
  regeneration added 163 windows but **zero new patients**, so more of the same
  corpus cannot fix it.
- Window-level rather than patient-level classification.

**Do not spend more compute here without changing one of those.** Per the
stopping rule, TUH becomes the scope panel and the poster rests on the synthetic
calibration, which is now considerably stronger: two fully gated regimes
(2.5a), the order-specificity result (2.5c), and a measured signature shift in a
pre-registered direction.

## 3. Failures and what they cost

### 3.1 R1 (nonlinear coupling) is confounded — result withdrawn

R1 injected the nonlinearity into the state update:
`child += alpha*(parent^2 - s2)/s2`. That changes each node's effective AR
structure by an amount depending on its in-degree, and the motifs differ there
(chain/fork `[0,1,1]`, collider `[2,0,0]`).

Measured on the raw series with node features alone, no SPIs involved:

| design | chain vs collider | chain vs fork |
|---|---|---|
| R0 linear VAR | 0.540 | 0.535 |
| **R1 nonlinear coupling** | **0.913** | 0.580 |
| R1b linear VAR + x^2 observation | 0.468 | 0.485 |

(chance 0.500). Lag-1 autocorrelation alone reaches 0.848 on R1, and the AC
profile is a motif fingerprint: chain `[0.974, 0.945, 0.821]` vs collider
`[0.948, 0.824, 0.812]`.

R1's apparent headline — signature flipping from `dir&linear` (3.07x) to
`dir&nonlinear` (5.93x), top module M01 -> M03 at 23.9x, z=11.6 — **cannot be
claimed**, because the task was solvable without any pairwise information.
R1 reaching F1 0.9987 at n=100 (easier than R0's 0.9322) was the symptom.

Patching failed: equalising lag-1 AC post-hoc moved 0.93 -> 0.88, because
squaring alters the whole spectrum.

**Replacement (R1b, `var_obs_nonlinear_a`):** keep the LINEAR VAR dynamics and
apply a non-monotone transform at OBSERVATION, identically to every channel and
motif, so it cannot encode the class by construction. Validated at chance.
M/T/instances match R0 exactly so regime is not confounded with problem size.

*Honest limitation:* linear statistics are **mis-specified, not blind** — for
Gaussian latents `corr(x_i^2, x_j^2) = 2*corr^2`. Predict a RELATIVE shift
toward information-theoretic modules, not a reversal.

### 3.2 Two architectural fixes that did not work

Both were attempts to make `w` load-bearing. Recorded as negative results.

- **Interpret `w` from `edge-ablation`** (phi zeroed, so all information must
  pass through `w`): F1 collapses to **0.434**, and the directed fraction of
  |w| *falls* (37% vs 52%). The model stops working rather than becoming
  interpretable.
- **Gate edge features by `w`** (`phi(w * E_ij)`, `--gate-edges`): F1
  **0.669** vs 0.988 ungated, directed fraction 36% vs 57%. Forcing all
  information through `w` costs 0.32 F1 and does not concentrate it on
  directed statistics.

Conclusion: discrimination flows through `phi` over the full vocabulary;
`w` selects topology. This is consistent with `fixed-spi` tying.

### 3.3 Dead ends — things built, measured, and abandoned

Recorded because re-deriving them is expensive and because several look
attractive enough to be tried again.

**Effective-dimension group weighting** (`--group-size effective`, kept in the
code, default off). The sqrt(p_g) group-lasso weight is derived under
within-group ORTHONORMAL design (Yuan & Lin recommend orthonormalising each
group first). This vocabulary violates that badly: the causal family has 48
members spanning ~4.3 independent directions (participation ratio), spectral 52
spanning 7.6. So redundant groups are over-penalised ~3x. Replacing p_g with the
measured effective dimension *did* stabilise the recovered family across lambda
(distance at both 0.002 and 0.01, where raw flipped) — but it did **not** solve
the actual problem, which turned out to be that lambda was simply mis-tuned. The
literature modules plus a lambda path made it unnecessary. Diagnosis correct,
remedy superfluous.

**Binary axes as the primary readout.** After finding the hand families
incoherent, I moved to reporting only `directed x nonlinear`. That was an
over-correction: it is statistically clean but discards the resolution that
makes a 300-statistic vocabulary worth having, and "directed statistics win on a
directed problem" largely restates the Markov-equivalence theorem. Measured
against modules it also loses: eta^2 0.197 vs 0.575. Axes survive as a coarse
summary (z=22 from 4 cells is efficient), not as the deliverable.

**Kuramoto as a symmetric-coupling negative control.** Planned, then dropped. A
separate dataset would change coupling type AND generator AND M AND T AND the
SPI distribution simultaneously, so a null result there would be
uninterpretable. The collider-vs-{chain,fork} binary task is a tighter control
and free: a collider's parents are independent so corr~0 separates it
symmetrically, while chain-vs-fork provably cannot be separated symmetrically —
same instances, same vocabulary, only the labels change (`--class-labels 0 0 1`).

**tanh as the R1 nonlinearity** (the multi-regime spec's suggestion). Measured
to be a weak test: tanh is monotone, so linear GC tracks it nearly as well as a
nonlinear measure (|pearson| 0.734 vs MI 0.389; the corr gap stays under 0.02
even at gain 40). Kept in the generator for completeness, not used. A
non-monotone coupling is required for linear statistics to actually fail.

**GPU training.** Investigated with profiling, rejected. At M=20/K=297/batch=32
the per-graph Python loop is 46% of forward time and issues one `.item()` device
sync PER GRAPH (32 stalls/batch); topk on (20,20) tensors is pure kernel-launch
overhead. More fundamentally the parallelism is ACROSS thousands of independent
fits, not within one, and a gpuvolta node gives 12 CPU cores per GPU — the same
concurrency at a higher SU rate. CPU `normal` queue is correct.

**Patching R1's marginals post-hoc.** Equalising each channel's lag-1
autocorrelation to a common target moved node-feature leakage 0.93 -> 0.88 only,
because squaring alters the whole spectrum rather than just first-order
structure. Abandoned in favour of moving the nonlinearity to the observation
stage, where it is motif-independent by construction.

**Sequential battery execution.** The first battery pinned BLAS to one thread
(correct — many small independent fits) and then ran the ten configurations
one after another, using 1 of 12 cores. Measured on the live job: r0_baselines
alone needed ~6.7 h of an 8 h wall. Fixed by dispatching with `xargs -P`, then
by sharding the two long runs across seeds via `--seed-offset`.

### 3.4 Bugs found (all fixed; each would have corrupted conclusions)

| bug | effect |
|---|---|
| `--group-lambda` never retuned after sqrt(\|g\|) normalisation | ~7x over-penalty; F1 0.627 vs 0.973 at n=100; recovered family flipped to `linear` |
| early stopping active during LR warmup (patience 20 < warmup 60) | runs halted with `spi_w ~ 0`; one seed at F1 0.31 vs 0.96 siblings |
| `_dataset_complete` required `calc.csv` | `--skip-existing` would recompute every dataset when resuming a `--no-csv` run |
| sample-efficiency pool keyed on LABEL | merged classes would sample the same instances twice |
| average-linkage clustering for empirical modules | 79% of SPIs in one cluster; switched to ward |
| non-converged fits voting in stability selection | flattened every frequency toward noise |
| `PARQUET` not forwarded through `qsub -v` | 3.2 MB/dataset of unused output (~90% of storage) |
| TUH montage confounded with class | `02_tcp_le` 73% FNSZ vs `01_tcp_ar` 44%; manifest now single-montage |

### 3.5 Process errors worth recording

- lambda was tuned twice on a single point of the n-curve and was wrong both
  times (0.002 chosen at n=100, 0.01 at n=400). There is a genuine
  accuracy/signature trade-off across n; select on both.
- A commit was pushed with a failing regression test. The test caught the
  lambda regression it was written for.
- Cluster jobs were sized from a 9-worker probe and run at 48 workers;
  per-dataset cost is ~3.5x higher under memory-bandwidth contention
  (850 s -> 3000 s). Per-node throughput is bandwidth-bound and roughly fixed,
  so the only lever is more nodes.

---

## 4. Open questions

| question | status |
|---|---|
| Does R1b show a shift toward nonlinear modules? | generation running |
| Does a fair latent model (temporal encoder) match the vocabulary? | queued; **never yet run on a full n curve** |
| Is message passing needed, or only structure? | untested (flat baseline is lossy) |
| Does any of this transfer to real data? | TUH; conversion code untested against real EDFs |

---

### 2.5a R1b: a SECOND valid regime, fully gated (2026-07-29)

R1b (`var_obs_nonlinear_a`: linear VAR observed through `x^2`) is established.
The first pass was void -- at chance, with no controls and lambda left at the
R0-tuned 0.01. At **lambda=0.001** with the full control panel (n=700, 3 seeds):

| model | F1 |
|---|---|
| `spi-mpnn` | **0.6727** +/- 0.098 |
| `fixed-spi` | 0.6837 +/- 0.164 |
| `node-only` | 0.3304 +/- 0.02 |
| `shuffled` | 0.3066 +/- 0.02 |
| `latent-directed` | 0.3071 +/- 0.03 |

Chance is 0.333. `validity_report` returns **INTERPRETABLE** -- the first
regime to pass every gate in one file.

Three things this establishes that R0 does not:
- **No marginal leakage** (`node-only` at chance), unlike withdrawn R1.
- **Pair correspondence is necessary** (`shuffled` at chance at every n),
  unlike R0 where `shuffled` climbs to 0.82 by n=1000.
- **The vocabulary is necessary.** `latent-directed` sits at chance at every n
  while the SPI probe reaches 0.67. A capacity-validated latent model on raw
  series cannot do this task at all.

Independent confirmation of the accuracy: lambda=0.0002 gave 0.6721 and
lambda=0.001 gave 0.6727, from separate runs.

**The pre-registered prediction held.** `nonlinear` enrichment 1.46x (z=3.5);
top modules **M09 4.26x (z=4.7)** and **M10 3.05x (z=5.9)**, both DTW/LCS
families; the top six SPIs are all nonlinear (`bary-sq_dtw_mean`, `gwtau`,
`lcss`, `dtw`). Against R0, where M05 (parametric Granger) and M01
(phase-spectral) lead and both are LINEAR. Cross-regime, measured against three
R0 lambdas: `directed & linear` **falls in 3/3** (-14 to -28pp, CI excludes 0
each time), `directed & nonlinear` rises in 2/3. Stability is better than R0
too: `top1-SPI = 1.00`, all three seeds agreeing on `bary-sq_dtw_mean`.

**The caveat, and it matters.** 0.6727 is suspiciously close to 2/3, and the
report warns `little enrichment -- the model is not preferentially using
directed statistics` (directed 1.27x, z=2.1). The DTW/LCS measures carrying the
signature are SYMMETRIC. So the likely reading is that `x^2` renders the
directional information unusable and the probe falls back to an undirected
solution, which is ceiling-limited. That would explain the weak directed
enrichment, the symmetric-measure dominance and the plateau simultaneously.

Not yet resolved: accuracy is still rising (n=400 -> 0.528, n=700 -> 0.673), so
it may exceed 2/3 with more data. **The decisive test is cheap**: re-run R1b on
the UNDIRECTED sub-vocabulary only. If that also reaches ~0.67, the probe is not
using direction and the claim becomes "recovers THAT channels couple, not which
way, once direction is obscured" -- a scope boundary with a mechanism, weaker
than mechanism recovery but honest and publishable.

### 2.5b Parametric beats nonparametric WITHIN the directed-spectral class

The strongest synthetic result, and the one that is not the tautology.

Measured across **10 independent lambda runs** on R0-297 (two separate sweeps,
lambda 0.001-0.05, a 50x range):

| module | what it contains | K | enrichment (median) | range |
|---|---|---|---|---|
| M01 | CoherencePhase, PhaseSlopeIndex, GroupDelay, PhaseLagIndex (band) | 6 | **6.98x** | 3.12-14.16, enriched in 10/10 |
| M05 | **parametric** spectral GC, TE-gaussian, TE-symbolic, phi*, SI-gaussian | 20 | **4.41x** | 3.20-6.63, enriched in 10/10 |
| M06 | **nonparametric** directed spectral: DTF, DC, PDC, GPDC, dDTF, sGC-nonparametric | 42 | **0.54x** | 0.27-1.45, depleted in 9/10 |

M05 and M06 are **both directed and both spectral**. A directed-vs-undirected
analysis lumps them together; the probe separates them, and does so consistently
across a 50x lambda range and two independent sweeps. 42 directed-spectral SPIs
are actively *depleted* -- they carry less weight than their share of the
vocabulary.

**Why this is a claim and not a restatement of the generator.** "Granger wins on
a VAR" is entailed. "Among estimators that all target directed spectral
influence, the *correctly specified parametric* ones dominate and the
factorization-based nonparametric ones are depleted" is a statement about
**estimator efficiency under correct specification** -- R0 is a VAR(1), so the
parametric AR model is correctly specified, and parametric spectral GC estimates
its coefficients directly where DTF/PDC/nonparametric-sGC go through spectral
density factorization. That is a statistical fact the probe recovered, not a
physics fact built into the generator.

**It is falsifiable.** On a generator that is not an AR process, or where AR
order is misspecified, M06 should stop being depleted relative to M05, or the
ordering should flip. That prediction is the reason a second valid regime
matters.

**Caveats, stated.** M01's enrichment varies 3.12-14.16 (K=6, high variance) --
its rank is not identified. M06 is enriched (1.45x) at the smallest lambda
tested, so the depletion is 9/10, not universal. And this is one dataset: it
describes R0 until a second regime tests the prediction.

### 2.4b The 2/3 ceiling: two DIFFERENT ceilings, do not conflate them

I previously recorded that "both earlier claims were wrong" and that chain and
collider are the confusable pair. That entry was itself wrong: it generalised a
measurement of ONE symmetric statistic into a claim about all of them.

**Under a general symmetric statistic (lagged included).** Chain (0->1->2) and
fork (0->1, 0->2) are Markov-equivalent: same unlabelled skeleton, no
v-structure, and identical weighted skeletons {alpha, alpha, alpha^2}. The
collider (1->0, 2->0) has {alpha, alpha, 0} because its parents are uncoupled.
So chain and fork collapse, collider separates, and the ceiling is **2/3**.
This is the abstract's claim (Verma & Pearl 1990) and it is correct.

**Under CONTEMPORANEOUS Pearson specifically.** The VAR has no self-persistence
(A has a zero diagonal), so X_0[t] is independent of X_0[t-1] and a directly
coupled pair has ~zero INSTANTANEOUS correlation. Only the fork produces
instantaneous coupling, because its two children are driven by the same parent
value. Measured over 400 instances per motif at M=3, sorted |rho| triples:

| motif | min | mid | max |
|---|---|---|---|
| chain | 0.015 | 0.032 | 0.056 |
| collider | 0.015 | 0.032 | 0.058 |
| fork | 0.020 | 0.048 | 0.206 |

Separability from that triple alone: chain vs collider **0.529** (chance 0.5),
chain vs fork 0.840, 3-way **0.582**. So under contemporaneous Pearson it is
CHAIN and COLLIDER that collapse -- the opposite pair -- and the achievable
accuracy is 0.582, well BELOW the general 2/3 ceiling.

**Both are true, at different scopes**, and the pipeline confirms it: the
`correlation` baseline uses contemporaneous Pearson |r| and measures 0.59,
matching the 0.582 predicted here rather than the 2/3 bound. The abstract
already says this ("the gap between ceiling and observed 59% arises because
top-d sparsification selects high-correlation nuisance pairs"); the measurement
here gives a second and simpler reason -- contemporaneous Pearson is a strictly
weaker symmetric statistic than the bound assumes.

For any figure: annotate 2/3 as the SYMMETRIC-STATISTIC ceiling and 0.59 as what
contemporaneous Pearson actually achieves. Do not present 0.582 as "the"
ceiling.

### 2.5a R1b: a SECOND valid regime, fully gated (2026-07-29)

R1b (`var_obs_nonlinear_a`: linear VAR observed through `x^2`) is established.
The first pass was void -- at chance, with no controls and lambda left at the
R0-tuned 0.01. At **lambda=0.001** with the full control panel (n=700, 3 seeds):

| model | F1 |
|---|---|
| `spi-mpnn` | **0.6727** +/- 0.098 |
| `fixed-spi` | 0.6837 +/- 0.164 |
| `node-only` | 0.3304 +/- 0.02 |
| `shuffled` | 0.3066 +/- 0.02 |
| `latent-directed` | 0.3071 +/- 0.03 |

Chance is 0.333. `validity_report` returns **INTERPRETABLE** -- the first
regime to pass every gate in one file.

Three things this establishes that R0 does not:
- **No marginal leakage** (`node-only` at chance), unlike withdrawn R1.
- **Pair correspondence is necessary** (`shuffled` at chance at every n),
  unlike R0 where `shuffled` climbs to 0.82 by n=1000.
- **The vocabulary is necessary.** `latent-directed` sits at chance at every n
  while the SPI probe reaches 0.67. A capacity-validated latent model on raw
  series cannot do this task at all.

Independent confirmation of the accuracy: lambda=0.0002 gave 0.6721 and
lambda=0.001 gave 0.6727, from separate runs.

**The pre-registered prediction held.** `nonlinear` enrichment 1.46x (z=3.5);
top modules **M09 4.26x (z=4.7)** and **M10 3.05x (z=5.9)**, both DTW/LCS
families; the top six SPIs are all nonlinear (`bary-sq_dtw_mean`, `gwtau`,
`lcss`, `dtw`). Against R0, where M05 (parametric Granger) and M01
(phase-spectral) lead and both are LINEAR. Cross-regime, measured against three
R0 lambdas: `directed & linear` **falls in 3/3** (-14 to -28pp, CI excludes 0
each time), `directed & nonlinear` rises in 2/3. Stability is better than R0
too: `top1-SPI = 1.00`, all three seeds agreeing on `bary-sq_dtw_mean`.

**The caveat, and it matters.** 0.6727 is suspiciously close to 2/3, and the
report warns `little enrichment -- the model is not preferentially using
directed statistics` (directed 1.27x, z=2.1). The DTW/LCS measures carrying the
signature are SYMMETRIC. So the likely reading is that `x^2` renders the
directional information unusable and the probe falls back to an undirected
solution, which is ceiling-limited. That would explain the weak directed
enrichment, the symmetric-measure dominance and the plateau simultaneously.

Not yet resolved: accuracy is still rising (n=400 -> 0.528, n=700 -> 0.673), so
it may exceed 2/3 with more data. **The decisive test is cheap**: re-run R1b on
the UNDIRECTED sub-vocabulary only. If that also reaches ~0.67, the probe is not
using direction and the claim becomes "recovers THAT channels couple, not which
way, once direction is obscured" -- a scope boundary with a mechanism, weaker
than mechanism recovery but honest and publishable.

### 2.5b Parametric beats nonparametric WITHIN the directed-spectral class

The strongest synthetic result, and the one that is not the tautology.

Measured across **10 independent lambda runs** on R0-297 (two separate sweeps,
lambda 0.001-0.05, a 50x range):

| module | what it contains | K | enrichment (median) | range |
|---|---|---|---|---|
| M01 | CoherencePhase, PhaseSlopeIndex, GroupDelay, PhaseLagIndex (band) | 6 | **6.98x** | 3.12-14.16, enriched in 10/10 |
| M05 | **parametric** spectral GC, TE-gaussian, TE-symbolic, phi*, SI-gaussian | 20 | **4.41x** | 3.20-6.63, enriched in 10/10 |
| M06 | **nonparametric** directed spectral: DTF, DC, PDC, GPDC, dDTF, sGC-nonparametric | 42 | **0.54x** | 0.27-1.45, depleted in 9/10 |

M05 and M06 are **both directed and both spectral**. A directed-vs-undirected
analysis lumps them together; the probe separates them, and does so consistently
across a 50x lambda range and two independent sweeps. 42 directed-spectral SPIs
are actively *depleted* -- they carry less weight than their share of the
vocabulary.

**Why this is a claim and not a restatement of the generator.** "Granger wins on
a VAR" is entailed. "Among estimators that all target directed spectral
influence, the *correctly specified parametric* ones dominate and the
factorization-based nonparametric ones are depleted" is a statement about
**estimator efficiency under correct specification** -- R0 is a VAR(1), so the
parametric AR model is correctly specified, and parametric spectral GC estimates
its coefficients directly where DTF/PDC/nonparametric-sGC go through spectral
density factorization. That is a statistical fact the probe recovered, not a
physics fact built into the generator.

**It is falsifiable.** On a generator that is not an AR process, or where AR
order is misspecified, M06 should stop being depleted relative to M05, or the
ordering should flip. That prediction is the reason a second valid regime
matters.

**Caveats, stated.** M01's enrichment varies 3.12-14.16 (K=6, high variance) --
its rank is not identified. M06 is enriched (1.45x) at the smallest lambda
tested, so the depletion is 9/10, not universal. And this is one dataset: it
describes R0 until a second regime tests the prediction.

### 2.4b The 2/3 ceiling: CHAIN and COLLIDER are the confusable pair (measured)

Asserted two different wrong ways before being measured. The correct statement
is empirical, and it explains the `correlation` baseline exactly.

Sorted |rho| triples over 400 instances per motif at M=3 (sorted, so
label-invariant -- motif nodes are randomly permuted, so a symmetric statistic
sees only the multiset):

| motif | min | mid | max |
|---|---|---|---|
| chain | 0.015 | 0.032 | **0.056** |
| collider | 0.015 | 0.032 | **0.058** |
| fork | 0.020 | 0.048 | **0.206** |

Separability from that triple alone (logistic regression, 5-fold CV):

| pair | accuracy | chance |
|---|---|---|
| chain vs collider | **0.529** | 0.5 |
| chain vs fork | 0.840 | 0.5 |
| fork vs collider | 0.841 | 0.5 |
| 3-way | **0.582** | 0.333 |

**Mechanism.** The VAR has no self-persistence (`A` has zero diagonal), so
`X_0[t]` is independent of `X_0[t-1]` and a directly-coupled pair has ~zero
CONTEMPORANEOUS correlation. Only the fork produces instantaneous coupling,
because its two children are driven by the same parent value. Chain and
collider therefore look alike to any contemporaneous symmetric statistic, and
separating them requires lag or direction.

The measured 3-way 0.582 matches the `correlation` baseline's 0.59 to within
noise -- the baseline is not underfitting, it is at its information ceiling.

**Both earlier claims were wrong.** "Chain and fork are Markov-equivalent" is
true for conditional independence but not what a contemporaneous statistic sees;
"fork and collider share a skeleton" is true of the labelled skeletons but node
labels are randomised, so the skeleton is not observable. The
CLAUDE.md statement about Kuramoto is a different and correct claim: there the
GENERATOR is symmetric, so fork and collider are literally the same process.

This also explains `shuffled` reaching 0.82 at n=1000 (3.1): shuffling permutes
values across pairs but preserves each SPI's multiset, and for DIRECTED lagged
SPIs the multisets do differ between chain and collider ({a, a, a^2} vs
{a, a, 0}), so the vocabulary retains chain/collider information that Pearson
alone does not.

### 2.5c The probe recovers the correct AR ORDER (controlled, 10/10)

The sharpest anti-tautology evidence in the project, and it needs no new data.

pyspi's vocabulary contains `sgc_parametric` at several model orders with
everything else matched -- same estimator, same statistic (mean/max), same
frequency bands, **6 SPIs per arm**. The only varying factor is the order of the
AR model the estimator fits. Measured across the same 10 lambda runs, as mean
|w| relative to the vocabulary average (1.0 = a typical SPI):

| arm | median |w| | vs order-20 |
|---|---|---|
| `sgc_parametric` **order-1** (correct for a VAR(1)) | **5.22x** | -- |
| `sgc_parametric` order-None (auto) | 4.45x | -- |
| `sgc_parametric` **order-20** (20x overparameterised) | **0.65x** | order-1 wins **10/10**, median **8.63x** |
| `sgc_nonparametric` | 0.64x | order-1 wins **10/10**, median **9.06x** |

Note this also corrects 2.5b: the M05/M06 contrast is NOT parametric vs
nonparametric. M06 contains `sgc_parametric_*_order-20`, i.e. the same estimator
at the wrong order. The within-estimator table above is the clean version and
the module-level one is confounded by M06's 36 non-sGC members.

**Why this is discovery, not confirmation.** "Granger wins on a VAR" is entailed
by the generator. Nothing in the generator encodes a preference over *model
order*: order-1 is correctly specified for a VAR(1), order-20 pays a variance
cost for 20x the parameters. The probe recovered a **statistical efficiency**
property of the estimator, not the physics property built into the data. And it
is matched on estimator, statistic and band, so it cannot be a module-size,
directedness or linearity artifact -- the three confounds that weakened every
earlier claim.

**The experiment this demands.** On VAR(p) data with p > 1 the preferred order
should track p. A dose-response curve (true order 1, 2, 3, 5, 10 vs recovered
order preference) would turn a single observation into a calibration curve, and
a flat preference for order-1 regardless of true p would falsify it. Same
generator, one changed parameter, same cost as R0. This is the highest-value
synthetic experiment remaining.

### 2.6 The vocabulary: ~125/136 SPIs retired, ~297 is the standard

The EEML abstract used `configs/pyspi/eeml.yaml` (~136 SPIs, ~125 after
filtering). Everything from R0-297 onward uses
`configs/pyspi/benchmarked90_amortized_config.yaml`: **297 SPIs**, the p90
amortized-cost cut of pyspi's 328 (cutoff: fastest kept 5.52 s, slowest dropped
6.79 s, benchmarked at M=16/T=800), plus two `te_kraskov` DCE variants added
back by hand for methodological coverage. After robust-scale filtering, runs
report K = 283-284.

Why it replaced the small config, not merely supplemented it:
- **Cheaper per SPI and broader.** It drops only the 31 slowest (`hhg` 506 s,
  `ccm` ~332 s, `gpfit` 411 s) and keeps more than twice as many statistics.
  Bigger vocabulary at lower cost -- this is what makes larger-M, more numerous
  datasets affordable.
- **Module coverage.** M01-M14 are all populated at 297; the small config left
  several modules with too few members for module-level enrichment to have a
  meaningful permutation null.
- **The reported numbers are all at 297.** eta^2 = 0.575 for module identity,
  the M01/M05/M06 enrichments, lambda-robustness over 50x.

Consequences to respect: `--group-lambda` **must** be re-tuned when K changes
(more groups shifts the sqrt(|g|) penalty scale); 0.005 at K~125, ~0.01 at
K~284. Legacy result JSONs at K~125 are not comparable dimension-for-dimension
and should not be pooled with 297 runs.

The identical file exists in both repos (`configs/pyspi/` here,
`configs/pyspi-v2/` in the generator repo); **verified byte-identical
2026-07-28**. Generation uses the generator repo's copy.

**pyspi is a fork, pinned.** `~/pyspi-fork` branch `v2`, commit `703c73a`
(`willedibam/pyspi`), editable-installed in the generator venv -- confirmed via
`pyspi.__file__` on the running job. It carries estimator **correctness** fixes,
not only speed work: kernel-entropy NORMALISE scale term, Theiler-MI counting,
KSG-AIS auto-embed for kraskov TE, plus the cache amortization the benchmarked
config's cost model depends on. SPI values computed against upstream PyPI pyspi
are therefore *different numbers*, not just slower ones. Any reproduction must
use this fork. (Verified: commit subjects and import path. Not verified: the
diffs themselves.)

## 5. Current state (2026-07-28, update when it changes)

**Jobs in flight** (all batch; nothing depends on an SSH session)

| job | what | status |
|---|---|---|
| `eeml_baselines` | validity gates + latent-vs-vocabulary on R0/R1 | running |
| `regime` x3 | R1b lambda sweep (0.0002, 0.005) + control/upper-bound panel (0.001) | running |
| `tuh_gen` array x14 | TUH pyspi generation, **all 322 sessions**, CHUNK=24 | queued |

**TUH staging is complete and verified** (2026-07-28): 322 EDFs at
`/scratch/ql44/we2614/tusz/edf`, 0 missing against the manifest, 0 outside the
`01_tcp_ar` montage, 0 truncated, 161 fnsz / 161 gnsz.

**The pilot gate was dropped, deliberately.** The plan was to withhold full
generation until the 114-session pilot's `node-only` control passed. That gate
was unsatisfiable as designed: `CHUNK=8` takes the first 8 manifest rows, which
are all fnsz, so the pilot could not produce a two-class dataset and could never
have run a control. With 138 kSU of 151 available and full generation costing
~2-3.6 kSU (<3%), generating everything and gating the *interpretation* on the
controls is cheaper than another pilot round-trip. The controls still gate every
conclusion -- `tuh_train.pbs` runs and reports them first.

**Outputs to collect**
```
logs/r1b_report.txt                    # R1b enrichment
eeml_baselines.o*  (BASELINES block)   # latent vs vocabulary
logs/tuh_report.txt                    # TUH battery -- READ CONTROLS FIRST
```

**Paths**
| what | where |
|---|---|
| eeml repo (Gadi) | `/scratch/ql44/we2614/eeml-2026-application` |
| generator repo (Gadi) | `~/mts-spi-study-cluster` (branch `refactor-lagged-warping`) |
| synthetic data | `/scratch/ql44/we2614/mts-spi-data/{260727_r0_297,260728_r1b_obs}` |
| staged EDFs | `/scratch/ql44/we2614/tusz/edf` (322, verified) |
| TUH output | `/scratch/ql44/we2614/mts-spi-data/260728_tuh` |

**Operational lessons**
- Anything over a few minutes goes in the queue. `run_regime.sh` run
  interactively on a login node died with the SSH session and lost a training
  run; NCI also reaps long login-node processes.
- Staging belongs on `copyq` (internet AND batch durability), not tmux.
- `qsub -v` splits on COMMAS, so a comma inside a value breaks submission with
  an unhelpful message. `regime.pbs` takes `+` instead.

**Decided, do not relitigate**
- Report literature modules; families are retired (eta^2 0.575 vs 0.135).
- The vocabulary is the **297-SPI benchmarked config** (2.6). The ~125/136-SPI
  abstract config is retired; do not mix K~125 and K~297 results.
- `w` is a diagnostic overlay, not a performance component.
- Skip Kuramoto; the collider-vs-rest binary task is tighter and free.
- After R1b, stop adding synthetic regimes and go to TUH.

**Next decisions, gated on the pending outputs**
1. TUH `node-only` clears chance -> coupling claim void there; report the scope
   limit, do not generate the remaining 208 sessions.
2. `latent-directed` ties `spi-mpnn` at every n -> drop sample-efficiency
   language; interpretability is the whole contribution.
3. R1b shows no shift -> the method reads directedness but not mechanism type;
   publishable as a limitation.

**R1b, first pass: at chance and uninterpretable.** F1 0.347/0.306/0.415 at
n=100/400/700 against 0.333 chance, non-monotonic. The enrichment printed
underneath it (M05 4.66x, z=11.7) is void: a signature from a classifier at
chance is group-lasso geometry, not recovered mechanism. Three things were
missing and all three were process failures, not data failures --
`run_regime.sh` hardcoded `--models spi-mpnn`, so there was **no control and no
upper bound**, and lambda was left at the R0-tuned 0.01.

The live hypothesis is over-regularisation, not an unsolvable task: R1b's `x^2`
observation squares the coupling (`alpha` 0.2 -> effective 0.04), so R1b is R0
at a fraction of the effect size, and the reported |w| RMS values (M05 0.0046,
all others 0.0006-0.0012) are consistent with `w` crushed toward zero, which
degenerates the adjacency to a constant. Unverified until the sweep returns.
This repeats a documented past mistake: the un-retuned lambda after the sqrt(|g|)
change cost F1 0.627 vs 0.973.

---

## 6. Reproducing

```bash
# regime report: integrity, accuracy, 2x2 + module enrichment, stability, CIs
bash docs/run_regime.sh <data-dir> <c1,c2,c3> <tag> [lambda] [seeds]

# grouping resolution comparison
PYTHONPATH=. python docs/compare_resolutions.py <results.json>

# does the probe rank statistics by standalone utility?
PYTHONPATH=. python docs/probe_vs_utility.py <data-dir> <c1,c2,c3> <results.json>

# validity audit over every result JSON (seconds, no compute)
PYTHONPATH=. python docs/validity_report.py results/*.json

# paired cross-regime signature shift (only if BOTH runs clear chance)
PYTHONPATH=. python docs/compare_regimes.py results/<ref>.json results/<test>.json

# regression guard (~30 s) — run before any commit touching train/model/CLI
python tests/test_regression.py
```

Result JSONs carry `hyperparameters`, `spi_names`, `spi_families`,
`spi_asymmetry` and per-seed `learned_w`, so every analysis above reproduces
from the JSON alone.

---

## 7. The validity protocol (candidate contribution)

Most interpretability methods report a signature and offer no test that the
signature means anything. This project has accumulated one, largely by being
burned. Stated as a protocol rather than a list of incidents, it is a
contribution in its own right — and unusually, it has **rejected three of its
own results**, which is the evidence that it is not decorative.

### 7.1 The gates

| gate | what it asks | void if |
|---|---|---|
| `node-only` | can node features alone (mean, std, lag-1 autocorr, dominant FFT mag) do it? | clears chance → classes differ in what channels DO, not how they couple; vocabulary mismatched |
| `shuffled` | are SPI values permuted across pairs enough? | clears chance → signal is the marginal distribution of SPI values, not pair correspondence |
| at-chance | is the primary model itself above chance? | not separated from chance → the signature is group-lasso geometry, not mechanism |
| Markov equivalence | is the discriminating statistic symmetric? | chain/fork indistinguishable under undirected statistics (Verma & Pearl 1990) → hard 2/3 ceiling |
| permutation null | is module enrichment larger than under permuted `w`? | not exceeded → enrichment is a size artifact |
| stability selection | is the same SPI chosen across seeds? | low → within-family specificity unidentified (collinearity; Zhao & Yu 2006) |

The first three are empirical and now audited automatically by
`docs/validity_report.py`. The fourth is a **proof**, not a measurement.

### 7.1b Signature stability under lambda, measured

The sharpest attack on a learned-weight claim is "you tuned lambda until the
answer looked right". Measured over 10 runs spanning 50x
(`docs/lambda_path_signature.py`):

- **M05 is top-1 in 9/10 runs.** The exception is a near-tie (M01 30% vs M05
  27%).
- Module-share rank correlation across the path: mean rho +0.86, min +0.74.
- **Ranks below 1 are NOT identified**: the top-2 set is {M01, M05} in only 6/10,
  and the top-3 set takes 6 distinct values across 10 runs.

So the reportable unit is the enrichment of a module against a permutation null,
not a ranking of modules against each other. The M05-enriched / M06-depleted
contrast (2.5b) holds in 10/10 and 9/10 respectively; "M05 is the top module"
holds in 9/10 and should be stated with that number attached.

### 7.2 Cases where it rejected our own results

1. **R1 (`var_nonlinear_a`) withdrawn.** F1 0.9987, better than R0 — backwards
   for a supposedly harder regime. `node-only` scored **0.913** (chance 0.5):
   the nonlinearity sat in the state update, so each node's AR structure varied
   with its in-degree, and the motif leaked through the marginals. The task was
   solvable with no pairwise information at all.
2. **Kuramoto abandoned before spending compute.** Fork and collider are
   Markov-equivalent under symmetric coupling, capping any undirected method at
   2/3. Structural, so no experiment was needed.
3. **R1b's first report void.** Module M05 at 4.66×, z=11.7, P(top)=1.00 — from
   a classifier at F1 0.4149 ± 0.0728 against 0.333 chance. The enrichment
   machinery works fine on a model that learned nothing.

Also caught by the same instinct: the abstract's "causal family carries ~3.5×
the L2 norm" was **size-confounded** (48 of ~125 SPIs) and is replaced by
enrichment against a permutation null, which has a null distribution and is not
size-dependent.

### 7.3 Why the at-chance test is statistical, not a margin

R1b at n=700 scored 0.4149 against 0.333. A fixed margin of 0.05 calls that
"above chance". The per-seed SD is 0.0728, so it is well inside two SDs of
chance. `validity_report.py` requires **both** `mean - 2*sd > chance` and
`mean > chance + margin`; on the real R1b numbers a margin-only rule returned
INTERPRETABLE and the statistical rule returns VOID.

### 7.4 What is missing before this is publishable as a contribution

- The gates are **necessary, not sufficient**. Passing all of them does not
  establish that `w` recovers mechanism; R0 passes and is still tautological.
- No **false-negative** characterisation: no case where the protocol wrongly
  voided a good result. Without that, its specificity is unmeasured.
- `UNGATED` was the normal state until now — `run_regime.sh` hardcoded
  `--models spi-mpnn`, so regime reports had no gates at all. The protocol
  existed in `baselines.pbs` and in judgement, not in the default path.
