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

### 3.3 Bugs found (all fixed; each would have corrupted conclusions)

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

### 3.4 Process errors worth recording

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

## 5. Reproducing

```bash
# regime report: integrity, accuracy, 2x2 + module enrichment, stability, CIs
bash docs/run_regime.sh <data-dir> <c1,c2,c3> <tag> [lambda] [seeds]

# grouping resolution comparison
PYTHONPATH=. python docs/compare_resolutions.py <results.json>

# does the probe rank statistics by standalone utility?
PYTHONPATH=. python docs/probe_vs_utility.py <data-dir> <c1,c2,c3> <results.json>

# regression guard (~30 s) — run before any commit touching train/model/CLI
python tests/test_regression.py
```

Result JSONs carry `hyperparameters`, `spi_names`, `spi_families`,
`spi_asymmetry` and per-seed `learned_w`, so every analysis above reproduces
from the JSON alone.
