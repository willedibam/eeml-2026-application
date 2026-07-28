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
