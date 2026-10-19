# Information lattice

**Question.** For a supervised task on multichannel recordings, where does the decodable label information live? The candidates are single channels, dependence between channels regardless of lag, the relative timing of channels, and second-order versus higher-order structure. Do current learners extract it? Direction adopted 19 October 2026. [Rationale and positioning](../../docs/research/information-lattice.md).

**Status.** Parked 19 October 2026: judged too narrow to be the paper's question; see [context](../../docs/context/research-reset.md). Kept as an instrument for exact ablation of context content. The LP validity block of Gate 1 ran; see [Gate 1 results](#gate-1-results).

## Cells

Every record is first z-scored per channel. All groups preserve each channel's mean and variance, so this commutes with them. Cells 2–6 apply a Haar-random element of a compact group to each record independently. Each cell therefore carries exactly the information in its maximal invariant: I(Y; gX) = I(Y; T(X)).

| Cell | Operation | Keeps exactly | Removes |
| --- | --- | --- | --- |
| X | none | everything | — |
| CS | common cyclic shift | all but time origin | absolute timing |
| IS | independent cyclic shift per channel | each channel up to rotation, *and* their dependence up to relative lag | relative timing: lag, order, direction |
| G1 | common Fourier phase rotation | cross-periodogram (all circular auto- and cross-covariances) | higher-order structure |
| G0 | independent phase rotation per channel | the channels' periodogram magnitudes, jointly | cross-spectral phase, higher order |
| IO | channel k cropped at an independent offset from a 2T record | as IS, without wrap-around seams; not a group action | as IS |
| XR | channel k taken from a random record of the same class and split | the class-conditional law of each channel | all within-record dependence |
| U0 | XR followed by G0 | each channel's periodogram law | dependence and higher order |

IS and G0 are **not** univariate cells. A coupled pair stays visibly coupled at a scrambled lag: the maximum circular cross-correlation over all lags survives IS. Periodogram magnitudes still co-fluctuate under G0. Only XR (classical within-class trial shuffling) removes dependence, and it is not a function of a single record. Contrasts involving XR or U0 therefore use unpaired tests.

**Contrasts** (a difference of decoding accuracies; positive values mean information is lost by the coarser cell):

| Name | Contrast | Meaning |
| --- | --- | --- |
| dep | IS − XR | lag-free dependence between channels |
| time | CS − IS | relative timing (lag, order, direction) |
| time2 | G1 − G0 | second-order relative timing (cross-spectral phase) |
| high | CS − G1 | higher-order structure, any source |
| highU | XR − U0 | univariate higher-order structure |
| dep2 | G0 − U0 | lag-free second-order dependence (magnitude co-fluctuation) |
| seam | IS − IO | wrap-around leakage of relative timing |

## Gate 1 protocol (pre-specified 19 October 2026, before any run)

**Effects.** Channels 0–1 carry the effect. The other channels are independent unit-variance AR(1), coefficient 0.6. Class 0 replaces the driving source with an independent copy of the same law, so each channel's law is identical across classes for every effect except U2 and U4. θ=0 makes the classes identical; θ ∈ [0, 1] scales strength. The lag is 5 samples.

| Effect | Class difference | Strength at θ |
| --- | --- | --- |
| U2 | ch0 AR(1) coefficient 0.6 vs 0.6+0.2θ, unit variance | univariate second-order |
| U4 | ch0 innovations √(1−θ)·N(0,1) + √θ·(Exp(1)−1) vs Gaussian, same spectrum | univariate higher-order |
| R2 | ch1 = c·ch0(t−5) + √(1−c²)·noise vs independent driver, c = 0.6θ | linear dependence |
| R2dir | ch0 leads ch1 vs ch1 leads ch0, c = 0.6θ, identical channel laws | linear direction only |
| R2dirP | R2dir with AR coefficient 0.95 (persistent; seam stress test) | linear direction only |
| R4env | shared log-normal envelope, depth 1.0θ, AR(0.9), carriers independent | envelope co-modulation |
| R4quad | ch1 = c·(ch0(t−5)² − 1)/√2 + noise, c = 0.6θ | quadratic, zero cross-covariance |
| R4pac | ch1 amplitude (1 + θ·cos φ_slow(t−5)) × fast carrier | phase–amplitude coupling |

**Predicted information** (+ present; 0 exactly absent; ~0 absent up to an O(1/T) cumulant trace in periodogram fluctuations; ~ small, not pre-specified):

| Effect | X | CS | IS | IO | XR | G1 | G0 | U0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U2 | + | + | + | + | + | + | + | + |
| U4 | + | + | + | + | + | ~0 | ~0 | ~0 |
| R2 | + | + | + | + | 0 | + | + | 0 |
| R2dir, R2dirP | + | + | 0 | 0 | 0 | + | 0 | 0 |
| R4env, R4quad, R4pac | + | + | + | + | 0 | ~ | ~ | 0 |

The **hard nulls** are every cell at θ=0; XR and U0 for every R effect; IS, IO, XR, G0 and U0 for R2dir and R2dirP. Under these, any learner's above-chance accuracy indicates a defect or leakage. IS for R2dir and R2dirP is zero only up to the wrap-around seam; the seam contrast tests this.

**Learners** (identical across cells; each fitted only on that cell's training surrogates, one draw per record):

- **LP**, a lattice probe with features chosen to cover the effect library. Per channel: 16 log-spaced log band powers, skewness, excess kurtosis and the time-irreversibility E[(Δz)³]. Per channel pair, for z and z² − 1: the circular cross-correlation at lags −8…8 and its maximum |·| over all lags; the mixed z × (z² − 1) correlations in both orientations at the same lags; and the correlation across frequencies of the log-periodograms. Classifier: standardisation, then LogisticRegressionCV (C ∈ 10^[−3,1], 9 values; 3-fold). LP stands in for the information ceiling and is tailored, not generic.
- **MR**, aeon MiniRocket with 10,000 kernels, then standardisation and RidgeClassifierCV (α ∈ 10^[−3,3], 10 values). This is the standard generic multivariate classifier.
- **CNN**, four dilated Conv1d layers (32 channels, kernel 9, dilations 1/2/4/8, BatchNorm, GELU; receptive field 121 samples), mean and standard-deviation pooling, then a linear head. Training: AdamW (lr 1e−3, weight decay 1e−2), batch 64, at most 30 epochs, early stopping on a stratified 20% validation split (patience 6), CPU.

**Tests.** The test set has n_test = 500 balanced records per dataset. Above chance: exact one-sided binomial test against 0.5. Paired contrasts among X, CS, IS, IO, G1 and G0: exact one-sided McNemar test on the same test records. Contrasts involving XR or U0: one-sided Fisher exact test. α = 0.05. A *rate* is the fraction of independent simulated datasets (seeds) that reject.

**Grid.**

| Block | Learners × seeds | θ | n_train | T | M | Cells |
| --- | --- | --- | --- | --- | --- | --- |
| validity | LP×20, MR×20 | 0, 1 | 1000 | 1024 | 4 | all 8 |
| validity-cnn | CNN×5 | 1 | 1000 | 1024 | 4 | all 8 |
| power | LP×20, MR×10 | 0.125, 0.25, 0.5, 1 | 100, 300, 1000 | 1024 | 4 | CS IS XR G1 G0 U0 |
| scale | LP×20, MR×10 | 0.5 | 300 | 256, 4096 (M=4); 1024 (M=16, 32) | — | CS IS XR G1 G0 U0 |

The power block omits R2dirP and excludes (θ=1, n=1000), which the validity block covers.

**Pass criteria.**

- **P1 calibration.** At θ=0, the above-chance rate pooled over the eight effects is ≤ 0.09 for every cell and learner (160 tests for LP and MR; P(≥15 | p=0.05) ≈ 0.012). The same holds for every contrast.
- **P2 specificity.** At θ=1, every hard-null cell has ≤ 4/20 rejections for LP and MR (P(≥5 | p=0.05) ≈ 0.003), and ≤ 1/5 for CNN.
- **P3 ceiling.** At θ=1, LP detects every + cell in ≥ 16/20 seeds.
- **P4 seam.** For R2dirP, IS stays within P2 and the seam contrast has ≤ 4/20 rejections.

MR and CNN sensitivity is an *outcome* (their extraction profile against the LP ceiling), not a criterion. If P1, P2 or P4 fails, Gate 1 fails and the cell definition or implementation is revised before Gate 2. If P3 fails for an effect, the LP feature set — not the lattice — is revised, and the failure is reported.

## Reproduce

```bash
uv sync --locked
uv run --locked pytest -q tests/test_information_lattice.py
uv run --locked python -m research.information_lattice.gate1 --config research/information_lattice/configs/gate1.toml --block validity --learner lp --output results/information_lattice/gate1
uv run --locked python -m research.information_lattice.analyse_gate1 --input results/information_lattice/gate1
```

The runner writes one JSON file per (block, effect, θ, n, T, M, seed, learner) under `tasks/`. It resumes by skipping completed tasks, and refuses to run against a `manifest.json` from a different source commit, config or lock. Seeds derive from `numpy.random.SeedSequence(20261019, spawn_key=...)`, with separate streams for data, surrogates and the learner.

## Gate 1 results

**Compute policy (user, 19 October):** run nothing locally beyond small tests; experiments run on Gadi. The first, local run (clean commit `74a0ab7`, `results/information_lattice/gate1-261019/`) was stopped once that was stated. Only the LP validity block completed (320/320); LP power is partial (961/1540) and will be rerun on Gadi.

**LP validity (θ ∈ {0, 1}, n=1000, T=1024, M=4, 20 seeds):** P1 calibration 15/15 pass; P2 specificity 18/18 hard nulls pass; P4 seam pass (R2dirP IS 0/20, seam 2/20). P3 has 35/37 pass; it fails only R4env/IS (14/20, accuracy 0.550) and R4env/IO (1/20). The LP feature set barely detects envelope coupling at a scrambled lag. This is a probe limitation, to be revised, not a lattice failure. Confirmed predictions:

- R2 is visible through lag-free dependence (dep 20/20, time 0/20). G0 retains it at 0.967 accuracy via periodogram co-fluctuation, confirming that G0 is not a univariate cell.
- R2dir lives only in timing (time 20/20, time2 20/20, dep 0/20).
- U4 lives only in higher-order univariate structure (highU 20/20).
- R4quad and R4pac live in higher-order dependence (high and dep 20/20; G1 at chance).

Deviations to carry forward:

- R4env's G1 cell is 0.561 (15/20; pre-specified "~"). Second-order sample statistics carry a measurable trace of envelope co-modulation, so "second-order" must be read as information in second-order sample statistics.
- IS − IO is not a pure seam test for dependence effects. IO also loses coupling through partial overlap of its offset crops (R2 IO 0.844). P4 rests on the timing-only R2dirP, as pre-specified.
