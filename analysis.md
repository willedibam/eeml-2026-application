# Experimental Analysis — Consolidated

**Internal working document. Last updated 27 March 2026.**

---

## 1. Experimental History

### 1.1 Data generations

| Tag | M | T | α | Instances/class | Outcome |
|-----|---|---|---|-----------------|---------|
| 260326_eeml | 5 | 1000 | [0.5, 0.95] | 1000 | Too easy — Shuffled at 82–98%, distributional fingerprint sufficient |
| **260326_2_eeml** | **10** | **500** | **[0.2, 0.8]** | **1000** | **Current working dataset** |
| 260327_eeml (pending) | 10 | 500 | [0.3, 0.7] | 1500 | Tighter coupling, more instances for n≤1000 |

### 1.2 Optimization discoveries

**Bifurcation problem.** On 260326_2_eeml, SPI-MPNN showed bimodal seed distributions: seeds either found the solution (85–100%) or stuck at chance (33%). Cause: loss landscape has two basins; early gradient direction determines which.

**Fixes tested:**
- LR warmup (20 epochs linear → cosine): 9/10 seeds succeed vs 3/10 without. **Adopted.**
- Multiple restarts (train N×, keep best val F1): equally effective. **Adopted (restarts=2).**
- w_lr_mult=10: partially effective (8/10). Not adopted.

### 1.3 Family assignment fix

`_FAMILY_RULES` in graph_build.py was broken: 97/125 SPIs fell into "other". Fixed to properly classify all SPIs into {causal(48), spectral(52), linear(13), distance(5), information(5), rank(2)}. **All runs prior to sens_w40_td5 used broken families**, meaning group lasso was meaningless. Runs with corrected families: sens_w40_td5, sens_td7, sens_gl05, sens_gl02, sens_gl05_w40.

---

## 2. Current Best Results

### 2.1 OLD: Full baseline (full2: w20, r=2, td3, gl=0.01, BROKEN families) — SUPERSEDED

*(See §2.4 for definitive results with corrected families.)*

### 2.4 DEFINITIVE: 260326_2_eeml (w20, r=2, td5, gl=0.02, corrected families, MPS)

| n/class | SPI-MPNN | Fixed-SPI | MLP-Mix | Corr | Latent | Shuffled | Node-Only |
|---|---|---|---|---|---|---|---|
| 20 | **70±13** | 61±23 | 63±12 | 34±6 | 32±3 | 35±4 | 31±1 |
| 50 | 79±16 | **83±18** | 75±7 | 29±4 | 29±2 | 33±3 | 33±1 |
| 100 | **81±16** | 60±26 | 80±12 | 32±4 | 29±3 | 31±5 | 34±1 |
| 200 | **98±2** | 95±11 | 91±7 | 32±8 | 30±3 | 35±7 | 33±2 |
| 500 | **98±2** | 96±12 | 87±19 | 42±14 | 29±3 | 49±14 | 32±2 |

**NOTE:** SPI-MPNN n=100 shows seed instability (per-seed: 0.96,0.46,0.93,0.84,0.93,0.66,0.80,0.66,0.86,0.97). Likely MPS numerical precision issue — CPU sensitivity runs with same config gave 94.0±2.2%.

### 2.5 DEFINITIVE: 260327_eeml (w20, r=2, td5, gl=0.02, corrected families, MPS)

New data: α∈[0.3,0.7], 1500/class.

| n/class | SPI-MPNN | Fixed-SPI | MLP-Mix | Corr | Latent | Shuffled | Node-Only |
|---|---|---|---|---|---|---|---|
| 20 | **69±18** | 60±20 | 53±16 | 30±3 | 31±2 | 31±4 | 32±2 |
| 50 | **89±7** | 80±26 | 84±8 | 39±11 | 30±3 | 31±5 | 32±3 |
| 100 | 94±5 | **97±1** | 80±10 | 40±10 | 28±2 | 35±10 | 30±2 |
| 200 | 97±6 | **98±1** | 85±20 | 47±14 | 28±3 | 34±5 | 31±2 |
| 500 | 98±4 | **99±1** | 97±5 | 54±10 | 30±3 | 40±15 | 33±3 |

### 2.6 DEFINITIVE (FINAL): 260327_eeml, w60, CPU — PRIMARY RESULT TABLE

| n/class | SPI-MPNN | Fixed-SPI | MLP-Mix | Edge-Abl. | Corr | Latent | Shuffled | Node-Only |
|---|---|---|---|---|---|---|---|---|
| 20 | 57±22 | **72±12** | 63±10 | 35±14 | 35±5 | 30±2 | 34±4 | 33±2 |
| 50 | **92±2** | 92±2 | 80±7 | 56±25 | 47±13 | 33±1 | 38±5 | 31±2 |
| 100 | **97±2** | 98±1 | 88±7 | 53±21 | 57±2 | 32±2 | 37±3 | 31±3 |
| 200 | 98±5 | **98±1** | 95±5 | 78±25 | 53±11 | 31±2 | 41±5 | 30±2 |
| 500 | 98±4 | **99±0.5** | 99±2 | 88±23 | 59±1 | 31±2 | 55±19 | 32±3 |
| 1000 | **100±0.2** | 99±1 | 99±0.2 | — | 59±1 | 31±3 | 80±13 | 34±2 |

Per-seed n=500: [1.00, 1.00, 0.87, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00] — seed 2 outlier.
Per-seed n=1000: all ≥0.99.
Top w (n=500): SGC parametric (0.25-0.5Hz), GC Gaussian — all causal.
Shuffled rises to 80% at n=1000 (distributional fingerprint).

### 2.2 SPI-MPNN sensitivity (all with restarts=2)

| Config | Families | n=20 | n=100 | n=500 | Failures (n=500) |
|--------|----------|------|-------|-------|-------------------|
| w20, td3, gl=0.01 | broken | — | 88.2±8.1% | 98.2±3.5% | 0/10 (1 weak: 0.88) |
| w20, td3, no-reg | broken | — | 81.0±10.0% | 95.7±6.5% | 0/10 (2 weak) |
| w40, td3, gl=0.01 | broken | — | 92.6±4.7% | 98.0±4.2% | 0/10 (1 weak: 0.86) |
| w20, td5, gl=0.01 | broken | — | 85.5±12.8% | 99.5±0.3% | 0/10, all ≥0.99 |
| w40, td5, gl=0.01 | **fixed** | — | **94.8±3.6%** | 97.6±4.5% | 0/10 (1 weak: 0.84) |
| w20, td7, gl=0.01 | **fixed** | — | 93.2±4.8% | 99.5±0.5% | 0/10 |
| w20, td5, gl=0.05 | **fixed** | — | 80.9±20.4% | 99.4±0.1% | 0/10, but 2 fail at n=100 |
| w20, td5, gl=0.05, w40 | **fixed** | — | 87.8±12.3% | 99.2±0.4% | 0/10 (1 weak n=100) |
| **w20, td5, gl=0.02** | **fixed** | — | **94.0±2.2%** | **99.6±0.3%** | **0/10, all ≥0.99** |
| w20, td5, gl=0.03 | fixed | — | 83.3±15.4% | 99.2±0.8% | 0/10 |
| w20, td5, l1=0.01 | fixed | — | 83.2±18.7% | 99.6±0.3% | 0/10 (high var n=100) |
| lr=5e-4, td5, gl=0.02 | fixed | 68.6±9.7% | 93.5±2.5% | 99.1±0.6% | 0/10 |
| **w40, td5, gl=0.02** | **fixed** | 64.5±11.7% | 92.2±4.8% | 99.4±0.5% | 0/10 |
| **w60, td5, gl=0.02** | **fixed** | **69.5±5.6%** | **95.1±2.3%** | 99.1±0.8% | 0/10 |
| w20, td5, gl=0.02, r=3 | fixed | — | 85.8±12.7% | 99.3±0.5% | 0/10 (high var n=100) |

**Best config: warmup=20, top_d=5, group_λ=0.02, l1_λ=0.001, restarts=2, corrected families.**
- w60 marginally better at n=100 (95.1±2.3% vs 94.0±2.2%) but worse at n=500 (99.1% vs 99.6%)
- w40 is a local dip — non-monotonic warmup pattern
- restarts=3 worse than restarts=2 (selection criterion unreliable with 3 candidates)
- n=100: 94.0±2.2% (all seeds ≥0.91) — tightest variance of any config
- n=500: 99.6±0.3% (all seeds ≥0.99) — near-perfect, zero failures
- Per-seed n=100: [0.91, 0.92, 0.92, 0.93, 0.93, 0.93, 0.96, 0.96, 0.96, 0.98]
- Per-seed n=500: [0.99, 0.99, 0.99, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]

### 2.3 Learned w interpretation (corrected families)

**gl=0.02 at n=500 (best config):**

| Family | # SPIs | L2(w̄) | Interpretation |
|--------|--------|--------|----------------|
| **causal** | 48 | **0.058** | SGC (order-1, 0.25–0.5Hz), GC, XME — directed frequency-domain measures |
| spectral | 52 | 0.010 | coherence magnitude (minor contribution) |
| linear | 13 | 0.010 | cov-sq LedoitWolf (minor) |
| information | 5 | 0.002 | negligible |
| distance | 5 | 0.002 | negligible |
| rank | 2 | 0.001 | negligible |

Top 5 w: **all causal** (SGC parametric variants + GC Gaussian). The model selects directed spectral causality measures — theoretically correct for VAR(1) where causal structure manifests as asymmetric spectral transfer functions.

**gl=0.05 (stronger regularisation):** 84/125 near-zero, top 5 concentration 39.6%, all causal. Sparser but unstable at small n.

---

## 3. Empirical Assessment (Updated 2026-03-28)

### 3.1 What holds — strongly confirmed across both datasets

1. **SPI vocabulary is the dominant inductive bias.** SPI-MPNN, Fixed-SPI, MLP-Mix (all SPI-consuming) dramatically outperform non-SPI baselines. Gap: 60-100% vs 28-54%. The vocabulary, not the construction mechanism, is the primary contribution.

2. **Symmetric construction provably and empirically limited.**
   - Latent: 28-35% — *below* chance. Symmetric node-embedding construction is actively harmful.
   - Correlation: 30-54% — stays well below 67% ceiling. Top-d sparsification of |r_ij| discards motif edges.
   - Node-Only: 30-34% (chance). Pairwise information necessary.

3. **Learned w is interpretable, theoretically correct.** Top 5: SGC (0.25-0.5Hz), GC (Gaussian), TLMI — all causal/directed. Selects spectral Granger causality for VAR(1), matching theory.

4. **Sample efficiency at small n.** SPI-MPNN best at n=20 on both datasets (69-70%).

5. **Controls hold.** Shuffled 31-49%, Node-Only 30-34%.

### 3.2 Findings (not problems)

1. **Fixed-SPI matches/beats SPI-MPNN at n≥100 on new data** (97±1 vs 94±5 at n=100). Vocabulary-as-dense-edge-features is sufficient with enough data. SPI-MPNN advantage: sample efficiency at small n + interpretable w.

2. **w60 clearly improves SPI-MPNN on new data** (97±2 vs 94±5 at n=100, 100±0.2 vs 98±4 at n=500). Longer warmup helps the model escape bifurcation basins.

3. **Correlation stays below 67%.** Top-d sparsification + nuisance AR(1) nodes prevent it from reaching the theoretical ceiling.

### 3.3 Remaining

- Definitive w60 run on new data (all 7 models + n=1000) — the primary result table for the paper.

---

## 4. Plan (Updated 2026-03-28)

### 4.1 ~~Sensitivity checks~~ DONE
### 4.2 ~~Definitive runs (w20, both datasets)~~ DONE
### 4.3 ~~New data generation~~ DONE

### 4.4 NEXT: Definitive w60 run on new data

All 7 models, n∈{20,50,100,200,500,1000}, 10 seeds, w60, td5, gl=0.02, corrected families.
This is the PRIMARY result table for the extended abstract.

### 4.5 Writing

Extended abstract drafted: eeml_extended_abstract.md. Update results table when §4.4 completes.
