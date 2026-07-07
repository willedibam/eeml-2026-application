# Multi-Regime Comparative Study — Experiment Spec

Status: proposed (2026-07). Owner: Will Edibam. Consumes `src/run_pipeline.py`
+ new generators in the sibling repo `../mts-spi-study-cluster/`.

## 1. Thesis

The abstract's single result — "on a VAR, the learned `w` recovers spectral
Granger causality" — is **near-tautological**: a linear autoregression is, by
construction, what Granger causality detects. Recovery is confirmation the
pipeline works, not a scientific finding.

This study removes the tautology by making family recovery a **prediction that
can be wrong**. We hold the *classification task fixed* (chain / fork / collider
motif identification) and vary only the **coupling physics** across a battery of
generative regimes. For each regime, theory predicts *which SPI family should
carry the learned weight*. The deliverable is a **"phase diagram"**: regime ×
family, showing that the recovered signature tracks the generating mechanism.

If it does, the contribution becomes: *"a principled, linear-probe method that
reads which of ~300 pairwise-dependence measures a relational task actually
needs, recovering the correct one across coupling regimes it was never told
about."* That is a real, falsifiable, group-aligned result — not a tautology.

## 2. Why this design is tight

- **Reuses the entire pipeline.** Only the generator changes; the task, models,
  loss, and analysis are unchanged. Every existing baseline/ablation transfers.
- **Recovery is predictive, not post-hoc.** The winning family is written down
  *before* the run (Table in §4). Wrong-family or wrong-band recovery falsifies.
- **Controls for the size confound.** With the √|g|-normalised group lasso and
  per-SPI-RMS reporting (both now default), a family can only dominate on merit.
  A pre-registration check: the *linear-undirected* regime must NOT light up the
  causal family — if it does, residual family bias remains.

## 3. Fixed task

Identical to the abstract: `M`-node processes, one of three directed motifs
(chain `A→B→C`, fork `A←B→C`, collider `A→B←C`) embedded among nuisance
channels; 3-class macro-F1; chain/fork Markov-equivalent under symmetric stats
(≤2/3 ceiling). Recommended scale-up with the amortized config: `M=15`,
`T=800`, 1500 instances/class, 30 seeds, SPI set =
`configs/pyspi/benchmarked90_amortized_config.yaml` (297 SPIs → richer family
coverage, esp. spectral/directed).

## 4. Regime matrix (pre-registered predictions)

Each regime keeps the motif topology but swaps the coupling function `f` in
`x_child(t) = f(x_parent, ·) + noise`. "Predicted family" is the SPI family that
should carry top per-SPI RMS weight; "specificity" is the finer within-family
prediction that must also hold.

| # | Regime | Coupling physics | Predicted family | Specificity prediction | Falsifier |
|---|--------|------------------|------------------|------------------------|-----------|
| R0 | **VAR-linear** (anchor / positive control) | linear, directed, lagged, broadband (existing `var_chat_a`) | causal | linear GC / spectral GC top | — (known) |
| R1 | **Nonlinear-directed** | directed, lagged, *nonlinear* map (e.g. `x_c = α·tanh(x_p^{t−1}) + ε`) | causal | **transfer entropy / directed-info (Kraskov) > linear GC** within family | if linear GC still beats TE, the method can't see nonlinear direction |
| R2 | **Band-limited-coherence** | undirected, linear, coupling *only* in 0.25–0.5 Hz band | spectral | **coherence/PLV in the matched band > other bands**; direction families near zero | wrong-band or causal family wins |
| R3 | **Instantaneous-nonlinear** | undirected, contemporaneous, nonlinear (e.g. `x_c = α·x_p^2 + ε`) | information | **MI / distance-corr > covariance**; rank moderate | covariance (linear) wins |
| R4 | **Monotone-heavy-tailed** | undirected, monotone nonlinearity + heavy-tailed noise | rank | **Spearman/Kendall > Pearson/cov** | linear family wins |
| R5 | **Directed-phase (Sakaguchi–Kuramoto)** | directed phase coupling with lag/frustration | causal + spectral | **phase-slope-index / directed-coherence + PLI** top; undirected coherence secondary | symmetric phase stats win (would reproduce the dead undirected-Kuramoto failure) |

R0 is the sanity anchor (must reproduce the abstract). R1–R5 are the genuine
tests. R5 also *resurrects the abandoned Kuramoto* correctly: the earlier failure
was undirected phase coupling (fork≡collider); Sakaguchi frustration makes it
directed and should now be separable — a nice narrative recovery.

## 5. Models to run

Per regime, the standard set **plus the new fair baseline**:
```
spi-mpnn fixed-spi correlation latent latent-directed mlp-mix node-only edge-ablation shuffled
```
`latent-directed` is essential: it is the *directed* latent competitor. Its
result decides the framing —
- if it fails on R0/R2 but spi-mpnn succeeds → latent learning lacks the
  inductive bias/sample-efficiency; the vocabulary prior is doing real work;
- if it matches spi-mpnn on discrimination → the accuracy advantage is gone, and
  the paper's differentiator is *interpretability of `w`* (the recovered family),
  full stop. Either way, we learn the true contribution. Do not omit it.

## 6. Metrics

1. **Discrimination** — macro-F1 vs `n_train` (existing). Confirms the task is
   solvable in each regime.
2. **Family recovery** (the novel metric) — for spi-mpnn, is the top per-SPI-RMS
   family the predicted one? Report recovery rate across 30 seeds, and a
   regime×family RMS matrix (→ the phase-diagram figure).
3. **Within-family specificity** — is the top *member* the predicted one (correct
   band in R2; TE>GC in R1; MI>cov in R3)? Report top-5 SPIs per regime.
4. **Baseline contrast** — spi-mpnn vs latent-directed on discrimination *and*
   interpretability (latent has none). This is the framing decider (§5).

## 7. Deliverable figures

- **F1 — phase diagram** (headline): heatmap, rows = R0–R5, cols = 6 families,
  cell = mean per-SPI RMS weight (row-normalised). Predicted-family cells should
  form the bright diagonal. This single panel replaces "we recover GC on a VAR"
  with "we recover the *right* statistic across six mechanisms."
- **F2** — sample-efficiency curves per regime (spi-mpnn vs latent-directed vs
  correlation), small multiples.
- **F3** — within-family specificity: top-5 SPI bar per regime (esp. R2 band
  selectivity, R1 TE-vs-GC).

## 8. Kill / falsification criteria (pre-registered)

- If **every** regime recovers the causal family → residual family bias, not
  mechanism reading. Re-audit the group-lasso normalisation and scaling; do not
  publish the phase-diagram claim.
- If recovery only holds where F1 ≥ ~0.95 → interpretation is post-hoc; report as
  a limitation, not a capability.
- If R2 lights the wrong frequency band → the frequency-specificity claim (a
  selling point) is false; retract it.
- If latent-directed equals spi-mpnn everywhere → drop all accuracy-superiority
  language; reframe strictly as interpretability.

## 9. Generators (to implement in `mts-spi-study-cluster`)

New generator functions (siblings of `var_chat_a`), each parameterised by
`motif_class ∈ {0,1,2}` and coupling strength `α∼U(0.3,0.7)`:

- `var_nonlinear_a` (R1): lagged `tanh`/cubic child update.
- `coherence_band_a` (R2): couple parents→children via a band-pass-filtered
  drive in `[fmin,fmax]`; expose `fmin,fmax` (default 0.25–0.5).
- `instant_nonlinear_a` (R3): contemporaneous quadratic/`|·|` coupling.
- `monotone_heavytail_a` (R4): monotone map + Student-t noise (`df` param).
- `sakaguchi_kuramoto_a` (R5): directed phase coupling with phase-lag `β`
  (frustration) and per-motif directed adjacency; emit `sin(θ)` observables.

Each must embed the motif among nuisance channels exactly as `var_chat_a` (same
`rho_nuisance`, `noise_std`, `zscore`) so *only* the coupling physics differs.

## 10. Config skeleton

`configs/generate/eeml-gen-regimes.yaml` (one block per regime; R1 shown):
```yaml
base_output_dir: data/260707_regimes/
pyspi_config: configs/pyspi/benchmarked90_amortized_config.yaml
normalise: false
rng_seed: 110305
defaults: {M_values: [15], T_values: [800], instances: 1500}
mts_classes:
  - {name: "r1-nl-chain",    generator: "var_nonlinear_a", base_params: {motif_class: 0, alpha_lo: 0.3, alpha_hi: 0.7, rho_nuisance: 0.8, noise_std: 0.1, zscore: true}}
  - {name: "r1-nl-fork",     generator: "var_nonlinear_a", base_params: {motif_class: 1, ...}}
  - {name: "r1-nl-collider", generator: "var_nonlinear_a", base_params: {motif_class: 2, ...}}
```

## 11. Run (per regime)

```bash
python -m src.run_pipeline \
  --data-dir data/260707_regimes/r1_nonlinear \
  --class-names r1-nl-chain r1-nl-fork r1-nl-collider \
  --mode sample-efficiency \
  --n-train 20 50 100 200 500 1000 \
  --test-per-class 200 --val-per-class 100 --seeds 30 \
  --models spi-mpnn fixed-spi correlation latent latent-directed node-only edge-ablation shuffled \
  --device cpu --output-dir results --tag r1_nonlinear
```
Group lasso stays default (√|g| ON). **Re-tune `--group-lambda`** on R0 first
(scale changed post-fix), then hold it fixed across R1–R5 for comparability.

## 12. Compute

- 6 regimes × ~4500 instances × 297 SPIs. The amortized config caps ~5.5s/SPI at
  M=16/T=800; generation is the cost, training is cheap on CPU. Parallelise
  generation across NCI Gadi / USYD Physics nodes (embarrassingly parallel over
  instances). Verify current cluster paths/modules before submitting —
  `CLUSTER_SETUP.md` is stale.
- Estimated training: minutes/regime/seed on CPU; fully parallel over seeds.

## 13. Milestones

1. Re-run R0 with normalised group lasso; reproduce the motif result and re-tune λ_g.
2. Implement + validate `var_nonlinear_a` (R1); confirm TE>GC recovery.
3. Add R2 (band selectivity) — the strongest standalone selling point.
4. Fill R3–R5; assemble the phase diagram (F1).
5. Decide framing from the latent-directed contrast; then commit to TUH real-world.
