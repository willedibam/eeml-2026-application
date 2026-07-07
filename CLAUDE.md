# CLAUDE.md — eeml-2026-application

Guidance for Claude Code working in this repo. Read this first.

## What this project is

Graph construction for GNNs on multivariate time series (MTS) by learning a
**linear probe over a vocabulary of ~125 named pairwise statistics (SPIs)**,
computed with `pyspi` (Cliff et al. 2023). The learned weight vector `w` is the
scientific output: a named, falsifiable hypothesis about which form of coupling
drives a task. Core adjacency: `A_ij = softplus(b + wᵀ E_ij)`, top-d sparsified.

**Status.** Accepted as a **poster at EEML 2026** on the strength of the extended
abstract in `overleaf/`. The abstract is *work in preparation*, not a finished
paper. The active goal is to grow this into a frontier-venue submission
(ICML / ICLR / LoG / NeurIPS / a computational-neuroscience methods venue).
See "Open scientific issues" — the current evidence base is not yet sufficient
for that bar.

## Repo map

```
src/
  run_pipeline.py   CLI entry: data → graph → train → eval. Two modes. Model registry.
  model.py          All models (registry below). Shared _SPIGraphBase backbone.
  train.py          Training loop, TrainConfig, sparse-group-lasso loss.
  graph_build.py    SPI tensor loading, filtering, robust scaling, family assignment.
  features.py       Node features: (mean, std, lag-1 autocorr, dominant FFT mag) = 4 dims.
  analysis.py       Result plots (matplotlib). plot_family_norms etc.
  utils.py          json/path helpers.
visualization.py    Paper-quality figures (LaTeX/CM fonts). plot_family_weights = Fig 1b/1c.
overleaf/           The submitted extended abstract (eeml_extended_abstract.tex) + refs.
configs/pyspi/      pyspi configs. eeml.yaml = the ~136-SPI config used for the abstract.
                    benchmarked90_amortized_config.yaml = faster 297-SPI config (see below).
data/260327_eeml/   Primary VAR dataset: var-chain / var-fork / var-collider.
figures*.ipynb      Figure generation notebooks.
```

Data compute (raw MTS + pyspi SPIs) is generated in the **sibling repo**
`../mts-spi-study-cluster/` (see `run_experiments.py` there). This repo consumes
the resulting per-instance dirs: `data/<dataset>/<class>/M{M}_T{T}_I{idx}/` each
containing `timeseries.npy`, `spi_mpis.npz`, `meta.json`.

## Model registry (`--models`)

| key | role | notes |
|---|---|---|
| `spi-mpnn` | **proposed** | learned linear `w` over SPIs; inspectable, group-lasso-regularised |
| `fixed-spi` | ablation | fully-connected + SPI edge features, no learned topology; matches spi-mpnn at n≥100 |
| `correlation` | baseline | fixed graph from Pearson \|r\|; symmetric → ≤2/3 ceiling |
| `latent` | baseline | **symmetric** dot-product embeddings; cannot represent directed edges (weak control) |
| `latent-directed` | baseline | **directed** ordered-pair MLP; the *fair* latent competitor (added 2026-07) |
| `mlp-mix` | ablation | nonlinear MLP adjacency; tests whether linear `w` suffices |
| `node-only` | ablation | no graph; sanity check that the task needs pairwise structure |
| `edge-ablation` | ablation | SPI topology, zeroed edge features |
| `shuffled` | control | SPI values permuted across pairs |
| `single-spi` / `subset-spi` | ablation | 1 or k named SPIs (needs `--single-spi` / `--subset-spi` prefixes) |

## Canonical run

The **CLI defaults already encode the validated hyperparameters** (found via
sensitivity analysis on the 30-seed VAR runs). Prefer the CLI; do not hand-build
`TrainConfig` — its *dataclass* defaults are zeros (l1/group/warmup = 0, top-d = 3,
restarts = 1) and will silently produce non-canonical runs.

```bash
python -m src.run_pipeline \
  --data-dir data/260327_eeml \
  --class-names var-chain var-fork var-collider \
  --mode sample-efficiency \
  --n-train 20 50 100 200 500 1000 \
  --test-per-class 200 --val-per-class 100 --seeds 30 \
  --models spi-mpnn fixed-spi correlation latent latent-directed node-only \
  --device cpu --output-dir results --tag <descriptive_tag>
```

Validated set (all are CLI defaults): `--warmup-epochs 60 --top-d 5
--group-lambda 0.02 --l1-lambda 0.001 --lr 1e-3 --restarts 2 --batch-size 32
--max-epochs 200 --patience 20 --hidden 64 --n-layers 2 --dropout 0.1`.

## Loss (train.py)

Sparse group lasso on `spi_w` on top of cross-entropy:
`L = CE + λ₁‖w‖₁ + λ_g Σ_g √|g| ‖w_g‖₂`.
- `λ₁‖w‖₁` → within-family sparsity. `λ_g Σ √|g|‖w_g‖₂` → between-family sparsity.
- **The √|g| weighting (Yuan & Lin 2006) is ON by default** (`group_size_norm`,
  added 2026-07). Without it, the causal family (48 SPIs, ~38% of the vocabulary)
  is under-penalised and dominates the signature by size alone. `--no-group-size-norm`
  restores the old behaviour for reproducing legacy runs. **Because the penalty
  scale changed, `--group-lambda` likely needs re-tuning; the memory'd 30-seed
  numbers were produced pre-fix.**
- Family importance is now reported as **per-SPI RMS** `‖w_g‖₂/√|g|`
  (`analysis.plot_family_norms`, `visualization.plot_family_weights`), not raw
  summed L2. The abstract's "causal family carries ~3.5× the L2 norm" claim is
  computed the old (confounded) way and **must be recomputed** before reuse.
- `AdamW(weight_decay=1e-4)` also applies implicit L2 to `spi_w`, double-regularising
  the parameter whose sparsity is being interpreted. Consider excluding `spi_w`
  from weight decay if the interpretability story needs to be clean.

## Scope boundaries — what does NOT work (do not re-run blind)

- **Symmetric-coupling tasks** (Kuramoto): fork/collider are indistinguishable
  under undirected adjacency; method fails at the ceiling. Abandoned.
- **Univariate-signal tasks** (motor-imagery BCI IV-2a, SelfRegulationSCP1): the
  class signal is per-channel power (ERD/ERS) or voltage polarity, not pairwise
  coupling. SPI vocabulary is mismatched; models sit at chance. Abandoned.
- **Pooled cross-subject EEG**: inter-subject SPI variance ≫ inter-class variance.
- The method works **only** when classes differ in *how* channels couple, not in
  *what* individual channels do. State this scope explicitly in any writeup.

Real-world bet: **TUH-EEG Seizure Corpus (TUSZ)**, focal (FNSZ, directed causal
outflow) vs generalized (GNSZ, symmetric bilateral coherence) — genuinely
separable by the directed-vs-undirected SPI families. SSH key: `id_ed25519.pub`
(approved). This is the make-or-break real result.

## Open scientific issues (the reason it's not yet frontier-ready)

1. **Near-tautological evidence.** The headline experiment is a VAR(1) on which a
   Granger-family statistic *must* win; "w recovers spectral GC" is confirmation,
   not discovery. Need results on tasks whose optimal statistic is non-obvious
   (nonlinear / directed-nonlinear) so recovery becomes a finding.
2. **No successful real-world result yet** (see scope). TUH is the plan.
3. **fixed-spi matches spi-mpnn**, so the contribution is "SPIs are good edge
   features" + interpretability of `w`, not topology-learning superiority. Frame
   accordingly; do not overclaim.
4. Novelty vs multi-connectivity / attention GNNs (e.g. sriramulu2023) needs a
   head-to-head, not a dismissal.

The recommended reframe is **interpretable scientific-discovery method**, not
accuracy-SOTA GNN. See `docs/multi_regime_experiment_spec.md` for the
generative-regime → recovered-family "phase diagram" study that turns the
tautology into the contribution.

## Compute

Clusters: **NCI Gadi** and **USYD Physics**. `CLUSTER_SETUP.md` is likely stale —
re-verify paths/modules by asking the user to run commands there. The
`benchmarked90_amortized_config.yaml` (297 SPIs, ~5.5s/SPI cap) is much cheaper
than the abstract's config and enables **more, larger-M, faster** MTS datasets —
the enabling lever for the multi-regime study.

## Conventions

- Minimal, surgical changes; preserve architecture/naming/style. Fix root causes.
- Split seeds `_SPLIT_SEED`/`_POOL_SEED = 42` are fixed — never change mid-experiment.
- New models consuming `spi_tensor`/`pearson_corr` MUST use the `_unbatch_*` helpers
  (PyG concatenates along dim 0; see the note at the top of model.py).
- Result JSONs carry full `hyperparameters` + per-seed `learned_w`; keep it that way
  so plots and audits are reproducible from the JSON alone.
- Label claims verified / inferred / speculation. Do not overstate novelty or
  empirical support — the user prefers admitted uncertainty over confident guesses.
