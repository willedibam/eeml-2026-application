# TUH-EEG (TUSZ) Real-World Experiment — Spec

Status: proposed (2026-07). Owner: Will Edibam. The make-or-break real result
(CLAUDE.md). Consumes `src/run_pipeline.py` **unchanged**; only a new generator
(`docs/tuh_generator_reference.py`, to be ported into the sibling compute repo
`../mts-spi-study-cluster`) is needed.

## 1. Thesis

Focal (**FNSZ**) vs generalized (**GNSZ**) seizures should be separable by *how*
channels couple, not *what* individual channels do — the only regime where this
method works (CLAUDE.md scope). FNSZ = focal, directed causal outflow →
directed/causal SPI families; GNSZ = generalized, symmetric bilateral coherence
→ spectral/undirected families. The deliverable is not accuracy SOTA but a
**recovered, bootstrap-calibrated family signature** that differs between the two
in the theory-predicted direction. This is the first non-tautological, real-world
instance of the phase-diagram claim.

Predicted family recovery (pre-registered):

| class | mechanism | predicted top family | falsifier |
|---|---|---|---|
| FNSZ | focal directed outflow | causal (GC/TE/PSI/DI) | symmetric family tops, or no separation from GNSZ |
| GNSZ | generalized symmetric coherence | spectral (coherence/PLV) | causal family tops |

If the recovered signatures do **not** differ between FNSZ and GNSZ (or the
classifier sits at chance), the real-world bet fails — report as a negative
result and stop, do not fish for other contrasts (that would repeat the dead
pooled-EEG failure mode).

## 2. Why the pipeline transfers unchanged

The generator emits per-instance dirs identical to the VAR format
(`timeseries.npy` `(T,M)`, `spi_mpis.npz`, `meta.json`). `run_pipeline.py`
already: fits the robust `SPIScaler` on train only, assigns families from pyspi
names, runs the model registry, and now (post-fix) reports family recovery with
`report_weight_stability` + `report_family_uncertainty`. Nothing in the
consuming repo needs to change except `--class-names fnsz gnsz` and possibly
`--top-d` for larger M.

**Leakage guard (critical):** split by PATIENT. TUSZ ships patient-disjoint
`train/dev/eval` dirs — map them to train pool / val / test directly. Never pool
a patient's windows across splits, and never let `--n-train` sampling cross
patients between splits. (The VAR nested-pool sampler assumes exchangeable
instances; for TUH, sample the training pool at the PATIENT level, not the
window level, or the sample-efficiency curve leaks.)

## 3. Instance design (defaults; verify in pilot)

- Instance = one window from an annotated seizure interval, labelled by type.
- Montage: bipolar **TCP AR** derivations (M ≈ 22 — `docs/tuh_generator_reference.py`
  `TCP_AR_PAIRS`, **UNVERIFIED**, confirm against DOCS/ montage file).
- Resample **128 Hz**, window **8 s** (T = 1024) — keeps T near pyspi's tested
  regime (the amortized config caps ~5.5 s/SPI at M=16/T=800; T-driven blowup on
  DTW / some info-theoretic SPIs is the main compute risk to measure).
- Bandpass 0.5–45 Hz, 60 Hz notch, per-channel z-score per window.
- ≤3 non-overlapping windows/seizure (augmentation; same patient → same split).

## 4. Phase 0 — discovery (laptop, minutes) — DO THIS FIRST

Pull only the tiny term-annotation `.csv` files, then tally:

```bash
rsync -auvxL -r -e "ssh -i ~/.ssh/id_ed25519" \
  --include='*/' --include='*.csv' --exclude='*' \
  nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/tuh_eeg_seizure/v2.0.6/edf/ \
  ./tusz_labels/
python docs/tuh_discovery.py ./tusz_labels
```

Decision gate: read FNSZ vs **GNSZ** patient counts per split (GNSZ is the
minority → sets balanced instances/class). If GNSZ has too few *patients* in
dev/eval to make a patient-disjoint test set, the contrast is underpowered —
reconsider before generating anything.

## 5. Phase 1 — pilot (~50–100 instances/class) — chosen first cut

Pull EDFs for a handful of FNSZ + GNSZ sessions **to the cluster** (laptop is
22 GiB, samples only), run `tuh_generator_reference.py` on them, and measure:

1. **Per-instance pyspi wall-clock** at real M/T (the compute-budget input).
2. **NaN/degenerate-SPI rate** on real EEG (VAR is clean; EEG is not — the
   `filter_spi_dimensions` drop rate tells you the usable vocabulary size).
3. **Montage correctness** — channel names resolve, M is as expected, no missing
   electrodes silently dropped.
4. **Sanity separation** — even at n≈50/class, does spi-mpnn beat chance and does
   the causal-vs-spectral family split show *any* FNSZ/GNSZ difference?

Kill criteria: if per-instance cost ≫ estimate, downsample harder / shorten
windows; if NaN rate guts the causal family, the contrast can't be read; if no
separation at all appears, escalate to more data only if Phase 0 shows the
patients exist.

## 6. Phase 2 — full generation (Gadi) + run

Only after the pilot validates cost and separation. Transfer FNSZ+GNSZ EDFs
(~10–30 GB est.) to **Gadi scratch** via rsync from a login node (compute nodes
have no internet). Fan generation out as array jobs over sessions
(embarrassingly parallel). Then:

```bash
python -m src.run_pipeline --data-dir data/tusz_fnsz_gnsz \
  --class-names fnsz gnsz --mode sample-efficiency \
  --n-train <patient-level sizes> --seeds 30 \
  --models spi-mpnn fixed-spi correlation latent latent-directed node-only \
  --device cpu --tag tusz_fnsz_gnsz
python -m src.analysis results/sample_efficiency_tusz_fnsz_gnsz_results.json
```

## 7. Compute budget (estimates — the pilot measures the true numbers)

- Per instance: pyspi at M≈22/T≈1024 ≈ **0.7–1.1 CPU-hr** (scaled from the
  benchmark M16/T800 ≈ 0.45 CPU-hr; **unverified**, T-driven SPIs are the risk).
- Target ~2000/class ≈ 4000 instances → ~4000 CPU-hr → **~8k SU** (Gadi normal
  queue, 2 SU/core-hr). Even 10k instances ≈ 20k SU. Against **150 kSU** this is
  ~5–13% — compute is not the binding constraint; class availability and clean
  instances are. Do not sacrifice signal to save SU.

## 8. Metrics (same as the multi-regime spec)

1. Discrimination: macro-F1 vs patient-level n_train (does the task solve?).
2. Family recovery: top per-SPI-RMS family per class + `report_family_uncertainty`
   bootstrap CI and P(top). The FNSZ causal vs GNSZ spectral contrast is the
   headline.
3. Member specificity: `report_weight_stability` — trust within-family claims
   only where top-SPI agreement / Jaccard is high (on VAR it was 0.67 / 0.40).
4. Baseline contrast: spi-mpnn vs latent-directed (fair encoder now) — decides
   whether the story is accuracy+interpretability or interpretability-only.

## 9. Open risks

- Montage pairs unverified (correctness flag 1 in the reference generator).
- Per-file fs varies — always read and resample (flag 2).
- GNSZ patient scarcity may cap the test set (Phase 0 decides).
- Real-EEG SPI degeneracy may shrink the causal family below what's needed to
  read a signature (Phase 1 decides).
