# Research assessment — 16 October 2026

## Judgment

The current EEML story is not yet defensible as a workshop paper in its existing form. Its strongest theoretical premise is false, the learned weights do not establish a mechanistic interpretation, and the experiments do not isolate the claimed advantage. These are reasons to rebuild the argument, not evidence that statistical interaction features or the real applications cannot work. The user reports that substantially faster pyspi and abundant Gadi compute now remove important historical constraints; the new study should not inherit the old small-system, short-window limits.

What is worth retaining: the idea of explicitly representing several notions of dependence, controlled simulations, attempts to falsify interpretations, and the comparison between graph topology and edge content. What is not a contribution by itself: putting existing descriptors into a GAT, changing the sparsity penalty, increasing the catalogue, or obtaining a small average accuracy gain. The [agenda](agenda.md) evaluates more consequential questions against current literature rather than assuming the abstract should survive.

## Findings that change the scientific argument

**The universal symmetry ceiling is false.** The archived abstract asserts that any symmetric statistic makes the VAR chain and fork indistinguishable, yielding a 2/3 accuracy ceiling. Markov equivalence concerns conditional-independence structure; it does not equate arbitrary parametrised distributions, and the time-unrolled VAR is not the contemporaneous three-node DAG invoked there. For the actual VAR mechanism, symmetric absolute lagged correlations distinguish the motifs: the fork has contemporaneously correlated children, the chain has a two-step endpoint relation, and the collider has neither. The executable [counterexample](../../research/premise_audit/README.md) provides exact stationary moments and a finite-sample check. This invalidates the universal claim; it does not establish that every symmetric descriptor or short-record estimator is sufficient.

**Information, estimation and optimisation were conflated.** A poor contemporaneous-Pearson model does not bound all symmetric statistics; a chance-level latent model does not prove latent architectures cannot access the relation. Current archived latent models use raw-series encoders, but the headline result JSON does not bind its code revision, so its exact historical information access cannot be certified from current source. Compare strong temporal models, lagged linear predictors and pooled interaction summaries before attributing differences to a graph. A neural random seed is not an independent dataset or subject.

**The weight vector is not a validated explanation.** In `archive/eeml-2026/src/model.py`, `_SPIGraphBase._edge_features` normally sends the entire SPI vector through the edge MLP, irrespective of `spi_w`. The selector therefore does not describe all information used by the classifier. Gating both paths removes this bypass but does not identify mechanisms: collinearity, rescaling and downstream compensation remain. Sparse penalties and stable coefficient ranks do not convert predictive weights into causal effects. Haufe et al. show why even linear decoding weights need care; the attention literature similarly requires an explicit explanation target and empirical validation, rather than a blanket claim that attention either is or is not explanatory. [Haufe et al.](https://www.sciencedirect.com/science/article/pii/S1053811913010914), [Jain and Wallace](https://aclanthology.org/N19-1357/), [Wiegreffe and Pinter](https://aclanthology.org/D19-1002/).

**The historical controls do not establish topology necessity.** The archive reports fixed-SPI matching the learned-topology model and shuffled edges reaching high accuracy at larger training sizes. These observations motivate a pooled-edge baseline and a matched graph/no-graph factorial experiment. They do not certify that topology is useless: optimisation, uncertainty, information access and training budget were not sufficiently matched. `ShuffledEdgeMPNN` resamples permutations during evaluation, so its reported test metric additionally depends on evaluation RNG; a fixed per-record shuffle or explicit Monte Carlo average is needed.

**Interpretive claims need task-specific limits.** A preference for low-order Granger estimators on a short VAR(1) sample is consistent with known finite-sample efficiency; it does not by itself clear the “well, obviously” test. Spectral bands with `fs=1` describe cycles/sample, not physical Hz unless the sampling rate is actually 1 Hz. The assertion that matching fixed-SPI establishes “no accuracy cost” needs a prespecified equivalence margin and appropriate paired intervals, not just similar means.

## Implementation and evaluation defects

| Finding | Evidence in archived source | Consequence / active replacement |
| --- | --- | --- |
| Missing values converted to zeros before validity filtering | `src/graph_build.py:38` | Missingness gate cannot measure the original failure rate. Active preprocessing preserves non-finite values until training-only selection and imputation; returns a validity mask. |
| Feature selection and empirical grouping use the largest training pool for every smaller budget | `src/run_pipeline.py:752` onward | Small-n curves have extra unlabelled information; not validation/test leakage, but not strictly n-sample inductive training. Refit all learned preprocessing per fold and budget. |
| `--group-by-patient` is ignored in standard mode | `_run_standard` uses sample-level `train_test_split` | Potential patient leakage despite recorded flag. No old standard-mode patient result should be trusted without its manifest. |
| Missing patient IDs fall back to a window-directory path | `src/run_pipeline.py:337` | Group split can appear valid while related windows are treated as independent. Active split contract rejects empty groups. |
| Samples are skipped on any load exception, with warnings | `_load_all_data` | Incomplete/class-dependent cohorts can silently change. New studies must retain exclusion reasons and verify planned versus observed cohort counts. |
| Model RNG depends on model-list order | seed set before looping over models | Seed labels do not define independent, reproducible model-specific streams. Derive split/init/shuffle seeds separately and save them. |
| Every restart evaluates test; winning weights are not restored to the supplied model | `src/train.py:388` and selection loop | Selection uses validation, so test leakage is not proven, but repeated test visibility and model/result mismatch are avoidable. Evaluate once after restoring the selected checkpoint; retain predictions and checkpoints. |
| Regression test uses the same data for validation and test, and returns normally when data are absent | `tests/test_regression.py` | It is not independent scientific validation and can look passed without executing. Active correctness tests require no private dataset. |
| Merged-label training count uses number of labels, sampling uses source directories | `src/run_pipeline.py:864` | Reported training budget can be wrong. Save actual sample/group IDs and counts. |
| Dense `(M,M,K)` arrays and fixed-M/T batching assumptions | model helpers, graph construction | Faster extraction does not make the old training implementation suitable for arbitrary larger systems. Memory/profile checks and ragged/edge-list batching are needed. |

These archived defects have been isolated, not silently repaired in a way that would relabel old results. The active package implements reusable contracts, not a replacement training pipeline. Identity declarations still require verified source adapters; validating a supplied hash cannot prove the caller hashed the correct input or assigned the correct subject.

## What the historical negatives mean

| Observation in old writeup | Defensible present interpretation | Experiment needed before ruling anything out |
| --- | --- | --- |
| TUH FNSZ/GNSZ weak or flat learning curves | Unresolved task/measurement/design result; no local patient-complete bank was found | Verify labels, official split, montage and independent patient counts; strong waveform/band-power baselines; longer windows where biologically justified; estimate uncertainty over patients. |
| Nonlinear R1 solved by node-only features | The particular generator does not isolate relational information | Match marginal distributions across labels and measure the residual separability before neural training. |
| R1b around 2/3 | No justified universal symmetric ceiling; possible finite-sample/observation loss | Include symmetric lagged features, directed features and undirected subsets; squaring is non-invertible and may destroy information. |
| Kuramoto fork/collider failure | Inspect symmetry of the specific generator and available labels | Verify distinguishability of the observation distributions before treating nonlearning as a method failure. |
| Latent baseline at chance | Architecture, optimisation or information-access failure remains possible | Equal raw history, modern temporal baseline, training diagnostics, matched tuning and replicated splits. |

None of these warrants abandoning EEG, nonlinear dynamics or graph learning. Nor does more compute alone validate the original explanations.

## Local data and result provenance

The metadata inventory found 3,363 records in `260327_eeml`: 658 lack `timeseries.npy`, 844 lack `spi_mpis.npz`. The 24-record nonlinear demo has those files. `260726_r1_nonlinear` has 2,224 metadata paths, of which 951 are unreadable; the first inspected unreadable file is empty. Among its 1,273 readable metadata records, one waveform and two SPI archives are missing. `data/tuh` contains no `meta.json` records. These are local-copy findings only; Gadi data were not inspected, changed or submitted to during this review.

All readable metadata inspected by the inventory lack the checked `pyspi.version`, `pyspi.computation_version` and `pyspi.config_sha256` fields. This does not prove that no historical provenance can be reconstructed elsewhere. It prevents silently certifying these local banks as current computation. The headline `sample_efficiency_30seeds_main_results.json` also lacks a hyperparameter section. Keep the banks for archaeology; recompute clean banks for confirmatory evidence.

Reproduce with `python scripts/audit_legacy.py --data-dir data --output results/<new-run>/legacy-inventory.json`. The detailed initial inventory is local at `results/review-2026-10-16/legacy-inventory.json`. No scientific files were deleted. [Current reproduction policy](../operations/reproduction.md).

## Submission threshold

A credible workshop submission needs one narrow question, a defensible relationship to the closest papers, a reliable positive or informative negative answer, strong simple comparators, correctly grouped uncertainty, and executable provenance. The existing abstract acceptance is a useful starting point, not evidence that the current claims meet that threshold. Conference-specific workshop fit and deadlines must be checked after the contribution is known; this review does not promise acceptance or claim main-track novelty.
