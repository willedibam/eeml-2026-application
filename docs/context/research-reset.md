# Research reset and submission direction

The user authorised a substantial research/repository reset on a new branch, without backward compatibility, prioritising a meaningful and interpretable question over incremental architecture gains. Application choice is open. Their initial `docs/` → `docs_old/` move was intentional and is preserved in `a7b089e`. They emphasised that old negatives may reflect inadequate design/compute, and that current pyspi is dramatically faster and supports larger systems. Do not turn those old failures into evidence against EEG, graph learning or nonlinear dependence.

The active branch is `codex/research-foundation`. [Assessment](../research/assessment.md) is authoritative for verified defects and evidence limits; [agenda](../research/agenda.md) is authoritative for literature positioning and the proposed decision experiments. The accepted EEML abstract is historical, not an evidence standard for a new submission.

## What now holds

- The universal 2/3 symmetric-statistic ceiling is false for the specified VAR motifs. Exact symmetric lagged moments separate all three. Fresh simulations plus a fixed logistic baseline obtain mean accuracy .6502 at T=500 and .9738 at T=5000. The latter was an exploratory follow-up, not a preregistered comparison. [Argument, caveats and reproduction](../../research/premise_audit/README.md).
- Verified old defects include zero-filling before missingness filtering, fitting small-n feature selection on the largest pool, ignoring patient grouping in standard mode, fallback window IDs for missing patients, and an ungated edge path that bypasses interpreted weights. Historical training code is archived rather than silently changing the meaning of old outputs.
- The local banks are incomplete: 951 nonlinear metadata files unreadable; missing waveform/SPI files in the older VAR bank; checked computation-identity fields absent. Detailed local inventory is `results/review-2026-10-16/legacy-inventory.json`. This is not an audit of Gadi storage.
- Current extraction reference is the sibling study at `bd5041cb143ec9535de2339727aaa3d9f0523caf` and pyspi at `c2d723f9422617d668bca1688cb06ada67395aaa`, computation `3.0.0.r9`. Use the bundled 283-statistic p90, not frozen 289/297 or older EEML YAMLs. Both sibling worktrees had unrelated dirty files and were left untouched. [Reproduction boundary](../operations/reproduction.md).

## Direction and its uncertainty

**Lattice parked (19 October 2026).** The user judged the information lattice "very narrow" even when broadened, and asked for its problem to be motivated as important or replaced. Assessment: it is a diagnostic with no downstream decision that model validation does not already make. Its exactness holds for the ablation but not the learner-relative estimate; our own probe missed R4env at a scrambled lag. Its strongest use, auditing benchmarks for relational signal, has been made twice (Abdelmalak 2025; Pinet 2026). Code and LP validity results remain in `research/information_lattice/` as a reusable instrument. An earlier correction stands: IS and G0 keep lag-free dependence, so XR is the univariate cell.

**Candidate under discussion (19 October):** when can a pretrained transformer infer, from context alone, how the variables of a new multivariate system interact? Is the limit the pretraining prior, the architecture, or the information in the context? Gaps verified this session, novelty unestablished:

- Won (arXiv 2610.08118) attributes foundation-model covariate attenuation to the prior, but only for single-input single-output plants, via fine-tuning.
- Raventós et al. (NeurIPS 2023) show a task-diversity transition for regression in-context learning; Forgione/Piga study single-system class models; Panda adds channel attention for chaotic systems.
- No controlled-prior study of coupling inference in multivariate networks was found.

Proposed decisive pilot: from-scratch pretraining on linear-Gaussian VAR networks with a computable Bayes reference. Vary coupling-graph diversity and M, and compare channel-independent, group, full and statistic-edge attention.

**Demoted:** the 16 October forecasting selector — whether interaction reliability and diversity predict cross-channel benefit on unseen sensor systems. Abdelmalak et al. (2025) already recommend Granger-based dataset analysis for the channel-independent versus channel-dependent choice. Won (October 2026) diagnoses attenuated covariate use in time-series foundation models, and Pinet et al. (2026) find anomaly benchmarks mostly univariate. Together with CoRA, ChaTSFM, Choi et al. (September 2026) and FFORMA, the selector fails the "well, obviously" test. Details are in the [agenda addendum](../research/agenda.md). A generic "stats + GAT", "correlation is not causation", "graphs sometimes hurt" or "gate channel mixing" paper would still not meet the user's standard.

## Repository state

The active package is `src/spigraph/`: missingness-preserving training-only preprocessing, split validation and computation-identity checks. It is a foundation, not a replacement GNN/forecasting pipeline. Forecast temporal leakage checks and verified extraction adapters still need implementation for a settled study. Tests are independent of private data; 19 passed and lint passed during the reset.

Old code, configs, pyproject/lock, notebooks, figures, manuscript and `docs_old/` are under `archive/eeml-2026/`; 128 moved non-cache files were byte-checked. Original pre-move state is recoverable at `60b1288`. Scientific data/results were untouched. Personal application materials/public key/Office lock were preserved locally in ignored `local_archive/application/`, removed from the current tracked tree, and remain in Git history. The old `.venv` was retained; `.venv-foundation` and the new root lock verified the foundation. No Gadi jobs or schedules were created.
