# Research reset and submission direction

The user authorised a substantial research/repository reset on a new branch, without backward compatibility, prioritising a meaningful and interpretable question over incremental architecture gains. Application choice is open. Their initial `docs/` → `docs_old/` move was intentional and is preserved in `a7b089e`. They emphasised that old negatives may reflect inadequate design/compute, and that current pyspi is dramatically faster and supports larger systems. Do not turn those old failures into evidence against EEG, graph learning or nonlinear dependence.

The active branch is `codex/research-foundation`. [Assessment](../research/assessment.md) is authoritative for verified defects and evidence limits; [agenda](../research/agenda.md) is authoritative for literature positioning and the proposed decision experiments. The accepted EEML abstract is historical, not an evidence standard for a new submission.

## What now holds

- The universal 2/3 symmetric-statistic ceiling is false for the specified VAR motifs. Exact symmetric lagged moments separate all three. Fresh simulations plus a fixed logistic baseline obtain mean accuracy .6502 at T=500 and .9738 at T=5000. The latter was an exploratory follow-up, not a preregistered comparison. [Argument, caveats and reproduction](../../research/premise_audit/README.md).
- Verified old defects include zero-filling before missingness filtering, fitting small-n feature selection on the largest pool, ignoring patient grouping in standard mode, fallback window IDs for missing patients, and an ungated edge path that bypasses interpreted weights. Historical training code is archived rather than silently changing the meaning of old outputs.
- The local banks are incomplete: 951 nonlinear metadata files unreadable; missing waveform/SPI files in the older VAR bank; checked computation-identity fields absent. Detailed local inventory is `results/review-2026-10-16/legacy-inventory.json`. This is not an audit of Gadi storage.
- Current extraction reference is the sibling study at `bd5041cb143ec9535de2339727aaa3d9f0523caf` and pyspi at `c2d723f9422617d668bca1688cb06ada67395aaa`, computation `3.0.0.r9`. Use the bundled 283-statistic p90, not frozen 289/297 or older EEML YAMLs. Both sibling worktrees had unrelated dirty files and were left untouched. [Reproduction boundary](../operations/reproduction.md).

## Direction and its uncertainty

The best current candidate is whether finite-record interaction reliability/diversity predicts when cross-channel forecasting helps or harms on unseen sensor systems. Traffic/sensor-network deployment is a provisional application, not a locked study. No new model or production dataset has been chosen. GATv2/graph transformers are comparators, not an assumed contribution.

Literature materially narrowed the proposal: CoRA and ChaTSFM already cover channel-aware adapters; Chronos-2 and TimesFM-3 are multivariate; Choi et al. (September 2026 preprint) already separate dependence, usefulness and reliance, and test temporal routing. FFORMA and model recommendation already cover feature-based selection. A generic “stats + GAT”, “correlation is not causation”, “graphs sometimes hurt”, or “gate channel mixing” paper would not meet the user's standard. The agenda documents primary-source links and the remaining candidate distinction; novelty is not established.

The next consequential evidence is whether competent forecasters leave useful selection headroom and whether p90/reliability descriptors predict it out of system beyond cheap summaries and equal-budget historical validation. If not, reject the vocabulary-selector framing rather than grow the architecture. Adequate independent systems, licences, pretraining contamination and the precise difference from the closest papers remain unresolved. No finite history can guarantee successful selection under arbitrary unannounced future mechanism changes.

## Repository state

The active package is `src/spigraph/`: missingness-preserving training-only preprocessing, split validation and computation-identity checks. It is a foundation, not a replacement GNN/forecasting pipeline. Forecast temporal leakage checks and verified extraction adapters still need implementation for a settled study. Tests are independent of private data; 19 passed and lint passed during the reset.

Old code, configs, pyproject/lock, notebooks, figures, manuscript and `docs_old/` are under `archive/eeml-2026/`; 128 moved non-cache files were byte-checked. Original pre-move state is recoverable at `60b1288`. Scientific data/results were untouched. Personal application materials/public key/Office lock were preserved locally in ignored `local_archive/application/`, removed from the current tracked tree, and remain in Git history. The old `.venv` was retained; `.venv-foundation` and the new root lock verified the foundation. No Gadi jobs or schedules were created.
