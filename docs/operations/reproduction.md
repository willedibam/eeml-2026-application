# Reproduction and extraction boundary

## Active versus historical environments

The root `pyproject.toml` and `uv.lock` define the small active analysis/test environment. Historical Torch/PyG/Java dependencies are in `archive/eeml-2026/`; they are not required to audit premises or validate data contracts. The old local `.venv` was retained; the review used `.venv-foundation`. Neither directory is tracked. Do not install pyspi 2.x to make a historical YAML appear runnable.

## Current pyspi source of truth

Inspect `../mts-spi-study-cluster/README.md`, `configs/pyspi/README.md`, its `pyproject.toml`, and `../pyspi-fork/CHANGELOG.md` before extraction. The inspected study revision was `bd5041cb143ec9535de2339727aaa3d9f0523caf`; pyspi was `c2d723f9422617d668bca1688cb06ada67395aaa`, package 3.0.0, computation `3.0.0.r9`. Both sibling checkouts had unrelated working-tree changes; they were read only and are not clean production snapshots. Pin and inspect the intended commits before reuse.

New runs should resolve pyspi's bundled `benchmarked_p90` (283 statistics in that revision), rather than copy the study's frozen historical 289-statistic catalogue. The local changelog records numerical/identifier changes and direction corrections as well as speed improvements. Six directed spectral measures were transposed in the transition to 3.0. Existing 2.x/r7 banks cannot be certified by renaming columns or applying an unverified blanket transpose. Recompute into a new bank and verify source→target orientation using a known delayed driver.

Reuse the study's current `mtsspi.run_corpus` extraction runner after preparing a study-specific configuration; avoid maintaining a second cluster farm here. This repository has no approved production extraction config or current replacement GNN training runner yet. The research question must determine those interfaces. No Gadi jobs were launched during the reset.

## Minimum bank/run contract

- Each input has a stable sample ID, a parent subject/trajectory/system ID, shape, sampling rate, channel identities/order, preprocessing recipe and content hash. Group cropped, transformed and overlapping observations by the same parent. Record exclusions explicitly.
- A bank declares pyspi version and full revision, computation version, resolved config hash, exact ordered catalogue, extraction-recipe hash, direction convention, RNG policy and environment/lock identity. A source commit alone does not describe a dirty/editable dependency.
- Store undefined estimates as missing, not zero. Distinguish missing input, constant channel, estimator failure and structural undefinedness where the extractor provides those statuses. Retain validity masks through training preprocessing.
- A study run binds the bank, immutable split manifest, source commit, lock hash, all hyperparameters and separate split/init/perturbation seeds. Store actual per-budget IDs, fitted preprocessing, selected checkpoint, predictions, losses, runtime and memory. Refuse incompatible resumes and output overwrites.
- Persist split manifests before tuning; reject group/source overlap and missing groups. Classification requires explicit class coverage. Forecasting additionally needs temporal-boundary/target-availability checks; the active classification validator does **not** implement those forecasting checks.

`spigraph.contracts` and `spigraph.preprocessing` implement the first reusable checks, not all of this pipeline. `ComputationIdentity` detects mismatched declarations; the future adapter still has to verify them against actual files/environment. An unknown historical identity stays unknown.

## Cluster sizing

Follow the account-specific [HPC instructions](../../HPC.md). First measure an end-to-end smoke on each relevant M/T range and data type, then a representative parallel scout. Include failed-estimator rates, end-to-end walltime, whole-job memory, output size and SU accounting. Use one frozen source snapshot while jobs run; source moves local → origin → cluster via `git pull --ff-only`. Install dependencies before PBS submission. Gadi uses a bounded farm, not job arrays. Store retained data on gdata, temporary work on scratch, and never treat scratch as the sole durable copy.
