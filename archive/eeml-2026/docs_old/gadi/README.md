# Gadi harness — TUSZ pyspi generation

Skeleton for generating TUSZ per-instance dirs on NCI Gadi (PBS Pro). Runs
**after** Phase 0 discovery + the local pilot validate cost and separation
(see `../tuh_experiment_spec.md`). Do NOT bulk-generate before the pilot.

## The internet constraint (drives the whole layout)

Gadi **compute nodes have no internet**. Only **login nodes** and the **copyq**
queue can reach the outside. So:

1. **Stage** (`stage_data.sh`) — rsync FNSZ/GNSZ EDFs from ISIP to `/scratch`.
   Run on a login node (small) or as a copyq job. Needs the SSH key.
2. **Generate** (`tuh_generate.pbs` → `generate_chunk.py`) — array job on
   `normal`; reads staged EDFs, runs pyspi, writes per-instance dirs. No
   internet needed. This is the compute.
3. **Train** — cheap; rsync the per-instance dirs back to the laptop, or run
   `src.run_pipeline` on a single normal node.

## Pieces

| file | where it runs | what |
|---|---|---|
| `stage_data.sh` | login / copyq | rsync EDFs listed in a manifest to scratch |
| `tuh_generate.pbs` | normal (array) | fan generation over manifest chunks |
| `generate_chunk.py` | inside the array task | multiprocess pyspi over one chunk |

## Manifest

One line per session: `edf_path,csv_path,split` (paths on scratch after
staging). Build it from the discovery output — filter to FNSZ+GNSZ sessions,
one bipolar-montage token per session. `tuh_discovery.py` already knows which
sessions carry which type; emit the manifest from that walk.

## Sizing (verify against the pilot's measured per-instance cost)

- Gadi `normal` node = 48 cores, ~190 GB. Per instance ≈ 0.7–1.1 CPU-hr (est).
- With 48 workers/node and 6 h walltime → ~250 instances/task.
- 4000 instances → ~16 array tasks (`-J 0-15`), ~4000 CPU-hr ≈ 8k SU. You have
  150k, so widen the array rather than lengthen walltime if throughput matters.

## Before submitting — verify (CLUSTER_SETUP.md is stale)

- [ ] `-P` project code and `-l storage=scratch/<proj>+gdata/<proj>`
- [ ] `module avail` → real python/mne/pyspi module or a conda/venv path
- [ ] `ssh -i ~/.ssh/id_ed25519 nedc-tuh-eeg@www.isip.piconepress.com` works
      from a login node (publickey)
- [ ] montage pairs in `../tuh_generator_reference.py` checked vs DOCS/
- [ ] run ONE array task (`-J 0-0`) end-to-end before the full sweep
