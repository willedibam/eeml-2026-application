# NCI Gadi — verified setup

Everything here was checked on 2026-07-28. Supersedes `CLUSTER_SETUP.md`.

## Identity and paths

| | |
|---|---|
| project | `ql44` |
| user | `we2614` |
| eeml repo | `/scratch/ql44/we2614/eeml-2026-application` (branch `main`) |
| generator repo | `~/mts-spi-study-cluster` (branch **`refactor-lagged-warping`**, not main) |
| pyspi | editable install from `~/pyspi-fork` (branch `v2`) |
| synthetic data | `/scratch/ql44/we2614/mts-spi-data/` |
| staged TUH EDFs | `/scratch/ql44/tusz/edf` |

`~/mts-spi-study-cluster/{data,logs}` are symlinks onto `/scratch`. **Home has a
10 GB quota** and fills easily — keep caches and outputs off it:
`export UV_CACHE_DIR=/scratch/ql44/we2614/uv-cache`.

## Environments

Two separate venvs; they are not interchangeable.

```bash
# training / analysis
cd /scratch/ql44/we2614/eeml-2026-application && source .venv/bin/activate
# generation (pyspi)
cd ~/mts-spi-study-cluster && source .venv/bin/activate
```

Built with `uv`, python `module load python3/3.12.1`. **Install CPU-only torch**
(`--index-url https://download.pytorch.org/whl/cpu`): the CUDA build pulls ~3 GB
of nvidia runtime that is never used and will blow the home quota.

A conda env named `mts-spi` may be active in your shell and will shadow the
venv. If `which python` is not under `.venv/bin`, run `source .venv/bin/activate`
again.

**TUH generation needs `mne` AND `pyspi`, and only the generator venv can have
both.** It reads EDFs with mne and computes SPIs with pyspi.

**pyspi cannot be installed into a fresh Python 3.12 venv.** It pins
`tslearn==0.6.3`, which pins `numba==0.53.1`, which refuses to build:
`RuntimeError: Cannot install on Python version 3.12.1; only versions
>=3.6,<3.10 are supported`. The generator venv predates that resolution and has
a working pyspi, so use it rather than trying to rebuild:

```bash
source ~/mts-spi-study-cluster/.venv/bin/activate
uv pip install mne          # the only thing it is missing
```

`tuh_generate_full.pbs` therefore sources the GENERATOR venv (overridable with
`VENV=`) and preflights `import pyspi, mne`, exiting 3 with instructions if
either is absent. Without that preflight the failure appears as every session
erroring inside worker processes with zero windows written — not as an import
error at job start.

Do **not** attempt `uv pip install -e ~/pyspi-fork` into the eeml venv; it will
fail on numba. If pyspi is ever needed there, it would require relaxing the
tslearn/numba pins in the fork.

## Queues

| queue | use | note |
|---|---|---|
| `normal` | all compute | 48 cores / 190 GB per node, 2 SU per core-hour |
| `copyq` | anything needing internet | compute nodes have **no** internet |

- `max_array_size = 10` per submission; `submit_pipeline.sh` and `submit_tuh.sh`
  batch around it. `max_queued = 1000`, so job count is not the constraint.
- **Login nodes are not for work.** Long processes get reaped and die with your
  SSH session; a `run_regime.sh` run this way lost a full training. tmux helps
  only until the node restarts. Use the queue.
- `qsub -v` splits its argument on **commas** to separate variables, so a comma
  inside a value breaks submission with "cannot send environment with the job".
  `regime.pbs` takes `+` instead.

## Measured performance (do not re-derive)

- **CPU only.** Profiling at M=20/K=297/batch=32: the per-graph Python loop is
  46% of forward time and issues one `.item()` device sync *per graph* (32
  stalls per batch). More importantly the parallelism is *across* thousands of
  independent fits, not within one, and a `gpuvolta` node gives 12 CPU cores per
  GPU — same concurrency, higher SU rate.
- **Pin BLAS to 1 thread** (`OMP_NUM_THREADS=1` etc.) and run one fit per core.
- **pyspi generation is memory-bandwidth bound.** ~3.6 GB per worker regardless
  of M/T (imports are only ~150 MB; the rest is runtime allocation). 48 workers
  need ~173 GB, which fits 190 GB but not the 130 GB that was once requested.
  Per-node throughput is roughly fixed, so the only lever is **more nodes**.
- Per-dataset generation cost, 297 SPIs, 48 concurrent workers:
  | config | cost |
  |---|---|
  | M=10, T=300 | ~295 s |
  | M=10, T=500 | ~500 s |
  | M=20, T=1000 | ~3000 s |
  A 9-worker probe measured 850 s for the last of these — **3.5x optimistic**,
  because contention only appears at full occupancy. Size jobs from full-
  occupancy numbers.
- Storage: set `PARQUET=0 CSV=0` (now the default). `calc.parquet` is 2.7 MB and
  `calc.csv` 0.6 MB per dataset against 0.2 MB for `spi_mpis.npz`, the only one
  training reads — ~94% waste.

## Storage: the binding limit is INODES, not bytes

`/scratch` for ql44 is 1 TB but only **202,000 inodes**. This project writes
**3 files per instance** (`timeseries.npy`, `spi_mpis.npz`, `meta.json`), so a
3,000-instance dataset costs ~9,000 inodes and the 322-session TUH corpus costs
~5,300. Four datasets plus a uv cache exhausted the quota at **72 GB of 1 TB --
7% by bytes, 100% by file count.**

What happens when you go over: PBS puts new and queued jobs in state `H` with
`Hold_Types = o`, and **a user cannot release an `o` hold** (`qrls` returns
"Unauthorized Request"). A job already holding an `o` hold must be `qdel`ed and
resubmitted after you are back under quota. Watch for this on chained jobs --
`tuh_train` acquired `Hold_Types = so` and would never have started.

Check with `lquota` (shows iUsage/iQuota) or `nci_account`; `du --inodes` is not
available in Gadi's coreutils, so count with
`for d in <dir>/*/; do echo "$(find "$d" | wc -l) $d"; done | sort -rn`.

First reclaim should always be `$UV_CACHE_DIR` (~30-60k inodes, pure cache,
regenerable, and no existing venv breaks -- uv hard-links at install time).
That alone took this project from 202.5k back to 172.8k.

The real fix, if it recurs: pack each class into a single `.npz` instead of a
directory per instance. That is a change to `graph_build.py`'s loader, not more
cleanup.

## TUH corpus access

Real data, `rsync` over SSH from `nedc-tuh-eeg@www.isip.piconepress.com`,
corpus root `data/tuh_eeg/tuh_eeg_seizure/v2.0.6/`.

- **publickey only** since Jan 2026 — no password works. Key at
  `~/.ssh/id_ed25519` on Gadi, and the host key must be in `~/.ssh/known_hosts`
  or you get "Host key verification failed".
- Staging must run on `copyq` or a login node (internet), never a compute node.
- ~4–13 MB per EDF; the 322-session manifest is ~7 GB at roughly 1 MB/s.

## Job scripts

| script | purpose |
|---|---|
| `docs/gadi/regime.pbs` | one regime report (`CLASSES` uses `+` separators) |
| `docs/gadi/baselines.pbs` | validity gates + latent-vs-vocabulary, sharded by seed |
| `docs/gadi/stability.pbs` | lambda path + stability selection |
| `docs/gadi/stage_copyq.pbs` | TUH staging |
| `docs/gadi/tuh_generate_full.pbs` + `submit_tuh.sh` | TUH EDF → windows → pyspi |
| `docs/gadi/tuh_train.pbs` | TUH battery, controls reported first |
| `~/mts-spi-study-cluster/jobs/gadi/submit_pipeline.sh` | synthetic generation |

Chain dependent work with `qsub -W depend=afterok:<jobid>` so a failed upstream
job cannot silently feed a partial dataset downstream.
