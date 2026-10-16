# Running on Gadi and USyd Physics

Reusable cluster guidance for these accounts. Replace `<repo>` with the repository name and use that repository's own entry points, dependency requirements and run configurations. This file does not assume any particular application or launcher. Query current quotas, queue limits and balances rather than treating past allocations as available.

## Source and environments

- Source moves from the local checkout to `origin`, then to the cluster. Commit and push locally; update an existing cluster checkout with `git pull --ff-only`. Do not edit tracked files on clusters or overwrite a dirty checkout. Follow the repository's branch policy.
- Before submitting, fetch from `origin` and confirm the cluster checkout is clean and at the intended pushed commit. For long runs, use a dedicated checkout or source snapshot of that commit; do not update code or dependencies while its jobs are using them.
- Use uv for Python environments. Keep `pyproject.toml` and `uv.lock` in Git; change dependencies and update the lock locally. On the cluster, use `uv sync --locked` with the project's required extras/groups and Python version. Resolve Linux-specific installation issues locally in the dependency definition, then push the fix.
- Keep one environment per project on shared working storage; a checkout's `.venv` may link to it. Create an additional environment only for an incompatible or deliberately frozen dependency set. Do not copy a macOS environment to Linux or create one for every run.
- Install dependencies and stage inputs before submitting. Gadi compute nodes have no external internet. Batch jobs should invoke the prepared environment directly, such as `.venv/bin/python`, rather than resolve or install dependencies at runtime.
- Record the source commit, configuration, seeds, input identity, environment/lockfile and any editable dependency revisions with each run. Use a new output directory when a computation change would make results incompatible.

## Storage

| Cluster | Working files, environments and temporary outputs | Retained outputs |
| --- | --- | --- |
| Gadi | `/scratch/ql44/we2614/<repo>/` | `/g/data/ql44/we2614/<repo>/` |
| Physics | `/import/taiji1/wedi0306/<repo>/` | Same storage; arrange a separate backup for important results |

These are project roots, not a required internal folder schema. Group data and results by purpose/run; add nesting only when useful. Keep dependencies, logs and disposable experiments distinct from scientific outputs. Source checkouts may live in home; keep large data and environments out of quota-limited home directories.

- Scratch is temporary and subject to expiry. Neither a Scratch link nor an archive kept on Scratch is durable storage. Gdata is retained project storage, not an independent backup; confirm backup arrangements for irreplaceable data. Physics working storage has no backup guarantee assumed here.
- Check both bytes and inodes before large runs. Prefer compact numeric files over thousands of tiny files. Keep one canonical copy; use direct links only when they make access clearer, and remember that a link does not copy or protect its target.
- Do not delete scientific data without the user's approval. When archiving, verify the archive against the originals before removing loose files, retain a checksum and leave a short restore pointer.
- `$PBS_JOBFS` on Gadi is node-local and disappears when the job ends. Copy needed outputs back before exit; multi-node jobs need code, environments and shared inputs on storage visible to every node.

## Submission and sizing

Run compute work through PBS, including interactive allocations. Login nodes are for light management. Start with a small end-to-end smoke, then measure one representative task before scaling.

- Request CPUs, memory and walltime explicitly. Match CPUs to actual concurrent work; do not reserve a full node for a small serial job. Account for memory across all simultaneous processes, and use PBS's whole-job peak rather than summing process maxima that may share memory.
- Size from measured elapsed time and peak memory, with headroom for variation. Group tasks with similar resource needs. Extra workers help only if the application can use them; extra memory can also increase Gadi charges.
- Prevent nested oversubscription: for process-parallel workloads, normally set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and `NUMEXPR_NUM_THREADS=1`. Deliberately threaded workloads should instead fit their threads within the allocated CPUs.
- Retain the job ID, PBS output, exit status and run metadata. Check output completeness and validity after execution; an exit status alone does not establish a complete dataset.

## NCI Gadi

- SSH: `we2614@gadi.nci.org.au`; allocation project: `ql44`. Use `we2614@gadi-dm.nci.org.au` for `scp`/`rsync` transfers.
- Check `nci_account`, `lquota` and `qstat -Qf` before a large submission. Storage quotas can hold jobs even when compute allocation remains.
- Declare every required filesystem in `storage=`, without leading slashes. Missing mounts can look like missing data inside the job.
- Gadi does not permit PBS job arrays. For many independent tasks, use a farm within one allocation, for example `nci-parallel`, rather than flooding the scheduler with small jobs. Multi-node requests must follow the selected queue's node-allocation rules.

Minimal single-node CPU example; replace the script path and size the request for the workload. Submit from the prepared repository root with `qsub run.pbs`:

```bash
#!/bin/bash
#PBS -P ql44
#PBS -q normal
#PBS -l ncpus=1,mem=4GB,walltime=00:10:00
#PBS -l storage=scratch/ql44+gdata/ql44
#PBS -l wd
#PBS -j oe

set -euo pipefail
# Load any modules required by the project's Python/native dependencies here.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec .venv/bin/python path/to/script.py
```

Monitor with `qstat -u "$USER"`; inspect a job with `qstat -f JOB_ID` or, while retained in scheduler history, `qstat -xf JOB_ID`. Review `resources_used` and the logs before sizing the next batch. Cancel only the intended job with `qdel JOB_ID`.

## USyd Physics

- SSH: `wedi0306@headnode.physics.usyd.edu.au`. Access may require the university network/VPN; the published external gateway is `gateway.physics.usyd.edu.au`.
- PBS Pro is used here too, but its queue/resource syntax differs from Gadi. For a basic CPU job, replace the Gadi directives above with `#PBS -q defaultQ`, `#PBS -l select=1:ncpus=1:mem=4gb`, `#PBS -l walltime=00:10:00` and `#PBS -j oe`. Omit Gadi's project, storage and `wd` directives; add `cd "${PBS_O_WORKDIR:?}"` before running the application.
- Query `qstat -Qf` for current queues and limits, and confirm account access before selecting a GPU or specialist queue. Physics supports arrays via `qsub -J START-END run.pbs`; the application must use `$PBS_ARRAY_INDEX` to select distinct work. Check the current array limit rather than assuming Gadi's policy applies.
- Use the same source, environment and output checks as on Gadi. Submit through the scheduler rather than running directly on compute hosts.

References: [NCI PBS documentation](https://opus.nci.org.au/display/Help/4.+PBS+Jobs), [USyd Gadi job guide](https://sydney-informatics-hub.github.io/usyd-gadi-onboarding-guide/notebooks/08_job_script.html), [Physics computing](https://www.physics.usyd.edu.au/computing/index.html), and [uv locking/syncing](https://docs.astral.sh/uv/concepts/projects/sync/). Account paths above are specific to this user; repository paths and commands are intentionally generic.
