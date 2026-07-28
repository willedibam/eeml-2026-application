# 2026-07-28 — order specificity, validity audit, TUH generation

- transcript `bc4dca00` · git `b52aa5a..498c4d1` · Gadi jobs `174918270`, `174926381`,
  `174937348-350`, `174937954[]`, `174937956[]`, `174938424`, `174942124`

## Realisations

- **The probe recovers a statistical-efficiency property, not just the encoded
  physics.** `sgc_parametric` order-1 carries 5.22x the vocabulary-average |w|;
  the *same estimator* at order-20 carries 0.65x. 10/10 lambda runs, median ratio
  8.63x, arms matched on estimator/statistic/band with K=6 each. A VAR(1)
  generator encodes no preference over model order, so this is not entailed by
  the data-generating process the way "Granger wins on a VAR" is. Best available
  answer to the tautology objection.
- Prediction it generates: on VAR(p), p>1, preferred order should track p. Flat
  preference falsifies. Not yet run.
- **`shuffled` is not a clean gate on R0.** At chance for n<=100, 0.82 by n=1000
  (30 seeds, sd 0.14; reproduced at 0.80 in an independent 10-seed run).
  Shuffling preserves the within-instance *multiset* of SPI values, and a
  collider has a near-zero pair where chain/fork have a medium indirect one.
  Mechanism inferred, not measured. Bounds the topology claim to small n; does
  not void it.
- **Gate tests need the SEM, signature tests need the SD.** A gate asks whether
  the mean is reliably above chance (SEM). Reading a signature needs the *typical
  seed* above chance, because `learned_w` is averaged and chance-level seeds
  contribute noise (SD). R1b is significantly above chance by SEM (t=2.5) and
  correctly void by SD.
- Module ranking is **not** identified: top-2 set holds 6/10, top-3 takes 6
  distinct values across 10 runs. Report enrichment against a permutation null,
  never a ranking.

## Failures and corrections

- **R1b (`var_obs_nonlinear_a`) at chance** on first pass: F1 0.347/0.306/0.415
  vs 0.333. Run with no control, no upper bound, and lambda left at the R0-tuned
  0.01 — `run_regime.sh` hardcoded `--models spi-mpnn`. The enrichment printed
  beneath it (M05 4.66x, z=11.7) is void. Lambda sweep + panel in flight.
- I wrote that M05-vs-M06 was parametric vs nonparametric. **Wrong**: M06
  contains `sgc_parametric_order-20`. The module contrast is confounded; the
  within-estimator comparison is the clean one.
- Four consecutive TUH generation failures, all environment/path assumptions
  asserted rather than checked: double `edf/`, silent zero-output exiting 0,
  missing pyspi, and `PYSPI_CONFIG` pointing at `/scratch/...` when the generator
  repo is in `$HOME`. Now preflighted (exit 3/4/5) and echoed at job start.
- `qsub -J` needs `#PBS -r y` (PBS Pro requires arrays to be rerunnable).
  `afterokarray` is Torque syntax, not PBS Pro — use `afterok:<id>[].gadi-pbs`,
  quoted.
- Staging writes to `/scratch/ql44/${USER}/tusz/edf`; I spent an hour reading a
  different path as a stall.

- **Inode quota, not disk, is the storage limit.** 3 files per instance x 4
  datasets exhausted 202k inodes at 7% of the byte quota. Over-quota puts jobs
  in `Hold_Types = o`, which a user cannot release -- `qdel` + resubmit is the
  only route, and it silently blocks chained jobs (`tuh_train` reached `so`).
  Deleting `$UV_CACHE_DIR` alone recovered 30k inodes at zero scientific cost.
- I went looking for further deletion targets after the safe one had already
  fixed it, and nominated another project's data. Wrong: bytes were never the
  constraint, and the fix was complete before I proposed more.

## Settings that mattered

- Vocabulary: **297-SPI** `benchmarked90_amortized_config.yaml`, K=283-284 after
  filtering. The ~125/136-SPI abstract config is retired.
- pyspi is a **pinned fork**: `~/pyspi-fork` `v2` `703c73a`, carrying estimator
  correctness fixes. Upstream PyPI pyspi gives different numbers.
- `--group-lambda` 0.01 at K~284; must be re-tuned when K changes.
- TUH generation: CHUNK=24, ncpus=24, mem=96GB, walltime 5h. Measured 3.66
  GB/worker peak, ~36 min/window at 24-worker contention (20.6 at 8).
  322 sessions -> ~1771 windows, ~3.3 h.

## Open

- R1b lambda sweep accuracy — the number that decides the synthetic programme.
- TUH controls (`node-only` first).
- VAR(p) order dose-response: not submitted.
- `shuffled` multiset mechanism: not verified.
