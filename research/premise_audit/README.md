# Symmetric statistics can distinguish the legacy VAR motifs

The archived abstract's universal 2/3 accuracy ceiling for symmetric statistics is false. This audit supplies a constructive counterexample; it does not claim causal identifiability or that every symmetric statistic suffices.

## Exact argument

For the specified VAR, `X_t = A X_(t-1) + epsilon_t`, innovations are independent with common variance and `A[target,source] = alpha` on the motif edges. The remaining channels are independent AR(1) nuisances. The stationary covariance solves `Sigma = A Sigma A^T + I`; lag-l covariance is `A^l Sigma`. Standardise by the stationary channel variances and define `S_l = (abs(R_l) + abs(R_l^T))/2`, with diagonal zero. Every `S_l` is symmetric.

The off-diagonal sum of `S_0` is positive only for the fork (its children share the previous innovation). Among the remaining two motifs, the off-diagonal sum of `S_2` is positive only for the chain (its endpoints are related after two time steps). The collider has neither. These statements hold for nonzero coupling in the specified family and survive node permutations and independent nuisance channels. The code checks alpha = 0.3, 0.5 and 0.7. This is sufficient to refute a universal impossibility claim without fitting any classifier.

This does not dispute the Markov-equivalence theorem. The old argument incorrectly applied it to arbitrary numerical statistics of temporally structured, parametrically specified processes. Symmetry under exchanging two channels does not mean discarding all lag information.

## Finite-sample diagnostic

Fresh mechanism-matched simulations use M=10, 500 records/class, alpha uniform on [0.3,0.7], independent Gaussian noise, nuisance AR coefficient 0.8, 200 transient samples, per-channel z-scoring and random channel permutations. They are not regenerated historical bytes. A recursive-VAR unit test checks the vectorised simulator independently.

Features are sorted upper-triangle values of `S_0`, `S_1`, `S_2` (135 features). Sorting discards pair/node identity and the symmetrisation discards directed orientation. StandardScaler is fitted on training data only; LogisticRegression uses fixed C=1 and max_iter=2000. Five stratified 70/30 splits use seeds 17, 29, 43, 71 and 101. No hyperparameter search was performed. Each split trains on 1,050 records and tests on 450; the repeated test sets overlap, so their spread is not a confidence interval over independent experiments.

| Samples per record T | Mean accuracy | Split SD | Mean macro-F1 |
| --- | --- | --- | --- |
| 500 | 0.6502 | 0.0225 | 0.6489 |
| 5000 | 0.9738 | 0.0036 | 0.9738 |

The longer-window run was chosen **after** seeing the T=500 result to probe the finite-sample explanation, as motivated by the exact moments. Both results are retained; this is an exploratory premise audit, not a preregistered benchmark. Changing T changes the random realisations, so the difference is not a paired estimate of a duration effect. It nevertheless provides a finite-sample counterexample to the claimed universal ceiling, consistent with the population calculation. The short-window result is a reminder that identifiable information can be difficult to estimate.

## Reproduce

From the repository root after `uv sync --locked`:

```bash
uv run --locked python -m research.premise_audit.symmetry --output results/premise/analytic.json
uv run --locked python -m research.premise_audit.symmetry --simulate --n-time 500 --output results/premise/t500.json
uv run --locked python -m research.premise_audit.symmetry --simulate --n-time 5000 --output results/premise/t5000.json
```

Use fresh output paths. JSON outputs retain the exact input hashes, partitions, test labels/predictions, runner and lock hashes, source revision/dirty state and library versions. The initial review outputs are under ignored `results/review-2026-10-16/`; final provenance-enabled replays are `t500-final.json` and `t5000-final.json` there. Earlier exploratory outputs are retained too. The original local waveform bank is not needed. This is not pyspi extraction and does not create or extend any SPI corpus.
