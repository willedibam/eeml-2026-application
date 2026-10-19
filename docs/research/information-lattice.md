# Candidate direction: where does task information live in multichannel recordings?

Proposed 19 October 2026 by a second review of the reset. **Not adopted; awaiting the user's decision.** It replaces the [forecasting selector](agenda.md) as the lead candidate because that direction is largely pre-empted (see the agenda's addendum). Novelty is plausible after a targeted search, not established.

## Question

For a supervised task on multichannel recordings, how much of the decodable label information is univariate versus cross-channel, and second-order (power and cross-spectral) versus higher-order, at which time resolution? Do current models — covariance/Riemannian pipelines, CNNs, graph models with statistic edges, EEG foundation models — use the information that exists?

## Why it matters, and why it is not obvious

- Second-order, contemporaneous Riemannian pipelines match or beat deep models across MOABB paradigms ([Chevallier et al., 2024](https://arxiv.org/abs/2404.15319)). Fine-tuned EEG foundation models often fail to beat compact or classical decoders ([ICLR 2026](https://mlanthology.org/iclr/2026/yang2026iclr-eeg/); [OmniEEG-Bench](https://arxiv.org/abs/2606.00815); [2601.17883](https://arxiv.org/abs/2601.17883)). Multivariate forecasting and anomaly benchmarks turned out to carry little cross-channel signal ([Abdelmalak et al., 2025](https://arxiv.org/abs/2502.09683); [Pinet et al., 2026](https://arxiv.org/abs/2606.02670)).
- A commercial anaesthesia monitor (BIS) is built on bispectral, third-order EEG analysis; [Miller et al., 2004](https://researchcommons.waikato.ac.nz/handle/10289/1438) could not show information beyond the power spectrum. [Andrzejak et al., 2012](https://repositori.upf.edu/items/f6f85adb-1114-4dd9-a8dd-5af4f31e47ed/full) asked a similar question for epileptic-focus localisation with surrogates. Each answer covers one statistic and one task. There is no general, model-agnostic measurement of how much headroom nonlinear or relational models have.
- Not obvious because: signal non-Gaussianity does not imply label-relevant higher-order information; deep-model wins can come from better estimation of time-varying second-order structure; sensor-level "relational" information can arise from volume conduction. The per-task answers are unknown, and they decide whether nonlinear relational architectures are worth building for that task.

## Construction

If a compact group G acts on records and g is Haar-random and independent of (X, Y), then S = gX is a randomised function of the orbit T(X) and T(S) = T(X). Hence **I(Y; S) = I(Y; T(X)) exactly**. This is elementary invariance theory, not a claimed result. It needs no null model to be true and holds for non-stationary windows; stationarity only affects how T is interpreted. Schreiber's [constrained randomisation](https://arxiv.org/abs/chao-dyn/9909042) samples the same fibres for hypothesis testing; here they measure supervised information.

| Group | Preserves exactly (maximal invariant) | Removes |
| --- | --- | --- |
| CS: common cyclic shift | everything except time origin | absolute timing |
| IS: independent cyclic shift per channel | each channel's full univariate structure | cross-channel alignment |
| G1: common Fourier phase rotation | cross-periodogram = all circular auto/cross-covariances | all higher-order structure |
| G0: independent phase rotation per channel | auto-periodograms | cross-spectral phase and higher order |

Since CS ⊂ IS ⊂ G0 and CS ⊂ G1 ⊂ G0, information falls monotonically along the lattice. Cross-channel information is I_CS − I_IS. Second-order cross-channel information is I_G1 − I_G0. Higher-order information is I_CS − I_G1. Their difference is an interaction term, which can be negative. Applying a group within blocks of length L gives a time-resolution curve. Adding time reversal isolates direction and irreversibility; shifting channel subsets localises cross-channel information between channel groups.

G1 commutes with invertible per-channel linear filtering and instantaneous spatial mixing, so **the second-order versus higher-order split is invariant to filtering, re-referencing and volume conduction.** The univariate versus cross-channel split is not, and it must be reported as sensor-level.

**Estimation.** Train the same architecture on each surrogate dataset, with fresh draws as augmentation, and estimate information from held-out cross-entropy or accuracy. Differences of lower bounds are not bounds, so use several strong learner families, planted-effect power analysis, and uncertainty grouped by subject. Training on X and testing on S measures **reliance**. Training and testing on S measures **information**. The gap between them shows unused information or a shortcut.

**Role of pyspi p90.** First, as a taxonomy: classify each of the 283 statistics by which groups leave it invariant, comparing SPI(X) with SPI(gX). This is an exact, symmetry-based complement to [Cliff et al.](https://doi.org/10.1038/s43588-023-00519-x)'s empirical modules. Second, as explanation: identify which named statistics carry label information in each lattice cell. Third, as a completeness test: does the named catalogue capture the cross-channel and higher-order information that deep learners extract? Graph models with statistic edges, such as GATv2 or a graph transformer, are one model family here, not the contribution.

## Toy check (exploratory, 19 October)

`python -m research.information_lattice.toy_check` (seed 0, N=400, T=1024, M=2, fixed features and logistic regression, 5-fold accuracy). G1 preserves the cross-periodogram to 7.3e−12; G0 preserves auto-periodograms to 5.7e−14. Each planted effect lands in its predicted cell:

| Planted difference | X | CS | IS | G1 | G0 |
| --- | --- | --- | --- | --- | --- |
| Lagged linear coupling | 1.000 | 1.000 | 0.547 | 0.997 | 0.525 |
| Quadratic coupling, zero cross-covariance | 1.000 | 1.000 | 0.497 | 0.510 | 0.487 |
| Univariate skewed innovations | 0.997 | 0.997 | 0.992 | 0.492 | 0.497 |

This verifies the implementation logic only. The classes are built to separate.

## Gates

1. **Validity and power.** Planted effects — linear lagged coupling, quadratic or envelope coupling with matched spectra, univariate non-Gaussianity, non-stationary bursts — are recovered in the right cell, with measured power against n, T and M. Null tasks give zero gaps within calibrated error. Kill if wrap-around or edge artefacts create label-dependent signal that cannot be controlled, or if estimation noise swamps realistic gaps at feasible n.
2. **Real-data atlas.** At least eight subject-grouped tasks: EEG sleep staging; TUSZ seizure; TUAB abnormality; TUEV events; HMS rhythmic/periodic patterns; MOABB motor imagery; plus iEEG, multi-lead ECG, IMU and an industrial process (Tennessee Eastman). Prior expectation, to be tested: morphology tasks (TUEV, HMS) show higher-order gaps, and motor imagery shows second-order cross-channel information only. The old TUH focal-versus-generalised null fits here; its never-run band-power baseline is the G0 cell.
3. **Predictive validity.** This gate turns an audit into a claim. Across tasks, do the measured gaps predict where nonlinear or relational models beat covariance baselines? If gaps are near zero everywhere and power is demonstrated, the result is still an informative negative, but weaker.

## Closest work and the distinction

- Surrogate testing ([Theiler et al., 1992](https://doi.org/10.1016/0167-2789(92)90102-S); Prichard and Theiler, 1994; [Lancaster et al., 2018](https://doi.org/10.1016/j.physrep.2018.06.001)) tests whether a signal is nonlinear. Here, label information is decomposed with exact equalities.
- Shuffle analyses in population coding (Nirenberg and Latham, 2003; Averbeck et al., 2006) estimate information in noise correlations between trial-shuffled spike counts. Here, the recordings are continuous, the groups act within records, and the second-order split is added.
- [Choi et al., 2026](https://arxiv.org/abs/2609.32187) separates related, useful and used sources in forecasting. Here, information is decomposed by type, for classification, using exact ablations.

## Open risks

- A largely null atlas could read as "well, obviously". Gate 3 and the planted controls are the mitigation.
- Possible overlap with the sibling SPI–SPI "dependence character" line, which aims to say how dependence differs between groups. The boundary needs the user's decision.
- Data licences are unverified: TUH data use agreement, HMS terms, NSRR.
- Classifier capacity limits every estimate. The atlas reports the best achieved lower bounds, not true information.
