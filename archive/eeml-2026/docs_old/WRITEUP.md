# Graph Construction from a Statistical Vocabulary

Consolidated account as of 2026-07-29. Numbers are measured; every claim here is
traceable to a result JSON or a named script in this repo.

## 1. Background and motivation

Message-passing neural networks propagate information along a relational graph.
For multivariate time series that graph is unobserved and must be constructed,
and the construction is the critical upstream design variable: Han et al. (2026)
show message passing consistently degrades when it is naive, and Liu et al.
(2025) find substantial variation across 239 pairwise statistics benchmarked on
the same neuroimaging data.

Practice occupies two regimes. The first fixes a single statistic — Pearson
correlation for functional connectivity, transfer entropy for causal graphs,
spatial distance for traffic. It is interpretable, but commits to one notion of
dependence before the task is seen; if the task-relevant coupling is directed,
nonlinear or band-limited, the operator becomes the bottleneck. The second
learns edges from latent embeddings (NRI, GraphWaveNet, adaptive dependency
learning). It recovers flexibility but sacrifices semantics: even NRI's discrete
edge types are learned categories with no pre-specified statistical meaning. In
domains where the graph *is* the object of study — neuroscience, climate,
genomics — that absence is the limitation.

We parameterise construction over a **vocabulary of K named pairwise
statistics**, and let end-to-end learning select from it. The learned weight
vector is itself the scientific output: a named, testable hypothesis about which
form of coupling drives the task.

## 2. Method

Given `X ∈ R^{M×T}`, compute K statistics per ordered channel pair via `pyspi`,
yielding a descriptor tensor `E ∈ R^{M×M×K}`. Graph construction is a linear
probe of this space:

    A_ij = softplus(b + wᵀ E_ij),      top-d sparsification per node

with K+1 learnable parameters. The linearity is deliberate (cf. Alain & Bengio,
2017): if the vocabulary is well structured a linear function suffices, and `w`
remains directly interpretable.

Selection is by sparse group lasso over literature modules:

    L = CE + λ₁‖w‖₁ + λ_g Σ_g √|g| ‖w_g‖₂

The √|g| weighting (Yuan & Lin, 2006) is not cosmetic: without it the largest
module wins on size alone. Grouping uses the **published M01–M14 modules** of
Cliff et al. (2023) rather than hand-assigned families — module identity
explains η² = 0.575 of the variance in log|w| (z = 25.4) against 0.135 for
families, which additionally misfile lagged correlations as "other".

Retained edges carry `E_ij` as attributes, and an edge network
`φ(E_ij) = MLP(E_ij)` conditions messages. **The vocabulary therefore plays a
dual role**: determining *which edges exist* (via `w`) and *what information
flows along them* (via `φ`). Node features are four-dimensional — mean, standard
deviation, lag-1 autocorrelation, dominant FFT magnitude. Global pooling gives
graph-level predictions.

`λ_g` must be re-tuned whenever K changes, because the √|g| penalty scales with
the number of groups; it is also an SNR knob, and a task with weaker coupling
needs a smaller value (§4c).

## 3. Case study: motif recovery

Three directed 3-node motifs — chain (A→B→C), fork (A←B→C), collider (A→B←C) —
embedded among 7 nuisance AR(1) channels at M=10, T=500, coupling α ~ U(0.2,0.8),
with motif nodes **randomly permuted per instance**, so the model is not told
which three of ten carry the structure.

Two ceilings govern the task, and they are different.

**Symmetric-statistic ceiling, 2/3.** Chain and fork are Markov-equivalent
(Verma & Pearl, 1990): identical weighted skeletons {α, α, α²}, while the
collider has {α, α, 0} because its parents are uncoupled. No symmetric statistic
separates the first two at any sample size.

**Contemporaneous-Pearson ceiling, 0.58.** Weaker still, and measured. The VAR
has no self-persistence, so a directly coupled pair has ≈ 0 *instantaneous*
correlation; only the fork's shared parent produces one. Over 400 instances per
motif, sorted |ρ| triples give chain (0.015, 0.032, 0.056) and collider
(0.015, 0.032, 0.058) — near-identical — against fork (0.020, 0.048, 0.206).
Separability from that triple alone: chain vs collider **0.529** (chance 0.5),
three-way **0.582**. The `correlation` baseline measures 0.59, so it is at its
information ceiling rather than underfitting.

**Results** (macro F1, 30 seeds): SPI-MPNN 67±14 at n=20 rising to 100±0.4 at
n=1000; `correlation` plateaus at 59; `latent` remains at 31–32 at every n;
`node-only` at 33. The vocabulary is necessary, and pairwise structure is
necessary.

**Signature.** Module enrichment against a permutation null (GSEA-style,
Subramanian et al., 2005): M01 ≈ 7×, M05 ≈ 4.4×, and M06 **depleted** at 0.54×.

## 4. Extensions

**(a) The probe recovers correct model order.** `pyspi` carries
`sgc_parametric` at order-1, order-auto and order-20 with estimator, statistic
and frequency band all matched, 6 SPIs per arm. Across 10 independent λ runs,
mean |w| relative to the vocabulary average: order-1 **5.22×**, order-auto
4.45×, order-20 **0.65×**, nonparametric 0.64×. Order-1 beats order-20 in
**10/10** runs, median ratio **8.63×**.

This is the strongest evidence against the tautology objection. "Granger wins on
a VAR" is entailed by the generator; a preference over *model order* is not.
Order-1 is correctly specified for a VAR(1) and order-20 is 20× over-
parameterised, so the probe has recovered a **statistical efficiency** property
of the estimator. Being matched across arms, it cannot be a module-size,
directedness or linearity artifact.

**(b) The signature is not a tuning artifact.** M05 is top-ranked in 9/10 runs
across a 50× λ range; module-share rank correlation is mean ρ = +0.86, min
+0.74. Ranks *below* first are not identified — the top-2 set holds in only
6/10 runs — so enrichment against a null is the reportable unit, never a
ranking.

**(c) The signature moves when the mechanism moves.** R1b observes the same
linear VAR through a squaring nonlinearity. `λ_g` must fall roughly tenfold,
since x² squares the coupling. At n=700: F1 **0.673**, with `node-only` 0.330,
`shuffled` 0.307 and `latent-directed` 0.307 — every control at the 0.333 chance
level. The signature moves to nonlinear DTW/LCS families (M09 4.26×, z=4.7;
M10 3.05×, z=5.9; the six highest-weighted SPIs all nonlinear), and the
`directed & linear` share falls 14–28 pp against three R0 λ references with CIs
excluding zero. The direction was stated in the generator's docstring before the
result was seen.

Caveat: 0.673 sits at the 2/3 symmetric ceiling and directed enrichment is weak
(1.27×, z=2.1), so the probe may be solving R1b *undirected*. Restricting to the
undirected sub-vocabulary would settle it.

**(d) A validity protocol.** `node-only` (do per-channel marginals suffice?),
`shuffled` (does pair correspondence matter?), permutation nulls, and stability
selection. It has rejected **four of our own results**: R1 (`node-only` 0.913
where chance is 0.5 — the nonlinearity sat in the state update and leaked the
motif through marginals), Kuramoto (symmetric generator, fork ≡ collider),
R1b's first pass (a signature read from a chance-level classifier), and TUH.

## 5. Real data: focal vs generalized seizure

TUH-EEG Seizure Corpus, FNSZ vs GNSZ, 22-channel bipolar `01_tcp_ar` montage,
8 s windows, patient-disjoint splits, 98 patients / 1590 windows.

The pre-registered hypothesis was mechanistic: focal seizures propagate from a
source (directed outflow), generalized are bilaterally synchronous (symmetric),
so the signature should load on directed measures.

**Falsified.** Every model sits in 0.52–0.61 macro F1 and none improves across a
17× increase in training data, where the same model goes 0.67 → 0.99 on R0.
`shuffled` matches or beats SPI-MPNN at 4 of 5 sample sizes. Directed enrichment
is **0.88×, z = −2.6** — significantly depleted — with weight concentrating on
generic precision-matrix estimators.

The failure is **not method-specific**: `latent-directed` (0.560) and `fixed-spi`
(0.567) fail equally, so no model class extracted signal. Three causes remain
undiscriminated: 8 s windows give ~1024 samples for 231 pairwise estimates, which
may leave estimator variance above the class difference; seizure-type labels
carry known inter-rater ambiguity; or the classes genuinely do not differ in
window-level pairwise coupling. A per-channel band-power baseline would separate
the first from the rest.

## 6. Conclusions

- A linear probe over a named statistical vocabulary can be attached to an MPNN
  **at no accuracy cost** and yields a falsifiable hypothesis about coupling
  mode. `fixed-spi` ties SPI-MPNN, so `w` is a **diagnostic overlay, not a
  performance component** — the honest framing.
- The readout **tracks the mechanism** rather than restating the generator: it
  recovers correct AR model order (10/10, matched arms), and it shifts in a
  pre-registered direction when the observation model changes.
- Scope is real and stated: the method requires classes to differ in *how*
  channels couple. `shuffled` reaching 0.82 by n=1000 on R0 bounds the topology
  claim to small n; TUH is a case where the requirement is not met.
- The protocol that produced these caveats has rejected four of our own results.
  A method that reports when its own interpretation is void is the part most
  worth keeping.
