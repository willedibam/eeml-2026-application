# Context Document: SPI-SPI Feature Space for Comparing Multivariate Time Series of Arbitrary Dimension

## Purpose of This Document
This document provides complete methodological context for ongoing work toward an EEML 2026 extended abstract submission. It describes a method for embedding multivariate time series (MTS) of arbitrary spatial dimension $M$ and temporal length $T$ into a common feature space. This document covers: (1) the problem and motivation, (2) the baseline method (SPI-SPI space), (3) its known limitations, and (4) illustrative case studies. A companion document describes the proposed extension using geometric deep learning.

---

## 1. Problem Statement

### 1.1 Feature-Based Univariate Time Series Analysis
Feature-based time series analysis transforms a univariate time series $x \in \mathbb{R}^T$ into a fixed-dimensional feature vector $\mathbf{f} \in \mathbb{R}^K$ by computing $K$ summary statistics (e.g., mean, autocorrelation, spectral entropy). This eliminates dependence on $T$, enabling comparison of time series of different lengths in a common $K$-dimensional feature space amenable to standard statistical and machine learning methods.

### 1.2 The Multivariate Extension Problem
For an $M$-channelled multivariate time series $\mathbf{X} \in \mathbb{R}^{M \times T}$, the spatial dimension $M$ introduces a second axis of variation. Feature-based methods that summarise pairwise interactions between channels (producing, e.g., $M \times M$ matrices) eliminate dependence on $T$ but remain sensitive to $M$. Specifically: if two MTS processes $\mathbf{X} \in \mathbb{R}^{M_1 \times T_1}$ and $\mathbf{Y} \in \mathbb{R}^{M_2 \times T_2}$ have $M_1 \neq M_2$, their pairwise interaction matrices live in different-dimensional spaces and cannot be directly compared.

Naïve solutions such as truncating to the minimum common channel count discard channels and, with them, potentially important dynamical regimes and inter-channel dependency structures. The problem is therefore: **how can we embed MTS of arbitrary $M$ and $T$ into a common feature space that preserves informative dependency structures?**

### 1.3 Existing Approaches
This problem is not entirely unaddressed. Several families of methods can compare graphs (or systems) of different sizes:

- **Graph kernels** (Weisfeiler-Leman subtree, random walk, graphlet kernels) define kernel functions between graphs of arbitrary size and are well-established in graph classification.
- **Global graph pooling in GNNs** (sum/mean/attention readout) produces fixed-dimensional embeddings regardless of node count.
- **NetSimile and similar feature-extraction approaches** compute distributions of local structural features (degree, clustering coefficient, etc.) and compare their statistics.

The method described here is a novel instantiation within this broader family, distinguished by: (a) using a diverse, interpretable library of pairwise interaction statistics as the input representation, and (b) exploiting the *relationships between* different statistical characterisations as the embedding, rather than structural graph features.

---

## 2. The SPI-SPI Feature Space (Baseline Method)

### 2.1 Pairwise Interaction Matrices (MPIs)
Let $\mathbf{X} \in \mathbb{R}^{M \times T}$ be an MTS with channels $\{X_t^{(i)}\}_{i=1}^M$. The Python library `pyspi` computes a library of $K > 250$ statistics of pairwise interaction (SPIs) between channels. For the $k$-th SPI, computing it on every ordered pair $(i, j)$ yields an $M \times M$ adjacency matrix

$$\mathrm{MPI}_k \in \mathbb{R}^{M \times M}, \quad [\mathrm{MPI}_k]_{ij} = \mathrm{SPI}_k(X^{(i)}, X^{(j)}).$$

We call this a **matrix of pairwise interaction (MPI)**. Geometrically, this is a weighted directed graph on $M$ nodes, where node $i$ represents channel $X^{(i)}$ and edge weight $(i,j)$ is the value of the $k$-th SPI computed between channels $i$ and $j$.

Computing all $K$ SPIs yields a tensor $\mathcal{M} \in \mathbb{R}^{K \times M \times M}$ — a collection of $K$ interaction graphs on the same $M$ nodes.

**Note on temporal invariance.** Each SPI maps a pair of time series (of any length $T$) to a scalar. Therefore, $\mathcal{M}$ is invariant to $T$. Two MTS $\mathbf{X} \in \mathbb{R}^{M \times T_1}$ and $\mathbf{Y} \in \mathbb{R}^{M \times T_2}$ with $T_1 \neq T_2$ produce MPIs of identical dimension, enabling direct comparison of their interaction structures.

**Remaining limitation.** The tensor $\mathcal{M} \in \mathbb{R}^{K \times M \times M}$ still depends on $M$. Systems with different channel counts produce tensors of different size.

### 2.2 Constructing the SPI-SPI Feature Vector
To eliminate dependence on $M$, we compute a second-order summary: the correlation between pairs of SPIs across all pairwise channel interactions within a single MTS.

For the $k$-th MPI, extract the $\binom{M}{2}$ off-diagonal entries (or all $M(M-1)$ entries if the SPI is asymmetric) into a vector $\mathbf{v}_k \in \mathbb{R}^{M(M-1)/2}$. For two SPIs $k$ and $k'$, compute

$$f_{kk'} = \mathrm{corr}(\mathbf{v}_k, \mathbf{v}_{k'})$$

where $\mathrm{corr}$ is Pearson's product-moment correlation. This yields a scalar summarising how similarly the two SPIs rank the pairwise interactions within the system.

For $K$ SPIs, we obtain a $\binom{K}{2}$-dimensional feature vector

$$\mathbf{f} = \{f_{kk'}\}_{k < k'} \in \mathbb{R}^{\binom{K}{2}}$$

which we term the **SPI-SPI feature vector**, and the resulting space **SPI-SPI space**.

**Key property.** The dimension of $\mathbf{f}$ depends only on $K$ (the number of SPIs chosen), not on $M$ or $T$. Therefore, any two MTS — regardless of their spatial or temporal dimensions — can be embedded into this common $\binom{K}{2}$-dimensional space.

### 2.3 Interpretation
Each feature $f_{kk'}$ encodes the empirical relationship between two statistical characterisations of pairwise interaction across the channels of a single system. Comparing $\mathbf{f}$ across different MTS processes reveals similarities and differences in the **character** or **nature** of dependency structures, rather than in the strength of specific channel-pair interactions.

This is a critical distinction: SPI-SPI space is agnostic to *which* channels are coupled. It captures the statistical fingerprint of the coupling regime but discards the topological arrangement. This is both a strength (enabling comparison across different $M$) and a limitation (see §2.4).

### 2.4 Choice of Correlation Measure
The off-diagonal entries of an MPI are not statistically independent — entries sharing a node (e.g., $\mathrm{SPI}_k(1,2)$ and $\mathrm{SPI}_k(1,3)$) are confounded by the marginal properties of the shared channel. This **network autocorrelation** inflates the effective sample size when computing $\mathrm{corr}(\mathbf{v}_k, \mathbf{v}_{k'})$, potentially leading to overconfident correlation estimates.

We use Pearson correlation rather than Spearman rank correlation for $f_{kk'}$. The rationale: rank transformation may corrupt subtle quantitative signatures — specifically, when the signal lies in the relative *magnitudes* of SPI values across channel pairs (not just their ordering), Spearman's rank space discards this information. Since the method's discriminative power relies on detecting fine-grained shifts in the covariation of SPIs, Pearson is preferred.

However, this choice should be validated empirically. A robustness check comparing Pearson, Spearman, and Kendall's $\tau$ for the inter-MPI correlation is warranted. Additionally, permutation-based significance testing or QAP (quadratic assignment procedure, standard in social network analysis) should be considered to account for the non-independence structure.

### 2.5 Known Limitations

1. **Information bottleneck.** Reducing an $M \times M$ adjacency matrix to a single scalar per SPI pair is aggressive. Graph topology — community structure, hub-spoke patterns, motifs, degree distribution — is entirely discarded. Two systems with identical coupling *character* but different network architectures (e.g., star vs. ring topology) are indistinguishable in SPI-SPI space.

2. **Scalability of feature space.** For $K$ SPIs, the feature vector has $\binom{K}{2}$ entries. With $K = 250$, this is 31,125 features — high-dimensional relative to typical sample sizes, requiring careful regularisation or feature selection.

3. **No learned representation.** The features are hand-crafted. There is no mechanism to learn which SPIs or which inter-SPI relationships are most informative for a given task.

4. **Network autocorrelation in the correlation estimate.** As noted in §2.4, the non-independence of off-diagonal entries biases the correlation. This is a statistical subtlety that does not invalidate the method but requires careful treatment.

---

## 3. Illustrative Case Studies

These case studies demonstrate the interpretive logic of SPI-SPI space on controlled synthetic systems. They serve as pedagogical motivation, not as the primary contribution.

### 3.1 Differentiating Linear, Nonlinear, and Non-Monotonic Dependencies

**Setup.** We construct an $M \times T$ MTS where each channel is a noisy, filtered copy of a latent autoregressive process

$$z_t = a \cdot z_{t-1} + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, 1),$$

passed through a nonlinear filter $g(z; \alpha)$ and corrupted by observation noise:

$$X_t^{(i)} = g(z_t; \alpha_i) + \eta_t^{(i)}, \quad \eta_t^{(i)} \sim \mathcal{N}(0, \sigma_\eta^2).$$

The filter $g(z; \alpha)$ is defined as: (1) rescale $z_t$ to $[0, 1]$ via min-max normalisation, (2) map to the interval $[-\alpha, \alpha]$, (3) apply $\sin(\cdot)$, and (4) normalise to $[-1, 1]$ by dividing by the theoretical maximum $g_{\max} = \min(\sin(\alpha), 1)$. The normalisation preserves a constant signal-to-noise ratio across filter configurations.

**Experimental logic.** The parameter $\alpha$ controls the degree of nonlinearity:
- For $\alpha \ll 1$: $\sin(x) \approx x$ (linear regime).
- For $\alpha$ approaching $\pi/2$: the mapping becomes nonlinear but monotonic.
- For $\alpha > \pi/2$: the mapping becomes non-monotonic.

Each channel $i$ draws $\alpha_i \sim \mathrm{Uniform}(\pi/\epsilon, A)$ where $\epsilon$ is small (e.g., $\epsilon = 1/64$). The control parameter $A$ governs the maximum nonlinearity across channels. As $A$ increases toward $\pi$, a greater proportion of channels are filtered through nonlinear (and eventually non-monotonic) transformations.

**Expected behaviour of $f_{kk'} = \mathrm{corr}(r, \rho)$.** In the linear regime, Pearson's $r$ and Spearman's $\rho$ agree closely across channel pairs, yielding $f_{kk'} \approx 1$. As more channels undergo nonlinear filtering, the Pearson MPI values degrade (violating linearity/Gaussianity assumptions) while Spearman values remain stable (rank-invariance under monotonic transformation). The result is a progressive decoupling: $f_{kk'}$ decreases as a function of $A$, with a sharper transition near $A = \pi/2$ where non-monotonicity begins to also degrade Spearman.

**Interpretation.** The degree to which $r$ and $\rho$ are correlated across channel pairs (the feature $f_{kk'}$) provides a signature of the proportion and character of nonlinear interactions in the system. This illustrates how SPI-SPI features encode dependency *character* without reference to specific channel identities.

**Important caveat.** The latent driver $z_t$ is itself a non-monotonic autoregressive process. The method characterises the nature of *inter-channel* dependence, not the marginal dynamics of individual channels. The filter $g$ modulates how each channel relates to the shared driver, and it is these inter-channel relationships that the SPIs (and hence SPI-SPI features) capture.

### 3.2 Temporal Misalignment via DTW and Euclidean Distance

**Setup.** Channels are noisy, *temporally warped* copies of the latent driver $z_t$. The warping for channel $i$ is generated by a random walk in $(t_{\mathrm{orig}}, t_{\mathrm{warp}})$-space: at each timestep, with probability $p_{\mathrm{step}}^{(i)}$, an "L-shaped" excursion deviates from the identity line. The excursion size is geometric: $s \sim \mathrm{Geom}(q)$, with $q = 0.5$ (giving $\mathbb{E}[s] = 2$).

The warp intensity parameter is $p_{\mathrm{step}}^{(i)} \sim \mathrm{Uniform}[0, a]$, making $a$ the single control parameter governing the degree of temporal misalignment across the MTS.

**Warpedness measure.** Per-channel warpedness is defined as the $L_1$ deviation from the identity alignment:

$$\mathcal{W}^{(i)} = \sum_{t=1}^T |t_{\mathrm{orig}, t} - t_{\mathrm{warp}, t}^{(i)}|.$$

The per-MTS aggregate $\mathcal{W} = M^{-1} \sum_i \mathcal{W}^{(i)}$ satisfies $\mathbb{E}[\mathcal{W}] \approx 2aT$, scaling linearly in $a$. The normalised form $\bar{\mathcal{W}} = (MT)^{-1} \sum_i \mathcal{W}^{(i)}$ satisfies $\mathbb{E}[\bar{\mathcal{W}}] \approx 2a$, depending only on $a$.

**Expected behaviour of $f_{kk'} = \mathrm{corr}(\mathrm{ED}, \mathrm{DTW})$.** For unwarped signals ($a \approx 0$), DTW reduces to pointwise Euclidean distance (the optimal alignment path is the identity). As $a$ increases, DTW's flexible alignment compensates for warping while Euclidean distance degrades. The decoupling of ED and DTW across channel pairs — captured by a decrease in $f_{kk'}$ — is a signature of the degree of temporal misalignment in the system.

### 3.3 Extension Beyond Toy Examples
In practice, individual features $f_{kk'}$ carry subtle signals, and it is the collective pattern across the full $\binom{K}{2}$-dimensional vector that constitutes the system's dynamical signature. By selecting $K$ SPIs spanning diverse statistical families — correlation measures, distance measures, information-theoretic quantities, spectral measures, causal/directed measures — the SPI-SPI feature space forms a "meshgrid" of statistical assumptions capable of jointly characterising complex dependency regimes.

**Preliminary results.** Clustering analyses in SPI-SPI space, visualised via PCA, UMAP, and t-SNE, show clear separation of synthetic MTS classes differing in dependency structure, with separation robust to variation in $M$ and $T$. Formal benchmarks against baselines are pending.

---

## 4. Open Questions and Status

- The correlation choice for $f_{kk'}$ (§2.4) requires empirical validation.
- The method has not been benchmarked against graph kernel baselines.
- Real-world MTS experiments are needed.
- The known limitation of discarding graph topology (§2.5) motivates the geometric deep learning extension described in the companion document.
