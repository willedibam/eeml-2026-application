# Graph Construction from a Statistical Vocabulary: Structured Inductive Bias for Relational Learning on Time Series

**[Author Name]**$^{1}$ and **[Supervisor Name]**$^{1}$

$^{1}$The University of Sydney

---

## Abstract

Graph neural networks for multivariate time series require a graph that is rarely given. We parameterise graph construction over a vocabulary of $K \approx 125$ named pairwise statistics, spanning causal, spectral, linear, and information-theoretic families, computed via `pyspi` [Cliff et al., 2023]. A learned weight vector $\mathbf{w}$, jointly optimised with a message-passing network, selects which statistical notion of coupling is task-relevant and produces an interpretable statistical signature of the relational structure. We prove that symmetric construction operators cannot distinguish Markov-equivalent topologies, and confirm this empirically: symmetric baselines remain at or below 59% while directed-statistic models achieve near-perfect classification. On synthetic VAR(1) topology classification (30 seeds), the learned $\mathbf{w}$ recovers spectral Granger causality as the dominant coupling mode, matching the generating process.

---

## 1 Introduction

Message passing neural networks (MPNNs) derive their power from propagating information along relational structure [Gilmer et al., 2017; Battaglia et al., 2018]. When applied to multivariate time series (MTS), the relational structure is unobserved and must be constructed from data. Two regimes dominate current practice, each with a characteristic limitation.

The first regime fixes the graph from a single pre-specified statistic. Pearson correlation is the default in functional connectivity studies [Bullmore & Sporns, 2009], while spatial distance is standard in traffic forecasting [Li et al., 2018]. This approach commits to one mathematical notion of dependence. If the task-relevant coupling is directed, nonlinear, or frequency-specific, the limitation lies in the graph construction operator, not the data. Liu et al. [2025] recently benchmarked 239 pairwise statistics for functional connectivity mapping, finding "substantial quantitative and qualitative variation across FC methods," confirming that the choice of construction statistic is consequential yet understudied.

The second regime learns the graph from latent embeddings. Neural Relational Inference [Kipf et al., 2018], Graph WaveNet [Wu et al., 2019], and MTGNN [Wu et al., 2020] optimise adjacency matrices jointly with the downstream task. This recovers flexibility at the cost of semantics: an edge weight of 0.7 encodes no information about what form of dependence it represents. Such graphs cannot be validated against domain knowledge, compared across tasks, or used to generate scientific hypotheses about the data-generating process.

Veličković & Blundell [2021] argued that aligning neural architectures with algorithmic structure yields favourable sample complexity; Veličković et al. [2022] demonstrated that injecting algorithmic priors improves sample efficiency over learning from scratch. We apply this principle to graph construction. Rather than re-learning what "connectivity" means from raw data, we supply a vocabulary of named pairwise statistics with known mathematical properties and let end-to-end learning identify which elements are task-relevant.

For each ordered pair $(i, j)$ in an MTS, we precompute a $K$-dimensional descriptor $\mathbf{E}_{ij}$ whose coordinates are named pairwise statistics — transfer entropy, spectral Granger causality, phase locking value, covariance, and approximately 125 others — drawn from six complementary mathematical families using `pyspi` [Cliff et al., 2023]. Graph construction is then a linear probe of this descriptor space:

$$A_{ij} = \mathrm{softplus}(b + \mathbf{w}^\top \mathbf{E}_{ij}), \quad \text{top-}d \text{ sparsification per node}$$

where $\mathbf{w} \in \mathbb{R}^K$ is jointly optimised with the downstream MPNN [cf. Alain & Bengio, 2017]. Group lasso regularisation [Yuan & Lin, 2006] over SPI families encourages family-level selection, producing an interpretable statistical signature of the task's relational structure.

---

## 2 Expressiveness of Graph Construction Operators

We formalise the intuition that the choice of construction statistic constrains what an MPNN can learn, drawing an analogy to Weisfeiler-Lehman expressiveness results for message passing [Xu et al., 2019; Morris et al., 2019].

**Proposition.** Consider three 3-node VAR(1) motifs over nodes $A, B, C$: chain ($A \to B \to C$), fork ($A \leftarrow B \to C$), and collider ($A \to B \leftarrow C$). All three share skeleton $A{-}B{-}C$. Chain and fork are Markov equivalent [Verma & Pearl, 1990]: they encode identical conditional independence relations ($A \perp C \mid B$, and no other). Any graph construction operator using only symmetric statistics assigns identical edge weights to chain and fork, regardless of sample size. No MPNN operating on such a graph can exceed $2/3$ accuracy on the three-class task.

Two DAGs are Markov equivalent if and only if they share the same skeleton and v-structures [Verma & Pearl, 1990]. Chain and fork share skeleton $\{A{-}B,\; B{-}C\}$ and have no v-structures, so any function of the joint distribution $P(X_i, X_j)$ that is symmetric in its arguments must assign identical values to both. In finite samples, estimation noise introduces small asymmetries, but these are insufficient for reliable discrimination, as confirmed empirically (Correlation $\leq 59$% across all 30 seeds and sample sizes). The collider has a v-structure and encodes a different independence ($A \perp C$ marginally), making it distinguishable by all methods and serving as an internal control.

Directed statistics break the chain-fork equivalence because they distinguish the ordered pair $(i, j)$ from $(j, i)$. In the chain, $\mathrm{SGC}(A \to B) \gg \mathrm{SGC}(B \to A)$; in the fork, $\mathrm{SGC}(B \to A) \approx \mathrm{SGC}(B \to C) \gg 0$ but $\mathrm{SGC}(A \to C) \approx 0$. The asymmetric descriptor tensor $\mathbf{E}_{ij} \neq \mathbf{E}_{ji}$ provides the information needed for discrimination.

---

## 3 Method

Given an MTS $X \in \mathbb{R}^{M \times T}$ (z-scored), we compute $K$ pairwise statistics per ordered pair via `pyspi` [Cliff et al., 2023], yielding a descriptor tensor $\mathbf{E} \in \mathbb{R}^{M \times M \times K}$. The statistics are partitioned into six families: causal (transfer entropy [Schreiber, 2000], Granger causality [Granger, 1969], spectral GC [Geweke, 1982], phase slope index), spectral (coherence, PLV, PLI), linear (covariance, precision, cross-correlation), information-theoretic (mutual information), distance-based (DTW, Euclidean), and rank-based (Spearman, Kendall). Causal measures are asymmetric ($\mathbf{E}_{ij} \neq \mathbf{E}_{ji}$), enabling directed graph construction that symmetric operators cannot represent.

The adjacency is computed as $A_{ij} = \mathrm{softplus}(b + \mathbf{w}^\top \mathbf{E}_{ij})$ with $K + 1$ learnable parameters and sparsified to the top-$d$ outgoing edges per node. The loss combines the task objective with group lasso regularisation: $\mathcal{L} = \mathcal{L}_{\mathrm{task}} + \lambda_1 \|\mathbf{w}\|_1 + \lambda_g \sum_{g \in \mathcal{G}} \|\mathbf{w}_g\|_2$. Training uses Adam with learning rate warmup over 60 epochs and restart selection (best of 2 initialisations), addressing the non-convex landscape of the joint graph-construction optimisation.

Retained edges carry the full descriptor $\mathbf{E}_{ij}$ as attributes. An edge network $\phi(\mathbf{E}_{ij}) = \mathrm{MLP}_\phi(\mathbf{E}_{ij})$ conditions messages on the descriptor:

$$\mathbf{m}_{ij} = \mathrm{MLP}_m([\mathbf{h}_j \| \mathbf{h}_i \| \phi(\mathbf{E}_{ij})]), \quad \mathbf{h}_i' = \mathbf{h}_i + \mathrm{LN}\!\left(\sum_j A_{ij} \cdot \mathbf{m}_{ij}\right)$$

The vocabulary therefore informs both topology (via $\mathbf{w}$) and message content (via $\phi$). Global mean+max pooling and a classifier produce graph-level predictions.

---

## 4 Experiments

We generate $M{=}10$ node VAR(1) processes ($T{=}500$) with a 3-node directed motif ($A{\to}B{\to}C$, $A{\leftarrow}B{\to}C$, or $A{\to}B{\leftarrow}C$) embedded among 7 nuisance AR(1) channels ($\rho{=}0.8$). Coupling strengths are drawn as $\alpha \sim \mathrm{Uniform}(0.3, 0.7)$, with 1500 instances per class. The `pyspi` vocabulary yields $K{=}125$ statistics after variance filtering. This setup directly tests the expressiveness proposition: chain and fork are Markov-equivalent, so symmetric construction operators are provably limited.

We evaluate models that each isolate a specific component. SPI-MPNN is the full proposed method with learned $\mathbf{w}$. Fixed-SPI uses the same vocabulary as dense edge features on a fully connected graph, without learned topology. Correlation and Latent represent the two dominant existing paradigms. Edge-Ablation uses the SPI-derived adjacency but zeros edge features during message passing. Shuffled randomly permutes SPI values across pairs, controlling for pair-correspondence. SGC-Only uses the oracle-best single SPI identified retrospectively from the learned $\mathbf{w}$.

**Results.** Macro F1 (%), 30 seeds, top-$d{=}5$, group $\lambda{=}0.02$.

| $n$/class | SPI-MPNN | Fixed-SPI | Correlation | Latent | Edge-Abl. | Shuffled | Node-Only |
|---|---|---|---|---|---|---|---|
| 20 | 67±14 | 58±22 | 35±5 | 32±2 | 40±15 | 35±3 | 33±2 |
| 50 | 86±8 | 84±19 | 41±11 | 32±2 | 42±16 | 36±6 | 33±2 |
| 100 | 95±4 | 98±1 | 53±9 | 32±2 | 58±26 | 38±8 | 33±2 |
| 200 | 98±3 | 99±1 | 59±1 | 31±3 | 75±24 | 44±12 | 33±2 |
| 500 | 99±2 | 100±0.3 | 59±1 | 31±2 | 93±11 | 67±19 | 32±2 |
| 1000 | 100±0.4 | 100±0.3 | 59±2 | 31±2 | 96±7 | 82±14 | 33±2 |

Additional ablations: SGC-Only (79–99.7%), Top-3 oracle SPIs (88–99.7%), TE-only (29–18%, chance-level), MLP-Mix (63–98%).

**[FIGURE PLACEHOLDER]** *Sample efficiency curves (F1 vs n/class) with ±1 s.d. bands, log-scale x-axis.*

Three findings emerge from these results.

First, the SPI vocabulary is the dominant inductive bias. All SPI-consuming models dramatically outperform non-SPI baselines across all sample sizes. The gap is not incremental: at $n{=}500$, SPI models achieve 93–100% while Correlation reaches only 59% and Latent remains at chance (31%). The vocabulary provides the structured prior knowledge that makes the task tractable in the small-sample regime. The oracle-best single SPI (spectral Granger causality) achieves near-perfect performance at $n \geq 200$ but exhibits catastrophic seed failures at low $n$ (2/30 at $n{=}20$); the full vocabulary eliminates these failures through the redundancy of 125 complementary estimators.

Second, symmetric construction is empirically limited, exactly as the Markov equivalence argument predicts. Correlation remains at or below 59% across all sample sizes, never approaching the theoretical $2/3$ ceiling, because top-$d$ sparsification of $|r_{ij}|$ discards motif edges in favour of high-correlation nuisance pairs. The latent model performs at chance across all 30 seeds and all sample sizes, confirming that symmetric node-embedding construction with no statistical prior cannot access the chain-fork distinction. These are among the cleanest results in the study.

Third, edge content is essential, not merely topology. The edge-ablation model uses the same SPI-derived adjacency as SPI-MPNN but zeros edge features during message passing. At $n{=}20$, 24 of 30 seeds catastrophically fail; even at $n{=}500$, performance is 93±11% versus 99±2% for the full model. The gap demonstrates that the MPNN actively uses the SPI descriptor to condition messages, and that the vocabulary informs both which edges exist and what information flows along them.

The learned $\mathbf{w}$ constitutes an interpretable statistical signature of the task. Under group lasso ($\lambda_g{=}0.02$), the causal family (48 SPIs) carries $6\times$ the $L_2$ norm of the next family. The five highest-weighted individual SPIs are all directed temporal measures: spectral Granger causality in the 0.25–0.5 Hz band, time-domain Granger causality, and extended-lag variants. For a VAR(1) process, causal structure manifests as asymmetric spectral transfer functions, so this recovery is a prediction from the generating process confirmed by the learned parameters.

Fixed-SPI matches or exceeds SPI-MPNN at most sample sizes, establishing that the vocabulary provides sufficient information as dense edge features for effective relational reasoning, even without learned topology. The advantage of SPI-MPNN is not accuracy but interpretability: the learned $\mathbf{w}$ produces a named statistical signature that Fixed-SPI cannot. Both models converge to near-perfect performance, confirming that the vocabulary itself, not the construction mechanism, is the primary contribution.

---

## 5 Discussion

Graph structure learning methods based on metric learning, neural optimisation, and attention mechanisms [Zhu et al., 2021] universally use latent embeddings that sacrifice interpretability. CauGNN [Duan et al., 2023] constructs graphs from a single pre-specified statistic (transfer entropy), while ADLGNN [Sriramulu et al., 2023] initialises from statistics but overwrites them with attention, losing interpretability. Our approach occupies an unoccupied point in this design space: a vocabulary of named statistics with learnable per-statistic weights that remain inspectable after training. Nguyen et al. [2025] argued that named statistical features capture interactions that raw-value methods miss; we operationalise this insight within a GNN framework.

Several limitations should be noted. SPI computation is expensive ($O(K \cdot M^2)$ per sample), though this cost is amortisable and independent of training. The vocabulary is a design choice, though group sparsity mitigates sensitivity to its composition. All SPIs are bivariate, so higher-order interactions require multi-hop message passing to compose. Most importantly, the current study is synthetic. Real-world validation on domains with established ground truth (e.g., EEG motor imagery, fMRI connectivity) is needed to confirm that the learned statistical signature produces domain-interpretable findings.

The statistical signature produced by the learned $\mathbf{w}$ identifies spectral Granger causality as the dominant coupling mode, constituting a testable hypothesis about the data-generating process rather than merely a classification artefact. This is a form of scientific output that latent graph learners structurally cannot produce. On real-world MTS, the same mechanism would identify which notion of coupling is relevant to the task, yielding domain-interpretable graph structure directly from the learned parameters.

---

## References

Alain, G. & Bengio, Y. (2017). Understanding intermediate layers using linear classifier probes. *ICLR Workshop*.

Battaglia, P.W. et al. (2018). Relational inductive biases, deep learning, and graph networks. *arXiv:1806.01261*.

Bullmore, E. & Sporns, O. (2009). Complex brain networks. *Nature Reviews Neuroscience*, 10(3), 186–198.

Cliff, O.M. et al. (2023). Unifying pairwise interactions in complex dynamics. *Nature Computational Science*, 3(10), 883–893.

Duan, Z. et al. (2023). Multivariate time series forecasting with transfer entropy graph. *Tsinghua Science and Technology*, 28(1), 141–149.

Geweke, J. (1982). Measurement of linear dependence and feedback between multiple time series. *JASA*, 77(378), 304–313.

Gilmer, J. et al. (2017). Neural message passing for quantum chemistry. *ICML*.

Granger, C.W.J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 37(3), 424–438.

Kipf, T. et al. (2018). Neural relational inference for interacting systems. *ICML*.

Li, Y. et al. (2018). Diffusion convolutional recurrent neural network. *ICLR*.

Liu, Z.-Q. et al. (2025). Benchmarking methods for mapping functional connectivity in the brain. *Nature Methods*.

Morris, C. et al. (2019). Weisfeiler and Leman go neural. *AAAI*, 4602–4609.

Nguyen, A. et al. (2025). A feature-based information-theoretic approach for detecting interpretable, long-timescale pairwise interactions. *Physical Review Research*, 7, 043283.

Schreiber, T. (2000). Measuring information transfer. *Physical Review Letters*, 85(2), 461–464.

Sriramulu, A. et al. (2023). Adaptive dependency learning graph neural networks. *Information Sciences*, 625, 700–714.

Veličković, P. & Blundell, C. (2021). Neural algorithmic reasoning. *Patterns*, 2(7), 100273.

Veličković, P. et al. (2022). Reasoning-modulated representations. *LoG*, PMLR 198.

Verma, T.S. & Pearl, J. (1990). Equivalence and synthesis of causal models. *UAI*, 220–227.

Wu, Z. et al. (2019). Graph WaveNet for deep spatial-temporal graph modeling. *IJCAI*.

Wu, Z. et al. (2020). Connecting the dots: Multivariate time series forecasting with graph neural networks. *KDD*.

Xu, K. et al. (2019). How powerful are graph neural networks? *ICLR*.

Yuan, M. & Lin, Y. (2006). Model selection and estimation in regression with grouped variables. *JRSSB*, 68(1), 49–67.

Zhu, Y. et al. (2021). Deep graph structure learning for robust representations: A survey. *arXiv:2103.03036*.
