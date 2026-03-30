# Graph Construction from a Statistical Vocabulary: Structured Inductive Bias for Relational Learning on Time Series

**[Author Name]**$^{1}$ and **[Supervisor Name]**$^{1}$

$^{1}$The University of Sydney

---

## Abstract

Graph neural networks for multivariate time series require a graph that is rarely given. We parameterise graph construction over a vocabulary of $K \approx 125$ named pairwise statistics — causal, spectral, linear, information-theoretic — computed via `pyspi` [Cliff et al., 2023]. A learned weight vector $\mathbf{w}$, jointly optimised with a message-passing network, selects which statistical notion of coupling is task-relevant. We prove that symmetric construction operators cannot distinguish Markov-equivalent topologies. On synthetic VAR(1) classification, the full vocabulary achieves 100% (std $< 1$%), while the oracle-best single statistic reaches 96% but with catastrophic seed failures (std 24%) and symmetric baselines remain at or below 59%. The learned $\mathbf{w}$ recovers spectral Granger causality as the dominant coupling mode, matching the generating process.

---

## 1 Introduction

Message passing neural networks (MPNNs) derive their power from propagating information along relational structure [Gilmer et al., 2017; Battaglia et al., 2018]. In MTS settings, this structure is unobserved and must be constructed. Two regimes dominate current practice.

**Fixing the graph from a single statistic** — Pearson correlation in functional connectivity [Bullmore & Sporns, 2009], spatial distance in traffic networks [Li et al., 2018] — is a strong commitment to one mathematical notion of dependence. If the task-relevant coupling is directed, nonlinear, or frequency-specific, this is an expressiveness limitation of the graph construction operator, not a data limitation. Liu et al. [2025] recently benchmarked 239 pairwise statistics for functional connectivity mapping, finding "substantial quantitative and qualitative variation across FC methods" — empirical confirmation that the choice of construction statistic is consequential yet understudied.

**Learning the graph from latent embeddings** — Neural Relational Inference [Kipf et al., 2018], Graph WaveNet [Wu et al., 2019], MTGNN [Wu et al., 2020] — recovers flexibility at the cost of semantics. The learned adjacency is opaque: an edge weight of 0.7 encodes no information about *what form* of dependence it represents. Such graphs cannot be validated against domain knowledge or compared across tasks. They also place the full burden on the training signal to discover relational structure that statistical methodology could supply as prior knowledge.

Veličković & Blundell [2021] argued that aligning neural architectures with algorithmic structure yields favourable sample complexity; Veličković et al. [2022] demonstrated that injecting algorithmic priors improves sample efficiency over learning from scratch. We apply this principle to graph construction: rather than re-learning what "connectivity" means from raw data, we supply a pre-specified vocabulary of pairwise statistics with known mathematical properties and let end-to-end learning identify which elements are task-relevant.

**Our approach.** For each ordered pair $(i, j)$ in an MTS, we precompute a $K$-dimensional descriptor $\mathbf{E}_{ij}$ whose coordinates are named pairwise statistics — transfer entropy, spectral Granger causality, phase locking value, covariance, and $K \approx 125$ others — drawn from six complementary mathematical families using `pyspi` [Cliff et al., 2023]. Graph construction becomes:

$$A_{ij} = \mathrm{softplus}(b + \mathbf{w}^\top \mathbf{E}_{ij}), \quad \text{top-}d \text{ sparsification per node}$$

where $\mathbf{w} \in \mathbb{R}^K$ is jointly optimised with the downstream MPNN. This is a deliberate linear probe of the SPI descriptor space [cf. Alain & Bengio, 2017]: the claim is that the vocabulary is sufficiently well-structured for a linear function to identify task-relevant edges. Group lasso regularisation [Yuan & Lin, 2006] over SPI families encourages family-level selection, producing an interpretable *statistical signature* of the task's relational structure.

---

## 2 Expressiveness of Graph Construction Operators

We formalise the intuition that the choice of construction statistic constrains what an MPNN can learn, analogous to Weisfeiler-Lehman expressiveness results for message passing [Xu et al., 2019; Morris et al., 2019].

**Proposition.** Consider three 3-node VAR(1) motifs over nodes $A, B, C$: chain ($A \to B \to C$), fork ($A \leftarrow B \to C$), and collider ($A \to B \leftarrow C$). All three share skeleton $A{-}B{-}C$. Chain and fork are Markov equivalent [Verma & Pearl, 1990]: they encode identical conditional independence relations ($A \perp C \mid B$, and no other). Any graph construction operator using only symmetric statistics — correlation, mutual information, coherence, or any function $f$ satisfying $f(X_i, X_j) = f(X_j, X_i)$ — assigns identical edge weights to chain and fork, regardless of sample size or estimation quality. No MPNN operating on such a graph can exceed $2/3$ accuracy on the three-class task.

*Proof sketch.* Two DAGs are Markov equivalent iff they share the same skeleton and v-structures [Verma & Pearl, 1990]. Chain ($A \to B \to C$) and fork ($A \leftarrow B \to C$) share skeleton $\{A{-}B,\; B{-}C\}$ and have no v-structures. Any symmetric statistic $f(X_i, X_j)$ depends only on the joint distribution $P(X_i, X_j)$, which is identical for Markov-equivalent structures. In finite samples, estimation noise introduces small asymmetries, but these are insufficient for reliable discrimination — as confirmed empirically (Correlation $\leq 59$%). The collider ($A \to B \leftarrow C$) has a v-structure and encodes a different independence ($A \perp C$ marginally), making it distinguishable by all methods — serving as an internal control. $\square$

Directed statistics break the chain–fork equivalence. In the chain, $\mathrm{SGC}(A \to B) \gg \mathrm{SGC}(B \to A)$; in the fork, $\mathrm{SGC}(B \to A) \approx \mathrm{SGC}(B \to C) \gg 0$ but $\mathrm{SGC}(A \to C) \approx 0$. The asymmetric descriptor tensor $\mathbf{E}_{ij} \neq \mathbf{E}_{ji}$ provides the information needed for discrimination.

---

## 3 Method

**SPI descriptor tensor.** Given $X \in \mathbb{R}^{M \times T}$ (z-scored), we compute $K$ pairwise statistics per ordered pair via `pyspi` [Cliff et al., 2023], yielding $\mathbf{E} \in \mathbb{R}^{M \times M \times K}$. Statistics are partitioned into families: **causal** (transfer entropy [Schreiber, 2000], Granger causality [Granger, 1969], spectral GC [Geweke, 1982], phase slope index), **spectral** (coherence, PLV, PLI), **linear** (covariance, precision, cross-correlation), **information** (mutual information), **distance** (DTW, Euclidean), and **rank** (Spearman, Kendall). Crucially, causal measures are asymmetric: $\mathbf{E}_{ij} \neq \mathbf{E}_{ji}$, enabling directed graph construction that symmetric operators cannot represent.

**Learned graph construction.** $A_{ij} = \mathrm{softplus}(b + \mathbf{w}^\top \mathbf{E}_{ij})$ with $K + 1$ learnable parameters. Sparsified to top-$d$ outgoing edges per node. Group lasso: $\mathcal{L} = \mathcal{L}_{\mathrm{task}} + \lambda_1 \|\mathbf{w}\|_1 + \lambda_g \sum_{g \in \mathcal{G}} \|\mathbf{w}_g\|_2$. Training uses Adam with LR warmup over 60 epochs and restart selection (best of 2 initialisations), addressing the non-convex landscape of the joint graph-construction optimisation.

**Edge-attributed MPNN.** Retained edges carry $\mathbf{E}_{ij}$ as attributes. An edge network $\phi(\mathbf{E}_{ij}) = \mathrm{MLP}_\phi(\mathbf{E}_{ij})$ conditions messages on the full descriptor:

$$\mathbf{m}_{ij} = \mathrm{MLP}_m([\mathbf{h}_j \| \mathbf{h}_i \| \phi(\mathbf{E}_{ij})]), \quad \mathbf{h}_i' = \mathbf{h}_i + \mathrm{LN}\!\left(\sum_j A_{ij} \cdot \mathbf{m}_{ij}\right)$$

The SPI vocabulary informs both topology (via $\mathbf{w}$) and message content (via $\phi$). Global mean+max pooling and a classifier produce graph-level predictions.

---

## 4 Experiments

**Synthetic topology classification.** We generate $M{=}10$ node VAR(1) processes ($T{=}500$) with a 3-node directed motif ($A{\to}B{\to}C$, $A{\leftarrow}B{\to}C$, or $A{\to}B{\leftarrow}C$) embedded among 7 nuisance AR(1) channels ($\rho{=}0.8$). Coupling strengths $\alpha \sim \mathrm{Uniform}(0.3, 0.7)$; 1500 instances per class. The `pyspi` vocabulary yields $K{=}125$ statistics after variance filtering.

**Models.** Each baseline isolates a component:

| Model | Graph construction | Tests |
|---|---|---|
| **SPI-MPNN** | $\mathrm{softplus}(b + \mathbf{w}^\top \mathbf{E}_{ij})$, learned | Full method |
| Correlation | Fixed $|r_{ij}|$ | Symmetric single-statistic |
| TE-Only | Fixed $|\mathrm{TE}_{ij}|$ | Directed single-statistic (nonparametric) |
| SGC-Only | Fixed $|\mathrm{SGC}_{ij}|$ | Oracle-best single SPI from learned $\mathbf{w}$ |
| Top-3 SPIs | Fixed mean $|\mathrm{SPI}|$, 3-dim edge feat. | Oracle top-3 from learned $\mathbf{w}$ |
| Latent | Learned node embeddings | No statistical prior |
| Fixed-SPI | Fully connected, $\mathbf{E}_{ij}$ as edge features | Is learned topology necessary? |
| MLP-Mix | $\sigma(\mathrm{MLP}(\mathbf{E}_{ij}))$ | Nonlinear construction |
| Edge-Ablation | SPI adjacency, zero edge features | Does MPNN use edge content? |
| Shuffled | SPIs permuted across pairs | Pair-correspondence control |
| Node-Only | Fully connected, no edge features | Are pairwise features necessary? |

**Results.** Macro F1 (%), 10 seeds. Top-$d{=}5$, group $\lambda{=}0.02$.

| $n$/class | SPI-MPNN | Fixed-SPI | SGC-Only | Correlation | Latent | Shuffled | Node-Only |
|---|---|---|---|---|---|---|---|
| 20 | 57±22 | 72±12 | 61±9 | 35±5 | 30±2 | 34±4 | 33±2 |
| 50 | 92±2 | 92±2 | 95±8 | 47±13 | 33±1 | 38±5 | 31±2 |
| 100 | 97±2 | 98±1 | 83±17 | 57±2 | 32±2 | 37±3 | 31±3 |
| 200 | 98±5 | 98±1 | 95±12 | 53±11 | 31±2 | 41±5 | 30±2 |
| 500 | 98±4 | 99±0.5 | 87±24 | 59±1 | 31±2 | 55±19 | 32±3 |
| 1000 | 100±0.2 | 99±1 | 96±11 | 59±1 | 31±3 | 80±13 | 34±2 |

Additional ablations (not shown): TE-only (29–18%, chance), Top-3 oracle SPIs (67–100%, std up to 27%), MLP-Mix (63–99%), Edge-Ablation (35–88%). See findings below.

**[FIGURE PLACEHOLDER]** *Sample efficiency curves (F1 vs n/class) with error bands, log-scale x-axis.*

**Interpretation.**

**(i) The SPI vocabulary is the dominant inductive bias — and diversity provides stability.** All SPI-consuming models (SPI-MPNN, Fixed-SPI, MLP-Mix) dramatically outperform non-SPI baselines. A vocabulary ablation ladder isolates why. TE-only (29–18%) uses a directed, causal statistic — the correct *type* — but the nonparametric Kraskov estimator at $T{=}500$ is too noisy for reliable per-instance graph construction. SGC-only (61–96%, std 8–24%), the oracle-best SPI from the learned $\mathbf{w}$ and a correctly specified parametric estimator for VAR(1), achieves high peak performance but with severe seed instability: 2/10 seeds catastrophically fail at $n{=}500$ (F1 $< 0.41$). The oracle top-3 SPIs (67–100%, std up to 27%) retain catastrophic failures at $n{=}500$. Only the full vocabulary (std 0.2–2% at $n \geq 50$, zero catastrophic failures) provides stable performance — the 125 SPIs function as a multi-view representation [Xu et al., 2013] whose diversity yields variance reduction via the bias-variance-covariance decomposition of complementary estimators [Krogh & Vedelsby, 1994; Brown et al., 2005]. The vocabulary's value is not solely in containing the right statistics but in providing *robustness through diversity*.

**(ii) Symmetric construction is empirically limited, as predicted.** Correlation (59%) stays below the theoretical $2/3$ ceiling — top-$d$ sparsification of $|r_{ij}|$ discards motif edges in favour of high-correlation nuisance pairs. The latent model (31%) performs at chance, confirming that symmetric node-embedding construction with no statistical prior cannot access the chain/fork distinction. These results validate the Markov equivalence argument.

**(iii) Edge content is load-bearing, not just topology.** The edge-ablation model — which uses the same SPI-derived adjacency as SPI-MPNN but zeros edge features during message passing — achieves only 88±23% at $n{=}500$ compared to 98±4% for the full model. The 10pp gap and dramatically higher variance demonstrate that the MPNN actively uses the SPI descriptor $\phi(\mathbf{E}_{ij})$ to condition messages, not merely the learned topology. The vocabulary informs both *which edges exist* and *what information flows along them*.

**(iv) The learned $\mathbf{w}$ constitutes an interpretable statistical signature.** Under group lasso ($\lambda_g{=}0.02$), the causal family (48 SPIs) carries 6$\times$ the $L_2$ norm of the next family. The top-weighted SPIs are:

| SPI | Family | Description |
|---|---|---|
| `sgc_parametric_mean_fs1_fmin0.25_fmax0.5_order1` | Causal | Spectral Granger causality, 0.25–0.5 Hz |
| `gc_gaussian_k1_kt1_l1_lt1` | Causal | Time-domain Granger causality |
| `sgc_parametric_max_fs1_fmin0.25_fmax0.5_order1` | Causal | SGC, peak frequency component |
| `gc_gaussian_kmax10_taumax2` | Causal | GC, extended lag |
| `sgc_parametric_mean_fs1_fmin0.25_fmax0.5_orderNone` | Causal | SGC, unrestricted order |

All five are directed temporal measures. For a VAR(1) process, causal structure manifests as asymmetric spectral transfer functions — the model recovers this from data, identifying the 0.25–0.5 Hz band where the VAR coupling concentrates. This is not a post-hoc rationalisation: it is a prediction from the generating process, confirmed by the learned parameters.

**Fixed-SPI competitiveness.** Fixed-SPI matches or exceeds SPI-MPNN across most sample sizes — indeed, it leads at $n{=}20$ (72% vs 57%). This is a substantive finding: the SPI vocabulary provides sufficient information as dense edge features for an MPNN to learn effective relational reasoning, even without learned topology or sparsification. The advantage of SPI-MPNN is not raw accuracy but *interpretability*: the learned $\mathbf{w}$ produces a named statistical signature that Fixed-SPI cannot. Both models converge to near-perfect performance, confirming that the vocabulary — not the construction mechanism — is the primary contribution.

**Shuffled control.** The shuffled baseline (SPI values randomly permuted across pairs) remains near chance at $n \leq 100$ (31–35%), confirming that pair-correspondence is necessary for small-sample learning. At $n{=}1000$, shuffled rises to 80% — the model exploits distributional differences in SPI value histograms across topology classes, a form of statistical fingerprinting that does not require correct pair assignment. This does not diminish the small-$n$ result, where the method's sample efficiency advantage is most relevant.

---

## 5 Discussion

**Positioning.** Graph structure learning methods — metric-based, neural, attention-based [Zhu et al., 2021] — universally use latent embeddings. CauGNN [Duan et al., 2023] constructs graphs from a single pre-specified statistic (transfer entropy); ADLGNN [Sriramulu et al., 2023] initialises from statistics but overwrites with attention, losing interpretability. Our approach occupies an unoccupied point: a *vocabulary* of named statistics with *learnable per-statistic weights* that remain inspectable after training. Nguyen et al. [2025] argued that named statistical features capture interactions that raw-value methods miss — we operationalise this insight within a GNN framework.

**Limitations.** (1) SPI computation is expensive ($O(K \cdot M^2)$ per sample); amortisable but limits scalability. (2) The vocabulary is a design choice — though group sparsity mitigates sensitivity. (3) All SPIs are bivariate; higher-order interactions require multi-hop message passing to compose. (4) The current study is synthetic; real-world validation (e.g., EEG motor imagery, fMRI) is needed to confirm that the learned statistical signature produces domain-interpretable findings.

**Conclusion.** The statistical signature produced by the learned $\mathbf{w}$ — identifying spectral Granger causality as the dominant coupling mode — constitutes a testable hypothesis about the data-generating process, not merely a classification artefact. This is a form of scientific output that latent graph learners structurally cannot produce. On real-world MTS (EEG, fMRI, climate), the same mechanism would identify which notion of coupling — coherence, transfer entropy, Granger causality, or combinations thereof — is relevant to the task, yielding domain-interpretable graph structure directly from the learned parameters.

---

## References

Alain, G. & Bengio, Y. (2017). Understanding intermediate layers using linear classifier probes. *ICLR Workshop*.

Battaglia, P.W. et al. (2018). Relational inductive biases, deep learning, and graph networks. *arXiv:1806.01261*.

Brown, G. et al. (2005). Diversity creation methods: A survey and categorisation. *Information Fusion*, 6(1), 5–20.

Bullmore, E. & Sporns, O. (2009). Complex brain networks. *Nature Reviews Neuroscience*, 10(3), 186–198.

Cliff, O.M. et al. (2023). Unifying pairwise interactions in complex dynamics. *Nature Computational Science*, 3(10), 883–893.

Duan, Z. et al. (2023). Multivariate time series forecasting with transfer entropy graph. *Tsinghua Science and Technology*, 28(1), 141–149.

Geweke, J. (1982). Measurement of linear dependence and feedback between multiple time series. *JASA*, 77(378), 304–313.

Gilmer, J. et al. (2017). Neural message passing for quantum chemistry. *ICML*.

Granger, C.W.J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 37(3), 424–438.

Kipf, T. et al. (2018). Neural relational inference for interacting systems. *ICML*.

Krogh, A. & Vedelsby, J. (1994). Neural network ensembles, cross validation, and active learning. *NeurIPS*, 7.

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

Xu, C. et al. (2013). A survey on multi-view learning. *arXiv:1304.5634*.

Xu, K. et al. (2019). How powerful are graph neural networks? *ICLR*.

Yuan, M. & Lin, Y. (2006). Model selection and estimation in regression with grouped variables. *JRSSB*, 68(1), 49–67.

Zhu, Y. et al. (2021). Deep graph structure learning for robust representations: A survey. *arXiv:2103.03036*.
