# Literature References — EEML 2026 Extended Abstract

**Compiled 2026-03-27. For use when writing the extended abstract and research document.**

---

## Tier 1: Must-Cite (directly load-bearing for our arguments)

### Core framework references
- **Gilmer et al. (2017).** Neural message passing for quantum chemistry. *ICML*. — MPNN framework.
- **Battaglia et al. (2018).** Relational inductive biases, deep learning, and graph networks. *arXiv:1806.01261*. — Relational inductive bias taxonomy. Our work operationalises their principle for graph construction.
- **Cliff et al. (2023).** Unifying pairwise interactions in complex dynamics. *Nature Computational Science*, 3(10), 883–893. DOI: 10.1038/s43588-023-00519-x — pyspi toolkit. Foundational data source.

### GNN expressiveness (argument: symmetric construction is provably limited)
- **Xu et al. (2019).** How powerful are graph neural networks? *ICLR*. — 1-WL ceiling for symmetric message-passing. Our Markov equivalence argument is analogous.
- **Morris et al. (2019).** Weisfeiler and Leman go neural: Higher-order graph neural networks. *AAAI*, 4602–4609. — Independent 1-WL proof + k-WL extensions.

### Markov equivalence (argument: symmetric statistics cannot distinguish chain/fork)
- **Verma & Pearl (1990).** Equivalence and synthesis of causal models. *UAI*, 220–227. — Canonical proof that Markov-equivalent DAGs share skeleton + v-structures.
- **Spirtes, Glymour & Scheines (2000).** *Causation, Prediction, and Search* (2nd ed.). MIT Press. — Foundational causal discovery textbook.

### Structured inductive biases improve sample efficiency
- **Velickovic et al. (2022).** Reasoning-modulated representations. *LoG*, PMLR 198:50:1–50:17. — Injecting algorithmic priors improves sample efficiency. We apply this principle to graph construction.
- **Velickovic & Blundell (2021).** Neural algorithmic reasoning. *Patterns*, 2(7), 100273. — Alignment with algorithmic structure → better sample complexity.

### Choice of statistic matters
- **Liu et al. (2025).** Benchmarking methods for mapping functional connectivity in the brain. *Nature Methods*. DOI: 10.1038/s41592-025-02704-4. — **KEY.** Used pyspi (239 measures) on fMRI. Found substantial variation across FC methods. Covariance/precision outperformed Pearson on several benchmarks. Same pyspi ecosystem. Supports our core premise.
- **Mohanty et al. (2020).** Rethinking measures of functional connectivity via feature extraction. *Scientific Reports*, 10(1), 1298. — Pearson correlation "cannot comprehensively capture BOLD inter-dependencies"; 8 alternative FC measures improved classification.
- **Qi et al. (2025).** Rethinking functional brain connectome analysis: do graph deep learning models help? *npj AI*. — **KEY.** Message-passing "consistently degrades" on FC graphs when construction is naive. Graph construction (not architecture) is the critical variable. Strongest existing empirical support for our core premise.

### Group sparsity
- **Yuan & Lin (2006).** Model selection and estimation in regression with grouped variables. *JRSSB*, 68(1), 49–67. — Group lasso.

### Latent graph learning baselines
- **Wu et al. (2019).** Graph WaveNet for deep spatial-temporal graph modeling. *IJCAI*. — Adaptive adjacency via E₁E₂ᵀ.
- **Wu et al. (2020).** Connecting the dots: Multivariate time series forecasting with graph neural networks. *KDD*. — MTGNN. Same paradigm, no pre-defined graph needed.
- **Kipf et al. (2018).** Neural relational inference for interacting systems. *ICML*. — VAE-based latent edge types. Edges not grounded in statistical vocabulary.

---

## Tier 2: Strongly Recommended

### Graph structure learning surveys
- **Zhu et al. (2021/2022).** A survey on graph structure learning: Progress and opportunities. *IJCAI 2022*. arXiv:2103.03036. — GSL taxonomy: metric-based, neural, direct. All use latent embeddings.
- **Jin et al. (2024).** A survey on graph neural networks for time series. *IEEE TPAMI*, 46(12), 10466–10485. — GNN4TS survey. Notes correlation/DTW heuristics exist but field overwhelmingly favors end-to-end learned adjacency.

### Closest existing work (must differentiate)
- **Sriramulu et al. (2023).** Adaptive dependency learning graph neural networks. *Information Sciences*, 625, 700–714. — **ADLGNN.** Statistical initialization + attention-based reweighting. Closest competitor. Key difference: loses interpretability of which statistic matters; our w remains inspectable.
- **Nguyen et al. (2025).** A feature-based information-theoretic approach for detecting interpretable, long-timescale pairwise interactions. *Physical Review Research*, 7, 043283. — **From USyd (Fulcher/Lizier group).** Uses catch22 features + MI for pairwise interaction detection. Strongest conceptual predecessor. Not a GNN paper.
- **Duan et al. (2023).** Multivariate time series forecasting with transfer entropy graph. *Tsinghua Science and Technology*, 28(1), 141–149. — **CauGNN/TEGNN.** Single statistic (TE) for directed graph construction. Our approach generalises to K statistics with learned weighting.

### Over-squashing and graph rewiring (complementary line of work)
- **Di Giovanni et al. (2023).** On over-squashing in message passing neural networks. *ICML*. — Topology (commute time) is the dominant factor in over-squashing. Justifies principled graph construction.
- **Topping et al. (2022).** Understanding over-squashing and bottlenecks on graphs via curvature. *ICLR* (Outstanding Paper Honorable Mention). — Curvature-based rewiring. Complementary to our work (construction vs rewiring).

### Directed GNNs
- **Tong et al. (2020).** Directed graph convolutional network. arXiv:2004.13970. — Converting directed→undirected "misleads message passing." Supports our directed construction.
- **Zhang et al. (2021).** MagNet: A neural network for directed graphs. *NeurIPS*. — Magnetic Laplacian for directed GNNs.
- **Cao et al. (2024).** Dementia classification using a GNN on imaging of effective brain connectivity. *Computers in Biology and Medicine*, 168, 107701. — DSL-GNN: directed edges from effective connectivity improved classification (94.0% AD vs HC).

### Causal discovery / DAG learning
- **Zheng et al. (2018).** DAGs with NO TEARS: Continuous optimization for structure learning. *NeurIPS* (Spotlight). — Continuous DAG learning. Related approach to differentiable graph construction.
- **Yu et al. (2019).** DAG-GNN: DAG structure learning with graph neural networks. *ICML*, PMLR 97:7154–7163. — GNN-based causal structure learning but latent/uninterpretable.

### Group sparsity in deep learning
- **Lemhadri et al. (2021).** LassoNet: A neural network with feature sparsity. *JMLR*, 22(127), 1–29. — Group sparsity for neural net feature selection. Precedent for our approach.

### Statistical foundations
- **Granger (1969).** Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 37(3), 424–438. — Original Granger causality.
- **Geweke (1982).** Measurement of linear dependence and feedback between multiple time series. *JASA*, 77(378), 304–313. — Spectral decomposition of GC. Directly relevant since our model selects SGC.
- **Schreiber (2000).** Measuring information transfer. *Physical Review Letters*, 85(2), 461–464. — Transfer entropy.

---

## Tier 3: Optional / Supporting

### Additional GSL methods
- **Bai et al. (2020).** Adaptive graph convolutional recurrent network for traffic forecasting. *NeurIPS*. — AGCRN. E*Eᵀ paradigm.
- **Franceschi et al. (2019).** Learning discrete structures for graph neural networks. *ICML*. — LDS. Bilevel optimization.
- **Shang et al. (2021).** Discrete graph structure learning for forecasting multiple time series. *ICLR*. — GTS. Gumbel-Softmax.
- **Fatemi et al. (2021).** SLAPS: Self-supervision improves structure learning for GNNs. *NeurIPS*. — Self-supervised GSL.
- **Graber & Schwing (2020).** Dynamic neural relational inference. *CVPR*. — dNRI. Dynamic extension of NRI.

### Additional expressiveness results
- **Maron et al. (2019).** Provably powerful graph networks. *NeurIPS*. — 3-WL power via higher-order tensors.
- **Zhang et al. (2023).** A complete expressiveness hierarchy for subgraph GNNs. *ICML*. — "Graph generation policy" as an ingredient in expressiveness.

### Neuroscience-specific
- **Bullmore & Sporns (2009).** Complex brain networks. *Nature Reviews Neuroscience*, 10(3), 186–198. — Foundational FC review.
- **Fornito, Zalesky & Bullmore (2016).** *Fundamentals of Brain Network Analysis*. Academic Press. — FC methods textbook.
- **Smith et al. (2011).** Network modelling methods for FMRI. *NeuroImage*, 54(2), 875–891. — Partial correlation >> Pearson for simulated fMRI.
- **Pervaiz et al. (2020).** Optimising network modelling methods for fMRI. *NeuroImage*, 211, 116604. — >9,000 pipeline variants, choice significantly affects prediction.
- **Bielczyk et al. (2019).** Disentangling causal webs in the brain using fMRI. *Network Neuroscience*, 3(2), 237–273. — No single causal method dominates.
- **Li et al. (2022).** BrainGB: A benchmark for brain network analysis with GNNs. *IEEE TMI*, 42(5), 1446–1456. — 375 architecture variants but only correlation-based FC.
- **Barnett & Seth (2014).** The MVGC multivariate Granger causality toolbox. *J Neurosci Methods*, 223, 50–68.

### Other directed/causal GNN work
- **Zheng et al. (2024).** CI-GNN: A Granger causality-inspired GNN for interpretable brain network-based psychiatric diagnosis. *Neural Networks*, 172, 106147. — GC-inspired architecture but not pre-computed edge statistics.
- **MTE-MTGNN (2025).** Beyond spatial neighbors: Utilizing multivariate transfer entropy for interpretable graph-based forecasting. *Engineering Applications of AI*. — Single-measure (MTE) directed construction.

### Benchmarks
- **GSLB (2023).** The graph structure learning benchmark. *NeurIPS Datasets and Benchmarks*. arXiv:2310.05174. — 16 GSL algorithms benchmarked.

---

## Novelty Gap Summary

**Confirmed through exhaustive search: no existing paper combines a vocabulary of named statistical pairwise interaction measures with learnable per-SPI weights for GNN graph construction.**

Closest works and how we differ:
1. **ADLGNN** — statistical init + attention reweighting → loses interpretability
2. **Nguyen et al. 2025** — named features for interaction detection → not a GNN paper
3. **CauGNN/TEGNN** — single statistic (TE) → not a vocabulary, not learnable weighting
4. **Liu et al. 2025** — pyspi for FC benchmarking → not GNN graph construction
5. **CI-GNN** — GC-inspired architecture → not pre-computed edge statistics from pyspi
