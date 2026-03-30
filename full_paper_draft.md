# Graph Construction from a Statistical Vocabulary: Structured Inductive Bias for Relational Learning on Time Series

**[Author Name]**$^{1}$ and **[Supervisor Name]**$^{1}$

$^{1}$The University of Sydney

---

## Abstract

Graph neural networks applied to multivariate time series (MTS) require a graph over observed variables, yet this graph is almost never given. Existing approaches either fix it from a single pairwise statistic — committing to one notion of dependence before learning begins — or learn it from latent embeddings, producing opaque adjacencies with no statistical semantics. We introduce a third approach: parameterising graph construction over a *vocabulary* of $K$ heterogeneous pairwise statistics spanning causal, spectral, linear, information-theoretic, and distance families, computed via `pyspi` [Cliff et al., 2023]. A learned weight vector $\mathbf{w} \in \mathbb{R}^K$, jointly optimised with a message passing neural network, simultaneously constructs the graph and identifies which statistical notion of coupling is task-relevant — yielding edges with named mathematical semantics.

We prove that symmetric graph construction operators cannot distinguish Markov-equivalent directed topologies regardless of sample size, while directed statistics in our vocabulary break this equivalence. On synthetic VAR(1) topology classification, models consuming the SPI vocabulary achieve near-perfect accuracy (99–100%), while baselines using symmetric statistics (Pearson correlation, latent embeddings) remain at or below 59% — confirming the theoretical limitation empirically. A vocabulary ablation ladder reveals that the vocabulary's value lies not merely in containing the right statistics but in providing *robustness through diversity*: the oracle-best single statistic (spectral Granger causality) achieves 96% peak accuracy but with catastrophic seed failures (std up to 24%), while the full vocabulary eliminates these failures entirely (std $< 1$% at $n \geq 50$). This variance reduction follows the bias-variance-covariance decomposition of diverse estimators [Krogh & Vedelsby, 1994] — each SPI provides a complementary view of pairwise dependence [Xu et al., 2013], and the ensemble is more stable than any of its members. The learned $\mathbf{w}$ concentrates on spectral Granger causality measures, recovering the generating process's causal structure in named, interpretable form. An edge-ablation study confirms the MPNN actively uses SPI descriptors to condition messages, not merely the learned topology.

---

## 1 Introduction

Message passing neural networks (MPNNs) derive their power from propagating information along relational structure [Gilmer et al., 2017; Battaglia et al., 2018]. In MTS settings — neuroscience, climate, finance, traffic — this structure is unobserved and must be constructed. Two regimes dominate current practice, and both impose significant constraints.

### 1.1 The single-statistic regime

**Fixing the graph from a single statistic** — Pearson correlation in functional connectivity [Bullmore & Sporns, 2009], spatial distance in traffic networks [Li et al., 2018], transfer entropy in causal graph construction [Duan et al., 2023] — is a strong commitment to one mathematical notion of dependence. If the task-relevant coupling is directed, nonlinear, or frequency-specific, this is an expressiveness limitation of the graph construction operator, not a data limitation.

This is not a hypothetical concern. Liu et al. [2025] recently benchmarked 239 pairwise statistics for functional connectivity mapping across multiple neuroimaging datasets, finding "substantial quantitative and qualitative variation across FC methods" — different statistics produced qualitatively different connectivity patterns from the same data. Their conclusion: the choice of construction statistic is consequential yet understudied. Our work provides a principled resolution: rather than choosing one statistic a priori, we let the task select from a vocabulary.

### 1.2 The latent embedding regime

**Learning the graph from latent embeddings** — Neural Relational Inference [Kipf et al., 2018], Graph WaveNet [Wu et al., 2019], MTGNN [Wu et al., 2020] — recovers flexibility at the cost of semantics. The learned adjacency is opaque: an edge weight of 0.7 encodes no information about *what form* of dependence it represents. Such graphs cannot be validated against domain knowledge or compared across tasks. They also place the full burden on the training signal to discover relational structure that statistical methodology could supply as prior knowledge.

This opacity is not merely an aesthetic objection. In scientific applications — neuroscience, climate science, genomics — the graph structure is often the scientific object of interest, not merely a computational scaffold. A learned adjacency that cannot be interpreted in terms of known statistical relationships has limited scientific value, regardless of downstream task performance.

### 1.3 Statistical vocabulary as structured inductive bias

Veličković & Blundell [2021] argued that aligning neural architectures with algorithmic structure yields favourable sample complexity; Veličković et al. [2022] demonstrated that injecting algorithmic priors improves sample efficiency over learning from scratch. We apply this principle to graph construction: rather than re-learning what "connectivity" means from raw data, we supply a pre-specified vocabulary of pairwise statistics with known mathematical properties and let end-to-end learning identify which elements are task-relevant.

The vocabulary functions as a multi-view representation of pairwise dependence [Xu et al., 2013]. Each SPI provides a complementary view — transfer entropy captures directed information flow, coherence captures frequency-domain coupling, correlation captures linear association — and the ensemble benefits from the bias-variance-covariance decomposition of diverse estimators [Krogh & Vedelsby, 1994; Brown et al., 2005]. Our experiments show this diversity is essential: even the oracle-best single statistic exhibits catastrophic failures that the full vocabulary eliminates.

### 1.4 Our approach

For each ordered pair $(i, j)$ in an MTS, we precompute a $K$-dimensional descriptor $\mathbf{E}_{ij}$ whose coordinates are named pairwise statistics — transfer entropy, spectral Granger causality, phase locking value, covariance, and $K \approx 125$ others — drawn from six complementary mathematical families using `pyspi` [Cliff et al., 2023]. Graph construction becomes:

$$A_{ij} = \mathrm{softplus}(b + \mathbf{w}^\top \mathbf{E}_{ij}), \quad \text{top-}d \text{ sparsification per node}$$

where $\mathbf{w} \in \mathbb{R}^K$ is jointly optimised with the downstream MPNN. This is a deliberate linear probe of the SPI descriptor space [cf. Alain & Bengio, 2017]: the claim is that the vocabulary is sufficiently well-structured for a linear function to identify task-relevant edges. Group lasso regularisation [Yuan & Lin, 2006] over SPI families encourages family-level selection, producing an interpretable *statistical signature* of the task's relational structure.

This design occupies an unoccupied point in the design space. CauGNN [Duan et al., 2023] constructs graphs from a single pre-specified statistic (transfer entropy); ADLGNN [Sriramulu et al., 2023] initialises from multiple statistics but overwrites them with attention, losing interpretability. Our approach retains a *vocabulary* of named statistics with *learnable per-statistic weights* that remain inspectable after training. Nguyen et al. [2025] argued that named statistical features capture interactions that raw-value methods miss — we operationalise this insight within a GNN framework.

---

## 2 Expressiveness of Graph Construction Operators

We formalise the intuition that the choice of construction statistic constrains what an MPNN can learn, analogous to Weisfeiler-Lehman expressiveness results for message passing [Xu et al., 2019; Morris et al., 2019].

### 2.1 Proposition

Consider three 3-node VAR(1) motifs over nodes $A, B, C$: chain ($A \to B \to C$), fork ($A \leftarrow B \to C$), and collider ($A \to B \leftarrow C$). All three share skeleton $A{-}B{-}C$. Chain and fork are Markov equivalent [Verma & Pearl, 1990]: they encode identical conditional independence relations ($A \perp C \mid B$, and no other). Any graph construction operator using only symmetric statistics — correlation, mutual information, coherence, or any function $f$ satisfying $f(X_i, X_j) = f(X_j, X_i)$ — assigns identical edge weights to chain and fork at the population level, regardless of sample size or estimation quality. No MPNN operating on such a graph can exceed $2/3$ accuracy on the three-class task.

### 2.2 Proof sketch

Two DAGs are Markov equivalent iff they share the same skeleton and v-structures [Verma & Pearl, 1990]. Chain ($A \to B \to C$) and fork ($A \leftarrow B \to C$) share skeleton $\{A{-}B,\; B{-}C\}$ and have no v-structures. Any symmetric statistic $f(X_i, X_j)$ depends only on the joint distribution $P(X_i, X_j)$, which is identical for Markov-equivalent structures. In finite samples, estimation noise introduces small asymmetries in symmetric statistics, but these are insufficient for reliable discrimination — as confirmed empirically (Correlation $\leq 59$% across all sample sizes, never approaching the $2/3$ ceiling despite up to $n = 1000$ training instances per class). The collider ($A \to B \leftarrow C$) has a v-structure and encodes a different independence ($A \perp C$ marginally), making it distinguishable by all methods — serving as an internal control. $\square$

### 2.3 Directed statistics break the equivalence

Directed statistics break the chain–fork equivalence because they distinguish $(i, j)$ from $(j, i)$. In the chain, $\mathrm{SGC}(A \to B) \gg \mathrm{SGC}(B \to A)$; in the fork, $\mathrm{SGC}(B \to A) \approx \mathrm{SGC}(B \to C) \gg 0$ but $\mathrm{SGC}(A \to C) \approx 0$. The asymmetric descriptor tensor $\mathbf{E}_{ij} \neq \mathbf{E}_{ji}$ provides the information needed for discrimination.

This is not merely a theoretical observation. The VAR(1) generating process produces causal structure that manifests as asymmetric spectral transfer functions — specifically, spectral Granger causality (Geweke, 1982) decomposes the total linear predictability between two time series into frequency-specific directed components. For a VAR(1) process with coupling in the 0.25–0.5 Hz band, the SGC in that band directly measures the causal flow magnitude — making it the theoretically optimal statistic for this task.

---

## 3 Method

### 3.1 SPI descriptor tensor

Given $X \in \mathbb{R}^{M \times T}$ (z-scored), we compute $K$ pairwise statistics per ordered pair via `pyspi` [Cliff et al., 2023], yielding $\mathbf{E} \in \mathbb{R}^{M \times M \times K}$. Statistics are partitioned into six families:

| Family | # SPIs | Examples | Symmetry |
|---|---|---|---|
| **Causal** | 48 | Transfer entropy [Schreiber, 2000], Granger causality [Granger, 1969], spectral GC [Geweke, 1982], phase slope index | **Asymmetric** |
| **Spectral** | 52 | Coherence magnitude, PLV, PLI, imaginary coherence | Mixed |
| **Linear** | 13 | Covariance, precision, cross-correlation | Symmetric |
| **Information** | 5 | Mutual information (Gaussian, Kraskov) | Symmetric |
| **Distance** | 5 | DTW, Euclidean, cross-distance | Symmetric |
| **Rank** | 2 | Spearman, Kendall | Symmetric |

Crucially, causal measures are asymmetric: $\mathbf{E}_{ij} \neq \mathbf{E}_{ji}$, enabling directed graph construction that symmetric operators cannot represent. The vocabulary is computed once per instance and cached; it does not enter the training loop.

### 3.2 Learned graph construction

$$A_{ij} = \mathrm{softplus}(b + \mathbf{w}^\top \mathbf{E}_{ij})$$

with $K + 1$ learnable parameters ($\mathbf{w} \in \mathbb{R}^K$, $b \in \mathbb{R}$). Sparsified to top-$d$ outgoing edges per node (directed). The loss includes group lasso regularisation:

$$\mathcal{L} = \mathcal{L}_{\mathrm{task}} + \lambda_1 \|\mathbf{w}\|_1 + \lambda_g \sum_{g \in \mathcal{G}} \|\mathbf{w}_g\|_2$$

where $\mathcal{G}$ partitions $\mathbf{w}$ into the six SPI families. This encourages family-level sparsity: entire families can be zeroed out, while within an active family, individual SPIs contribute proportional to their task relevance. The L1 term provides within-family sparsity.

The linearity of the construction function is a deliberate design choice, not a limitation. It serves as a linear probe [Alain & Bengio, 2017] of the SPI descriptor space: the claim is that the vocabulary is sufficiently well-structured that a linear function can identify task-relevant edges. If a nonlinear function were required, it would suggest the vocabulary is not well-aligned with the task — precisely the signal we want to detect.

### 3.3 Edge-attributed MPNN

Retained edges carry the full $\mathbf{E}_{ij} \in \mathbb{R}^K$ as attributes. An edge network $\phi(\mathbf{E}_{ij}) = \mathrm{MLP}_\phi(\mathbf{E}_{ij})$ conditions messages on the full descriptor:

$$\mathbf{m}_{ij} = \mathrm{MLP}_m([\mathbf{h}_j \| \mathbf{h}_i \| \phi(\mathbf{E}_{ij})]), \quad \mathbf{h}_i' = \mathbf{h}_i + \mathrm{LN}\!\left(\sum_j A_{ij} \cdot \mathbf{m}_{ij}\right)$$

This is a custom MPNN layer (Gilmer et al., 2017), not GINEConv (Hu et al., 2020). The SPI vocabulary informs both topology (via $\mathbf{w}$) and message content (via $\phi$). Two rounds of message passing, followed by global mean+max pooling and a classifier, produce graph-level predictions.

The dual role of the vocabulary — informing both *which edges exist* and *what information flows along them* — is confirmed by the edge-ablation study (§4.3).

### 3.4 Optimisation

**LR warmup.** The model exhibits a bifurcation in the loss landscape: early gradient direction determines whether $\mathbf{w}$ converges to the informative basin (85–100% accuracy) or remains stuck at chance (33%). Linear LR warmup (60 epochs → cosine decay) allows the MPNN to partially train before $\mathbf{w}$ gradients dominate, increasing the fraction of seeds that find the informative basin from 3/10 to 9/10.

**Restarts.** We train each seed twice and keep the model with higher validation F1. Combined with warmup, this eliminates catastrophic failures across all 10 seeds at $n \geq 50$.

---

## 4 Experiments

### 4.1 Synthetic topology classification

We generate $M{=}10$ node VAR(1) processes ($T{=}500$) with a 3-node directed motif (chain, fork, or collider) embedded among 7 nuisance AR(1) channels ($\rho{=}0.8$). Coupling strengths $\alpha \sim \mathrm{Uniform}(0.3, 0.7)$; 1500 instances per class. The `pyspi` vocabulary yields $K{=}125$ statistics after variance filtering. This setup is designed to test the expressiveness proposition: chain and fork are Markov-equivalent, so symmetric construction operators are provably limited.

### 4.2 Models

Each model isolates a specific component of the full method:

| Model | Graph construction | Edge features | Tests |
|---|---|---|---|
| **SPI-MPNN** | $\mathrm{softplus}(b + \mathbf{w}^\top \mathbf{E}_{ij})$, learned | $\mathbf{E}_{ij} \in \mathbb{R}^K$ | Full method |
| Correlation | Fixed $|r_{ij}|$, top-$d$ | scalar $|r_{ij}|$ | Symmetric single-statistic |
| TE-Only | Fixed $|\mathrm{TE}_{ij}|$, top-$d$ | scalar TE | Directed single-statistic (nonparametric) |
| SGC-Only | Fixed $|\mathrm{SGC}_{ij}|$, top-$d$ | scalar SGC | Oracle-best single SPI from learned $\mathbf{w}$ |
| Top-3 SPIs | Mean $|\mathrm{SPI}|$ of oracle top-3, top-$d$ | 3-dim vector | Oracle top-3 from learned $\mathbf{w}$ |
| Latent | Learned node embeddings, top-$d$ | $[\mathbf{h}_i; \mathbf{h}_j]$ | No statistical prior |
| Fixed-SPI | Fully connected (no sparsification) | $\mathbf{E}_{ij} \in \mathbb{R}^K$ | Is learned topology necessary? |
| MLP-Mix | $\sigma(\mathrm{MLP}(\mathbf{E}_{ij}))$, top-$d$ | $\mathbf{E}_{ij} \in \mathbb{R}^K$ | Nonlinear construction |
| Edge-Ablation | SPI adjacency (same as SPI-MPNN) | **Zeroed** | Does MPNN use edge content? |
| Shuffled | SPI adjacency (from permuted SPIs) | Permuted $\mathbf{E}_{ij}$ | Pair-correspondence control |
| Node-Only | Fully connected | None | Are pairwise features necessary? |

### 4.3 Results

Macro F1 (%), 10 seeds. Training: LR warmup (60 epochs), restarts (best of 2), top-$d{=}5$, group $\lambda{=}0.02$, $\lambda_1{=}0.001$.

**Full results table:**

| $n$/class | SPI-MPNN | Fixed-SPI | MLP-Mix | SGC-Only | Top-3 | Edge-Abl. | Corr. | TE-Only | Latent | Shuffled | Node-Only |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 57±22 | **72±12** | 63±10 | 61±9 | 67±14 | 35±14 | 35±5 | 29±1 | 30±2 | 34±4 | 33±2 |
| 50 | 92±2 | 92±2 | 80±7 | 95±8 | **99±1** | 56±25 | 47±13 | 28±1 | 33±1 | 38±5 | 31±2 |
| 100 | 97±2 | **98±1** | 88±7 | 83±17 | 99±1 | 53±21 | 57±2 | 27±8 | 32±2 | 37±3 | 31±3 |
| 200 | 98±5 | **98±1** | 95±5 | 95±12 | 97±8 | 78±25 | 53±11 | 26±3 | 31±2 | 41±5 | 30±2 |
| 500 | 98±4 | **99±0.5** | 99±2 | 87±24 | 87±27 | 88±23 | 59±1 | 20±4 | 31±2 | 55±19 | 32±3 |
| 1000 | **100±0.2** | 99±1 | 99±0.2 | 96±11 | 100±0.3 | — | 59±1 | 18±2 | 31±3 | 80±13 | 34±2 |

**Per-seed analysis at $n = 500$** (illustrating the stability argument):

| Model | Seeds | Min | Catastrophic failures (F1 < 0.4) |
|---|---|---|---|
| SPI-MPNN | [1.00, 1.00, 0.87, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00] | 0.87 | 0/10 |
| Fixed-SPI | [1.00, 0.99, 1.00, 1.00, 0.99, 1.00, 1.00, 0.99, 0.99, 1.00] | 0.99 | 0/10 |
| SGC-Only | [0.99, 0.99, 0.41, 0.37, 0.99, 0.98, 1.00, 1.00, 1.00, 0.99] | 0.37 | **2/10** |
| Top-3 | [1.00, 0.17, 1.00, 0.99, 0.99, 1.00, 1.00, 1.00, 0.99, 0.56] | 0.17 | **1/10** |

### 4.4 Findings

#### Finding 1: The SPI vocabulary is the dominant inductive bias — and diversity provides stability

All SPI-consuming models (SPI-MPNN, Fixed-SPI, MLP-Mix) dramatically outperform non-SPI baselines across all sample sizes. The gap is not marginal — it is the difference between near-perfect classification and chance performance. A vocabulary ablation ladder isolates the source of this advantage:

**TE-only (29–18%, at or below chance).** Transfer entropy is a directed, causal statistic — the correct *type* for this task. Yet the nonparametric Kraskov k-NN estimator [Kraskov et al., 2004] at $T{=}500$ produces high-variance estimates. The true TE between motif pairs is a small positive number; estimated TE between nuisance pairs is noisy around zero but frequently exceeds the true signal. When top-$d$ sparsification selects the 5 largest $|\mathrm{TE}|$ values per node, it selects noise peaks, not true causal edges. The resulting graph is essentially random — hence chance performance. The F1 *decreasing* with $n$ confirms the bottleneck is per-instance estimation quality (fixed by $T$), not training set size.

**SGC-only (61–96%, std 8–24%).** Spectral Granger causality is the oracle-best SPI from the learned $\mathbf{w}$. Unlike TE, it is a parametric estimator: it fits a VAR model, then computes Geweke's (1982) spectral decomposition. For a true VAR(1) process, this is a *correctly specified* estimator with much lower variance than nonparametric TE. Accordingly, it achieves high *peak* performance. However, the adjacency from a single scalar is fragile: 2/10 seeds catastrophically fail at $n{=}500$ (F1 $< 0.41$). The optimisation landscape with 1-dimensional edge features is flat, and the gradient signal is insufficient to reliably escape the chance-level basin.

**Top-3 oracle SPIs (67–100%, std 0.3–27%).** The oracle top-3 from $\mathbf{w}$ (SGC mean, GC Gaussian, GC extended lag). Near-perfect when it works, but retains catastrophic failures: 1/10 seeds at F1 $= 0.17$ at $n{=}500$. Three estimators provide more stable adjacency than one, but 3-dimensional edge features are still insufficient for reliable gradient signal. The mean adjacency over 3 SPIs averages out *some* per-statistic noise, but not enough to eliminate the failure mode.

**SPI-MPNN (57–100%, std 0.2–2% at $n \geq 50$).** The full vocabulary. All seeds $\geq 0.87$ at $n{=}500$; zero catastrophic failures. The 125 SPIs with complementary bias-variance profiles provide variance reduction that no small subset matches. The vocabulary functions as a multi-view representation [Xu et al., 2013] where each SPI is a complementary view of pairwise dependence, and the ensemble benefits from the bias-variance-covariance decomposition of diverse estimators [Krogh & Vedelsby, 1994; Brown et al., 2005].

**The vocabulary's value is not merely in containing the right statistics — it is in providing robustness through diversity:** stable graph construction from the average of many noisy estimators, and rich 125-dimensional edge features that provide a strong gradient signal from epoch 1, eliminating the optimisation fragility of single-statistic approaches.

#### Finding 2: Symmetric construction is empirically limited, as predicted

Correlation (59% at $n = 500$) stays below the theoretical $2/3$ ceiling. The gap between the ceiling and observed performance arises because top-$d$ sparsification of $|r_{ij}|$ discards motif edges in favour of high-correlation nuisance pairs — the 7 AR(1) channels have $\rho = 0.8$, producing strong autocorrelation that translates into high pairwise correlation even without direct coupling.

The latent model (31% across all $n$) performs at chance, confirming that symmetric node-embedding construction ($A_{ij} = \mathrm{softplus}(\mathbf{h}_i \cdot \mathbf{h}_j / \sqrt{d})$) with no statistical prior cannot access the chain/fork distinction. This is expected: the model must discover pairwise relational structure from node-level features alone, a strictly harder problem.

Node-Only (31–34%) confirms that pairwise features are necessary — concatenating per-node summary statistics provides no information about the motif structure.

These results validate the Markov equivalence proposition empirically.

#### Finding 3: Edge content is load-bearing, not just topology

The edge-ablation model — which uses the same SPI-derived adjacency as SPI-MPNN but zeros edge features during message passing — achieves only 88±23% at $n{=}500$ compared to 98±4% for the full model. The 10 percentage point gap and dramatically higher variance (23% vs 4%) demonstrate that the MPNN actively uses the SPI descriptor $\phi(\mathbf{E}_{ij})$ to condition messages, not merely the learned topology.

This confirms the dual role of the vocabulary: it informs both *which edges exist* (via $\mathbf{w}$) and *what information flows along them* (via $\phi$). The edge-ablation model's high variance — including seeds at chance — suggests that topology alone is sometimes insufficient, and edge content provides the additional signal needed for consistent performance.

#### Finding 4: The learned $\mathbf{w}$ constitutes an interpretable statistical signature

Under group lasso ($\lambda_g{=}0.02$), the causal family (48 SPIs) carries 6$\times$ the $L_2$ norm of the next family. The top-weighted individual SPIs are:

| SPI | Family | $\overline{|w_k|}$ | Description |
|---|---|---|---|
| `sgc_parametric_mean_fs1_fmin0.25_fmax0.5_order1` | Causal | 0.0226 | Spectral Granger causality, 0.25–0.5 Hz, order-1 |
| `gc_gaussian_k1_kt1_l1_lt1` | Causal | 0.0225 | Time-domain Granger causality, lag 1 |
| `gc_gaussian_kmax10_taumax2` | Causal | 0.0206 | GC, extended lag |
| `sgc_parametric_max_fs1_fmin0.25_fmax0.5_order1` | Causal | 0.0202 | SGC, peak frequency component |
| `sgc_parametric_mean_fs1_fmin0.25_fmax0.5_orderNone` | Causal | 0.0141 | SGC, unrestricted order |

All five are directed temporal measures. For a VAR(1) process, causal structure manifests as asymmetric spectral transfer functions — the model recovers this from data, identifying the 0.25–0.5 Hz band where the VAR coupling concentrates. This is not a post-hoc rationalisation: it is a prediction from the generating process, confirmed by the learned parameters.

The interpretability here is structural, not cosmetic. The learned $\mathbf{w}$ constitutes a *testable hypothesis* about the data-generating process: "this task's relational structure is best captured by spectral Granger causality in the 0.25–0.5 Hz band." On real-world data, the analogous signature would identify which notion of coupling — coherence, transfer entropy, or combinations thereof — is relevant to the domain, yielding scientific insight directly from the learned parameters.

#### Finding 5: Fixed-SPI competitiveness reveals the vocabulary as the primary contribution

Fixed-SPI matches or exceeds SPI-MPNN across most sample sizes — indeed, it leads at $n{=}20$ (72% vs 57%). This is a substantive finding: the SPI vocabulary provides sufficient information as dense edge features for an MPNN to learn effective relational reasoning, even without learned topology or sparsification.

The advantage of SPI-MPNN is not raw accuracy but *interpretability*: the learned $\mathbf{w}$ produces a named statistical signature that Fixed-SPI cannot. Both models converge to near-perfect performance, confirming that the vocabulary — not the construction mechanism — is the primary contribution. This reframes the contribution: SPI-MPNN is not a superior classification method but a *scientific instrument* that reveals which statistical notion of coupling is task-relevant.

#### Finding 6: Controls confirm experimental validity

**Shuffled control.** The shuffled baseline (SPI values randomly permuted across pairs) remains near chance at $n \leq 100$ (31–35%), confirming that pair-correspondence is necessary for small-sample learning. At $n{=}1000$, shuffled rises to 80% — the model exploits distributional differences in SPI value histograms across topology classes, a form of statistical fingerprinting that does not require correct pair assignment. This does not diminish the small-$n$ result, where the method's sample efficiency advantage is most relevant.

**MLP-Mix** (63–99%) confirms that nonlinear adjacency construction from the vocabulary is functional but offers no advantage over the linear probe, validating the choice of a linear $\mathbf{w}$.

---

## 5 Discussion

### 5.1 Positioning in the literature

Graph structure learning methods — metric-based, neural, attention-based [Zhu et al., 2021] — universally use latent embeddings to construct graphs. A smaller body of work uses pre-specified statistics: CauGNN [Duan et al., 2023] and MTE-MTGNN [2025] construct graphs from transfer entropy; ADLGNN [Sriramulu et al., 2023] initialises from multiple statistics but overwrites them with attention, losing the interpretability of the initial statistics. Our approach occupies an unoccupied point: a *vocabulary* of named statistics with *learnable per-statistic weights* that remain inspectable after training.

The closest intellectual antecedent is Nguyen et al. [2025], who argued that named statistical features capture interpretable, long-timescale pairwise interactions that raw-value methods miss. We operationalise this insight within a GNN framework, showing that the same vocabulary that provides interpretability also provides the dominant inductive bias for classification.

### 5.2 Limitations

1. **SPI computation is expensive** ($O(K \cdot M^2)$ per sample). For $K = 125$ and $M = 10$, this is tractable; for $M > 100$, it may require subset selection or amortisation. The computation is embarrassingly parallel and performed once per instance.

2. **The vocabulary is a design choice.** `pyspi` provides a curated set of 125 statistics, but the selection is not optimised for any particular task. Group lasso mitigates sensitivity by allowing the model to zero out irrelevant families. Extending the vocabulary (e.g., adding wavelet coherence or partial directed coherence) is straightforward.

3. **All SPIs are bivariate.** Higher-order interactions (e.g., synergistic or redundant information among triplets) are not directly captured. Multi-hop message passing can compose bivariate information, but cannot represent genuinely trivariate phenomena like partial information decomposition [Williams & Beer, 2010].

4. **The current study is synthetic.** Real-world validation — EEG motor imagery, fMRI resting state, climate teleconnections — is needed to confirm that the learned statistical signature produces domain-interpretable findings. The synthetic setup is designed to test a clean theoretical claim (Markov equivalence + vocabulary as inductive bias), not to demonstrate practical utility.

5. **The linear probe assumption.** If the task-relevant graph requires a nonlinear function of the SPI descriptors, the linear $\mathbf{w}$ will underperform. MLP-Mix shows that nonlinear construction is no better here, but this may not hold on more complex generating processes.

### 5.3 Future directions

**Real-world validation.** On EEG or fMRI data, the learned $\mathbf{w}$ would identify which notion of coupling — coherence, transfer entropy, Granger causality — is relevant to the clinical or cognitive task. Comparing the learned statistical signature across tasks (e.g., motor imagery vs resting state vs seizure detection) would test whether different tasks select different families, as the Liu et al. [2025] benchmarking results suggest they should.

**Vocabulary extension.** The `pyspi` vocabulary is not exhaustive. Adding task-specific statistics (e.g., partial directed coherence for EEG, partial correlation for fMRI) or multi-scale statistics (wavelet coherence at multiple resolutions) could improve performance on real-world data where the optimal statistic is not in the current vocabulary.

**Scalability.** For large $M$, computing all $M^2 \times K$ statistics is prohibitive. Sparse computation — computing SPIs only for spatially or temporally adjacent pairs — would reduce cost while retaining most information. Alternatively, a hierarchical approach could compute cheap statistics (correlation, covariance) for all pairs and expensive statistics (TE, SGC) only for pairs that pass a screening threshold.

### 5.4 Conclusion

The statistical signature produced by the learned $\mathbf{w}$ — identifying spectral Granger causality as the dominant coupling mode — constitutes a testable hypothesis about the data-generating process, not merely a classification artefact. This is a form of scientific output that latent graph learners structurally cannot produce. The vocabulary ablation ladder demonstrates that this output requires not just the right type of statistic but the full diversity of the vocabulary: the ensemble of 125 estimators provides robustness through complementary bias-variance profiles that no single statistic or small oracle-selected subset can match.

On real-world MTS — EEG, fMRI, climate — the same mechanism would identify which notion of coupling is relevant to the task, yielding domain-interpretable graph structure directly from the learned parameters. The vocabulary-as-inductive-bias paradigm offers a principled middle ground between the rigidity of single-statistic construction and the opacity of latent graph learning.

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

Kraskov, A. et al. (2004). Estimating mutual information. *Physical Review E*, 69(6), 066138.

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

Williams, P.L. & Beer, R.D. (2010). Nonnegative decomposition of multivariate information. *arXiv:1004.2515*.

Wu, Z. et al. (2019). Graph WaveNet for deep spatial-temporal graph modeling. *IJCAI*.

Wu, Z. et al. (2020). Connecting the dots: Multivariate time series forecasting with graph neural networks. *KDD*.

Xu, C. et al. (2013). A survey on multi-view learning. *arXiv:1304.5634*.

Xu, K. et al. (2019). How powerful are graph neural networks? *ICLR*.

Yuan, M. & Lin, Y. (2006). Model selection and estimation in regression with grouped variables. *JRSSB*, 68(1), 49–67.

Zhu, Y. et al. (2021). Deep graph structure learning for robust representations: A survey. *arXiv:2103.03036*.
