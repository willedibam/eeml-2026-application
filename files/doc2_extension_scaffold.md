# Context Document: Extension — Learned Representations on Interaction Graphs via Edge-Conditioned Message Passing

## Purpose of This Document
This document describes the proposed extension of the SPI-SPI baseline (Document 1) using geometric deep learning. The core idea: rather than hand-crafting second-order statistics over pairwise interaction matrices, we learn a representation of the MTS by treating the collection of SPIs as a $K$-dimensional edge-feature vector on a single graph and applying an edge-conditioned message passing neural network (MPNN) with global pooling. This document provides: (1) motivation and positioning, (2) formal problem setup, (3) architecture specification, (4) training and evaluation protocol, (5) experimental plan, and (6) practical implementation notes. It is written to be self-contained when paired with Document 1.

---

## 1. Motivation

### 1.1 Why Learn the Representation?
The SPI-SPI baseline (Document 1, §2) reduces each $M \times M$ MPI to a single scalar via correlation, discarding all graph topology. This is a severe information bottleneck: two systems with identical statistical coupling *character* but different network architectures (star vs. ring, clustered vs. uniform) are indistinguishable.

The natural fix is to use a Graph Neural Network (GNN) that preserves topological information through message passing, while producing a fixed-dimensional output via global pooling (handling variable $M$).

### 1.2 Why Not $K$ Independent GNNs? (Extension 1 — Rejected)
A naïve approach is to run a GNN independently on each of the $K$ MPIs, then aggregate the $K$ embeddings. This was considered and rejected because it processes each SPI in isolation: the inter-SPI relationships — which are the entire basis of the SPI-SPI method's interpretive power — are only recovered at a late aggregation stage with no mechanism for pairwise SPI interaction during the learned representation. This is a structural mismatch with the problem.

### 1.3 The Key Reframing
For a given MTS, the $K$ MPIs are not $K$ independent graphs. They are $K$ **views** of the same $M$-node system, where each view captures a different statistical property of pairwise channel interactions. This is a **multiplex network**: same nodes, $K$ edge types.

The cleanest formulation avoids multiplex-specific architectures entirely. For each node pair $(i, j)$, we have $K$ SPI values:

$$\mathbf{e}_{ij} = [\mathrm{SPI}_1(i,j),\; \mathrm{SPI}_2(i,j),\; \dots,\; \mathrm{SPI}_K(i,j)] \in \mathbb{R}^K.$$

This is a $K$-dimensional **edge feature vector**. The MTS becomes a single graph on $M$ nodes with $K$-dimensional edge features. Standard edge-conditioned MPNNs handle this natively.

### 1.4 What This Architecture Recovers
During message passing, the network jointly sees all $K$ interaction types for every edge simultaneously. It can learn, within a single forward pass, that specific combinations of SPI values on an edge indicate specific coupling regimes — exactly the kind of cross-SPI reasoning that the SPI-SPI correlation approximates as a scalar, but now:

- It is **learned** (task-adaptive, not hand-crafted).
- It is **local** (per-edge reasoning, not whole-matrix correlation).
- It is **topology-preserving** (message passing propagates structural context).

After global pooling over the $M$ nodes, the resulting embedding captures both the *character* of dependency (via edge feature interactions) and the *structural arrangement* of that dependency (via the GNN's sensitivity to graph topology).

---

## 2. Formal Problem Setup

### 2.1 Input Representation
Given an MTS $\mathbf{X} \in \mathbb{R}^{M \times T}$ and a set of $K$ selected SPIs, the `pyspi` computation yields a tensor $\mathcal{M} \in \mathbb{R}^{K \times M \times M}$. We reinterpret this as a single weighted graph:

- **Nodes:** $\mathcal{V} = \{1, 2, \dots, M\}$, one per channel.
- **Edges:** For every ordered pair $(i, j)$ with $i \neq j$, an edge with feature vector $\mathbf{e}_{ij} \in \mathbb{R}^K$ as defined above. If certain SPIs are symmetric, we may restrict to unordered pairs, but in general SPIs can be asymmetric (e.g., Granger causality), so directed edges should be used.
- **Node features:** Initial node features $\mathbf{h}_i^{(0)}$ must be defined. Options discussed in §3.2.
- **Graph:** $G = (\mathcal{V}, \mathcal{E}, \{\mathbf{e}_{ij}\}, \{\mathbf{h}_i^{(0)}\})$.

### 2.2 Task Formulation
We consider **graph-level classification/regression**: given a dataset of MTS samples $\{(\mathbf{X}_n, y_n)\}_{n=1}^N$ with (potentially different) spatial dimensions $M_n$ and temporal lengths $T_n$, learn a mapping $\mathbf{X}_n \mapsto \hat{y}_n$ via the intermediate graph representation.

The pipeline is:

$$\mathbf{X}^{M \times T} \xrightarrow{\text{pyspi}} \mathcal{M}^{K \times M \times M} \xrightarrow{\text{graph construction}} G^{(M \text{ nodes}, K\text{-dim edges})} \xrightarrow{\text{MPNN + pooling}} \mathbf{z} \in \mathbb{R}^D \xrightarrow{\text{classifier}} \hat{y}.$$

**Variable $M$ is handled by the global pooling step** (which maps any $M$-node graph to a fixed-dimensional vector). **Variable $T$ is handled by the pyspi step** (each SPI maps a pair of time series to a scalar regardless of length).

---

## 3. Architecture

### 3.1 Edge-Conditioned Message Passing
We use a message passing neural network where edge features modulate the message. At layer $\ell$:

$$\mathbf{m}_{j \to i}^{(\ell)} = \phi_{\mathrm{msg}}^{(\ell)}\!\left(\mathbf{h}_i^{(\ell)},\; \mathbf{h}_j^{(\ell)},\; \mathbf{e}_{ij}\right)$$

$$\mathbf{h}_i^{(\ell+1)} = \phi_{\mathrm{upd}}^{(\ell)}\!\left(\mathbf{h}_i^{(\ell)},\; \bigoplus_{j \in \mathcal{N}(i)} \mathbf{m}_{j \to i}^{(\ell)}\right)$$

where $\oplus$ is a permutation-invariant aggregation (sum or mean), $\phi_{\mathrm{msg}}$ is the message function, and $\phi_{\mathrm{upd}}$ is the update function.

**Concrete choice: NNConv (Gilmer et al., 2017).** The message function is:

$$\mathbf{m}_{j \to i} = f_\theta(\mathbf{e}_{ij}) \cdot \mathbf{h}_j^{(\ell)}$$

where $f_\theta: \mathbb{R}^K \to \mathbb{R}^{d_{\mathrm{hidden}} \times d_{\mathrm{hidden}}}$ is a small MLP that maps the $K$-dimensional edge feature to a weight matrix. This is the standard `NNConv` layer in PyTorch Geometric.

**Alternative: GINEConv (Hu et al., 2020).** Extends the expressive GIN (Graph Isomorphism Network) to edge features:

$$\mathbf{h}_i^{(\ell+1)} = \mathrm{MLP}\!\left((1 + \epsilon^{(\ell)}) \cdot \mathbf{h}_i^{(\ell)} + \sum_{j \in \mathcal{N}(i)} \mathrm{ReLU}\!\left(\mathbf{h}_j^{(\ell)} + \phi_{\mathrm{edge}}(\mathbf{e}_{ij})\right)\right).$$

GINEConv is simpler and cheaper than NNConv (no weight matrix generation), and has stronger theoretical expressivity guarantees (as powerful as the 1-WL test with edge features). **Recommended as default** unless NNConv's additional flexibility proves necessary.

**Number of layers.** 2–3 layers is standard for small graphs. For $M = 5$–$100$, the graph diameter is small (typically 2 for fully connected interaction graphs), so 2 layers are likely sufficient. Deeper networks risk oversmoothing.

### 3.2 Node Features
The graphs derived from pyspi are fully connected (every pair of channels has an SPI value), so initial node features require care. Options:

1. **Degree-based:** Uninformative for fully connected graphs (all nodes have degree $M-1$). **Not recommended.**

2. **Aggregated edge statistics:** For node $i$, compute summary statistics of its incident edge features: $\mathbf{h}_i^{(0)} = [\mathrm{mean}_j(\mathbf{e}_{ij}),\; \mathrm{std}_j(\mathbf{e}_{ij})]$, yielding $\mathbf{h}_i^{(0)} \in \mathbb{R}^{2K}$. This gives each node a "profile" of how it interacts with the rest of the system. **Recommended default.**

3. **Learnable node embeddings:** Use a single shared learnable vector $\mathbf{h}^{(0)} \in \mathbb{R}^{d}$ for all nodes (since nodes have no intrinsic identity across samples with different $M$). This is simple but provides no initial differentiation between nodes — the GNN must learn all node-level information from message passing alone.

4. **Channel-level time series features:** Compute univariate time series features (e.g., via `catch22` or `tsfresh`) for each channel and use these as node features. This injects marginal channel information but adds computational cost and complexity. **Consider as an extension, not for the initial submission.**

### 3.3 Global Pooling (Readout)
After $L$ layers of message passing, each node has a representation $\mathbf{h}_i^{(L)} \in \mathbb{R}^{d_{\mathrm{hidden}}}$. We need a graph-level vector $\mathbf{z} \in \mathbb{R}^D$:

- **Mean pooling:** $\mathbf{z} = \frac{1}{M} \sum_{i=1}^M \mathbf{h}_i^{(L)}$. Simple, handles variable $M$.
- **Sum pooling:** $\mathbf{z} = \sum_{i=1}^M \mathbf{h}_i^{(L)}$. Distinguishes graphs of different sizes (may be desirable or undesirable depending on whether $M$ itself is informative).
- **Attention-based pooling** (e.g., `GlobalAttention` in PyG): Learns which nodes are most important. Slightly more expressive, marginal additional cost.
- **Set2Set** (Vinyals et al., 2016): LSTM-based pooling. More expressive but heavier. Likely unnecessary for this problem scale.

**Recommendation:** Start with mean pooling. If performance is insufficient, try `GlobalAttention`.

### 3.4 Classification Head
A 2-layer MLP on the graph embedding $\mathbf{z}$:

$$\hat{y} = \mathrm{softmax}\!\left(W_2 \cdot \mathrm{ReLU}(W_1 \cdot \mathbf{z} + b_1) + b_2\right).$$

With dropout between layers (rate 0.3–0.5).

### 3.5 Full Architecture Summary

```
Input: MTS X ∈ R^{M × T}
  ↓ pyspi (K SPIs)
Graph: M nodes, K-dim edge features, node features from edge statistics
  ↓ GINEConv layer 1 (d_hidden = 64 or 128)
  ↓ ReLU + BatchNorm
  ↓ GINEConv layer 2
  ↓ ReLU + BatchNorm
  ↓ Global mean pooling → z ∈ R^{d_hidden}
  ↓ MLP classifier (2-layer, dropout 0.3)
Output: class prediction ŷ
```

---

## 4. Relationship to SPI-SPI Baseline

The SPI-SPI correlation method and the MPNN operate on the same input ($K$ MPIs from pyspi) but extract information differently:

| Aspect | SPI-SPI (baseline) | MPNN (proposed) |
|--------|---------------------|-----------------|
| Representation | Hand-crafted $\binom{K}{2}$-dim vector | Learned $D$-dim embedding |
| Cross-SPI reasoning | Global (whole-matrix correlation) | Local (per-edge, per layer) |
| Topology awareness | None (off-diagonals flattened) | Full (message passing on graph) |
| Invariance | $M$ and $T$ | $M$ and $T$ |
| Interpretability | Direct (each $f_{kk'}$ is a named correlation) | Requires post-hoc analysis |
| Trainability | No learning | End-to-end |

**In the submission, SPI-SPI is positioned as the interpretable, closed-form baseline. The MPNN is the learned generalisation.**

The key experimental question: does the MPNN outperform SPI-SPI, and if so, is the improvement attributable to topology preservation, learned cross-SPI interactions, or both?

---

## 5. Experimental Plan

### 5.1 Synthetic Experiments

**Experiment 1: Dependency character classification (adapted from Document 1, §3.1).**
- Generate MTS samples with controlled proportions of linear, nonlinear monotonic, and non-monotonic inter-channel coupling (using the sin-filter construction).
- Vary $M$ across samples (e.g., $M \in \{8, 16, 32, 64\}$) and $T$ (e.g., $T \in \{200, 500, 1000\}$).
- Task: classify each MTS by its dominant coupling regime.
- Compare: SPI-SPI + standard classifier (SVM, random forest) vs. MPNN.
- **Expected result:** Both methods should perform well when coupling character is the discriminative signal. The MPNN should show less degradation with noisy or heterogeneous data.

**Experiment 2: Topology-dependent classification (the critical test).**
- Generate MTS with **identical coupling character** but **different network topologies**. For example: (a) hub-spoke: one node drives all others; (b) chain: coupling propagates sequentially; (c) clustered: two groups with strong intra-group coupling, weak inter-group. All use the same type of coupling (e.g., linear Gaussian).
- Vary $M$ across samples.
- Task: classify each MTS by its topology class.
- **Expected result:** SPI-SPI should fail (it discards topology). The MPNN should succeed (it preserves graph structure through message passing). This is the experiment that justifies the extension.

**Experiment 3: Warping detection (adapted from Document 1, §3.2).**
- Generate MTS with controlled temporal warping.
- Task: classify or regress on the degree of warping.
- Compare SPI-SPI vs. MPNN.
- **Expected result:** Comparable performance (warping is a global property captured by both approaches), but useful to demonstrate the MPNN does not sacrifice performance on tasks where the baseline already works.

### 5.2 Real-World Experiment (if feasible)
A single real-world demonstration with naturally varying $M$ would significantly strengthen the submission. Candidate domains:

- **Neuroimaging (EEG/fMRI):** Different electrode montages or parcellation schemes give different $M$. Classification task: disease state, cognitive task, or arousal level.
- **Ecological sensor networks:** Environmental monitoring stations vary in count across regions. Task: classify environmental regime.
- **Financial data:** Market indices/sectors with different numbers of constituent instruments. Task: identify market regime.

**For a 2-page extended abstract, one clean real-world result is sufficient.** If no suitable dataset is readily available, the submission can proceed with synthetic-only, but should explicitly frame real-world application as immediate future work.

### 5.3 Ablation Studies (space permitting)
- **Edge features only vs. edge + node features:** Does the MPNN benefit from node feature initialisation (§3.2)?
- **Number of SPIs ($K$):** Performance as a function of $K$. Start with $K = 20$–50 curated SPIs spanning distinct statistical families. Does performance improve with more?
- **Pooling strategy:** Mean vs. sum vs. attention.

### 5.4 Interpretability Analysis
After training, inspect which edge feature dimensions (SPIs) are most influential:

- **Edge feature attribution:** Gradient-based or attention-based attribution on $\mathbf{e}_{ij}$ to identify which SPIs the network relies on for classification.
- **Comparison with SPI-SPI feature importance:** Does the MPNN attend to the same SPI pairs that are discriminative in the SPI-SPI baseline? Where does it diverge?

This analysis connects the learned representation back to the statistical reasoning that motivates the work, and is likely to be the most engaging part of the paper for reviewers.

---

## 6. Practical Implementation Notes

### 6.1 PyTorch Geometric Setup
The implementation uses PyTorch Geometric (PyG). Key components:

```python
# Core imports
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool
import torch
import torch.nn as nn

# Graph construction from pyspi output
def mts_to_pyg_graph(mpi_tensor, label):
    """
    Args:
        mpi_tensor: np.ndarray of shape (K, M, M) — K MPIs for one MTS
        label: int — class label
    Returns:
        torch_geometric.data.Data
    """
    K, M, _ = mpi_tensor.shape

    # Build edge index (fully connected, directed)
    src, dst = [], []
    for i in range(M):
        for j in range(M):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Build K-dimensional edge features
    edge_attr = []
    for i in range(M):
        for j in range(M):
            if i != j:
                edge_attr.append(mpi_tensor[:, i, j])
    edge_attr = torch.tensor(np.stack(edge_attr), dtype=torch.float)  # (num_edges, K)

    # Node features: mean and std of incident edge features
    x = []
    for i in range(M):
        incident = mpi_tensor[:, i, :]  # (K, M)
        mask = np.ones(M, dtype=bool)
        mask[i] = False
        incident = incident[:, mask]  # (K, M-1)
        node_feat = np.concatenate([incident.mean(axis=1), incident.std(axis=1)])
        x.append(node_feat)
    x = torch.tensor(np.stack(x), dtype=torch.float)  # (M, 2K)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([label]))
```

### 6.2 Model Definition (Sketch)

```python
class InteractionGraphNet(nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim, num_classes):
        super().__init__()
        # Edge feature MLP for GINEConv
        nn1 = nn.Sequential(nn.Linear(num_node_features, hidden_dim), nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim))
        nn2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim))
        # Edge feature projection to match node feature dim
        self.edge_proj1 = nn.Linear(num_edge_features, num_node_features)
        self.edge_proj2 = nn.Linear(num_edge_features, hidden_dim)

        self.conv1 = GINEConv(nn1)
        self.conv2 = GINEConv(nn2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        # GINEConv requires edge features of same dim as node features
        x = torch.relu(self.bn1(self.conv1(x, edge_index, self.edge_proj1(edge_attr))))
        x = torch.relu(self.bn2(self.conv2(x, edge_index, self.edge_proj2(edge_attr))))
        x = global_mean_pool(x, batch)
        return self.classifier(x)
```

**Note:** This is a starting sketch. GINEConv in PyG requires edge features to have the same dimensionality as node features (they are added element-wise). The `edge_proj` layers handle this. Verify the current PyG API at implementation time; parameter names and conventions may have changed.

### 6.3 Training Protocol
- **Loss:** Cross-entropy for classification; MSE for regression.
- **Optimiser:** Adam, lr = 1e-3 to 1e-4, with cosine annealing or ReduceLROnPlateau.
- **Batch size:** 32–128 (graphs are small; batching is cheap).
- **Evaluation:** Stratified k-fold cross-validation ($k = 5$ or 10). Report mean ± std accuracy.
- **Baselines to compare against:**
    1. SPI-SPI correlation vector + SVM (linear and RBF)
    2. SPI-SPI correlation vector + Random Forest
    3. Graph kernel on MPIs (Weisfeiler-Leman subtree kernel, from `grakel` library) + SVM — this is an important baseline
    4. The MPNN (proposed)

### 6.4 Compute Requirements
The graphs are tiny ($M = 5$–$100$ nodes, fully connected). The bottleneck is `pyspi` computation (computing 250+ SPIs on MTS data), not the GNN training. A single consumer GPU is more than sufficient for training. CPU-only training is feasible for small-scale experiments. PBS/NCI Gadi resources are relevant only if running large parameter sweeps or generating very large synthetic datasets where pyspi computation is the bottleneck.

### 6.5 SPI Selection
Using all 250+ SPIs from pyspi as $K$-dimensional edge features is feasible but introduces redundancy and noise. Recommended approach:

1. **Start with a curated subset of $K = 20$–50 SPIs** spanning distinct statistical families:
   - Correlation: Pearson $r$, Spearman $\rho$, Kendall $\tau$
   - Distance: Euclidean distance, DTW, cosine distance, distance correlation
   - Information-theoretic: mutual information, time-lagged mutual information, joint entropy, transfer entropy
   - Spectral: coherence magnitude, phase slope index
   - Causal/directed: Granger causality (parametric and nonparametric)
   - Nonlinear: kernel-based measures, Hilbert-Schmidt independence criterion

2. **Scale to full SPI set** as an ablation. The GNN's edge MLP should in principle learn to ignore uninformative SPIs, but verifying this empirically is valuable.

---

## 7. Submission Structure (2-Page Extended Abstract for EEML)

Suggested structure for the 2-page limit:

**Title:** "Learned Representations on Statistical Interaction Graphs for Comparing Multivariate Time Series of Arbitrary Dimension"

**Section 1 — Introduction and Problem (≈0.4 pages).**
- The problem: comparing MTS with different $M$ and $T$.
- Why existing feature-based methods fail for variable $M$.
- One-sentence summary of approach: convert MTS to interaction graphs with multi-dimensional edge features, learn representations via edge-conditioned MPNN.

**Section 2 — Method (≈0.6 pages).**
- pyspi → $K$ MPIs → single graph with $K$-dim edge features.
- MPNN architecture (brief: GINEConv, 2 layers, global mean pool).
- SPI-SPI baseline as the closed-form, interpretable counterpart.

**Section 3 — Experiments (≈0.6 pages).**
- Synthetic: dependency character classification (varying $M$, $T$) and topology classification (the critical discriminating experiment).
- (Optional) One real-world dataset.
- Baselines: SPI-SPI + SVM/RF, graph kernel + SVM.

**Section 4 — Results and Discussion (≈0.3 pages).**
- Key results. Focus on: (a) MPNN matches SPI-SPI when topology is irrelevant, (b) MPNN outperforms when topology is relevant, (c) interpretability analysis showing which SPIs the network attends to.

**Figure budget (critical for 2 pages):**
- **Figure 1:** Method overview diagram. Left: MTS → pyspi → $K$ MPIs (shown as adjacency matrices or small network diagrams). Right: reinterpretation as single graph with $K$-dim edge features → MPNN → embedding → classifier. This figure does significant expository work and should be carefully designed.
- **Figure 2:** Key experimental results. Possibly a 2-panel figure: (a) classification accuracy vs. number of channels $M$ for all methods on the dependency character task; (b) same for the topology task, showing SPI-SPI failure and MPNN success.

---

## 8. Risks and Honest Assessment

1. **Novelty concern.** Using an MPNN with multi-dimensional edge features for graph classification is not itself novel — it is standard in molecular property prediction and similar domains. The novelty is in the *input representation* (pyspi-derived interaction graphs) and the *application domain* (comparing MTS of arbitrary dimension). A reviewer may view this as "standard GNN on a new input," which is a valid criticism. The interpretability analysis (§5.4) and the principled construction of the edge features from a diverse statistical library are the main counters.

2. **Experimental risk.** The topology experiment (§5.1, Experiment 2) is the most important and the most uncertain. If the fully connected graph structure (all pairs have edges) makes it difficult for the GNN to learn topology — because the graph is always a complete graph and topology differences manifest only in edge weight patterns — then the advantage over SPI-SPI may be smaller than expected. This needs to be tested early.

3. **pyspi computation cost.** Computing 250+ SPIs on many MTS samples is slow. For the submission timeline, precompute on a manageable dataset size and cache results. This is an engineering concern, not a methodological one, but it affects what is achievable before the deadline.

4. **2-page limit.** The mathematical setup is involved. Compression to 2 pages requires ruthless prioritisation. The case studies from Document 1 almost certainly cannot appear in full — at most, one is briefly referenced as motivation.

---

## 9. Suggested Figures for the Submission

### Figure 1: Method Overview (full-width, top of page 1 or spanning columns)
A schematic showing:
- Left: An MTS $\mathbf{X} \in \mathbb{R}^{M \times T}$ (depicted as $M$ time series traces).
- Middle-left: Arrow labelled "pyspi" → $K$ adjacency matrices (small heatmaps or network diagrams), labelled as MPIs.
- Middle: Arrow → single graph with nodes coloured uniformly, edges coloured or thickened to represent the $K$-dimensional feature vector. Annotation: "$\mathbf{e}_{ij} \in \mathbb{R}^K$".
- Middle-right: Arrow labelled "MPNN" → schematic of message passing (one round, showing messages flowing along edges).
- Right: Arrow labelled "global pool" → single vector $\mathbf{z}$ → classifier → $\hat{y}$.

Below or alongside: show the SPI-SPI baseline path as a simpler alternative (MPIs → flatten → correlate → $\mathbf{f} \in \mathbb{R}^{\binom{K}{2}}$).

This figure is the most important visual in the submission. It should be clean, schematic (not a screenshot), and use consistent colour coding.

### Figure 2: Key Results (2-panel)
- Panel (a): Accuracy on dependency-character classification as a function of $M$ for MPNN, SPI-SPI+SVM, SPI-SPI+RF, graph kernel+SVM.
- Panel (b): Same for topology classification, showing the expected failure of SPI-SPI methods and the success of the MPNN.

Error bars from cross-validation folds.

---

## 10. Questions Requiring User Input

1. **EEML deadline.** When is it? This determines whether the full experimental plan is feasible or needs to be trimmed.

2. **Existing pyspi computations.** Have you already run pyspi on any datasets (synthetic or real)? If so, what $K$, $M$, $T$?

3. **Real-world data access.** Do you have any MTS datasets with naturally varying $M$? Even one would help.

4. **SPI selection.** Do you have a preferred subset of SPIs, or should we curate one? The choice of $K$ SPIs is a design decision that affects both computational cost and the richness of the edge features.

5. **PyG version.** Which Python and PyTorch versions are available on your cluster? PyG installation can be sensitive to CUDA/PyTorch version compatibility.
