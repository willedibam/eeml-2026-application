"""
Graph neural network models for SPI-based MTS classification.

Models
------
SPIEdgeMPNN      Proposed method: learned linear adjacency from K SPI descriptors.
CorrelationMPNN  Baseline: fixed graph from Pearson |r_ij|, scalar edge weight.
LatentGraphMPNN  Baseline: per-sample adjacency from learned node embeddings.
FixedSPIMPNN     Ablation: fully connected + K-dim SPI edge features, no learned adj.
MLPMixMPNN       Ablation: nonlinear adjacency from MLP(SPI descriptors).
NodeOnlyMLP      Ablation: no graph, MLP over concatenated node features.
EdgeAblationMPNN Ablation: SPI adjacency + zero edge features in message passing.
ShuffledEdgeMPNN Control: SPI adjacency + randomly permuted SPI edge vectors.
SingleSPIMPNN    Ablation: fixed graph from one named SPI (e.g. transfer entropy).
SubsetSPIMPNN    Ablation: fixed graph from mean of k named SPIs, k-dim edge features.

Message passing
---------------
All edge-attributed models use a custom MPNN-style layer (Gilmer et al. 2017):
    m_ij = MLP([h_j, h_i, phi_e(e_ij)])
    h_i' = h_i + LayerNorm(sum_j A_ij * m_ij)

This is NOT GINEConv (Hu et al. 2020), which has a different update rule.

PyG batching note
-----------------
spi_tensor is stored as (M, M, K) per graph. PyG concatenates along dim 0
to (B*M, M, K). _unbatch_spi recovers (B, M, M, K) via to_dense_batch.
pearson_corr follows the same pattern: (M, M) → (B*M, M) → (B, M, M).
Any new model using these tensors MUST use the corresponding unbatch helper.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import to_dense_batch


# ---------------------------------------------------------------------------
# Batching helpers
# ---------------------------------------------------------------------------

def _unbatch_spi(batch: Batch) -> torch.Tensor:
    """(B*M, M, K) → (B, M, M, K)."""
    dense, _ = to_dense_batch(batch.spi_tensor, batch.batch)
    return dense


def _unbatch_pearson(batch: Batch) -> torch.Tensor:
    """(B*M, M) → (B, M, M)."""
    dense, _ = to_dense_batch(batch.pearson_corr, batch.batch)
    return dense


def _sparsify(adj: torch.Tensor, top_d: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Retain top_d outgoing edges per node (directed, no self-loops).

    Args:
        adj: (M, M) non-negative weights. Diagonal is zeroed out.
        top_d: edges to retain per source node.

    Returns:
        edge_index: (2, E) long tensor.
        edge_weight: (E,) float tensor.
    """
    M = adj.shape[0]
    device = adj.device
    adj = adj * (~torch.eye(M, dtype=torch.bool, device=device)).float()
    d = min(top_d, M - 1)
    _, top_idx = adj.topk(d, dim=1)
    src = torch.arange(M, device=device).unsqueeze(1).expand(-1, d).reshape(-1)
    dst = top_idx.reshape(-1)
    return torch.stack([src, dst]), adj[src, dst]


# ---------------------------------------------------------------------------
# Message passing layer
# ---------------------------------------------------------------------------

class MPLayer(nn.Module):
    """
    MPNN-style message passing layer (Gilmer et al. 2017).

    m_ij = MLP([h_j, h_i, phi_e(e_ij)])
    h_i' = h_i + LayerNorm(sum_j A_ij * m_ij)
    """

    def __init__(self, hidden: int, edge_dim: int, dropout: float = 0.1):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * hidden + edge_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        msg = self.msg_mlp(torch.cat([h[dst], h[src], edge_attr], dim=1))
        msg = msg * edge_weight.unsqueeze(1)
        agg = h.new_zeros(h.shape[0], msg.shape[1])
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
        return h + self.drop(self.norm(agg))


# ---------------------------------------------------------------------------
# Shared SPI-graph backbone
# ---------------------------------------------------------------------------

class _SPIGraphBase(nn.Module):
    """
    Shared backbone for all SPI-based models.

    Subclasses implement compute_adjacency() and optionally override forward().
    """

    def __init__(
        self,
        n_spi: int,
        n_node_features: int,
        n_classes: int,
        hidden: int = 64,
        n_layers: int = 2,
        top_d: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.top_d = top_d
        self.n_spi = n_spi
        self.hidden = hidden

        self.node_proj = nn.Linear(n_node_features, hidden)
        self.edge_net = nn.Sequential(
            nn.Linear(n_spi, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.mp_layers = nn.ModuleList(
            [MPLayer(hidden, hidden, dropout) for _ in range(n_layers)]
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def compute_adjacency(self, spi_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _readout(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_weight: torch.Tensor,
        batch_vec: torch.Tensor,
    ) -> torch.Tensor:
        h = self.node_proj(node_feats)
        for layer in self.mp_layers:
            h = layer(h, edge_index, edge_attr, edge_weight)
        out = torch.cat(
            [global_mean_pool(h, batch_vec), global_max_pool(h, batch_vec)], dim=1
        )
        return self.classifier(out)

    def forward(self, batch: Batch) -> torch.Tensor:
        device = batch.x.device
        spi_dense = _unbatch_spi(batch)  # (B, M, M, K)

        all_ei, all_ea, all_ew, bvec = [], [], [], []
        offset = 0
        for i in range(batch.num_graphs):
            M = (batch.batch == i).sum().item()
            spi_i = spi_dense[i, :M, :M, :]
            adj = self.compute_adjacency(spi_i)
            ei, ew = _sparsify(adj, self.top_d)
            ea = self.edge_net(spi_i[ei[0], ei[1]])
            all_ei.append(ei + offset)
            all_ea.append(ea)
            all_ew.append(ew)
            bvec.append(torch.full((M,), i, dtype=torch.long, device=device))
            offset += M

        return self._readout(
            batch.x,
            torch.cat(all_ei, dim=1),
            torch.cat(all_ea),
            torch.cat(all_ew),
            torch.cat(bvec),
        )


# ---------------------------------------------------------------------------
# SPIEdgeMPNN — proposed model
# ---------------------------------------------------------------------------

class SPIEdgeMPNN(_SPIGraphBase):
    """
    Proposed: learned linear adjacency from K SPI descriptors.

    A_ij = softplus(b + w^T E_ij)

    spi_w (K,) is a global learnable weight vector. It is inspectable after
    training to determine which SPI families the model relied on.
    Group sparsity regularisation on spi_w is applied in train.py.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spi_w = nn.Parameter(torch.zeros(self.n_spi))
        self.spi_b = nn.Parameter(torch.tensor(-2.0))

    def compute_adjacency(self, spi_tensor: torch.Tensor) -> torch.Tensor:
        scores = torch.einsum("ijk,k->ij", spi_tensor, self.spi_w) + self.spi_b
        return F.softplus(scores)


# ---------------------------------------------------------------------------
# CorrelationMPNN — baseline
# ---------------------------------------------------------------------------

class CorrelationMPNN(nn.Module):
    """
    Baseline: fixed adjacency from absolute Pearson correlation.

    Pearson |r_ij| is computed from raw MTS in graph_build.py and stored as
    pearson_corr in the Data object. No learnable adjacency parameters.
    Edge attribute: scalar |r_ij| processed by edge network.
    """

    def __init__(
        self,
        n_node_features: int,
        n_classes: int,
        hidden: int = 64,
        n_layers: int = 2,
        top_d: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.top_d = top_d
        self.node_proj = nn.Linear(n_node_features, hidden)
        self.edge_net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.mp_layers = nn.ModuleList(
            [MPLayer(hidden, hidden, dropout) for _ in range(n_layers)]
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        device = batch.x.device
        corr_dense = _unbatch_pearson(batch)  # (B, M, M)

        all_ei, all_ea, all_ew, bvec = [], [], [], []
        offset = 0
        for i in range(batch.num_graphs):
            M = (batch.batch == i).sum().item()
            corr_i = corr_dense[i, :M, :M].abs()
            ei, ew = _sparsify(corr_i, self.top_d)
            ea = self.edge_net(ew.unsqueeze(1))
            all_ei.append(ei + offset)
            all_ea.append(ea)
            all_ew.append(ew)
            bvec.append(torch.full((M,), i, dtype=torch.long, device=device))
            offset += M

        edge_index = torch.cat(all_ei, dim=1)
        edge_attr = torch.cat(all_ea)
        edge_weight = torch.cat(all_ew)
        bvec = torch.cat(bvec)

        h = self.node_proj(batch.x)
        for layer in self.mp_layers:
            h = layer(h, edge_index, edge_attr, edge_weight)
        out = torch.cat(
            [global_mean_pool(h, bvec), global_max_pool(h, bvec)], dim=1
        )
        return self.classifier(out)


# ---------------------------------------------------------------------------
# LatentGraphMPNN — baseline
# ---------------------------------------------------------------------------

class LatentGraphMPNN(nn.Module):
    """
    Baseline: per-sample adjacency from learned node embeddings.

    A_ij = softplus(h_i . h_j / sqrt(embed_dim))
    where h_i = adj_encoder(x_i) are data-derived, per-sample embeddings.

    Edge attributes: [h_i; h_j] passed through edge_net.

    This is a fair comparison to SPIEdgeMPNN: both produce per-sample graphs
    and both have edge features. The difference is the feature source:
        - SPIEdgeMPNN: precomputed statistical descriptors (SPIs)
        - LatentGraphMPNN: learned embeddings from raw node statistics

    Parameter budget for adjacency:
        adj_encoder: n_node_features * embed_dim params
        SPIEdgeMPNN has K + 1 params (spi_w + spi_b)
        Set embed_dim so n_node_features * embed_dim ≈ K.
    """

    def __init__(
        self,
        n_node_features: int,
        n_classes: int,
        hidden: int = 64,
        n_layers: int = 2,
        top_d: int = 3,
        embed_dim: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.top_d = top_d
        self.embed_dim = embed_dim

        self.adj_encoder = nn.Linear(n_node_features, embed_dim)
        self.node_proj = nn.Linear(n_node_features, hidden)
        self.edge_net = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.mp_layers = nn.ModuleList(
            [MPLayer(hidden, hidden, dropout) for _ in range(n_layers)]
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        device = batch.x.device

        all_ei, all_ea, all_ew, bvec = [], [], [], []
        offset = 0
        for i in range(batch.num_graphs):
            node_mask = batch.batch == i
            x_i = batch.x[node_mask]
            M = x_i.shape[0]

            h_emb = self.adj_encoder(x_i)  # (M, embed_dim)
            scores = (h_emb @ h_emb.T) / (self.embed_dim ** 0.5)
            adj = F.softplus(scores)
            ei, ew = _sparsify(adj, self.top_d)
            ea = self.edge_net(torch.cat([h_emb[ei[0]], h_emb[ei[1]]], dim=1))

            all_ei.append(ei + offset)
            all_ea.append(ea)
            all_ew.append(ew)
            bvec.append(torch.full((M,), i, dtype=torch.long, device=device))
            offset += M

        edge_index = torch.cat(all_ei, dim=1)
        edge_attr = torch.cat(all_ea)
        edge_weight = torch.cat(all_ew)
        bvec = torch.cat(bvec)

        h = self.node_proj(batch.x)
        for layer in self.mp_layers:
            h = layer(h, edge_index, edge_attr, edge_weight)
        out = torch.cat(
            [global_mean_pool(h, bvec), global_max_pool(h, bvec)], dim=1
        )
        return self.classifier(out)


# ---------------------------------------------------------------------------
# FixedSPIMPNN — ablation
# ---------------------------------------------------------------------------

class FixedSPIMPNN(nn.Module):
    """
    Ablation: fully connected graph + K-dim SPI edge features, no learned adjacency.

    Tests whether SPIs are useful as edge features independent of learned topology.
    If FixedSPIMPNN ≈ SPIEdgeMPNN, learned topology adds little beyond rich features.
    If SPIEdgeMPNN >> FixedSPIMPNN, learned sparsification is essential.
    """

    def __init__(
        self,
        n_spi: int,
        n_node_features: int,
        n_classes: int,
        hidden: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_spi = n_spi
        self.node_proj = nn.Linear(n_node_features, hidden)
        self.edge_net = nn.Sequential(
            nn.Linear(n_spi, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.mp_layers = nn.ModuleList(
            [MPLayer(hidden, hidden, dropout) for _ in range(n_layers)]
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        device = batch.x.device
        spi_dense = _unbatch_spi(batch)

        all_ei, all_ea, all_ew, bvec = [], [], [], []
        offset = 0
        for i in range(batch.num_graphs):
            M = (batch.batch == i).sum().item()
            spi_i = spi_dense[i, :M, :M, :]

            # Fully connected directed graph (all M*(M-1) pairs)
            idx = torch.arange(M, device=device)
            src, dst = torch.meshgrid(idx, idx, indexing="ij")
            mask = src != dst
            src, dst = src[mask], dst[mask]
            ei = torch.stack([src, dst])
            ew = torch.ones(ei.shape[1], device=device)
            ea = self.edge_net(spi_i[ei[0], ei[1]])

            all_ei.append(ei + offset)
            all_ea.append(ea)
            all_ew.append(ew)
            bvec.append(torch.full((M,), i, dtype=torch.long, device=device))
            offset += M

        edge_index = torch.cat(all_ei, dim=1)
        edge_attr = torch.cat(all_ea)
        edge_weight = torch.cat(all_ew)
        bvec = torch.cat(bvec)

        h = self.node_proj(batch.x)
        for layer in self.mp_layers:
            h = layer(h, edge_index, edge_attr, edge_weight)
        out = torch.cat(
            [global_mean_pool(h, bvec), global_max_pool(h, bvec)], dim=1
        )
        return self.classifier(out)


# ---------------------------------------------------------------------------
# MLPMixMPNN — ablation (Tier 2)
# ---------------------------------------------------------------------------

class MLPMixMPNN(_SPIGraphBase):
    """
    Ablation: nonlinear adjacency from MLP(SPI) instead of linear w^T E_ij.

    Tests whether the SPI descriptor space requires nonlinear combination
    for graph construction. Does NOT have spi_w (not inspectable, no L1 reg).

    If MLPMixMPNN ≈ SPIEdgeMPNN: linear probing of SPI space suffices.
    If MLPMixMPNN >> SPIEdgeMPNN: nonlinear interactions between SPIs matter.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adj_mlp = nn.Sequential(
            nn.Linear(self.n_spi, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def compute_adjacency(self, spi_tensor: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.adj_mlp(spi_tensor).squeeze(-1))


# ---------------------------------------------------------------------------
# NodeOnlyMLP — ablation
# ---------------------------------------------------------------------------

class NodeOnlyMLP(nn.Module):
    """
    Ablation: no graph. MLP over concatenated node features.

    Critical sanity check: if this matches graph-based models, the task is
    solvable from univariate channel statistics alone and graph reasoning
    provides no benefit. Must be run before reporting other results.
    """

    def __init__(
        self,
        n_node_features: int,
        n_nodes: int,
        n_classes: int,
        hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(n_node_features * n_nodes, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        # to_dense_batch pads with zeros if graphs have different sizes.
        # For fixed-M datasets, no padding occurs.
        x_dense, _ = to_dense_batch(batch.x, batch.batch)  # (B, M, F_n)
        x_flat = x_dense.view(x_dense.shape[0], -1)        # (B, M*F_n)
        return self.classifier(x_flat)


# ---------------------------------------------------------------------------
# EdgeAblationMPNN — ablation
# ---------------------------------------------------------------------------

class EdgeAblationMPNN(SPIEdgeMPNN):
    """
    Ablation: SPI-derived adjacency with zero edge features in message passing.

    Adjacency and top-d sparsification are identical to SPIEdgeMPNN.
    The edge network output is replaced with zeros before message passing.

    Tests whether the MPNN uses edge attribute information (beyond edge
    topology and weights) to produce its predictions.
    """

    def forward(self, batch: Batch) -> torch.Tensor:
        device = batch.x.device
        spi_dense = _unbatch_spi(batch)

        all_ei, all_ew, bvec = [], [], []
        offset = 0
        for i in range(batch.num_graphs):
            M = (batch.batch == i).sum().item()
            spi_i = spi_dense[i, :M, :M, :]
            adj = self.compute_adjacency(spi_i)
            ei, ew = _sparsify(adj, self.top_d)
            all_ei.append(ei + offset)
            all_ew.append(ew)
            bvec.append(torch.full((M,), i, dtype=torch.long, device=device))
            offset += M

        edge_index = torch.cat(all_ei, dim=1)
        edge_weight = torch.cat(all_ew)
        bvec = torch.cat(bvec)
        zero_ea = edge_weight.new_zeros(edge_index.shape[1], self.hidden)

        h = self.node_proj(batch.x)
        for layer in self.mp_layers:
            h = layer(h, edge_index, zero_ea, edge_weight)
        out = torch.cat(
            [global_mean_pool(h, bvec), global_max_pool(h, bvec)], dim=1
        )
        return self.classifier(out)


# ---------------------------------------------------------------------------
# ShuffledEdgeMPNN — control
# ---------------------------------------------------------------------------

class ShuffledEdgeMPNN(SPIEdgeMPNN):
    """
    Control: SPI adjacency + randomly permuted SPI edge vectors per graph.

    Both adjacency and edge features use shuffled SPI vectors (shuffled across
    off-diagonal pairs within each graph, independently per forward call).

    Near-chance performance confirms that descriptor-pair correspondence
    (i.e. the specific statistic attached to the specific pair) is essential.
    If ShuffledEdgeMPNN ≈ SPIEdgeMPNN, the model is not using content.
    """

    def forward(self, batch: Batch) -> torch.Tensor:
        device = batch.x.device
        spi_dense = _unbatch_spi(batch)

        all_ei, all_ea, all_ew, bvec = [], [], [], []
        offset = 0
        for i in range(batch.num_graphs):
            M = (batch.batch == i).sum().item()
            spi_i = spi_dense[i, :M, :M, :]

            off_diag = ~torch.eye(M, dtype=torch.bool, device=device)
            flat = spi_i[off_diag]  # (M*(M-1), K)
            perm = torch.randperm(flat.shape[0], device=device)
            spi_sh = torch.zeros_like(spi_i)
            spi_sh[off_diag] = flat[perm]

            adj = self.compute_adjacency(spi_sh)
            ei, ew = _sparsify(adj, self.top_d)
            ea = self.edge_net(spi_sh[ei[0], ei[1]])

            all_ei.append(ei + offset)
            all_ea.append(ea)
            all_ew.append(ew)
            bvec.append(torch.full((M,), i, dtype=torch.long, device=device))
            offset += M

        edge_index = torch.cat(all_ei, dim=1)
        edge_attr = torch.cat(all_ea)
        edge_weight = torch.cat(all_ew)
        bvec = torch.cat(bvec)

        h = self.node_proj(batch.x)
        for layer in self.mp_layers:
            h = layer(h, edge_index, edge_attr, edge_weight)
        out = torch.cat(
            [global_mean_pool(h, bvec), global_max_pool(h, bvec)], dim=1
        )
        return self.classifier(out)


# ---------------------------------------------------------------------------
# SingleSPIMPNN — ablation (single named statistic)
# ---------------------------------------------------------------------------

class SingleSPIMPNN(nn.Module):
    """
    Ablation: fixed adjacency from one SPI dimension (e.g. transfer entropy).

    Identical architecture to CorrelationMPNN but uses a specific SPI
    dimension from spi_tensor instead of pre-computed Pearson correlation.
    The SPI dimension index is provided at construction time.
    """

    def __init__(
        self,
        spi_index: int,
        n_node_features: int,
        n_classes: int,
        hidden: int = 64,
        n_layers: int = 2,
        top_d: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.spi_index = spi_index
        self.top_d = top_d
        self.node_proj = nn.Linear(n_node_features, hidden)
        self.edge_net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.mp_layers = nn.ModuleList(
            [MPLayer(hidden, hidden, dropout) for _ in range(n_layers)]
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        device = batch.x.device
        spi_dense = _unbatch_spi(batch)  # (B, M, M, K)

        all_ei, all_ea, all_ew, bvec = [], [], [], []
        offset = 0
        for i in range(batch.num_graphs):
            M = (batch.batch == i).sum().item()
            stat_i = spi_dense[i, :M, :M, self.spi_index].abs()
            ei, ew = _sparsify(stat_i, self.top_d)
            ea = self.edge_net(ew.unsqueeze(1))
            all_ei.append(ei + offset)
            all_ea.append(ea)
            all_ew.append(ew)
            bvec.append(torch.full((M,), i, dtype=torch.long, device=device))
            offset += M

        edge_index = torch.cat(all_ei, dim=1)
        edge_attr = torch.cat(all_ea)
        edge_weight = torch.cat(all_ew)
        bvec = torch.cat(bvec)

        h = self.node_proj(batch.x)
        for layer in self.mp_layers:
            h = layer(h, edge_index, edge_attr, edge_weight)
        out = torch.cat(
            [global_mean_pool(h, bvec), global_max_pool(h, bvec)], dim=1
        )
        return self.classifier(out)


# ---------------------------------------------------------------------------
# SubsetSPIMPNN — ablation (small subset of named statistics)
# ---------------------------------------------------------------------------

class SubsetSPIMPNN(nn.Module):
    """
    Ablation: fixed adjacency from mean of k SPI dimensions, k-dim edge features.

    Adjacency: mean |SPI| across the k selected dimensions, top-d sparsified.
    Edge features: the k-dimensional sub-vector processed by edge network.

    Interpolates between SingleSPIMPNN (k=1) and FixedSPIMPNN (k=K).
    Tests whether a small oracle-selected subset of SPIs (e.g. the top-3
    by learned |w|) suffices, or whether the full vocabulary contributes
    through variance reduction across diverse estimators.
    """

    def __init__(
        self,
        spi_indices: list[int],
        n_node_features: int,
        n_classes: int,
        hidden: int = 64,
        n_layers: int = 2,
        top_d: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.spi_indices = spi_indices
        self.n_subset = len(spi_indices)
        self.top_d = top_d
        self.node_proj = nn.Linear(n_node_features, hidden)
        self.edge_net = nn.Sequential(
            nn.Linear(self.n_subset, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.mp_layers = nn.ModuleList(
            [MPLayer(hidden, hidden, dropout) for _ in range(n_layers)]
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        device = batch.x.device
        spi_dense = _unbatch_spi(batch)  # (B, M, M, K)

        all_ei, all_ea, all_ew, bvec = [], [], [], []
        offset = 0
        for i in range(batch.num_graphs):
            M = (batch.batch == i).sum().item()
            subset = spi_dense[i, :M, :M, :][:, :, self.spi_indices]  # (M, M, k)
            adj = subset.abs().mean(dim=-1)  # (M, M)
            ei, ew = _sparsify(adj, self.top_d)
            ea = self.edge_net(subset[ei[0], ei[1]])  # (E, k) -> (E, hidden)
            all_ei.append(ei + offset)
            all_ea.append(ea)
            all_ew.append(ew)
            bvec.append(torch.full((M,), i, dtype=torch.long, device=device))
            offset += M

        edge_index = torch.cat(all_ei, dim=1)
        edge_attr = torch.cat(all_ea)
        edge_weight = torch.cat(all_ew)
        bvec = torch.cat(bvec)

        h = self.node_proj(batch.x)
        for layer in self.mp_layers:
            h = layer(h, edge_index, edge_attr, edge_weight)
        out = torch.cat(
            [global_mean_pool(h, bvec), global_max_pool(h, bvec)], dim=1
        )
        return self.classifier(out)
