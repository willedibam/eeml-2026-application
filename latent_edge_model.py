"""
Fair latent baseline: learned adjacency AND learned edge features from
node embeddings, using the SAME message passing architecture as SPIEdgeMPNN.

This isolates the comparison to: precomputed SPI descriptors vs. learned
pairwise representations. The message passing, edge network, classifier,
pooling, and sparsification are all identical.

Add this class to model.py alongside the existing baselines.
Register "latent-edge" in _ALL_MODELS in run_pipeline.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool, global_max_pool

# Import MPLayer from the existing model.py — same message passing
# architecture as SPIEdgeMPNN uses.
# from .model import MPLayer


class LatentEdgeMPNN(nn.Module):
    """
    Fair latent baseline with learned edge features.

    Learns adjacency AND edge features from trainable node embeddings,
    using the same downstream architecture as SPIEdgeMPNN:
        - Same edge network MLP (edge_dim → 64)
        - Same MPLayer (residual + LayerNorm message passing)
        - Same classifier (mean+max pool → MLP)
        - Same top-d sparsification

    The ONLY difference from SPIEdgeMPNN is the source of pairwise
    information:
        - SPIEdgeMPNN: precomputed K-dim SPI descriptor per pair
        - LatentEdgeMPNN: learned K-dim vector from node embeddings

    Adjacency: A_ij = softplus(MLP_adj([u_i; u_j]))
    Edge features: e_ij = MLP_edge([u_i; u_j]) ∈ R^K

    Then e_ij is passed through the same edge_net and MPLayers.

    Args:
        n_edge_dim: Dimension of learned edge features. Should match
            n_spi from SPIEdgeMPNN for a fair comparison.
        n_node_features: F_n, node feature dimension.
        n_classes: C, number of output classes.
        n_nodes: M, number of nodes (fixed across samples).
        hidden: H, hidden width. Default 64.
        n_layers: Number of message passing layers. Default 3.
        top_d: Number of outgoing edges per node. Default 4.
        dropout: Dropout rate. Default 0.1.
        embed_dim: Dimension of per-node learnable embeddings. Default 16.
    """

    def __init__(
        self,
        n_edge_dim: int,
        n_node_features: int,
        n_classes: int,
        n_nodes: int = 5,
        hidden: int = 64,
        n_layers: int = 3,
        top_d: int = 4,
        dropout: float = 0.1,
        embed_dim: int = 16,
    ):
        super().__init__()
        self.hidden = hidden
        self.top_d = top_d
        self.n_nodes = n_nodes
        self.n_edge_dim = n_edge_dim

        # Learnable node embeddings (one per node position, shared across samples)
        self.node_embed = nn.Parameter(torch.randn(n_nodes, embed_dim) * 0.1)

        # Pairwise → adjacency scalar
        self.adj_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # Pairwise → edge descriptor (same dim as SPI vector)
        self.pair_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, n_edge_dim),
        )

        # --- Everything below is identical to SPIEdgeMPNN ---

        # Node projection
        self.node_proj = nn.Linear(n_node_features, hidden)

        # Edge network: n_edge_dim → 64 (same as SPIEdgeMPNN.edge_net)
        self.edge_net = nn.Sequential(
            nn.Linear(n_edge_dim, 32),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.GELU(),
        )

        # Message passing layers (same MPLayer as SPIEdgeMPNN)
        self.mp_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.mp_layers.append(MPLayer(hidden, edge_dim=64, dropout=dropout))

        # Classifier (same as SPIEdgeMPNN)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def _compute_pairwise(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute adjacency and edge features from node embeddings.

        Returns:
            adj: (M, M) non-negative adjacency weights
            edge_feats: (M, M, n_edge_dim) learned edge descriptors
        """
        M = self.n_nodes
        u = self.node_embed  # (M, embed_dim)

        # All pairs: (M, M, 2*embed_dim)
        u_i = u.unsqueeze(1).expand(-1, M, -1)  # (M, M, embed_dim)
        u_j = u.unsqueeze(0).expand(M, -1, -1)  # (M, M, embed_dim)
        pairs = torch.cat([u_i, u_j], dim=-1)    # (M, M, 2*embed_dim)

        # Adjacency
        adj = F.softplus(self.adj_mlp(pairs).squeeze(-1))  # (M, M)
        mask = ~torch.eye(M, dtype=torch.bool, device=u.device)
        adj = adj * mask.float()

        # Edge features
        edge_feats = self.pair_mlp(pairs)  # (M, M, n_edge_dim)

        return adj, edge_feats

    def forward(self, batch: Batch) -> torch.Tensor:
        device = batch.x.device
        batch_size = batch.num_graphs
        M = self.n_nodes

        # Compute shared adjacency and edge features
        adj, edge_feats = self._compute_pairwise()

        # Sparsify: top-d per node
        d = min(self.top_d, M - 1)
        _, top_idx = adj.topk(d, dim=1)  # (M, d)
        src = torch.arange(M, device=device).unsqueeze(1).expand(-1, d).reshape(-1)
        dst = top_idx.reshape(-1)

        # Edge features for retained edges
        edge_spi = edge_feats[src, dst]        # (E, n_edge_dim)
        edge_attr = self.edge_net(edge_spi)    # (E, 64)
        edge_weight = adj[src, dst]            # (E,)

        # Replicate across batch
        all_edge_index = []
        all_edge_attr = []
        all_edge_weight = []
        all_batch_vec = []
        node_offset = 0

        for i in range(batch_size):
            all_edge_index.append(
                torch.stack([src + node_offset, dst + node_offset])
            )
            all_edge_attr.append(edge_attr)
            all_edge_weight.append(edge_weight)
            all_batch_vec.append(
                torch.full((M,), i, dtype=torch.long, device=device)
            )
            node_offset += M

        edge_index = torch.cat(all_edge_index, dim=1)
        edge_attr = torch.cat(all_edge_attr, dim=0)
        edge_weight = torch.cat(all_edge_weight, dim=0)
        batch_vec = torch.cat(all_batch_vec, dim=0)

        # Node projection
        h = self.node_proj(batch.x)

        # Message passing (same MPLayer as SPIEdgeMPNN)
        for layer in self.mp_layers:
            h = layer(h, edge_index, edge_attr, edge_weight)

        # Graph readout (same as SPIEdgeMPNN)
        h_mean = global_mean_pool(h, batch_vec)
        h_max = global_max_pool(h, batch_vec)
        h_graph = torch.cat([h_mean, h_max], dim=1)

        return self.classifier(h_graph)


# ---------------------------------------------------------------------------
# Copy of MPLayer from model.py — included here so this file is
# self-contained for review. In the actual repo, import from model.py.
# ---------------------------------------------------------------------------

class MPLayer(nn.Module):
    """
    Single message passing layer with edge network.
    Identical to the MPLayer in model.py.
    """

    def __init__(self, hidden: int, edge_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden + edge_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, hidden),
        )
        self.layer_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_index, edge_attr, edge_weight):
        src, dst = edge_index
        msg_input = torch.cat([h[dst], h[src], edge_attr], dim=1)
        messages = self.message_mlp(msg_input)
        messages = messages * edge_weight.unsqueeze(1)
        N = h.shape[0]
        agg = torch.zeros(N, messages.shape[1], device=h.device, dtype=h.dtype)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)
        h = h + self.dropout(self.layer_norm(agg))
        return h