"""
Visualization functions for EEML 2026 extended abstract figures.

All functions return (fig, ax) or (fig, axes) for easy composition.
Pass ax= to plot into an existing axes (for subplot assembly).
Designed to produce PDF-quality components for Inkscape assembly.

Usage:
    python visualization.py              # generate all figure components
    python visualization.py --fig1-only  # Figure 1 panels only
    python visualization.py --fig2-only  # Figure 2 panels only
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import networkx as nx
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------

STYLE = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# Model colours — warm = SPI vocabulary, cool = controls
MODEL_COLORS = {
    "spi-mpnn": "#D32F2F",
    "fixed-spi": "#FF7043",
    "mlp-mix": "#FFB300",
    "edge-ablation": "#8D6E63",
    "correlation": "#1976D2",
    "latent": "#7B1FA2",
    "shuffled": "#78909C",
    "node-only": "#9E9E9E",
}

MODEL_LABELS = {
    "spi-mpnn": "SPI-MPNN (ours)",
    "fixed-spi": "Fixed-SPI",
    "mlp-mix": "MLP-Mix",
    "edge-ablation": "Edge Ablation",
    "correlation": "Correlation",
    "latent": "Latent",
    "shuffled": "Shuffled",
    "node-only": "Node-Only",
}

MODEL_MARKERS = {
    "spi-mpnn": "o",
    "fixed-spi": "s",
    "mlp-mix": "^",
    "edge-ablation": "D",
    "correlation": "d",
    "latent": "v",
    "shuffled": "x",
    "node-only": "+",
}

MODEL_LINESTYLES = {
    "spi-mpnn": "-",
    "fixed-spi": "-",
    "mlp-mix": "-",
    "edge-ablation": "--",
    "correlation": "--",
    "latent": "--",
    "shuffled": ":",
    "node-only": ":",
}

FAMILY_COLORS = {
    "causal": "#D32F2F",
    "spectral": "#1976D2",
    "linear": "#388E3C",
    "information": "#7B1FA2",
    "distance": "#FF8F00",
    "rank": "#78909C",
}

MOTIF_NODE_LABELS = {0: r"$A$", 1: r"$B$", 2: r"$C$"}


def apply_style():
    """Apply publication-quality style globally."""
    plt.rcParams.update(STYLE)


# ---------------------------------------------------------------------------
# Figure 1 components
# ---------------------------------------------------------------------------

# Motif definitions: (edge_list, motif_node_indices)
MOTIFS = {
    "Chain": ([(0, 1), (1, 2)], [0, 1, 2]),
    "Fork": ([(1, 0), (2, 0)], [0, 1, 2]),
    "Collider": ([(0, 1), (2, 1)], [0, 1, 2]),
}


def plot_motif_graph(
    motif_name: str,
    M: int = 10,
    *,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (2.0, 2.0),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Draw a directed motif graph with M nodes.
    3 motif nodes in A-B-C linear layout; nuisance nodes in outer ring.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    edges, motif_nodes = MOTIFS[motif_name]

    G = nx.DiGraph()
    G.add_nodes_from(range(M))
    G.add_edges_from(edges)

    # Layout: motif nodes in a horizontal line (A-B-C)
    pos = {}
    motif_positions = [(-0.35, 0.0), (0.0, 0.0), (0.35, 0.0)]
    for i, node in enumerate(motif_nodes):
        pos[node] = motif_positions[i]

    # Nuisance nodes in outer ring with faint dotted inter-connections
    nuisance = [n for n in range(M) if n not in motif_nodes]
    for i, node in enumerate(nuisance):
        angle = 2 * np.pi * i / len(nuisance) - np.pi / 2
        pos[node] = (0.7 * np.cos(angle), 0.7 * np.sin(angle))

    # Draw faint dotted edges between nuisance nodes (visual only)
    if len(nuisance) > 1:
        nuisance_edges = []
        for i in range(len(nuisance)):
            for j in range(i + 1, len(nuisance)):
                nuisance_edges.append((nuisance[i], nuisance[j]))
        nx.draw_networkx_edges(
            G.to_undirected(), pos, edgelist=nuisance_edges,
            edge_color="#E0E0E0", width=0.3, style="dotted", ax=ax,
        )

    # Draw nuisance nodes
    nx.draw_networkx_nodes(
        G, pos, nodelist=nuisance, node_color="#E0E0E0",
        node_size=120, edgecolors="#BDBDBD", linewidths=0.5, ax=ax,
    )
    # Draw motif nodes
    nx.draw_networkx_nodes(
        G, pos, nodelist=motif_nodes, node_color="#1565C0",
        node_size=200, edgecolors="#0D47A1", linewidths=1.0, ax=ax,
    )
    # Draw motif edges
    nx.draw_networkx_edges(
        G, pos, edgelist=edges, edge_color="#1565C0",
        width=2.0, arrows=True, arrowsize=12,
        connectionstyle="arc3,rad=0.1", ax=ax,
    )
    # Labels on motif nodes: A, B, C
    motif_labels = {n: MOTIF_NODE_LABELS[i] for i, n in enumerate(motif_nodes)}
    nx.draw_networkx_labels(
        G, pos, labels=motif_labels, font_size=7,
        font_color="white", font_weight="bold", ax=ax,
    )

    ax.set_title(motif_name, fontsize=9, fontweight="bold", pad=4)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")

    return fig, ax


def plot_time_series(
    data,
    *,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (4.0, 2.0),
    max_T: int | None = 200,
    title: str | None = None,
    alpha: float = 0.7,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot univariate or multivariate time series.

    Parameters
    ----------
    data : array-like or str/Path
        (T,) or (T, M) array, or path to .npy file.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    if isinstance(data, (str, Path)):
        data = np.load(data)
    data = np.atleast_2d(data)
    if data.shape[0] < data.shape[1]:
        data = data.T  # ensure (T, M)

    if max_T is not None and data.shape[0] > max_T:
        data = data[:max_T]

    T, M = data.shape
    for m in range(M):
        ax.plot(range(T), data[:, m], linewidth=0.6, alpha=alpha)

    ax.set_xlabel(r"Time ($t$)")
    ax.set_ylabel("Amplitude")
    if title:
        ax.set_title(title, fontsize=8)

    return fig, ax


def plot_mts_heatmap(
    timeseries_path: str | Path,
    *,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (3.0, 1.5),
    cmap: str = "RdBu_r",
    title: str | None = None,
    max_T: int | None = 200,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Carpet plot (M x T heatmap) from a timeseries.npy file using imshow.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    ts = np.load(timeseries_path)  # (T, M)
    if max_T is not None and ts.shape[0] > max_T:
        ts = ts[:max_T]

    vmax = np.abs(ts).max()
    ax.imshow(
        ts.T, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xlabel(r"Time ($t$)", fontsize=7)
    ax.set_ylabel(r"Channel ($m$)", fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)

    return fig, ax


def plot_mpi_heatmap(
    npz_path: str | Path,
    spi_name: str,
    *,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (1.8, 1.5),
    cmap: str | None = None,
    title: str | None = None,
    show_colorbar: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot one SPI matrix (M x M) from spi_mpis.npz using imshow.

    Automatically selects diverging cmap for asymmetric SPIs (causal family),
    sequential for symmetric ones.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    with np.load(npz_path) as npz:
        mat = np.asarray(npz[spi_name], dtype=np.float64)

    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

    # Detect asymmetry
    is_asymmetric = not np.allclose(mat, mat.T, atol=1e-6, equal_nan=True)

    if cmap is None:
        cmap = "RdBu_r" if is_asymmetric else "YlOrRd"

    vmax = np.abs(mat).max() if np.abs(mat).max() > 0 else 1.0

    if is_asymmetric:
        im = ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax,
                        interpolation="nearest", aspect="equal")
    else:
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax,
                        interpolation="nearest", aspect="equal")

    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([])
    ax.set_yticks([])

    label = title if title else spi_name
    sym_tag = "directed" if is_asymmetric else "symmetric"
    ax.set_title(f"{label}\n({sym_tag})", fontsize=7, pad=2)

    return fig, ax


# Alias for backwards compatibility
plot_spi_heatmap = plot_mpi_heatmap


# ---------------------------------------------------------------------------
# Figure 2 components
# ---------------------------------------------------------------------------

def load_results(results_path: str | Path) -> dict:
    """Load a results JSON file."""
    with open(results_path) as f:
        return json.load(f)


def overlay_model(
    ax: plt.Axes,
    results_path: str | Path,
    model: str,
    *,
    metric: str = "f1",
    color: str | None = None,
    label: str | None = None,
    marker: str | None = None,
    linestyle: str | None = None,
) -> plt.Axes:
    """
    Overlay a single model's sample efficiency curve onto an existing axes.
    Useful for composing curves from different result files.
    """
    data = load_results(results_path)
    n_values = sorted(int(k) for k in data["results"].keys())

    c = color or MODEL_COLORS.get(model, "#333333")
    lbl = label or MODEL_LABELS.get(model, model)
    mk = marker or MODEL_MARKERS.get(model, "o")
    ls = linestyle or MODEL_LINESTYLES.get(model, "-")

    means, stds, ns = [], [], []
    for n in n_values:
        n_str = str(n)
        model_data = data["results"][n_str]["models"].get(model)
        if model_data is None:
            continue
        means.append(model_data[f"{metric}_mean"])
        stds.append(model_data[f"{metric}_std"])
        ns.append(n)

    means, stds = np.array(means), np.array(stds)
    ax.plot(ns, means, color=c, marker=mk, label=lbl,
            linestyle=ls, linewidth=1.5, markersize=4, zorder=3)
    ax.fill_between(ns, means - stds, np.minimum(means + stds, 1.0),
                    color=c, alpha=0.12, zorder=2)
    return ax


def plot_sample_efficiency(
    results_path: str | Path,
    *,
    models: list[str] | None = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (5.5, 3.5),
    show_chance: bool = True,
    show_symmetric_ceiling: bool = True,
    metric: str = "f1",
    band_mode: str = "std",
    grid: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Sample efficiency curves: F1 (or acc) vs n/class for all models.

    Parameters
    ----------
    band_mode : str
        "std" (default), "sem", or "minmax".
    grid : bool
        Show light grid lines (default True).
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    data = load_results(results_path)
    n_values = sorted(int(k) for k in data["results"].keys())

    if models is None:
        first_n = data["results"][str(n_values[0])]
        models = list(first_n["models"].keys())

    for model in models:
        color = MODEL_COLORS.get(model, "#333333")
        label = MODEL_LABELS.get(model, model)
        marker = MODEL_MARKERS.get(model, "o")
        ls = MODEL_LINESTYLES.get(model, "-")

        means, stds, ns, all_seeds = [], [], [], []
        for n in n_values:
            n_str = str(n)
            if n_str not in data["results"]:
                continue
            model_data = data["results"][n_str]["models"].get(model)
            if model_data is None:
                continue
            means.append(model_data[f"{metric}_mean"])
            stds.append(model_data[f"{metric}_std"])
            ns.append(n)
            # Collect per-seed values for sem/minmax
            per_seed = model_data.get("per_seed", [])
            seed_vals = [s.get(f"test_{metric}", s.get(metric, 0))
                         for s in per_seed]
            all_seeds.append(seed_vals)

        means, stds = np.array(means), np.array(stds)

        ax.plot(ns, means, color=color, marker=marker, label=label,
                linestyle=ls, linewidth=1.5, markersize=4, zorder=3)

        if band_mode == "std":
            lo = means - stds
            hi = np.minimum(means + stds, 1.0)
        elif band_mode == "sem":
            n_seeds = np.array([len(s) for s in all_seeds])
            sem = stds / np.sqrt(np.maximum(n_seeds, 1))
            lo = means - sem
            hi = np.minimum(means + sem, 1.0)
        elif band_mode == "minmax":
            lo = np.array([min(s) if s else m for s, m in zip(all_seeds, means)])
            hi = np.array([max(s) if s else m for s, m in zip(all_seeds, means)])
        else:
            lo = means - stds
            hi = np.minimum(means + stds, 1.0)

        ax.fill_between(ns, lo, hi, color=color, alpha=0.12, zorder=2)

    if show_chance:
        ax.axhline(1 / 3, color="#D32F2F", linestyle="--", linewidth=1.2,
                    zorder=1, label="Chance (1/3)")
    if show_symmetric_ceiling:
        ax.axhline(2 / 3, color="#9E9E9E", linestyle=":", linewidth=0.8,
                    zorder=1, label="Symmetric ceiling (2/3)")

    ax.set_xscale("log")
    ax.set_xticks(n_values)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel(r"Training samples per class ($n$)")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0.15, 1.05)
    ax.legend(loc="lower right", framealpha=0.9, ncol=2)

    if grid:
        ax.grid(True, alpha=0.2, linewidth=0.5)

    return fig, ax


def plot_sample_efficiency_multi(
    results_paths: dict[str, str | Path],
    *,
    models: list[str] | None = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (5.5, 3.5),
    metric: str = "f1",
    band_mode: str = "std",
    grid: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Overlay sample efficiency curves from multiple result files.

    results_paths: {model_name: path} — each file contributes one model's data.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    all_n = set()
    for path in results_paths.values():
        data = load_results(path)
        all_n.update(int(k) for k in data["results"].keys())

    if models is None:
        models = list(results_paths.keys())

    for model in models:
        path = results_paths.get(model)
        if path is None:
            continue
        data = load_results(path)

        color = MODEL_COLORS.get(model, "#333333")
        label = MODEL_LABELS.get(model, model)
        marker = MODEL_MARKERS.get(model, "o")
        ls = MODEL_LINESTYLES.get(model, "-")

        means, stds, ns = [], [], []
        for n in sorted(all_n):
            n_str = str(n)
            if n_str not in data["results"]:
                continue
            model_data = data["results"][n_str]["models"].get(model)
            if model_data is None:
                continue
            means.append(model_data[f"{metric}_mean"])
            stds.append(model_data[f"{metric}_std"])
            ns.append(n)

        if not ns:
            continue
        means, stds = np.array(means), np.array(stds)

        ax.plot(ns, means, color=color, marker=marker, label=label,
                linestyle=ls, linewidth=1.5, markersize=4, zorder=3)
        ax.fill_between(ns, means - stds, np.minimum(means + stds, 1.0),
                        color=color, alpha=0.12, zorder=2)

    ax.axhline(1 / 3, color="#D32F2F", linestyle="--", linewidth=1.2, zorder=1)
    ax.axhline(2 / 3, color="#9E9E9E", linestyle=":", linewidth=0.8, zorder=1)

    ax.set_xscale("log")
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel(r"Training samples per class ($n$)")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0.15, 1.05)
    ax.legend(loc="lower right", framealpha=0.9, ncol=2)

    if grid:
        ax.grid(True, alpha=0.2, linewidth=0.5)

    return fig, ax


def plot_family_weights(
    results_path: str | Path,
    n_value: int = 500,
    *,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (3.5, 2.5),
    top_k_labels: int = 3,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Violin plot of learned |w| by SPI family, averaged across seeds.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    data = load_results(results_path)
    spi_names = data["spi_names"]
    spi_families_dict = data.get("spi_families", {})

    idx_to_family = {}
    for fam, indices in spi_families_dict.items():
        for idx in indices:
            idx_to_family[idx] = fam

    n_str = str(n_value)
    per_seed = data["results"][n_str]["models"]["spi-mpnn"]["per_seed"]
    w_matrix = np.array([s["learned_w"] for s in per_seed])  # (n_seeds, K)
    w_mean = np.abs(w_matrix).mean(axis=0)  # (K,)

    import pandas as pd
    records = []
    for k in range(len(spi_names)):
        records.append({
            "SPI": spi_names[k],
            "Family": idx_to_family.get(k, "other"),
            r"$|w|$": w_mean[k],
        })
    df = pd.DataFrame(records)

    family_l2 = df.groupby("Family")[r"$|w|$"].apply(
        lambda x: np.sqrt((x**2).sum())
    ).sort_values(ascending=False)
    family_order = family_l2.index.tolist()

    sns.violinplot(
        data=df, y="Family", x=r"$|w|$", hue="Family", order=family_order,
        hue_order=family_order,
        palette={f: FAMILY_COLORS.get(f, "#999") for f in family_order},
        inner="quart", linewidth=0.8, cut=0, ax=ax, orient="h", legend=False,
    )

    top_spis = df.nlargest(top_k_labels, r"$|w|$")
    for _, row in top_spis.iterrows():
        family_idx = family_order.index(row["Family"])
        short_name = _short_spi_name(row["SPI"])
        ax.annotate(
            short_name, (row[r"$|w|$"], family_idx),
            textcoords="offset points", xytext=(5, 0),
            fontsize=5.5, fontstyle="italic", color="#333",
        )

    ax.set_xlabel(r"Mean $|w_k|$ across seeds")
    ax.set_ylabel("")
    ax.set_title(rf"Learned weights by family ($n$={n_value})", fontsize=8)

    return fig, ax


def plot_family_bar(
    results_path: str | Path,
    n_value: int = 500,
    *,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (3.0, 2.0),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Horizontal bar chart: family L2 norm of w.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    data = load_results(results_path)
    spi_names = data["spi_names"]
    spi_families_dict = data.get("spi_families", {})

    idx_to_family = {}
    for fam, indices in spi_families_dict.items():
        for idx in indices:
            idx_to_family[idx] = fam

    n_str = str(n_value)
    per_seed = data["results"][n_str]["models"]["spi-mpnn"]["per_seed"]
    w_matrix = np.array([s["learned_w"] for s in per_seed])
    w_mean = np.abs(w_matrix).mean(axis=0)

    family_l2 = {}
    for k in range(len(spi_names)):
        fam = idx_to_family.get(k, "other")
        family_l2.setdefault(fam, []).append(w_mean[k])
    family_l2 = {f: np.sqrt(np.sum(np.array(v) ** 2))
                 for f, v in family_l2.items()}

    sorted_fams = sorted(family_l2.items(), key=lambda x: x[1], reverse=True)
    names = [f[0] for f in sorted_fams]
    vals = [f[1] for f in sorted_fams]
    colors = [FAMILY_COLORS.get(n, "#999") for n in names]

    ax.barh(names, vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel(r"$\|\mathbf{w}_g\|_2$", fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Statistical signature", fontsize=8)

    return fig, ax


def plot_per_seed_strip(
    results_path: str | Path,
    n_value: int = 500,
    models: list[str] | None = None,
    *,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (4.0, 2.5),
    metric: str = "f1",
    show_violin: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Strip/swarm plot of per-seed F1 for selected models at a given n.

    Parameters
    ----------
    show_violin : bool
        If True, overlay a translucent violin behind the strip points.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    import pandas as pd

    data = load_results(results_path)
    n_str = str(n_value)

    if models is None:
        models = ["spi-mpnn", "fixed-spi", "correlation", "latent"]

    records = []
    for model in models:
        model_data = data["results"][n_str]["models"].get(model)
        if model_data is None:
            continue
        for seed_data in model_data["per_seed"]:
            key = f"test_{metric}"
            records.append({
                "Model": MODEL_LABELS.get(model, model),
                "F1": seed_data[key],
                "model_key": model,
            })

    df = pd.DataFrame(records)
    palette = {MODEL_LABELS.get(m, m): MODEL_COLORS.get(m, "#333")
               for m in models}

    if show_violin:
        sns.violinplot(
            data=df, x="Model", y="F1", hue="Model", palette=palette,
            inner=None, linewidth=0.5, cut=0, ax=ax, legend=False,
            alpha=0.15,
        )

    sns.stripplot(
        data=df, x="Model", y="F1", hue="Model", palette=palette,
        jitter=0.15, size=5, alpha=0.8, ax=ax, legend=False,
    )
    ax.axhline(1 / 3, color="#D32F2F", linestyle="--", linewidth=0.7)
    ax.set_ylabel("Per-seed Macro F1")
    ax.set_xlabel("")
    ax.set_title(rf"Per-seed performance ($n$={n_value})", fontsize=8)

    return fig, ax


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short_spi_name(name: str) -> str:
    """Shorten SPI names for annotations."""
    replacements = [
        ("sgc_parametric_mean_fs-1_fmin-0-25_fmax-0-5_order-1", "SGC(0.25-0.5Hz)"),
        ("sgc_parametric_max_fs-1_fmin-0-25_fmax-0-5_order-1", "SGC(max)"),
        ("sgc_parametric_mean_fs-1_fmin-0-25_fmax-0-5_order-None", "SGC(all)"),
        ("gc_gaussian_k-1_kt-1_l-1_lt-1", "GC(Gaussian)"),
        ("gc_gaussian_k-max-10_tau-max-2", "GC(ext. lag)"),
        ("tlmi_gaussian", "TLMI"),
        ("xme_kozachenko_k10", "XME(k=10)"),
        ("xcorr_Pearson", "Pearson r"),
    ]
    for long, short in replacements:
        if name == long:
            return short
    if len(name) > 25:
        return name[:22] + "..."
    return name


# ---------------------------------------------------------------------------
# CLI: generate all figure components as PDFs
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate figure components")
    parser.add_argument("--fig1-only", action="store_true")
    parser.add_argument("--fig2-only", action="store_true")
    parser.add_argument("--outdir", type=str, default="figures")
    args = parser.parse_args()

    apply_style()
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    data_dir = Path("data/260327_eeml")
    results_main = Path("results/sample_efficiency_30seeds_main_results.json")
    results_edge = Path("results/sample_efficiency_edge_ablation_30seeds_results.json")

    # Example instance for SPI heatmaps
    instance = data_dir / "var-chain" / "M10_T500_I0"
    npz_path = instance / "spi_mpis.npz"
    ts_path = instance / "timeseries.npy"

    do_fig1 = not args.fig2_only
    do_fig2 = not args.fig1_only

    if do_fig1:
        print("=== Figure 1 components ===")

        # Panel A: motif graphs (individual)
        for name in ["Chain", "Fork", "Collider"]:
            fig, _ = plot_motif_graph(name, M=10)
            safe = name.lower()
            fig.savefig(outdir / f"fig1a_motif_{safe}.pdf")
            plt.close(fig)
        print("  Saved motif graphs")

        # Panel A: MTS heatmaps (one per class)
        for cls in ["var-chain", "var-fork", "var-collider"]:
            ts = data_dir / cls / "M10_T500_I0" / "timeseries.npy"
            if ts.exists():
                fig, _ = plot_mts_heatmap(ts, title=cls.replace("var-", ""))
                fig.savefig(outdir / f"fig1a_mts_{cls}.pdf")
                plt.close(fig)
        print("  Saved MTS heatmaps")

        # Panel B: SPI heatmaps (chain vs fork comparison)
        for cls in ["var-chain", "var-fork"]:
            cls_npz = data_dir / cls / "M10_T500_I0" / "spi_mpis.npz"
            if cls_npz.exists():
                with np.load(cls_npz) as npz:
                    available = set(npz.files)
                for spi, title in [
                    ("xcorr_mean_sig-True", r"$\bar{r}$ (symmetric)"),
                    ("sgc_parametric_mean_fs-1_fmin-0-25_fmax-0-5_order-1",
                     "SGC (directed)"),
                ]:
                    if spi in available:
                        fig, _ = plot_mpi_heatmap(
                            cls_npz, spi,
                            title=f"{cls.replace('var-', '')} -- {title}",
                        )
                        safe = spi.replace("-", "_")[:20]
                        fig.savefig(outdir / f"fig1b_{cls}_{safe}.pdf")
                        plt.close(fig)
        print("  Saved chain/fork comparison heatmaps")

        # Panel B: MPI stack (selected SPIs)
        if npz_path.exists():
            with np.load(npz_path) as npz:
                available = set(npz.files)
            spis_to_show = [
                ("xcorr_mean_sig-True", r"Cross-corr $\bar{r}$"),
                ("sgc_parametric_mean_fs-1_fmin-0-25_fmax-0-5_order-1",
                 "SGC (0.25--0.5 Hz)"),
            ]
            for spi_name, spi_title in spis_to_show:
                if spi_name in available:
                    fig, _ = plot_mpi_heatmap(
                        npz_path, spi_name, title=spi_title,
                    )
                    safe = spi_name.replace("-", "_")[:20]
                    fig.savefig(outdir / f"fig1b_mpi_{safe}.pdf")
                    plt.close(fig)
            print("  Saved MPI heatmaps")

        # Panel C: family bar chart
        if results_main.exists():
            fig, _ = plot_family_bar(results_main, n_value=500)
            fig.savefig(outdir / "fig1c_family_bar.pdf")
            plt.close(fig)
            print("  Saved fig1c_family_bar.pdf")

    if do_fig2:
        print("=== Figure 2 components ===")

        if results_main.exists():
            # Panel A: sample efficiency curves
            fig, ax = plot_sample_efficiency(
                results_main,
                models=["spi-mpnn", "fixed-spi", "mlp-mix",
                        "correlation", "latent", "shuffled", "node-only"],
            )
            fig.savefig(outdir / "fig2a_sample_efficiency.pdf")
            plt.close(fig)
            print("  Saved fig2a_sample_efficiency.pdf")

            # With edge-ablation overlaid
            if results_edge.exists():
                fig, ax = plot_sample_efficiency(
                    results_main,
                    models=["spi-mpnn", "fixed-spi", "mlp-mix",
                            "correlation", "latent", "shuffled", "node-only"],
                )
                overlay_model(ax, results_edge, "edge-ablation")
                ax.legend(loc="lower right", framealpha=0.9, ncol=2)
                fig.savefig(outdir / "fig2a_sample_efficiency_with_edge_abl.pdf")
                plt.close(fig)
                print("  Saved fig2a_sample_efficiency_with_edge_abl.pdf")

            # Panel B: family weights violin
            fig, _ = plot_family_weights(results_main, n_value=500)
            fig.savefig(outdir / "fig2b_family_violin.pdf")
            plt.close(fig)
            print("  Saved fig2b_family_violin.pdf")

            # Panel C: per-seed strip
            fig, _ = plot_per_seed_strip(
                results_main, n_value=500,
                models=["spi-mpnn", "fixed-spi", "mlp-mix",
                        "correlation", "latent"],
                show_violin=True,
            )
            fig.savefig(outdir / "fig2c_per_seed_strip.pdf")
            plt.close(fig)
            print("  Saved fig2c_per_seed_strip.pdf")

    print(f"\nAll outputs in {outdir}/")


if __name__ == "__main__":
    main()
