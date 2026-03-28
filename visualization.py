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
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
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


def apply_style():
    """Apply publication-quality style globally."""
    plt.rcParams.update(STYLE)


# ---------------------------------------------------------------------------
# Figure 1 components
# ---------------------------------------------------------------------------

# Motif definitions: (edge_list, motif_node_indices)
MOTIFS = {
    "Chain": ([(0, 1), (1, 2)], [0, 1, 2]),
    "Fork": ([(0, 1), (0, 2)], [0, 1, 2]),
    "Collider": ([(1, 0), (2, 0)], [0, 1, 2]),
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
    3 motif nodes are highlighted; remaining are faint nuisance nodes.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    edges, motif_nodes = MOTIFS[motif_name]

    G = nx.DiGraph()
    G.add_nodes_from(range(M))
    G.add_edges_from(edges)

    # Layout: motif nodes in a triangle at centre, nuisance nodes in a ring
    pos = {}
    # Motif triangle
    motif_positions = [(-0.3, 0.3), (0.3, 0.3), (0.0, -0.3)]
    for i, node in enumerate(motif_nodes):
        pos[node] = motif_positions[i]

    # Nuisance nodes in outer ring
    nuisance = [n for n in range(M) if n not in motif_nodes]
    for i, node in enumerate(nuisance):
        angle = 2 * np.pi * i / len(nuisance) - np.pi / 2
        pos[node] = (0.7 * np.cos(angle), 0.7 * np.sin(angle))

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
    # Labels on motif nodes only
    motif_labels = {n: str(n) for n in motif_nodes}
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


def plot_motif_graphs_row(
    M: int = 10,
    figsize: tuple = (6.0, 2.2),
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Draw all three motif graphs side by side."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, name in zip(axes, ["Chain", "Fork", "Collider"]):
        plot_motif_graph(name, M=M, ax=ax)
    fig.tight_layout(pad=0.5)
    return fig, axes


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
    Carpet plot (T x M heatmap) from a timeseries.npy file.
    Truncates to max_T timesteps for visual clarity.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    ts = np.load(timeseries_path)  # (T, M)
    if max_T is not None and ts.shape[0] > max_T:
        ts = ts[:max_T]

    vmax = np.abs(ts).max()
    sns.heatmap(
        ts.T, ax=ax, cmap=cmap, center=0, vmin=-vmax, vmax=vmax,
        cbar=False, xticklabels=False, yticklabels=False,
    )
    ax.set_xlabel("Time ($t$)", fontsize=7)
    ax.set_ylabel("Channel ($m$)", fontsize=7)
    if title:
        ax.set_title(title, fontsize=8)

    return fig, ax


def plot_spi_heatmap(
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
    Plot one SPI matrix (M x M) from spi_mpis.npz.

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
    kwargs = dict(cmap=cmap, xticklabels=False, yticklabels=False, square=True)

    if is_asymmetric:
        kwargs.update(center=0, vmin=-vmax, vmax=vmax)
    else:
        kwargs.update(vmin=0, vmax=vmax)

    sns.heatmap(mat, ax=ax, cbar=show_colorbar, **kwargs)

    label = title if title else spi_name
    sym_tag = "directed" if is_asymmetric else "symmetric"
    ax.set_title(f"{label}\n({sym_tag})", fontsize=7, pad=2)

    return fig, ax


def plot_spi_stack(
    npz_path: str | Path,
    spi_names: list[str],
    titles: list[str] | None = None,
    figsize_per: tuple = (1.5, 1.3),
    ncols: int = 1,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Plot a vertical stack of SPI heatmaps — the Panel B component.
    """
    nrows = int(np.ceil(len(spi_names) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
    )
    axes_flat = np.atleast_1d(axes).flatten()

    if titles is None:
        titles = spi_names

    for i, (name, title) in enumerate(zip(spi_names, titles)):
        plot_spi_heatmap(npz_path, name, ax=axes_flat[i], title=title)

    # Hide unused axes
    for j in range(len(spi_names), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout(pad=0.3)
    return fig, list(axes_flat[:len(spi_names)])


# ---------------------------------------------------------------------------
# Figure 2 components
# ---------------------------------------------------------------------------

def load_results(results_path: str | Path) -> dict:
    """Load a results JSON file."""
    with open(results_path) as f:
        return json.load(f)


def plot_sample_efficiency(
    results_path: str | Path,
    *,
    models: list[str] | None = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (5.5, 3.5),
    show_chance: bool = True,
    show_symmetric_ceiling: bool = True,
    metric: str = "f1",
) -> tuple[plt.Figure, plt.Axes]:
    """
    Sample efficiency curves: F1 (or acc) vs n/class for all models.

    Reads from the standard results JSON format.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    data = load_results(results_path)
    n_values = sorted(int(k) for k in data["results"].keys())

    if models is None:
        # Determine from first n entry
        first_n = data["results"][str(n_values[0])]
        models = list(first_n["models"].keys())

    for model in models:
        color = MODEL_COLORS.get(model, "#333333")
        label = MODEL_LABELS.get(model, model)
        marker = MODEL_MARKERS.get(model, "o")
        ls = MODEL_LINESTYLES.get(model, "-")

        means, stds, ns = [], [], []
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

        means, stds = np.array(means), np.array(stds)

        ax.plot(ns, means, color=color, marker=marker, label=label,
                linestyle=ls, linewidth=1.5, markersize=4, zorder=3)
        ax.fill_between(ns, means - stds, np.minimum(means + stds, 1.0),
                        color=color, alpha=0.12, zorder=2)

    if show_chance:
        ax.axhline(1 / 3, color="#BDBDBD", linestyle="--", linewidth=0.8,
                    zorder=1, label="Chance (1/3)")
    if show_symmetric_ceiling:
        ax.axhline(2 / 3, color="#BDBDBD", linestyle=":", linewidth=0.8,
                    zorder=1, label="Symmetric ceiling (2/3)")

    ax.set_xscale("log")
    ax.set_xticks(n_values)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("Training samples per class ($n$)")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0.15, 1.05)
    ax.legend(loc="lower right", framealpha=0.9, ncol=2)

    return fig, ax


def plot_sample_efficiency_multi(
    results_paths: dict[str, str | Path],
    *,
    models: list[str] | None = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (5.5, 3.5),
    metric: str = "f1",
) -> tuple[plt.Figure, plt.Axes]:
    """
    Overlay sample efficiency curves from multiple result files.

    results_paths: {model_name: path} — each file contributes one model's data.
    Useful when edge-ablation is in a separate results file.
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

    ax.axhline(1 / 3, color="#BDBDBD", linestyle="--", linewidth=0.8, zorder=1)
    ax.axhline(2 / 3, color="#BDBDBD", linestyle=":", linewidth=0.8, zorder=1)

    ax.set_xscale("log")
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("Training samples per class ($n$)")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0.15, 1.05)
    ax.legend(loc="lower right", framealpha=0.9, ncol=2)

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
    Shows distribution of per-SPI weights within each family.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    data = load_results(results_path)
    spi_names = data["spi_names"]
    spi_families_dict = data.get("spi_families", {})

    # Build index→family mapping from {family: [indices]} dict
    idx_to_family = {}
    for fam, indices in spi_families_dict.items():
        for idx in indices:
            idx_to_family[idx] = fam

    # Get learned w across seeds
    n_str = str(n_value)
    per_seed = data["results"][n_str]["models"]["spi-mpnn"]["per_seed"]
    w_matrix = np.array([s["learned_w"] for s in per_seed])  # (n_seeds, K)
    w_mean = np.abs(w_matrix).mean(axis=0)  # (K,)

    # Build dataframe for seaborn
    import pandas as pd
    records = []
    for k in range(len(spi_names)):
        records.append({
            "SPI": spi_names[k],
            "Family": idx_to_family.get(k, "other"),
            "|w|": w_mean[k],
        })
    df = pd.DataFrame(records)

    # Order families by total L2 norm
    family_l2 = df.groupby("Family")["|w|"].apply(
        lambda x: np.sqrt((x**2).sum())
    ).sort_values(ascending=False)
    family_order = family_l2.index.tolist()

    sns.violinplot(
        data=df, y="Family", x="|w|", hue="Family", order=family_order,
        hue_order=family_order,
        palette={f: FAMILY_COLORS.get(f, "#999") for f in family_order},
        inner="quart", linewidth=0.8, cut=0, ax=ax, orient="h", legend=False,
    )

    # Annotate top-k individual SPIs
    top_spis = df.nlargest(top_k_labels, "|w|")
    for _, row in top_spis.iterrows():
        family_idx = family_order.index(row["Family"])
        short_name = _short_spi_name(row["SPI"])
        ax.annotate(
            short_name, (row["|w|"], family_idx),
            textcoords="offset points", xytext=(5, 0),
            fontsize=5.5, fontstyle="italic", color="#333",
        )

    ax.set_xlabel("Mean $|w_k|$ across seeds")
    ax.set_ylabel("")
    ax.set_title(f"Learned weights by family ($n$={n_value})", fontsize=8)

    return fig, ax


def plot_family_bar(
    results_path: str | Path,
    n_value: int = 500,
    *,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (3.0, 2.0),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Simple horizontal bar chart: family L2 norm of w.
    Compact version for use inside Figure 1 Panel C.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    data = load_results(results_path)
    spi_names = data["spi_names"]
    spi_families_dict = data.get("spi_families", {})

    # Build index→family mapping
    idx_to_family = {}
    for fam, indices in spi_families_dict.items():
        for idx in indices:
            idx_to_family[idx] = fam

    n_str = str(n_value)
    per_seed = data["results"][n_str]["models"]["spi-mpnn"]["per_seed"]
    w_matrix = np.array([s["learned_w"] for s in per_seed])
    w_mean = np.abs(w_matrix).mean(axis=0)

    # Compute per-family L2
    family_l2 = {}
    for k in range(len(spi_names)):
        fam = idx_to_family.get(k, "other")
        family_l2.setdefault(fam, []).append(w_mean[k])
    family_l2 = {f: np.sqrt(np.sum(np.array(v) ** 2))
                 for f, v in family_l2.items()}

    # Sort
    sorted_fams = sorted(family_l2.items(), key=lambda x: x[1], reverse=True)
    names = [f[0] for f in sorted_fams]
    vals = [f[1] for f in sorted_fams]
    colors = [FAMILY_COLORS.get(n, "#999") for n in names]

    ax.barh(names, vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("$\\|\\mathbf{w}_g\\|_2$", fontsize=8)
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
) -> tuple[plt.Figure, plt.Axes]:
    """
    Strip/swarm plot of per-seed F1 for selected models at a given n.
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

    sns.stripplot(
        data=df, x="Model", y="F1", hue="Model", palette=palette,
        jitter=0.15, size=5, alpha=0.8, ax=ax, legend=False,
    )
    ax.axhline(1 / 3, color="#BDBDBD", linestyle="--", linewidth=0.7)
    ax.set_ylabel("Per-seed Macro F1")
    ax.set_xlabel("")
    ax.set_title(f"Per-seed performance ($n$={n_value})", fontsize=8)

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
    # Generic shortening
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
    results_main = Path("results/sample_efficiency_definitive_w60_cpu_results.json")
    results_edge = Path("results/sample_efficiency_edge_ablation_cpu_results.json")

    # Example instance for SPI heatmaps
    instance = data_dir / "var-chain" / "M10_T500_I0"
    npz_path = instance / "spi_mpis.npz"
    ts_path = instance / "timeseries.npy"

    do_fig1 = not args.fig2_only
    do_fig2 = not args.fig1_only

    if do_fig1:
        print("=== Figure 1 components ===")

        # Panel A: motif graphs
        fig, _ = plot_motif_graphs_row(M=10)
        fig.savefig(outdir / "fig1a_motif_graphs.pdf")
        print(f"  Saved {outdir}/fig1a_motif_graphs.pdf")
        plt.close(fig)

        # Panel A: MTS heatmaps (one per class)
        for cls in ["var-chain", "var-fork", "var-collider"]:
            ts = data_dir / cls / "M10_T500_I0" / "timeseries.npy"
            if ts.exists():
                fig, _ = plot_mts_heatmap(ts, title=cls.replace("var-", ""))
                fig.savefig(outdir / f"fig1a_mts_{cls}.pdf")
                plt.close(fig)
                print(f"  Saved fig1a_mts_{cls}.pdf")

        # Panel B: SPI heatmap stack
        spis_to_show = [
            ("xcorr_mean_sig-True", "Cross-corr $\\bar{r}$"),
            ("cov_EmpiricalCovariance", "Covariance"),
            ("mi_gaussian", "MI (Gaussian)"),
            ("sgc_parametric_mean_fs-1_fmin-0-25_fmax-0-5_order-1", "SGC (0.25–0.5 Hz)"),
            ("gc_gaussian_k-1_kt-1_l-1_lt-1", "GC (Gaussian)"),
            ("te_kraskov_k-4", "TE (Kraskov)"),
        ]
        spi_names_show = [s[0] for s in spis_to_show]
        spi_titles = [s[1] for s in spis_to_show]

        # Check which exist
        with np.load(npz_path) as npz:
            available = set(npz.files)
        valid = [(n, t) for n, t in zip(spi_names_show, spi_titles) if n in available]
        if valid:
            fig, _ = plot_spi_stack(
                npz_path,
                [v[0] for v in valid],
                titles=[v[1] for v in valid],
            )
            fig.savefig(outdir / "fig1b_spi_stack.pdf")
            print(f"  Saved fig1b_spi_stack.pdf")
            plt.close(fig)

        # Panel B: chain vs fork comparison (Markov equivalence visual)
        for cls in ["var-chain", "var-fork"]:
            cls_npz = data_dir / cls / "M10_T500_I0" / "spi_mpis.npz"
            if cls_npz.exists():
                for spi, title in [("xcorr_mean_sig-True", "$\\bar{r}$ (symmetric)"),
                                   ("sgc_parametric_mean_fs-1_fmin-0-25_fmax-0-5_order-1", "SGC (directed)")]:
                    if spi in available:
                        fig, _ = plot_spi_heatmap(
                            cls_npz, spi,
                            title=f"{cls.replace('var-', '')} — {title}",
                        )
                        safe = spi.replace("-", "_")[:20]
                        fig.savefig(outdir / f"fig1b_{cls}_{safe}.pdf")
                        plt.close(fig)

        print("  Saved chain/fork comparison heatmaps")

        # Panel C: family bar chart (compact, for inset)
        if results_main.exists():
            fig, _ = plot_family_bar(results_main, n_value=500)
            fig.savefig(outdir / "fig1c_family_bar.pdf")
            print(f"  Saved fig1c_family_bar.pdf")
            plt.close(fig)

    if do_fig2:
        print("=== Figure 2 components ===")

        if results_main.exists():
            # Panel A: sample efficiency curves (main models)
            fig, ax = plot_sample_efficiency(
                results_main,
                models=["spi-mpnn", "fixed-spi", "mlp-mix",
                        "correlation", "latent", "shuffled", "node-only"],
            )
            fig.savefig(outdir / "fig2a_sample_efficiency.pdf")
            print(f"  Saved fig2a_sample_efficiency.pdf")
            plt.close(fig)

            # With edge-ablation overlaid
            if results_edge.exists():
                fig, ax = plot_sample_efficiency(
                    results_main,
                    models=["spi-mpnn", "fixed-spi", "mlp-mix",
                            "correlation", "latent", "shuffled", "node-only"],
                )
                # Overlay edge-ablation from separate file
                edge_data = load_results(results_edge)
                ns_e, means_e, stds_e = [], [], []
                for n_str, v in edge_data["results"].items():
                    m = v["models"].get("edge-ablation")
                    if m:
                        ns_e.append(int(n_str))
                        means_e.append(m["f1_mean"])
                        stds_e.append(m["f1_std"])
                if ns_e:
                    ns_e, means_e, stds_e = zip(*sorted(zip(ns_e, means_e, stds_e)))
                    means_e, stds_e = np.array(means_e), np.array(stds_e)
                    ax.plot(ns_e, means_e, color=MODEL_COLORS["edge-ablation"],
                            marker="D", label=MODEL_LABELS["edge-ablation"],
                            linestyle="--", linewidth=1.5, markersize=4, zorder=3)
                    ax.fill_between(ns_e, means_e - stds_e,
                                    np.minimum(means_e + stds_e, 1.0),
                                    color=MODEL_COLORS["edge-ablation"],
                                    alpha=0.12, zorder=2)
                    ax.legend(loc="lower right", framealpha=0.9, ncol=2)

                fig.savefig(outdir / "fig2a_sample_efficiency_with_edge_abl.pdf")
                print(f"  Saved fig2a_sample_efficiency_with_edge_abl.pdf")
                plt.close(fig)

            # Panel B: family weights violin
            fig, _ = plot_family_weights(results_main, n_value=500)
            fig.savefig(outdir / "fig2b_family_violin.pdf")
            print(f"  Saved fig2b_family_violin.pdf")
            plt.close(fig)

            # Panel C: per-seed strip
            fig, _ = plot_per_seed_strip(
                results_main, n_value=500,
                models=["spi-mpnn", "fixed-spi", "mlp-mix", "correlation", "latent"],
            )
            fig.savefig(outdir / "fig2c_per_seed_strip.pdf")
            print(f"  Saved fig2c_per_seed_strip.pdf")
            plt.close(fig)

    print(f"\nAll outputs in {outdir}/")


if __name__ == "__main__":
    main()
