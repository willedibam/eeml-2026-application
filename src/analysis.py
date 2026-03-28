"""
Post-hoc analysis and figure generation.

Reads JSON results files from run_pipeline.py and produces figures.

Usage
-----
# Sample efficiency results (primary)
python -m src.analysis results/sample_efficiency_results.json

# Standard results
python -m src.analysis results/standard_results.json

# Directory of JSONs
python -m src.analysis results/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .utils import project_root

# Consistent colours and display names across all figures
MODEL_COLORS = {
    "spi-mpnn":      "#1565C0",   # deep blue
    "correlation":   "#E65100",   # deep orange
    "latent":        "#6A1B9A",   # deep purple
    "fixed-spi":     "#2E7D32",   # deep green
    "mlp-mix":       "#00838F",   # teal
    "node-only":     "#4E342E",   # brown
    "edge-ablation": "#546E7A",   # blue-grey
    "shuffled":      "#B71C1C",   # dark red
}
MODEL_LABELS = {
    "spi-mpnn":      "SPI-Graph (ours)",
    "correlation":   "Correlation",
    "latent":        "Latent",
    "fixed-spi":     "Fixed+SPI",
    "mlp-mix":       "MLP-Mix",
    "node-only":     "Node-Only MLP",
    "edge-ablation": "Edge Ablation",
    "shuffled":      "Shuffled",
}
MAIN_MODELS = ["spi-mpnn", "correlation", "latent"]


def _label(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def _color(name: str) -> str:
    return MODEL_COLORS.get(name, "black")


# ---------------------------------------------------------------------------
# Sample efficiency plot (primary figure)
# ---------------------------------------------------------------------------

def plot_sample_efficiency(
    results: dict,
    out_dir: Path,
    metric: str = "acc",
    models: list[str] | None = None,
    tag: str = "",
) -> None:
    """
    Accuracy or F1 vs log(n_train) for all models.

    This is the make-or-break figure for the abstract.
    Produces two versions: all models, and main-models-only.
    """
    se = results.get("results", {})
    if not se:
        print("[WARN] No sample_efficiency results to plot")
        return

    n_values = sorted([int(n) for n in se.keys()])
    if not n_values:
        return

    all_model_names = list(se[str(n_values[0])]["models"].keys())
    if models:
        model_names = [m for m in models if m in all_model_names]
    else:
        model_names = all_model_names

    key_mean = f"{metric}_mean"
    key_std = f"{metric}_std"
    ylabel = "Test Accuracy" if metric == "acc" else "Test Macro-F1"

    for subset, fname_suffix in [(model_names, "all"), (MAIN_MODELS, "main")]:
        subset = [m for m in subset if m in all_model_names]
        if not subset:
            continue

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for model in subset:
            means, stds = [], []
            for n in n_values:
                model_data = se[str(n)]["models"].get(model, {})
                means.append(model_data.get(key_mean, float("nan")))
                stds.append(model_data.get(key_std, 0.0))

            means = np.array(means)
            stds = np.array(stds)
            color = _color(model)
            ax.plot(
                n_values, means,
                label=_label(model), color=color,
                marker="o", linewidth=2, markersize=5,
            )
            ax.fill_between(
                n_values, means - stds, means + stds,
                alpha=0.15, color=color,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Training samples per class", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(n_values)
        ax.set_xticklabels([str(n) for n in n_values])
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3, which="both")
        ax.set_ylim(bottom=0)

        classes = results.get("classes", [])
        n_classes = len(classes)
        if n_classes > 0:
            ax.axhline(1.0 / n_classes, color="grey", linestyle=":", linewidth=1,
                       label="Chance")

        title = f"Sample efficiency — {', '.join(classes)}"
        ax.set_title(title, fontsize=11)
        plt.tight_layout()

        tag_str = f"_{tag}" if tag else ""
        path = out_dir / f"sample_efficiency_{metric}_{fname_suffix}{tag_str}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# SPI weight inspection (Experiment 4)
# ---------------------------------------------------------------------------

def plot_weight_inspection(results: dict, out_dir: Path) -> None:
    """
    Bar chart of learned spi_w grouped by SPI family, for spi-mpnn.

    Works with both standard and sample-efficiency results.
    For SE results, uses the largest n_train (most stable weights).
    """
    spi_names = results.get("spi_names", [])
    spi_families = results.get("spi_families", {})
    if not spi_names:
        return

    # Recover per-seed weights
    if results.get("mode") == "sample_efficiency":
        # Use largest n_train
        se = results.get("results", {})
        if not se:
            return
        max_n = str(max(int(n) for n in se.keys()))
        model_data = se[max_n]["models"].get("spi-mpnn")
        label_suffix = f"n_train={max_n}"
    else:
        model_data = results.get("models", {}).get("spi-mpnn")
        label_suffix = f"n={results.get('n_train', '?')}"

    if not model_data:
        return

    seeds = model_data.get("per_seed", [])
    w_list = [s["learned_w"] for s in seeds if "learned_w" in s]
    if not w_list:
        print("[WARN] No learned_w found in spi-mpnn results")
        return

    w_matrix = np.array(w_list)  # (n_seeds, K)
    mean_w = w_matrix.mean(axis=0)
    std_w = w_matrix.std(axis=0)
    K = len(mean_w)

    # Assign family colours
    family_color_map = {
        "linear":      "#1565C0",
        "rank":        "#43A047",
        "spectral":    "#F4511E",
        "causal":      "#8E24AA",
        "information": "#00838F",
        "distance":    "#6D4C41",
        "other":       "#757575",
    }
    name_to_family = {}
    for family, indices in spi_families.items():
        for idx in indices:
            if idx < len(spi_names):
                name_to_family[spi_names[idx]] = family

    colors = [family_color_map.get(name_to_family.get(n, "other"), "#757575")
              for n in spi_names]

    # Sort by |mean_w|
    order = np.argsort(np.abs(mean_w))[::-1]

    fig, ax = plt.subplots(figsize=(9, max(4, K * 0.28)))
    y = np.arange(K)
    ax.barh(y, mean_w[order], xerr=std_w[order],
            color=[colors[i] for i in order], alpha=0.85,
            capsize=2, edgecolor="none", height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([spi_names[i] for i in order], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Learned weight w", fontsize=11)
    ax.set_title(
        f"SPI-Graph learned weights ({len(w_list)} seeds, {label_suffix})",
        fontsize=11
    )
    ax.invert_yaxis()

    # Family legend
    used_families = set(name_to_family.values())
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=family_color_map.get(f, "#757575"), alpha=0.85)
        for f in used_families
    ]
    ax.legend(handles, list(used_families), fontsize=8, loc="lower right",
              title="Family", title_fontsize=8)

    plt.tight_layout()
    path = out_dir / "weight_inspection.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_family_norms(results: dict, out_dir: Path) -> None:
    """Group-level ||w_g||_2 per SPI family, for spi-mpnn."""
    spi_names = results.get("spi_names", [])
    spi_families = results.get("spi_families", {})
    if not spi_names or not spi_families:
        return

    if results.get("mode") == "sample_efficiency":
        se = results.get("results", {})
        if not se:
            return
        max_n = str(max(int(n) for n in se.keys()))
        model_data = se[max_n]["models"].get("spi-mpnn")
    else:
        model_data = results.get("models", {}).get("spi-mpnn")

    if not model_data:
        return

    seeds = model_data.get("per_seed", [])
    w_list = [s["learned_w"] for s in seeds if "learned_w" in s]
    if not w_list:
        return

    w_matrix = np.array(w_list)
    fam_names = list(spi_families.keys())
    fam_norms = np.array([
        [np.linalg.norm(w_matrix[s, spi_families[f]]) for f in fam_names]
        for s in range(w_matrix.shape[0])
    ])
    mean_norms = fam_norms.mean(axis=0)
    std_norms = fam_norms.std(axis=0)
    order = np.argsort(mean_norms)[::-1]

    fig, ax = plt.subplots(figsize=(7, max(3, len(fam_names) * 0.55)))
    y = np.arange(len(fam_names))
    ax.barh(y, mean_norms[order], xerr=std_norms[order],
            color="#1565C0", alpha=0.8, capsize=3, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels([fam_names[i] for i in order], fontsize=10)
    ax.set_xlabel("||w_g||₂", fontsize=11)
    ax.set_title(f"Family importance ({w_matrix.shape[0]} seeds)", fontsize=11)
    ax.invert_yaxis()
    plt.tight_layout()
    path = out_dir / "family_norms.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Model comparison bar chart (for standard mode)
# ---------------------------------------------------------------------------

def plot_model_comparison(results: dict, out_dir: Path, metric: str = "acc") -> None:
    """Bar chart of test accuracy/F1 across models (standard mode)."""
    models_data = results.get("models", {})
    if not models_data:
        return

    key_mean = f"{metric}_mean"
    key_std = f"{metric}_std"
    ylabel = "Test Accuracy" if metric == "acc" else "Test Macro-F1"

    names = list(models_data.keys())
    means = [models_data[n].get(key_mean, 0.0) for n in names]
    stds = [models_data[n].get(key_std, 0.0) for n in names]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 4))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=4,
                  color=[_color(n) for n in names], alpha=0.85, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels([_label(n) for n in names], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, 1.05)
    classes = results.get("classes", [])
    ax.set_title(f"Model comparison — {', '.join(classes)}", fontsize=11)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.01,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = out_dir / f"model_comparison_{metric}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Training curves (standard mode, seed 0)
# ---------------------------------------------------------------------------

def plot_training_curves(results: dict, out_dir: Path) -> None:
    models_data = results.get("models", {})
    if not models_data:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for name, mdata in models_data.items():
        seeds = mdata.get("per_seed", [])
        if not seeds:
            continue
        s0 = seeds[0]
        losses = s0.get("train_losses", [])
        val_f1s = s0.get("val_f1s", [])
        if losses:
            ax1.plot(losses, label=_label(name), color=_color(name), alpha=0.85)
        if val_f1s:
            ax2.plot(val_f1s, label=_label(name), color=_color(name), alpha=0.85)

    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train loss")
    ax1.set_title("Training loss (seed 0)"); ax1.legend(fontsize=7)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val macro-F1")
    ax2.set_title("Validation F1 (seed 0)"); ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=7)

    plt.tight_layout()
    path = out_dir / "training_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _analyze(results_path: Path, out_dir: Path) -> None:
    with results_path.open() as f:
        results = json.load(f)

    out_dir.mkdir(parents=True, exist_ok=True)
    mode = results.get("mode", "standard")
    print(f"\nAnalyzing {results_path.name} (mode={mode})...")

    if mode == "sample_efficiency":
        plot_sample_efficiency(results, out_dir, metric="acc")
        plot_sample_efficiency(results, out_dir, metric="f1")
        plot_weight_inspection(results, out_dir)
        plot_family_norms(results, out_dir)
    else:
        plot_model_comparison(results, out_dir, metric="acc")
        plot_model_comparison(results, out_dir, metric="f1")
        plot_training_curves(results, out_dir)
        plot_weight_inspection(results, out_dir)
        plot_family_norms(results, out_dir)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Analyze EEML results")
    p.add_argument("path", help="Results JSON or directory of JSONs")
    p.add_argument("--out-dir", help="Output directory for figures")
    args = p.parse_args(argv)

    path = Path(args.path)
    if not path.is_absolute():
        path = project_root() / path

    if path.is_dir():
        jsons = sorted(path.glob("*_results.json"))
        if not jsons:
            print(f"No *_results.json in {path}"); sys.exit(1)
        for j in jsons:
            out = Path(args.out_dir) if args.out_dir else j.parent / "figures"
            _analyze(j, out)
    elif path.is_file():
        out = Path(args.out_dir) if args.out_dir else path.parent / "figures"
        _analyze(path, out)
    else:
        print(f"Not found: {path}"); sys.exit(1)


if __name__ == "__main__":
    main()
