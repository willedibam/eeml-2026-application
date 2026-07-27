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
    """Size-normalised family importance (||w_g||_2 / sqrt(|g|)) for spi-mpnn.

    Uses RMS weight magnitude per family, NOT the raw group L2 norm. Raw L2
    grows with family size, so a bar chart of ||w_g||_2 would rank the
    largest family (causal, 48 SPIs) top by construction. RMS reports
    per-SPI magnitude and is comparable across unequally sized families.
    """
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
        [
            np.linalg.norm(w_matrix[s, spi_families[f]])
            / max(len(spi_families[f]), 1) ** 0.5
            for f in fam_names
        ]
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
    ax.set_xlabel("||w_g||₂ / √|g|  (per-SPI RMS)", fontsize=11)
    ax.set_title(f"Family importance ({w_matrix.shape[0]} seeds)", fontsize=11)
    ax.invert_yaxis()
    plt.tight_layout()
    path = out_dir / "family_norms.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def _spi_mpnn_weight_matrix(results: dict) -> np.ndarray | None:
    """Extract the per-seed learned_w matrix (n_seeds, K) for spi-mpnn."""
    if results.get("mode") == "sample_efficiency":
        se = results.get("results", {})
        if not se:
            return None
        max_n = str(max(int(n) for n in se.keys()))
        model_data = se[max_n]["models"].get("spi-mpnn")
    else:
        model_data = results.get("models", {}).get("spi-mpnn")
    if not model_data:
        return None
    w_list = [s["learned_w"] for s in model_data.get("per_seed", []) if "learned_w" in s]
    if not w_list:
        return None
    return np.asarray(w_list)


def report_weight_stability(results: dict, top_k: int = 5) -> dict | None:
    """Cross-seed stability of the recovered signature (identifiability check).

    Collinear SPIs (e.g. the many GC/SGC variants in the 48-member causal
    family) make the *individual* top-weighted statistic unstable across
    seeds even when the *family* is stable. From learned_w alone this reports:
      - top-1 family agreement across seeds,
      - top-1 SPI agreement across seeds,
      - mean pairwise Jaccard overlap of the per-seed top-k SPI sets.

    High family agreement with low SPI agreement is the honest, expected
    signature. Within-family 'specificity' claims (TE>GC in R1, band
    selection in R2 of the multi-regime spec) should only be trusted where
    SPI-level agreement / Jaccard is high — otherwise the winning member is
    an arbitrary pick among collinear estimators.
    """
    spi_names = results.get("spi_names", [])
    spi_families = results.get("spi_families", {})
    w_matrix = _spi_mpnn_weight_matrix(results)
    if w_matrix is None or not spi_names:
        return None

    n_seeds = w_matrix.shape[0]
    abs_w = np.abs(w_matrix)

    # top-1 SPI per seed
    top1_spi = abs_w.argmax(axis=1)
    _, counts = np.unique(top1_spi, return_counts=True)
    spi_agree = counts.max() / n_seeds

    # top-1 family per seed (by per-SPI RMS)
    fam_names = list(spi_families.keys())
    fam_agree = float("nan")
    if fam_names:
        rms = np.stack([
            np.array([
                np.linalg.norm(w_matrix[s, spi_families[f]])
                / max(len(spi_families[f]), 1) ** 0.5
                for f in fam_names
            ])
            for s in range(n_seeds)
        ])
        top1_fam = rms.argmax(axis=1)
        _, fcounts = np.unique(top1_fam, return_counts=True)
        fam_agree = fcounts.max() / n_seeds

    # mean pairwise Jaccard of per-seed top-k SPI sets
    k = min(top_k, w_matrix.shape[1])
    topk_sets = [set(np.argsort(abs_w[s])[::-1][:k].tolist()) for s in range(n_seeds)]
    jac = []
    for a in range(n_seeds):
        for b in range(a + 1, n_seeds):
            inter = len(topk_sets[a] & topk_sets[b])
            union = len(topk_sets[a] | topk_sets[b])
            jac.append(inter / union if union else 0.0)
    mean_jaccard = float(np.mean(jac)) if jac else float("nan")

    modal_spi = spi_names[int(np.bincount(top1_spi).argmax())]
    report = {
        "n_seeds": int(n_seeds),
        "top1_spi_agreement": float(spi_agree),
        "top1_family_agreement": float(fam_agree),
        f"top{k}_mean_jaccard": mean_jaccard,
        "modal_top1_spi": modal_spi,
    }
    print(
        f"  [stability] seeds={n_seeds}  "
        f"top1-family={fam_agree:.2f}  top1-SPI={spi_agree:.2f}  "
        f"top{k}-Jaccard={mean_jaccard:.2f}  modal-SPI={modal_spi}"
    )
    if spi_agree < 0.5:
        print("  [stability] WARN: top SPI unstable across seeds (collinear family); "
              "treat within-family specificity claims as unidentified.")
    return report


def report_family_uncertainty(
    results: dict, n_boot: int = 2000, seed: int = 0
) -> dict | None:
    """Bootstrap CI on per-family importance and top-family selection confidence.

    The seeds are the sample; we resample them with replacement to put a
    calibrated interval on each family's mean per-SPI RMS and on the claim
    "family X is the top-weighted family". A recovered family with P(top) near
    1.0 and a CI separated from the runner-up is a defensible scientific
    statement; overlapping CIs mean the ranking is not resolved at this sample
    size. Complements report_weight_stability (which is member-level).
    """
    spi_families = results.get("spi_families", {})
    w = _spi_mpnn_weight_matrix(results)
    if w is None or not spi_families:
        return None

    fam = list(spi_families.keys())
    n_seeds = w.shape[0]
    rms = np.stack([
        np.array([
            np.linalg.norm(w[s, spi_families[f]]) / max(len(spi_families[f]), 1) ** 0.5
            for f in fam
        ])
        for s in range(n_seeds)
    ])  # (n_seeds, n_fam)

    rng = np.random.default_rng(seed)
    boot_means = np.empty((n_boot, len(fam)))
    top_counts = np.zeros(len(fam))
    for b in range(n_boot):
        idx = rng.integers(0, n_seeds, n_seeds)
        m = rms[idx].mean(axis=0)
        boot_means[b] = m
        top_counts[int(m.argmax())] += 1

    mean = rms.mean(axis=0)
    lo, hi = np.percentile(boot_means, [2.5, 97.5], axis=0)
    top_conf = top_counts / n_boot
    order = np.argsort(mean)[::-1]

    print(f"  [uncertainty] {n_seeds} seeds, {n_boot} bootstraps")
    for i in order:
        print(
            f"    {fam[i]:12s} RMS={mean[i]:.4f}  "
            f"95% CI [{lo[i]:.4f}, {hi[i]:.4f}]  P(top)={top_conf[i]:.2f}"
        )
    if len(order) >= 2 and hi[order[1]] >= lo[order[0]]:
        print("  [uncertainty] WARN: top-2 family CIs overlap; family ranking "
              "not resolved at this seed count.")
    return {
        "families": fam,
        "mean_rms": mean.tolist(),
        "ci_low": lo.tolist(),
        "ci_high": hi.tolist(),
        "p_top": top_conf.tolist(),
    }


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


def report_directed_enrichment(results: dict, n_perm: int = 5000,
                               seed: int = 0) -> dict | None:
    """How much of the learned |w| lands on DIRECTED statistics, vs a null.

    This is GSEA-style enrichment (Subramanian et al. 2005) applied to a ranked
    SPI list: take the set of directed statistics, measure the share of total
    |w| it captures, and calibrate against permuting w across SPIs.

    Why this is the readout to trust for motif tasks. Chain and fork are
    Markov-equivalent under any symmetric statistic (Verma & Pearl 1990), so
    directedness is *necessary*, not merely helpful -- the enrichment is the
    empirical shadow of that theorem. And unlike family-level L2, the set is
    MEASURED (spi_asymmetry, from the data) rather than hand-assigned, so the
    claim does not inherit the taxonomy's arbitrariness or its size confound.

    Measured on VAR (lambda_g=0.01, n=400, 5 seeds): 72% of |w| on the 33% of
    SPIs that are directed, 2.20x, z=6.8. The shuffled control -- pair
    correspondence destroyed -- gives 1.11x, z=1.0, confirming the enrichment
    tracks the task rather than the vocabulary or the optimiser.
    """
    asym = results.get("spi_asymmetry")
    W = _spi_mpnn_weight_matrix(results)
    if asym is None or W is None:
        return None
    asym = np.asarray(asym)
    directed = asym > 0.01
    if directed.sum() == 0 or directed.all():
        return None

    w = np.abs(W).mean(axis=0)
    base = directed.sum() / directed.size
    obs = w[directed].sum() / w.sum()

    rng = np.random.default_rng(seed)
    null = np.array([(w[rng.permutation(w.size)][directed]).sum() / w.sum()
                     for _ in range(n_perm)])
    z = (obs - null.mean()) / (null.std() + 1e-12)
    p = float((null >= obs).mean())

    print(f"  [directed] {directed.sum()}/{directed.size} SPIs directed "
          f"({100*base:.0f}% of vocabulary)")
    print(f"  [directed] |w| on directed: {100*obs:.0f}%  "
          f"enrichment {obs/base:.2f}x  z={z:.1f}  p={p:.4f}")
    if obs / base < 1.3:
        print("  [directed] WARN: little enrichment -- the model is not "
              "preferentially using directed statistics.")
    return {"n_directed": int(directed.sum()), "n_spi": int(directed.size),
            "frac_weight_directed": float(obs), "baseline": float(base),
            "enrichment": float(obs / base), "z": float(z), "p": p}


# ---------------------------------------------------------------------------
# Shard merging (cluster runs)
# ---------------------------------------------------------------------------

def merge_shards(paths: list[Path]) -> dict:
    """Merge per-seed result shards from a sharded cluster run into one result.

    docs/gadi/train.pbs shards the seed range across array tasks, so a 30-seed
    run lands as 30 files (`<tag>_s0`, `<tag>_s1`, ...) each holding one seed.
    Every plot and diagnostic here expects a single result with all seeds in
    `per_seed`, so concatenate them and recompute the aggregates.

    Shards must agree on the experiment: identical spi_names and
    hyperparameters apart from `seed_offset`. Mismatches raise rather than
    silently averaging across different experiments.
    """
    if not paths:
        raise ValueError("no shards to merge")

    shards = []
    for p in sorted(paths):
        with p.open() as f:
            shards.append((p, json.load(f)))

    base_path, base = shards[0]
    base = json.loads(json.dumps(base))          # deep copy; don't mutate input
    base_names = base.get("spi_names")
    base_hp = {k: v for k, v in base.get("hyperparameters", {}).items()
               if k != "seed_offset"}

    for p, s in shards[1:]:
        if s.get("spi_names") != base_names:
            raise ValueError(f"{p.name}: spi_names differ from {base_path.name}")
        hp = {k: v for k, v in s.get("hyperparameters", {}).items()
              if k != "seed_offset"}
        if hp != base_hp:
            diff = {k for k in set(hp) | set(base_hp) if hp.get(k) != base_hp.get(k)}
            raise ValueError(f"{p.name}: hyperparameters differ in {sorted(diff)}")

    def _merge_models(dst: dict, srcs: list[dict]) -> None:
        for model in dst:
            per = list(dst[model].get("per_seed", []))
            for s in srcs:
                per.extend(s.get(model, {}).get("per_seed", []))
            f1s = [r["test_f1"] for r in per]
            accs = [r["test_acc"] for r in per]
            dst[model].update({
                "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
                "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                "per_seed": per,
            })

    if base.get("mode") == "sample_efficiency":
        for n, blk in base.get("results", {}).items():
            _merge_models(blk["models"],
                          [s["results"][n]["models"] for _, s in shards[1:]])
    else:
        _merge_models(base["models"], [s["models"] for _, s in shards[1:]])

    n_seeds = len(next(iter(
        (base["results"][next(iter(base["results"]))]["models"]
         if base.get("mode") == "sample_efficiency" else base["models"])
        .values()))["per_seed"])
    base["seeds"] = n_seeds
    base["merged_from"] = [p.name for p, _ in shards]
    base.get("hyperparameters", {}).pop("seed_offset", None)
    print(f"  [merge] {len(shards)} shards -> {n_seeds} seeds")
    return base


def find_shards(directory: Path, tag: str) -> list[Path]:
    """Result shards for `tag`, i.e. <prefix>_<tag>_s<N>_results.json."""
    return sorted(directory.glob(f"*{tag}_s*_results.json"))


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
        report_weight_stability(results)
        report_family_uncertainty(results)
        report_directed_enrichment(results)
    else:
        plot_model_comparison(results, out_dir, metric="acc")
        plot_model_comparison(results, out_dir, metric="f1")
        plot_training_curves(results, out_dir)
        plot_weight_inspection(results, out_dir)
        plot_family_norms(results, out_dir)
        report_weight_stability(results)
        report_family_uncertainty(results)
        report_directed_enrichment(results)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Analyze EEML results")
    p.add_argument("path", help="Results JSON or directory of JSONs")
    p.add_argument("--out-dir", help="Output directory for figures")
    p.add_argument("--merge", metavar="TAG",
                   help="Merge sharded cluster results before analysing: "
                        "combines <dir>/*<TAG>_s*_results.json into one result "
                        "(docs/gadi/train.pbs writes one file per seed shard). "
                        "Writes <dir>/<TAG>_merged_results.json.")
    args = p.parse_args(argv)

    path = Path(args.path)
    if not path.is_absolute():
        path = project_root() / path

    if args.merge:
        if not path.is_dir():
            print("--merge expects a directory"); sys.exit(1)
        shards = find_shards(path, args.merge)
        if not shards:
            print(f"No shards matching *{args.merge}_s*_results.json in {path}")
            sys.exit(1)
        merged = merge_shards(shards)
        out_path = path / f"{args.merge}_merged_results.json"
        with out_path.open("w") as f:
            json.dump(merged, f, indent=2)
        print(f"  Saved {out_path.name}")
        out = Path(args.out_dir) if args.out_dir else path / "figures"
        _analyze(out_path, out)
        return

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
