"""
EEML pipeline: SPI data → graph construction → train MPNN → evaluate.

Two modes
---------
standard          Single stratified split. Quick validation.
sample-efficiency Variable training set sizes with fixed val/test.
                  This is the primary experimental mode for the abstract.

Usage (sample-efficiency, primary)
-----------------------------------
python -m src.run_pipeline \\
    --data-dir data/topology/ \\
    --class-names chain star ring random \\
    --mode sample-efficiency \\
    --n-train 20 50 100 200 500 1000 \\
    --test-per-class 200 \\
    --val-per-class 100 \\
    --seeds 10 \\
    --models spi-mpnn correlation latent node-only \\
    --device cuda

Usage (standard, pilot)
------------------------
python -m src.run_pipeline \\
    --data-dir data/topology/ \\
    --class-names chain star ring random \\
    --mode standard \\
    --seeds 3 \\
    --max-epochs 50 \\
    --device cpu

Available models
----------------
spi-mpnn       Proposed: learned linear adj from K SPI descriptors
correlation    Baseline: fixed adj from Pearson |r_ij|
latent         Baseline: per-sample adj from learned node embeddings
fixed-spi      Ablation: fully connected + SPI edge features
mlp-mix        Ablation: nonlinear adj from MLP(SPI)
node-only      Ablation: no graph, MLP over node features
edge-ablation  Ablation: SPI adj + zero edge features
shuffled       Control: SPI adj + randomly shuffled SPI edge vectors

Note: nested sampling in sample-efficiency mode
------------------------------------------------
The training pool is permuted once with a fixed seed. For each n_train,
the first n_train samples from that permutation are used. This means:
    n_train=50 subset ⊂ n_train=100 subset ⊂ n_train=200 subset ⊂ ...
Differences across n values therefore reflect true sample efficiency,
not variance from which samples happened to be selected.
The scaler is fit on the training subset for each n_train independently.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

from .features import node_features
from .graph_build import (
    SPIScaler,
    assign_spi_families,
    build_graph,
    filter_spi_dimensions,
    load_spi_names,
    load_spi_tensor,
)
from .model import (
    CorrelationMPNN,
    EdgeAblationMPNN,
    FixedSPIMPNN,
    LatentGraphMPNN,
    MLPMixMPNN,
    NodeOnlyMLP,
    ShuffledEdgeMPNN,
    SingleSPIMPNN,
    SPIEdgeMPNN,
    SubsetSPIMPNN,
)
from .train import TrainConfig, TrainResult, train_model
from .utils import dump_json, load_json, project_root, to_relative

# Fixed seeds for data splitting — never change these mid-experiment
_SPLIT_SEED = 42   # stratified split (standard mode)
_POOL_SEED = 42    # pool permutation (sample-efficiency mode)

ALL_MODELS = [
    "spi-mpnn",
    "correlation",
    "latent",
    "fixed-spi",
    "mlp-mix",
    "node-only",
    "edge-ablation",
    "shuffled",
    "single-spi",
    "subset-spi",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EEML GNN pipeline")
    p.add_argument("--data-dir", required=True)
    p.add_argument(
        "--class-names", nargs="+", required=True,
        help="Class directory names under data-dir, in label order. "
             "E.g.: chain star ring random"
    )
    p.add_argument(
        "--mode", choices=["standard", "sample-efficiency"],
        default="sample-efficiency",
    )
    # Sample-efficiency mode
    p.add_argument(
        "--n-train", type=int, nargs="+",
        default=[20, 50, 100, 200, 500, 1000],
        help="Training set sizes per class (sample-efficiency mode)"
    )
    p.add_argument("--test-per-class", type=int, default=200)
    p.add_argument("--val-per-class", type=int, default=100)
    # Standard mode
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    # Models
    p.add_argument(
        "--models", nargs="+", default=None,
        help=f"Models to run. Default: all. Options: {ALL_MODELS}"
    )
    # Architecture
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--top-d", type=int, default=5)
    p.add_argument("--embed-dim", type=int, default=8,
                   help="Node embed dim for latent baseline (approx K/n_node_feat)")
    p.add_argument("--dropout", type=float, default=0.1)
    # Training
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--no-cosine-decay", action="store_true")
    # Regularisation
    p.add_argument("--l1-lambda", type=float, default=0.001)
    p.add_argument("--group-lambda", type=float, default=0.02)
    # Optimisation stability
    p.add_argument("--warmup-epochs", type=int, default=60,
                   help="Linear LR warmup epochs before cosine decay")
    p.add_argument("--w-lr-mult", type=float, default=1.0,
                   help="LR multiplier for spi_w/spi_b parameters")
    p.add_argument("--restarts", type=int, default=2,
                   help="Train N times per seed, keep best by val F1")
    # Output
    p.add_argument("--output-dir", default=None)
    p.add_argument("--tag", default="",
                   help="Optional tag appended to output filename")
    p.add_argument("--single-spi", default=None,
                   help="SPI name prefix for single-spi model (e.g. 'te_kraskov'). "
                        "Matches first retained SPI starting with this prefix.")
    p.add_argument("--subset-spi", nargs="+", default=None,
                   help="SPI name prefixes for subset-spi model (e.g. 'sgc_parametric_mean' "
                        "'gc_gaussian'). Each prefix matches the first retained SPI.")
    p.add_argument("--instance-range", type=int, nargs=2, default=None,
                   metavar=("START", "END"),
                   help="Only load instances with index in [START, END] (inclusive). "
                        "Useful for per-subject runs when subjects are blocked by index.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _extract_instance_index(path: Path) -> int | None:
    """Extract the integer instance index from a directory name like 'classfeet_I42'."""
    name = path.name
    if "_I" in name:
        try:
            return int(name.rsplit("_I", 1)[1])
        except ValueError:
            pass
    return None


def _discover_datasets(
    data_dir: Path,
    class_names: list[str],
    instance_range: tuple[int, int] | None = None,
) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for name in class_names:
        class_dir = data_dir / name
        if not class_dir.exists():
            print(f"[WARN] Class directory not found: {class_dir}")
            continue
        dirs = sorted(
            d for d in class_dir.iterdir()
            if d.is_dir()
            and (d / "spi_mpis.npz").exists()
            and (d / "timeseries.npy").exists()
            and (d / "meta.json").exists()
        )
        if instance_range is not None:
            lo, hi = instance_range
            dirs = [
                d for d in dirs
                if (idx := _extract_instance_index(d)) is not None
                and lo <= idx <= hi
            ]
        if dirs:
            result[name] = dirs
        else:
            print(f"[WARN] No completed samples in {class_dir}")
    return result


def _load_all_data(
    datasets_by_class: dict[str, list[Path]],
    spi_names: list[str],
    class_to_label: dict[str, int],
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[Path]]:
    spi_tensors, mts_arrays, labels, paths = [], [], [], []
    for class_name, dirs in datasets_by_class.items():
        label = class_to_label[class_name]
        for d in dirs:
            try:
                tensor = load_spi_tensor(d, spi_names)
                mts = np.load(d / "timeseries.npy").astype(np.float64)
                spi_tensors.append(tensor)
                mts_arrays.append(mts)
                labels.append(label)
                paths.append(d)
            except Exception as e:
                print(f"[WARN] Skipping {d}: {e}")
    return spi_tensors, mts_arrays, labels, paths


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def _find_spi_index(spi_names: list[str], prefix: str) -> int:
    """Find the index of the first SPI name starting with prefix."""
    for i, name in enumerate(spi_names):
        if name.startswith(prefix):
            return i
    raise ValueError(
        f"No SPI matching prefix '{prefix}'. "
        f"Available: {[n for n in spi_names if any(p in n for p in prefix.split('_')[:1])]}"
    )


def _make_model(
    name: str,
    K: int,
    F_n: int,
    n_classes: int,
    M: int,
    args: argparse.Namespace,
    spi_names: list[str] | None = None,
) -> torch.nn.Module:
    common = dict(
        n_node_features=F_n,
        n_classes=n_classes,
        hidden=args.hidden,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    if name == "spi-mpnn":
        return SPIEdgeMPNN(n_spi=K, top_d=args.top_d, **common)
    elif name == "correlation":
        return CorrelationMPNN(top_d=args.top_d, **common)
    elif name == "latent":
        return LatentGraphMPNN(top_d=args.top_d, embed_dim=args.embed_dim, **common)
    elif name == "fixed-spi":
        return FixedSPIMPNN(n_spi=K, **common)
    elif name == "mlp-mix":
        return MLPMixMPNN(n_spi=K, top_d=args.top_d, **common)
    elif name == "node-only":
        return NodeOnlyMLP(n_node_features=F_n, n_nodes=M, n_classes=n_classes,
                           hidden=args.hidden, dropout=args.dropout)
    elif name == "edge-ablation":
        return EdgeAblationMPNN(n_spi=K, top_d=args.top_d, **common)
    elif name == "shuffled":
        return ShuffledEdgeMPNN(n_spi=K, top_d=args.top_d, **common)
    elif name == "single-spi":
        if spi_names is None or args.single_spi is None:
            raise ValueError("single-spi model requires --single-spi prefix")
        idx = _find_spi_index(spi_names, args.single_spi)
        return SingleSPIMPNN(spi_index=idx, top_d=args.top_d, **common)
    elif name == "subset-spi":
        if spi_names is None or args.subset_spi is None:
            raise ValueError("subset-spi model requires --subset-spi prefixes")
        indices = [_find_spi_index(spi_names, p) for p in args.subset_spi]
        return SubsetSPIMPNN(spi_indices=indices, top_d=args.top_d, **common)
    else:
        raise ValueError(f"Unknown model '{name}'")


# ---------------------------------------------------------------------------
# Result serialisation helpers
# ---------------------------------------------------------------------------

def _result_to_dict(result: TrainResult) -> dict[str, Any]:
    d: dict[str, Any] = {
        "test_f1": result.test_f1,
        "test_acc": result.test_acc,
        "best_epoch": result.best_epoch,
        "best_val_f1": result.best_val_f1,
        "train_seconds": result.train_seconds,
    }
    if result.learned_w.size > 0:
        d["learned_w"] = result.learned_w.tolist()
        d["learned_b"] = result.learned_b
    if result.train_losses:
        d["train_losses"] = result.train_losses
    if result.val_f1s:
        d["val_f1s"] = result.val_f1s
    if result.restart_used > 0:
        d["restart_used"] = result.restart_used
    return d


def _aggregate_results(
    per_seed: list[TrainResult],
) -> dict[str, Any]:
    f1s = [r.test_f1 for r in per_seed]
    accs = [r.test_acc for r in per_seed]
    return {
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "per_seed": [_result_to_dict(r) for r in per_seed],
    }


# ---------------------------------------------------------------------------
# Core: build and split dataset
# ---------------------------------------------------------------------------

def _build_dataset(
    spi_tensors: list[np.ndarray],
    mts_arrays: list[np.ndarray],
    labels: list[int],
) -> list[Data]:
    return [
        build_graph(t, m, l)
        for t, m, l in zip(spi_tensors, mts_arrays, labels)
    ]


def _scale_tensors(
    spi_tensors: list[np.ndarray],
    train_indices: list[int],
    retained_indices: list[int],
) -> tuple[list[np.ndarray], SPIScaler]:
    """Subset to retained dims, fit scaler on train subset, apply to all."""
    tensors = [t[:, :, retained_indices] for t in spi_tensors]
    scaler = SPIScaler()
    scaler.fit([tensors[i] for i in train_indices])
    return [scaler.transform(t) for t in tensors], scaler


# ---------------------------------------------------------------------------
# Mode 1: Standard (single split)
# ---------------------------------------------------------------------------

def _run_standard(args: argparse.Namespace, data_dir: Path, output_dir: Path) -> None:
    class_names = args.class_names
    class_to_label = {c: i for i, c in enumerate(class_names)}
    n_classes = len(class_names)
    models_to_run = args.models or list(ALL_MODELS)

    print(f"\n{'='*70}")
    print(f"Mode: standard  |  Classes: {class_names}")
    print(f"{'='*70}")

    datasets_by_class = _discover_datasets(data_dir, class_names, args.instance_range)
    if not datasets_by_class:
        print("[ERROR] No data found"); return

    for name, dirs in datasets_by_class.items():
        print(f"  {name}: {len(dirs)} samples")

    first_dir = next(iter(datasets_by_class.values()))[0]
    all_spi_names = load_spi_names(first_dir)

    print("[STAGE] Loading data...")
    raw_spi, raw_mts, labels, _ = _load_all_data(
        datasets_by_class, all_spi_names, class_to_label
    )
    n = len(raw_spi)
    labels_arr = np.array(labels)

    # Stratified split
    idx = np.arange(n)
    train_idx, temp_idx = train_test_split(
        idx, test_size=1 - args.train_ratio, stratify=labels_arr,
        random_state=_SPLIT_SEED
    )
    val_frac = args.val_ratio / (1 - args.train_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=1 - val_frac, stratify=labels_arr[temp_idx],
        random_state=_SPLIT_SEED
    )

    print("[STAGE] Filtering and scaling SPIs...")
    retained_names, retained_indices = filter_spi_dimensions(
        all_spi_names, [raw_spi[i] for i in train_idx]
    )
    K = len(retained_names)
    if K == 0:
        print("[ERROR] All SPI dimensions dropped"); return

    scaled_spi, _ = _scale_tensors(raw_spi, list(train_idx), retained_indices)
    _, spi_family_indices = assign_spi_families(retained_names)

    print("[STAGE] Building PyG graphs...")
    dataset = _build_dataset(scaled_spi, raw_mts, labels)
    M = dataset[0].num_nodes
    F_n = dataset[0].x.shape[1]

    train_data = [dataset[i] for i in train_idx]
    val_data = [dataset[i] for i in val_idx]
    test_data = [dataset[i] for i in test_idx]
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    config = TrainConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=args.device,
        l1_lambda=args.l1_lambda,
        group_lambda=args.group_lambda,
        spi_family_indices=list(spi_family_indices.values()) if args.group_lambda > 0 else None,
        use_cosine_decay=not args.no_cosine_decay,
        warmup_epochs=args.warmup_epochs,
        w_lr_mult=args.w_lr_mult,
        restarts=args.restarts,
    )

    all_results: dict[str, list[TrainResult]] = defaultdict(list)
    for seed in range(args.seeds):
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        for model_name in models_to_run:
            print(f"  [{model_name}]")
            model = _make_model(model_name, K, F_n, n_classes, M, args, retained_names)
            result = train_model(model, train_data, val_data, test_data, config)
            all_results[model_name].append(result)

    summary: dict[str, Any] = {
        "mode": "standard",
        "classes": class_names,
        "n_samples": n,
        "n_spi": K,
        "spi_names": retained_names,
        "spi_families": {k: v for k, v in spi_family_indices.items()},
        "n_train": len(train_data),
        "n_val": len(val_data),
        "n_test": len(test_data),
        "n_nodes": M,
        "hyperparameters": {
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "l1_lambda": args.l1_lambda,
            "group_lambda": args.group_lambda,
            "warmup_epochs": args.warmup_epochs,
            "w_lr_mult": args.w_lr_mult,
            "restarts": args.restarts,
            "top_d": args.top_d,
            "no_cosine_decay": args.no_cosine_decay,
        },
        "models": {},
    }
    for model_name, results in all_results.items():
        agg = _aggregate_results(results)
        print(
            f"  {model_name:16s}  F1={agg['f1_mean']:.4f}±{agg['f1_std']:.4f}"
            f"  Acc={agg['acc_mean']:.4f}±{agg['acc_std']:.4f}"
        )
        summary["models"][model_name] = agg

    tag = f"_{args.tag}" if args.tag else ""
    out_path = output_dir / f"standard{tag}_results.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(out_path, summary)
    print(f"\n  Saved: {to_relative(out_path)}")


# ---------------------------------------------------------------------------
# Mode 2: Sample efficiency
# ---------------------------------------------------------------------------

def _run_sample_efficiency(
    args: argparse.Namespace, data_dir: Path, output_dir: Path
) -> None:
    class_names = args.class_names
    class_to_label = {c: i for i, c in enumerate(class_names)}
    n_classes = len(class_names)
    models_to_run = args.models or list(ALL_MODELS)
    n_train_values = sorted(args.n_train)
    n_val = args.val_per_class
    n_test = args.test_per_class

    print(f"\n{'='*70}")
    print(f"Mode: sample-efficiency  |  Classes: {class_names}")
    print(f"n_train values: {n_train_values}  |  val/class: {n_val}  |  test/class: {n_test}")
    print(f"{'='*70}")

    datasets_by_class = _discover_datasets(data_dir, class_names, args.instance_range)
    if not datasets_by_class:
        print("[ERROR] No data found"); return

    n_available = {name: len(dirs) for name, dirs in datasets_by_class.items()}
    for name, cnt in n_available.items():
        needed = max(n_train_values) + n_val + n_test
        print(f"  {name}: {cnt} available (need {needed})")
        if cnt < needed:
            print(f"  [WARN] Insufficient data for {name}: {cnt} < {needed}")

    first_dir = next(iter(datasets_by_class.values()))[0]
    all_spi_names = load_spi_names(first_dir)

    print("[STAGE] Loading all data...")
    raw_spi, raw_mts, labels, _ = _load_all_data(
        datasets_by_class, all_spi_names, class_to_label
    )
    n = len(raw_spi)
    labels_arr = np.array(labels)

    # Fixed test and val sets (stratified, same across all n_train values)
    # Use remaining samples as training pool
    needed_fixed = (n_val + n_test) * n_classes
    if n < needed_fixed + n_classes:
        print("[ERROR] Insufficient total samples"); return

    # First carve out fixed test set
    idx = np.arange(n)
    pool_idx, test_idx = train_test_split(
        idx,
        test_size=n_test * n_classes,
        stratify=labels_arr,
        random_state=_POOL_SEED,
    )
    # Then carve out fixed val set from the pool
    pool_idx, val_idx = train_test_split(
        pool_idx,
        test_size=n_val * n_classes,
        stratify=labels_arr[pool_idx],
        random_state=_POOL_SEED,
    )

    print(f"  Fixed test: {len(test_idx)}, Fixed val: {len(val_idx)}, Pool: {len(pool_idx)}")

    # Create per-class nested permutation of training pool
    # This ensures n_train=50 subset ⊆ n_train=100 subset ⊆ ...
    rng = np.random.default_rng(_POOL_SEED)
    pool_by_class: dict[str, np.ndarray] = {}
    for class_name in class_names:
        class_label = class_to_label[class_name]
        class_pool = pool_idx[labels_arr[pool_idx] == class_label]
        pool_by_class[class_name] = rng.permutation(class_pool)

    max_pool = min(len(v) for v in pool_by_class.values())
    max_n = max(n for n in n_train_values if n <= max_pool)
    n_train_values = [n for n in n_train_values if n <= max_pool]
    print(f"  Effective n_train values: {n_train_values} (pool limit: {max_pool}/class)")

    # Pre-filter SPI dimensions using the largest possible training set
    # (conservative: if a dim survives at max n, it survives at smaller n too)
    largest_train_idx = np.concatenate([
        pool_by_class[c][:max_n] for c in class_names
    ])
    print("[STAGE] Filtering SPI dimensions on largest training pool...")
    retained_names, retained_indices = filter_spi_dimensions(
        all_spi_names, [raw_spi[i] for i in largest_train_idx]
    )
    K = len(retained_names)
    if K == 0:
        print("[ERROR] All SPI dimensions dropped"); return

    _, spi_family_indices = assign_spi_families(retained_names)

    # Build unscaled graph dataset (scaling happens per-n below)
    raw_spi_retained = [t[:, :, retained_indices] for t in raw_spi]

    val_data_unscaled: list[tuple[np.ndarray, np.ndarray, int]] = [
        (raw_spi_retained[i], raw_mts[i], labels[i]) for i in val_idx
    ]
    test_data_unscaled: list[tuple[np.ndarray, np.ndarray, int]] = [
        (raw_spi_retained[i], raw_mts[i], labels[i]) for i in test_idx
    ]

    # First sample from dataset to get M and F_n
    _dummy_scaler = SPIScaler()
    _dummy_scaler.fit([raw_spi_retained[i] for i in largest_train_idx])
    _sample = build_graph(
        _dummy_scaler.transform(raw_spi_retained[val_idx[0]]),
        raw_mts[val_idx[0]],
        labels[val_idx[0]],
    )
    M = _sample.num_nodes
    F_n = _sample.x.shape[1]

    config = TrainConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=args.device,
        l1_lambda=args.l1_lambda,
        group_lambda=args.group_lambda,
        spi_family_indices=list(spi_family_indices.values()) if args.group_lambda > 0 else None,
        use_cosine_decay=not args.no_cosine_decay,
        warmup_epochs=args.warmup_epochs,
        w_lr_mult=args.w_lr_mult,
        restarts=args.restarts,
    )

    results_by_n: dict[str, dict[str, Any]] = {}

    for n_train in n_train_values:
        print(f"\n{'─'*60}")
        print(f"n_train = {n_train} per class ({n_train * n_classes} total)")
        print(f"{'─'*60}")

        # Nested training indices
        train_idx_n = np.concatenate([
            pool_by_class[c][:n_train] for c in class_names
        ])

        # Fit scaler on THIS training subset (never on val/test)
        scaler = SPIScaler()
        scaler.fit([raw_spi_retained[i] for i in train_idx_n])

        def _scale_and_build(
            pairs: list[tuple[np.ndarray, np.ndarray, int]]
        ) -> list[Data]:
            return [build_graph(scaler.transform(s), m, l) for s, m, l in pairs]

        train_data = _scale_and_build([
            (raw_spi_retained[i], raw_mts[i], labels[i]) for i in train_idx_n
        ])
        val_data = _scale_and_build(val_data_unscaled)
        test_data = _scale_and_build(test_data_unscaled)

        model_results: dict[str, list[TrainResult]] = defaultdict(list)

        for seed in range(args.seeds):
            print(f"\n  -- Seed {seed} --")
            torch.manual_seed(seed)
            np.random.seed(seed)

            for model_name in models_to_run:
                print(f"    [{model_name}]")
                model = _make_model(model_name, K, F_n, n_classes, M, args, retained_names)
                result = train_model(model, train_data, val_data, test_data, config)
                model_results[model_name].append(result)

        n_summary: dict[str, Any] = {
            "n_train_total": n_train * n_classes,
            "models": {},
        }
        for model_name, results in model_results.items():
            agg = _aggregate_results(results)
            n_summary["models"][model_name] = agg
            print(
                f"  {model_name:16s}  F1={agg['f1_mean']:.4f}±{agg['f1_std']:.4f}"
                f"  Acc={agg['acc_mean']:.4f}±{agg['acc_std']:.4f}"
            )

        results_by_n[str(n_train)] = n_summary

    summary: dict[str, Any] = {
        "mode": "sample_efficiency",
        "classes": class_names,
        "n_spi": K,
        "spi_names": retained_names,
        "spi_families": {k: v for k, v in spi_family_indices.items()},
        "n_val_per_class": n_val,
        "n_test_per_class": n_test,
        "n_nodes": M,
        "n_node_features": F_n,
        "models_run": models_to_run,
        "seeds": args.seeds,
        "hyperparameters": {
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "l1_lambda": args.l1_lambda,
            "group_lambda": args.group_lambda,
            "warmup_epochs": args.warmup_epochs,
            "w_lr_mult": args.w_lr_mult,
            "restarts": args.restarts,
            "top_d": args.top_d,
            "no_cosine_decay": args.no_cosine_decay,
        },
        "results": results_by_n,
    }

    tag = f"_{args.tag}" if args.tag else ""
    out_path = output_dir / f"sample_efficiency{tag}_results.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(out_path, summary)
    print(f"\n  Saved: {to_relative(out_path)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Validate models
    if args.models:
        for m in args.models:
            if m not in ALL_MODELS:
                print(f"[ERROR] Unknown model '{m}'. Available: {ALL_MODELS}")
                sys.exit(1)

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = project_root() / data_dir

    output_dir = Path(args.output_dir) if args.output_dir else project_root() / "results"
    if not output_dir.is_absolute():
        output_dir = project_root() / output_dir

    print(f"Data directory : {to_relative(data_dir)}")
    print(f"Output directory: {to_relative(output_dir)}")
    print(f"Classes        : {args.class_names}")
    print(f"Mode           : {args.mode}")
    print(f"Models         : {args.models or ALL_MODELS}")
    print(f"Seeds          : {args.seeds}")
    print(f"Device         : {args.device}")

    if args.mode == "standard":
        _run_standard(args, data_dir, output_dir)
    else:
        _run_sample_efficiency(args, data_dir, output_dir)


if __name__ == "__main__":
    main()
