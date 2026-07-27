"""Training loop for EEML MPNN pipeline."""
from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 200
    patience: int = 20
    grad_clip: float = 1.0
    l1_lambda: float = 0.0          # L1 on spi_w; 0 = disabled
    group_lambda: float = 0.0       # group lasso on SPI families; 0 = disabled
    spi_family_indices: list[list[int]] | None = None
    group_size_norm: bool = True    # scale each family penalty by sqrt(size)
    # Per-group sizes used by the sqrt() weighting. None = raw member count
    # (Yuan & Lin). Supply effective dimensions (graph_build.effective_group_dims)
    # to stop redundant groups being over-penalised; see that docstring.
    spi_family_sizes: list[float] | None = None
    use_cosine_decay: bool = True
    device: str = "cpu"
    # --- new options ---
    warmup_epochs: int = 0          # linear LR warmup; 0 = disabled
    w_lr_mult: float = 1.0          # LR multiplier for spi_w/spi_b params
    restarts: int = 1               # train N times, keep best (by val F1)


@dataclass
class TrainResult:
    best_val_f1: float = 0.0
    best_epoch: int = 0
    test_f1: float = 0.0
    test_acc: float = 0.0
    learned_w: np.ndarray = field(default_factory=lambda: np.array([]))
    learned_b: float = 0.0
    train_losses: list[float] = field(default_factory=list)
    val_f1s: list[float] = field(default_factory=list)
    train_seconds: float = 0.0
    restart_used: int = 0           # which restart produced best result


@torch.no_grad()
def _evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        labels.append(batch.y.cpu().numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels, preds)
    return float(f1), float(acc)


def _build_optimizer_and_scheduler(
    model: nn.Module, config: TrainConfig
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
    """Build optimizer, isolating spi_w/spi_b into their own param group.

    spi_w is the interpreted scientific output; its sparsity is imposed
    explicitly by the L1 + group-lasso terms in the loss. AdamW weight decay
    would add a second, unreported L2 shrinkage on the same parameter,
    confounding the recovered family signature. We therefore always put
    spi_w/spi_b in a group with weight_decay=0 (and apply the w_lr_mult LR
    there). All other parameters keep the configured weight decay.
    """
    if hasattr(model, "spi_w"):
        w_params = [model.spi_w, model.spi_b]
        w_param_ids = {id(p) for p in w_params}
        other_params = [p for p in model.parameters() if id(p) not in w_param_ids]
        param_groups = [
            {"params": other_params, "lr": config.lr,
             "weight_decay": config.weight_decay},
            {"params": w_params, "lr": config.lr * config.w_lr_mult,
             "weight_decay": 0.0},
        ]
    else:
        param_groups = [
            {"params": model.parameters(), "lr": config.lr,
             "weight_decay": config.weight_decay}
        ]

    optimizer = torch.optim.AdamW(param_groups)

    if not config.use_cosine_decay:
        return optimizer, None

    if config.warmup_epochs > 0:
        warmup_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: min(1.0, (epoch + 1) / config.warmup_epochs),
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=config.max_epochs - config.warmup_epochs
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config.warmup_epochs],
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=config.max_epochs)

    return optimizer, scheduler


def _train_single(
    model: nn.Module,
    train_data: list[Data],
    val_data: list[Data],
    test_data: list[Data],
    config: TrainConfig,
) -> TrainResult:
    """Single training run (one restart)."""
    device = torch.device(config.device)
    model = model.to(device)

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

    optimizer, scheduler = _build_optimizer_and_scheduler(model, config)
    criterion = nn.CrossEntropyLoss()

    result = TrainResult()
    best_state = None
    patience_counter = 0
    start_time = time.perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y)

            if config.l1_lambda > 0 and hasattr(model, "spi_w"):
                loss = loss + config.l1_lambda * model.spi_w.abs().sum()

            if (
                config.group_lambda > 0
                and config.spi_family_indices
                and hasattr(model, "spi_w")
            ):
                for gi, idx in enumerate(config.spi_family_indices):
                    penalty = model.spi_w[idx].norm(2)
                    if config.group_size_norm:
                        size = (config.spi_family_sizes[gi]
                                if config.spi_family_sizes else len(idx))
                        # Yuan & Lin (2006): scale each group's penalty by
                        # sqrt(|g|) so families are penalised per-unit rather
                        # than by raw member count. Without this, large families
                        # (causal: 48 SPIs) are under-penalised and dominate the
                        # learned signature purely by size — the confound behind
                        # the "causal family carries 3.5x the L2 norm" claim.
                        penalty = penalty * (size ** 0.5)
                    loss = loss + config.group_lambda * penalty

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            epoch_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        result.train_losses.append(avg_loss)

        val_f1, val_acc = _evaluate(model, val_loader, device)
        result.val_f1s.append(val_f1)

        if val_f1 > result.best_val_f1:
            result.best_val_f1 = val_f1
            result.best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"  [Epoch {epoch:3d}] loss={avg_loss:.4f}  "
                f"val_f1={val_f1:.4f}  val_acc={val_acc:.4f}"
            )

        # Never early-stop while the LR is still warming up. During warmup the
        # LR is only (epoch/warmup)*lr, so progress is slow and noisy; with
        # patience < warmup_epochs a run could halt before the LR ever reached
        # full value, leaving spi_w at ~0. That produced catastrophic seed
        # variance (e.g. one seed at F1 0.31 vs 0.96 for its siblings, best
        # epoch 8 of a 60-epoch warmup) and made stronger group-lambda settings
        # look worse than they are, since heavier regularisation slows early
        # progress and triggered the failure more often.
        if epoch > config.warmup_epochs and patience_counter >= config.patience:
            print(f"  Early stopping at epoch {epoch} (best={result.best_epoch})")
            break

    result.train_seconds = time.perf_counter() - start_time

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    test_f1, test_acc = _evaluate(model, test_loader, device)
    result.test_f1 = test_f1
    result.test_acc = test_acc

    if hasattr(model, "spi_w"):
        result.learned_w = model.spi_w.detach().cpu().numpy().copy()
        result.learned_b = float(model.spi_b.detach().cpu().item())

    print(
        f"  [Done] test_f1={test_f1:.4f}  test_acc={test_acc:.4f}  "
        f"best_epoch={result.best_epoch}  time={result.train_seconds:.1f}s"
    )
    return result


def stability_selection(
    make_model: Callable[[], nn.Module],
    train_data: list[Data],
    val_data: list[Data],
    config: TrainConfig,
    *,
    n_subsamples: int = 50,
    subsample_frac: float = 0.5,
    top_q: int = 10,
    seed: int = 0,
    min_val_f1: float | None = None,
) -> dict[str, Any]:
    """Stability selection for spi_w (Meinshausen & Buhlmann 2010).

    Why this exists. The L1 / group-lasso terms are added to the loss and
    differentiated, i.e. a *subgradient* method: spi_w is shrunk towards zero
    but never exactly zero, and Adam's per-parameter scaling further distorts
    the effective penalty. So "the top-weighted SPIs" is a statement about a
    soft point estimate of a weakly-identified parameter -- fragile ground for
    a scientific claim, especially with collinear SPIs where the winner among
    near-duplicates is close to arbitrary.

    Stability selection replaces the point estimate with a *frequency*: refit
    on many random subsamples of the training set and record how often each
    SPI lands in the top-q by |w|. An SPI selected in ~all subsamples is a
    robust finding; one selected half the time is not, however large its mean
    weight. This is the honest unit for the paper's interpretability claim,
    and it degrades gracefully under collinearity (a collinear pair simply
    splits its frequency).

    Only *converged* fits may vote. Meinshausen-Buhlmann assumes each
    subsample yields a valid fit; a run that failed to optimise has an
    essentially arbitrary spi_w, and letting it vote dilutes every frequency
    toward noise (observed: fits ranging val F1 0.55-0.997 in one batch
    produced a spuriously flat selection profile). `min_val_f1` drops
    non-converged fits; pass None to accept all (and inspect n_used).

    Returns per-SPI selection frequency, sign consistency, and the fits' val
    F1s (to confirm the subsampled fits are actually learning).
    """
    rng = np.random.default_rng(seed)
    n = len(train_data)
    n_sub = max(2, int(round(subsample_frac * n)))

    counts: np.ndarray | None = None
    sign_sum: np.ndarray | None = None
    val_f1s: list[float] = []
    n_used = 0

    # Subsampled fits are a diagnostic, not model selection: one restart and
    # no test evaluation keeps the cost proportionate.
    sub_config = copy.copy(config)
    sub_config.restarts = 1

    for b in range(n_subsamples):
        idx = rng.choice(n, size=n_sub, replace=False)
        subset = [train_data[i] for i in idx]

        torch.manual_seed(seed * 100003 + b)
        model = make_model()
        res = _train_single(model, subset, val_data, val_data, sub_config)
        w = res.learned_w
        if w.size == 0:
            raise ValueError("stability_selection requires a model with spi_w")

        if counts is None:
            counts = np.zeros(w.size)
            sign_sum = np.zeros(w.size)

        val_f1s.append(res.best_val_f1)
        if min_val_f1 is not None and res.best_val_f1 < min_val_f1:
            continue  # non-converged fit: its w is arbitrary, no vote

        q = min(top_q, w.size)
        top_idx = np.argsort(np.abs(w))[::-1][:q]
        counts[top_idx] += 1.0
        sign_sum[top_idx] += np.sign(w[top_idx])
        n_used += 1

    if n_used == 0:
        raise ValueError(
            f"No subsample fit reached min_val_f1={min_val_f1}; "
            f"val F1s ranged {min(val_f1s):.3f}-{max(val_f1s):.3f}. "
            "Lower the threshold or fix the training config."
        )

    freq = counts / n_used
    # Sign consistency among the fits that selected each SPI (1.0 = always
    # same direction). A high-frequency SPI with inconsistent sign is a
    # magnitude artifact, not a coherent effect.
    with np.errstate(invalid="ignore", divide="ignore"):
        sign_consistency = np.where(counts > 0, np.abs(sign_sum) / np.maximum(counts, 1), np.nan)

    return {
        "selection_frequency": freq.tolist(),
        "sign_consistency": sign_consistency.tolist(),
        "n_subsamples": n_subsamples,
        "n_used": n_used,                     # converged fits that voted
        "min_val_f1": min_val_f1,
        "subsample_frac": subsample_frac,
        "top_q": top_q,
        "val_f1_mean": float(np.mean(val_f1s)),
        "val_f1_std": float(np.std(val_f1s)),
        "val_f1_used_mean": float(np.mean([v for v in val_f1s
                                           if min_val_f1 is None or v >= min_val_f1])),
    }


def train_model(
    model: nn.Module,
    train_data: list[Data],
    val_data: list[Data],
    test_data: list[Data],
    config: TrainConfig,
) -> TrainResult:
    """
    Train with optional multiple restarts.

    When config.restarts > 1, the model is re-initialised and trained
    from scratch each time. The restart with the highest val F1 is
    selected and evaluated on the test set.
    """
    if config.restarts <= 1:
        return _train_single(model, train_data, val_data, test_data, config)

    # Save initial state for re-initialisation
    init_state = copy.deepcopy(model.state_dict())
    best_result: TrainResult | None = None

    for r in range(config.restarts):
        print(f"    [Restart {r+1}/{config.restarts}]")

        # Re-initialise model weights (fresh random init, not the saved zeros)
        if r > 0:
            model.load_state_dict(copy.deepcopy(init_state))
            # Re-randomise all parameters so each restart is independent
            for m in model.modules():
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()
            # spi_w and spi_b don't have reset_parameters — reinit manually
            if hasattr(model, "spi_w"):
                nn.init.zeros_(model.spi_w)
                nn.init.constant_(model.spi_b, -2.0)

        result = _train_single(model, train_data, val_data, test_data, config)
        result.restart_used = r

        if best_result is None or result.best_val_f1 > best_result.best_val_f1:
            best_result = result
            print(f"    [Restart {r+1}] New best: val_f1={result.best_val_f1:.4f}")
        else:
            print(
                f"    [Restart {r+1}] val_f1={result.best_val_f1:.4f} "
                f"(best so far: {best_result.best_val_f1:.4f})"
            )

    best_result.train_seconds = sum(
        r.train_seconds for r in [best_result]  # total is tracked per-restart
    )
    return best_result
