"""Training loop for EEML MPNN pipeline."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

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
    """Build optimizer with optional separate LR for spi_w/spi_b params."""
    if config.w_lr_mult != 1.0 and hasattr(model, "spi_w"):
        w_params = [model.spi_w, model.spi_b]
        w_param_ids = {id(p) for p in w_params}
        other_params = [p for p in model.parameters() if id(p) not in w_param_ids]
        param_groups = [
            {"params": other_params, "lr": config.lr},
            {"params": w_params, "lr": config.lr * config.w_lr_mult},
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr": config.lr}]

    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=config.weight_decay
    )

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
                for idx in config.spi_family_indices:
                    loss = loss + config.group_lambda * model.spi_w[idx].norm(2)

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

        if patience_counter >= config.patience:
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
