"""Training loop for Covertype tabular classification experiment."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from covertype_config import CovertypeConfig, get_device, save_json
from covertype_models import count_parameters
from plotting import plot_training_history
from utils import set_seed


def compute_metrics(
    all_targets: np.ndarray,
    all_preds: np.ndarray,
    num_classes: int = 7,
) -> dict[str, Any]:
    """Compute classification metrics from collected predictions."""
    accuracy = (all_preds == all_targets).mean()
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted")
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))

    # Per-class accuracy
    per_class_acc = []
    for c in range(num_classes):
        mask = all_targets == c
        if mask.sum() > 0:
            per_class_acc.append(float((all_preds[mask] == c).mean()))
        else:
            per_class_acc.append(0.0)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": cm.tolist(),
        "per_class_accuracy": per_class_acc,
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    limit_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    t0 = time.time()

    for i, (features, targets) in enumerate(dataloader):
        if limit_batches is not None and i >= limit_batches:
            break

        features, targets = features.to(device), targets.to(device)
        logits = model(features)
        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        total_correct += (logits.argmax(-1) == targets).sum().item()
        total_examples += features.size(0)

    elapsed = time.time() - t0

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
        "num_examples": total_examples,
        "elapsed_seconds": elapsed,
        "examples_per_second": total_examples / max(elapsed, 1e-6),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int = 7,
    limit_batches: int | None = None,
) -> dict[str, Any]:
    """Evaluate model and compute full metrics including F1."""
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_preds = []
    all_targets = []

    for i, (features, targets) in enumerate(dataloader):
        if limit_batches is not None and i >= limit_batches:
            break

        features, targets = features.to(device), targets.to(device)
        logits = model(features)
        loss = F.cross_entropy(logits, targets)

        total_loss += loss.item() * features.size(0)
        total_examples += features.size(0)

        all_preds.append(logits.argmax(-1).cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    metrics = compute_metrics(all_targets, all_preds, num_classes=num_classes)
    metrics["loss"] = total_loss / max(total_examples, 1)
    metrics["num_examples"] = total_examples

    return metrics


def fit_covertype(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    config: CovertypeConfig,
) -> dict[str, Any]:
    """Full training loop for Covertype."""
    device = get_device(config.device)
    model = model.to(device)
    set_seed(config.seed)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = None
    if config.use_scheduler and config.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "train_seconds": [],
        "train_examples_per_second": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "val_weighted_f1": [],
    }

    best_val_macro_f1 = -1.0
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict = None

    pbar = tqdm(range(config.epochs), desc="Training")
    for epoch in pbar:
        train_metrics = train_one_epoch(
            model, dataloaders["train"], optimizer, device,
            limit_batches=config.limit_train_batches,
        )
        val_metrics = evaluate(
            model, dataloaders["val"], device,
            num_classes=config.num_classes,
            limit_batches=config.limit_eval_batches,
        )

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["train_seconds"].append(train_metrics["elapsed_seconds"])
        history["train_examples_per_second"].append(train_metrics["examples_per_second"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["val_weighted_f1"].append(val_metrics["weighted_f1"])

        # Track best by val macro F1 (tie-break: accuracy, then loss)
        is_best = (
            val_metrics["macro_f1"] > best_val_macro_f1
            or (val_metrics["macro_f1"] == best_val_macro_f1 and val_metrics["accuracy"] > best_val_accuracy)
            or (val_metrics["macro_f1"] == best_val_macro_f1 and val_metrics["accuracy"] == best_val_accuracy
                and val_metrics["loss"] < best_val_loss)
        )
        if is_best:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_val_accuracy = val_metrics["accuracy"]
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        pbar.set_postfix(
            best_f1=f"{best_val_macro_f1:.4f}",
            train_acc=f"{train_metrics['accuracy']:.4f}",
            val_f1=f"{val_metrics['macro_f1']:.4f}",
        )

    # Restore best model and evaluate test
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model = model.to(device)

    test_metrics = evaluate(
        model, dataloaders["test"], device,
        num_classes=config.num_classes,
        limit_batches=config.limit_eval_batches,
    )

    # Save artifacts
    run_name = f"{config.model_type}_{config.scale}_seed{config.seed}"
    results_dir = Path(config.results_dir) / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    param_count = count_parameters(model)
    total_train_seconds = sum(history["train_seconds"])
    mean_epoch_seconds = total_train_seconds / max(len(history["train_seconds"]), 1)

    # Save checkpoint
    checkpoint_path = checkpoint_dir / f"{run_name}_best.pt"
    torch.save({
        "model_state_dict": best_state_dict,
        "config": config.to_dict(),
        "model_type": config.model_type,
        "scale": config.scale,
        "seed": config.seed,
        "param_count": param_count,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_accuracy": best_val_accuracy,
        "best_val_loss": best_val_loss,
        "test_metrics": test_metrics,
    }, checkpoint_path)

    # Save metrics JSON
    save_json({
        "model_type": config.model_type,
        "scale": config.scale,
        "seed": config.seed,
        "param_count": param_count,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_accuracy": best_val_accuracy,
        "best_val_loss": best_val_loss,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "test_loss": test_metrics["loss"],
        "train_time_total_seconds": total_train_seconds,
        "mean_epoch_seconds": mean_epoch_seconds,
    }, results_dir / "metrics.json")

    # Save history JSON
    save_json(history, results_dir / "history.json")

    # Save confusion matrix
    save_json({
        "confusion_matrix": test_metrics["confusion_matrix"],
        "per_class_accuracy": test_metrics["per_class_accuracy"],
    }, results_dir / "confusion_matrix.json")

    # Save training plot
    plot_training_history(history, results_dir / "training_history.png")

    return {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_accuracy": best_val_accuracy,
        "test_metrics": test_metrics,
        "checkpoint_path": str(checkpoint_path),
        "param_count": param_count,
        "mean_epoch_seconds": mean_epoch_seconds,
    }
