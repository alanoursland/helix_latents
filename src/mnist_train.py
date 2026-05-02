"""Training and evaluation loops for MNIST models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from mnist_config import MNISTConfig, get_device, save_json
from mnist_models import count_parameters
from plotting import plot_training_history
from utils import set_seed


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    limit_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for i, (images, targets) in enumerate(dataloader):
        if limit_batches is not None and i >= limit_batches:
            break

        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.shape[0]
        total_correct += (logits.argmax(dim=-1) == targets).sum().item()
        total_examples += images.shape[0]

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "num_examples": total_examples,
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    limit_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for i, (images, targets) in enumerate(dataloader):
        if limit_batches is not None and i >= limit_batches:
            break

        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = F.cross_entropy(logits, targets)

        total_loss += loss.item() * images.shape[0]
        total_correct += (logits.argmax(dim=-1) == targets).sum().item()
        total_examples += images.shape[0]

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "num_examples": total_examples,
    }


def fit_mnist(
    model: torch.nn.Module,
    dataloaders: dict[str, DataLoader],
    config: MNISTConfig,
) -> dict[str, Any]:
    """Full training loop for MNIST. Returns history and metrics."""
    device = get_device(config.device)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    history: dict[str, list] = {
        "train_loss": [], "train_accuracy": [],
        "val_loss": [], "val_accuracy": [],
    }

    best_val_accuracy = -1.0
    best_epoch = 0
    best_state_dict = None

    pbar = tqdm(range(config.epochs), desc="Training")
    for epoch in pbar:
        train_metrics = train_one_epoch(
            model, dataloaders["train"], optimizer, device, config.limit_train_batches
        )
        val_metrics = evaluate(
            model, dataloaders["val"], device, config.limit_eval_batches
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        pbar.set_postfix(
            train_acc=f"{train_metrics['accuracy']:.4f}",
            val_acc=f"{val_metrics['accuracy']:.4f}",
            best=f"{best_val_accuracy:.4f}",
        )

    # Restore best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model = model.to(device)

    # Evaluate test set
    test_metrics = evaluate(model, dataloaders["test"], device, config.limit_eval_batches)

    # Save checkpoint
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{config.model_type}_best.pt"

    param_count = count_parameters(model)
    checkpoint = {
        "model_state_dict": best_state_dict or model.state_dict(),
        "config": config.to_dict(),
        "model_type": config.model_type,
        "param_count": param_count,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "test_metrics": test_metrics,
    }
    torch.save(checkpoint, checkpoint_path)

    # Save metrics JSON
    results_dir = Path(config.results_dir) / config.model_type
    results_dir.mkdir(parents=True, exist_ok=True)

    save_json(history, results_dir / "history.json")
    save_json({
        "model_type": config.model_type,
        "param_count": param_count,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "test_accuracy": test_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
    }, results_dir / "metrics.json")

    # Plot
    plot_training_history(history, str(results_dir / "training_history.png"))

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "test_metrics": test_metrics,
        "param_count": param_count,
        "checkpoint_path": str(checkpoint_path),
    }
