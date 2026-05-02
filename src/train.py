"""Training and evaluation loops."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ExperimentConfig, get_device
from utils import set_seed


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in dataloader:
        a = batch["a"].to(device)
        b = batch["b"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()
        output = model(a, b)
        loss = F.cross_entropy(output.logits, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * a.shape[0]
        total_correct += (output.logits.argmax(dim=-1) == target).sum().item()
        total_examples += a.shape[0]

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
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in dataloader:
        a = batch["a"].to(device)
        b = batch["b"].to(device)
        target = batch["target"].to(device)

        output = model(a, b)
        loss = F.cross_entropy(output.logits, target)

        total_loss += loss.item() * a.shape[0]
        total_correct += (output.logits.argmax(dim=-1) == target).sum().item()
        total_examples += a.shape[0]

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "num_examples": total_examples,
    }


def fit(
    model: torch.nn.Module,
    dataloaders: dict[str, DataLoader],
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Full training loop with early stopping and checkpointing."""
    set_seed(config.seed)
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
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict = None
    patience_counter = 0

    pbar = tqdm(range(config.max_epochs), desc="Training")
    for epoch in pbar:
        train_metrics = train_one_epoch(model, dataloaders["train"], optimizer, device)
        val_metrics = evaluate(model, dataloaders["val"], device)

        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        improved = (
            val_metrics["accuracy"] > best_val_accuracy
            or (val_metrics["accuracy"] == best_val_accuracy
                and val_metrics["loss"] < best_val_loss)
        )

        if improved:
            best_val_accuracy = val_metrics["accuracy"]
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        pbar.set_postfix(
            train_acc=f"{train_metrics['accuracy']:.4f}",
            val_acc=f"{val_metrics['accuracy']:.4f}",
            best=f"{best_val_accuracy:.4f}",
        )

        if patience_counter >= config.early_stopping_patience:
            break

    # Restore best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model = model.to(device)

    # Save checkpoint
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best.pt"

    checkpoint = {
        "model_state_dict": best_state_dict or model.state_dict(),
        "config": config.to_dict(),
        "history": history,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
    }
    torch.save(checkpoint, checkpoint_path)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "checkpoint_path": str(checkpoint_path),
    }
