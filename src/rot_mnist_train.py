"""
Rotated MNIST Training Loop
=============================

Train, evaluate, and checkpoint routines for the rotated MNIST experiment.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from rot_mnist_config import RotMNISTConfig, get_device, save_json
from rot_mnist_models import count_parameters
from utils import set_seed


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    limit_batches: int | None = None,
    quad_reg_lambda: float = 0.0,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    t0 = time.time()

    for i, (images, targets) in enumerate(dataloader):
        if limit_batches is not None and i >= limit_batches:
            break
        images, targets = images.to(device), targets.to(device)

        logits = model(images)
        loss = F.cross_entropy(logits, targets)
        if quad_reg_lambda > 0:
            loss = loss + quad_reg_lambda * model.quadrature_penalty()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(1) == targets).sum().item()
        total_examples += images.size(0)

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
    limit_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for i, (images, targets) in enumerate(dataloader):
        if limit_batches is not None and i >= limit_batches:
            break
        images, targets = images.to(device), targets.to(device)

        logits = model(images)
        loss = F.cross_entropy(logits, targets)

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(1) == targets).sum().item()
        total_examples += images.size(0)

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
        "num_examples": total_examples,
    }


def plot_training_history(
    history: dict[str, list],
    output_path: str | Path,
    test_rotated_acc: float | None = None,
    test_unrotated_acc: float | None = None,
) -> None:
    """Plot training history with optional test accuracy lines."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_accuracy"], label="Train")
    ax2.plot(epochs, history["val_accuracy"], label="Val")
    if test_rotated_acc is not None:
        ax2.axhline(y=test_rotated_acc, color="red", linestyle="--",
                     label=f"Test Rotated ({test_rotated_acc:.4f})")
    if test_unrotated_acc is not None:
        ax2.axhline(y=test_unrotated_acc, color="green", linestyle="--",
                     label=f"Test Unrotated ({test_unrotated_acc:.4f})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def fit_rot_mnist(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    config: RotMNISTConfig,
) -> dict[str, Any]:
    """Full training loop for one rotated MNIST model variant."""
    device = get_device(config.device)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = None
    if config.use_scheduler and config.scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    # Run naming
    run_name = f"{config.model_type}_{config.scale}_seed{config.seed}"
    results_dir = Path(config.results_dir) / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    quad_reg_lambda = (
        config.quad_reg_lambda if config.model_type == "helix_conv_quadreg" else 0.0
    )
    track_alignment = hasattr(model, "quadrature_alignment")

    history: dict[str, list] = {
        "train_loss": [],
        "train_accuracy": [],
        "train_seconds": [],
        "train_examples_per_second": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    if track_alignment:
        # Mean per-unit correlation between W_v and rotate(W_u, 90°), per
        # conv block, recorded at init and after each epoch. This is the
        # preserve-or-destroy trace for the quadrature experiment.
        history["quad_alignment"] = [model.quadrature_alignment()]

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            model, dataloaders["train"], optimizer, device,
            limit_batches=config.limit_train_batches,
            quad_reg_lambda=quad_reg_lambda,
        )
        val_metrics = evaluate(
            model, dataloaders["val"], device,
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
        if track_alignment:
            history["quad_alignment"].append(model.quadrature_alignment())

        # Best checkpoint by val accuracy (tiebreak: val loss)
        is_best = False
        if val_metrics["accuracy"] > best_val_acc:
            is_best = True
        elif val_metrics["accuracy"] == best_val_acc and val_metrics["loss"] < best_val_loss:
            is_best = True

        if is_best:
            best_val_acc = val_metrics["accuracy"]
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"  Epoch {epoch:3d}/{config.epochs}  "
            f"train_loss={train_metrics['loss']:.4f}  "
            f"train_acc={train_metrics['accuracy']:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  "
            + (
                f"quad_align={history['quad_alignment'][-1][0]:.3f}  "
                if track_alignment else ""
            )
            + f"{'*' if is_best else ''}"
        )

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    # Test on both rotated and un-rotated
    test_rotated = evaluate(
        model, dataloaders["test_rotated"], device,
        limit_batches=config.limit_eval_batches,
    )
    test_unrotated = evaluate(
        model, dataloaders["test_unrotated"], device,
        limit_batches=config.limit_eval_batches,
    )

    # Save checkpoint
    ckpt_path = ckpt_dir / f"{run_name}_best.pt"
    torch.save({
        "model_state_dict": best_state,
        "config": config.to_dict(),
        "model_type": config.model_type,
        "scale": config.scale,
        "seed": config.seed,
        "param_count": count_parameters(model),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "best_val_loss": best_val_loss,
    }, ckpt_path)

    # Save metrics
    n_params = count_parameters(model)
    epoch_times = history["train_seconds"]
    mean_epoch_seconds = float(np.mean(epoch_times)) if epoch_times else 0.0

    metrics = {
        "model_type": config.model_type,
        "scale": config.scale,
        "seed": config.seed,
        "param_count": n_params,
        "epochs": config.epochs,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "best_val_loss": best_val_loss,
        "test_rotated_accuracy": test_rotated["accuracy"],
        "test_rotated_loss": test_rotated["loss"],
        "test_unrotated_accuracy": test_unrotated["accuracy"],
        "test_unrotated_loss": test_unrotated["loss"],
        "mean_epoch_seconds": mean_epoch_seconds,
    }
    if track_alignment:
        metrics["quad_alignment_init"] = history["quad_alignment"][0]
        metrics["quad_alignment_final"] = model.quadrature_alignment()
        metrics["quad_reg_lambda"] = quad_reg_lambda
    save_json(metrics, results_dir / "metrics.json")
    save_json(history, results_dir / "history.json")

    # Quadrature alignment plot (preserve-or-destroy trace)
    if track_alignment:
        align = np.array(history["quad_alignment"])  # [epochs+1, num_blocks]
        fig, ax = plt.subplots(figsize=(8, 5))
        for block in range(align.shape[1]):
            ax.plot(range(align.shape[0]), align[:, block], marker="o",
                    markersize=3, label=f"Layer {block}")
        ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5,
                   label="Perfect quadrature")
        ax.axhline(y=0.0, color="gray", linestyle="--", alpha=0.5,
                   label="Independent filters")
        ax.set_xlabel("Epoch (0 = init)")
        ax.set_ylabel("Mean corr(W_v, rotate(W_u, 90°))")
        ax.set_title(f"Quadrature Alignment During Training\n{run_name}")
        ax.set_ylim(-0.3, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(results_dir / "quad_alignment.png", dpi=150)
        plt.close(fig)

    # Training history plot
    plot_training_history(
        history,
        results_dir / "training_history.png",
        test_rotated_acc=test_rotated["accuracy"],
        test_unrotated_acc=test_unrotated["accuracy"],
    )

    return {
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "best_val_loss": best_val_loss,
        "test_rotated": test_rotated,
        "test_unrotated": test_unrotated,
        "checkpoint_path": str(ckpt_path),
        "param_count": n_params,
        "mean_epoch_seconds": mean_epoch_seconds,
        "run_name": run_name,
        "results_dir": str(results_dir),
    }
