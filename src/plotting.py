"""Plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history: dict, output_path: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.set_yscale("log")

    ax2.plot(epochs, history["train_accuracy"], label="Train")
    ax2.plot(epochs, history["val_accuracy"], label="Validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_intervention_accuracy(results: dict, output_path: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shifts = sorted(int(k) for k in results["accuracy_by_shift"].keys())
    accuracies = [results["accuracy_by_shift"][str(k)] for k in shifts]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(shifts)), accuracies, tick_label=[str(s) for s in shifts])
    ax.axhline(y=1.0 / results["modulus"], color="r", linestyle="--", label="Chance")
    ax.set_xlabel("Shift k")
    ax.set_ylabel("Intervention Accuracy")
    ax.set_title(f"Intervention Accuracy ({results['model_type']}, mode={results['mode']})")
    ax.set_ylim(0, 1.05)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrix(
    expected: np.ndarray, predicted: np.ndarray, modulus: int, output_path: str
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = np.zeros((modulus, modulus), dtype=int)
    for e, p in zip(expected, predicted):
        cm[e, p] += 1

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Expected")
    ax.set_title(f"Intervention Confusion Matrix (N={modulus})")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
