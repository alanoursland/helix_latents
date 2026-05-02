"""Experiment configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch


@dataclass
class ExperimentConfig:
    modulus: int = 59
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15

    model_type: Literal[
        "baseline_mlp",
        "circle_bottleneck_mlp",
        "helix_bottleneck_mlp",
    ] = "helix_bottleneck_mlp"

    embedding_dim: int = 32
    hidden_dim: int = 128
    num_hidden_layers: int = 2
    dropout: float = 0.0

    helix_alpha: float | None = None

    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 500
    early_stopping_patience: int = 50

    seed: int = 0
    device: str = "cuda"

    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"

    def effective_alpha(self) -> float:
        if self.helix_alpha is not None:
            return self.helix_alpha
        return 1.0 / self.modulus

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_device(requested: str = "cuda") -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
