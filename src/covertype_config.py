"""Configuration for Covertype tabular classification experiment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch


SCALE_PRESETS = {
    "small": {
        "hidden_dim": 64,
        "circle_units": 32,
        "helix_units": 32,
        "matched_hidden_dim": 96,
    },
    "medium": {
        "hidden_dim": 128,
        "circle_units": 64,
        "helix_units": 64,
        "matched_hidden_dim": 192,
    },
    "large": {
        "hidden_dim": 256,
        "circle_units": 128,
        "helix_units": 128,
        "matched_hidden_dim": 384,
    },
}


@dataclass
class CovertypeConfig:
    model_type: Literal[
        "standard_mlp",
        "standard_mlp_matched",
        "circle_mlp",
        "helix_mlp",
    ] = "helix_mlp"

    scale: Literal["small", "medium", "large"] = "medium"

    batch_size: int = 1024
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.05
    use_layernorm: bool = True

    input_dim: int = 54
    num_classes: int = 7

    hidden_dim: int = 128
    matched_hidden_dim: int = 192
    circle_units: int = 64
    helix_units: int = 64
    num_layers: int = 2

    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15

    seed: int = 0
    device: str = "cuda"

    data_dir: str = "covertype_data"
    results_dir: str = "covertype_results"
    checkpoint_dir: str = "covertype_checkpoints"

    limit_train_batches: int | None = None
    limit_eval_batches: int | None = None

    use_scheduler: bool = False
    scheduler_type: Literal["none", "cosine"] = "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_scale_preset(config: CovertypeConfig) -> CovertypeConfig:
    """Apply scale preset dimensions to the config."""
    preset = SCALE_PRESETS[config.scale]
    config.hidden_dim = preset["hidden_dim"]
    config.circle_units = preset["circle_units"]
    config.helix_units = preset["helix_units"]
    config.matched_hidden_dim = preset["matched_hidden_dim"]
    return config


def get_device(requested: str = "cuda") -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
