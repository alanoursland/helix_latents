"""
Rotated MNIST HelixConv Configuration
======================================

Config dataclass, scale presets, and utility functions for the rotated
MNIST experiment.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch


@dataclass
class RotMNISTConfig:
    model_type: Literal[
        "standard_cnn",
        "standard_cnn_matched",
        "circle_conv",
        "helix_conv",
    ] = "helix_conv"

    scale: Literal["small", "medium", "large"] = "small"

    batch_size: int = 128
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.0
    use_layernorm: bool = False

    input_channels: int = 1
    num_classes: int = 10

    hidden_channels: int = 32
    matched_hidden_channels: int = 48
    circle_units: int = 16
    helix_units: int = 16
    kernel_size: int = 5
    num_conv_blocks: int = 2

    rotation_max_degrees: int = 180
    rotation_fill: float = 0.0

    seed: int = 0
    device: str = "cuda"

    data_dir: str = "rot_mnist_data"
    results_dir: str = "rot_mnist_results"
    checkpoint_dir: str = "rot_mnist_checkpoints"
    figures_dir: str = "rot_mnist_figures"

    limit_train_batches: int | None = None
    limit_eval_batches: int | None = None

    use_scheduler: bool = False
    scheduler_type: Literal["none", "cosine"] = "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SCALE_PRESETS: dict[str, dict[str, int]] = {
    "small": {
        "hidden_channels": 32,
        "circle_units": 16,
        "helix_units": 16,
        "matched_hidden_channels": 48,
    },
    "medium": {
        "hidden_channels": 64,
        "circle_units": 32,
        "helix_units": 32,
        "matched_hidden_channels": 96,
    },
    "large": {
        "hidden_channels": 128,
        "circle_units": 64,
        "helix_units": 64,
        "matched_hidden_channels": 192,
    },
}


def apply_scale_preset(config: RotMNISTConfig) -> RotMNISTConfig:
    preset = SCALE_PRESETS[config.scale]
    for key, value in preset.items():
        setattr(config, key, value)
    return config


def resolve_data_dir(config: RotMNISTConfig, cli_data_dir: str | None = None) -> str:
    if cli_data_dir is not None:
        return cli_data_dir
    env = os.environ.get("HELIX_DATA_DIR")
    if env is not None:
        return env
    return config.data_dir


def get_device(requested: str = "cuda") -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
