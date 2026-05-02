"""Configuration for Flattened CIFAR-10 HelixLayer experiment."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch


SCALE_PRESETS = {
    "small": {
        "hidden_dim": 256,
        "circle_units": 128,
        "helix_units": 128,
        "matched_hidden_dim": 384,
    },
    "medium": {
        "hidden_dim": 512,
        "circle_units": 256,
        "helix_units": 256,
        "matched_hidden_dim": 768,
    },
    "large": {
        "hidden_dim": 1024,
        "circle_units": 512,
        "helix_units": 512,
        "matched_hidden_dim": 1536,
    },
}


@dataclass
class CIFAR10Config:
    model_type: Literal[
        "standard_mlp",
        "standard_mlp_matched",
        "circle_mlp",
        "helix_mlp",
    ] = "helix_mlp"

    scale: Literal["small", "medium", "large"] = "medium"

    batch_size: int = 128
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.1
    use_layernorm: bool = True

    hidden_dim: int = 512
    matched_hidden_dim: int = 768
    circle_units: int = 256
    helix_units: int = 256
    num_layers: int = 2

    val_size: int = 5000
    seed: int = 0
    device: str = "cuda"

    data_dir: str = ""
    results_dir: str = "cifar10_results"
    checkpoint_dir: str = "cifar10_checkpoints"

    limit_train_batches: int | None = None
    limit_eval_batches: int | None = None

    use_scheduler: bool = False
    scheduler_type: Literal["none", "cosine"] = "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_scale_preset(config: CIFAR10Config) -> CIFAR10Config:
    """Apply scale preset dimensions to the config."""
    preset = SCALE_PRESETS[config.scale]
    config.hidden_dim = preset["hidden_dim"]
    config.circle_units = preset["circle_units"]
    config.helix_units = preset["helix_units"]
    config.matched_hidden_dim = preset["matched_hidden_dim"]
    return config


def resolve_data_dir(cli_value: str | None = None, verbose: bool = True) -> str:
    """Resolve data directory using priority chain:
    1. CLI flag (if non-empty)
    2. HELIX_DATA_DIR environment variable (if set)
    3. ./data/ relative to working directory (fallback)
    """
    if verbose:
        print("Resolving data directory:")

    # 1. CLI flag
    if cli_value:
        if verbose:
            print(f"  [1] CLI --data-dir provided: {cli_value}")
            print(f"  => Using: {cli_value}")
        return cli_value
    elif verbose:
        print(f"  [1] CLI --data-dir: not provided")

    # 2. Environment variable
    env_value = os.environ.get("HELIX_DATA_DIR", "")
    if env_value:
        if verbose:
            print(f"  [2] HELIX_DATA_DIR env var found: {env_value}")
            print(f"  => Using: {env_value}")
        return env_value
    elif verbose:
        print(f"  [2] HELIX_DATA_DIR env var: not set")

    # 3. Default
    default = "data"
    if verbose:
        print(f"  [3] Falling back to default: ./{default}/")
        print(f"  => Using: {default}")
    return default


def get_device(requested: str = "cuda") -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
