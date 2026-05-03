"""Data loading and preprocessing for Covertype tabular classification experiment."""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from covertype_config import CovertypeConfig


class TabularTensorDataset(Dataset):
    """Simple dataset wrapping numpy arrays as tensors."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def load_covertype_arrays(data_dir: str = "covertype_data") -> tuple[np.ndarray, np.ndarray]:
    """Load Covertype dataset using sklearn.

    Returns X (n_samples, 54) and y (n_samples,) with labels 0-indexed.
    """
    from sklearn.datasets import fetch_covtype

    dataset = fetch_covtype(data_home=data_dir, download_if_missing=True)
    X = dataset.data
    y = dataset.target.astype(np.int64) - 1  # Convert 1..7 to 0..6
    return X, y


def make_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    seed: int = 0,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Create stratified train/val/test split."""
    from sklearn.model_selection import train_test_split

    test_fraction = 1.0 - train_fraction - val_fraction
    temp_fraction = val_fraction + test_fraction

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=temp_fraction,
        random_state=seed,
        stratify=y,
    )

    # Split temp into val and test (equal halves of the temp portion)
    val_ratio = val_fraction / temp_fraction
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1.0 - val_ratio),
        random_state=seed,
        stratify=y_temp,
    )

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }


def preprocess_covertype_features(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Standardize continuous features using train-set statistics.

    Features 0:10 are continuous (standardized).
    Features 10:54 are binary (left as 0/1).
    """
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    # Compute mean/std from training continuous features only
    mean = X_train[:, :10].mean(axis=0, keepdims=True)
    std = X_train[:, :10].std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)

    # Apply standardization to continuous features
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    X_train[:, :10] = (X_train[:, :10] - mean) / std
    X_val[:, :10] = (X_val[:, :10] - mean) / std
    X_test[:, :10] = (X_test[:, :10] - mean) / std

    return {
        "train": (X_train.astype(np.float32), y_train),
        "val": (X_val.astype(np.float32), y_val),
        "test": (X_test.astype(np.float32), y_test),
    }


def resolve_data_dir(cli_value: str | None = None, config_value: str = "") -> str:
    """Resolve data directory: CLI flag > env var > config default."""
    if cli_value:
        return cli_value

    env_value = os.environ.get("HELIX_DATA_DIR", "")
    if env_value:
        return env_value

    return config_value or "covertype_data"


def make_covertype_dataloaders(
    config: CovertypeConfig,
    cli_data_dir: str | None = None,
) -> dict[str, DataLoader]:
    """Load, split, preprocess Covertype and return dataloaders."""
    data_dir = resolve_data_dir(cli_value=cli_data_dir, config_value=config.data_dir)

    X, y = load_covertype_arrays(data_dir=data_dir)

    splits = make_train_val_test_split(
        X, y,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )

    splits = preprocess_covertype_features(splits)

    train_ds = TabularTensorDataset(*splits["train"])
    val_ds = TabularTensorDataset(*splits["val"])
    test_ds = TabularTensorDataset(*splits["test"])

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
