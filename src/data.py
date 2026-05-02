"""Modular addition dataset."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from config import ExperimentConfig


class ModularAdditionDataset(Dataset):
    def __init__(self, pairs: torch.Tensor, targets: torch.Tensor):
        self.pairs = pairs
        self.targets = targets

    def __len__(self) -> int:
        return self.pairs.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "a": self.pairs[idx, 0],
            "b": self.pairs[idx, 1],
            "target": self.targets[idx],
        }


def make_modular_addition_data(modulus: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate all (a, b) pairs and targets = (a+b) % N."""
    a_vals = torch.arange(modulus)
    b_vals = torch.arange(modulus)
    grid_a, grid_b = torch.meshgrid(a_vals, b_vals, indexing="ij")
    pairs = torch.stack([grid_a.flatten(), grid_b.flatten()], dim=1)
    targets = (pairs[:, 0] + pairs[:, 1]) % modulus
    return pairs, targets


def split_dataset(
    pairs: torch.Tensor,
    targets: torch.Tensor,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[ModularAdditionDataset, ModularAdditionDataset, ModularAdditionDataset]:
    """Split into train/val/test with seeded permutation."""
    n = pairs.shape[0]
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    return (
        ModularAdditionDataset(pairs[train_idx], targets[train_idx]),
        ModularAdditionDataset(pairs[val_idx], targets[val_idx]),
        ModularAdditionDataset(pairs[test_idx], targets[test_idx]),
    )


def make_dataloaders(config: ExperimentConfig) -> dict[str, DataLoader]:
    pairs, targets = make_modular_addition_data(config.modulus)
    train_ds, val_ds, test_ds = split_dataset(
        pairs, targets, config.train_frac, config.val_frac, config.test_frac, config.seed
    )
    return {
        "train": DataLoader(train_ds, batch_size=config.batch_size, shuffle=True),
        "val": DataLoader(val_ds, batch_size=config.batch_size, shuffle=False),
        "test": DataLoader(test_ds, batch_size=config.batch_size, shuffle=False),
    }
