"""MNIST data loading with train/val split."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from mnist_config import MNISTConfig, resolve_data_dir

MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def make_mnist_dataloaders(
    config: MNISTConfig, cli_data_dir: str | None = None
) -> dict[str, DataLoader]:
    """Load MNIST and return train/val/test dataloaders.

    Data directory is resolved via the priority chain:
    CLI flag > HELIX_DATA_DIR env var > ./data/
    """
    data_dir = resolve_data_dir(cli_data_dir or config.data_dir or None)

    full_train = datasets.MNIST(data_dir, train=True, download=True, transform=MNIST_TRANSFORM)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=MNIST_TRANSFORM)

    # Deterministic train/val split
    train_size = len(full_train) - config.val_size
    generator = torch.Generator().manual_seed(config.seed)
    train_ds, val_ds = random_split(full_train, [train_size, config.val_size], generator=generator)

    return {
        "train": DataLoader(train_ds, batch_size=config.batch_size, shuffle=True),
        "val": DataLoader(val_ds, batch_size=config.batch_size, shuffle=False),
        "test": DataLoader(test_ds, batch_size=config.batch_size, shuffle=False),
    }
