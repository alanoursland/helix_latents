"""Data loading for Flattened CIFAR-10 HelixLayer experiment."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from cifar10_config import CIFAR10Config, resolve_data_dir


CIFAR10_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
])


def make_cifar10_dataloaders(
    config: CIFAR10Config,
    cli_data_dir: str | None = None,
) -> dict[str, DataLoader]:
    """Create train/val/test dataloaders for CIFAR-10."""
    data_dir = resolve_data_dir(cli_value=cli_data_dir or config.data_dir or None)

    # Load full training set
    full_train_ds = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=CIFAR10_TRANSFORM,
    )

    # Deterministic train/val split
    train_size = len(full_train_ds) - config.val_size
    generator = torch.Generator().manual_seed(config.seed)
    train_ds, val_ds = random_split(
        full_train_ds,
        [train_size, config.val_size],
        generator=generator,
    )

    # Test set
    test_ds = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=CIFAR10_TRANSFORM,
    )

    num_workers = 2
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
