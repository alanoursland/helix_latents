"""
Rotated MNIST Data Loading
===========================

MNIST with rotation augmentation, train/val/test splits, and a
fixed-rotation-grid utility for trajectory analysis.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

from rot_mnist_config import RotMNISTConfig, resolve_data_dir


MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def make_train_transform(config: RotMNISTConfig) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomRotation(
            degrees=config.rotation_max_degrees,
            fill=config.rotation_fill,
        ),
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])


def make_rotated_test_transform(config: RotMNISTConfig) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomRotation(
            degrees=config.rotation_max_degrees,
            fill=config.rotation_fill,
        ),
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])


def make_unrotated_test_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])


def make_rot_mnist_dataloaders(
    config: RotMNISTConfig,
    cli_data_dir: str | None = None,
) -> dict[str, DataLoader]:
    """Build train, val, test_rotated, and test_unrotated dataloaders."""
    data_dir = resolve_data_dir(config, cli_data_dir)

    train_transform = make_train_transform(config)
    rotated_test_transform = make_rotated_test_transform(config)
    unrotated_test_transform = make_unrotated_test_transform()

    # Full training set with rotation augmentation
    full_train = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=train_transform,
    )

    # Split train into train + val
    val_size = 5000
    train_size = len(full_train) - val_size
    gen = torch.Generator().manual_seed(config.seed)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=gen)

    # Test sets
    test_rotated = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=rotated_test_transform,
    )
    test_unrotated = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=unrotated_test_transform,
    )

    pin = torch.cuda.is_available()
    return {
        "train": DataLoader(
            train_set, batch_size=config.batch_size, shuffle=True,
            num_workers=2, pin_memory=pin,
        ),
        "val": DataLoader(
            val_set, batch_size=config.batch_size, shuffle=False,
            num_workers=2, pin_memory=pin,
        ),
        "test_rotated": DataLoader(
            test_rotated, batch_size=config.batch_size, shuffle=False,
            num_workers=2, pin_memory=pin,
        ),
        "test_unrotated": DataLoader(
            test_unrotated, batch_size=config.batch_size, shuffle=False,
            num_workers=2, pin_memory=pin,
        ),
    }


def make_fixed_rotation_grid(
    base_image: torch.Tensor,
    num_angles: int = 72,
    fill: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate a single image through evenly-spaced angles for trajectory analysis.

    Args:
        base_image: [1, H, W] or [H, W] tensor in [0, 1] range (un-normalized).
        num_angles: Number of rotation angles (default 72 = 5 degree steps).
        fill: Fill value for rotation padding.

    Returns:
        images: [num_angles, 1, H, W] normalized tensor.
        angles: [num_angles] tensor of rotation angles in degrees.
    """
    if base_image.dim() == 2:
        base_image = base_image.unsqueeze(0)  # [1, H, W]

    angles_deg = torch.linspace(0, 360, num_angles + 1)[:num_angles]

    images = []
    for angle in angles_deg:
        # Rotate un-normalized image
        rotated = TF.rotate(base_image, angle.item(), fill=[fill])
        # Normalize after rotation
        normalized = TF.normalize(rotated, MNIST_MEAN, MNIST_STD)
        images.append(normalized)

    return torch.stack(images), angles_deg
