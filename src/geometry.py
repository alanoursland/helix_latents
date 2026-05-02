"""Circle and helix encoding, rotation, and shifting."""

from __future__ import annotations

import math

import torch


def number_to_theta(x: torch.Tensor, modulus: int) -> torch.Tensor:
    """theta = 2*pi*x / modulus"""
    return 2.0 * math.pi * x.float() / modulus


def circle_encode(x: torch.Tensor, modulus: int) -> torch.Tensor:
    """Encode integers as [cos(theta), sin(theta)]. Output shape: [..., 2]."""
    theta = number_to_theta(x, modulus)
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)


def helix_encode(
    x: torch.Tensor, modulus: int, alpha: float | None = None
) -> torch.Tensor:
    """Encode integers as [cos(theta), sin(theta), alpha*x]. Output shape: [..., 3]."""
    if alpha is None:
        alpha = 1.0 / modulus
    theta = number_to_theta(x, modulus)
    z = alpha * x.float()
    return torch.stack([torch.cos(theta), torch.sin(theta), z], dim=-1)


def rotate_circle(
    xy: torch.Tensor, k: int | torch.Tensor, modulus: int
) -> torch.Tensor:
    """Rotate circle coords by k modular steps. Input/output shape: [..., 2]."""
    if isinstance(k, int):
        delta = 2.0 * math.pi * k / modulus
        cos_d = math.cos(delta)
        sin_d = math.sin(delta)
    else:
        delta = 2.0 * math.pi * k.float() / modulus
        cos_d = torch.cos(delta)
        sin_d = torch.sin(delta)

    x = xy[..., 0]
    y = xy[..., 1]
    x_new = x * cos_d - y * sin_d
    y_new = x * sin_d + y * cos_d
    return torch.stack([x_new, y_new], dim=-1)


def shift_helix(
    xyz: torch.Tensor,
    k: int | torch.Tensor,
    modulus: int,
    alpha: float | None = None,
    shift_axis: bool = True,
) -> torch.Tensor:
    """Shift helix coords by k steps. Rotates phase, optionally shifts axis."""
    if alpha is None:
        alpha = 1.0 / modulus

    xy = xyz[..., :2]
    z = xyz[..., 2]

    xy_rotated = rotate_circle(xy, k, modulus)

    if shift_axis:
        if isinstance(k, int):
            z_new = z + alpha * k
        else:
            z_new = z + alpha * k.float()
    else:
        z_new = z

    return torch.cat([xy_rotated, z_new.unsqueeze(-1)], dim=-1)


def infer_phase_step(xy: torch.Tensor, modulus: int) -> torch.Tensor:
    """Recover approximate modular number from circle coords using atan2."""
    theta = torch.atan2(xy[..., 1], xy[..., 0])
    theta = theta % (2.0 * math.pi)
    return theta * modulus / (2.0 * math.pi)
