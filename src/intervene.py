"""Latent intervention logic."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from config import ExperimentConfig, get_device
from geometry import rotate_circle, shift_helix
from models import ModelOutput


def expected_intervention_targets(
    a: torch.Tensor, b: torch.Tensor, k: int, modulus: int
) -> torch.Tensor:
    """Compute (a + k + b) mod N."""
    return (a + k + b) % modulus


@torch.no_grad()
def intervene_on_a(
    model: torch.nn.Module,
    a: torch.Tensor,
    b: torch.Tensor,
    k: int,
    modulus: int,
    model_type: str,
    alpha: float | None = None,
    mode: str = "phase_plus_axis",
) -> ModelOutput:
    """Run model with intervention on latent(a).

    1. Normal forward to get latents.
    2. Modify latents["a"].
    3. Re-run with latent_override.
    """
    model.eval()
    normal_output = model(a, b)
    latent_a = normal_output.latents["a"]

    if model_type == "circle_bottleneck_mlp":
        if mode in ("phase_plus_axis", "phase_only"):
            modified = rotate_circle(latent_a, k, modulus)
        elif mode == "random":
            random_angle = torch.randn(1, device=latent_a.device).item() * 6.2832
            cos_r = torch.cos(torch.tensor(random_angle, device=latent_a.device))
            sin_r = torch.sin(torch.tensor(random_angle, device=latent_a.device))
            x, y = latent_a[..., 0], latent_a[..., 1]
            modified = torch.stack([x * cos_r - y * sin_r, x * sin_r + y * cos_r], dim=-1)
        elif mode == "axis_only":
            modified = latent_a  # no axis for circle
        else:
            raise ValueError(f"Unsupported mode '{mode}' for circle model")

    elif model_type == "helix_bottleneck_mlp":
        if alpha is None:
            alpha = 1.0 / modulus
        if mode == "phase_plus_axis":
            modified = shift_helix(latent_a, k, modulus, alpha, shift_axis=True)
        elif mode == "phase_only":
            modified = shift_helix(latent_a, k, modulus, alpha, shift_axis=False)
        elif mode == "axis_only":
            modified = latent_a.clone()
            modified[..., 2] = modified[..., 2] + alpha * k
        elif mode == "random":
            noise = torch.randn_like(latent_a) * 0.5
            modified = latent_a + noise
        else:
            raise ValueError(f"Unsupported mode '{mode}' for helix model")

    elif model_type == "baseline_mlp":
        if mode == "random":
            noise = torch.randn_like(latent_a) * latent_a.std()
            modified = latent_a + noise
        else:
            raise ValueError(f"Baseline model only supports 'random' mode, got '{mode}'")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model(a, b, latent_override={"a": modified})


@torch.no_grad()
def evaluate_intervention(
    model: torch.nn.Module,
    dataloader: DataLoader,
    config: ExperimentConfig,
    shifts: list[int],
    mode: str = "phase_plus_axis",
) -> dict[str, Any]:
    """Evaluate intervention accuracy across multiple shifts."""
    device = get_device(config.device)
    model = model.to(device)
    model.eval()
    alpha = config.effective_alpha()

    accuracy_by_shift: dict[str, float] = {}
    total_correct = 0
    total_examples = 0

    for k in shifts:
        shift_correct = 0
        shift_total = 0

        for batch in dataloader:
            a = batch["a"].to(device)
            b = batch["b"].to(device)
            expected = expected_intervention_targets(a, b, k, config.modulus)
            output = intervene_on_a(model, a, b, k, config.modulus, config.model_type, alpha, mode)
            predictions = output.logits.argmax(dim=-1)
            shift_correct += (predictions == expected).sum().item()
            shift_total += a.shape[0]

        accuracy_by_shift[str(k)] = shift_correct / shift_total
        total_correct += shift_correct
        total_examples += shift_total

    return {
        "model_type": config.model_type,
        "mode": mode,
        "modulus": config.modulus,
        "accuracy_by_shift": accuracy_by_shift,
        "overall_accuracy": total_correct / total_examples if total_examples > 0 else 0.0,
        "num_examples": total_examples,
    }
