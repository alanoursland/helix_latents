"""
Rotated MNIST Causal Intervention Analysis (Measurement 4, Optional)
=====================================================================

Tests whether rotating (a, b) at a hidden layer produces the same output
as rotating the input image.

Only run if Measurements 2 and 3 show self-organization.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from rot_mnist_config import save_json
from rot_mnist_data import make_fixed_rotation_grid


def intervene_at_layer(
    model: nn.Module,
    images: torch.Tensor,
    delta_theta_rad: float,
    layer_idx: int = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Run forward with (a, b) rotated by delta_theta at the specified layer.

    Args:
        model: HelixCNN model.
        images: [batch, 1, H, W] input images.
        delta_theta_rad: Angle in radians to rotate (a, b).
        layer_idx: Which HelixConv layer to intervene on.
        device: Computation device.

    Returns:
        logits: [batch, num_classes] model output after intervention.
    """
    if device is None:
        device = next(model.parameters()).device

    images = images.to(device)
    model.eval()

    cos_d = float(np.cos(delta_theta_rad))
    sin_d = float(np.sin(delta_theta_rad))

    with torch.no_grad():
        x = images
        for i, (conv, act, pool) in enumerate(
            zip(model.conv_blocks, model.activations, model.pools)
        ):
            if i == layer_idx:
                _, inter = conv.forward_with_intermediates(x)
                a = inter["a"]
                b = inter["b"]

                # Rotate (a, b) by delta_theta
                a_new = a * cos_d - b * sin_d
                b_new = a * sin_d + b * cos_d

                # Reconstruct features with rotated (a, b)
                z = inter.get("z", None)
                r = torch.sqrt(a_new * a_new + b_new * b_new + conv.eps)
                sin_t = b_new / r
                cos_t = a_new / r

                if z is not None:
                    feats = torch.cat([
                        sin_t, cos_t, r, z,
                        r * sin_t, r * cos_t,
                        torch.tanh(z), r * torch.tanh(z),
                    ], dim=1)
                else:
                    feats = torch.cat([
                        sin_t, cos_t, r,
                        r * sin_t, r * cos_t,
                    ], dim=1)

                x = pool(act(conv.project(feats)))
            else:
                x = pool(act(conv(x)))

        x = x.flatten(1)
        logits = model.classifier(x)

    return logits


def intervention_match_accuracy(
    model: nn.Module,
    base_images: torch.Tensor,
    delta_theta_deg: float,
    layer_idx: int = 0,
    num_angles: int = 72,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Compare intervened output to output on actually-rotated input.

    For each base image:
      1. Run model on base image, intervene with delta_theta -> get predictions.
      2. Rotate base image by delta_theta, run model normally -> get predictions.
      3. Check if predictions match.

    Args:
        model: HelixCNN model.
        base_images: [N, 1, H, W] un-normalized images.
        delta_theta_deg: Intervention angle in degrees.
        layer_idx: Which layer to intervene on.
        device: Computation device.

    Returns:
        dict with match_accuracy and details.
    """
    if device is None:
        device = next(model.parameters()).device

    delta_theta_rad = np.radians(delta_theta_deg)
    model.eval()

    from rot_mnist_data import MNIST_MEAN, MNIST_STD
    import torchvision.transforms.functional as TF

    matches = 0
    total = 0

    for i in range(base_images.size(0)):
        base_img = base_images[i]  # [1, H, W]

        # Normalize base image for model
        base_norm = TF.normalize(base_img, MNIST_MEAN, MNIST_STD).unsqueeze(0).to(device)

        # Intervened prediction
        logits_intervened = intervene_at_layer(
            model, base_norm, delta_theta_rad, layer_idx=layer_idx, device=device,
        )
        pred_intervened = logits_intervened.argmax(1).item()

        # Actually rotate the image, then normalize
        rotated_img = TF.rotate(base_img, delta_theta_deg, fill=[0.0])
        rotated_norm = TF.normalize(rotated_img, MNIST_MEAN, MNIST_STD).unsqueeze(0).to(device)

        with torch.no_grad():
            logits_rotated = model(rotated_norm)
        pred_rotated = logits_rotated.argmax(1).item()

        if pred_intervened == pred_rotated:
            matches += 1
        total += 1

    return {
        "delta_theta_deg": delta_theta_deg,
        "match_accuracy": matches / max(total, 1),
        "matches": matches,
        "total": total,
    }


def sweep_intervention_angles(
    model: nn.Module,
    test_dataset,
    results_dir: str | Path,
    delta_thetas: list[float] | None = None,
    num_images: int = 100,
    layer_idx: int = 0,
    device: torch.device | None = None,
) -> dict:
    """Run intervention analysis at multiple angles.

    Args:
        model: HelixCNN model.
        test_dataset: Un-rotated MNIST test dataset.
        results_dir: Where to save results.
        delta_thetas: List of intervention angles in degrees.
        num_images: Number of test images to use.
        layer_idx: Which layer to intervene on.
        device: Computation device.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if delta_thetas is None:
        delta_thetas = [15.0, 30.0, 60.0, 90.0, 180.0]

    if device is None:
        device = next(model.parameters()).device

    # Collect un-normalized base images
    base_images = []
    for i in range(min(num_images, len(test_dataset))):
        img, _ = test_dataset[i]
        # Denormalize
        base = img * 0.3081 + 0.1307
        base = base.clamp(0, 1)
        base_images.append(base)
    base_images = torch.stack(base_images)

    results = []
    for dt in delta_thetas:
        result = intervention_match_accuracy(
            model, base_images, dt, layer_idx=layer_idx, device=device,
        )
        results.append(result)
        print(f"  Δθ={dt:6.1f}°  match_accuracy={result['match_accuracy']:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    angles = [r["delta_theta_deg"] for r in results]
    accs = [r["match_accuracy"] for r in results]
    ax.bar(range(len(angles)), accs, tick_label=[f"{a:.0f}°" for a in angles], alpha=0.7)
    ax.set_xlabel("Intervention Angle Δθ")
    ax.set_ylabel("Match Accuracy")
    ax.set_title("Causal Intervention: Does rotating (a,b) match rotating input?")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.1, color="red", linestyle="--", alpha=0.5, label="Chance (10%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "intervention_summary.png", dpi=150)
    plt.close(fig)

    # Save JSON
    summary = {
        "layer_idx": layer_idx,
        "num_images": num_images,
        "results": results,
    }
    save_json(summary, results_dir / "intervention.json")

    return summary
