"""
Rotated MNIST Trajectory Analysis (Measurement 3)
===================================================

Tests whether (a, b) traces a circle as the input image rotates.

For a fixed un-rotated MNIST digit:
  1. Rotate through 0-360 degrees in fine increments.
  2. Record (a, b, z, r) at a fixed spatial position for each helix unit.
  3. Plot trajectories in 2D.

Produces:
  trajectory_grid_layer<N>.png
  trajectory_summary_layer<N>.png
  trajectory.json
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rot_mnist_config import save_json
from rot_mnist_data import make_fixed_rotation_grid


def trajectory_for_unit(
    model,
    base_image: torch.Tensor,
    layer_idx: int = 0,
    unit_idx: int = 0,
    spatial_position: tuple[int, int] = (14, 14),
    num_angles: int = 72,
    fill: float = 0.0,
    device: torch.device | None = None,
) -> dict[str, np.ndarray]:
    """Compute (a, b, z, r) trajectory for one helix unit as input rotates.

    Args:
        model: HelixCNN model.
        base_image: [1, H, W] or [H, W] un-normalized tensor.
        layer_idx: Which HelixConv layer to probe.
        unit_idx: Which helix unit within that layer.
        spatial_position: (row, col) to sample.
        num_angles: Number of rotation steps through 360 degrees.
        fill: Fill value for rotation.
        device: Device for computation.

    Returns:
        dict with keys: angles, a, b, z, r (each [num_angles] numpy array)
    """
    if device is None:
        device = next(model.parameters()).device

    images, angles_deg = make_fixed_rotation_grid(base_image, num_angles=num_angles, fill=fill)
    # images: [num_angles, 1, H, W], angles_deg: [num_angles]

    model.eval()
    with torch.no_grad():
        images = images.to(device)
        _, intermediates = model.forward_with_intermediates_at_layer(images, layer_idx=layer_idx)

    # Clamp spatial position to actual feature map size
    feat_h, feat_w = intermediates["a"].shape[2], intermediates["a"].shape[3]
    h = min(spatial_position[0], feat_h - 1)
    w = min(spatial_position[1], feat_w - 1)

    a_vals = intermediates["a"][:, unit_idx, h, w].cpu().numpy()
    b_vals = intermediates["b"][:, unit_idx, h, w].cpu().numpy()
    r_vals = intermediates["r"][:, unit_idx, h, w].cpu().numpy()
    z_vals = intermediates.get("z", torch.zeros_like(intermediates["a"]))[:, unit_idx, h, w].cpu().numpy()

    return {
        "angles": angles_deg.numpy(),
        "a": a_vals,
        "b": b_vals,
        "z": z_vals,
        "r": r_vals,
    }


def trajectory_for_layer(
    model,
    base_image: torch.Tensor,
    layer_idx: int = 0,
    spatial_position: tuple[int, int] = (14, 14),
    num_angles: int = 72,
    fill: float = 0.0,
    device: torch.device | None = None,
) -> dict[str, np.ndarray]:
    """Compute trajectories for all units in one layer at once.

    Returns:
        dict with keys: angles [num_angles], a [num_angles, units],
        b [num_angles, units], z [num_angles, units], r [num_angles, units]
    """
    if device is None:
        device = next(model.parameters()).device

    images, angles_deg = make_fixed_rotation_grid(base_image, num_angles=num_angles, fill=fill)

    model.eval()
    with torch.no_grad():
        images = images.to(device)
        _, intermediates = model.forward_with_intermediates_at_layer(images, layer_idx=layer_idx)

    # Clamp spatial position to actual feature map size
    feat_h, feat_w = intermediates["a"].shape[2], intermediates["a"].shape[3]
    h = min(spatial_position[0], feat_h - 1)
    w = min(spatial_position[1], feat_w - 1)

    a_all = intermediates["a"][:, :, h, w].cpu().numpy()  # [num_angles, units]
    b_all = intermediates["b"][:, :, h, w].cpu().numpy()
    r_all = intermediates["r"][:, :, h, w].cpu().numpy()
    z_all = intermediates.get("z", torch.zeros_like(intermediates["a"]))[:, :, h, w].cpu().numpy()

    return {
        "angles": angles_deg.numpy(),
        "a": a_all,
        "b": b_all,
        "z": z_all,
        "r": r_all,
    }


def compute_circularity_score(a: np.ndarray, b: np.ndarray) -> float:
    """Ratio of minor/major axis of best-fit ellipse. 1.0 = perfect circle."""
    points = np.stack([a, b], axis=1)  # [N, 2]
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 0)
    if eigenvalues.max() < 1e-12:
        return 0.0
    return float(np.sqrt(eigenvalues.min() / eigenvalues.max()))


def compute_radius_variance(r: np.ndarray) -> float:
    """Coefficient of variation of r. Low = rotation-invariant radius."""
    mean_r = r.mean()
    if mean_r < 1e-12:
        return 0.0
    return float(r.std() / mean_r)


def compute_z_variance(z: np.ndarray) -> float:
    """Coefficient of variation of z. Low = rotation-invariant axis."""
    mean_z = np.abs(z).mean()
    if mean_z < 1e-12:
        return float(z.std())
    return float(z.std() / mean_z)


def compute_winding_number(a: np.ndarray, b: np.ndarray) -> float:
    """Number of full rotations the trajectory makes (via cumulative angle change)."""
    angles = np.arctan2(b, a)
    diffs = np.diff(angles)
    # Unwrap jumps
    diffs = (diffs + np.pi) % (2 * np.pi) - np.pi
    total_angle = np.abs(diffs.sum())
    return float(total_angle / (2 * np.pi))


def plot_trajectory_grid(
    layer_data: dict[str, np.ndarray],
    output_path: str | Path,
    layer_idx: int = 0,
    title_suffix: str = "",
) -> None:
    """Grid of (a, b) trajectory plots, one per unit, color-coded by rotation angle."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    angles = layer_data["angles"]
    a_all = layer_data["a"]  # [num_angles, units]
    b_all = layer_data["b"]
    num_units = a_all.shape[1]

    cols = min(num_units, 8)
    rows = (num_units + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for ax in axes.flat:
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    colors = plt.cm.hsv(angles / 360.0)

    for u in range(num_units):
        row = u // cols
        col = u % cols
        ax = axes[row, col]

        ax.scatter(a_all[:, u], b_all[:, u], c=colors, s=8, alpha=0.8)
        # Draw lines connecting sequential points
        ax.plot(a_all[:, u], b_all[:, u], color="gray", alpha=0.3, linewidth=0.5)

        circ = compute_circularity_score(a_all[:, u], b_all[:, u])
        winding = compute_winding_number(a_all[:, u], b_all[:, u])
        ax.set_title(f"Unit {u}\ncirc={circ:.2f} wind={winding:.1f}", fontsize=7)

    # Hide extra axes
    for u in range(num_units, rows * cols):
        row = u // cols
        col = u % cols
        axes[row, col].set_visible(False)

    fig.suptitle(
        f"(a, b) Trajectories Under Input Rotation\n"
        f"Layer {layer_idx}{title_suffix}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_trajectory_summary(
    layer_data: dict[str, np.ndarray],
    output_path: str | Path,
    layer_idx: int = 0,
) -> None:
    """Bar charts of per-unit circularity, radius variance, z variance, winding number."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    a_all = layer_data["a"]
    b_all = layer_data["b"]
    r_all = layer_data["r"]
    z_all = layer_data["z"]
    num_units = a_all.shape[1]

    circularities = [compute_circularity_score(a_all[:, u], b_all[:, u]) for u in range(num_units)]
    r_variances = [compute_radius_variance(r_all[:, u]) for u in range(num_units)]
    z_variances = [compute_z_variance(z_all[:, u]) for u in range(num_units)]
    windings = [compute_winding_number(a_all[:, u], b_all[:, u]) for u in range(num_units)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    units = np.arange(num_units)

    axes[0, 0].bar(units, circularities, alpha=0.7)
    axes[0, 0].set_title("Circularity (1.0 = circle)")
    axes[0, 0].set_xlabel("Unit")
    axes[0, 0].set_ylim(0, 1.1)

    axes[0, 1].bar(units, r_variances, alpha=0.7, color="orange")
    axes[0, 1].set_title("Radius CV (low = invariant)")
    axes[0, 1].set_xlabel("Unit")

    axes[1, 0].bar(units, z_variances, alpha=0.7, color="green")
    axes[1, 0].set_title("Z CV (low = invariant)")
    axes[1, 0].set_xlabel("Unit")

    axes[1, 1].bar(units, windings, alpha=0.7, color="purple")
    axes[1, 1].set_title("Winding Number")
    axes[1, 1].set_xlabel("Unit")

    fig.suptitle(f"Trajectory Summary - Layer {layer_idx}", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return {
        "circularities": circularities,
        "r_variances": r_variances,
        "z_variances": z_variances,
        "windings": windings,
    }


def run_trajectory_analysis(
    model,
    test_dataset,
    results_dir: str | Path,
    layer_idx: int = 0,
    num_digits: int = 3,
    spatial_positions: list[tuple[int, int]] | None = None,
    num_angles: int = 72,
    device: torch.device | None = None,
) -> dict:
    """Run trajectory analysis for multiple digits and positions.

    Args:
        model: HelixCNN model.
        test_dataset: un-rotated MNIST test dataset (returns (image, label)).
        results_dir: Where to save figures and JSON.
        layer_idx: Which HelixConv layer to probe.
        num_digits: How many distinct base digits to analyze.
        spatial_positions: List of (row, col) to probe. Default: center and two offsets.
        num_angles: Rotation steps.
        device: Computation device.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if spatial_positions is None:
        spatial_positions = [(14, 14), (10, 10), (18, 18)]

    if device is None:
        device = next(model.parameters()).device

    all_summaries = []

    for digit_idx in range(min(num_digits, len(test_dataset))):
        raw_img, label = test_dataset[digit_idx]
        # raw_img is already a tensor from ToTensor (normalized)
        # We need un-normalized for rotation grid, so denormalize
        base_image = raw_img * 0.3081 + 0.1307  # undo normalization
        base_image = base_image.clamp(0, 1)

        for pos in spatial_positions:
            layer_data = trajectory_for_layer(
                model, base_image, layer_idx=layer_idx,
                spatial_position=pos, num_angles=num_angles,
                device=device,
            )

            suffix = f"_digit{digit_idx}_pos{pos[0]}_{pos[1]}"

            plot_trajectory_grid(
                layer_data,
                results_dir / f"trajectory_grid_layer{layer_idx}{suffix}.png",
                layer_idx=layer_idx,
                title_suffix=f" | digit={label}, pos={pos}",
            )

            summary_metrics = plot_trajectory_summary(
                layer_data,
                results_dir / f"trajectory_summary_layer{layer_idx}{suffix}.png",
                layer_idx=layer_idx,
            )

            all_summaries.append({
                "digit_idx": digit_idx,
                "digit_label": int(label),
                "spatial_position": list(pos),
                "layer_idx": layer_idx,
                **{k: [float(v) for v in vals] for k, vals in summary_metrics.items()},
            })

    # Primary trajectory grid: first digit, center position
    primary_data = trajectory_for_layer(
        model, (test_dataset[0][0] * 0.3081 + 0.1307).clamp(0, 1),
        layer_idx=layer_idx, spatial_position=(14, 14),
        num_angles=num_angles, device=device,
    )
    plot_trajectory_grid(
        primary_data,
        results_dir / f"trajectory_grid_layer{layer_idx}.png",
        layer_idx=layer_idx,
    )
    plot_trajectory_summary(
        primary_data,
        results_dir / f"trajectory_summary_layer{layer_idx}.png",
        layer_idx=layer_idx,
    )

    # Aggregate summary
    aggregate = {
        "layer_idx": layer_idx,
        "num_digits": num_digits,
        "spatial_positions": [list(p) for p in spatial_positions],
        "num_angles": num_angles,
        "per_digit_summaries": all_summaries,
    }
    save_json(aggregate, results_dir / "trajectory.json")

    return aggregate
