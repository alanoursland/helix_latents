"""
Rotated MNIST Filter Analysis (Measurement 2)
===============================================

Analyzes whether (W_u, W_v) filter pairs self-organized into rotated
quadrature pairs. This is the central analysis of the experiment.

For each helix/circle unit in the first conv layer:
  1. Extract W_u and W_v filters.
  2. Sweep rotation angles phi to find argmax correlation.
  3. Record phi_star and correlation at phi_star.

Produces:
  filter_pair_grid.png
  phi_star_histogram.png
  correlation_histogram.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate as scipy_rotate

from rot_mnist_config import save_json


def normalized_correlation(A: np.ndarray, B: np.ndarray) -> float:
    """Pearson correlation between two arrays, treating them as vectors."""
    A = A.astype(np.float64) - A.mean()
    B = B.astype(np.float64) - B.mean()
    denom = np.sqrt((A * A).sum() * (B * B).sum()) + 1e-12
    return float((A * B).sum() / denom)


def rotate_filter(filt: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a 2D filter by angle_deg using bilinear interpolation."""
    return scipy_rotate(filt, angle_deg, reshape=False, order=1, mode="constant", cval=0.0)


def sweep_phi_star_for_unit(
    W_u: np.ndarray,
    W_v: np.ndarray,
    angle_step: float = 5.0,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Find the rotation angle that maximizes correlation between W_u and rotate(W_v, phi).

    Returns:
        phi_star: angle in degrees that maximizes correlation
        corr_star: correlation at phi_star
        angles: array of tested angles
        correlations: array of correlation values
    """
    angles = np.arange(0, 360, angle_step)
    correlations = np.array([
        normalized_correlation(W_u, rotate_filter(W_v, phi))
        for phi in angles
    ])
    best_idx = int(np.argmax(correlations))
    return float(angles[best_idx]), float(correlations[best_idx]), angles, correlations


def analyze_layer_filters(
    model,
    layer_idx: int = 0,
    layer_type: str = "helix",
    angle_step: float = 5.0,
) -> dict:
    """Analyze filter pairs in the first HelixConv or CircleConv layer.

    Args:
        model: HelixCNN or CircleCNN model.
        layer_idx: Which conv block to analyze.
        layer_type: "helix" or "circle".
        angle_step: Rotation step in degrees for the sweep.

    Returns:
        dict with phi_stars, corr_stars, per-unit data, and raw weights.
    """
    conv_block = model.conv_blocks[layer_idx]
    W_u = conv_block.conv_u.weight.detach().cpu().numpy()  # [units, in_ch, k, k]
    W_v = conv_block.conv_v.weight.detach().cpu().numpy()

    num_units = W_u.shape[0]
    in_channels = W_u.shape[1]

    phi_stars = []
    corr_stars = []
    unit_data = []

    for u in range(num_units):
        # For single-channel input, use in_channel=0
        # For multi-channel, average across input channels
        if in_channels == 1:
            wu = W_u[u, 0]
            wv = W_v[u, 0]
        else:
            wu = W_u[u].mean(axis=0)
            wv = W_v[u].mean(axis=0)

        phi_star, corr_star, angles, correlations = sweep_phi_star_for_unit(
            wu, wv, angle_step=angle_step
        )
        phi_stars.append(phi_star)
        corr_stars.append(corr_star)
        unit_data.append({
            "unit": u,
            "phi_star": phi_star,
            "corr_star": corr_star,
        })

    return {
        "phi_stars": phi_stars,
        "corr_stars": corr_stars,
        "unit_data": unit_data,
        "W_u": W_u,
        "W_v": W_v,
        "num_units": num_units,
        "layer_idx": layer_idx,
        "layer_type": layer_type,
    }


def plot_filter_pair_grid(
    analysis: dict,
    output_path: str | Path,
) -> None:
    """Visualize W_u and W_v side by side for each unit, sorted by phi_star."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    W_u = analysis["W_u"]
    W_v = analysis["W_v"]
    num_units = analysis["num_units"]
    phi_stars = analysis["phi_stars"]
    corr_stars = analysis["corr_stars"]

    # Sort by phi_star
    order = np.argsort(phi_stars)

    cols = min(num_units, 8)
    rows = (num_units + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 2.5, rows * 1.5))
    if rows == 1:
        axes = axes[np.newaxis, :]

    for ax in axes.flat:
        ax.axis("off")

    for pos, idx in enumerate(order):
        row = pos // cols
        col = pos % cols

        in_ch = W_u.shape[1]
        wu = W_u[idx, 0] if in_ch == 1 else W_u[idx].mean(axis=0)
        wv = W_v[idx, 0] if in_ch == 1 else W_v[idx].mean(axis=0)

        vmax = max(abs(wu).max(), abs(wv).max())

        ax_u = axes[row, col * 2]
        ax_v = axes[row, col * 2 + 1]

        ax_u.imshow(wu, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax_v.imshow(wv, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax_u.set_title(f"u{idx}", fontsize=7)
        ax_v.set_title(f"v{idx} φ*={phi_stars[idx]:.0f}° r={corr_stars[idx]:.2f}", fontsize=6)

    fig.suptitle(
        f"Filter Pairs (Layer {analysis['layer_idx']}, {analysis['layer_type']})\n"
        f"Sorted by φ*",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_phi_star_histogram(
    analysis: dict,
    output_path: str | Path,
) -> None:
    """Histogram of phi_star across all units."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    phi_stars = analysis["phi_stars"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(phi_stars, bins=np.arange(0, 365, 15), edgecolor="black", alpha=0.7)
    ax.axvline(x=90, color="red", linestyle="--", alpha=0.5, label="90° (quadrature)")
    ax.set_xlabel("φ* (degrees)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Filter Pair Rotation Angle φ*\n"
        f"Layer {analysis['layer_idx']}, {analysis['layer_type']}, "
        f"N={analysis['num_units']} units"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_correlation_histogram(
    analysis: dict,
    output_path: str | Path,
) -> None:
    """Histogram of best correlation values across all units."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    corr_stars = analysis["corr_stars"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(corr_stars, bins=20, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Best Correlation")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Filter Pair Correlation at φ*\n"
        f"Layer {analysis['layer_idx']}, {analysis['layer_type']}, "
        f"N={analysis['num_units']} units"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_filter_analysis(
    model,
    results_dir: str | Path,
    model_type: str = "helix_conv",
) -> dict:
    """Run the full filter analysis pipeline and save figures + JSON."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    layer_type = "helix" if "helix" in model_type else "circle"

    analysis = analyze_layer_filters(model, layer_idx=0, layer_type=layer_type)

    plot_filter_pair_grid(analysis, results_dir / "filter_pair_grid.png")
    plot_phi_star_histogram(analysis, results_dir / "phi_star_histogram.png")
    plot_correlation_histogram(analysis, results_dir / "correlation_histogram.png")

    # Save JSON summary (without numpy arrays)
    summary = {
        "layer_idx": analysis["layer_idx"],
        "layer_type": analysis["layer_type"],
        "num_units": analysis["num_units"],
        "phi_stars": analysis["phi_stars"],
        "corr_stars": analysis["corr_stars"],
        "mean_corr_star": float(np.mean(analysis["corr_stars"])),
        "fraction_near_90": float(
            np.mean([80 <= p <= 100 for p in analysis["phi_stars"]])
        ),
    }
    save_json(summary, results_dir / "filter_analysis.json")

    return summary
