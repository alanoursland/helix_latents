"""
Rotated MNIST HelixConv Tests
==============================

Usage:
  cd src
  python rot_mnist_test_all.py               # fast tests (no data download)
  python rot_mnist_test_all.py --data         # + data tests (downloads MNIST)
  python rot_mnist_test_all.py --slow         # + overfit tests (GPU recommended)
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from rot_mnist_config import RotMNISTConfig, SCALE_PRESETS, apply_scale_preset, get_device, save_json
from rot_mnist_models import (
    StandardCNN, CircleConv2d, HelixConv2d, CircleCNN, HelixCNN,
    build_rot_mnist_model, count_parameters,
)
from rot_mnist_analyze_filters import normalized_correlation, rotate_filter, sweep_phi_star_for_unit
from rot_mnist_analyze_trajectory import (
    compute_circularity_score, compute_radius_variance, compute_winding_number,
)
from utils import set_seed


# ===== Config Tests =====

def test_config_defaults():
    c = RotMNISTConfig()
    assert c.model_type == "helix_conv"
    assert c.scale == "small"
    assert c.batch_size == 128
    assert c.epochs == 30
    assert c.learning_rate == 1e-3
    assert c.input_channels == 1
    assert c.num_classes == 10
    assert c.kernel_size == 5
    assert c.num_conv_blocks == 2
    assert c.rotation_max_degrees == 180

def test_config_to_dict():
    c = RotMNISTConfig()
    d = c.to_dict()
    assert isinstance(d, dict)
    assert d["model_type"] == "helix_conv"
    assert d["kernel_size"] == 5

def test_apply_scale_preset_small():
    c = RotMNISTConfig(scale="small")
    apply_scale_preset(c)
    assert c.hidden_channels == 32
    assert c.helix_units == 16
    assert c.circle_units == 16
    assert c.matched_hidden_channels == 48

def test_apply_scale_preset_medium():
    c = RotMNISTConfig(scale="medium")
    apply_scale_preset(c)
    assert c.hidden_channels == 64
    assert c.helix_units == 32
    assert c.circle_units == 32
    assert c.matched_hidden_channels == 96

def test_apply_scale_preset_large():
    c = RotMNISTConfig(scale="large")
    apply_scale_preset(c)
    assert c.hidden_channels == 128
    assert c.helix_units == 64
    assert c.circle_units == 64
    assert c.matched_hidden_channels == 192

def test_get_device_falls_back_to_cpu():
    dev = get_device("cpu")
    assert dev == torch.device("cpu")

def test_default_kernel_size_is_5():
    c = RotMNISTConfig()
    assert c.kernel_size == 5


# ===== Model Tests =====

def test_standard_cnn_forward_shape():
    model = StandardCNN(input_channels=1, num_classes=10, hidden_channels=16, kernel_size=5)
    x = torch.randn(4, 1, 28, 28)
    y = model(x)
    assert y.shape == (4, 10)

def test_standard_cnn_no_nans():
    model = StandardCNN(input_channels=1, num_classes=10, hidden_channels=16)
    for scale in [0.01, 1.0, 10.0]:
        x = torch.randn(4, 1, 28, 28) * scale
        y = model(x)
        assert torch.isfinite(y).all(), f"NaN/Inf at scale {scale}"

def test_standard_cnn_backward_pass():
    model = StandardCNN(input_channels=1, num_classes=10, hidden_channels=16)
    x = torch.randn(4, 1, 28, 28)
    targets = torch.randint(0, 10, (4,))
    loss = F.cross_entropy(model(x), targets)
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None

def test_circle_conv_forward_shape():
    layer = CircleConv2d(in_channels=1, units=8, out_channels=16, kernel_size=5, padding=2)
    x = torch.randn(4, 1, 28, 28)
    y = layer(x)
    assert y.shape == (4, 16, 28, 28)

def test_circle_conv_no_nans():
    layer = CircleConv2d(in_channels=1, units=8, out_channels=16, kernel_size=5, padding=2)
    for scale in [0.01, 1.0, 10.0]:
        x = torch.randn(4, 1, 28, 28) * scale
        y = layer(x)
        assert torch.isfinite(y).all(), f"NaN/Inf at scale {scale}"

def test_circle_conv_backward_pass():
    layer = CircleConv2d(in_channels=1, units=8, out_channels=16, kernel_size=5, padding=2)
    x = torch.randn(4, 1, 28, 28)
    y = layer(x)
    y.sum().backward()
    assert layer.conv_u.weight.grad is not None
    assert layer.conv_v.weight.grad is not None
    assert layer.project.weight.grad is not None

def test_circle_conv_filters_independent_after_init():
    layer = CircleConv2d(in_channels=1, units=8, out_channels=16, kernel_size=5, padding=2)
    wu = layer.conv_u.weight.detach().numpy().flatten()
    wv = layer.conv_v.weight.detach().numpy().flatten()
    corr = np.corrcoef(wu, wv)[0, 1]
    assert abs(corr) < 0.5, f"Filters suspiciously correlated: {corr:.3f}"

def test_helix_conv_forward_shape():
    layer = HelixConv2d(in_channels=1, units=8, out_channels=16, kernel_size=5, padding=2)
    x = torch.randn(4, 1, 28, 28)
    y = layer(x)
    assert y.shape == (4, 16, 28, 28)

def test_helix_conv_no_nans():
    layer = HelixConv2d(in_channels=1, units=8, out_channels=16, kernel_size=5, padding=2)
    for scale in [0.01, 1.0, 10.0]:
        x = torch.randn(4, 1, 28, 28) * scale
        y = layer(x)
        assert torch.isfinite(y).all(), f"NaN/Inf at scale {scale}"

def test_helix_conv_backward_pass():
    layer = HelixConv2d(in_channels=1, units=8, out_channels=16, kernel_size=5, padding=2)
    x = torch.randn(4, 1, 28, 28)
    y = layer(x)
    y.sum().backward()
    assert layer.conv_u.weight.grad is not None
    assert layer.conv_v.weight.grad is not None
    assert layer.conv_w.weight.grad is not None
    assert layer.project.weight.grad is not None

def test_helix_conv_forward_with_intermediates():
    layer = HelixConv2d(in_channels=1, units=8, out_channels=16, kernel_size=5, padding=2)
    x = torch.randn(4, 1, 28, 28)
    y, intermediates = layer.forward_with_intermediates(x)
    assert y.shape == (4, 16, 28, 28)
    assert intermediates["a"].shape == (4, 8, 28, 28)
    assert intermediates["b"].shape == (4, 8, 28, 28)
    assert intermediates["z"].shape == (4, 8, 28, 28)
    assert intermediates["r"].shape == (4, 8, 28, 28)
    assert torch.isfinite(y).all()
    assert torch.isfinite(intermediates["r"]).all()
    assert (intermediates["r"] >= 0).all()

def test_helix_conv_filters_independent_after_init():
    layer = HelixConv2d(in_channels=1, units=8, out_channels=16, kernel_size=5, padding=2)
    wu = layer.conv_u.weight.detach().numpy().flatten()
    wv = layer.conv_v.weight.detach().numpy().flatten()
    corr = np.corrcoef(wu, wv)[0, 1]
    assert abs(corr) < 0.5, f"Filters suspiciously correlated: {corr:.3f}"

def test_circle_cnn_forward_shape():
    model = CircleCNN(input_channels=1, num_classes=10, circle_units=8, hidden_channels=16)
    x = torch.randn(4, 1, 28, 28)
    y = model(x)
    assert y.shape == (4, 10)

def test_circle_cnn_no_nans():
    model = CircleCNN(input_channels=1, num_classes=10, circle_units=8, hidden_channels=16)
    x = torch.randn(4, 1, 28, 28)
    y = model(x)
    assert torch.isfinite(y).all()

def test_circle_cnn_backward_pass():
    model = CircleCNN(input_channels=1, num_classes=10, circle_units=8, hidden_channels=16)
    x = torch.randn(4, 1, 28, 28)
    targets = torch.randint(0, 10, (4,))
    loss = F.cross_entropy(model(x), targets)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"

def test_helix_cnn_forward_shape():
    model = HelixCNN(input_channels=1, num_classes=10, helix_units=8, hidden_channels=16)
    x = torch.randn(4, 1, 28, 28)
    y = model(x)
    assert y.shape == (4, 10)

def test_helix_cnn_no_nans():
    model = HelixCNN(input_channels=1, num_classes=10, helix_units=8, hidden_channels=16)
    x = torch.randn(4, 1, 28, 28)
    y = model(x)
    assert torch.isfinite(y).all()

def test_helix_cnn_backward_pass():
    model = HelixCNN(input_channels=1, num_classes=10, helix_units=8, hidden_channels=16)
    x = torch.randn(4, 1, 28, 28)
    targets = torch.randint(0, 10, (4,))
    loss = F.cross_entropy(model(x), targets)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"

def test_helix_cnn_forward_with_intermediates_at_layer():
    model = HelixCNN(input_channels=1, num_classes=10, helix_units=8, hidden_channels=16)
    x = torch.randn(4, 1, 28, 28)
    logits, intermediates = model.forward_with_intermediates_at_layer(x, layer_idx=0)
    assert logits.shape == (4, 10)
    assert intermediates["a"].shape == (4, 8, 28, 28)

def test_count_parameters_positive():
    for mt in ["standard_cnn", "standard_cnn_matched", "circle_conv", "helix_conv"]:
        config = RotMNISTConfig(model_type=mt)
        apply_scale_preset(config)
        model = build_rot_mnist_model(config)
        n = count_parameters(model)
        assert n > 0, f"{mt}: {n} parameters"

def test_build_all_model_types():
    for mt in ["standard_cnn", "standard_cnn_matched", "circle_conv", "helix_conv"]:
        config = RotMNISTConfig(model_type=mt)
        apply_scale_preset(config)
        model = build_rot_mnist_model(config)
        x = torch.randn(2, 1, 28, 28)
        y = model(x)
        assert y.shape == (2, 10), f"{mt}: got {y.shape}"

def test_all_models_random_stress_no_nans():
    for mt in ["standard_cnn", "circle_conv", "helix_conv"]:
        config = RotMNISTConfig(model_type=mt)
        apply_scale_preset(config)
        model = build_rot_mnist_model(config)
        for scale in [0.01, 1.0, 10.0]:
            x = torch.randn(8, 1, 28, 28) * scale
            y = model(x)
            assert torch.isfinite(y).all(), f"{mt} NaN at scale {scale}"

def test_synthetic_training_step():
    config = RotMNISTConfig(model_type="helix_conv")
    apply_scale_preset(config)
    model = build_rot_mnist_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(8, 1, 28, 28)
    targets = torch.randint(0, 10, (8,))

    logits = model(x)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)


# ===== Filter Analysis Pipeline Tests =====

def test_normalized_correlation_self_is_one():
    A = np.random.randn(5, 5)
    corr = normalized_correlation(A, A)
    assert abs(corr - 1.0) < 1e-6, f"Self-correlation: {corr}"

def test_normalized_correlation_orthogonal_is_zero():
    # Use arrays that are truly orthogonal under Pearson correlation
    A = np.array([[1, -1], [-1, 1]], dtype=float)
    B = np.array([[1, 1], [-1, -1]], dtype=float)
    corr = normalized_correlation(A, B)
    assert abs(corr) < 1e-6, f"Orthogonal correlation: {corr}"

def test_filter_rotation_90_recovers_original_after_4_steps():
    np.random.seed(42)
    A = np.random.randn(5, 5)
    rotated = A.copy()
    for _ in range(4):
        rotated = rotate_filter(rotated, 90.0)
    corr = normalized_correlation(A, rotated)
    assert corr > 0.9, f"4x90 rotation correlation: {corr}"

def test_phi_star_recovers_known_rotation():
    np.random.seed(42)
    # Make a filter with clear oriented pattern
    A = np.zeros((7, 7))
    A[3, :] = 1.0  # horizontal bar
    A[2, :] = 0.5
    A[4, :] = 0.5

    B = rotate_filter(A, 90.0)
    phi_star, corr_star, _, _ = sweep_phi_star_for_unit(A, B, angle_step=5.0)
    assert 80 <= phi_star <= 100, f"phi_star: {phi_star}"
    assert corr_star > 0.8, f"corr_star: {corr_star}"

def test_phi_star_handles_sign_flip_correctly():
    # For B = -A, all correlations are negated: corr(A, rotate(-A, phi)) = -corr(A, rotate(A, phi)).
    # The best correlation should be negative or near zero (since self-corr at phi=0 is -1).
    np.random.seed(42)
    A = np.random.randn(5, 5)
    B = -A
    phi_star, corr_star, _, correlations = sweep_phi_star_for_unit(A, B, angle_step=5.0)
    # corr_star should be substantially below 1.0 — sign flip cannot produce clean pairing
    assert corr_star < 0.5, \
        f"Sign flip produced suspiciously high correlation: corr_star={corr_star}"


# ===== Trajectory Analysis Tests =====

def test_circularity_score_perfect_circle_is_one():
    t = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    a = np.cos(t)
    b = np.sin(t)
    circ = compute_circularity_score(a, b)
    assert circ > 0.95, f"Circle circularity: {circ}"

def test_circularity_score_line_is_zero():
    a = np.linspace(-1, 1, 72)
    b = np.zeros(72)
    circ = compute_circularity_score(a, b)
    assert circ < 0.1, f"Line circularity: {circ}"

def test_radius_variance_constant_signal_is_zero():
    r = np.ones(72)
    rv = compute_radius_variance(r)
    assert rv < 1e-6, f"Constant radius variance: {rv}"

def test_winding_number_one_full_loop():
    t = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    a = np.cos(t)
    b = np.sin(t)
    wn = compute_winding_number(a, b)
    assert 0.9 < wn < 1.1, f"Winding number: {wn}"


# ===== Trajectory Spatial Clamping Tests =====

def test_trajectory_for_layer_clamps_spatial_position():
    """Layer 1 has 14x14 feature maps; position (14,14) should be clamped to (13,13)."""
    from rot_mnist_analyze_trajectory import trajectory_for_layer
    model = HelixCNN(input_channels=1, num_classes=10, helix_units=4, hidden_channels=8)
    base_image = torch.rand(1, 28, 28)
    # Layer 1 spatial dim is 14x14, so (14,14) is out of bounds without clamping
    result = trajectory_for_layer(
        model, base_image, layer_idx=1,
        spatial_position=(14, 14), num_angles=4, device=torch.device("cpu"),
    )
    assert result["a"].shape == (4, 4)  # [num_angles, units]

def test_trajectory_for_unit_clamps_spatial_position():
    """Same test for the per-unit variant."""
    from rot_mnist_analyze_trajectory import trajectory_for_unit
    model = HelixCNN(input_channels=1, num_classes=10, helix_units=4, hidden_channels=8)
    base_image = torch.rand(1, 28, 28)
    result = trajectory_for_unit(
        model, base_image, layer_idx=1, unit_idx=0,
        spatial_position=(20, 20), num_angles=4, device=torch.device("cpu"),
    )
    assert result["a"].shape == (4,)  # [num_angles]

def test_trajectory_for_layer_works_at_all_layers():
    """Trajectory analysis should work at every conv block without index errors."""
    from rot_mnist_analyze_trajectory import trajectory_for_layer
    model = HelixCNN(input_channels=1, num_classes=10, helix_units=4, hidden_channels=8)
    base_image = torch.rand(1, 28, 28)
    for layer_idx in range(model.num_conv_blocks):
        result = trajectory_for_layer(
            model, base_image, layer_idx=layer_idx,
            spatial_position=(14, 14), num_angles=4, device=torch.device("cpu"),
        )
        assert result["a"].ndim == 2

def test_circle_cnn_trajectory_for_layer():
    """CircleCNN trajectory should work and produce zeros for z."""
    from rot_mnist_analyze_trajectory import trajectory_for_layer
    model = CircleCNN(input_channels=1, num_classes=10, circle_units=4, hidden_channels=8)
    base_image = torch.rand(1, 28, 28)
    result = trajectory_for_layer(
        model, base_image, layer_idx=0,
        spatial_position=(14, 14), num_angles=4, device=torch.device("cpu"),
    )
    assert result["a"].shape == (4, 4)
    assert np.allclose(result["z"], 0.0), "CircleCNN should have zero z values"


# ===== Data Tests (require download) =====

def test_mnist_loads():
    from rot_mnist_data import make_rot_mnist_dataloaders
    config = RotMNISTConfig()
    config.limit_train_batches = 1
    config.limit_eval_batches = 1
    dataloaders = make_rot_mnist_dataloaders(config)
    assert "train" in dataloaders
    assert "val" in dataloaders
    assert "test_rotated" in dataloaders
    assert "test_unrotated" in dataloaders

def test_split_sizes():
    from rot_mnist_data import make_rot_mnist_dataloaders
    config = RotMNISTConfig()
    dataloaders = make_rot_mnist_dataloaders(config)
    assert len(dataloaders["train"].dataset) == 55000
    assert len(dataloaders["val"].dataset) == 5000
    assert len(dataloaders["test_rotated"].dataset) == 10000
    assert len(dataloaders["test_unrotated"].dataset) == 10000

def test_split_deterministic():
    from rot_mnist_data import make_rot_mnist_dataloaders
    config = RotMNISTConfig()
    dl1 = make_rot_mnist_dataloaders(config)
    dl2 = make_rot_mnist_dataloaders(config)
    assert len(dl1["train"].dataset) == len(dl2["train"].dataset)
    assert len(dl1["val"].dataset) == len(dl2["val"].dataset)

def test_batch_shapes():
    from rot_mnist_data import make_rot_mnist_dataloaders
    config = RotMNISTConfig(batch_size=32)
    dataloaders = make_rot_mnist_dataloaders(config)
    images, targets = next(iter(dataloaders["train"]))
    assert images.shape == (32, 1, 28, 28)
    assert targets.shape == (32,)

def test_train_transform_actually_rotates():
    from torchvision import datasets
    from rot_mnist_data import make_train_transform
    config = RotMNISTConfig()
    transform = make_train_transform(config)
    ds = datasets.MNIST(root=config.data_dir, train=True, download=True, transform=transform)
    # Load same image twice — should differ due to random rotation
    torch.manual_seed(0)
    img1, _ = ds[0]
    torch.manual_seed(1)
    img2, _ = ds[0]
    # They might be the same by chance, but very unlikely
    diff = (img1 - img2).abs().sum().item()
    assert diff > 0.01, "Train transform did not appear to rotate"

def test_unrotated_transform_does_not_rotate():
    from torchvision import datasets
    from rot_mnist_data import make_unrotated_test_transform
    transform = make_unrotated_test_transform()
    config = RotMNISTConfig()
    ds = datasets.MNIST(root=config.data_dir, train=False, download=True, transform=transform)
    torch.manual_seed(0)
    img1, _ = ds[0]
    torch.manual_seed(1)
    img2, _ = ds[0]
    diff = (img1 - img2).abs().sum().item()
    assert diff < 1e-6, "Unrotated transform changed between seeds"

def test_fixed_rotation_grid_shapes():
    from rot_mnist_data import make_fixed_rotation_grid
    base = torch.rand(1, 28, 28)
    images, angles = make_fixed_rotation_grid(base, num_angles=36)
    assert images.shape == (36, 1, 28, 28)
    assert angles.shape == (36,)

def test_fixed_rotation_grid_first_image_unrotated():
    from rot_mnist_data import make_fixed_rotation_grid
    base = torch.rand(1, 28, 28)
    images, angles = make_fixed_rotation_grid(base, num_angles=72)
    assert abs(angles[0].item()) < 1e-6, f"First angle: {angles[0]}"


# ===== Slow Overfit Tests =====

def _overfit_tiny_batch(model_type: str, threshold: float = 0.90, steps: int = 300):
    set_seed(0)
    config = RotMNISTConfig(model_type=model_type)
    apply_scale_preset(config)
    model = build_rot_mnist_model(config)
    device = get_device("cuda")
    model = model.to(device)

    # Fixed tiny batch
    x = torch.randn(128, 1, 28, 28, device=device)
    targets = torch.randint(0, 10, (128,), device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(steps):
        logits = model(x)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = model(x)
        acc = (logits.argmax(1) == targets).float().mean().item()

    assert acc >= threshold, \
        f"{model_type} overfit accuracy {acc:.4f} < {threshold} after {steps} steps"
    return acc

def test_overfit_tiny_batch_standard():
    acc = _overfit_tiny_batch("standard_cnn", threshold=0.90)
    print(f"  standard_cnn overfit acc: {acc:.4f}")

def test_overfit_tiny_batch_circle():
    acc = _overfit_tiny_batch("circle_conv", threshold=0.85)
    print(f"  circle_conv overfit acc: {acc:.4f}")

def test_overfit_tiny_batch_helix():
    acc = _overfit_tiny_batch("helix_conv", threshold=0.85)
    print(f"  helix_conv overfit acc: {acc:.4f}")


# ===== Runner =====

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", action="store_true", help="Run data tests (downloads MNIST)")
    parser.add_argument("--slow", action="store_true", help="Run slow overfit tests")
    args = parser.parse_args()

    fast_tests = [
        # Config
        test_config_defaults,
        test_config_to_dict,
        test_apply_scale_preset_small,
        test_apply_scale_preset_medium,
        test_apply_scale_preset_large,
        test_get_device_falls_back_to_cpu,
        test_default_kernel_size_is_5,
        # Standard CNN
        test_standard_cnn_forward_shape,
        test_standard_cnn_no_nans,
        test_standard_cnn_backward_pass,
        # CircleConv2d
        test_circle_conv_forward_shape,
        test_circle_conv_no_nans,
        test_circle_conv_backward_pass,
        test_circle_conv_filters_independent_after_init,
        # HelixConv2d
        test_helix_conv_forward_shape,
        test_helix_conv_no_nans,
        test_helix_conv_backward_pass,
        test_helix_conv_forward_with_intermediates,
        test_helix_conv_filters_independent_after_init,
        # CircleCNN
        test_circle_cnn_forward_shape,
        test_circle_cnn_no_nans,
        test_circle_cnn_backward_pass,
        # HelixCNN
        test_helix_cnn_forward_shape,
        test_helix_cnn_no_nans,
        test_helix_cnn_backward_pass,
        test_helix_cnn_forward_with_intermediates_at_layer,
        # Factory
        test_count_parameters_positive,
        test_build_all_model_types,
        test_all_models_random_stress_no_nans,
        test_synthetic_training_step,
        # Filter analysis pipeline
        test_normalized_correlation_self_is_one,
        test_normalized_correlation_orthogonal_is_zero,
        test_filter_rotation_90_recovers_original_after_4_steps,
        test_phi_star_recovers_known_rotation,
        test_phi_star_handles_sign_flip_correctly,
        # Trajectory analysis
        test_circularity_score_perfect_circle_is_one,
        test_circularity_score_line_is_zero,
        test_radius_variance_constant_signal_is_zero,
        test_winding_number_one_full_loop,
        # Trajectory spatial clamping
        test_trajectory_for_layer_clamps_spatial_position,
        test_trajectory_for_unit_clamps_spatial_position,
        test_trajectory_for_layer_works_at_all_layers,
        test_circle_cnn_trajectory_for_layer,
    ]

    data_tests = [
        test_mnist_loads,
        test_split_sizes,
        test_split_deterministic,
        test_batch_shapes,
        test_train_transform_actually_rotates,
        test_unrotated_transform_does_not_rotate,
        test_fixed_rotation_grid_shapes,
        test_fixed_rotation_grid_first_image_unrotated,
    ]

    slow_tests = [
        test_overfit_tiny_batch_standard,
        test_overfit_tiny_batch_circle,
        test_overfit_tiny_batch_helix,
    ]

    tests = fast_tests[:]
    if args.data or args.slow:
        tests.extend(data_tests)
    if args.slow:
        tests.extend(slow_tests)

    passed = 0
    failed = 0
    errors = []
    t0 = time.time()

    for test_fn in tests:
        name = test_fn.__name__
        try:
            test_fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            errors.append((name, e))
            print(f"  FAIL  {name}: {e}")
            traceback.print_exc()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print(f"Time: {elapsed:.1f}s")

    if errors:
        print(f"\nFailed tests:")
        for name, e in errors:
            print(f"  {name}: {e}")
        sys.exit(1)
    else:
        print("\nAll tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
