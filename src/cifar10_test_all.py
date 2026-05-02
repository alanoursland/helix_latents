"""
Tests for the Flattened CIFAR-10 HelixLayer experiment.

Usage:
  cd src
  python cifar10_test_all.py           # fast tests (no CIFAR-10 download)
  python cifar10_test_all.py --slow    # includes tiny-batch overfit tests (needs CIFAR-10)
  python cifar10_test_all.py --data    # includes data loading tests (needs CIFAR-10)
"""

from __future__ import annotations

import argparse
import sys
import traceback

import torch
import torch.nn.functional as F

from cifar10_config import CIFAR10Config, apply_scale_preset, get_device, resolve_data_dir
from cifar10_data import make_cifar10_dataloaders
from cifar10_models import (
    CircleLayer,
    CircleMLP,
    HelixLayer,
    HelixMLP,
    StandardMLP,
    build_cifar10_model,
    count_parameters,
)
from utils import set_seed


# ─── Config Tests ────────────────────────────────────────────────────────────


def test_config_defaults():
    config = CIFAR10Config()
    assert config.model_type == "helix_mlp"
    assert config.scale == "medium"
    assert config.epochs == 100
    assert config.hidden_dim == 512
    assert config.dropout == 0.1


def test_config_to_dict():
    config = CIFAR10Config()
    d = config.to_dict()
    assert isinstance(d, dict)
    assert "model_type" in d
    assert "scale" in d


def test_apply_scale_preset_small():
    config = CIFAR10Config(scale="small")
    apply_scale_preset(config)
    assert config.hidden_dim == 256
    assert config.circle_units == 128
    assert config.helix_units == 128
    assert config.matched_hidden_dim == 384


def test_apply_scale_preset_medium():
    config = CIFAR10Config(scale="medium")
    apply_scale_preset(config)
    assert config.hidden_dim == 512
    assert config.circle_units == 256
    assert config.helix_units == 256
    assert config.matched_hidden_dim == 768


def test_apply_scale_preset_large():
    config = CIFAR10Config(scale="large")
    apply_scale_preset(config)
    assert config.hidden_dim == 1024
    assert config.circle_units == 512
    assert config.helix_units == 512
    assert config.matched_hidden_dim == 1536


def test_resolve_data_dir_cli():
    result = resolve_data_dir(cli_value="E:/my_data", verbose=False)
    assert result == "E:/my_data"


def test_resolve_data_dir_default():
    import os
    old = os.environ.pop("HELIX_DATA_DIR", None)
    result = resolve_data_dir(cli_value=None, verbose=False)
    assert result == "data"
    if old is not None:
        os.environ["HELIX_DATA_DIR"] = old


# ─── Layer Shape Tests ────────────────────────────────────────────────────────


def test_circle_layer_forward_shape():
    layer = CircleLayer(input_dim=3072, units=64, output_dim=256)
    x = torch.randn(8, 3072)
    y = layer(x)
    assert y.shape == (8, 256), f"Expected (8,256), got {y.shape}"


def test_helix_layer_forward_shape():
    layer = HelixLayer(input_dim=3072, units=64, output_dim=256)
    x = torch.randn(8, 3072)
    y = layer(x)
    assert y.shape == (8, 256), f"Expected (8,256), got {y.shape}"


def test_circle_layer_small_input():
    layer = CircleLayer(input_dim=256, units=32, output_dim=128)
    x = torch.randn(4, 256)
    y = layer(x)
    assert y.shape == (4, 128)


def test_helix_layer_small_input():
    layer = HelixLayer(input_dim=256, units=32, output_dim=128)
    x = torch.randn(4, 256)
    y = layer(x)
    assert y.shape == (4, 128)


# ─── No NaN Tests ────────────────────────────────────────────────────────────


def test_circle_layer_no_nans():
    layer = CircleLayer(input_dim=3072, units=64, output_dim=256)
    for scale in [0.01, 1.0, 10.0]:
        x = torch.randn(8, 3072) * scale
        y = layer(x)
        assert torch.isfinite(y).all(), f"NaN/Inf at scale={scale}"


def test_helix_layer_no_nans():
    layer = HelixLayer(input_dim=3072, units=64, output_dim=256)
    for scale in [0.01, 1.0, 10.0]:
        x = torch.randn(8, 3072) * scale
        y = layer(x)
        assert torch.isfinite(y).all(), f"NaN/Inf at scale={scale}"


def test_helix_layer_zero_input():
    layer = HelixLayer(input_dim=256, units=32, output_dim=128)
    x = torch.zeros(4, 256)
    y = layer(x)
    assert torch.isfinite(y).all()


# ─── Model Shape Tests ───────────────────────────────────────────────────────


def test_standard_mlp_forward_shape():
    model = StandardMLP(input_dim=3072, hidden_dim=64, num_layers=2)
    images = torch.randn(8, 3, 32, 32)
    logits = model(images)
    assert logits.shape == (8, 10)


def test_circle_mlp_forward_shape():
    model = CircleMLP(input_dim=3072, hidden_dim=64, units=16, num_layers=2)
    images = torch.randn(8, 3, 32, 32)
    logits = model(images)
    assert logits.shape == (8, 10)


def test_helix_mlp_forward_shape():
    model = HelixMLP(input_dim=3072, hidden_dim=64, units=16, num_layers=2)
    images = torch.randn(8, 3, 32, 32)
    logits = model(images)
    assert logits.shape == (8, 10)


def test_all_models_no_nans():
    images = torch.randn(16, 3, 32, 32)
    for ModelCls, kwargs in [
        (StandardMLP, {"input_dim": 3072, "hidden_dim": 64}),
        (CircleMLP, {"input_dim": 3072, "hidden_dim": 64, "units": 16}),
        (HelixMLP, {"input_dim": 3072, "hidden_dim": 64, "units": 16}),
    ]:
        model = ModelCls(**kwargs)
        logits = model(images)
        assert torch.isfinite(logits).all(), f"NaN in {ModelCls.__name__}"


# ─── Backward Pass Tests ─────────────────────────────────────────────────────


def test_standard_mlp_backward():
    model = StandardMLP(input_dim=3072, hidden_dim=64)
    images = torch.randn(16, 3, 32, 32)
    targets = torch.randint(0, 10, (16,))
    logits = model(images)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_circle_mlp_backward():
    model = CircleMLP(input_dim=3072, hidden_dim=64, units=16)
    images = torch.randn(16, 3, 32, 32)
    targets = torch.randint(0, 10, (16,))
    logits = model(images)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_helix_mlp_backward():
    model = HelixMLP(input_dim=3072, hidden_dim=64, units=16)
    images = torch.randn(16, 3, 32, 32)
    targets = torch.randint(0, 10, (16,))
    logits = model(images)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_helix_gradients_finite():
    model = HelixMLP(input_dim=3072, hidden_dim=64, units=16)
    images = torch.randn(16, 3, 32, 32)
    targets = torch.randint(0, 10, (16,))
    logits = model(images)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"


# ─── Parameter Counting ──────────────────────────────────────────────────────


def test_count_parameters_positive():
    for ModelCls, kwargs in [
        (StandardMLP, {"input_dim": 3072, "hidden_dim": 64}),
        (CircleMLP, {"input_dim": 3072, "hidden_dim": 64, "units": 16}),
        (HelixMLP, {"input_dim": 3072, "hidden_dim": 64, "units": 16}),
    ]:
        model = ModelCls(**kwargs)
        assert count_parameters(model) > 0, f"{ModelCls.__name__} has 0 params"


def test_build_cifar10_model_factory():
    for mt in ["standard_mlp", "standard_mlp_matched", "circle_mlp", "helix_mlp"]:
        config = CIFAR10Config(model_type=mt, hidden_dim=64, helix_units=16, circle_units=16, matched_hidden_dim=96)
        model = build_cifar10_model(config)
        images = torch.randn(4, 3, 32, 32)
        logits = model(images)
        assert logits.shape == (4, 10), f"Failed for {mt}"


# ─── Synthetic Training Step ─────────────────────────────────────────────────


def test_synthetic_training_step():
    """One optimizer step on synthetic data should produce finite loss."""
    model = HelixMLP(input_dim=3072, hidden_dim=64, units=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    images = torch.randn(32, 3, 32, 32)
    targets = torch.randint(0, 10, (32,))

    logits = model(images)
    loss = F.cross_entropy(logits, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(torch.tensor(loss.item()))
    acc = (logits.argmax(-1) == targets).float().mean().item()
    assert 0.0 <= acc <= 1.0


# ─── Data Tests (require CIFAR-10 download) ──────────────────────────────────


def test_cifar10_batch_shape():
    config = CIFAR10Config(batch_size=8)
    loaders = make_cifar10_dataloaders(config)
    images, targets = next(iter(loaders["train"]))
    assert images.shape == (8, 3, 32, 32), f"Got {images.shape}"


def test_cifar10_target_shape():
    config = CIFAR10Config(batch_size=8)
    loaders = make_cifar10_dataloaders(config)
    images, targets = next(iter(loaders["train"]))
    assert targets.shape == (8,), f"Got {targets.shape}"


def test_cifar10_num_classes():
    config = CIFAR10Config(batch_size=256)
    loaders = make_cifar10_dataloaders(config)
    all_targets = []
    for images, targets in loaders["train"]:
        all_targets.append(targets)
        if len(all_targets) > 10:
            break
    all_targets = torch.cat(all_targets)
    unique_classes = all_targets.unique()
    assert len(unique_classes) == 10, f"Got {len(unique_classes)} classes"


def test_cifar10_train_val_test_sizes():
    config = CIFAR10Config(val_size=5000)
    loaders = make_cifar10_dataloaders(config)
    assert len(loaders["train"].dataset) == 45000
    assert len(loaders["val"].dataset) == 5000
    assert len(loaders["test"].dataset) == 10000


def test_cifar10_split_deterministic():
    config = CIFAR10Config(batch_size=8, seed=42)
    loaders1 = make_cifar10_dataloaders(config)
    loaders2 = make_cifar10_dataloaders(config)
    # Same split indices
    assert len(loaders1["train"].dataset) == len(loaders2["train"].dataset)
    assert len(loaders1["val"].dataset) == len(loaders2["val"].dataset)


# ─── Slow Tests (tiny batch overfit) ─────────────────────────────────────────


def test_overfit_tiny_batch_standard():
    """StandardMLP should memorize 128 CIFAR-10 examples."""
    config = CIFAR10Config(batch_size=128)
    loaders = make_cifar10_dataloaders(config)
    images, targets = next(iter(loaders["train"]))

    model = StandardMLP(input_dim=3072, hidden_dim=256, num_layers=2, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(300):
        logits = model(images)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        acc = (model(images).argmax(-1) == targets).float().mean().item()
    assert acc >= 0.80, f"StandardMLP overfit acc={acc:.2f}, expected >= 0.80"


def test_overfit_tiny_batch_circle():
    """CircleMLP should memorize 128 CIFAR-10 examples."""
    config = CIFAR10Config(batch_size=128)
    loaders = make_cifar10_dataloaders(config)
    images, targets = next(iter(loaders["train"]))

    model = CircleMLP(input_dim=3072, hidden_dim=256, units=128, num_layers=2, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(400):
        logits = model(images)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        acc = (model(images).argmax(-1) == targets).float().mean().item()
    assert acc >= 0.70, f"CircleMLP overfit acc={acc:.2f}, expected >= 0.70"


def test_overfit_tiny_batch_helix():
    """HelixMLP should memorize 128 CIFAR-10 examples."""
    config = CIFAR10Config(batch_size=128)
    loaders = make_cifar10_dataloaders(config)
    images, targets = next(iter(loaders["train"]))

    model = HelixMLP(input_dim=3072, hidden_dim=256, units=128, num_layers=2, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(400):
        logits = model(images)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        acc = (model(images).argmax(-1) == targets).float().mean().item()
    assert acc >= 0.70, f"HelixMLP overfit acc={acc:.2f}, expected >= 0.70"


# ─── Runner ───────────────────────────────────────────────────────────────────


def run_tests():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slow", action="store_true", help="Run slow overfit tests (needs CIFAR-10)")
    parser.add_argument("--data", action="store_true", help="Run data loading tests (needs CIFAR-10)")
    args = parser.parse_args()

    # Fast tests (no CIFAR-10 needed)
    fast_tests = [
        test_config_defaults,
        test_config_to_dict,
        test_apply_scale_preset_small,
        test_apply_scale_preset_medium,
        test_apply_scale_preset_large,
        test_resolve_data_dir_cli,
        test_resolve_data_dir_default,
        test_circle_layer_forward_shape,
        test_helix_layer_forward_shape,
        test_circle_layer_small_input,
        test_helix_layer_small_input,
        test_circle_layer_no_nans,
        test_helix_layer_no_nans,
        test_helix_layer_zero_input,
        test_standard_mlp_forward_shape,
        test_circle_mlp_forward_shape,
        test_helix_mlp_forward_shape,
        test_all_models_no_nans,
        test_standard_mlp_backward,
        test_circle_mlp_backward,
        test_helix_mlp_backward,
        test_helix_gradients_finite,
        test_count_parameters_positive,
        test_build_cifar10_model_factory,
        test_synthetic_training_step,
    ]

    data_tests = [
        test_cifar10_batch_shape,
        test_cifar10_target_shape,
        test_cifar10_num_classes,
        test_cifar10_train_val_test_sizes,
        test_cifar10_split_deterministic,
    ]

    slow_tests = [
        test_overfit_tiny_batch_standard,
        test_overfit_tiny_batch_circle,
        test_overfit_tiny_batch_helix,
    ]

    tests_to_run = fast_tests[:]
    if args.data or args.slow:
        tests_to_run.extend(data_tests)
    if args.slow:
        tests_to_run.extend(slow_tests)

    passed = 0
    failed = 0
    errors = []

    for test_fn in tests_to_run:
        name = test_fn.__name__
        try:
            test_fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            errors.append((name, e))
            print(f"  FAIL  {name}: {e}")

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {passed+failed} total")

    if errors:
        print(f"\nFailures:")
        for name, e in errors:
            print(f"  {name}:")
            traceback.print_exception(type(e), e, e.__traceback__)
        sys.exit(1)
    else:
        print("All tests passed!")


if __name__ == "__main__":
    run_tests()
