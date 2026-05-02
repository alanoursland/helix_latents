"""
Tests for the MNIST HelixLayer experiment.

Usage:
  cd src
  python mnist_test_all.py           # fast tests (no MNIST download)
  python mnist_test_all.py --slow    # includes tiny-batch overfit tests (needs MNIST)
  python mnist_test_all.py --data    # includes data loading tests (needs MNIST)
"""

from __future__ import annotations

import argparse
import sys
import traceback

import torch
import torch.nn.functional as F

from mnist_config import MNISTConfig, get_device, resolve_data_dir
from mnist_data import make_mnist_dataloaders
from mnist_models import (
    CircleLayer,
    CircleMLP,
    HelixLayer,
    HelixMLP,
    StandardMLP,
    build_mnist_model,
    count_parameters,
)
from utils import set_seed


# ─── Layer Shape Tests ────────────────────────────────────────────────────────


def test_helix_layer_forward_shape():
    layer = HelixLayer(input_dim=784, units=16, output_dim=32)
    x = torch.randn(8, 784)
    y = layer(x)
    assert y.shape == (8, 32), f"Expected (8,32), got {y.shape}"


def test_circle_layer_forward_shape():
    layer = CircleLayer(input_dim=784, units=16, output_dim=32)
    x = torch.randn(8, 784)
    y = layer(x)
    assert y.shape == (8, 32), f"Expected (8,32), got {y.shape}"


def test_helix_layer_small_input():
    layer = HelixLayer(input_dim=32, units=8, output_dim=16)
    x = torch.randn(4, 32)
    y = layer(x)
    assert y.shape == (4, 16)


def test_circle_layer_small_input():
    layer = CircleLayer(input_dim=32, units=8, output_dim=16)
    x = torch.randn(4, 32)
    y = layer(x)
    assert y.shape == (4, 16)


# ─── No NaN Tests ────────────────────────────────────────────────────────────


def test_helix_layer_no_nans():
    layer = HelixLayer(input_dim=784, units=16, output_dim=32)
    for scale in [0.01, 1.0, 10.0]:
        x = torch.randn(8, 784) * scale
        y = layer(x)
        assert torch.isfinite(y).all(), f"NaN/Inf at scale={scale}"


def test_circle_layer_no_nans():
    layer = CircleLayer(input_dim=784, units=16, output_dim=32)
    for scale in [0.01, 1.0, 10.0]:
        x = torch.randn(8, 784) * scale
        y = layer(x)
        assert torch.isfinite(y).all(), f"NaN/Inf at scale={scale}"


def test_helix_layer_zero_input():
    """Edge case: all-zeros should still produce finite output (eps protects)."""
    layer = HelixLayer(input_dim=32, units=8, output_dim=16)
    x = torch.zeros(4, 32)
    y = layer(x)
    assert torch.isfinite(y).all()


# ─── Model Shape Tests ───────────────────────────────────────────────────────


def test_standard_mlp_forward_shape():
    model = StandardMLP(hidden_dim=32, num_layers=2)
    images = torch.randn(8, 1, 28, 28)
    logits = model(images)
    assert logits.shape == (8, 10)


def test_circle_mlp_forward_shape():
    model = CircleMLP(hidden_dim=32, units=8, num_layers=2)
    images = torch.randn(8, 1, 28, 28)
    logits = model(images)
    assert logits.shape == (8, 10)


def test_helix_mlp_forward_shape():
    model = HelixMLP(hidden_dim=32, units=8, num_layers=2)
    images = torch.randn(8, 1, 28, 28)
    logits = model(images)
    assert logits.shape == (8, 10)


def test_all_models_no_nans():
    images = torch.randn(16, 1, 28, 28)
    for ModelCls, kwargs in [
        (StandardMLP, {"hidden_dim": 32}),
        (CircleMLP, {"hidden_dim": 32, "units": 8}),
        (HelixMLP, {"hidden_dim": 32, "units": 8}),
    ]:
        model = ModelCls(**kwargs)
        logits = model(images)
        assert torch.isfinite(logits).all(), f"NaN in {ModelCls.__name__}"


# ─── Backward Pass Tests ─────────────────────────────────────────────────────


def test_standard_mlp_backward():
    model = StandardMLP(hidden_dim=32)
    images = torch.randn(16, 1, 28, 28)
    targets = torch.randint(0, 10, (16,))
    logits = model(images)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_circle_mlp_backward():
    model = CircleMLP(hidden_dim=32, units=8)
    images = torch.randn(16, 1, 28, 28)
    targets = torch.randint(0, 10, (16,))
    logits = model(images)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_helix_mlp_backward():
    model = HelixMLP(hidden_dim=32, units=8)
    images = torch.randn(16, 1, 28, 28)
    targets = torch.randint(0, 10, (16,))
    logits = model(images)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_helix_gradients_finite():
    model = HelixMLP(hidden_dim=32, units=8)
    images = torch.randn(16, 1, 28, 28)
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
        (StandardMLP, {"hidden_dim": 32}),
        (CircleMLP, {"hidden_dim": 32, "units": 8}),
        (HelixMLP, {"hidden_dim": 32, "units": 8}),
    ]:
        model = ModelCls(**kwargs)
        assert count_parameters(model) > 0, f"{ModelCls.__name__} has 0 params"


def test_build_mnist_model_factory():
    for mt in ["standard_mlp", "standard_mlp_matched", "circle_mlp", "helix_mlp"]:
        config = MNISTConfig(model_type=mt, hidden_dim=32, helix_units=8, circle_units=8)
        model = build_mnist_model(config)
        images = torch.randn(4, 1, 28, 28)
        logits = model(images)
        assert logits.shape == (4, 10), f"Failed for {mt}"


# ─── Config Tests ────────────────────────────────────────────────────────────


def test_config_defaults():
    config = MNISTConfig()
    assert config.model_type == "helix_mlp"
    assert config.epochs == 20
    assert config.hidden_dim == 128


def test_config_to_dict():
    config = MNISTConfig()
    d = config.to_dict()
    assert isinstance(d, dict)
    assert "model_type" in d
    assert "epochs" in d


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


# ─── Data Tests (require MNIST download) ─────────────────────────────────────


def test_mnist_dataloader_keys():
    config = MNISTConfig(batch_size=8)
    loaders = make_mnist_dataloaders(config)
    assert "train" in loaders
    assert "val" in loaders
    assert "test" in loaders


def test_mnist_batch_shapes():
    config = MNISTConfig(batch_size=8)
    loaders = make_mnist_dataloaders(config)
    images, targets = next(iter(loaders["train"]))
    assert images.shape == (8, 1, 28, 28), f"Got {images.shape}"
    assert targets.shape == (8,), f"Got {targets.shape}"


def test_mnist_split_sizes():
    config = MNISTConfig(val_size=5000)
    loaders = make_mnist_dataloaders(config)
    assert len(loaders["train"].dataset) == 55000
    assert len(loaders["val"].dataset) == 5000
    assert len(loaders["test"].dataset) == 10000


# ─── Slow Tests (tiny batch overfit) ─────────────────────────────────────────


def test_overfit_tiny_batch_standard():
    """StandardMLP should memorize 128 MNIST examples."""
    config = MNISTConfig(batch_size=128)
    loaders = make_mnist_dataloaders(config)
    images, targets = next(iter(loaders["train"]))

    model = StandardMLP(hidden_dim=128, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(200):
        logits = model(images)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        acc = (model(images).argmax(-1) == targets).float().mean().item()
    assert acc >= 0.90, f"StandardMLP overfit acc={acc:.2f}, expected >= 0.90"


def test_overfit_tiny_batch_circle():
    """CircleMLP should memorize 128 MNIST examples."""
    config = MNISTConfig(batch_size=128)
    loaders = make_mnist_dataloaders(config)
    images, targets = next(iter(loaders["train"]))

    model = CircleMLP(hidden_dim=128, units=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(300):
        logits = model(images)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        acc = (model(images).argmax(-1) == targets).float().mean().item()
    assert acc >= 0.80, f"CircleMLP overfit acc={acc:.2f}, expected >= 0.80"


def test_overfit_tiny_batch_helix():
    """HelixMLP should memorize 128 MNIST examples."""
    config = MNISTConfig(batch_size=128)
    loaders = make_mnist_dataloaders(config)
    images, targets = next(iter(loaders["train"]))

    model = HelixMLP(hidden_dim=128, units=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(300):
        logits = model(images)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        acc = (model(images).argmax(-1) == targets).float().mean().item()
    assert acc >= 0.80, f"HelixMLP overfit acc={acc:.2f}, expected >= 0.80"


# ─── Runner ───────────────────────────────────────────────────────────────────


def run_tests():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slow", action="store_true", help="Run slow overfit tests (needs MNIST)")
    parser.add_argument("--data", action="store_true", help="Run data loading tests (needs MNIST)")
    args = parser.parse_args()

    # Fast tests (no MNIST needed)
    fast_tests = [
        test_helix_layer_forward_shape,
        test_circle_layer_forward_shape,
        test_helix_layer_small_input,
        test_circle_layer_small_input,
        test_helix_layer_no_nans,
        test_circle_layer_no_nans,
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
        test_build_mnist_model_factory,
        test_config_defaults,
        test_config_to_dict,
        test_resolve_data_dir_cli,
        test_resolve_data_dir_default,
    ]

    data_tests = [
        test_mnist_dataloader_keys,
        test_mnist_batch_shapes,
        test_mnist_split_sizes,
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
