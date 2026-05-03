"""
Tests for Covertype tabular classification experiment.

Usage:
  cd src
  python covertype_test_all.py              # fast tests only (no dataset needed)
  python covertype_test_all.py --data       # include data tests (needs dataset)
  python covertype_test_all.py --slow       # include slow overfit tests (needs dataset)
"""

from __future__ import annotations

import argparse
import sys
import traceback

import numpy as np
import torch
import torch.nn.functional as F


# ─── Test Infrastructure ──────────────────────────────────────────────────────

_results: list[tuple[str, bool, str]] = []


def run_test(name: str, fn):
    try:
        fn()
        _results.append((name, True, ""))
        print(f"  PASS  {name}")
    except Exception as e:
        _results.append((name, False, str(e)))
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc()


def print_summary():
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed")
    if passed < total:
        print("\nFailed tests:")
        for name, ok, msg in _results:
            if not ok:
                print(f"  - {name}: {msg}")
    print(f"{'='*60}")
    return passed == total


# ─── Config Tests ─────────────────────────────────────────────────────────────

def test_config_defaults():
    from covertype_config import CovertypeConfig
    c = CovertypeConfig()
    assert c.input_dim == 54
    assert c.num_classes == 7
    assert c.batch_size == 1024
    assert c.epochs == 100
    assert c.learning_rate == 1e-3
    assert c.weight_decay == 1e-4
    assert c.dropout == 0.05
    assert c.use_layernorm is True
    assert c.num_layers == 2


def test_config_to_dict():
    from covertype_config import CovertypeConfig
    c = CovertypeConfig()
    d = c.to_dict()
    assert isinstance(d, dict)
    assert d["input_dim"] == 54
    assert d["num_classes"] == 7


def test_apply_scale_preset_small():
    from covertype_config import CovertypeConfig, apply_scale_preset
    c = CovertypeConfig()
    c.scale = "small"
    apply_scale_preset(c)
    assert c.hidden_dim == 64
    assert c.circle_units == 32
    assert c.helix_units == 32
    assert c.matched_hidden_dim == 96


def test_apply_scale_preset_medium():
    from covertype_config import CovertypeConfig, apply_scale_preset
    c = CovertypeConfig()
    c.scale = "medium"
    apply_scale_preset(c)
    assert c.hidden_dim == 128
    assert c.circle_units == 64
    assert c.helix_units == 64
    assert c.matched_hidden_dim == 192


def test_apply_scale_preset_large():
    from covertype_config import CovertypeConfig, apply_scale_preset
    c = CovertypeConfig()
    c.scale = "large"
    apply_scale_preset(c)
    assert c.hidden_dim == 256
    assert c.circle_units == 128
    assert c.helix_units == 128
    assert c.matched_hidden_dim == 384


def test_get_device_falls_back_to_cpu():
    from covertype_config import get_device
    device = get_device("cpu")
    assert device == torch.device("cpu")


# ─── Model Tests ──────────────────────────────────────────────────────────────

def test_standard_mlp_forward_shape():
    from covertype_models import StandardMLP
    model = StandardMLP(input_dim=54, num_classes=7, hidden_dim=64, num_layers=2)
    x = torch.randn(16, 54)
    out = model(x)
    assert out.shape == (16, 7)


def test_standard_mlp_no_nans():
    from covertype_models import StandardMLP
    model = StandardMLP(input_dim=54, num_classes=7, hidden_dim=64)
    x = torch.randn(16, 54)
    out = model(x)
    assert not torch.isnan(out).any()


def test_standard_mlp_backward_pass():
    from covertype_models import StandardMLP
    model = StandardMLP(input_dim=54, num_classes=7, hidden_dim=64)
    x = torch.randn(16, 54)
    targets = torch.randint(0, 7, (16,))
    logits = model(x)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None


def test_circle_layer_forward_shape():
    from covertype_models import CircleLayer
    layer = CircleLayer(input_dim=54, units=32, output_dim=64)
    x = torch.randn(16, 54)
    out = layer(x)
    assert out.shape == (16, 64)


def test_circle_layer_no_nans():
    from covertype_models import CircleLayer
    layer = CircleLayer(input_dim=54, units=32, output_dim=64)
    for scale in [0.01, 1.0, 10.0]:
        x = torch.randn(16, 54) * scale
        out = layer(x)
        assert not torch.isnan(out).any(), f"NaN at scale {scale}"


def test_circle_layer_backward_pass():
    from covertype_models import CircleLayer
    layer = CircleLayer(input_dim=54, units=32, output_dim=64)
    x = torch.randn(16, 54)
    out = layer(x)
    out.sum().backward()
    assert layer.u.grad is not None
    assert layer.v.grad is not None


def test_circle_mlp_forward_shape():
    from covertype_models import CircleMLP
    model = CircleMLP(input_dim=54, num_classes=7, hidden_dim=64, units=32)
    x = torch.randn(16, 54)
    out = model(x)
    assert out.shape == (16, 7)


def test_circle_mlp_no_nans():
    from covertype_models import CircleMLP
    model = CircleMLP(input_dim=54, num_classes=7, hidden_dim=64, units=32)
    x = torch.randn(16, 54)
    out = model(x)
    assert not torch.isnan(out).any()


def test_circle_mlp_backward_pass():
    from covertype_models import CircleMLP
    model = CircleMLP(input_dim=54, num_classes=7, hidden_dim=64, units=32)
    x = torch.randn(16, 54)
    targets = torch.randint(0, 7, (16,))
    logits = model(x)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None


def test_helix_layer_forward_shape():
    from covertype_models import HelixLayer
    layer = HelixLayer(input_dim=54, units=32, output_dim=64)
    x = torch.randn(16, 54)
    out = layer(x)
    assert out.shape == (16, 64)


def test_helix_layer_no_nans():
    from covertype_models import HelixLayer
    layer = HelixLayer(input_dim=54, units=32, output_dim=64)
    for scale in [0.01, 1.0, 10.0]:
        x = torch.randn(16, 54) * scale
        out = layer(x)
        assert not torch.isnan(out).any(), f"NaN at scale {scale}"


def test_helix_layer_backward_pass():
    from covertype_models import HelixLayer
    layer = HelixLayer(input_dim=54, units=32, output_dim=64)
    x = torch.randn(16, 54)
    out = layer(x)
    out.sum().backward()
    assert layer.u.grad is not None
    assert layer.v.grad is not None
    assert layer.w.grad is not None


def test_helix_mlp_forward_shape():
    from covertype_models import HelixMLP
    model = HelixMLP(input_dim=54, num_classes=7, hidden_dim=64, units=32)
    x = torch.randn(16, 54)
    out = model(x)
    assert out.shape == (16, 7)


def test_helix_mlp_no_nans():
    from covertype_models import HelixMLP
    model = HelixMLP(input_dim=54, num_classes=7, hidden_dim=64, units=32)
    x = torch.randn(16, 54)
    out = model(x)
    assert not torch.isnan(out).any()


def test_helix_mlp_backward_pass():
    from covertype_models import HelixMLP
    model = HelixMLP(input_dim=54, num_classes=7, hidden_dim=64, units=32)
    x = torch.randn(16, 54)
    targets = torch.randint(0, 7, (16,))
    logits = model(x)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None


def test_count_parameters_positive():
    from covertype_models import StandardMLP, CircleMLP, HelixMLP, count_parameters
    for Model, kwargs in [
        (StandardMLP, {"input_dim": 54, "num_classes": 7, "hidden_dim": 64}),
        (CircleMLP, {"input_dim": 54, "num_classes": 7, "hidden_dim": 64, "units": 32}),
        (HelixMLP, {"input_dim": 54, "num_classes": 7, "hidden_dim": 64, "units": 32}),
    ]:
        model = Model(**kwargs)
        n = count_parameters(model)
        assert n > 0, f"{Model.__name__} has {n} params"


def test_build_all_model_types():
    from covertype_config import CovertypeConfig, apply_scale_preset
    from covertype_models import build_covertype_model
    for mt in ["standard_mlp", "standard_mlp_matched", "circle_mlp", "helix_mlp"]:
        c = CovertypeConfig()
        c.model_type = mt
        c.scale = "small"
        apply_scale_preset(c)
        model = build_covertype_model(c)
        x = torch.randn(4, 54)
        out = model(x)
        assert out.shape == (4, 7), f"{mt} output shape {out.shape}"


def test_all_models_random_stress_no_nans():
    from covertype_config import CovertypeConfig, apply_scale_preset
    from covertype_models import build_covertype_model
    for mt in ["standard_mlp", "circle_mlp", "helix_mlp"]:
        c = CovertypeConfig()
        c.model_type = mt
        c.scale = "small"
        apply_scale_preset(c)
        model = build_covertype_model(c)
        for scale in [0.01, 1.0, 10.0]:
            x = torch.randn(32, 54) * scale
            out = model(x)
            assert not torch.isnan(out).any(), f"NaN in {mt} at scale {scale}"


def test_synthetic_training_step():
    from covertype_config import CovertypeConfig, apply_scale_preset
    from covertype_models import build_covertype_model
    c = CovertypeConfig()
    c.model_type = "helix_mlp"
    c.scale = "small"
    apply_scale_preset(c)
    model = build_covertype_model(c)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randn(32, 54)
    targets = torch.randint(0, 7, (32,))

    logits = model(x)
    loss = F.cross_entropy(logits, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss)
    assert loss.item() > 0


# ─── Metrics Tests ────────────────────────────────────────────────────────────

def test_accuracy_helper():
    from covertype_train import compute_metrics
    targets = np.array([0, 1, 2, 3, 4, 5, 6])
    preds = np.array([0, 1, 2, 3, 4, 5, 6])
    m = compute_metrics(targets, preds, num_classes=7)
    assert abs(m["accuracy"] - 1.0) < 1e-6


def test_macro_f1_helper():
    from covertype_train import compute_metrics
    targets = np.array([0, 0, 1, 1, 2, 2])
    preds = np.array([0, 0, 1, 1, 2, 2])
    m = compute_metrics(targets, preds, num_classes=7)
    assert m["macro_f1"] > 0.0


def test_weighted_f1_helper():
    from covertype_train import compute_metrics
    targets = np.array([0, 0, 0, 1, 1, 2])
    preds = np.array([0, 0, 1, 1, 1, 2])
    m = compute_metrics(targets, preds, num_classes=7)
    assert 0.0 < m["weighted_f1"] <= 1.0


def test_confusion_matrix_shape():
    from covertype_train import compute_metrics
    targets = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2])
    preds = np.array([0, 1, 2, 3, 4, 5, 6, 1, 2, 0])
    m = compute_metrics(targets, preds, num_classes=7)
    cm = m["confusion_matrix"]
    assert len(cm) == 7
    assert len(cm[0]) == 7


def test_per_class_accuracy_shape():
    from covertype_train import compute_metrics
    targets = np.array([0, 1, 2, 3, 4, 5, 6])
    preds = np.array([0, 1, 2, 3, 4, 5, 6])
    m = compute_metrics(targets, preds, num_classes=7)
    assert len(m["per_class_accuracy"]) == 7
    assert all(a == 1.0 for a in m["per_class_accuracy"])


# ─── Data Tests ───────────────────────────────────────────────────────────────

def test_covertype_loads():
    from covertype_data import load_covertype_arrays, resolve_data_dir
    data_dir = resolve_data_dir()
    X, y = load_covertype_arrays(data_dir=data_dir)
    assert X.shape[0] > 0
    assert y.shape[0] == X.shape[0]


def test_feature_shape_54():
    from covertype_data import load_covertype_arrays, resolve_data_dir
    data_dir = resolve_data_dir()
    X, y = load_covertype_arrays(data_dir=data_dir)
    assert X.shape[1] == 54


def test_num_classes_7():
    from covertype_data import load_covertype_arrays, resolve_data_dir
    data_dir = resolve_data_dir()
    X, y = load_covertype_arrays(data_dir=data_dir)
    unique = set(y.tolist())
    assert unique == {0, 1, 2, 3, 4, 5, 6}


def test_split_sizes():
    from covertype_data import load_covertype_arrays, make_train_val_test_split, resolve_data_dir
    data_dir = resolve_data_dir()
    X, y = load_covertype_arrays(data_dir=data_dir)
    splits = make_train_val_test_split(X, y)
    total = X.shape[0]
    train_n = splits["train"][0].shape[0]
    val_n = splits["val"][0].shape[0]
    test_n = splits["test"][0].shape[0]
    assert train_n + val_n + test_n == total
    # Check rough proportions (within 1%)
    assert abs(train_n / total - 0.70) < 0.01
    assert abs(val_n / total - 0.15) < 0.01
    assert abs(test_n / total - 0.15) < 0.01


def test_split_deterministic():
    from covertype_data import load_covertype_arrays, make_train_val_test_split, resolve_data_dir
    data_dir = resolve_data_dir()
    X, y = load_covertype_arrays(data_dir=data_dir)
    s1 = make_train_val_test_split(X, y, seed=42)
    s2 = make_train_val_test_split(X, y, seed=42)
    assert np.array_equal(s1["train"][1][:100], s2["train"][1][:100])


def test_continuous_features_standardized():
    from covertype_data import (
        load_covertype_arrays, make_train_val_test_split,
        preprocess_covertype_features, resolve_data_dir,
    )
    data_dir = resolve_data_dir()
    X, y = load_covertype_arrays(data_dir=data_dir)
    splits = make_train_val_test_split(X, y)
    processed = preprocess_covertype_features(splits)
    X_train = processed["train"][0]
    # Training continuous features should be approximately mean=0, std=1
    mean = X_train[:, :10].mean(axis=0)
    std = X_train[:, :10].std(axis=0)
    assert np.allclose(mean, 0, atol=0.01), f"Mean not ~0: {mean}"
    assert np.allclose(std, 1, atol=0.01), f"Std not ~1: {std}"


def test_binary_features_remain_binary():
    from covertype_data import (
        load_covertype_arrays, make_train_val_test_split,
        preprocess_covertype_features, resolve_data_dir,
    )
    data_dir = resolve_data_dir()
    X, y = load_covertype_arrays(data_dir=data_dir)
    splits = make_train_val_test_split(X, y)
    processed = preprocess_covertype_features(splits)
    X_train = processed["train"][0]
    binary_features = X_train[:, 10:]
    unique_vals = set(np.unique(binary_features).tolist())
    assert unique_vals <= {0.0, 1.0}, f"Binary features have values: {unique_vals}"


def test_batch_shapes():
    from covertype_config import CovertypeConfig, apply_scale_preset
    from covertype_data import make_covertype_dataloaders
    c = CovertypeConfig()
    c.scale = "small"
    apply_scale_preset(c)
    c.batch_size = 64
    dataloaders = make_covertype_dataloaders(c)
    batch_x, batch_y = next(iter(dataloaders["train"]))
    assert batch_x.shape == (64, 54)
    assert batch_y.shape == (64,)
    assert batch_x.dtype == torch.float32
    assert batch_y.dtype == torch.int64


# ─── Slow Overfit Tests ───────────────────────────────────────────────────────

def test_overfit_tiny_batch_standard():
    from covertype_models import StandardMLP
    _overfit_tiny_batch(StandardMLP(input_dim=54, num_classes=7, hidden_dim=128, num_layers=2, dropout=0.0), 0.90)


def test_overfit_tiny_batch_circle():
    from covertype_models import CircleMLP
    _overfit_tiny_batch(CircleMLP(input_dim=54, num_classes=7, hidden_dim=128, units=64, dropout=0.0, use_layernorm=True), 0.85)


def test_overfit_tiny_batch_helix():
    from covertype_models import HelixMLP
    _overfit_tiny_batch(HelixMLP(input_dim=54, num_classes=7, hidden_dim=128, units=64, dropout=0.0, use_layernorm=True), 0.85)


def _overfit_tiny_batch(model, threshold: float, steps: int = 500, batch_size: int = 32):
    """Verify a model can overfit a small fixed batch."""
    torch.manual_seed(0)
    x = torch.randn(batch_size, 54)
    targets = torch.randint(0, 7, (batch_size,))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    for _ in range(steps):
        logits = model(x)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(x)
        preds = logits.argmax(-1)
        acc = (preds == targets).float().mean().item()

    assert acc >= threshold, f"Overfit accuracy {acc:.4f} < {threshold}"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", action="store_true", help="Run data tests (requires dataset)")
    parser.add_argument("--slow", action="store_true", help="Run slow overfit tests")
    args = parser.parse_args()

    print("=" * 60)
    print("Covertype Experiment Tests")
    print("=" * 60)

    # Fast tests (no dataset needed)
    print("\n--- Config Tests ---")
    run_test("test_config_defaults", test_config_defaults)
    run_test("test_config_to_dict", test_config_to_dict)
    run_test("test_apply_scale_preset_small", test_apply_scale_preset_small)
    run_test("test_apply_scale_preset_medium", test_apply_scale_preset_medium)
    run_test("test_apply_scale_preset_large", test_apply_scale_preset_large)
    run_test("test_get_device_falls_back_to_cpu", test_get_device_falls_back_to_cpu)

    print("\n--- Model Tests ---")
    run_test("test_standard_mlp_forward_shape", test_standard_mlp_forward_shape)
    run_test("test_standard_mlp_no_nans", test_standard_mlp_no_nans)
    run_test("test_standard_mlp_backward_pass", test_standard_mlp_backward_pass)
    run_test("test_circle_layer_forward_shape", test_circle_layer_forward_shape)
    run_test("test_circle_layer_no_nans", test_circle_layer_no_nans)
    run_test("test_circle_layer_backward_pass", test_circle_layer_backward_pass)
    run_test("test_circle_mlp_forward_shape", test_circle_mlp_forward_shape)
    run_test("test_circle_mlp_no_nans", test_circle_mlp_no_nans)
    run_test("test_circle_mlp_backward_pass", test_circle_mlp_backward_pass)
    run_test("test_helix_layer_forward_shape", test_helix_layer_forward_shape)
    run_test("test_helix_layer_no_nans", test_helix_layer_no_nans)
    run_test("test_helix_layer_backward_pass", test_helix_layer_backward_pass)
    run_test("test_helix_mlp_forward_shape", test_helix_mlp_forward_shape)
    run_test("test_helix_mlp_no_nans", test_helix_mlp_no_nans)
    run_test("test_helix_mlp_backward_pass", test_helix_mlp_backward_pass)
    run_test("test_count_parameters_positive", test_count_parameters_positive)
    run_test("test_build_all_model_types", test_build_all_model_types)
    run_test("test_all_models_random_stress_no_nans", test_all_models_random_stress_no_nans)
    run_test("test_synthetic_training_step", test_synthetic_training_step)

    print("\n--- Metrics Tests ---")
    run_test("test_accuracy_helper", test_accuracy_helper)
    run_test("test_macro_f1_helper", test_macro_f1_helper)
    run_test("test_weighted_f1_helper", test_weighted_f1_helper)
    run_test("test_confusion_matrix_shape", test_confusion_matrix_shape)
    run_test("test_per_class_accuracy_shape", test_per_class_accuracy_shape)

    # Data tests
    if args.data or args.slow:
        print("\n--- Data Tests ---")
        run_test("test_covertype_loads", test_covertype_loads)
        run_test("test_feature_shape_54", test_feature_shape_54)
        run_test("test_num_classes_7", test_num_classes_7)
        run_test("test_split_sizes", test_split_sizes)
        run_test("test_split_deterministic", test_split_deterministic)
        run_test("test_continuous_features_standardized", test_continuous_features_standardized)
        run_test("test_binary_features_remain_binary", test_binary_features_remain_binary)
        run_test("test_batch_shapes", test_batch_shapes)

    # Slow overfit tests
    if args.slow:
        print("\n--- Slow Overfit Tests ---")
        run_test("test_overfit_tiny_batch_standard", test_overfit_tiny_batch_standard)
        run_test("test_overfit_tiny_batch_circle", test_overfit_tiny_batch_circle)
        run_test("test_overfit_tiny_batch_helix", test_overfit_tiny_batch_helix)

    all_passed = print_summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
