"""
Tests for the helix latents experiment.

Run with:
  cd src
  python test_all.py
"""

from __future__ import annotations

import sys
import traceback

import torch

from config import ExperimentConfig, get_device
from data import make_dataloaders, make_modular_addition_data, split_dataset
from geometry import circle_encode, helix_encode, infer_phase_step, rotate_circle, shift_helix
from intervene import evaluate_intervention, expected_intervention_targets, intervene_on_a
from models import BaselineMLP, CircleBottleneckMLP, HelixBottleneckMLP, build_model
from train import evaluate, fit, train_one_epoch
from utils import set_seed


def assert_close(a, b, atol=1e-5, msg=""):
    if not torch.allclose(a, b, atol=atol):
        raise AssertionError(f"Not close: {msg}\n  got {a}\n  expected {b}")


# ─── Data Tests ───────────────────────────────────────────────────────────────

def test_data_size():
    pairs, targets = make_modular_addition_data(5)
    assert pairs.shape == (25, 2), f"Expected (25,2), got {pairs.shape}"
    assert targets.shape == (25,), f"Expected (25,), got {targets.shape}"


def test_data_targets_correct():
    pairs, targets = make_modular_addition_data(7)
    expected = (pairs[:, 0] + pairs[:, 1]) % 7
    assert torch.equal(targets, expected)


def test_all_pairs_present():
    N = 5
    pairs, _ = make_modular_addition_data(N)
    pair_set = {(pairs[i, 0].item(), pairs[i, 1].item()) for i in range(pairs.shape[0])}
    expected = {(a, b) for a in range(N) for b in range(N)}
    assert pair_set == expected


def test_splits_disjoint_and_cover():
    pairs, targets = make_modular_addition_data(11)
    tr, va, te = split_dataset(pairs, targets, 0.7, 0.15, 0.15, seed=42)
    total = len(tr) + len(va) + len(te)
    assert total == 121, f"Expected 121, got {total}"

    tr_set = set(map(tuple, tr.pairs.tolist()))
    va_set = set(map(tuple, va.pairs.tolist()))
    te_set = set(map(tuple, te.pairs.tolist()))
    assert len(tr_set & va_set) == 0
    assert len(tr_set & te_set) == 0
    assert len(va_set & te_set) == 0


def test_splits_deterministic():
    pairs, targets = make_modular_addition_data(7)
    tr1, _, _ = split_dataset(pairs, targets, 0.7, 0.15, 0.15, seed=123)
    tr2, _, _ = split_dataset(pairs, targets, 0.7, 0.15, 0.15, seed=123)
    assert torch.equal(tr1.pairs, tr2.pairs)


def test_dataloaders():
    config = ExperimentConfig(modulus=5, batch_size=4)
    loaders = make_dataloaders(config)
    batch = next(iter(loaders["train"]))
    assert set(batch.keys()) == {"a", "b", "target"}


# ─── Geometry Tests ───────────────────────────────────────────────────────────

def test_circle_encode_shape_and_norm():
    x = torch.arange(10)
    result = circle_encode(x, 10)
    assert result.shape == (10, 2)
    norms = torch.norm(result, dim=-1)
    assert_close(norms, torch.ones(10), msg="circle norms")


def test_helix_encode_shape():
    x = torch.arange(10)
    result = helix_encode(x, 10)
    assert result.shape == (10, 3)
    phase_norms = torch.norm(result[:, :2], dim=-1)
    assert_close(phase_norms, torch.ones(10), msg="helix phase norms")


def test_rotate_zero_is_identity():
    xy = circle_encode(torch.arange(10), 10)
    rotated = rotate_circle(xy, 0, 10)
    assert_close(xy, rotated, msg="rotate by 0")


def test_rotate_N_is_identity():
    N = 13
    xy = circle_encode(torch.arange(N), N)
    rotated = rotate_circle(xy, N, N)
    assert_close(xy, rotated, msg="rotate by N")


def test_rotate_matches_encode():
    N = 59
    for a in [0, 1, 17, 58]:
        for k in [1, 3, -2, 60]:
            x = torch.tensor([a])
            rotated = rotate_circle(circle_encode(x, N), k, N)
            expected = circle_encode(torch.tensor([(a + k) % N]), N)
            assert_close(rotated, expected, msg=f"a={a}, k={k}")


def test_shift_helix_axis():
    N = 11
    alpha = 1.0 / N
    x = torch.tensor([4])
    xyz = helix_encode(x, N, alpha)
    shifted = shift_helix(xyz, 3, N, alpha, shift_axis=True)
    expected_z = alpha * 4.0 + alpha * 3
    assert_close(shifted[..., 2], torch.tensor([expected_z]), msg="helix axis shift")


def test_shift_helix_no_axis():
    N = 11
    alpha = 1.0 / N
    x = torch.tensor([4])
    xyz = helix_encode(x, N, alpha)
    shifted = shift_helix(xyz, 3, N, alpha, shift_axis=False)
    assert_close(shifted[..., 2], xyz[..., 2], msg="helix axis preserved")


# ─── Model Tests ──────────────────────────────────────────────────────────────

def test_model_forward_shapes():
    a = torch.tensor([0, 1, 2])
    b = torch.tensor([3, 4, 5])

    m1 = BaselineMLP(modulus=11, embedding_dim=16, hidden_dim=32, num_hidden_layers=1)
    out1 = m1(a, b)
    assert out1.logits.shape == (3, 11)
    assert out1.latents["a"].shape == (3, 16)

    m2 = CircleBottleneckMLP(modulus=11, hidden_dim=32, num_hidden_layers=1)
    out2 = m2(a, b)
    assert out2.logits.shape == (3, 11)
    assert out2.latents["a"].shape == (3, 2)

    m3 = HelixBottleneckMLP(modulus=11, hidden_dim=32, num_hidden_layers=1)
    out3 = m3(a, b)
    assert out3.logits.shape == (3, 11)
    assert out3.latents["a"].shape == (3, 3)


def test_latent_override():
    model = CircleBottleneckMLP(modulus=11, hidden_dim=32, num_hidden_layers=1)
    a = torch.tensor([0, 1, 2])
    b = torch.tensor([3, 4, 5])
    out1 = model(a, b)
    override = torch.randn(3, 2)
    out2 = model(a, b, latent_override={"a": override})
    assert not torch.allclose(out1.logits, out2.logits)


def test_build_model_factory():
    for mt in ["baseline_mlp", "circle_bottleneck_mlp", "helix_bottleneck_mlp"]:
        config = ExperimentConfig(modulus=7, model_type=mt, hidden_dim=32)
        model = build_model(config)
        out = model(torch.tensor([0, 1]), torch.tensor([2, 3]))
        assert out.logits.shape == (2, 7), f"Failed for {mt}"


# ─── Intervention Tests ───────────────────────────────────────────────────────

def test_expected_targets():
    a = torch.tensor([3, 10])
    b = torch.tensor([4, 10])
    expected = expected_intervention_targets(a, b, k=5, modulus=11)
    manual = (a + 5 + b) % 11
    assert torch.equal(expected, manual)


def test_intervene_circle_shape():
    model = CircleBottleneckMLP(modulus=11, hidden_dim=32, num_hidden_layers=1)
    a = torch.tensor([0, 1, 2, 3])
    b = torch.tensor([4, 5, 6, 7])
    output = intervene_on_a(model, a, b, k=1, modulus=11, model_type="circle_bottleneck_mlp")
    assert output.logits.shape == (4, 11)


def test_intervene_helix_shape():
    model = HelixBottleneckMLP(modulus=11, hidden_dim=32, num_hidden_layers=1)
    a = torch.tensor([0, 1, 2, 3])
    b = torch.tensor([4, 5, 6, 7])
    output = intervene_on_a(model, a, b, k=1, modulus=11, model_type="helix_bottleneck_mlp")
    assert output.logits.shape == (4, 11)


def test_intervene_baseline_random():
    model = BaselineMLP(modulus=7, embedding_dim=16, hidden_dim=32, num_hidden_layers=1)
    a = torch.tensor([0, 1, 2])
    b = torch.tensor([3, 4, 5])
    output = intervene_on_a(model, a, b, k=1, modulus=7, model_type="baseline_mlp", mode="random")
    assert output.logits.shape == (3, 7)


def test_intervene_baseline_phase_raises():
    model = BaselineMLP(modulus=7, embedding_dim=16, hidden_dim=32, num_hidden_layers=1)
    a = torch.tensor([0, 1, 2])
    b = torch.tensor([3, 4, 5])
    try:
        intervene_on_a(model, a, b, k=1, modulus=7, model_type="baseline_mlp", mode="phase_only")
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass


# ─── Training Smoke Test ──────────────────────────────────────────────────────

def test_training_smoke():
    config = ExperimentConfig(
        modulus=7, model_type="circle_bottleneck_mlp", hidden_dim=64,
        num_hidden_layers=2, max_epochs=200, batch_size=16, seed=42,
        learning_rate=3e-3, device="cpu", early_stopping_patience=200,
        checkpoint_dir="checkpoints_test", results_dir="results_test",
    )
    set_seed(config.seed)
    dataloaders = make_dataloaders(config)
    model = build_model(config)
    result = fit(model, dataloaders, config)
    assert result["best_val_accuracy"] > 0.30, (
        f"Expected > 0.30, got {result['best_val_accuracy']}"
    )


# ─── Reproducibility ─────────────────────────────────────────────────────────

def test_reproducibility():
    config = ExperimentConfig(modulus=7, hidden_dim=32, model_type="circle_bottleneck_mlp")
    set_seed(42)
    m1 = build_model(config)
    p1 = list(m1.parameters())[0].data.clone()
    set_seed(42)
    m2 = build_model(config)
    p2 = list(m2.parameters())[0].data.clone()
    assert torch.equal(p1, p2)


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_tests():
    tests = [
        # Data
        test_data_size,
        test_data_targets_correct,
        test_all_pairs_present,
        test_splits_disjoint_and_cover,
        test_splits_deterministic,
        test_dataloaders,
        # Geometry
        test_circle_encode_shape_and_norm,
        test_helix_encode_shape,
        test_rotate_zero_is_identity,
        test_rotate_N_is_identity,
        test_rotate_matches_encode,
        test_shift_helix_axis,
        test_shift_helix_no_axis,
        # Models
        test_model_forward_shapes,
        test_latent_override,
        test_build_model_factory,
        # Interventions
        test_expected_targets,
        test_intervene_circle_shape,
        test_intervene_helix_shape,
        test_intervene_baseline_random,
        test_intervene_baseline_phase_raises,
        # Training
        test_training_smoke,
        # Reproducibility
        test_reproducibility,
    ]

    passed = 0
    failed = 0
    errors = []

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
