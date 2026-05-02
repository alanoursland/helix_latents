"""
Flattened CIFAR-10 HelixLayer Experiment
==========================================

Tests whether HelixLayer can remain competitive with dense MLP baselines
on a harder, nonlocal image-classification problem.

Usage:
  cd src
  python cifar10_run_experiment.py --quick                             # fast sanity check
  python cifar10_run_experiment.py --quick --all-models                # quick all models
  python cifar10_run_experiment.py --model-type helix_mlp --scale small
  python cifar10_run_experiment.py --all-models --scale medium
  python cifar10_run_experiment.py --all-models --sweep-scales
  python cifar10_run_experiment.py --data-dir E:/ml_datasets --all-models --scale small
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cifar10_config import CIFAR10Config, apply_scale_preset, get_device, save_json
from cifar10_data import make_cifar10_dataloaders
from cifar10_models import build_cifar10_model, count_parameters
from cifar10_train import evaluate, fit_cifar10
from utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flattened CIFAR-10 HelixLayer Experiment")
    parser.add_argument("--model-type", type=str, default=None,
                        choices=["standard_mlp", "standard_mlp_matched", "circle_mlp", "helix_mlp"])
    parser.add_argument("--all-models", action="store_true", help="Train all 4 model variants")
    parser.add_argument("--scale", type=str, default=None, choices=["small", "medium", "large"])
    parser.add_argument("--sweep-scales", action="store_true", help="Run all scales")
    parser.add_argument("--quick", action="store_true", help="Fast sanity check (1 epoch, limited batches)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--matched-hidden-dim", type=int, default=None)
    parser.add_argument("--circle-units", type=int, default=None)
    parser.add_argument("--helix-units", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset directory")
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-eval-batches", type=int, default=None)
    parser.add_argument("--use-scheduler", action="store_true")
    parser.add_argument("--scheduler-type", type=str, default=None, choices=["none", "cosine"])
    return parser.parse_args()


def run_single(config: CIFAR10Config, cli_data_dir: str | None) -> dict:
    """Train one CIFAR-10 model variant."""
    print(f"\n{'='*60}")
    print(f"Model: {config.model_type} | Scale: {config.scale}")
    print(f"{'='*60}")

    device = get_device(config.device)
    print(f"Device: {device}")

    # Data
    dataloaders = make_cifar10_dataloaders(config, cli_data_dir)
    print(f"Data splits: train={len(dataloaders['train'].dataset)}, "
          f"val={len(dataloaders['val'].dataset)}, "
          f"test={len(dataloaders['test'].dataset)}")

    # Model
    set_seed(config.seed)
    model = build_cifar10_model(config)
    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,}")

    # Train
    result = fit_cifar10(model, dataloaders, config)

    print(f"\nResults:")
    print(f"  Best epoch: {result['best_epoch']}")
    print(f"  Best val accuracy: {result['best_val_accuracy']:.4f}")
    print(f"  Test accuracy: {result['test_metrics']['accuracy']:.4f}")
    print(f"  Test loss: {result['test_metrics']['loss']:.4f}")
    print(f"  Mean epoch time: {result['mean_epoch_seconds']:.2f}s")
    print(f"  Checkpoint: {result['checkpoint_path']}")

    return {
        "model_type": config.model_type,
        "scale": config.scale,
        "param_count": n_params,
        "best_epoch": result["best_epoch"],
        "best_val_accuracy": result["best_val_accuracy"],
        "test_accuracy": result["test_metrics"]["accuracy"],
        "test_loss": result["test_metrics"]["loss"],
        "mean_epoch_seconds": result["mean_epoch_seconds"],
    }


def main() -> None:
    args = parse_args()

    # Build base config
    config = CIFAR10Config()

    if args.quick:
        config.epochs = 1
        config.scale = "small"
        config.limit_train_batches = 100
        config.limit_eval_batches = 50

    # Apply scale (CLI overrides quick default)
    if args.scale is not None:
        config.scale = args.scale

    # Apply CLI overrides
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "matched_hidden_dim": args.matched_hidden_dim,
        "circle_units": args.circle_units,
        "helix_units": args.helix_units,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": args.device,
        "limit_train_batches": args.limit_train_batches,
        "limit_eval_batches": args.limit_eval_batches,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)

    if args.use_scheduler:
        config.use_scheduler = True
    if args.scheduler_type is not None:
        config.scheduler_type = args.scheduler_type
        config.use_scheduler = True

    # Determine scales to run
    if args.sweep_scales:
        scales = ["small", "medium", "large"]
    else:
        scales = [config.scale]

    # Determine models to run
    if args.all_models:
        model_types = ["standard_mlp", "standard_mlp_matched", "circle_mlp", "helix_mlp"]
    elif args.model_type:
        model_types = [args.model_type]
    else:
        model_types = ["helix_mlp"]

    # Run experiments
    all_results = []
    for scale in scales:
        config.scale = scale
        apply_scale_preset(config)

        for mt in model_types:
            config.model_type = mt
            result = run_single(config, args.data_dir)
            all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Scale':<8} {'Params':>10} {'Best Val':>10} {'Test Acc':>10} {'Time/Ep':>10}")
    print("-" * 75)
    for r in all_results:
        print(f"{r['model_type']:<25} {r['scale']:<8} {r['param_count']:>10,} "
              f"{r['best_val_accuracy']:>10.4f} {r['test_accuracy']:>10.4f} "
              f"{r['mean_epoch_seconds']:>9.2f}s")

    # Save comparison
    if len(all_results) > 1:
        results_dir = Path(config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        if args.sweep_scales:
            filename = f"comparison_all_scales_seed{config.seed}.json"
        else:
            filename = f"comparison_{config.scale}_seed{config.seed}.json"

        save_json({
            "dataset": "CIFAR-10",
            "input": "flattened_pixels",
            "seed": config.seed,
            "epochs": config.epochs,
            "results": all_results,
        }, results_dir / filename)
        print(f"\nComparison saved to {results_dir / filename}")


if __name__ == "__main__":
    main()
