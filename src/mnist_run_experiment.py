"""
MNIST HelixLayer Experiment
============================

Tests whether HelixLayer can function as a trainable feedforward primitive
on standard image classification.

Usage:
  cd src
  python mnist_run_experiment.py --quick                    # fast sanity check
  python mnist_run_experiment.py --model-type helix_mlp     # single model
  python mnist_run_experiment.py --all-models               # full comparison
  python mnist_run_experiment.py --data-dir E:/ml_datasets  # custom data path
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mnist_config import MNISTConfig, get_device, resolve_data_dir, save_json
from mnist_data import make_mnist_dataloaders
from mnist_models import build_mnist_model, count_parameters
from mnist_train import evaluate, fit_mnist
from utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MNIST HelixLayer Experiment")
    parser.add_argument("--model-type", type=str, default=None,
                        choices=["standard_mlp", "standard_mlp_matched", "circle_mlp", "helix_mlp"])
    parser.add_argument("--all-models", action="store_true", help="Train all 4 model variants")
    parser.add_argument("--quick", action="store_true", help="Fast sanity check (1 epoch, limited batches)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--matched-hidden-dim", type=int, default=None)
    parser.add_argument("--helix-units", type=int, default=None)
    parser.add_argument("--circle-units", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset directory")
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-eval-batches", type=int, default=None)
    return parser.parse_args()


def run_single_model(config: MNISTConfig, cli_data_dir: str | None) -> dict:
    """Train one MNIST model variant."""
    print(f"\n{'='*60}")
    print(f"Model: {config.model_type}")
    print(f"{'='*60}")

    device = get_device(config.device)
    print(f"Device: {device}")

    # Data
    dataloaders = make_mnist_dataloaders(config, cli_data_dir)
    print(f"Data splits: train={len(dataloaders['train'].dataset)}, "
          f"val={len(dataloaders['val'].dataset)}, "
          f"test={len(dataloaders['test'].dataset)}")

    # Model
    set_seed(config.seed)
    model = build_mnist_model(config)
    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,}")

    # Train
    result = fit_mnist(model, dataloaders, config)

    print(f"\nResults:")
    print(f"  Best epoch: {result['best_epoch']}")
    print(f"  Best val accuracy: {result['best_val_accuracy']:.4f}")
    print(f"  Test accuracy: {result['test_metrics']['accuracy']:.4f}")
    print(f"  Test loss: {result['test_metrics']['loss']:.4f}")
    print(f"  Checkpoint: {result['checkpoint_path']}")

    return {
        "model_type": config.model_type,
        "param_count": n_params,
        "best_epoch": result["best_epoch"],
        "best_val_accuracy": result["best_val_accuracy"],
        "test_accuracy": result["test_metrics"]["accuracy"],
        "test_loss": result["test_metrics"]["loss"],
    }


def main() -> None:
    args = parse_args()

    # Build base config
    config = MNISTConfig()

    if args.quick:
        config.epochs = 1
        config.limit_train_batches = 100
        config.limit_eval_batches = 50

    # Apply CLI overrides
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "matched_hidden_dim": args.matched_hidden_dim,
        "helix_units": args.helix_units,
        "circle_units": args.circle_units,
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

    # Determine models to run
    if args.all_models:
        model_types = ["standard_mlp", "standard_mlp_matched", "circle_mlp", "helix_mlp"]
    elif args.model_type:
        model_types = [args.model_type]
    else:
        model_types = ["helix_mlp"]

    # Run experiments
    all_results = []
    for mt in model_types:
        config.model_type = mt
        result = run_single_model(config, args.data_dir)
        all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Params':>10} {'Best Val':>10} {'Test Acc':>10}")
    print("-" * 57)
    for r in all_results:
        print(f"{r['model_type']:<25} {r['param_count']:>10,} "
              f"{r['best_val_accuracy']:>10.4f} {r['test_accuracy']:>10.4f}")

    # Save comparison
    if len(all_results) > 1:
        results_dir = Path(config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        save_json({
            "dataset": "MNIST",
            "seed": config.seed,
            "epochs": config.epochs,
            "results": all_results,
        }, results_dir / "comparison.json")
        print(f"\nComparison saved to {results_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
