"""
Helix Latents: Modular Addition Experiment
==========================================

Run the full experiment pipeline:
  1. Train all three model variants (baseline, circle, helix)
  2. Evaluate normal test accuracy
  3. Run causal latent interventions
  4. Compare results and save plots

Usage:
  cd src
  python run_experiment.py                    # full default (N=59, all models)
  python run_experiment.py --modulus 11       # smaller/faster run
  python run_experiment.py --model-type circle_bottleneck_mlp  # single model
  python run_experiment.py --quick            # fast smoke test (N=11, 100 epochs)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import ExperimentConfig, get_device, save_json
from data import make_dataloaders
from intervene import evaluate_intervention
from models import build_model
from plotting import plot_intervention_accuracy, plot_training_history
from train import evaluate, fit
from utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Helix Latents Experiment")
    parser.add_argument("--modulus", type=int, default=None)
    parser.add_argument("--model-type", type=str, default=None,
                        choices=["baseline_mlp", "circle_bottleneck_mlp", "helix_bottleneck_mlp"])
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-hidden-layers", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--quick", action="store_true", help="Fast smoke test (N=11, 100 epochs)")
    parser.add_argument("--shifts", type=int, nargs="+", default=None,
                        help="Intervention shifts to test")
    parser.add_argument("--skip-intervention", action="store_true",
                        help="Skip intervention evaluation")
    parser.add_argument("--all-models", action="store_true",
                        help="Train and compare all 3 model types")
    return parser.parse_args()


def run_single_model(config: ExperimentConfig, shifts: list[int]) -> dict:
    """Train one model, evaluate, and run interventions."""
    print(f"\n{'='*60}")
    print(f"Model: {config.model_type}")
    print(f"Modulus: {config.modulus}, Hidden: {config.hidden_dim}, "
          f"Layers: {config.num_hidden_layers}")
    print(f"{'='*60}")

    device = get_device(config.device)
    print(f"Device: {device}")

    # Data
    dataloaders = make_dataloaders(config)
    print(f"Dataset: {config.modulus}^2 = {config.modulus**2} examples")
    print(f"  Train: {len(dataloaders['train'].dataset)}, "
          f"Val: {len(dataloaders['val'].dataset)}, "
          f"Test: {len(dataloaders['test'].dataset)}")

    # Build & train
    set_seed(config.seed)
    model = build_model(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    result = fit(model, dataloaders, config)
    print(f"\nTraining complete:")
    print(f"  Best epoch: {result['best_epoch']}")
    print(f"  Best val accuracy: {result['best_val_accuracy']:.4f}")

    # Test evaluation
    model = model.to(device)
    test_metrics = evaluate(model, dataloaders["test"], device)
    print(f"  Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test loss: {test_metrics['loss']:.4f}")

    # Save training results
    results_dir = Path(config.results_dir) / config.model_type
    results_dir.mkdir(parents=True, exist_ok=True)

    save_json(result["history"], results_dir / "training_history.json")
    plot_training_history(result["history"], str(results_dir / "training_history.png"))

    # Interventions
    intervention_results = {}
    if shifts and config.model_type != "baseline_mlp":
        modes = ["phase_plus_axis", "phase_only"]
        if config.model_type == "helix_bottleneck_mlp":
            modes.append("axis_only")
        modes.append("random")

        for mode in modes:
            print(f"\n  Intervention mode={mode}:")
            int_result = evaluate_intervention(model, dataloaders["test"], config, shifts, mode)
            print(f"    Overall accuracy: {int_result['overall_accuracy']:.4f} "
                  f"(chance: {1.0/config.modulus:.4f})")
            for k, acc in sorted(int_result["accuracy_by_shift"].items(), key=lambda x: int(x[0])):
                print(f"      k={k}: {acc:.4f}")

            save_json(int_result, results_dir / f"intervention_{mode}.json")
            plot_intervention_accuracy(int_result, str(results_dir / f"intervention_{mode}.png"))
            intervention_results[mode] = int_result

    elif shifts and config.model_type == "baseline_mlp":
        # Only random control for baseline
        print(f"\n  Intervention mode=random (control):")
        int_result = evaluate_intervention(model, dataloaders["test"], config, shifts, "random")
        print(f"    Overall accuracy: {int_result['overall_accuracy']:.4f} "
              f"(chance: {1.0/config.modulus:.4f})")
        save_json(int_result, results_dir / "intervention_random.json")
        intervention_results["random"] = int_result

    return {
        "model_type": config.model_type,
        "test_accuracy": test_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
        "best_val_accuracy": result["best_val_accuracy"],
        "best_epoch": result["best_epoch"],
        "interventions": intervention_results,
    }


def main() -> None:
    args = parse_args()

    # Build config
    config = ExperimentConfig()

    if args.quick:
        config.modulus = 11
        config.max_epochs = 300
        config.hidden_dim = 128
        config.batch_size = 32
        config.learning_rate = 3e-3
        config.early_stopping_patience = 80

    # Apply CLI overrides
    for key in ["modulus", "hidden_dim", "num_hidden_layers", "max_epochs",
                "batch_size", "learning_rate", "seed", "device", "model_type"]:
        val = getattr(args, key.replace("-", "_"), None) if "-" not in key else getattr(args, key.replace("-", "_"), None)
        if val is not None:
            setattr(config, key, val)

    # Determine shifts
    if args.skip_intervention:
        shifts = []
    elif args.shifts:
        shifts = args.shifts
    elif config.modulus <= 13:
        shifts = [1, 2, 3, 5]
    else:
        shifts = [1, 2, 3, 5, 10, 17, 29]

    # Determine which models to run
    if args.all_models or args.model_type is None:
        model_types = ["circle_bottleneck_mlp", "helix_bottleneck_mlp", "baseline_mlp"]
    else:
        model_types = [config.model_type]

    # Run experiments
    all_results = []
    for mt in model_types:
        config.model_type = mt
        result = run_single_model(config, shifts)
        all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<30} {'Test Acc':>10} {'Best Val':>10}")
    print("-" * 52)
    for r in all_results:
        print(f"{r['model_type']:<30} {r['test_accuracy']:>10.4f} {r['best_val_accuracy']:>10.4f}")

    if shifts:
        print(f"\nIntervention Results (chance = {1.0/config.modulus:.4f}):")
        print(f"{'Model':<30} {'Mode':<20} {'Accuracy':>10}")
        print("-" * 62)
        for r in all_results:
            for mode, int_r in r["interventions"].items():
                print(f"{r['model_type']:<30} {mode:<20} {int_r['overall_accuracy']:>10.4f}")

    # Save comparison
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {"modulus": config.modulus, "shifts": shifts, "results": [
            {k: v for k, v in r.items() if k != "interventions"}
            for r in all_results
        ]},
        results_dir / "comparison.json",
    )
    print(f"\nResults saved to {results_dir}/")


if __name__ == "__main__":
    main()
