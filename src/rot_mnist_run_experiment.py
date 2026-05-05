"""
Rotated MNIST HelixConv Experiment Runner
==========================================

Tests whether HelixConv self-organizes to track input orientation when
trained on rotation-augmented MNIST.

Usage:
  cd src
  python rot_mnist_run_experiment.py --quick                             # fast sanity check
  python rot_mnist_run_experiment.py --quick --all-models                # quick all models
  python rot_mnist_run_experiment.py --model-type helix_conv --scale small
  python rot_mnist_run_experiment.py --all-models --scale small --run-filter-analysis --run-trajectory-analysis
  python rot_mnist_run_experiment.py --all-models --sweep-scales
  python rot_mnist_run_experiment.py --model-type helix_conv --scale small --run-intervention-analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rot_mnist_config import RotMNISTConfig, apply_scale_preset, get_device, save_json
from rot_mnist_data import make_rot_mnist_dataloaders
from rot_mnist_models import build_rot_mnist_model, count_parameters
from rot_mnist_train import fit_rot_mnist
from utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rotated MNIST HelixConv Experiment")
    parser.add_argument("--model-type", type=str, default=None,
                        choices=["standard_cnn", "standard_cnn_matched", "circle_conv", "helix_conv"])
    parser.add_argument("--all-models", action="store_true", help="Train all 4 model variants")
    parser.add_argument("--scale", type=str, default=None, choices=["small", "medium", "large"])
    parser.add_argument("--sweep-scales", action="store_true", help="Run all scales")
    parser.add_argument("--quick", action="store_true", help="Fast sanity check (1 epoch, limited batches)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hidden-channels", type=int, default=None)
    parser.add_argument("--matched-hidden-channels", type=int, default=None)
    parser.add_argument("--circle-units", type=int, default=None)
    parser.add_argument("--helix-units", type=int, default=None)
    parser.add_argument("--kernel-size", type=int, default=None)
    parser.add_argument("--num-conv-blocks", type=int, default=None)
    parser.add_argument("--rotation-max-degrees", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset directory")
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-eval-batches", type=int, default=None)
    parser.add_argument("--use-scheduler", action="store_true")
    parser.add_argument("--scheduler-type", type=str, default=None, choices=["none", "cosine"])
    parser.add_argument("--run-filter-analysis", action="store_true",
                        help="Run filter-pair analysis for circle/helix models")
    parser.add_argument("--run-trajectory-analysis", action="store_true",
                        help="Run trajectory analysis for helix models")
    parser.add_argument("--run-intervention-analysis", action="store_true",
                        help="Run causal intervention analysis for helix models")
    return parser.parse_args()


def run_single(config: RotMNISTConfig, cli_data_dir: str | None, args: argparse.Namespace) -> dict:
    """Train one rotated MNIST model variant, optionally run analyses."""
    print(f"\n{'='*60}")
    print(f"Model: {config.model_type} | Scale: {config.scale}")
    print(f"{'='*60}")

    device = get_device(config.device)
    print(f"Device: {device}")

    # Data
    dataloaders = make_rot_mnist_dataloaders(config, cli_data_dir)
    print(f"Data splits: train={len(dataloaders['train'].dataset)}, "
          f"val={len(dataloaders['val'].dataset)}, "
          f"test_rotated={len(dataloaders['test_rotated'].dataset)}, "
          f"test_unrotated={len(dataloaders['test_unrotated'].dataset)}")

    # Model
    set_seed(config.seed)
    model = build_rot_mnist_model(config)
    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,}")

    # Train
    result = fit_rot_mnist(model, dataloaders, config)

    print(f"\nResults:")
    print(f"  Best epoch: {result['best_epoch']}")
    print(f"  Best val accuracy: {result['best_val_accuracy']:.4f}")
    print(f"  Test rotated accuracy: {result['test_rotated']['accuracy']:.4f}")
    print(f"  Test unrotated accuracy: {result['test_unrotated']['accuracy']:.4f}")
    print(f"  Rotated-unrotated gap: "
          f"{result['test_rotated']['accuracy'] - result['test_unrotated']['accuracy']:+.4f}")
    print(f"  Mean epoch time: {result['mean_epoch_seconds']:.2f}s")
    print(f"  Checkpoint: {result['checkpoint_path']}")

    # Optional analyses for geometric models
    run_dir = Path(result["results_dir"])

    if args.run_filter_analysis and config.model_type in ("circle_conv", "helix_conv"):
        print(f"\nRunning filter analysis...")
        from rot_mnist_analyze_filters import run_filter_analysis
        filter_summary = run_filter_analysis(model, run_dir, config.model_type)
        print(f"  Mean corr_star: {filter_summary['mean_corr_star']:.4f}")
        print(f"  Fraction near 90°: {filter_summary['fraction_near_90']:.4f}")

    if args.run_trajectory_analysis and config.model_type in ("circle_conv", "helix_conv"):
        print(f"\nRunning trajectory analysis...")
        from rot_mnist_analyze_trajectory import run_trajectory_analysis
        from torchvision import datasets, transforms
        from rot_mnist_config import resolve_data_dir

        # Get un-rotated test dataset for trajectory base images
        data_dir = resolve_data_dir(config, cli_data_dir)
        unrot_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        unrot_test = datasets.MNIST(root=data_dir, train=False, download=True,
                                    transform=unrot_transform)

        for layer_idx in range(config.num_conv_blocks):
            print(f"  Layer {layer_idx}...")
            run_trajectory_analysis(
                model, unrot_test, run_dir,
                layer_idx=layer_idx, num_digits=3,
                device=device,
            )

    if args.run_intervention_analysis and config.model_type == "helix_conv":
        print(f"\nRunning intervention analysis...")
        from rot_mnist_analyze_intervention import sweep_intervention_angles
        from torchvision import datasets, transforms
        from rot_mnist_config import resolve_data_dir

        data_dir = resolve_data_dir(config, cli_data_dir)
        unrot_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        unrot_test = datasets.MNIST(root=data_dir, train=False, download=True,
                                    transform=unrot_transform)

        sweep_intervention_angles(
            model, unrot_test, run_dir,
            num_images=100, layer_idx=0, device=device,
        )

    return {
        "model_type": config.model_type,
        "scale": config.scale,
        "param_count": n_params,
        "best_epoch": result["best_epoch"],
        "best_val_accuracy": result["best_val_accuracy"],
        "test_rotated_accuracy": result["test_rotated"]["accuracy"],
        "test_rotated_loss": result["test_rotated"]["loss"],
        "test_unrotated_accuracy": result["test_unrotated"]["accuracy"],
        "test_unrotated_loss": result["test_unrotated"]["loss"],
        "mean_epoch_seconds": result["mean_epoch_seconds"],
    }


def main() -> None:
    args = parse_args()

    # Build base config
    config = RotMNISTConfig()

    if args.quick:
        config.epochs = 1
        config.scale = "small"
        config.limit_train_batches = 50
        config.limit_eval_batches = 20

    # Apply scale (CLI overrides quick default)
    if args.scale is not None:
        config.scale = args.scale

    # Apply CLI overrides
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_channels": args.hidden_channels,
        "matched_hidden_channels": args.matched_hidden_channels,
        "circle_units": args.circle_units,
        "helix_units": args.helix_units,
        "kernel_size": args.kernel_size,
        "num_conv_blocks": args.num_conv_blocks,
        "rotation_max_degrees": args.rotation_max_degrees,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
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
        model_types = ["standard_cnn", "standard_cnn_matched", "circle_conv", "helix_conv"]
    elif args.model_type:
        model_types = [args.model_type]
    else:
        model_types = ["helix_conv"]

    # Run experiments
    all_results = []
    for scale in scales:
        config.scale = scale
        apply_scale_preset(config)

        for mt in model_types:
            config.model_type = mt
            result = run_single(config, args.data_dir, args)
            all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Scale':<8} {'Params':>10} {'Val Acc':>10} "
          f"{'Rot Acc':>10} {'Unrot Acc':>10} {'Time/Ep':>10}")
    print("-" * 95)
    for r in all_results:
        print(f"{r['model_type']:<25} {r['scale']:<8} {r['param_count']:>10,} "
              f"{r['best_val_accuracy']:>10.4f} {r['test_rotated_accuracy']:>10.4f} "
              f"{r['test_unrotated_accuracy']:>10.4f} {r['mean_epoch_seconds']:>9.2f}s")

    # Save comparison
    if len(all_results) > 1:
        results_dir = Path(config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        if args.sweep_scales:
            filename = f"comparison_all_scales_seed{config.seed}.json"
        else:
            filename = f"comparison_{config.scale}_seed{config.seed}.json"

        save_json({
            "dataset": "Rotated MNIST",
            "input": "1x28x28_rotated",
            "seed": config.seed,
            "epochs": config.epochs,
            "results": all_results,
        }, results_dir / filename)
        print(f"\nComparison saved to {results_dir / filename}")


if __name__ == "__main__":
    main()
