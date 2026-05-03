# Experiments

All experiments are run from the `src/` directory.

```bash
cd src
```

## Experiment 1: Modular Addition

Tests whether HelixLayer learns causal helical structure on modular arithmetic.

```bash
# Tests
python test_all.py

# Quick sanity check
python run_experiment.py --quick

# Full run
python run_experiment.py --all-models
```

## Experiment 2: MNIST

Tests trainability of geometric MLP layers on easy image classification.

```bash
# Tests
python mnist_test_all.py
python mnist_test_all.py --data
python mnist_test_all.py --slow

# Quick sanity check
python mnist_run_experiment.py --quick --all-models

# Full run
python mnist_run_experiment.py --all-models
```

## Experiment 3: Flattened CIFAR-10

Tests geometric MLPs on hard nonlocal pixel classification (no convolution, no augmentation).

```bash
# Tests
python cifar10_test_all.py
python cifar10_test_all.py --data
python cifar10_test_all.py --slow

# Quick sanity check
python cifar10_run_experiment.py --quick --all-models

# Single scale
python cifar10_run_experiment.py --all-models --scale small
python cifar10_run_experiment.py --all-models --scale medium
python cifar10_run_experiment.py --all-models --scale large

# All scales
python cifar10_run_experiment.py --all-models --sweep-scales
```

## Experiment 4: Covertype

Tests geometric MLPs on heterogeneous tabular classification (54 mixed features, 7 classes).

```bash
# Tests
python covertype_test_all.py
python covertype_test_all.py --data
python covertype_test_all.py --slow

# Quick sanity check
python covertype_run_experiment.py --quick --all-models

# Single scale
python covertype_run_experiment.py --all-models --scale small
python covertype_run_experiment.py --all-models --scale medium
python covertype_run_experiment.py --all-models --scale large

# All scales
python covertype_run_experiment.py --all-models --sweep-scales
```

## Common Options

Most experiment runners support:

| Flag | Description |
|---|---|
| `--quick` | 1 epoch, limited batches (smoke test) |
| `--all-models` | Run all 4 model variants |
| `--model-type X` | Run a single model (standard_mlp, standard_mlp_matched, circle_mlp, helix_mlp) |
| `--scale X` | Set scale (small, medium, large) |
| `--sweep-scales` | Run all scales |
| `--seed N` | Set random seed |
| `--epochs N` | Override epoch count |
| `--device cpu` | Force CPU |
| `--data-dir PATH` | Override dataset directory |
| `--use-scheduler` | Enable cosine LR scheduler |

## Data Directory

Datasets are resolved in order:

1. `--data-dir` CLI flag
2. `HELIX_DATA_DIR` environment variable
3. Default (experiment-specific, e.g. `./covertype_data/`)

To point all experiments at a shared dataset location:

```bash
export HELIX_DATA_DIR=E:/ml_datasets
```

## Dependencies

```
torch
torchvision
numpy
matplotlib
tqdm
scikit-learn
```
