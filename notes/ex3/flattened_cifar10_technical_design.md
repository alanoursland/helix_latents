# Technical Design: Flattened CIFAR-10 HelixLayer Experiment

## 1. Purpose

This document specifies the implementation design for a flattened CIFAR-10 classification experiment using dense, circle, and helix feedforward models.

The previous MNIST experiment showed that `CircleLayer` and `HelixLayer` can train and generalize on a simple image-classification task. Flattened CIFAR-10 is a harder test: images are RGB, higher-dimensional, and more visually complex, while the flattened setup removes the locality bias that convolutional models normally exploit.

The central question is:

```text
Can Helix MLP remain competitive with dense MLP baselines on flattened CIFAR-10?
```

This experiment is not intended to compete with CNNs or modern vision models. It compares fully connected architectures on intentionally bad terrain.

## 2. Scope

Dataset:

```text
CIFAR-10
```

Input:

```text
RGB image, shape [3, 32, 32]
flattened input dimension: 3072
```

Output:

```text
10-way class prediction
```

Model families:

```text
standard_mlp
standard_mlp_matched
circle_mlp
helix_mlp
```

Main comparison:

```text
accuracy vs parameter count
```

Secondary comparisons:

```text
training stability
time per epoch
examples per second
train-test gap
```

## 3. Non-Goals

This experiment should not:

1. use convolutional layers;
2. use data augmentation in v1;
3. compare against CNN SOTA;
4. introduce rotation, scale, or other synthetic geometry;
5. claim causal learned helix structure;
6. use external experiment trackers;
7. convert the repo into an installable package.

This is a local, flat-script experiment.

## 4. Local File Layout

Add the following files alongside the existing local experiment files:

```text
cifar10_config.py
cifar10_data.py
cifar10_models.py
cifar10_train.py
cifar10_run_experiment.py
cifar10_test_all.py
cifar10_results.md
```

Generated local directories:

```text
cifar10_data/
cifar10_results/
cifar10_checkpoints/
```

Recommended layout:

```text
src/
  utils.py
  plotting.py

  mnist_config.py
  mnist_data.py
  mnist_models.py
  mnist_train.py
  mnist_run_experiment.py
  mnist_test_all.py

  cifar10_config.py
  cifar10_data.py
  cifar10_models.py
  cifar10_train.py
  cifar10_run_experiment.py
  cifar10_test_all.py
  cifar10_results.md

  cifar10_data/
  cifar10_results/
  cifar10_checkpoints/
```

Do not add package scaffolding.

## 5. Dependencies

Required:

```text
torch
torchvision
numpy
matplotlib
tqdm
```

Optional:

```text
pandas
```

The implementation should run without `wandb`, `sklearn`, or any cloud service.

## 6. Configuration Design

File:

```text
cifar10_config.py
```

### 6.1 CIFAR10Config

Define:

```python
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
```

```python
@dataclass
class CIFAR10Config:
    model_type: Literal[
        "standard_mlp",
        "standard_mlp_matched",
        "circle_mlp",
        "helix_mlp",
    ] = "helix_mlp"

    scale: Literal["small", "medium", "large"] = "medium"

    batch_size: int = 128
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.1
    use_layernorm: bool = True

    hidden_dim: int = 512
    matched_hidden_dim: int = 768
    circle_units: int = 256
    helix_units: int = 256
    num_layers: int = 2

    val_size: int = 5000
    seed: int = 0
    device: str = "cuda"

    data_dir: str = "cifar10_data"
    results_dir: str = "cifar10_results"
    checkpoint_dir: str = "cifar10_checkpoints"

    limit_train_batches: int | None = None
    limit_eval_batches: int | None = None

    use_scheduler: bool = False
    scheduler_type: Literal["none", "cosine"] = "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

### 6.2 Scale Presets

Implement:

```python
SCALE_PRESETS = {
    "small": {
        "hidden_dim": 256,
        "circle_units": 128,
        "helix_units": 128,
        "matched_hidden_dim": 384,
    },
    "medium": {
        "hidden_dim": 512,
        "circle_units": 256,
        "helix_units": 256,
        "matched_hidden_dim": 768,
    },
    "large": {
        "hidden_dim": 1024,
        "circle_units": 512,
        "helix_units": 512,
        "matched_hidden_dim": 1536,
    },
}
```

Implement:

```python
def apply_scale_preset(config: CIFAR10Config) -> CIFAR10Config:
    ...
```

This should mutate or return a config with dimensions set from `config.scale`.

### 6.3 Utilities

Implement:

```python
def get_device(requested: str = "cuda") -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
```

Implement:

```python
def save_json(data: dict, path: str | Path) -> None:
    ...
```

Optional:

```python
def load_json(path: str | Path) -> dict:
    ...
```

## 7. Data Design

File:

```text
cifar10_data.py
```

Use:

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
```

### 7.1 Transform

Use standard CIFAR-10 normalization.

```python
CIFAR10_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
])
```

Do not use augmentation in v1.

### 7.2 DataLoader Function

Implement:

```python
def make_cifar10_dataloaders(config: CIFAR10Config) -> dict[str, DataLoader]:
    ...
```

Behavior:

1. Download CIFAR-10 into `config.data_dir`.
2. Load training set.
3. Split training set into train/validation using `config.val_size`.
4. Load test set.
5. Return:

```text
train
val
test
```

### 7.3 Split

CIFAR-10 training set size is 50,000.

For default `val_size=5000`:

```text
train: 45,000
val: 5,000
test: 10,000
```

Use a seeded generator:

```python
generator = torch.Generator().manual_seed(config.seed)
train_ds, val_ds = random_split(
    full_train_ds,
    [50000 - config.val_size, config.val_size],
    generator=generator,
)
```

### 7.4 DataLoader Settings

Recommended:

```python
num_workers = 2
pin_memory = torch.cuda.is_available()
```

Use:

```python
shuffle=True
```

for training, and:

```python
shuffle=False
```

for validation/test.

### 7.5 Shape Contract

Images are returned as:

```text
[batch, 3, 32, 32]
```

Models should flatten internally.

## 8. Model Design

File:

```text
cifar10_models.py
```

This file should provide:

```python
count_parameters
StandardMLP
CircleLayer
HelixLayer
CircleMLP
HelixMLP
build_cifar10_model
find_param_matched_hidden_dim  # optional
```

If MNIST model code already exists, copy and adapt it rather than introducing shared package abstractions. Keep this local and simple.

## 9. StandardMLP Design

### 9.1 Constructor

```python
class StandardMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 3072,
        num_classes: int = 10,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        ...
```

### 9.2 Architecture

For `num_layers=2`:

```text
Flatten
Linear(3072, hidden_dim)
GELU
Dropout
Linear(hidden_dim, hidden_dim)
GELU
Dropout
Linear(hidden_dim, 10)
```

For `num_layers > 2`, add additional hidden blocks.

### 9.3 Forward

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.flatten(x)
    return self.net(x)
```

Output:

```text
[batch, 10]
```

## 10. CircleLayer Design

This should match the MNIST CircleLayer design but support larger input dimensions.

### 10.1 Constructor

```python
class CircleLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        units: int,
        output_dim: int,
        eps: float = 1e-6,
        use_layernorm: bool = True,
        dropout: float = 0.1,
    ):
        ...
```

### 10.2 Parameters

```python
self.u = nn.Parameter(torch.empty(input_dim, units))
self.v = nn.Parameter(torch.empty(input_dim, units))

self.bias_u = nn.Parameter(torch.zeros(units))
self.bias_v = nn.Parameter(torch.zeros(units))

self.out = nn.Linear(units * 5, output_dim)
```

### 10.3 Initialization

```python
scale = 1.0 / math.sqrt(input_dim)
nn.init.normal_(self.u, mean=0.0, std=scale)
nn.init.normal_(self.v, mean=0.0, std=scale)
```

### 10.4 Forward

```python
a = x @ self.u + self.bias_u
b = x @ self.v + self.bias_v
r = torch.sqrt(a * a + b * b + self.eps)

sin_theta = b / r
cos_theta = a / r

features = torch.cat(
    [
        sin_theta,
        cos_theta,
        r,
        r * sin_theta,
        r * cos_theta,
    ],
    dim=-1,
)

features = self.dropout(features)
y = self.out(features)
y = self.layernorm(y)
return y
```

Output:

```text
[batch, output_dim]
```

## 11. HelixLayer Design

### 11.1 Constructor

```python
class HelixLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        units: int,
        output_dim: int,
        eps: float = 1e-6,
        use_layernorm: bool = True,
        dropout: float = 0.1,
    ):
        ...
```

### 11.2 Parameters

```python
self.u = nn.Parameter(torch.empty(input_dim, units))
self.v = nn.Parameter(torch.empty(input_dim, units))
self.w = nn.Parameter(torch.empty(input_dim, units))

self.bias_u = nn.Parameter(torch.zeros(units))
self.bias_v = nn.Parameter(torch.zeros(units))
self.bias_w = nn.Parameter(torch.zeros(units))

self.out = nn.Linear(units * 8, output_dim)
```

### 11.3 Initialization

```python
scale = 1.0 / math.sqrt(input_dim)
nn.init.normal_(self.u, mean=0.0, std=scale)
nn.init.normal_(self.v, mean=0.0, std=scale)
nn.init.normal_(self.w, mean=0.0, std=scale)
```

### 11.4 Forward

```python
a = x @ self.u + self.bias_u
b = x @ self.v + self.bias_v
z = x @ self.w + self.bias_w

r = torch.sqrt(a * a + b * b + self.eps)

sin_theta = b / r
cos_theta = a / r

features = torch.cat(
    [
        sin_theta,
        cos_theta,
        r,
        z,
        r * sin_theta,
        r * cos_theta,
        torch.tanh(z),
        r * torch.tanh(z),
    ],
    dim=-1,
)

features = self.dropout(features)
y = self.out(features)
y = self.layernorm(y)
return y
```

### 11.5 Numerical Stability

Do not use `atan2` in the forward pass.

Use:

```python
eps = 1e-6
```

Assert with tests that:

```python
torch.isfinite(output).all()
```

for random input scales:

```text
0.01
1.0
10.0
```

## 12. CircleMLP and HelixMLP

### 12.1 CircleMLP

```python
class CircleMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 3072,
        num_classes: int = 10,
        hidden_dim: int = 512,
        units: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_layernorm: bool = True,
    ):
        ...
```

For `num_layers=2`:

```text
Flatten
CircleLayer(3072, units, hidden_dim)
GELU
Dropout
CircleLayer(hidden_dim, units, hidden_dim)
GELU
Dropout
Linear(hidden_dim, 10)
```

### 12.2 HelixMLP

```python
class HelixMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 3072,
        num_classes: int = 10,
        hidden_dim: int = 512,
        units: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_layernorm: bool = True,
    ):
        ...
```

For `num_layers=2`:

```text
Flatten
HelixLayer(3072, units, hidden_dim)
GELU
Dropout
HelixLayer(hidden_dim, units, hidden_dim)
GELU
Dropout
Linear(hidden_dim, 10)
```

### 12.3 LayerNorm Placement

Each CircleLayer/HelixLayer can apply LayerNorm internally after its output projection.

The model then applies GELU after the layer.

```python
x = F.gelu(self.layer1(x))
x = F.gelu(self.layer2(x))
logits = self.classifier(x)
```

Do not apply LayerNorm after the final classifier.

## 13. Model Factory

Implement:

```python
def build_cifar10_model(config: CIFAR10Config) -> nn.Module:
    ...
```

Dispatch:

```python
if config.model_type == "standard_mlp":
    return StandardMLP(
        input_dim=3072,
        num_classes=10,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )

if config.model_type == "standard_mlp_matched":
    return StandardMLP(
        input_dim=3072,
        num_classes=10,
        hidden_dim=config.matched_hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )

if config.model_type == "circle_mlp":
    return CircleMLP(
        input_dim=3072,
        num_classes=10,
        hidden_dim=config.hidden_dim,
        units=config.circle_units,
        num_layers=config.num_layers,
        dropout=config.dropout,
        use_layernorm=config.use_layernorm,
    )

if config.model_type == "helix_mlp":
    return HelixMLP(
        input_dim=3072,
        num_classes=10,
        hidden_dim=config.hidden_dim,
        units=config.helix_units,
        num_layers=config.num_layers,
        dropout=config.dropout,
        use_layernorm=config.use_layernorm,
    )
```

Raise `ValueError` on unknown model type.

## 14. Parameter Counting

Implement:

```python
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

Every run must save and print parameter count.

### 14.1 Optional Parameter Matching Helper

Implement later if useful:

```python
def find_param_matched_hidden_dim(
    target_params: int,
    candidate_dims: list[int],
    config: CIFAR10Config,
) -> int:
    ...
```

Process:

1. Build target HelixMLP.
2. Count params.
3. Build dense MLPs for candidate hidden dims.
4. Pick closest parameter count.

For v1, manual `matched_hidden_dim` presets are acceptable.

## 15. Training Design

File:

```text
cifar10_train.py
```

### 15.1 Functions

Implement:

```python
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    limit_batches: int | None = None,
) -> dict[str, float]:
    ...
```

Implement:

```python
@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    limit_batches: int | None = None,
) -> dict[str, float]:
    ...
```

Implement:

```python
def fit_cifar10(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    config: CIFAR10Config,
) -> dict[str, Any]:
    ...
```

### 15.2 Metrics

Track:

```text
loss
accuracy
num_examples
elapsed_seconds
examples_per_second
```

For `train_one_epoch`, track epoch duration.

### 15.3 Optimizer

Use AdamW:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
)
```

### 15.4 Scheduler

V1 can use no scheduler.

If `config.use_scheduler` is true and `config.scheduler_type == "cosine"`:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config.epochs,
)
```

Call `scheduler.step()` once per epoch after validation.

### 15.5 Fit Loop Behavior

The fit loop should:

1. move model to device;
2. create optimizer;
3. optionally create scheduler;
4. train for `config.epochs`;
5. evaluate validation after each epoch;
6. save best checkpoint by validation accuracy;
7. break ties by validation loss;
8. restore best model state;
9. evaluate test set;
10. save metrics and history JSON;
11. return a result dictionary.

### 15.6 History Keys

```text
train_loss
train_accuracy
train_seconds
train_examples_per_second
val_loss
val_accuracy
```

Optional:

```text
learning_rate
```

### 15.7 Checkpoint Format

Save to:

```text
cifar10_checkpoints/<model_type>_<scale>_seed<seed>_best.pt
```

Checkpoint dictionary:

```python
{
    "model_state_dict": best_state_dict,
    "config": config.to_dict(),
    "model_type": config.model_type,
    "scale": config.scale,
    "seed": config.seed,
    "param_count": param_count,
    "history": history,
    "best_epoch": best_epoch,
    "best_val_accuracy": best_val_accuracy,
    "best_val_loss": best_val_loss,
    "test_metrics": test_metrics,
}
```

### 15.8 Metrics JSON

Save to:

```text
cifar10_results/<model_type>_<scale>_seed<seed>/metrics.json
```

Contents:

```python
{
    "model_type": config.model_type,
    "scale": config.scale,
    "seed": config.seed,
    "param_count": param_count,
    "best_epoch": best_epoch,
    "best_val_accuracy": best_val_accuracy,
    "best_val_loss": best_val_loss,
    "test_accuracy": test_metrics["accuracy"],
    "test_loss": test_metrics["loss"],
    "train_time_total_seconds": ...,
    "mean_epoch_seconds": ...,
}
```

## 16. Experiment Runner

File:

```text
cifar10_run_experiment.py
```

### 16.1 CLI Arguments

Use `argparse`.

Required/optional arguments:

```text
--model-type
--all-models
--scale
--sweep-scales
--quick
--epochs
--batch-size
--hidden-dim
--matched-hidden-dim
--circle-units
--helix-units
--num-layers
--dropout
--learning-rate
--weight-decay
--seed
--device
--limit-train-batches
--limit-eval-batches
--use-scheduler
--scheduler-type
```

### 16.2 Model Types

Allowed:

```text
standard_mlp
standard_mlp_matched
circle_mlp
helix_mlp
```

### 16.3 Quick Mode

If `--quick` is passed, override:

```text
epochs = 1
scale = small
limit_train_batches = 100
limit_eval_batches = 50
```

Quick mode should be explicitly labeled as non-reportable.

### 16.4 All Models

If `--all-models`, run:

```text
standard_mlp
standard_mlp_matched
circle_mlp
helix_mlp
```

### 16.5 Sweep Scales

If `--sweep-scales`, run:

```text
small
medium
large
```

If combined with `--all-models`, run all models at all scales.

### 16.6 Run Function

Implement:

```python
def run_single(config: CIFAR10Config) -> dict[str, Any]:
    ...
```

Steps:

1. apply scale preset;
2. set seed;
3. build dataloaders;
4. build model;
5. count params;
6. print run summary;
7. train with `fit_cifar10`;
8. return compact result.

### 16.7 Comparison JSON

After multiple runs, save:

```text
cifar10_results/comparison_<scale>_seed<seed>.json
```

For scale sweeps:

```text
cifar10_results/comparison_all_scales_seed<seed>.json
```

Format:

```python
{
    "dataset": "CIFAR-10",
    "input": "flattened_pixels",
    "seed": seed,
    "results": [
        {
            "model_type": "...",
            "scale": "...",
            "param_count": 123,
            "best_epoch": 12,
            "best_val_accuracy": 0.55,
            "test_accuracy": 0.54,
            "test_loss": 1.23,
            "mean_epoch_seconds": 8.5,
        }
    ]
}
```

### 16.8 Console Table

Print:

```text
Model                  Scale    Params      Best Val    Test Acc    Time/Epoch
----------------------------------------------------------------------------
standard_mlp           small    ...
standard_mlp_matched   small    ...
circle_mlp             small    ...
helix_mlp              small    ...
```

## 17. Plotting Design

Reuse or extend `plotting.py`.

### 17.1 Training History Plot

Generate:

```text
cifar10_results/<run_name>/training_history.png
```

Plot:

```text
train_loss
val_loss
train_accuracy
val_accuracy
```

### 17.2 Accuracy vs Parameters Plot

Generate:

```text
cifar10_results/accuracy_vs_params.png
```

X-axis:

```text
parameter count
```

Y-axis:

```text
test accuracy
```

Use one point per model/scale.

Label points with:

```text
model_type + scale
```

### 17.3 Accuracy vs Time Plot

Optional:

```text
cifar10_results/accuracy_vs_epoch_time.png
```

## 18. Testing Design

File:

```text
cifar10_test_all.py
```

Use local script-style tests, as in prior experiments.

### 18.1 Test Runner

Support flags:

```text
--data
--slow
```

Default fast tests should not require downloading CIFAR-10.

### 18.2 Fast Tests

Run with:

```bash
python cifar10_test_all.py
```

Tests:

```text
test_config_defaults
test_apply_scale_preset
test_standard_mlp_forward_shape
test_circle_layer_forward_shape
test_helix_layer_forward_shape
test_circle_mlp_forward_shape
test_helix_mlp_forward_shape
test_all_models_no_nans
test_all_models_backward_pass
test_count_parameters_positive
test_synthetic_training_step
```

### 18.3 Shape Tests

Expected model input:

```text
[batch, 3, 32, 32]
```

Expected logits:

```text
[batch, 10]
```

Example:

```python
images = torch.randn(8, 3, 32, 32)
model = HelixMLP(hidden_dim=64, units=16)
logits = model(images)
assert logits.shape == (8, 10)
```

### 18.4 No-NaN Tests

Test random input scales:

```text
0.01
1.0
10.0
```

For all model types, assert:

```python
torch.isfinite(logits).all()
```

### 18.5 Backward Tests

For each model:

```python
logits = model(images)
loss = F.cross_entropy(logits, targets)
loss.backward()
```

Assert:

```python
at least one trainable parameter has finite nonzero gradient
```

### 18.6 Synthetic Training Step

Create a tiny synthetic dataloader:

```text
N = 64
images ~ Normal(0, 1), shape [N, 3, 32, 32]
labels ~ randint(0, 10)
```

Run one training epoch and assert:

```text
loss is finite
accuracy is between 0 and 1
```

### 18.7 Data Tests

Run with:

```bash
python cifar10_test_all.py --data
```

Tests:

```text
test_cifar10_batch_shape
test_cifar10_target_shape
test_cifar10_num_classes
test_cifar10_train_val_test_sizes
test_cifar10_split_deterministic
```

### 18.8 Slow Tiny-Batch Overfit Tests

Run with:

```bash
python cifar10_test_all.py --slow
```

Tests:

```text
test_overfit_tiny_batch_standard
test_overfit_tiny_batch_circle
test_overfit_tiny_batch_helix
```

Procedure:

1. Load or create one batch of 128 CIFAR-10 examples.
2. Train on the same batch for 200-500 steps.
3. Evaluate on that same batch.

Suggested thresholds:

```text
standard_mlp: >= 80%
circle_mlp:   >= 70%
helix_mlp:    >= 70%
```

If HelixMLP cannot overfit a tiny batch, debug before full runs.

## 19. Acceptance Criteria

The v1 implementation is complete when:

1. `python cifar10_test_all.py` passes.
2. `python cifar10_test_all.py --data` passes.
3. `python cifar10_test_all.py --slow` passes or gives useful diagnostics.
4. `python cifar10_run_experiment.py --quick --model-type helix_mlp` runs.
5. `python cifar10_run_experiment.py --quick --all-models` runs.
6. A full small-scale run completes for all four models.
7. Metrics JSON files are saved.
8. Checkpoints are saved.
9. Training plots are saved.
10. A comparison JSON is saved.
11. Parameter counts are reported.
12. The results writeup is updated with cautious interpretation.

## 20. Recommended Run Order

### 20.1 Initial Sanity

```bash
python cifar10_test_all.py
python cifar10_run_experiment.py --quick --model-type helix_mlp --device cpu
```

### 20.2 Data and Slow Tests

```bash
python cifar10_test_all.py --data
python cifar10_test_all.py --slow
```

### 20.3 Small Scale

```bash
python cifar10_run_experiment.py --all-models --scale small
```

### 20.4 Medium Scale

```bash
python cifar10_run_experiment.py --all-models --scale medium
```

### 20.5 Large Scale

```bash
python cifar10_run_experiment.py --all-models --scale large
```

### 20.6 Full Sweep

```bash
python cifar10_run_experiment.py --all-models --sweep-scales
```

## 21. Interpretation Rules

### 21.1 Helix Matches Dense MLP

If Helix MLP comes within a small gap of the parameter-matched dense MLP:

```text
HelixLayer remains competitive on a hard nonlocal pixel task.
```

This is a good result.

### 21.2 Helix Beats Dense MLP

If Helix MLP beats the parameter-matched dense MLP:

```text
Potential evidence that HelixLayer is an efficient nonlinear primitive in this setup.
```

Required follow-up:

```text
multi-seed runs
parameter sweep
time/compute comparison
feature ablations
```

### 21.3 Helix Trails Slightly

If Helix MLP trails by a few percentage points:

```text
HelixLayer is viable but not advantaged here.
```

This is still useful.

### 21.4 Helix Trails Badly

If Helix MLP falls far behind:

```text
HelixLayer may struggle with high-dimensional nonlocal pixel data.
```

Debug:

```text
tiny-batch overfitting
learning rate
weight decay
dropout
LayerNorm placement
units
feature set
```

### 21.5 Circle Beats Helix

If Circle MLP beats Helix MLP:

```text
Axis features are not helping in this setting.
```

Do not make helix-specific performance claims.

## 22. Feature Ablation Plan

If Helix MLP is competitive, run ablations.

### 22.1 Helix Feature Groups

Current full features:

```text
sin(theta)
cos(theta)
r
z
r * sin(theta)
r * cos(theta)
tanh(z)
r * tanh(z)
```

Ablation variants:

```text
phase_only:       sin(theta), cos(theta)
phase_radius:     sin(theta), cos(theta), r
raw_projection:   r*sin(theta), r*cos(theta), z
axis_only:        z, tanh(z)
no_axis:          sin(theta), cos(theta), r, r*sin(theta), r*cos(theta)
full:             all features
```

### 22.2 Why This Matters

Some features are equivalent to raw projections:

```text
r * sin(theta) = b
r * cos(theta) = a
```

These terms provide a dense-layer-like route through the HelixLayer. That is useful for training stability, but ablations are needed to determine whether the normalized phase or axis features are actually contributing.

## 23. Multi-Seed Plan

Once one scale works, run seeds:

```text
0, 1, 2
```

For stronger claims:

```text
0, 1, 2, 3, 4
```

Save:

```text
cifar10_results/multiseed_comparison.json
```

Report:

```text
mean test accuracy
std test accuracy
mean best val accuracy
std best val accuracy
mean epoch time
```

Do not claim a model wins based on one seed unless the margin is large.

## 24. Results Writeup Plan

File:

```text
cifar10_results.md
```

Structure:

```text
Overview
Summary table
Standard MLP
Parameter-matched Standard MLP
Circle MLP
Helix MLP
Accuracy vs parameter count
Main takeaways
Scope of claim
Implications for tabular experiment
Source artifacts
```

Use cautious language.

Good:

```text
Helix MLP remained within 2 points of the dense matched baseline.
```

Avoid:

```text
HelixLayer is generally better than dense layers.
```

## 25. Next Experiment

If this experiment succeeds or partially succeeds, move to tabular classification.

Recommended first tabular benchmark:

```text
Covertype
```

Other candidates:

```text
HIGGS
UCI Adult
Letter Recognition
Spambase
```

The purpose of tabular classification is to test whether HelixLayer is a general nonlinear feature primitive in a setting with:

```text
no image locality
no sequence order
no obvious phase variable
heterogeneous features
```

## 26. Core Design Summary

Flattened CIFAR-10 is the first serious architecture stress test.

All models receive the same flattened pixels. No model gets convolution, augmentation, or an explicit geometry target. The comparison is about whether the HelixLayer primitive can survive and remain competitive in a hostile high-dimensional setting.

The central number is not absolute CIFAR-10 accuracy.

The central number is:

```text
accuracy gap between Helix MLP and parameter-matched dense MLP
```

In plain terms:

```text
No synthetic cyclic task this time.
Just raw flattened pixels with no spatial structure.
Compare dense and geometric MLPs on difficult ground.
