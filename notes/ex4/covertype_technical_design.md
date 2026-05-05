# Technical Design: Covertype Classification with Geometric MLP Layers

## 1. Purpose

This document specifies the implementation design for a Covertype tabular classification experiment using dense, circle, and helix MLP models.

The experiment follows the flattened CIFAR-10 result, where Circle MLP and Helix MLP outperformed dense MLP baselines on a hard flattened-pixel benchmark. Covertype tests a different domain: heterogeneous tabular data.

The central question is:

```text
Can Circle MLP and Helix MLP remain competitive with dense MLP baselines on tabular classification?
```

## 2. Scope

Dataset:

```text
Forest CoverType / Covertype
```

Input:

```text
54 tabular features
```

Output:

```text
7-class forest cover type label
```

Model families:

```text
standard_mlp
standard_mlp_matched
circle_mlp
helix_mlp
```

Primary metrics:

```text
accuracy
macro F1
test loss
```

Secondary metrics:

```text
weighted F1
per-class accuracy
confusion matrix
parameter count
mean epoch time
examples per second
```

## 3. Non-Goals

This experiment should not:

1. attempt to beat all tabular ML methods;
2. use XGBoost/LightGBM as the main baseline in v1;
3. claim tabular data is naturally helical;
4. claim learned helix axes are causal;
5. use external experiment trackers;
6. convert the repo into an installable package.

This is a local script-based experiment comparing neural feedforward primitives.

## 4. Local File Layout

Add these files:

```text
covertype_config.py
covertype_data.py
covertype_models.py
covertype_train.py
covertype_run_experiment.py
covertype_test_all.py
covertype_results.md
```

Generated directories:

```text
covertype_data/
covertype_results/
covertype_checkpoints/
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

Recommended:

```text
sklearn
```

`sklearn` is useful for:

```text
fetch_covtype
train_test_split
accuracy_score
f1_score
confusion_matrix
classification_report
```

If avoiding `sklearn` is preferred, download/load the dataset manually and implement metrics locally. But `sklearn` is acceptable for this experiment.

## 6. Configuration Design

File:

```text
covertype_config.py
```

### 6.1 CovertypeConfig

```python
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
```

```python
@dataclass
class CovertypeConfig:
    model_type: Literal[
        "standard_mlp",
        "standard_mlp_matched",
        "circle_mlp",
        "helix_mlp",
    ] = "helix_mlp"

    scale: Literal["small", "medium", "large"] = "medium"

    batch_size: int = 1024
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.05
    use_layernorm: bool = True

    input_dim: int = 54
    num_classes: int = 7

    hidden_dim: int = 128
    matched_hidden_dim: int = 192
    circle_units: int = 64
    helix_units: int = 64
    num_layers: int = 2

    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15

    seed: int = 0
    device: str = "cuda"

    data_dir: str = "covertype_data"
    results_dir: str = "covertype_results"
    checkpoint_dir: str = "covertype_checkpoints"

    limit_train_batches: int | None = None
    limit_eval_batches: int | None = None

    use_scheduler: bool = False
    scheduler_type: Literal["none", "cosine"] = "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

### 6.2 Scale Presets

Use smaller scales than CIFAR-10 because the input dimension is only 54.

```python
SCALE_PRESETS = {
    "small": {
        "hidden_dim": 64,
        "circle_units": 32,
        "helix_units": 32,
        "matched_hidden_dim": 96,
    },
    "medium": {
        "hidden_dim": 128,
        "circle_units": 64,
        "helix_units": 64,
        "matched_hidden_dim": 192,
    },
    "large": {
        "hidden_dim": 256,
        "circle_units": 128,
        "helix_units": 128,
        "matched_hidden_dim": 384,
    },
}
```

Implement:

```python
def apply_scale_preset(config: CovertypeConfig) -> CovertypeConfig:
    ...
```

### 6.3 Utility Functions

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
def set_seed(seed: int) -> None:
    ...
```

## 7. Data Design

File:

```text
covertype_data.py
```

### 7.1 Loading the Dataset

Preferred v1 approach:

```python
from sklearn.datasets import fetch_covtype
```

Load:

```python
dataset = fetch_covtype(data_home=config.data_dir, download_if_missing=True)
X = dataset.data
y = dataset.target
```

Labels from `fetch_covtype` are typically 1-indexed. Convert to 0-indexed:

```python
y = y.astype(np.int64) - 1
```

### 7.2 Feature Preprocessing

Covertype has 54 features.

Use:

```text
features 0:10   continuous numerical features
features 10:54  binary indicator features
```

Preprocessing:

1. Split first.
2. Fit mean/std on training continuous features only.
3. Standardize continuous features in train/val/test.
4. Leave binary features as 0/1.
5. Convert all features to `float32`.

Implementation:

```python
continuous = X[:, :10]
binary = X[:, 10:]
```

For train statistics:

```python
mean = X_train[:, :10].mean(axis=0, keepdims=True)
std = X_train[:, :10].std(axis=0, keepdims=True)
std = np.maximum(std, 1e-6)
```

Apply:

```python
X_train[:, :10] = (X_train[:, :10] - mean) / std
X_val[:, :10] = (X_val[:, :10] - mean) / std
X_test[:, :10] = (X_test[:, :10] - mean) / std
```

### 7.3 Split

Use stratified splits if `sklearn` is available:

```python
from sklearn.model_selection import train_test_split
```

Procedure:

```text
first split train vs temp
then split temp into val/test
```

Default fractions:

```text
train = 70%
val   = 15%
test  = 15%
```

Example:

```python
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=config.seed,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=config.seed,
    stratify=y_temp,
)
```

### 7.4 Dataset Class

Implement:

```python
class TabularTensorDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
```

### 7.5 Dataloader Function

Implement:

```python
def make_covertype_dataloaders(config: CovertypeConfig) -> dict[str, DataLoader]:
    ...
```

Return:

```text
train
val
test
```

Use:

```python
shuffle=True
```

for train and `False` for validation/test.

Suggested dataloader settings:

```python
num_workers = 0
pin_memory = torch.cuda.is_available()
```

For tabular data, `num_workers=0` is often sufficient.

### 7.6 Shape Contract

Batch features:

```text
[batch, 54]
```

Batch targets:

```text
[batch]
```

Model logits:

```text
[batch, 7]
```

## 8. Model Design

File:

```text
covertype_models.py
```

Provide:

```python
count_parameters
StandardMLP
CircleLayer
HelixLayer
CircleMLP
HelixMLP
build_covertype_model
```

The model code can be adapted from CIFAR-10, but without image flattening.

## 9. StandardMLP

```python
class StandardMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 54,
        num_classes: int = 7,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.05,
    ):
        ...
```

For `num_layers=2`:

```text
Linear(54, hidden_dim)
GELU
Dropout
Linear(hidden_dim, hidden_dim)
GELU
Dropout
Linear(hidden_dim, 7)
```

No flattening is needed.

## 10. CircleLayer

Same as previous experiments, but with tabular input.

```python
class CircleLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        units: int,
        output_dim: int,
        eps: float = 1e-6,
        use_layernorm: bool = True,
        dropout: float = 0.05,
    ):
        ...
```

Parameters:

```python
self.u = nn.Parameter(torch.empty(input_dim, units))
self.v = nn.Parameter(torch.empty(input_dim, units))
self.bias_u = nn.Parameter(torch.zeros(units))
self.bias_v = nn.Parameter(torch.zeros(units))
self.out = nn.Linear(units * 5, output_dim)
```

Forward:

```python
a = x @ self.u + self.bias_u
b = x @ self.v + self.bias_v
r = torch.sqrt(a * a + b * b + self.eps)

sin_theta = b / r
cos_theta = a / r

features = torch.cat(
    [sin_theta, cos_theta, r, r * sin_theta, r * cos_theta],
    dim=-1,
)

features = self.dropout(features)
y = self.out(features)
y = self.layernorm(y)
return y
```

## 11. HelixLayer

```python
class HelixLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        units: int,
        output_dim: int,
        eps: float = 1e-6,
        use_layernorm: bool = True,
        dropout: float = 0.05,
    ):
        ...
```

Parameters:

```python
self.u = nn.Parameter(torch.empty(input_dim, units))
self.v = nn.Parameter(torch.empty(input_dim, units))
self.w = nn.Parameter(torch.empty(input_dim, units))

self.bias_u = nn.Parameter(torch.zeros(units))
self.bias_v = nn.Parameter(torch.zeros(units))
self.bias_w = nn.Parameter(torch.zeros(units))

self.out = nn.Linear(units * 8, output_dim)
```

Forward:

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

Do not use `atan2` in the forward pass.

## 12. CircleMLP and HelixMLP

### CircleMLP

```python
class CircleMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 54,
        num_classes: int = 7,
        hidden_dim: int = 128,
        units: int = 64,
        num_layers: int = 2,
        dropout: float = 0.05,
        use_layernorm: bool = True,
    ):
        ...
```

For `num_layers=2`:

```text
CircleLayer(54, units, hidden_dim)
GELU
Dropout
CircleLayer(hidden_dim, units, hidden_dim)
GELU
Dropout
Linear(hidden_dim, 7)
```

### HelixMLP

```python
class HelixMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 54,
        num_classes: int = 7,
        hidden_dim: int = 128,
        units: int = 64,
        num_layers: int = 2,
        dropout: float = 0.05,
        use_layernorm: bool = True,
    ):
        ...
```

For `num_layers=2`:

```text
HelixLayer(54, units, hidden_dim)
GELU
Dropout
HelixLayer(hidden_dim, units, hidden_dim)
GELU
Dropout
Linear(hidden_dim, 7)
```

## 13. Model Factory

Implement:

```python
def build_covertype_model(config: CovertypeConfig) -> nn.Module:
    ...
```

Dispatch on:

```text
standard_mlp
standard_mlp_matched
circle_mlp
helix_mlp
```

Use `config.input_dim` and `config.num_classes`.

## 14. Parameter Counting

Implement:

```python
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

Print and save parameter counts for every run.

## 15. Metrics Design

File:

```text
covertype_train.py
```

Use local metric helpers.

### 15.1 Accuracy

```python
def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()
```

### 15.2 Macro F1

Preferred: use sklearn.

```python
from sklearn.metrics import f1_score
```

At evaluation time, collect all predictions and targets:

```python
macro_f1 = f1_score(y_true, y_pred, average="macro")
weighted_f1 = f1_score(y_true, y_pred, average="weighted")
```

### 15.3 Confusion Matrix

Use:

```python
from sklearn.metrics import confusion_matrix
```

Save confusion matrix to JSON as nested lists.

## 16. Training Design

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
) -> dict[str, Any]:
    ...
```

Implement:

```python
def fit_covertype(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    config: CovertypeConfig,
) -> dict[str, Any]:
    ...
```

### 16.1 Loss

Use:

```python
F.cross_entropy(logits, targets)
```

For v1, do not use class weighting unless class imbalance causes obvious issues. Report macro F1 to handle imbalance in evaluation.

### 16.2 Optimizer

Use:

```python
torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
)
```

### 16.3 Scheduler

V1 can skip scheduler.

Optional:

```python
torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config.epochs,
)
```

### 16.4 Checkpoint Selection

Use best validation macro F1 as the primary checkpoint criterion.

Tie-breaker:

```text
validation accuracy
then validation loss
```

This is tabular and potentially imbalanced, so macro F1 is a better primary criterion than accuracy.

### 16.5 History

Track per epoch:

```text
train_loss
train_accuracy
train_seconds
train_examples_per_second
val_loss
val_accuracy
val_macro_f1
val_weighted_f1
```

### 16.6 Test Metrics

After restoring the best checkpoint, evaluate on test set and save:

```text
test_loss
test_accuracy
test_macro_f1
test_weighted_f1
confusion_matrix
per_class_accuracy
```

## 17. Artifacts

Per run directory:

```text
covertype_results/<model_type>_<scale>_seed<seed>/
```

Save:

```text
metrics.json
history.json
confusion_matrix.json
training_history.png
```

Checkpoint:

```text
covertype_checkpoints/<model_type>_<scale>_seed<seed>_best.pt
```

Comparison files:

```text
covertype_results/comparison_<scale>_seed<seed>.json
covertype_results/comparison_all_scales_seed<seed>.json
```

## 18. Experiment Runner

File:

```text
covertype_run_experiment.py
```

### 18.1 CLI Arguments

Support:

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
--data-dir
--limit-train-batches
--limit-eval-batches
--use-scheduler
--scheduler-type
```

### 18.2 Quick Mode

Quick mode:

```text
epochs = 1
scale = small
limit_train_batches = 50
limit_eval_batches = 20
```

Quick mode is only for smoke testing.

### 18.3 All Models

Run:

```text
standard_mlp
standard_mlp_matched
circle_mlp
helix_mlp
```

### 18.4 Sweep Scales

Run:

```text
small
medium
large
```

## 19. Plotting

Generate training-history plots.

For each run:

```text
training_history.png
```

Plot:

```text
train_loss
val_loss
train_accuracy
val_accuracy
val_macro_f1
```

Comparison plots:

```text
accuracy_vs_params.png
macro_f1_vs_params.png
macro_f1_vs_epoch_time.png
```

## 20. Testing Design

File:

```text
covertype_test_all.py
```

Support:

```text
--data
--slow
```

### 20.1 Fast Tests

Do not require dataset download.

Tests:

```text
test_config_defaults
test_apply_scale_preset
test_standard_mlp_forward_shape
test_circle_layer_forward_shape
test_helix_layer_forward_shape
test_all_models_no_nans
test_all_models_backward_pass
test_count_parameters_positive
test_synthetic_training_step
test_macro_f1_helper
```

Synthetic batch shape:

```text
features: [batch, 54]
targets:  [batch]
logits:   [batch, 7]
```

### 20.2 Data Tests

Run with:

```bash
python covertype_test_all.py --data
```

Tests:

```text
test_covertype_loads
test_feature_shape_54
test_num_classes_7
test_split_sizes
test_split_deterministic
test_continuous_features_standardized
test_binary_features_remain_binary
```

### 20.3 Slow Tests

Run with:

```bash
python covertype_test_all.py --slow
```

Tiny overfit tests:

```text
test_overfit_tiny_batch_standard
test_overfit_tiny_batch_circle
test_overfit_tiny_batch_helix
```

Use a small fixed batch and verify models can fit it.

Suggested thresholds:

```text
standard_mlp tiny-batch accuracy >= 90%
circle_mlp tiny-batch accuracy >= 85%
helix_mlp tiny-batch accuracy >= 85%
```

## 21. Acceptance Criteria

The implementation is complete when:

1. `python covertype_test_all.py` passes.
2. `python covertype_test_all.py --data` passes.
3. `python covertype_run_experiment.py --quick --model-type helix_mlp` runs.
4. `python covertype_run_experiment.py --quick --all-models` runs.
5. Full small-scale run completes for all models.
6. Full medium-scale run completes for all models.
7. Metrics include accuracy, macro F1, weighted F1, test loss, and epoch time.
8. Confusion matrices are saved.
9. Comparison JSON is saved.
10. Results document is written cautiously.

## 22. Recommended Run Order

```bash
python covertype_test_all.py
python covertype_test_all.py --data

python covertype_run_experiment.py --quick --model-type helix_mlp
python covertype_run_experiment.py --quick --all-models

python covertype_run_experiment.py --all-models --scale small
python covertype_run_experiment.py --all-models --scale medium
python covertype_run_experiment.py --all-models --scale large

python covertype_run_experiment.py --all-models --sweep-scales
```

Then repeat key scales with more seeds.

## 23. Results Writeup Plan

File:

```text
covertype_results.md
```

Include:

```text
overview
dataset and preprocessing
model variants
summary table
accuracy and macro F1 comparisons
training-rate comparison
confusion matrix notes
per-class behavior
scope of claim
next steps
```

## 24. Core Design Summary

Covertype tests whether geometric MLP layers work on heterogeneous tabular data.

The key comparison is not against CNNs or gradient boosting. It is:

```text
dense MLP vs Circle MLP vs Helix MLP
```

The central metrics are:

```text
accuracy
macro F1
test loss
mean epoch time
```

In plain terms:

```text
Give the models 54 mixed tabular columns.
No images. No cyclic structure. No easy wins.
See which architecture generalizes.
```
