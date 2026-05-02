# Technical Design: MNIST HelixLayer Experiment

## 1. Purpose

This document specifies the implementation design for a local MNIST experiment using a helix-native neural network layer.

The experiment extends the previous modular addition work from explicit task-aligned geometric bottlenecks into a more general classification setting. The goal is not to show that helices are universally better than standard neural layers. The goal is to test whether a `HelixLayer` can function as a trainable feedforward primitive on ordinary image classification.

The primary question is:

```text
Can a model using HelixLayer units train, overfit small batches, and generalize on MNIST?
```

The secondary question is:

```text
How does HelixMLP compare to ordinary MLP and CircleMLP baselines at similar model sizes?
```

## 2. Scope

This experiment uses MNIST classification.

Input:

```text
28 x 28 grayscale image
```

Output:

```text
digit class in {0, 1, ..., 9}
```

The implementation should remain local and flat. Do not convert the repo into an installable package.

The design assumes files are run from one local source directory, similar to the existing modular addition experiment.

## 3. Non-Goals

This experiment should not attempt to:

1. build a convolutional helix model;
2. beat state-of-the-art MNIST performance;
3. prove helices are generally superior;
4. implement causal phase interventions yet;
5. use CIFAR-10 yet;
6. add package installation or a `src/helix_latents` namespace;
7. depend on external experiment trackers.

This is a viability and comparison experiment.

## 4. Local File Layout

Add the following files alongside the existing flat local files:

```text
mnist_config.py
mnist_data.py
mnist_models.py
mnist_train.py
mnist_run_experiment.py
mnist_test_all.py
mnist_results.md
```

Existing files that can be reused:

```text
utils.py
plotting.py
```

Generated local directories:

```text
data/
mnist_results/
mnist_checkpoints/
```

Recommended final local layout:

```text
src/
  config.py
  data.py
  geometry.py
  intervene.py
  models.py
  plotting.py
  run_experiment.py
  test_all.py
  train.py
  utils.py

  mnist_config.py
  mnist_data.py
  mnist_models.py
  mnist_train.py
  mnist_run_experiment.py
  mnist_test_all.py
  mnist_results.md

  data/
  mnist_results/
  mnist_checkpoints/
```

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

No `wandb`, package install, or cloud dependency should be required.

## 6. Configuration

File:

```text
mnist_config.py
```

Define:

```python
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
```

### 6.1 MNISTConfig

```python
@dataclass
class MNISTConfig:
    model_type: Literal[
        "standard_mlp",
        "standard_mlp_matched",
        "circle_mlp",
        "helix_mlp",
    ] = "helix_mlp"

    batch_size: int = 128
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    hidden_dim: int = 128
    matched_hidden_dim: int = 256
    circle_units: int = 64
    helix_units: int = 64
    num_layers: int = 2
    dropout: float = 0.0
    use_layernorm: bool = True

    val_size: int = 5000
    seed: int = 0
    device: str = "cuda"

    data_dir: str = "data"
    results_dir: str = "mnist_results"
    checkpoint_dir: str = "mnist_checkpoints"

    limit_train_batches: int | None = None
    limit_eval_batches: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

### 6.2 Utility Functions

Add:

```python
def get_device(requested: str = "cuda") -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
```

Add:

```python
def save_json(data: dict, path: str | Path) -> None:
    ...
```

This can mirror the existing `config.py`.

## 7. Data Design

File:

```text
mnist_data.py
```

Use `torchvision.datasets.MNIST`.

### 7.1 Transform

Use standard MNIST normalization:

```python
from torchvision import transforms

MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
```

### 7.2 Data Function

Implement:

```python
def make_mnist_dataloaders(config: MNISTConfig) -> dict[str, DataLoader]:
    ...
```

Behavior:

1. Download MNIST into `config.data_dir`.
2. Load full training set.
3. Split into train and validation using `config.val_size`.
4. Load test set.
5. Return dataloaders for:

```text
train
val
test
```

### 7.3 Deterministic Split

Use a seeded `torch.Generator`.

```python
generator = torch.Generator().manual_seed(config.seed)
train_ds, val_ds = random_split(
    full_train_ds,
    [60000 - config.val_size, config.val_size],
    generator=generator,
)
```

### 7.4 DataLoader Settings

Recommended:

```python
num_workers = 2
pin_memory = torch.cuda.is_available()
```

But keep defaults simple and robust.

```python
DataLoader(
    dataset,
    batch_size=config.batch_size,
    shuffle=True,
)
```

Use `shuffle=True` for train, `False` for validation/test.

## 8. Model Design

File:

```text
mnist_models.py
```

This file contains:

```python
count_parameters
CircleLayer
HelixLayer
StandardMLP
CircleMLP
HelixMLP
build_mnist_model
```

## 9. HelixLayer Design

The `HelixLayer` is the central object.

It maps:

```text
x ∈ R[input_dim] -> y ∈ R[output_dim]
```

using `units` learned helix subspaces.

Each unit learns three input directions:

```text
u_i, v_i, w_i ∈ R[input_dim]
```

For input `x`, compute:

```text
a_i = x · u_i + bias_u_i
b_i = x · v_i + bias_v_i
z_i = x · w_i + bias_w_i
r_i = sqrt(a_i^2 + b_i^2 + eps)
cos_i = a_i / r_i
sin_i = b_i / r_i
```

The layer then constructs nonlinear features per unit.

### 9.1 Why Avoid atan2 in v1

The mathematical phase is:

```text
theta_i = atan2(b_i, a_i)
```

But v1 should not use `atan2` in the forward pass. Instead use:

```text
cos(theta_i) = a_i / r_i
sin(theta_i) = b_i / r_i
```

This avoids explicit angle wrapping and tends to be more stable for gradients.

`atan2` can be added later for analysis and visualization.

### 9.2 Feature Set

For each helix unit, use 8 features:

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

Since:

```text
r * sin(theta) = b
r * cos(theta) = a
```

the feature set includes both normalized phase information and raw projection information. This makes the layer less brittle.

### 9.3 Constructor

```python
class HelixLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        units: int,
        output_dim: int,
        eps: float = 1e-6,
        use_layernorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        ...
```

### 9.4 Parameters

```python
self.u = nn.Parameter(torch.empty(input_dim, units))
self.v = nn.Parameter(torch.empty(input_dim, units))
self.w = nn.Parameter(torch.empty(input_dim, units))

self.bias_u = nn.Parameter(torch.zeros(units))
self.bias_v = nn.Parameter(torch.zeros(units))
self.bias_w = nn.Parameter(torch.zeros(units))

self.out = nn.Linear(units * 8, output_dim)
```

Optional:

```python
self.layernorm = nn.LayerNorm(output_dim) if use_layernorm else nn.Identity()
self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
```

### 9.5 Initialization

Use variance-scaled initialization:

```python
scale = 1.0 / math.sqrt(input_dim)
nn.init.normal_(self.u, mean=0.0, std=scale)
nn.init.normal_(self.v, mean=0.0, std=scale)
nn.init.normal_(self.w, mean=0.0, std=scale)
```

Let `self.out` use PyTorch default initialization or initialize with Xavier uniform.

### 9.6 Forward Pass

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
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

### 9.7 Shape Contract

Input:

```text
x.shape == [batch, input_dim]
```

Output:

```text
y.shape == [batch, output_dim]
```

## 10. CircleLayer Design

The `CircleLayer` is a control for phase/radius features without axis.

Each unit learns two directions:

```text
u_i, v_i ∈ R[input_dim]
```

Compute:

```text
a_i = x · u_i + bias_u_i
b_i = x · v_i + bias_v_i
r_i = sqrt(a_i^2 + b_i^2 + eps)
cos_i = a_i / r_i
sin_i = b_i / r_i
```

Feature set per unit:

```text
sin(theta)
cos(theta)
r
r * sin(theta)
r * cos(theta)
```

So `num_features_per_unit = 5`.

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
        dropout: float = 0.0,
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

### 10.3 Forward Pass

```python
a = x @ self.u + self.bias_u
b = x @ self.v + self.bias_v
r = torch.sqrt(a * a + b * b + eps)

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

return layernorm(out(dropout(features)))
```

## 11. Model Architectures

### 11.1 StandardMLP

```python
class StandardMLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        ...
```

For `num_layers = 2`:

```text
Flatten
Linear(784, hidden_dim)
GELU
Dropout
Linear(hidden_dim, hidden_dim)
GELU
Dropout
Linear(hidden_dim, 10)
```

### 11.2 CircleMLP

```python
class CircleMLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        units: int = 64,
        num_layers: int = 2,
        use_layernorm: bool = True,
        dropout: float = 0.0,
    ):
        ...
```

For `num_layers = 2`:

```text
Flatten
CircleLayer(784, units, hidden_dim)
GELU
CircleLayer(hidden_dim, units, hidden_dim)
GELU
Linear(hidden_dim, 10)
```

### 11.3 HelixMLP

```python
class HelixMLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        units: int = 64,
        num_layers: int = 2,
        use_layernorm: bool = True,
        dropout: float = 0.0,
    ):
        ...
```

For `num_layers = 2`:

```text
Flatten
HelixLayer(784, units, hidden_dim)
GELU
HelixLayer(hidden_dim, units, hidden_dim)
GELU
Linear(hidden_dim, 10)
```

### 11.4 Model Factory

```python
def build_mnist_model(config: MNISTConfig) -> nn.Module:
    if config.model_type == "standard_mlp":
        return StandardMLP(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
    if config.model_type == "standard_mlp_matched":
        return StandardMLP(
            hidden_dim=config.matched_hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
    if config.model_type == "circle_mlp":
        return CircleMLP(
            hidden_dim=config.hidden_dim,
            units=config.circle_units,
            num_layers=config.num_layers,
            use_layernorm=config.use_layernorm,
            dropout=config.dropout,
        )
    if config.model_type == "helix_mlp":
        return HelixMLP(
            hidden_dim=config.hidden_dim,
            units=config.helix_units,
            num_layers=config.num_layers,
            use_layernorm=config.use_layernorm,
            dropout=config.dropout,
        )
    raise ValueError(...)
```

## 12. Parameter Counting and Fairness

Add:

```python
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

Each experiment run should print and save parameter counts.

The main comparison should include:

```text
standard_mlp
standard_mlp_matched
circle_mlp
helix_mlp
```

`standard_mlp_matched` should use a larger or smaller hidden dimension chosen to roughly match HelixMLP parameter count.

It is acceptable for v1 to do approximate matching manually. Later, add a helper that searches hidden sizes.

### 12.1 Optional Hidden-Dim Search

Add:

```python
def find_param_matched_hidden_dim(
    target_params: int,
    candidate_dims: list[int],
    config: MNISTConfig,
) -> int:
    ...
```

This can build `StandardMLP` candidates and return the one closest to target params.

## 13. Training Design

File:

```text
mnist_train.py
```

Implement:

```python
train_one_epoch
evaluate
fit_mnist
```

### 13.1 Metrics

Track:

```text
loss
accuracy
num_examples
```

### 13.2 Train One Epoch

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

Use:

```python
model.train()
loss = F.cross_entropy(logits, targets)
```

### 13.3 Evaluate

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

Use:

```python
model.eval()
```

### 13.4 Fit Loop

```python
def fit_mnist(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    config: MNISTConfig,
) -> dict[str, Any]:
    ...
```

Behavior:

1. set seed before model creation in script, not inside fit;
2. move model to device;
3. create AdamW optimizer;
4. train for `config.epochs`;
5. evaluate validation after each epoch;
6. save best checkpoint by validation accuracy;
7. restore best model state at the end;
8. evaluate test set;
9. save history and metrics.

Optimizer:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
)
```

### 13.5 Checkpoint Format

```python
checkpoint = {
    "model_state_dict": best_state_dict,
    "config": config.to_dict(),
    "model_type": config.model_type,
    "param_count": param_count,
    "history": history,
    "best_epoch": best_epoch,
    "best_val_accuracy": best_val_accuracy,
    "test_metrics": test_metrics,
}
```

Save to:

```text
mnist_checkpoints/<model_type>_best.pt
```

### 13.6 Metrics JSON

Save to:

```text
mnist_results/<model_type>/metrics.json
```

With:

```python
{
    "model_type": config.model_type,
    "param_count": param_count,
    "best_epoch": best_epoch,
    "best_val_accuracy": best_val_accuracy,
    "test_accuracy": test_metrics["accuracy"],
    "test_loss": test_metrics["loss"],
}
```

### 13.7 History JSON

Save to:

```text
mnist_results/<model_type>/history.json
```

History keys:

```text
train_loss
train_accuracy
val_loss
val_accuracy
```

## 14. Experiment Runner

File:

```text
mnist_run_experiment.py
```

### 14.1 CLI

Use `argparse`.

Arguments:

```text
--model-type
--all-models
--quick
--epochs
--batch-size
--hidden-dim
--matched-hidden-dim
--helix-units
--circle-units
--learning-rate
--weight-decay
--seed
--device
--limit-train-batches
--limit-eval-batches
```

### 14.2 Quick Mode

`--quick` should set:

```text
epochs = 1
batch_size = 128
limit_train_batches = 100
limit_eval_batches = 50
```

This mode is for import/training sanity only.

### 14.3 All Models Mode

If `--all-models` is passed, run:

```text
standard_mlp
standard_mlp_matched
circle_mlp
helix_mlp
```

For each:

1. build fresh config;
2. set seed;
3. build dataloaders;
4. build model;
5. count params;
6. train;
7. save results.

### 14.4 Comparison Output

After all models, save:

```text
mnist_results/comparison.json
```

Format:

```python
{
    "dataset": "MNIST",
    "seed": config.seed,
    "results": [
        {
            "model_type": "...",
            "param_count": 12345,
            "best_val_accuracy": 0.98,
            "test_accuracy": 0.97,
            "test_loss": 0.09,
        }
    ]
}
```

Also print a table:

```text
Model                  Params      Best Val    Test Acc
-------------------------------------------------------
standard_mlp           ...
standard_mlp_matched   ...
circle_mlp             ...
helix_mlp              ...
```

## 15. Plotting

Reuse or extend `plotting.py`.

Add local functions if preferred:

```python
def plot_mnist_training_history(history: dict, output_path: str) -> None:
    ...
```

Generate:

```text
mnist_results/<model_type>/training_history.png
```

Optional:

```text
mnist_results/comparison.png
```

The comparison plot can show:

```text
x-axis: parameter count
y-axis: test accuracy
point label: model type
```

Do not make plots required for tests.

## 16. Testing Design

File:

```text
mnist_test_all.py
```

Use local script-style tests like the existing `test_all.py`. It should run with:

```bash
python mnist_test_all.py
```

No `pytest` dependency is required.

### 16.1 Test Runner Structure

Define test functions and a simple runner:

```python
def run_tests():
    tests = [
        test_helix_layer_forward_shape,
        ...
    ]
    ...
```

Print:

```text
PASS test_name
FAIL test_name
```

Exit with nonzero status on failure.

### 16.2 Test: Layer Shapes

```python
def test_helix_layer_forward_shape():
    layer = HelixLayer(input_dim=784, units=16, output_dim=32)
    x = torch.randn(8, 784)
    y = layer(x)
    assert y.shape == (8, 32)
```

Circle:

```python
def test_circle_layer_forward_shape():
    layer = CircleLayer(input_dim=784, units=16, output_dim=32)
    x = torch.randn(8, 784)
    y = layer(x)
    assert y.shape == (8, 32)
```

### 16.3 Test: No NaNs

```python
def test_layers_no_nans():
    ...
    assert torch.isfinite(y).all()
```

Run random inputs through:

```text
CircleLayer
HelixLayer
StandardMLP
CircleMLP
HelixMLP
```

### 16.4 Test: Backward Pass

```python
def test_helix_backward_pass():
    model = HelixMLP(hidden_dim=32, units=8)
    images = torch.randn(16, 1, 28, 28)
    targets = torch.randint(0, 10, (16,))
    logits = model(images)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
    )
```

Repeat for standard and circle models.

### 16.5 Test: Parameter Count

```python
def test_count_parameters_positive():
    model = HelixMLP(hidden_dim=32, units=8)
    assert count_parameters(model) > 0
```

### 16.6 Test: Tiny Batch Overfit

This is the most important behavioral test, but it may be slow. Include it as an optional test controlled by a CLI flag.

Default:

```bash
python mnist_test_all.py
```

runs fast tests.

Optional:

```bash
python mnist_test_all.py --slow
```

runs:

```text
test_overfit_tiny_batch_standard
test_overfit_tiny_batch_circle
test_overfit_tiny_batch_helix
```

Tiny batch test:

1. load one batch of MNIST;
2. train model on that same batch for 100-300 steps;
3. assert final batch accuracy exceeds a threshold.

Suggested threshold:

```text
standard_mlp > 90%
circle_mlp > 80%
helix_mlp > 80%
```

If this fails, do not run full MNIST.

### 16.7 Test: Quick Training Smoke

Optional but useful:

```python
def test_quick_training_smoke():
    config = MNISTConfig(
        model_type="helix_mlp",
        epochs=1,
        limit_train_batches=10,
        limit_eval_batches=5,
        device="cpu",
    )
    ...
```

Assert training returns a metrics dictionary with finite loss and accuracy.

## 17. Acceptance Criteria

The implementation is complete when:

1. `python mnist_test_all.py` passes.
2. `python mnist_test_all.py --slow` passes or produces understandable diagnostics.
3. `python mnist_run_experiment.py --quick --model-type helix_mlp` runs.
4. `python mnist_run_experiment.py --quick --all-models` runs.
5. Full MNIST training works for `standard_mlp`.
6. Full MNIST training works for `circle_mlp`.
7. Full MNIST training works for `helix_mlp`.
8. Results are saved under `mnist_results/`.
9. Checkpoints are saved under `mnist_checkpoints/`.
10. A comparison JSON is produced.
11. The results file documents parameter counts and test accuracy.

## 18. Expected Results

This experiment is exploratory. Plausible outcomes:

### 18.1 HelixMLP Matches StandardMLP

This is a good first result.

Interpretation:

```text
HelixLayer is viable as a trainable classification layer.
```

### 18.2 HelixMLP Slightly Underperforms

Still useful if it trains stably.

Interpretation:

```text
The primitive works, but this task does not reward its inductive bias.
```

### 18.3 HelixMLP Outperforms Matched MLP

Interesting, but require confirmation.

Next steps:

```text
rerun multiple seeds
check parameter counts
check optimizer settings
try Fashion-MNIST
try CIFAR-10
```

### 18.4 HelixMLP Cannot Overfit One Batch

Stop and debug.

Likely issues:

```text
initialization
too few units
radius normalization
LayerNorm placement
learning rate
weight decay
feature set too constrained
```

## 19. Result Reporting

Create or update:

```text
mnist_results.md
```

Include:

```text
dataset
training settings
model definitions
parameter counts
best validation accuracy
test accuracy
notes on training stability
```

Suggested table:

```text
| Model | Params | Best Val Acc | Test Acc | Notes |
|---|---:|---:|---:|---|
| Standard MLP | ... | ... | ... | Dense baseline |
| Param-matched MLP | ... | ... | ... | Dense fairer comparison |
| Circle MLP | ... | ... | ... | Phase/radius layer |
| Helix MLP | ... | ... | ... | Phase/radius/axis layer |
```

Use cautious language.

Good:

```text
HelixMLP trained successfully and achieved comparable MNIST accuracy to the dense baseline.
```

Avoid:

```text
Helices are generally superior neural primitives.
```

## 20. Risks and Mitigations

### Risk: HelixLayer Is Too Linear

Because `r * sin(theta)` and `r * cos(theta)` recover `b` and `a`, parts of the feature set are linear projections.

Mitigation:

Keep nonlinear normalized phase features, `r`, `tanh(z)`, and interactions. Later try richer features:

```text
sin(2theta)
cos(2theta)
r^2
z^2
sin(theta) * tanh(z)
cos(theta) * tanh(z)
```

### Risk: Radius Near Zero Causes Instability

Mitigation:

Use:

```python
eps = 1e-6
r = torch.sqrt(a*a + b*b + eps)
```

Test for NaNs and gradient stability.

### Risk: Parameter Matching Is Unfair

Mitigation:

Always report parameter counts. Include `standard_mlp_matched`.

### Risk: MNIST Is Too Easy

Mitigation:

This is acceptable for v1. Move to Fashion-MNIST and CIFAR-10 after viability is established.

### Risk: No Clear Interpretability Story

Mitigation:

Do not overclaim. This experiment tests layer viability. Later experiments can inspect learned bases, phase distributions, and interventions.

## 21. Future Extensions

After MNIST:

1. Fashion-MNIST.
2. Flattened CIFAR-10.
3. Rotated MNIST with class and angle heads.
4. Rotated CIFAR-10 with class and angle heads.
5. HelixConv2d.
6. Learned multi-frequency helix features.
7. Causal latent intervention on learned phase channels.

## 22. Implementation Order

Recommended order:

```text
1. mnist_config.py
2. mnist_models.py with StandardMLP, CircleLayer, HelixLayer
3. mnist_test_all.py layer shape/no-NaN/backward tests
4. mnist_data.py
5. mnist_train.py
6. mnist_run_experiment.py quick mode
7. tiny batch overfit tests
8. full MNIST runs
9. comparison JSON and plots
10. mnist_results.md
```

## 23. Minimal First Run

After implementation:

```bash
python mnist_test_all.py
python mnist_test_all.py --slow
python mnist_run_experiment.py --quick --model-type helix_mlp
python mnist_run_experiment.py --quick --all-models
```

Then:

```bash
python mnist_run_experiment.py --model-type standard_mlp
python mnist_run_experiment.py --model-type circle_mlp
python mnist_run_experiment.py --model-type helix_mlp
```

Finally:

```bash
python mnist_run_experiment.py --all-models
```

## 24. Core Design Summary

The `HelixLayer` is a learned geometric feature layer.

For each input, each unit computes:

```text
two phase-plane projections
one axis projection
radius in the phase plane
normalized sine/cosine phase features
axis nonlinearities
phase-axis interactions
```

These features are projected back into an ordinary hidden vector.

This makes the first version practical:

```text
ordinary tensors in
ordinary tensors out
helix-native computation inside
```

Goblin version:

```text
Let the layer sniff a 3D tunnel through the input,
measure spin, distance, and depth,
then hand the next layer a useful pile of shiny features.
```
