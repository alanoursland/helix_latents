# Experiment Design: PyTorch Helix Bottleneck Modular Addition

## Purpose

This document specifies a minimal PyTorch implementation for testing whether circular or helical latent representations can function as direct causal objects inside a neural network.

The core experiment is modular addition:

```text
(a + b) mod N = c
```

The model predicts `c` from `a` and `b`. For structured models, part of the representation is explicitly encoded as a circle or helix. After training, we intervene on that latent representation by rotating it and checking whether the output shifts predictably.

The critical test is:

```text
rotate latent(a) by k steps
model should predict (a + k + b) mod N
```

This design is intended to be detailed enough for an implementation agent to build the first version directly.

## Non-Goals

This first experiment should not attempt to build a full transformer, a language model, or an automatic manifold-discovery system.

The first version should be small, deterministic, inspectable, and easy to test.

Do not add complexity until the minimal causal intervention result is working.

## Recommended Repository Layout

```text
.
├── README.md
├── overview.md
├── experiment_description.md
├── experiment_design.md
├── pyproject.toml
├── configs
│   └── modular_addition.yaml
├── src
│   └── helix_latents
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── geometry.py
│       ├── models.py
│       ├── train.py
│       ├── evaluate.py
│       ├── intervene.py
│       ├── plotting.py
│       └── utils.py
├── scripts
│   ├── train_modular_addition.py
│   ├── evaluate_modular_addition.py
│   └── run_interventions.py
└── tests
    ├── test_data.py
    ├── test_geometry.py
    ├── test_models.py
    ├── test_training_smoke.py
    ├── test_interventions.py
    └── test_reproducibility.py
```

Package name can be changed to match the repo name.

## Dependencies

Use a small dependency set.

```text
python >= 3.10
torch
numpy
pydantic or dataclasses
pyyaml
pytest
matplotlib
tqdm
```

Optional:

```text
ruff
mypy
wandb
pandas
```

The first implementation should not require Weights & Biases or any external service.

## Configuration

Create a single config object, preferably a dataclass.

Example:

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class ExperimentConfig:
    modulus: int = 59
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15

    model_type: Literal[
        "baseline_mlp",
        "circle_bottleneck_mlp",
        "helix_bottleneck_mlp",
    ] = "helix_bottleneck_mlp"

    embedding_dim: int = 32
    hidden_dim: int = 128
    num_hidden_layers: int = 2
    dropout: float = 0.0

    helix_alpha: float | None = None

    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 500
    early_stopping_patience: int = 50

    seed: int = 0
    device: str = "cuda"  # fallback to cpu if unavailable

    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
```

If `helix_alpha` is `None`, set it to:

```python
helix_alpha = 1.0 / modulus
```

The YAML config should mirror these fields.

## Data Module

File:

```text
src/helix_latents/data.py
```

### Dataset Definition

The dataset contains all ordered pairs:

```text
a, b ∈ {0, ..., N - 1}
c = (a + b) mod N
```

Each example should be a dictionary or tuple containing:

```python
{
    "a": torch.LongTensor scalar,
    "b": torch.LongTensor scalar,
    "target": torch.LongTensor scalar,
}
```

A simple `torch.utils.data.Dataset` is enough.

Suggested class:

```python
class ModularAdditionDataset(torch.utils.data.Dataset):
    def __init__(self, pairs: torch.Tensor, targets: torch.Tensor):
        ...
```

Where:

```text
pairs.shape == [num_examples, 2]
targets.shape == [num_examples]
```

### Dataset Generation

Function:

```python
def make_modular_addition_data(modulus: int) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

Return:

```text
pairs: LongTensor [N*N, 2]
targets: LongTensor [N*N]
```

The order may be deterministic:

```python
pairs = [(a, b) for a in range(N) for b in range(N)]
```

### Splitting

Function:

```python
def split_dataset(
    pairs: torch.Tensor,
    targets: torch.Tensor,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[Dataset, Dataset, Dataset]:
    ...
```

Requirements:

1. Splits must be deterministic for a given seed.
2. Splits must be disjoint.
3. Splits must cover all examples.
4. Fractions must sum approximately to `1.0`.
5. Use a generated permutation, not contiguous slicing of the original ordered list.

### DataLoaders

Function:

```python
def make_dataloaders(config: ExperimentConfig) -> dict[str, DataLoader]:
    ...
```

Return keys:

```text
train
val
test
```

Use `shuffle=True` for train, `shuffle=False` for val/test.

## Geometry Module

File:

```text
src/helix_latents/geometry.py
```

This module owns all circle and helix math.

### Angle Conversion

Function:

```python
def number_to_theta(x: torch.Tensor, modulus: int) -> torch.Tensor:
    ...
```

Definition:

```text
theta = 2πx / modulus
```

Input `x` may be integer or float tensor.

Output should be float tensor.

### Circle Encoding

Function:

```python
def circle_encode(x: torch.Tensor, modulus: int) -> torch.Tensor:
    ...
```

Return shape:

```text
[*x.shape, 2]
```

Definition:

```text
[cos(theta), sin(theta)]
```

### Helix Encoding

Function:

```python
def helix_encode(
    x: torch.Tensor,
    modulus: int,
    alpha: float | None = None,
) -> torch.Tensor:
    ...
```

Return shape:

```text
[*x.shape, 3]
```

Definition:

```text
[cos(theta), sin(theta), alpha * x]
```

Default:

```python
alpha = 1.0 / modulus
```

### Rotation

Function:

```python
def rotate_circle(
    xy: torch.Tensor,
    k: int | torch.Tensor,
    modulus: int,
) -> torch.Tensor:
    ...
```

Input shape:

```text
[..., 2]
```

Output shape:

```text
[..., 2]
```

Definition:

```text
delta = 2πk / modulus

x' = x cos(delta) - y sin(delta)
y' = x sin(delta) + y cos(delta)
```

Function should support scalar `k` and batched `k`.

### Helix Shift

Function:

```python
def shift_helix(
    xyz: torch.Tensor,
    k: int | torch.Tensor,
    modulus: int,
    alpha: float | None = None,
    shift_axis: bool = True,
) -> torch.Tensor:
    ...
```

Input shape:

```text
[..., 3]
```

Behavior:

1. Rotate the first two coordinates by `k` modular steps.
2. If `shift_axis=True`, add `alpha * k` to the third coordinate.
3. If `shift_axis=False`, leave the third coordinate unchanged.

### Optional Projection Helpers

Useful helper functions:

```python
def infer_phase_step(xy: torch.Tensor, modulus: int) -> torch.Tensor:
    ...
```

This can convert a circular coordinate back into an approximate modular number using `atan2`.

Not necessary for training, but useful for debugging.

## Model Interfaces

File:

```text
src/helix_latents/models.py
```

All models should support ordinary forward passes and intervention-friendly forward passes.

Recommended output structure:

```python
@dataclass
class ModelOutput:
    logits: torch.Tensor
    latents: dict[str, torch.Tensor]
```

All models should implement:

```python
def forward(
    self,
    a: torch.Tensor,
    b: torch.Tensor,
    latent_override: dict[str, torch.Tensor] | None = None,
) -> ModelOutput:
    ...
```

`latent_override` allows intervention code to replace internal representations without changing input tokens.

The key convention:

```text
latents["a"] is the representation of a before combination
latents["b"] is the representation of b before combination
```

For baseline models, `latents["a"]` and `latents["b"]` may be learned embeddings.

For circle and helix models, these should be the explicit circle or helix encodings, or their learned projections depending on the architecture chosen.

## Model 1: BaselineMLP

Class:

```python
class BaselineMLP(nn.Module):
    ...
```

Inputs:

```text
a: LongTensor [batch]
b: LongTensor [batch]
```

Architecture:

```text
a_embed = Embedding(N, embedding_dim)(a)
b_embed = Embedding(N, embedding_dim)(b)
x = concat(a_embed, b_embed)
hidden = MLP(x)
logits = Linear(hidden_dim, N)
```

Latents:

```python
latents = {
    "a": a_embed,
    "b": b_embed,
}
```

For baseline intervention controls, `latent_override` may replace these embeddings.

## Model 2: CircleBottleneckMLP

Class:

```python
class CircleBottleneckMLP(nn.Module):
    ...
```

Minimal architecture:

```text
a_circle = circle_encode(a, N)  # [batch, 2]
b_circle = circle_encode(b, N)  # [batch, 2]
x = concat(a_circle, b_circle)
hidden = MLP(x)
logits = Linear(hidden_dim, N)
```

This is the most direct version.

Optional learned projection:

```text
a_repr = Linear(2, bottleneck_projection_dim)(a_circle)
b_repr = Linear(2, bottleneck_projection_dim)(b_circle)
```

For the first version, avoid projection so the intervention is directly interpretable.

Latents:

```python
latents = {
    "a": a_circle,
    "b": b_circle,
}
```

Override behavior:

```python
if latent_override and "a" in latent_override:
    a_circle = latent_override["a"]
```

## Model 3: HelixBottleneckMLP

Class:

```python
class HelixBottleneckMLP(nn.Module):
    ...
```

Minimal architecture:

```text
a_helix = helix_encode(a, N, alpha)  # [batch, 3]
b_helix = helix_encode(b, N, alpha)  # [batch, 3]
x = concat(a_helix, b_helix)
hidden = MLP(x)
logits = Linear(hidden_dim, N)
```

Latents:

```python
latents = {
    "a": a_helix,
    "b": b_helix,
}
```

Override behavior should mirror the circle model.

## MLP Builder

Function:

```python
def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_hidden_layers: int,
    dropout: float,
) -> nn.Sequential:
    ...
```

Use:

```text
Linear
ReLU or GELU
Dropout if dropout > 0
...
Linear to output_dim
```

Default activation should be `GELU`.

## Model Factory

Function:

```python
def build_model(config: ExperimentConfig) -> nn.Module:
    ...
```

Dispatch by:

```python
config.model_type
```

Validate model type and dimensions.

## Training

File:

```text
src/helix_latents/train.py
```

### Training Step

Function:

```python
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    ...
```

For each batch:

```python
a = batch["a"].to(device)
b = batch["b"].to(device)
target = batch["target"].to(device)

output = model(a, b)
loss = F.cross_entropy(output.logits, target)
```

Track:

```text
loss
accuracy
```

### Evaluation Step

Function:

```python
@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    ...
```

Track:

```text
loss
accuracy
num_examples
```

### Fit Loop

Function:

```python
def fit(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    config: ExperimentConfig,
) -> dict[str, Any]:
    ...
```

Behavior:

1. Set seed.
2. Move model to device.
3. Create AdamW optimizer.
4. Train for up to `max_epochs`.
5. Evaluate on validation after each epoch.
6. Save best checkpoint by validation accuracy, breaking ties by validation loss.
7. Use early stopping after `early_stopping_patience`.
8. Return training history.

Checkpoint contents:

```python
{
    "model_state_dict": model.state_dict(),
    "config": asdict(config),
    "history": history,
    "best_epoch": best_epoch,
    "best_val_accuracy": best_val_accuracy,
}
```

## Evaluation

File:

```text
src/helix_latents/evaluate.py
```

Should contain:

```python
def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    ...
```

and possibly:

```python
def evaluate_checkpoint(checkpoint_path: str, split: str = "test") -> dict[str, float]:
    ...
```

The script should print normal test accuracy and save a JSON result file.

## Intervention Logic

File:

```text
src/helix_latents/intervene.py
```

This is the heart of the experiment.

### Intervention Function

Function:

```python
@torch.no_grad()
def intervene_on_a(
    model: nn.Module,
    a: torch.Tensor,
    b: torch.Tensor,
    k: int,
    modulus: int,
    model_type: str,
    alpha: float | None = None,
    mode: str = "phase_plus_axis",
) -> ModelOutput:
    ...
```

Process:

1. Run model once normally to get latents.
2. Extract `latents["a"]`.
3. Modify it according to model type and mode.
4. Re-run the model with `latent_override={"a": modified_latent}`.
5. Return the intervened output.

For circle model:

```text
modified = rotate_circle(latents["a"], k, modulus)
```

For helix model:

```text
mode == "phase_plus_axis":
    modified = shift_helix(latents["a"], k, modulus, alpha, shift_axis=True)

mode == "phase_only":
    modified = shift_helix(latents["a"], k, modulus, alpha, shift_axis=False)

mode == "axis_only":
    modified = latents["a"].clone()
    modified[..., 2] += alpha * k

mode == "random":
    modified = random perturbation or random rotation control
```

For baseline model:

```text
mode == "random":
    modified = random rotated or permuted embedding
```

Do not expect semantic intervention behavior from the baseline.

### Intervention Evaluation

Function:

```python
@torch.no_grad()
def evaluate_intervention(
    model: nn.Module,
    dataloader: DataLoader,
    config: ExperimentConfig,
    shifts: list[int],
    mode: str = "phase_plus_axis",
) -> dict[str, Any]:
    ...
```

For each batch and each `k`:

```python
expected = (a + k + b) % N
output = intervene_on_a(...)
prediction = output.logits.argmax(dim=-1)
```

Track:

```text
accuracy_by_shift[k]
overall_intervention_accuracy
num_examples
```

Also track ordinary baseline accuracy in the same function or separately.

### Output Format

Return a dictionary like:

```python
{
    "model_type": config.model_type,
    "mode": mode,
    "modulus": config.modulus,
    "accuracy_by_shift": {
        "1": 0.997,
        "2": 0.996,
        ...
    },
    "overall_accuracy": 0.996,
    "num_examples": 522,
}
```

Save this as JSON.

## Plotting

File:

```text
src/helix_latents/plotting.py
```

Minimum plots:

1. Training and validation accuracy by epoch.
2. Intervention accuracy by shift.
3. Confusion matrix for one selected shift.

Use Matplotlib only.

Functions:

```python
def plot_training_history(history: dict, output_path: str) -> None:
    ...

def plot_intervention_accuracy(results: dict, output_path: str) -> None:
    ...

def plot_confusion_matrix(
    expected: np.ndarray,
    predicted: np.ndarray,
    modulus: int,
    output_path: str,
) -> None:
    ...
```

Do not require plots for CI tests.

## Scripts

### Training Script

File:

```text
scripts/train_modular_addition.py
```

CLI:

```bash
python scripts/train_modular_addition.py --config configs/modular_addition.yaml
```

Optional overrides:

```bash
python scripts/train_modular_addition.py --model-type helix_bottleneck_mlp --seed 0
```

Expected behavior:

1. Load config.
2. Set seed.
3. Build data.
4. Build model.
5. Train.
6. Save checkpoint.
7. Save training history.

### Evaluation Script

File:

```text
scripts/evaluate_modular_addition.py
```

CLI:

```bash
python scripts/evaluate_modular_addition.py --checkpoint checkpoints/best.pt --split test
```

Expected behavior:

1. Load checkpoint and config.
2. Rebuild model.
3. Load weights.
4. Rebuild data split with same seed.
5. Evaluate selected split.
6. Save JSON metrics.

### Intervention Script

File:

```text
scripts/run_interventions.py
```

CLI:

```bash
python scripts/run_interventions.py \
  --checkpoint checkpoints/best.pt \
  --shifts 1 2 3 5 10 \
  --mode phase_plus_axis
```

Expected behavior:

1. Load checkpoint and config.
2. Rebuild model and test loader.
3. Run intervention evaluation.
4. Save JSON results.
5. Save intervention accuracy plot.

## Reproducibility

File:

```text
src/helix_latents/utils.py
```

Function:

```python
def set_seed(seed: int) -> None:
    ...
```

Should set:

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

For deterministic behavior, optionally set:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

This may slow training. It is acceptable for this small experiment.

## Testing Strategy

Testing should verify the mathematical contract before training anything expensive.

Use `pytest`.

### Unit Tests: Data

File:

```text
tests/test_data.py
```

Test cases:

1. `make_modular_addition_data(5)` returns exactly 25 examples.
2. Every pair `(a, b)` appears exactly once.
3. Every target equals `(a + b) % N`.
4. Dataset splits are disjoint.
5. Dataset splits cover all examples.
6. Splits are deterministic for the same seed.
7. Splits differ for different seeds.

Example assertions:

```python
def test_modular_targets_are_correct():
    pairs, targets = make_modular_addition_data(7)
    expected = (pairs[:, 0] + pairs[:, 1]) % 7
    assert torch.equal(targets, expected)
```

### Unit Tests: Geometry

File:

```text
tests/test_geometry.py
```

Test cases:

1. `circle_encode` returns shape `[batch, 2]`.
2. Every circle vector has norm approximately `1.0`.
3. `helix_encode` returns shape `[batch, 3]`.
4. First two helix coordinates have norm approximately `1.0`.
5. Rotating by `k=0` returns the same coordinates.
6. Rotating by `k=N` returns the same coordinates.
7. Rotating `circle_encode(a)` by `k` equals `circle_encode((a + k) % N)`.
8. Shifting `helix_encode(a)` by `k` matches the circular part of `helix_encode((a + k) % N)`.
9. For helix axis, `z` increases by `alpha * k` when `shift_axis=True`.
10. For `shift_axis=False`, `z` is unchanged.

Use tolerant comparisons:

```python
torch.allclose(x, y, atol=1e-6)
```

### Unit Tests: Models

File:

```text
tests/test_models.py
```

Test cases for each model type:

1. Forward pass accepts batch tensors `a` and `b`.
2. Logits shape is `[batch, N]`.
3. Output includes `latents`.
4. `latents["a"]` and `latents["b"]` exist.
5. Circle model latent shape is `[batch, 2]`.
6. Helix model latent shape is `[batch, 3]`.
7. `latent_override={"a": ...}` changes the forward path without raising.
8. Invalid override shape raises a helpful error.

Example:

```python
def test_helix_model_shapes():
    model = HelixBottleneckMLP(modulus=11, hidden_dim=32, num_hidden_layers=1)
    a = torch.tensor([0, 1, 2])
    b = torch.tensor([3, 4, 5])
    output = model(a, b)
    assert output.logits.shape == (3, 11)
    assert output.latents["a"].shape == (3, 3)
```

### Unit Tests: Intervention

File:

```text
tests/test_interventions.py
```

These should test intervention mechanics independent of learned behavior.

Use a fake or minimal model if needed.

Test cases:

1. `intervene_on_a` returns logits with shape `[batch, N]`.
2. For circle model, intervention modifies `latents["a"]` as expected.
3. For helix model, `phase_only` leaves axis unchanged.
4. For helix model, `axis_only` leaves phase unchanged.
5. Expected target calculation is correct:

```python
expected = (a + k + b) % N
```

6. Shifts larger than `N` work:

```text
k = N + 3
```

7. Negative shifts work:

```text
k = -1
```

### Smoke Test: Training

File:

```text
tests/test_training_smoke.py
```

This should be small and fast.

Use:

```text
N = 7
max_epochs = 50
hidden_dim = 32
batch_size = 16
```

Test that:

1. Training loop runs without error.
2. Loss decreases from first epoch to last epoch, or final accuracy is above a low threshold.
3. A checkpoint dictionary can be created.
4. Evaluation returns loss and accuracy.

Avoid requiring 100% accuracy in CI; that can be flaky. Use a modest threshold such as:

```text
accuracy > 0.50 for N = 7
```

For a local non-CI test, add an optional longer test that should reach near-perfect accuracy.

Mark it:

```python
@pytest.mark.slow
```

### Integration Test: Circle Intervention Learns

Optional slow test:

```text
tests/test_intervention_learning_slow.py
```

Use:

```text
N = 11
circle_bottleneck_mlp
max_epochs = 300
```

Expected:

```text
test_accuracy > 0.95
intervention_accuracy for k=1 > 0.90
```

Mark as slow so it does not run by default:

```python
@pytest.mark.slow
```

### Reproducibility Test

File:

```text
tests/test_reproducibility.py
```

Test cases:

1. Same seed gives same dataset split.
2. Same seed gives same initial model parameters.
3. If deterministic mode is enabled, a short training run produces identical metrics on CPU.

Do not require bitwise reproducibility on GPU in the first version.

## Acceptance Criteria for First Implementation

The first implementation is complete when all of the following are true:

1. `pytest` passes for all non-slow tests.
2. A baseline MLP trains on modular addition.
3. A circle bottleneck MLP trains on modular addition.
4. A helix bottleneck MLP trains on modular addition.
5. The intervention script can rotate or shift `latents["a"]`.
6. Intervention results are saved as JSON.
7. At least one plot of intervention accuracy by shift is saved.
8. For a trained circle or helix model, intervention accuracy is substantially above chance.
9. The README or experiment docs explain how to reproduce the run.

For `N = 59`, chance accuracy is:

```text
1 / 59 ≈ 1.7%
```

A successful first result does not need to be perfect, but it should show a clear intervention effect.

## Suggested Default Config

```yaml
modulus: 59
train_frac: 0.7
val_frac: 0.15
test_frac: 0.15

model_type: helix_bottleneck_mlp

embedding_dim: 32
hidden_dim: 128
num_hidden_layers: 2
dropout: 0.0

helix_alpha: null

batch_size: 128
learning_rate: 0.001
weight_decay: 0.0
max_epochs: 500
early_stopping_patience: 50

seed: 0
device: cuda

checkpoint_dir: checkpoints
results_dir: results
```

## Implementation Notes and Pitfalls

### Do Not Hide the Bottleneck

For the first version, the structured bottleneck should be directly inspectable. Avoid immediately projecting it into a high-dimensional learned space before storing it in `latents`.

If a projection is added later, store both:

```python
latents["a_raw"]
latents["a_projected"]
```

Interventions should initially operate on `a_raw`.

### Be Careful With Axis Semantics

For modular addition, the axial coordinate in the helix is not strictly necessary. The phase is the important part. This means the helix model may not outperform the circle model on this task.

That is fine.

The helix becomes more meaningful in later tasks where both cyclic and monotonic information matter.

### Avoid Leaky Intervention Code

The intervention function should not recompute `a + k` and feed that into the model. It must modify the latent only.

Bad:

```python
model((a + k) % N, b)
```

Good:

```python
normal_output = model(a, b)
modified_latent = rotate_circle(normal_output.latents["a"], k, N)
intervened_output = model(a, b, latent_override={"a": modified_latent})
```

### Interpret Failure Carefully

If normal accuracy is high but intervention accuracy is low, possible explanations include:

1. The model solved the task using `b` plus memorized decision boundaries over the bottleneck.
2. The MLP does not use the latent in a geometrically smooth way.
3. The train/test split allowed lookup-like behavior.
4. The intervention target does not match the learned representation semantics.
5. The model needs architectural support for equivariance.

A failure is still informative. It suggests that imposing helical coordinates is not enough; the downstream computation must also respect the geometry.

## Possible Second Version

After the direct bottleneck version works, build a more ambitious model with learned helix bases:

```text
helix parameters → learned basis vectors → residual vector
```

For example:

```python
h = (
    r * cos(theta) * u
    + r * sin(theta) * v
    + z * w
)
```

where `u`, `v`, and `w` are learned vectors in a larger hidden space.

Then interventions rotate the coefficient coordinates while leaving the learned basis fixed.

This version is closer to mechanistic interpretability in real networks, where the helix lives in a subspace of a higher-dimensional residual stream.

## Summary

The first software milestone is a tiny, fully tested PyTorch experiment where:

1. circular and helical latents are explicit;
2. models can solve modular addition;
3. latent interventions are easy to perform;
4. rotating the latent representation predictably shifts model behavior.

Dragon version:

> The implementation should make the hidden spiral graspable: build it, train it, twist it, and measure whether the model obeys the twist.
