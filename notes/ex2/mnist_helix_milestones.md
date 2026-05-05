# Milestone Plan: MNIST HelixLayer Experiment

## Purpose

This document breaks the MNIST HelixLayer technical design into implementation milestones.

The goal is to move from the current flat local repo to a working MNIST experiment that compares:

```text
standard_mlp
standard_mlp_matched
circle_mlp
helix_mlp
```

The implementation should remain local and script-based. Do not create an installable package.

## Target Local Workflow

The final workflow should be:

```bash
python mnist_test_all.py
python mnist_test_all.py --slow

python mnist_run_experiment.py --quick --model-type helix_mlp
python mnist_run_experiment.py --quick --all-models

python mnist_run_experiment.py --model-type standard_mlp
python mnist_run_experiment.py --model-type circle_mlp
python mnist_run_experiment.py --model-type helix_mlp
python mnist_run_experiment.py --all-models
```

Generated artifacts should be local:

```text
data/
mnist_results/
mnist_checkpoints/
```

## Milestone 0: Add Local MNIST Files

### Goal

Create the flat-file structure for the MNIST experiment.

### Files to Add

```text
mnist_config.py
mnist_data.py
mnist_models.py
mnist_train.py
mnist_run_experiment.py
mnist_test_all.py
mnist_results.md
```

### Deliverables

Each file can start as a minimal placeholder with imports and docstrings.

No package structure should be added.

Do not add:

```text
src/helix_latents/
setup.py
pyproject.toml
editable install workflow
```

### Acceptance Checks

From the local source directory:

```bash
python -c "import mnist_config, mnist_data, mnist_models, mnist_train"
```

should run without import errors.

### Definition of Done

The repo has a local MNIST experiment skeleton without changing the modular addition experiment.

## Milestone 1: MNIST Configuration

### Goal

Implement a local configuration object for the MNIST experiment.

### File

```text
mnist_config.py
```

### Deliverables

Implement:

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
```

Also implement:

```python
def to_dict(self) -> dict
def get_device(requested: str = "cuda") -> torch.device
def save_json(data: dict, path: str | Path) -> None
```

### Tests

Add tests to `mnist_test_all.py`:

```text
test_config_defaults
test_config_to_dict
test_get_device_falls_back_to_cpu
```

### Acceptance Checks

```bash
python mnist_test_all.py
```

should pass the config tests.

### Definition of Done

The MNIST experiment can construct a config and save JSON files locally.

## Milestone 2: Implement StandardMLP

### Goal

Build the simplest dense MNIST classifier first.

This gives a known-good model and tests the local MNIST training path before introducing geometric layers.

### File

```text
mnist_models.py
```

### Deliverables

Implement:

```python
def count_parameters(model: nn.Module) -> int
```

Implement:

```python
class StandardMLP(nn.Module):
    ...
```

Architecture:

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

For `num_layers > 2`, add more hidden `Linear -> GELU -> Dropout` blocks.

Implement partial factory:

```python
def build_mnist_model(config: MNISTConfig) -> nn.Module:
    ...
```

Support initially:

```text
standard_mlp
standard_mlp_matched
```

### Tests

Add tests:

```text
test_standard_mlp_forward_shape
test_standard_mlp_no_nans
test_standard_mlp_backward_pass
test_count_parameters_positive
test_build_standard_models
```

### Acceptance Checks

```bash
python mnist_test_all.py
```

should pass.

Manual sanity:

```bash
python - <<'PY'
from mnist_config import MNISTConfig
from mnist_models import build_mnist_model, count_parameters
config = MNISTConfig(model_type="standard_mlp")
model = build_mnist_model(config)
print(count_parameters(model))
PY
```

### Definition of Done

A standard MLP can be built, run forward, backpropagate, and report parameter count.

## Milestone 3: Implement CircleLayer and CircleMLP

### Goal

Add the circle-only geometric control model.

This tests phase/radius features without the helix axis.

### File

```text
mnist_models.py
```

### Deliverables

Implement:

```python
class CircleLayer(nn.Module):
    ...
```

Layer input/output shape:

```text
[batch, input_dim] -> [batch, output_dim]
```

Per-unit projections:

```text
a = x @ u + bias_u
b = x @ v + bias_v
r = sqrt(a*a + b*b + eps)
sin_theta = b / r
cos_theta = a / r
```

Feature set:

```text
sin_theta
cos_theta
r
r * sin_theta
r * cos_theta
```

Implement:

```python
class CircleMLP(nn.Module):
    ...
```

Architecture:

```text
Flatten
CircleLayer(784, circle_units, hidden_dim)
GELU
CircleLayer(hidden_dim, circle_units, hidden_dim)
GELU
Linear(hidden_dim, 10)
```

Update factory to support:

```text
circle_mlp
```

### Tests

Add tests:

```text
test_circle_layer_forward_shape
test_circle_layer_no_nans
test_circle_layer_backward_pass
test_circle_mlp_forward_shape
test_circle_mlp_no_nans
test_circle_mlp_backward_pass
test_build_circle_model
```

### Acceptance Checks

```bash
python mnist_test_all.py
```

should pass.

### Definition of Done

CircleLayer and CircleMLP are stable on random inputs and support backpropagation.

## Milestone 4: Implement HelixLayer and HelixMLP

### Goal

Add the main helix-native layer.

### File

```text
mnist_models.py
```

### Deliverables

Implement:

```python
class HelixLayer(nn.Module):
    ...
```

Layer input/output shape:

```text
[batch, input_dim] -> [batch, output_dim]
```

Per-unit projections:

```text
a = x @ u + bias_u
b = x @ v + bias_v
z = x @ w + bias_w
r = sqrt(a*a + b*b + eps)
sin_theta = b / r
cos_theta = a / r
```

Feature set:

```text
sin_theta
cos_theta
r
z
r * sin_theta
r * cos_theta
tanh(z)
r * tanh(z)
```

Implement:

```python
class HelixMLP(nn.Module):
    ...
```

Architecture:

```text
Flatten
HelixLayer(784, helix_units, hidden_dim)
GELU
HelixLayer(hidden_dim, helix_units, hidden_dim)
GELU
Linear(hidden_dim, 10)
```

Update factory to support:

```text
helix_mlp
```

### Initialization Requirements

Initialize `u`, `v`, and `w` with:

```python
scale = 1.0 / math.sqrt(input_dim)
nn.init.normal_(param, mean=0.0, std=scale)
```

Use:

```python
eps = 1e-6
```

### Tests

Add tests:

```text
test_helix_layer_forward_shape
test_helix_layer_no_nans
test_helix_layer_backward_pass
test_helix_mlp_forward_shape
test_helix_mlp_no_nans
test_helix_mlp_backward_pass
test_build_helix_model
```

Also add a stronger random stress test:

```text
test_helix_layer_random_stress_no_nans
```

It should run several random input scales:

```text
scale = 0.01, 1.0, 10.0
```

### Acceptance Checks

```bash
python mnist_test_all.py
```

should pass.

### Definition of Done

HelixLayer and HelixMLP are numerically stable on random inputs and support backpropagation.

## Milestone 5: MNIST Data Loading

### Goal

Load MNIST locally with a deterministic train/validation split.

### File

```text
mnist_data.py
```

### Deliverables

Implement transform:

```python
MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
```

Implement:

```python
def make_mnist_dataloaders(config: MNISTConfig) -> dict[str, DataLoader]:
    ...
```

Return:

```text
train
val
test
```

Behavior:

```text
download MNIST if needed
train/val split with seeded generator
shuffle train only
```

### Tests

Add tests:

```text
test_mnist_dataloader_keys
test_mnist_batch_shapes
test_mnist_target_shapes
test_mnist_split_sizes
test_mnist_split_deterministic
```

Because data download can be slow, allow these tests to be skipped by default unless a flag is passed.

Recommended:

```bash
python mnist_test_all.py --data
```

### Acceptance Checks

```bash
python mnist_test_all.py --data
```

should pass.

Manual sanity:

```bash
python - <<'PY'
from mnist_config import MNISTConfig
from mnist_data import make_mnist_dataloaders

loaders = make_mnist_dataloaders(MNISTConfig(batch_size=8))
batch = next(iter(loaders["train"]))
print(batch[0].shape, batch[1].shape)
PY
```

Expected:

```text
torch.Size([8, 1, 28, 28]) torch.Size([8])
```

### Definition of Done

MNIST data loads locally and produces expected batch shapes.

## Milestone 6: Training and Evaluation Loops

### Goal

Implement reusable training code for MNIST models.

### File

```text
mnist_train.py
```

### Deliverables

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
def fit_mnist(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    config: MNISTConfig,
) -> dict[str, Any]:
    ...
```

Metrics:

```text
loss
accuracy
num_examples
```

Fit should:

```text
train for config.epochs
evaluate validation each epoch
track history
save best checkpoint by validation accuracy
restore best state
evaluate test set
save metrics JSON
save history JSON
```

### Tests

Add fast synthetic-loader tests that do not require downloading MNIST:

```text
test_train_one_epoch_synthetic
test_evaluate_synthetic
test_fit_mnist_synthetic_quick
```

Use a tiny fake dataset:

```text
images: random [N, 1, 28, 28]
labels: random [N]
```

### Acceptance Checks

```bash
python mnist_test_all.py
```

should pass training-loop tests without MNIST download.

### Definition of Done

Training logic works on synthetic data and can save local metrics/checkpoints.

## Milestone 7: Experiment Runner CLI

### Goal

Create the main local command-line entry point.

### File

```text
mnist_run_experiment.py
```

### Deliverables

Implement CLI args:

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

Supported model types:

```text
standard_mlp
standard_mlp_matched
circle_mlp
helix_mlp
```

Quick mode should set:

```text
epochs = 1
limit_train_batches = 100
limit_eval_batches = 50
```

Script behavior:

```text
build config
apply CLI overrides
set seed
build dataloaders
build model
print parameter count
train
save metrics
save checkpoint
print result summary
```

`--all-models` should run all four models and save:

```text
mnist_results/comparison.json
```

### Tests

Add a subprocess smoke test if practical:

```text
test_runner_quick_standard_subprocess
```

If subprocess tests are too slow, keep this as a manual acceptance check.

### Acceptance Checks

Manual:

```bash
python mnist_run_experiment.py --quick --model-type standard_mlp --device cpu
python mnist_run_experiment.py --quick --model-type helix_mlp --device cpu
python mnist_run_experiment.py --quick --all-models --device cpu
```

### Definition of Done

The MNIST experiment can be run from CLI without writing Python code.

## Milestone 8: Tiny Batch Overfit Tests

### Goal

Verify that each model can fit a tiny dataset.

This is the most important diagnostic before running full MNIST.

### File

```text
mnist_test_all.py
```

### Deliverables

Add slow tests enabled by:

```bash
python mnist_test_all.py --slow
```

Tests:

```text
test_overfit_tiny_batch_standard
test_overfit_tiny_batch_circle
test_overfit_tiny_batch_helix
```

Procedure:

1. Load one MNIST batch or create a small fixed subset.
2. Train the model on the same batch for 100-300 steps.
3. Measure accuracy on that same batch.

Suggested thresholds:

```text
standard_mlp >= 90%
circle_mlp >= 80%
helix_mlp >= 80%
```

If MNIST download in tests is undesirable, use synthetic structured labels first, but eventually run the real MNIST tiny-batch test manually.

### Acceptance Checks

```bash
python mnist_test_all.py --slow
```

should pass before running full experiments.

### Definition of Done

All three model families can overfit a tiny batch.

## Milestone 9: Full MNIST Single-Model Runs

### Goal

Run full MNIST training for each model independently.

### Commands

```bash
python mnist_run_experiment.py --model-type standard_mlp
python mnist_run_experiment.py --model-type standard_mlp_matched
python mnist_run_experiment.py --model-type circle_mlp
python mnist_run_experiment.py --model-type helix_mlp
```

### Deliverables

For each model, save:

```text
mnist_results/<model_type>/history.json
mnist_results/<model_type>/metrics.json
mnist_results/<model_type>/training_history.png
mnist_checkpoints/<model_type>_best.pt
```

### Acceptance Checks

Each model should:

```text
train without NaNs
save metrics
save checkpoint
achieve test accuracy above chance
```

Minimum initial bar:

```text
test accuracy > 80%
```

This is intentionally low for the first full run. MNIST is easy; final models should likely do better.

### Definition of Done

All four model variants complete a full MNIST run.

## Milestone 10: All-Models Comparison Run

### Goal

Produce one comparison artifact across model variants.

### Command

```bash
python mnist_run_experiment.py --all-models
```

### Deliverables

Save:

```text
mnist_results/comparison.json
```

Optional:

```text
mnist_results/comparison.png
```

The comparison JSON should include:

```text
model_type
param_count
best_epoch
best_val_accuracy
test_accuracy
test_loss
```

### Acceptance Checks

The script prints a table:

```text
Model                  Params      Best Val    Test Acc
-------------------------------------------------------
standard_mlp           ...
standard_mlp_matched   ...
circle_mlp             ...
helix_mlp              ...
```

### Definition of Done

There is a single local artifact comparing all models.

## Milestone 11: Parameter Matching Pass

### Goal

Make the dense MLP comparison fairer.

### Deliverables

Either manually tune:

```text
matched_hidden_dim
```

or implement:

```python
def find_param_matched_hidden_dim(
    target_params: int,
    candidate_dims: list[int],
    config: MNISTConfig,
) -> int:
    ...
```

The parameter-matched MLP should be close to HelixMLP parameter count.

Suggested tolerance:

```text
within 10%
```

### Acceptance Checks

Comparison output should include parameter counts showing:

```text
abs(params_standard_mlp_matched - params_helix_mlp) / params_helix_mlp <= 0.10
```

If this is not achieved, document the mismatch.

### Definition of Done

The main comparison includes a credible parameter-matched dense baseline.

## Milestone 12: Basic Plotting

### Goal

Make results inspectable.

### Files

Either reuse:

```text
plotting.py
```

or create simple plotting in:

```text
mnist_train.py
```

### Deliverables

Generate:

```text
mnist_results/<model_type>/training_history.png
```

Optional comparison plot:

```text
mnist_results/comparison.png
```

Plot training history:

```text
train_loss
val_loss
train_accuracy
val_accuracy
```

### Acceptance Checks

Plots are created after a run and open without error.

### Definition of Done

Each model run produces a visual training history.

## Milestone 13: Results Writeup

### Goal

Document the first MNIST findings carefully.

### File

```text
mnist_results.md
```

### Deliverables

Include:

```text
experiment purpose
dataset
model definitions
training settings
parameter counts
test accuracy
known limitations
next steps
```

Suggested table:

```text
| Model | Params | Best Val Acc | Test Acc | Notes |
|---|---:|---:|---:|---|
| Standard MLP | ... | ... | ... | Dense baseline |
| Param-matched MLP | ... | ... | ... | Fairer dense comparison |
| Circle MLP | ... | ... | ... | Phase/radius layer |
| Helix MLP | ... | ... | ... | Phase/radius/axis layer |
```

Use cautious language.

Good:

```text
HelixMLP trained successfully and reached comparable MNIST accuracy.
```

Avoid:

```text
Helices are better than standard neural layers.
```

### Acceptance Checks

The writeup is consistent with `comparison.json`.

### Definition of Done

The repo has a clear local summary of MNIST results.

## Milestone 14: Multi-Seed Confirmation

### Goal

Check whether results are stable.

### Commands

Run at least three seeds:

```bash
python mnist_run_experiment.py --all-models --seed 0
python mnist_run_experiment.py --all-models --seed 1
python mnist_run_experiment.py --all-models --seed 2
```

To avoid overwriting, include seed in result directories or filenames:

```text
mnist_results/seed_0/
mnist_results/seed_1/
mnist_results/seed_2/
```

This can be a later improvement.

### Deliverables

Save:

```text
mnist_results/multiseed_comparison.json
```

With:

```text
mean test accuracy
std test accuracy
mean best val accuracy
std best val accuracy
```

### Acceptance Checks

The writeup reports mean and standard deviation.

### Definition of Done

Any claims about one model beating another are supported by multiple seeds.

## Milestone 15: Decide Whether to Continue

### Goal

Use MNIST results to decide the next experiment.

### Decision Tree

If `helix_mlp` cannot overfit a tiny batch:

```text
debug HelixLayer before any new dataset
```

If `helix_mlp` trains but badly underperforms:

```text
try initialization/feature-set changes
try Fashion-MNIST only after basic fixes
```

If `helix_mlp` matches dense MLP:

```text
proceed to Fashion-MNIST or rotated MNIST
```

If `helix_mlp` beats matched dense MLP:

```text
rerun multiple seeds
try Fashion-MNIST
try flattened CIFAR-10
```

If `circle_mlp` matches or beats `helix_mlp`:

```text
do not claim axis utility on MNIST
move to a task where axis should matter
```

### Definition of Done

There is a written next-step decision based on the data.

## Suggested Implementation Order

The shortest reliable path is:

```text
0. local MNIST files
1. config
2. StandardMLP
3. CircleLayer/CircleMLP
4. HelixLayer/HelixMLP
5. data loading
6. training loops
7. runner CLI
8. tiny-batch overfit tests
9. full single-model runs
10. all-models comparison
11. parameter matching
12. plots
13. results writeup
14. multi-seed runs
15. next-step decision
```

## Critical Early Stop Conditions

Stop and debug if any of these happen:

```text
HelixLayer forward pass produces NaNs
HelixLayer backward pass produces NaN gradients
HelixMLP cannot overfit one tiny batch
HelixMLP quick run stays at chance accuracy
CircleMLP trains but HelixMLP does not
```

Likely fixes:

```text
lower learning rate
remove weight decay
disable LayerNorm or move it
increase eps
increase helix_units
add richer nonlinear features
change initialization scale
gradient clipping
```

## First-Day Target

A good first implementation target is:

```bash
python mnist_test_all.py
python mnist_run_experiment.py --quick --model-type helix_mlp --device cpu
```

If that works, run:

```bash
python mnist_test_all.py --slow
```

Then proceed to full MNIST.

## Final Acceptance Criteria

The MNIST experiment v1 is complete when:

1. The local files exist and import cleanly.
2. Fast tests pass.
3. Slow tiny-batch overfit tests pass.
4. Quick mode works for all models.
5. Full MNIST runs complete for all models.
6. Parameter counts are reported.
7. Results are saved locally.
8. Checkpoints are saved locally.
9. A comparison JSON exists.
10. A cautious `mnist_results.md` writeup exists.
11. The next experiment decision is documented.

## Summary

Build the layer.

Make sure it does not explode.

Make it memorize a small batch.

Make it learn all the digits.

Only then ask whether the geometric layer is actually better than a plain dense MLP.
