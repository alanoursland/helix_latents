# Milestone Plan: Flattened CIFAR-10 HelixLayer Experiment

## Purpose

This document breaks the flattened CIFAR-10 technical design into implementation milestones.

The goal is to extend the local classification experiments from MNIST to a harder, nonlocal pixel benchmark. All models receive flattened CIFAR-10 images. No convolutional layers, no data augmentation, and no synthetic geometric variables are used in v1.

The central comparison is:

```text
Helix MLP vs parameter-matched dense MLP on flattened CIFAR-10
```

The central measurement is not absolute CIFAR-10 performance. The central measurement is the performance gap between model families under the same fully connected setup.

## Target Workflow

Final local commands should look like:

```bash
python cifar10_test_all.py
python cifar10_test_all.py --data
python cifar10_test_all.py --slow

python cifar10_run_experiment.py --quick --model-type helix_mlp
python cifar10_run_experiment.py --quick --all-models

python cifar10_run_experiment.py --all-models --scale small
python cifar10_run_experiment.py --all-models --scale medium
python cifar10_run_experiment.py --all-models --scale large

python cifar10_run_experiment.py --all-models --sweep-scales
```

Generated artifacts should be local:

```text
cifar10_data/
cifar10_results/
cifar10_checkpoints/
```

Do not turn the repo into a package.

---

## Milestone 0: Add Local CIFAR-10 Files

### Goal

Create the flat local file structure for the CIFAR-10 experiment.

### Files to Add

```text
cifar10_config.py
cifar10_data.py
cifar10_models.py
cifar10_train.py
cifar10_run_experiment.py
cifar10_test_all.py
cifar10_results.md
```

### Deliverables

Each file can begin as a minimal placeholder with imports and docstrings.

No package scaffolding should be added.

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
python -c "import cifar10_config, cifar10_data, cifar10_models, cifar10_train"
```

should run without import errors.

### Definition of Done

The CIFAR-10 experiment has a local skeleton and does not disturb the existing modular addition or MNIST experiments.

---

## Milestone 1: CIFAR-10 Configuration

### Goal

Implement the configuration object and scale presets.

### File

```text
cifar10_config.py
```

### Deliverables

Implement:

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
```

Implement:

```python
def to_dict(self) -> dict
def get_device(requested: str = "cuda") -> torch.device
def save_json(data: dict, path: str | Path) -> None
def apply_scale_preset(config: CIFAR10Config) -> CIFAR10Config
```

Scale presets:

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

### Tests

Add to `cifar10_test_all.py`:

```text
test_config_defaults
test_config_to_dict
test_get_device_falls_back_to_cpu
test_apply_scale_preset_small
test_apply_scale_preset_medium
test_apply_scale_preset_large
```

### Acceptance Checks

```bash
python cifar10_test_all.py
```

should pass the config tests.

### Definition of Done

The experiment can construct configs, apply scale presets, select device, and save JSON results.

---

## Milestone 2: Implement StandardMLP

### Goal

Build the dense baseline first.

### File

```text
cifar10_models.py
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

Constructor:

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

Architecture for `num_layers=2`:

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

Implement model factory support for:

```text
standard_mlp
standard_mlp_matched
```

### Tests

Add:

```text
test_standard_mlp_forward_shape
test_standard_mlp_no_nans
test_standard_mlp_backward_pass
test_count_parameters_positive
test_build_standard_mlp
test_build_standard_mlp_matched
```

Expected model input:

```text
[batch, 3, 32, 32]
```

Expected logits:

```text
[batch, 10]
```

### Acceptance Checks

```bash
python cifar10_test_all.py
```

should pass.

### Definition of Done

Dense CIFAR-10 MLPs can be built, run forward, backpropagate, and report parameter counts.

---

## Milestone 3: Implement CircleLayer and CircleMLP

### Goal

Add the circle-only geometric model.

### File

```text
cifar10_models.py
```

### Deliverables

Implement:

```python
class CircleLayer(nn.Module):
    ...
```

Per-unit computations:

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

Architecture for `num_layers=2`:

```text
Flatten
CircleLayer(3072, circle_units, hidden_dim)
GELU
Dropout
CircleLayer(hidden_dim, circle_units, hidden_dim)
GELU
Dropout
Linear(hidden_dim, 10)
```

Update `build_cifar10_model` to support:

```text
circle_mlp
```

### Tests

Add:

```text
test_circle_layer_forward_shape
test_circle_layer_no_nans
test_circle_layer_backward_pass
test_circle_mlp_forward_shape
test_circle_mlp_no_nans
test_circle_mlp_backward_pass
test_build_circle_mlp
```

### Acceptance Checks

```bash
python cifar10_test_all.py
```

should pass.

### Definition of Done

CircleLayer and CircleMLP are numerically stable on random CIFAR-shaped inputs and support backpropagation.

---

## Milestone 4: Implement HelixLayer and HelixMLP

### Goal

Add the main helix-native model.

### File

```text
cifar10_models.py
```

### Deliverables

Implement:

```python
class HelixLayer(nn.Module):
    ...
```

Per-unit computations:

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

Architecture for `num_layers=2`:

```text
Flatten
HelixLayer(3072, helix_units, hidden_dim)
GELU
Dropout
HelixLayer(hidden_dim, helix_units, hidden_dim)
GELU
Dropout
Linear(hidden_dim, 10)
```

Update `build_cifar10_model` to support:

```text
helix_mlp
```

### Initialization Requirements

For `u`, `v`, and `w`:

```python
scale = 1.0 / math.sqrt(input_dim)
nn.init.normal_(param, mean=0.0, std=scale)
```

Use:

```python
eps = 1e-6
```

Do not use `atan2` in the forward pass.

### Tests

Add:

```text
test_helix_layer_forward_shape
test_helix_layer_no_nans
test_helix_layer_backward_pass
test_helix_mlp_forward_shape
test_helix_mlp_no_nans
test_helix_mlp_backward_pass
test_build_helix_mlp
```

Add random stress tests:

```text
test_helix_layer_random_stress_no_nans
test_all_models_random_stress_no_nans
```

Use input scales:

```text
0.01
1.0
10.0
```

### Acceptance Checks

```bash
python cifar10_test_all.py
```

should pass.

### Definition of Done

HelixLayer and HelixMLP run stably, avoid NaNs, and support gradients on random CIFAR-shaped tensors.

---

## Milestone 5: CIFAR-10 Data Loading

### Goal

Load CIFAR-10 locally and create deterministic train/validation/test dataloaders.

### File

```text
cifar10_data.py
```

### Deliverables

Implement transform:

```python
CIFAR10_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
])
```

Implement:

```python
def make_cifar10_dataloaders(config: CIFAR10Config) -> dict[str, DataLoader]:
    ...
```

Return:

```text
train
val
test
```

Default split:

```text
train: 45,000
val:    5,000
test:  10,000
```

Use a seeded `torch.Generator` for deterministic validation split.

### Tests

Data tests should only run when requested:

```bash
python cifar10_test_all.py --data
```

Add:

```text
test_cifar10_batch_shape
test_cifar10_target_shape
test_cifar10_num_classes
test_cifar10_train_val_test_sizes
test_cifar10_split_deterministic
```

Expected batch shape:

```text
[batch, 3, 32, 32]
```

Expected target shape:

```text
[batch]
```

### Acceptance Checks

```bash
python cifar10_test_all.py --data
```

should pass.

### Definition of Done

CIFAR-10 downloads, loads, splits deterministically, and produces correctly shaped batches.

---

## Milestone 6: Training and Evaluation Loops

### Goal

Implement CIFAR-10 training code.

### File

```text
cifar10_train.py
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
def fit_cifar10(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    config: CIFAR10Config,
) -> dict[str, Any]:
    ...
```

Metrics:

```text
loss
accuracy
num_examples
elapsed_seconds
examples_per_second
```

Use optimizer:

```python
torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
)
```

Optional scheduler:

```python
torch.optim.lr_scheduler.CosineAnnealingLR
```

if `config.use_scheduler` is true.

### Checkpoints

Save to:

```text
cifar10_checkpoints/<model_type>_<scale>_seed<seed>_best.pt
```

Checkpoint should include:

```text
model_state_dict
config
model_type
scale
seed
param_count
history
best_epoch
best_val_accuracy
best_val_loss
test_metrics
```

### Metrics

Save to:

```text
cifar10_results/<model_type>_<scale>_seed<seed>/metrics.json
cifar10_results/<model_type>_<scale>_seed<seed>/history.json
```

### Tests

Use synthetic data for default tests.

Add:

```text
test_train_one_epoch_synthetic
test_evaluate_synthetic
test_fit_cifar10_synthetic_quick
```

These should not download CIFAR-10.

### Acceptance Checks

```bash
python cifar10_test_all.py
```

should pass the synthetic training tests.

### Definition of Done

Training logic works on synthetic data and can save local checkpoint and metrics artifacts.

---

## Milestone 7: Experiment Runner CLI

### Goal

Create the main local command-line runner.

### File

```text
cifar10_run_experiment.py
```

### Deliverables

Implement CLI arguments:

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

Allowed model types:

```text
standard_mlp
standard_mlp_matched
circle_mlp
helix_mlp
```

Quick mode should set:

```text
epochs = 1
scale = small
limit_train_batches = 100
limit_eval_batches = 50
```

Implement:

```python
def run_single(config: CIFAR10Config) -> dict[str, Any]:
    ...
```

Behavior:

1. apply scale preset;
2. set seed;
3. build dataloaders;
4. build model;
5. count parameters;
6. print run summary;
7. train;
8. save artifacts;
9. return compact result.

### Comparison JSON

For `--all-models`, save:

```text
cifar10_results/comparison_<scale>_seed<seed>.json
```

For `--sweep-scales`, save:

```text
cifar10_results/comparison_all_scales_seed<seed>.json
```

### Acceptance Checks

Manual:

```bash
python cifar10_run_experiment.py --quick --model-type standard_mlp --device cpu
python cifar10_run_experiment.py --quick --model-type helix_mlp --device cpu
python cifar10_run_experiment.py --quick --all-models --device cpu
```

### Definition of Done

CIFAR-10 experiments can be launched from CLI without writing Python code.

---

## Milestone 8: Quick Mode Validation

### Goal

Verify the full pipeline runs end to end before full training.

### Commands

```bash
python cifar10_run_experiment.py --quick --model-type standard_mlp
python cifar10_run_experiment.py --quick --model-type circle_mlp
python cifar10_run_experiment.py --quick --model-type helix_mlp
python cifar10_run_experiment.py --quick --all-models
```

### Expected Behavior

Quick mode should:

```text
download or load CIFAR-10
train for 1 epoch
limit train/eval batches
save metrics
save history
save checkpoint
produce finite accuracy/loss
```

Accuracy does not need to be high.

Minimum expectation:

```text
no NaNs
loss finite
accuracy > 10% or trending upward
```

Chance accuracy is:

```text
10%
```

### Definition of Done

All model families complete a quick CIFAR-10 run.

---

## Milestone 9: Tiny-Batch Overfit Tests

### Goal

Check whether each model can fit a small fixed CIFAR-10 batch.

This is the key diagnostic before interpreting full training failures.

### File

```text
cifar10_test_all.py
```

### Deliverables

Add slow tests enabled by:

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

1. Load one CIFAR-10 batch of 128 examples.
2. Train on that same batch for 200-500 steps.
3. Evaluate on the same batch.

Suggested thresholds:

```text
standard_mlp >= 80%
circle_mlp >= 70%
helix_mlp >= 70%
```

If CIFAR-10 download is not desired in slow tests, use a fixed small synthetic dataset first, then run the real tiny-batch test manually.

### Acceptance Checks

```bash
python cifar10_test_all.py --slow
```

should pass or produce clear diagnostics.

### Definition of Done

All models can overfit a small CIFAR-10 batch, or failures are documented before full training.

---

## Milestone 10: Small-Scale Full Run

### Goal

Run the first reportable CIFAR-10 comparison at small scale.

### Command

```bash
python cifar10_run_experiment.py --all-models --scale small
```

### Deliverables

Save one run directory per model:

```text
cifar10_results/standard_mlp_small_seed0/
cifar10_results/standard_mlp_matched_small_seed0/
cifar10_results/circle_mlp_small_seed0/
cifar10_results/helix_mlp_small_seed0/
```

Each should contain:

```text
history.json
metrics.json
training_history.png
```

Save:

```text
cifar10_results/comparison_small_seed0.json
```

### Acceptance Checks

Each model should:

```text
train without NaNs
save metrics
save checkpoint
achieve accuracy meaningfully above chance
```

Minimum initial bar:

```text
test accuracy > 30%
```

This is intentionally low for the first full small-scale run.

### Definition of Done

A complete small-scale comparison exists.

---

## Milestone 11: Medium-Scale Full Run

### Goal

Run the main default comparison.

### Command

```bash
python cifar10_run_experiment.py --all-models --scale medium
```

### Deliverables

Save:

```text
cifar10_results/comparison_medium_seed0.json
```

and per-model artifacts.

### Acceptance Checks

Expected rough dense baseline range:

```text
55%–60% test accuracy
```

This depends on training budget and regularization. If dense MLP is far below this, debug before interpreting helix results.

Key comparison:

```text
helix_mlp test accuracy vs standard_mlp_matched test accuracy
```

### Definition of Done

The main reportable CIFAR-10 comparison exists.

---

## Milestone 12: Large-Scale Full Run

### Goal

Check whether behavior changes with more capacity.

### Command

```bash
python cifar10_run_experiment.py --all-models --scale large
```

### Deliverables

Save:

```text
cifar10_results/comparison_large_seed0.json
```

and per-model artifacts.

### Acceptance Checks

All models should complete training within available compute.

If large-scale training is too slow, document the limitation and prioritize medium-scale multi-seed runs instead.

### Definition of Done

A large-scale comparison exists or is explicitly deferred for compute reasons.

---

## Milestone 13: Scale Sweep

### Goal

Compare accuracy-vs-parameter curves.

### Command

```bash
python cifar10_run_experiment.py --all-models --sweep-scales
```

### Deliverables

Save:

```text
cifar10_results/comparison_all_scales_seed0.json
cifar10_results/accuracy_vs_params.png
```

The comparison should include:

```text
model_type
scale
param_count
best_val_accuracy
test_accuracy
test_loss
mean_epoch_seconds
```

### Acceptance Checks

Plot shows all model/scale points.

### Definition of Done

The project can report performance as a function of parameter count, not just a single point.

---

## Milestone 14: Parameter-Matching Pass

### Goal

Tighten the dense baseline comparison.

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
    config: CIFAR10Config,
) -> int:
    ...
```

Parameter matching target:

```text
standard_mlp_matched should be within 10% of helix_mlp parameters
```

If this is not achieved, document the mismatch.

### Acceptance Checks

Comparison JSON includes parameter counts and the writeup comments on any mismatch.

### Definition of Done

The main dense baseline is credibly parameter-matched to Helix MLP.

---

## Milestone 15: Plotting

### Goal

Make training and comparison results inspectable.

### Files

Reuse or extend:

```text
plotting.py
```

or implement local helpers in:

```text
cifar10_train.py
```

### Deliverables

Training history plot per run:

```text
cifar10_results/<run_name>/training_history.png
```

Accuracy-vs-parameter plot:

```text
cifar10_results/accuracy_vs_params.png
```

Optional:

```text
cifar10_results/accuracy_vs_time.png
```

### Acceptance Checks

Plots are produced and open successfully.

### Definition of Done

Each full run produces visual training history and comparison plots.

---

## Milestone 16: First Results Writeup

### Goal

Document the first CIFAR-10 results cautiously.

### File

```text
cifar10_results.md
```

### Deliverables

Include:

```text
overview
dataset and preprocessing
model variants
training settings
parameter counts
summary table
per-model results
accuracy-vs-parameters
main takeaways
scope of claim
next steps
source artifacts
```

Use cautious language.

Good:

```text
Helix MLP remained within 2.1 points of the parameter-matched dense baseline.
```

Avoid:

```text
HelixLayer is better than dense layers.
```

### Acceptance Checks

The writeup numbers match JSON artifacts.

### Definition of Done

A repo-ready results document exists.

---

## Milestone 17: Multi-Seed Runs

### Goal

Check whether the observed gaps are stable.

### Commands

Run at least three seeds:

```bash
python cifar10_run_experiment.py --all-models --scale medium --seed 0
python cifar10_run_experiment.py --all-models --scale medium --seed 1
python cifar10_run_experiment.py --all-models --scale medium --seed 2
```

For stronger claims:

```text
seed ∈ {0, 1, 2, 3, 4}
```

### Deliverables

Save:

```text
cifar10_results/multiseed_medium_comparison.json
```

Report:

```text
mean test accuracy
std test accuracy
mean best val accuracy
std best val accuracy
mean epoch time
```

### Acceptance Checks

The writeup reports means and standard deviations before claiming one architecture is better.

### Definition of Done

At least medium-scale CIFAR-10 results are replicated across seeds.

---

## Milestone 18: Feature Ablations

### Goal

Determine what parts of HelixLayer contribute to performance if it is competitive.

### Deliverables

Add configurable HelixLayer feature modes:

```text
full
phase_only
phase_radius
raw_projection
axis_only
no_axis
```

Feature groups:

```text
phase_only:
  sin(theta), cos(theta)

phase_radius:
  sin(theta), cos(theta), r

raw_projection:
  r*sin(theta), r*cos(theta), z

axis_only:
  z, tanh(z)

no_axis:
  sin(theta), cos(theta), r, r*sin(theta), r*cos(theta)

full:
  sin(theta), cos(theta), r, z,
  r*sin(theta), r*cos(theta), tanh(z), r*tanh(z)
```

Run at least the medium scale:

```bash
python cifar10_run_experiment.py --model-type helix_mlp --scale medium --helix-feature-mode full
python cifar10_run_experiment.py --model-type helix_mlp --scale medium --helix-feature-mode no_axis
python cifar10_run_experiment.py --model-type helix_mlp --scale medium --helix-feature-mode raw_projection
```

### Acceptance Checks

Results show whether normalized phase/axis features are doing work beyond raw projections.

### Definition of Done

Feature ablation results are available for the most competitive Helix MLP setting.

---

## Milestone 19: Decision Gate for Tabular Experiment

### Goal

Use the CIFAR-10 result to decide the tabular experiment setup.

### Decision Rules

If Helix MLP matches or nearly matches dense MLP:

```text
Proceed to tabular classification with confidence that the primitive is viable beyond MNIST.
```

If Helix MLP trails badly but can overfit tiny batches:

```text
Proceed to tabular classification anyway, but frame CIFAR-10 as a limitation on nonlocal pixel data.
```

If Helix MLP cannot overfit CIFAR-10 tiny batches:

```text
Debug architecture before tabular, because optimization may be the limiting factor.
```

If Circle MLP beats Helix MLP:

```text
Do not make axis-utility claims. Consider no-axis ablation and tabular test.
```

### Definition of Done

The repo documents whether CIFAR-10 supports moving to tabular classification and why.

---

## Milestone 20: Final CIFAR-10 Experiment Package

### Goal

Complete the v1 flattened CIFAR-10 experiment.

### Final Deliverables

```text
cifar10_config.py
cifar10_data.py
cifar10_models.py
cifar10_train.py
cifar10_run_experiment.py
cifar10_test_all.py
cifar10_results.md

cifar10_results/
  comparison_small_seed0.json
  comparison_medium_seed0.json
  comparison_large_seed0.json
  comparison_all_scales_seed0.json
  accuracy_vs_params.png
  <per-run directories>/

cifar10_checkpoints/
  <model checkpoints>
```

### Final Acceptance Criteria

The experiment is complete when:

1. Fast tests pass.
2. Data tests pass.
3. Tiny-batch overfit tests pass or failures are documented.
4. Quick mode works.
5. Small-scale comparison exists.
6. Medium-scale comparison exists.
7. Parameter counts are reported.
8. Training time is reported.
9. Accuracy-vs-parameter plot exists.
10. Results writeup exists.
11. Interpretation is cautious and gap-focused.

---

## Recommended Implementation Order

Shortest reliable path:

```text
0. Add local files
1. Config and scale presets
2. StandardMLP
3. CircleLayer and CircleMLP
4. HelixLayer and HelixMLP
5. CIFAR-10 data loading
6. Training loops
7. Runner CLI
8. Quick mode validation
9. Tiny-batch overfit tests
10. Small-scale full run
11. Medium-scale full run
12. Large-scale full run
13. Scale sweep
14. Parameter matching pass
15. Plotting
16. First results writeup
17. Multi-seed runs
18. Feature ablations
19. Tabular decision gate
20. Final experiment package
```

## Critical Stop Conditions

Stop and debug if any of these happen:

```text
HelixLayer produces NaNs
HelixMLP gradients become NaN
HelixMLP cannot overfit a tiny batch
CircleMLP trains but HelixMLP fails completely
Dense MLP baseline is far below expected range
Quick mode fails to save metrics/checkpoints
```

Likely fixes:

```text
lower learning rate
remove or reduce weight decay
reduce dropout
disable or move LayerNorm
increase eps
increase helix_units
change initialization scale
add gradient clipping
run with scheduler
run longer
```

## Expected Research Outcomes

### Outcome A: Helix MLP is close to dense MLP

This is a good result.

Interpretation:

```text
HelixLayer generalizes as a viable primitive to a hard nonlocal pixel task.
```

### Outcome B: Helix MLP beats dense MLP

This is strong but needs confirmation.

Required follow-up:

```text
multi-seed
parameter sweep
compute comparison
feature ablation
```

### Outcome C: Helix MLP trails slightly

Still useful.

Interpretation:

```text
HelixLayer is viable but not advantaged on flattened CIFAR-10.
```

### Outcome D: Helix MLP trails badly

Also useful.

Interpretation:

```text
HelixLayer may struggle with high-dimensional nonlocal pixel data.
```

This would motivate tabular classification as a different kind of generality test.

## Goblin Summary

First make sure the spiral beast can stand up.

Then make it memorize one tiny pile of CIFAR goblins.

Then make it fight dense goblins at small, medium, and large sizes.

Do not ask whether it conquered vision.

Ask how far behind or ahead it is on the same bad terrain.
