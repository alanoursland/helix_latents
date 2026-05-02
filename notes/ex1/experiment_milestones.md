# Experiment Milestones: Helix Bottleneck Modular Addition

## Purpose

This document breaks the PyTorch experiment design into implementation milestones.

The goal is to move from an empty repository to a working, tested experiment where circular and helical latent representations can be trained and causally intervened on.

The milestones are ordered to make failure points small and visible.

## Milestone 0: Repository Skeleton

### Goal

Create the project structure and basic tooling.

### Deliverables

```text
pyproject.toml
README.md
configs/modular_addition.yaml
src/helix_latents/__init__.py
tests/
scripts/
```

Recommended layout:

```text
src/helix_latents/
    config.py
    data.py
    geometry.py
    models.py
    train.py
    evaluate.py
    intervene.py
    plotting.py
    utils.py

scripts/
    train_modular_addition.py
    evaluate_modular_addition.py
    run_interventions.py

tests/
    test_data.py
    test_geometry.py
    test_models.py
    test_training_smoke.py
    test_interventions.py
    test_reproducibility.py
```

### Acceptance Checks

```bash
pytest
```

should run, even if there are only placeholder tests.

The package should be importable:

```python
import helix_latents
```

### Notes

Keep this milestone boring. No experiment logic yet. The little goblin cave needs shelves before it needs treasure.

## Milestone 1: Configuration and Utilities

### Goal

Add a typed configuration object and reproducibility helpers.

### Deliverables

In `src/helix_latents/config.py`:

```python
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
    device: str = "cuda"

    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
```

Also add helpers:

```python
def load_config(path: str) -> ExperimentConfig
def save_json(data: dict, path: str) -> None
def get_device(requested: str) -> torch.device
```

In `src/helix_latents/utils.py`:

```python
def set_seed(seed: int, deterministic: bool = False) -> None
```

### Acceptance Checks

Tests should verify:

1. Default config can be constructed.
2. YAML config can be loaded.
3. `helix_alpha` can be left as `None`.
4. `get_device("cuda")` falls back to CPU if CUDA is unavailable.
5. Same seed produces same random tensors on CPU.

### Definition of Done

A script can load `configs/modular_addition.yaml` and print an `ExperimentConfig`.

## Milestone 2: Modular Addition Dataset

### Goal

Generate and split the modular addition dataset.

### Deliverables

In `src/helix_latents/data.py`:

```python
def make_modular_addition_data(modulus: int) -> tuple[torch.Tensor, torch.Tensor]
```

Returns:

```text
pairs: LongTensor [N*N, 2]
targets: LongTensor [N*N]
```

where:

```text
target = (a + b) mod N
```

Add:

```python
class ModularAdditionDataset(torch.utils.data.Dataset)
```

Add:

```python
def split_dataset(
    pairs: torch.Tensor,
    targets: torch.Tensor,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[Dataset, Dataset, Dataset]
```

Add:

```python
def make_dataloaders(config: ExperimentConfig) -> dict[str, DataLoader]
```

### Acceptance Checks

Tests in `tests/test_data.py` should verify:

1. `N = 5` produces 25 examples.
2. Every ordered pair appears exactly once.
3. Targets are exactly correct.
4. Splits are disjoint.
5. Splits cover all examples.
6. Splits are deterministic for the same seed.
7. Splits differ for different seeds.
8. Dataloaders produce batches containing `a`, `b`, and `target`.

### Definition of Done

This should work:

```python
config = ExperimentConfig(modulus=7, batch_size=4)
loaders = make_dataloaders(config)
batch = next(iter(loaders["train"]))
assert set(batch.keys()) == {"a", "b", "target"}
```

## Milestone 3: Geometry Primitives

### Goal

Implement the direct mathematical objects: circle encoding, helix encoding, circle rotation, and helix shifting.

### Deliverables

In `src/helix_latents/geometry.py`:

```python
def number_to_theta(x: torch.Tensor, modulus: int) -> torch.Tensor
def circle_encode(x: torch.Tensor, modulus: int) -> torch.Tensor
def helix_encode(
    x: torch.Tensor,
    modulus: int,
    alpha: float | None = None,
) -> torch.Tensor
def rotate_circle(
    xy: torch.Tensor,
    k: int | torch.Tensor,
    modulus: int,
) -> torch.Tensor
def shift_helix(
    xyz: torch.Tensor,
    k: int | torch.Tensor,
    modulus: int,
    alpha: float | None = None,
    shift_axis: bool = True,
) -> torch.Tensor
```

Optional debugging helper:

```python
def infer_phase_step(xy: torch.Tensor, modulus: int) -> torch.Tensor
```

### Acceptance Checks

Tests in `tests/test_geometry.py` should verify:

1. Circle encodings have shape `[batch, 2]`.
2. Circle encodings have norm approximately `1`.
3. Helix encodings have shape `[batch, 3]`.
4. Helix phase coordinates have norm approximately `1`.
5. Rotating by `0` is identity.
6. Rotating by `N` is identity.
7. Rotating `circle_encode(a)` by `k` matches `circle_encode((a + k) % N)`.
8. Negative shifts work.
9. Shifts larger than `N` work.
10. `shift_helix(..., shift_axis=True)` increments the axis by `alpha * k`.
11. `shift_helix(..., shift_axis=False)` preserves the axis.

### Definition of Done

This property should hold:

```python
xy = circle_encode(a, N)
xy_shifted = rotate_circle(xy, k, N)
xy_expected = circle_encode((a + k) % N, N)
torch.testing.assert_close(xy_shifted, xy_expected)
```

## Milestone 4: Model Classes

### Goal

Implement the baseline, circle bottleneck, and helix bottleneck models with a shared intervention-friendly interface.

### Deliverables

In `src/helix_latents/models.py`:

```python
@dataclass
class ModelOutput:
    logits: torch.Tensor
    latents: dict[str, torch.Tensor]
```

Add:

```python
def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_hidden_layers: int,
    dropout: float,
) -> nn.Sequential
```

Add model classes:

```python
class BaselineMLP(nn.Module)
class CircleBottleneckMLP(nn.Module)
class HelixBottleneckMLP(nn.Module)
```

Each model must implement:

```python
def forward(
    self,
    a: torch.Tensor,
    b: torch.Tensor,
    latent_override: dict[str, torch.Tensor] | None = None,
) -> ModelOutput
```

Add:

```python
def build_model(config: ExperimentConfig) -> nn.Module
```

### Required Latent Conventions

For all models:

```python
output.latents["a"]
output.latents["b"]
```

For `CircleBottleneckMLP`:

```text
latents["a"].shape == [batch, 2]
latents["b"].shape == [batch, 2]
```

For `HelixBottleneckMLP`:

```text
latents["a"].shape == [batch, 3]
latents["b"].shape == [batch, 3]
```

For `BaselineMLP`:

```text
latents["a"].shape == [batch, embedding_dim]
latents["b"].shape == [batch, embedding_dim]
```

### Acceptance Checks

Tests in `tests/test_models.py` should verify:

1. Each model runs a forward pass.
2. Each model returns logits of shape `[batch, N]`.
3. Each model returns `latents["a"]` and `latents["b"]`.
4. Circle latent shapes are correct.
5. Helix latent shapes are correct.
6. Baseline latent shapes are correct.
7. `latent_override={"a": tensor}` changes the forward path without error.
8. Invalid override shapes raise useful errors.

### Definition of Done

This should run:

```python
model = HelixBottleneckMLP(modulus=59, hidden_dim=128, num_hidden_layers=2)
a = torch.tensor([0, 1, 2])
b = torch.tensor([3, 4, 5])
out = model(a, b)
assert out.logits.shape == (3, 59)
assert out.latents["a"].shape == (3, 3)
```

## Milestone 5: Training and Evaluation Loops

### Goal

Train a model on modular addition and evaluate normal task accuracy.

### Deliverables

In `src/helix_latents/train.py`:

```python
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]
```

Add:

```python
@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]
```

Add:

```python
def fit(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    config: ExperimentConfig,
) -> dict[str, Any]
```

In `src/helix_latents/evaluate.py`:

```python
def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float
```

### Required Fit Behavior

The fit loop should:

1. Set up optimizer.
2. Train for up to `max_epochs`.
3. Evaluate validation after each epoch.
4. Track loss and accuracy.
5. Save best model by validation accuracy, breaking ties by validation loss.
6. Support early stopping.
7. Return a history dictionary.

Checkpoint format:

```python
{
    "model_state_dict": model.state_dict(),
    "config": asdict(config),
    "history": history,
    "best_epoch": best_epoch,
    "best_val_accuracy": best_val_accuracy,
}
```

### Acceptance Checks

Tests in `tests/test_training_smoke.py` should verify:

1. A tiny model trains without crashing.
2. Evaluation returns `loss`, `accuracy`, and `num_examples`.
3. Training improves loss or reaches a modest accuracy threshold.
4. A checkpoint dictionary can be saved and loaded.

Use a tiny setting:

```text
N = 7
hidden_dim = 32
max_epochs = 50
batch_size = 16
```

Do not require perfect accuracy in default CI tests.

### Definition of Done

This should work locally:

```bash
python scripts/train_modular_addition.py --model-type helix_bottleneck_mlp
```

and produce a checkpoint.

## Milestone 6: Training Script

### Goal

Add a user-facing script that trains any model variant from config or CLI overrides.

### Deliverables

In `scripts/train_modular_addition.py`:

CLI should support:

```bash
python scripts/train_modular_addition.py --config configs/modular_addition.yaml
```

Optional overrides:

```bash
python scripts/train_modular_addition.py \
  --model-type circle_bottleneck_mlp \
  --modulus 59 \
  --seed 0
```

Script behavior:

1. Load config.
2. Apply CLI overrides.
3. Set seed.
4. Build dataloaders.
5. Build model.
6. Train model.
7. Save checkpoint.
8. Save training history as JSON.
9. Print final train, validation, and test metrics.

### Acceptance Checks

Manual check:

```bash
python scripts/train_modular_addition.py --model-type circle_bottleneck_mlp --modulus 11 --max-epochs 20
```

should finish quickly and write outputs.

Automated check can run the script with very small settings in a subprocess.

### Definition of Done

A contributor can train a first model without writing Python code.

## Milestone 7: Intervention Mechanics

### Goal

Implement latent interventions independently of full experiment reporting.

### Deliverables

In `src/helix_latents/intervene.py`:

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
) -> ModelOutput
```

Supported modes:

```text
phase_plus_axis
phase_only
axis_only
random
```

Behavior by model type:

For `circle_bottleneck_mlp`:

```text
phase_plus_axis and phase_only both rotate the 2D circle.
axis_only should raise a clear error or behave as no-op with a warning.
```

For `helix_bottleneck_mlp`:

```text
phase_plus_axis rotates phase and shifts axis.
phase_only rotates phase only.
axis_only shifts axis only.
random applies a control perturbation.
```

For `baseline_mlp`:

```text
random is supported as a control.
phase modes should raise clear errors.
```

Add:

```python
def expected_intervention_targets(
    a: torch.Tensor,
    b: torch.Tensor,
    k: int,
    modulus: int,
) -> torch.Tensor
```

Definition:

```text
(a + k + b) mod N
```

### Acceptance Checks

Tests in `tests/test_interventions.py` should verify:

1. Expected target calculation is correct.
2. Intervened logits have shape `[batch, N]`.
3. Circle intervention modifies the phase latent correctly.
4. Helix `phase_only` leaves axis unchanged.
5. Helix `axis_only` leaves phase unchanged.
6. Negative shifts work.
7. Large shifts work.
8. Unsupported mode/model combinations raise helpful errors.

### Definition of Done

The model can be run normally, its `a` latent can be modified, and the modified latent can be passed back through the same forward path without changing input tokens.

## Milestone 8: Intervention Evaluation

### Goal

Measure whether latent rotation causally shifts model predictions.

### Deliverables

In `src/helix_latents/intervene.py` or `src/helix_latents/evaluate.py`:

```python
@torch.no_grad()
def evaluate_intervention(
    model: nn.Module,
    dataloader: DataLoader,
    config: ExperimentConfig,
    shifts: list[int],
    mode: str = "phase_plus_axis",
) -> dict[str, Any]
```

For each shift `k`:

1. Compute intervened output.
2. Compute expected target:

```text
(a + k + b) mod N
```

3. Compute accuracy.
4. Store per-shift and overall metrics.

Required output:

```python
{
    "model_type": config.model_type,
    "mode": mode,
    "modulus": config.modulus,
    "accuracy_by_shift": {
        "1": 0.99,
        "2": 0.98
    },
    "overall_accuracy": 0.985,
    "num_examples": 522,
}
```

Optional but useful:

```python
{
    "predictions_by_shift": ...,
    "targets_by_shift": ...
}
```

Only include full arrays if saved to disk in a compact format.

### Acceptance Checks

Tests should verify:

1. Output dictionary has expected keys.
2. All requested shifts appear.
3. Accuracy values are between `0` and `1`.
4. `num_examples` is correct.
5. Evaluation works on CPU.

### Definition of Done

A trained circle or helix model can be evaluated with:

```python
results = evaluate_intervention(model, test_loader, config, shifts=[1, 2, 5])
```

## Milestone 9: Intervention Script

### Goal

Add a command-line script to run interventions on a saved checkpoint.

### Deliverables

In `scripts/run_interventions.py`:

CLI:

```bash
python scripts/run_interventions.py \
  --checkpoint checkpoints/best.pt \
  --shifts 1 2 3 5 10 \
  --mode phase_plus_axis
```

Script behavior:

1. Load checkpoint.
2. Rebuild config.
3. Rebuild model.
4. Load weights.
5. Rebuild data split using original seed.
6. Run intervention evaluation on test set.
7. Save JSON results.
8. Print a compact summary.

### Acceptance Checks

Manual check:

```bash
python scripts/run_interventions.py --checkpoint checkpoints/best.pt --shifts 1 2
```

should print something like:

```text
shift=1 intervention_accuracy=0.98
shift=2 intervention_accuracy=0.97
overall=0.975
```

### Definition of Done

A contributor can train a model and run causal latent interventions entirely from CLI.

## Milestone 10: Plotting and Result Artifacts

### Goal

Generate basic plots and result files for human inspection.

### Deliverables

In `src/helix_latents/plotting.py`:

```python
def plot_training_history(history: dict, output_path: str) -> None
def plot_intervention_accuracy(results: dict, output_path: str) -> None
def plot_confusion_matrix(
    expected: np.ndarray,
    predicted: np.ndarray,
    modulus: int,
    output_path: str,
) -> None
```

Update scripts to save:

```text
results/training_history.json
results/intervention_results.json
results/training_history.png
results/intervention_accuracy.png
```

Optional:

```text
results/confusion_shift_1.png
```

### Acceptance Checks

Manual check:

1. Training history plot exists.
2. Intervention accuracy plot exists.
3. Plots open without error.
4. JSON files are readable.

Automated tests can verify that plotting functions create files from toy data, but avoid requiring exact image comparison.

### Definition of Done

After training and intervention, the repo contains enough artifacts to understand whether the experiment worked.

## Milestone 11: First End-to-End Run

### Goal

Run the complete experiment locally for a small modulus and confirm the pipeline works.

### Recommended Settings

```text
modulus = 11
model_type = circle_bottleneck_mlp
hidden_dim = 64
num_hidden_layers = 2
max_epochs = 300
batch_size = 32
seed = 0
```

### Steps

```bash
python scripts/train_modular_addition.py \
  --model-type circle_bottleneck_mlp \
  --modulus 11 \
  --max-epochs 300

python scripts/run_interventions.py \
  --checkpoint checkpoints/best.pt \
  --shifts 1 2 3 5 \
  --mode phase_only
```

### Expected Result

Normal test accuracy should be high.

Intervention accuracy should be substantially above chance.

For `N = 11`, chance accuracy is approximately:

```text
9.1%
```

A good first target:

```text
test_accuracy > 95%
intervention_accuracy > 80%
```

### Definition of Done

There is at least one saved run where:

1. Normal accuracy is high.
2. Intervention accuracy is clearly above chance.
3. Results are saved in JSON.
4. Intervention plot is saved.

## Milestone 12: Full Default Run

### Goal

Run the intended default experiment with `N = 59`.

### Recommended Settings

```text
modulus = 59
model_type = helix_bottleneck_mlp
hidden_dim = 128
num_hidden_layers = 2
max_epochs = 500
batch_size = 128
seed = 0
```

### Steps

```bash
python scripts/train_modular_addition.py \
  --model-type helix_bottleneck_mlp \
  --modulus 59

python scripts/run_interventions.py \
  --checkpoint checkpoints/best.pt \
  --shifts 1 2 3 5 10 17 29 \
  --mode phase_plus_axis
```

Also run:

```bash
python scripts/run_interventions.py \
  --checkpoint checkpoints/best.pt \
  --shifts 1 2 3 5 10 17 29 \
  --mode phase_only

python scripts/run_interventions.py \
  --checkpoint checkpoints/best.pt \
  --shifts 1 2 3 5 10 17 29 \
  --mode axis_only

python scripts/run_interventions.py \
  --checkpoint checkpoints/best.pt \
  --shifts 1 2 3 5 10 17 29 \
  --mode random
```

### Expected Result

For `N = 59`, chance accuracy is approximately:

```text
1.7%
```

A strong result:

```text
test_accuracy > 95%
phase intervention accuracy > 80%
random control near chance or no systematic shift
```

Perfect results are not required.

### Definition of Done

The default experiment produces interpretable results and control comparisons.

## Milestone 13: Documentation Update

### Goal

Make the project usable by someone who did not write it.

### Deliverables

Update `README.md` with:

1. Project purpose.
2. Installation instructions.
3. How to run tests.
4. How to train a model.
5. How to run interventions.
6. How to interpret result files.
7. A short explanation of the causal intervention.

Add a short result summary once the first successful run exists.

Example:

```text
The circle bottleneck model reached 99.8% normal test accuracy on N=59.
Rotating the latent phase by k produced 97.4% intervention accuracy averaged over shifts [1, 2, 3, 5, 10].
Random latent perturbations did not produce systematic shifts.
```

Use actual numbers from runs.

### Acceptance Checks

A new contributor can clone the repo, install dependencies, run tests, train a small model, and run interventions by following the README.

### Definition of Done

Documentation matches the implemented commands.

## Milestone 14: Slow Tests and Regression Checks

### Goal

Add optional tests that verify learned intervention behavior without making normal CI flaky.

### Deliverables

Add:

```text
tests/test_intervention_learning_slow.py
```

Mark tests with:

```python
@pytest.mark.slow
```

Suggested test:

```text
N = 11
model_type = circle_bottleneck_mlp
max_epochs = 300
target test_accuracy > 0.95
target intervention_accuracy for k=1 > 0.80
```

Add pytest config so slow tests are skipped by default.

Example:

```ini
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow",
]
```

### Acceptance Checks

Default:

```bash
pytest
```

runs fast.

Slow:

```bash
pytest -m slow
```

runs learned-behavior tests.

### Definition of Done

There is a regression test for the core phenomenon, but it does not slow down ordinary development.

## Milestone 15: Result Comparison Across Models

### Goal

Compare baseline, circle bottleneck, and helix bottleneck models under the same conditions.

### Deliverables

Add a script or notebook-like Python script:

```text
scripts/compare_models.py
```

It should train or load:

```text
baseline_mlp
circle_bottleneck_mlp
helix_bottleneck_mlp
```

For each model, report:

```text
normal test accuracy
phase intervention accuracy
axis-only intervention accuracy, if applicable
random control behavior
```

Save:

```text
results/model_comparison.json
results/model_comparison.md
```

### Acceptance Checks

The comparison result should make it clear whether structured bottlenecks offer cleaner causal control than the baseline.

### Definition of Done

The repo has a repeatable comparison across all first-generation model types.

## Milestone 16: Optional Learned-Basis Helix

### Goal

Move one step closer to mechanistic interpretability in high-dimensional neural networks.

### Idea

Instead of feeding `[cos θ, sin θ, z]` directly into the MLP, learn basis vectors:

```text
h = cos(θ)u + sin(θ)v + z w
```

where:

```text
u, v, w ∈ R^d
```

are learned parameters.

### Deliverables

Add model:

```python
class LearnedBasisHelixMLP(nn.Module)
```

The model should store both:

```python
latents["a_coefficients"]  # [cos θ, sin θ, z]
latents["a_projected"]     # high-dimensional vector
```

Interventions should operate first on coefficients.

### Acceptance Checks

1. Forward pass works.
2. Coefficient latents are inspectable.
3. Projected latents are high-dimensional.
4. Intervention on coefficients changes output.
5. Learned basis vectors can be saved and inspected.

### Definition of Done

A helix can live inside a learned subspace rather than only as a raw three-dimensional input.

## Milestone 17: Optional Non-Modular Task

### Goal

Test whether a true helix helps when both cycle and progression matter.

### Candidate Tasks

```text
integer addition with range extrapolation
counting with periodic resets
line wrapping
calendar arithmetic
musical beat and measure prediction
bracket depth with periodic markers
```

### Recommended First Extension

Line wrapping:

```text
Given current character count and line width, predict whether the next token crosses the boundary.
```

Why this task is good:

1. It has cyclic structure: position within the line.
2. It has monotonic structure: total progress through text.
3. It connects to real transformer interpretability work.
4. It should favor helix-like representations over pure circles.

### Definition of Done

There is at least one task where the axial coordinate is meaningfully useful.

## Overall Project Acceptance Criteria

The first complete version of the project is done when:

1. Non-slow tests pass.
2. A small `N = 11` end-to-end run works.
3. A default `N = 59` run works.
4. Circle and helix bottleneck models train successfully.
5. Latent interventions can be run from CLI.
6. Intervention accuracy is reported by shift.
7. Random controls are implemented.
8. Result plots are generated.
9. Documentation explains how to reproduce the experiment.
10. The repo contains a saved example result or result summary.

## Suggested Implementation Order

The shortest path to a meaningful result is:

```text
0. repo skeleton
1. config and utils
2. data
3. geometry
4. models
5. training
6. train script
7. intervention mechanics
8. intervention evaluation
9. intervention script
10. plots
11. small end-to-end run
12. default run
13. docs
```

Everything after that is optional expansion.

## Risk Register

### Risk: Model Solves Task But Intervention Fails

Interpretation:

The model may use the bottleneck as a lookup coordinate rather than as a smooth geometric variable.

Mitigation:

Add stronger architectural equivariance or train with latent augmentation.

### Risk: Circle Performs as Well as Helix

Interpretation:

Modular addition only needs phase. The axis is not necessary.

Mitigation:

Move to tasks that require both cycle and progression.

### Risk: Baseline Also Shows Intervention-Like Behavior

Interpretation:

The control may be poorly designed, or the baseline embedding may have accidentally learned circular geometry.

Mitigation:

Probe baseline embeddings. Test whether learned embeddings form a circle. Add random-subspace and permutation controls.

### Risk: Training Is Flaky

Interpretation:

Small models and random splits may create variance.

Mitigation:

Use fixed seeds, smaller `N` for smoke tests, and mark learned-behavior tests as slow.

### Risk: Results Look Good But Are Leaky

Interpretation:

The intervention code may accidentally feed `(a + k)` as input.

Mitigation:

Test that inputs remain unchanged during intervention. Review intervention code carefully.

## Final North Star

The first milestone chain should produce a simple causal statement:

> We trained a tiny model with a circular or helical latent bottleneck. When we rotated that latent by `k`, the model behaved as though the represented number had increased by `k`.

That is the tiny proof-of-life for helix-like objects as direct model components.
