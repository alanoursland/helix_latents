# Technical Design: Rotated MNIST with HelixConv Layers

## 1. Purpose

This document specifies the implementation design for a rotated MNIST experiment using vanilla CNN, parameter-matched CNN, CircleConv, and HelixConv models.

The experiment is the first in the project to directly test whether the geometric layers **self-organize** on a task with input symmetry. Unlike the modular addition experiment, the geometry is not imposed at the input. Unlike MNIST, CIFAR-10, and Covertype, the task has a clear underlying continuous symmetry (rotation) that the layer's primitives are built to express.

The central question is:

```text
Does HelixConv spontaneously organize so that phase tracks input orientation,
when the only signal it receives is a rotation-invariant classification objective?
```

## 2. Scope

Dataset:

```text
Rotated MNIST
```

Input:

```text
[1, 28, 28] grayscale image
random rotation θ ∈ [0, 2π) applied per example
```

Output:

```text
10-class digit label
```

Model families:

```text
standard_cnn
standard_cnn_matched
circle_conv
helix_conv
```

Primary measurements (in order of importance):

```text
1. filter-pair self-organization  (W_u vs rotated W_v)
2. (a, b) trajectory under input rotation sweep
3. test accuracy on rotated test set
```

Secondary measurements:

```text
test accuracy on un-rotated test set
test accuracy on fixed-rotation grid
mean epoch time
parameter count
optional: causal intervention via (a, b) rotation
```

The visualization analyses are the core deliverables of this experiment. The accuracy comparison is a sanity check, not the main result.

## 3. Non-Goals

This experiment should not:

1. attempt to beat rotation-equivariant CNNs (e.g., Harmonic Networks, E(2)-CNN);
2. enforce architectural rotation equivariance via tied weights;
3. claim helix layers are generally better than vanilla convs;
4. extend to color or natural images in v1;
5. require a specific batch size or hardware footprint;
6. convert the repo into an installable package.

This is a local script-based experiment that asks whether the layer learns the geometric thing on its own.

## 4. Local File Layout

Add these files:

```text
rot_mnist_config.py
rot_mnist_data.py
rot_mnist_models.py
rot_mnist_train.py
rot_mnist_run_experiment.py
rot_mnist_analyze_filters.py
rot_mnist_analyze_trajectory.py
rot_mnist_analyze_intervention.py
rot_mnist_test_all.py
rot_mnist_results.md
```

Generated directories:

```text
rot_mnist_data/
rot_mnist_results/
rot_mnist_checkpoints/
rot_mnist_figures/
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
scipy
```

`scipy` is useful for `scipy.ndimage.rotate` when rotating learned filters at non-90° angles for the filter-pair correlation analysis. `torch.nn.functional.affine_grid` + `grid_sample` is an acceptable alternative if avoiding scipy.

`sklearn` is not required for this experiment.

## 6. Configuration Design

File:

```text
rot_mnist_config.py
```

### 6.1 RotMNISTConfig

```python
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
```

```python
@dataclass
class RotMNISTConfig:
    model_type: Literal[
        "standard_cnn",
        "standard_cnn_matched",
        "circle_conv",
        "helix_conv",
    ] = "helix_conv"

    scale: Literal["small", "medium", "large"] = "small"

    batch_size: int = 128
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.0
    use_layernorm: bool = False

    input_channels: int = 1
    num_classes: int = 10

    hidden_channels: int = 32
    matched_hidden_channels: int = 48
    circle_units: int = 16
    helix_units: int = 16
    kernel_size: int = 5
    num_conv_blocks: int = 2

    rotation_max_degrees: int = 180
    rotation_fill: float = 0.0

    seed: int = 0
    device: str = "cuda"

    data_dir: str = "rot_mnist_data"
    results_dir: str = "rot_mnist_results"
    checkpoint_dir: str = "rot_mnist_checkpoints"
    figures_dir: str = "rot_mnist_figures"

    limit_train_batches: int | None = None
    limit_eval_batches: int | None = None

    use_scheduler: bool = False
    scheduler_type: Literal["none", "cosine"] = "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

### 6.2 Scale Presets

The default scale for this experiment is **small**, because the analysis figures are easier to read with a small number of helix units.

```python
SCALE_PRESETS = {
    "small": {
        "hidden_channels": 32,
        "circle_units": 16,
        "helix_units": 16,
        "matched_hidden_channels": 48,
    },
    "medium": {
        "hidden_channels": 64,
        "circle_units": 32,
        "helix_units": 32,
        "matched_hidden_channels": 96,
    },
    "large": {
        "hidden_channels": 128,
        "circle_units": 64,
        "helix_units": 64,
        "matched_hidden_channels": 192,
    },
}
```

Implement:

```python
def apply_scale_preset(config: RotMNISTConfig) -> RotMNISTConfig:
    ...
```

### 6.3 Kernel Size Choice

Default kernel size is 5, larger than the typical 3 used for MNIST.

The reason is analysis-driven: filter-pair rotation is much easier to detect visually and numerically at kernel sizes ≥ 5. A 3×3 kernel has so few degrees of freedom that "rotated by 90°" is hard to distinguish from "permuted" or "sign-flipped."

If kernel size is changed, the filter-pair correlation analysis should still work, but interpretation becomes harder.

### 6.4 Utility Functions

Same as in previous experiments:

```python
def get_device(requested: str = "cuda") -> torch.device:
    ...

def save_json(data: dict, path: str | Path) -> None:
    ...

def set_seed(seed: int) -> None:
    ...
```

## 7. Data Design

File:

```text
rot_mnist_data.py
```

### 7.1 Loading the Dataset

Use `torchvision.datasets.MNIST` with rotation augmentation:

```python
from torchvision import datasets, transforms
```

### 7.2 Three Distinct Transforms

This experiment uses **three different transform pipelines** for training and evaluation. The distinction is important.

**Training transform** (random rotation):

```python
train_transform = transforms.Compose([
    transforms.RandomRotation(
        degrees=config.rotation_max_degrees,
        fill=config.rotation_fill,
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
```

**Rotated test transform** (random rotation, deterministic seed for reproducibility):

```python
rotated_test_transform = transforms.Compose([
    transforms.RandomRotation(
        degrees=config.rotation_max_degrees,
        fill=config.rotation_fill,
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
```

**Un-rotated test transform** (no rotation; transfer evaluation):

```python
unrotated_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
```

Reporting test accuracy on both rotated and un-rotated test sets gives a sense of whether the model learned rotation invariance versus ignored rotation entirely.

### 7.3 Split

Standard MNIST split:

```text
train: 55,000   (random rotation)
val:    5,000   (random rotation; split from train)
test:  10,000   (random rotation)
```

Use:

```python
torch.utils.data.random_split
```

with a fixed generator seeded by `config.seed`.

### 7.4 Fixed-Rotation-Grid Test Set

Required for trajectory analysis (Measurement 3 in the experiment description).

Implement:

```python
def make_fixed_rotation_grid(
    base_image: torch.Tensor,
    num_angles: int = 72,
    fill: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      images: [num_angles, 1, 28, 28] tensor of the same image at evenly-spaced rotations
      angles: [num_angles] tensor of rotation angles in radians
    """
    ...
```

The default `num_angles=72` corresponds to 5° increments through a full 360° sweep. Use `torchvision.transforms.functional.rotate` or `torch.nn.functional.affine_grid` + `grid_sample` for the rotation.

The base image should be drawn from the **un-normalized** test set (raw [0, 1] tensor), then normalized after rotation, to avoid normalization artifacts at the rotation boundary.

### 7.5 Dataloader Function

Implement:

```python
def make_rot_mnist_dataloaders(config: RotMNISTConfig) -> dict[str, DataLoader]:
    ...
```

Return:

```text
train               (rotated, shuffled)
val                 (rotated, not shuffled)
test_rotated        (rotated, not shuffled)
test_unrotated      (not rotated, not shuffled)
```

### 7.6 Shape Contract

Batch images:

```text
[batch, 1, 28, 28]
```

Batch targets:

```text
[batch]
```

Model logits:

```text
[batch, 10]
```

## 8. Model Design

File:

```text
rot_mnist_models.py
```

Provide:

```python
count_parameters
StandardCNN
CircleConv2d
HelixConv2d
CircleCNN
HelixCNN
build_rot_mnist_model
```

## 9. StandardCNN

```python
class StandardCNN(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        hidden_channels: int = 32,
        kernel_size: int = 5,
        num_conv_blocks: int = 2,
        dropout: float = 0.0,
    ):
        ...
```

For `num_conv_blocks=2`:

```text
Conv2d(1, hidden_channels, kernel_size, padding=k//2)
GELU
MaxPool2d(2)
Conv2d(hidden_channels, hidden_channels, kernel_size, padding=k//2)
GELU
MaxPool2d(2)
Flatten
Linear(hidden_channels * 7 * 7, num_classes)
```

For 28×28 input with two MaxPool2d(2) stages: spatial dim after pooling is 7×7.

The matched variant uses `matched_hidden_channels` instead.

## 10. CircleConv2d

```python
class CircleConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        units: int,
        out_channels: int,
        kernel_size: int = 5,
        padding: int = 2,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.units = units

        self.conv_u = nn.Conv2d(in_channels, units, kernel_size, padding=padding)
        self.conv_v = nn.Conv2d(in_channels, units, kernel_size, padding=padding)

        # 5 features per unit: sin_t, cos_t, r, r*sin_t, r*cos_t
        self.project = nn.Conv2d(units * 5, out_channels, kernel_size=1)
```

Forward pass:

```python
def forward(self, x):
    a = self.conv_u(x)
    b = self.conv_v(x)
    r = torch.sqrt(a * a + b * b + self.eps)
    sin_t = b / r
    cos_t = a / r

    feats = torch.cat([
        sin_t,
        cos_t,
        r,
        r * sin_t,
        r * cos_t,
    ], dim=1)

    return self.project(feats)
```

### Critical: Independent Filter Banks

`conv_u` and `conv_v` are independent learnable convolutions. They are **not** constrained to be 90°-rotated copies of each other. Tying them would convert this experiment into Harmonic Networks and invalidate the self-organization claim.

## 11. HelixConv2d

```python
class HelixConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        units: int,
        out_channels: int,
        kernel_size: int = 5,
        padding: int = 2,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.units = units

        self.conv_u = nn.Conv2d(in_channels, units, kernel_size, padding=padding)
        self.conv_v = nn.Conv2d(in_channels, units, kernel_size, padding=padding)
        self.conv_w = nn.Conv2d(in_channels, units, kernel_size, padding=padding)

        # 8 features per unit: sin_t, cos_t, r, z, r*sin_t, r*cos_t, tanh(z), r*tanh(z)
        self.project = nn.Conv2d(units * 8, out_channels, kernel_size=1)
```

Forward pass:

```python
def forward(self, x):
    a = self.conv_u(x)
    b = self.conv_v(x)
    z = self.conv_w(x)

    r = torch.sqrt(a * a + b * b + self.eps)
    sin_t = b / r
    cos_t = a / r

    feats = torch.cat([
        sin_t,
        cos_t,
        r,
        z,
        r * sin_t,
        r * cos_t,
        torch.tanh(z),
        r * torch.tanh(z),
    ], dim=1)

    return self.project(feats)
```

`conv_u`, `conv_v`, `conv_w` are independent.

### Hooks for Analysis

To make Measurement 3 (trajectory analysis) tractable, expose the intermediate `(a, b, z)` tensors via a forward hook or by adding an alternative method:

```python
def forward_with_intermediates(self, x):
    a = self.conv_u(x)
    b = self.conv_v(x)
    z = self.conv_w(x)
    r = torch.sqrt(a * a + b * b + self.eps)
    sin_t = b / r
    cos_t = a / r

    feats = torch.cat([sin_t, cos_t, r, z, r * sin_t, r * cos_t,
                       torch.tanh(z), r * torch.tanh(z)], dim=1)
    out = self.project(feats)

    return out, {"a": a, "b": b, "z": z, "r": r}
```

This avoids the need for forward hooks during analysis.

## 12. CircleCNN and HelixCNN

```python
class HelixCNN(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        helix_units: int = 16,
        hidden_channels: int = 32,
        kernel_size: int = 5,
        num_conv_blocks: int = 2,
        dropout: float = 0.0,
    ):
        ...
```

For `num_conv_blocks=2`:

```text
HelixConv2d(1, helix_units, hidden_channels)
GELU
MaxPool2d(2)
HelixConv2d(hidden_channels, helix_units, hidden_channels)
GELU
MaxPool2d(2)
Flatten
Linear(hidden_channels * 7 * 7, num_classes)
```

CircleCNN follows the same pattern with CircleConv2d.

## 13. Model Factory

Implement:

```python
def build_rot_mnist_model(config: RotMNISTConfig) -> nn.Module:
    ...
```

Dispatch on:

```text
standard_cnn
standard_cnn_matched
circle_conv
helix_conv
```

## 14. Parameter Counting

```python
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

Report parameter counts for every run.

The matched dense baseline (`standard_cnn_matched`) should be sized by the implementer so its parameter count is approximately equal to `helix_conv` at the same scale. Exact matching is not required, but the count must be reported.

## 15. Training Design

File:

```text
rot_mnist_train.py
```

### 15.1 Loss

```python
F.cross_entropy(logits, targets)
```

### 15.2 Optimizer

```python
torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
)
```

### 15.3 Scheduler

V1 skips scheduler. Optional cosine decay if training is unstable.

### 15.4 Checkpoint Selection

Use best validation accuracy on the rotated validation set. This is balanced (10 classes, equal frequency), so accuracy and macro F1 will agree closely. Accuracy is sufficient.

### 15.5 History

Track per epoch:

```text
train_loss
train_accuracy
train_seconds
train_examples_per_second
val_loss
val_accuracy
```

### 15.6 Test Metrics

After restoring the best checkpoint, evaluate on:

```text
test_rotated_loss
test_rotated_accuracy
test_unrotated_loss
test_unrotated_accuracy
```

The gap between rotated and un-rotated test accuracy is informative. A model that has learned rotation invariance should perform similarly on both.

## 16. Filter Analysis: Measurement 2

File:

```text
rot_mnist_analyze_filters.py
```

This is the central analysis figure of the experiment.

### 16.1 Goal

For each helix unit in the **first** HelixConv layer (or each circle unit in the first CircleConv layer):

1. Extract the learned 2D filter `W_u[unit, in_channel, :, :]` and `W_v[unit, in_channel, :, :]`.
2. For single-channel MNIST input, `in_channel = 0`.
3. Find the rotation angle `φ*` that maximizes correlation between `W_u` and `rotate(W_v, φ)`.
4. Record `φ*` and the correlation value at `φ*`.

### 16.2 Rotation of Filters

Use bilinear rotation. Two acceptable approaches:

```python
# Option A: scipy
from scipy.ndimage import rotate
W_v_rotated = rotate(W_v, angle_deg, reshape=False, order=1, mode="constant")
```

```python
# Option B: torch
import torch.nn.functional as F
# Build affine grid at angle_rad and use grid_sample
```

### 16.3 Correlation Metric

```python
def normalized_correlation(A: np.ndarray, B: np.ndarray) -> float:
    A = A - A.mean()
    B = B - B.mean()
    denom = np.sqrt((A * A).sum() * (B * B).sum()) + 1e-12
    return float((A * B).sum() / denom)
```

### 16.4 Sweep

For each unit, sweep `φ ∈ {0°, 5°, 10°, ..., 355°}`. Find:

```python
phi_star = argmax_phi  normalized_correlation(W_u, rotate(W_v, phi))
corr_star = max_phi    normalized_correlation(W_u, rotate(W_v, phi))
```

### 16.5 Reported Figures

```text
filter_pair_grid.png
  Side-by-side visualization of W_u and W_v for every unit.
  Sorted by phi_star.

phi_star_histogram.png
  Histogram of phi_star across all units.
  Peak near 90° = self-organized into rotated quadrature pairs.
  Uniform = no self-organization.
  Peak near 0° or 180° = duplicates or sign flips, not rotations.

correlation_histogram.png
  Histogram of corr_star across all units.
  High values mean clean pairing; low values mean noisy or absent pairing.
```

These three figures together largely determine the result of the experiment.

### 16.6 Layer Selection

The default analysis applies to the **first** HelixConv layer, where filters operate directly on the input image. Filters in deeper layers operate on already-transformed feature maps and are harder to interpret as oriented edge detectors.

If desired, run the same analysis on the second HelixConv layer for comparison, but expect noisier results.

## 17. Trajectory Analysis: Measurement 3

File:

```text
rot_mnist_analyze_trajectory.py
```

### 17.1 Goal

For a fixed un-rotated MNIST digit, rotate the input through θ ∈ [0°, 360°) at fine resolution. At each rotation, record the value of `(a, b)` at one spatial position for one helix unit. Plot the trajectory in 2D.

If the trajectory is a clean circle, that unit's phase tracks input orientation.

### 17.2 Procedure

```python
def trajectory_for_unit(
    model: HelixCNN,
    base_image: torch.Tensor,        # [1, 1, 28, 28], un-normalized
    layer_idx: int = 0,              # which HelixConv layer
    unit_idx: int = 0,               # which helix unit in that layer
    spatial_position: tuple[int, int] = (14, 14),
    num_angles: int = 72,
    fill: float = 0.0,
) -> dict[str, np.ndarray]:
    """
    Returns:
      angles: [num_angles] in radians
      a, b, z, r: [num_angles] arrays
    """
```

Steps:

1. Generate `num_angles` rotated copies of the base image.
2. Normalize after rotation.
3. Run forward through the model up to `layer_idx`, capturing `(a, b, z, r)` via `forward_with_intermediates`.
4. Index into `[unit_idx, spatial_position[0], spatial_position[1]]`.

### 17.3 Reported Figures

```text
trajectory_grid_<layer>.png
  Grid of (a, b) trajectory plots, one per unit.
  Color-coded by input rotation angle.

trajectory_summary_<layer>.png
  Per-unit summary metrics:
    - circularity score (ratio of minor/major axis of best-fit ellipse)
    - radius variance under rotation (low = r is rotation-invariant)
    - z variance under rotation (low = axis is rotation-invariant)
    - winding number (how many full circles in 360° rotation)
```

### 17.4 Multiple Inputs and Positions

Run the analysis for at least:

```text
3 distinct base digits
3 spatial positions
all units in the first HelixConv layer
```

Save aggregated results to:

```text
rot_mnist_results/<run>/trajectory.json
```

A unit that traces clean circles for multiple inputs at multiple positions is genuinely tracking orientation. A unit that traces circles for one digit only might be tracking that digit's specific features rather than orientation per se.

## 18. Causal Intervention: Optional Measurement 4

File:

```text
rot_mnist_analyze_intervention.py
```

This is the strongest version of the test, analogous to the original modular addition experiment. Run only if Measurements 2 and 3 already show self-organization.

### 18.1 Procedure

For a fixed input image:

1. Run the model and record the activations.
2. At the first HelixConv layer, replace `(a, b)` at every spatial position with a rotated version:

```text
a' = a cos(Δθ) - b sin(Δθ)
b' = a sin(Δθ) + b cos(Δθ)
```

3. Pass the modified activations through the rest of the model.
4. Compare the output to running the model on a version of the input rotated by `Δθ`.

### 18.2 Metric

```python
intervention_match_accuracy = mean(
    argmax(logits_intervened) == argmax(logits_actually_rotated_input)
)
```

If self-organization happened, this should be high. If not, it will be near chance or near "always predict the original class regardless of intervention."

### 18.3 Sweeps

```text
Δθ ∈ {15°, 30°, 60°, 90°, 180°}
multiple seeds, multiple test images
```

## 19. Artifacts

Per run directory:

```text
rot_mnist_results/<model_type>_<scale>_seed<seed>/
```

Save:

```text
metrics.json
history.json
training_history.png
```

For helix and circle runs, additionally save:

```text
filter_pair_grid.png
phi_star_histogram.png
correlation_histogram.png
trajectory_grid_layer0.png
trajectory_grid_layer1.png
trajectory_summary_layer0.png
trajectory.json
```

If intervention analysis is run:

```text
intervention.json
intervention_summary.png
```

Checkpoint:

```text
rot_mnist_checkpoints/<model_type>_<scale>_seed<seed>_best.pt
```

## 20. Experiment Runner

File:

```text
rot_mnist_run_experiment.py
```

### 20.1 CLI Arguments

Support:

```text
--model-type
--all-models
--scale
--sweep-scales
--quick
--epochs
--batch-size
--hidden-channels
--matched-hidden-channels
--circle-units
--helix-units
--kernel-size
--num-conv-blocks
--rotation-max-degrees
--learning-rate
--weight-decay
--seed
--device
--data-dir
--limit-train-batches
--limit-eval-batches
--use-scheduler
--scheduler-type
--run-filter-analysis
--run-trajectory-analysis
--run-intervention-analysis
```

### 20.2 Quick Mode

Quick mode:

```text
epochs = 1
scale = small
limit_train_batches = 50
limit_eval_batches = 20
```

Quick mode is for smoke testing only. Analysis figures from a quick run are not interpretable; they exist only to verify the analysis pipeline.

### 20.3 All Models

Run:

```text
standard_cnn
standard_cnn_matched
circle_conv
helix_conv
```

### 20.4 Sweep Scales

Run:

```text
small
medium
large
```

Default first run is `--all-models --scale small` because the analysis figures are most readable at small unit counts.

## 21. Plotting

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
test_rotated_accuracy   (single horizontal line)
test_unrotated_accuracy (single horizontal line)
```

Comparison plots:

```text
accuracy_vs_params.png
rotated_vs_unrotated_accuracy.png
```

The rotated-vs-un-rotated comparison is informative: a model that learned rotation invariance should sit near the diagonal.

## 22. Testing Design

File:

```text
rot_mnist_test_all.py
```

Support:

```text
--data
--slow
```

### 22.1 Fast Tests

Do not require dataset download.

```text
test_config_defaults
test_apply_scale_preset
test_standard_cnn_forward_shape
test_circle_conv_forward_shape
test_helix_conv_forward_shape
test_helix_conv_forward_with_intermediates
test_circle_cnn_forward_shape
test_helix_cnn_forward_shape
test_all_models_no_nans
test_all_models_backward_pass
test_count_parameters_positive
test_synthetic_training_step
```

Synthetic batch shape:

```text
images:  [batch, 1, 28, 28]
targets: [batch]
logits:  [batch, 10]
```

For `test_helix_conv_forward_with_intermediates`:

```python
layer = HelixConv2d(in_channels=1, units=8, out_channels=16, kernel_size=5, padding=2)
x = torch.randn(4, 1, 28, 28)
y, intermediates = layer.forward_with_intermediates(x)
assert y.shape == (4, 16, 28, 28)
assert intermediates["a"].shape == (4, 8, 28, 28)
assert intermediates["b"].shape == (4, 8, 28, 28)
assert intermediates["z"].shape == (4, 8, 28, 28)
assert torch.isfinite(y).all()
assert torch.isfinite(intermediates["r"]).all()
```

### 22.2 Data Tests

Run with:

```bash
python rot_mnist_test_all.py --data
```

```text
test_mnist_loads
test_train_transform_actually_rotates
test_unrotated_transform_does_not_rotate
test_split_sizes
test_split_deterministic
test_fixed_rotation_grid_shapes
test_fixed_rotation_grid_first_image_unrotated
```

For `test_train_transform_actually_rotates`: load the same image twice through the train transform with different random seeds, and assert the resulting tensors differ.

### 22.3 Slow Tests

Run with:

```bash
python rot_mnist_test_all.py --slow
```

Tiny overfit tests:

```text
test_overfit_tiny_batch_standard
test_overfit_tiny_batch_circle
test_overfit_tiny_batch_helix
```

Use a small fixed batch (128 examples). All three models should reach >= 90% training accuracy in ~300 steps. If `helix_conv` cannot overfit a tiny batch, debug optimization before running full experiments.

Common failure modes:

```text
NaNs from r ≈ 0  (increase eps, check init scale)
gradient explosion  (reduce learning rate)
dead units  (check init of conv_u, conv_v)
```

### 22.4 Analysis Pipeline Tests

```text
test_normalized_correlation_self_is_one
test_normalized_correlation_orthogonal_is_zero
test_filter_rotation_90_recovers_original_after_4_steps
test_phi_star_recovers_known_rotation
```

For `test_phi_star_recovers_known_rotation`: construct two filters where `B = rotate(A, 90°)` synthetically. Run the φ* analysis and assert it recovers φ* ≈ 90° with high correlation. This validates the analysis pipeline before running it on real models.

## 23. Acceptance Criteria

The implementation is complete when:

1. `python rot_mnist_test_all.py` passes.
2. `python rot_mnist_test_all.py --data` passes.
3. `python rot_mnist_run_experiment.py --quick --model-type helix_conv` runs.
4. `python rot_mnist_run_experiment.py --quick --all-models` runs.
5. Full small-scale run completes for all four models.
6. Filter analysis produces `phi_star_histogram.png` and `filter_pair_grid.png` for circle and helix runs.
7. Trajectory analysis produces `trajectory_grid_layer0.png` for the helix run.
8. Test accuracy on both rotated and un-rotated test sets is recorded.
9. Comparison JSON is saved.
10. Results document is written cautiously.

## 24. Recommended Run Order

```bash
python rot_mnist_test_all.py
python rot_mnist_test_all.py --data

python rot_mnist_run_experiment.py --quick --model-type helix_conv
python rot_mnist_run_experiment.py --quick --all-models

python rot_mnist_run_experiment.py --all-models --scale small \
    --run-filter-analysis --run-trajectory-analysis

# Look at the figures. Decide whether to continue.

python rot_mnist_run_experiment.py --all-models --scale medium \
    --run-filter-analysis --run-trajectory-analysis

# Only if filter and trajectory analyses showed self-organization:
python rot_mnist_run_experiment.py --model-type helix_conv --scale small \
    --run-intervention-analysis
```

The "look at the figures" step is the most important one. The filter histogram and trajectory plots are the result of the experiment. If they show no structure, the medium-scale run and intervention analysis can be skipped.

## 25. Results Writeup Plan

File:

```text
rot_mnist_results.md
```

Include:

```text
overview
dataset and rotation augmentation
model variants
training summary table (accuracy, params, epoch time)
rotated vs un-rotated test accuracy
filter analysis
  phi_star histogram
  correlation histogram
  filter pair visualization grid
  qualitative description of what was found
trajectory analysis
  trajectory grid for first helix layer
  per-unit circularity / radius-variance / z-variance summary
  qualitative description of what was found
optional: intervention analysis
scope of claim
next steps
```

The writeup should explicitly state which of the three outcomes from the experiment description occurred:

```text
optimistic   (clean self-organization)
middle       (some units organize, most don't)
pessimistic  (no structure found)
```

Each is informative and should be reported honestly.

## 26. Core Design Summary

This experiment tests whether HelixConv self-organizes to track input orientation when trained on rotated MNIST.

The key design choices are:

```text
W_u, W_v, W_w are independent — no enforced equivariance
kernel size 5 — large enough for filter rotation to be visually clear
forward_with_intermediates exposed — analysis needs (a, b, z) at every position
fixed-rotation-grid test set — required for trajectory analysis
small scale by default — analysis figures are easier to read
```

The central output of the experiment is not a number. It is two figures:

```text
phi_star_histogram.png       — did the filters self-organize?
trajectory_grid_layer0.png   — does (a, b) trace a circle as input rotates?
```

The test accuracy comparison is a sanity check.

In plain terms:

```text
Rotate the digit through every angle.
Plot where the helix layer's phase lands.
If the points trace a circle, the layer learned rotation structure on its own.
If they form a blob, it was just fitting a nonlinear function.
```
