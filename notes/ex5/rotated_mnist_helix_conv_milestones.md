# Milestone Plan: Rotated MNIST HelixConv Experiment

## Purpose

This document breaks the rotated MNIST HelixConv technical design into implementation milestones.

The goal is to test whether HelixConv self-organizes to track input orientation when trained on a rotation-augmented classification task.

The central question is:

```text
Do (W_u, W_v) filter pairs become rotated quadrature pairs,
and does (a, b) trace a circle as the input rotates?
```

The central deliverables are not numbers. They are figures:

```text
phi_star_histogram.png       — did the filters self-organize?
trajectory_grid_layer0.png   — does (a, b) trace a circle as input rotates?
```

The accuracy comparison is a sanity check, not the result.

## Target Workflow

Final local commands should look like:

```bash
python rot_mnist_test_all.py
python rot_mnist_test_all.py --data
python rot_mnist_test_all.py --slow

python rot_mnist_run_experiment.py --quick --model-type helix_conv
python rot_mnist_run_experiment.py --quick --all-models

python rot_mnist_run_experiment.py --all-models --scale small \
    --run-filter-analysis --run-trajectory-analysis

python rot_mnist_run_experiment.py --all-models --scale medium \
    --run-filter-analysis --run-trajectory-analysis

python rot_mnist_run_experiment.py --model-type helix_conv --scale small \
    --run-intervention-analysis
```

Generated artifacts should be local:

```text
rot_mnist_data/
rot_mnist_results/
rot_mnist_checkpoints/
rot_mnist_figures/
```

Do not turn the repo into a package.

---

## Milestone 0: Add Local Rotated MNIST Files

### Goal

Create the local file structure.

### Files to Add

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

### Deliverables

Minimal placeholder files with imports and docstrings.

### Acceptance Checks

```bash
python -c "import rot_mnist_config, rot_mnist_data, rot_mnist_models, rot_mnist_train"
```

should run without import errors.

### Definition of Done

The rotated MNIST experiment skeleton exists and does not disturb prior experiments.

---

## Milestone 1: Configuration and Scale Presets

### Goal

Implement the config object and scale presets.

### File

```text
rot_mnist_config.py
```

### Deliverables

Implement:

```python
RotMNISTConfig
SCALE_PRESETS
apply_scale_preset
get_device
save_json
set_seed
```

Default config:

```text
input_channels = 1
num_classes = 10
batch_size = 128
epochs = 30
learning_rate = 1e-3
weight_decay = 1e-4
dropout = 0.0
kernel_size = 5
num_conv_blocks = 2
rotation_max_degrees = 180
```

Scale presets:

```text
small:  hidden_channels=32,  circle_units=16, helix_units=16, matched_hidden_channels=48
medium: hidden_channels=64,  circle_units=32, helix_units=32, matched_hidden_channels=96
large:  hidden_channels=128, circle_units=64, helix_units=64, matched_hidden_channels=192
```

### Tests

Add:

```text
test_config_defaults
test_config_to_dict
test_apply_scale_preset_small
test_apply_scale_preset_medium
test_apply_scale_preset_large
test_get_device_falls_back_to_cpu
test_default_kernel_size_is_5
```

### Acceptance Checks

```bash
python rot_mnist_test_all.py
```

passes config tests.

### Definition of Done

Configs can be created, scaled, serialized, and used to select device. Default scale is small because analysis figures are easier to read at small unit counts.

---

## Milestone 2: Implement Standard CNN

### Goal

Build the dense baseline first.

### File

```text
rot_mnist_models.py
```

### Deliverables

Implement:

```python
count_parameters
StandardCNN
build_rot_mnist_model
```

Support:

```text
standard_cnn
standard_cnn_matched
```

Architecture for `num_conv_blocks=2`:

```text
Conv2d(1, hidden_channels, k, padding=k//2)
GELU
MaxPool2d(2)
Conv2d(hidden_channels, hidden_channels, k, padding=k//2)
GELU
MaxPool2d(2)
Flatten
Linear(hidden_channels * 7 * 7, 10)
```

### Tests

Add:

```text
test_standard_cnn_forward_shape
test_standard_cnn_no_nans
test_standard_cnn_backward_pass
test_count_parameters_positive
test_build_standard_cnn
test_build_standard_cnn_matched
```

Expected shape:

```text
input:  [batch, 1, 28, 28]
output: [batch, 10]
```

### Acceptance Checks

```bash
python rot_mnist_test_all.py
```

passes dense model tests.

### Definition of Done

Standard CNN can run forward, backpropagate, and report parameter counts.

---

## Milestone 3: Implement CircleConv2d and CircleCNN

### Goal

Add the circle convolutional model.

### File

```text
rot_mnist_models.py
```

### Deliverables

Implement:

```python
CircleConv2d
CircleCNN
```

Feature set per unit:

```text
sin(theta)
cos(theta)
r
r * sin(theta)
r * cos(theta)
```

Three independent learnable filter banks per unit:

```text
conv_u
conv_v
```

Update factory support:

```text
circle_conv
```

**Critical design constraint:** `conv_u` and `conv_v` are independent. They are not constrained to be 90°-rotated copies of each other. Tying them invalidates the self-organization claim.

### Tests

Add:

```text
test_circle_conv_forward_shape
test_circle_conv_no_nans
test_circle_conv_backward_pass
test_circle_conv_filters_independent_after_init
test_circle_cnn_forward_shape
test_circle_cnn_no_nans
test_circle_cnn_backward_pass
test_build_circle_conv
```

For `test_circle_conv_filters_independent_after_init`:

```text
Construct a fresh CircleConv2d.
Compute correlation between conv_u.weight and rotate(conv_v.weight, 90°).
Assert correlation is below some loose threshold (e.g., < 0.5)
to confirm filters are not initialized as a tied pair.
```

This guards against accidentally introducing weight tying.

### Acceptance Checks

```bash
python rot_mnist_test_all.py
```

passes Circle tests.

### Definition of Done

CircleConv2d and CircleCNN are stable on image-shaped tensors.

---

## Milestone 4: Implement HelixConv2d and HelixCNN

### Goal

Add the helix convolutional model.

### File

```text
rot_mnist_models.py
```

### Deliverables

Implement:

```python
HelixConv2d
HelixCNN
```

Feature set per unit:

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

Three independent learnable filter banks per unit:

```text
conv_u
conv_v
conv_w
```

Update factory support:

```text
helix_conv
```

Implement `forward_with_intermediates`:

```python
def forward_with_intermediates(self, x):
    ...
    return out, {"a": a, "b": b, "z": z, "r": r}
```

This is required for trajectory analysis.

Do not use `atan2` in forward pass.

Use:

```text
eps = 1e-6
```

### Tests

Add:

```text
test_helix_conv_forward_shape
test_helix_conv_no_nans
test_helix_conv_backward_pass
test_helix_conv_forward_with_intermediates
test_helix_conv_filters_independent_after_init
test_helix_cnn_forward_shape
test_helix_cnn_no_nans
test_helix_cnn_backward_pass
test_build_helix_conv
test_all_models_random_stress_no_nans
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
assert torch.isfinite(intermediates["r"]).all()
```

Use input scales:

```text
0.01
1.0
10.0
```

### Acceptance Checks

```bash
python rot_mnist_test_all.py
```

passes Helix tests.

### Definition of Done

HelixConv2d and HelixCNN are numerically stable, expose intermediate activations, and have independent filter banks.

---

## Milestone 5: Data Loading and Rotation Augmentation

### Goal

Load MNIST with three distinct transform pipelines: train (rotated), rotated test, un-rotated test.

### File

```text
rot_mnist_data.py
```

### Deliverables

Implement:

```python
make_train_transform
make_rotated_test_transform
make_unrotated_test_transform
make_rot_mnist_dataloaders
make_fixed_rotation_grid
```

Three transforms:

```text
train_transform           = RandomRotation + ToTensor + Normalize
rotated_test_transform    = RandomRotation + ToTensor + Normalize
unrotated_test_transform  = ToTensor + Normalize
```

Standard MNIST split:

```text
train: 55,000 (random rotation)
val:    5,000 (random rotation; split from train)
test:  10,000 (rotated AND un-rotated copies)
```

Dataloaders return:

```text
train
val
test_rotated
test_unrotated
```

`make_fixed_rotation_grid` builds a tensor of one image at 72 evenly-spaced rotations through 360°.

### Tests

Run with:

```bash
python rot_mnist_test_all.py --data
```

Add:

```text
test_mnist_loads
test_train_transform_actually_rotates
test_unrotated_transform_does_not_rotate
test_split_sizes
test_split_deterministic
test_batch_shapes
test_fixed_rotation_grid_shapes
test_fixed_rotation_grid_first_image_unrotated
test_fixed_rotation_grid_normalization_after_rotation
```

For `test_train_transform_actually_rotates`:

```text
Load the same image twice through the train transform with different seeds.
Assert the resulting tensors differ.
```

### Acceptance Checks

```bash
python rot_mnist_test_all.py --data
```

passes.

### Definition of Done

MNIST loads correctly, three transforms work as intended, and the fixed-rotation grid is available for trajectory analysis.

---

## Milestone 6: Training and Evaluation Loops

### Goal

Implement reusable training code.

### File

```text
rot_mnist_train.py
```

### Deliverables

Implement:

```python
train_one_epoch
evaluate
fit_rot_mnist
```

Use:

```text
cross entropy loss
AdamW optimizer
best checkpoint by validation accuracy
```

Track:

```text
train_loss
train_accuracy
train_seconds
train_examples_per_second
val_loss
val_accuracy
```

Test metrics evaluate on **both** test sets:

```text
test_rotated_loss
test_rotated_accuracy
test_unrotated_loss
test_unrotated_accuracy
```

Save:

```text
metrics.json
history.json
training_history.png
checkpoint
```

### Tests

Use synthetic image-shaped data.

Add:

```text
test_train_one_epoch_synthetic
test_evaluate_synthetic
test_fit_rot_mnist_synthetic_quick
```

### Acceptance Checks

```bash
python rot_mnist_test_all.py
```

passes synthetic training tests.

### Definition of Done

Training works on synthetic data and saves both rotated-test and un-rotated-test metrics.

---

## Milestone 7: Filter Analysis Pipeline

### Goal

Implement and validate the filter-pair self-organization analysis.

This is the central analysis of the experiment.

### File

```text
rot_mnist_analyze_filters.py
```

### Deliverables

Implement:

```python
normalized_correlation
rotate_filter
sweep_phi_star_for_unit
analyze_first_helix_layer_filters
analyze_first_circle_layer_filters
plot_filter_pair_grid
plot_phi_star_histogram
plot_correlation_histogram
```

Sweep:

```text
phi in {0°, 5°, 10°, ..., 355°}
```

For each unit, find:

```text
phi_star = argmax_phi normalized_correlation(W_u, rotate(W_v, phi))
corr_star = max_phi    normalized_correlation(W_u, rotate(W_v, phi))
```

Generate three figures:

```text
filter_pair_grid.png
phi_star_histogram.png
correlation_histogram.png
```

### Critical Tests

The analysis pipeline itself must be validated before being applied to a trained model. Add:

```text
test_normalized_correlation_self_is_one
test_normalized_correlation_orthogonal_is_zero
test_filter_rotation_90_recovers_original_after_4_steps
test_phi_star_recovers_known_rotation
test_phi_star_handles_sign_flip_correctly
```

For `test_phi_star_recovers_known_rotation`:

```text
Construct a 5x5 filter A with a clear oriented pattern.
Construct B = rotate(A, 90°) synthetically.
Run the phi_star analysis.
Assert phi_star is in [85°, 95°] and corr_star > 0.9.
```

For `test_phi_star_handles_sign_flip_correctly`:

```text
Construct A and B = -A.
Confirm phi_star reports either 0° or 180° (not 90°).
This guards against duplicates being mistakenly labeled as rotations.
```

If these tests do not pass, the histogram is meaningless regardless of the trained model.

### Acceptance Checks

```bash
python rot_mnist_test_all.py
```

passes filter-analysis pipeline tests.

### Definition of Done

The filter analysis recovers known synthetic rotations correctly and produces all three figures.

---

## Milestone 8: Trajectory Analysis Pipeline

### Goal

Implement the (a, b) trajectory measurement under input rotation.

### File

```text
rot_mnist_analyze_trajectory.py
```

### Deliverables

Implement:

```python
trajectory_for_unit
trajectory_for_layer
plot_trajectory_grid
plot_trajectory_summary
compute_circularity_score
compute_radius_variance
compute_z_variance
compute_winding_number
```

Procedure:

```text
1. Generate fixed-rotation grid for one base image.
2. Run forward through model up to chosen helix layer.
3. Capture (a, b, z, r) at chosen spatial position for each unit.
4. Plot (a, b) trajectory in 2D.
```

Per-unit summary metrics:

```text
circularity score   (minor/major axis ratio of best-fit ellipse)
radius variance     (low = r is rotation-invariant)
z variance          (low = axis is rotation-invariant)
winding number      (full circles in 360° rotation)
```

Generate:

```text
trajectory_grid_layer0.png
trajectory_grid_layer1.png
trajectory_summary_layer0.png
trajectory.json
```

### Tests

Add:

```text
test_trajectory_shapes
test_circularity_score_perfect_circle_is_one
test_circularity_score_line_is_zero
test_radius_variance_constant_signal_is_zero
test_winding_number_one_full_loop
```

### Acceptance Checks

```bash
python rot_mnist_test_all.py
```

passes trajectory analysis tests.

### Definition of Done

Trajectory analysis runs end-to-end on a small trained model and produces interpretable figures.

---

## Milestone 9: Causal Intervention Analysis (Optional)

### Goal

Implement the strongest version of the test: rotate (a, b) at a hidden layer and check whether the output matches running the model on a rotated input.

### File

```text
rot_mnist_analyze_intervention.py
```

### When to Run

Only run this milestone if Milestones 7 and 8 already showed self-organization. If filter pairs look random and trajectories are blobs, intervention will not produce coherent results.

### Deliverables

Implement:

```python
intervene_at_layer
intervention_match_accuracy
sweep_intervention_angles
```

Sweep:

```text
delta_theta in {15°, 30°, 60°, 90°, 180°}
multiple test images
multiple seeds (if available)
```

Generate:

```text
intervention.json
intervention_summary.png
```

### Tests

Add:

```text
test_intervene_at_layer_zero_angle_is_identity
test_intervene_at_layer_360_is_identity
test_intervention_match_accuracy_shape
```

### Definition of Done

Causal intervention analysis exists and is documented as an optional milestone.

---

## Milestone 10: Experiment Runner CLI

### Goal

Create the command-line runner.

### File

```text
rot_mnist_run_experiment.py
```

### Deliverables

Support CLI args:

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

Implement:

```python
run_single(config)
```

For `--all-models`, run:

```text
standard_cnn
standard_cnn_matched
circle_conv
helix_conv
```

For `--sweep-scales`, run:

```text
small
medium
large
```

### Quick Mode

Set:

```text
epochs = 1
scale = small
limit_train_batches = 50
limit_eval_batches = 20
```

### Acceptance Checks

```bash
python rot_mnist_run_experiment.py --quick --model-type helix_conv
python rot_mnist_run_experiment.py --quick --all-models
```

both run.

### Definition of Done

Experiments can be launched from the CLI, including optional analysis flags.

---

## Milestone 11: Quick Mode Validation

### Goal

Verify the full pipeline end-to-end.

### Commands

```bash
python rot_mnist_run_experiment.py --quick --model-type standard_cnn
python rot_mnist_run_experiment.py --quick --model-type circle_conv
python rot_mnist_run_experiment.py --quick --model-type helix_conv
python rot_mnist_run_experiment.py --quick --all-models
```

### Expected Behavior

Quick mode should:

```text
load data
train for 1 epoch
evaluate on rotated and un-rotated test sets
save metrics
save history
save checkpoint
print summary
```

### Definition of Done

All models complete quick mode without NaNs or artifact failures.

---

## Milestone 12: Tiny-Batch Overfit Tests

### Goal

Verify that each model can fit a small fixed batch.

### File

```text
rot_mnist_test_all.py
```

### Deliverables

Slow tests:

```text
test_overfit_tiny_batch_standard
test_overfit_tiny_batch_circle
test_overfit_tiny_batch_helix
```

Run with:

```bash
python rot_mnist_test_all.py --slow
```

Suggested thresholds:

```text
standard_cnn >= 90%
circle_conv >= 85%
helix_conv >= 85%
```

If `helix_conv` cannot overfit a tiny batch, do not run full experiments. Common failure modes:

```text
NaNs from r ≈ 0  (increase eps, check init scale)
gradient explosion  (reduce learning rate)
dead units  (check init of conv_u, conv_v)
```

### Definition of Done

All model families can overfit a small batch or failures are documented and resolved.

---

## Milestone 13: Small-Scale Full Run with Filter Analysis

### Goal

Run the first reportable rotated MNIST comparison and produce the central analysis figures.

This is the milestone that actually answers the experiment's question.

### Command

```bash
python rot_mnist_run_experiment.py --all-models --scale small \
    --run-filter-analysis --run-trajectory-analysis
```

### Deliverables

Per-run directories:

```text
rot_mnist_results/standard_cnn_small_seed0/
rot_mnist_results/standard_cnn_matched_small_seed0/
rot_mnist_results/circle_conv_small_seed0/
rot_mnist_results/helix_conv_small_seed0/
```

Each should include:

```text
metrics.json
history.json
training_history.png
```

For circle and helix runs, additionally:

```text
filter_pair_grid.png
phi_star_histogram.png
correlation_histogram.png
trajectory_grid_layer0.png
trajectory_grid_layer1.png
trajectory_summary_layer0.png
trajectory.json
```

Comparison:

```text
rot_mnist_results/comparison_small_seed0.json
```

### Decision Gate

This is the first opportunity to look at the central figures. Possible outcomes:

```text
optimistic   — phi_star peaks at 90°, trajectories are circles
middle       — some units organize, most don't
pessimistic  — phi_star is uniform, trajectories are blobs
```

The decision to continue to medium scale depends on what the figures show.

### Definition of Done

A complete small-scale comparison and analysis exists. The central question has a preliminary answer.

---

## Milestone 14: Medium-Scale Full Run with Analyses

### Goal

Confirm or refine the small-scale finding.

### Command

```bash
python rot_mnist_run_experiment.py --all-models --scale medium \
    --run-filter-analysis --run-trajectory-analysis
```

### Deliverables

```text
rot_mnist_results/comparison_medium_seed0.json
```

Plus all filter and trajectory figures for circle and helix runs.

### Notes

Medium scale has more units, which makes the histogram analysis more statistically meaningful but the filter-pair grid harder to read. Both scales should be reported.

### Definition of Done

A complete medium-scale comparison and analysis exists.

---

## Milestone 15: Rotated vs Un-rotated Test Accuracy Analysis

### Goal

Quantify how well each model learned rotation invariance.

### Deliverables

For each model, record:

```text
test_rotated_accuracy
test_unrotated_accuracy
gap = test_rotated_accuracy - test_unrotated_accuracy
```

Generate:

```text
rot_mnist_results/rotated_vs_unrotated_accuracy.png
```

A model that learned rotation invariance should sit near the diagonal. A model that ignored rotation will be much worse on rotated than un-rotated.

### Interpretation

```text
small gap, both high    — model learned rotation invariance
large gap, rotated low  — model did not generalize across rotations
small gap, both low     — model failed; investigate training
```

### Definition of Done

The rotated-vs-un-rotated comparison is reported for all four model families.

---

## Milestone 16: Causal Intervention Run (Optional)

### Goal

Run the intervention analysis if Milestones 13 and 14 showed self-organization.

### Command

```bash
python rot_mnist_run_experiment.py --model-type helix_conv --scale small \
    --run-intervention-analysis
```

### Decision Rule

Run this milestone only if:

```text
phi_star histogram peaks near 90° with high correlation
AND
trajectory grid shows circular patterns for at least some units
```

If both Milestones 13 and 14 showed pessimistic results, skip this milestone and document why.

### Definition of Done

Intervention analysis is either completed or formally skipped with reason.

---

## Milestone 17: Large-Scale Full Run (Optional)

### Goal

Check whether self-organization improves or degrades with more capacity.

### Command

```bash
python rot_mnist_run_experiment.py --all-models --scale large \
    --run-filter-analysis --run-trajectory-analysis
```

### Notes

Large scale produces the most parameters but the noisiest analysis figures. The filter histogram is the most useful summary at this scale; the per-unit grid becomes hard to read.

### Definition of Done

A complete large-scale comparison exists or is deferred for compute reasons.

---

## Milestone 18: First Results Writeup

### Goal

Write the first results document.

### File

```text
rot_mnist_results.md
```

### Include

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
  qualitative description
trajectory analysis
  trajectory grid for first helix layer
  per-unit circularity / radius-variance / z-variance summary
  qualitative description
optional: intervention analysis
scope of claim
next steps
```

The writeup must explicitly state which outcome occurred:

```text
optimistic   (clean self-organization)
middle       (some units organize, most don't)
pessimistic  (no structure found)
```

Each is informative. Report honestly.

### Definition of Done

A cautious results document exists and matches saved JSON artifacts and figures.

---

## Milestone 19: Multi-Seed Confirmation

### Goal

Check whether the result is stable across seeds.

### Commands

At least:

```bash
python rot_mnist_run_experiment.py --all-models --scale small --seed 0 \
    --run-filter-analysis --run-trajectory-analysis
python rot_mnist_run_experiment.py --all-models --scale small --seed 1 \
    --run-filter-analysis --run-trajectory-analysis
python rot_mnist_run_experiment.py --all-models --scale small --seed 2 \
    --run-filter-analysis --run-trajectory-analysis
```

Prefer:

```text
seeds = 0, 1, 2, 3, 4
```

### Deliverables

```text
rot_mnist_results/multiseed_small_comparison.json
rot_mnist_results/multiseed_phi_star_aggregate.png
```

Report:

```text
mean test accuracy (rotated)
std test accuracy (rotated)
mean test accuracy (un-rotated)
fraction of units with phi_star in [80°, 100°]
mean corr_star across units
```

The "fraction of units near 90°" metric is what to track across seeds. If it's reliably high, self-organization is real. If it fluctuates wildly, the small-scale result was lucky.

### Definition of Done

Any self-organization claim is backed by multiple seeds.

---

## Milestone 20: Final Decision Gate

### Goal

Decide what the rotated MNIST result means for the research program.

### Decision Rules

If filter pairs cleanly self-organize and trajectories trace circles:

```text
Strong evidence that helix layers self-organize on data with input symmetry.
This is the result that retroactively backs the project's original framing.
Next: try another naturally-rotated dataset (galaxy, satellite imagery)
to test whether the result transfers.
```

If some units organize but most don't:

```text
Partial self-organization. The layer can find geometric structure
when useful but doesn't waste capacity on it otherwise.
Investigate: what distinguishes organized from unorganized units?
```

If filter pairs look random and trajectories are blobs:

```text
Helix layers do not self-organize even when input symmetry is built in.
The geometric primitives function as a generic nonlinearity.
This is consistent with the parameter-efficiency findings on Covertype.
The project's original framing requires revision.
```

If models fail to train:

```text
Check rotation augmentation, kernel size, init scale, eps.
Filter analysis on a non-trained model is meaningless.
```

### Definition of Done

The next experiment direction is documented based on the figures, not the accuracy numbers.

---

## Final Acceptance Criteria

The rotated MNIST experiment v1 is complete when:

1. Fast tests pass.
2. Data tests pass.
3. Analysis pipeline tests pass (especially `test_phi_star_recovers_known_rotation`).
4. Quick mode works for all models.
5. Tiny-batch overfit tests pass.
6. Small-scale comparison and analyses exist.
7. Filter histograms are produced for circle and helix runs.
8. Trajectory grids are produced for the helix run.
9. Rotated vs un-rotated test accuracies are recorded.
10. Parameter counts are reported.
11. Results document is written cautiously and states which outcome occurred.
12. Multi-seed follow-up is planned or completed.

## Summary

Train on rotation-augmented MNIST.

Look at the filter pairs.

If they line up at right angles, the layer learned rotation structure on its own.

If they look like random noise, it was just fitting a nonlinear function.

Either answer is honest.
