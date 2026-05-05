# Milestone Plan: Covertype Classification Experiment

## Purpose

This document breaks the Covertype classification technical design into implementation milestones.

The goal is to test dense, Circle, and Helix MLPs on heterogeneous tabular classification.

The central comparison is:

```text
Circle/Helix MLP vs parameter-matched dense MLP
```

The central metrics are:

```text
test accuracy
macro F1
test loss
mean epoch time
```

## Target Workflow

Final local commands should look like:

```bash
python covertype_test_all.py
python covertype_test_all.py --data
python covertype_test_all.py --slow

python covertype_run_experiment.py --quick --model-type helix_mlp
python covertype_run_experiment.py --quick --all-models

python covertype_run_experiment.py --all-models --scale small
python covertype_run_experiment.py --all-models --scale medium
python covertype_run_experiment.py --all-models --scale large

python covertype_run_experiment.py --all-models --sweep-scales
```

Generated artifacts should be local:

```text
covertype_data/
covertype_results/
covertype_checkpoints/
```

Do not turn the repo into a package.

---

## Milestone 0: Add Local Covertype Files

### Goal

Create the local file structure.

### Files to Add

```text
covertype_config.py
covertype_data.py
covertype_models.py
covertype_train.py
covertype_run_experiment.py
covertype_test_all.py
covertype_results.md
```

### Deliverables

Minimal placeholder files with imports and docstrings.

### Acceptance Checks

```bash
python -c "import covertype_config, covertype_data, covertype_models, covertype_train"
```

should run without import errors.

### Definition of Done

The Covertype experiment skeleton exists and does not disturb prior experiments.

---

## Milestone 1: Configuration and Scale Presets

### Goal

Implement the config object and scale presets.

### File

```text
covertype_config.py
```

### Deliverables

Implement:

```python
CovertypeConfig
SCALE_PRESETS
apply_scale_preset
get_device
save_json
set_seed
```

Default config:

```text
input_dim = 54
num_classes = 7
batch_size = 1024
epochs = 100
learning_rate = 1e-3
weight_decay = 1e-4
dropout = 0.05
```

Scale presets:

```text
small:  hidden_dim=64,  circle_units=32,  helix_units=32,  matched_hidden_dim=96
medium: hidden_dim=128, circle_units=64,  helix_units=64,  matched_hidden_dim=192
large:  hidden_dim=256, circle_units=128, helix_units=128, matched_hidden_dim=384
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
```

### Acceptance Checks

```bash
python covertype_test_all.py
```

passes config tests.

### Definition of Done

Configs can be created, scaled, serialized, and used to select device.

---

## Milestone 2: Implement Dense MLP

### Goal

Build the dense baseline first.

### File

```text
covertype_models.py
```

### Deliverables

Implement:

```python
count_parameters
StandardMLP
build_covertype_model
```

Support:

```text
standard_mlp
standard_mlp_matched
```

Architecture:

```text
Linear(54, hidden_dim)
GELU
Dropout
Linear(hidden_dim, hidden_dim)
GELU
Dropout
Linear(hidden_dim, 7)
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

Expected shape:

```text
input:  [batch, 54]
output: [batch, 7]
```

### Acceptance Checks

```bash
python covertype_test_all.py
```

passes dense model tests.

### Definition of Done

Dense Covertype MLPs can run forward, backpropagate, and report parameter counts.

---

## Milestone 3: Implement CircleLayer and CircleMLP

### Goal

Add the circle geometric model.

### File

```text
covertype_models.py
```

### Deliverables

Implement:

```python
CircleLayer
CircleMLP
```

Feature set:

```text
sin(theta)
cos(theta)
r
r * sin(theta)
r * cos(theta)
```

Update factory support:

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
python covertype_test_all.py
```

passes Circle tests.

### Definition of Done

CircleLayer and CircleMLP are stable on tabular-shaped tensors.

---

## Milestone 4: Implement HelixLayer and HelixMLP

### Goal

Add the helix geometric model.

### File

```text
covertype_models.py
```

### Deliverables

Implement:

```python
HelixLayer
HelixMLP
```

Feature set:

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

Update factory support:

```text
helix_mlp
```

Do not use `atan2` in forward pass.

Use:

```text
eps = 1e-6
```

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
python covertype_test_all.py
```

passes Helix tests.

### Definition of Done

HelixLayer and HelixMLP are numerically stable and trainable on synthetic tabular inputs.

---

## Milestone 5: Data Loading and Preprocessing

### Goal

Load Covertype, split it, and preprocess features without leakage.

### File

```text
covertype_data.py
```

### Deliverables

Implement:

```python
load_covertype_arrays
make_train_val_test_split
preprocess_covertype_features
TabularTensorDataset
make_covertype_dataloaders
```

Use stratified splits:

```text
train: 70%
val:   15%
test:  15%
```

Preprocessing:

```text
standardize first 10 continuous features using train statistics
leave remaining 44 binary features unchanged
convert labels from 1..7 to 0..6
```

### Tests

Run with:

```bash
python covertype_test_all.py --data
```

Add:

```text
test_covertype_loads
test_feature_shape_54
test_num_classes_7
test_split_sizes
test_split_deterministic
test_continuous_features_standardized
test_binary_features_remain_binary
test_batch_shapes
```

### Acceptance Checks

```bash
python covertype_test_all.py --data
```

passes.

### Definition of Done

Covertype loads, splits, preprocesses correctly, and produces dataloaders.

---

## Milestone 6: Metrics

### Goal

Implement tabular classification metrics.

### File

```text
covertype_train.py
```

### Deliverables

Implement helpers for:

```text
accuracy
macro F1
weighted F1
per-class accuracy
confusion matrix
```

Use sklearn if available:

```text
f1_score
confusion_matrix
```

### Tests

Add:

```text
test_accuracy_helper
test_macro_f1_helper
test_weighted_f1_helper
test_confusion_matrix_shape
test_per_class_accuracy_shape
```

### Acceptance Checks

```bash
python covertype_test_all.py
```

passes metric tests.

### Definition of Done

Evaluation can report accuracy, macro F1, weighted F1, per-class accuracy, and confusion matrix.

---

## Milestone 7: Training and Evaluation Loops

### Goal

Implement reusable training code.

### File

```text
covertype_train.py
```

### Deliverables

Implement:

```python
train_one_epoch
evaluate
fit_covertype
```

Use:

```text
cross entropy loss
AdamW optimizer
best checkpoint by validation macro F1
accuracy as tie-breaker
loss as second tie-breaker
```

Track:

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

Save:

```text
metrics.json
history.json
confusion_matrix.json
checkpoint
```

### Tests

Use synthetic tabular data.

Add:

```text
test_train_one_epoch_synthetic
test_evaluate_synthetic
test_fit_covertype_synthetic_quick
```

### Acceptance Checks

```bash
python covertype_test_all.py
```

passes synthetic training tests.

### Definition of Done

Training and evaluation work on synthetic data and save artifacts.

---

## Milestone 8: Experiment Runner CLI

### Goal

Create the command-line runner.

### File

```text
covertype_run_experiment.py
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

Implement:

```python
run_single(config)
```

For `--all-models`, run:

```text
standard_mlp
standard_mlp_matched
circle_mlp
helix_mlp
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
python covertype_run_experiment.py --quick --model-type helix_mlp
python covertype_run_experiment.py --quick --all-models
```

both run.

### Definition of Done

Experiments can be launched from the CLI.

---

## Milestone 9: Quick Mode Validation

### Goal

Verify the full pipeline end-to-end.

### Commands

```bash
python covertype_run_experiment.py --quick --model-type standard_mlp
python covertype_run_experiment.py --quick --model-type circle_mlp
python covertype_run_experiment.py --quick --model-type helix_mlp
python covertype_run_experiment.py --quick --all-models
```

### Expected Behavior

Quick mode should:

```text
load data
train for 1 epoch
evaluate
save metrics
save history
save checkpoint
print summary
```

### Definition of Done

All models complete quick mode without NaNs or artifact failures.

---

## Milestone 10: Tiny-Batch Overfit Tests

### Goal

Verify that each model can fit a small fixed tabular batch.

### File

```text
covertype_test_all.py
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
python covertype_test_all.py --slow
```

Suggested thresholds:

```text
standard_mlp >= 90%
circle_mlp >= 85%
helix_mlp >= 85%
```

### Definition of Done

All model families can overfit a small batch or failures are documented.

---

## Milestone 11: Small-Scale Full Run

### Goal

Run the first reportable Covertype comparison.

### Command

```bash
python covertype_run_experiment.py --all-models --scale small
```

### Deliverables

Per-run directories:

```text
covertype_results/standard_mlp_small_seed0/
covertype_results/standard_mlp_matched_small_seed0/
covertype_results/circle_mlp_small_seed0/
covertype_results/helix_mlp_small_seed0/
```

Each should include:

```text
metrics.json
history.json
confusion_matrix.json
training_history.png
```

Comparison:

```text
covertype_results/comparison_small_seed0.json
```

### Definition of Done

A complete small-scale comparison exists.

---

## Milestone 12: Medium-Scale Full Run

### Goal

Run the main default comparison.

### Command

```bash
python covertype_run_experiment.py --all-models --scale medium
```

### Deliverables

```text
covertype_results/comparison_medium_seed0.json
```

### Definition of Done

A complete medium-scale comparison exists.

---

## Milestone 13: Large-Scale Full Run

### Goal

Check whether behavior changes with more capacity.

### Command

```bash
python covertype_run_experiment.py --all-models --scale large
```

### Deliverables

```text
covertype_results/comparison_large_seed0.json
```

### Definition of Done

A complete large-scale comparison exists or is deferred for compute reasons.

---

## Milestone 14: Scale Sweep

### Goal

Compare accuracy and macro F1 against parameter count.

### Command

```bash
python covertype_run_experiment.py --all-models --sweep-scales
```

### Deliverables

```text
covertype_results/comparison_all_scales_seed0.json
covertype_results/accuracy_vs_params.png
covertype_results/macro_f1_vs_params.png
```

### Definition of Done

A scale sweep exists across all model families.

---

## Milestone 15: Parameter-Matching Pass

### Goal

Improve fairness of dense comparison.

### Deliverables

Tune:

```text
matched_hidden_dim
```

or implement:

```python
find_param_matched_hidden_dim
```

Target:

```text
standard_mlp_matched within 10% of helix_mlp or circle_mlp parameter count
```

Because Circle and Helix have different parameter counts, it may be useful to report two matched dense baselines later:

```text
standard_mlp_matched_circle
standard_mlp_matched_helix
```

For v1, one matched dense baseline is acceptable if parameter counts are reported clearly.

### Definition of Done

Parameter mismatch is either reduced or documented.

---

## Milestone 16: First Results Writeup

### Goal

Write the first results document.

### File

```text
covertype_results.md
```

### Include

```text
overview
dataset
preprocessing
model variants
summary table
accuracy
macro F1
test loss
training rate
confusion matrix notes
scope of claim
next steps
```

### Definition of Done

A cautious results document exists and matches saved JSON artifacts.

---

## Milestone 17: Multi-Seed Runs

### Goal

Check whether the result is stable.

### Commands

At least:

```bash
python covertype_run_experiment.py --all-models --scale medium --seed 0
python covertype_run_experiment.py --all-models --scale medium --seed 1
python covertype_run_experiment.py --all-models --scale medium --seed 2
```

Prefer:

```text
seeds = 0, 1, 2, 3, 4
```

### Deliverables

```text
covertype_results/multiseed_medium_comparison.json
```

Report:

```text
mean test accuracy
std test accuracy
mean macro F1
std macro F1
mean epoch time
```

### Definition of Done

Any architecture-win claim is backed by multiple seeds.

---

## Milestone 18: Feature Ablations

### Goal

Understand which geometric features matter.

### Helix Ablations

```text
full
no_axis
phase_radius
raw_projection
axis_only
```

### Circle Ablations

```text
full
phase_only
radius_only
raw_projection
```

### Definition of Done

Ablations clarify whether performance comes from normalized geometric features, raw projections, or axis features.

---

## Milestone 19: Optional Tree Baseline

### Goal

Contextualize neural performance on tabular data.

### Deliverables

Optional simple baselines:

```text
RandomForestClassifier
HistGradientBoostingClassifier
```

Do not make these the main comparison.

### Definition of Done

Tree baseline numbers are reported as context, not as the main architecture comparison.

---

## Milestone 20: Final Decision Gate

### Goal

Decide what the Covertype result means for the research program.

### Decision Rules

If Circle/Helix beat dense across seeds:

```text
Strong evidence that geometric layers generalize beyond image-origin data.
```

If Circle beats Helix again:

```text
Phase/radius expansion is likely the main useful primitive; axis utility remains unproven.
```

If Helix beats Circle:

```text
Run axis ablations before claiming helix-specific benefit.
```

If dense wins:

```text
Geometric layers may be less useful on heterogeneous tabular data, or need tabular-specific tuning.
```

If all models are weak:

```text
Check preprocessing, class imbalance, and training setup before interpreting architecture.
```

### Definition of Done

The next experiment direction is documented.

---

## Final Acceptance Criteria

The Covertype experiment v1 is complete when:

1. Fast tests pass.
2. Data tests pass.
3. Quick mode works.
4. Small-scale comparison exists.
5. Medium-scale comparison exists.
6. Metrics include accuracy and macro F1.
7. Confusion matrices are saved.
8. Training rates are reported.
9. Parameter counts are reported.
10. Results document is written cautiously.
11. Multi-seed follow-up is planned or completed.

## Summary

Build the tabular data pipeline.

Check the preprocessing is clean.

Make each model overfit a small batch.

Then compare dense, circle, and helix across scales.

If the circle model wins again, that result is worth understanding.
