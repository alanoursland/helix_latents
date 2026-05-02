# Flattened CIFAR-10 Experiment

## Purpose

This experiment tests whether `CircleLayer` and `HelixLayer` remain viable on a harder, nonlocal image-classification problem.

The previous MNIST experiment showed that Circle MLP and Helix MLP can train and generalize on ordinary digit classification. MNIST is useful, but it is still relatively simple. Flattened CIFAR-10 is a more serious stress test for feedforward architectures because it removes the spatial inductive bias that convolutional models normally use.

The task is:

```text
flattened RGB image ∈ R^(32×32×3) → class ∈ {0, ..., 9}
```

or equivalently:

```text
x ∈ R^3072 → y ∈ {airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck}
```

The goal is not to beat convolutional networks. The goal is to compare helix-native feedforward layers against dense feedforward baselines in a setting where all fully connected models are expected to struggle.

The central question is:

```text
Can Helix MLP stay competitive with dense MLPs on flattened CIFAR-10?
```

## Why Flattened CIFAR-10?

CIFAR-10 with flattened pixels is intentionally harsh. It discards the locality and translation structure that CNNs exploit. A dense MLP has to learn useful nonlocal pixel interactions directly.

That makes this a good architecture stress test.

This experiment does **not** hand the model a cyclic or helical variable. There is no explicit phase target, no rotation angle, and no synthetic geometry. If Helix MLP performs competitively here, that is evidence that the primitive can function beyond tasks designed around geometric structure.

A positive result is not necessarily "Helix MLP beats dense MLP." A positive result can be:

```text
Helix MLP trains stably and comes within a small gap of a parameter-matched dense MLP.
```

A negative result is also informative:

```text
Helix MLP falls far behind dense MLP on nonlocal pixel data.
```

That would reveal a limitation of the architecture.

## Dataset

Use `torchvision.datasets.CIFAR10`.

Input images:

```text
shape: [3, 32, 32]
flattened dimension: 3072
classes: 10
train examples: 50,000
test examples: 10,000
```

Recommended split:

```text
train: 45,000
val:   5,000
test:  10,000
```

Use a deterministic validation split from the training set.

## Preprocessing

Use standard CIFAR-10 normalization.

```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
])
```

For the first experiment, do **not** use data augmentation.

Do not use:

```text
random crop
horizontal flip
color jitter
cutout
mixup
autoaugment
```

Reason: augmentation adds another variable. The first run should test architecture behavior under a clean, controlled setup.

Later experiments can add augmentation once the baseline comparison is established.

## Model Variants

Train the same four model families as in the MNIST experiment:

```text
standard_mlp
standard_mlp_matched
circle_mlp
helix_mlp
```

### 1. Standard MLP

Dense feedforward baseline.

Example architecture:

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

### 2. Parameter-Matched Standard MLP

A dense MLP with hidden width chosen to roughly match the parameter count of the Helix MLP.

This is the main baseline.

Parameter matching does not need to be perfect in the first run, but the result table must report parameter counts.

### 3. Circle MLP

Uses CircleLayer blocks.

Each CircleLayer computes learned 2D projections and phase/radius features:

```text
a = x · u
b = x · v
r = sqrt(a² + b²)
sin(theta) = b / r
cos(theta) = a / r
```

Feature set:

```text
sin(theta)
cos(theta)
r
r * sin(theta)
r * cos(theta)
```

The layer projects these features back into an ordinary hidden vector.

### 4. Helix MLP

Uses HelixLayer blocks.

Each HelixLayer computes learned 3D projections and phase/radius/axis features:

```text
a = x · u
b = x · v
z = x · w
r = sqrt(a² + b²)
sin(theta) = b / r
cos(theta) = a / r
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

The layer projects these features back into an ordinary hidden vector.

## Recommended Architecture

Start with two hidden layers, matching the MNIST setup as closely as possible.

For CIFAR-10, use a larger hidden dimension than MNIST.

Suggested first configuration:

```text
hidden_dim: 512
circle_units: 256
helix_units: 256
num_layers: 2
dropout: 0.1
layernorm: true
```

Why larger than MNIST:

```text
input dimension is 3072 instead of 784
CIFAR-10 is substantially harder
fully connected models need more capacity
```

If this is too slow, run a smaller quick pass:

```text
hidden_dim: 256
circle_units: 128
helix_units: 128
num_layers: 2
```

## Parameter Scale Sweep

A single model size is not enough. Run at least three parameter scales.

Suggested scale grid:

| Scale | hidden_dim | circle_units | helix_units |
|---|---:|---:|---:|
| small | 256 | 128 | 128 |
| medium | 512 | 256 | 256 |
| large | 1024 | 512 | 512 |

For each scale, run:

```text
standard_mlp
standard_mlp_matched
circle_mlp
helix_mlp
```

The exact parameter counts will differ. Report them.

The most useful plot is:

```text
test accuracy vs parameter count
```

This is more informative than a single headline number.

## Training Configuration

Suggested default:

```text
batch_size: 128
epochs: 100
learning_rate: 1e-3
weight_decay: 1e-4
optimizer: AdamW
scheduler: cosine decay or none for v1
dropout: 0.1
seed: 0
```

For the first implementation, keeping the scheduler off is acceptable. If all models plateau early or overfit badly, add cosine decay in a second pass.

Loss:

```python
loss = F.cross_entropy(logits, target)
```

Metrics:

```text
train loss
train accuracy
val loss
val accuracy
test loss
test accuracy
parameter count
time per epoch
examples per second
```

## Expected Baseline Range

Flattened CIFAR-10 is a bad setup for fully connected networks. Do not compare these numbers to CNNs.

A strong dense MLP baseline on flattened CIFAR-10 is expected to land roughly in the range:

```text
55%–60% test accuracy
```

Exact accuracy depends on width, depth, regularization, augmentation, and training budget.

The relevant comparison is the gap between model families under the same setup.

## Success Criteria

### Minimal success

Helix MLP trains stably, avoids NaNs, and achieves nontrivial CIFAR-10 accuracy.

Example:

```text
test accuracy > 45%
```

This would show that HelixLayer can handle harder image data at all.

### Useful success

Helix MLP comes within a few percentage points of the parameter-matched dense MLP.

Example:

```text
standard_mlp_matched: 58%
helix_mlp:            55%–58%
```

This would suggest that the primitive generalizes reasonably to nonlocal pixel classification.

### Strong success

Helix MLP matches or beats the parameter-matched dense MLP across at least one parameter scale.

Example:

```text
standard_mlp_matched: 58%
helix_mlp:            58%+
```

This would be interesting, but should be confirmed with multiple seeds.

### Negative but informative result

Helix MLP falls far behind the dense baseline.

Example:

```text
standard_mlp_matched: 58%
helix_mlp:            42%
```

This would suggest that the helix primitive has difficulty with high-dimensional, nonlocal pixel data. That is a real finding about the architecture's limitations.

## Key Comparisons

### Helix MLP vs Standard MLP

Tests whether the helix layer improves over a simple dense baseline.

### Helix MLP vs Parameter-Matched Standard MLP

This is the main comparison.

If Helix MLP has more parameters, it should not be credited for matching a smaller dense model. Report parameter counts carefully.

### Helix MLP vs Circle MLP

Tests whether the axial features add value beyond phase/radius features.

If Circle MLP matches or beats Helix MLP, do not claim helix-specific benefit.

### Circle MLP vs Standard MLP

Tests whether phase/radius expansion alone is useful as a generic feedforward layer.

## Important Caveats

### This is not a vision SOTA experiment

CNNs and modern vision models will do much better. That is not the comparison.

The comparison is:

```text
dense FFN vs circle FFN vs helix FFN on flattened pixels
```

### Flattened CIFAR-10 is hostile to all FFNs

Poor absolute accuracy is expected. The important signal is relative performance.

### Parameter matching is difficult

HelixLayer has more internal structure and more operations than a dense layer. Report:

```text
parameter count
training time
examples per second
```

If possible, add approximate FLOP estimates later.

### No causal helix claim yet

This experiment does not test whether learned helix channels are causally meaningful. It only tests whether the architecture trains and performs competitively.

Causal probing can come later if the model is viable.

## Implementation Files

Use a flat local layout, mirroring the MNIST experiment.

Suggested files:

```text
cifar10_config.py
cifar10_data.py
cifar10_models.py
cifar10_train.py
cifar10_run_experiment.py
cifar10_test_all.py
cifar10_results.md
```

Alternatively, reuse the MNIST files by generalizing them to classification experiments:

```text
classification_config.py
classification_data.py
classification_models.py
classification_train.py
classification_run_experiment.py
```

For speed, the first version can copy the MNIST files and adapt them for CIFAR-10.

## Suggested CLI

```bash
python cifar10_test_all.py

python cifar10_run_experiment.py --quick --model-type helix_mlp
python cifar10_run_experiment.py --quick --all-models

python cifar10_run_experiment.py --model-type standard_mlp --scale small
python cifar10_run_experiment.py --model-type standard_mlp_matched --scale small
python cifar10_run_experiment.py --model-type circle_mlp --scale small
python cifar10_run_experiment.py --model-type helix_mlp --scale small

python cifar10_run_experiment.py --all-models --scale medium
python cifar10_run_experiment.py --all-models --scale large
```

Optional:

```bash
python cifar10_run_experiment.py --sweep-scales
```

## Quick Mode

Quick mode is for verifying the pipeline, not for reporting results.

Suggested quick settings:

```text
epochs: 1
limit_train_batches: 100
limit_eval_batches: 50
hidden_dim: 256
circle_units: 128
helix_units: 128
device: cpu or cuda
```

Expected quick result:

```text
training runs without NaNs
metrics are finite
accuracy is above chance or moving upward
```

Chance accuracy is:

```text
10%
```

## Tests

Use local script-style tests, as in previous experiments.

### Fast tests

Run with:

```bash
python cifar10_test_all.py
```

Tests:

```text
test_config_defaults
test_standard_mlp_forward_shape
test_circle_layer_forward_shape
test_helix_layer_forward_shape
test_all_models_no_nans
test_all_models_backward_pass
test_count_parameters_positive
test_synthetic_training_step
```

### Data tests

Run with:

```bash
python cifar10_test_all.py --data
```

Tests:

```text
test_cifar10_batch_shape
test_cifar10_flatten_dim
test_cifar10_num_classes
test_train_val_split_sizes
test_split_deterministic
```

Expected batch shape before flattening:

```text
[batch, 3, 32, 32]
```

Expected flattened dimension:

```text
3072
```

### Slow tests

Run with:

```bash
python cifar10_test_all.py --slow
```

Tiny overfit tests:

```text
test_overfit_tiny_batch_standard
test_overfit_tiny_batch_circle
test_overfit_tiny_batch_helix
```

Suggested thresholds are lower than MNIST:

```text
standard_mlp tiny-batch accuracy > 80%
circle_mlp tiny-batch accuracy > 70%
helix_mlp tiny-batch accuracy > 70%
```

If Helix MLP cannot overfit a small CIFAR-10 batch, full training results will be hard to interpret.

## Result Artifacts

Save results locally.

Suggested output structure:

```text
cifar10_results/
  comparison_scale_small.json
  comparison_scale_medium.json
  comparison_scale_large.json
  accuracy_vs_params.png

  standard_mlp_small/
    history.json
    metrics.json
    training_history.png

  standard_mlp_matched_small/
    history.json
    metrics.json
    training_history.png

  circle_mlp_small/
    history.json
    metrics.json
    training_history.png

  helix_mlp_small/
    history.json
    metrics.json
    training_history.png
```

Checkpoints:

```text
cifar10_checkpoints/
  standard_mlp_small_best.pt
  standard_mlp_matched_small_best.pt
  circle_mlp_small_best.pt
  helix_mlp_small_best.pt
```

## Reporting Template

The final results document should include:

```text
dataset
preprocessing
model variants
parameter counts
training budget
test accuracy
test loss
training stability
time per epoch
accuracy vs parameter count
```

Suggested table:

| Model | Scale | Params | Best Val Acc | Test Acc | Test Loss | Time/Epoch |
|---|---|---:|---:|---:|---:|---:|
| Standard MLP | small | ... | ... | ... | ... | ... |
| Dense matched | small | ... | ... | ... | ... | ... |
| Circle MLP | small | ... | ... | ... | ... | ... |
| Helix MLP | small | ... | ... | ... | ... | ... |

## Interpretation Rules

Before seeing results, use these rules.

### If Helix MLP matches dense MLP

Interpretation:

```text
HelixLayer remains competitive on a hard nonlocal pixel task.
```

Do not claim superiority unless confirmed across seeds and parameter scales.

### If Helix MLP beats dense MLP

Interpretation:

```text
Potential evidence that HelixLayer is an efficient nonlinear primitive for this setup.
```

Required follow-up:

```text
multiple seeds
parameter sweep
compute/time comparison
ablation of feature groups
```

### If Helix MLP trails by a small gap

Interpretation:

```text
HelixLayer is viable but not better than dense layers here.
```

This is still a useful result.

### If Helix MLP trails badly

Interpretation:

```text
HelixLayer may struggle with high-dimensional nonlocal pixel data.
```

Follow-up:

```text
check tiny-batch overfitting
check optimization
try larger units
try lower learning rate
try feature ablations
```

### If Circle MLP beats Helix MLP

Interpretation:

```text
Axis features are not helping in this setup.
```

Do not make helix-specific claims.

## Follow-Up Ablations

If Helix MLP is competitive, run feature ablations:

```text
phase-only features
raw projection features
axis-only features
phase-axis interaction features
full feature set
```

This matters because some features are equivalent to raw projections:

```text
r * sin(theta) = b
r * cos(theta) = a
```

Those terms may function like ordinary dense paths. Ablations are needed to determine what part of the layer is doing useful work.

## Next Experiment

If flattened CIFAR-10 is successful or partially successful, the next experiment should move to tabular classification.

Candidate datasets:

```text
Covertype
HIGGS
UCI Adult
UCI Letter Recognition
UCI Spambase
```

Tabular data has no image locality, no rotation, and no obvious phase variable. It is a strong test of whether HelixLayer is a genuinely general nonlinear feature primitive.

## Goblin Summary

Flatten the tiny pictures.

Take away the convolution crutch.

Make the dense goblin and spiral goblin fight on bad terrain.

The number that matters is not whether either one becomes a vision champion.

The number that matters is the gap.
