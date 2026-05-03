# Covertype Classification Experiment

## Purpose

This experiment tests whether `CircleLayer` and `HelixLayer` remain useful on tabular classification.

The previous flattened CIFAR-10 experiment gave the first strong non-toy result: Circle MLP and Helix MLP outperformed dense MLP baselines on flattened image pixels across all tested scales, while training at roughly similar epoch rates.

Covertype is a different kind of test.

It has:

```text
no image locality
no convolutional structure
no rotation target
no explicit phase variable
no sequence order
heterogeneous numerical and binary/categorical features
```

This makes it a strong next benchmark for asking whether the geometric layers are useful as general feedforward primitives rather than only as vision or synthetic-geometry tools.

The central question is:

```text
Can Circle MLP and Helix MLP remain competitive with dense MLP baselines on a heterogeneous tabular classification task?
```

## Dataset

Use the Forest CoverType dataset.

Task:

```text
predict forest cover type from cartographic and environmental features
```

Input:

```text
54 tabular features
```

Output:

```text
7 forest cover type classes
```

The dataset contains numerical terrain features and binary indicator features for wilderness areas and soil types.

The problem is multiclass classification.

## Why Covertype?

Covertype is a useful first tabular benchmark because it is:

```text
standard
large enough to be meaningful
fast enough to iterate locally
multiclass
not image-like
not sequence-like
not designed around phase or geometry
```

This makes it a good follow-up to flattened CIFAR-10.

Flattened CIFAR-10 still has image-origin structure, even though the models receive flattened pixels. Covertype is more directly tabular and heterogeneous.

A positive result on Covertype would strengthen the claim that CircleLayer and HelixLayer are not merely exploiting image-specific or cyclic structure.

## Models

Train the same model families as before:

```text
standard_mlp
standard_mlp_matched
circle_mlp
helix_mlp
```

### Standard MLP

Ordinary dense feedforward network.

### Parameter-Matched Standard MLP

Dense MLP with hidden width chosen to roughly match the geometric models.

This remains the main baseline.

### Circle MLP

Uses CircleLayer blocks with phase/radius features.

For each unit:

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

### Helix MLP

Uses HelixLayer blocks with phase/radius/axis features.

For each unit:

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

## Primary Metrics

Use several metrics because tabular datasets may be imbalanced.

Primary:

```text
test accuracy
macro F1
test loss / negative log likelihood
```

Secondary:

```text
weighted F1
per-class accuracy
confusion matrix
mean epoch time
examples per second
parameter count
```

Accuracy alone is not sufficient. Macro F1 should be reported prominently.

## Baseline Framing

This experiment compares neural feedforward primitives.

It should not be framed as an attempt to beat all tabular ML methods. Tree-based models such as random forests, gradient boosting, XGBoost, LightGBM, and CatBoost are traditionally strong on tabular data.

The relevant comparison is:

```text
dense MLP vs Circle MLP vs Helix MLP
```

Optional later baseline:

```text
RandomForestClassifier or HistGradientBoostingClassifier
```

But the first version should keep the neural comparison clean.

## Preprocessing

Use a fixed train/validation/test split.

Recommended:

```text
train: 70%
val:   15%
test:  15%
```

Use stratified splits if available.

Preprocess features as follows:

1. Standardize continuous numerical features using train-set statistics.
2. Leave binary indicator features as 0/1.
3. Convert labels to integer class IDs from 0 to 6.
4. Do not one-hot encode labels.
5. Do not leak validation/test statistics into preprocessing.

The Covertype features are commonly structured as:

```text
10 continuous/cartographic features
4 wilderness-area binary features
40 soil-type binary features
```

So the default preprocessing should standardize the first 10 features and leave the remaining 44 binary features unchanged.

## Experimental Scales

Run three scale settings:

| Scale | hidden_dim | circle_units | helix_units |
|---|---:|---:|---:|
| small | 64 | 32 | 32 |
| medium | 128 | 64 | 64 |
| large | 256 | 128 | 128 |

Tabular input dimension is only 54, so these are much smaller than the CIFAR-10 models.

The exact matched dense hidden widths should be chosen after checking parameter counts. Approximate matching is acceptable in v1, but parameter counts must be reported.

## Training Configuration

Suggested default:

```text
epochs: 100
batch_size: 1024
optimizer: AdamW
learning_rate: 1e-3
weight_decay: 1e-4
dropout: 0.05
use_layernorm: true
scheduler: optional
seed: 0
```

Because Covertype is larger than MNIST but much lower-dimensional than CIFAR-10, a larger batch size is reasonable.

Start without a scheduler. Add cosine decay only if learning plateaus or instability appears.

## Success Criteria

### Minimal success

Circle MLP and Helix MLP train stably, avoid NaNs, and achieve reasonable accuracy and macro F1.

This establishes basic tabular viability.

### Useful success

Circle MLP or Helix MLP comes within a small gap of the parameter-matched dense MLP.

This would show that the geometric layers can function as general tabular feedforward primitives.

### Strong success

Circle MLP or Helix MLP beats the parameter-matched dense MLP on both:

```text
test accuracy
macro F1
```

This is the strongest single-run result.

### Very strong success

The geometric advantage repeats across multiple seeds and scales.

Required before making strong claims:

```text
seeds: 0, 1, 2
```

Preferably:

```text
seeds: 0, 1, 2, 3, 4
```

## Interpretation Rules

### If Circle and Helix both beat dense MLPs

Interpretation:

```text
Geometric feature layers appear useful beyond image-origin data.
```

This would be an important generality result.

### If Circle beats Helix again

Interpretation:

```text
The phase/radius expansion may be the useful part; the helix axis is not clearly helping.
```

This should be treated as a good finding, not a failure.

### If Helix beats Circle

Interpretation:

```text
Axis-related features may help with heterogeneous tabular feature interactions.
```

This requires ablation before claiming axis utility.

### If dense MLP wins

Interpretation:

```text
The geometric layers did not improve tabular performance under this setup.
```

This is still useful. It would identify a limitation or a need for better tabular-specific preprocessing, regularization, or layer design.

### If all neural models perform poorly

Interpretation:

```text
The training setup or preprocessing may be inadequate.
```

Check preprocessing, class balance, label encoding, learning rate, and baseline accuracy.

## Important Caveats

This experiment should not overclaim.

It does not prove:

```text
geometric layers are generally better than dense layers
helix axes are semantically meaningful
tabular data has helical structure
the models beat gradient boosting
```

It can show:

```text
CircleLayer and/or HelixLayer are viable and competitive feedforward primitives on tabular classification.
```

## Recommended Follow-Up Ablations

If Circle or Helix performs well, run ablations.

For HelixLayer:

```text
full
no_axis
phase_radius
raw_projection
axis_only
```

For CircleLayer:

```text
full
phase_only
radius_only
raw_projection
```

These ablations are important because some features act like raw learned projections:

```text
r * sin(theta) = b
r * cos(theta) = a
```

Ablations can separate ordinary projection capacity from normalized geometric features.

## Expected Result Shapes

A strong result might look like:

```text
dense matched:  X% accuracy, Y macro F1
circle_mlp:     X+1% accuracy, Y+1 macro F1
helix_mlp:      X+0.5% accuracy, Y+0.5 macro F1
```

A still-useful result might look like:

```text
dense matched:  best overall
circle_mlp:     close but slightly behind
helix_mlp:      close but slightly behind
```

A limitation result might look like:

```text
dense matched:  clearly better
circle_mlp:     underfits or overfits
helix_mlp:      underfits or overfits
```

All three outcomes are informative.

## Relationship to Prior Experiments

The experiment ladder is now:

```text
Experiment 1: modular addition
  causal geometry proof of concept

Experiment 2: MNIST
  trainability on easy classification

Experiment 3: flattened CIFAR-10
  hard nonlocal pixel classification

Experiment 4: Covertype
  heterogeneous tabular classification
```

Covertype is the first test where the input is not image-origin and not explicitly cyclic.

## Goblin Summary

No pixels.

No rotations.

No spiral-shaped steak.

Just mixed little columns, some smooth, some binary, all annoying.

If the circle or helix goblin still fights well here, the shiny gets shinier.
