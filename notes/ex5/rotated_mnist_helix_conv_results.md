# Rotated MNIST HelixConv Results

## Outcome

**Pessimistic.** HelixConv trains comparably to vanilla CNN but does not self-organize into orientation-tracking geometric variables. Filter pairs are scattered; trajectories are not circles. The geometric primitives function as a generic nonlinearity.

## Overview

This experiment tested whether a HelixConv layer — given independent filter banks `W_u`, `W_v`, `W_w` and the mathematical structure to compute phase, radius, and axis features — would spontaneously learn to pair `(W_u, W_v)` as rotated quadrature filters when trained on rotation-augmented MNIST.

It did not.

The accuracy comparison across all four model variants is roughly equivalent. The filter analysis and trajectory analysis, which are the actual measurements of interest, show no self-organization.

## Dataset

MNIST with random rotation augmentation (uniform over ±180°).

```
train:           55,000 (with random rotation)
val:              5,000 (with random rotation)
test_rotated:    10,000 (with random rotation)
test_unrotated:  10,000 (no rotation)
```

## Training Summary

Scale: small. Seed: 0. Epochs: 30. Kernel size: 5. AdamW, lr=1e-3, weight_decay=1e-4.

| Model | Params | Val Acc | Test Rot Acc | Test Unrot Acc | Gap | Time/Ep |
|---|---|---|---|---|---|---|
| Standard CNN | 42,154 | 96.06% | 95.97% | 96.45% | -0.48% | 13.4s |
| Standard CNN (matched) | 82,426 | 96.84% | 96.20% | 97.00% | -0.80% | 13.9s |
| CircleConv CNN | 47,338 | 96.58% | 96.11% | 96.83% | -0.72% | 14.2s |
| HelixConv CNN | 63,642 | 96.60% | 96.36% | 96.61% | -0.25% | 14.3s |

All four models reach comparable accuracy. The rotated-unrotated gap is small for all models (< 1%), indicating that all learned reasonable rotation invariance from the augmented training data alone.

HelixConv is not measurably better or worse than vanilla convolution at this task. This is the expected precondition — it confirms the layer is not broken and earns the right to examine the internal structure. But the accuracy comparison is not the result.

## Measurement 2: Filter Pair Self-Organization

This is the central measurement. For each unit in the first HelixConv (or CircleConv) layer, we extracted `W_u` and `W_v`, swept rotation angles `φ ∈ {0°, 5°, ..., 355°}`, and found the angle `φ*` that maximizes the normalized correlation between `W_u` and `rotate(W_v, φ)`.

### HelixConv (16 units)

```
φ* values:  35, 275, 200, 265, 230, 315, 255, 0, 175, 265, 320, 305, 270, 195, 305, 350
corr*:      0.56, 0.21, 0.56, 0.56, 0.29, 0.58, 0.58, 0.01, 0.36, 0.44, 0.66, 0.40, -0.08, 0.71, 0.48, 0.38

Mean corr*:          0.42
Fraction near 90°:   0.00  (0 of 16 units)
```

### CircleConv (16 units)

```
φ* values:  50, 270, 200, 270, 250, 300, 230, 325, 175, 5, 330, 90, 310, 0, 305, 5
corr*:      0.42, 0.39, 0.49, 0.48, 0.50, 0.45, 0.65, 0.35, 0.35, 0.35, 0.52, 0.60, 0.45, 0.06, -0.01, 0.03

Mean corr*:          0.38
Fraction near 90°:   0.0625  (1 of 16 units)
```

### Interpretation

The `φ*` distribution is scattered uniformly across angles. There is no peak near 90°. One CircleConv unit landed at 90° by chance (1/16 ≈ 6.25%, consistent with a uniform distribution over 72 angle bins).

The correlation values are moderate (mean ~0.4). For self-organized quadrature pairs, we would expect `φ*` concentrated near 90° with correlations above 0.8. What we observe instead is consistent with `W_u` and `W_v` remaining effectively independent after training.

The `φ*` histogram is the single most important figure in this experiment. It shows that the filters did not self-organize.

## Measurement 3: Trajectory Analysis

For each helix unit, we rotated a fixed input digit through 360° in 5° steps and tracked the `(a, b)` activations at a fixed spatial position.

### Key observations

At layer 0, digit 0 (a "7"), center position:

- **Circularities** range from 0.11 to 0.74 (median ~0.26). A perfect circle would score > 0.95. Most units produce distorted or collapsed trajectories, not circles.

- **Winding numbers** are mostly 0.0, with 3 of 16 units showing winding number ≈ 1.0. The majority of units do not complete a full rotation in `(a, b)` space as the input rotates through 360°. The three units with winding ≈ 1.0 are interesting but are not accompanied by high circularity or low radius variance.

- **Radius variance** (coefficient of variation) ranges from 0.16 to 1.21. For rotation-invariant radius, this should be near 0. The high values indicate that `r` fluctuates substantially with input rotation, meaning it is not functioning as a pure "feature strength" variable independent of orientation.

- **Z variance** ranges from 0.16 to 2.47. The axis variable `z` is not rotation-invariant either.

### Interpretation

The trajectories are blobs and distorted curves, not circles. The `(a, b)` representation does not track input orientation in a structured way. The helix unit's phase, radius, and axis features all respond to rotation, but not in the clean decomposition that would constitute self-organization.

This is consistent with the filter analysis: since `W_u` and `W_v` are not rotated pairs, there is no reason `(a, b)` would trace a circle.

## Measurement 4: Causal Intervention

Not run. The design document specifies that intervention analysis should only be performed if Measurements 2 and 3 show self-organization. Since filter pairs are scattered and trajectories are blobs, intervening on `(a, b)` would not produce coherent results.

## Parameter Counts

| Model | Params |
|---|---|
| Standard CNN | 42,154 |
| CircleConv CNN | 47,338 |
| HelixConv CNN | 63,642 |
| Standard CNN (matched) | 82,426 |

HelixConv uses ~1.5x the parameters of the standard CNN. The matched CNN uses ~2x. Despite having more parameters, HelixConv does not achieve measurably better accuracy. This is consistent with the Covertype finding that geometric layers can match dense baselines but are not more parameter-efficient.

## What This Means

The rotated MNIST experiment was designed as the cleanest possible test of self-organization. The input symmetry (rotation) is exactly the symmetry that the helix layer is built to express. The filters are unconstrained. The kernel size (5) is large enough for rotation structure to be numerically detectable.

Despite these favorable conditions, gradient descent did not discover that pairing `W_u` and `W_v` as 90°-rotated copies would let `(a, b)` cleanly track input orientation. Instead, the layer found a different solution that achieves the same accuracy — presumably treating the geometric features as a generic nonlinear expansion of the conv outputs.

This is the pessimistic outcome described in the experiment design, and it is consistent with everything in the project so far:

- Experiment 1 (modular addition): Causal interventions worked, but the geometry was imposed, not learned.
- Experiment 2 (MNIST): HelixLayer trained but showed no advantage over dense baselines.
- Experiment 3 (flattened CIFAR-10): Geometric layers beat dense, but the advantage may be parameter count.
- Experiment 4 (Covertype): Best absolute metrics for geometric layers, but dense MLPs are more parameter-efficient.
- **Experiment 5 (rotated MNIST HelixConv): No self-organization. Geometric primitives function as generic nonlinearity.**

The pattern across all five experiments is that helix layers can train and sometimes achieve good accuracy, but there is no evidence that they discover or exploit geometric structure on their own.

## Scope of Claim

This result applies to:

```
Single-channel 28x28 rotated MNIST
HelixConv2d with 16 units, kernel size 5, 2 conv blocks
30 epochs of AdamW training
Seed 0, small scale
```

It does not rule out self-organization under different conditions:

- Larger scale (more units might specialize)
- More training epochs
- Different initialization
- Explicit regularization encouraging quadrature pairing
- Multi-seed runs (the result has not been confirmed across seeds)
- Different datasets with stronger rotational structure

However, if self-organization were a robust emergent property of the architecture, it should have appeared under these favorable conditions. Its absence here is evidence against the strong version of the self-organization hypothesis.

## Artifacts

### Per-model results

```
rot_mnist_results/standard_cnn_small_seed0/
rot_mnist_results/standard_cnn_matched_small_seed0/
rot_mnist_results/circle_conv_small_seed0/
rot_mnist_results/helix_conv_small_seed0/
```

Each contains `metrics.json`, `history.json`, `training_history.png`.

### Filter analysis (circle and helix only)

```
filter_pair_grid.png
phi_star_histogram.png
correlation_histogram.png
filter_analysis.json
```

### Trajectory analysis (circle and helix only)

```
trajectory_grid_layer0.png
trajectory_grid_layer1.png
trajectory_summary_layer0.png
trajectory_summary_layer1.png
trajectory.json
```

Plus per-digit, per-position variants for 3 digits and 3 spatial positions.

### Comparison

```
rot_mnist_results/comparison_small_seed0.json
```

## Next Steps

The experiment design's decision gate for the pessimistic outcome:

> Helix layers do not self-organize even when input symmetry is built in. The geometric primitives function as a generic nonlinearity. This is consistent with the parameter-efficiency findings on Covertype. The project's original framing requires revision.

Options:

1. **Multi-seed confirmation.** Run seeds 0-4 at small scale to verify the result is not seed-dependent. If even one seed shows a peak at 90°, the picture changes.

2. **Medium-scale run.** More units (32 helix units) would make the histogram more statistically meaningful. It is possible that a few units self-organize while most do not (the "middle outcome").

3. **Constrained variant.** Tie `W_v = rotate_90(W_u)` explicitly, making the layer a simplified harmonic network. Compare accuracy to the unconstrained version. If constrained is better, the structure helps but gradient descent cannot find it. If constrained is worse, the geometric features genuinely are not useful for this task.

4. **Move on.** Accept the result and revise the project framing. The helix layer is a functional nonlinearity but not a self-organizing geometric primitive.
