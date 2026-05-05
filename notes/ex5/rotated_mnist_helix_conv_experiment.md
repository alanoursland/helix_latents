# Rotated MNIST HelixConv Experiment

## Purpose

This experiment tests whether a helix-style convolutional layer **self-organizes** to track input orientation when trained on a dataset with rotational structure.

Previous experiments in this project used HelixLayer as a feedforward primitive and asked whether it could train and generalize. This experiment asks a different question:

```text
If we give a conv layer the mathematical primitives for phase/radius/axis features,
and train it on rotated images,
does the layer spontaneously organize so that phase tracks orientation?
```

This is the "self-organization" test. The whole point is to **not** enforce rotation equivariance. We give the layer the right structure and let gradient descent decide whether to use it.

## Why This Test

The four previous experiments produced a mixed picture:

```text
Experiment 1: modular addition
  Imposed circular geometry supports causal interventions.
  Bottleneck was the input, so equivariance was forced, not learned.

Experiment 2: MNIST
  HelixLayer trains and generalizes.
  No clear advantage over dense baselines.

Experiment 3: flattened CIFAR-10
  Geometric layers beat dense baselines, gap widens with scale.
  Circle beats Helix; axis features did not earn their keep.

Experiment 4: Covertype
  Helix achieves best absolute metrics but is parameter-inefficient.
  Dense scales fine; geometric advantage may be capacity, not structure.
```

None of these tested whether the geometric layers self-organize internally into manipulable geometric variables. The original modular-addition causal-intervention result was on imposed geometry, not discovered geometry.

Rotated MNIST is the cleanest available test because the input symmetry is exactly the symmetry the helix layer is built to express. If self-organization is going to happen anywhere, it should happen here. If it does not happen here, the broader self-organization claim is in serious trouble.

## Dataset

Use MNIST through `torchvision.datasets.MNIST`, with a random rotation applied at load time.

```text
image shape: [1, 28, 28]
classes:     10
rotation:    θ ∈ [0, 2π) sampled uniformly per example
```

Suggested transform:

```python
transforms.Compose([
    transforms.RandomRotation(degrees=180, fill=0),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
```

Use the standard MNIST split:

```text
train: 60,000  (with random rotation augmentation)
val:    5,000  (split from train; with random rotation)
test:  10,000  (with random rotation)
```

For evaluation analyses (not training), it is also useful to keep an **un-rotated MNIST test set** and a **fixed-rotation-grid test set** where the same image is replayed at controlled rotation angles. The fixed-rotation test set is required for the trajectory analysis below.

## The HelixConv Layer

A standard 2D conv filter at spatial position `(i, j)` produces a scalar:

```text
y[i, j] = (W * x)[i, j]
```

A HelixConv unit produces three scalars per spatial position by learning **three independent filter banks**:

```text
a[i, j] = (W_u * x)[i, j]
b[i, j] = (W_v * x)[i, j]
z[i, j] = (W_w * x)[i, j]
```

Then computes phase, radius, and axis features:

```text
r[i, j]      = sqrt(a[i, j]² + b[i, j]² + ε)
sin_θ[i, j]  = b[i, j] / r[i, j]
cos_θ[i, j]  = a[i, j] / r[i, j]
```

Final per-unit features (concatenated along the channel dimension):

```text
sin_θ
cos_θ
r
z
r * sin_θ
r * cos_θ
tanh(z)
r * tanh(z)
```

These features are passed through a learned 1×1 conv to project to the desired output channel count.

### Critical design constraint

`W_u`, `W_v`, and `W_w` are **independent** learnable filter banks. There is no constraint that `W_v` is a 90°-rotated version of `W_u`. The whole experiment hinges on whether gradient descent **discovers** that pairing rotated filters makes `(a, b)` behave like `(cos, sin)`.

If we constrain `W_v = rotate_90(W_u)` ahead of time, we are doing Harmonic Networks, not testing self-organization.

### HelixConv v1 Sketch

```python
class HelixConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        units: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.units = units

        # Three independent filter banks per unit
        self.conv_u = nn.Conv2d(in_channels, units, kernel_size, padding=padding)
        self.conv_v = nn.Conv2d(in_channels, units, kernel_size, padding=padding)
        self.conv_w = nn.Conv2d(in_channels, units, kernel_size, padding=padding)

        # 1x1 projection from 8*units feature channels to out_channels
        self.project = nn.Conv2d(units * 8, out_channels, kernel_size=1)

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

A CircleConv variant is the same with no `W_w`/`z` and 5 features per unit instead of 8.

## Model Variants

Train four variants for comparison:

### 1. Standard CNN

Ordinary 2D convolutional baseline.

```text
Conv2d(1, hidden_channels, 3, padding=1)
GELU
MaxPool
Conv2d(hidden_channels, hidden_channels, 3, padding=1)
GELU
MaxPool
Flatten
Linear -> 10
```

### 2. Parameter-Matched Standard CNN

Same architecture with hidden width adjusted to roughly match HelixConv parameter count. This is the fair dense baseline.

### 3. CircleConv CNN

Replaces the conv layers with CircleConv2d blocks (phase/radius features only).

### 4. HelixConv CNN

Replaces the conv layers with HelixConv2d blocks (phase/radius/axis features).

## What This Experiment Measures

There are three distinct measurements, and they answer different questions. The headline accuracy comparison is **not the most important one**. The visualization analyses are.

### Measurement 1: Does the model train?

Standard sanity check. Report:

```text
test accuracy on rotated MNIST
test accuracy on un-rotated MNIST  (transfer check)
mean epoch time
parameter count
```

A precondition, not a result. If HelixConv is wildly worse than vanilla conv, the layer is broken and the rest of the analysis is moot. If it is roughly comparable, we have earned the right to look at the more interesting measurements.

Do not read too much into either model winning by a small margin. This is not the result of the experiment.

### Measurement 2: Did `(W_u, W_v)` self-organize into rotated filter pairs?

This is the central plot.

For each helix unit in the first HelixConv layer:

1. Extract `W_u` and `W_v` (each is a 2D `k × k` filter, ignoring input channels for single-channel MNIST).
2. For each candidate angle `φ ∈ {0°, 5°, 10°, ..., 355°}`, compute the correlation between `W_u` and `rotate(W_v, φ)`.
3. Record the angle `φ*` that maximizes this correlation, and the correlation value at `φ*`.

Then make two histograms across all helix units:

```text
histogram of φ*           (where do filter pairs sit on the rotation circle?)
histogram of correlation  (how cleanly do they pair?)
```

Interpretation:

```text
φ* peaks near 90°, high correlation
  -> layer self-organized into orthogonal-quadrature filter pairs
     (a, b) is structured like (cos, sin) of one underlying filter

φ* uniformly distributed
  -> layer did not find the structure; W_u and W_v are independent

φ* peaks near 0° or 180°, high correlation
  -> filter pairs are duplicates or sign-flips, not rotated pairs
     (a, b) is degenerate; no useful phase
```

This single figure largely determines the result of the experiment.

### Measurement 3: Does `(a, b)` trace a circle as input rotation varies?

This is the **direct behavioral test** of "twist the input, watch the activations twist."

Procedure:

1. Pick one un-rotated MNIST digit.
2. Rotate it through `θ ∈ [0°, 360°)` in 5° increments (72 rotated copies).
3. For each rotated copy, run forward through the trained model, and record `(a, b)` at a fixed spatial position for each helix unit in the first HelixConv layer.
4. Plot the trajectory of `(a, b)` in 2D as `θ` varies.

Repeat for several digits, several spatial positions, and several units. Produce a grid of trajectory plots.

Interpretation:

```text
clean circle, period 360°
  -> unit's phase tracks input orientation directly
     this is the strongest possible self-organization result

clean circle, period 180° or smaller
  -> unit tracks orientation modulo a smaller symmetry
     also a good result; consistent with Gabor-like behavior

ellipse or distorted curve
  -> unit responds to rotation but not in pure phase
     interesting and worth understanding

blob with no structure
  -> unit does not track rotation; using helix features as
     generic nonlinearity
```

A complementary measurement: track `r` and `z` across the same rotation sweep. Ideally, `r` should be approximately rotation-invariant (it is the strength of the feature, not its orientation), and `z` should also be approximately rotation-invariant or vary slowly.

### Measurement 4 (optional): Does twisting `(a, b)` predictably transform downstream activations?

This is the full causal-intervention version, analogous to the original modular addition test.

For a fixed input image, rotate the `(a, b)` representation at one helix unit by `Δθ` (without changing the input):

```text
a' = a cos(Δθ) - b sin(Δθ)
b' = a sin(Δθ) + b cos(Δθ)
```

Pass the modified activations through the rest of the network. Compare the output to running the network on a version of the input image that was actually rotated by `Δθ`.

If the two match, the helix unit is functioning as a learned, manipulable orientation variable — exactly the property the project's original framing wanted.

This measurement is harder than the trajectory plot, and only worth running if Measurements 2 and 3 already show self-organization. If `(W_u, W_v)` are independent and `(a, b)` is a blob, intervention will not produce coherent results.

## Suggested Default Config

```python
@dataclass
class RotMNISTConfig:
    model_type: str = "helix_conv"   # or standard_cnn, standard_cnn_matched, circle_conv

    batch_size: int = 128
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    hidden_channels: int = 32
    helix_units: int = 16
    circle_units: int = 16
    kernel_size: int = 5             # larger kernel makes filter-rotation visualization clearer

    seed: int = 0
    device: str = "cuda"

    rotation_max_degrees: int = 180  # full rotation range
    data_dir: str = "data"
    results_dir: str = "rot_mnist_results"
    checkpoint_dir: str = "rot_mnist_checkpoints"
```

A larger kernel size (5 or 7 rather than 3) is recommended for this experiment. Filter-rotation pairings are more visually obvious and more numerically stable to detect at larger spatial extents.

## Local File Layout

Match the existing flat structure:

```text
rot_mnist_data.py
rot_mnist_models.py
rot_mnist_train.py
rot_mnist_run_experiment.py
rot_mnist_analyze_filters.py    # Measurement 2
rot_mnist_analyze_trajectory.py # Measurement 3
rot_mnist_test_all.py
rot_mnist_results.md
```

## Tests

Following the project pattern.

### Test 1: Data Shapes

```text
batch image shape    == [batch, 1, 28, 28]
batch target shape   == [batch]
rotation augmentation actually rotates (verify visually on a few samples)
```

### Test 2: HelixConv Forward Shape

```python
layer = HelixConv2d(in_channels=1, units=8, out_channels=16, kernel_size=5, padding=2)
x = torch.randn(4, 1, 28, 28)
y = layer(x)
assert y.shape == (4, 16, 28, 28)
```

### Test 3: No NaNs

Especially important because of `b / r` division. Run random inputs through all models and assert finiteness.

### Test 4: Backward Pass

Cross-entropy loss, single backward pass, assert gradients flow to `conv_u`, `conv_v`, `conv_w`, and `project`.

### Test 5: Overfit Tiny Batch

Take 128 rotated MNIST examples and train for ~300 steps. All four models should reach near-100% training accuracy. If HelixConv cannot overfit a tiny batch, debug optimization before running full experiments.

### Test 6: Filter Visualization Pipeline

Train for one epoch on a small subset, then run the filter-correlation analysis end-to-end. Verify that the histogram of `φ*` is being computed correctly. The point is to validate the analysis pipeline, not to interpret the result of a one-epoch model.

## Expected Outcomes

### Optimistic outcome

```text
HelixConv trains comparably to vanilla CNN.
Filter-pair correlation histogram peaks near φ = 90°.
(a, b) trajectories trace clean circles as input rotates.
r is approximately rotation-invariant.
Causal intervention (Measurement 4) produces coherent rotated outputs.
```

This would be the strongest single result for the entire project. It would show that helix layers self-organize into manipulable geometric variables when the input has the right symmetry, without being told to.

### Middle outcome

```text
HelixConv trains comparably to vanilla CNN.
Some helix units self-organize, most do not.
Trajectory plots are circles for some units, blobs for others.
```

Still informative. Would suggest the layer can find geometric structure when useful but does not waste capacity on it otherwise. Worth understanding which units organize and why.

### Pessimistic outcome

```text
HelixConv trains comparably to vanilla CNN.
Filter pairs look random; φ* histogram is uniform.
(a, b) trajectories are blobs.
```

Consistent with everything else in the project so far. Would indicate that helix layers, given the option to use phase structure or to ignore it, ignore it. The geometric primitives function as a generic nonlinearity rather than as the foundation for self-organized orientation tracking.

This outcome is genuinely possible. The project's existing results on non-arithmetic tasks lean this direction.

### Failure outcome

```text
HelixConv fails to overfit tiny batch.
HelixConv training diverges or stalls.
NaNs appear in r or sin_θ / cos_θ.
```

Stop. Debug optimization. Likely causes: bad initialization of `W_u`, `W_v`; insufficient `eps`; kernel too small for stable phase computation; LayerNorm placement.

## Scope of the Claim

This experiment can show:

```text
HelixConv layers do (or do not) self-organize into orientation-tracking
geometric variables when trained on data with rotational input symmetry.
```

This experiment cannot show:

```text
HelixConv beats rotation-equivariant CNNs.
HelixConv self-organizes on data without rotational symmetry.
HelixConv is generally useful for image tasks.
The result transfers to natural images, color, or larger datasets.
```

The result, whatever it is, applies to single-channel rotated MNIST with this specific layer design. It does not immediately generalize. But it is the cleanest single test of the self-organization claim, because the right structure to find is unambiguous and the dataset is small enough to iterate quickly.

## Relationship to Prior Experiments

```text
Experiment 1: modular addition
  causal interventions on imposed circular geometry

Experiment 2: MNIST
  trainability of HelixLayer on simple classification

Experiment 3: flattened CIFAR-10
  geometric MLPs outperform dense baselines on hard pixel classification

Experiment 4: Covertype
  geometric MLPs achieve best metrics but trade parameters for accuracy

Experiment 5: rotated MNIST HelixConv (this experiment)
  does HelixConv self-organize to track input orientation?
```

This is the first experiment in the ladder that directly tests whether the geometric layers learn the geometric thing on their own.

## Minimal First Version

If you want to see the central thing fastest, the minimal version is:

```text
1. One HelixConv2d layer with 8 units, kernel size 5.
2. One pooling + one classifier head.
3. Train on rotated MNIST for ~10 epochs.
4. Visualize W_u and W_v for each of the 8 units side by side.
5. Compute the φ* histogram.
6. Look at the figure.
```

If `W_u` and `W_v` look like rotated pairs at φ ≈ 90°, run the trajectory analysis next. If they look like noise, the deeper experiments probably will not recover anything either.

This is a few hours of code and one figure. The figure decides whether the rest of the experiment is worth running.

## Summary

Rotate the digit, check if the learned phase rotates with it.

If it does, the layer learned rotation-equivariant structure on its own.

If it doesn't, the layer was just a generic nonlinearity all along.
