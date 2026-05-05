# MNIST HelixLayer Experiment

## Purpose

This experiment tests whether a helix-native feedforward layer is a viable general-purpose neural network primitive on a non-arithmetic classification task.

The previous modular addition experiment showed that explicit circular/helix latents can act as causal variables when the task has known cyclic structure. MNIST is different: the labels are ordinary image classes, and there is no obvious ground-truth helix variable.

The question here is narrower and more practical:

```text
Can a model built around HelixLayer units train and generalize on MNIST classification?
```

This is not intended to prove that helices are superior. It is a viability test.

## Main Hypothesis

A HelixLayer may be useful because each unit reads a local 3D subspace of the input and decomposes it into:

```text
phase  θ
radius r
axis   z
```

A standard linear neuron computes roughly:

```text
x -> w · x
```

A helix unit computes something more structured:

```text
x -> phase/radius/axis features in a learned 3D subspace
```

The resulting layer may be a useful alternative to ordinary affine + activation layers.

## Local File Layout

Keep the repo flat. Do not make a package.

Add these files alongside the existing local scripts:

```text
mnist_data.py
mnist_models.py
mnist_train.py
mnist_run_experiment.py
mnist_test_all.py
mnist_results.md
```

Existing shared files can still be reused if useful:

```text
utils.py
config.py
plotting.py
```

No `src/helix_latents/` package structure is needed.

## Dataset

Use MNIST through `torchvision.datasets.MNIST`.

Inputs:

```text
image shape: [1, 28, 28]
flattened shape: [784]
classes: 10
```

Default normalization:

```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
```

Use the standard split:

```text
train: 60,000
test: 10,000
```

Optionally create a validation split from train:

```text
train: 55,000
val: 5,000
test: 10,000
```

## Model Variants

Train at least four variants.

### 1. Standard MLP

A normal dense baseline:

```text
784 -> hidden -> hidden -> 10
```

Example:

```text
Linear
GELU
Linear
GELU
Linear
```

### 2. Parameter-Matched Standard MLP

A standard MLP with adjusted hidden width so its total parameter count roughly matches the HelixLayer model.

This is important because helix units have more parameters than ordinary neurons.

### 3. CircleLayer MLP

A circle-only version of the helix layer.

Each unit learns a 2D subspace using basis vectors `u` and `v`, then computes:

```text
a = x · u
b = x · v
theta = atan2(b, a)
r = sqrt(a^2 + b^2)
```

The output features can include:

```text
cos(theta)
sin(theta)
r
```

This tests whether phase/radius alone are enough, without an axial coordinate.

### 4. HelixLayer MLP

Each unit learns a 3D subspace using basis vectors:

```text
u, v, w
```

For input `x`, compute:

```text
a = x · u
b = x · v
z = x · w
r = sqrt(a^2 + b^2 + eps)
theta = atan2(b, a)
```

Then emit features based on:

```text
cos(theta)
sin(theta)
r
z
```

A simple output per helix unit is:

```text
features_i = [
    r_i * cos(theta_i),
    r_i * sin(theta_i),
    z_i
]
```

This is actually equivalent to returning the three projections `[a, b, z]`, so by itself it is too linear. To make the layer genuinely nonlinear, include nonlinear phase/radius/axis features.

Recommended v1 features:

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

Then pass the concatenated features through a learned output projection.

## HelixLayer v1

The first implementation should be simple and stable.

### Constructor

```python
class HelixLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        units: int,
        output_dim: int,
        eps: float = 1e-6,
        use_layernorm: bool = True,
    ):
        ...
```

### Parameters

```python
self.u = nn.Parameter(torch.randn(input_dim, units) * scale)
self.v = nn.Parameter(torch.randn(input_dim, units) * scale)
self.w = nn.Parameter(torch.randn(input_dim, units) * scale)

self.bias_u = nn.Parameter(torch.zeros(units))
self.bias_v = nn.Parameter(torch.zeros(units))
self.bias_w = nn.Parameter(torch.zeros(units))

self.out = nn.Linear(units * num_features_per_unit, output_dim)
```

Optional:

```python
self.layernorm = nn.LayerNorm(output_dim)
```

### Forward Pass

```python
def forward(self, x):
    a = x @ self.u + self.bias_u
    b = x @ self.v + self.bias_v
    z = x @ self.w + self.bias_w

    r = torch.sqrt(a * a + b * b + eps)

    sin_theta = b / r
    cos_theta = a / r

    features = torch.cat([
        sin_theta,
        cos_theta,
        r,
        z,
        r * sin_theta,
        r * cos_theta,
        torch.tanh(z),
        r * torch.tanh(z),
    ], dim=-1)

    y = self.out(features)

    if use_layernorm:
        y = self.layernorm(y)

    return y
```

Important note:

```text
Do not use atan2 in v1 unless needed for analysis.
```

Using `a/r` and `b/r` gives sine and cosine directly and avoids some gradient awkwardness around angle wrapping.

## CircleLayer v1

Same idea, but no `w` and no `z`.

Features:

```text
sin_theta
cos_theta
r
r * sin_theta
r * cos_theta
```

Layer:

```python
class CircleLayer(nn.Module):
    ...
```

This gives a direct comparison:

```text
circle geometry vs helix geometry
```

## HelixMLP Architecture

A small MNIST classifier:

```text
Flatten
HelixLayer(784 -> hidden)
GELU
HelixLayer(hidden -> hidden)
GELU
Linear(hidden -> 10)
```

The HelixLayer itself outputs `output_dim`, so hidden activations remain ordinary vectors between layers. This is a pragmatic hybrid: the layer computes using helix features internally but returns a standard vector.

Example:

```python
class HelixMLP(nn.Module):
    def __init__(self, hidden_dim=128, units=64):
        self.flatten = nn.Flatten()
        self.h1 = HelixLayer(784, units=units, output_dim=hidden_dim)
        self.h2 = HelixLayer(hidden_dim, units=units, output_dim=hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.gelu(self.h1(x))
        x = F.gelu(self.h2(x))
        return self.classifier(x)
```

## Baseline MLP Architecture

```python
class StandardMLP(nn.Module):
    def __init__(self, hidden_dim=128):
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 10),
        )
```

## Parameter Counting

Every run should print:

```text
model name
parameter count
train accuracy
validation accuracy
test accuracy
```

Function:

```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

The comparison should include:

```text
standard_mlp_small
standard_mlp_param_matched
circle_mlp
helix_mlp
```

## Suggested Default Config

```python
@dataclass
class MNISTConfig:
    model_type: str = "helix_mlp"

    batch_size: int = 128
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    hidden_dim: int = 128
    helix_units: int = 64
    circle_units: int = 64

    seed: int = 0
    device: str = "cuda"

    data_dir: str = "data"
    results_dir: str = "mnist_results"
    checkpoint_dir: str = "mnist_checkpoints"
```

For a very fast smoke test:

```text
epochs = 1
limit_train_batches = 100
```

## Training

Use ordinary cross-entropy:

```python
loss = F.cross_entropy(logits, target)
```

Optimizer:

```python
torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
)
```

Track:

```text
train_loss
train_accuracy
val_loss
val_accuracy
test_loss
test_accuracy
```

Save:

```text
mnist_results/<model_type>/history.json
mnist_results/<model_type>/metrics.json
mnist_checkpoints/<model_type>_best.pt
```

## Commands

Suggested CLI:

```bash
python mnist_test_all.py

python mnist_run_experiment.py --model-type standard_mlp
python mnist_run_experiment.py --model-type circle_mlp
python mnist_run_experiment.py --model-type helix_mlp

python mnist_run_experiment.py --all-models
python mnist_run_experiment.py --quick
```

## Tests

Use the same style as the current `test_all.py`: a local script with assertions.

### Test 1: Data Shapes

Verify:

```text
batch image shape == [batch, 1, 28, 28]
batch target shape == [batch]
```

### Test 2: HelixLayer Forward Shape

```python
layer = HelixLayer(input_dim=784, units=16, output_dim=32)
x = torch.randn(8, 784)
y = layer(x)
assert y.shape == (8, 32)
```

### Test 3: CircleLayer Forward Shape

```python
layer = CircleLayer(input_dim=784, units=16, output_dim=32)
x = torch.randn(8, 784)
y = layer(x)
assert y.shape == (8, 32)
```

### Test 4: No NaNs

Run random inputs through all models and assert:

```python
torch.isfinite(logits).all()
```

This is especially important for radius normalization.

### Test 5: Backward Pass

For each model:

```python
logits = model(images)
loss = F.cross_entropy(logits, targets)
loss.backward()
```

Assert at least one parameter has a nonzero gradient.

### Test 6: Overfit Tiny Batch

Take one batch of 128 MNIST examples and train for 100-300 steps.

Expected:

```text
standard_mlp should overfit
circle_mlp should overfit
helix_mlp should overfit
```

This is the most important early test. If HelixMLP cannot overfit one batch, do not run full MNIST yet.

### Test 7: Quick Training Smoke Test

Train for one epoch or 100 batches and assert accuracy is better than chance:

```text
accuracy > 20%
```

MNIST chance is 10%.

## Expected Outcomes

Possible outcomes:

### HelixMLP trains and matches MLP

This is a good first result.

It means the helix layer is a viable local primitive on a standard classification task.

### HelixMLP beats parameter-matched MLP

This would be more interesting, but do not expect it on the first try.

If it happens, rerun with multiple seeds.

### HelixMLP loses slightly but trains well

Still useful. It means the primitive is viable, but may not help generic classification.

Pairing this with a win on phase-progress tasks would be a coherent result.

### HelixMLP fails to overfit a tiny batch

Stop. Debug optimization before running full experiments.

Likely causes:

```text
bad initialization
unstable radius normalization
too few units
too much weight decay
LayerNorm placed poorly
features too constrained
```

## Suggested Reporting Table

```text
| Model | Params | Test Acc | Notes |
|---|---:|---:|---|
| Standard MLP | ... | ... | Dense baseline |
| Param-matched MLP | ... | ... | Fairer comparison |
| Circle MLP | ... | ... | Phase/radius features |
| Helix MLP | ... | ... | Phase/radius/axis features |
```

## Cautious Interpretation

MNIST is not a task with obvious helical structure. A positive result here should be interpreted as:

```text
HelixLayer can function as a trainable neural network layer on ordinary classification.
```

Not as:

```text
Helices are generally better than standard neural layers.
```

A stronger claim would require:

1. multiple seeds;
2. matched parameter counts;
3. matched compute where possible;
4. additional datasets;
5. a task where phase/progress structure is known to matter.

## Next Step After MNIST

If MNIST works, run:

```text
Fashion-MNIST
flattened CIFAR-10
rotated MNIST
rotated CIFAR-10
```

The most interesting follow-up is rotated MNIST or rotated CIFAR-10 with two heads:

```text
class head
rotation angle head
```

That would test both:

```text
classification utility
phase representation
```

## Summary

First prove the helix layer can learn digits.

Then move to harder benchmarks.
