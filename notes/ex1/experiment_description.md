# Experiment Description: Helix Bottleneck for Modular Addition

## Goal

Test whether a neural network can use a helical latent representation as a causal computational object, rather than merely forming a helical-looking representation after training.

The smallest useful test is modular addition:

```text
(a + b) mod N = c
```

We train a tiny model to predict `c` from `a` and `b`, while forcing part of the model's internal state through a circular or helical bottleneck. Then we directly intervene on that bottleneck by rotating it and checking whether the model's output changes in the predicted way.

The core question is:

> If we twist the internal helix by `k` steps, does the model behave as though the represented number changed by `k`?

If yes, the helix is not just decorative geometry. It is doing computational work.

## Task

Choose a small prime modulus, such as:

```text
N = 59
```

Generate all ordered pairs:

```text
a, b ∈ {0, 1, ..., N - 1}
```

The label is:

```text
c = (a + b) mod N
```

This gives `N² = 3481` total examples when `N = 59`.

Split into train, validation, and test sets. A simple split is:

```text
70% train
15% validation
15% test
```

For an even cleaner generalization test, hold out specific values of `a`, `b`, or sums, but the first version can use a random split.

## Model Variants

### 1. Baseline Model

A small MLP that receives learned embeddings for `a` and `b`.

Example structure:

```text
embedding_a = Embed(a)
embedding_b = Embed(b)

x = concat(embedding_a, embedding_b)

hidden = MLP(x)

logits = Linear(hidden)
```

The output dimension is `N`, one logit for each possible answer.

This model answers the question:

> Can a normal network solve the task?

It probably can.

### 2. Circle Bottleneck Model

A model with an explicit circular latent representation for one or both input numbers.

For a number `a`, define:

```text
theta_a = 2πa / N

circle(a) = [cos(theta_a), sin(theta_a)]
```

The model receives this circular representation either as its only representation of `a`, or as an additional structured latent.

Minimal version:

```text
h_a = [cos(2πa/N), sin(2πa/N)]
h_b = [cos(2πb/N), sin(2πb/N)]

x = concat(h_a, h_b)

hidden = MLP(x)

logits = Linear(hidden)
```

This tests whether circular phase alone is enough for the model to solve modular addition.

### 3. Helix Bottleneck Model

A helix adds an axial coordinate to the circular representation:

```text
theta_a = 2πa / N

helix(a) = [
  cos(theta_a),
  sin(theta_a),
  alpha * a
]
```

where `alpha` is a scale factor, for example:

```text
alpha = 1 / N
```

So the representation contains both:

1. **phase**, which captures modular/cyclic structure;
2. **axis position**, which captures monotonic progression.

For pure modular addition, the circular part may be sufficient. The axial coordinate becomes more useful in later tasks involving both modular and non-modular quantities.

## Training Objective

Use ordinary cross-entropy loss:

```text
loss = CrossEntropy(logits, c)
```

Track:

```text
train accuracy
validation accuracy
test accuracy
```

For the smallest first experiment, no extra geometric regularizer is necessary because the helix is explicitly imposed.

Later experiments can allow the model to learn the helix basis itself and add manifold regularization.

## Causal Intervention

After training, test whether the bottleneck representation is used causally.

For a chosen shift `k`, rotate the circular part of `h_a` by:

```text
delta = 2πk / N
```

Given:

```text
h_a = [x, y]
```

apply:

```text
x' = x cos(delta) - y sin(delta)
y' = x sin(delta) + y cos(delta)
```

For a helix representation:

```text
h_a = [x, y, z]
```

apply:

```text
x' = x cos(delta) - y sin(delta)
y' = x sin(delta) + y cos(delta)
z' = z + alpha * k
```

Then pass the modified bottleneck into the rest of the model without changing the original input tokens.

The expected output becomes:

```text
c_intervened = (a + k + b) mod N
```

The central metric is:

```text
intervention_accuracy = mean(argmax(logits_intervened) == c_intervened)
```

## Controls

### Random Subspace Control

Apply a random rotation to an unrelated hidden vector or random latent subspace.

Expected result:

```text
No consistent shift in the output.
```

This checks that the effect is specific to the structured circular or helical bottleneck.

### Wrong Rotation Control

Rotate by a value that does not correspond to an integer step, such as:

```text
delta = 0.37 radians
```

Expected result:

```text
The model may become uncertain or interpolate between neighboring outputs.
```

This can reveal whether the model treats the helix continuously or only as a discrete code.

### Axis-Only Control

For the helix model, change only the axial coordinate `z` while leaving phase unchanged.

Expected result for modular addition:

```text
Little or no systematic modular shift.
```

If the output changes strongly, the model may be relying on the axial coordinate rather than the circular phase.

### Phase-Only Control

For the helix model, rotate phase while leaving `z` unchanged.

Expected result:

```text
The output may shift correctly if phase is the dominant causal variable.
```

Comparing phase-only and phase-plus-axis interventions helps identify which part of the helix is doing the work.

## Success Criteria

A strong positive result would show:

```text
test_accuracy ≈ 100%
intervention_accuracy ≈ 100%
random_control_accuracy ≈ chance or no systematic shift
```

For `N = 59`, chance accuracy is roughly:

```text
1 / 59 ≈ 1.7%
```

The most important number is intervention accuracy. A model that solves the task but fails the intervention has not learned to use the helix as a clean causal object.

## Interpretation

If the intervention works, we can say:

> The model's structured latent representation behaves like a manipulable computational variable.

This is stronger than saying:

> A helix appears when we visualize the activations.

The experiment tests whether the representation has causal semantics. Rotating the latent should produce a predictable transformation in model behavior.

In short: twist the learned helix in the hidden representation and check whether the output rotates with it.

## Minimal Implementation Plan

1. Generate all modular addition examples for `N = 59`.
2. Build a tiny baseline MLP.
3. Build a circle bottleneck MLP.
4. Build a helix bottleneck MLP.
5. Train each model with cross-entropy.
6. Evaluate normal test accuracy.
7. Run latent rotation interventions for several values of `k`.
8. Compare intervention accuracy against controls.
9. Plot predicted output shift versus intended shift.
10. Save model checkpoints and intervention logs.

## Suggested Files

```text
experiments/modular_addition/data.py
experiments/modular_addition/models.py
experiments/modular_addition/train.py
experiments/modular_addition/intervene.py
experiments/modular_addition/evaluate.py
experiments/modular_addition/plots.py
```

## Suggested First Plots

### Accuracy by Model

Compare normal task accuracy:

```text
baseline
circle_bottleneck
helix_bottleneck
```

### Intervention Accuracy by Shift

For each shift `k`, plot:

```text
k → intervention_accuracy
```

### Confusion Matrix After Intervention

Rows:

```text
expected shifted answer
```

Columns:

```text
model predicted answer
```

A successful model should place most mass on the diagonal.

### Phase Interpolation Plot

Rotate by non-integer angles and plot output probabilities. This can show whether the model has learned a smooth circular computation or a brittle lookup table.

## Extensions

After the smallest experiment works, try tasks where a true helix should be more useful than a pure circle.

Examples:

```text
integer addition without modulo
calendar arithmetic
line wrapping
bracket depth tracking
musical beat and measure prediction
counting with periodic resets
```

These tasks combine cyclic structure with monotonic progression, which is exactly the natural habitat of a helix.

## Core Hypothesis

A helix is useful because it combines two kinds of information in one object:

```text
Where am I in the cycle?
How far along am I overall?
```

The experiment tests whether a neural network can use that object directly and causally.
