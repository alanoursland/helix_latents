# Helix Latents

This repo explores a class of neural network models that treat helical, circular, toroidal, and related geometric structures as first-class computational objects.

The motivating observation is simple: neural networks often discover low-dimensional geometric structure when they need to represent ordered, periodic, or phase-like variables. In mechanistic interpretability, these structures sometimes appear as circles, spirals, helices, or more general manifolds inside activation space. Examples include representations of token position, modular arithmetic, counting, line length, calendar-like cycles, and other variables that combine progression with recurrence.

The goal of this project is to ask: what happens if we do not merely discover these structures after training, but build models that can use them directly?

## Core Idea

A helix is a compact way to represent two things at once:

1. Phase: where something is within a cycle.
2. Progression: how far along it is overall.

For example, a scalar variable `t` can be embedded as a helix:

```text
x(t) = c + t v + r cos(ωt) u + r sin(ωt) w
```

Here `u` and `w` span the circular part of the representation, while `v` gives the axis of progression. The circular component can encode modular or periodic information. The axial component can encode monotonic progress.

A model with helix-native latent variables could represent states like:

```text
H = (θ, z, r, confidence)
```

where `θ` is phase, `z` is position along the axis, `r` is radius or strength, and `confidence` is how strongly the model is using that geometric variable.

Instead of treating the hidden state as an unconstrained vector, the model can write to, read from, rotate, translate, and compare structured geometric objects.

## Why This Might Matter

Many cognitive and language-modeling problems involve variables that are not purely linear.

Examples:

- token position
- line length
- indentation depth
- bracket depth
- musical rhythm
- rhyme and meter
- turn-taking in dialogue
- repeated sequence continuation
- modular arithmetic
- calendar arithmetic
- syntactic or semantic cycles

Ordinary neural networks can learn internal representations for these variables, but the representations are often implicit. A helix-native model would make some of these latent variables more explicit and more causally accessible.

The hope is not to replace transformers, attention, or MLPs. The hope is to add a small number of geometric organs: modules specialized for tracking phase, recurrence, progression, and structured state.

## Mechanistic Interpretability Angle

The central interpretability question is not merely:

> Does a helix appear in activation space?

The stronger question is:

> Is the helix doing causal work?

A representation is useful if we can intervene on it and predict the result. For example, if a model uses a helix to represent a number, then rotating the phase of the helix should make the model behave as though the number changed.

A successful helix-native model should allow interventions like:

```text
rotate(H, Δθ)
translate(H, Δz)
project_phase(H)
project_axis(H)
compare_phase(H1, H2)
```

These operations should have predictable downstream effects.

For instance, if a model has learned modular addition, rotating the latent representation of `a` by `+k` should cause the output to shift from:

```text
a + b mod N
```

to:

```text
a + k + b mod N
```

without changing the input tokens.

That kind of intervention would show that the geometric object is not just decorative. It is part of the computation.

## Smallest Experiment

The smallest experiment is modular addition.

Task:

```text
input:  a, b
output: (a + b) mod N
```

Use a small prime or odd integer such as `N = 59`.

Train two models:

1. A baseline MLP or tiny transformer.
2. A similar model with an explicit circular or helical bottleneck.

A simple circular encoding is:

```text
h(a) = [cos(2πa/N), sin(2πa/N)]
```

A simple helical encoding is:

```text
h(a) = [cos(2πa/N), sin(2πa/N), αa]
```

After training, perform a causal intervention. Rotate the hidden representation for `a` by `k` steps, while leaving the input unchanged.

Expected result:

```text
model(a, b)        -> a + b mod N
model(rotate(a,k), b) -> a + k + b mod N
```

The key metric is intervention accuracy:

> How often does rotating the latent helix by `k` produce the answer corresponding to adding `k` to the represented value?

Controls should include random rotations in unrelated subspaces and interventions on baseline models.

## Research Questions

This repo is meant to explore questions like:

- Can explicit geometric latent variables improve generalization on periodic or structured tasks?
- Are helix-native models more interpretable than unconstrained models?
- Do models naturally learn to use helix modules when they are available?
- Can learned helices specialize to different features?
- Can geometric interventions produce predictable behavioral changes?
- What tasks prefer circles, helices, tori, cylinders, or more general manifolds?
- Can this scale from toy arithmetic to language-modeling features like line length, indentation, meter, or dialogue rhythm?

## Possible Model Components

A helix module might include:

- learned basis vectors for circular and axial subspaces
- a parameterization of phase, radius, pitch, and axis location
- readout functions for phase and progression
- operations for rotation, translation, and comparison
- regularizers that keep states close to intended manifolds
- intervention hooks for mechanistic interpretability experiments

A generic embedding function might look like:

```text
embed(H) = r cos(θ) u + r sin(θ) w + z v
```

Multiple helices could be combined into richer structures:

```text
S¹ × R
S¹ × S¹
S¹ × S¹ × R
```

These correspond roughly to cylinders, tori, and coupled periodic-plus-linear systems.

## Initial Milestones

1. Implement circular and helical latent modules.
2. Train on modular addition.
3. Add causal intervention tests.
4. Compare against unconstrained MLP baselines.
5. Extend to sequence tasks such as repeated patterns and bracket depth.
6. Add visualization tools for latent trajectories.
7. Explore learned rather than hard-coded helix bases.
8. Test whether helix modules emerge as useful components in tiny transformers.

## Philosophy

The project is based on a small bet:

> Some neural computations become easier to understand when their latent geometry is made explicit.

A helix is not magic. It is just a useful mathematical shape for binding recurrence to direction. But that combination appears often: in sequence position, rhythm, counting, modular arithmetic, and ordered cycles.

If ordinary networks keep rediscovering these structures, maybe we should give models direct access to them and see what happens.

## Status

This is an exploratory research repo. Early experiments should prioritize small, testable, causal demonstrations over ambitious scaling.

The first win is not state-of-the-art performance. The first win is a clean intervention:

> Twist the latent geometry, and the model's behavior twists with it.
