# Experiment 6: Quadrature Initialization and Regularization

## Motivation

Experiment 5 showed that HelixConv does not self-organize: given unconstrained
filter banks `W_u`, `W_v` and the machinery to compute phase and radius,
gradient descent on rotation-augmented MNIST leaves the filter pairs scattered
uniformly in rotation angle, with no concentration at the 90° quadrature
relationship that would let phase track input orientation.

That result is consistent with two different explanations:

1. **No basin.** The loss landscape has no attractor at the quadrature
   solution — even a model placed there would drift away, because the
   structure carries no advantage for the classification loss.
2. **Never found.** A quadrature basin exists (the structure is at least
   neutral, possibly useful), but gradient descent starting from independent
   random filters never finds it.

Experiment 5 cannot distinguish these. This experiment can, by starting the
model at (or pulling it toward) the quadrature solution and watching what
training does.

## Setup

Identical to Experiment 5 (rotated MNIST, ±180° augmentation, 30 epochs,
small scale, seed 0, AdamW lr=1e-3 wd=1e-4, kernel 5), with three model
variants:

| Variant | Description |
|---|---|
| `helix_conv` | Ex5 baseline: unconstrained random init (control). |
| `helix_conv_quadinit` | `W_v` initialized as `rotate(W_u, 90°)` (exact `rot90`, plus copied bias), then trained with no constraint. Tests whether training **preserves or destroys** imposed quadrature structure. |
| `helix_conv_quadreg` | Random init, plus a soft penalty `λ · ‖W_v − rotate(W_u, 90°)‖² / ‖rotate(W_u, 90°)‖²` summed over conv blocks, λ = 0.1. Tests whether a soft pull toward quadrature is **compatible with** the classification loss, and whether the structure then helps. |

The rotation convention (`torch.rot90`, k=−1) was verified numerically to
read out as φ* = 90° under the Experiment 5 filter-analysis sweep.

## Measurements

1. **Quadrature alignment trace** (new): mean per-unit Pearson correlation
   between `W_v` and `rotate(W_u, 90°)`, per conv block, recorded at init and
   after every epoch (`quad_alignment` in `history.json`,
   `quad_alignment.png`). This is the preserve-or-destroy signal:
   - quadinit starts at 1.0. If it stays high, the quadrature solution is at
     least a local basin. If it decays toward 0, the loss actively pulls the
     filters apart — there is no basin.
   - quadreg starts near 0. Where it equilibrates measures how hard the task
     loss resists the pull.
2. **φ\* filter analysis** (Experiment 5's Measurement 2, unchanged): final
   distribution of best-rotation angles and correlations.
3. **Accuracy** on rotated and unrotated test sets, against the `helix_conv`
   control. If quadrature structure helps, the constrained variants should
   match or beat the control; if it costs capacity, they should lag.

## Predictions

- If Experiment 5's negative result is "no basin": quadinit alignment decays
  substantially over 30 epochs and final φ* scatters; quadreg alignment stays
  low or costs accuracy.
- If it is "never found": quadinit alignment stays near 1.0 with no accuracy
  cost (the structure is neutral — a flat region, not repulsive), and quadreg
  reaches high alignment at equal accuracy. Either way accuracy parity with
  the control would indicate the geometry is permitted but not preferred.
- A quadrature variant *beating* the control would be the surprising outcome,
  suggesting the structure is useful but unreachable from random init.

## Running

```bash
cd src
python rot_mnist_run_experiment.py --quad-models --scale small --run-filter-analysis
```

Single seed (0), consistent with the rest of the project: these are smoke
tests, and one trial is enough to falsify.
