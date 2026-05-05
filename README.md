# Helix Latents

An experimental investigation into whether neural network layers with built-in geometric structure — circular phase, radial magnitude, helical axis — learn to use that structure, or just treat it as another nonlinearity.

## The Question

Standard neural network layers map inputs through dense linear transformations and pointwise nonlinearities. The representations that emerge are unstructured — they work, but they are not organized around any particular geometric primitive.

What happens if you give a layer the mathematical machinery for phase and radius (circle) or phase, radius, and axis position (helix)? Specifically:

1. Does the layer learn to use phase as phase, or does it ignore the structure and treat `sin(θ)`, `cos(θ)`, `r` as three more nonlinear features?
2. If you give a convolutional layer two independent filter banks and the formula for converting their outputs into phase/radius, does gradient descent discover that pairing the filters as 90°-rotated copies would let the phase track input orientation?
3. When the geometric structure is imposed (not learned), does it support clean causal interventions — can you rotate the phase and get the output you'd expect?

Five experiments, run in sequence, tested these questions with increasing directness.

## Experiments

### Experiment 1: Modular Addition

**Question:** Does imposed circular geometry support causal interventions?

**Setup:** Modular arithmetic (`(a + b) mod N`), where the correct latent structure is known to be circular. Circle and helix bottleneck MLPs with explicit phase representations.

**Result:** Yes. Rotating the phase representation by `2πk/N` shifts the output by exactly `k`, with 100% intervention accuracy. The helix model correctly uses only the phase component and ignores the axis. This confirms the pipeline works and the geometric features are causally meaningful — when the structure is imposed.

**Limitation:** The geometry was designed in, not discovered. This does not tell us whether a layer would find this structure on its own.

[Design](notes/ex1/experiment_description.md) | [Results](notes/ex1/experiment_results.md)

### Experiment 2: MNIST

**Question:** Can geometric MLP layers train on ordinary image classification?

**Setup:** Standard MNIST digit classification (flattened 784-pixel input). Circle MLP and Helix MLP compared against dense MLP baselines.

**Result:** All models reach 97-98% accuracy. HelixLayer trains stably and reaches competitive performance, but does not beat parameter-matched dense baselines. This is a viability check, not a claim of advantage.

[Design](notes/ex2/mnist_helix_experiment.md) | [Results](notes/ex2/mnist_experiment_results.md)

### Experiment 3: Flattened CIFAR-10

**Question:** Do geometric layers help on harder classification without convolution?

**Setup:** CIFAR-10 with flattened 3072-pixel input (no convolution, no augmentation). This forces the network to learn nonlocal pixel relationships purely through feedforward layers.

**Result:** Geometric MLPs outperform dense baselines by 5-6 percentage points at large scale (58.3% circle vs 52.7% dense). The gap widens with scale. This is the strongest positive result in the project — the geometric expansion appears to provide a useful inductive bias for this particular task.

**Caveat:** Single seed. The geometric models also use more parameters. The advantage may partially reflect capacity rather than structural utility.

[Design](notes/ex3/flattened_cifar10_experiment.md) | [Results](notes/ex3/flattened_cifar10_results.md)

### Experiment 4: Covertype (Tabular)

**Question:** Do geometric layers help on heterogeneous tabular data?

**Setup:** Forest Covertype dataset — 54 features (10 continuous, 44 binary), 7 classes. Tests whether circle/helix structure helps when the input has no obvious geometric symmetry.

**Result:** Helix MLP achieves the best absolute accuracy (95.75%) and macro F1 (0.932) at large scale. But the dense baselines reach 94.76% with far fewer parameters. When measured by accuracy error reduction per parameter, the dense MLP is more efficient. The geometric models win on absolute metrics by using roughly 4x the parameters.

[Design](notes/ex4/covertype_experiment.md) | [Results](notes/ex4/covertype_results.md)

### Experiment 5: Rotated MNIST HelixConv

**Question:** Does HelixConv self-organize to track input orientation?

**Setup:** MNIST with random rotation augmentation. A `HelixConv2d` layer uses three independent convolutional filter banks (`W_u`, `W_v`, `W_w`) and computes phase, radius, and axis features per spatial position. The filters are not constrained — the experiment tests whether gradient descent discovers that pairing `W_u` and `W_v` as 90°-rotated copies would let the phase representation track input orientation.

**Result:** It does not. The filter pair rotation angle `φ*` is scattered uniformly across 0°-360° with no concentration at 90°. Activation trajectories under input rotation are blobs, not circles. All four model variants reach equivalent accuracy (~96%), confirming the layer functions correctly — it just does not organize its internal structure around orientation.

This was the cleanest possible test of self-organization: the input symmetry exactly matches the layer's built-in structure, the kernel is large enough for rotation to be detectable, and the filters are unconstrained. The negative result is informative.

[Design](notes/ex5/rotated_mnist_helix_conv_experiment.md) | [Results](notes/ex5/rotated_mnist_helix_conv_results.md)

## Summary Table

| # | Task | Geometry | Accuracy vs Dense | Self-Org? | Verdict |
|---|---|---|---|---|---|
| 1 | Modular addition | Imposed | N/A (different task) | N/A (imposed) | Causal interventions work |
| 2 | MNIST (flat) | Learned MLP | Comparable | Not tested | Viable, not superior |
| 3 | CIFAR-10 (flat) | Learned MLP | +5.6 pts | Not tested | Best positive result |
| 4 | Covertype (tabular) | Learned MLP | +1.0 pts (4x params) | Not tested | Wins on absolute, loses on efficiency |
| 5 | Rotated MNIST | Learned Conv | Comparable | **No** | No self-organization |

## Conclusion

The geometric layers work. They train reliably, produce finite gradients, and achieve competitive or better accuracy across every task tested. They are not broken. On flattened CIFAR-10, they provide a genuine advantage that dense baselines cannot match at the same scale.

But there is no evidence that they learn geometric structure.

When circular geometry is imposed on a task with known circular symmetry (Experiment 1), the representation is causally manipulable. When the same geometric primitives are offered as learnable features on tasks with clear rotational structure (Experiment 5), gradient descent ignores the structure and uses the features as a generic nonlinear expansion. The layer has the mathematical vocabulary for phase and orientation. It does not learn to speak it.

The flattened CIFAR-10 result (Experiment 3) is real but ambiguous. The geometric layers outperform dense baselines by a meaningful margin, and the gap grows with scale. Whether this reflects genuine structural utility or simply a favorable parameter expansion is not resolved. The Covertype result (Experiment 4) leans toward the latter interpretation — the geometric models win on absolute metrics but lose on parameter efficiency.

Across five experiments, the pattern is:

- **Imposed geometry + matching task symmetry = clean, manipulable structure.**
- **Learned geometry + matching task symmetry = generic nonlinearity.**
- **Learned geometry + non-geometric task = sometimes better accuracy, but via parameter count, not structural discovery.**

The project's original motivation was that helix-shaped latent representations might self-organize when the architecture provides the right primitives. The evidence does not support this. The geometric layers are a functional architectural choice — sometimes useful, never self-organizing.

## Running the Code

All experiment code is in [`src/`](src/). See [`src/README.md`](src/README.md) for per-experiment instructions.

```bash
cd src

# Example: run one experiment
python rot_mnist_run_experiment.py --quick --all-models

# Example: run with analysis
python rot_mnist_run_experiment.py --all-models --scale small \
    --run-filter-analysis --run-trajectory-analysis
```

## Dependencies

```
torch
torchvision
numpy
matplotlib
tqdm
scikit-learn
scipy
```
