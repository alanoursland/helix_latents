# Helix Latents

An experimental investigation into whether neural network layers with built-in geometric structure — circular phase, radial magnitude, helical axis — learn to use that structure, or just treat it as another nonlinearity.

## Motivation

Mechanistic interpretability work has found that large language models represent numbers internally as generalized helices and manipulate these helices to perform arithmetic. Nanda et al. first showed circular number representations in toy transformers trained on modular addition ([Progress Measures for Grokking via Mechanistic Interpretability](https://arxiv.org/abs/2301.05217)). Kantamneni & Tegmark then showed that full-scale LLMs solve `a + b` by operating on helical number representations — the "Clock algorithm" ([Language Models Use Trigonometry to Do Addition](https://arxiv.org/abs/2502.00873)). These helices are not designed in; they emerge from training on next-token prediction over text that happens to contain arithmetic.

This raises a question: if helical geometry emerges naturally in large models trained on general data, can we build layers that provide the geometric primitives explicitly and get the same structure to emerge in small models trained on specific tasks? If the answer is yes, these layers would give us architectures with built-in interpretable geometric variables — phase, radius, axis — that we could read out and intervene on directly, without reverse-engineering them from unstructured representations.

## The Question

What happens if you give a layer the mathematical machinery for phase and radius (circle) or phase, radius, and axis position (helix)?

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

The geometric layers train reliably and do not break anything. They reach accuracy roughly comparable to standard layers across all tasks tested, though after accounting for parameter count they are generally slightly less efficient than dense linear layers (Experiments 2, 4). The flattened CIFAR-10 result (Experiment 3) is the one case where geometric layers clearly outperform dense baselines on absolute accuracy, but we did not control tightly enough for parameter count to know whether that advantage is structural or just capacity.

The one experiment that directly tested whether the geometric features are used geometrically (Experiment 5) found that they are not. When given independent filter banks and the mathematical machinery to compute phase and radius, gradient descent on a rotation-augmented classification task does not learn to pair the filters as rotated copies. The phase variable does not track input orientation. The geometric features are treated as a generic nonlinear expansion of the convolutional outputs.

For Experiments 2-4, we did not inspect whether the learned representations use phase as phase internally — we only compared final accuracy. It is possible that some geometric structure is present but was not measured. The only direct test is Experiment 5, and it was negative.

When geometry is imposed rather than learned (Experiment 1), it works cleanly — rotating phase by a known amount shifts the output predictably. This confirms the mathematical pipeline is sound. The gap is between "the structure works when you build it in" and "the structure emerges when you let gradient descent find it."

The project's original motivation was that helix-shaped representations emerge in LLMs trained on general text (Kantamneni & Tegmark), and that providing the geometric primitives explicitly might encourage the same structure to emerge in smaller models trained on specific tasks. The evidence does not support this. The conditions under which helices emerge in LLMs — massive scale, diverse training data, incidental arithmetic in a language modeling objective — may be essential, not incidental. Providing the geometric vocabulary is not enough; gradient descent on a classification loss finds other solutions that work equally well without organizing the representation geometrically.

These layers are a viable but unremarkable architectural choice for the tasks tested here.

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
