# Paper Proposal

## Working Title

**Permitted but Not Preferred: Imposed Geometric Structure in Neural Layers Survives Training but Never Emerges From It**

Alternative titles:
- *Neutral Plateaus, Not Basins: Why Helical Latents Don't Self-Organize*
- *You Can Lead a Network to Quadrature: A Negative Result on Built-In Geometric Latents*

## Target Venue and Format

A 4–6 page workshop note. Candidate venues, in order of fit:

1. **"I Can't Believe It's Not Better" (ICBINB) workshop** (NeurIPS/ICLR) —
   purpose-built for well-executed negative results with a diagnosis.
2. **Workshop on Mechanistic Interpretability** (ICML/NeurIPS) — the
   motivation (Kantamneni & Tegmark's Clock algorithm; Nanda et al.'s
   grokking circles) and the "can we build interpretability in?" question
   are squarely in scope.
3. **Workshop on Symmetry and Geometry in Neural Representations (NeurReps)**
   — the equivariance-adjacent framing fits, and the result is a useful
   counterpoint to the hard-constraint literature.

Not a main-track paper: scale is MNIST-class, single seed per condition (to
be fixed, see §Required Work), and the headline is negative. As a workshop
note the completeness of the argument is the selling point.

## One-Paragraph Pitch

Mechanistic interpretability has shown that helical/circular number
representations *emerge* in trained networks (modular-addition transformers,
LLM arithmetic). We ask the converse engineering question: if you hand a
small network the geometric primitives — explicit phase, radius, and axis
computed from learned filter banks — does gradient descent on a matched task
*use* them geometrically? Across five tasks the answer is no, and a final
pair of experiments explains the failure mode precisely. On rotation-augmented
MNIST, unconstrained filter banks never converge to the 90°-quadrature
pairing that would make phase track orientation (final angles uniform over
[0°, 360°)); but quadrature *imposed at initialization* largely survives 30
epochs of unconstrained training, and a mild soft regularizer (λ=0.1) holds
it perfectly at zero accuracy cost. The quadrature manifold is therefore a
**neutral plateau** in the loss landscape — not a basin gradient descent
seeks, and not a hill it flees. Imposed geometry remains causally
manipulable (phase rotations shift outputs exactly, 100% intervention
accuracy on modular addition), so the obstruction to "interpretability by
construction" is not expressivity or trainability but the absence of any
optimization pressure toward using geometry as geometry. This suggests that
emergent helices in large models are driven by something these supervised
classification settings lack, and that practitioners who want geometric
latents should impose them — cheaply, via init or soft regularization —
rather than hope they emerge.

## Core Claims

1. **No emergence (C1).** Layers equipped with circle/helix primitives
   (`sin θ, cos θ, r, z` from independent learned filter banks) train
   stably and competitively on five tasks, but treat the primitives as a
   generic nonlinear expansion. Direct test: on rotated MNIST, filter-pair
   rotation angles φ* are uniform with no mass at 90° (0/16 helix units;
   activation trajectories under input rotation are blobs, not circles).
   *(Experiments 2–5.)*
2. **Neutrality, not repulsion (C2) — the new and central claim.**
   Quadrature initialized at layer 0 decays only from alignment 1.0 to 0.71
   over 30 epochs and plateaus, with 16/16 units retaining φ* = 90°;
   a soft penalty maintains alignment ≈ 1.0 in all layers at no accuracy
   cost. So the structure is *compatible* with the task loss but carries
   zero gradient signal: a flat shelf, unreachable from random init by
   anything but chance. *(Experiment 6.)*
3. **Depth boundary (C3).** Imposed quadrature over raw pixels survives;
   imposed quadrature over learned layer-1 feature channels collapses within
   5 epochs. Geometric priors are only even neutral where the input carries
   the matching symmetry. *(Experiment 6.)*
4. **Imposed geometry is causally sound (C4).** When the latent structure is
   designed in (modular addition with circle/helix bottlenecks), phase
   rotation by 2πk/N shifts the output by exactly k (100% intervention
   accuracy) — so the failure of C1 is not a failure of the primitives.
   *(Experiment 1.)*

## Narrative Structure

1. **Motivation:** emergent helices in LLM arithmetic (Kantamneni & Tegmark
   2025; Nanda et al. 2023) → the "interpretability by construction" hope:
   provide the vocabulary, get readable latents for free.
2. **Method:** Circle/Helix MLP layers and CircleConv/HelixConv2d; the φ*
   filter-rotation sweep and alignment-trace measurements (these
   measurement tools are a minor methodological contribution in themselves).
3. **Emergence fails** (C1), with rotated MNIST as the clean test — input
   symmetry exactly matches layer structure, and it still doesn't organize.
4. **Why it fails** (C2, C3): the init/regularize interventions that
   separate "no basin" from "never found."
5. **Implications:** (a) for interp-by-construction: impose, don't hope —
   soft regularization is free; (b) for emergent-structure research: the
   driver of helix formation in LLMs is absent from supervised
   classification, candidate hypotheses being task-mandated algorithmic
   structure (the *output* must be circular in modular arithmetic) versus
   mere input symmetry, which we show is insufficient; (c) for the
   equivariance literature: a data point on why hard constraints
   (steerable/group-equivariant CNNs) are used in practice — the soft
   version is not discovered.

## Related Work (to position against)

- **Emergent geometric representations:** Nanda et al. (grokking modular
  addition, arXiv:2301.05217); Kantamneni & Tegmark (LLM addition helices,
  arXiv:2502.00873); Engels et al. (circular features in LLMs); Liu et al.
  (grokking and representation formation).
- **Hard-wired equivariance:** Cohen & Welling (group-equivariant CNNs),
  Worrall et al. (harmonic networks — which *fix* quadrature by
  construction; our negative result explains why they have to). We must be
  clear we test the *soft/learnable* middle ground they skip.
- **Lottery-ticket / reachability-vs-existence:** framing C2 as a
  reachability result connects to work distinguishing what solutions exist
  from what SGD finds (e.g., Frankle & Carlin; mode connectivity
  literature).
- **Learned invariance from augmentation:** all our models become
  rotation-robust (gap < 1%) without geometric structure — consistent with
  augmentation-driven invariance not requiring equivariant mechanism.

## Evidence Inventory (already in repo)

| Claim | Experiment | Key artifact |
|---|---|---|
| C4 | ex1 | 100% intervention accuracy table |
| C1 viability | ex2–4 | accuracy tables (MNIST, flat CIFAR-10, Covertype) |
| C1 direct | ex5 | φ* histogram (uniform), trajectory plots, 4-model accuracy table |
| C2, C3 | ex6 | alignment traces (per-epoch, per-layer), final φ* (16/16 at 90°), λ=0.1 at no cost |

The flattened CIFAR-10 advantage (+5.6 pts, ex3) is *not* a claim — it is
confounded by parameter count and single-seeded; the paper mentions it only
as "geometric expansion can act as a useful generic nonlinearity," or drops
it entirely.

## Required Work Before Submission

Ordered by importance; items 1–2 are necessary, the rest strengthen.

1. **Seeds.** 5 seeds for ex5 control and the three ex6 variants. The whole
   argument rests on these; per-run cost is ~30 GPU-minutes. Report
   mean ± sd accuracy and alignment, and pooled φ* histograms (80 units
   instead of 16). *(Without this, reviewers at even a workshop will balk.)*
2. **Longer-horizon decay.** Train quadinit for 100–150 epochs to bound the
   layer-0 plateau: does 0.71 hold or keep eroding? This decides between
   "neutral plateau" and "slow repulsion," i.e., it could *falsify C2* —
   say so in the paper either way.
3. **λ sweep.** quadreg at λ ∈ {0.001, 0.01, 0.1, 1.0} to show the
   compatibility result isn't an artifact of one value, and locate where (if
   anywhere) alignment starts costing accuracy.
4. **Anneal-to-zero probe.** Train quadreg for 15 epochs, then set λ=0 and
   watch the alignment trace — the cleanest single figure for "plateau, not
   basin" (structure formed *by training itself* then released).
5. **One task where geometry should matter at the output.** Optional but
   high-value: rotation *regression* (predict the digit's rotation angle)
   with the same HelixConv. If phase organizes there, C2 sharpens into "the
   pressure must come from the task's output structure, not input
   symmetry" — turning a negative result into a characterization.
6. Pin environment (`requirements.txt`), add a `make reproduce-ex6` entry
   point, and a stats note (uniformity test on φ*, e.g., Rayleigh test,
   instead of eyeballing the histogram).

## Proposed Figures

1. **Fig 1 (headline):** three alignment traces overlaid (control flat at 0,
   quadinit decaying to plateau, quadreg pinned at 1) — the entire argument
   in one panel. Layer 0 and layer 1 as two subpanels gives C3 for free.
2. **Fig 2:** φ* histograms, control vs quadinit (pooled over seeds), with
   the 90° line.
3. **Fig 3:** accuracy bars across all variants (all equal) — the
   "no use for it" panel.
4. **Fig 4 (if item 4 above is run):** anneal-to-zero trace.
5. **Table 1:** ex1 intervention accuracy (C4) + ex2–4 accuracy summary,
   compressed.

## Risks and Honest Limitations

- **Scale:** MNIST-class tasks and 16–64 unit layers. The defense: the
  question is about optimization pressure, not capability, and the setting
  is deliberately the *easiest possible* case for emergence (input symmetry
  exactly matches layer structure). If it doesn't emerge here, "provide the
  vocabulary" is dead as a general recipe.
- **Item 2 could overturn C2.** If long training erodes the plateau to
  zero, the paper's thesis becomes "slow repulsion" rather than
  "neutrality" — still publishable, but the title changes. Run it first.
- **One architecture family.** The circle/helix parameterization is one of
  many; a reviewer may ask whether e.g. learned Fourier features behave
  differently. Scope the claim to "filter-bank phase/radius layers" and
  note the generalization as open.
- **Novelty check before writing:** search for prior work on
  soft-equivariance regularization and on "does equivariance emerge from
  augmentation" (e.g., Gruver et al., *The Lie Derivative for Measuring
  Learned Equivariance*, which found partial emergent equivariance in large
  vision models — must be cited and reconciled: their models are far larger
  and their measure is behavioral, ours is mechanistic/weight-level).

## Estimated Effort

- Compute: items 1–4 ≈ 25–35 runs ≈ a few GPU-days at small scale; trivial
  on any single modern GPU.
- Writing: the repo's notes/ex5 and notes/ex6 documents are ~70% of the
  paper's content already. Realistic total: 2–3 weeks part-time to a
  submittable 5-page note.
