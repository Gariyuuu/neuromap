# NeuroMap: Auditing the Alignment Between Artificial Visual Representations and Measured Visual-Cortex Responses

*A small-scale, pre-registered audit using public mouse visual-cortex recordings. Not a clinical
tool, not a claim of cognitive similarity, not a replacement for Brain-Score.*

## Abstract

We measure how well linear ridge-encoding models built on artificial vision-model activations
predict trial-averaged calcium-imaging responses in six mouse visual cortical areas (Allen Brain
Observatory Visual Coding, 18 sessions, natural_scenes stimulus), and we ask whether the resulting
model rankings agree across three representational-similarity metrics (ridge encoding, RSA, linear
CKA), across cortical areas, and across recording sessions of the same area. Contrary to our
pre-registered H1, raw downsampled pixel intensities outperformed every pretrained deep network
(ResNet-18, ResNet-50, ViT-B/16, self-supervised DINO ViT-S/16) on mean held-out encoding score
(pixels r=0.199 vs. 0.108–0.121 for deep models), and higher ImageNet top-1 accuracy was associated
with *lower*, not higher, mean neural alignment across the three supervised models tested
(Spearman ρ=-1.0, n=3, exploratory). These findings replicate, in a new dataset and with a broader
model set, prior reports that ImageNet-optimized CNNs are a comparatively poor model of mouse
visual cortex. We do not claim any model "thinks like" the mouse brain; we report where a specific,
falsifiable set of correlational predictions holds and where it does not, and we quantify how much
that answer depends on which metric or session one asks.

## 1. Introduction

Comparing artificial neural networks to biological visual systems has produced an influential
research program — most visibly Brain-Score (Schrimpf, Kubilius, Yamins & DiCarlo, 2018/2020),
which benchmarks CNN/ViT alignment against macaque V1/V2/V4/IT and human psychophysics. That
program is primate-focused and uses linear/PLS encoding plus behavioral consistency as its
primary tools. Two questions it does not address at scale are (1) whether the same encoding
methodology, applied to a different species with a demonstrably flatter visual hierarchy — mouse
— tells the same story, and (2) whether conclusions about *which* model is most brain-like are
stable across the similarity metric used to ask the question. Recent work on primate/general
vision data (Soni, Srivastava, Kording & Khosla, 2024; Bo, Soni, Srivastava & Khosla, 2024; Sexton
& Love, 2022) shows metric choice can change both layer-area correspondence and overall model
rankings. We ask the same question in mouse visual cortex, where it has not been explicitly
tested, and add a secondary activation-compression experiment distinct from prior
model-parameter-compression work (Cowley, Stan, Pillow & Smith, 2026).

This is an audit, not a new benchmark suite: 18 sessions, 6 models, 3 similarity metrics, one
public dataset. We report what we found, including a result we did not expect and did not tune
for after seeing it (pixels beating every deep network — see Results §4.1).

## 2. Related Work

- **Yamins, Hong, Cadieu & DiCarlo (2014, PNAS)** — categorization-optimized hierarchical CNNs
  predict macaque V4/IT; task performance correlates with brain-predictivity.
- **Cadieu, Hong, Yamins et al. (2014, PLOS Comp Bio)** — deep nets match IT population geometry.
- **Cadena et al. (2019, PLOS Comp Bio)** — CNN features beat classical LN/wavelet models for
  macaque V1 spiking responses.
- **Schrimpf, Kubilius, Yamins & DiCarlo (2018 bioRxiv / 2020 Neuron)** — Brain-Score, the
  dominant primate/human neural-and-behavioral benchmarking framework; not mouse, not
  RSA/CKA-primary, no systematic cross-metric agreement analysis.
- **Soni, Srivastava, Kording & Khosla (2024, bioRxiv)** and **Bo, Soni, Srivastava & Khosla
  (2024, arXiv:2411.14633)** — similarity-metric choice materially changes both layer-area
  correspondence and model rankings for primate/general vision alignment ("fragile foundations").
- **Sexton & Love (2022, Science Advances)** — a stricter direct-interface test overturns standard
  hierarchical layer-area correspondence claims.
- **Siegle et al. (2021, Nature)** — Allen Institute Neuropixels survey: mouse visual cortex has a
  real but much flatter functional hierarchy than primate ventral stream.
- **Shi, Shea-Brown & Buice (NeurIPS 2019)** and **Nayebi et al.** — using Allen Brain Observatory
  data, standard ImageNet-supervised CNNs are a comparatively poor fit for mouse visual cortex;
  best-ImageNet-accuracy does not track best-brain-similarity in mice.
- **Cowley, Stan, Pillow & Smith (2026, Nature)** — a 60M-parameter DNN can be compressed ~5000x in
  *parameter count* while retaining macaque V1/V4/IT predictivity. NeuroMap's compression
  experiment (§5) instead compresses *activations* via PCA for a fixed architecture, on mouse data.

**Novelty position.** Neither "ranking instability across metrics" nor "compression preserves
predictivity" is a new phenomenon in this literature. What has not been explicitly reported is
their combination, in mouse Allen Observatory data, across natural_scenes/gratings/movies-style
stimulus statistics with a three-metric (ridge, RSA, CKA) design. We frame this work accordingly:
a rigorous small-scale replication/extension into an untested regime, not a discovery of a new
effect. See `docs/novelty_memo.md` for the full literature audit behind this framing.

## 3. Neural Dataset

**Allen Brain Observatory, Visual Coding (2-photon calcium imaging).** Public, no authentication
required (AllenSDK `BrainObservatoryCache`), Allen Institute Terms of Use. Mouse (*Mus musculus*),
GCaMP6f indicator. We use the `natural_scenes` stimulus (118 unique grayscale natural images,
presented for ~250 ms each, ~50 repeats/image per session) across 6 visual areas — VISp, VISl,
VISal, VISpm, VISam, VISrl — with 3 sessions (distinct mice) per area, 18 sessions total.

**Response processing.** Mean ΔF/F over a 7-frame window beginning 2 frames after stimulus onset
(accounts for GCaMP6f rise latency); trial-averaged per image (blank/gray-screen trials excluded);
per-cell split-half reliability with Spearman-Brown correction. All choices are recorded in
`configs/dataset.yaml` and `data/manifests/neural_data_manifest.json`, including a SHA-256 hash of
every stimulus image and of the full stimulus set, so stimulus identity is auditable rather than
assumed.

**Sample size.** 18 sessions, mean ~150 neurons/session (range 38–269), 118 images/session. This
is a small sample for subject-level statistics; every confidence interval reported below reflects
that (n=3 sessions/area).

## 4. Artificial Models

| Model | Family | Objective | Weights |
|---|---|---|---|
| Raw pixels (32×32 downsample) | classical | none | — |
| Gabor filter bank (4 orientations × 3 frequencies, 8×8-pooled) | classical | none | — |
| ResNet-18 | CNN, supervised | ImageNet-1k classification | torchvision IMAGENET1K_V1 (69.76% top-1) |
| ResNet-50 | CNN, supervised | ImageNet-1k classification | torchvision IMAGENET1K_V2 (80.86% top-1) |
| ViT-B/16 | ViT, supervised | ImageNet-1k classification | torchvision IMAGENET1K_V1 (81.07% top-1) |
| DINO ViT-S/16 | ViT, self-supervised | self-distillation (no labels) | facebookresearch/dino, dino_deitsmall16_pretrain |

For each CNN we extract `layer1–layer4` and `avgpool`; for each ViT, transformer blocks 2, 5, 8,
11. Each layer's output is global-average-pooled (CNN spatial maps) or mean-pooled over tokens
(ViT) to one feature vector per image — deliberately collapsing spatial/token structure to keep
the encoding-model comparison about representational content, not mapping flexibility.
ImageNet top-1 accuracies above are read directly from torchvision's weight metadata, not
estimated or fabricated.

## 5. Encoding Models

Ridge regression, one model per neuron, features shared across neurons in a session.
5-fold outer cross-validation split **by stimulus image**, never by trial (repeats of the same
image never span train/test). `RidgeCV` (`sklearn`) selects each neuron's regularization strength
using only the outer-training fold (generalized cross-validation). Features are standardized and
PCA-reduced to 50 components, fit on the training fold only. The mean-response baseline scores 0
by convention for every neuron (a constant prediction has undefined Pearson correlation with a
non-constant target; scoring it exactly 0 is the honest way to state "explains no image-to-image
variance").

## 6. Representational Similarity

RDMs use 1 − Pearson r between image-response vectors. RSA compares the Spearman rank correlation
of RDM upper triangles, bootstrapped over stimuli (1000 resamples) for a CI. Linear CKA is
reported separately — it is sensitive to representational geometry, not just distance ranks, and
the two are not treated as interchangeable anywhere in this project.

## 7. Results

### 7.1 RQ1 — can visual features predict neural responses at all?

Yes, modestly: every real feature space beats the r=0 mean-response floor for essentially every
session (n=378 session×model×layer rows above the floor). Absolute encoding scores are small
(session means from ~0.05 to ~0.35 depending on area/model), consistent with linear encoding of
trial-averaged single-session mouse calcium responses — this is expected, not a failure of the
pipeline (see Methods discussion of linear-mapping choice).

### 7.2 RQ2/H1 — do pretrained deep models beat simple baselines? **H1 not supported.**

| Model | Mean r (± across 18 sessions) |
|---|---|
| Raw pixels | **0.199** (± 0.161) |
| Gabor filter bank | 0.110 (± 0.052) |
| ResNet-18 | 0.121 (± 0.051) |
| ResNet-50 | 0.112 (± 0.050) |
| ViT-B/16 | 0.108 (± 0.060) |
| DINO ViT-S/16 | 0.109 (± 0.050) |
| Mean-response baseline | 0.000 (by convention) |

Raw pixels had the *highest* mean encoding score of any feature space, and the highest
variance across sessions. Paired by session, the best pretrained deep model beat pixels in only
8/18 sessions (mean difference −0.069, deep model minus pixels). This is not driven by a
numerical artifact: the strongest single pixel result (VISrl, one session, r=0.725 averaged over
134 neurons) came from a session with high measured reliability (mean split-half r=0.70) and 121/134
neurons individually predicted at r>0.5 — a real, reliable effect, not noise fit by an
overly-flexible model (PCA-50 ridge is deliberately low-flexibility). We interpret this as evidence
that population activity in some mouse visual-cortex sessions tracks low-level luminance/contrast
statistics more strongly than object-recognition-optimized deep features — consistent with, and
sharper than, prior reports (Shi, Shea-Brown & Buice 2019; Nayebi et al.) that ImageNet-CNNs are a
comparatively poor model of mouse visual cortex.

### 7.3 RQ3/H2 — does alignment vary by area, and does layer depth track cortical hierarchy?

See the [Brain Areas](/brain) and [Layers](/layers) pages for the full area×model and
layer×area breakdowns (Figure `layer_depth_curves.png`). Area-to-area variation in which model
wins is large (see §7.5) — itself evidence against a single stable "best model" story. Layer
depth does **not** track cortical hierarchy in the way a primate-ventral-stream story would
predict: for every deep model (ResNet-18/50, ViT-B/16, DINO ViT-S/16), mean encoding score
*decreases monotonically or near-monotonically with layer depth for essentially every area at
once* — VISp, VISl, VISal, VISpm, VISam, and VISrl all prefer earlier layers over later ones, with
roughly parallel curves rather than each area peaking at a different depth. This is a stronger and
more specific null than "no correspondence": it is not that different mouse areas prefer different
depths (which would still support a hierarchy, just one not aligned to network depth), but that
*all* areas uniformly prefer low-level, early-layer features — consistent with H2's expectation
that mouse cortex's known flatter functional hierarchy (Siegle et al. 2021) would not reproduce a
clean primate-like early-layer/early-area correspondence, and consistent with the pixel-baseline
result in §7.2.

### 7.4 RQ4/H6 — does ImageNet accuracy predict neural alignment? (exploratory, n=3)

Across the three supervised models with known ImageNet top-1 accuracy (ResNet-18 69.8%, ResNet-50
80.9%, ViT-B/16 81.1%), mean encoding score is *inversely* related to accuracy (ResNet-18 highest
alignment, ViT-B/16 lowest; Spearman ρ=−1.0). With n=3 this is not a confirmatory statistic — it
is mechanically bound to ±1 — but the direction is opposite to a naive "better classifier, more
brain-like" expectation, and consistent with the H1 result above.

### 7.5 RQ5/H4 — ranking stability across metrics, areas, and sessions (primary result). **H4 supported.**

Kendall's W and pairwise Spearman agreement are computed for three groupings: model rankings
across the 6 brain areas, across the 18 individual sessions, and across the 3 similarity metrics
(ridge encoding, RSA, CKA) pooled over sessions. All three are far below the W < 0.7 threshold set
in advance (`configs/hypotheses.yaml`) as informal support for instability:

| Grouping | Kendall's W | Mean pairwise Spearman ρ | Range |
|---|---|---|---|
| Across 6 brain areas | **0.124** | −0.051 | [−0.886, 0.829] |
| Across 18 sessions | **0.087** | 0.034 | [−1.0, 1.0] |
| Across 3 metrics (ridge/RSA/CKA) | **0.263** | −0.105 | [−0.771, 0.600] |

The metric-level disagreement is the sharpest illustration: **RSA ranks the deep models above
pixels** (mean ρ: DINO ViT-S/16 0.063, ResNet-18 0.061, ResNet-50 0.046, ViT-B/16 0.039, pixels
0.030, Gabor −0.006) — the *opposite* ordering from the ridge-encoding result in §7.2, where pixels
led every deep model. The same six feature spaces, the same 18 sessions, the same stimuli — two
different similarity metrics disagree about which representation is "most brain-like." This
directly replicates, in mouse Allen Observatory data, the metric-fragility finding Soni et al.
(2024) and Bo et al. (2024) reported for primate/general vision alignment, and is this project's
primary empirical contribution (see §10).

### 7.4b RQ4/H6 confirmation and H3

The inverse ImageNet-accuracy relationship in §7.4 holds using the full ranking-stability report:
ResNet-18 (69.8% top-1) mean r=0.121, ResNet-50 (80.9%) r=0.112, ViT-B/16 (81.1%) r=0.108 — Spearman
ρ=−1.0 (n=3, mechanically bound, not confirmatory). For H3, self-supervised DINO ViT-S/16
(mean r=0.109) was statistically indistinguishable from its matched-architecture supervised
counterpart ViT-B/16 (mean r=0.108) — a self-supervised objective matched supervised
ImageNet-classification performance for mouse neural alignment, supporting H3.

### 7.6 RQ6/H5 — compression. **H5 supported.**

Restricted to the four learned model families (ResNet-18/50, ViT-B/16, DINO ViT-S/16; one
representative session per area, 3 PCA dimensionalities — see `scripts/05_run_compression.py`
docstring for this scope-management choice) — mean encoding score is nearly flat as PCA
dimensionality drops from 64 to 4 components, while variance retained drops sharply:

| PCA components | Mean encoding r | Mean variance retained |
|---|---|---|
| 64 | 0.071 | 95.2% |
| 16 | 0.067 | 72.6% |
| 4 | 0.054 | 43.7% |

Going from 64 to 4 components (a 16x reduction in stored dimensionality, and from 95% to 44% of
variance retained) cost only 0.017 mean encoding-r — a small fraction of the already-modest
encoding scores in §7.2. This supports H5: most of the (modest) neural predictivity these
activations carry is concentrated in a low-dimensional subspace, echoing Cowley et al. (2026)'s
parameter-compression result in macaque cortex, but via activation-dimensionality compression on
mouse data — a different compression mechanism, consistent conclusion.

## 8. Failure Analysis

See the site's [Failures](/failures) page for the frozen numbers: the fraction of neurons no
model predicts above r=0.1 in any layer, the stimuli with the lowest out-of-fold
population-pattern correlation under each session's best model, and where area-level rankings
disagree most sharply.

## 9. Limitations

- **Species.** Mouse visual cortex has a flatter functional hierarchy than primate ventral
  stream; none of these findings generalize to primate or human claims.
- **Sample size.** n=3 sessions/area; subject-level confidence intervals are wide, and per-neuron
  distributions are reported descriptively only (neurons within a session are not independent
  subjects).
- **Linear encoding choice.** Ridge regression on a 50-component PCA basis is a deliberately
  low-flexibility mapping; a more flexible (e.g., non-linear or higher-dimensional linear) mapping
  could change the ranking between deep-model layers, though it would not explain away the
  pixel-baseline result, which used the identical mapping.
- **RQ4/H6 statistical power.** Only 3 supervised models have known ImageNet accuracy;
  Spearman ρ with n=3 is directional, not confirmatory.
- **No semantic stimulus categories.** AllenSDK's natural_scenes metadata has no category labels,
  so stimulus-level failure analysis is per-image, not per-category.
- **Single stimulus set.** All results use natural_scenes; gratings/movies stimuli in the same
  dataset were not analyzed and may show different area/model patterns.

## 10. Conclusion

In mouse visual cortex, under a linear ridge-encoding methodology, pretrained ImageNet-optimized
deep networks did not outperform a deliberately simple raw-pixel baseline, and higher ImageNet
accuracy did not predict higher neural alignment — both consistent with, and sharper than, prior
reports for this species. Whichever model is called "best" depends materially on area, session,
and similarity metric. We take that instability, not a leaderboard entry, to be this project's
main empirical contribution.

## References

- Bo, W., Soni, A., Srivastava, S., & Khosla, M. (2024). *On the Fragility of Brain-Similarity
  Metrics for Deep Neural Networks*. arXiv:2411.14633.
- Cadena, S. A., Denfield, G. H., Walker, E. Y., Gatys, L. A., Tolias, A. S., Bethge, M., &
  Ecker, A. S. (2019). Deep convolutional models improve predictions of macaque V1 responses to
  natural images. *PLOS Computational Biology*, 15(4).
- Cadieu, C. F., Hong, H., Yamins, D. L. K., Pinto, N., Ardila, D., Solomon, E. A., Majaj, N. J.,
  & DiCarlo, J. J. (2014). Deep neural networks rival the representation of primate IT cortex for
  core visual object recognition. *PLOS Computational Biology*, 10(12).
- Cowley, B. R., Stan, P. L., Pillow, J. W., & Smith, M. A. (2026). Compact deep neural network
  models of the visual cortex. *Nature*.
- Nayebi, A., Sagastuy-Brena, J., Bear, D. M., Kar, K., Kubilius, J., Ganguli, S., Sussillo, D.,
  DiCarlo, J. J., & Yamins, D. L. K. Mouse visual cortex as a limited resource system that
  self-learns an ecologically-general representation. (bioRxiv / PLOS Computational Biology.)
- Schrimpf, M., Kubilius, J., Hong, H., Majaj, N. J., Rajalingham, R., Issa, E. B., Kar, K.,
  Bashivan, P., Prescott-Roy, J., Geiger, F., Schmidt, K., Yamins, D. L. K., & DiCarlo, J. J.
  (2018/2020). Brain-Score: Which Artificial Neural Network for Object Recognition is most
  Brain-Like? *bioRxiv* (2018); integrative benchmarking framework further described in
  *Neuron* (2020).
- Sexton, N. J., & Love, B. C. (2022). Reassessing hierarchical correspondences between brain and
  deep networks through direct interface. *Science Advances*, 8(28).
- Shi, J., Shea-Brown, E., & Buice, M. (2019). Comparison Against Task Driven Artificial Neural
  Networks Reveals Functional Properties of Mouse Visual Cortex. *NeurIPS 2019*.
- Siegle, J. H., Jia, X., Durand, S., et al. (2021). Survey of spiking in the mouse visual system
  reveals functional hierarchy. *Nature*, 592, 86–92.
- Soni, A., Srivastava, S., Kording, K., & Khosla, M. (2024). Conclusions about Neural Network to
  Brain Alignment are Profoundly Impacted by the Similarity Measure. *bioRxiv*.
- Yamins, D. L. K., Hong, H., Cadieu, C. F., Solomon, E. A., Seibert, D., & DiCarlo, J. J. (2014).
  Performance-optimized hierarchical models predict neural responses in higher visual cortex.
  *PNAS*, 111(23), 8619–8624.

These citations were verified by an independent literature-review pass (see
`docs/novelty_memo.md`) via web search against publisher/preprint listings; none were generated
from memory alone. Exact venue/year details for the Nayebi et al. and Schrimpf et al. entries
reflect what could be confirmed at review time (bioRxiv preprint vs. later peer-reviewed venue) —
treat page/volume numbers not given above as unconfirmed rather than omitted by accident.

---

*Data, code, and frozen results: see the project repository. Reproduction commands in
`README`/`Makefile`. This paper and all numbers in it are generated from the canonical result
files under `results/` and `data/manifests/` — see each file's `git_sha` field for exact
provenance.*
