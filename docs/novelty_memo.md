# Novelty Memo

## What already exists

**Brain-Score** (Schrimpf, Kubilius, Yamins, DiCarlo et al., 2018 bioRxiv / 2020 *Neuron*) is a
composite leaderboard scoring ANNs on ~33 neural + behavioral benchmarks spanning **macaque**
V1/V2/V4/IT and human object-recognition psychophysics, using linear/PLS encoding plus behavioral
consistency. It does **not** cover: mouse visual cortex, 2-photon calcium imaging, RSA/CKA as
primary metrics, systematic cross-metric agreement analysis, or activation compression. NeuroMap
sits outside its scope but reuses its encoding-model logic.

**Foundational alignment results:** Yamins, Hong, Cadieu & DiCarlo (2014, *PNAS*) — categorization-
optimized CNNs predict macaque V4/IT, performance correlates with brain-predictivity. Cadieu et al.
(2014, *PLOS Comp Bio*) — deep nets match IT population geometry. Cadena et al. (2019, *PLOS Comp
Bio*) — CNN features beat classical LN/wavelet models for macaque V1.

**Ranking (in)stability across metrics — already studied for primate/general vision, NOT novel as a
general claim.** Soni, Srivastava, Kording & Khosla (2024, bioRxiv) show metric choice changes
layer-area correspondence and model rankings ("fragile foundations"). Bo, Soni, Srivastava & Khosla
(2024, arXiv:2411.14633) find 9 similarity measures × 10 behavioral metrics × 19 models frequently
disagree. Sexton & Love (2022, *Science Advances*) show a stricter test overturns standard
hierarchical layer-area correspondence claims.

**Compression vs. predictivity — partially studied, not on mouse data.** Cowley, Stan, Pillow &
Smith (2026, *Nature*) compress a 60M-parameter DNN ~5000x while retaining macaque V1/V4/IT
predictivity — but this compresses model *parameters/architecture*, not activation
dimensionality via PCA for a linear encoding model, and it is primate.

**Mouse-specific hierarchy — the load-bearing prior work.** Siegle et al. (2021, *Nature*, Allen
Institute Neuropixels survey) find a real but much *flatter* functional hierarchy in mouse visual
cortex than the primate ventral stream. Shi, Shea-Brown & Buice (NeurIPS 2019, using Allen Brain
Observatory) and Nayebi et al. (Yamins lab) both find standard ImageNet-supervised CNNs are a poor
fit for mouse cortex — best-ImageNet-accuracy does not track best-brain-similarity in mice.

## Novelty verdict

Both "ranking instability across metrics" and "compression vs. predictivity" have prior art, but
**not combined, and not on mouse Allen Observatory data with a multi-metric (RSA + CKA + ridge
encoding) design across natural-scenes / gratings / movies stimulus classes.** No paper does this
specific cross-metric ranking-stability audit on mouse calcium-imaging data.

## Chosen contribution (honest framing)

**Primary (RQ5):** a cross-metric / cross-stimulus-class ranking-stability audit on Allen Brain
Observatory mouse data — do CNN vs. ViT vs. self-supervised model rankings for neural alignment
agree across RSA, CKA, and ridge-encoding, and across stimulus classes, given mouse cortex's known
flat hierarchy? This extends Soni/Khosla's primate finding into an untested regime rather than
discovering a new phenomenon.

**Secondary/exploratory (RQ6):** PCA-compressed activations vs. ridge predictivity, framed
explicitly as "extends a primate finding (Cowley et al.) into mouse territory with a different
compression mechanism (activation dimensionality, not model parameters)" — not as untouched ground.

This project is **a rigorous small-scale replication/extension of a known metric-fragility problem,
moved into the mouse/Allen-Observatory setting where it has not been explicitly tested** — not a
new-phenomenon discovery. State this plainly in the paper's Related Work and Limitations sections.
