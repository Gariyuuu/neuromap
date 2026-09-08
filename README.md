# NeuroMap

**Auditing the alignment between artificial visual representations and measured mouse
visual-cortex responses.** A small-scale, pre-registered research artifact — not a clinical tool,
not a claim that any model "thinks like" the brain, not a replacement for Brain-Score.

See `docs/novelty_memo.md` for the literature review behind the project's scope, and
`configs/hypotheses.yaml` for the pre-registered hypotheses tested below.

## What this is

Ridge-regression encoding models predict trial-averaged calcium-imaging responses in 6 mouse
visual cortical areas (Allen Brain Observatory Visual Coding, 18 sessions, natural_scenes
stimulus) from 6 artificial vision-model feature spaces (raw pixels, Gabor filters, ResNet-18/50,
ViT-B/16, self-supervised DINO ViT-S/16). We ask whether the resulting model ranking is stable
across similarity metrics (ridge encoding, RSA, linear CKA), cortical areas, and sessions —
and report a genuine, unexpected finding: raw pixels outperformed every pretrained deep model
on mean encoding score (see `paper/paper.md` §7.2).

## Repository layout

```
configs/            dataset selection, pre-registered hypotheses
neuromap/            importable pipeline library (neural processing, features, encoding, RSA, stats)
scripts/             numbered pipeline stages (01_fetch_neural_data.py ... 09_build_site_data.py)
data/manifests/      provenance: stimulus hashes, session metadata, preprocessing config (git-tracked)
data/raw/            AllenSDK-cached NWB files (gitignored, ~6GB, regenerate with `make data`)
data/processed/      extracted neural response tables + stimulus PNGs (gitignored, regenerate)
features/            cached model activations + provenance JSON per (model, layer) (gitignored)
results/             canonical result parquets/JSON — encoding, RSA, compression, failures (git-tracked)
tests/               27 tests: encoding correctness, leakage prevention, RSA/CKA sanity, stats, provenance
site/                Next.js research site consuming compact JSON exported from results/
paper/paper.md       conference-style manuscript, rendered at the site's /paper route
docs/novelty_memo.md literature review and honest novelty framing
```

## Reproduction

```
make setup            # python3.11 venv + pip install -r requirements.txt
make data             # fetch + process 18 Allen Brain Observatory sessions (~6GB, ~20-30 min)
make features-smoke   # extract artificial-model activations for 118 stimuli (~2 min, CPU)
make benchmark        # ridge encoding models, 378 (session, model, layer) rows (~90 min, CPU)
make rsa              # RSA + linear CKA (~90-120 min, CPU, bootstrap-heavy)
make compression       # PCA compression experiment
make analyze          # ranking-stability + failure-analysis reports
make site             # export compact JSON into site/public/data/
make test             # pytest — synthetic fixtures only, no data download
```

All scripts are resumable: re-running `make data`/`make benchmark`/`make rsa`/`make compression`
skips already-completed (session, model, layer) combinations.

## Scientific integrity notes

- No fabricated recordings, subjects, neurons, brain regions, citations, correlations, noise
  ceilings, p-values, or accuracy figures. ImageNet top-1 accuracies are read directly from
  `torchvision` weight metadata (verified against the installed package, not recalled from memory).
  Literature citations were verified via web search by an independent review pass; anything not
  independently verifiable is flagged as such in `docs/novelty_memo.md`.
- Every canonical result file's provenance (`git_sha`, config hash, stimulus hash) is written by
  `neuromap/provenance.py` at generation time.
- Stimulus-response alignment is guaranteed by SHA-256 hashes of the raw stimulus images, recorded
  in `data/manifests/neural_data_manifest.json`.
- Train/test splits are by stimulus image, never by trial; PCA/scaling inside the encoding
  pipeline is fit on the training fold only (see `tests/test_leakage.py`).

## Known limitations (see `paper/paper.md` §9 for the full list)

Mouse-only (does not generalize to primate/human claims); n=3 sessions/area; linear
low-flexibility encoding by design; RQ4 exploratory with n=3 supervised models; no semantic
stimulus category labels available in AllenSDK.
