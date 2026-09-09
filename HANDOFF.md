# NeuroMap — Handoff

**Status: complete.** All canonical results are frozen (`results/RELEASE.json`, git SHA
`d6b2f61`), the paper has no placeholders left, the site is deployed and public, and
31/31 tests pass. See the final report delivered to the user for the full scientific summary.

## What's done

- Dataset: 18 Allen Brain Observatory Visual Coding (2P) sessions across 6 mouse visual areas,
  natural_scenes stimulus, cached under `data/raw/allen_boc` (gitignored, ~6GB — `make data`
  regenerates it, resumable).
- Neural response processing + reliability, with full provenance (stimulus hashes, preprocessing
  config) in `data/manifests/neural_data_manifest.json`.
- Feature extraction: 6 model families x layers (20 feature sets total), cached + provenance JSON.
- Canonical encoding (360 rows), RSA/CKA (360 rows, 18/18 sessions), compression (324 rows,
  6 sessions x 18 deep-model feature-sets x 3 PCA dims — see `scripts/05_run_compression.py`
  docstring for how to widen this scope), ranking-stability (`results/encoding/ranking_stability.json`),
  and failure analysis — all complete.
- **Primary result (RQ5/H4):** model rankings for neural alignment are NOT stable — Kendall's W =
  0.087-0.263 across areas/sessions/metrics (pre-registered instability threshold was W<0.7). RSA
  ranks deep nets above pixels; ridge encoding ranks pixels above every deep net, on the same data.
- **Secondary finding:** raw pixels beat every pretrained deep model on ridge-encoding score
  (0.199 vs. 0.077-0.121), and ImageNet accuracy is inversely related to encoding score (n=3,
  exploratory). Both replicate prior mouse-cortex literature (Shi et al. 2019, Nayebi et al.).
- 31/31 tests pass (`make test`); CI green (`.github/workflows/ci.yml`).
- Next.js site (13 routes) builds and deploys cleanly: **https://site-two-nu-61.vercel.app**
  (public, no auth wall — verified with a plain `curl`).
- GitHub repo: **https://github.com/Gariyuuu/neuromap** (public), pushed through commit `d6b2f61`.
- `paper/paper.md` has all numbers filled in from frozen result files (no placeholders).

## If you pick this up later

Everything is reproducible from a clean checkout via `make setup data features-smoke benchmark
rsa compression analyze figures site` (see README for per-step time estimates), then
`cd site && vercel --prod` to redeploy (project is already linked in this account under the name
"site"; a fresh `vercel` login would need to re-link via `vercel link`).

Natural directions to extend, not started: the `drifting_gratings`/`static_gratings`/
`natural_movie_*` stimuli in the same Allen dataset (only `natural_scenes` was analyzed), a
non-linear encoding-model robustness check, and widening the compression experiment's scope
(currently 1 session/area x 3 dims x deep-model-only, by design — see script docstring).

## Gotchas hit and fixed during this build

- `pkg_resources` removed from modern `setuptools` (>=81) — AllenSDK needs `setuptools<81`.
- AllenSDK 2.16.2 + modern `statsmodels` breaks (`multipletests` import path moved) — pin
  `statsmodels==0.13.5`. AllenSDK otherwise tolerates a newer `numpy`/`pandas` than its declared
  upper bound (pip warns but everything works).
- Allen's `api.brain-map.org` file-download endpoint times out intermittently on large NWB
  files — `scripts/01_fetch_neural_data.py` retries each session up to 5x and is safe to just
  re-run to pick up where it left off.
- Running `04_run_rsa.py` and `05_run_compression.py` concurrently on the same 8-core machine
  roughly doubled both scripts' wall-clock time (BLAS/sklearn thread contention) — running them
  sequentially was faster in wall-clock terms despite less "parallelism."
- `lib/data.ts`'s `fs`/`path` imports leak into the client bundle if a `"use client"` component
  imports anything from that file — constants (`MODEL_LABELS`, types) live in `lib/constants.ts`
  instead; only server components import `lib/data.ts` directly.
