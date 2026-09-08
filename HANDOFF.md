# NeuroMap — Handoff

Status as of this writing: **encoding complete, RSA/CKA and compression running in background,
ranking-stability/failure-analysis/site-export/figures/release pending on those.** If you're
picking this up mid-run, check `results/rsa/rsa_results.parquet` and
`results/compression/compression_results.parquet` for row counts against the 18-session (RSA) /
6-session (compression, deliberately reduced scope — see script docstring) targets before
re-running anything; both scripts are resumable and skip completed rows.

## What's definitely done and correct

- Dataset: 18 Allen Brain Observatory Visual Coding (2P) sessions across 6 mouse visual areas,
  natural_scenes stimulus, cached under `data/raw/allen_boc` (gitignored, ~6GB).
- Neural response processing + reliability: `data/processed/neural/<area>/<exp_id>_{trials,images,reliability}.parquet`,
  manifest at `data/manifests/neural_data_manifest.json` (stimulus hashes, preprocessing config).
- Feature extraction: 6 model families x layers, cached in `features/*.npy` + provenance JSON.
- **Canonical encoding run** (`results/encoding/encoding_results.parquet`, 378 rows): complete.
  Headline finding — raw pixels (mean r=0.199) beat every pretrained deep model (0.108–0.121).
  See `paper/paper.md` §7.2 for the full writeup and the reliability check that rules out a
  numerical artifact.
- 31/31 tests pass (`make test`); CI configured (`.github/workflows/ci.yml`).
- Next.js site scaffolded (10 routes + methods/paper), builds cleanly (`npm run build` in `site/`),
  lab-atlas dark theme, accessible (non-color-only) charts.
- GitHub repo live: https://github.com/Gariyuuu/neuromap (public, pushed through the
  "add activation-extraction tests" commit).

## What's still pending when you resume

1. **Wait for or resume** `scripts/04_run_rsa.py` (RSA + CKA, all 18 sessions, 1000 bootstraps —
   slow, ~10 min/session) and `scripts/05_run_compression.py` (deliberately scoped to 1
   session/area x 3 PCA dims x 4 deep models, to keep runtime sane — see its docstring for how to
   widen scope later).
2. Run `scripts/06_ranking_stability.py` (needs both encoding + RSA results) — this produces the
   project's primary result (RQ5/H4 cross-metric ranking stability).
3. Run `scripts/07_failure_analysis.py` (needs encoding results only, can run any time).
4. Run `scripts/08_make_figures.py` again (regenerates `rsa_matrix.png`, `ranking_uncertainty.png`,
   `compression_curves.png`, `stimulus_residuals.png` with complete data — it was run once
   already with partial data to sanity-check the pipeline).
5. Run `scripts/09_build_site_data.py` to export `site/public/data/*.json` from final results,
   then `cp data/processed/stimuli/natural_scenes/*.png site/public/stimuli/` if not already done.
6. Fill in the two `[Numbers below are filled in from ...]` placeholders in `paper/paper.md`
   (§7.5 ranking stability, §7.6 compression) with the real frozen numbers.
7. `npm run build` in `site/` to verify, then deploy: `cd site && vercel --prod` (new Vercel
   project defaults to an SSO wall — disable it in the project's Deployment Protection settings
   before sharing the URL, per this account's usual gotcha).
8. `make release` (`scripts/11_freeze_release.py`) to write `results/RELEASE.json`.
9. Commit + push everything, including `site/public/data/*.json` and `site/public/stimuli/*.png`
   (NOT gitignored — they're the site's build input, needed for a plain `vercel --prod` from a
   fresh clone to reproduce the deployed site without re-running the Python pipeline).
10. Final report to the user: dataset/models/areas/sample sizes, the pixel-beats-deep-nets
    finding, RQ5 ranking-stability numbers, RQ6 compression curve, tests, site URL, repo URL,
    limitations, reproduction commands — per the original task's "FINAL RESPONSE" spec.

## Gotchas hit and fixed during this build

- `pkg_resources` removed from modern `setuptools` (>=81) — AllenSDK needs `setuptools<81`.
- AllenSDK 2.16.2 + modern `statsmodels` breaks (`multipletests` import path moved) — pin
  `statsmodels==0.13.5`. AllenSDK otherwise tolerates a newer `numpy`/`pandas` than its declared
  upper bound (pip warns but everything works).
- Allen's `api.brain-map.org` file-download endpoint times out intermittently on large NWB
  files — `scripts/01_fetch_neural_data.py` retries each session up to 5x and is safe to just
  re-run to pick up where it left off.
- `lib/data.ts`'s `fs`/`path` imports leak into the client bundle if a `"use client"` component
  imports anything from that file — constants (`MODEL_LABELS`, types) live in `lib/constants.ts`
  instead; only server components import `lib/data.ts` directly.
- Compression experiment's naive full grid (20 feature-sets x 6 dims x 18 sessions = 2160
  `run_encoding` calls) was projected at several hours — rescoped to deep-model-only families,
  3 dims, 1 session/area (see script docstring for how to widen it back up).
