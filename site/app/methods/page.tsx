import { Panel } from "@/components/ui";

export default function MethodsPage() {
  return (
    <div className="space-y-6 max-w-3xl">
      <div>
        <h1 className="text-xl font-medium">Methodology &amp; limitations</h1>
        <p className="text-dim text-sm mt-1">
          Read this before interpreting any result on this site.
        </p>
      </div>

      <Panel title="What this project is">
        <p className="text-sm text-dim leading-relaxed">
          NeuroMap measures how well linear ridge-regression maps from artificial vision model
          activations predict trial-averaged calcium-imaging responses in mouse visual cortex, and
          how consistent the resulting model rankings are across similarity metrics (ridge
          encoding, RSA, linear CKA), brain areas, and recording sessions. It is a small-scale,
          honest audit — not a new benchmark suite and not a claim about how the brain works.
        </p>
      </Panel>

      <Panel title="What this project is not">
        <ul className="text-sm text-dim list-disc list-inside space-y-1.5 leading-relaxed">
          <li>Not a clinical or diagnostic tool.</li>
          <li>Not evidence that any artificial network &ldquo;thinks like&rdquo; or is
            functionally equivalent to a biological visual system — correlation between
            representations is not identity of mechanism.</li>
          <li>Not a primate or human result — mouse visual cortex has a demonstrably flatter
            functional hierarchy than the primate ventral stream (Siegle et al. 2021, <em>Nature</em>),
            so findings here should not be extrapolated across species.</li>
          <li>Not a replacement for Brain-Score (Schrimpf, Kubilius, Yamins &amp; DiCarlo) — that
            project benchmarks primate/human data at far larger scale. NeuroMap sits in
            deliberately narrower, adjacent territory (see below).</li>
        </ul>
      </Panel>

      <Panel title="Data & preprocessing">
        <p className="text-sm text-dim leading-relaxed">
          Allen Brain Observatory Visual Coding (2-photon), natural_scenes stimulus, 18 sessions
          across 6 areas (3 sessions/mice per area). Response window: mean ΔF/F over 7 frames
          starting 2 frames after stimulus onset (accounts for GCaMP6f rise latency). Trials are
          averaged per image (~50 repeats); blank/gray-screen trials excluded. Reliability is
          split-half correlation with Spearman-Brown correction. All choices are persisted in{" "}
          <code className="font-mono text-xs">configs/dataset.yaml</code> and{" "}
          <code className="font-mono text-xs">data/manifests/neural_data_manifest.json</code>, not left implicit.
        </p>
      </Panel>

      <Panel title="Encoding models">
        <p className="text-sm text-dim leading-relaxed">
          Ridge regression, one model per neuron, features shared across neurons in a session.
          5-fold outer cross-validation split by stimulus image (never by trial, so repeats of the
          same image never span train/test); RidgeCV picks the regularization strength per neuron
          using only the outer-training fold. Model features are PCA-reduced to 50 components,
          fit on the training fold only, before ridge regression — this deliberately favors a
          linear, low-flexibility mapping so encoding score differences reflect representational
          quality rather than mapping flexibility (per project design; see limitations below on
          why this may still underestimate high-dimensional layers).
        </p>
      </Panel>

      <Panel title="RSA and CKA are reported separately, not averaged">
        <p className="text-sm text-dim leading-relaxed">
          RSA compares the rank order of pairwise representational distances (Spearman ρ on
          RDM upper triangles, bootstrapped over stimuli). Linear CKA compares representational
          geometry more directly via kernel alignment. The two frequently disagree — that
          disagreement is itself part of the ranking-stability finding (RQ5), not noise to average away.
        </p>
      </Panel>

      <Panel title="Honest, pre-registered hypothesis outcomes">
        <ul className="text-sm text-dim list-disc list-inside space-y-1.5">
          <li><span className="text-fg">H1 (pretrained &gt; pixels) — NOT SUPPORTED.</span> Raw
            downsampled pixel intensities outperformed every pretrained deep model on mean
            encoding score in this dataset. This is a genuine, surprising result, consistent
            with — and stronger than — prior findings that ImageNet-optimized CNNs are a
            comparatively poor fit for mouse visual cortex (Shi, Shea-Brown &amp; Buice 2019;
            Nayebi et al.). It may partly reflect that a 32×32 pixel PCA basis captures
            low-level luminance/contrast statistics that correlate with calcium-signal amplitude
            in ways object-recognition-optimized features do not prioritize.</li>
          <li>See <a href="/models" className="underline text-model">Models</a> and{" "}
            <a href="/brain" className="underline text-model">Brain Areas</a> for H2–H6 outcomes
            with numbers, and <code className="font-mono text-xs">configs/hypotheses.yaml</code> for
            the full pre-registration.</li>
        </ul>
      </Panel>

      <Panel title="Multiple comparisons & aggregation">
        <p className="text-sm text-dim leading-relaxed">
          Neurons within a session are not independent subjects — per-neuron distributions are
          reported descriptively only. Confirmatory claims (hypothesis tests, CIs meant to
          generalize) are computed at the session/subject level (n=3 per area), with
          Benjamini-Hochberg FDR correction where multiple comparisons are run across areas or
          models. n=3 sessions/area gives wide confidence intervals; this is stated wherever a CI
          is shown, not hidden.
        </p>
      </Panel>

      <Panel title="Reproduction">
        <pre className="text-xs font-mono text-dim overflow-x-auto">{`make setup            # create venv, install deps
make data             # fetch + process Allen Brain Observatory sessions
make features-smoke   # extract artificial-model activations (118 stimuli, fast)
make benchmark        # run ridge encoding models (main result)
make rsa              # RSA + CKA
make compression       # activation-compression experiment
make analyze          # ranking stability + failure analysis
make site             # export compact JSON for this website
make test             # 27 unit/leakage/sanity tests`}</pre>
      </Panel>
    </div>
  );
}
