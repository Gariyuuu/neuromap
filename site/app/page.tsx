import { loadJSON, Overview } from "@/lib/data";
import { Panel, StatTile, EmptyState } from "@/components/ui";

export default function OverviewPage() {
  const overview = loadJSON<Overview>("overview.json");

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-medium tracking-tight">
          NeuroMap: Auditing the Alignment Between Artificial Visual Representations and
          Measured Visual-Cortex Responses
        </h1>
        <p className="text-dim mt-3 max-w-3xl text-sm leading-relaxed">
          NeuroMap compares how well artificial vision models predict real neural responses
          recorded in mouse visual cortex, using public 2-photon calcium-imaging data from the
          Allen Brain Observatory. It is a rigorous small-scale audit — not a claim that any
          model &ldquo;thinks like the brain.&rdquo; See <a href="/methods" className="underline text-model">
          Methods &amp; Limitations</a> for what this project does and does not support.
        </p>
      </div>

      {overview ? (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <StatTile label="Visual areas" value={String(overview.n_areas)} sub={overview.areas.join(", ")} />
            <StatTile label="Recording sessions" value={String(overview.n_sessions)} sub="3 mice per area" />
            <StatTile label="Neurons (cells)" value={overview.n_neurons_total.toLocaleString()} sub="split-half reliability computed per cell" />
            <StatTile label="Model families" value={String(overview.n_models)} sub={overview.models.join(", ")} />
          </div>

          <Panel title="Dataset" subtitle={overview.source}>
            <p className="text-sm text-dim">
              Stimulus: <span className="text-fg font-mono">{overview.stimulus}</span> — 118 unique
              grayscale natural images, ~50 repeats each, mouse visual cortex (VISp, VISl, VISal,
              VISpm, VISam, VISrl).
            </p>
          </Panel>
        </>
      ) : (
        <EmptyState what="Overview data" />
      )}

      <Panel title="Central research questions">
        <ol className="text-sm space-y-2 list-decimal list-inside text-dim">
          <li><span className="text-fg">RQ1</span> — How well can visual stimulus features predict neural responses?</li>
          <li><span className="text-fg">RQ2</span> — Which artificial vision models align most with measured visual-cortex responses?</li>
          <li><span className="text-fg">RQ3</span> — Does alignment vary by cortical area or processing stage?</li>
          <li><span className="text-fg">RQ4</span> — Does higher ImageNet accuracy imply higher neural alignment?</li>
          <li><span className="text-fg">RQ5 (primary)</span> — How stable are model rankings across metrics, areas, and sessions?</li>
          <li><span className="text-fg">RQ6</span> — Can compressed, lower-dimensional model representations retain neural predictivity?</li>
        </ol>
      </Panel>

      <Panel title="Narrowest defensible contribution" subtitle="See docs/novelty_memo.md for the full literature review">
        <p className="text-sm text-dim leading-relaxed">
          Brain-Score (Schrimpf, Kubilius, Yamins &amp; DiCarlo) already benchmarks CNN/brain
          alignment for primate ventral stream. Model-ranking instability across similarity
          metrics has separately been shown for primate data (Soni, Srivastava, Kording &amp;
          Khosla, 2024). NeuroMap extends that specific question — cross-metric,
          cross-stimulus-class ranking stability — into mouse Allen Observatory data, where it has
          not been explicitly tested, and adds an activation-compression arm distinct from prior
          architecture-compression work (Cowley, Stan, Pillow &amp; Smith).
        </p>
      </Panel>
    </div>
  );
}
