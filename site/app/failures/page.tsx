import { loadJSON } from "@/lib/data";
import { Panel, EmptyState, StatTile } from "@/components/ui";

type FailureReport = {
  poor_neuron_threshold_r: number;
  overall_fraction_poorly_predicted: number;
  hardest_stimuli_frame_ids: number[];
  hardest_stimuli_mean_pattern_r: number[];
  note: string;
};

export default function FailuresPage() {
  const report = loadJSON<FailureReport>("failure_analysis.json");
  const ranking = loadJSON<any>("ranking_stability.json");

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-medium">Failure analysis</h1>
        <p className="text-dim text-sm mt-1 max-w-2xl">
          Where does every model fail, and where do models disagree with each other?
        </p>
      </div>

      {!report ? (
        <EmptyState what="Failure analysis" />
      ) : (
        <>
          <div className="grid grid-cols-2 gap-3">
            <StatTile
              label="Neurons no model predicts well"
              value={`${(report.overall_fraction_poorly_predicted * 100).toFixed(1)}%`}
              sub={`threshold r < ${report.poor_neuron_threshold_r}, across all models/layers`}
            />
            <StatTile label="Hardest stimulus IDs" value={report.hardest_stimuli_frame_ids.slice(0, 5).join(", ")} sub="lowest population-pattern correlation, held-out" />
          </div>

          <Panel title="Hardest stimuli (lowest out-of-fold population-pattern correlation)">
            <div className="scrollx">
              <table className="text-sm">
                <thead><tr><th>Frame ID</th><th>Mean pattern r (across sessions)</th></tr></thead>
                <tbody>
                  {report.hardest_stimuli_frame_ids.map((id, i) => (
                    <tr key={id}>
                      <td className="font-mono">scene_{String(id).padStart(3, "0")}</td>
                      <td className="font-mono">{report.hardest_stimuli_mean_pattern_r[i]?.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </>
      )}

      {ranking?.ranking_stability_across_areas_encoding && (
        <Panel title="Where do brain areas disagree on model ranking?" subtitle="Kendall's W across areas (ridge-encoding metric)">
          <p className="text-sm text-dim">
            Kendall&apos;s W = <span className="font-mono text-fg">{ranking.ranking_stability_across_areas_encoding.kendalls_w.toFixed(3)}</span>
            {" "}(1.0 = perfect agreement, 0 = chance). Mean pairwise Spearman ={" "}
            <span className="font-mono text-fg">{ranking.ranking_stability_across_areas_encoding.mean_pairwise_spearman.toFixed(3)}</span>.
          </p>
        </Panel>
      )}

      {ranking?.ranking_stability_across_metrics && (
        <Panel title="Models with strong ridge-encoding score but weak RSA/CKA (or vice versa)" subtitle="Full detail on the Models page and results/encoding/ranking_stability.json">
          <p className="text-sm text-dim">
            Cross-metric Kendall&apos;s W (ridge encoding vs. RSA vs. CKA) ={" "}
            <span className="font-mono text-fg">{ranking.ranking_stability_across_metrics.kendalls_w.toFixed(3)}</span>.
            Low agreement here is itself the primary finding of this project (RQ5) — see{" "}
            <a href="/methods" className="underline text-model">Methods</a>.
          </p>
        </Panel>
      )}
    </div>
  );
}
