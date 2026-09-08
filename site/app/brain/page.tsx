import { loadJSON, AreaModelHeatmapRow, Overview } from "@/lib/data";
import { Panel, EmptyState } from "@/components/ui";
import { Heatmap } from "@/components/Heatmap";

export default function BrainPage() {
  const rows = loadJSON<AreaModelHeatmapRow[]>("brain_area_model_heatmap.json");
  const overview = loadJSON<Overview>("overview.json");

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-medium">Brain-area results</h1>
        <p className="text-dim text-sm mt-1 max-w-2xl">
          Mean held-out ridge-encoding correlation (best layer per model), per visual area,
          averaged across 3 sessions/mice. RQ3: does alignment vary by cortical area?
        </p>
      </div>

      {!rows || !overview ? (
        <EmptyState what="Brain-area heatmap data" />
      ) : (
        <Panel title="Area × model mean encoding score (Pearson r, held-out images)">
          <Heatmap
            rows={overview.areas}
            cols={overview.models.filter((m) => m !== "mean_baseline")}
            valueOf={(area, model) => rows.find((r) => r.area === area && r.model === model)?.mean ?? null}
            countOf={(area, model) => rows.find((r) => r.area === area && r.model === model)?.count ?? null}
          />
          <p className="text-xs text-dimmer mt-4">
            Cells show mean r across 3 sessions per area, using each model&apos;s best-performing
            layer for that session. Scores are modest in absolute terms — expected for linear
            encoding of single-trial-averaged mouse calcium responses, and consistent with prior
            findings that mouse visual cortex is harder for ImageNet-trained CNNs to predict than
            primate ventral stream (see <a href="/methods" className="underline text-model">Methods</a>).
          </p>
        </Panel>
      )}
    </div>
  );
}
