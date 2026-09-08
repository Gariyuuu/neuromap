import { loadJSON, AreaModelHeatmapRow, MODEL_LABELS } from "@/lib/data";
import { Panel, EmptyState, Pill } from "@/components/ui";
import { ModelBarChart } from "@/components/charts/ModelBarChart";
import { ChartDescription } from "@/components/ui";

export default function ModelsPage() {
  const rows = loadJSON<AreaModelHeatmapRow[]>("brain_area_model_heatmap.json");
  const ranking = loadJSON<Record<string, any>>("ranking_stability.json");

  let chartData: { model: string; family: string; mean: number; std: number }[] = [];
  if (rows) {
    const byModel = new Map<string, number[]>();
    const familyOf = new Map<string, string>();
    for (const r of rows) {
      if (!byModel.has(r.model)) byModel.set(r.model, []);
      byModel.get(r.model)!.push(r.mean);
      familyOf.set(r.model, r.family ?? "");
    }
    chartData = Array.from(byModel.entries()).map(([model, vals]) => {
      const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
      const variance = vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length;
      return { model, family: familyOf.get(model) ?? "", mean, std: Math.sqrt(variance) };
    });
  }

  const rq4 = ranking?.imagenet_accuracy_vs_neural_alignment;
  const h3 = ranking?.self_supervised_vs_supervised_vit;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-medium">Artificial-model comparison</h1>
        <p className="text-dim text-sm mt-1 max-w-2xl">
          RQ2: which model family aligns most with mouse visual cortex? Averaged across all 6
          visual areas, best layer per session.
        </p>
      </div>

      {!rows ? (
        <EmptyState what="Model comparison data" />
      ) : (
        <Panel title="Mean encoding score by model, averaged across brain areas">
          <ChartDescription>
            Bar chart of mean ridge-encoding correlation per artificial model, averaged across
            six mouse visual areas, with standard-deviation error bars across areas.
          </ChartDescription>
          <ModelBarChart data={chartData} />
          <div className="flex gap-2 mt-3 flex-wrap">
            <Pill color="#8b96a3">classical baseline</Pill>
            <Pill color="#6ea8fe">CNN, supervised</Pill>
            <Pill color="#c792ea">ViT, supervised</Pill>
            <Pill color="#3ecf8e">self-supervised</Pill>
          </div>
        </Panel>
      )}

      {rq4 && (
        <Panel title="RQ4 — does ImageNet accuracy predict neural alignment?" subtitle="Exploratory, n=3 supervised models — underpowered">
          <p className="text-sm text-dim">
            Spearman ρ = <span className="font-mono text-fg">{rq4.spearman_rho.toFixed(3)}</span>,
            p = <span className="font-mono text-fg">{rq4.p_value.toFixed(3)}</span> across{" "}
            {rq4.models.map((m: string) => MODEL_LABELS[m] ?? m).join(", ")}.
          </p>
          <p className="text-xs text-dimmer mt-2">
            With only 3 supervised models, this is a directional check, not a confirmatory test —
            see configs/hypotheses.yaml (H6).
          </p>
        </Panel>
      )}

      {h3 && (
        <Panel title="H3 — self-supervised vs. supervised, matched ViT architecture">
          <p className="text-sm text-dim">
            DINO ViT-S/16 (self-supervised): r ={" "}
            <span className="font-mono text-fg">{h3.self_supervised_dino_vits16_mean_r.toFixed(4)}</span>
            {" "}&nbsp;·&nbsp; ViT-B/16 (supervised, ImageNet): r ={" "}
            <span className="font-mono text-fg">{h3.supervised_vit_b_16_mean_r.toFixed(4)}</span>
          </p>
        </Panel>
      )}
    </div>
  );
}
