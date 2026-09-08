import { loadJSON, LayerProfileRow, MODEL_LABELS } from "@/lib/data";
import { Panel, EmptyState, ChartDescription } from "@/components/ui";
import { LayerProfileChart } from "@/components/charts/LayerProfileChart";

const LAYER_ORDER: Record<string, string[]> = {
  resnet18: ["layer1", "layer2", "layer3", "layer4", "avgpool"],
  resnet50: ["layer1", "layer2", "layer3", "layer4", "avgpool"],
  vit_b_16: ["block2", "block5", "block8", "block11"],
  dino_vits16: ["block2", "block5", "block8", "block11"],
};

export default function LayersPage() {
  const rows = loadJSON<LayerProfileRow[]>("layer_profiles.json");

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-medium">Layer explorer</h1>
        <p className="text-dim text-sm mt-1 max-w-2xl">
          RQ2/H2: do earlier visual areas align more strongly with earlier network layers? Mouse
          visual cortex has a known <em>flatter</em> functional hierarchy than primate ventral
          stream (Siegle et al. 2021) — treated here as an open empirical question, not assumed.
        </p>
      </div>

      {!rows ? (
        <EmptyState what="Layer-profile data" />
      ) : (
        <div className="grid md:grid-cols-2 gap-5">
          {Object.entries(LAYER_ORDER).map(([model, layers]) => {
            const areas = Array.from(new Set(rows.filter((r) => r.model === model).map((r) => r.area)));
            const series = areas.map((area) => ({
              area,
              values: layers.map((layer) => rows.find((r) => r.model === model && r.area === area && r.layer === layer)?.mean_r ?? null),
            }));
            if (series.length === 0) return null;
            return (
              <Panel key={model} title={MODEL_LABELS[model] ?? model} subtitle="Mean encoding r by layer, one line per visual area">
                <ChartDescription>
                  Line chart, x-axis is network depth from {layers[0]} to {layers[layers.length - 1]},
                  y-axis is mean encoding correlation, one line per mouse visual area (area names
                  shown in the legend, not encoded by color alone).
                </ChartDescription>
                <LayerProfileChart layers={layers} series={series} />
              </Panel>
            );
          })}
        </div>
      )}
    </div>
  );
}
