import { loadJSON, CompressionRow, MODEL_LABELS } from "@/lib/data";
import { Panel, EmptyState, ChartDescription } from "@/components/ui";
import { CompressionChart } from "@/components/charts/CompressionChart";

export default function CompressionPage() {
  const rows = loadJSON<CompressionRow[]>("compression_results.json");

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-medium">Compression experiment</h1>
        <p className="text-dim text-sm mt-1 max-w-3xl">
          RQ6/H5: how much representational complexity is necessary to preserve brain alignment?
          Model activations are progressively PCA-reduced (fit on training folds only) and
          re-scored. This is distinct from Cowley, Stan, Pillow &amp; Smith&apos;s (2026, <em>Nature</em>)
          architecture-parameter compression — here we compress <em>activations</em>, not model weights.
        </p>
      </div>

      {!rows ? (
        <EmptyState what="Compression experiment results" />
      ) : (
        <>
          <Panel title="Encoding score vs. PCA dimensionality">
            <ChartDescription>
              Line chart of mean encoding correlation against number of retained PCA components
              (log scale), one line per model, averaged across sessions and areas.
            </ChartDescription>
            <CompressionChart data={rows} />
          </Panel>

          <Panel title="Variance retained vs. dimensionality (reference)">
            <div className="scrollx">
              <table className="text-sm">
                <thead>
                  <tr><th>Model / layer</th><th>Dims</th><th>Native dim</th><th>Variance retained</th><th>Mean r</th><th>RSA ρ</th></tr>
                </thead>
                <tbody>
                  {rows.slice(0, 60).map((r, i) => (
                    <tr key={i}>
                      <td className="font-mono">{(MODEL_LABELS[r.model] ?? r.model)} / {r.layer}</td>
                      <td className="font-mono">{r.n_dims}</td>
                      <td className="font-mono">{r.native_dim}</td>
                      <td className="font-mono">{(r.variance_retained * 100).toFixed(1)}%</td>
                      <td className="font-mono">{r.mean_r.toFixed(4)}</td>
                      <td className="font-mono">{r.rsa_rho.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-xs text-dimmer mt-2">Showing first 60 rows; full table in results/compression/compression_results.parquet.</p>
          </Panel>
        </>
      )}
    </div>
  );
}
