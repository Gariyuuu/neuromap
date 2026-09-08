import { loadJSON, RsaResultRow, MODEL_LABELS } from "@/lib/data";
import { Panel, EmptyState } from "@/components/ui";

export default function RsaPage() {
  const rows = loadJSON<RsaResultRow[]>("rsa_results.json");

  const byModel = new Map<string, RsaResultRow[]>();
  rows?.forEach((r) => {
    if (!byModel.has(r.model)) byModel.set(r.model, []);
    byModel.get(r.model)!.push(r);
  });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-medium">Representational similarity analysis</h1>
        <p className="text-dim text-sm mt-1 max-w-3xl">
          RSA compares the rank structure of pairwise stimulus (dis)similarities between neural
          and model representations (Spearman ρ on RDM upper triangles, bootstrap CI over
          stimuli). Linear CKA is reported alongside but is <em>not interchangeable</em> with RSA
          — it is sensitive to representational geometry, not just distance ranks.
        </p>
      </div>

      {!rows ? (
        <EmptyState what="RSA/CKA results" />
      ) : (
        Array.from(byModel.entries()).map(([model, modelRows]) => (
          <Panel key={model} title={MODEL_LABELS[model] ?? model}>
            <div className="scrollx">
              <table className="text-sm">
                <thead>
                  <tr>
                    <th>Area</th><th>Session</th><th>Layer</th>
                    <th>RSA ρ (95% CI)</th><th>Linear CKA</th>
                  </tr>
                </thead>
                <tbody>
                  {modelRows.map((r, i) => (
                    <tr key={i}>
                      <td className="font-mono">{r.area}</td>
                      <td className="font-mono">{r.experiment_id}</td>
                      <td className="font-mono">{r.layer}</td>
                      <td className="font-mono">
                        {r.rsa_rho.toFixed(3)} [{r.rsa_ci_lo.toFixed(3)}, {r.rsa_ci_hi.toFixed(3)}]
                      </td>
                      <td className="font-mono">{r.cka.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        ))
      )}
    </div>
  );
}
