import { loadJSON, DatasetsInfo } from "@/lib/data";
import { Panel, EmptyState } from "@/components/ui";

export default function DatasetsPage() {
  const info = loadJSON<DatasetsInfo>("datasets.json");

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-medium">Dataset explorer</h1>
        <p className="text-dim text-sm mt-1 max-w-2xl">Data card and provenance for the neural dataset used.</p>
      </div>

      <Panel title="Data card — Allen Brain Observatory, Visual Coding (2-photon)">
        <dl className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
          <div><dt className="text-dim text-xs uppercase">Species</dt><dd>Mouse (Mus musculus)</dd></div>
          <div><dt className="text-dim text-xs uppercase">Recording modality</dt><dd>2-photon calcium imaging (GCaMP6f)</dd></div>
          <div><dt className="text-dim text-xs uppercase">Brain areas</dt><dd>VISp, VISl, VISal, VISpm, VISam, VISrl</dd></div>
          <div><dt className="text-dim text-xs uppercase">Stimulus</dt><dd>natural_scenes (118 grayscale images, ~50 repeats)</dd></div>
          <div><dt className="text-dim text-xs uppercase">Sessions used</dt><dd>18 (3 mice × 6 areas)</dd></div>
          <div><dt className="text-dim text-xs uppercase">Access</dt><dd>Public, no authentication (AllenSDK)</dd></div>
          <div><dt className="text-dim text-xs uppercase">License</dt><dd>Allen Institute Terms of Use</dd></div>
          <div><dt className="text-dim text-xs uppercase">Citation</dt><dd>Allen Institute for Brain Science, Allen Brain Observatory (2016)</dd></div>
        </dl>
      </Panel>

      {!info ? (
        <EmptyState what="Detailed dataset/session manifest" />
      ) : (
        <>
          <Panel title="Response preprocessing (persisted config, not implicit)">
            <pre className="text-xs font-mono text-dim overflow-x-auto">
              {JSON.stringify(info.preprocessing_config, null, 2)}
            </pre>
          </Panel>

          <Panel title={`Sessions (${info.sessions.length})`}>
            <div className="scrollx">
              <table className="text-sm">
                <thead>
                  <tr>
                    <th>Area</th><th>Experiment ID</th><th>N neurons</th><th>N trials</th><th>Mean reliability</th>
                  </tr>
                </thead>
                <tbody>
                  {info.sessions.map((s: any, i: number) => (
                    <tr key={i}>
                      <td className="font-mono">{s.area}</td>
                      <td className="font-mono">{s.experiment_id}</td>
                      <td>{s.n_cells}</td>
                      <td>{s.n_trials}</td>
                      <td>{s.mean_reliability?.toFixed(3) ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </>
      )}

      <Panel title="Known limitations">
        <ul className="text-sm text-dim list-disc list-inside space-y-1">
          <li>Species: mouse visual cortex has a flatter functional hierarchy than primate ventral stream — findings do not transfer to primate/human claims.</li>
          <li>n=3 sessions per area is a small sample; subject-level confidence intervals are wide.</li>
          <li>No semantic category labels are provided for natural_scenes images in AllenSDK, so stimulus-level failure analysis is per-image, not per-category.</li>
          <li>Trial-averaged responses discard within-session temporal dynamics beyond the fixed response window.</li>
        </ul>
      </Panel>
    </div>
  );
}
