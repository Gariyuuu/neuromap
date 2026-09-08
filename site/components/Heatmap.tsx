import { MODEL_LABELS } from "@/lib/constants";

function colorForValue(v: number, min: number, max: number) {
  const t = max > min ? (v - min) / (max - min) : 0.5;
  // dark panel -> teal (neural-predictivity accent), interpolated
  const lightness = 14 + t * 30;
  return `hsl(158, ${40 + t * 30}%, ${lightness}%)`;
}

export function Heatmap({
  rows,
  cols,
  valueOf,
  countOf,
  rowLabel = "Brain area",
  colLabel = "Model",
}: {
  rows: string[];
  cols: string[];
  valueOf: (row: string, col: string) => number | null;
  countOf?: (row: string, col: string) => number | null;
  rowLabel?: string;
  colLabel?: string;
}) {
  const values = rows.flatMap((r) => cols.map((c) => valueOf(r, c))).filter((v): v is number => v !== null);
  const min = Math.min(...values, 0);
  const max = Math.max(...values, 0.001);

  return (
    <div>
      <p className="sr-only">
        Heatmap of mean ridge-encoding correlation (r) between {rowLabel.toLowerCase()} and{" "}
        {colLabel.toLowerCase()}, cell darkness and printed value both encode the score — color
        is never the only signal.
      </p>
      <div className="scrollx">
        <table className="text-sm">
          <thead>
            <tr>
              <th className="sticky left-0 bg-[#12161b]">{rowLabel}</th>
              {cols.map((c) => (
                <th key={c} className="text-center min-w-[110px]">
                  {MODEL_LABELS[c] ?? c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r}>
                <td className="font-mono sticky left-0 bg-[#12161b] font-medium">{r}</td>
                {cols.map((c) => {
                  const v = valueOf(r, c);
                  const n = countOf?.(r, c);
                  return (
                    <td key={c} className="text-center">
                      {v === null ? (
                        <span className="text-dimmer">—</span>
                      ) : (
                        <div
                          className="rounded px-2 py-1.5 font-mono text-xs"
                          style={{ background: colorForValue(v, min, max), color: "#06110c" }}
                          title={n ? `n=${n} sessions` : undefined}
                        >
                          {v.toFixed(3)}
                        </div>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
