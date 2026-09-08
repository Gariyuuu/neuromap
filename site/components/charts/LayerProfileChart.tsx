"use client";

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

const AREA_COLORS: Record<string, string> = {
  VISp: "#3ecf8e", VISl: "#6ea8fe", VISal: "#c792ea",
  VISpm: "#e0a458", VISam: "#e0685f", VISrl: "#5cc8e0",
};
const AREA_DASH: Record<string, string> = {
  VISp: "0", VISl: "0", VISal: "4 2",
  VISpm: "4 2", VISam: "1 3", VISrl: "1 3",
};

export function LayerProfileChart({
  layers,
  series,
}: {
  layers: string[];
  series: { area: string; values: (number | null)[] }[];
}) {
  const chartData = layers.map((layer, i) => {
    const point: Record<string, string | number | null> = { layer };
    for (const s of series) point[s.area] = s.values[i];
    return point;
  });

  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#232a32" />
        <XAxis dataKey="layer" tick={{ fill: "#8b96a3", fontSize: 10 }} />
        <YAxis tick={{ fill: "#8b96a3", fontSize: 10 }} />
        <Tooltip contentStyle={{ background: "#12161b", border: "1px solid #232a32", fontSize: 12 }} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {series.map((s) => (
          <Line
            key={s.area}
            type="monotone"
            dataKey={s.area}
            stroke={AREA_COLORS[s.area] ?? "#8b96a3"}
            strokeDasharray={AREA_DASH[s.area]}
            dot={{ r: 2 }}
            connectNulls
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
