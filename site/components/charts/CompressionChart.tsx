"use client";

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

export function CompressionChart({
  data,
}: {
  data: { n_dims: number; model: string; mean_r: number }[];
}) {
  const dims = Array.from(new Set(data.map((d) => d.n_dims))).sort((a, b) => a - b);
  const models = Array.from(new Set(data.map((d) => d.model)));
  const chartData = dims.map((d) => {
    const point: Record<string, number> = { n_dims: d };
    for (const m of models) {
      const match = data.filter((x) => x.n_dims === d && x.model === m);
      if (match.length) point[m] = match.reduce((a, b) => a + b.mean_r, 0) / match.length;
    }
    return point;
  });

  const colors = ["#3ecf8e", "#6ea8fe", "#c792ea", "#e0a458", "#e0685f", "#5cc8e0"];

  return (
    <ResponsiveContainer width="100%" height={320}>
      <LineChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#232a32" />
        <XAxis dataKey="n_dims" scale="log" domain={["auto", "auto"]} type="number" tick={{ fill: "#8b96a3", fontSize: 11 }} label={{ value: "PCA components", position: "bottom", fill: "#8b96a3", fontSize: 11 }} />
        <YAxis tick={{ fill: "#8b96a3", fontSize: 11 }} label={{ value: "Mean encoding r", angle: -90, fill: "#8b96a3", fontSize: 11 }} />
        <Tooltip contentStyle={{ background: "#12161b", border: "1px solid #232a32", fontSize: 12 }} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {models.map((m, i) => (
          <Line key={m} type="monotone" dataKey={m} stroke={colors[i % colors.length]} dot={{ r: 3 }} connectNulls />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
