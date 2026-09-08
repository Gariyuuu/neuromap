"use client";

import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ErrorBar, Cell,
} from "recharts";
import { MODEL_LABELS, MODEL_FAMILY_COLOR } from "@/lib/constants";

export type ModelBarDatum = { model: string; family: string; mean: number; std: number };

export function ModelBarChart({ data }: { data: ModelBarDatum[] }) {
  const chartData = data.map((d) => ({
    ...d,
    label: MODEL_LABELS[d.model] ?? d.model,
    color: MODEL_FAMILY_COLOR[d.family] ?? "#8b96a3",
  }));

  return (
    <ResponsiveContainer width="100%" height={360}>
      <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 60 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#232a32" />
        <XAxis
          dataKey="label"
          angle={-30}
          textAnchor="end"
          interval={0}
          height={80}
          tick={{ fill: "#8b96a3", fontSize: 11 }}
        />
        <YAxis tick={{ fill: "#8b96a3", fontSize: 11 }} label={{ value: "Mean encoding r", angle: -90, fill: "#8b96a3", fontSize: 11 }} />
        <Tooltip
          contentStyle={{ background: "#12161b", border: "1px solid #232a32", fontSize: 12 }}
          formatter={(value) => (typeof value === "number" ? value.toFixed(4) : value)}
        />
        <Bar dataKey="mean" radius={[4, 4, 0, 0]}>
          {chartData.map((d, i) => (
            <Cell key={i} fill={d.color} />
          ))}
          <ErrorBar dataKey="std" width={4} strokeWidth={1} stroke="#e6ebf0" />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
