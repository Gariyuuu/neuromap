import fs from "fs";
import path from "path";

const DATA_DIR = path.join(process.cwd(), "public", "data");

export function loadJSON<T>(name: string): T | null {
  const p = path.join(DATA_DIR, name);
  if (!fs.existsSync(p)) return null;
  try {
    return JSON.parse(fs.readFileSync(p, "utf-8")) as T;
  } catch {
    return null;
  }
}

export type Overview = {
  dataset: string;
  source: string;
  stimulus: string;
  n_sessions: number;
  n_areas: number;
  areas: string[];
  n_models: number;
  models: string[];
  n_neurons_total: number;
  git_sha: string;
};

export type AreaModelHeatmapRow = {
  area: string;
  model: string;
  mean: number;
  std: number;
  count: number;
  family: string;
};

export type SessionDetailRow = {
  area: string;
  experiment_id: number;
  model: string;
  layer: string;
  mean_r: number;
  n_neurons: number;
  n_images: number;
};

export type LayerProfileRow = {
  area: string;
  model: string;
  layer: string;
  mean_r: number;
};

export type RsaResultRow = {
  area: string;
  experiment_id: number;
  model: string;
  layer: string;
  rsa_rho: number;
  rsa_ci_lo: number;
  rsa_ci_hi: number;
  cka: number;
};

export type CompressionRow = {
  area: string;
  experiment_id: number;
  model: string;
  layer: string;
  family: string;
  n_dims: number;
  native_dim: number;
  variance_retained: number;
  mean_r: number;
  rsa_rho: number;
  runtime_s: number;
  storage_bytes_per_image: number;
};

export type DatasetsInfo = {
  preprocessing_config: Record<string, unknown>;
  sessions: Array<Record<string, unknown>>;
  dataset_config: Record<string, unknown>;
};

export const MODEL_LABELS: Record<string, string> = {
  pixels: "Raw pixels",
  gabor: "Gabor filter bank",
  resnet18: "ResNet-18 (supervised)",
  resnet50: "ResNet-50 (supervised)",
  vit_b_16: "ViT-B/16 (supervised)",
  dino_vits16: "DINO ViT-S/16 (self-supervised)",
};

export const MODEL_FAMILY_COLOR: Record<string, string> = {
  classical: "#8b96a3",
  cnn_supervised: "#6ea8fe",
  vit_supervised: "#c792ea",
  self_supervised: "#3ecf8e",
};
