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

export * from "./constants";
