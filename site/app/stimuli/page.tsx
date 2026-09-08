import Image from "next/image";
import { loadJSON } from "@/lib/data";
import { Panel, EmptyState } from "@/components/ui";

type FailureReport = { hardest_stimuli_frame_ids: number[]; hardest_stimuli_mean_pattern_r: number[] };

export default function StimuliPage() {
  const failures = loadJSON<FailureReport>("failure_analysis.json");
  const hardIds = new Set(failures?.hardest_stimuli_frame_ids ?? []);
  const hardScoreById = new Map(
    (failures?.hardest_stimuli_frame_ids ?? []).map((id, i) => [id, failures!.hardest_stimuli_mean_pattern_r[i]])
  );

  const images = Array.from({ length: 118 }, (_, i) => i);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-medium">Stimulus explorer</h1>
        <p className="text-dim text-sm mt-1 max-w-2xl">
          All 118 natural_scenes stimuli (Allen Brain Observatory, public, Allen Institute Terms
          of Use). Highlighted images are the &ldquo;hardest stimuli&rdquo; — lowest out-of-fold
          population-pattern correlation under each session&apos;s best model (see{" "}
          <a href="/failures" className="underline text-model">Failure analysis</a>).
        </p>
      </div>

      {!failures && <EmptyState what="Failure analysis (for hardest-stimulus highlighting)" />}

      <Panel title="Stimulus set (natural_scenes, frame IDs 0–117)">
        <div className="grid grid-cols-6 sm:grid-cols-8 md:grid-cols-10 lg:grid-cols-12 gap-2">
          {images.map((id) => {
            const isHard = hardIds.has(id);
            return (
              <div
                key={id}
                className={`relative rounded overflow-hidden border ${isHard ? "border-[#e0685f]" : "border-hairline"}`}
                title={isHard ? `scene_${String(id).padStart(3, "0")} — hardest stimulus, pattern r=${hardScoreById.get(id)?.toFixed(3)}` : `scene_${String(id).padStart(3, "0")}`}
              >
                <Image
                  src={`/stimuli/scene_${String(id).padStart(3, "0")}.png`}
                  alt={`Natural scene stimulus ${id}${isHard ? ", flagged as a hard-to-predict stimulus" : ""}`}
                  width={100}
                  height={78}
                  className="w-full h-auto grayscale"
                  unoptimized
                />
                {isHard && (
                  <span className="absolute bottom-0 right-0 bg-[#e0685f] text-[9px] px-1 text-[#1a0806] font-mono">hard</span>
                )}
              </div>
            );
          })}
        </div>
      </Panel>
    </div>
  );
}
