import fs from "fs";
import path from "path";
import { marked } from "marked";
import { Panel, EmptyState } from "@/components/ui";

export default async function PaperPage() {
  const paperPath = path.join(process.cwd(), "..", "paper", "paper.md");
  const exists = fs.existsSync(paperPath);
  const html = exists ? await marked.parse(fs.readFileSync(paperPath, "utf-8")) : "";

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-medium">Paper</h1>
        <p className="text-dim text-sm mt-1">Full conference-style manuscript, rendered from paper/paper.md.</p>
      </div>
      {!exists ? (
        <EmptyState what="Paper manuscript" />
      ) : (
        <Panel>
          <article
            className="prose prose-invert prose-sm max-w-none prose-headings:font-medium prose-a:text-model"
            dangerouslySetInnerHTML={{ __html: html }}
          />
        </Panel>
      )}
    </div>
  );
}
