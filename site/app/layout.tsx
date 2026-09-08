import type { Metadata } from "next";
import "./globals.css";
import { NavLinks } from "@/components/NavLinks";

export const metadata: Metadata = {
  title: "NeuroMap — Artificial vs. Biological Vision",
  description:
    "Auditing the alignment between artificial visual representations and measured mouse visual-cortex responses.",
};

const ROUTES = [
  { href: "/", label: "Overview" },
  { href: "/datasets", label: "Datasets" },
  { href: "/brain", label: "Brain Areas" },
  { href: "/models", label: "Models" },
  { href: "/layers", label: "Layers" },
  { href: "/stimuli", label: "Stimuli" },
  { href: "/rsa", label: "RSA" },
  { href: "/compression", label: "Compression" },
  { href: "/failures", label: "Failures" },
  { href: "/methods", label: "Methods" },
  { href: "/paper", label: "Paper" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full">
      <body className="min-h-full flex flex-col">
        <header className="border-b border-hairline sticky top-0 z-10 backdrop-blur bg-[#0b0d10]/90">
          <div className="max-w-[1400px] mx-auto px-6 py-3 flex items-center gap-8">
            <a href="/" className="font-mono text-sm tracking-tight">
              <span className="text-neural">Neuro</span>
              <span className="text-model">Map</span>
            </a>
            <NavLinks routes={ROUTES} />
          </div>
        </header>
        <main className="flex-1 grid-lab-bg">
          <div className="max-w-[1400px] mx-auto px-6 py-8">{children}</div>
        </main>
        <footer className="border-t border-hairline text-dimmer text-xs">
          <div className="max-w-[1400px] mx-auto px-6 py-4 flex justify-between">
            <span>
              NeuroMap · Allen Brain Observatory Visual Coding (mouse, 2-photon calcium imaging) ·
              research artifact, not a clinical or cognitive-similarity claim
            </span>
            <a href="/methods" className="hover:text-fg">
              Methodology & limitations
            </a>
          </div>
        </footer>
      </body>
    </html>
  );
}
