"use client";

import { useRef } from "react";
import { WovenLightHero } from "@/components/ui/woven-light-hero";
import { ResearchConsole } from "@/components/research-console";

export default function Home() {
  const consoleRef = useRef<HTMLDivElement>(null);

  return (
    <main className="min-h-screen bg-black">
      <WovenLightHero
        headline="ResearchOS"
        subline="Ask questions about AI research. Paste a citation you saw in a paper and get it explained in plain language — grounded in the paper itself, never invented."
        ctaLabel="Start reading"
        brand="ResearchOS"
        onCta={() =>
          consoleRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })
        }
      />
      <div ref={consoleRef}>
        <ResearchConsole />
      </div>
      <footer className="border-t border-white/10 px-4 py-8 text-center text-xs text-slate-600">
        Retrieval: BM25 + bge-small-en-v1.5, fused with RRF, reranked by bge-reranker-base.
        Generation: Groq. Answers are grounded in retrieved passages only.
      </footer>
    </main>
  );
}
