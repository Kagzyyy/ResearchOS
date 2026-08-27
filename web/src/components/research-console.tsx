"use client";

import { useEffect, useRef, useState } from "react";
import {
  ArrowUpRight,
  BookOpen,
  Loader2,
  MessageSquareText,
  Quote,
  Search,
  TriangleAlert,
} from "lucide-react";
import {
  ask,
  explain,
  health,
  logClick,
  type AskResult,
  type ExplainResult,
  type Health,
  type Passage,
} from "@/lib/api";

type Mode = "ask" | "explain";

const EXAMPLES: Record<Mode, string[]> = {
  ask: [
    "How do joint-embedding predictive architectures avoid representation collapse?",
    "What is the difference between JEPA and masked autoencoders?",
    "How is reinforcement learning used to improve LLM reasoning?",
    "What are world models used for in robotics?",
  ],
  explain: [
    "[23] M. Assran et al. Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. CVPR 2023.",
    "arXiv:2301.08243",
    "Masked Autoencoders Are Scalable Vision Learners",
  ],
};

export function ResearchConsole() {
  const [mode, setMode] = useState<Mode>("ask");
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [askResult, setAskResult] = useState<AskResult | null>(null);
  const [explainResult, setExplainResult] = useState<ExplainResult | null>(null);
  const [status, setStatus] = useState<Health | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    health().then(setStatus).catch(() => setStatus(null));
  }, []);

  async function run(value?: string) {
    const text = (value ?? input).trim();
    if (text.length < 3 || loading) return;
    setInput(text);
    setLoading(true);
    setError(null);
    setAskResult(null);
    setExplainResult(null);
    try {
      if (mode === "ask") setAskResult(await ask(text));
      else setExplainResult(await explain(text));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  const queryId = askResult?.query_id ?? explainResult?.query_id ?? null;

  return (
    <section id="console" className="mx-auto w-full max-w-5xl px-4 py-16 md:py-24">
      <header className="mb-8">
        <h2 className="font-serif text-3xl text-white md:text-4xl">Research console</h2>
        <p className="mt-2 text-sm text-slate-400">
          {status
            ? `${status.papers.toLocaleString()} papers · ${status.chunks.toLocaleString()} passages · reranked by ${status.reranker_model?.split("/").pop()}`
            : "Backend not reachable — start the API on port 8100."}
          {status && !status.generation_enabled && (
            <span className="ml-2 text-amber-400">
              · GROQ_API_KEY unset, passages only
            </span>
          )}
        </p>
      </header>

      <div className="mb-4 flex gap-2">
        <ModeButton active={mode === "ask"} onClick={() => setMode("ask")} icon={<MessageSquareText size={15} />}>
          Ask a question
        </ModeButton>
        <ModeButton active={mode === "explain"} onClick={() => setMode("explain")} icon={<Quote size={15} />}>
          Explain a citation
        </ModeButton>
      </div>

      <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-3 backdrop-blur-sm focus-within:border-white/25">
        <textarea
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) run();
          }}
          rows={mode === "explain" ? 3 : 2}
          placeholder={
            mode === "ask"
              ? "Ask anything about AI research…"
              : "Paste a reference: [23] Author et al. Title. Venue. — or an arXiv id"
          }
          className="w-full resize-none bg-transparent px-2 py-1 text-[15px] text-slate-100 placeholder:text-slate-500 focus:outline-none"
        />
        <div className="flex items-center justify-between px-2 pt-2">
          <span className="text-xs text-slate-500">⌘/Ctrl + Enter</span>
          <button
            onClick={() => run()}
            disabled={loading || input.trim().length < 3}
            className="inline-flex items-center gap-2 rounded-lg bg-white px-4 py-2 text-sm font-semibold text-black transition hover:bg-slate-200 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {loading ? <Loader2 size={15} className="animate-spin" /> : <Search size={15} />}
            {mode === "ask" ? "Ask" : "Explain"}
          </button>
        </div>
      </div>

      <div className="mt-3 flex flex-wrap gap-2">
        {EXAMPLES[mode].map((ex) => (
          <button
            key={ex}
            onClick={() => run(ex)}
            className="max-w-full truncate rounded-full border border-white/10 px-3 py-1.5 text-xs text-slate-400 transition hover:border-white/25 hover:text-slate-200"
          >
            {ex.length > 68 ? `${ex.slice(0, 68)}…` : ex}
          </button>
        ))}
      </div>

      {error && (
        <div className="mt-6 flex items-start gap-3 rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-200">
          <TriangleAlert size={18} className="mt-0.5 shrink-0" />
          <div>
            <p className="font-semibold">Request failed</p>
            <p className="mt-1 text-red-300/80">{error}</p>
          </div>
        </div>
      )}

      {askResult && <AskView result={askResult} queryId={queryId} />}
      {explainResult && <ExplainView result={explainResult} queryId={queryId} />}
    </section>
  );
}

function ModeButton({
  active,
  onClick,
  icon,
  children,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`inline-flex items-center gap-2 rounded-lg px-3.5 py-2 text-sm font-medium transition ${
        active
          ? "bg-white/10 text-white ring-1 ring-white/20"
          : "text-slate-400 hover:text-slate-200"
      }`}
    >
      {icon}
      {children}
    </button>
  );
}

function AskView({ result, queryId }: { result: AskResult; queryId: number | null }) {
  return (
    <div className="mt-8 space-y-6">
      {!result.confident && (
        <div className="flex items-start gap-3 rounded-xl border border-amber-500/30 bg-amber-500/10 p-4 text-sm text-amber-100">
          <TriangleAlert size={18} className="mt-0.5 shrink-0" />
          <div>
            <p className="font-semibold">Outside this corpus</p>
            <p className="mt-1 text-amber-200/80">
              Top passages averaged {result.mean_top_score.toFixed(3)}. The closest
              matches are shown, but treat them as unrelated.
            </p>
          </div>
        </div>
      )}

      {result.answer && (
        <article className="rounded-2xl border border-white/10 bg-white/[0.03] p-6">
          <Prose text={result.answer} />
        </article>
      )}

      <Meta
        items={[
          `${result.distinct_papers} papers`,
          `${result.timing_ms.retrieval_and_rerank.toFixed(0)} ms retrieval`,
          result.timing_ms.generation > 0
            ? `${(result.timing_ms.generation / 1000).toFixed(1)} s generation`
            : "no generation",
          `mean ${result.mean_top_score.toFixed(3)}`,
        ]}
      />

      <PassageList title="Sources" passages={result.citations} queryId={queryId} />
    </div>
  );
}

function ExplainView({ result, queryId }: { result: ExplainResult; queryId: number | null }) {
  if (!result.resolved) {
    return (
      <div className="mt-8 space-y-4">
        <div className="flex items-start gap-3 rounded-xl border border-amber-500/30 bg-amber-500/10 p-4 text-sm text-amber-100">
          <TriangleAlert size={18} className="mt-0.5 shrink-0" />
          <div>
            <p className="font-semibold">Could not resolve that citation</p>
            <p className="mt-1 text-amber-200/80">
              Searched for <span className="font-mono">{result.query_used}</span>. Try the
              arXiv id, or just the title.
            </p>
          </div>
        </div>
        {result.candidates.length > 0 && (
          <div className="rounded-xl border border-white/10 p-4">
            <p className="mb-2 text-xs tracking-wide text-slate-400 uppercase">Did you mean</p>
            <ul className="space-y-1.5 text-sm text-slate-300">
              {result.candidates.map((c) => (
                <li key={c.paper_id} className="flex gap-2">
                  <span className="font-mono text-slate-500">{c.score}</span>
                  <span>{c.title}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    );
  }

  const paper = result.paper!;
  return (
    <div className="mt-8 space-y-6">
      <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h3 className="font-serif text-xl text-white">{paper.title}</h3>
            <p className="mt-1.5 text-sm text-slate-400">
              {paper.authors.slice(0, 4).join(", ")}
              {paper.authors.length > 4 && " et al."} · {paper.published} ·{" "}
              {paper.primary_category}
            </p>
          </div>
          <a
            href={paper.arxiv_url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex shrink-0 items-center gap-1 rounded-lg border border-white/15 px-3 py-1.5 text-xs text-slate-300 transition hover:border-white/30 hover:text-white"
          >
            arXiv <ArrowUpRight size={13} />
          </a>
        </div>
      </div>

      <Meta
        items={[
          `resolved via ${result.method.replace("_", " ")}`,
          result.from_corpus ? "from local corpus" : "fetched from arXiv",
          paper.has_full_text ? "full text" : "abstract only",
          `${(result.timing_ms.total / 1000).toFixed(1)} s`,
        ]}
      />

      {result.explanation && (
        <article className="rounded-2xl border border-white/10 bg-white/[0.03] p-6">
          <Prose text={result.explanation} />
        </article>
      )}

      <PassageList title="From this paper" passages={result.passages} queryId={queryId} />
      {result.related.length > 0 && (
        <PassageList
          title="What other papers say about it"
          passages={result.related}
          queryId={queryId}
        />
      )}
    </div>
  );
}

function Meta({ items }: { items: string[] }) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-500">
      {items.map((i) => (
        <span key={i}>{i}</span>
      ))}
    </div>
  );
}

/** Minimal markdown: **bold**, `code`, and paragraph breaks. */
function Prose({ text }: { text: string }) {
  return (
    <div className="space-y-3 text-[15px] leading-relaxed text-slate-200">
      {text.split("\n").filter((l) => l.trim()).map((line, i) => (
        <p key={i}>
          {line.split(/(\*\*[^*]+\*\*|`[^`]+`|\[\d{4}\.\d{4,5}\])/g).map((part, j) => {
            if (part.startsWith("**") && part.endsWith("**"))
              return <strong key={j} className="font-semibold text-white">{part.slice(2, -2)}</strong>;
            if (part.startsWith("`") && part.endsWith("`"))
              return <code key={j} className="rounded bg-white/10 px-1.5 py-0.5 font-mono text-[13px]">{part.slice(1, -1)}</code>;
            if (/^\[\d{4}\.\d{4,5}\]$/.test(part))
              return (
                <a key={j} href={`https://arxiv.org/abs/${part.slice(1, -1)}`} target="_blank" rel="noopener noreferrer" className="text-sky-400 hover:underline">
                  {part}
                </a>
              );
            return <span key={j}>{part}</span>;
          })}
        </p>
      ))}
    </div>
  );
}

function PassageList({
  title,
  passages,
  queryId,
}: {
  title: string;
  passages: Passage[];
  queryId: number | null;
}) {
  if (passages.length === 0) return null;
  return (
    <div>
      <h4 className="mb-3 flex items-center gap-2 text-xs tracking-wider text-slate-400 uppercase">
        <BookOpen size={13} /> {title}
      </h4>
      <div className="space-y-2">
        {passages.map((p) => (
          <PassageCard key={p.chunk_id} passage={p} queryId={queryId} />
        ))}
      </div>
    </div>
  );
}

function PassageCard({ passage, queryId }: { passage: Passage; queryId: number | null }) {
  const [open, setOpen] = useState(false);
  const [logged, setLogged] = useState(false);

  function toggle() {
    const next = !open;
    setOpen(next);
    // Opening a citation is the feedback signal the retraining loop would consume.
    if (next && !logged && queryId !== null) {
      logClick(queryId, passage.doc_id).catch(() => {});
      setLogged(true);
    }
  }

  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.02] transition hover:border-white/20">
      <button onClick={toggle} className="flex w-full items-start gap-3 p-4 text-left">
        <span className="mt-0.5 font-mono text-xs text-slate-500">
          {passage.score.toFixed(2)}
        </span>
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm font-medium text-slate-200">
            {passage.title}
          </span>
          <span className="mt-0.5 block text-xs text-slate-500">
            {passage.paper_id} · {passage.section} · {passage.published}
          </span>
        </span>
      </button>
      {open && (
        <div className="border-t border-white/10 px-4 py-3">
          <p className="text-sm leading-relaxed text-slate-300">{passage.text}</p>
          <a
            href={passage.arxiv_url}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-3 inline-flex items-center gap-1 text-xs text-sky-400 hover:underline"
          >
            Open on arXiv <ArrowUpRight size={12} />
          </a>
        </div>
      )}
    </div>
  );
}
