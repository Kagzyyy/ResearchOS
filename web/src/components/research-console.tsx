"use client";

import { useEffect, useRef, useState } from "react";
import {
  ArrowUpRight,
  BookOpen,
  FileText,
  Layers,
  Loader2,
  Search,
  Sparkles,
  ThumbsDown,
  Paperclip,
  X,
  ThumbsUp,
  TriangleAlert,
} from "lucide-react";
import {
  ask,
  explain,
  findPapers,
  health,
  logClick,
  uploadPdf,
  vote,
  type UploadedDoc,
  type AskResult,
  type ExplainResult,
  type Health,
  type PaperHit,
  type PapersResult,
  type Passage,
} from "@/lib/api";
import BoxLoader from "@/components/ui/box-loader";

type Mode = "papers" | "explain";

const EXAMPLES: Record<Mode, string[]> = {
  papers: [
    "world models for robot manipulation",
    "mixture-of-experts routing",
    "reinforcement learning for LLM reasoning",
    "diffusion models for video generation",
    "self-supervised learning without contrastive pairs",
  ],
  explain: [
    "[23] M. Assran et al. Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. CVPR 2023.",
    "arXiv:2312.00752",
    "How do JEPAs avoid representation collapse?",
  ],
};

export function ResearchConsole() {
  const [mode, setMode] = useState<Mode>("papers");
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [papersResult, setPapersResult] = useState<PapersResult | null>(null);
  const [explainResult, setExplainResult] = useState<ExplainResult | null>(null);
  const [askResult, setAskResult] = useState<AskResult | null>(null);
  const [status, setStatus] = useState<Health | null>(null);
  const [doc, setDoc] = useState<UploadedDoc | null>(null);
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    health().then(setStatus).catch(() => setStatus(null));
  }, []);

  function clear() {
    setPapersResult(null);
    setExplainResult(null);
    setAskResult(null);
    setError(null);
  }

  async function run(value?: string) {
    const text = (value ?? input).trim();
    if (text.length < 3 || loading) return;
    setInput(text);
    setLoading(true);
    clear();
    try {
      if (mode === "papers") {
        setPapersResult(await findPapers(text));
      } else {
        const res = await explain(text, doc?.document_id);
        if (res.resolved) {
          setExplainResult(res);
        } else {
          // Not a citation — treat it as a question about the corpus instead.
          setAskResult(await ask(text));
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  async function handleFile(file: File | undefined) {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const uploaded = await uploadPdf(file);
      setDoc(uploaded);
      setMode("explain");
      clear();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  const queryId =
    papersResult?.query_id ?? explainResult?.query_id ?? askResult?.query_id ?? null;

  return (
    <section id="console" className="mx-auto w-full max-w-5xl px-4 py-16 md:py-24">
      <header className="mb-8">
        <h2 className="font-serif text-3xl text-white md:text-4xl">Research console</h2>
        <p className="mt-2 text-sm text-slate-400">
          {status
            ? `${status.papers.toLocaleString()} papers · ${status.chunks.toLocaleString()} passages · reranked by ${status.reranker_model?.split("/").pop()}`
            : "Backend not reachable — start the API on port 8100."}
          {status && !status.generation_enabled && (
            <span className="ml-2 text-amber-400">· GROQ_API_KEY unset, no written answers</span>
          )}
        </p>
      </header>

      <div className="mb-4 flex flex-wrap gap-2">
        <ModeButton
          active={mode === "papers"}
          onClick={() => {
            setMode("papers");
            clear();
          }}
          icon={<Layers size={15} />}
          hint="a topic → the papers on it"
        >
          Find papers
        </ModeButton>
        <ModeButton
          active={mode === "explain"}
          onClick={() => {
            setMode("explain");
            clear();
          }}
          icon={<Sparkles size={15} />}
          hint="a citation or a concept → explained"
        >
          Explain
        </ModeButton>
      </div>

      <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-3 backdrop-blur-sm focus-within:border-white/25">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              run();
            }
          }}
          rows={mode === "explain" ? 3 : 2}
          placeholder={
            mode === "papers"
              ? "A topic, e.g. world models for robot manipulation…"
              : doc
                ? `Ask anything about "${doc.title.slice(0, 48)}"…`
                : "Paste a reference, an arXiv id, or ask about a concept…"
          }
          className="w-full resize-none bg-transparent px-2 py-1 text-[15px] text-slate-100 placeholder:text-slate-500 focus:outline-none"
        />
        {mode === "explain" && doc && (
          <div className="mx-2 mb-2 flex items-center gap-2 rounded-lg border border-sky-500/30 bg-sky-500/10 px-3 py-2 text-xs">
            <Paperclip size={13} className="shrink-0 text-sky-300" />
            <span className="min-w-0 flex-1 truncate text-sky-100">
              Asking about <span className="font-medium">{doc.title}</span>
            </span>
            <span className="shrink-0 text-sky-400/70">
              {doc.n_chunks} passages · {doc.n_words.toLocaleString()} words
            </span>
            <button
              onClick={() => setDoc(null)}
              className="shrink-0 rounded p-0.5 text-sky-300 transition hover:bg-white/10 hover:text-white"
              aria-label="Stop using this document"
            >
              <X size={13} />
            </button>
          </div>
        )}
        <div className="flex items-center justify-between px-2 pt-2">
          <div className="flex items-center gap-3">
            {mode === "explain" && (
              <>
                <input
                  ref={fileRef}
                  type="file"
                  accept="application/pdf,.pdf"
                  className="hidden"
                  onChange={(e) => handleFile(e.target.files?.[0])}
                />
                <button
                  onClick={() => fileRef.current?.click()}
                  disabled={uploading}
                  className="inline-flex items-center gap-1.5 text-xs text-slate-400 transition hover:text-slate-200 disabled:opacity-50"
                >
                  {uploading ? (
                    <Loader2 size={13} className="animate-spin" />
                  ) : (
                    <Paperclip size={13} />
                  )}
                  {uploading ? "Reading PDF…" : doc ? "Replace PDF" : "Attach a PDF"}
                </button>
              </>
            )}
            <span className="text-xs text-slate-500">Enter to run · Shift+Enter for newline</span>
          </div>
          <button
            onClick={() => run()}
            disabled={loading || input.trim().length < 3}
            className="inline-flex items-center gap-2 rounded-lg bg-white px-4 py-2 text-sm font-semibold text-black transition hover:bg-slate-200 disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Search size={15} />
            {mode === "papers" ? "Find papers" : doc ? "Ask this paper" : "Explain"}
          </button>
        </div>
      </div>

      <div className="mt-3 flex flex-wrap gap-2">
        {(mode === "explain" && doc
          ? ["What problem does this solve?", "Explain the method in simple terms", "What are the limitations?"]
          : EXAMPLES[mode]
        ).map((ex) => (
          <button
            key={ex}
            onClick={() => run(ex)}
            className="max-w-full truncate rounded-full border border-white/10 px-3 py-1.5 text-xs text-slate-400 transition hover:border-white/25 hover:text-slate-200"
          >
            {ex.length > 64 ? `${ex.slice(0, 64)}…` : ex}
          </button>
        ))}
      </div>

      {loading && (
        <div className="mt-10 flex flex-col items-center gap-6 py-10">
          <BoxLoader />
          <p className="text-sm text-slate-400">
            {mode === "papers"
              ? "Searching 40,379 passages…"
              : doc
                ? `Reading ${doc.title.slice(0, 40)}…`
                : "Retrieving, reranking, then writing an answer…"}
          </p>
        </div>
      )}

      {error && (
        <Callout tone="red" title="Request failed">
          {error}
        </Callout>
      )}

      {!loading && papersResult && <PapersView result={papersResult} queryId={queryId} />}
      {!loading && explainResult && <ExplainView result={explainResult} queryId={queryId} />}
      {!loading && askResult && <AskView result={askResult} queryId={queryId} />}
    </section>
  );
}

function ModeButton({
  active,
  onClick,
  icon,
  hint,
  children,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  hint: string;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`group inline-flex items-center gap-2 rounded-lg px-3.5 py-2 text-sm font-medium transition ${
        active ? "bg-white/10 text-white ring-1 ring-white/20" : "text-slate-400 hover:text-slate-200"
      }`}
    >
      {icon}
      {children}
      <span className="hidden text-xs font-normal text-slate-500 sm:inline">· {hint}</span>
    </button>
  );
}

function Callout({
  tone,
  title,
  children,
}: {
  tone: "red" | "amber";
  title: string;
  children: React.ReactNode;
}) {
  const colours =
    tone === "red"
      ? "border-red-500/30 bg-red-500/10 text-red-100"
      : "border-amber-500/30 bg-amber-500/10 text-amber-100";
  return (
    <div className={`mt-6 flex items-start gap-3 rounded-xl border p-4 text-sm ${colours}`}>
      <TriangleAlert size={18} className="mt-0.5 shrink-0" />
      <div>
        <p className="font-semibold">{title}</p>
        <p className="mt-1 opacity-80">{children}</p>
      </div>
    </div>
  );
}

/** Explicit thumbs up/down. The scarcest and most useful signal - you cannot
 *  collect it retroactively, which is why it ships with the first version. */
function FeedbackBar({ queryId }: { queryId: number | null }) {
  const [sent, setSent] = useState<null | boolean>(null);
  const [comment, setComment] = useState("");
  const [showComment, setShowComment] = useState(false);

  if (queryId === null) return null;

  function send(helpful: boolean, text?: string) {
    setSent(helpful);
    vote(queryId as number, helpful, text).catch(() => {});
  }

  if (sent !== null && !showComment) {
    return (
      <p className="text-xs text-slate-500">
        Thanks — logged as {sent ? "helpful" : "not helpful"}.
      </p>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <span className="text-xs text-slate-500">Was this useful?</span>
        <button
          onClick={() => send(true)}
          className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 px-2.5 py-1.5 text-xs text-slate-300 transition hover:border-emerald-500/40 hover:text-emerald-300"
        >
          <ThumbsUp size={13} /> Yes
        </button>
        <button
          onClick={() => setShowComment(true)}
          className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 px-2.5 py-1.5 text-xs text-slate-300 transition hover:border-red-500/40 hover:text-red-300"
        >
          <ThumbsDown size={13} /> No
        </button>
      </div>
      {showComment && (
        <div className="flex gap-2">
          <input
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                send(false, comment);
                setShowComment(false);
              }
            }}
            placeholder="What was wrong? (optional)"
            className="flex-1 rounded-lg border border-white/10 bg-white/[0.03] px-3 py-1.5 text-xs text-slate-200 placeholder:text-slate-600 focus:border-white/25 focus:outline-none"
          />
          <button
            onClick={() => {
              send(false, comment);
              setShowComment(false);
            }}
            className="rounded-lg bg-white/10 px-3 py-1.5 text-xs text-slate-200 transition hover:bg-white/20"
          >
            Send
          </button>
        </div>
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

/* ------------------------------------------------------------------ papers */

function PapersView({ result, queryId }: { result: PapersResult; queryId: number | null }) {
  return (
    <div className="mt-8 space-y-6">
      {result.source === "arxiv_live" && (
        <Callout tone="amber" title="Not in the local corpus — fetched live from arXiv">
          The indexed sample does not cover this topic, so these came straight from the
          arXiv API. Abstracts only, so no full-text passages.
        </Callout>
      )}
      {!result.confident && result.source !== "arxiv_live" && (
        <Callout tone="amber" title="Outside this corpus">
          Top matches averaged {result.mean_top_score.toFixed(3)}. Results are shown but
          are probably unrelated to what you asked.
        </Callout>
      )}
      <Meta
        items={[
          `${result.papers.length} papers`,
          `${result.timing_ms.total.toFixed(0)} ms`,
          `top ${result.top_score.toFixed(3)}`,
        ]}
      />
      <div className="space-y-3">
        {result.papers.map((p, i) => (
          <PaperCard key={p.paper_id} paper={p} rank={i + 1} queryId={queryId} />
        ))}
      </div>
      <FeedbackBar queryId={queryId} />
    </div>
  );
}

function PaperCard({
  paper,
  rank,
  queryId,
}: {
  paper: PaperHit;
  rank: number;
  queryId: number | null;
}) {
  const [open, setOpen] = useState(false);
  const [logged, setLogged] = useState(false);

  function toggle() {
    const next = !open;
    setOpen(next);
    if (next && !logged && queryId !== null) {
      logClick(queryId, paper.doc_id).catch(() => {});
      setLogged(true);
    }
  }

  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.02] transition hover:border-white/20">
      <button onClick={toggle} className="flex w-full items-start gap-4 p-4 text-left">
        <span className="mt-1 w-5 shrink-0 text-right font-mono text-xs text-slate-600">
          {rank}
        </span>
        <span className="min-w-0 flex-1">
          <span className="block text-[15px] leading-snug font-medium text-slate-100">
            {paper.title}
          </span>
          <span className="mt-1 block text-xs text-slate-500">
            {paper.authors.slice(0, 3).join(", ")}
            {paper.authors.length > 3 && " et al."} · {paper.published} ·{" "}
            {paper.primary_category}
          </span>
          <span className="mt-2 flex flex-wrap items-center gap-2 text-[11px] text-slate-500">
            <span className="rounded bg-white/5 px-1.5 py-0.5 font-mono">
              {paper.score.toFixed(3)}
            </span>
            <span>
              {paper.matched_passages} passage{paper.matched_passages === 1 ? "" : "s"} matched
            </span>
            {paper.matched_sections.length > 0 && (
              <span className="text-slate-600">{paper.matched_sections.join(" · ")}</span>
            )}
            {paper.has_full_text && (
              <span className="inline-flex items-center gap-1 text-slate-600">
                <FileText size={11} /> full text
              </span>
            )}
            {paper.from_arxiv && (
              <span className="rounded bg-sky-500/15 px-1.5 py-0.5 text-sky-300">
                live from arXiv
              </span>
            )}
          </span>
        </span>
      </button>

      {open && (
        <div className="space-y-3 border-t border-white/10 px-4 py-3 pl-13">
          <div>
            <p className="mb-1 text-[11px] tracking-wider text-slate-500 uppercase">
              Why it matched · {paper.why_section}
            </p>
            <p className="text-sm leading-relaxed text-slate-300">{paper.why}…</p>
          </div>
          {paper.abstract && (
            <div>
              <p className="mb-1 text-[11px] tracking-wider text-slate-500 uppercase">Abstract</p>
              <p className="text-sm leading-relaxed text-slate-400">{paper.abstract}</p>
            </div>
          )}
          <a
            href={paper.arxiv_url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-xs text-sky-400 hover:underline"
          >
            {paper.paper_id} on arXiv <ArrowUpRight size={12} />
          </a>
        </div>
      )}
    </div>
  );
}

/* ----------------------------------------------------------------- explain */

function ExplainView({ result, queryId }: { result: ExplainResult; queryId: number | null }) {
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
          result.from_corpus ? "from local corpus" : "fetched live from arXiv",
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
      <FeedbackBar queryId={queryId} />
    </div>
  );
}

function AskView({ result, queryId }: { result: AskResult; queryId: number | null }) {
  return (
    <div className="mt-8 space-y-6">
      <Callout tone="amber" title="Not a citation — answered as a question instead">
        Nothing resolved to a specific paper, so this was answered from the corpus.
      </Callout>
      {result.answer && (
        <article className="rounded-2xl border border-white/10 bg-white/[0.03] p-6">
          <Prose text={result.answer} />
        </article>
      )}
      <Meta
        items={[
          `${result.distinct_papers} papers`,
          `${result.timing_ms.total.toFixed(0)} ms`,
          `mean ${result.mean_top_score.toFixed(3)}`,
        ]}
      />
      <PassageList title="Sources" passages={result.citations} queryId={queryId} />
      <FeedbackBar queryId={queryId} />
    </div>
  );
}

/* -------------------------------------------------------------- primitives */

function Prose({ text }: { text: string }) {
  return (
    <div className="space-y-3 text-[15px] leading-relaxed text-slate-200">
      {text
        .split("\n")
        .filter((l) => l.trim())
        .map((line, i) => (
          <p key={i}>
            {line
              .split(/(\*\*[^*]+\*\*|`[^`]+`|\[\d{4}\.\d{4,5}[^\]]*\])/g)
              .map((part, j) => {
                if (part.startsWith("**") && part.endsWith("**"))
                  return (
                    <strong key={j} className="font-semibold text-white">
                      {part.slice(2, -2)}
                    </strong>
                  );
                if (part.startsWith("`") && part.endsWith("`"))
                  return (
                    <code key={j} className="rounded bg-white/10 px-1.5 py-0.5 font-mono text-[13px]">
                      {part.slice(1, -1)}
                    </code>
                  );
                if (/^\[\d{4}\.\d{4,5}[^\]]*\]$/.test(part))
                  return (
                    <a
                      key={j}
                      href={`https://arxiv.org/abs/${part.slice(1, -1).split("†")[0]}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sky-400 hover:underline"
                    >
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
