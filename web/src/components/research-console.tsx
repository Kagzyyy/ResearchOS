"use client";

import { useEffect, useRef, useState } from "react";
import {
  ArrowUpRight,
  BookOpen,
  FileText,
  Layers,
  Loader2,
  Search,
  Check,
  Clock,
  Copy,
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
  IS_MISCONFIGURED,
  linkLabel,
  logClick,
  uploadPdf,
  vote,
  type UploadedDoc,
  type AskResult,
  type Diagnosis,
  type ExplainResult,
  type Health,
  type PaperHit,
  type PapersResult,
  type Passage,
} from "@/lib/api";
import BoxLoader from "@/components/ui/box-loader";
import MorphOrb, { type MorphOrbHandle } from "@/components/ui/ai-thiking-orb-and-input";
import MathText, { extractMath, withMath } from "@/components/ui/math-text";
import PaperFigures from "@/components/ui/paper-figures";
import HalftoneNebula from "@/components/ui/halftone-nebula";
import { LeaderboardRankings } from "@/components/ui/leaderboard-rankings";
import {
  addHistory,
  clearHistory,
  readHistory,
  relativeTime,
  type HistoryEntry,
} from "@/lib/history";

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

// The planet moves right, out from under the headline.
const SKY = { planetX: 0.78, planetY: 0.24 };

export function ResearchConsole() {
  const [mode, setMode] = useState<Mode>("papers");
  const [loading, setLoading] = useState(false);
  // True when the orb is already animating this search, so the page's own
  // skeleton loader would only be a second spinner under it.
  const [animated, setAnimated] = useState(false);
  const orbRef = useRef<MorphOrbHandle>(null);
  // The results panel. `revealed` follows the orb's animation (the panel opens as
  // the orb collapses, not before), `dismissed` is the user closing it.
  const [revealed, setRevealed] = useState(true);
  const [dismissed, setDismissed] = useState(false);
  const [lastQuery, setLastQuery] = useState("");
  const [lastMode, setLastMode] = useState<Mode>("papers");
  const [error, setError] = useState<string | null>(null);
  const [papersResult, setPapersResult] = useState<PapersResult | null>(null);
  const [explainResult, setExplainResult] = useState<ExplainResult | null>(null);
  const [askResult, setAskResult] = useState<AskResult | null>(null);
  const [status, setStatus] = useState<Health | null>(null);
  const [doc, setDoc] = useState<UploadedDoc | null>(null);
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);

  useEffect(() => {
    setHistory(readHistory());
  }, []);

  // The extension hands off here when it cannot draw a panel on the page -
  // Chrome blocks content scripts inside its PDF viewer, which is exactly
  // where papers get read. ?explain=... runs the query on arrival.
  useEffect(() => {
    const handoff = new URLSearchParams(window.location.search).get("explain");
    if (!handoff) return;
    setMode("explain");
    // Clear the parameter so a refresh does not silently re-run it.
    window.history.replaceState({}, "", window.location.pathname);
    document.getElementById("console")?.scrollIntoView({ behavior: "smooth" });
    void run(handoff);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    health().then(setStatus).catch(() => setStatus(null));
  }, []);

  function clear() {
    setPapersResult(null);
    setExplainResult(null);
    setAskResult(null);
    setError(null);
  }

  const clip = (t: string, n: number) => (t.length > n ? `${t.slice(0, n - 1)}…` : t);

  /** Runs a search. Resolves to one short line for the orb's result card. */
  async function run(
    value: string,
    retryOf: number | null = null,
    viaOrb = false,
    asMode: Mode = mode,
  ): Promise<string> {
    const text = value.trim();
    if (text.length < 3) return "Type at least three characters.";
    if (loading) return "Still working on the previous search.";
    setAnimated(viaOrb);
    setDismissed(false);
    setLastQuery(text);
    setLastMode(asMode);
    setLoading(true);
    clear();
    try {
      let summary = "";
      let spoken = "";
      if (asMode === "papers") {
        const res = await findPapers(text, 10, retryOf);
        setPapersResult(res);
        summary = `${res.papers.length} papers`;
        spoken =
          res.papers.length === 0
            ? "Nothing matched that."
            : !res.confident
              ? `${res.papers.length} weak matches. Treat them as leads.`
              : `${res.papers.length} papers found. Top: ${clip(res.papers[0].title, 70)}`;
      } else {
        const res = await explain(text, doc?.document_id, retryOf);
        if (res.resolved) {
          setExplainResult(res);
          summary = res.paper?.title ?? "explained";
          spoken = `Explained: ${clip(res.paper?.title ?? "that paper", 80)}`;
        } else {
          // /explain already answered a question server-side; use that rather
          // than calling /ask, which re-ran the whole pipeline a second time.
          const answered = res.answer_result
            ? { ...res.answer_result, query_id: res.query_id }
            : await ask(text, 6, retryOf);
          setAskResult(answered);
          summary = `${answered.distinct_papers} papers`;
          spoken =
            answered.source === "general_knowledge"
              ? "Answered from general knowledge, no citations."
              : answered.answer
                ? `Answered from ${answered.distinct_papers} papers.`
                : "No answer found.";
        }
      }
      setHistory(addHistory({ query: text, mode: asMode, summary }));
      return spoken;
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      setError(message);
      return `Something went wrong: ${clip(message, 90)}`;
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

  const hasContent = Boolean(papersResult || explainResult || askResult || error);
  const spinner = loading && !animated;
  const open = !dismissed && (spinner || (!loading && hasContent && (!animated || revealed)));
  const chips = (
    mode === "explain" && doc
      ? ["What problem does this solve?", "Explain the method in simple terms", "What are the limitations?"]
      : EXAMPLES[mode]
  ).slice(0, 5);

  // Esc closes the results (while the orb is animating, Esc cancels the animation instead).
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setDismissed(true);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  // The panel opens as the orb collapses; a new search closes it first.
  const onOrbPhase = (phase: string) => {
    if (phase === "launch") setRevealed(false);
    else if (["condense", "unfold", "answered", "reset", "idle"].includes(phase)) setRevealed(true);
  };

  return (
    <main className="h-svh w-full overflow-hidden bg-[#050309]">
      <HalftoneNebula params={SKY} maxDpr={1.5} touch="draw">
        {/* ── top bar ── */}
        <header className="flex items-start justify-between gap-4 p-5 font-mono text-[10px] tracking-[0.3em] text-[#f6e2e8]/60 uppercase sm:p-8">
          <span className="text-[#f6e2e8]/90">ResearchOS</span>
          <span className="text-right tracking-[0.12em] normal-case">
            {status
              ? `${status.papers.toLocaleString()} papers · ${status.chunks.toLocaleString()} passages`
              : IS_MISCONFIGURED
                ? "not connected"
                : "connecting…"}
            {status && !status.generation_enabled && (
              <span className="ml-2 text-amber-400">· no written answers</span>
            )}
          </span>
        </header>

        {/* ── the headline, while nothing is being asked ── */}
        <div
          className={`pointer-events-none absolute inset-x-0 top-[20%] px-6 transition-all duration-500 sm:px-12 ${
            open || loading ? "translate-y-3 opacity-0" : "opacity-100"
          }`}
        >
          <p className="mb-4 font-mono text-[11px] tracking-[0.34em] text-[#ff1f5a] uppercase">
            ✦ {status ? status.papers.toLocaleString() : "…"} papers indexed
          </p>
          <h1 className="max-w-2xl text-5xl leading-[0.95] font-semibold tracking-tight text-[#f6e2e8] sm:text-7xl">
            Ask the
            <br />
            literature.
          </h1>
          <p className="mt-5 max-w-md text-sm leading-relaxed text-[#f6e2e8]/65">
            Find the papers on a topic, or ask what anything means. Every answer is grounded in
            the papers it cites, with their equations and figures.
          </p>

          {!status && (
            <div className="pointer-events-auto mt-6 max-w-md">
              <Callout
                tone="amber"
                title={IS_MISCONFIGURED ? "This deployment isn't finished" : "Waiting for the backend"}
              >
                {IS_MISCONFIGURED ? (
                  <>
                    The site is live but no search backend is reachable from here. Whoever
                    deployed it needs to set <code>NEXT_PUBLIC_API_BASE</code> to a public HTTPS
                    address and redeploy.
                  </>
                ) : (
                  <>
                    The backend loads its index and two models on start, which takes a few
                    minutes. This will clear on its own, no need to refresh.
                  </>
                )}
              </Callout>
            </div>
          )}

          {history.length > 0 && status && (
            // Recent searches are the first thing to give up space: not on phones,
            // and not on short windows where they would run into the example chips.
            <div className="pointer-events-auto mt-6 hidden max-w-xl sm:block [@media(max-height:760px)]:hidden">
              <div className="mb-2 flex items-center gap-3 font-mono text-[10px] tracking-[0.28em] text-[#f6e2e8]/45 uppercase">
                <span className="inline-flex items-center gap-1.5">
                  <Clock size={11} /> Recent
                </span>
                <button
                  onClick={() => setHistory(clearHistory())}
                  className="tracking-[0.2em] transition hover:text-[#f6e2e8]"
                >
                  clear
                </button>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {history.slice(0, 4).map((h) => (
                  <button
                    key={h.id}
                    onClick={() => {
                      setMode(h.mode);
                      void run(h.query, null, false, h.mode);
                    }}
                    title={`${h.summary ?? ""} · ${relativeTime(h.at)}`}
                    className="inline-flex max-w-full items-center gap-1.5 rounded-md border border-white/10 bg-black/30 px-2.5 py-1 text-xs text-[#f6e2e8]/70 transition hover:border-[#ff1f5a]/60 hover:text-[#f6e2e8]"
                  >
                    <span className="shrink-0 text-[10px] text-[#ff1f5a]/80 uppercase">
                      {h.mode === "papers" ? "find" : "explain"}
                    </span>
                    <span className="truncate">{h.query.slice(0, 40)}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* ── above the bar: examples, the mode switch, the PDF ── */}
        <div className="absolute inset-x-0 bottom-[140px] z-10 flex flex-col items-center gap-3 px-4">
          <div
            className={`flex max-w-3xl flex-wrap justify-center gap-2 transition-opacity duration-300 ${
              open || loading ? "pointer-events-none opacity-0" : "opacity-100"
            }`}
          >
            {chips.map((ex, i) => (
              <button
                key={ex}
                onClick={() => orbRef.current?.submit(ex)}
                className={`max-w-[18rem] truncate rounded-full border border-white/12 bg-black/35 px-3 py-1.5 text-xs text-[#f6e2e8]/70 transition hover:border-[#ff1f5a]/60 hover:text-[#f6e2e8] ${
                  i >= 3 ? "hidden sm:inline-block" : ""
                }`}
              >
                {ex.length > 52 ? `${ex.slice(0, 52)}…` : ex}
              </button>
            ))}
          </div>

          <div className="pointer-events-auto flex flex-wrap items-center justify-center gap-2">
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
            {mode === "explain" && (
              <>
                <input
                  ref={fileRef}
                  type="file"
                  accept="application/pdf,.pdf"
                  className="hidden"
                  onChange={(e) => handleFile(e.target.files?.[0])}
                />
                {doc ? (
                  <span className="inline-flex max-w-[16rem] items-center gap-2 rounded-lg border border-sky-400/30 bg-black/50 px-3 py-2 text-xs text-sky-100">
                    <Paperclip size={13} className="shrink-0 text-sky-300" />
                    <span className="truncate">{doc.title}</span>
                    <button
                      onClick={() => setDoc(null)}
                      className="shrink-0 rounded p-0.5 text-sky-300 transition hover:bg-white/10 hover:text-white"
                      aria-label="Stop using this document"
                    >
                      <X size={13} />
                    </button>
                  </span>
                ) : (
                  <button
                    onClick={() => fileRef.current?.click()}
                    disabled={uploading}
                    className="inline-flex items-center gap-1.5 rounded-lg bg-black/35 px-3 py-2 text-xs text-[#f6e2e8]/70 transition hover:text-white disabled:opacity-50"
                  >
                    {uploading ? <Loader2 size={13} className="animate-spin" /> : <Paperclip size={13} />}
                    {uploading ? "Reading PDF…" : "Attach a PDF"}
                  </button>
                )}
              </>
            )}
          </div>
        </div>

        {/* ── results: a panel over the sky, never a new page or a scroll down ── */}
        <div
          data-nebula-ignore
          inert={!open}
          aria-hidden={!open}
          className={`absolute inset-x-3 top-16 bottom-[208px] z-20 mx-auto max-w-4xl transition-all duration-500 ease-out sm:inset-x-6 ${
            open ? "pointer-events-auto translate-y-0 opacity-100" : "pointer-events-none translate-y-6 opacity-0"
          }`}
        >
          <div className="flex h-full flex-col overflow-hidden rounded-2xl border border-white/12 bg-[#06040b]/92 shadow-2xl shadow-black/70">
            <div className="flex items-center gap-3 border-b border-white/10 px-5 py-3">
              <span className="shrink-0 font-mono text-[10px] tracking-[0.28em] text-[#ff1f5a] uppercase">
                {lastMode === "papers" ? "papers" : "explain"}
              </span>
              <span className="min-w-0 flex-1 truncate text-sm text-slate-300">{lastQuery}</span>
              <span className="hidden shrink-0 font-mono text-[10px] tracking-[0.2em] text-slate-600 uppercase sm:inline">
                esc to close
              </span>
              <button
                onClick={() => setDismissed(true)}
                aria-label="Close results"
                className="shrink-0 rounded-md p-1 text-slate-400 transition hover:bg-white/10 hover:text-white"
              >
                <X size={16} />
              </button>
            </div>
            <div
              className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-5 pb-8 [&>div:first-child]:mt-4"
              tabIndex={open ? 0 : -1}
            >
              {spinner && (
                <div className="mt-10 flex flex-col items-center gap-6 py-10">
                  <BoxLoader />
                  <p className="text-sm text-slate-400">
                    {lastMode === "papers"
                      ? "Searching the papers…"
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
              {!loading && papersResult && (
                <PapersView
                  result={papersResult}
                  queryId={queryId}
                  onRetry={(q) => void run(q, queryId, false, lastMode)}
                />
              )}
              {!loading && explainResult && <ExplainView result={explainResult} queryId={queryId} />}
              {!loading && askResult && (
                <AskView
                  result={askResult}
                  queryId={queryId}
                  onRetry={(q) => void run(q, queryId, false, lastMode)}
                />
              )}
            </div>
          </div>
        </div>
      </HalftoneNebula>

      <MorphOrb
        ref={orbRef}
        onSubmit={(text) => run(text, null, true)}
        onPhase={onOrbPhase}
        placeholder={
          mode === "papers"
            ? "A topic, e.g. world models for robot manipulation…"
            : doc
              ? `Ask anything about "${doc.title.slice(0, 40)}"…`
              : "Paste a reference, an arXiv id, a DOI, or ask a question…"
        }
        maxLength={1000}
        answerTitle="Results"
        // The animation's own "thinking" floor is cosmetic (the default is 3.4 s);
        // the real search sets the pace, with just enough for the labels to turn.
        minThinkMs={700}
        speed={1.6}
        autoResetMs={1300}
      />
    </main>
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
        active
          ? "bg-black/55 text-white ring-1 ring-[#ff1f5a]/60"
          : "bg-black/30 text-[#f6e2e8]/65 hover:text-white"
      }`}
    >
      {icon}
      {children}
      <span className="hidden text-xs font-normal text-slate-500 sm:inline">· {hint}</span>
    </button>
  );
}

/**
 * v2: why a search failed, and what to try instead.
 *
 * Shown only when the confidence gate fired and nothing rescued the query -
 * exactly where v1 could only say "nothing scored well enough". The blamed
 * document and the reason come from a CRITIC-R1-format diagnosis; the button
 * runs the critic's suggested query, closing the paper's critique -> refine
 * loop with the user in control of the second search.
 */
function DiagnosisPanel({
  diagnosis,
  onRetry,
}: {
  diagnosis: Diagnosis;
  onRetry: (query: string) => void;
}) {
  const blamed =
    diagnosis.location === "search"
      ? "The query itself"
      : diagnosis.location_title
        ? `“${diagnosis.location_title}”`
        : diagnosis.location;
  return (
    <section
      aria-label="Why this search struggled"
      className="rounded-2xl border border-sky-400/25 bg-sky-400/[0.05] p-5"
    >
      <div className="flex items-center gap-2 text-xs font-medium tracking-wide text-sky-300 uppercase">
        <TriangleAlert size={14} aria-hidden />
        Why this search struggled
      </div>
      <p className="mt-3 text-sm text-slate-200">
        <span className="text-slate-400">Weakest evidence: </span>
        {blamed}
      </p>
      <p className="mt-1.5 text-sm leading-relaxed text-slate-300">{diagnosis.reason}</p>
      {diagnosis.suggested_query && (
        <button
          type="button"
          onClick={() => onRetry(diagnosis.suggested_query!)}
          className="mt-4 inline-flex items-center gap-2 rounded-lg border border-sky-400/40 bg-sky-400/10 px-3.5 py-2 text-sm text-sky-100 transition hover:bg-sky-400/20 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-300/60"
        >
          <Search size={14} aria-hidden />
          Try: {diagnosis.suggested_query}
        </button>
      )}
      <p className="mt-3 text-[11px] text-slate-500">
        Structured diagnosis in the CRITIC-R1 format · {(diagnosis.latency_ms / 1000).toFixed(1)}s
      </p>
    </section>
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

function CopyAnswer({ text, cites }: { text: string; cites: { paper_id: string; title: string; arxiv_url: string }[] }) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    // Citations are appended as real references - an answer pasted into notes
    // is worthless if the [2312.00752] tags lead nowhere.
    const seen = new Set<string>();
    const refs = cites
      .filter((c) => c.paper_id && !seen.has(c.paper_id) && seen.add(c.paper_id))
      .map((c) => `[${c.paper_id}] ${c.title}${c.arxiv_url ? ` — ${c.arxiv_url}` : ""}`)
      .join("\n");
    try {
      const payload = refs ? `${text}\n\nReferences\n${refs}` : text;
      await navigator.clipboard.writeText(payload);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      /* clipboard blocked - the text is still on screen */
    }
  }

  return (
    <button
      onClick={copy}
      className="inline-flex items-center gap-1.5 text-xs text-slate-500 transition hover:text-slate-300"
    >
      {copied ? <Check size={13} /> : <Copy size={13} />}
      {copied ? "Copied with references" : "Copy answer"}
    </button>
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

function PapersView({
  result,
  queryId,
  onRetry,
}: {
  result: PapersResult;
  queryId: number | null;
  onRetry: (query: string) => void;
}) {
  return (
    <div className="mt-8 space-y-6">
      {result.diagnosis && <DiagnosisPanel diagnosis={result.diagnosis} onRetry={onRetry} />}
      <Meta
        items={[
          `${result.papers.length} papers`,
          `${result.timing_ms.total.toFixed(0)} ms`,
          `top ${result.top_score.toFixed(3)}`,
        ]}
      />
      {/* Ranked results as a leaderboard: rank number, medal for the top three,
          title, byline, and the reranker score on the right. Clicking a row
          logs the click (the feedback signal the backend learns from) and opens
          the paper on arXiv.
          `border-border` is explicit because Tailwind v4's bare `border`
          utility sets width only and falls back to currentColor. */}
      <LeaderboardRankings
        className="border-border bg-transparent"
        showAvatar={false}
        rankings={result.papers.map((p, i) => ({
          userId: p.paper_id,
          rank: i + 1,
          userName: p.title,
          byline: [
            p.authors.slice(0, 3).join(", ") + (p.authors.length > 3 ? " et al." : ""),
            p.published,
            p.venue && p.venue !== "arXiv" ? p.venue : p.primary_category,
            typeof p.cited_by_count === "number" && p.cited_by_count > 0
              ? `${p.cited_by_count.toLocaleString()} citations`
              : "",
            p.from_arxiv
              ? `live · ${p.live_source === "openalex" ? "OpenAlex" : p.live_source === "arxiv+openalex" ? "arXiv + OpenAlex" : "arXiv"}`
              : "",
          ]
            .filter(Boolean)
            .join(" · "),
          value: p.score,
          avatarUrl: null,
          displayed: true,
        }))}
        onUserClick={(item) => {
          const paper = result.papers.find((p) => p.paper_id === item.userId);
          if (!paper) return;
          if (queryId !== null) logClick(queryId, paper.doc_id).catch(() => {});
          window.open(paper.arxiv_url, "_blank", "noopener,noreferrer");
        }}
      />
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
                live · {paper.live_source === "openalex" ? "OpenAlex"
                  : paper.live_source === "arxiv+openalex" ? "arXiv + OpenAlex" : "arXiv"}
              </span>
            )}
            {paper.venue && paper.venue !== "arXiv" && (
              <span className="text-slate-500">{paper.venue}</span>
            )}
            {typeof paper.cited_by_count === "number" && paper.cited_by_count > 0 && (
              <span className="text-slate-500">
                {paper.cited_by_count.toLocaleString()} citations
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
            <p className="text-sm leading-relaxed text-slate-300"><MathText text={paper.why} />…</p>
          </div>
          {paper.abstract && (
            <div>
              <p className="mb-1 text-[11px] tracking-wider text-slate-500 uppercase">Abstract</p>
              <p className="text-sm leading-relaxed text-slate-400">
                <MathText text={paper.abstract} />
              </p>
            </div>
          )}
          <a
            href={paper.arxiv_url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-xs text-sky-400 hover:underline"
          >
            Open on {linkLabel(paper.arxiv_url)} <ArrowUpRight size={12} />
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
            {linkLabel(paper.arxiv_url)} <ArrowUpRight size={13} />
          </a>
        </div>
      </div>

      <Meta
        items={[
          `resolved via ${result.method.replace("_", " ")}`,
          result.from_corpus ? "from local corpus" : `fetched live from ${linkLabel(paper.arxiv_url) === "arXiv" ? "arXiv" : "OpenAlex"}`,
          paper.has_full_text ? "full text" : "abstract only",
          `${(result.timing_ms.total / 1000).toFixed(1)} s`,
        ]}
      />

      {result.explanation && (
        <article className="rounded-2xl border border-white/10 bg-white/[0.03] p-6">
          <Prose text={result.explanation} />
          <div className="mt-4 border-t border-white/10 pt-3">
            <CopyAnswer text={result.explanation} cites={[...result.passages, ...result.related]} />
          </div>
        </article>
      )}

      <PaperFigures
        paperIds={[paper.paper_id]}
        question={result.citation}
        titles={{ [paper.paper_id]: paper.title }}
        perPaper={3}
      />

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

function AskView({
  result,
  queryId,
  onRetry,
}: {
  result: AskResult;
  queryId: number | null;
  onRetry: (query: string) => void;
}) {
  return (
    <div className="mt-8 space-y-6">
      {result.diagnosis && <DiagnosisPanel diagnosis={result.diagnosis} onRetry={onRetry} />}
      {result.answer && (
        <article className="rounded-2xl border border-white/10 bg-white/[0.03] p-6">
          <Prose text={result.answer} />
          <div className="mt-4 border-t border-white/10 pt-3">
            <CopyAnswer text={result.answer} cites={result.citations} />
          </div>
        </article>
      )}
      {/* No banner: an answer that is not drawn from papers is marked by a quiet
          tag here and by the "Related papers" heading below, which is enough to
          tell it apart from a cited answer without a warning box. */}
      <Meta
        items={[
          ...(result.grounded === false ? ["general knowledge · no citations"] : []),
          `${result.distinct_papers} papers`,
          `${result.timing_ms.total.toFixed(0)} ms`,
          `mean ${result.mean_top_score.toFixed(3)}`,
        ]}
      />
      {result.grounded !== false && (
        <PaperFigures
          paperIds={result.citations.map((c) => c.paper_id)}
          question={result.question}
          titles={Object.fromEntries(result.citations.map((c) => [c.paper_id, c.title]))}
        />
      )}
      <PassageList
        title={result.grounded === false ? "Related papers" : "Sources"}
        passages={result.citations}
        queryId={queryId}
      />
      <FeedbackBar queryId={queryId} />
    </div>
  );
}

/* -------------------------------------------------------------- primitives */

function Prose({ text }: { text: string }) {
  // Equations come out before the text is split into lines: a display equation
  // spans several, and "*" or "_" inside one is not emphasis.
  const { text: body, items } = extractMath(text);
  return (
    <div className="space-y-3 text-[15px] leading-relaxed text-slate-200">
      {body
        .split("\n")
        .filter((l) => l.trim())
        .map((line, i) => (
          <p key={i}>
            {line
              .split(/(\*\*[^*]+\*\*|`[^`]+`|\[\d{4}\.\d{4,5}[^\]]*\])/g)
              .map((part, j) => {
                const key = `m${i}-${j}-`;
                if (part.startsWith("**") && part.endsWith("**"))
                  return (
                    <strong key={j} className="font-semibold text-white">
                      {withMath(part.slice(2, -2), items, key)}
                    </strong>
                  );
                if (part.startsWith("`") && part.endsWith("`"))
                  return (
                    <code key={j} className="rounded bg-white/10 px-1.5 py-0.5 font-mono text-[13px]">
                      {withMath(part.slice(1, -1), items, key)}
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
                return <span key={j}>{withMath(part, items, key)}</span>;
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
          <p className="text-sm leading-relaxed text-slate-300">
            <MathText text={passage.text} />
          </p>
          <a
            href={passage.arxiv_url}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-3 inline-flex items-center gap-1 text-xs text-sky-400 hover:underline"
          >
            Open on {linkLabel(passage.arxiv_url)} <ArrowUpRight size={12} />
          </a>
        </div>
      )}
    </div>
  );
}
