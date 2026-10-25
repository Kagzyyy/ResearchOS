/** Typed client for the ResearchOS-ML FastAPI backend. */

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8100";

/**
 * A deployed site pointing at localhost is broken for every visitor except the
 * person who built it - their browser resolves localhost to their own machine.
 * Detecting it lets the UI explain that, instead of showing a generic network
 * error that looks like the whole app is down.
 */
export const IS_MISCONFIGURED =
  typeof window !== "undefined" &&
  !/^(localhost|127\.0\.0\.1)$/.test(window.location.hostname) &&
  /^https?:\/\/(localhost|127\.0\.0\.1)/.test(API_BASE);

/**
 * Shared access token. Inlined into the client bundle, so it is a deterrent
 * against drive-by abuse of your Groq quota, not a secret. Anyone who can load
 * the page can read it. Do not use it to protect anything that matters.
 */
const API_TOKEN = process.env.NEXT_PUBLIC_API_TOKEN ?? "";

function headers(): Record<string, string> {
  const h: Record<string, string> = { "Content-Type": "application/json" };
  if (API_TOKEN) h["Authorization"] = `Bearer ${API_TOKEN}`;
  return h;
}

export interface Passage {
  chunk_id: string;
  doc_id: string;
  paper_id: string;
  arxiv_url: string;
  score: number;
  title: string;
  section: string;
  published: string;
  text: string;
}

/** v2: structured failure diagnosis (CRITIC-R1 format), present only when
 *  the confidence gate fired and nothing rescued the query. */
export interface Diagnosis {
  location: string;              // "search" or "information:DocK"
  location_title: string | null; // title of the document blamed, if any
  reason: string;
  fix: string;
  suggested_query: string | null;
  model: string;
  latency_ms: number;
}

export interface AskResult {
  query_id: number;
  question: string;
  answer: string | null;
  confident: boolean;
  top_score: number;
  mean_top_score: number;
  citations: Passage[];
  distinct_papers: number;
  generation_configured: boolean;
  timing_ms: { retrieval_and_rerank: number; generation: number; total: number };
  diagnosis?: Diagnosis | null;
  /** "corpus" | "arxiv_live" | "corpus+arxiv_live" | "general_knowledge" | "upload" */
  source?: string;
  /** False when the answer is the model's own knowledge, not drawn from the papers. */
  grounded?: boolean;
}

export interface PaperMeta {
  paper_id: string;
  title: string;
  abstract: string;
  authors: string[];
  published: string;
  primary_category: string;
  arxiv_url: string;
  has_full_text: boolean;
  from_arxiv?: boolean;
}

export interface ExplainResult {
  query_id: number;
  citation: string;
  resolved: boolean;
  method: string;
  query_used: string;
  from_corpus?: boolean;
  candidates: { paper_id: string; title: string; score: string }[];
  paper: PaperMeta | null;
  explanation: string | null;
  passages: Passage[];
  related: Passage[];
  generation_configured: boolean;
  timing_ms: { total: number; generation?: number };
  /** Full /ask payload when a question was answered here, so the client
   *  never re-runs the pipeline with a second /ask call. */
  answer_result?: AskResult;
}

export interface PaperHit {
  paper_id: string;
  doc_id: string;
  title: string;
  authors: string[];
  abstract: string;
  published: string;
  primary_category: string;
  arxiv_url: string;
  score: number;
  matched_passages: number;
  matched_sections: string[];
  why: string;
  why_section: string;
  has_full_text: boolean;
  /** Fetched live rather than from the index (any source, despite the name). */
  from_arxiv?: boolean;
  live_source?: "arxiv" | "openalex" | "arxiv+openalex";
  venue?: string;
  cited_by_count?: number | null;
}

export interface PapersResult {
  query_id: number;
  topic: string;
  source?: string;
  papers: PaperHit[];
  confident: boolean;
  top_score: number;
  mean_top_score: number;
  timing_ms: { total: number; diagnosis?: number };
  diagnosis?: Diagnosis | null;
}

export interface Health {
  status: string;
  chunks: number;
  papers: number;
  reranker_model: string | null;
  llm_model: string | null;
  generation_enabled: boolean;
  auth_required?: boolean;
  rate_limit_per_minute?: number;
  live_sources?: Record<string, { available: boolean; remaining_usd?: number | null }>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: headers(),
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    if (res.status === 429) throw new Error("Rate limited — wait a minute and try again.");
    if (res.status === 401) throw new Error("Access token rejected. Check NEXT_PUBLIC_API_TOKEN.");
    throw new Error(`${path} failed: ${res.status} ${detail}`);
  }
  return res.json() as Promise<T>;
}

/**
 * `retryOf` is the query_id whose v2 diagnosis suggested this query. It is
 * what lets the backend measure whether following a suggestion helps.
 */
export const findPapers = (topic: string, topK = 10, retryOf: number | null = null) =>
  post<PapersResult>("/papers", { topic, top_k: topK, retry_of: retryOf });

export const ask = (query: string, topK = 6, retryOf: number | null = null) =>
  post<AskResult>("/ask", { query, top_k: topK, generate: true, retry_of: retryOf });

export const explain = (
  citation: string,
  documentId?: string | null,
  retryOf: number | null = null,
) =>
  post<ExplainResult>("/explain", {
    citation,
    generate: true,
    document_id: documentId ?? null,
    retry_of: retryOf,
  });

/** Where a paper link actually goes, for its label. */
export function linkLabel(url: string): string {
  if (/arxiv\.org/i.test(url)) return "arXiv";
  if (/doi\.org/i.test(url)) return "DOI";
  return "source";
}

export interface UploadedDoc {
  document_id: string;
  title: string;
  n_chunks: number;
  n_words: number;
  filename: string;
}

/** Upload a PDF. Multipart, so it does not go through post()/JSON headers. */
export async function uploadPdf(file: File): Promise<UploadedDoc> {
  const body = new FormData();
  body.append("file", file);
  const auth: Record<string, string> = {};
  if (API_TOKEN) auth["Authorization"] = `Bearer ${API_TOKEN}`;
  const res = await fetch(`${API_BASE}/upload`, { method: "POST", headers: auth, body });
  if (!res.ok) {
    let detail = await res.text().catch(() => "");
    try {
      detail = JSON.parse(detail).detail ?? detail;
    } catch {
      /* plain-text body */
    }
    throw new Error(detail || `Upload failed (${res.status})`);
  }
  return res.json() as Promise<UploadedDoc>;
}

export const logClick = (queryId: number, docId: string) =>
  post<{ logged: boolean }>("/feedback/click", { query_id: queryId, doc_id: docId });

export const vote = (queryId: number, helpful: boolean, comment?: string) =>
  post<{ logged: boolean }>("/feedback/vote", {
    query_id: queryId,
    helpful,
    comment: comment ?? null,
  });

export async function health(): Promise<Health> {
  const res = await fetch(`${API_BASE}/health`, { headers: headers() });
  if (!res.ok) throw new Error(`health failed: ${res.status}`);
  return res.json();
}
