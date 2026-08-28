/** Typed client for the ResearchOS-ML FastAPI backend. */

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8100";

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
}

export interface PapersResult {
  query_id: number;
  topic: string;
  papers: PaperHit[];
  confident: boolean;
  top_score: number;
  mean_top_score: number;
  timing_ms: { total: number };
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

export const findPapers = (topic: string, topK = 10) =>
  post<PapersResult>("/papers", { topic, top_k: topK });

export const ask = (query: string, topK = 6) =>
  post<AskResult>("/ask", { query, top_k: topK, generate: true });

export const explain = (citation: string) =>
  post<ExplainResult>("/explain", { citation, generate: true });

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
