/**
 * Local search history.
 *
 * Someone reading papers daily re-runs the same handful of queries and wants
 * yesterday's answer back without retyping it. Kept in localStorage rather
 * than the server: it is per-person, per-device, and nobody should need an
 * account to get their own history back.
 *
 * Every access is wrapped - localStorage throws outright in private windows
 * and when a browser blocks site data, and a history feature must never be
 * the reason the page fails to render.
 */

export interface HistoryEntry {
  id: string;
  query: string;
  mode: "papers" | "explain";
  at: number;
  /** Enough to show what came back without storing the whole payload. */
  summary?: string;
}

const KEY = "researchos.history.v1";
const LIMIT = 40;

export function readHistory(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as HistoryEntry[]) : [];
  } catch {
    return [];
  }
}

export function addHistory(entry: Omit<HistoryEntry, "id" | "at">): HistoryEntry[] {
  const list = readHistory();
  // Re-running a query moves it to the top rather than duplicating it.
  const deduped = list.filter(
    (e) => !(e.query === entry.query && e.mode === entry.mode),
  );
  const next: HistoryEntry[] = [
    { ...entry, id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`, at: Date.now() },
    ...deduped,
  ].slice(0, LIMIT);
  try {
    localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    /* quota or blocked storage - history is a convenience, not a requirement */
  }
  return next;
}

export function clearHistory(): HistoryEntry[] {
  try {
    localStorage.removeItem(KEY);
  } catch {
    /* ignore */
  }
  return [];
}

export function relativeTime(at: number): string {
  const mins = Math.round((Date.now() - at) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}
