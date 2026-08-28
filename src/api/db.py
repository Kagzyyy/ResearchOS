"""SQLite logging: queries, citations, clicks, votes and failures.

This is the instrumentation that makes a small deployment worth running. With
several people using it, the interesting question is not "did it work" but
"where is it failing, and on what kind of question" - which needs the failures
recorded as deliberately as the successes.

Four signals, in ascending order of how much they tell you:

  clicks   which citation someone opened. Weak, plentiful, unbiased.
  votes    an explicit thumbs up/down. Strong, scarce, self-selected.
  low      queries the retriever itself flagged as out-of-corpus. Free, and the
           fastest route to "what do people ask that this cannot answer".
  errors   exceptions, with endpoint and query. Silent otherwise.
"""

from __future__ import annotations

import sqlite3
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    query       TEXT    NOT NULL,
    answer      TEXT,
    latency_ms  REAL,
    backend     TEXT
);

CREATE TABLE IF NOT EXISTS citations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id    INTEGER NOT NULL REFERENCES queries(id),
    doc_id      TEXT    NOT NULL,
    rank        INTEGER NOT NULL,
    score       REAL,
    title       TEXT
);

CREATE TABLE IF NOT EXISTS citation_clicks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    query_id    INTEGER NOT NULL REFERENCES queries(id),
    doc_id      TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS votes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    query_id    INTEGER NOT NULL REFERENCES queries(id),
    helpful     INTEGER NOT NULL,
    comment     TEXT
);

CREATE TABLE IF NOT EXISTS errors (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    endpoint    TEXT    NOT NULL,
    query       TEXT,
    error_type  TEXT    NOT NULL,
    message     TEXT,
    stack       TEXT
);

CREATE INDEX IF NOT EXISTS idx_citations_query ON citations(query_id);
CREATE INDEX IF NOT EXISTS idx_clicks_query    ON citation_clicks(query_id);
CREATE INDEX IF NOT EXISTS idx_votes_query     ON votes(query_id);
CREATE INDEX IF NOT EXISTS idx_errors_ts       ON errors(ts);
"""

# Columns added after the first release. SQLite has no "ADD COLUMN IF NOT
# EXISTS", so they are applied one at a time and duplicates ignored.
MIGRATIONS = [
    ("queries", "mode", "TEXT"),
    ("queries", "confident", "INTEGER"),
    ("queries", "top_score", "REAL"),
    ("queries", "mean_top_score", "REAL"),
    ("queries", "n_results", "INTEGER"),
    ("queries", "generated", "INTEGER"),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class FeedbackLog:
    """Thin wrapper over a SQLite file. Safe to construct per request."""

    def __init__(self, path: str = "results/feedback.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            for table, column, coltype in MIGRATIONS:
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
                except sqlite3.OperationalError:
                    pass  # already present

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------- recording

    def log_query(
        self,
        query: str,
        answer: str | None,
        latency_ms: float,
        backend: str,
        citations: list[dict[str, Any]],
        mode: str = "ask",
        confident: bool | None = None,
        top_score: float | None = None,
        mean_top_score: float | None = None,
        generated: bool | None = None,
    ) -> int:
        """Record a query, its answer, and the passages shown."""
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO queries (ts, query, answer, latency_ms, backend, mode, "
                "confident, top_score, mean_top_score, n_results, generated) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    _now(), query, answer, latency_ms, backend, mode,
                    None if confident is None else int(confident),
                    top_score, mean_top_score, len(citations),
                    None if generated is None else int(generated),
                ),
            )
            query_id = int(cursor.lastrowid)
            conn.executemany(
                "INSERT INTO citations (query_id, doc_id, rank, score, title) VALUES (?, ?, ?, ?, ?)",
                [
                    (query_id, c.get("doc_id", ""), rank, c.get("score"), c.get("title"))
                    for rank, c in enumerate(citations, start=1)
                ],
            )
        return query_id

    def log_click(self, query_id: int, doc_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO citation_clicks (ts, query_id, doc_id) VALUES (?, ?, ?)",
                (_now(), query_id, doc_id),
            )

    def log_vote(self, query_id: int, helpful: bool, comment: str | None = None) -> None:
        """Explicit thumbs up/down. The scarcest and most valuable signal."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO votes (ts, query_id, helpful, comment) VALUES (?, ?, ?, ?)",
                (_now(), query_id, int(helpful), comment),
            )

    def log_error(
        self, endpoint: str, error: BaseException, query: str | None = None
    ) -> None:
        """Record an exception. Without this, failures are invisible."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO errors (ts, endpoint, query, error_type, message, stack) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    _now(), endpoint, query, type(error).__name__, str(error)[:2000],
                    "".join(traceback.format_exception(type(error), error, error.__traceback__))[-4000:],
                ),
            )

    # ------------------------------------------------------------- reporting

    def stats(self) -> dict[str, int]:
        with self._connect() as conn:
            def count(sql: str) -> int:
                return int(conn.execute(sql).fetchone()[0])

            return {
                "queries": count("SELECT COUNT(*) FROM queries"),
                "citations": count("SELECT COUNT(*) FROM citations"),
                "clicks": count("SELECT COUNT(*) FROM citation_clicks"),
                "votes_up": count("SELECT COUNT(*) FROM votes WHERE helpful=1"),
                "votes_down": count("SELECT COUNT(*) FROM votes WHERE helpful=0"),
                "errors": count("SELECT COUNT(*) FROM errors"),
                "low_confidence": count("SELECT COUNT(*) FROM queries WHERE confident=0"),
            }

    def recent_failures(self, limit: int = 50) -> dict[str, list[dict[str, Any]]]:
        """Everything worth reading when asking 'how is this failing?'."""
        with self._connect() as conn:
            def rows(sql: str, *args: Any) -> list[dict[str, Any]]:
                return [dict(r) for r in conn.execute(sql, args).fetchall()]

            return {
                "errors": rows(
                    "SELECT ts, endpoint, query, error_type, message FROM errors "
                    "ORDER BY id DESC LIMIT ?", limit
                ),
                "downvoted": rows(
                    "SELECT v.ts, q.query, q.mode, v.comment FROM votes v "
                    "JOIN queries q ON q.id = v.query_id WHERE v.helpful = 0 "
                    "ORDER BY v.id DESC LIMIT ?", limit
                ),
                "out_of_corpus": rows(
                    "SELECT ts, query, mode, mean_top_score FROM queries "
                    "WHERE confident = 0 ORDER BY id DESC LIMIT ?", limit
                ),
                "slowest": rows(
                    "SELECT ts, query, mode, latency_ms FROM queries "
                    "WHERE latency_ms IS NOT NULL ORDER BY latency_ms DESC LIMIT ?", limit
                ),
                "never_clicked": rows(
                    "SELECT q.ts, q.query, q.mode FROM queries q "
                    "LEFT JOIN citation_clicks c ON c.query_id = q.id "
                    "WHERE c.id IS NULL AND q.confident = 1 ORDER BY q.id DESC LIMIT ?", limit
                ),
            }
