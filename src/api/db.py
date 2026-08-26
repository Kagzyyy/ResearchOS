"""SQLite logging for queries, the citations shown, and which ones got clicked.

The click log is the feedback signal a future retraining loop would consume.
It is recorded now; the loop itself is designed but not running (see README,
"Limitations and next steps").
"""

from __future__ import annotations

import sqlite3
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

CREATE INDEX IF NOT EXISTS idx_citations_query ON citations(query_id);
CREATE INDEX IF NOT EXISTS idx_clicks_query    ON citation_clicks(query_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class FeedbackLog:
    """Thin wrapper over a SQLite file. Safe to construct per request."""

    def __init__(self, path: str = "results/feedback.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def log_query(
        self,
        query: str,
        answer: str | None,
        latency_ms: float,
        backend: str,
        citations: list[dict[str, Any]],
    ) -> int:
        """Record a query, its answer, and the passages that were shown."""
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO queries (ts, query, answer, latency_ms, backend) VALUES (?, ?, ?, ?, ?)",
                (_now(), query, answer, latency_ms, backend),
            )
            query_id = int(cursor.lastrowid)
            conn.executemany(
                "INSERT INTO citations (query_id, doc_id, rank, score, title) VALUES (?, ?, ?, ?, ?)",
                [
                    (query_id, c["doc_id"], rank, c.get("score"), c.get("title"))
                    for rank, c in enumerate(citations, start=1)
                ],
            )
        return query_id

    def log_click(self, query_id: int, doc_id: str) -> None:
        """Record that a user opened a specific cited passage."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO citation_clicks (ts, query_id, doc_id) VALUES (?, ?, ?)",
                (_now(), query_id, doc_id),
            )

    def stats(self) -> dict[str, int]:
        """Counts, for the UI footer and for sanity-checking the log."""
        with self._connect() as conn:
            return {
                "queries": conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0],
                "citations": conn.execute("SELECT COUNT(*) FROM citations").fetchone()[0],
                "clicks": conn.execute("SELECT COUNT(*) FROM citation_clicks").fetchone()[0],
            }
