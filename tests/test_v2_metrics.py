"""Tests for the live v2 funnel and label export. Offline, temp SQLite.

Run: `python tests/test_v2_metrics.py`
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.db import FeedbackLog  # noqa: E402

DIAG = {"location": "information:Doc2", "suggested_query": "conditional computation sparse experts"}


def _log() -> FeedbackLog:
    return FeedbackLog(str(Path(tempfile.mkdtemp()) / "f.db"))


def _q(log, query, confident, diagnosis=None, retry_of=None, source="corpus"):
    return log.log_query(query=query, answer=None, latency_ms=10.0, backend="retrieval",
                         citations=[{"doc_id": f"{query}-d1"}], mode="papers",
                         confident=confident, mean_top_score=0.2, source=source,
                         diagnosis=diagnosis, retry_of=retry_of)


def test_funnel_counts_retries_and_clicks():
    log = _log()
    a = _q(log, "vague a", False, DIAG)
    b = _q(log, "vague b", False, DIAG)
    _q(log, "vague c", False, DIAG)                    # shown, never retried
    ra = _q(log, DIAG["suggested_query"], True, retry_of=a)
    _q(log, DIAG["suggested_query"], True, retry_of=b)
    log.log_click(ra, "paper-1")                       # retry a led to a paper
    m = log.v2_metrics()
    assert m["diagnoses_shown"] == 3 and m["diagnoses_with_query"] == 3
    assert m["retries"] == 2 and m["retry_rate"] == round(2 / 3, 4)
    assert m["retry_click_through"] == 0.5
    assert m["baseline_queries"] == 1                  # only "vague c"
    assert m["diagnosis_locations"] == {"information": 3}


def test_baseline_excludes_queries_that_were_retried():
    """A retried query's own clicks belong to the retry arm, not the baseline."""
    log = _log()
    a = _q(log, "vague", False, DIAG)
    log.log_click(a, "x")
    _q(log, "better", True, retry_of=a)
    assert log.v2_metrics()["baseline_queries"] == 0


def test_forged_retry_id_is_dropped():
    log = _log()
    rid = _q(log, "q", True, retry_of=999)
    with sqlite3.connect(log.path) as conn:
        assert conn.execute("SELECT retry_of FROM queries WHERE id=?", (rid,)).fetchone()[0] is None


def test_labels_are_retried_then_clicked_pairs():
    log = _log()
    a = _q(log, "half-remembered moe paper", False, DIAG)
    r = _q(log, DIAG["suggested_query"], True, retry_of=a)
    log.log_click(r, "1701.06538")
    _q(log, "other", False, DIAG)
    labels = log.v2_labels()
    assert len(labels) == 1
    assert labels[0]["original_query"] == "half-remembered moe paper"
    assert labels[0]["clicked_doc"] == "1701.06538"


def test_old_database_is_migrated_in_place():
    """A feedback.db from before v2 must gain the columns, not break."""
    path = Path(tempfile.mkdtemp()) / "old.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE queries (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, "
                     "query TEXT NOT NULL, answer TEXT, latency_ms REAL, backend TEXT)")
    log = FeedbackLog(str(path))
    _q(log, "q", False, DIAG)
    assert log.v2_metrics()["diagnoses_shown"] == 1


def test_empty_database_reports_no_rates():
    m = _log().v2_metrics()
    assert m["retries"] == 0 and m["retry_rate"] is None and m["retry_click_through"] is None


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
