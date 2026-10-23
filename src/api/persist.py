"""Keep the feedback database across restarts of a free Hugging Face Space.

A free Space's disk is ephemeral: every restart, rebuild or sleep starts from
the repository again. The feedback database - queries, clicks, votes and the
v2 retry funnel, including the retried-then-clicked pairs meant to become
training data - would be wiped each time, so the live measurement could
never accumulate.

When FEEDBACK_REPO (a dataset repo id, e.g. "you/researchos-feedback") and
HF_TOKEN (a write token) are set as Space secrets, the database is restored
from that private dataset on boot and snapshotted back every few minutes.
Unset, nothing happens: local runs keep their file on a real disk anyway.

The snapshot uses SQLite's online backup API, so it is consistent even while
requests are writing; copying the file directly could capture a torn page.
It is only uploaded when the database actually changed.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
import threading
import time
from contextlib import closing
from pathlib import Path

SYNC_EVERY_S = 600.0


def _settings() -> tuple[str, str] | None:
    repo, token = os.environ.get("FEEDBACK_REPO"), os.environ.get("HF_TOKEN")
    return (repo, token) if repo and token else None


def restore(db_path: str) -> bool:
    """Pull the last snapshot down before the app opens the database."""
    settings = _settings()
    if settings is None or Path(db_path).exists():
        return False
    repo, token = settings
    try:
        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(repo, Path(db_path).name, repo_type="dataset", token=token)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, db_path)
        print(f"[persist] restored {db_path} from {repo}", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001 - first boot has nothing to restore
        print(f"[persist] nothing restored ({type(exc).__name__}); starting fresh", flush=True)
        return False


def _fingerprint(db_path: str) -> tuple[int, ...]:
    # closing(): sqlite3's own context manager commits but never closes.
    with closing(sqlite3.connect(db_path)) as conn:
        return tuple(
            conn.execute(f"SELECT COALESCE(MAX(id), 0) FROM {t}").fetchone()[0]
            for t in ("queries", "citation_clicks", "votes", "errors")
        )


def snapshot_once(db_path: str, repo: str, token: str, last: tuple | None) -> tuple | None:
    """Upload a consistent copy if anything changed. Returns the new fingerprint."""
    current = _fingerprint(db_path)
    if current == last:
        return last
    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory() as tmp:
        snap = Path(tmp) / Path(db_path).name
        with closing(sqlite3.connect(db_path)) as src, closing(sqlite3.connect(snap)) as dst:
            src.backup(dst)
        api = HfApi(token=token)
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=str(snap), path_in_repo=snap.name, repo_id=repo,
                        repo_type="dataset", commit_message="feedback snapshot")
    return current


def start_sync(db_path: str, every_s: float = SYNC_EVERY_S) -> threading.Thread | None:
    """Snapshot in the background for the life of the process."""
    settings = _settings()
    if settings is None:
        return None
    repo, token = settings

    def loop() -> None:
        last = None
        while True:
            time.sleep(every_s)
            try:
                last = snapshot_once(db_path, repo, token, last)
            except Exception as exc:  # noqa: BLE001 - never take the app down
                print(f"[persist] snapshot failed: {type(exc).__name__}: {exc}", flush=True)

    thread = threading.Thread(target=loop, name="feedback-sync", daemon=True)
    thread.start()
    print(f"[persist] syncing {db_path} to {repo} every {every_s / 60:.0f} min", flush=True)
    return thread
