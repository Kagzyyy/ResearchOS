"""Make sure the corpus and its prebuilt indices are on disk before the app loads them.

The 273k-paper corpus is ~1.6 GB, past what a free Hugging Face Space can hold
in its own repo (1 GB). It lives in a public dataset repo instead and is
downloaded once per container start, straight to the paths the config names.
Files already on disk (a laptop, a warm container) are used as they are and
nothing touches the network.

If the download fails - dataset repo unreachable, a rate limit, a half-finished
upload - the app must still come up. The config can name a smaller `fallback`
corpus bundled with the Space; resolution switches to it and says so loudly,
so a degraded deploy is visible in /health instead of being a boot failure.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


def _complete(paths: dict[str, Any]) -> bool:
    """Are the corpus files and both index caches present?"""
    cache = Path(paths["index_cache"])
    return (
        Path(paths["papers"]).exists()
        and Path(paths["chunks"]).exists()
        and (cache / "bm25" / "corpus.json").exists()
        and any(cache.glob("dense_*.npy"))
    )


def _patterns(paths: dict[str, Any]) -> list[str]:
    corpus_dir = Path(paths["papers"]).parent.as_posix()
    return [f"{corpus_dir}/*", f"{Path(paths['index_cache']).as_posix()}/**"]


def ensure_corpus(cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve the corpus, mutating cfg["paths"] only if it has to fall back.

    Returns {"source": "local" | "downloaded" | "fallback", ...} for /health.
    """
    spec = cfg.get("corpus") or {}
    paths = cfg["paths"]
    if _complete(paths):
        return {"source": "local"}

    repo = spec.get("repo")
    error = "no corpus.repo configured"
    if repo:
        try:
            from huggingface_hub import snapshot_download

            started = time.time()
            print(f"[corpus] downloading {repo} (first start of this container) ...", flush=True)
            snapshot_download(repo_id=repo, repo_type="dataset", local_dir=".",
                              allow_patterns=_patterns(paths), max_workers=8)
            if _complete(paths):
                took = time.time() - started
                print(f"[corpus] ready in {took:.0f}s", flush=True)
                return {"source": "downloaded", "repo": repo, "seconds": round(took)}
            error = "download finished but files are missing"
        except Exception as exc:  # noqa: BLE001 - any failure must degrade, not crash
            error = f"{type(exc).__name__}: {str(exc)[:160]}"

    fallback = spec.get("fallback")
    if fallback:
        merged = dict(paths)
        merged.update(fallback)
        if _complete(merged):
            paths.update(fallback)
            print(f"[corpus] WARNING: using the bundled fallback corpus ({error})", flush=True)
            return {"source": "fallback", "reason": error}
    raise RuntimeError(f"corpus unavailable and no usable fallback ({error})")
