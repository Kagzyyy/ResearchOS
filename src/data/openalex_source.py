"""OpenAlex: a second live source, for papers arXiv does not have.

arXiv covers preprints. A large share of what researchers cite never went
there: conference papers with no preprint, journal articles, DOI-only
references. OpenAlex indexes ~250M scholarly works under CC0, so it fills
both gaps this project had: live search beyond arXiv, and citation lookup
by DOI or by a title arXiv cannot find.

It is wrapped in the same rails as the arXiv client, because it sits on the
request path and is always an enhancement, never the only way to answer:

  * a hard wall-clock budget per call - a slow API costs the user nothing
    beyond that budget
  * a circuit breaker - after a failure, stop asking for a while instead of
    paying the timeout on every request
  * an LRU cache - the same paper gets looked up repeatedly
  * budget awareness - OpenAlex meters usage in dollars per day (keyless:
    $0.10/day shared by everyone behind the same IP, about 100 searches; a
    free key: $1/day). Every response reports what is left, so the client
    stops *before* the budget is gone, leaving headroom, instead of
    discovering the limit as a stream of 429s.

Set OPENALEX_API_KEY to use a free key (https://openalex.org/settings/api).
Without one everything works, within the keyless budget.
"""

from __future__ import annotations

import os
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any

import httpx

BASE_URL = "https://api.openalex.org"

# OpenAlex's own topic taxonomy: field 17 is Computer Science. Without this,
# an off-topic question ("sourdough fermentation") is answered from food
# science, and the app's honest "this covers AI/ML research only" decline
# turns into a confident answer from the wrong literature.
#
# Restricted to papers, in English: unfiltered, a PageRank query ranked a
# Spanish thesis and a university course page above the PageRank paper itself.
# OpenAlex types conference papers as "article", so venues are kept.
DEFAULT_FILTER = ("primary_topic.field.id:fields/17,has_abstract:true,"
                  "type:article|preprint|review,language:en")

SELECT = ",".join((
    "id", "doi", "title", "publication_year", "publication_date", "type",
    "abstract_inverted_index", "authorships", "primary_location", "locations",
    "cited_by_count", "open_access",
))

REQUEST_BUDGET_S = 5.0
BREAKER_COOLDOWN_S = 120.0
# Stop with this much of the daily budget left. Keyless usage is shared with
# everyone on the host's IP, so running it to zero would also starve them.
BUDGET_RESERVE_USD = 0.01
MAX_BUDGET_PAUSE_S = 3600.0

_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="openalex")
_LOCK = threading.Lock()
_STATE: dict[str, Any] = {
    "open_until": 0.0,
    "reason": "",
    "remaining_usd": None,
    "limit_usd": None,
    "calls": 0,
    "failures": 0,
}
_CACHE: OrderedDict[str, Any] = OrderedDict()
_CACHE_MAX = 512

_ARXIV_IN_URL = re.compile(r"arxiv\.org/(?:abs|pdf)/(?P<id>\d{4}\.\d{4,5})", re.IGNORECASE)
_ARXIV_DOI = re.compile(r"10\.48550/arxiv\.(?P<id>\d{4}\.\d{4,5})", re.IGNORECASE)
DOI_RE = re.compile(r"\b(?P<doi>10\.\d{4,9}/[^\s\"'<>]+)", re.IGNORECASE)


class OpenAlexUnavailable(RuntimeError):
    """OpenAlex did not answer in time, refused, or is paused."""


# ---------------------------------------------------------------- state ----
def breaker_open() -> bool:
    with _LOCK:
        return time.monotonic() < _STATE["open_until"]


def _pause(seconds: float, reason: str) -> None:
    with _LOCK:
        _STATE["open_until"] = time.monotonic() + seconds
        _STATE["reason"] = reason
    print(f"[openalex] {reason} - pausing live lookups for {seconds:.0f}s", flush=True)


def status() -> dict[str, Any]:
    """For /health: is the source usable right now, and how much budget is left."""
    with _LOCK:
        paused_for = max(0.0, _STATE["open_until"] - time.monotonic())
        return {
            "available": paused_for == 0.0,
            "paused_for_s": round(paused_for, 1),
            "reason": _STATE["reason"] if paused_for else "",
            "keyed": bool(os.environ.get("OPENALEX_API_KEY")),
            "remaining_usd": _STATE["remaining_usd"],
            "limit_usd": _STATE["limit_usd"],
            "calls": _STATE["calls"],
            "failures": _STATE["failures"],
        }


def _record_budget(headers: httpx.Headers) -> None:
    """Read the metering headers and pause ahead of exhaustion."""
    try:
        remaining = float(headers.get("X-RateLimit-Remaining-USD", "nan"))
        limit = float(headers.get("X-RateLimit-Limit-USD", "nan"))
        reset_s = float(headers.get("X-RateLimit-Reset", "3600"))
    except ValueError:
        return
    if remaining != remaining:      # NaN: header absent
        return
    with _LOCK:
        _STATE["remaining_usd"] = round(remaining, 4)
        _STATE["limit_usd"] = None if limit != limit else round(limit, 4)
    if remaining < BUDGET_RESERVE_USD:
        _pause(min(reset_s, MAX_BUDGET_PAUSE_S),
               f"daily budget nearly spent (${remaining:.3f} left)")


# -------------------------------------------------------------- parsing ----
def reconstruct_abstract(inverted: dict[str, list[int]] | None) -> str:
    """OpenAlex ships abstracts as {word: [positions]} for licensing reasons."""
    if not inverted:
        return ""
    slots: dict[int, str] = {}
    for word, positions in inverted.items():
        for pos in positions:
            slots[pos] = word
    return " ".join(slots[i] for i in sorted(slots))


def _arxiv_id(work: dict[str, Any]) -> str | None:
    """The arXiv id of a work, if any copy of it lives on arXiv."""
    doi = work.get("doi") or ""
    m = _ARXIV_DOI.search(doi)
    if m:
        return m.group("id")
    for loc in work.get("locations") or []:
        for key in ("landing_page_url", "pdf_url"):
            m = _ARXIV_IN_URL.search(loc.get(key) or "")
            if m:
                return m.group("id")
    return None


def normalise_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    doi = doi.strip().rstrip(".,;)")
    doi = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)", "", doi, flags=re.IGNORECASE)
    return doi.lower() or None


def parse_work(work: dict[str, Any]) -> dict[str, Any]:
    """One OpenAlex work -> the fields the pipeline uses."""
    openalex_id = (work.get("id") or "").rsplit("/", 1)[-1]
    arxiv_id = _arxiv_id(work)
    doi = normalise_doi(work.get("doi"))
    primary = work.get("primary_location") or {}
    # Some venues arrive as "Name/Name/Proceedings - Name"; the first is enough.
    venue = ((primary.get("source") or {}).get("display_name") or "").split("/")[0].strip()
    if arxiv_id:
        url = f"https://arxiv.org/abs/{arxiv_id}"
    elif doi:
        url = f"https://doi.org/{doi}"
    else:
        url = primary.get("landing_page_url") or work.get("id") or ""
    authors = [
        (a.get("author") or {}).get("display_name", "")
        for a in (work.get("authorships") or [])[:20]
    ]
    return {
        "openalex_id": openalex_id,
        # arXiv id when there is one, so the same paper found through arXiv,
        # OpenAlex and the corpus collapses to a single entry.
        "paper_id": arxiv_id or f"openalex:{openalex_id}",
        "arxiv_id": arxiv_id,
        "doi": doi,
        "title": re.sub(r"\s+", " ", work.get("title") or "").strip(),
        "abstract": re.sub(r"\s+", " ", reconstruct_abstract(work.get("abstract_inverted_index"))).strip(),
        "authors": [a for a in authors if a],
        "published": work.get("publication_date") or str(work.get("publication_year") or ""),
        "year": work.get("publication_year"),
        "venue": venue,
        "type": work.get("type") or "",
        "cited_by_count": int(work.get("cited_by_count") or 0),
        "url": url,
        "pdf_url": (work.get("open_access") or {}).get("oa_url") or "",
    }


# ------------------------------------------------------------------ http ----
def _get(path: str, params: dict[str, Any]) -> dict[str, Any]:
    headers = {"User-Agent": "ResearchOS (https://github.com/Kagzyyy/ResearchOS)"}
    key = os.environ.get("OPENALEX_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    with _LOCK:
        _STATE["calls"] += 1
    r = httpx.get(f"{BASE_URL}{path}", params=params, headers=headers, timeout=10.0)
    _record_budget(r.headers)
    if r.status_code == 404:
        return {}
    if r.status_code == 429:
        retry = float(r.headers.get("Retry-After", "60") or 60)
        _pause(min(max(retry, 30.0), MAX_BUDGET_PAUSE_S), "rate limited (429)")
        raise OpenAlexUnavailable("rate limited")
    r.raise_for_status()
    return r.json()


def _bounded(key: str, fn, budget_s: float):
    """Run fn() inside the budget, through the cache and the breaker."""
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    if breaker_open():
        raise OpenAlexUnavailable("paused after a recent failure")
    future = _POOL.submit(fn)
    try:
        result = future.result(timeout=max(budget_s, 0.1))
    except FuturesTimeout:
        with _LOCK:
            _STATE["failures"] += 1
        _pause(BREAKER_COOLDOWN_S, f"no answer within {budget_s:.1f}s")
        raise OpenAlexUnavailable(f"timed out after {budget_s:.1f}s") from None
    except OpenAlexUnavailable:
        with _LOCK:
            _STATE["failures"] += 1
        raise
    except Exception as exc:  # noqa: BLE001
        with _LOCK:
            _STATE["failures"] += 1
        _pause(BREAKER_COOLDOWN_S, f"{type(exc).__name__}: {str(exc)[:80]}")
        raise OpenAlexUnavailable(str(exc)) from exc
    with _LOCK:
        _CACHE[key] = result
        if len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
    return result


# ------------------------------------------------------------------- api ----
def search_works(query: str, max_results: int = 8, budget_s: float = REQUEST_BUDGET_S,
                 filter_: str = DEFAULT_FILTER) -> list[dict[str, Any]]:
    """Relevance-ranked search over titles, abstracts and full text."""
    query = (query or "").strip()
    if not query:
        return []
    params = {"search": query[:300], "per_page": max(1, min(max_results, 25)),
              "select": SELECT, "filter": filter_}

    def run() -> list[dict[str, Any]]:
        works = _get("/works", params).get("results") or []
        return [w for w in (parse_work(x) for x in works) if w["title"] and w["abstract"]]

    return _bounded(f"search|{filter_}|{max_results}|{query.lower()}", run, budget_s)


def search_title(title: str, max_results: int = 8,
                 budget_s: float = REQUEST_BUDGET_S) -> list[dict[str, Any]]:
    """Title-only search, for citation resolution. No field filter: a cited
    paper can be from any field, and the title match itself is the check."""
    title = re.sub(r"[,:;()\[\]\"]", " ", title or "").strip()
    if not title:
        return []
    params = {"filter": f"title.search:{title[:250]}",
              "per_page": max(1, min(max_results, 25)), "select": SELECT}

    def run() -> list[dict[str, Any]]:
        works = _get("/works", params).get("results") or []
        return [parse_work(x) for x in works if x.get("title")]

    return _bounded(f"title|{max_results}|{title.lower()}", run, budget_s)


def get_by_doi(doi: str, budget_s: float = REQUEST_BUDGET_S) -> dict[str, Any] | None:
    doi = normalise_doi(doi)
    if not doi:
        return None

    def run() -> dict[str, Any] | None:
        work = _get(f"/works/doi:{doi}", {"select": SELECT})
        return parse_work(work) if work.get("id") else None

    return _bounded(f"doi|{doi}", run, budget_s)


def extract_doi(text: str) -> str | None:
    m = DOI_RE.search(text or "")
    return normalise_doi(m.group("doi")) if m else None
