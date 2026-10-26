"""Figures from a paper, taken from arXiv's own HTML rendering of it.

A search result that says "the architecture is shown in Figure 1" is much more
useful with Figure 1 next to it. arXiv renders papers to HTML
(arxiv.org/html/<id>), where every figure is an <img> or an SVG <object> with
its caption in a <figcaption>. This fetches that page, pulls the figures out,
and ranks them against the question. Nothing is stored: the images stay on
arXiv's servers and the browser loads them from there.

The HTML belongs to a paper's author, so it is treated as untrusted input:

  * image URLs are resolved against the page and must be https on an arXiv
    host, or they are dropped - a paper cannot make the app point a visitor's
    browser at an arbitrary server
  * captions are returned as plain text, never as markup. Equations inside them
    are converted to $...$ LaTeX from the MathML `alttext`, which the frontend
    typesets
  * the page is size-capped, so one enormous paper cannot stall a request

Like the other optional live lookups it has a time budget, a circuit breaker
(a 404 is a normal answer - many papers have no HTML - and does not trip it)
and a cache.
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

HTML_URL = "https://arxiv.org/html/{id}"
ALLOWED_HOSTS = {"arxiv.org", "www.arxiv.org", "ar5iv.labs.arxiv.org", "ar5iv.org"}
ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")

MAX_PAGE_BYTES = 6_000_000
REQUEST_BUDGET_S = 7.0
BREAKER_COOLDOWN_S = 120.0
HIT_TTL_S = 24 * 3600.0          # figures of a published paper rarely change
MISS_TTL_S = 3600.0              # no HTML today may mean "not rendered yet"
CACHE_MAX = 256
MAX_FIGURES = 12
MAX_IMAGES_PER_FIGURE = 4
MAX_CAPTION_CHARS = 600

# Two pools: figures_for() runs one get_figures() per paper, and each of those
# waits on a _fetch(). Sharing one pool would let the waiting tasks occupy every
# worker the fetches need.
_FETCH_POOL = ThreadPoolExecutor(max_workers=6, thread_name_prefix="figures-fetch")
_PAPER_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="figures-paper")
_LOCK = threading.Lock()
_BREAKER = {"open_until": 0.0}
_CACHE: OrderedDict[str, tuple[float, list[dict[str, Any]]]] = OrderedDict()

# A top-level figure that is not one of these is a table, algorithm or listing.
_NOT_A_FIGURE = ("ltx_figure_panel", "ltx_float_algorithm", "ltx_float_listing", "ltx_table")


def valid_id(paper_id: str) -> bool:
    """A modern arXiv id. YYMM.NNNN before 2015, YYMM.NNNNN from January 2015."""
    m = ID_RE.match(paper_id or "")
    if not m:
        return False
    stem = paper_id.split("v")[0]
    yymm, number = stem.split(".")
    return len(number) == (4 if int(yymm) < 1501 else 5)


def safe_image_url(src: str, base: str) -> str | None:
    """Resolve a media reference against its page; keep it only if it is arXiv's."""
    if not src or src.startswith(("data:", "javascript:")):
        return None
    url = urljoin(base, src.strip())
    parts = urlparse(url)
    if parts.scheme != "https" or (parts.hostname or "") not in ALLOWED_HOSTS:
        return None
    return url


class _FigureParser(HTMLParser):
    """Collects top-level figures: their images and their caption text."""

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base = base_url
        self.figures: list[dict[str, Any]] = []
        self._stack: list[bool] = []         # one entry per open <figure>: are we capturing it?
        self._cur: dict[str, Any] | None = None
        self._in_caption = False
        self._math_depth = 0
        self._caption: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: (v or "") for k, v in attrs}
        classes = a.get("class", "").split()
        if tag == "figure":
            if not self._stack:
                capture = "ltx_figure" in classes and not any(c in classes for c in _NOT_A_FIGURE)
                if capture:
                    self._cur = {"id": a.get("id", ""), "images": [], "caption": ""}
                    self._caption = []
            else:
                capture = self._stack[-1]    # a sub-figure belongs to its parent
            self._stack.append(capture)
            return
        if not (self._stack and self._stack[-1] and self._cur is not None):
            return
        if tag == "img" and "ltx_graphics" in classes:
            width = a.get("width", "")
            if width.isdigit() and int(width) < 60:
                return                       # an icon, not a figure
            self._add_image(a.get("src", ""))
        elif tag == "object" and a.get("type", "") == "image/svg+xml":
            self._add_image(a.get("data", ""))
        elif tag == "figcaption" and len(self._stack) == 1:
            self._in_caption = True
        elif tag == "math" and self._in_caption:
            alt = a.get("alttext", "").strip()
            if alt and self._math_depth == 0:
                self._caption.append(f" ${alt}$ ")
            self._math_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "math" and self._math_depth:
            self._math_depth -= 1
        elif tag == "figcaption":
            self._in_caption = False
        elif tag == "figure" and self._stack:
            captured = self._stack.pop()
            if not self._stack and captured and self._cur is not None:
                self._finish()

    def handle_data(self, data: str) -> None:
        if self._in_caption and self._math_depth == 0:
            self._caption.append(data)

    def _add_image(self, src: str) -> None:
        url = safe_image_url(src, self.base)
        if url and url not in self._cur["images"] and len(self._cur["images"]) < MAX_IMAGES_PER_FIGURE:
            self._cur["images"].append(url)

    def _finish(self) -> None:
        cur, self._cur = self._cur, None
        text = re.sub(r"\s+", " ", "".join(self._caption)).strip()
        text = re.sub(r"\s+([,.;:!?)])", r"\1", text)         # "Figure 1 :" -> "Figure 1:"
        text = re.sub(r"\(\s+", "(", text)
        label = ""
        m = re.match(r"^(Figure|Fig\.)\s*(\d+[a-zA-Z]?)\s*[:.]\s*", text)
        if m:
            label = f"Figure {m.group(2)}"
            text = text[m.end():]
        if cur and cur["images"]:
            number = int(re.sub(r"\D", "", label) or 0) or len(self.figures) + 1
            self.figures.append({
                "label": label or f"Figure {number}",
                "number": number,
                "caption": text[:MAX_CAPTION_CHARS],
                "images": cur["images"],
            })


def extract_figures(html: str, base_url: str) -> list[dict[str, Any]]:
    """Every top-level figure with at least one image, in paper order."""
    parser = _FigureParser(base_url)
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001 - malformed markup: keep what was parsed
        pass
    return parser.figures[:MAX_FIGURES]


_STOP = set("the a an of for to in on and or is are with by from as at be this that how what why which "
            "does do can it its using use used into between versus vs".split())


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z0-9-]{2,}", (text or "").lower()) if w not in _STOP}


def rank_figures(figures: list[dict[str, Any]], query: str, k: int = 3) -> list[dict[str, Any]]:
    """The k figures most worth showing for this question.

    Word overlap between the question and the caption decides, with a small
    bonus for Figure 1 (usually the overview or architecture diagram) so that
    a question with no overlap still gets the figure most papers lead with.
    Ties keep paper order.
    """
    q = _tokens(query)
    scored = []
    for fig in figures:
        overlap = len(q & _tokens(fig["caption"])) / max(len(q), 1) if q else 0.0
        bonus = 0.15 if fig["number"] == 1 else (0.05 if fig["number"] == 2 else 0.0)
        scored.append((overlap + bonus, -fig["number"], fig))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [dict(f, score=round(s, 3)) for s, _n, f in scored[:k]]


# ----------------------------------------------------------------- fetching ----
def breaker_open() -> bool:
    with _LOCK:
        return time.monotonic() < _BREAKER["open_until"]


def _trip(reason: str) -> None:
    with _LOCK:
        _BREAKER["open_until"] = time.monotonic() + BREAKER_COOLDOWN_S
    print(f"[figures] {reason} - pausing figure lookups for {BREAKER_COOLDOWN_S:.0f}s", flush=True)


def _fetch(paper_id: str) -> list[dict[str, Any]]:
    headers = {"User-Agent": "ResearchOS (https://github.com/Kagzyyy/ResearchOS)"}
    with httpx.stream("GET", HTML_URL.format(id=paper_id), headers=headers,
                      follow_redirects=True, timeout=REQUEST_BUDGET_S) as r:
        if r.status_code == 404:
            return []                         # no HTML rendering: a normal answer
        r.raise_for_status()
        chunks, size = [], 0
        for chunk in r.iter_bytes():
            size += len(chunk)
            if size > MAX_PAGE_BYTES:
                return []                     # too large to be worth parsing
            chunks.append(chunk)
        html = b"".join(chunks).decode(r.encoding or "utf-8", errors="replace")
        return extract_figures(html, str(r.url))


def get_figures(paper_id: str, budget_s: float = REQUEST_BUDGET_S) -> list[dict[str, Any]]:
    """All figures of one paper, cached. Returns [] whenever it cannot help."""
    if not valid_id(paper_id):
        return []
    now = time.monotonic()
    with _LOCK:
        hit = _CACHE.get(paper_id)
        if hit and hit[0] > now:
            _CACHE.move_to_end(paper_id)
            return hit[1]
    if breaker_open():
        return []
    future = _FETCH_POOL.submit(_fetch, paper_id)
    try:
        figures = future.result(timeout=max(budget_s, 0.1))
    except FuturesTimeout:
        _trip(f"no answer within {budget_s:.0f}s")
        return []
    except Exception as exc:  # noqa: BLE001
        _trip(f"{type(exc).__name__}: {str(exc)[:80]}")
        return []
    ttl = HIT_TTL_S if figures else MISS_TTL_S
    with _LOCK:
        _CACHE[paper_id] = (now + ttl, figures)
        while len(_CACHE) > CACHE_MAX:
            _CACHE.popitem(last=False)
    return figures


def figures_for(paper_ids: list[str], query: str, per_paper: int = 2,
                budget_s: float = REQUEST_BUDGET_S) -> dict[str, list[dict[str, Any]]]:
    """Ranked figures for several papers, fetched in parallel under one budget."""
    ids = list(dict.fromkeys(i for i in paper_ids if valid_id(i)))[:4]
    futures = {i: _PAPER_POOL.submit(get_figures, i, budget_s) for i in ids}
    deadline = time.monotonic() + budget_s + 1.0
    out: dict[str, list[dict[str, Any]]] = {}
    for pid, fut in futures.items():
        try:
            figs = fut.result(timeout=max(deadline - time.monotonic(), 0.05))
        except Exception:  # noqa: BLE001 - one slow paper must not sink the rest
            figs = []
        out[pid] = rank_figures(figs, query, per_paper)
    return out
