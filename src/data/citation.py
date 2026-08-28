"""Resolve a citation you saw in a paper to the actual paper.

The use case: you are reading a paper, you hit "...following [23] (Assran et al.,
2023)...", and you want to know what that is without leaving what you are doing.

Accepted inputs, in the order they are tried:

  arXiv id       2301.08243, arXiv:2301.08243v2, https://arxiv.org/abs/2301.08243
  bracketed ref  [23] M. Assran et al. "Self-Supervised Learning from Images..."
  bare title     Self-Supervised Learning from Images with a Joint-Embedding...

Resolution is local-first: the ingested corpus is checked before arXiv is
called, because most of what you are reading about is already indexed. Anything
not found locally is fetched on demand, so the tool is not limited to the
corpus snapshot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.data.arxiv_source import Paper, download_pdf, extract_text, fetch_metadata

# 2301.08243 / 2301.08243v2, with optional arXiv: prefix or abs/pdf URL
ARXIV_ID_RE = re.compile(
    r"(?:arxiv[:\s/]*|/abs/|/pdf/)?(?P<id>\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE
)
# Leading "[23]" or "23." reference markers
REF_MARKER_RE = re.compile(r"^\s*[\[\(]?\d{1,3}[\]\).]?\s*")
# "Assran et al., 2023" style author-year tails
AUTHOR_YEAR_RE = re.compile(r"\b[A-Z][a-z]+\s+et\s+al\.?,?\s*\(?\d{4}\)?[.,]?", re.IGNORECASE)
# A leading author block terminated by "et al.", with or without initials:
#   "M. Assran et al." / "Kaiming He et al." / "Vaswani et al.,"
AUTHOR_ETAL_RE = re.compile(
    r"^(?:[A-Z][a-zA-Z.'-]*\s+)*?[A-Z][a-zA-Z'-]+,?\s+et\s+al\.?,?\s*", re.IGNORECASE
)
QUOTED_RE = re.compile(r"[\"“‘']([^\"”’']{12,})[\"”’']")


def _safe_fetch(query: str, max_results: int) -> list[Paper]:
    """An arXiv lookup that never takes down the request.

    Interactive lookups use a 1s delay and a single retry - the 3s/3-retry
    defaults are for bulk ingestion and turn one 429 into a minute of blocking.
    A rate-limited search returns nothing and the caller degrades.
    """
    try:
        return fetch_metadata(
            query, max_results=max_results, delay_seconds=0.0,
            sort_by_relevance=True, num_retries=0, use_cache=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[arxiv] lookup failed ({type(exc).__name__}); degrading")
        return []


@dataclass
class Resolution:
    """What a citation string resolved to, and how."""

    paper: Paper | None
    method: str  # arxiv_id | local_title | arxiv_search | unresolved
    query_used: str
    from_corpus: bool
    candidates: list[dict[str, str]]


QUESTION_STARTS = (
    "what", "how", "why", "when", "where", "which", "who", "explain", "describe",
    "compare", "does", "do ", "is ", "are ", "can ", "should", "tell me", "give me",
    "summarise", "summarize", "define",
)

# What a citation actually looks like. Anything matching one of these is worth
# the cost of resolution; nothing else is.
_CITATION_SHAPES = (
    re.compile(r"et\s+al", re.IGNORECASE),          # Assran et al.
    re.compile(r"^\s*[\[\(]\d{1,3}[\]\)]"),              # [23] or (23)
    re.compile(r"^\s*\d{1,3}\.\s+[A-Z]"),                # 23. Author
    re.compile(r"\((?:19|20)\d{2}\)"),                   # (2023)
    re.compile(r",\s*(?:19|20)\d{2}"),                 # , 2023
    re.compile(r"(?:CVPR|NeurIPS|ICML|ICLR|ACL|EMNLP|AAAI|SIGGRAPH)"),
    re.compile(r"arxiv", re.IGNORECASE),
    re.compile(r"[\"“‘']([^\"”’']{15,})[\"”’']"),  # a quoted title
)


def looks_like_a_citation(text: str) -> bool:
    """True only when the text has the shape of a reference.

    The default matters. Selecting a phrase in a paper - "MLP block of
    Transformers" - is asking about a concept, not naming a paper. Treating it
    as a title costs two arXiv title searches that cannot succeed, and arXiv
    rate-limits: one such selection took 34 seconds to fail. Citations announce
    themselves (an id, "et al.", a bracketed number, a year, a venue); concepts
    do not, so concepts are the default and resolution is opt-in.
    """
    raw = (text or "").strip()
    if not raw:
        return False
    if extract_arxiv_id(raw):
        return True
    if any(shape.search(raw) for shape in _CITATION_SHAPES):
        return True
    # A bare title: long, no trailing question mark, and mostly capitalised
    # words - "Masked Autoencoders Are Scalable Vision Learners".
    words = raw.split()
    if 4 <= len(words) <= 25 and not raw.endswith("?"):
        capitalised = sum(1 for w in words if w[:1].isupper())
        if capitalised >= max(3, len(words) * 0.5):
            return True
    return False


def looks_like_a_question(text: str) -> bool:
    """Anything that is not shaped like a citation is treated as a question."""
    raw = (text or "").strip()
    if not raw:
        return False
    if raw.endswith("?") or raw.lower().startswith(QUESTION_STARTS):
        return True
    return not looks_like_a_citation(raw)


def extract_arxiv_id(text: str) -> str | None:
    """Pull an arXiv id out of free text, if there is one."""
    match = ARXIV_ID_RE.search(text or "")
    return match.group("id") if match else None


def clean_citation(text: str) -> str:
    """Strip reference markers, author-year tails and quotes down to a title-ish string."""
    cleaned = REF_MARKER_RE.sub("", text or "").strip()

    # A quoted span in a reference is almost always the title.
    quoted = QUOTED_RE.search(cleaned)
    if quoted:
        return quoted.group(1).strip()

    cleaned = AUTHOR_ETAL_RE.sub("", cleaned)
    cleaned = AUTHOR_YEAR_RE.sub(" ", cleaned)
    # Drop a leading author list: "M. Assran, Q. Duval, I. Misra. Title here"
    cleaned = re.sub(r"^(?:[A-Z]\.\s*){1,4}[A-Z][a-z]+(?:,\s*(?:[A-Z]\.\s*)*[A-Z][a-z]+)*[.,]\s*", "", cleaned)
    # A bare year left over once the author block is gone: "2017. Attention Is..."
    cleaned = re.sub(r"^\(?(?:19|20)\d{2}\)?[.,]?\s*", "", cleaned)
    # Drop trailing venue noise
    cleaned = re.sub(r"\b(?:In\s+)?(?:Proceedings|Proc\.|CVPR|NeurIPS|ICML|ICLR|arXiv preprint)\b.*$", "", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip(" .,;:")


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower())


STOPWORDS = {
    "the", "a", "an", "of", "for", "with", "and", "in", "on", "to", "via", "using",
    "all", "you", "need", "is", "are", "we", "our", "its", "that", "this", "from",
}


def _tokens(text: str) -> set[str]:
    return {w for w in _normalise(text).split() if w not in STOPWORDS and len(w) > 2}


def _token_overlap(a: str, b: str) -> float:
    """Jaccard overlap on content words. Used for ranking arXiv search results."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _build_idf(papers_by_id: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Inverse document frequency over corpus titles."""
    import math

    df: dict[str, int] = {}
    for meta in papers_by_id.values():
        for token in _tokens(meta.get("title", "")):
            df[token] = df.get(token, 0) + 1
    n = max(1, len(papers_by_id))
    return {t: math.log(n / (1 + c)) for t, c in df.items()}


def _weighted_coverage(query: str, title: str, idf: dict[str, float], default_idf: float) -> float:
    """How much of the query's *information* the title covers.

    Plain Jaccard fails badly on titles built from common words. "Attention Is
    All You Need" against "All you need is log" scores 0.60 on Jaccard - three
    throwaway tokens match and the one word that carries the meaning,
    "attention", does not. Weighting each token by IDF makes the miss decisive:
    a rare token that fails to match costs far more than three common ones that
    do.
    """
    qt = _tokens(query)
    if not qt:
        return 0.0
    tt = _tokens(title)
    total = sum(idf.get(t, default_idf) for t in qt)
    if total <= 0:
        return 0.0
    matched = sum(idf.get(t, default_idf) for t in qt & tt)
    # Penalise a title that is much longer than the query - it is a different
    # paper that happens to contain these words.
    length_penalty = min(1.0, len(qt) / max(1, len(tt))) ** 0.5
    return (matched / total) * length_penalty


def find_in_corpus(
    query: str, papers_by_id: dict[str, dict[str, Any]], threshold: float = 0.72
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    """Best local title match, plus runners-up for disambiguation."""
    idf = _build_idf(papers_by_id)
    default_idf = max(idf.values(), default=1.0)
    scored = [
        (_weighted_coverage(query, meta.get("title", ""), idf, default_idf), meta)
        for meta in papers_by_id.values()
    ]
    scored.sort(key=lambda kv: kv[0], reverse=True)
    candidates = [
        {"paper_id": m.get("paper_id", ""), "title": m.get("title", ""), "score": f"{s:.2f}"}
        for s, m in scored[:5]
        if s > 0.2
    ]
    if scored and scored[0][0] >= threshold:
        return scored[0][1], candidates
    return None, candidates


def resolve(
    citation: str,
    papers_by_id: dict[str, dict[str, Any]],
    allow_remote: bool = True,
    fetch_full_text: bool = True,
    pdf_dir: str = "cache/arxiv/pdf",
    text_dir: str = "cache/arxiv/text",
    max_cache_gb: float = 5.0,
) -> Resolution:
    """Turn a citation string into a Paper. Local corpus first, then arXiv."""
    raw = (citation or "").strip()
    if not raw:
        return Resolution(None, "unresolved", "", False, [])

    # --- 1. explicit arXiv id ---
    arxiv_id = extract_arxiv_id(raw)
    if arxiv_id:
        local = papers_by_id.get(arxiv_id)
        if local:
            return Resolution(Paper(**local), "arxiv_id", arxiv_id, True, [])
        if allow_remote:
            found = _safe_fetch(f"id:{arxiv_id}", 1)
            if found:
                paper = found[0]
                if fetch_full_text:
                    _attach_text(paper, pdf_dir, text_dir, max_cache_gb)
                return Resolution(paper, "arxiv_id", arxiv_id, False, [])

    # --- 2. title match against the local corpus ---
    title_query = clean_citation(raw)
    local_match, candidates = find_in_corpus(title_query, papers_by_id)
    if local_match:
        return Resolution(Paper(**local_match), "local_title", title_query, True, candidates)

    # --- 3. ask arXiv ---
    if allow_remote and title_query:
        escaped = title_query.replace('"', "").strip()
        found = _safe_fetch(f'ti:"{escaped}"', 8)
        if not found:
            found = _safe_fetch(f"all:{escaped}", 8)
        if found:
            best = max(found, key=lambda p: _token_overlap(title_query, p.title))
            if _token_overlap(title_query, best.title) >= 0.4:
                if fetch_full_text:
                    _attach_text(best, pdf_dir, text_dir, max_cache_gb)
                return Resolution(best, "arxiv_search", title_query, False, candidates)
            candidates = [
                {"paper_id": p.paper_id, "title": p.title, "score": f"{_token_overlap(title_query, p.title):.2f}"}
                for p in found
            ] + candidates

    return Resolution(None, "unresolved", title_query, False, candidates[:5])


def _attach_text(paper: Paper, pdf_dir: str, text_dir: str, max_cache_gb: float = 5.0) -> None:
    """Best-effort full text for an on-demand paper. Abstract-only is acceptable."""
    pdf = download_pdf(paper, pdf_dir, max_cache_gb=max_cache_gb)
    if pdf is not None:
        text = extract_text(pdf, text_dir)
        if text and len(text.split()) > 200:
            paper.full_text = text
