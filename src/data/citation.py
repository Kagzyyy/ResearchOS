"""Resolve a citation you saw in a paper to the actual paper.

The use case: you are reading a paper, you hit "...following [23] (Assran et al.,
2023)...", and you want to know what that is without leaving what you are doing.

Accepted inputs, in the order they are tried:

  arXiv id       2301.08243, arXiv:2301.08243v2, https://arxiv.org/abs/2301.08243
  DOI            10.1109/CVPR.2016.90, https://doi.org/10.1109/CVPR.2016.90
  bracketed ref  [23] M. Assran et al. "Self-Supervised Learning from Images..."
  bare title     Self-Supervised Learning from Images with a Joint-Embedding...

Resolution is local-first: the ingested corpus is checked before any network
call, because most of what you are reading about is already indexed. Anything
not found locally is fetched on demand - arXiv first, then OpenAlex, which
covers the conference and journal papers that never had an arXiv preprint.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.data import openalex_source as openalex
from src.data.arxiv_source import (
    ArxivUnavailable, Paper, download_pdf, extract_text, fetch_metadata_bounded,
)

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
    """An arXiv lookup that never takes down - or stalls - the request.

    Bounded by fetch_metadata_bounded: retries off, results cached, and the
    request stops waiting after the budget. Disabling retries alone was not
    enough - a throttled arXiv still held each call ~46s before its 429, and
    resolve() can make two calls in a row. A failed lookup returns nothing
    and the caller degrades.
    """
    try:
        return fetch_metadata_bounded(query, max_results=max_results)
    except ArxivUnavailable as exc:
        print(f"[arxiv] lookup skipped ({exc}); degrading")
        return []


def _paper_from_openalex(work: dict[str, Any]) -> Paper:
    """An OpenAlex work in the Paper shape the rest of the pipeline expects."""
    arxiv_id = work.get("arxiv_id")
    return Paper(
        paper_id=work["paper_id"],
        version=1,
        title=work["title"],
        abstract=work["abstract"],
        authors=work["authors"],
        categories=[],
        primary_category=work.get("venue", ""),
        published=work.get("published", ""),
        updated="",
        # Only arXiv PDFs are ever fetched on the request path: a known host
        # with stable, polite access. Other open-access PDFs are linked, not
        # downloaded.
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else "",
        metadata={
            "source": "openalex",
            "url": work.get("url", ""),
            "venue": work.get("venue", ""),
            "doi": work.get("doi") or "",
            "cited_by_count": str(work.get("cited_by_count", 0)),
        },
    )


def _safe_openalex(fn, *args) -> Any:
    try:
        return fn(*args)
    except openalex.OpenAlexUnavailable as exc:
        print(f"[openalex] lookup skipped ({exc}); degrading")
        return None


def paper_url(paper: Paper) -> str:
    """Where a reader should go for this paper, whichever source it came from."""
    return (paper.metadata or {}).get("url") or f"https://arxiv.org/abs/{paper.paper_id}"


@dataclass
class Resolution:
    """What a citation string resolved to, and how."""

    paper: Paper | None
    method: str  # arxiv_id | doi | local_title | arxiv_search | openalex_title | unresolved
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
    re.compile(r"\bet\s+al\b", re.IGNORECASE),          # Assran et al.
    re.compile(r"^\s*[\[\(]\d{1,3}[\]\)]"),              # [23] or (23)
    re.compile(r"^\s*\d{1,3}\.\s+[A-Z]"),                # 23. Author
    re.compile(r"\((?:19|20)\d{2}\)"),                   # (2023)
    re.compile(r",\s*(?:19|20)\d{2}\b"),                 # , 2023
    re.compile(r"\b(?:CVPR|NeurIPS|ICML|ICLR|ACL|EMNLP|AAAI|SIGGRAPH)\b"),
    re.compile(r"\barxiv\b", re.IGNORECASE),
    re.compile(r"\b10\.\d{4,9}/\S+"),                    # a DOI
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


class TitleIndex:
    """Inverted index over corpus titles, built once per corpus.

    find_in_corpus used to rebuild the IDF table and score every title on
    every lookup: fine at 8k papers, seconds per lookup at 300k. This keeps
    the scoring identical and only changes which titles get scored.

    Pruning is exact, not heuristic. A title that contains none of a set S of
    query tokens can match at most (total - w(S)) of the query's IDF weight,
    so if w(S) >= (1 - floor) * total it cannot score above `floor`. Taking
    S as the rarest query tokens until that holds, the postings of S contain
    every title that could clear the floor - the same results as a full scan.
    """

    def __init__(self, papers_by_id: dict[str, dict[str, Any]]) -> None:
        import math

        self.papers = papers_by_id
        self.tokens: dict[str, set[str]] = {}
        postings: dict[str, list[str]] = {}
        for pid, meta in papers_by_id.items():
            toks = _tokens(meta.get("title", ""))
            self.tokens[pid] = toks
            for t in toks:
                postings.setdefault(t, []).append(pid)
        self.postings = postings
        n = max(1, len(papers_by_id))
        self.idf = {t: math.log(n / (1 + len(ids))) for t, ids in postings.items()}
        self.default_idf = max(self.idf.values(), default=1.0)

    def search(self, query: str, floor: float = 0.2) -> list[tuple[float, dict[str, Any]]]:
        qt = _tokens(query)
        if not qt:
            return []
        weight = {t: self.idf.get(t, self.default_idf) for t in qt}
        total = sum(weight.values())
        needed, covered, pivots = (1.0 - floor) * total, 0.0, []
        for t in sorted(qt, key=weight.get, reverse=True):
            pivots.append(t)
            covered += weight[t]
            if covered >= needed:
                break
        candidates = {pid for t in pivots for pid in self.postings.get(t, ())}
        scored = []
        for pid in candidates:
            tt = self.tokens[pid]
            matched = sum(weight[t] for t in qt & tt)
            score = (matched / total) * min(1.0, len(qt) / max(1, len(tt))) ** 0.5
            scored.append((score, self.papers[pid]))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        return scored

    def title_coverage(self, query: str, title: str) -> float:
        """Share of the TITLE's IDF weight that the query covers - the
        direction search() ignores."""
        tt, qt = _tokens(title), _tokens(query)
        if not tt:
            return 0.0
        w = {t: self.idf.get(t, self.default_idf) for t in tt}
        return sum(w[t] for t in tt & qt) / sum(w.values())


_TITLE_INDEX: dict[int, tuple[int, TitleIndex]] = {}

AMBIGUITY_MARGIN = 0.05      # best must beat the runner-up by this much...
TITLE_COVERAGE_MIN = 0.80    # ...or the query must cover the title's own words


def _title_index(papers_by_id: dict[str, dict[str, Any]]) -> TitleIndex:
    """One index per corpus dict, rebuilt only if the corpus changes size."""
    key = id(papers_by_id)
    cached = _TITLE_INDEX.get(key)
    if cached is None or cached[0] != len(papers_by_id):
        _TITLE_INDEX.clear()          # one live corpus per process
        cached = (len(papers_by_id), TitleIndex(papers_by_id))
        _TITLE_INDEX[key] = cached
    return cached[1]


def find_in_corpus(
    query: str, papers_by_id: dict[str, dict[str, Any]], threshold: float = 0.72
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    """Best local title match, plus runners-up for disambiguation.

    A high score alone is not enough at scale. The score asks how much of
    the query a title covers, so with 123k papers a cited title that is NOT
    in the corpus ("Self-Supervised Learning from Images with a
    Joint-Embedding Predictive Architecture") matched three different JEPA
    papers at an identical 0.82. A match is accepted only when it is also
    unambiguous - clearly ahead of the runner-up - or covers the title's own
    words too; otherwise resolution falls through to arXiv/OpenAlex.
    Measured on 600 corpus titles: recall unchanged on exact, word-dropped
    and word-added citations (100 / 99.5 / 98.7%); the tie case is rejected.
    """
    index = _title_index(papers_by_id)
    scored = index.search(query, floor=0.2)
    candidates = [
        {"paper_id": m.get("paper_id", ""), "title": m.get("title", ""), "score": f"{s:.2f}"}
        for s, m in scored[:5]
        if s > 0.2
    ]
    if scored and scored[0][0] >= threshold:
        best_score, best = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        if (best_score - runner_up >= AMBIGUITY_MARGIN
                or index.title_coverage(query, best.get("title", "")) >= TITLE_COVERAGE_MIN):
            return best, candidates
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

    # A DOI's digits can look like an arXiv id ("10.1145/3292500.3330701"
    # contains "2500.33307"), so a non-arXiv DOI takes precedence over it.
    doi = openalex.extract_doi(raw)
    arxiv_doi = bool(doi and doi.startswith("10.48550/arxiv."))

    # --- 1. explicit arXiv id ---
    arxiv_id = extract_arxiv_id(raw) if (not doi or arxiv_doi) else None
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
            # arXiv down or throttling: every arXiv paper also has a DataCite
            # DOI, which OpenAlex resolves - so an arXiv id never depends on
            # arXiv's own API being up.
            work = _safe_openalex(openalex.get_by_doi, f"10.48550/arxiv.{arxiv_id}")
            if work and work.get("title"):
                return Resolution(_paper_from_openalex(work), "arxiv_id", arxiv_id, False, [])

    # --- 1b. DOI: conference and journal papers arXiv does not know ---
    if doi and not arxiv_doi and allow_remote:
        work = _safe_openalex(openalex.get_by_doi, doi)
        if work and work.get("title"):
            if work.get("arxiv_id") and work["arxiv_id"] in papers_by_id:
                return Resolution(Paper(**papers_by_id[work["arxiv_id"]]), "doi", doi, True, [])
            paper = _paper_from_openalex(work)
            if fetch_full_text and paper.pdf_url:
                _attach_text(paper, pdf_dir, text_dir, max_cache_gb)
            return Resolution(paper, "doi", doi, False, [])

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

    # --- 4. ask OpenAlex: the paper may simply never have been on arXiv ---
    if allow_remote and title_query:
        works = _safe_openalex(openalex.search_title, title_query, 8) or []
        if works:
            best = max(works, key=lambda w: _token_overlap(title_query, w["title"]))
            overlap = _token_overlap(title_query, best["title"])
            if overlap >= 0.6 and best.get("abstract"):
                paper = _paper_from_openalex(best)
                if fetch_full_text and paper.pdf_url:
                    _attach_text(paper, pdf_dir, text_dir, max_cache_gb)
                return Resolution(paper, "openalex_title", title_query, False, candidates)
            candidates = candidates + [
                {"paper_id": w["paper_id"], "title": w["title"],
                 "score": f"{_token_overlap(title_query, w['title']):.2f}"}
                for w in works[:3]
            ]

    return Resolution(None, "unresolved", title_query, False, candidates[:5])


def _attach_text(paper: Paper, pdf_dir: str, text_dir: str, max_cache_gb: float = 5.0) -> None:
    """Best-effort full text for an on-demand paper. Abstract-only is acceptable."""
    pdf = download_pdf(paper, pdf_dir, max_cache_gb=max_cache_gb)
    if pdf is not None:
        text = extract_text(pdf, text_dir)
        if text and len(text.split()) > 200:
            paper.full_text = text
