"""Build a retrieval evaluation set from citation contexts.

arXiv ships no relevance judgements, so there is nothing to compute nDCG
against. This manufactures a test set from signal that is already in the
corpus and is genuinely human-authored:

    A paper writes  "...following the approach of [23]..."
    Reference [23]  resolves to arXiv paper 2301.08243
    => query  = that sentence
       gold   = 2301.08243

A researcher decided that citation belonged in that sentence. That is a real
relevance judgement, just not one collected for this purpose.

What this measures: "given a sentence that cites something, can the system
find the thing being cited?" That is exactly the Find-papers use case, so it
is a fair proxy for it. It is NOT a proxy for the Explain use case, and it is
weak supervision rather than a curated gold set - see the honesty notes in
README-ML.md.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.data.arxiv_source import Paper
from src.data.citation import _build_idf, _tokens, _weighted_coverage, clean_citation

# Where the bibliography starts.
REFS_HEADER_RE = re.compile(r"^\s*(references|bibliography)\s*$", re.IGNORECASE | re.MULTILINE)
# "[23] Author, Title..." at the start of a reference entry.
NUMBERED_REF_RE = re.compile(r"\[(\d{1,3})\]\s*(.+?)(?=\n\s*\[\d{1,3}\]|\Z)", re.DOTALL)
# Inline "[23]" or "[12, 23]" or "[8-23]" markers in the body.
INLINE_MARKER_RE = re.compile(r"\[(\d{1,3}(?:\s*[,\-–]\s*\d{1,3})*)\]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

MIN_QUERY_WORDS = 8
MAX_QUERY_WORDS = 60


@dataclass
class CitationPair:
    """One (citing sentence -> cited paper) judgement."""

    query_id: str
    query: str
    gold_paper_id: str
    citing_paper_id: str
    marker: str
    gold_title: str


def split_body_and_refs(full_text: str) -> tuple[str, str]:
    """Return (body, references_block). Empty refs when no header is found."""
    matches = list(REFS_HEADER_RE.finditer(full_text))
    if not matches:
        return full_text, ""
    last = matches[-1]
    return full_text[: last.start()], full_text[last.end() :]


def parse_numbered_references(refs_block: str) -> dict[str, str]:
    """Map "23" -> the raw reference string."""
    out: dict[str, str] = {}
    for match in NUMBERED_REF_RE.finditer(refs_block):
        number, body = match.group(1), re.sub(r"\s+", " ", match.group(2)).strip()
        if 20 < len(body) < 600:
            out[number] = body
    return out


def _expand_markers(raw: str) -> list[str]:
    """"[12, 23]" -> ["12", "23"]. Ranges are skipped as too ambiguous."""
    if re.search(r"[\-–]", raw):
        return []
    return [p.strip() for p in raw.split(",") if p.strip().isdigit()]


def find_citing_sentences(body: str) -> list[tuple[str, str]]:
    """Return (marker_number, sentence) for every unambiguous inline citation."""
    found: list[tuple[str, str]] = []
    for sentence in SENTENCE_SPLIT_RE.split(body):
        clean = re.sub(r"\s+", " ", sentence).strip()
        words = clean.split()
        if not (MIN_QUERY_WORDS <= len(words) <= MAX_QUERY_WORDS):
            continue
        markers = INLINE_MARKER_RE.findall(clean)
        # A sentence citing many works at once is about all of them; a sentence
        # citing exactly one is about that one. Only the latter is a clean label.
        expanded = [n for m in markers for n in _expand_markers(m)]
        if len(expanded) != 1:
            continue
        found.append((expanded[0], clean))
    return found


def build_pairs(
    papers: list[Paper],
    corpus_papers: dict[str, dict[str, Any]],
    match_threshold: float = 0.78,
    max_per_paper: int = 6,
) -> list[CitationPair]:
    """Mine (citing sentence -> cited corpus paper) pairs from full-text papers.

    A pair is kept only when the cited work resolves to a paper that is IN the
    corpus - otherwise the retriever could not possibly find it and the query
    would score zero for reasons that say nothing about retrieval quality.

    `match_threshold` is deliberately stricter than the interactive lookup: a
    wrong label is worse than a missing one when building an evaluation set.
    """
    idf = _build_idf(corpus_papers)
    default_idf = max(idf.values(), default=1.0)
    titles = {meta["paper_id"]: meta.get("title", "") for meta in corpus_papers.values()}

    # Inverted index over title tokens. Scoring a reference against all 8,330
    # titles is 45M comparisons across the corpus and takes over ten minutes;
    # candidates that share no rare token cannot clear the threshold anyway, so
    # only papers sharing the query's rarest tokens are ever scored.
    postings: dict[str, list[str]] = {}
    for pid, title in titles.items():
        for token in _tokens(title):
            postings.setdefault(token, []).append(pid)

    def candidates_for(query: str, max_tokens: int = 6) -> set[str]:
        query_tokens = sorted(
            _tokens(query), key=lambda t: idf.get(t, default_idf), reverse=True
        )[:max_tokens]
        found: set[str] = set()
        for token in query_tokens:
            found.update(postings.get(token, ()))
        return found

    pairs: list[CitationPair] = []
    resolved_cache: dict[str, str | None] = {}

    for paper in papers:
        if not paper.full_text:
            continue
        body, refs_block = split_body_and_refs(paper.full_text)
        if not refs_block:
            continue
        references = parse_numbered_references(refs_block)
        if not references:
            continue

        kept = 0
        for marker, sentence in find_citing_sentences(body):
            if kept >= max_per_paper:
                break
            raw_ref = references.get(marker)
            if not raw_ref:
                continue

            cache_key = f"{paper.paper_id}:{marker}"
            if cache_key not in resolved_cache:
                title_guess = clean_citation(raw_ref)
                gold_id = None
                if len(_tokens(title_guess)) >= 3:
                    best_id, best_score = None, 0.0
                    for pid in candidates_for(title_guess):
                        score = _weighted_coverage(
                            title_guess, titles[pid], idf, default_idf
                        )
                        if score > best_score:
                            best_id, best_score = pid, score
                    if best_score >= match_threshold and best_id != paper.paper_id:
                        gold_id = best_id
                resolved_cache[cache_key] = gold_id

            gold_id = resolved_cache[cache_key]
            if not gold_id:
                continue

            # Strip the marker so the query cannot leak its own answer.
            query = INLINE_MARKER_RE.sub("", sentence).strip()
            query = re.sub(r"\s+", " ", query)
            if len(query.split()) < MIN_QUERY_WORDS:
                continue

            pairs.append(
                CitationPair(
                    query_id=f"{paper.paper_id}#{marker}#{kept}",
                    query=query,
                    gold_paper_id=gold_id,
                    citing_paper_id=paper.paper_id,
                    marker=marker,
                    gold_title=corpus_papers[gold_id].get("title", ""),
                )
            )
            kept += 1

    return pairs


def write_pairs(pairs: list[CitationPair], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(asdict(pair), ensure_ascii=False) + "\n")
    print(f"[eval] wrote {len(pairs)} citation pairs -> {target}")


def read_pairs(path: str | Path) -> list[CitationPair]:
    with open(path, encoding="utf-8") as handle:
        return [CitationPair(**json.loads(line)) for line in handle if line.strip()]
