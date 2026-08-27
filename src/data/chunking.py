"""Section-aware chunking for full-text papers.

FR2 in REQUIREMENTS-ML.md. Three things this has to get right:

1. Chunks must fit the reranker window (384 tokens ~= 280 words), or the tail of
   every chunk is silently truncated at scoring time and never contributes.
2. Every chunk carries paper id, section name and character offsets, so a
   citation can point at a span rather than waving at a whole paper.
3. Chunking is deterministic - re-ingesting a paper yields identical chunks, so
   the index and any evaluation over it stay reproducible.

References sections are dropped. They are pure citation strings: high lexical
overlap with every query, no answer content, and a reliable source of false
positives for BM25.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from src.data.arxiv_source import Paper

# Headings we expect in an ML paper. Matched case-insensitively at line start,
# optionally numbered ("3.1 Method").
SECTION_PATTERN = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(abstract|introduction|background|related\s+work|preliminaries|method(?:s|ology)?|"
    r"approach|model|architecture|experiment(?:s|al\s+setup)?|results|evaluation|"
    r"analysis|ablation(?:\s+stud(?:y|ies))?|discussion|limitations|conclusion(?:s)?|"
    r"future\s+work|acknowledg(?:e)?ments|references|bibliography|appendix)"
    r"\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Everything from here on is not answer content.
STOP_SECTIONS = {"references", "bibliography", "acknowledgments", "acknowledgements"}

# Skipped when chunking full text. The abstract is already indexed verbatim from
# the arXiv metadata, so taking it again from the PDF produces a near-duplicate
# chunk that competes with itself in the ranking. "frontmatter" is the title
# block: author names and affiliations, no answer content.
SKIP_IN_FULL_TEXT = {"abstract", "frontmatter"}


@dataclass
class Chunk:
    """One retrievable unit."""

    chunk_id: str
    paper_id: str
    version: int
    title: str
    section: str
    text: str
    start: int
    end: int
    published: str
    primary_category: str

    def to_corpus_entry(self) -> dict[str, str]:
        """The shape the retrieval stack expects: title + text, plus metadata."""
        return {
            "title": self.title,
            "text": self.text,
            "paper_id": self.paper_id,
            "section": self.section,
            "published": self.published,
            "primary_category": self.primary_category,
        }


def split_sections(text: str) -> list[tuple[str, str, int]]:
    """Split full text into (section_name, section_text, start_offset).

    Falls back to one unnamed section when no headings are found, which happens
    with two-column PDFs whose text extraction interleaves badly.
    """
    matches = list(SECTION_PATTERN.finditer(text))
    if not matches:
        return [("body", text, 0)]

    sections: list[tuple[str, str, int]] = []
    if matches[0].start() > 0:
        sections.append(("frontmatter", text[: matches[0].start()], 0))

    for i, match in enumerate(matches):
        name = re.sub(r"\s+", " ", match.group(1)).strip().lower()
        if name in STOP_SECTIONS:
            break  # everything after references is discarded
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((name, body, start))
    return sections


def _split_words(text: str, max_words: int, overlap_words: int) -> list[tuple[str, int, int]]:
    """Sliding window over words. Returns (text, word_start, word_end) windows."""
    words = text.split()
    if not words:
        return []
    if len(words) <= max_words:
        return [(" ".join(words), 0, len(words))]

    stride = max(1, max_words - overlap_words)
    windows: list[tuple[str, int, int]] = []
    for start in range(0, len(words), stride):
        window = words[start : start + max_words]
        if len(window) < overlap_words and windows:
            break  # trailing scrap already covered by the previous window's overlap
        windows.append((" ".join(window), start, start + len(window)))
        if start + max_words >= len(words):
            break
    return windows


def chunk_paper(
    paper: Paper,
    max_words: int = 220,
    overlap_words: int = 40,
    min_words: int = 25,
) -> list[Chunk]:
    """Turn one paper into chunks. The abstract is always chunk 0.

    max_words defaults to 220 because the reranker window is 384 *tokens*, and
    English averages roughly 1.3 tokens per word - 220 words lands near 290
    tokens, leaving room for the query and special tokens on the same input.
    """
    chunks: list[Chunk] = []

    def add(section: str, text: str, start: int, end: int) -> None:
        text = re.sub(r"\s+", " ", text).strip()
        if len(text.split()) < min_words:
            return
        chunks.append(
            Chunk(
                chunk_id=f"{paper.paper_id}#{len(chunks)}",
                paper_id=paper.paper_id,
                version=paper.version,
                title=paper.title,
                section=section,
                text=text,
                start=start,
                end=end,
                published=paper.published,
                primary_category=paper.primary_category,
            )
        )

    # The abstract is the highest-signal passage in any paper - always index it.
    add("abstract", paper.abstract, 0, len(paper.abstract))

    if paper.full_text:
        for name, body, offset in split_sections(paper.full_text):
            if name in SKIP_IN_FULL_TEXT:
                continue
            for window_text, w_start, w_end in _split_words(body, max_words, overlap_words):
                add(name, window_text, offset + w_start, offset + w_end)

    return chunks


def chunk_papers(papers: list[Paper], **kwargs: int) -> list[Chunk]:
    """Chunk a whole corpus, reporting the shape of the result."""
    chunks: list[Chunk] = []
    for paper in papers:
        chunks.extend(chunk_paper(paper, **kwargs))
    with_text = sum(1 for p in papers if p.full_text)
    per_paper = len(chunks) / max(1, len(papers))
    print(
        f"[chunk] {len(chunks)} chunks from {len(papers)} papers "
        f"({with_text} with full text, {per_paper:.1f} chunks/paper)"
    )
    return chunks


def write_chunks(chunks: list[Chunk], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    print(f"[chunk] wrote {len(chunks)} chunks -> {target}")


def read_chunks(path: str | Path) -> list[Chunk]:
    with open(path, encoding="utf-8") as handle:
        return [Chunk(**json.loads(line)) for line in handle if line.strip()]


def chunks_to_corpus(chunks: list[Chunk]) -> dict[str, dict[str, str]]:
    """Adapt chunks to the Corpus shape the existing retrieval stack consumes."""
    return {chunk.chunk_id: chunk.to_corpus_entry() for chunk in chunks}
