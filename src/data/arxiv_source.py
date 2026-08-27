"""arXiv ingestion: metadata, full text, and version dedup.

Design notes that follow from REQUIREMENTS-ML.md:

FR1.4 - resumable. Every PDF and every extracted text file is cached on disk by
arXiv id. Re-running ingestion skips anything already fetched, so an interrupted
run costs nothing.

FR6.2 - version dedup. arXiv ids carry a version suffix (2401.12345v3). The
index is keyed on the *versionless* id, so a later version replaces an earlier
one instead of sitting beside it as a near-duplicate. Duplicate near-identical
papers corrupt both retrieval and any evaluation run over the index.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import arxiv

# "http://arxiv.org/abs/2401.12345v3" -> ("2401.12345", 3)
_ENTRY_RE = re.compile(r"/abs/(?P<id>[^v]+(?:v\d+)?)")
_VERSION_RE = re.compile(r"^(?P<base>.+?)v(?P<version>\d+)$")


@dataclass
class Paper:
    """One arXiv paper, versionless id as the primary key."""

    paper_id: str
    version: int
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    primary_category: str
    published: str
    updated: str
    pdf_url: str
    full_text: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def versioned_id(self) -> str:
        return f"{self.paper_id}v{self.version}"


def _split_version(entry_id: str) -> tuple[str, int]:
    """Turn an arXiv entry URL into (versionless_id, version)."""
    match = _ENTRY_RE.search(entry_id)
    raw = match.group("id") if match else entry_id.rsplit("/", 1)[-1]
    versioned = _VERSION_RE.match(raw)
    if versioned:
        return versioned.group("base"), int(versioned.group("version"))
    return raw, 1


def _clean(text: str) -> str:
    """Collapse the whitespace arXiv abstracts arrive with."""
    return re.sub(r"\s+", " ", text or "").strip()


def fetch_metadata(
    query: str,
    max_results: int,
    page_size: int = 100,
    delay_seconds: float = 3.0,
    sort_by_relevance: bool = False,
) -> list[Paper]:
    """Run one arXiv API query and return papers.

    Ingestion wants newest-first (build a current corpus). Citation lookup wants
    relevance - sorting a title search by date returns the most *recent* papers
    whose titles match, so a foundational paper like "Attention Is All You Need"
    is buried under every recent paper that echoes its title.
    """
    client = arxiv.Client(page_size=page_size, delay_seconds=delay_seconds, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=(
            arxiv.SortCriterion.Relevance
            if sort_by_relevance
            else arxiv.SortCriterion.SubmittedDate
        ),
    )

    papers: list[Paper] = []
    for result in client.results(search):
        paper_id, version = _split_version(result.entry_id)
        papers.append(
            Paper(
                paper_id=paper_id,
                version=version,
                title=_clean(result.title),
                abstract=_clean(result.summary),
                authors=[a.name for a in result.authors][:20],
                categories=list(result.categories),
                primary_category=result.primary_category,
                published=result.published.date().isoformat() if result.published else "",
                updated=result.updated.date().isoformat() if result.updated else "",
                pdf_url=result.pdf_url or "",
            )
        )
    print(f"[arxiv] {len(papers):>5} results for: {query[:80]}")
    return papers


def dedup_by_version(papers: list[Paper]) -> list[Paper]:
    """FR6.2 - keep only the highest version of each paper."""
    best: dict[str, Paper] = {}
    for paper in papers:
        existing = best.get(paper.paper_id)
        if existing is None or paper.version > existing.version:
            best[paper.paper_id] = paper
    dropped = len(papers) - len(best)
    if dropped:
        print(f"[arxiv] dedup: dropped {dropped} superseded/duplicate versions")
    return list(best.values())


def download_pdf(paper: Paper, pdf_dir: str | Path, delay_seconds: float = 1.0) -> Path | None:
    """Fetch a PDF, reusing the cached copy when present. Returns None on failure."""
    target = Path(pdf_dir) / f"{paper.versioned_id}.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return target

    try:
        import requests

        response = requests.get(
            paper.pdf_url,
            timeout=60,
            headers={"User-Agent": "ResearchOS-ML/1.0 (research assistant; arXiv ingest)"},
        )
        response.raise_for_status()
        target.write_bytes(response.content)
        time.sleep(delay_seconds)  # be polite to arXiv
        return target
    except Exception as exc:  # noqa: BLE001 - one bad PDF must not stop ingestion
        print(f"[arxiv] pdf failed {paper.versioned_id}: {type(exc).__name__}")
        return None


def extract_text(pdf_path: Path, text_dir: str | Path) -> str | None:
    """Extract text from a PDF, caching the result. Returns None on failure."""
    cached = Path(text_dir) / f"{pdf_path.stem}.txt"
    cached.parent.mkdir(parents=True, exist_ok=True)
    if cached.exists():
        return cached.read_text(encoding="utf-8")

    try:
        import pymupdf

        with pymupdf.open(pdf_path) as doc:
            text = "\n".join(page.get_text() for page in doc)
        cached.write_text(text, encoding="utf-8")
        return text
    except Exception as exc:  # noqa: BLE001
        print(f"[arxiv] text extraction failed {pdf_path.name}: {type(exc).__name__}")
        return None


def write_papers(papers: list[Paper], path: str | Path) -> None:
    """Persist papers as JSONL."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        for paper in papers:
            handle.write(json.dumps(asdict(paper), ensure_ascii=False) + "\n")
    print(f"[arxiv] wrote {len(papers)} papers -> {target}")


def read_papers(path: str | Path) -> list[Paper]:
    """Load papers back from JSONL."""
    with open(path, encoding="utf-8") as handle:
        return [Paper(**json.loads(line)) for line in handle if line.strip()]
