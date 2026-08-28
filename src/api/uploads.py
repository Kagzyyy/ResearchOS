"""Uploaded-PDF store: extract, chunk, and answer questions against one paper.

Why this is not just "add it to the corpus": rebuilding the BM25 and FAISS
indices over 40,000 chunks to accommodate one 30-chunk document would take
minutes and be thrown away. A single paper is small enough that the
cross-encoder can score every one of its chunks directly - no index, no
approximation, and better ranking than retrieval would give.

Documents are persisted as JSON so they survive a restart, and the store is
capped so an unbounded number of uploads cannot fill the disk.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.data.arxiv_source import Paper
from src.data.chunking import chunk_paper

import re

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB - larger than any sane paper
_ARXIV_STAMP_RE = re.compile(r"^\s*arxiv:\s*\d{4}\.\d{4,5}", re.IGNORECASE)
PDF_MAGIC = b"%PDF-"


class UploadError(ValueError):
    """Anything wrong with the uploaded file, phrased for the user."""


@dataclass
class UploadedDoc:
    document_id: str
    title: str
    n_chunks: int
    n_words: int
    uploaded_at: float
    filename: str

    def to_public(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "n_chunks": self.n_chunks,
            "n_words": self.n_words,
            "filename": self.filename,
        }


# Lines that appear before the title on real papers: licence grants, arXiv
# stamps, conference banners. "Attention Is All You Need" opens with Google's
# permission notice, so taking the first substantial line names the paper
# "Provided proper attribution is provided, Google hereby grants...".
_BOILERPLATE = (
    "provided proper attribution", "hereby grants", "permission to", "licence",
    "license", "copyright", "creative commons", "all rights reserved",
    "preprint", "under review", "accepted at", "to appear in", "proceedings of",
    "conference on", "workshop on", "arxiv", "doi:", "http", "www.",
    "equal contribution", "corresponding author",
)


def _title_from_layout(doc: Any) -> str | None:
    """Title = the largest text on page 1.

    Content heuristics lose to licence blocks: "Attention Is All You Need"
    opens with a multi-line Google permission notice, so both "first
    substantial line" and keyword filtering pick a fragment of the legal text.
    Font size does not have that problem - the title is set larger than
    everything around it, by typographic convention, in every paper format.
    """
    try:
        page = doc[0].get_text("dict")
    except Exception:  # noqa: BLE001
        return None

    spans: list[tuple[float, float, str]] = []
    for block in page.get("blocks", []):
        for line in block.get("lines", []):
            # arXiv stamps the id vertically down the left margin, in a large
            # font - it wins on size alone. Horizontal text only.
            direction = line.get("dir", (1.0, 0.0))
            if abs(direction[0]) < 0.9:
                continue
            for span in line.get("spans", []):
                text = " ".join(span.get("text", "").split())
                if not text or _ARXIV_STAMP_RE.match(text):
                    continue
                spans.append((round(span.get("size", 0), 1), span.get("bbox", [0, 0])[1], text))
    if not spans:
        return None

    largest = max(size for size, _, _ in spans)
    # Join every span at (near) the largest size, in reading order - titles
    # routinely wrap onto two or three lines.
    title_spans = sorted(
        [(y, t) for size, y, t in spans if size >= largest - 0.6], key=lambda st: st[0]
    )
    title = " ".join(t for _, t in title_spans).strip()
    return title if 10 <= len(title) <= 300 else None


def _guess_title(text: str, fallback: str) -> str:
    """Fallback when layout gives nothing: first plausible non-boilerplate line."""
    candidates: list[str] = []
    for raw in text.splitlines()[:60]:
        line = " ".join(raw.split())
        lowered = line.lower()
        if not (15 <= len(line) <= 250):
            continue
        if any(marker in lowered for marker in _BOILERPLATE):
            continue
        words = line.split()
        if len(words) < 3:
            continue
        if sum(c.isdigit() for c in line) > len(line) * 0.3:
            continue
        # Author lines are mostly capitalised short tokens separated by commas.
        if line.count(",") >= 3 and sum(w[:1].isupper() for w in words) > len(words) * 0.7:
            continue
        candidates.append(line)
        if len(candidates) >= 5:
            break
    return candidates[0] if candidates else fallback


class UploadStore:
    """Persisted store of uploaded documents and their chunks."""

    def __init__(self, directory: str = "cache/uploads", max_documents: int = 200) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_documents = max_documents

    def _path(self, document_id: str) -> Path:
        return self.directory / f"{document_id}.json"

    def _prune(self) -> None:
        files = sorted(self.directory.glob("*.json"), key=lambda f: f.stat().st_mtime)
        for path in files[: max(0, len(files) - self.max_documents)]:
            path.unlink(missing_ok=True)

    def add(self, data: bytes, filename: str) -> UploadedDoc:
        """Validate, extract, chunk and persist one PDF."""
        if not data:
            raise UploadError("The file is empty.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise UploadError(
                f"That file is {len(data) / 1e6:.1f} MB; the limit is "
                f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB."
            )
        if not data.startswith(PDF_MAGIC):
            raise UploadError("That does not look like a PDF.")

        # Content-addressed: re-uploading the same paper reuses the same id.
        document_id = hashlib.sha256(data).hexdigest()[:16]
        existing = self.get_meta(document_id)
        if existing is not None:
            return existing

        try:
            import pymupdf

            with pymupdf.open(stream=data, filetype="pdf") as doc:
                text = "\n".join(page.get_text() for page in doc)
                layout_title = _title_from_layout(doc)
                meta_title = " ".join((doc.metadata or {}).get("title", "").split())
        except Exception as exc:  # noqa: BLE001
            raise UploadError(f"Could not read that PDF ({type(exc).__name__}).") from exc

        words = text.split()
        if len(words) < 200:
            raise UploadError(
                "Almost no text came out of that PDF - it is probably a scan. "
                "This needs a text-based PDF, not images."
            )

        # Layout first, then embedded metadata, then the line heuristic.
        title = layout_title or (meta_title if len(meta_title) > 10 else "") or _guess_title(
            text, Path(filename).stem
        )
        paper = Paper(
            paper_id=document_id,
            version=1,
            title=title,
            abstract="",
            authors=[],
            categories=[],
            primary_category="upload",
            published="",
            updated="",
            pdf_url="",
            full_text=text,
        )
        chunks = chunk_paper(paper)
        if not chunks:
            raise UploadError("No usable passages could be extracted from that PDF.")

        meta = UploadedDoc(
            document_id=document_id,
            title=title,
            n_chunks=len(chunks),
            n_words=len(words),
            uploaded_at=time.time(),
            filename=filename,
        )
        self._path(document_id).write_text(
            json.dumps(
                {"meta": asdict(meta), "chunks": [asdict(c) for c in chunks]},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self._prune()
        return meta

    def _load(self, document_id: str) -> dict[str, Any] | None:
        path = self._path(document_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def get_meta(self, document_id: str) -> UploadedDoc | None:
        payload = self._load(document_id)
        return UploadedDoc(**payload["meta"]) if payload else None

    def passages(self, document_id: str) -> list[dict[str, Any]]:
        """Every chunk of one document, in the shape the LLM layer expects."""
        payload = self._load(document_id)
        if not payload:
            return []
        title = payload["meta"]["title"]
        return [
            {
                "chunk_id": c["chunk_id"],
                "doc_id": c["chunk_id"],
                "paper_id": document_id,
                "arxiv_url": "",
                "score": 0.0,
                "title": title,
                "section": c["section"],
                "published": "",
                "text": c["text"],
            }
            for c in payload["chunks"]
        ]
