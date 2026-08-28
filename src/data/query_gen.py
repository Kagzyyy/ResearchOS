"""Generate natural-language evaluation queries from papers (doc2query / InPars).

Why this exists
---------------
The sentence-based benchmark in scripts/eval_retrieval.py uses verbatim text
from the target paper as the query. That hands BM25 an exact lexical match and
is why it scores 0.89 there. It is not what a user types.

Here the model reads a paper and writes the question a researcher would type to
FIND it. The label is still structural, not an opinion: the question was written
from that paper, so that paper answers it. The model generates the *query*, never
the *verdict* - which is what separates this from LLM-as-judge.

The prompt forbids reusing the paper's title, acronyms and coined terms. Without
that the model paraphrases the title, the lexical gift comes straight back, and
the benchmark measures nothing new.

Known weakness: false negatives. A generated question about paper A may also be
answered by B and C, but only A is marked relevant, so absolute recall reads low.
Every configuration is penalised identically, so comparisons stay valid.
`judge.py` recovers most of the loss by judging a pooled candidate set.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from src.api.llm import _complete
from src.data.arxiv_source import Paper

QUERY_SYSTEM_PROMPT = """You write realistic search queries for a research-paper \
search engine.

You are shown one paper. Write ONE query that a researcher would type into a search \
box when they are looking for work like this but have never seen this paper.

Rules:
- 8 to 20 words. Natural language. A question or a topic phrase, not a title.
- Describe the PROBLEM and the APPROACH in generic terms.
- Do NOT reuse the paper's title, its acronym, its model name, or any term the \
authors coined. If the paper calls its method "FlowMPC", never write "FlowMPC".
- Do NOT copy distinctive multi-word phrases from the text verbatim.
- Write what someone searching would plausibly know to ask, not what the paper calls \
itself.

Output only the query. No preamble, no quotes, no explanation."""


@dataclass
class GeneratedQuery:
    query_id: str
    query: str
    gold_paper_id: str
    gold_title: str


def _looks_leaky(query: str, title: str, min_overlap: int = 4) -> bool:
    """True when the query copies a long run of words from the title."""
    def words(text: str) -> list[str]:
        return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()

    q, t = words(query), words(title)
    if len(q) < min_overlap or len(t) < min_overlap:
        return False
    t_grams = {tuple(t[i : i + min_overlap]) for i in range(len(t) - min_overlap + 1)}
    return any(tuple(q[i : i + min_overlap]) in t_grams for i in range(len(q) - min_overlap + 1))


def _clean(text: str) -> str:
    text = text.strip().strip('"').strip("'").strip()
    text = re.sub(r"^(query|question|search)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def generate_one(
    paper: Paper,
    context: str,
    model: str,
    max_tokens: int,
    temperature: float,
    retries: int = 4,
) -> tuple[str | None, str]:
    """One query for one paper.

    Returns (query, reason). reason is "ok", "leak", "malformed" or "api".
    These were previously collapsed into a single None, which meant Groq
    rate-limit errors were reported as "rejected for copying the title" - the
    run looked like a strict filter working when it was actually being throttled.
    """
    user = (
        f"Title: {paper.title}\n\n"
        f"Abstract: {paper.abstract[:1200]}\n\n"
        f"Excerpt: {context[:900]}\n\n"
        "Write the search query."
    )
    raw = None
    for attempt in range(retries):
        try:
            raw = _complete(QUERY_SYSTEM_PROMPT, user, model, max_tokens, temperature)
            break
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            transient = "rate" in message or "429" in message or "timeout" in message
            if attempt == retries - 1 or not transient:
                return None, "api"
            time.sleep(2.0 * (2**attempt) + random.random())
    if raw is None:
        return None, "api"

    query = _clean(raw)
    if not (5 <= len(query.split()) <= 40):
        return None, "malformed"
    if _looks_leaky(query, paper.title):
        return None, "leak"
    return query, "ok"


def generate_queries(
    papers: list[Paper],
    context_by_paper: dict[str, str],
    n_queries: int,
    model: str,
    out_path: str | Path,
    max_tokens: int = 900,
    temperature: float = 0.7,
    workers: int = 6,
    rng: random.Random | None = None,
) -> list[GeneratedQuery]:
    """Generate queries in parallel, writing incrementally so a crash costs nothing."""
    rng = rng or random.Random(0)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    done: dict[str, GeneratedQuery] = {}
    if target.exists():
        for line in target.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = GeneratedQuery(**json.loads(line))
                done[item.gold_paper_id] = item
        print(f"[qgen] resuming: {len(done)} already generated")

    candidates = [p for p in papers if p.paper_id in context_by_paper and p.paper_id not in done]
    rng.shuffle(candidates)
    # Oversample: roughly half get rejected for copying the title, so taking
    # exactly `n_queries` candidates permanently loses a slot per rejection.
    wanted = max(0, n_queries - len(done))
    candidates = candidates[: wanted * 3]
    if not candidates:
        return list(done.values())

    lock = threading.Lock()
    handle = open(target, "a", encoding="utf-8")
    counts = {"leak": 0, "api": 0, "malformed": 0}

    def work(paper: Paper) -> None:
        if len(done) >= n_queries:
            return
        query, reason = generate_one(
            paper, context_by_paper[paper.paper_id], model, max_tokens, temperature
        )
        if query is None:
            with lock:
                counts[reason] = counts.get(reason, 0) + 1
            return
        item = GeneratedQuery(
            query_id=f"g{paper.paper_id}",
            query=query,
            gold_paper_id=paper.paper_id,
            gold_title=paper.title,
        )
        with lock:
            done[paper.paper_id] = item
            handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
            handle.flush()
            if len(done) % 25 == 0:
                print(f"[qgen] {len(done)}/{n_queries} generated | {counts}")

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, candidates))
    finally:
        handle.close()

    print(
        f"[qgen] {len(done)} queries | {counts['leak']} rejected for copying the title, "
        f"{counts['api']} API failures, {counts['malformed']} malformed"
    )
    return list(done.values())


def read_generated(path: str | Path) -> list[GeneratedQuery]:
    with open(path, encoding="utf-8") as handle:
        return [GeneratedQuery(**json.loads(line)) for line in handle if line.strip()]
