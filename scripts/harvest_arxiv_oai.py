"""Bulk-harvest arXiv abstracts through OAI-PMH, resumably.

    python scripts/harvest_arxiv_oai.py --created-from 2024-01-01

The search API that built the v1 corpus is meant for queries, not bulk
download: it pages 100 results at a time, sorts poorly past ~10k results,
and throttles hard. OAI-PMH is arXiv's official bulk-metadata interface:
~1,300 records per response, a resumption token per page, no key.

Two filters are applied locally, because OAI-PMH can only filter by the
date a record was last *touched*:
  * primary category in the AI set the v1 corpus was built from
  * first submitted on or after --created-from - otherwise every 2009 paper
    whose metadata was edited last week comes along. The date comes from the
    arXiv id (2203.06424 = March 2022), not the <created> field, which
    measured wrong: it reported 2024-01-01 for that 2022 paper.

Resumable: the resumption token and the output are checkpointed after every
page, so an interruption (or arXiv's 503 "retry later") costs one page.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import httpx  # noqa: E402

from src.data.arxiv_source import Paper  # noqa: E402

ENDPOINT = "https://oaipmh.arxiv.org/oai"
NS = {"oai": "http://www.openarchives.org/OAI/2.0/", "a": "http://arxiv.org/OAI/arXiv/"}
AI_CATEGORIES = ("cs.LG", "cs.AI", "cs.CL", "cs.CV", "cs.NE", "cs.RO", "cs.IR", "stat.ML")


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def id_month(paper_id: str) -> str | None:
    """'2203.06424' -> '2022-03'. Old-style ids (cs/0101001) predate any window."""
    m = re.match(r"^(\d{2})(\d{2})\.\d{4,5}$", paper_id or "")
    return f"20{m.group(1)}-{m.group(2)}" if m else None


def parse_page(xml_text: str, categories: set[str], created_from: str
               ) -> tuple[list[Paper], str | None, int]:
    """Records kept, the next resumption token (None at the end), records seen."""
    root = ET.fromstring(xml_text)
    error = root.find("oai:error", NS)
    if error is not None:
        if error.get("code") == "noRecordsMatch":
            return [], None, 0
        raise RuntimeError(f"OAI error {error.get('code')}: {error.text}")
    kept: list[Paper] = []
    records = root.findall(".//oai:record", NS)
    for record in records:
        header = record.find("oai:header", NS)
        if header is not None and header.get("status") == "deleted":
            continue
        meta = record.find(".//a:arXiv", NS)
        if meta is None:
            continue
        cats = (meta.findtext("a:categories", "", NS) or "").split()
        if not cats or cats[0] not in categories:     # first listed = primary
            continue
        paper_id = _clean(meta.findtext("a:id", "", NS))
        month = id_month(paper_id)
        if month is None or month < created_from[:7]:
            continue
        created = meta.findtext("a:created", "", NS) or ""
        # <created> is only trusted when it agrees with the id's month.
        published = created if created[:7] == month else f"{month}-01"
        authors = []
        for a in meta.findall("a:authors/a:author", NS):
            name = " ".join(x for x in (_clean(a.findtext("a:forenames", "", NS)),
                                        _clean(a.findtext("a:keyname", "", NS))) if x)
            if name:
                authors.append(name)
        title, abstract = _clean(meta.findtext("a:title", "", NS)), _clean(meta.findtext("a:abstract", "", NS))
        if not (paper_id and title and abstract):
            continue
        kept.append(Paper(
            paper_id=paper_id, version=1, title=title, abstract=abstract,
            authors=authors[:20], categories=cats, primary_category=cats[0],
            published=published, updated=meta.findtext("a:updated", "", NS) or published,
            pdf_url=f"https://arxiv.org/pdf/{paper_id}",
        ))
    token_el = root.find(".//oai:resumptionToken", NS)
    token = (token_el.text or "").strip() if token_el is not None else ""
    return kept, (token or None), len(records)


def fetch(client: httpx.Client, params: dict[str, str]) -> str:
    """One page, honouring arXiv's 503 Retry-After. Patient: this is offline."""
    for attempt in range(12):
        try:
            r = client.get(ENDPOINT, params=params, timeout=120.0)
        except httpx.HTTPError as exc:
            wait = min(300, 15 * (attempt + 1))
            print(f"    {type(exc).__name__}; retrying in {wait}s", flush=True)
            time.sleep(wait)
            continue
        if r.status_code in (429, 503):
            wait = int(r.headers.get("Retry-After", "30") or 30)
            print(f"    {r.status_code}; arXiv asks to wait {wait}s", flush=True)
            time.sleep(min(wait, 600))
            continue
        r.raise_for_status()
        return r.text
    raise RuntimeError("arXiv OAI-PMH kept refusing; rerun later to resume")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sets", nargs="+", default=["cs", "stat:stat:ML"],
                    help="stat:stat:ML, not stat: the full stat set is mostly math")
    ap.add_argument("--created-from", default="2024-01-01")
    ap.add_argument("--categories", nargs="+", default=list(AI_CATEGORIES))
    ap.add_argument("--out", default="data/arxiv_abstracts.jsonl")
    ap.add_argument("--delay", type=float, default=3.0, help="seconds between pages")
    args = ap.parse_args()

    out = Path(args.out)
    state_path = out.with_suffix(".state.json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    out.parent.mkdir(parents=True, exist_ok=True)
    cats = set(args.categories)

    with httpx.Client(headers={"User-Agent": "ResearchOS harvester "
                               "(https://github.com/Kagzyyy/ResearchOS)"}) as client:
        for oai_set in args.sets:
            s = state.setdefault(oai_set, {"token": None, "done": False, "pages": 0,
                                           "seen": 0, "kept": 0})
            if s["done"]:
                print(f"[oai] {oai_set}: already complete ({s['kept']:,} kept)")
                continue
            while True:
                params = ({"verb": "ListRecords", "resumptionToken": s["token"]} if s["token"]
                          else {"verb": "ListRecords", "metadataPrefix": "arXiv",
                                "set": oai_set, "from": args.created_from})
                papers, token, seen = parse_page(fetch(client, params), cats, args.created_from)
                with open(out, "a", encoding="utf-8") as fh:
                    for p in papers:
                        fh.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
                s.update(token=token, pages=s["pages"] + 1, seen=s["seen"] + seen,
                         kept=s["kept"] + len(papers), done=token is None)
                state_path.write_text(json.dumps(state))     # checkpoint every page
                if s["pages"] % 10 == 0 or token is None:
                    print(f"[oai] {oai_set}: page {s['pages']}, seen {s['seen']:,}, "
                          f"kept {s['kept']:,}", flush=True)
                if token is None:
                    break
                time.sleep(args.delay)
    total = sum(v["kept"] for v in state.values())
    print(f"[oai] done: {total:,} records written to {out} (duplicates resolved at build)")


if __name__ == "__main__":
    main()
