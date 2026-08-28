"""Assemble a deploy/ folder ready to push to Hugging Face Spaces.

    python scripts/prepare_deploy.py

Two things happen here that matter.

**The papers file gets slimmed.** It carries `full_text` for every paper with a
PDF - 82 MB on disk and more once parsed into memory. The chunks file already
holds that text, split into the passages retrieval actually uses, so the copy in
the papers file is dead weight at serving time. Only citation lookup reads these
records, and only for metadata.

**The prebuilt indices come along.** Building BM25 and FAISS over 40k chunks
takes a few minutes; on a Space that would happen on every cold start, and free
Spaces sleep. Shipping `cache/ml/` turns a three-minute wake into seconds.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import load_config

# Fields the serving path never reads. `full_text` is the big one.
DROP_FIELDS = ("full_text", "metadata")


def slim_papers(source: Path, target: Path) -> tuple[int, int, int]:
    """Copy papers without their full text. Returns (count, before, after)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with open(source, encoding="utf-8") as src, open(target, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            for field in DROP_FIELDS:
                record.pop(field, None)
            # `full_text` is not optional in the dataclass, so keep the key with
            # a null value rather than removing it - read_papers() constructs
            # Paper(**record) and would fail on a missing field.
            record["full_text"] = None
            record["metadata"] = {}
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
    return kept, source.stat().st_size, target.stat().st_size


def copy_tree(source: Path, target: Path) -> int:
    if not source.exists():
        return 0
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    return sum(f.stat().st_size for f in target.rglob("*") if f.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ml.yaml")
    parser.add_argument("--out", default="deploy")
    parser.add_argument("--skip-index", action="store_true",
                        help="omit prebuilt indices; the Space rebuilds them on first boot")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mb = lambda n: f"{n / 1e6:.0f} MB"  # noqa: E731

    papers_src = Path(cfg["paths"]["papers"])
    chunks_src = Path(cfg["paths"]["chunks"])
    for path in (papers_src, chunks_src):
        if not path.exists():
            raise SystemExit(f"{path} is missing. Run scripts/ingest_arxiv.py first.")

    count, before, after = slim_papers(papers_src, out / "data" / papers_src.name)
    print(f"[deploy] papers   {count:,} records  {mb(before)} -> {mb(after)}")

    shutil.copy(chunks_src, out / "data" / chunks_src.name)
    chunk_bytes = (out / "data" / chunks_src.name).stat().st_size
    print(f"[deploy] chunks   {mb(chunk_bytes)}")

    index_bytes = 0
    if args.skip_index:
        print("[deploy] indices  skipped - the Space will build them on first boot (~3 min)")
    else:
        index_bytes = copy_tree(Path(cfg["paths"]["index_cache"]), out / "cache" / "ml")
        print(f"[deploy] indices  {mb(index_bytes)}" if index_bytes
              else "[deploy] indices  none found; run a query locally to build them first")

    total = after + chunk_bytes + index_bytes
    print(f"\n[deploy] total    {mb(total)} in {out}/")
    if total > 400e6:
        print("[deploy] over 400 MB - push with git-lfs, and expect a slow first upload")
    print("\nNext: copy deploy/data and deploy/cache into your Space repo alongside")
    print("Dockerfile.space (renamed to Dockerfile), src/, configs/ and scripts/.")
    print("See DEPLOY.md for the full sequence.")


if __name__ == "__main__":
    main()
