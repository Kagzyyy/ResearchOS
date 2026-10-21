"""Does v2 actually beat v1? End-to-end rescue evaluation.

    python scripts/eval_v2_rescue.py --n 150

The question a user cares about: "I typed a vague description of a paper I
half-remember, and the search failed. Does v2 get me there?" This measures
exactly that, and attributes the gain fairly.

Protocol
--------
1. Sample real papers from the corpus (seed 42). Because the target is in the
   index, a miss is retrieval's fault, not a coverage gap.
2. For each, an LLM writes the *vague* query a researcher who half-remembers it
   would type - forbidden from using its title, acronyms or method names. This
   is deliberately hard: v1's generated benchmark saturated at >= 0.958
   because its queries sat too close to the source text, so it could not rank
   anything.
3. Run v1 (find_papers, production config, live arXiv off for determinism).
   Keep every query where its confidence gate fires - exactly the condition
   under which v2 activates in production. That includes queries v1 actually
   got right, so a repair can harm as well as rescue, and both are counted.
4. On those queries, compare two repairs, both followed by a fresh search:
     v2       CRITIC-R1-format diagnosis of the retrieved documents, then its
              suggested query
     control  an LLM rewrite of the query that never sees the documents
   The control is what makes this an ablation: if v2 only matched it, the
   gain would be "an LLM rephrased the query", not "the critic saw what went
   wrong". Both use the same model and temperature.

Reported: recall@10 of v1, v2 and the control on the gate-fired queries with
Wilson 95% CIs, rescues (v1 miss -> hit) and harms (v1 hit -> miss), exact
McNemar tests between every pair on identical queries, and the gate's own
calibration: recall@10 when it is confident vs when it fires. Misses the gate
never flags are out of reach of any gate-triggered repair - reported, not
hidden.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import src.config  # noqa: F401,E402  - loads .env

from critic_localization import mcnemar_exact, wilson  # noqa: E402

GEN_SYSTEM = """You write search queries the way a researcher types them when they
half-remember a paper. You are given a paper's title and abstract.

Write ONE query of 6-14 words describing what the paper is about.
Rules:
- Do NOT use any distinctive word from the title.
- Do NOT use acronyms, model names, dataset names or method names from the paper.
- Describe the problem or idea in plain, slightly vague terms, as someone recalling it would.
Output only the query, nothing else."""

REWRITE_SYSTEM = """Rewrite the user's search query so it is more likely to find the
right AI/ML research paper in a paper search engine. Use the technical terms a paper
on this topic would use. Output only the new query (3-12 words), nothing else."""


def backoff(exc: Exception, attempt: int) -> float:
    """Seconds to wait before retrying.

    A per-minute limit clears in seconds, so short exponential steps suffice.
    The daily token cap (200K per model on Groq's free tier) is a rolling
    window: Groq names the wait in the error ("try again in 2m32.4s"), and
    sleeping exactly that long lets an eval bigger than one day's budget
    finish unattended instead of crashing on a 429.
    """
    msg = str(exc)
    if "tokens per day" in msg:
        m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", msg)
        if m:
            h, mnt, sec = (float(g) if g else 0.0 for g in m.groups())
            wait = h * 3600 + mnt * 60 + sec + 5
            print(f"    daily token cap reached; waiting {wait / 60:.1f} min", flush=True)
            return wait
    return min(60, 10 * (attempt + 1))


def groq(system: str, user: str, model: str, retries: int = 30) -> str:
    """One short completion with patient retry - this is an offline eval, so
    waiting out a rate limit is correct here, unlike in a user request."""
    from groq import Groq
    client = Groq(timeout=60.0, max_retries=0)
    extra = ({"reasoning_effort": "low"} if model.startswith("openai/gpt-oss")
             else {"reasoning_format": "hidden"})
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0.0, max_tokens=300,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                extra_body=extra)
            return (r.choices[0].message.content or "").strip().strip('"').strip()
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            time.sleep(backoff(exc, attempt))
            print(f"    retry after {type(exc).__name__}", flush=True)
    return ""


def robust_diagnose(query: str, papers: list[dict], model: str, retries: int = 30):
    """The production diagnosis, minus its production shortcuts.

    In a user request diagnose() returns None on a rate limit, by design.
    Inside an evaluation that would silently score a 429 as a v2 *failure*
    and understate v2. So the eval uses the same prompt and the same
    validator, but waits out rate limits instead of giving up.
    """
    import src.api.diagnose as dg
    for attempt in range(retries):
        try:
            t0 = time.perf_counter()
            text, used = dg._call(query, papers, None, [model])
            return dg.validate(text, papers, query, used, (time.perf_counter() - t0) * 1000)
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            time.sleep(backoff(exc, attempt))
            print(f"    diagnose retry after {type(exc).__name__}", flush=True)
    return None


def rank_of(target: str, papers: list[dict]) -> int | None:
    for i, p in enumerate(papers, 1):
        if p["paper_id"] == target:
            return i
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--config", default="configs/ml.cpu.yaml")
    ap.add_argument("--model", default="openai/gpt-oss-120b",
                    help="diagnosis + control model (both arms always share one)")
    ap.add_argument("--gen-model", default="qwen/qwen3.8-27b",
                    help="writes the vague test queries; a different model family "
                         "from the diagnoser, so it cannot win by recognising "
                         "its own phrasing")
    ap.add_argument("--out", default="results/v2_rescue.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    cache_path = out.with_suffix(".cache.json")
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    save = lambda: cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")  # noqa: E731

    from src.api.ml_pipeline import MLPipeline
    from src.config import load_config
    cfg = load_config(args.config)
    cfg["citation"]["allow_remote"] = False     # deterministic: corpus only
    cfg.setdefault("diagnosis", {})["enabled"] = False   # called explicitly below
    pipe = MLPipeline(cfg)

    # ---- 1. sample targets with real abstracts -------------------------
    rng = random.Random(args.seed)
    pool = [p for p in pipe.papers_by_id.values()
            if len((p.get("abstract") or "").split()) >= 80]
    targets = rng.sample(pool, min(args.n, len(pool)))
    print(f"[rescue] {len(targets)} target papers sampled from {len(pool)}")

    # ---- 2. vague queries -------------------------------------------------
    queries = cache.setdefault("queries", {})
    for i, p in enumerate(targets, 1):
        pid = p["paper_id"]
        if pid not in queries:
            queries[pid] = groq(GEN_SYSTEM, f"Title: {p['title']}\n\nAbstract: {p['abstract'][:1500]}",
                                args.gen_model)
            if i % 10 == 0:
                save()
                print(f"    queries {i}/{len(targets)}", flush=True)
    save()

    # ---- 3. v1 --------------------------------------------------------
    rows = []
    for p in targets:
        pid, q = p["paper_id"], queries[p["paper_id"]]
        r = pipe.find_papers(q, top_k=10)
        rows.append({"paper_id": pid, "title": p["title"], "query": q,
                     "v1_rank": rank_of(pid, r["papers"]), "v1_confident": r["confident"],
                     "v1_mean_top": r["mean_top_score"], "v1_papers": r["papers"]})
    # ---- the gate as a failure detector, over every query ----------------
    # The gate asks "is there strong evidence for this topic?", not "did we
    # find the one paper the user meant?". A vague query can match many
    # right-topic papers (confident, target unranked) or the target can rank
    # first among weak matches (target found, gate fires). Measuring how far
    # the two disagree says how good a trigger the gate is for v2.
    confident_rows = [r for r in rows if r["v1_confident"]]
    fired = [r for r in rows if not r["v1_confident"]]
    recall = lambda rs: sum(1 for r in rs if r["v1_rank"]) / max(len(rs), 1)  # noqa: E731
    print(f"[rescue] v1 recall@10: {recall(rows):.1%} overall | "
          f"{recall(confident_rows):.1%} when confident (n={len(confident_rows)}) | "
          f"{recall(fired):.1%} when the gate fires (n={len(fired)})")

    # ---- 4. repairs on every query where v2 would run in production -------
    # Not just the misses: v2 fires on every gate-fired query, including ones
    # v1 actually got right - so following it can also cause harm, and that
    # has to be counted.
    # Keyed by model: repairs from one model must never be scored as another's.
    diag_cache = cache.setdefault(f"diagnosis:{args.model}", {})
    ctrl_cache = cache.setdefault(f"control:{args.model}", {})
    for i, r in enumerate(fired, 1):
        pid = r["paper_id"]
        if pid not in diag_cache:
            d = robust_diagnose(r["query"], r["v1_papers"], args.model)
            diag_cache[pid] = d.to_json() if d else None
        if pid not in ctrl_cache:
            ctrl_cache[pid] = groq(REWRITE_SYSTEM, r["query"], args.model)
        if i % 5 == 0:
            save()
            print(f"    repairs {i}/{len(fired)}", flush=True)
    save()

    for r in fired:
        pid = r["paper_id"]
        d = diag_cache.get(pid)
        r["v2_diagnosis"] = d
        r["v2_query"] = d.get("suggested_query") if d else None
        r["control_query"] = ctrl_cache.get(pid) or None
        for arm in ("v2", "control"):
            q = r[f"{arm}_query"]
            if q:
                res = pipe.find_papers(q, top_k=10)
                r[f"{arm}_rank"] = rank_of(pid, res["papers"])
                r[f"{arm}_confident"] = res["confident"]
            else:
                # No runnable query means no button: the user keeps v1's result.
                r[f"{arm}_rank"] = r["v1_rank"]
                r[f"{arm}_confident"] = r["v1_confident"]
    for r in rows:
        r.pop("v1_papers", None)

    n = len(fired)
    hits = {arm: [bool(r.get(f"{arm}_rank")) for r in fired] for arm in ("v1", "v2", "control")}
    summary = {
        "targets": len(rows),
        "gen_model": args.gen_model,
        "repair_model": args.model,
        "gate": {
            "confident": len(confident_rows),
            "recall_at_10_when_confident": round(recall(confident_rows), 4),
            "fired": n,
            "recall_at_10_when_fired": round(recall(fired), 4),
        },
        "v1_recall_at_10_overall": round(recall(rows), 4),
        "diagnosis_produced": sum(1 for r in fired if r["v2_diagnosis"]),
        "diagnosis_with_query": sum(1 for r in fired if r["v2_query"]),
    }
    for arm in ("v1", "v2", "control"):
        k = sum(hits[arm])
        lo, hi = wilson(k, n)
        entry = {"recall_at_10": round(k / max(n, 1), 4), "ci95": [round(lo, 4), round(hi, 4)],
                 "found": k}
        if arm != "v1":
            entry["rescued"] = sum(1 for a, b in zip(hits["v1"], hits[arm]) if b and not a)
            entry["harmed"] = sum(1 for a, b in zip(hits["v1"], hits[arm]) if a and not b)
            entry["passes_gate_after"] = round(
                sum(1 for r in fired if r.get(f"{arm}_confident")) / max(n, 1), 4)
        summary[arm] = entry
    summary["paired"] = {}
    for a, b in (("v1", "v2"), ("v1", "control"), ("v2", "control")):
        only_a, only_b, p = mcnemar_exact(hits[a], hits[b])
        summary["paired"][f"{a}_vs_{b}"] = {f"only_{a}": only_a, f"only_{b}": only_b,
                                           "p_exact": round(p, 5)}

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
                   encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\n[rescue] saved -> {out}")


if __name__ == "__main__":
    main()
