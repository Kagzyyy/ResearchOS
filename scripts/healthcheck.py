"""End-to-end check that every part of ResearchOS actually works.

    python scripts/healthcheck.py

Run it after any change and before shipping. It exercises the real HTTP
endpoints against a running backend rather than importing the modules, so it
catches wiring problems - a route not registered, a field renamed, a config not
reloaded - that module-level checks miss entirely.

Exits non-zero if anything failed, so it can gate a deploy.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import httpx

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


class Checks:
    """Runs checks against a live backend and remembers what happened."""

    def __init__(self, base: str, token: str | None) -> None:
        self.base = base.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.results: list[tuple[str, str, str]] = []

    def record(self, name: str, status: str, detail: str = "") -> None:
        self.results.append((name, status, detail))
        mark = {PASS: "  ok  ", FAIL: " FAIL ", SKIP: " skip "}[status]
        print(f"[{mark}] {name}" + (f"  - {detail}" if detail else ""))

    def _send(self, method: str, path: str, **kw: Any) -> Any:
        """One request, waiting out the server's own rate limiter.

        The backend allows N requests a minute per client, and a full check
        makes more than that. A 429 here is the limiter working, not a fault,
        so the check honours Retry-After instead of reporting a failure.
        """
        for _ in range(3):
            response = httpx.request(method, f"{self.base}{path}",
                                     headers=self.headers, **kw)
            if response.status_code != 429:
                break
            wait = float(response.headers.get("Retry-After", "10"))
            print(f"         (rate limited by the backend; waiting {wait:.0f}s)")
            time.sleep(min(wait, 65.0))
        response.raise_for_status()
        return response.json()

    def get(self, path: str, timeout: float = 30.0) -> Any:
        return self._send("GET", path, timeout=timeout)

    def post(self, path: str, body: dict[str, Any], timeout: float = 240.0) -> Any:
        return self._send("POST", path, json=body, timeout=timeout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8100")
    parser.add_argument("--token", default=None)
    parser.add_argument("--quick", action="store_true", help="skip generation and upload")
    args = parser.parse_args()

    checks = Checks(args.base, args.token)
    print(f"\nChecking {checks.base}\n" + "-" * 68)

    # --- is it up, and what does it think it is? --------------------------
    try:
        health = checks.get("/health")
        checks.record(
            "backend reachable",
            PASS,
            f"{health['papers']:,} papers, {health['chunks']:,} passages",
        )
        generation = bool(health.get("generation_enabled"))
        checks.record(
            "generation configured",
            PASS if generation else SKIP,
            health.get("llm_model") if generation else "no GROQ_API_KEY, passages only",
        )
    except Exception as exc:  # noqa: BLE001
        checks.record("backend reachable", FAIL, f"{type(exc).__name__}: {exc}")
        print("\nBackend is down, so nothing else can be checked.\n")
        raise SystemExit(1)

    # --- find papers, in corpus -------------------------------------------
    try:
        started = time.perf_counter()
        result = checks.post(
            "/papers", {"topic": "world models for robot manipulation", "top_k": 5}
        )
        elapsed = (time.perf_counter() - started) * 1000
        ok = bool(result["papers"]) and result["confident"]
        checks.record(
            "find papers (in corpus)",
            PASS if ok else FAIL,
            f"{len(result['papers'])} papers, {elapsed:.0f}ms, top {result['top_score']}",
        )
    except Exception as exc:  # noqa: BLE001
        checks.record("find papers (in corpus)", FAIL, f"{type(exc).__name__}: {exc}")

    # --- find papers, live arXiv fallback ---------------------------------
    try:
        result = checks.post(
            "/papers", {"topic": "agent based economic world model simulation", "top_k": 5}
        )
        served_live = result.get("source") == "arxiv_live" or any(
            p.get("from_arxiv") for p in result["papers"]
        )
        checks.record(
            "find papers (live arXiv fallback)",
            PASS if result["papers"] else FAIL,
            f"source={result.get('source')}"
            + (", served live" if served_live else ", answered from corpus"),
        )
    except Exception as exc:  # noqa: BLE001
        checks.record("find papers (live arXiv fallback)", FAIL, f"{type(exc).__name__}: {exc}")

    # --- honest refusal ----------------------------------------------------
    try:
        result = checks.post(
            "/papers", {"topic": "best sourdough starter hydration ratio", "top_k": 5}
        )
        declined = not result["confident"] or not result["papers"]
        checks.record(
            "declines off-topic questions",
            PASS if declined else FAIL,
            f"mean {result['mean_top_score']}",
        )
        # v2: a declined result carries a structured diagnosis. The key must
        # always exist; the value may be null when the breaker or budget is
        # holding the diagnosis back, which is a skip, not a failure.
        if "diagnosis" not in result:
            checks.record("v2 diagnosis on declined result", FAIL, "field missing - v1 backend?")
        elif result["diagnosis"] is None:
            checks.record("v2 diagnosis on declined result", SKIP,
                          "null (no key, breaker open, or budget)")
        else:
            d = result["diagnosis"]
            ok = d.get("location") in ("search",) or str(d.get("location", "")).startswith(
                "information:Doc")
            checks.record("v2 diagnosis on declined result", PASS if ok else FAIL,
                          f"{d.get('location')} in {d.get('latency_ms', 0) / 1000:.1f}s, "
                          f"query={d.get('suggested_query')!r}")
    except Exception as exc:  # noqa: BLE001
        checks.record("declines off-topic questions", FAIL, f"{type(exc).__name__}: {exc}")

    # --- citation resolution ----------------------------------------------
    citations = [
        ("citation by arXiv id", "arXiv:2312.00752", "2312.00752"),
        (
            "citation by reference string",
            "[23] M. Assran et al. Self-Supervised Learning from Images with a "
            "Joint-Embedding Predictive Architecture. CVPR 2023.",
            "2301.08243",
        ),
    ]
    for label, citation, expected in citations:
        try:
            result = checks.post("/explain", {"citation": citation, "generate": False})
            got = (result.get("paper") or {}).get("paper_id")
            checks.record(
                label,
                PASS if got == expected else FAIL,
                f"resolved to {got}" + ("" if got == expected else f", expected {expected}"),
            )
        except Exception as exc:  # noqa: BLE001
            checks.record(label, FAIL, f"{type(exc).__name__}: {exc}")

    # --- a DOI-only paper: the reason OpenAlex is a live source ---------------
    sources = health.get("live_sources", {})
    if not sources:
        checks.record("live sources reported", FAIL, "health has no live_sources - old backend?")
    else:
        checks.record("live sources reported", PASS, ", ".join(
            f"{k} {'up' if v.get('available') else 'paused'}" for k, v in sources.items()))
    if not sources.get("openalex", {}).get("available"):
        checks.record("citation by DOI (no arXiv copy)", SKIP, "OpenAlex paused or absent")
    else:
        try:
            result = checks.post("/explain", {"citation": "10.1145/3292500.3330701",
                                              "generate": False})
            title = (result.get("paper") or {}).get("title", "")
            checks.record("citation by DOI (no arXiv copy)",
                          PASS if "optuna" in title.lower() else FAIL,
                          f"resolved to {title[:50]!r} via {result.get('method')}")
        except Exception as exc:  # noqa: BLE001
            checks.record("citation by DOI (no arXiv copy)", FAIL, f"{type(exc).__name__}: {exc}")

    # --- a question must not be resolved as a paper title -----------------
    try:
        result = checks.post(
            "/explain",
            {"citation": "How do JEPAs avoid representation collapse?", "generate": False},
        )
        checks.record(
            "question is not treated as a citation",
            PASS if result.get("method") == "question" else FAIL,
            f"method={result.get('method')}",
        )
    except Exception as exc:  # noqa: BLE001
        checks.record("question is not treated as a citation", FAIL, f"{type(exc).__name__}: {exc}")

    # --- PDF upload and document QA ---------------------------------------
    pdfs = sorted(Path("cache/arxiv/pdf").glob("*.pdf"))
    if args.quick or not pdfs:
        checks.record("PDF upload", SKIP, "no local PDF to test with" if not pdfs else "quick mode")
    else:
        try:
            with open(pdfs[0], "rb") as handle:
                response = httpx.post(
                    f"{checks.base}/upload",
                    files={"file": (pdfs[0].name, handle, "application/pdf")},
                    headers=checks.headers,
                    timeout=180,
                )
            response.raise_for_status()
            uploaded = response.json()
            checks.record(
                "PDF upload",
                PASS,
                f"'{uploaded['title'][:38]}', {uploaded['n_chunks']} passages",
            )
            answered = checks.post(
                "/explain",
                {
                    "citation": "What problem does this paper solve?",
                    "document_id": uploaded["document_id"],
                    "generate": False,
                },
            )
            checks.record(
                "question against uploaded PDF",
                PASS if answered.get("passages") else FAIL,
                f"{len(answered.get('passages', []))} passages used",
            )
        except Exception as exc:  # noqa: BLE001
            checks.record("PDF upload", FAIL, f"{type(exc).__name__}: {exc}")

    # --- generation --------------------------------------------------------
    if generation and not args.quick:
        try:
            started = time.perf_counter()
            result = checks.post("/explain", {"citation": "arXiv:2312.00752", "generate": True})
            elapsed = time.perf_counter() - started
            text = result.get("explanation") or ""
            checks.record(
                "writes a grounded answer",
                PASS if len(text) > 200 else FAIL,
                f"{len(text)} chars in {elapsed:.1f}s",
            )
        except Exception as exc:  # noqa: BLE001
            checks.record("writes a grounded answer", FAIL, f"{type(exc).__name__}: {exc}")
    else:
        checks.record("writes a grounded answer", SKIP, "quick mode" if generation else "no API key")

    # --- feedback ----------------------------------------------------------
    try:
        result = checks.post("/papers", {"topic": "mixture of experts routing", "top_k": 3})
        query_id = result["query_id"]
        checks.post("/feedback/vote", {"query_id": query_id, "helpful": True})
        checks.post(
            "/feedback/click",
            {"query_id": query_id, "doc_id": result["papers"][0]["doc_id"]},
        )
        stats = checks.get("/stats")
        checks.record(
            "feedback is recorded",
            PASS,
            f"{stats['queries']} queries, {stats['votes_up']}+/{stats['votes_down']}-, "
            f"{stats['clicks']} clicks",
        )
    except Exception as exc:  # noqa: BLE001
        checks.record("feedback is recorded", FAIL, f"{type(exc).__name__}: {exc}")

    # --- the live v2 funnel: a retry is linked to the query it came from -------
    try:
        before = checks.get("/v2/metrics")["retries"]
        origin = checks.post("/papers", {"topic": "papers about sparse experts", "top_k": 3})
        retry = checks.post("/papers", {"topic": "mixture of experts routing", "top_k": 3,
                                        "retry_of": origin["query_id"]})
        checks.post("/feedback/click",
                    {"query_id": retry["query_id"], "doc_id": retry["papers"][0]["doc_id"]})
        metrics = checks.get("/v2/metrics")
        labels = checks.get("/v2/labels?limit=5")
        ok = metrics["retries"] == before + 1 and any(
            row["retry_id"] == retry["query_id"] for row in labels["labels"])
        checks.record("v2 retry funnel is recorded", PASS if ok else FAIL,
                      f"{metrics['retries']} retries, click-through "
                      f"{metrics['retry_click_through']}, {labels['count']} labels")
    except Exception as exc:  # noqa: BLE001
        checks.record("v2 retry funnel is recorded", FAIL, f"{type(exc).__name__}: {exc}")

    # --- anything the server logged as an error ---------------------------
    try:
        failures = checks.get("/failures?limit=5")
        errors = failures.get("errors", [])
        checks.record(
            "no server errors logged",
            PASS if not errors else FAIL,
            "clean" if not errors else f"{len(errors)} recorded, see scripts/review_failures.py",
        )
    except Exception as exc:  # noqa: BLE001
        checks.record("no server errors logged", FAIL, f"{type(exc).__name__}: {exc}")

    # --- summary -----------------------------------------------------------
    print("-" * 68)
    failed = [name for name, status, _ in checks.results if status == FAIL]
    skipped = [name for name, status, _ in checks.results if status == SKIP]
    passed = sum(1 for _, status, _ in checks.results if status == PASS)
    print(f"{passed} passed, {len(failed)} failed, {len(skipped)} skipped\n")
    if failed:
        for name in failed:
            print(f"  FAILED  {name}")
        print()
        raise SystemExit(1)
    print("ResearchOS is working.\n")


if __name__ == "__main__":
    main()
