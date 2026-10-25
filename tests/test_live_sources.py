"""Tests for live paper sources: OpenAlex parsing, budget rails, merging,
and citation resolution. Offline - every network call is replaced.

Run: `python tests/test_live_sources.py`
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

import src.data.citation as citation  # noqa: E402
import src.data.openalex_source as oa  # noqa: E402
from src.api.ml_pipeline import MLPipeline  # noqa: E402

WORK = {
    "id": "https://openalex.org/W123",
    "doi": "https://doi.org/10.48550/arXiv.1706.03762",
    "title": "Attention  Is All You Need",
    "publication_date": "2017-06-12",
    "publication_year": 2017,
    "abstract_inverted_index": {"The": [0], "dominant": [1], "models": [2, 4], "sequence": [3]},
    "authorships": [{"author": {"display_name": "Ashish Vaswani"}}],
    "primary_location": {"source": {"display_name": "NeurIPS/NeurIPS Proceedings"}},
    "locations": [],
    "cited_by_count": 100000,
    "open_access": {"oa_url": "https://example.org/a.pdf"},
}


def _reset() -> None:
    oa._STATE.update(open_until=0.0, reason="", remaining_usd=None, limit_usd=None)
    oa._CACHE.clear()


# ---------------------------------------------------------------- parsing ----
def test_abstract_is_rebuilt_from_inverted_index():
    assert oa.reconstruct_abstract(WORK["abstract_inverted_index"]) == \
        "The dominant models sequence models"
    assert oa.reconstruct_abstract(None) == ""


def test_arxiv_doi_gives_arxiv_identity_and_link():
    w = oa.parse_work(WORK)
    assert w["paper_id"] == "1706.03762"          # collapses with arXiv/corpus copies
    assert w["url"] == "https://arxiv.org/abs/1706.03762"
    assert w["title"] == "Attention Is All You Need"
    assert w["venue"] == "NeurIPS"


def test_arxiv_id_found_in_locations_and_doi_link_otherwise():
    w = dict(WORK, doi="https://doi.org/10.1109/CVPR.2016.90",
             locations=[{"landing_page_url": "http://arxiv.org/abs/1512.03385"}])
    assert oa.parse_work(w)["paper_id"] == "1512.03385"
    w = dict(WORK, doi="https://doi.org/10.1109/CVPR.2016.90", locations=[])
    parsed = oa.parse_work(w)
    assert parsed["paper_id"] == "openalex:W123"
    assert parsed["url"] == "https://doi.org/10.1109/cvpr.2016.90"


def test_doi_extraction_and_normalisation():
    assert oa.extract_doi("He et al. doi:10.1109/CVPR.2016.90.") == "10.1109/cvpr.2016.90"
    assert oa.normalise_doi("https://dx.doi.org/10.1145/ABC") == "10.1145/abc"
    assert oa.extract_doi("no identifier here") is None


# ------------------------------------------------------------------ rails ----
def test_budget_headers_pause_before_exhaustion():
    _reset()
    oa._record_budget(httpx.Headers({"X-RateLimit-Remaining-USD": "0.004",
                                     "X-RateLimit-Limit-USD": "0.1",
                                     "X-RateLimit-Reset": "600"}))
    assert oa.breaker_open(), "must stop with budget left, not run it to zero"
    st = oa.status()
    assert st["available"] is False and st["remaining_usd"] == 0.004
    _reset()
    oa._record_budget(httpx.Headers({"X-RateLimit-Remaining-USD": "0.09"}))
    assert not oa.breaker_open()


def test_paused_source_costs_no_network_time():
    _reset()
    oa._pause(60, "test")
    t0 = time.time()
    try:
        oa.search_works("anything")
        raise AssertionError("expected OpenAlexUnavailable")
    except oa.OpenAlexUnavailable:
        pass
    assert time.time() - t0 < 0.05
    _reset()


def test_timeout_trips_breaker():
    _reset()
    try:
        oa._bounded("slow", lambda: time.sleep(1.0), budget_s=0.1)
        raise AssertionError("expected a timeout")
    except oa.OpenAlexUnavailable:
        pass
    assert oa.breaker_open()
    _reset()


def test_results_are_cached():
    _reset()
    calls = []
    oa._bounded("k", lambda: calls.append(1) or ["x"], 1.0)
    assert oa._bounded("k", lambda: calls.append(1) or ["y"], 1.0) == ["x"]
    assert len(calls) == 1
    _reset()


# ---------------------------------------------------------------- merging ----
def _live(pid, title, source, **kw):
    return dict(paper_id=pid, title=title, live_source=source, venue=kw.get("venue", ""),
                cited_by_count=kw.get("cites"))


def _fake_pipeline(arxiv_items, openalex_items, slow_openalex=False):
    def _oa(q, k, deadline):
        if slow_openalex:
            time.sleep(2.0)
        return openalex_items
    return types.SimpleNamespace(
        cfg={"citation": {"live_budget_s": 0.5, "live_sources": ["arxiv", "openalex"]}},
        _live_arxiv=lambda q, k, d: arxiv_items,
        _live_openalex=_oa,
    )


def test_same_paper_from_both_sources_collapses():
    fake = _fake_pipeline(
        [_live("1706.03762", "Attention Is All You Need", "arxiv", venue="arXiv")],
        [_live("1706.03762", "Attention is all you need", "openalex", venue="NeurIPS", cites=9),
         _live("openalex:W9", "Deep Residual Learning", "openalex")],
    )
    out = MLPipeline.search_live(fake, "q")
    assert [p["paper_id"] for p in out] == ["1706.03762", "openalex:W9"]
    assert out[0]["live_source"] == "arxiv+openalex" and out[0]["cited_by_count"] == 9


def test_title_dedup_when_no_arxiv_id():
    fake = _fake_pipeline([], [_live("openalex:W1", "Deep Residual Learning!", "openalex"),
                               _live("openalex:W2", "deep residual learning", "openalex")])
    assert len(MLPipeline.search_live(fake, "q")) == 1


def test_slow_source_does_not_extend_the_budget():
    fake = _fake_pipeline([_live("1", "A paper", "arxiv")], [], slow_openalex=True)
    t0 = time.time()
    out = MLPipeline.search_live(fake, "q")
    assert time.time() - t0 < 1.0, "one shared deadline, not one per source"
    assert [p["paper_id"] for p in out] == ["1"]


def test_question_marks_and_operators_never_reach_openalex():
    """A "?" is a wildcard to OpenAlex, which rejected every question-form query."""
    assert oa.clean_query("What is the bias-variance tradeoff?") == "What is the bias-variance tradeoff"
    assert oa.clean_query('a*b "quoted" (x) [y] | z') == "a b quoted x y z"
    assert oa.clean_query("Title: A, B; C", drop=",:;") == "Title A B C"
    assert oa.clean_query("???") == ""


def test_search_sends_the_cleaned_query():
    _reset()
    sent = {}

    def fake_get(path, params):
        sent.update(params)
        return {"results": []}
    saved = oa._get
    oa._get = fake_get
    try:
        oa.search_works("How does attention work?", 3)
    finally:
        oa._get = saved
        _reset()
    assert "?" not in sent["search"]


def test_a_rejected_query_does_not_pause_the_source():
    """One malformed query is not an outage: the next request must still go out."""
    _reset()

    class Resp:
        status_code = 400
        headers = httpx.Headers({})

        def json(self):
            return {}
    saved = httpx.get
    httpx.get = lambda *a, **k: Resp()
    try:
        try:
            oa.search_works("anything at all")
            raise AssertionError("expected OpenAlexUnavailable")
        except oa.OpenAlexUnavailable as exc:
            assert "rejected" in str(exc)
    finally:
        httpx.get = saved
    assert not oa.breaker_open(), "a 400 must not trip the breaker"
    _reset()


def test_a_server_error_does_pause_the_source():
    _reset()

    class Resp:
        status_code = 503
        headers = httpx.Headers({})

        def raise_for_status(self):
            raise httpx.HTTPStatusError("503", request=None, response=None)
    saved = httpx.get
    httpx.get = lambda *a, **k: Resp()
    try:
        try:
            oa.search_works("anything else")
        except oa.OpenAlexUnavailable:
            pass
    finally:
        httpx.get = saved
    assert oa.breaker_open(), "a real outage should still back off"
    _reset()


def test_fast_source_is_not_held_for_a_slow_one():
    """OpenAlex answered; arXiv needs 3 s. Return after the grace, not the 3 s."""
    def slow_arxiv(q, k, d):
        time.sleep(3.0)
        return [_live("9", "Late arXiv paper", "arxiv")]
    fake = _fake_pipeline([], [_live("openalex:W1", "Fast OpenAlex paper", "openalex")])
    fake._live_arxiv = slow_arxiv
    fake.cfg = {"citation": {"live_budget_s": 5.0, "live_grace_s": 0.3,
                             "live_sources": ["arxiv", "openalex"]}}
    t0 = time.time()
    out = MLPipeline.search_live(fake, "q")
    assert time.time() - t0 < 1.5, "should not wait out the slow source"
    assert [p["paper_id"] for p in out] == ["openalex:W1"]


def test_empty_source_does_not_cut_the_other_short():
    """A source that returns nothing must not start the grace clock."""
    def slow_arxiv(q, k, d):
        time.sleep(0.8)
        return [_live("9", "Late but useful", "arxiv")]
    fake = _fake_pipeline([], [])
    fake._live_arxiv = slow_arxiv
    fake.cfg = {"citation": {"live_budget_s": 3.0, "live_grace_s": 0.1,
                             "live_sources": ["arxiv", "openalex"]}}
    assert [p["paper_id"] for p in MLPipeline.search_live(fake, "q")] == ["9"]


# -------------------------------------------------------------- citations ----
def test_citation_shapes_regression():
    """These patterns once held literal backspaces instead of \\b and matched nothing."""
    for text in ("Assran et al., 2023", "In NeurIPS", "arXiv preprint", "10.1109/CVPR.2016.90"):
        assert citation.looks_like_a_citation(text), text
    for text in ("MLP block of Transformers", "how does attention work"):
        assert not citation.looks_like_a_citation(text), text


def _patch(name, fn):
    saved = getattr(citation.openalex, name)
    setattr(citation.openalex, name, fn)
    return saved


def test_doi_beats_an_arxiv_lookalike_inside_it():
    """10.1145/3292500.3330701 contains '2500.33307', which is not an arXiv id."""
    seen = {}
    saved_doi = _patch("get_by_doi", lambda d: seen.setdefault("doi", d) and oa.parse_work(
        dict(WORK, doi=f"https://doi.org/{d}", title="Optuna", locations=[])))
    saved_fetch = citation._safe_fetch
    citation._safe_fetch = lambda *a, **k: (_ for _ in ()).throw(AssertionError("no arXiv call"))
    try:
        r = citation.resolve("https://doi.org/10.1145/3292500.3330701", {},
                             fetch_full_text=False)
        assert r.method == "doi" and r.paper.title == "Optuna"
        assert seen["doi"] == "10.1145/3292500.3330701"
        assert citation.paper_url(r.paper) == "https://doi.org/10.1145/3292500.3330701"
    finally:
        _patch("get_by_doi", saved_doi)
        citation._safe_fetch = saved_fetch


def test_openalex_title_fallback_when_arxiv_has_nothing():
    saved_title = _patch("search_title", lambda t, n: [oa.parse_work(
        dict(WORK, doi="https://doi.org/10.1/x", locations=[],
             title="Semi-Supervised Classification with Graph Convolutional Networks"))])
    saved_fetch = citation._safe_fetch
    citation._safe_fetch = lambda *a, **k: []
    try:
        r = citation.resolve("Semi-Supervised Classification with Graph Convolutional Networks",
                             {}, fetch_full_text=False)
        assert r.method == "openalex_title", r.method
    finally:
        _patch("search_title", saved_title)
        citation._safe_fetch = saved_fetch


def test_weak_openalex_title_match_is_not_accepted():
    saved_title = _patch("search_title", lambda t, n: [oa.parse_work(
        dict(WORK, doi="https://doi.org/10.1/x", locations=[], title="Something Else Entirely"))])
    saved_fetch = citation._safe_fetch
    citation._safe_fetch = lambda *a, **k: []
    try:
        r = citation.resolve("Semi-Supervised Classification with Graph Convolutional Networks",
                             {}, fetch_full_text=False)
        assert r.paper is None and r.method == "unresolved"
    finally:
        _patch("search_title", saved_title)
        citation._safe_fetch = saved_fetch


def test_title_index_matches_full_scan_exactly():
    """The pruned index must return what scoring every title would."""
    import random

    rng = random.Random(1)
    vocab = ["attention", "graph", "neural", "network", "diffusion", "policy", "learning",
             "robust", "sparse", "expert", "vision", "language", "model", "reward", "token"]
    papers = {str(i): {"paper_id": str(i), "title": " ".join(rng.sample(vocab, rng.randint(2, 7)))}
              for i in range(400)}
    idf = citation._build_idf(papers)
    default = max(idf.values())
    index = citation.TitleIndex(papers)
    for _ in range(200):
        q = " ".join(rng.sample(vocab, rng.randint(1, 5)))
        brute = sorted(((citation._weighted_coverage(q, m["title"], idf, default), pid)
                        for pid, m in papers.items()), reverse=True)
        brute = {pid: round(s, 9) for s, pid in brute if s > 0.2}
        fast = {m["paper_id"]: round(s, 9) for s, m in index.search(q, floor=0.2) if s > 0.2}
        assert fast == brute, q


def test_tied_local_titles_are_not_accepted():
    """A cited paper absent from the corpus must not resolve to a look-alike.

    Regression: at 123k papers, I-JEPA's title tied three JEPA papers at 0.82.
    """
    papers = {
        "a": {"paper_id": "a", "title": "Connecting Joint-Embedding Predictive Architecture with Contrastive Self-supervised Learning"},
        "b": {"paper_id": "b", "title": "3D-JEPA: A Joint Embedding Predictive Architecture for 3D Self-Supervised Representation Learning"},
        "c": {"paper_id": "c", "title": "Joint-Embedding Predictive Architecture for Self-Supervised Learning of Mask Classification Architecture"},
        **{str(i): {"paper_id": str(i), "title": f"Unrelated paper on topic {i} with graphs"} for i in range(50)},
    }
    citation._TITLE_INDEX.clear()
    match, candidates = citation.find_in_corpus(
        "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture", papers)
    assert match is None, match
    assert candidates, "look-alikes are still offered as candidates"
    match, _ = citation.find_in_corpus(
        "3D-JEPA: A Joint Embedding Predictive Architecture for 3D Self-Supervised Representation Learning",
        papers)
    assert match and match["paper_id"] == "b", "an exact title still resolves"
    citation._TITLE_INDEX.clear()


def test_arxiv_id_resolves_through_openalex_when_arxiv_is_down():
    saved_doi = _patch("get_by_doi", lambda d: oa.parse_work(dict(WORK, doi=f"https://doi.org/{d}")))
    saved_fetch = citation._safe_fetch
    citation._safe_fetch = lambda *a, **k: []          # arXiv answered nothing
    try:
        r = citation.resolve("arXiv:1706.03762", {}, fetch_full_text=False)
        assert r.method == "arxiv_id" and r.paper.paper_id == "1706.03762"
    finally:
        _patch("get_by_doi", saved_doi)
        citation._safe_fetch = saved_fetch


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
