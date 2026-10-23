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
