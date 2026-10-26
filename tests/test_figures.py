"""Tests for pulling figures out of arXiv's HTML renderings.

Offline: the HTML is hand-written to mirror arXiv's LaTeXML output. What is
being checked is the extraction logic and, above all, that untrusted paper
markup cannot send a visitor's browser anywhere but arXiv.

Run: `python tests/test_figures.py`
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.data.figures as fg  # noqa: E402

BASE = "https://arxiv.org/html/2401.00001"

PAGE = """
<html><body>
<figure id="S1.F1" class="ltx_figure">
  <img src="2401.00001v2/Figures/arch.png" class="ltx_graphics" width="900" height="500" alt="Refer to caption">
  <figcaption class="ltx_caption"><span class="ltx_tag ltx_tag_figure">Figure 1: </span>The model
    architecture with <math alttext="N=6"><semantics><mrow><mi>N</mi><mo>=</mo><mn>6</mn></mrow>
    <annotation encoding="application/x-tex">N=6</annotation></semantics></math> layers.</figcaption>
</figure>

<figure id="S2.F2" class="ltx_figure">
  <div class="ltx_flex_figure">
    <figure class="ltx_figure ltx_figure_panel">
      <img src="a.png" class="ltx_graphics" width="300" height="200">
      <figcaption class="ltx_caption">(a) panel caption that must not become the figure caption</figcaption>
    </figure>
    <figure class="ltx_figure ltx_figure_panel">
      <object type="image/svg+xml" data="b.svg" class="ltx_graphics"></object>
    </figure>
  </div>
  <figcaption class="ltx_caption"><span class="ltx_tag ltx_tag_figure">Figure 2: </span>Attention
    patterns, left and right.</figcaption>
</figure>

<figure id="S3.T1" class="ltx_table"><img src="table.png" class="ltx_graphics" width="400" height="200">
  <figcaption class="ltx_caption">Table 1: results</figcaption></figure>

<figure class="ltx_float ltx_float_algorithm ltx_figure"><img src="algo.png" class="ltx_graphics" width="400" height="200">
  <figcaption>Algorithm 1</figcaption></figure>

<figure id="S4.F3" class="ltx_figure">
  <img src="icon.png" class="ltx_graphics" width="24" height="24">
  <figcaption class="ltx_caption">Figure 3: only an icon</figcaption>
</figure>

<figure id="S5.F4" class="ltx_figure">
  <img src="https://evil.example.com/track.png" class="ltx_graphics" width="400" height="300">
  <img src="http://arxiv.org/insecure.png" class="ltx_graphics" width="400" height="300">
  <img src="javascript:alert(1)" class="ltx_graphics" width="400" height="300">
  <img src="data:image/png;base64,AAAA" class="ltx_graphics" width="400" height="300">
  <figcaption class="ltx_caption">Figure 4: every image here is hostile</figcaption>
</figure>

<figure id="S6.F5" class="ltx_figure">
  <img src="ok.png" class="ltx_graphics" width="400" height="300">
  <figcaption class="ltx_caption">Figure 5: <b>bold</b> text and <script>alert(1)</script> markup</figcaption>
</figure>
</body></html>
"""


def test_figures_are_extracted_in_order_with_labels_and_resolved_urls():
    figs = fg.extract_figures(PAGE, BASE)
    assert [f["label"] for f in figs] == ["Figure 1", "Figure 2", "Figure 5"], figs
    assert figs[0]["images"] == ["https://arxiv.org/html/2401.00001v2/Figures/arch.png"]
    assert figs[0]["number"] == 1


def test_equations_in_captions_become_latex():
    cap = fg.extract_figures(PAGE, BASE)[0]["caption"]
    assert cap == "The model architecture with $N=6$ layers.", cap


def test_subfigure_panels_belong_to_the_parent_and_do_not_steal_its_caption():
    fig = fg.extract_figures(PAGE, BASE)[1]
    # resolved like a browser would: a relative "a.png" on /html/2401.00001 is /html/a.png
    assert fig["images"] == ["https://arxiv.org/html/a.png",
                             "https://arxiv.org/html/b.svg"], "img and svg <object> both kept"
    assert fig["caption"] == "Attention patterns, left and right."


def test_tables_algorithms_and_icons_are_not_figures():
    labels = [f["label"] for f in fg.extract_figures(PAGE, BASE)]
    assert "Figure 3" not in labels, "an icon-only figure has nothing to show"
    assert all("Table" not in f["caption"] and "Algorithm" not in f["caption"]
               for f in fg.extract_figures(PAGE, BASE))


def test_images_off_arxiv_or_not_https_are_dropped():
    """Paper HTML is untrusted: it must not aim a visitor's browser anywhere else."""
    for src in ("https://evil.example.com/x.png", "http://arxiv.org/x.png",
                "javascript:alert(1)", "data:image/png;base64,AAAA",
                "https://arxiv.org.evil.example.com/x.png", "//evil.example.com/x.png"):
        assert fg.safe_image_url(src, BASE) is None, src
    assert fg.safe_image_url("x.png", BASE) == "https://arxiv.org/html/x.png"
    assert fg.safe_image_url("https://ar5iv.labs.arxiv.org/a.png", BASE)
    # a figure whose every image was rejected has nothing to show
    assert "Figure 4" not in [f["label"] for f in fg.extract_figures(PAGE, BASE)]


def test_captions_are_plain_text_never_markup():
    cap = fg.extract_figures(PAGE, BASE)[2]["caption"]
    assert "<b>" not in cap and "</b>" not in cap, "tags are stripped, not passed through"
    assert "bold" in cap


def test_garbage_html_does_not_raise():
    assert fg.extract_figures("<figure class='ltx_figure'><img src=", BASE) == []
    assert fg.extract_figures("", BASE) == []
    assert fg.extract_figures("<<<>>><figure><figure></figure>", BASE) == []


def test_id_validation():
    assert fg.valid_id("2312.00752") and fg.valid_id("2312.00752v2")
    assert fg.valid_id("1412.6980"), "pre-2015 ids have four digits after the dot"
    for bad in ("", "openalex:W1", "../../etc/passwd", "2312.00752/../x", "abc",
                "2312.0075", "1412.69801"):
        assert not fg.valid_id(bad), bad


def test_ranking_prefers_caption_overlap_then_figure_one():
    figs = [{"label": f"Figure {n}", "number": n, "caption": c, "images": ["u"]} for n, c in
            [(1, "Overview of the system"), (2, "Scaled dot-product attention and multi-head attention"),
             (3, "Training loss curves")]]
    top = fg.rank_figures(figs, "how does scaled dot-product attention work", 2)
    assert [f["number"] for f in top] == [2, 1]
    # no overlap at all: Figure 1 (the overview most papers lead with) comes first
    assert fg.rank_figures(figs, "zzz qqq", 1)[0]["number"] == 1
    assert fg.rank_figures([], "anything", 3) == []


def _clear():
    fg._CACHE.clear()
    fg._BREAKER["open_until"] = 0.0


def test_results_are_cached_and_misses_are_cached_shorter():
    _clear()
    calls = []
    saved = fg._fetch
    fg._fetch = lambda pid: calls.append(pid) or [{"label": "Figure 1", "number": 1, "caption": "", "images": ["u"]}]
    try:
        fg.get_figures("2401.00001")
        fg.get_figures("2401.00001")
        assert calls == ["2401.00001"], "second call must come from the cache"
        fg._fetch = lambda pid: calls.append(pid) or []
        fg.get_figures("2401.00002")
        miss_expiry = fg._CACHE["2401.00002"][0] - time.monotonic()
        assert miss_expiry <= fg.MISS_TTL_S + 1, "no figures today is cached for less than a hit"
    finally:
        fg._fetch = saved
        _clear()


def test_a_failure_pauses_lookups_but_a_404_does_not():
    _clear()
    saved = fg._fetch
    try:
        fg._fetch = lambda pid: []                       # what a 404 returns
        fg.get_figures("2401.00003")
        assert not fg.breaker_open(), "a paper without HTML is a normal answer"

        def boom(pid):
            raise RuntimeError("503")
        fg._fetch = boom
        assert fg.get_figures("2401.00004") == []
        assert fg.breaker_open(), "a real failure backs off"
        fg._fetch = lambda pid: (_ for _ in ()).throw(AssertionError("must not be called while paused"))
        assert fg.get_figures("2401.00005") == []
    finally:
        fg._fetch = saved
        _clear()


def test_slow_fetch_is_abandoned_within_the_budget():
    _clear()
    saved = fg._fetch
    fg._fetch = lambda pid: time.sleep(1.5) or []
    try:
        t0 = time.time()
        assert fg.get_figures("2401.00006", budget_s=0.2) == []
        assert time.time() - t0 < 1.0
    finally:
        fg._fetch = saved
        _clear()


def test_several_papers_are_fetched_in_parallel_and_bad_ids_ignored():
    _clear()
    saved = fg._fetch
    fg._fetch = lambda pid: time.sleep(0.3) or [{"label": "Figure 1", "number": 1, "caption": pid, "images": ["u"]}]
    try:
        t0 = time.time()
        out = fg.figures_for(["2401.00007", "2401.00008", "2401.00009", "openalex:W1", "../x"], "q", 2)
        assert time.time() - t0 < 0.85, "three 0.3 s fetches should overlap, not add up"
        assert set(out) == {"2401.00007", "2401.00008", "2401.00009"}
        assert all(len(v) == 1 for v in out.values())
    finally:
        fg._fetch = saved
        _clear()


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
