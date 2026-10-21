"""Tests for v2 failure diagnosis: the rails that keep it from making v1 worse.

Offline - no API calls. The model call is the one part not tested here; what
is tested is everything that decides whether its output reaches a user.

Run: `python tests/test_diagnose.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.api.diagnose as dg  # noqa: E402

PASSAGES = [
    {"title": "Attention Is All You Need", "text": "We propose the Transformer..."},
    {"title": "Sourdough Fermentation Kinetics", "text": "Lactic acid bacteria..."},
    {"title": "BERT", "text": "Bidirectional encoder representations..."},
]
Q = "how do transformers handle long documents"


def crit(loc: str, reason: str = "Doc2 is about bread fermentation, not sequence models.",
         fix: str = "search: efficient attention for long sequences") -> str:
    return (f"<verdict>INCORRECT</verdict><location>{loc}</location>"
            f"<reason>{reason}</reason><fix>{fix}</fix>")


def check(text: str):
    return dg.validate(text, PASSAGES, Q, "test-model", 1234.5)


# ------------------------------------------------------------- accepted ----
def test_valid_document_diagnosis():
    d = check(crit("information:Doc2"))
    assert d is not None
    assert d.location == "information:Doc2"
    assert d.location_title == "Sourdough Fermentation Kinetics"
    assert d.suggested_query == "efficient attention for long sequences"


def test_search_stage_diagnosis_has_no_title():
    d = check(crit("search", reason="The query is too vague for the corpus terms used."))
    assert d is not None and d.location == "search" and d.location_title is None


def test_api_model_whitespace_is_tolerated():
    """The API model sometimes wraps output in newlines; the schema still holds."""
    d = check("\n  " + crit("information:Doc1") + "\n")
    assert d is not None


# ----------------------------------------------------------- suppressed ----
def test_hallucinated_document_is_suppressed():
    """Citing Doc7 when only three were shown is the critic inventing an address."""
    assert check(crit("information:Doc7")) is None
    assert check(crit("information:Doc0")) is None


def test_unparseable_output_is_suppressed():
    assert check("The search failed because the query was vague.") is None
    assert check("") is None


def test_reasoning_stages_are_rejected_for_retrieval_failures():
    """think/answer do not exist when the engine never used the evidence."""
    assert check(crit("think:Step1")) is None
    assert check(crit("answer")) is None


def test_empty_reason_is_suppressed():
    assert check(crit("information:Doc2", reason="  ")) is None


# ------------------------------------------------------ query extraction ----
def test_query_extraction():
    assert dg.extract_query("search: sparse attention transformers", Q) == "sparse attention transformers"
    assert dg.extract_query('Search: "long context language models"', Q) == "long context language models"
    # a fix that is advice, not a query, yields no button
    assert dg.extract_query("Try being more specific.", Q) is None


def test_placebo_query_is_rejected():
    """Re-running the identical query would be a button that does nothing."""
    assert dg.extract_query(f"search: {Q.upper()}?", Q) is None


def test_degenerate_query_lengths_rejected():
    assert dg.extract_query("search: x", Q) is None
    assert dg.extract_query("search: " + "word " * 30, Q) is None


# ------------------------------------------------------- never-raises ----
def test_no_key_means_no_diagnosis_not_an_error(monkeypatch=None):
    import os
    saved = os.environ.pop("GROQ_API_KEY", None)
    try:
        assert dg.diagnose(Q, PASSAGES) is None
    finally:
        if saved is not None:
            os.environ["GROQ_API_KEY"] = saved


def test_open_breaker_short_circuits():
    import os
    import time
    os.environ.setdefault("GROQ_API_KEY", "test")
    dg._BREAKER["open_until"] = time.monotonic() + 60
    try:
        t0 = time.time()
        assert dg.diagnose(Q, PASSAGES) is None
        assert time.time() - t0 < 0.05, "an open breaker must not touch the network"
    finally:
        dg._BREAKER["open_until"] = 0.0


def _fake_groq(responses: dict):
    """A stand-in groq module: per-model either text or an exception."""
    import types

    class _Completions:
        def create(self, model, **kw):
            r = responses[model]
            if isinstance(r, Exception):
                raise r
            msg = types.SimpleNamespace(content=r)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    class Groq:
        def __init__(self, **kw):
            self.chat = types.SimpleNamespace(completions=_Completions())

    return types.SimpleNamespace(Groq=Groq)


def test_quota_on_first_model_falls_back_to_second():
    """A daily-token 429 on one model must move to the next, not kill the feature."""
    import sys as _sys
    saved = _sys.modules.get("groq")
    _sys.modules["groq"] = _fake_groq({
        "big": RuntimeError("Error 429: rate limit on tokens per day (TPD)"),
        "small": crit("information:Doc2"),
    })
    try:
        text, used = dg._call(Q, PASSAGES, None, ["big", "small"])
        assert used == "small" and "information:Doc2" in text
    finally:
        if saved is not None:
            _sys.modules["groq"] = saved
        else:
            _sys.modules.pop("groq", None)


def test_non_quota_error_is_not_masked_by_fallback():
    """A real error (bad request) must surface, not be papered over by model #2."""
    import sys as _sys
    saved = _sys.modules.get("groq")
    _sys.modules["groq"] = _fake_groq({"big": ValueError("400 invalid request"),
                                      "small": crit("information:Doc2")})
    try:
        try:
            dg._call(Q, PASSAGES, None, ["big", "small"])
            raise AssertionError("expected the ValueError to propagate")
        except ValueError:
            pass
    finally:
        if saved is not None:
            _sys.modules["groq"] = saved
        else:
            _sys.modules.pop("groq", None)


def test_document_rendering_matches_training_format():
    out = dg.render_documents(PASSAGES)
    assert out.splitlines()[0].startswith("Doc1(Attention Is All You Need):")
    assert dg.render_documents([]) == "(no documents were retrieved)"


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
