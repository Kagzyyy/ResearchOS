"""Tests for how answer() chooses between papers, general knowledge and a decline.

Offline: retrieval and the LLM calls are replaced. What is under test is the
routing - which path a question takes and how that path is labelled.

Run: `python tests/test_general_answer.py`
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.api.llm as llm  # noqa: E402
import src.api.ml_pipeline as mp  # noqa: E402


def _passage(pid: str, score: float) -> dict:
    return {"paper_id": pid, "doc_id": f"{pid}#0", "chunk_id": f"{pid}#0", "score": score,
            "title": f"Paper {pid}", "text": "text", "section": "abstract",
            "published": "2025-01-01", "arxiv_url": f"https://arxiv.org/abs/{pid}"}


def _pipeline(passages: list[dict]):
    fake = types.SimpleNamespace(
        cfg={"answer": {"top_k_passages": 6, "min_mean_score": 0.2, "confidence_top_n": 5,
                        "enrich_below": 0.0, "max_tokens": 100, "live_min_score": 0.3},
             "citation": {"allow_remote": False}, "diagnosis": {"enabled": False}},
        retrieve=lambda q: [dict(p) for p in passages],
        _answer_models=lambda: ["m"],
        _diagnose=lambda *a: (None, 0.0),
    )
    fake._confidence = types.MethodType(mp.MLPipeline._confidence, fake)
    return fake


def _run(passages, grounded_reply, general_reply, generate=True):
    """Run answer() with scripted model replies; return (result, calls)."""
    calls = []
    saved = (mp.generate_answer, mp.general_answer)
    mp.generate_answer = lambda *a, **k: (calls.append("grounded"), grounded_reply)[1]
    mp.general_answer = lambda *a, **k: (calls.append("general"), general_reply)[1]
    try:
        return mp.MLPipeline.answer(_pipeline(passages), "q", generate=generate), calls
    finally:
        mp.generate_answer, mp.general_answer = saved


GOOD = [_passage("1", 0.9), _passage("2", 0.8)]


def test_covered_question_is_answered_from_papers_only():
    result, calls = _run(GOOD, "Answer [1]", "SHOULD NOT BE USED")
    assert result["answer"] == "Answer [1]" and result["grounded"] is True
    assert calls == ["grounded"], "no second model call when the papers cover it"


def test_not_covered_falls_back_to_general_knowledge_and_is_labelled():
    result, calls = _run(GOOD, "NOT_COVERED", "Batch norm normalises activations.")
    assert result["source"] == "general_knowledge"
    assert result["grounded"] is False and result["confident"] is False
    assert result["answer"] == "Batch norm normalises activations."
    assert calls == ["grounded", "general"]
    assert result["citations"], "the related papers stay visible as reading"


def test_nothing_retrieved_goes_straight_to_general_with_one_call():
    result, calls = _run([], "unused", "Gradient descent minimises a loss.")
    assert result["source"] == "general_knowledge" and calls == ["general"]
    assert result["diagnosis"] is None, "an answered question needs no failure diagnosis"


def test_out_of_scope_is_declined_not_answered():
    result, _ = _run([], "unused", None)
    assert result["source"] != "general_knowledge" and result["grounded"] is True
    assert result["answer"].startswith("Nothing answers that")


def test_not_covered_and_out_of_scope_says_papers_lack_it():
    result, _ = _run(GOOD, "NOT_COVERED", None)
    assert "do not contain an answer" in result["answer"]


def test_general_failure_degrades_to_the_plain_decline():
    saved = (mp.generate_answer, mp.general_answer)
    mp.generate_answer = lambda *a, **k: "NOT_COVERED"

    def boom(*a, **k):
        raise RuntimeError("429 daily cap")
    mp.general_answer = boom
    try:
        result = mp.MLPipeline.answer(_pipeline(GOOD), "q")
    finally:
        mp.generate_answer, mp.general_answer = saved
    assert result["source"] != "general_knowledge" and "do not contain" in result["answer"]


def test_retrieval_only_mode_never_calls_a_model():
    result, calls = _run([], None, "x", generate=False)
    assert calls == [] and result["answer"].startswith("Nothing answers that")


def test_sentinel_parsing():
    assert llm.is_not_covered("NOT_COVERED")
    assert llm.is_not_covered("  not_covered\n")
    assert not llm.is_not_covered("The passages are not covered by ...")
    assert not llm.is_not_covered(None)


def test_general_answer_scope_check(monkeypatch=None):
    saved = (llm._complete, llm.is_configured)
    llm.is_configured = lambda: True
    try:
        llm._complete = lambda *a, **k: "OUT_OF_SCOPE"
        assert llm.general_answer("how to bake bread") is None
        llm._complete = lambda *a, **k: "Overfitting is when ..."
        assert llm.general_answer("what is overfitting").startswith("Overfitting")
    finally:
        llm._complete, llm.is_configured = saved


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
