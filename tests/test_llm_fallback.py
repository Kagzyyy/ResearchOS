"""Tests for answer generation surviving Groq's daily token caps.

Offline - the groq module is replaced by a stand-in that fails or answers per
model. Run: `python tests/test_llm_fallback.py`
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.api.llm as llm  # noqa: E402

DAILY = RuntimeError("Error code: 429 - Rate limit reached on tokens per day (TPD): "
                     "Limit 200000, Used 199137. Please try again in 2m32.496s.")


def _fake_groq(responses: dict, calls: list):
    class _Completions:
        def create(self, model, **kw):
            calls.append((model, kw.get("extra_body")))
            r = responses[model]
            if isinstance(r, Exception):
                raise r
            msg = types.SimpleNamespace(content=r)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=msg, finish_reason="stop")])

    class Groq:
        def __init__(self, **kw):
            self.chat = types.SimpleNamespace(completions=_Completions())

    return types.SimpleNamespace(Groq=Groq)


def _with(responses: dict, fn):
    calls: list = []
    saved = sys.modules.get("groq")
    sys.modules["groq"] = _fake_groq(responses, calls)
    try:
        return fn(), calls
    finally:
        if saved is not None:
            sys.modules["groq"] = saved
        else:
            sys.modules.pop("groq", None)


def test_daily_cap_moves_to_next_model_without_waiting():
    """A daily cap must not be retried: it would fail 4x more slowly, then error."""
    t0 = time.time()
    out, calls = _with({"a": DAILY, "b": "answer from b"},
                       lambda: llm._complete("s", "u", ["a", "b"], 100, 0.0))
    assert out == "answer from b"
    assert [m for m, _ in calls] == ["a", "b"], calls     # exactly one try on "a"
    assert time.time() - t0 < 1.0


def test_all_models_capped_raises():
    try:
        _with({"a": DAILY, "b": DAILY}, lambda: llm._complete("s", "u", ["a", "b"], 100, 0.0))
        raise AssertionError("expected the last error to propagate")
    except RuntimeError as exc:
        assert "per day" in str(exc)


def test_single_model_string_still_works():
    out, _ = _with({"a": "ok"}, lambda: llm._complete("s", "u", "a", 100, 0.0))
    assert out == "ok"


def test_non_transient_error_is_not_masked():
    try:
        _with({"a": ValueError("400 invalid request"), "b": "ok"},
              lambda: llm._complete("s", "u", ["a", "b"], 100, 0.0))
        raise AssertionError("a real error must surface, not fall through to model b")
    except ValueError:
        pass


def test_qwen_hides_reasoning_gpt_oss_untouched():
    _, calls = _with({"qwen/x": "ok"}, lambda: llm._complete("s", "u", "qwen/x", 100, 0.0))
    assert calls[0][1] == {"reasoning_format": "hidden"}
    _, calls = _with({"openai/gpt-oss-20b": "ok"},
                     lambda: llm._complete("s", "u", "openai/gpt-oss-20b", 100, 0.0))
    assert calls[0][1] == {}


def test_wait_hint_parses_minutes_and_hours():
    assert llm._wait_hint("please try again in 28.13s") == 28.13
    assert abs(llm._wait_hint("try again in 2m32.5s") - 152.5) < 1e-9
    assert llm._wait_hint("try again in 1h0m5s") == 3605.0
    assert llm._wait_hint("no hint here") is None


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
