"""Structured failure diagnosis for low-confidence results (ResearchOS v2).

v1's weakest moment is when its confidence gate fires. It then says "nothing
answers that" - honest, and useless: no hint of what went wrong or what to try.
v2 attaches a structured diagnosis in the format of CRITIC-R1 (arXiv
2605.29886): which retrieved passage let the search down, why, and a concrete
query to run instead.

How the two halves divide the work is the result of an experiment, not a
design preference. Training CRITIC-R1 on this hardware split the critic's two
jobs apart:

  deciding WHETHER an answer is wrong   AUC 0.447 at 1.5B - no signal
  saying WHERE it went wrong            44-46% vs 21% chance (p < 1e-6)

So the critic is never asked whether to intervene. That decision belongs to
the existing mean-of-top-5 confidence gate, which already separates in-corpus
from out-of-corpus queries cleanly (0.67-0.996 vs 0.001-0.19). The critic only
diagnoses, with the verdict supplied - the same pre-filled protocol it was
evaluated under.

In production the diagnosis comes from gpt-oss on Groq rather than the
fine-tuned 1.5B critic. Speed decides it: an API call answers in about two
seconds, where the 1.5B model would take 15-30s on the Space's CPU. On
accuracy the two are complementary rather than ranked: given the CRITIC-R1
prompt, gpt-oss-20b names a document only 17 times in 100 (it mostly blames
the answer step), but is right 82% of those times; the trained critic always
names one and is right 46%. The prompt below therefore restricts the
location to a document or the query itself. scripts/critic_localization.py
has the head-to-head; scripts/eval_v2_rescue.py measures this path end to
end. The trained critic remains the measured research artifact.

Safety rails, because this is optional and must never make v1 worse:
  * fires only when the gate does - confident results never pay for it
  * a hard wall-clock budget and a circuit breaker, as for live arXiv search
  * every diagnosis is parsed with the same strict schema the critic was
    trained on; one that fails to parse, or cites a document that was not
    shown, is suppressed rather than displayed
"""

from __future__ import annotations

import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import asdict, dataclass
from typing import Any

from src.critic.schema import ParseError, Stage, parse_critique

DEFAULT_MODEL = "openai/gpt-oss-120b"

_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="diagnose")
_LOCK = threading.Lock()
_BREAKER = {"open_until": 0.0}
BREAKER_COOLDOWN_S = 120.0

# Prompt: CRITIC-R1's Figure 7(a) evidence rules, with the verdict supplied by
# the gate and the fix constrained to a runnable query. "search: <query>" is
# the paper's own fix form (its Figure 7 example is "search: retrieve correct
# entity"); requiring it makes the fix actionable rather than advisory.
SYSTEM = """You diagnose why a research-paper search engine failed to find good evidence.

The engine already decided its evidence is too weak to answer confidently. Your job is
NOT to judge that - it is decided. Your job is to say WHERE the retrieval went wrong,
WHY, and what to search for instead.

Rules:
- Use ONLY the documents shown. Do not invent facts about papers.
- Prefer pointing at a specific document (information:DocK) when a retrieved document
  is off-topic or answers a different question than the one asked.
- Use location "search" when the query itself was the problem (too vague, wrong terms).
- The fix MUST be a new search query, written as: search: <query>
  The query should use the technical terms a paper on this topic would actually use.
- If the question is not about AI/ML research at all, use location "search", say so
  in <reason>, and write the fix as exactly: search: none"""

INSTRUCTION = """Output (STRICT, one line, exactly this order, nothing else):
<verdict>INCORRECT</verdict><location>...</location><reason>...</reason><fix>search: ...</fix>

- location: search | information:DocK
- <reason> <= 30 words, specific to the documents shown
- <fix> must start with "search:" followed by a concrete query of 3-12 words"""


@dataclass
class Diagnosis:
    """What went wrong with a weak result, and what to try next."""

    location: str
    location_title: str | None
    reason: str
    fix: str
    suggested_query: str | None
    model: str
    latency_ms: float

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _breaker_open() -> bool:
    with _LOCK:
        return time.monotonic() < _BREAKER["open_until"]


def _trip(reason: str) -> None:
    with _LOCK:
        _BREAKER["open_until"] = time.monotonic() + BREAKER_COOLDOWN_S
    print(f"[diagnose] {reason} - pausing diagnoses for {BREAKER_COOLDOWN_S:.0f}s", flush=True)


def render_documents(passages: list[dict[str, Any]], max_docs: int = 6,
                     max_chars: int = 420) -> str:
    """Doc1..DocK in the trajectory format the critic was trained on."""
    lines = []
    for i, p in enumerate(passages[:max_docs], 1):
        title = (p.get("title") or "").strip()
        body = (p.get("text") or p.get("why") or p.get("abstract") or "").strip()
        body = re.sub(r"\s+", " ", body)[:max_chars]
        lines.append(f"Doc{i}({title}): {body}")
    return "\n".join(lines) if lines else "(no documents were retrieved)"


def extract_query(fix: str, original: str) -> str | None:
    """Pull a runnable query out of a `search: ...` fix.

    Rejected when it is empty, absurdly long, or just the original question
    again - re-running the same query would be a placebo button.
    """
    m = re.match(r"^\s*search\s*:\s*(.+)$", fix or "", re.IGNORECASE | re.S)
    if not m:
        return None
    query = m.group(1).strip().strip("\"'`").strip()
    query = re.sub(r"\s+", " ", query)
    words = query.split()
    if not 2 <= len(words) <= 20:
        return None
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()  # noqa: E731
    if norm(query) == norm(original):
        return None
    return query


def validate(text: str, passages: list[dict[str, Any]], question: str,
             model: str, latency_ms: float, max_docs: int = 6) -> Diagnosis | None:
    """Turn raw model output into a Diagnosis, or None if it is not trustworthy.

    Parsing reuses the exact schema the critic was trained and evaluated on.
    A citation of a document that was never shown is the critic hallucinating
    an address, so it is suppressed rather than displayed.
    """
    try:
        c = parse_critique(text, strict=False)
    except ParseError:
        return None
    if c.location is None or not c.reason.strip():
        return None

    shown = min(len(passages), max_docs)
    title = None
    if c.location.stage is Stage.INFORMATION:
        if c.location.index is None or not 1 <= c.location.index <= shown:
            return None
        title = (passages[c.location.index - 1].get("title") or "").strip() or None
    elif c.location.stage is not Stage.SEARCH:
        # think/answer stages do not exist in a retrieval failure - the engine
        # never reasoned over evidence it judged too weak to use.
        return None

    return Diagnosis(
        location=str(c.location), location_title=title,
        reason=c.reason.strip(), fix=c.fix.strip(),
        suggested_query=extract_query(c.fix, question),
        model=model, latency_ms=round(latency_ms, 1),
    )


def _call(question: str, passages: list[dict[str, Any]], answer: str | None,
          models: list[str]) -> tuple[str, str]:
    """Try each model in order; return (text, model_used).

    A quota rejection moves straight to the next model rather than failing.
    Groq's free tier caps each model separately - gpt-oss-120b at 200,000
    tokens per DAY, which at ~1,500 tokens a diagnosis is only ~130 a day
    across every user. Measured the hard way: a benchmark exhausted it
    mid-run, after which every full-size request returned 429 for hours while
    the per-minute headers still looked healthy. Separate per-model budgets
    roughly double capacity, and the feature degrades rather than vanishing.
    """
    from groq import Groq

    client = Groq(timeout=20.0, max_retries=0)
    user = (f"{INSTRUCTION}\n\n<question>{question}</question>\n"
            f"<information>\n{render_documents(passages)}\n</information>")
    if answer:
        user += f"\n<answer>{answer[:400]}</answer>"
    last: Exception | None = None
    for model in models:
        try:
            r = client.chat.completions.create(
                model=model, temperature=0.0, max_tokens=400,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": user}],
                # gpt-oss bills hidden reasoning against max_tokens before any
                # visible output; at default effort a long prompt burned the
                # whole budget and returned four characters.
                extra_body={"reasoning_effort": "low"},
            )
            return (r.choices[0].message.content or "").strip(), model
        except Exception as exc:  # noqa: BLE001
            last = exc
            if not any(k in str(exc).lower() for k in ("429", "rate", "quota")):
                raise           # a real error, not a quota: do not mask it
    raise last or RuntimeError("no diagnosis model available")


def diagnose(question: str, passages: list[dict[str, Any]], answer: str | None = None,
             *, model: str | list[str] = DEFAULT_MODEL,
             budget_s: float = 6.0) -> Diagnosis | None:
    """Diagnose a low-confidence result. Returns None whenever it cannot help.

    Never raises: a diagnosis is an enhancement, and a failure here must leave
    the response exactly as v1 would have returned it.
    """
    if not os.environ.get("GROQ_API_KEY") or _breaker_open():
        return None
    models = [model] if isinstance(model, str) else list(model)
    started = time.perf_counter()
    future = _POOL.submit(_call, question, passages, answer, models)
    try:
        text, used = future.result(timeout=budget_s)
    except FuturesTimeout:
        _trip(f"no answer within {budget_s:.0f}s")
        return None
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).lower()
        if any(k in msg for k in ("429", "rate", "503", "overload")):
            _trip(f"{type(exc).__name__}: {str(exc)[:80]}")
        return None
    return validate(text, passages, question, used,
                    (time.perf_counter() - started) * 1000.0)
