"""Consensus-based structured critique supervision (CRITIC-R1 Algorithm 2).

The paper samples K critiques from a strong teacher per trajectory, takes the
majority verdict as the reference, keeps the critiques that agree with it, and
picks the highest-quality one as the label.

Two adaptations, both forced by a free Groq tier that allows **1,000 requests
per day** (measured from the response headers, alongside 8,000 tokens/minute):

**The reference verdict comes from gold answers, not from the vote.**
HotpotQA ships gold answers, so `v*` is computable exactly and for free. That
is strictly better supervision than an LLM's opinion of correctness, and it
means stage 1 (CJA) - which trains on verdicts alone - needs *zero* API calls.
The whole budget goes to the diagnostic fields that stage 2 needs.

**Consensus filters rather than decides.** Algorithm 2's structure is kept:
sample K, discard the samples whose verdict disagrees with `v*`, score the
survivors, keep the best. A teacher that confabulates a location for a
trajectory it misjudged is exactly what this filter removes.

The budget is tracked on disk, so a run that exhausts the daily quota stops
cleanly and resumes the next day instead of failing mid-way.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import httpx

from src.critic.prompts import CRITIC_INSTRUCTION, CRITIC_SYSTEM
from src.critic.rewards import keyword_coverage, tokenize
from src.critic.schema import Critique, Location, ParseError, Stage, Verdict, parse_critique
from src.critic.trajectory import Trajectory

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_TEACHER = "openai/gpt-oss-120b"   # the strongest free Groq model

# Measured from x-ratelimit-* headers on this account.
DAILY_REQUEST_LIMIT = 1000
TOKENS_PER_MINUTE = 8000


class BudgetExhausted(RuntimeError):
    """Raised when the daily request quota is spent. Not an error - a stop."""


@dataclass
class Budget:
    """Persistent request/token accounting across runs and days."""

    path: Path
    day: str = ""
    requests: int = 0
    tokens: int = 0
    limit: int = DAILY_REQUEST_LIMIT

    @classmethod
    def load(cls, path: str | Path, limit: int = DAILY_REQUEST_LIMIT) -> "Budget":
        path = Path(path)
        today = time.strftime("%Y-%m-%d")
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            if d.get("day") == today:
                return cls(path=path, day=today, requests=d.get("requests", 0),
                           tokens=d.get("tokens", 0), limit=limit)
        return cls(path=path, day=today, limit=limit)

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.requests)

    def spend(self, tokens: int = 0) -> None:
        if self.remaining <= 0:
            raise BudgetExhausted(
                f"daily limit of {self.limit} requests reached; resumes tomorrow")
        self.requests += 1
        self.tokens += tokens
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {"day": self.day, "requests": self.requests, "tokens": self.tokens}), encoding="utf-8")


class Teacher:
    """Groq chat client with TPM throttling and daily budget enforcement."""

    def __init__(self, model: str = DEFAULT_TEACHER, budget: Budget | None = None,
                 tpm: int = TOKENS_PER_MINUTE) -> None:
        self.model = model
        self.budget = budget
        self.tpm = tpm
        self._window: list[tuple[float, int]] = []   # (timestamp, tokens)
        key = os.environ.get("GROQ_API_KEY", "").strip()
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set")
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {key}"}, timeout=120.0)

    def _throttle(self, incoming: int) -> None:
        """Block until `incoming` tokens fit inside the trailing minute."""
        while True:
            now = time.time()
            self._window = [(t, n) for t, n in self._window if now - t < 60.0]
            used = sum(n for _, n in self._window)
            # 85% of the nominal ceiling. Running to the limit produced a
            # continuous 429 retry storm: the server's accounting window and
            # ours never line up exactly, and each 429 costs a sleep.
            if used + incoming <= self.tpm * 0.85 or not self._window:
                return
            wait = 60.0 - (now - self._window[0][0]) + 0.5
            print(f"[teacher] TPM {used}/{self.tpm}; waiting {wait:.0f}s", flush=True)
            time.sleep(max(wait, 1.0))

    def complete(self, messages: list[dict[str, str]], *, temperature: float = 0.7,
                 max_tokens: int = 400, retries: int = 3) -> str:
        """One teacher call.

        `reasoning_effort="low"` is not optional for gpt-oss. These are
        reasoning models whose hidden chain-of-thought is billed against
        max_tokens *before* any visible content: at the default effort a long
        10-document trajectory burned the entire 220-token budget on reasoning
        and returned the 4 characters "<ver". Low effort cut completion tokens
        from 121 to 49 on an identical prompt.
        """
        approx = sum(len(m["content"]) for m in messages) // 4 + max_tokens
        for attempt in range(retries):
            self._throttle(approx)
            # Budget is charged on a *completed* call, below. Charging here
            # made every 429 retry cost quota: 452 requests were spent to
            # label 130 trajectories that should have cost ~390.
            if self.budget is not None and self.budget.remaining <= 0:
                raise BudgetExhausted(
                    f"daily limit of {self.budget.limit} requests reached")
            try:
                r = self._client.post(GROQ_URL, json={
                    "model": self.model, "messages": messages,
                    "temperature": temperature, "max_tokens": max_tokens,
                    "reasoning_effort": "low"})
            except httpx.HTTPError as exc:
                if attempt == retries - 1:
                    raise
                print(f"[teacher] {type(exc).__name__}; retrying", flush=True)
                time.sleep(2 ** attempt)
                continue

            if r.status_code == 429:
                # Honour the server's own hint rather than guessing.
                hint = re.search(r"try again in ([\d.]+)s", r.text)
                wait = float(hint.group(1)) + 1.0 if hint else 20.0
                print(f"[teacher] rate limited; sleeping {wait:.0f}s", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            body = r.json()
            used = body.get("usage", {}).get("total_tokens", approx)
            self._window.append((time.time(), used))
            if self.budget is not None:
                self.budget.spend(used)
            return (body["choices"][0]["message"].get("content") or "").strip()
        return ""


# --- quality scoring (Algorithm 2, line 14) ---------------------------------
def quality_score(c: Critique, trajectory: Trajectory) -> float:
    """Intrinsic usefulness of a critique, with no reference to compare against.

    The paper scores s_keyword + s_reason + s_fix + s_location. Without a
    reference critique the proxy is specificity: does it point at a concrete
    document, mention entities actually in the question, and propose an action
    naming something checkable?
    """
    score = 0.0

    # location: concrete beats vague
    if c.location is not None:
        score += 0.5
        if c.location.index is not None:
            score += 0.5
        # a cited document that actually exists
        if (c.location.stage is Stage.INFORMATION and c.location.index is not None
                and 1 <= c.location.index <= len(trajectory.passages)):
            score += 0.5

    # reason: substantive, and grounded in the question's own terms
    words = len(c.reason.split())
    if 6 <= words <= 40:
        score += 0.5
    score += 0.8 * keyword_coverage(c.reason, trajectory.question)

    # fix: actionable, and not a restatement of the reason
    fix_words = len(c.fix.split())
    if 3 <= fix_words <= 25:
        score += 0.4
    if re.search(r"\b(doc\d+|search|compare|verify|check|restrict|entity|date|year)\b",
                 c.fix, re.I):
        score += 0.4
    if len(set(tokenize(c.fix)) - set(tokenize(c.reason))) >= 3:
        score += 0.3
    return score


@dataclass
class SupervisedCritique:
    """A trajectory paired with its reference critique and how it was produced."""

    qid: str
    reference_verdict: str
    critique: dict[str, Any]
    n_samples: int = 0
    n_agreeing: int = 0
    quality: float = 0.0
    source: str = "teacher"     # teacher | derived | fallback
    raw_samples: list[str] = field(default_factory=list)

    def to_critique(self) -> Critique:
        d = self.critique
        loc = Location.parse(d["location"]) if d.get("location") else None
        return Critique(Verdict(d["verdict"]), loc, d.get("reason", ""), d.get("fix", ""))


def _derived_label(traj: Trajectory, v_star: Verdict) -> Critique:
    """A reference critique built without the teacher.

    Used for CORRECT trajectories (where the paper's own format fixes
    location="none" and fix="keep", so there is nothing to ask about) and as
    the fallback when every teacher sample disagrees with ground truth. Costs
    no API budget.
    """
    if v_star is Verdict.CORRECT:
        return Critique(Verdict.CORRECT, None,
                        "Answer is consistent with the retrieved evidence.", "keep")
    gold = ", ".join(traj.gold_titles) or "the supporting documents"
    loc_idx = traj.gold_indices[0] if traj.gold_indices else None
    location = Location(Stage.INFORMATION, loc_idx) if loc_idx else Location(Stage.ANSWER)

    if v_star is Verdict.UNSURE:
        # The UNSURE band is mostly lexical near-misses, not contradictions -
        # e.g. gold "ponderosa pine" against predicted "Pinus ponderosa",
        # which scores F1 0.5 and lands here. Labelling that "does not match
        # the evidence" would be false, and would train the critic to call
        # paraphrases wrong. Say what is actually true: unconfirmed, not
        # contradicted.
        return Critique(
            v_star, location,
            f"The answer may paraphrase the entity in {gold}; the evidence "
            "neither clearly confirms nor contradicts it.",
            f"Check Doc{loc_idx} for the exact wording the question asks for."
            if loc_idx else "Check whether the answer uses the exact wording required.",
        )

    return Critique(
        v_star, location,
        f"The answer does not match the evidence in {gold}.",
        f"Re-read Doc{loc_idx} and extract the exact entity the question asks for."
        if loc_idx else "Extract the exact answer span the question asks for.",
    )


def supervise_trajectory(
    traj: Trajectory, teacher: Teacher, *, k: int = 3, temperature: float = 0.7,
) -> SupervisedCritique:
    """Algorithm 2 for one trajectory, with v* taken from the gold answer.

    CORRECT trajectories are labelled without any API call: Figure 7(b) fixes
    their form to location="none", fix="keep", so sampling a teacher would buy
    nothing. In a 33%-CORRECT set that alone saves a third of the budget.
    """
    v_star = traj.verdict
    if v_star is Verdict.CORRECT:
        c = _derived_label(traj, v_star)
        return SupervisedCritique(traj.qid, v_star.value, asdict(c) | {
            "location": str(c.location) if c.location else "none"},
            n_samples=0, n_agreeing=0, quality=1.0, source="derived")

    messages = [{"role": "system", "content": CRITIC_SYSTEM},
                {"role": "user", "content": f"{CRITIC_INSTRUCTION}\n\n{traj.render(500)}"}]

    samples, parsed = [], []
    for _ in range(k):
        try:
            text = teacher.complete(messages, temperature=temperature)
        except BudgetExhausted:
            raise
        samples.append(text)
        try:
            # Non-strict: salvage a good critique wrapped in chatter rather
            # than spend another request on it (Algorithm 2, line 4).
            parsed.append(parse_critique(text, strict=False))
        except ParseError:
            continue

    agreeing = [c for c in parsed if c.verdict is v_star]
    if not agreeing:
        c = _derived_label(traj, v_star)
        return SupervisedCritique(traj.qid, v_star.value, asdict(c) | {
            "location": str(c.location) if c.location else "none"},
            n_samples=len(samples), n_agreeing=0, quality=0.0,
            source="fallback", raw_samples=samples)

    best = max(agreeing, key=lambda c: quality_score(c, traj))
    return SupervisedCritique(
        traj.qid, v_star.value,
        {"verdict": best.verdict.value,
         "location": str(best.location) if best.location else "none",
         "reason": best.reason, "fix": best.fix},
        n_samples=len(samples), n_agreeing=len(agreeing),
        quality=round(quality_score(best, traj), 3), source="teacher",
    )


# The teacher is a deliberately *lenient* judge: Figure 7(a) tells it to
# assume an answer is acceptable unless the evidence contradicts it. Measured
# consequence - on 64 trajectories whose answers did not match the gold, it
# returned CORRECT 61 times and UNSURE 3 times, never INCORRECT. So it
# disagreed with ground truth on 68% of non-correct trajectories, and
# Algorithm 2's consensus filter discarded nearly all of its output.
#
# That is not the teacher failing. It is answering a different question:
# "is this contradicted by the evidence?" rather than "is this the right
# answer?" - and a wrong answer is frequently not contradicted, merely
# unsupported.
#
# So stop asking it to judge. The verdict is already known from the gold
# answer; the teacher is asked only to localise and explain, which is what it
# is actually good at and all that stage 2 (DQA) needs.
DIAGNOSIS_SYSTEM = """You diagnose errors in a retrieval-augmented QA trajectory.

You are told the correct answer and that the model's answer is wrong or unverified.
Your job is NOT to judge correctness - that is already decided. Your job is to say
WHERE the trajectory went wrong, WHY, and HOW to fix it.

Use <information> as the only source of evidence. Be specific: name the document."""

DIAGNOSIS_INSTRUCTION = """Output (STRICT, one line, in this exact order):
<verdict>{verdict}</verdict><location>...</location><reason>...</reason><fix>...</fix>

- verdict is fixed: use exactly {verdict}.
- location: answer | information:DocK | search:stepK | think:stepK
- Keep <reason> <= 25 words. Keep <fix> <= 20 words.
- Output MUST start with "<verdict>". No leading text."""


def diagnose_trajectory(traj: Trajectory, teacher: Teacher, *, k: int = 2,
                        temperature: float = 0.7) -> SupervisedCritique:
    """Verdict-conditioned supervision: the teacher localises, never judges.

    Every request yields a usable label, so the API budget buys k times more
    training data than consensus voting did at the same k.
    """
    v_star = traj.verdict
    if v_star is Verdict.CORRECT:
        c = _derived_label(traj, v_star)
        return SupervisedCritique(traj.qid, v_star.value, asdict(c) | {
            "location": str(c.location) if c.location else "none"},
            n_samples=0, n_agreeing=0, quality=1.0, source="derived")

    messages = [
        {"role": "system", "content": DIAGNOSIS_SYSTEM},
        {"role": "user", "content": "\n\n".join([
            DIAGNOSIS_INSTRUCTION.format(verdict=v_star.value),
            traj.render(320),
            f"The correct answer is: {traj.gold_answer}",
            f"The model answered: {traj.answer}",
            "Diagnose where the trajectory went wrong.",
        ])},
    ]

    parsed, samples = [], []
    for _ in range(k):
        text = teacher.complete(messages, temperature=temperature)
        samples.append(text)
        try:
            c = parse_critique(text, strict=False)
        except ParseError:
            continue
        # Keep the diagnosis, pin the verdict to ground truth - the teacher
        # occasionally ignores the instruction and emits its own.
        parsed.append(Critique(v_star, c.location, c.reason, c.fix))

    usable = [c for c in parsed if c.location is not None and c.reason.strip()]
    if not usable:
        c = _derived_label(traj, v_star)
        return SupervisedCritique(traj.qid, v_star.value, asdict(c) | {
            "location": str(c.location) if c.location else "none"},
            n_samples=len(samples), n_agreeing=0, quality=0.0,
            source="fallback", raw_samples=samples)

    best = max(usable, key=lambda c: quality_score(c, traj))
    return SupervisedCritique(
        traj.qid, v_star.value,
        {"verdict": best.verdict.value, "location": str(best.location),
         "reason": best.reason, "fix": best.fix},
        n_samples=len(samples), n_agreeing=len(usable),
        quality=round(quality_score(best, traj), 3), source="teacher")


# --- persistence ------------------------------------------------------------
def write_supervision(path: str | Path, items: Iterable[SupervisedCritique]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")
    return path


def read_supervision(path: str | Path) -> list[SupervisedCritique]:
    with open(path, encoding="utf-8") as fh:
        return [SupervisedCritique(**json.loads(line)) for line in fh if line.strip()]


def supervision_summary(items: list[SupervisedCritique]) -> dict[str, Any]:
    src = Counter(i.source for i in items)
    verd = Counter(i.reference_verdict for i in items)
    teacher_items = [i for i in items if i.source == "teacher"]
    return {
        "total": len(items),
        "by_source": dict(src),
        "by_verdict": dict(verd),
        "mean_quality": round(
            sum(i.quality for i in teacher_items) / max(len(teacher_items), 1), 3),
        "mean_agreement": round(
            sum(i.n_agreeing / max(i.n_samples, 1) for i in teacher_items)
            / max(len(teacher_items), 1), 3),
    }
