"""RAG trajectories: the thing the critic audits.

A trajectory is one pass of a RAG system over a question, serialised with the
tags the paper's prompts expect:

    <question> ... </question>
    <search> ... </search>
    <information> Doc1: ... Doc2: ... </information>
    <think> ... </think>
    <answer> ... </answer>

Two choices here depart from the paper, both deliberate and both to fit an
8GB laptop instead of a cluster:

**HotpotQA in the distractor setting, not a retrieval corpus.** The paper
retrieves from HotpotWiki18 with E5-base-v2 over a FAISS index. The distractor
split ships 10 paragraphs per question - 2 genuinely supporting, 8 plausible
distractors - which is already a realistic retrieval result, complete with the
failure mode that matters (relevant-looking evidence that does not support the
answer). It removes a multi-million-passage index from the critical path and
costs nothing the critic cares about.

**Verdicts derived from gold answers, not purely from a teacher LLM.** The
paper labels every field with a consensus of teacher critiques. Groq's free
tier allows 1,000 requests per *day*, so labelling thousands of trajectories
that way is not possible here. But HotpotQA has gold answers, so the *verdict*
- the only field stage 1 (CJA) trains on - can be computed exactly and for
free. The teacher is then needed only for location/reason/fix in stage 2,
which is where the API budget goes. Ground truth is arguably better
supervision than an LLM's opinion of it, but it is a deviation and is recorded
as one.
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from src.critic.schema import Verdict

# Trajectories are graded SUCCESSFUL / FAILED / AMBIGUOUS (Figure 2b). These
# thresholds on answer F1 implement that split; the ambiguous band becomes
# UNSURE, which is exactly the case the paper wants the critic to abstain on.
F1_CORRECT = 0.80
F1_INCORRECT = 0.30


# --- answer scoring (standard SQuAD/HotpotQA normalisation) -----------------
def normalize_answer(s: str) -> str:
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def answer_f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    # yes/no/noanswer are atomic - partial credit on them is meaningless
    if set(g) <= {"yes", "no", "noanswer"} or set(p) <= {"yes", "no", "noanswer"}:
        return float(p == g)
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def verdict_from_answer(pred: str, gold: str) -> tuple[Verdict, float]:
    """Reference verdict, computed rather than asked for.

    Returns (verdict, f1). The UNSURE band is not a hedge: it marks
    trajectories where answer overlap genuinely cannot settle correctness, and
    teaching the critic to say UNSURE there is the point of having the label.
    """
    f1 = answer_f1(pred, gold)
    if f1 >= F1_CORRECT:
        return Verdict.CORRECT, f1
    if f1 <= F1_INCORRECT:
        return Verdict.INCORRECT, f1
    return Verdict.UNSURE, f1


# --- data types -------------------------------------------------------------
@dataclass
class Passage:
    """One retrieved paragraph. `index` is 1-based: Doc1 ... Doc10."""

    index: int
    title: str
    text: str
    is_gold: bool = False   # from HotpotQA supporting_facts

    def render(self) -> str:
        return f"Doc{self.index}({self.title}): {self.text}"


@dataclass
class Trajectory:
    """One RAG attempt, plus the labels derivable without an API call."""

    qid: str
    question: str
    passages: list[Passage]
    answer: str
    think: str = ""
    search: str = ""
    gold_answer: str = ""
    gold_titles: list[str] = field(default_factory=list)
    level: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    # -- derived labels, free of charge --
    @property
    def f1(self) -> float:
        return answer_f1(self.answer, self.gold_answer)

    @property
    def verdict(self) -> Verdict:
        return verdict_from_answer(self.answer, self.gold_answer)[0]

    @property
    def gold_indices(self) -> list[int]:
        return [p.index for p in self.passages if p.is_gold]

    def render(self, max_passage_chars: int = 600) -> str:
        """Serialise for the critic prompt, bounded so context stays predictable."""
        docs = "\n".join(
            f"Doc{p.index}({p.title}): {p.text[:max_passage_chars]}" for p in self.passages
        )
        parts = [f"<question>{self.question}</question>"]
        if self.search:
            parts.append(f"<search>{self.search}</search>")
        parts.append(f"<information>\n{docs}\n</information>")
        if self.think:
            parts.append(f"<think>{self.think}</think>")
        parts.append(f"<answer>{self.answer}</answer>")
        return "\n".join(parts)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["f1"] = round(self.f1, 4)
        d["reference_verdict"] = self.verdict.value
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Trajectory":
        d = dict(d)
        d.pop("f1", None)
        d.pop("reference_verdict", None)
        d["passages"] = [Passage(**p) for p in d["passages"]]
        return cls(**d)


# --- HotpotQA loading -------------------------------------------------------
def load_hotpotqa(split: str = "validation", limit: int | None = None,
                  seed: int = 42) -> Iterator[dict[str, Any]]:
    """Stream HotpotQA distractor examples, shuffled deterministically.

    Streaming keeps the 600MB train split off disk when only a few hundred
    examples are needed.
    """
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=2000)
    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            return
        yield ex


def example_to_passages(ex: dict[str, Any]) -> list[Passage]:
    """Turn HotpotQA's context into Doc1..Doc10, flagging the supporting ones."""
    ctx, sf = ex["context"], ex["supporting_facts"]
    titles = ctx["title"] if isinstance(ctx, dict) else [c[0] for c in ctx]
    sents = ctx["sentences"] if isinstance(ctx, dict) else [c[1] for c in ctx]
    gold = set(sf["title"] if isinstance(sf, dict) else [s[0] for s in sf])
    return [
        Passage(index=i + 1, title=t, text=" ".join(s).strip(), is_gold=t in gold)
        for i, (t, s) in enumerate(zip(titles, sents))
    ]


# --- persistence ------------------------------------------------------------
def write_trajectories(path: str | Path, trajectories: list[Trajectory]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for t in trajectories:
            fh.write(json.dumps(t.to_json(), ensure_ascii=False) + "\n")
    return path


def read_trajectories(path: str | Path) -> list[Trajectory]:
    with open(path, encoding="utf-8") as fh:
        return [Trajectory.from_json(json.loads(line)) for line in fh if line.strip()]


def verdict_distribution(trajectories: list[Trajectory]) -> dict[str, int]:
    """Label balance. A critic trained on 95% CORRECT learns to say CORRECT."""
    counts = Counter(t.verdict.value for t in trajectories)
    return {v.value: counts.get(v.value, 0) for v in Verdict}
