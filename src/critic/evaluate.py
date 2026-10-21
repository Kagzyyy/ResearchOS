"""Evaluation for the structured critic - the paper's own metrics.

Answer quality alone cannot tell you whether a critic works. CRITIC-R1's
central claim is about *intervention behaviour*: the trained critic's recall
drops by half (33.1% -> 17.7%) while its false alarm rate falls from 12.9% to
4.6%, and the paper presents that trade as the win. A harness that only
reported F1 would score that as a regression.

So this computes what Tables 5 and 6 report:

  precision    of the trajectories flagged INCORRECT, how many really were
  recall       of the really-incorrect trajectories, how many were flagged
  false alarm  of the really-CORRECT trajectories, how many were flagged
  corr/trig    of refinements that fired, how many fixed a wrong answer
  corr/wrong   of initially-wrong answers, how many ended up fixed
  harm rate    of initially-RIGHT answers, how many were broken by refining

`harm_rate` is ours, not the paper's. The paper argues false positives are
worse than false negatives; harm rate measures the damage directly rather than
inferring it, and it is the number that decides whether shipping the critic is
defensible at all.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from src.critic.schema import ParseError, Stage, Verdict, parse_critique
from src.critic.trajectory import Trajectory, answer_f1, verdict_from_answer


@dataclass
class CriticMetrics:
    """Intervention behaviour (Table 5) plus format health."""

    n: int = 0
    precision: float = 0.0
    recall: float = 0.0
    false_alarm: float = 0.0
    verdict_accuracy: float = 0.0
    valid_format_rate: float = 0.0
    abstention_rate: float = 0.0
    predicted: dict[str, int] = field(default_factory=dict)
    reference: dict[str, int] = field(default_factory=dict)
    verdict_confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    location_confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    location_accuracy: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _blank_confusion(labels: Sequence[str]) -> dict[str, dict[str, int]]:
    return {a: {b: 0 for b in labels} for a in labels}


def evaluate_critic(
    trajectories: Sequence[Trajectory],
    critique_texts: Sequence[str],
    reference_locations: dict[str, str] | None = None,
) -> CriticMetrics:
    """Score critiques against gold-derived verdicts.

    An unparseable critique counts as *no intervention*, not as a free pass:
    downstream it cannot trigger refinement, so treating it as CORRECT is the
    behaviour it actually produces.
    """
    labels = [v.value for v in Verdict]
    m = CriticMetrics(n=len(trajectories))
    confusion = _blank_confusion(labels)
    stages = [s.value for s in Stage]
    loc_conf = _blank_confusion(stages)

    valid = 0
    pred_counts: Counter[str] = Counter()
    ref_counts: Counter[str] = Counter()
    loc_hits = loc_total = 0

    for traj, text in zip(trajectories, critique_texts):
        ref_v = traj.verdict
        ref_counts[ref_v.value] += 1
        try:
            c = parse_critique(text, strict=True)
            valid += 1
            pred_v = c.verdict
        except ParseError:
            try:
                c = parse_critique(text, strict=False)
                pred_v = c.verdict
            except ParseError:
                c, pred_v = None, Verdict.CORRECT   # cannot trigger a refinement

        pred_counts[pred_v.value] += 1
        confusion[ref_v.value][pred_v.value] += 1

        if (reference_locations and c is not None and c.location is not None
                and traj.qid in reference_locations):
            ref_loc = reference_locations[traj.qid].split(":")[0].lower()
            if ref_loc in loc_conf:
                loc_total += 1
                loc_conf[ref_loc][c.location.stage.value] += 1
                loc_hits += int(c.location.stage.value == ref_loc)

    flagged = pred_counts[Verdict.INCORRECT.value]
    truly_wrong = ref_counts[Verdict.INCORRECT.value]
    truly_right = ref_counts[Verdict.CORRECT.value]
    tp = confusion[Verdict.INCORRECT.value][Verdict.INCORRECT.value]
    fp_on_correct = confusion[Verdict.CORRECT.value][Verdict.INCORRECT.value]

    m.precision = tp / flagged if flagged else 0.0
    m.recall = tp / truly_wrong if truly_wrong else 0.0
    m.false_alarm = fp_on_correct / truly_right if truly_right else 0.0
    m.verdict_accuracy = sum(confusion[v][v] for v in labels) / max(m.n, 1)
    m.valid_format_rate = valid / max(m.n, 1)
    m.abstention_rate = pred_counts[Verdict.UNSURE.value] / max(m.n, 1)
    m.predicted = dict(pred_counts)
    m.reference = dict(ref_counts)
    m.verdict_confusion = confusion
    m.location_confusion = loc_conf
    m.location_accuracy = loc_hits / loc_total if loc_total else 0.0
    return m


@dataclass
class RefinementMetrics:
    """End-to-end effect of critic-guided refinement (Table 6 + harm)."""

    n: int = 0
    triggered: int = 0
    trigger_rate: float = 0.0
    f1_before: float = 0.0
    f1_after: float = 0.0
    f1_delta: float = 0.0
    corrected_of_triggered: float = 0.0
    corrected_of_wrong: float = 0.0
    harmed_of_right: float = 0.0
    net_corrections: int = 0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_refinement(
    trajectories: Sequence[Trajectory],
    refined_answers: Sequence[str | None],
) -> RefinementMetrics:
    """Compare answers before and after refinement.

    `refined_answers[i]` is None when the critic did not trigger - that case
    keeps the original answer and counts toward neither correction nor harm.
    """
    m = RefinementMetrics(n=len(trajectories))
    f1_before: list[float] = []
    f1_after: list[float] = []
    corrected = harmed = wrong_before = right_before = 0

    for traj, refined in zip(trajectories, refined_answers):
        before = traj.f1
        was_right = verdict_from_answer(traj.answer, traj.gold_answer)[0] is Verdict.CORRECT
        wrong_before += (not was_right)
        right_before += was_right

        if refined is None:
            f1_before.append(before)
            f1_after.append(before)
            continue

        m.triggered += 1
        after = answer_f1(refined, traj.gold_answer)
        is_right = verdict_from_answer(refined, traj.gold_answer)[0] is Verdict.CORRECT
        f1_before.append(before)
        f1_after.append(after)
        if not was_right and is_right:
            corrected += 1
        elif was_right and not is_right:
            harmed += 1

    n = max(m.n, 1)
    m.trigger_rate = m.triggered / n
    m.f1_before = sum(f1_before) / n
    m.f1_after = sum(f1_after) / n
    m.f1_delta = m.f1_after - m.f1_before
    m.corrected_of_triggered = corrected / m.triggered if m.triggered else 0.0
    m.corrected_of_wrong = corrected / wrong_before if wrong_before else 0.0
    m.harmed_of_right = harmed / right_before if right_before else 0.0
    m.net_corrections = corrected - harmed
    return m


def format_report(critic: CriticMetrics, refine: RefinementMetrics | None = None,
                  title: str = "critic") -> str:
    """A console table shaped like the paper's, for eyeballing a run."""
    lines = [f"\n=== {title} ===",
             f"  trajectories            {critic.n}",
             f"  valid format            {critic.valid_format_rate:6.1%}",
             f"  verdict accuracy        {critic.verdict_accuracy:6.1%}",
             "",
             "  -- intervention behaviour (paper Table 5) --",
             f"  precision               {critic.precision:6.1%}",
             f"  recall                  {critic.recall:6.1%}",
             f"  FALSE ALARM             {critic.false_alarm:6.1%}   <- lower is better",
             f"  abstention (UNSURE)     {critic.abstention_rate:6.1%}",
             "",
             f"  predicted  {critic.predicted}",
             f"  reference  {critic.reference}"]
    if critic.location_accuracy:
        lines.append(f"  location stage accuracy {critic.location_accuracy:6.1%}")
    if refine is not None:
        lines += ["",
                  "  -- refinement effect (paper Table 6) --",
                  f"  triggered               {refine.triggered}/{refine.n} "
                  f"({refine.trigger_rate:.1%})",
                  f"  answer F1 before/after  {refine.f1_before:.3f} -> {refine.f1_after:.3f} "
                  f"({refine.f1_delta:+.3f})",
                  f"  corrected / triggered   {refine.corrected_of_triggered:6.1%}",
                  f"  corrected / wrong       {refine.corrected_of_wrong:6.1%}",
                  f"  HARMED / right          {refine.harmed_of_right:6.1%}   <- the risk",
                  f"  net corrections         {refine.net_corrections:+d}"]
    return "\n".join(lines)


def save_report(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
