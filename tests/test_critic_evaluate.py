"""Tests for the critic evaluation metrics.

These matter more than they look. The whole project is judged by false alarm
rate, precision, recall and harm rate, so a sign error or a swapped
denominator here would not fail loudly - it would produce a plausible,
publishable-looking table that happens to be wrong.

Run: `python tests/test_critic_evaluate.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.critic.evaluate import evaluate_critic, evaluate_refinement  # noqa: E402
from src.critic.trajectory import (  # noqa: E402
    Passage, Trajectory, answer_f1, normalize_answer, verdict_from_answer,
)
from src.critic.schema import Verdict  # noqa: E402

V = Verdict


def traj(qid: str, answer: str, gold: str) -> Trajectory:
    return Trajectory(
        qid=qid, question="q?", passages=[Passage(1, "T", "text", True)],
        answer=answer, gold_answer=gold, gold_titles=["T"])


def critique(verdict: str, loc: str = "information:Doc1") -> str:
    if verdict == "CORRECT":
        return ("<verdict>CORRECT</verdict><location>none</location>"
                "<reason>Consistent with the evidence.</reason><fix>keep</fix>")
    return (f"<verdict>{verdict}</verdict><location>{loc}</location>"
            f"<reason>The answer contradicts the cited document.</reason>"
            f"<fix>Re-read Doc1 and extract the exact entity.</fix>")


# ------------------------------------------------------- answer scoring ----
def test_normalisation_and_f1():
    assert normalize_answer("The  Beatles.") == "beatles"
    assert answer_f1("Brooklyn, New York", "Brooklyn New York") == 1.0
    assert answer_f1("Swiss", "American") == 0.0
    # yes/no are atomic: partial credit on them would be meaningless
    assert answer_f1("yes", "yes no") == 0.0


def test_verdict_bands():
    assert verdict_from_answer("Alistair Grant", "Alistair Grant")[0] is V.CORRECT
    assert verdict_from_answer("totally wrong", "Alistair Grant")[0] is V.INCORRECT
    # a lexical near-miss lands in the UNSURE band rather than being forced
    v, f1 = verdict_from_answer("Pinus ponderosa", "ponderosa pine")
    assert v is V.UNSURE and 0.3 < f1 < 0.8


# ----------------------------------------------------- critic behaviour ----
def test_perfect_critic():
    ts = [traj("a", "right", "right"), traj("b", "wrong", "right answer here")]
    cs = [critique("CORRECT"), critique("INCORRECT")]
    m = evaluate_critic(ts, cs)
    assert m.precision == 1.0 and m.recall == 1.0
    assert m.false_alarm == 0.0
    assert m.verdict_accuracy == 1.0 and m.valid_format_rate == 1.0


def test_false_alarm_is_measured_against_correct_only():
    """Flagging a right answer is the error the paper cares about most."""
    ts = [traj("a", "right", "right"),          # truly CORRECT
          traj("b", "right", "right"),          # truly CORRECT
          traj("c", "wrong", "something else")]  # truly INCORRECT
    cs = [critique("INCORRECT"), critique("CORRECT"), critique("INCORRECT")]
    m = evaluate_critic(ts, cs)
    assert m.false_alarm == 0.5      # 1 of the 2 correct answers was flagged
    assert m.precision == 0.5        # 2 flagged, 1 genuinely wrong
    assert m.recall == 1.0           # the 1 wrong one was caught


def test_unparseable_counts_as_no_intervention():
    """A critique nothing can parse cannot trigger refinement downstream."""
    ts = [traj("a", "wrong", "right answer")]
    m = evaluate_critic(ts, ["I think this looks wrong, honestly."])
    assert m.valid_format_rate == 0.0
    assert m.recall == 0.0          # it never actually flagged anything
    assert m.false_alarm == 0.0


def test_abstention_is_not_an_intervention():
    ts = [traj("a", "right", "right"), traj("b", "wrong", "other")]
    m = evaluate_critic(ts, [critique("UNSURE"), critique("UNSURE")])
    assert m.abstention_rate == 1.0
    assert m.false_alarm == 0.0, "UNSURE must not count as a false alarm"
    assert m.recall == 0.0


def test_conservative_critic_profile_matches_paper_direction():
    """A conservative critic: fewer flags, lower recall, far lower false alarm."""
    ts = ([traj(f"c{i}", "right", "right") for i in range(8)]
          + [traj(f"w{i}", "wrong", "right answer") for i in range(4)])
    aggressive = [critique("INCORRECT")] * 12
    conservative = ([critique("CORRECT")] * 8 + [critique("INCORRECT")] * 2
                    + [critique("CORRECT")] * 2)
    a, c = evaluate_critic(ts, aggressive), evaluate_critic(ts, conservative)
    assert a.recall > c.recall            # aggressive catches more
    assert c.false_alarm < a.false_alarm  # but cries wolf constantly
    assert c.precision > a.precision
    assert c.false_alarm == 0.0


# -------------------------------------------------------- refinement ----
def test_refinement_counts_fixes_and_harm():
    ts = [traj("a", "wrong", "right answer"),   # wrong, gets fixed
          traj("b", "right answer", "right answer"),  # right, gets broken
          traj("c", "wrong", "right answer")]   # wrong, stays wrong
    refined = ["right answer", "broken nonsense", "still wrong"]
    m = evaluate_refinement(ts, refined)
    assert m.triggered == 3
    assert m.corrected_of_wrong == 0.5        # 1 of 2 wrong answers fixed
    assert m.harmed_of_right == 1.0           # the 1 right answer was broken
    assert m.net_corrections == 0             # +1 fixed, -1 broken


def test_untriggered_items_are_left_alone():
    """None means the critic did not fire: no credit, no blame."""
    ts = [traj("a", "right", "right"), traj("b", "wrong", "right answer")]
    m = evaluate_refinement(ts, [None, None])
    assert m.triggered == 0 and m.trigger_rate == 0.0
    assert m.f1_before == m.f1_after
    assert m.net_corrections == 0


def test_harm_rate_detects_a_damaging_critic():
    """The number that decides whether shipping the critic is defensible."""
    ts = [traj(f"c{i}", "right answer", "right answer") for i in range(4)]
    m = evaluate_refinement(ts, ["nonsense"] * 4)
    assert m.harmed_of_right == 1.0
    assert m.net_corrections == -4
    assert m.f1_delta < 0


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
    print(f"\n{len(tests)-len(failed)}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
