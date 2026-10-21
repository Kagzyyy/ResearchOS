"""Tests for the CRITIC-R1 reward functions and critique schema.

Runs under pytest, or standalone: `python tests/test_critic_rewards.py`.

The point of these is that the paper's contribution is a set of pure
functions, so correctness can be established on CPU before any GPU time is
spent. Every published number (Table 7) is asserted exactly; every unpublished
hyperparameter is tested for the *property* the paper claims rather than a
value it never gives.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.critic.rewards import (  # noqa: E402
    DEFAULT, VERDICT_MATRIX, auxiliary_penalty, critique_reward, format_reward,
    keyword_coverage, location_reward, shaped, token_f1, verdict_reward,
)
from src.critic.schema import (  # noqa: E402
    Critique, Location, ParseError, Stage, Verdict, is_valid_format, parse_critique,
)

V = Verdict


def crit(verdict=V.INCORRECT, loc="information:Doc3",
         reason="Doc3 is about a different architecture than the question asks about",
         fix="Re-search restricting to papers that evaluate on ImageNet") -> Critique:
    return Critique(verdict, Location.parse(loc) if loc else None, reason, fix)


def render(verdict="INCORRECT", loc="information:Doc3",
           reason="Doc3 is about a different architecture than the question asks about",
           fix="Re-search restricting to papers that evaluate on ImageNet") -> str:
    return (f"<verdict>{verdict}</verdict>\n<location>{loc}</location>\n"
            f"<reason>{reason}</reason>\n<fix>{fix}</fix>")


# ---------------------------------------------------------------- schema ----
def test_parses_canonical_critique():
    c = parse_critique(render())
    assert c.verdict is V.INCORRECT
    assert c.location == Location(Stage.INFORMATION, 3)
    assert "different architecture" in c.reason


def test_location_forms():
    assert Location.parse("answer") == Location(Stage.ANSWER, None)
    assert Location.parse("information:Doc12") == Location(Stage.INFORMATION, 12)
    assert Location.parse("think:Step2") == Location(Stage.THINK, 2)
    assert Location.parse("INFORMATION : doc4") == Location(Stage.INFORMATION, 4)
    assert Location.parse("nonsense") is None


def test_strict_format_rules():
    """App A.1.1: each tag exactly once, fixed order, nothing outside."""
    assert is_valid_format(render())

    # missing tag
    assert not is_valid_format("<verdict>CORRECT</verdict>")
    # duplicate tag
    assert not is_valid_format(render() + "<fix>again</fix>")
    # prose outside the tags
    assert not is_valid_format("Sure! Here is my critique:\n" + render())
    # wrong order
    out_of_order = ("<location>answer</location><verdict>CORRECT</verdict>"
                    "<reason>r</reason><fix>f</fix>")
    assert not is_valid_format(out_of_order)
    # unknown verdict
    assert not is_valid_format(render(verdict="MAYBE"))


def test_correct_verdict_may_omit_location():
    c = parse_critique(render(verdict="CORRECT", loc="none", reason="", fix=""))
    assert c.verdict is V.CORRECT and c.location is None
    # but an INCORRECT verdict may not hand-wave the location
    try:
        parse_critique(render(verdict="INCORRECT", loc="none"))
        raise AssertionError("expected ParseError")
    except ParseError as e:
        assert e.code == "bad_location"


def test_non_strict_recovers_teacher_output():
    """Algorithm 2 line 4: salvage a good critique wrapped in chatter."""
    messy = "Here's my assessment:\n" + render() + "\nHope that helps!"
    assert not is_valid_format(messy)
    assert parse_critique(messy, strict=False).verdict is V.INCORRECT


# ------------------------------------------------------- Table 7 (exact) ----
def test_verdict_matrix_matches_paper_exactly():
    expected = {
        (V.CORRECT,   V.CORRECT):    0.7, (V.CORRECT,   V.INCORRECT): -1.0,
        (V.CORRECT,   V.UNSURE):    -0.1, (V.INCORRECT, V.CORRECT):   -0.3,
        (V.INCORRECT, V.INCORRECT):  0.5, (V.INCORRECT, V.UNSURE):    -0.1,
        (V.UNSURE,    V.CORRECT):    0.1, (V.UNSURE,    V.INCORRECT): -0.2,
        (V.UNSURE,    V.UNSURE):     0.0,
    }
    for (truth, pred), value in expected.items():
        assert verdict_reward(pred, truth) == value, f"{truth}->{pred}"
    assert len(VERDICT_MATRIX) == 3 and all(len(r) == 3 for r in VERDICT_MATRIX.values())


def test_false_alarm_punished_harder_than_miss():
    """The paper's whole thesis, as an inequality.

    Flagging a correct answer (-1.0) must hurt more than missing a real error
    (-0.3). This is what drives recall down and precision up.
    """
    false_alarm = verdict_reward(V.INCORRECT, V.CORRECT)
    miss = verdict_reward(V.CORRECT, V.INCORRECT)
    assert false_alarm < miss
    assert abs(false_alarm) / abs(miss) > 3.0


def test_abstention_is_cheap():
    """UNSURE must cost less than being wrong, so declining stays attractive."""
    for truth in (V.CORRECT, V.INCORRECT):
        wrong = V.INCORRECT if truth is V.CORRECT else V.CORRECT
        assert verdict_reward(V.UNSURE, truth) > verdict_reward(wrong, truth)
    assert verdict_reward(V.UNSURE, V.CORRECT) == -0.1


# ------------------------------------------------------------- shaping ----
def test_shaped_endpoints_and_convexity():
    """Eq 9/11: 0 -> 0, 1 -> R_max, and starves partial credit in between."""
    assert shaped(0.0, 0.6, 3.0) == 0.0
    assert abs(shaped(1.0, 0.6, 3.0) - 0.6) < 1e-9
    half = shaped(0.5, 0.6, 3.0)
    assert half < 0.5 * 0.6, "convex: half similarity must earn less than half reward"
    # monotone increasing
    vals = [shaped(s / 10, 0.6, 3.0) for s in range(11)]
    assert all(b > a for a, b in zip(vals, vals[1:]))
    # beta=0 degenerates to linear
    assert abs(shaped(0.5, 1.0, 0.0) - 0.5) < 1e-9


def test_token_f1_and_keyword_coverage():
    assert token_f1("the cat sat", "the cat sat") == 1.0
    assert token_f1("completely different", "nothing alike here") == 0.0
    assert 0.0 < token_f1("doc3 wrong architecture", "doc3 describes wrong architecture") < 1.0
    # coverage ignores stopwords and rewards naming the right things
    assert keyword_coverage("check the imagenet evaluation", "imagenet evaluation") == 1.0
    assert keyword_coverage("something else", "imagenet evaluation") == 0.0


# ------------------------------------------------------------ location ----
def test_location_reward_tiers():
    ref = crit(loc="information:Doc3")
    exact = location_reward(crit(loc="information:Doc3"), ref)
    stage_only = location_reward(crit(loc="information:Doc7"), ref)
    wrong = location_reward(crit(loc="think:Step1"), ref)
    assert exact > stage_only > wrong == 0.0
    assert exact == DEFAULT.lambda_type + DEFAULT.lambda_index
    assert stage_only == DEFAULT.lambda_type


# ----------------------------------------------------- auxiliary penalty ----
def test_generic_fixes_are_penalised():
    """App A.1.6: 'search again' with no specifics is a degenerate critique."""
    for generic in ("search again", "Re-search", "try again", "verify", "fix it"):
        assert auxiliary_penalty(crit(fix=generic)) < 0.0
    specific = crit(fix="Restrict the search to papers reporting ImageNet top-1 accuracy")
    assert auxiliary_penalty(specific) == 0.0


def test_correct_verdict_exempt_from_penalty():
    """Nothing to diagnose, so empty reason/fix is not degenerate."""
    assert auxiliary_penalty(Critique(V.CORRECT, None, "", "")) == 0.0


# ---------------------------------------------------------- the gate ----
def test_gate_closed_zeroes_the_diagnostic_layer():
    """The core of Eq 5: a perfect diagnosis of a non-existent error earns nothing."""
    reference = crit(verdict=V.CORRECT, loc=None, reason="", fix="")
    # Articulate, well-formed, correctly shaped - and wrong about there being a problem.
    bad = critique_reward(render(verdict="INCORRECT"), reference, stage=2)
    assert not bad.gate_open
    assert bad.location == 0.0 and bad.reason == 0.0 and bad.fix == 0.0
    assert bad.total == bad.format + bad.verdict


def test_gate_open_pays_for_diagnosis():
    reference = crit()
    good = critique_reward(render(), reference, stage=2)
    assert good.gate_open
    assert good.location > 0 and good.reason > 0 and good.fix > 0
    assert good.total > good.format + good.verdict


def test_gate_makes_honest_verdict_dominate():
    """A bare correct verdict must beat an eloquent wrong one.

    This is the incentive the gate creates: the model cannot buy its way out
    of a miscalibrated judgement with good prose.
    """
    reference = crit(verdict=V.CORRECT, loc=None, reason="", fix="")
    honest = critique_reward(render(verdict="CORRECT", loc="none", reason="", fix=""),
                             reference, stage=2)
    eloquent_but_wrong = critique_reward(render(verdict="INCORRECT"), reference, stage=2)
    assert honest.total > eloquent_but_wrong.total


# ----------------------------------------------------------- Algorithm 1 ----
def test_invalid_format_returns_gamma_alone():
    """Algorithm 1 lines 2-3: short-circuit, no verdict credit."""
    r = critique_reward("I think the answer looks wrong.", crit(), stage=2)
    assert r.total == -DEFAULT.gamma_format
    assert r.verdict == 0.0 and r.parse_error == "missing_tag"


def test_stage_one_ignores_diagnostics():
    """CJA stage: identical verdicts score the same regardless of diagnosis quality."""
    reference = crit()
    rich = critique_reward(render(), reference, stage=1)
    poor = critique_reward(
        render(reason="It is wrong because the document does not match the question",
               fix="Search for the correct supporting evidence instead"),
        reference, stage=1)
    assert rich.location == poor.location == 0.0
    assert rich.verdict == poor.verdict
    assert abs(rich.total - poor.total) < 1e-9


def test_stage_two_separates_diagnosis_quality():
    """DQA stage: same verdict, better diagnosis, strictly higher reward."""
    reference = crit()
    precise = critique_reward(render(), reference, stage=2)
    vague = critique_reward(
        render(loc="information:Doc9", reason="something is off", fix="search again"),
        reference, stage=2)
    assert precise.total > vague.total


def test_format_reward_signs():
    assert format_reward(render()) == DEFAULT.alpha_format
    assert format_reward("not a critique") == -DEFAULT.gamma_format


def test_reward_breakdown_is_loggable():
    d = critique_reward(render(), crit(), stage=2).as_dict()
    assert set(d) >= {"total", "format", "verdict", "location", "reason", "fix",
                      "auxiliary", "gate_open", "parse_error"}


# ------------------------------------------------------------------ main ----
if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
