"""CJA and DQA reward functions from CRITIC-R1 (arXiv 2605.29886, Sec 3.2, App A.1).

Two rewards with different jobs:

**CJA (Conservative Judgement Alignment)** teaches the critic *when not to
speak*. Its weight is the verdict matrix (Table 7), where calling a correct
answer wrong costs -1.0 while missing a real error costs only -0.3 - a false
alarm is punished 3.3x harder than a miss. That asymmetry is the entire
anti-over-intervention mechanism, and it is why the paper's recall halves
(33.1% -> 17.7%) and they report it as a success.

**DQA (Diagnostic Quality Alignment)** scores the usefulness of the diagnosis:
localisation, explanation, and fix.

**The gate** is the part worth understanding. In stage 2 the whole diagnostic
layer is multiplied by an indicator on the verdict being right:

    R = r_format + r_verdict + 1[v_hat == v_star] * (r_DQA + r_aux)

Without it, a model is rewarded for fluent, well-localised explanations of
errors that do not exist - it learns to *sound* diagnostic. The gate forces an
order of operations: be right about whether there is a problem, and only then
earn credit for describing it.

Everything here is a pure function of two strings, so all of it is unit-tested
on CPU. Hyperparameters the paper does not publish are marked below.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from src.critic.schema import Critique, ParseError, Stage, Verdict, parse_critique

# --- Table 7: verdict reward matrix -----------------------------------------
# Rows are ground truth, columns are the critic's prediction. Published values.
#
#                      predicted CORRECT  INCORRECT  UNSURE
VERDICT_MATRIX: dict[Verdict, dict[Verdict, float]] = {
    Verdict.CORRECT:   {Verdict.CORRECT: 0.7, Verdict.INCORRECT: -1.0, Verdict.UNSURE: -0.1},
    Verdict.INCORRECT: {Verdict.CORRECT: -0.3, Verdict.INCORRECT: 0.5, Verdict.UNSURE: -0.1},
    Verdict.UNSURE:    {Verdict.CORRECT: 0.1, Verdict.INCORRECT: -0.2, Verdict.UNSURE: 0.0},
}

# Fixes that name nothing. Appendix A.1.6 calls these out specifically:
# "simply suggesting to 'search again' without specifying what evidence or
# constraint should be checked".
_GENERIC_FIX_RE = re.compile(
    r"^\W*(?:please\s+)?(?:re[- ]?search|search\s+again|try\s+again|look\s+again|"
    r"retry|redo|check\s+again|verify|fix\s+(?:it|this)|correct\s+(?:it|this)|"
    r"improve\s+(?:it|this)|use\s+better\s+(?:evidence|sources?))\W*$",
    re.IGNORECASE,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class RewardConfig:
    """Reward hyperparameters.

    Values the paper publishes: the verdict matrix (Table 7). Everything else
    (alpha, gamma, lambdas, R_max, betas) is named in the paper but left
    unspecified, so these are our choices, picked so that a perfect stage-2
    critique scores about +3.0 and a format violation about -1.0 - keeping the
    diagnostic layer meaningful without letting it outweigh the verdict.
    """

    # Format (Eq 1 / Eq 6)
    alpha_format: float = 0.2       # small bonus for valid structure
    gamma_format: float = 1.0       # heavy penalty, returned alone on violation

    # Location (Eq 8)
    lambda_type: float = 0.4        # right stage
    lambda_index: float = 0.3       # right stage AND right index

    # Reason (Eq 9) and Fix (Eq 10, 11)
    reason_max: float = 0.6
    reason_beta: float = 3.0
    fix_max: float = 0.6
    fix_beta: float = 3.0
    fix_alpha: float = 0.5          # F1 vs keyword-coverage blend

    # Auxiliary penalties (App A.1.6)
    penalty_generic_fix: float = 0.3
    penalty_short_field: float = 0.2
    min_reason_chars: int = 25
    min_fix_chars: int = 15

    stopwords: frozenset[str] = field(
        default_factory=lambda: frozenset(
            "the a an and or but is are was were be been being to of in on at for "
            "with from by as it its this that these those there here which who whom "
            "not no do does did has have had will would should could can may might".split()
        )
    )


DEFAULT = RewardConfig()


# --- text similarity --------------------------------------------------------
def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def token_f1(pred: str, ref: str) -> float:
    """Token-level F1 over multisets - the paper's `s_reason` and `s_F1_fix`.

    Multiset (Counter) intersection, not set, so repeated terms count the way
    SQuAD-style F1 counts them.
    """
    p, r = Counter(tokenize(pred)), Counter(tokenize(ref))
    if not p or not r:
        return 0.0
    overlap = sum((p & r).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / sum(p.values()), overlap / sum(r.values())
    return 2 * precision * recall / (precision + recall)


def keyword_coverage(pred: str, ref: str, cfg: RewardConfig = DEFAULT) -> float:
    """Fraction of the reference's content words that appear in the prediction.

    Paired with F1 in `s_fix` because a fix can be worded completely
    differently and still name the right thing to check - coverage catches
    that where F1 punishes it.
    """
    ref_kw = {t for t in tokenize(ref) if t not in cfg.stopwords and len(t) > 2}
    if not ref_kw:
        return 0.0
    pred_tokens = set(tokenize(pred))
    return len(ref_kw & pred_tokens) / len(ref_kw)


def shaped(score: float, max_reward: float, beta: float) -> float:
    """The paper's normalised exponential (Eq 9 / Eq 11).

        r = R_max * (exp(beta*s) - 1) / (exp(beta) - 1)

    Maps s in [0,1] onto [0, R_max], but *convexly*: at beta=3, s=0.3 earns
    only 8% of R_max, s=0.5 earns 18%, and s=0.9 earns 73%. That shape deliberately starves
    partial credit, so vaguely-related waffle scores near zero instead of
    collecting half marks.
    """
    s = min(max(score, 0.0), 1.0)
    if beta <= 0:
        return max_reward * s
    return max_reward * (math.exp(beta * s) - 1.0) / (math.exp(beta) - 1.0)


# --- individual reward terms ------------------------------------------------
def format_reward(text: str, cfg: RewardConfig = DEFAULT) -> float:
    """Eq 1: +alpha for a valid structure, -gamma otherwise."""
    try:
        parse_critique(text, strict=True)
        return cfg.alpha_format
    except ParseError:
        return -cfg.gamma_format


def verdict_reward(predicted: Verdict, reference: Verdict) -> float:
    """Table 7 lookup. The asymmetry here *is* the conservatism."""
    return VERDICT_MATRIX[reference][predicted]


def location_reward(pred: Critique, ref: Critique, cfg: RewardConfig = DEFAULT) -> float:
    """Eq 8: lambda_t for the right stage, plus lambda_i for the right index.

    Index credit requires the stage to be right first - "Doc3" is meaningless
    if you also thought the failure was in reasoning.
    """
    if ref.location is None:
        # Nothing to localise (reference verdict is CORRECT). Predicting a
        # location anyway is not rewarded, but is not punished here either -
        # that case is already handled by the gate.
        return 0.0
    if pred.location is None or pred.location.stage is not ref.location.stage:
        return 0.0
    r = cfg.lambda_type
    if ref.location.index is not None and pred.location.index == ref.location.index:
        r += cfg.lambda_index
    return r


def reason_reward(pred: Critique, ref: Critique, cfg: RewardConfig = DEFAULT) -> float:
    """Eq 9: shaped token-F1 between the predicted and reference explanation."""
    return shaped(token_f1(pred.reason, ref.reason), cfg.reason_max, cfg.reason_beta)


def fix_reward(pred: Critique, ref: Critique, cfg: RewardConfig = DEFAULT) -> float:
    """Eq 10 + 11: blend F1 with keyword coverage, then shape it."""
    s = (cfg.fix_alpha * token_f1(pred.fix, ref.fix)
         + (1.0 - cfg.fix_alpha) * keyword_coverage(pred.fix, ref.fix, cfg))
    return shaped(s, cfg.fix_max, cfg.fix_beta)


def auxiliary_penalty(pred: Critique, cfg: RewardConfig = DEFAULT) -> float:
    """App A.1.6: discourage degenerate critiques. Returns a value <= 0.

    Only meaningful when the critic claims an error - a CORRECT verdict has no
    obligation to produce a detailed reason or fix.
    """
    if pred.verdict is Verdict.CORRECT:
        return 0.0
    penalty = 0.0
    if _GENERIC_FIX_RE.match(pred.fix.strip()):
        penalty -= cfg.penalty_generic_fix
    if len(pred.reason.strip()) < cfg.min_reason_chars:
        penalty -= cfg.penalty_short_field
    if len(pred.fix.strip()) < cfg.min_fix_chars:
        penalty -= cfg.penalty_short_field
    return penalty


# --- Algorithm 1: the two-stage gated reward --------------------------------
@dataclass(frozen=True)
class RewardBreakdown:
    """Per-term reward, so training logs show *why* a sample scored as it did."""

    total: float
    format: float = 0.0
    verdict: float = 0.0
    location: float = 0.0
    reason: float = 0.0
    fix: float = 0.0
    auxiliary: float = 0.0
    gate_open: bool = False
    parse_error: str | None = None

    def as_dict(self) -> dict[str, float | bool | str | None]:
        return {
            "total": round(self.total, 4), "format": round(self.format, 4),
            "verdict": round(self.verdict, 4), "location": round(self.location, 4),
            "reason": round(self.reason, 4), "fix": round(self.fix, 4),
            "auxiliary": round(self.auxiliary, 4), "gate_open": self.gate_open,
            "parse_error": self.parse_error,
        }


def critique_reward(
    generated: str,
    reference: Critique,
    *,
    stage: int = 2,
    cfg: RewardConfig = DEFAULT,
    trust_diagnosis: bool = True,
) -> RewardBreakdown:
    """Algorithm 1. `stage` 1 = CJA only, 2 = CJA + gated DQA.

    Note the early return on invalid format: the paper returns -gamma *alone*,
    not -gamma plus the other terms. An unparseable critique gets no chance to
    earn verdict credit, because nothing downstream could read it anyway.
    """
    try:
        pred = parse_critique(generated, strict=True)
    except ParseError as exc:
        return RewardBreakdown(total=-cfg.gamma_format, format=-cfg.gamma_format,
                               parse_error=exc.code)

    r_format = cfg.alpha_format
    r_verdict = verdict_reward(pred.verdict, reference.verdict)

    if stage == 1:
        # Stage 1 (CJA): structure and calibrated judgement only. The critic
        # must learn *when* to intervene before it learns how to diagnose.
        r_aux = auxiliary_penalty(pred, cfg)
        return RewardBreakdown(
            total=r_format + r_verdict + r_aux,
            format=r_format, verdict=r_verdict, auxiliary=r_aux, gate_open=False,
        )

    # Stage 2 (DQA): the gate. Diagnostic terms exist only if the verdict was right.
    gate_open = pred.verdict == reference.verdict
    if not gate_open:
        return RewardBreakdown(total=r_format + r_verdict,
                               format=r_format, verdict=r_verdict, gate_open=False)

    r_loc = location_reward(pred, reference, cfg)
    # `trust_diagnosis=False` marks a reference whose LOCATION is real (the
    # gold supporting document the generator ignored) but whose reason/fix are
    # templated boilerplate. Scoring text similarity against boilerplate would
    # train the critic to reproduce the boilerplate - a high reward for saying
    # nothing. Location is still scored, because that label is genuine, and
    # the auxiliary penalty still applies so degenerate output stays punished.
    if trust_diagnosis:
        r_reason = reason_reward(pred, reference, cfg)
        r_fix = fix_reward(pred, reference, cfg)
    else:
        r_reason = r_fix = 0.0
    r_aux = auxiliary_penalty(pred, cfg)
    return RewardBreakdown(
        total=r_format + r_verdict + r_loc + r_reason + r_fix + r_aux,
        format=r_format, verdict=r_verdict, location=r_loc, reason=r_reason,
        fix=r_fix, auxiliary=r_aux, gate_open=True,
    )
