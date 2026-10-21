"""The structured critique format from CRITIC-R1 (arXiv 2605.29886).

The paper's central move is to stop treating critique as free text and treat it
as *error diagnosis* with four fields:

    <verdict>   CORRECT | INCORRECT | UNSURE
    <location>  which stage failed, e.g. information:Doc3
    <reason>    why it failed
    <fix>       what to do about it

Format is enforced strictly, and that strictness is load-bearing rather than
fussiness: the critique is consumed by a refinement step, so an unparseable
critique is worthless no matter how insightful. The paper gives invalid format
a hard negative reward (-gamma) for exactly this reason.

Appendix A.1.1 requires that all four components appear exactly once, in a
fixed order, with nothing outside the tags.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Verdict(str, Enum):
    """High-level judgement. UNSURE exists so the critic can decline.

    Abstention matters: the verdict reward (Table 7) makes UNSURE cost only
    -0.1 either way, while a false alarm costs -1.0. The critic is given a
    cheap way to say "I don't know" precisely so it does not guess.
    """

    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    UNSURE = "UNSURE"


class Stage(str, Enum):
    """Where in the RAG trajectory an error lives.

    These are the paper's four stages, which happen to be exactly the stages a
    ResearchOS query already passes through: an arXiv search, the retrieved
    passages, the model's reasoning, and the written answer.
    """

    SEARCH = "search"            # the query sent to retrieval was wrong
    INFORMATION = "information"  # a retrieved passage is irrelevant/misused
    THINK = "think"              # the reasoning over correct evidence is wrong
    ANSWER = "answer"            # evidence and reasoning fine, answer malformed


# "information:Doc3" or "think:Step2" or bare "answer".
_LOCATION_RE = re.compile(
    r"^\s*(search|information|think|answer)\s*(?::\s*(?:Doc|Step)?\s*(\d+))?\s*$",
    re.IGNORECASE,
)

_TAGS = ("verdict", "location", "reason", "fix")


@dataclass(frozen=True)
class Location:
    """An error address: which stage, and optionally which item within it."""

    stage: Stage
    index: int | None = None

    def __str__(self) -> str:
        if self.index is None:
            return self.stage.value
        label = "Doc" if self.stage is Stage.INFORMATION else "Step"
        return f"{self.stage.value}:{label}{self.index}"

    @classmethod
    def parse(cls, raw: str) -> "Location | None":
        m = _LOCATION_RE.match(raw or "")
        if not m:
            return None
        idx = int(m.group(2)) if m.group(2) is not None else None
        return cls(Stage(m.group(1).lower()), idx)


@dataclass(frozen=True)
class Critique:
    """One structured critique. `raw` keeps the text it was parsed from."""

    verdict: Verdict
    location: Location | None
    reason: str
    fix: str
    raw: str = ""

    def render(self) -> str:
        """Emit the canonical form (Figure 7b). Used for SFT targets."""
        if self.verdict is Verdict.CORRECT:
            return ("<verdict>CORRECT</verdict><location>none</location>"
                    f"<reason>{self.reason.strip() or 'Answer is consistent with the evidence.'}</reason>"
                    "<fix>keep</fix>")
        loc = str(self.location) if self.location else "answer"
        return (
            f"<verdict>{self.verdict.value}</verdict>\n"
            f"<location>{loc}</location>\n"
            f"<reason>{self.reason.strip()}</reason>\n"
            f"<fix>{self.fix.strip()}</fix>"
        )


class ParseError(ValueError):
    """Raised with a machine-readable code so rewards can distinguish causes."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}{': ' + detail if detail else ''}")
        self.code = code


def _tag_spans(text: str, tag: str) -> list[tuple[int, int, str]]:
    """All (start, end, inner) for <tag>...</tag>, non-greedy, case-insensitive."""
    return [
        (m.start(), m.end(), m.group(1))
        for m in re.finditer(rf"<{tag}>(.*?)</{tag}>", text, re.S | re.I)
    ]


def parse_critique(text: str, *, strict: bool = True) -> Critique:
    """Parse a model's output into a Critique.

    `strict=True` enforces Appendix A.1.1 exactly: each tag once, in order,
    nothing outside. That is what the format reward scores.

    `strict=False` relaxes the "nothing outside" rule only. It exists for the
    paper's "format recovery" step during supervision construction (Algorithm
    2, line 4), where a teacher's otherwise-good critique should not be thrown
    away over stray prose.

    Raises ParseError with a code: missing_tag / duplicate_tag / bad_order /
    extra_content / bad_verdict / bad_location / empty_field.
    """
    if not text:
        raise ParseError("missing_tag", "empty output")

    spans: dict[str, tuple[int, int, str]] = {}
    for tag in _TAGS:
        found = _tag_spans(text, tag)
        if not found:
            raise ParseError("missing_tag", tag)
        if len(found) > 1:
            raise ParseError("duplicate_tag", tag)
        spans[tag] = found[0]

    starts = [spans[t][0] for t in _TAGS]
    if starts != sorted(starts):
        raise ParseError("bad_order", "tags must appear verdict, location, reason, fix")

    if strict:
        # Everything outside the four tag blocks must be whitespace only.
        covered, cursor = [], 0
        for tag in _TAGS:
            s, e, _ = spans[tag]
            covered.append(text[cursor:s])
            cursor = e
        covered.append(text[cursor:])
        if any(chunk.strip() for chunk in covered):
            raise ParseError("extra_content", "text outside the structured tags")

    raw_verdict = spans["verdict"][2].strip().upper()
    try:
        verdict = Verdict(raw_verdict)
    except ValueError as exc:
        raise ParseError("bad_verdict", raw_verdict[:40]) from exc

    raw_location = spans["location"][2].strip()
    reason = spans["reason"][2].strip()
    fix = spans["fix"][2].strip()

    # Figure 7(b) couples the verdict to the other fields:
    #   CORRECT             -> location "none", fix "keep"
    #   INCORRECT / UNSURE  -> location must NOT be "none"
    # Enforced here because r_format scores exactly this contract.
    is_none = raw_location.lower() in ("", "none", "-", "n/a")
    if verdict is Verdict.CORRECT:
        if not is_none:
            raise ParseError("bad_location", "CORRECT requires location 'none'")
        location = None
    else:
        if is_none:
            raise ParseError("bad_location", f"{verdict.value} requires a location")
        location = Location.parse(raw_location)
        if location is None:
            raise ParseError("bad_location", raw_location[:40])
        if not reason:
            raise ParseError("empty_field", f"reason required when verdict is {verdict.value}")

    return Critique(verdict=verdict, location=location, reason=reason, fix=fix, raw=text)


def is_valid_format(text: str) -> bool:
    """Cheap boolean form of parse_critique, for the format reward."""
    try:
        parse_critique(text, strict=True)
        return True
    except ParseError:
        return False
