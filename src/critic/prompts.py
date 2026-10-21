"""Prompt templates, transcribed from CRITIC-R1 Figure 7 and Appendix A.3.

Kept verbatim where the paper gives wording, because the prompts encode the
same conservatism the reward matrix does - "If there is no clear contradiction
with <information>, prefer CORRECT" is CJA expressed in English. Changing them
would quietly change the behaviour the rewards are trying to train.
"""

from __future__ import annotations

from src.critic.trajectory import Trajectory

# --- Figure 7(a): system prompt for the critic ------------------------------
CRITIC_SYSTEM = """You are an external critic for a retrieval-augmented QA system.
You audit a trajectory produced by another QA model.
The trajectory may contain reasoning, search steps, retrieved information, and a final answer.

**IMPORTANT RULES**
- You MUST use <information> as the ONLY source of knowledge.
- DO NOT use your own parametric knowledge or world knowledge.
- If the <information> does not contradict the answer, you should assume the answer is acceptable.
- Do NOT invent missing facts from your own knowledge.

**Evaluation policy**
- If the reasoning and answer are consistent with <information>, judge CORRECT.
- Only judge INCORRECT if there is a clear contradiction or a clear entity mismatch.
- If the evidence is insufficient but not contradictory, judge CORRECT or UNSURE."""

# --- Figure 7(b): instruction prompt defining the output contract -----------
CRITIC_INSTRUCTION = """Input:
- QUESTION
- TRAJECTORY with tags: <think> <search> <information> <answer>

Output (STRICT, one line, in this exact order):
<verdict>...</verdict><location>...</location><reason>...</reason><fix>...</fix>

Allowed values:
- verdict: CORRECT | INCORRECT | UNSURE
- location: none | answer | information:DocK | search:stepK | think:stepK

Constraints:
- Output MUST start with "<verdict>" (no leading text).
- Keep <reason> <= 25 words.
- Keep <fix> <= 20 words.
- If verdict is CORRECT: location must be "none" and fix must be "keep".
- If verdict is INCORRECT or UNSURE: location must NOT be "none".
- If there is no clear contradiction with <information>, prefer CORRECT.

Example (format only):
<verdict>INCORRECT</verdict><location>information:Doc1</location><reason>Doc1 contradicts the final answer.</reason><fix>search: retrieve correct entity</fix>"""

# --- Appendix A.3: refinement prompt ----------------------------------------
# The critique is a *hint to verify*, never an oracle. The paper is explicit
# that this is what keeps a wrong critique from destroying a right answer.
REFINE_SYSTEM = """You are answering a question using retrieved information.

You are also given:
1. A previous trajectory from an earlier attempt.
2. An external critique of that previous trajectory.

Important rules:
- The previous trajectory may contain mistakes.
- The previous final answer may be wrong.
- The external critique may also be wrong.
- Do NOT blindly trust the previous trajectory.
- Do NOT blindly trust the critique.
- Use the critique only as a hint about possible problems to check.
- Re-solve the question with fresh reasoning instead of simply copying the previous answer.
- If the critique points out a possible issue, verify it by your own reasoning and search.
- If the critique is unsupported or mistaken, ignore it.
- Do not change your answer just because the critique suggests a change.
- Base your final answer on your own reasoning process and the retrieved information.
- You MUST end with exactly one final answer inside <answer> and </answer>."""

# --- generator prompt (ours: the paper uses a Search-R1 checkpoint) ---------
GENERATOR_SYSTEM = """You answer multi-hop questions using only the provided documents.

Rules:
- Use ONLY the documents given. Do not use outside knowledge.
- Think briefly, then answer.
- The answer must be as short as possible: a name, a date, a number, or yes/no.

Output format, exactly:
<think>one or two sentences of reasoning</think><answer>short answer</answer>"""


def critic_prompt(trajectory: Trajectory, max_passage_chars: int = 600) -> list[dict[str, str]]:
    """Chat messages asking for a structured critique of one trajectory."""
    return [
        {"role": "system", "content": CRITIC_SYSTEM},
        {"role": "user", "content": f"{CRITIC_INSTRUCTION}\n\n"
                                    f"{trajectory.render(max_passage_chars)}"},
    ]


def generator_prompt(question: str, passages, max_passage_chars: int = 600) -> list[dict[str, str]]:
    """Chat messages asking for an answer, producing a critiquable trajectory."""
    docs = "\n".join(f"Doc{p.index}({p.title}): {p.text[:max_passage_chars]}" for p in passages)
    return [
        {"role": "system", "content": GENERATOR_SYSTEM},
        {"role": "user", "content": f"<information>\n{docs}\n</information>\n\n"
                                    f"<question>{question}</question>"},
    ]


def refine_prompt(trajectory: Trajectory, critique_text: str,
                  max_passage_chars: int = 600) -> list[dict[str, str]]:
    """Chat messages asking the generator to re-solve, given a critique."""
    docs = "\n".join(
        f"Doc{p.index}({p.title}): {p.text[:max_passage_chars]}" for p in trajectory.passages
    )
    return [
        {"role": "system", "content": REFINE_SYSTEM},
        {"role": "user", "content":
            f"<information>\n{docs}\n</information>\n\n"
            f"<question>{trajectory.question}</question>\n\n"
            f"<previous_trajectory>\n<think>{trajectory.think}</think>\n"
            f"<answer>{trajectory.answer}</answer>\n</previous_trajectory>\n\n"
            f"<critique>{critique_text}</critique>\n\n"
            "Re-solve the question. End with exactly one <answer>...</answer>."},
    ]
