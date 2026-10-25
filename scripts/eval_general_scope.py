"""Does the general-knowledge fallback answer AI questions and decline the rest?

    python scripts/eval_general_scope.py

The fallback is one model call that is both the scope check and the answer
(llm.general_answer). Two ways it can be wrong, and both matter:

  * refusing an AI question   -> the product fails its stated job
  * answering a non-AI one    -> it becomes a general chatbot, spending the
                                 free-tier token budget on cooking questions

It can also fail more quietly: an answer that cites an arXiv id or URL it
cannot know. The prompt forbids that, so this counts it.

Models run in the production fallback order. Results: results/general_scope.json.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import src.config  # noqa: F401,E402  - loads .env

from calibrate_gate import OFF_TOPIC  # noqa: E402
from critic_localization import wilson  # noqa: E402
from src.api.llm import general_answer  # noqa: E402

AI_QUESTIONS = [
    "What is backpropagation and why does it work?",
    "Explain the bias-variance tradeoff.",
    "What is the difference between L1 and L2 regularization?",
    "How does batch normalization help training?",
    "What is the vanishing gradient problem?",
    "Explain how self-attention works in a transformer.",
    "What is the difference between supervised, unsupervised and reinforcement learning?",
    "How does Q-learning update its value estimates?",
    "What is a convolutional neural network and why is it good for images?",
    "What is overfitting and how can it be prevented?",
    "Explain the Adam optimizer.",
    "What is the difference between precision and recall?",
    "How do generative adversarial networks work?",
    "What is retrieval-augmented generation?",
    "What is RLHF?",
    "What is the curse of dimensionality?",
    "How does k-means clustering work?",
    "What is transfer learning?",
    "Explain gradient boosting.",
    "What is the kernel trick in support vector machines?",
    "What is KL divergence and where is it used in machine learning?",
    "How do diffusion models generate images?",
    "What are embeddings in machine learning?",
    "What is the difference between bagging and boosting?",
    "How should I split data to avoid leakage when training a model?",
    "What is tokenization in language models?",
    "What are the main risks of deploying large language models, and how are they mitigated?",
    "How do I choose a learning rate schedule?",
    "What is the Bellman equation?",
    "What is a mixture-of-experts model?",
]
# Reported, not scored: reasonable people disagree about whether these are "AI".
BORDERLINE = [
    "How do I write a SQL join?",
    "What is a Python decorator?",
    "Explain big-O notation.",
    "Is it ethical for AI to replace workers?",
    "What is the current stock price of Nvidia?",
]
MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
CITATION = re.compile(r"arxiv\.org|arXiv:\s*\d{4}\.\d{4,5}|\b\d{4}\.\d{4,5}\b|https?://", re.IGNORECASE)


def ask(question: str) -> str | None:
    return general_answer(question, model=MODELS, max_tokens=1300)


def main() -> None:
    out = Path("results/general_scope.json")
    cache_path = out.with_suffix(".cache.json")
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    def run(questions: list[str]) -> dict[str, str | None]:
        for i, q in enumerate(questions, 1):
            if q not in cache:
                cache[q] = ask(q)
                cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                print(f"    {i}/{len(questions)} {'answered' if cache[q] else 'declined'}: {q[:55]}",
                      flush=True)
        return {q: cache[q] for q in questions}

    ai, off, border = run(AI_QUESTIONS), run(OFF_TOPIC), run(BORDERLINE)
    answered = sum(1 for a in ai.values() if a)
    wrongly = sum(1 for a in off.values() if a)
    cited = [q for q, a in {**ai, **border}.items() if a and CITATION.search(a)]
    lengths = sorted(len(a.split()) for a in ai.values() if a)
    summary = {
        "ai_questions": len(ai), "ai_answered": answered,
        "ai_answered_rate": round(answered / len(ai), 4),
        "ai_answered_ci95": [round(x, 4) for x in wilson(answered, len(ai))],
        "off_topic": len(off), "off_topic_answered": wrongly,
        "off_topic_declined_rate": round(1 - wrongly / len(off), 4),
        "off_topic_declined_ci95": [round(x, 4) for x in wilson(len(off) - wrongly, len(off))],
        "answers_with_a_url_or_arxiv_id": len(cited),
        "median_answer_words": lengths[len(lengths) // 2] if lengths else 0,
        "borderline": {q: bool(a) for q, a in border.items()},
        "off_topic_wrongly_answered": [q for q, a in off.items() if a],
        "ai_wrongly_declined": [q for q, a in ai.items() if not a],
        "flagged_citations": cited,
    }
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
