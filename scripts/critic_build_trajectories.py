"""Stage 1: produce RAG trajectories for the critic to audit.

    python scripts/critic_build_trajectories.py --limit 200 --split validation

Runs a local generator over HotpotQA (distractor setting) and records what it
did: the documents it saw, its reasoning, and its answer. Gold answers turn
each trajectory into a labelled training example for free - no API calls - by
scoring the answer and mapping it to CORRECT / UNSURE / INCORRECT.

A deliberately imperfect generator is the right tool here. The critic learns
from mistakes, so a 1.5B model answering multi-hop questions is a *feature*:
it produces the distribution of real failures (wrong entity, distractor
misuse, multi-hop shortcuts) that a strong model would not.

Resumable: re-running appends only the questions not already present.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.critic.local_llm import GenerationConfig, LocalLLM, vram_report  # noqa: E402
from src.critic.prompts import generator_prompt  # noqa: E402
from src.critic.trajectory import (  # noqa: E402
    Trajectory, example_to_passages, load_hotpotqa, read_trajectories,
    verdict_distribution, write_trajectories,
)

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S | re.I)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)


def parse_generation(text: str) -> tuple[str, str]:
    """Pull (think, answer) out of a generation, tolerating sloppy output.

    Small models drop closing tags often enough that strict parsing would
    discard usable trajectories. A malformed generation is still a perfectly
    good thing for a critic to audit - it just needs an answer to score.
    """
    think_m = _THINK_RE.search(text)
    think = think_m.group(1).strip() if think_m else ""

    # Qwen2.5-1.5B almost never emits <think> - measured ~0% across five
    # prompt variants, including one that pre-filled the opening tag. It
    # answers extractively rather than reasoning. Whatever it wrote *before*
    # the answer is still its visible working, so that becomes the think
    # field; discarding it would leave the critic with no `think` stage to
    # localise reasoning errors to at all.
    def _working(before: str) -> str:
        before = before.strip()
        return before if len(before) > 20 else think

    answer_m = _ANSWER_RE.search(text)
    if answer_m:
        return (think or _working(text[:answer_m.start()])), answer_m.group(1).strip()

    # Unclosed <answer>: take the tail.
    open_m = re.search(r"<answer>(.*)", text, re.S | re.I)
    if open_m:
        answer = open_m.group(1).strip().split("\n")[0].strip()
        return (think or _working(text[:open_m.start()])), answer

    # No tags at all: the last non-empty line is the usual answer shape, and
    # anything above it is the working.
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    if not lines:
        return think, ""
    return (think or (" ".join(lines[:-1]) if len(lines) > 1 else "")), lines[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--split", default="validation", choices=["train", "validation"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="data/critic/trajectories_val.jsonl")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    # 0.3 measured best on both answer F1 (0.302 vs 0.251 at 0.7) and verdict
    # balance; hotter sampling made answers verbose and listy, which scores as
    # INCORRECT even when the right span is present.
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-passage-chars", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    existing = read_trajectories(out) if out.exists() else []
    seen = {t.qid for t in existing}
    if existing:
        print(f"[traj] resuming: {len(existing)} already in {out}")

    print(f"[traj] loading HotpotQA {args.split} (distractor)...")
    wanted, examples = args.limit - len(existing), []
    if wanted <= 0:
        print(f"[traj] already have {len(existing)} >= limit {args.limit}; nothing to do")
        return
    for ex in load_hotpotqa(args.split, limit=args.limit * 3, seed=args.seed):
        if ex["id"] in seen:
            continue
        examples.append(ex)
        if len(examples) >= wanted:
            break
    print(f"[traj] {len(examples)} new questions to answer")

    llm = LocalLLM(args.model, max_input_tokens=2048)
    passages = [example_to_passages(ex) for ex in examples]
    prompts = [generator_prompt(ex["question"], ps, args.max_passage_chars)
               for ex, ps in zip(examples, passages)]

    started = time.time()

    def progress(done: int, total: int, elapsed: float) -> None:
        rate = done / max(elapsed, 1e-6)
        eta = (total - done) / max(rate, 1e-6)
        print(f"[traj] {done}/{total}  {rate*60:.1f}/min  ETA {eta/60:.1f} min  "
              f"{vram_report()}", flush=True)

    outputs = llm.chat(
        prompts,
        GenerationConfig(max_new_tokens=args.max_new_tokens, temperature=args.temperature),
        batch_size=args.batch_size, on_progress=progress,
    )

    new: list[Trajectory] = []
    for ex, ps, text in zip(examples, passages, outputs):
        think, answer = parse_generation(text)
        new.append(Trajectory(
            qid=ex["id"], question=ex["question"], passages=ps, answer=answer,
            think=think, search=ex["question"], gold_answer=ex["answer"],
            gold_titles=[p.title for p in ps if p.is_gold],
            level=ex.get("level", ""),
            meta={"type": ex.get("type", ""), "generator": args.model,
                  "raw_len": len(text)},
        ))

    all_traj = existing + new
    write_trajectories(out, all_traj)

    dist = verdict_distribution(all_traj)
    total = sum(dist.values()) or 1
    mean_f1 = sum(t.f1 for t in all_traj) / total
    empty = sum(1 for t in all_traj if not t.answer.strip())
    print(f"\n[traj] wrote {len(all_traj)} trajectories to {out} "
          f"in {(time.time()-started)/60:.1f} min")
    print(f"[traj] generator answer F1: {mean_f1:.3f}   empty answers: {empty}")
    print("[traj] reference verdict distribution (what the critic must learn):")
    for k, v in dist.items():
        print(f"         {k:<10} {v:4d}  ({100*v/total:5.1f}%)")
    if dist["CORRECT"] / total > 0.9:
        print("[traj] WARNING: >90% CORRECT - too easy; the critic will learn to "
              "always say CORRECT. Use a weaker generator or harder questions.")
    if dist["INCORRECT"] / total > 0.9:
        print("[traj] WARNING: >90% INCORRECT - generator may be broken; check "
              "that answers are being parsed.")


if __name__ == "__main__":
    main()
