"""Stage 6: evaluate the critic, reproducing the paper's ablation.

    python scripts/critic_evaluate.py --adapters base sft cja dqa --refine

Mirrors Table 3's ablation arms:

    base   no training        - the paper's "w/o Training" row, which scores
                               *worse* than using no critic at all (41.4 vs
                               44.5 F1). Reproducing that is the point: it is
                               the result that justifies the whole method.
    sft    cold-start only    - knows the schema, not the judgement
    cja    + stage 1          - conservative verdicts
    dqa    + stage 2 (gated)  - full CRITIC-R1

With `--refine`, the generator re-answers every flagged trajectory using the
critique as a hint (never an oracle - Appendix A.3), which gives the
end-to-end numbers: how many wrong answers got fixed, and how many right ones
got broken.

Memory note: critic and generator are the same base model, so the generator is
obtained by switching the LoRA adapters *off* rather than loading a second
copy. On an 8GB card that is the difference between fitting and not.
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

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from src.critic.evaluate import (  # noqa: E402
    evaluate_critic, evaluate_refinement, format_report, save_report,
)
from src.critic.local_llm import release, vram_report  # noqa: E402
from src.critic.prompts import (  # noqa: E402
    CRITIC_INSTRUCTION, CRITIC_SYSTEM, REFINE_SYSTEM,
)
from src.critic.schema import ParseError, Verdict, parse_critique  # noqa: E402
from src.critic.supervise import read_supervision  # noqa: E402
from src.critic.trajectory import read_trajectories  # noqa: E402

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)


@torch.inference_mode()
def batched_generate(model, tok, prompts: list[str], max_new_tokens: int,
                     batch_size: int = 4, temperature: float = 0.0,
                     max_input: int = 1536, label: str = "") -> list[str]:
    """Greedy by default: evaluation should not be a lottery."""
    outputs: list[str] = []
    bs, i, t0 = batch_size, 0, time.time()
    while i < len(prompts):
        chunk = prompts[i:i + bs]
        try:
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_input).to(model.device)
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=tok.pad_token_id)
            outputs.extend(tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                            skip_special_tokens=True))
            i += len(chunk)
            if i % 40 < bs:
                rate = i / max(time.time() - t0, 1e-6)
                print(f"    {label} {i}/{len(prompts)}  {rate*60:.0f}/min", flush=True)
        except torch.cuda.OutOfMemoryError:
            release()
            if bs == 1:
                outputs.append("")
                i += 1
            else:
                bs = max(1, bs // 2)
                print(f"    OOM; batch -> {bs}")
    release()
    return outputs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapters", nargs="+", default=["base", "sft", "cja", "dqa"])
    ap.add_argument("--models-dir", default="models/critic")
    ap.add_argument("--trajectories", default="data/critic/trajectories_eval.jsonl")
    ap.add_argument("--supervision", default="data/critic/supervision_eval.jsonl")
    ap.add_argument("--out", default="results/critic_eval.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--refine", action="store_true", help="also measure end-to-end refinement")
    args = ap.parse_args()

    trajectories = read_trajectories(args.trajectories)
    if args.limit:
        trajectories = trajectories[: args.limit]
    ref_locs = {}
    if Path(args.supervision).exists():
        ref_locs = {s.qid: s.critique.get("location", "")
                    for s in read_supervision(args.supervision)}
    print(f"[eval] {len(trajectories)} eval trajectories | {vram_report()}")

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    critic_prompts = [
        tok.apply_chat_template(
            [{"role": "system", "content": CRITIC_SYSTEM},
             {"role": "user", "content": f"{CRITIC_INSTRUCTION}\n\n{t.render(450)}"}],
            tokenize=False, add_generation_prompt=True)
        for t in trajectories
    ]

    report: dict = {"n": len(trajectories), "base_model": args.base, "arms": {}}

    for arm in args.adapters:
        adapter_path = Path(args.models_dir) / arm
        if arm != "base" and not adapter_path.exists():
            print(f"[eval] skipping '{arm}': {adapter_path} not found")
            continue

        print(f"\n[eval] === arm: {arm} ===")
        model = AutoModelForCausalLM.from_pretrained(
            args.base, dtype=torch.bfloat16, device_map="cuda")
        if arm != "base":
            model = PeftModel.from_pretrained(model, str(adapter_path))
        model.eval()

        critiques = batched_generate(model, tok, critic_prompts, 200,
                                     args.batch_size, label=f"{arm} critique")
        metrics = evaluate_critic(trajectories, critiques, ref_locs or None)
        arm_report: dict = {"critic": metrics.to_json()}

        refine_metrics = None
        if args.refine:
            # Refine only what the critic flagged - that is the behaviour in
            # production, and refining everything would measure a different
            # system.
            flagged: list[int] = []
            for i, text in enumerate(critiques):
                try:
                    v = parse_critique(text, strict=False).verdict
                except ParseError:
                    v = Verdict.CORRECT
                if v is Verdict.INCORRECT:
                    flagged.append(i)
            print(f"    critic flagged {len(flagged)}/{len(trajectories)}")

            refined: list[str | None] = [None] * len(trajectories)
            if flagged:
                prompts = []
                for i in flagged:
                    t = trajectories[i]
                    docs = "\n".join(f"Doc{p.index}({p.title}): {p.text[:450]}"
                                     for p in t.passages)
                    prompts.append(tok.apply_chat_template(
                        [{"role": "system", "content": REFINE_SYSTEM},
                         {"role": "user", "content":
                             f"<information>\n{docs}\n</information>\n\n"
                             f"<question>{t.question}</question>\n\n"
                             f"<previous_trajectory><think>{t.think}</think>"
                             f"<answer>{t.answer}</answer></previous_trajectory>\n\n"
                             f"<critique>{critiques[i]}</critique>\n\n"
                             "Re-solve. End with exactly one <answer>...</answer>."}],
                        tokenize=False, add_generation_prompt=True))

                # The generator is this same model with the adapters off - no
                # second set of weights.
                ctx = model.disable_adapter() if arm != "base" else torch.no_grad()
                with ctx:
                    outs = batched_generate(model, tok, prompts, 120,
                                            args.batch_size, label=f"{arm} refine")
                for i, text in zip(flagged, outs):
                    m = _ANSWER_RE.search(text)
                    refined[i] = (m.group(1).strip() if m
                                  else text.strip().split("\n")[-1].strip())
            refine_metrics = evaluate_refinement(trajectories, refined)
            arm_report["refinement"] = refine_metrics.to_json()

        report["arms"][arm] = arm_report
        print(format_report(metrics, refine_metrics, title=f"{arm}"))

        del model
        release()

    save_report(args.out, report)
    print(f"\n[eval] saved -> {args.out}")

    # Side-by-side, the shape the paper reports.
    if len(report["arms"]) > 1:
        print(f"\n{'arm':<8}{'falseAlarm':>12}{'precision':>11}{'recall':>9}"
              f"{'fmt':>7}{'F1 after':>10}{'harm':>8}")
        print("-" * 65)
        for arm, r in report["arms"].items():
            c, rf = r["critic"], r.get("refinement")
            print(f"{arm:<8}{c['false_alarm']:>11.1%}{c['precision']:>11.1%}"
                  f"{c['recall']:>9.1%}{c['valid_format_rate']:>7.0%}"
                  f"{(rf['f1_after'] if rf else 0):>10.3f}"
                  f"{(rf['harmed_of_right'] if rf else 0):>8.1%}")


if __name__ == "__main__":
    main()
