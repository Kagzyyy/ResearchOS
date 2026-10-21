"""Stage 3-5: cold-start SFT, then two-stage GRPO (CRITIC-R1 Sec 3.3.2).

    python scripts/critic_train.py --phase all --steps 60

Phases, in the paper's order:

  sft     teach the output schema, so RL starts above the format cliff
  cja     GRPO stage 1 - conservative judgement: when NOT to intervene
  dqa     GRPO stage 2 - gated diagnostic quality, initialised from stage 1

Each phase saves LoRA adapters (a few MB) and a JSONL log of per-step reward
statistics, so a run can be inspected or resumed without re-training.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from src.critic.grpo import GRPOConfig, GRPOTrainer  # noqa: E402
from src.critic.local_llm import release, vram_report  # noqa: E402
from src.critic.prompts import CRITIC_INSTRUCTION, CRITIC_SYSTEM  # noqa: E402
from src.critic.rewards import critique_reward  # noqa: E402
from src.critic.schema import (  # noqa: E402
    Critique, Location, ParseError, Verdict, parse_critique,
)
from src.critic.sft import SFTConfig, train_sft  # noqa: E402
from src.critic.supervise import read_supervision  # noqa: E402
from src.critic.trajectory import read_trajectories  # noqa: E402

# Table 9: LoRA r=16, alpha=16, all linear layers.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


def build_dataset(traj_path: str, sup_path: str | None, tokenizer,
                  max_passage_chars: int = 450) -> list[dict]:
    """Pair each trajectory with its reference critique and a rendered prompt.

    When supervision is missing (stage 1 needs only verdicts, which are free),
    a reference is derived from the gold answer so CJA can still train.
    """
    trajectories = read_trajectories(traj_path)
    supervision = {s.qid: s for s in read_supervision(sup_path)} if sup_path and Path(
        sup_path).exists() else {}

    dataset = []
    for t in trajectories:
        sup = supervision.get(t.qid)
        if sup is not None:
            ref = sup.to_critique()
        elif t.verdict is Verdict.CORRECT:
            ref = Critique(Verdict.CORRECT, None,
                           "Answer is consistent with the retrieved evidence.", "keep")
        else:
            idx = t.gold_indices[0] if t.gold_indices else None
            ref = Critique(
                t.verdict,
                Location.parse(f"information:Doc{idx}") if idx else Location.parse("answer"),
                "The answer does not match the supporting evidence.",
                f"Re-read Doc{idx} and extract the exact entity requested."
                if idx else "Extract the exact answer span the question asks for.")

        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": CRITIC_SYSTEM},
             {"role": "user", "content":
                 f"{CRITIC_INSTRUCTION}\n\n{t.render(max_passage_chars)}"}],
            tokenize=False, add_generation_prompt=True)
        # Stage 2 supplies the verdict instead of asking for it.
        #
        # The paper gates DQA on the critic's own verdict being right, which
        # assumes stage 1 produced a usable one. Ours did not: measured AUC
        # for verdict discrimination is 0.447 - no signal - so the critic
        # always says CORRECT, the gate never opens on an INCORRECT
        # trajectory, and r_DQA is identically zero. Observed directly: DQA
        # sat at reward -0.100 with 75% flat groups, learning nothing.
        #
        # So the verdict is pre-filled from ground truth and the critic
        # generates only location/reason/fix. That trains the capability
        # which demonstrably works (localisation 44% vs 21% chance on 100
        # held-out wrong answers, binomial p=2e-7) instead of one that does
        # not - and it matches how this
        # would deploy in ResearchOS, where the existing mean-of-top-5
        # confidence gate already supplies a reliable verdict.
        prefix = f"<verdict>{ref.verdict.value}</verdict><location>"
        dataset.append({"qid": t.qid, "prompt": prompt, "reference": ref,
                        "target": ref.render(), "prefix": prefix,
                        "has_teacher": sup is not None
                        and sup.source == "teacher"})
    return dataset


def make_reward_fn(stage: int, prefilled: bool = False):
    """Wrap critique_reward, returning (reward, diagnostics) for logging.

    `prefilled` reattaches the verdict prefix that was appended to the prompt,
    so the reward still scores a complete, parseable critique.
    """
    def fn(meta: dict, text: str):
        ref: Critique = meta["reference"]
        if prefilled:
            text = meta["prefix"] + text
        bd = critique_reward(text, ref, stage=stage,
                             trust_diagnosis=meta.get("has_teacher", True))
        # Compare verdicts directly. Inferring a match from `bd.verdict > 0`
        # scored a correct UNSURE->UNSURE abstention as a mismatch, because
        # that cell of Table 7 is exactly 0.0 - and abstention is the very
        # behaviour CJA exists to encourage, so the metric would have hidden
        # the effect being trained for.
        try:
            verdict_match = parse_critique(text, strict=True).verdict is ref.verdict
        except ParseError:
            verdict_match = False
        return bd.total, {"parse_error": bd.parse_error, "gate_open": bd.gate_open,
                          "verdict_match": verdict_match}
    return fn


def load_model(base: str, adapter: str | None, lr_target_dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"
    # Truncate the START of an over-long prompt, never the end. The chat
    # template's "<|im_start|>assistant" cue is the last thing in the prompt,
    # and right-truncation (the default) deletes it - after which the model
    # has no signal that it is its turn and simply continues the trajectory.
    # 37% of these prompts exceed the 1536-token cap, so this was live.
    tok.truncation_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(base, dtype=lr_target_dtype,
                                                 device_map="cuda")
    if adapter and Path(adapter).exists():
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
        print(f"[train] resumed adapters from {adapter}")
    else:
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=16, lora_dropout=0.0, task_type="CAUSAL_LM",
            target_modules=LORA_TARGETS))
    # Checkpointing only engages in train() mode - a silent 3x memory cost if
    # this is forgotten, and the first VRAM probe for this project hit exactly
    # that, reporting a false OOM at batch size 1.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] {base} + LoRA: {trainable/1e6:.1f}M trainable | {vram_report()}")
    return model, tok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", default="all", choices=["all", "sft", "cja", "dqa"])
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--trajectories", default="data/critic/trajectories_train.jsonl")
    ap.add_argument("--supervision", default="data/critic/supervision_train.jsonl")
    ap.add_argument("--out", default="models/critic")
    ap.add_argument("--steps", type=int, default=60, help="GRPO steps per stage")
    ap.add_argument("--sft-epochs", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--prompts-per-step", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=200)  # 120 truncated <fix>
    ap.add_argument("--max-prompt-tokens", type=int, default=1536)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    phases = ["sft", "cja", "dqa"] if args.phase == "all" else [args.phase]
    report: dict = {"base": args.base, "phases": {}}
    started = time.time()

    # ---------------- SFT ----------------
    if "sft" in phases:
        model, tok = load_model(args.base, None)
        data = build_dataset(args.trajectories, args.supervision, tok)
        if args.limit:
            data = data[: args.limit]
        print(f"[train] SFT on {len(data)} examples")
        report["phases"]["sft"] = train_sft(
            model, tok, data, SFTConfig(epochs=args.sft_epochs), out / "sft")
        del model
        release()

    grpo_cfg = GRPOConfig(
        group_size=args.group_size, prompts_per_step=args.prompts_per_step,
        max_new_tokens=args.max_new_tokens, max_prompt_tokens=args.max_prompt_tokens,
        gen_batch=min(8, args.group_size * args.prompts_per_step),
    )

    # ---------------- GRPO stage 1: CJA ----------------
    if "cja" in phases:
        model, tok = load_model(args.base, str(out / "sft"))
        data = build_dataset(args.trajectories, args.supervision, tok)
        if args.limit:
            data = data[: args.limit]
        trainer = GRPOTrainer(model, tok, make_reward_fn(stage=1), grpo_cfg, out)
        trainer.train(data, args.steps, tag="cja")
        trainer.save("cja")
        report["phases"]["cja"] = {
            "steps": args.steps,
            "reward_first": round(trainer.history[0].reward_mean, 4),
            "reward_last": round(trainer.history[-1].reward_mean, 4),
            "format_first": round(trainer.history[0].valid_format_rate, 3),
            "format_last": round(trainer.history[-1].valid_format_rate, 3),
        }
        del model, trainer
        release()

    # ---------------- GRPO stage 2: gated DQA ----------------
    if "dqa" in phases:
        # Table 9: "Stage-2 initialization: Stage-1 LoRA adapter".
        init = out / "cja" if (out / "cja").exists() else out / "sft"
        model, tok = load_model(args.base, str(init))
        data = build_dataset(args.trajectories, args.supervision, tok)
        # Stage 2 trains on every trajectory that has a real error location -
        # the gold supporting document the generator ignored. That label is
        # free and correct for all 343 non-CORRECT trajectories, where only 25
        # have teacher reason/fix text. Filtering to the teacher-labelled 25
        # discarded 93% of the usable localisation signal; the reason/fix
        # rewards are instead switched off per-example via trust_diagnosis.
        data = [d for d in data
                if d["reference"].verdict is not Verdict.CORRECT
                and d["reference"].location is not None]
        if args.limit:
            data = data[: args.limit]
        for d in data:
            d["prompt"] = d["prompt"] + d["prefix"]
        print(f"[train] DQA on {len(data)} located trajectories "
              f"(verdict pre-filled; critic writes location/reason/fix)")
        trainer = GRPOTrainer(model, tok, make_reward_fn(stage=2, prefilled=True),
                              grpo_cfg, out)
        trainer.train(data, args.steps, tag="dqa")
        trainer.save("dqa")
        report["phases"]["dqa"] = {
            "steps": args.steps,
            "reward_first": round(trainer.history[0].reward_mean, 4),
            "reward_last": round(trainer.history[-1].reward_mean, 4),
            "gate_first": round(trainer.history[0].gate_open_rate, 3),
            "gate_last": round(trainer.history[-1].gate_open_rate, 3),
        }
        del model, trainer
        release()

    report["minutes"] = round((time.time() - started) / 60, 1)
    (out / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
