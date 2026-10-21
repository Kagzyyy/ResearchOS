"""Cold-start supervised fine-tuning (CRITIC-R1 Appendix A.1, first paragraph).

"Before the two-stage RL training, we first perform a lightweight cold-start
SFT step using the structured critique annotations. This step initializes the
critic to follow the required output schema."

Why RL cannot start cold. The format reward is a cliff: valid structure earns
+alpha, anything else earns -gamma with no partial credit (Algorithm 1 returns
immediately on a parse failure). A base model that emits "Sure! Here's my
critique:" before the first tag scores -gamma on every sample in the group, so
every advantage is identical, the group-relative baseline cancels to zero, and
GRPO receives no gradient at all. SFT exists to get the model over that cliff
cheaply, so RL starts where rewards can actually differentiate.

Deliberately short - one or two epochs. Training to convergence here would
teach the model to copy the teacher, which is the behaviour RL is supposed to
improve on.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from src.critic.local_llm import GB, release, vram_report


@dataclass
class SFTConfig:
    epochs: int = 2
    micro_batch: int = 1            # memory-bound, as measured
    grad_accum: int = 8             # effective batch 8
    lr: float = 1e-5                # higher than RL's 1e-6: this is format-learning
    max_length: int = 1792
    grad_clip: float = 1.0
    log_every: int = 10
    seed: int = 42


def build_sft_example(prompt: str, target: str, tokenizer, max_length: int) -> dict[str, torch.Tensor] | None:
    """Tokenise one (prompt, critique) pair, masking the prompt out of the loss.

    Loss is computed on the critique only. Training on the prompt tokens would
    spend capacity learning to reproduce trajectories, which is not the task.
    """
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    t_ids = tokenizer(target + tokenizer.eos_token, add_special_tokens=False)["input_ids"]

    # Truncate the prompt from the left: the trajectory's tail (the answer being
    # judged) matters more than the head (the first documents).
    budget = max_length - len(t_ids)
    if budget < 64:
        return None
    if len(p_ids) > budget:
        p_ids = p_ids[-budget:]

    input_ids = p_ids + t_ids
    labels = [-100] * len(p_ids) + t_ids
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
    }


def train_sft(model, tokenizer, examples: Sequence[dict[str, str]],
              cfg: SFTConfig | None = None,
              output_dir: str | Path = "models/critic/sft") -> dict[str, Any]:
    """Run cold-start SFT. `examples` are {"prompt": ..., "target": ...}."""
    cfg = cfg or SFTConfig()
    torch.manual_seed(cfg.seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)

    encoded = [e for e in (build_sft_example(x["prompt"], x["target"], tokenizer, cfg.max_length)
                           for x in examples) if e is not None]
    print(f"[sft] {len(encoded)}/{len(examples)} examples usable | {vram_report()}")

    losses: list[float] = []
    started = time.time()
    step = 0
    for epoch in range(cfg.epochs):
        order = torch.randperm(len(encoded), generator=torch.Generator().manual_seed(
            cfg.seed + epoch)).tolist()
        opt.zero_grad(set_to_none=True)
        for i, idx in enumerate(order, 1):
            ex = encoded[idx]
            try:
                ids = ex["input_ids"].unsqueeze(0).to(model.device)
                labels = ex["labels"].unsqueeze(0).to(model.device)
                mask = ex["attention_mask"].unsqueeze(0).to(model.device)

                # Score only the critique, never the prompt. Passing `labels`
                # to the model instead would compute logits for every
                # position: with Qwen's 151,936-token vocabulary that is a
                # [1, 1792, 151936] tensor - 0.54 GB in bf16, 1.1 GB upcast
                # for cross-entropy, and as much again for its gradient.
                # Measured peak with that approach: 6.83 GB of a 7.45 GB
                # budget. Restricting logits to the ~80 target tokens removes
                # almost all of it, and the prompt logits were being discarded
                # by the -100 mask anyway.
                n_target = int((labels[0] != -100).sum().item())
                if n_target < 2:
                    continue
                logits = model(input_ids=ids, attention_mask=mask,
                               logits_to_keep=n_target).logits
                # logits[:, i] predicts token i+1, so drop the last and align.
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1, :].reshape(-1, logits.size(-1)).float(),
                    labels[:, -(n_target - 1):].reshape(-1),
                    ignore_index=-100)
                (loss / cfg.grad_accum).backward()
                losses.append(loss.item())
            except torch.cuda.OutOfMemoryError:
                release()
                print(f"[sft] OOM on example {idx} (len {len(ex['input_ids'])}); skipped")
                continue

            if i % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % cfg.log_every == 0:
                    recent = sum(losses[-cfg.grad_accum * cfg.log_every:]) / max(
                        len(losses[-cfg.grad_accum * cfg.log_every:]), 1)
                    peak = (torch.cuda.max_memory_allocated() / GB
                            if torch.cuda.is_available() else 0.0)
                    print(f"[sft] epoch {epoch+1} step {step}  loss {recent:.4f}  "
                          f"peak {peak:.2f}GB", flush=True)
        opt.zero_grad(set_to_none=True)

    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    summary = {
        "examples": len(encoded),
        "epochs": cfg.epochs,
        "steps": step,
        "final_loss": round(sum(losses[-32:]) / max(len(losses[-32:]), 1), 4),
        "initial_loss": round(sum(losses[:32]) / max(len(losses[:32]), 1), 4),
        "minutes": round((time.time() - started) / 60, 2),
    }
    (out / "sft_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[sft] done: loss {summary['initial_loss']} -> {summary['final_loss']} "
          f"in {summary['minutes']} min, saved to {out}")
    release()
    return summary
