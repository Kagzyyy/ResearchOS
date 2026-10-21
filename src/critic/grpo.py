"""GRPO (Group Relative Policy Optimization) for the structured critic.

GRPO comes from DeepSeekMath (Shao et al., 2024) and is what CRITIC-R1 trains
with. Its one idea, against PPO: **drop the value network**. PPO learns a
critic network to estimate how good a state is, so it can compute advantage =
reward - baseline. That network is as large as the policy, so it roughly
doubles memory. GRPO instead samples a *group* of G completions for the same
prompt and uses the group's own mean as the baseline:

    A_i = (r_i - mean(r)) / std(r)

The comparison is between siblings answering the identical prompt, so prompt
difficulty cancels out - a hard prompt where every sample scores badly still
produces a useful gradient, because what matters is which sibling scored
*relatively* better. On an 8GB card this is not a nicety: a second
value-network-sized model simply would not fit.

Written directly rather than pulled from TRL for two reasons: this venv also
runs the production backend and a torch/transformers dependency bump would put
that at risk, and generation here has to be memory-governed on a card that
silently spills to system RAM instead of raising OutOfMemoryError.

Memory plan, from measurements on this GPU (8.59 GB total, 7.45 GB free):

    weights + LoRA r=16                 3.16 GB
    rollouts, 16 seqs x 2048 KV cache   0.94 GB   (inference_mode, no graph)
    fwd/bwd at micro-batch 1            4.34 GB peak
                                        -------
    worst phase                        ~4.4 GB, leaving ~3 GB of headroom

The reference policy for the KL term costs nothing extra: with LoRA, disabling
the adapters *is* the reference model, so there is no second copy of weights.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F

from src.critic.local_llm import GB, MIN_HEADROOM_GB, release, vram_report

# A reward function sees the sample's metadata and the generated text.
RewardFn = Callable[[dict[str, Any], str], float]


@dataclass
class GRPOConfig:
    """Defaults follow the paper's Table 9 where it specifies a value."""

    group_size: int = 4             # G: completions per prompt
    prompts_per_step: int = 2       # prompts per optimizer step
    gen_batch: int = 8              # sequences per generation call
    lr: float = 1e-6                # Table 9
    kl_coef: float = 0.003          # Table 9
    clip_eps: float = 0.2
    inner_epochs: int = 1           # updates per rollout batch
    max_prompt_tokens: int = 1536
    max_new_tokens: int = 160
    temperature: float = 1.0
    top_p: float = 0.95
    grad_clip: float = 1.0
    log_every: int = 1
    save_every: int = 25
    seed: int = 42


@dataclass
class StepStats:
    step: int
    reward_mean: float
    reward_std: float
    reward_max: float
    reward_min: float
    kl: float
    loss: float
    valid_format_rate: float
    gate_open_rate: float
    verdict_match_rate: float
    seconds: float
    vram_peak_gb: float


def _sequence_logprobs(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    prompt_len: int, *, requires_grad: bool,
) -> torch.Tensor:
    """Per-token log-probs of the completion portion only.

    `logits_to_keep` is what makes this affordable: Qwen's vocabulary is
    151,936, so logits over a full 2048-token sequence would be a
    [B, 2048, 151936] tensor - about 0.7 GB in bf16 before its gradient. Only
    the generated tokens are scored, so only those logits are materialised.
    """
    n_completion = input_ids.shape[1] - prompt_len
    if n_completion <= 0:
        return torch.zeros((input_ids.shape[0], 0), device=input_ids.device)

    ctx = torch.enable_grad() if requires_grad else torch.no_grad()
    with ctx:
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    logits_to_keep=n_completion + 1)
        # logits[:, i] predicts token i+1, so drop the last and align.
        logits = out.logits[:, :-1, :]
        targets = input_ids[:, prompt_len:]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        return torch.gather(logprobs, 2, targets.unsqueeze(-1)).squeeze(-1)


def group_advantages(rewards: Sequence[float], eps: float = 1e-4) -> torch.Tensor:
    """GRPO's advantage: a sample's reward relative to its own group.

        A_i = (r_i - mean(r)) / (std(r) + eps)

    This single line is what removes PPO's value network. The baseline is the
    group mean, and because every member of the group answered the *identical*
    prompt, prompt difficulty cancels: a hard prompt where all four samples
    score -0.9 still yields a useful gradient, since what is learned is which
    sibling did better, not whether the absolute score was high.

    A group whose samples all scored the same carries no preference
    information, so its advantages are forced to exactly zero. Without that,
    dividing float noise by a near-zero std manufactures a large spurious
    gradient - which is also the honest signal that training is stuck: it is
    exactly what a model failing the format cliff produces, every sample
    scoring -1.0.
    """
    rt = torch.as_tensor(list(rewards), dtype=torch.float32)
    std = rt.std(unbiased=False)
    if std < 1e-6:
        return torch.zeros_like(rt)
    return (rt - rt.mean()) / (std + eps)


class GRPOTrainer:
    """Two-stage GRPO over a LoRA-adapted causal LM."""

    def __init__(self, model, tokenizer, reward_fn: RewardFn,
                 cfg: GRPOConfig | None = None, output_dir: str | Path = "models/critic") -> None:
        self.cfg = cfg or GRPOConfig()
        # Validate config before allocating anything: a bad setting should
        # fail in the first millisecond, not after a model load.
        if self.cfg.inner_epochs != 1:
            # old_lp is re-derived from new_lp each pass, pinning the ratio at
            # 1, so extra inner epochs would repeat the same gradient rather
            # than performing PPO-style reuse. Fail loudly instead of training
            # something subtly wrong for hours.
            raise ValueError(
                "inner_epochs must be 1: this implementation is on-policy and "
                "does not cache old log-probs across passes.")
        if self.cfg.group_size < 2:
            # A group of one has no baseline to be relative to, so every
            # advantage is zero and no learning can occur.
            raise ValueError("group_size must be >= 2 for a group-relative baseline")

        self.model = model
        self.tok = tokenizer
        self.reward_fn = reward_fn
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=self.cfg.lr)
        self.history: list[StepStats] = []
        torch.manual_seed(self.cfg.seed)

    # -- rollouts -----------------------------------------------------------
    @contextmanager
    def _generation_mode(self):
        """Switch the model into a state where generation is actually correct.

        Training needs gradient checkpointing (it is what fits 1.5B into 8GB),
        but checkpointing forces `use_cache=False`, and generating without a
        KV cache in train mode produced pure garbage here - random tokens,
        mixed scripts, code fragments - while reporting no error at all. Every
        rollout scored the format penalty, so the group had zero variance and
        GRPO silently learned nothing.

        Restoring the cache also removes the quadratic cost of re-encoding the
        whole prefix at every decode step, which is most of the rollout time.
        """
        self.model.gradient_checkpointing_disable()
        self.model.config.use_cache = True
        self.model.eval()
        try:
            yield
        finally:
            self.model.config.use_cache = False
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.train()

    @torch.inference_mode()
    def _rollout(self, prompts: list[str]) -> list[list[str]]:
        """Sample G completions for each prompt.

        inference_mode (not no_grad) because it also disables version counters
        and autograd bookkeeping, which matters when generating 16 sequences
        on a card with 3 GB of slack.
        """
        cfg = self.cfg
        flat = [p for p in prompts for _ in range(cfg.group_size)]
        texts: list[str] = []
        bs = cfg.gen_batch

        i = 0
        with self._generation_mode():
          while i < len(flat):
              chunk = flat[i:i + bs]
              try:
                  enc = self.tok(chunk, return_tensors="pt", padding=True, truncation=True,
                                 max_length=cfg.max_prompt_tokens).to(self.model.device)
                  out = self.model.generate(
                      **enc, max_new_tokens=cfg.max_new_tokens, do_sample=True,
                      temperature=cfg.temperature, top_p=cfg.top_p,
                      pad_token_id=self.tok.pad_token_id,
                      # A finished critique ends at </fix> - about 80 tokens.
                      # Without this every rollout ran the full 200, and
                      # generation is the slowest phase of the step.
                      stop_strings=["</fix>"], tokenizer=self.tok)
                  gen = out[:, enc["input_ids"].shape[1]:]
                  texts.extend(self.tok.batch_decode(gen, skip_special_tokens=True))
                  i += len(chunk)
              except torch.cuda.OutOfMemoryError:
                release()
                if bs == 1:
                    texts.append("")
                    i += 1
                else:
                    bs = max(1, bs // 2)
                    print(f"[grpo] rollout OOM; gen_batch -> {bs}")
        release()
        return [texts[j * cfg.group_size:(j + 1) * cfg.group_size] for j in range(len(prompts))]

    # -- one optimizer step -------------------------------------------------
    def step(self, batch: list[dict[str, Any]], step_idx: int) -> StepStats:
        """One GRPO step over `prompts_per_step` prompts."""
        cfg = self.cfg
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        prompts = [b["prompt"] for b in batch]
        groups = self._rollout(prompts)

        # --- rewards and group-relative advantages -------------------------
        samples: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        for meta, completions in zip(batch, groups):
            rewards = []
            for text in completions:
                r = self.reward_fn(meta, text)
                rewards.append(float(r if not isinstance(r, tuple) else r[0]))
                if isinstance(r, tuple):
                    diagnostics.append(r[1])
            adv = group_advantages(rewards)
            for text, a, r in zip(completions, adv.tolist(), rewards):
                samples.append({"meta": meta, "text": text, "advantage": a, "reward": r})

        all_rewards = torch.tensor([s["reward"] for s in samples])

        # --- policy update --------------------------------------------------
        total_loss, total_kl, n_micro = 0.0, 0.0, 0
        for _ in range(cfg.inner_epochs):
            self.opt.zero_grad(set_to_none=True)
            for sample in samples:
                loss, kl = self._micro_step(sample, len(samples))
                if loss is None:
                    continue
                total_loss += loss
                total_kl += kl
                n_micro += 1
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], cfg.grad_clip)
            self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        release()

        peak = torch.cuda.max_memory_allocated() / GB if torch.cuda.is_available() else 0.0
        valid = sum(1 for d in diagnostics if not d.get("parse_error")) / max(len(diagnostics), 1)
        gate = sum(1 for d in diagnostics if d.get("gate_open")) / max(len(diagnostics), 1)
        match = sum(1 for d in diagnostics if d.get("verdict_match")) / max(len(diagnostics), 1)

        stats = StepStats(
            step=step_idx,
            reward_mean=all_rewards.mean().item(), reward_std=all_rewards.std().item(),
            reward_max=all_rewards.max().item(), reward_min=all_rewards.min().item(),
            kl=total_kl / max(n_micro, 1), loss=total_loss / max(n_micro, 1),
            valid_format_rate=valid, gate_open_rate=gate, verdict_match_rate=match,
            seconds=time.time() - t0, vram_peak_gb=peak,
        )
        self.history.append(stats)

        # The spill trap: exceeding free VRAM does not raise on Windows, it
        # silently moves tensors to system RAM and runs ~13x slower. Warn
        # loudly rather than let an overnight run quietly crawl.
        if torch.cuda.is_available():
            total_gb = torch.cuda.mem_get_info()[1] / GB
            if peak > total_gb - MIN_HEADROOM_GB:
                print(f"[grpo] WARNING peak {peak:.2f} GB is within {MIN_HEADROOM_GB} GB "
                      f"of the {total_gb:.2f} GB card - risk of silent spill to "
                      f"system RAM. Reduce group_size or prompts_per_step.")
        return stats

    def _micro_step(self, sample: dict[str, Any], n_total: int) -> tuple[float | None, float]:
        """Clipped policy-gradient loss with a KL leash, for ONE sample.

        Deliberately one sample at a time. The tokenizer is left-padded for
        generation, so a batched version would have to slice the completion
        at a per-row offset; getting that subtly wrong gathers log-probs from
        pad tokens and silently trains on noise. Memory forces micro_batch=1
        on this card anyway (measured: 4.34 GB peak at batch 1), so batching
        here would buy nothing and risk a bug that does not announce itself.

        Concatenating prompt and completion ids directly also means no pad
        tokens exist, so there is no mask to get wrong and no compute spent on
        padding.
        """
        cfg = self.cfg
        text = sample["text"]
        if not text.strip():
            return None, 0.0

        p_ids = self.tok(sample["meta"]["prompt"], return_tensors="pt", truncation=True,
                         max_length=cfg.max_prompt_tokens)["input_ids"]
        c_ids = self.tok(text, return_tensors="pt", add_special_tokens=False,
                         truncation=True, max_length=cfg.max_new_tokens)["input_ids"]
        if c_ids.shape[1] < 2:
            return None, 0.0

        ids = torch.cat([p_ids, c_ids], dim=1).to(self.model.device)
        attn = torch.ones_like(ids)
        plen = p_ids.shape[1]

        try:
            new_lp = _sequence_logprobs(self.model, ids, attn, plen, requires_grad=True)
            # Reference policy = the same weights with LoRA switched off, so
            # the KL term costs no extra memory.
            with self.model.disable_adapter():
                ref_lp = _sequence_logprobs(self.model, ids, attn, plen, requires_grad=False)

            adv = float(sample["advantage"])

            # One update per rollout keeps this on-policy, so old == new and
            # the ratio is exactly 1 in value - but the gradient still flows
            # through new_lp, which is what makes this the policy gradient.
            old_lp = new_lp.detach()
            ratio = torch.exp(new_lp - old_lp)
            pg = -torch.min(ratio * adv,
                            torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv)

            # k3 KL estimator (Schulman): exp(d) - d - 1, always >= 0 and
            # lower variance than the naive difference.
            #
            # Clamped because exp() makes this explode: a smoke run logged
            # KL=70.09 on a single step, from one token the SFT'd policy was
            # confident about and the frozen base model was not. Unclamped,
            # a rare token can dominate the whole batch's gradient and
            # destabilise a long run. +-10 keeps every realistic divergence
            # intact while bounding the tail.
            d = torch.clamp(ref_lp - new_lp, -10.0, 10.0)
            kl = torch.exp(d) - d - 1.0

            loss = (pg + cfg.kl_coef * kl).mean()
            (loss / max(n_total, 1)).backward()
            return loss.item(), kl.mean().item()
        except torch.cuda.OutOfMemoryError:
            release()
            print("[grpo] OOM in micro-step; skipping this sample")
            return None, 0.0

    # -- training loop ------------------------------------------------------
    def train(self, dataset: Sequence[dict[str, Any]], steps: int,
              tag: str = "stage1") -> list[StepStats]:
        cfg = self.cfg
        self.model.train()
        log_path = self.out / f"{tag}_log.jsonl"
        started = time.time()
        print(f"[grpo] {tag}: {steps} steps x {cfg.prompts_per_step} prompts "
              f"x G={cfg.group_size} | {vram_report()}")

        cursor = 0
        for s in range(1, steps + 1):
            batch = [dataset[(cursor + j) % len(dataset)] for j in range(cfg.prompts_per_step)]
            cursor += cfg.prompts_per_step
            stats = self.step(batch, s)

            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(stats)) + "\n")
            if s % cfg.log_every == 0:
                eta = (time.time() - started) / s * (steps - s)
                print(f"[grpo] {tag} {s}/{steps}  R={stats.reward_mean:+.3f}"
                      f"+-{stats.reward_std:.3f}  KL={stats.kl:.4f}  "
                      f"fmt={stats.valid_format_rate:.0%}  "
                      f"verdict={stats.verdict_match_rate:.0%}  "
                      f"{stats.seconds:.1f}s  peak={stats.vram_peak_gb:.2f}GB  "
                      f"ETA {eta/60:.0f}m", flush=True)
            if s % cfg.save_every == 0 or s == steps:
                self.save(f"{tag}_step{s}")
        return self.history

    def save(self, name: str) -> Path:
        path = self.out / name
        self.model.save_pretrained(path)      # LoRA adapters only, a few MB
        self.tok.save_pretrained(path)
        return path
