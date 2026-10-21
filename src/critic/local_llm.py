"""Batched local generation, sized for an 8GB laptop GPU.

Everything here exists because of one measurement: on this card, a config that
needs slightly more than the free VRAM does **not** raise OutOfMemoryError. On
Windows, CUDA silently spills into system RAM over PCIe and the same step goes
from 3.5s to 44.9s - a 13x slowdown that looks like a hang, not a crash. An
overnight run would still be going at breakfast with no error in the log.

So this module:
  * refuses to start if the model will not leave `MIN_HEADROOM_GB` free,
  * halves the batch and retries on OOM rather than dying mid-run,
  * watches peak allocation and shrinks the batch when it approaches the spill
    threshold, before the slowdown happens,
  * frees the cache between batches to limit fragmentation.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GB = 1e9
# Leave this much VRAM unused. Below it, the allocator starts spilling to
# system memory instead of failing, which is far worse than an exception.
MIN_HEADROOM_GB = 0.8


def free_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.mem_get_info()[0] / GB


def vram_report() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    free, total = torch.cuda.mem_get_info()
    return (f"{torch.cuda.memory_allocated()/GB:.2f} GB used / "
            f"{free/GB:.2f} GB free of {total/GB:.2f} GB")


def release() -> None:
    """Drop cached blocks. Called between phases, not inside hot loops."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True
    repetition_penalty: float = 1.0


class LocalLLM:
    """A chat model with memory-aware batched generation.

    `batch_size` is an upper bound, not a promise: it is reduced automatically
    on OOM or when peak allocation gets close to the spill threshold.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        max_input_tokens: int = 2048,
    ) -> None:
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_input_tokens = max_input_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Decoder-only batched generation requires LEFT padding: with right
        # padding the pads land between the prompt and the first generated
        # token, and short prompts in a batch produce garbage.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        before = free_vram_gb()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, device_map=self.device)
        self.model.eval()
        after = free_vram_gb()
        print(f"[llm] {model_name} loaded: {before - after:.2f} GB weights, {vram_report()}")

        if self.device == "cuda" and after < MIN_HEADROOM_GB:
            raise RuntimeError(
                f"only {after:.2f} GB free after loading {model_name}; "
                f"need >{MIN_HEADROOM_GB} GB of headroom. Use a smaller model "
                f"or close other GPU applications.")

    # -- internals ----------------------------------------------------------
    def _render(self, messages: Sequence[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    @torch.inference_mode()
    def _generate_batch(self, prompts: list[str], cfg: GenerationConfig) -> list[str]:
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.model.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature if cfg.do_sample else None,
            top_p=cfg.top_p if cfg.do_sample else None,
            repetition_penalty=cfg.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # Strip the prompt: with left padding every row shares an input length.
        gen = out[:, enc["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)

    # -- public API ---------------------------------------------------------
    def chat(
        self,
        conversations: Iterable[Sequence[dict[str, str]]],
        cfg: GenerationConfig | None = None,
        batch_size: int = 4,
        on_progress: Callable[[int, int, float], None] | None = None,
    ) -> list[str]:
        """Generate one completion per conversation, in order."""
        cfg = cfg or GenerationConfig()
        prompts = [self._render(c) for c in conversations]
        results: list[str] = []
        i, bs, started = 0, max(1, batch_size), time.time()

        while i < len(prompts):
            chunk = prompts[i:i + bs]
            try:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                results.extend(self._generate_batch(chunk, cfg))

                # Pre-emptive shrink: if this batch came within the headroom of
                # spilling, the next one probably would spill. Spilling does not
                # raise, so reacting to OOM alone is not enough.
                if torch.cuda.is_available() and bs > 1:
                    peak = torch.cuda.max_memory_allocated() / GB
                    _, total = torch.cuda.mem_get_info()
                    if peak > (total / GB) - MIN_HEADROOM_GB:
                        bs = max(1, bs // 2)
                        print(f"[llm] peak {peak:.2f} GB near spill threshold; "
                              f"batch -> {bs}")
                i += len(chunk)
                if on_progress:
                    on_progress(i, len(prompts), time.time() - started)
            except torch.cuda.OutOfMemoryError:
                release()
                if bs == 1:
                    # One prompt alone will not fit. Record a failure rather
                    # than abort the whole run.
                    print(f"[llm] OOM at batch=1 on item {i}; emitting empty result")
                    results.append("")
                    i += 1
                else:
                    bs = max(1, bs // 2)
                    print(f"[llm] OOM; batch -> {bs}, retrying")
        release()
        return results

    def unload(self) -> None:
        """Free the weights. Call before loading a second model."""
        del self.model
        release()
        print(f"[llm] unloaded, {vram_report()}")
