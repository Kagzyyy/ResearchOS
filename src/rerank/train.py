"""Fine-tune bge-reranker-base with a listwise objective.

Loss: MultipleNegativesRankingLoss over a group of 1 positive and N sampled
negatives. For a cross-encoder this is softmax cross-entropy across the group -
listwise, not pointwise BCE. The model has to rank the positive above its
negatives, which is exactly the task at inference time.

Rows 4 and 5 of the results table run this file with identical settings and a
different negatives file. Nothing else may differ.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

# Must be set before the first CUDA allocation. Reduces fragmentation, which is
# what kills the second attempt after an OOM retry on an 8GB card.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import Dataset
from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.losses import MultipleNegativesRankingLoss

from src.config import report_vram
from src.data.negatives import read_jsonl


def build_dataset(jsonl_path: str | Path, num_negatives: int) -> Dataset:
    """Turn mined groups into the column layout MultipleNegativesRankingLoss wants.

    Columns: query, positive, negative_1 .. negative_N (in that order).
    """
    groups = read_jsonl(jsonl_path)
    if not groups:
        raise ValueError(f"No training groups found in {jsonl_path}")

    columns: dict[str, list[str]] = {"query": [], "positive": []}
    for i in range(1, num_negatives + 1):
        columns[f"negative_{i}"] = []

    for group in groups:
        negatives = group["negatives"]
        if len(negatives) < num_negatives:
            raise ValueError(
                f"Group {group['query_id']} has {len(negatives)} negatives, "
                f"need {num_negatives}"
            )
        columns["query"].append(group["query"])
        columns["positive"].append(group["positive"])
        for i in range(1, num_negatives + 1):
            columns[f"negative_{i}"].append(negatives[i - 1])

    dataset = Dataset.from_dict(columns)
    print(f"[train] {len(dataset)} groups from {jsonl_path} | columns: {dataset.column_names}")
    return dataset


def _apply_lora(model: CrossEncoder, rank: int) -> CrossEncoder:
    """Wrap the encoder in LoRA adapters. Used only as an OOM fallback."""
    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.05,
        target_modules=["query", "key", "value"],
    )
    model.model = get_peft_model(model.model, lora_config)
    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.model.parameters())
    print(f"[train] LoRA r={rank}: {trainable:,} trainable / {total:,} total params")
    return model


def _free_cuda(reset_peak: bool = False) -> None:
    """Release everything before an OOM retry, or the retry inherits the problem.

    `reset_peak` is opt-in: resetting the counter on teardown would wipe the
    measurement before it can be reported, which is the whole point of tracking it.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if reset_peak:
            torch.cuda.reset_peak_memory_stats()


def _training_args(
    cfg: dict[str, Any], output_dir: str, seed: int, gradient_checkpointing: bool
) -> CrossEncoderTrainingArguments:
    tc = cfg["train"]
    # W&B only if the user actually has credentials; never block a run on it.
    report_to = "wandb" if os.environ.get("WANDB_API_KEY") else "none"
    if report_to == "none":
        print("[train] WANDB_API_KEY not set - tracking disabled, metrics still written to results/")
    else:
        os.environ.setdefault("WANDB_PROJECT", cfg["run"].get("wandb_project", "researchos"))

    return CrossEncoderTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=tc["epochs"],
        per_device_train_batch_size=tc["batch_size"],
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        learning_rate=float(tc["learning_rate"]),
        warmup_ratio=tc["warmup_ratio"],
        lr_scheduler_type=tc["lr_scheduler"],
        fp16=tc["fp16"],
        bf16=False,
        gradient_checkpointing=gradient_checkpointing,
        save_strategy="epoch",
        save_total_limit=tc.get("save_total_limit", 3),
        logging_steps=tc.get("logging_steps", 20),
        dataloader_num_workers=0,
        seed=seed,
        report_to=report_to,
        run_name=cfg["run"]["name"],
    )


def train_reranker(cfg: dict[str, Any], seed: int) -> str:
    """Train one arm of the ablation. Returns the saved model directory."""
    mc = cfg["model"]
    tc = cfg["train"]
    output_dir = tc["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(tc["negatives_file"], tc["num_negatives"])

    def run(use_lora: bool, gradient_checkpointing: bool) -> str:
        _free_cuda(reset_peak=True)
        model = CrossEncoder(mc["base"], num_labels=1, max_length=mc["max_length"])
        if use_lora:
            model = _apply_lora(model, tc.get("lora_rank", 16))
        loss = MultipleNegativesRankingLoss(model, num_negatives=tc["num_negatives"])
        trainer = CrossEncoderTrainer(
            model=model,
            args=_training_args(cfg, output_dir, seed, gradient_checkpointing),
            train_dataset=dataset,
            loss=loss,
        )
        try:
            trainer.train()
            # Read the peak BEFORE teardown; the counter must survive to be reported.
            report_vram(f"{cfg['run']['name']} (training)")
            final_dir = str(Path(output_dir) / "final")
            model.save_pretrained(final_dir)
            print(f"[train] saved -> {final_dir}")
            return final_dir
        finally:
            # Hold no references before a possible retry.
            del trainer, loss, model
            _free_cuda()

    # Memory ladder for an 8GB card. Sequence length and dataset size are never
    # reduced - the spec forbids it. Batch size stays at the configured value.
    attempts: list[tuple[str, bool, bool]] = [
        ("full fine-tune + gradient checkpointing", False, True),
        ("LoRA r={} + gradient checkpointing".format(tc.get("lora_rank", 16)), True, True),
    ]
    if tc.get("use_lora", False):
        attempts = attempts[1:]

    last_error: BaseException | None = None
    for label, use_lora, checkpointing in attempts:
        print(f"[train] attempting: {label}")
        try:
            return run(use_lora=use_lora, gradient_checkpointing=checkpointing)
        except torch.cuda.OutOfMemoryError as exc:
            last_error = exc
            if not tc.get("oom_fallback_lora", True):
                raise
            print(f"[train] CUDA OOM during '{label}' - falling back")
    raise RuntimeError("All training attempts ran out of memory") from last_error
