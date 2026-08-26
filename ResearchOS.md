# CLAUDE.md

Project context for Claude Code. Read this before making changes.

## What this project is

A scientific-literature research assistant whose retrieval quality comes from a
**cross-encoder reranker I fine-tuned myself**, not from an off-the-shelf model.

The RAG app is the harness. The reranker is the point. Every architectural
decision should favor producing a defensible experimental result over adding
product features.

The deliverable is a six-row ablation table (see below) plus a working demo.

## Hardware constraints — these are hard limits

Training runs on a laptop **RTX 4060, 8GB VRAM**. No cloud GPU. Do not suggest
one.

Non-negotiable settings:

- Reranker base model: `BAAI/bge-reranker-base` (110M). **Never** `bge-reranker-v2-m3` (568M) — it will not fit.
- Embedding model: `BAAI/bge-small-en-v1.5` (33M).
- `max_length=384`, batch size 8, `gradient_accumulation_steps=4`, fp16 mixed precision.
- If OOM: switch to LoRA (`peft`, r=16) before reducing dataset size or sequence length.
- Checkpoint every epoch. A crash must never cost more than one epoch.

Print `torch.cuda.max_memory_allocated()` after every training run so VRAM
headroom stays visible.

## Stack

| Layer | Choice |
|---|---|
| Dataset | SciFact via `beir` |
| Lexical retrieval | `bm25s` |
| Dense retrieval | `bge-small-en-v1.5` + FAISS flat index |
| Fusion | Reciprocal rank fusion |
| Reranker | `bge-reranker-base`, fine-tuned |
| Training | PyTorch + `sentence-transformers` CrossEncoder |
| Tracking | Weights & Biases |
| Inference optimization | ONNX Runtime + INT8 dynamic quantization |
| API | FastAPI |
| UI | Gradio |
| Generation | Claude API |
| Logging | SQLite |

**Do not add:** LangChain, LlamaIndex, a vector database service, Next.js,
Triton, Docker Compose, or an agent framework. If a dependency is not in the
table, ask before adding it.

## Build order

Stages are sequential. **Do not start a stage until the previous stage's metric
is written into `results/metrics.json` and committed.** A half-finished later
stage is worth less than a completed earlier one.

1. **Baseline retrieval** — Load SciFact. Build BM25 and dense indices. Fuse with RRF. Evaluate nDCG@10 and Recall@50 on the test split. → rows 1 and 2.
2. **Off-the-shelf reranker** — Rerank baseline top-50 with untrained `bge-reranker-base`. Evaluate. → row 3.
3. **Mine training data** — For each train query, retrieve top-50, drop the gold passage, sample negatives from ranks 10–50. Build two datasets: `hard_negatives.jsonl` and `random_negatives.jsonl` (same size, same positives).
4. **Train on random negatives** — Baseline arm of the ablation. → row 4.
5. **Train on hard negatives** — Identical hyperparameters, different data. The delta between rows 4 and 5 is the headline result. → row 5.
6. **Optimize inference** — Export the row-5 model to ONNX, apply INT8 dynamic quantization. Benchmark p50/p95 latency and throughput for PyTorch / ONNX / ONNX-INT8. Re-evaluate quality. → row 6.
7. **Application** — FastAPI: retrieve → rerank → Claude with retrieved context → answer with span citations. Gradio UI. Log every citation click to SQLite.
8. **Ship** — README, architecture diagram, 60s demo recording, Dockerfile.

## The result table

`results/metrics.json` is the source of truth; `README.md` renders it.

| Setup | nDCG@10 | Recall@50 | p95 latency | Params |
|---|---|---|---|---|
| BM25 only | | | | — |
| BM25 + dense (RRF) | | | | 33M |
| + off-the-shelf reranker | | | | 110M |
| + fine-tuned, random negatives | | | | 110M |
| + fine-tuned, hard negatives | | | | 110M |
| Fine-tuned, ONNX INT8 | | | | 110M |

## Training details

- Loss: listwise cross-entropy, 1 positive vs. 7 sampled negatives per group. **Not** pointwise BCE.
- LR 2e-5, cosine schedule, 10% warmup, 2–3 epochs.
- Seed everything. Log the seed.
- Rows 4 and 5 must differ **only** in the negatives file. Same seed, same hyperparameters, same epochs. If anything else differs the ablation proves nothing.

## Evaluation rules

- Evaluate only on the SciFact **test** split. Never tune on it.
- Use the official BEIR evaluator, not a hand-rolled nDCG.
- Report the off-the-shelf baseline next to every fine-tuned number.
- Never report a synthetic-data metric as a headline result.

## Repository layout

```
src/
  data/      loading, chunking, negative mining
  retrieval/ bm25, dense, rrf fusion
  rerank/    training, evaluation, onnx export
  api/       fastapi app
  ui/        gradio app
configs/     yaml, one per experiment
results/     metrics.json, latency.json, plots
scripts/     entrypoints (run_baseline.py, mine_negatives.py, train.py, ...)
```

Every script takes a config file. No hardcoded hyperparameters.

## Conventions

- Type hints on all function signatures. Small, testable functions.
- Any script that trains or evaluates writes its result to `results/` — never print-only.
- Cache indices and embeddings to disk; rebuilding is slow and unnecessary.
- Commit after each completed stage with the metric in the message, e.g. `feat: hard-negative reranker, nDCG@10 0.74`.
- No secrets in code. `ANTHROPIC_API_KEY` from environment.

## Prompt injection

Retrieved passages are untrusted input. Keep them structurally delimited in the
prompt to Claude, never concatenated into the instruction block. Retrieved text
must not reach any tool-calling path.

## README requirements

The public README must include, in order: a 60-second demo GIF, an architecture
diagram, the results table, an "Engineering decisions" section explaining
hard-negative mining and the quantization tradeoff, and a **"Limitations and
next steps"** section stating honestly that:

- evaluation uses a public benchmark rather than a bespoke labeled set;
- citation-click feedback is logged but the retraining loop is designed, not running;
- distillation to a ~30M student and Triton serving are planned.

Bounding the claims honestly is a feature. Do not remove this section.

## Out of scope

Screen capture / desktop overlay, multi-tenancy, auth, Terraform, distillation,
Triton, drift monitoring, fine-tuning the generator. Generation stays a Claude
API call — retrieval is the measured bottleneck.
