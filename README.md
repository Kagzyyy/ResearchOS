# ResearchOS

A scientific-literature research assistant whose retrieval quality comes from a
**cross-encoder reranker fine-tuned here**, not from an off-the-shelf model.

The RAG app is the harness. The reranker is the point. The deliverable is the
ablation table below: the same model, the same hyperparameters, the same seed,
trained twice on datasets that differ **only** in how their negatives were chosen.

## What the experiment found

Two findings, and the second one is a null result:

**1. Fine-tuning the reranker works, and matters a lot.** Off the shelf,
`bge-reranker-base` scores **0.7124** nDCG@10 on SciFact — *worse* than the BM25 +
dense baseline it is supposed to improve (0.7236). Fine-tuned on mined data it reaches
**0.7602**: +0.048 over the off-the-shelf model, +0.037 over the baseline.

**2. Hard negatives did *not* measurably beat random negatives here.** The headline
hypothesis was that mining negatives from the retriever's own top-50 would beat
sampling them uniformly from the corpus. The measured gap is **+0.0019 nDCG@10**, with
a paired permutation test giving **p = 0.81** and a 95% bootstrap CI of
**[−0.013, +0.017]** — straddling zero. Per query: 40 better, 35 worse, **225
identical**. On this benchmark, at this scale, the intervention is not detectable.

That is reported as a null result rather than buried, because a 0.25% difference
presented as a win would be dishonest. `scripts/significance.py` reproduces it, and
[Why hard negatives did not help here](#why-hard-negatives-did-not-help-here) discusses
what would need to change to give the hypothesis a fair test.

---

## Demo

<!-- Record a 60s capture of the Gradio UI and drop it here as docs/demo.gif -->
![60-second demo](docs/demo.gif)

> **Not yet recorded.** Start the app (below), run one of the example claims, expand a
> passage, and click "Mark as useful" — that is the full loop the GIF should show.

---

## Architecture

```mermaid
flowchart LR
    Q["Claim<br/><i>Vitamin D reduces<br/>infection risk</i>"]

    Q --> BM25["<b>BM25</b><br/>bm25s<br/>k1=1.5, b=0.75"]
    Q --> DENSE["<b>Dense</b><br/>bge-small-en-v1.5 (33M)<br/>FAISS flat, cosine"]

    BM25 -- "top 50" --> RRF["<b>RRF fusion</b><br/>k=60"]
    DENSE -- "top 50" --> RRF

    RRF -- "50 candidates" --> RR["<b>Cross-encoder reranker</b><br/>bge-reranker-base (278M)<br/><i>fine-tuned on hard negatives</i>"]

    RR -- "top 5" --> GEN["<b>Claude</b><br/>answer with<br/>span citations"]

    GEN --> API["FastAPI"]
    API --> UI["Gradio UI"]
    UI -- "citation clicks" --> DB[("SQLite<br/>feedback log")]

    style RR fill:#2d5016,stroke:#7cb342,color:#fff
    style RRF fill:#1a3a52,stroke:#4a90d9,color:#fff
```

**Why two retrievers.** BM25 matches words; dense retrieval matches meaning. SciFact
punishes word-matching badly — its very first test claim, *"0-dimensional biomaterials
show inductive properties,"* is answered by an abstract about *"quantum dots for stem
cell labeling"* that shares **zero content words** with the claim. RRF merges the two
ranked lists without requiring their scores to be comparable.

**Why a separate reranker.** Scoring all 5,183 documents with a cross-encoder is
impossible at interactive latency. Retrieval gets recall up cheaply (94.2% of gold
passages land in the top 50); the reranker then reads all 50 carefully and reorders.

---

## Results

SciFact **test** split, 300 queries, 5,183 documents. Metrics from the official BEIR
evaluator. `results/metrics.json` is the source of truth; this table is rendered from
it by `scripts/render_readme.py`.

<!-- RESULTS:START -->

| Setup | nDCG@10 | Recall@50 | p95 latency | Params |
|---|---|---|---|---|
| BM25 only | 0.6863 | 0.8852 | 1 ms | — |
| BM25 + dense (RRF) | 0.7236 | 0.9420 | 38 ms | 33M |
| + off-the-shelf reranker | 0.7124 | 0.9420 | 250 ms | 278M |
| + fine-tuned, random negatives | 0.7583 | 0.9420 | 235 ms | 278M |
| + fine-tuned, hard negatives | 0.7602 | 0.9420 | 215 ms | 278M |
| Fine-tuned, ONNX INT8 | — | — | — | — |

**Fine-tuning works.** The fine-tuned reranker scores nDCG@10 0.7602 against 0.7124 off the shelf (+0.0478), and the off-the-shelf model is itself *below* the retrieval baseline it is meant to improve.

**Hard vs. random negatives: +0.0019 nDCG@10 — NOT statistically significant.** Paired permutation p = 0.806; 95% bootstrap CI [-0.0134, +0.0172] over 300 queries. Per query: 40 better, 35 worse, 225 identical. Same seed, same hyperparameters, same epochs — only the negatives file differs. Reproduce with `scripts/significance.py`.

<!-- RESULTS:END -->

Latency is end-to-end per query (retrieval + fusion + reranking), measured
single-query on an RTX 4060 Laptop, 8GB.

**On the parameter counts.** `bge-reranker-base` is commonly described as a 110M model,
and the original spec for this project assumed that. It is not: the checkpoint holds
**278.0M** parameters, because it is XLM-RoBERTa-based with a 250,002-token multilingual
vocabulary. The word-embedding matrix alone is 192.0M; the transformer body and
classification head are 86.0M. Verified by counting `model.parameters()`, and visible in
the 1,112 MB fp32 ONNX export. The 8GB choice still holds — 278M fits, 568M
(`bge-reranker-v2-m3`) would not — but the table reports the measured number rather than
the folklore one. `bge-small-en-v1.5` is 33.4M, as advertised.

---

## Engineering decisions

### Hard-negative mining

A reranker's job is discriminating between passages that *all* look plausible — that is
what it is handed at inference time, since the retriever already filtered out the
obviously irrelevant. Training it on random negatives teaches the wrong task: telling
biomedical text from unrelated biomedical text is trivially easy, the loss collapses
early, and the model never learns the fine distinctions it will actually face.

So negatives are mined from the retriever's own output:

```
for each training query:
    retrieve top-50 with the same BM25 + dense + RRF pipeline used at inference
    drop every gold passage
    sample 7 negatives from ranks 10-50
```

Ranks 10–50, not 1–50, deliberately. The very top of the list is where unlabeled
positives hide — SciFact's judgments are sparse, and a passage ranked 3rd for a claim
is quite likely relevant even if no annotator marked it. Training on those as negatives
teaches the model to suppress correct answers. Starting at rank 10 trades a little
hardness for a lot of label safety.

The control arm draws its negatives uniformly from the corpus. Both datasets have the
same size, the same positives, in the same order, with the same number of negatives per
group. `configs/train_random.yaml` and `configs/train_hard.yaml` differ only in the
negatives file, the output directory, and the run name — verifiable with `diff`.

How different the two datasets actually are, measured on the 919 mined groups:

| | hard | random |
|---|---|---|
| Groups | 919 | 919 |
| Negatives per group | 7 | 7 |
| Positives | identical | identical |
| Negatives that the retriever surfaced in the top-50 **for their own query** | 100.0% | 0.9% |
| Negative instances shared between the two files | 6 / 6,433 | |

That 100% vs 0.9% is the intervention, stated numerically. A hard negative is a passage
the retrieval stack genuinely confused for an answer to *this* claim; a random negative
almost never is.

The intervention was therefore applied cleanly. It just did not produce a measurable
effect — see below.

### Why hard negatives did not help here

The training signal behaved exactly as the hypothesis predicted. Training loss, same
steps, same schedule, same seed — only the negatives file differs:

| | step 20 | step 40 | step 60 | step 80 |
|---|---|---|---|---|
| Random negatives | 0.4987 | 0.1366 | 0.0893 | 0.0924 |
| Hard negatives | 0.8770 | 0.5025 | 0.3966 | 0.3233 |
| **Ratio** | 1.8× | 3.7× | **4.4×** | 3.5× |

This is the clearest result in the project, and it is not the one that was being tested.
The random arm **solves its task by step 40** and flatlines at ~0.09 — it spent the back
half of training with essentially nothing left to learn. The hard arm is still at 0.32
after three epochs and still descending: it never ran out of difficulty. The two arms
were unmistakably doing different work.

**And the evaluation did not move at all.** A 3.5× difference in training loss produced a
+0.0019 difference in nDCG@10 and left 225 of 300 queries byte-identical. That gap
between "the optimisation clearly changed" and "the metric clearly did not" is the actual
finding, and it points hard at explanation 1 below.

Three explanations, in descending order of how much I believe them:

1. **The evaluation has no room left.** Reranking only reorders the top-50, so
   Recall@50 is pinned at 0.9420 for every reranked row — the ceiling is fixed before
   the reranker runs. 225 of 300 queries produced byte-identical rankings under both
   models. Most SciFact claims have exactly one gold passage that both arms already
   place first, so there is nothing for a better model to fix.
2. **Domain adaptation dominates negative quality.** Most of the +0.048 comes from
   teaching a general-purpose multilingual reranker what biomedical text looks like.
   Both arms get that equally. Negative *selection* is a second-order effect on top of a
   much larger first-order one.
3. **919 training groups is small.** With 87 optimizer steps total, there may simply not
   be enough gradient updates for a subtler training signal to express itself.

What a fair test would need: a larger benchmark where recall is not already 94% (BEIR's
NFCorpus or TREC-COVID), more than one seed per arm so within-arm variance can be
compared against between-arm variance, and enough training data that the ceiling is the
model rather than the dataset. None of that fits on one laptop evening, which is a
limitation of this project, not evidence against the technique — the literature reports
hard-negative mining working well at larger scale.

### Listwise loss, not pointwise

The objective is `MultipleNegativesRankingLoss` over a group of 1 positive and 7
negatives — softmax cross-entropy across the group. Pointwise BCE would ask "is this
pair relevant, yes or no," which is not the question at inference time. Listwise asks
"which of these eight is the right one," which is exactly the question.

### Fitting 278M params into 8GB

Full fine-tuning at batch size 8 OOMs immediately, because
`MultipleNegativesRankingLoss` expands each batch item into 8 pairs — 64 sequences of
384 tokens per forward pass. The escape ladder in `src/rerank/train.py`, in order:

1. **Gradient checkpointing.** Recompute activations during the backward pass instead
   of storing them. Costs ~30% throughput, saves most of the activation memory.
2. **LoRA (r=16).** Freeze the backbone, train low-rank adapters on the attention
   projections. Drops optimizer state from ~1.3GB to a few MB.

Sequence length, batch size, and dataset size are never reduced — they are pinned by
the experiment design, and shrinking them would make the ablation incomparable.
`torch.cuda.max_memory_allocated()` is printed after every run so headroom stays visible.

### The quantization tradeoff

Row 6 exports the row-5 model to ONNX and applies INT8 dynamic quantization: weights
stored as int8, activation scales chosen at runtime, no calibration set needed.

Measured on 50 test queries × 50 candidates each (`results/latency.json`):

| Backend | Device | p50 | p95 | Throughput | File size |
|---|---|---|---|---|---|
| PyTorch fp16 | CUDA | **174 ms** | **177 ms** | **289 pairs/s** | — |
| ONNX fp32 | CPU | 7,790 ms | 7,856 ms | 6.4 pairs/s | 1,112 MB |
| ONNX INT8 | CPU | 3,879 ms | 6,990 ms | 10.3 pairs/s | **279 MB** |

**Quantization did what it promised — and it is still 22× slower than the GPU.** INT8
is a genuine 2.0× speedup over fp32 on the same CPU, at 4.0× smaller on disk. But the
honest comparison is the top row: reranking 50 candidates takes 174 ms on the 4060 and
3.9 seconds on the CPU. No amount of quantization closes that.

So INT8 is **not a latency optimization here — it is a deployment option.** It buys the
ability to serve without a GPU at all, in a 279 MB artifact that fits comfortably in a
container (which is exactly what the `Dockerfile` ships). If you have the GPU, use
PyTorch. If you cannot depend on one, INT8 makes CPU serving merely slow rather than
unusable. Framing it as "we made inference faster" would be true only against a baseline
nobody would deploy.

Note also the p50/p95 spread: PyTorch is 174/177 ms — almost perfectly flat — while INT8
is 3,879/6,990 ms. The CPU path's tail is nearly 2× its median, because it is competing
for cores with everything else on the machine. Tail latency, not median, is what a user
actually feels.

Quantization is also lossy, so row 6 **re-evaluates quality** rather than reporting speed
alone. Reporting a latency number without the accuracy column beside it would be
dishonest — the whole question is whether the speedup costs more than it is worth.

### Prompt injection

Retrieved passages are untrusted input. In `src/api/generate.py` they are never
concatenated into the instruction block — instructions are fixed in the system prompt
before any passage is seen, and passages arrive in the user turn wrapped in
`<passage id="...">` tags with tag-like sequences stripped so a passage cannot close its
own wrapper. The API call passes **no tools at all**, so retrieved text has no
tool-calling path to reach.

---

## Reproducing

Stages are sequential. Each writes its metric to `results/` before the next begins.

```bash
python scripts/run_baseline.py       --config configs/baseline.yaml          # rows 1, 2
python scripts/run_rerank.py         --config configs/rerank_offshelf.yaml   # row 3
python scripts/mine_negatives.py     --config configs/mine_negatives.yaml
python scripts/train.py              --config configs/train_random.yaml
python scripts/run_rerank.py         --config configs/rerank_random_ft.yaml  # row 4
python scripts/train.py              --config configs/train_hard.yaml
python scripts/run_rerank.py         --config configs/rerank_hard_ft.yaml    # row 5
python scripts/export_onnx.py        --config configs/onnx.yaml
python scripts/benchmark_latency.py  --config configs/onnx.yaml
python scripts/run_rerank.py         --config configs/rerank_onnx_int8.yaml  # row 6
python scripts/render_readme.py
```

Every script takes a config file. No hyperparameter is hardcoded.

### Verifying the ablation

The headline claim is a delta between two training runs, which is worth nothing unless
those runs were identical apart from the intervention. That is checked mechanically
rather than taken on trust:

```bash
python scripts/verify_ablation.py
```

It asserts that the two configs differ only in the three permitted fields, that the two
datasets carry the same positives in the same order with the same group sizes, that the
negative sets barely overlap, and that both runs took the same number of optimizer steps
under the same learning-rate schedule. It exits non-zero if any check fails.

### Setup

```bash
uv venv --python 3.11
.venv\Scripts\activate.bat
uv pip install -r requirements.txt
```

Set `ANTHROPIC_API_KEY` in your environment (or a `.env` file) for answer generation.
Without it, retrieval and reranking still work and the UI shows ranked passages only.
`WANDB_API_KEY` is optional; training logs to Weights & Biases when it is present and
runs normally when it is not.

### Running the app

```bash
python scripts/serve_api.py --config configs/app.yaml
```

```bash
python scripts/serve_ui.py --config configs/app.yaml
```

API docs at `http://127.0.0.1:8000/docs`, UI at `http://127.0.0.1:7860`.

---

## Hardware

Everything here runs on one laptop: **RTX 4060, 8GB VRAM**. No cloud GPU was used, and
none is required. Training a full arm of the ablation takes roughly 20 minutes.

---

## Repository layout

```
src/
  config.py    config loading, seeding, VRAM reporting
  data/        loading, negative mining
  retrieval/   bm25, dense, rrf fusion
  rerank/      training, evaluation, onnx export and runtime
  api/         fastapi app, pipeline, claude generation, sqlite log
  ui/          gradio app
configs/       one yaml per experiment
results/       metrics.json, latency.json, feedback.db
scripts/       entrypoints
```

---

## Limitations and next steps

Bounding the claims honestly is a feature of this project, not an apology.

- **The headline hypothesis was not confirmed.** Hard negatives did not measurably beat
  random negatives (+0.0019 nDCG@10, p = 0.81, CI [−0.013, +0.017]). What the project
  *does* demonstrate is that fine-tuning the reranker at all is worth +0.048 over the
  off-the-shelf model. The hard-vs-random question remains open, and this setup was
  underpowered to answer it — see
  [Why hard negatives did not help here](#why-hard-negatives-did-not-help-here).
- **Only one seed per arm.** With a single run each, there is no way to separate the
  between-arm difference from ordinary run-to-run variance. Three seeds per arm would
  have made the null result much stronger; it would also have tripled the training time.
- **Evaluation uses a public benchmark, not a bespoke labeled set.** SciFact is a
  standard BEIR dataset. That makes the numbers comparable to published work, but it
  also means the model has been tuned against a benchmark that many models have seen.
  Nothing here demonstrates performance on a proprietary corpus.
- **Citation-click feedback is logged but the retraining loop is designed, not
  running.** Every click lands in `results/feedback.db` with its query and the
  passages that were shown. Turning that into a training signal — deduplication,
  position-bias correction, a periodic mining job — is specified and not built.
- **Distillation to a ~30M student and Triton serving are planned, not done.** The
  current serving path is a single-process FastAPI app with the 278M reranker. That is
  adequate for a demo and would not survive real traffic.
- **The SciFact test split is small, and it is the binding constraint.** 300 queries put
  the 95% CI on a paired difference at roughly ±0.015 nDCG@10 — which is why a +0.0019
  effect was never going to be resolvable here. Any future arm of this experiment needs
  either a larger benchmark or many more seeds before its delta means anything.
- **Generation is not evaluated.** Retrieval is the measured bottleneck and the only
  thing this project makes claims about. Answer quality, faithfulness, and citation
  correctness are unmeasured.
- **The 60-second demo GIF has not been recorded.** Everything it would show works and
  is exercised by the API and UI, but the capture itself is still to do.
