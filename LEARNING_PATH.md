# ResearchOS — Learning Path

I am building this myself. Claude's job is to teach, review, and unblock — not to write
the project code. See "Teaching contract" at the bottom.

## The one-sentence goal

Prove, with numbers, that a reranker I fine-tuned on *hard negatives* retrieves
scientific papers better than the same model fine-tuned on *random negatives*.

Everything else (the search, the API, the UI) exists to make that claim demonstrable.

## The pipeline I am building

```
  question
     |
     +--> BM25 search        (keyword matching)     --> 50 candidates
     +--> Dense search       (meaning matching)     --> 50 candidates
     |
     +--> RRF fusion         (merge the two lists)  --> 50 candidates
     |
     +--> Reranker           (score each one)       --> top 5
     |
     +--> Claude             (answer + citations)
     |
     +--> Gradio UI / FastAPI / SQLite log
```

## Lessons

| # | Lesson | What I learn | Checkpoint (proof it works) |
|---|---|---|---|
| 0 | Environment | venv, uv, dependency pinning, GPU check | `torch.cuda.is_available()` is True |
| 1 | The data | SciFact, corpus/queries/qrels, JSON, dicts | Print 1 query + its gold passage |
| 2 | BM25 | Inverted index, TF-IDF intuition, ranking | Row 1 of the table: nDCG@10 |
| 3 | Embeddings + FAISS | Vectors, cosine similarity, ANN index | Row 2: dense + RRF fusion |
| 4 | Evaluation | nDCG, Recall@k, why test split is sacred | `results/metrics.json` exists |
| 5 | Cross-encoders | Bi-encoder vs cross-encoder tradeoff | Row 3: off-the-shelf reranker |
| 6 | Negative mining | Hard vs random negatives, data leakage | Two `.jsonl` files, same size |
| 7 | Training | Loss functions, LR schedule, fp16, VRAM | Rows 4 and 5 — the headline delta |
| 8 | Inference optimization | ONNX, INT8 quantization, p50/p95 latency | Row 6 |
| 9 | The app | FastAPI, Gradio, SQLite, prompt injection | Working demo |
| 10 | Shipping | README, diagram, honest limitations | Public repo |

Rule: do not start lesson N+1 until lesson N's number is in `results/metrics.json`.

## Python I need to pick up along the way

Already assumed: variables, `if`, `for`, functions, lists, `print`.

Learned on demand, in the lesson that first needs it:
- dicts and nested dicts (L1)
- reading/writing files, JSON, JSONL (L1)
- list comprehensions (L2)
- imports, modules, packages, `__init__.py` (L2)
- type hints (L2 onward — required by the project conventions)
- numpy arrays and vectorized math (L3)
- classes (L5)
- decorators (L9, for FastAPI routes)
- `argparse` / YAML configs (L2 onward — no hardcoded hyperparameters)

## Teaching contract

1. Claude explains the concept and *why* it is needed here.
2. Claude gives a skeleton with `TODO:` comments and function signatures — never the filled-in logic.
3. I write the logic.
4. Claude reviews what I wrote, points at bugs, and asks me to explain my own code back.
5. Claude only writes complete code when I explicitly say "just show me".
