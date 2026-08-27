# ResearchOS-ML — Requirements Specification

Requirements discovery output. **This is not a design document.** No architecture,
no schemas, no implementation. Next step is `/sc:design` or `/sc:workflow`.

Parent project: [ResearchOS](ResearchOS.md) — same shape, different corpus, much
harder evaluation problem.

---

## 1. Goal

Answer questions about machine learning, deep learning, and AI research papers,
including recent ones, with answers grounded in retrieved passages and traceable to
specific papers.

**Delivery order (decided):** build the usable tool first, then retrofit an evaluation
harness once the corpus and chunking have stabilised.

Phase 1 succeeds when the author uses it in preference to searching arXiv by hand.
Phase 2 succeeds when there is a measured, defensible retrieval claim with a
significance test attached.

---

## 2. Scope

### In scope

| Decision | Value |
|---|---|
| Domain | arXiv `cs.LG`, `cs.CV`, `cs.CL`, `cs.AI` |
| Freshness | **Periodic refresh** — scheduled pull, incremental index, version dedup |
| Question types | Explain a concept · Find the paper · Compare approaches |
| Grounding | Every factual claim cites a specific paper and passage |

### Out of scope

- **"Survey what's new"** queries (date-filtered multi-document summarisation) — deferred.
- Generation quality metrics. Retrieval is the measured component; answer faithfulness
  is unmeasured, as in the parent project.
- Full-text figures, tables, and equations as retrievable units. Text only for v1.
- Multi-tenancy, auth, non-arXiv sources.

---

## 3. What differs from the parent project

These are the requirement drivers. Each one invalidates an assumption that held for
SciFact.

| Assumption in ResearchOS | Status here | Consequence |
|---|---|---|
| 5,183 docs, fits a FAISS flat index | **Broken** — ~250k–350k papers, millions of chunks | Exact flat search will not meet latency; approximate index required |
| Documents are ~250-word abstracts | **Broken** — 8k–15k words per paper | Chunking becomes a first-class design problem |
| Corpus is frozen | **Broken** — refreshed on a schedule | Incremental indexing, version dedup, index rebuild policy |
| Relevance judgements ship with the data | **Broken** — none exist | Labels must be manufactured (see §5) |
| One document = one retrievable unit = one citation | **Broken** — many chunks per paper | Results must be diversified; citations must locate a span |
| Query vocabulary is stable | **Broken** — jargon coined monthly | Lexical retrieval arm becomes load-bearing |

**Note on the last row.** `bge-small-en-v1.5` has frozen weights and has never seen
"JEPA", "GRPO", or "Mamba"; it embeds them as near-noise. BM25 gives a rare token a very
high IDF weight, so exact-matching novel terminology is precisely its strength. **The
BM25 arm must not be dropped as an optimisation.** It is the system's only defence
against vocabulary it was never trained on.

---

## 4. Functional requirements

### FR1 — Ingestion
- **FR1.1** Pull papers from arXiv for the configured categories and date range.
- **FR1.2** Extract full text, preserving section structure (title, abstract, and named
  sections).
- **FR1.3** Record per paper: arXiv id, version, title, authors, categories, submission
  date, latest revision date.
- **FR1.4** Ingestion must be resumable — an interrupted run does not corrupt the index
  or require a full restart.

### FR2 — Chunking
- **FR2.1** Split papers into retrievable chunks that fit the reranker's input window.
- **FR2.2** Every chunk carries its paper id, section name, and character offsets into
  the source text, so a citation can point at a span rather than a whole paper.
- **FR2.3** Chunking is deterministic — the same paper produces the same chunks on
  re-ingestion.

### FR3 — Retrieval
- **FR3.1** Hybrid lexical + dense retrieval, fused, as in the parent project.
- **FR3.2** The lexical arm is mandatory (see §3 note on vocabulary drift).
- **FR3.3** Return candidates from the full corpus, not a pre-filtered subset.

### FR4 — Reranking and diversification
- **FR4.1** Rerank fused candidates with a cross-encoder.
- **FR4.2** **Cap chunks per paper in the final result set.** Five chunks from one paper
  is a failure for "explain" and "compare" queries, which need multiple sources. This
  requirement did not exist in the parent project, where one document was one abstract.
- **FR4.3** For "compare" queries the result set must span at least two distinct papers.

### FR5 — Generation
- **FR5.1** Answer only from retrieved passages; state plainly when they are insufficient.
- **FR5.2** Every factual claim carries a citation resolving to a paper and a span.
- **FR5.3** Retrieved text is untrusted: structurally delimited, never concatenated into
  the instruction block, never routed to a tool-calling path. Inherited from the parent
  project and non-negotiable.

### FR6 — Freshness
- **FR6.1** A scheduled job ingests new submissions and adds them to the index without a
  full rebuild.
- **FR6.2** **Version dedup:** when `2401.12345v3` arrives and `v2` is indexed, the old
  version's chunks are *replaced*, not appended. Duplicate near-identical papers in the
  index corrupt both retrieval and any evaluation run over it.
- **FR6.3** Every answer surfaces how current the index is, so the user can tell whether
  a gap means "no such paper" or "not ingested yet".

### FR7 — Feedback
- **FR7.1** Log every query, the passages shown, and which citations the user opened.
  Carried over from the parent project, and worth more here: these logs are a source of
  realistic evaluation queries (see §5, tier 2).

### FR8 — Evaluation harness
- **FR8.1** Build tier-1 and tier-2 label sets as specified in §5.
- **FR8.2** Report nDCG@10 and Recall@k using the official BEIR evaluator.
- **FR8.3** Any comparison between two systems reports a confidence interval and a
  significance test, not a bare delta. Inherited directly from what the parent project's
  null result taught.

---

## 5. Evaluation strategy

**The decision that gates everything else.** arXiv ships no relevance judgements.

### Tier 1 — citation contexts (train + dev)

Mine citation sentences from full text: where paper A writes *"we adopt grouped-query
attention [23]"*, the sentence becomes a query and reference `[23]` is its gold answer.

- **Free and large-scale** — hundreds of thousands of pairs.
- **Real human signal**, not model-generated. This matters: the parent spec forbids
  reporting a synthetic-data metric as a headline result.
- Maps naturally onto the "find the paper" query type.

**Known bias, and the mitigation.** Citation labels skew toward well-cited papers, which
skew *older* — the opposite of this project's stated goal. Restrict citing papers to a
recent window so the cited set also skews recent, and never report a headline number on
tier 1 alone.

### Tier 2 — hand-labelled gold set (test only)

~50 questions written by the author across all three query types, with the top-20
retrieved results judged for relevance. Roughly 1,000 judgements; one to two days.

- Small, but genuinely bespoke — this **retires the parent project's largest stated
  limitation**.
- Covers "explain" and "compare", which citation contexts structurally cannot: those
  queries have no single gold passage.
- Stratify by publication date so failures on recent papers are visible rather than
  averaged away.

### The rule

> Train and tune on tier 1. Report headline numbers on tier 2. **Never tune on tier 2.**

Same discipline as the SciFact test split, for the same reason.

### Rejected alternatives

| Option | Why not |
|---|---|
| LLM-as-judge | Synthetic metric; the parent spec forbids it as a headline result. Acceptable only as a dev-loop smoke signal. |
| SCIDOCS / existing CS benchmark | Task is citation prediction, not question answering, and it contains nothing recent. Useful as a sanity check, not as the deliverable. |

---

## 6. Non-functional requirements

- **NFR1 — Hardware.** Single laptop, RTX 4060, 8GB VRAM. No cloud GPU. Any component
  that cannot be trained or served within that budget is out of scope.
- **NFR2 — Interactive latency.** p95 under ~3 s end-to-end for retrieval and reranking
  on GPU. The parent project's 215 ms was over 5,183 documents; this corpus is ~60×
  larger before chunking.
- **NFR3 — Index scale.** Must remain queryable at millions of chunks. An exact flat
  index will not satisfy NFR2 at this scale — approximate search is required. *(Which
  index: a design decision, not a requirement.)*
- **NFR4 — Storage.** Full text for ~300k papers plus embeddings plus indices. Disk
  budget must be established before ingestion starts.
- **NFR5 — Reproducibility.** Seed everything; log the seed; every script takes a config
  file. Inherited.
- **NFR6 — Cost.** Generation is a paid API call. Per-query cost must be bounded and
  visible.

---

## 7. User stories and acceptance criteria

**US1 — Find the paper.**
> As a researcher, I ask *"which paper introduced grouped-query attention?"* and get the
> correct paper with the passage that introduces it.

*Accepts when:* the correct paper is rank 1 for ≥80% of tier-2 "find the paper"
questions, and the cited span actually contains the definition.

**US2 — Explain a concept.**
> As a researcher, I ask *"explain JEPA"* and get an explanation synthesised from several
> papers, each claim cited.

*Accepts when:* the result set spans ≥3 distinct papers including the originating one;
no single paper supplies more than the FR4.2 cap; every claim in the answer carries a
resolvable citation.

**US3 — Compare approaches.**
> As a researcher, I ask *"how does JEPA differ from MAE?"* and get a contrast grounded in
> both papers.

*Accepts when:* the result set contains passages from both named approaches; the answer
cites both.

**US4 — Recent work is reachable.**
> As a researcher, I ask about a paper published since the last refresh and get either the
> paper or an honest statement that the index does not yet include it.

*Accepts when:* papers ingested in the most recent refresh are retrievable, and the
index date is visible in the response.

**US5 — Honest failure.**
> As a researcher, I ask something the corpus does not cover and am told so, rather than
> given a confident wrong answer.

*Accepts when:* out-of-corpus queries produce an explicit "not found" rather than
low-scoring unrelated passages presented as answers. *(The parent project already
demonstrates the raw signal exists: an out-of-domain query scored +0.06 against +0.60
for an in-domain one — an order of magnitude. What is missing is a threshold that acts
on it.)*

---

## 8. Open questions

1. **Corpus cutoff.** From what date? This sets corpus size, disk, and ingest time, and
   nothing can be estimated until it is fixed.
2. **Refresh cadence.** Daily, weekly, monthly? Drives the ingestion job's design and how
   stale US4 is allowed to be.
3. **Full text or abstracts only?** Abstracts alone make this dramatically easier —
   no PDF parsing, no chunking, an order of magnitude less storage — but cannot support
   "explain" or "compare" well, since methods live in the body. **This is the single
   biggest remaining scope lever.**
4. **Is a fine-tuned reranker still the centrepiece?** The parent project's headline
   ablation was a null result, and most of its measured gain came from domain adaptation
   rather than negative selection. Adapting a reranker from biomedical to ML text is a
   genuine domain shift and would likely repeat that gain — but if the goal is a *useful
   tool*, off-the-shelf retrieval plus good chunking may deliver more per hour spent.
5. **Rejection threshold.** US5 needs a score below which the system declines to answer.
   How is that calibrated, and on which label tier?
6. **Do citation-context queries resemble real user questions?** They are declarative
   sentences, not questions. If the gap is large, tier-1 training may not transfer to
   tier-2 evaluation — this should be checked early, not discovered at the end.

---

## 9. Next step

`/sc:design` for architecture (index choice, chunking strategy, ingestion pipeline), or
`/sc:workflow` for an implementation plan. Question 3 in §8 should be settled first —
it changes the scope of everything downstream.
