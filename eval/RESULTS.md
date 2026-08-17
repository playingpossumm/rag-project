# Retrieval evaluation results

Corpus: `attention_is_all_you_need.pdf`, 15 pages → **64 chunks** (240 tokens, 40 overlap).
Golden set: 24 answerable cases + 4 adversarial, labelled at page level.
Reproduce with `python src/evaluate.py [--candidate-k N] [--per-case]`.

## Headline: the pipeline vs. the naive baseline

| pipeline | hit@5 | MRR | NDCG |
|---|---|---|---|
| dense only, no rerank *(naive RAG)* | 0.958 | 0.771 | 0.807 |
| dense + rerank | 1.000 | 0.941 | 0.921 |
| **RRF hybrid + rerank** | **1.000** | **0.941** | **0.928** |

## The result that changed a default: corpus size distorts the comparison

Measured only at `candidate_k=20`, hybrid search looks worthless — identical hit@5
and MRR to dense-only. That reading is **wrong**, and the reason matters more than
the number.

On a 64-chunk corpus, a 20-candidate pool is **31% of the entire corpus**. Recall is
trivially 1.000 for any first stage, so fusion has no recall headroom to fill. It
*does* improve pool ordering (candidate MRR 0.774 → 0.883), but the reranker
re-scores every candidate from scratch and discards first-stage ordering entirely.
Both effects vanish.

Sweeping candidate depth puts the pool into a realistic selectivity regime —
the one a corpus of hundreds of documents would actually operate in:

| candidate_k | % of corpus | dense recall | RRF recall | dense MRR (e2e) | RRF MRR (e2e) |
|---|---|---|---|---|---|
| 3 | 4.7% | 0.875 | **0.917** | 0.806 | **0.889** |
| 5 | 7.8% | 0.958 | 0.958 | 0.878 | **0.931** |
| 10 | 16% | 0.958 | **1.000** | 0.899 | **0.941** |
| 20 | 31% | 1.000 | 1.000 | 0.941 | 0.941 |

RRF is never worse than dense-only and increasingly better as the pool tightens.
**`DEFAULT_FUSION = "rrf"`** on that basis — the equivalence at k=20 is an artifact
of a small corpus, not a property of the method, and a default chosen from that row
alone would silently degrade as the corpus grows.

Weighted fusion (`alpha=0.5`) performs comparably but adds a parameter that needs
per-corpus tuning and relies on per-query min-max normalization, which distorts
absolute scores. RRF is scale-free and parameter-light, so it wins on robustness
rather than on the numbers.

## Context expansion (parent-document retrieval)

Expansion changes what text is *returned*, never the ranking, so `hit@k` / MRR / NDCG
are identical by construction across all three modes. It is measured instead by
**context recall**: does the text handed to the generator actually contain the answer?
Each answerable case carries a verified `answer_contains` string for this.

| mode | context recall | tokens/query | blocks/query |
|---|---|---|---|
| none (240-token chunks) | 0.958 | 1,138 | 5.0 |
| window ±1 chunk | 0.958 | 2,716 | 5.0 |
| **page** | **1.000** | 2,680 | **3.0** |

**Window expansion is strictly dominated** — it costs the same as page expansion and
recovers nothing. Page expansion reaches full context recall, and returns *fewer*
blocks because hits sharing a page are deduplicated into one, so the model reads each
page once instead of seeing overlapping fragments of it repeatedly.

`expansion="page"` is the default in `generate.py`. The 2.4× token cost is real but
cheap in absolute terms (~2.7k tokens), and it buys the property the whole pipeline
exists for: a citation the model can actually substantiate from the text it was given.

### The case that justifies it — and what it revealed about the metrics

Only one case separates the modes: `compound` ("how many parameters and what dropout
rate did the base model use?").

| | pages returned | hit@k | context recall |
|---|---|---|---|
| chunks | [8, 8, 9, 8, 9] | **1.000** | **0.000** |
| page | [8, 9] | 1.000 | **1.000** |

Chunk retrieval *did* return page 9 — the gold page — so `hit@k` scored it a success.
But the page-9 chunks it returned came from elsewhere on that page and did not contain
the Table 3 row holding the answer. **The ranking metrics reported success on a query
the system could not answer.**

This is the page-level-labelling caveat biting in practice: judging relevance per page
over-credits any chunk that merely shares a page with the answer. Context recall is the
check that catches it, which is the argument for building the measurement before the
feature rather than after.

## Abstention signal

Adversarial cases have no correct answer; a good system should be visibly less
confident. Top-1 score on the four unanswerable questions:

| stage | scores |
|---|---|
| dense first stage (cosine) | +0.43, +0.34, +0.35, +0.27 |
| after reranking (logits) | **−7.04**, +0.96, **−8.13**, **−9.90** |

Cosine similarity cannot separate answerable from unanswerable — the scores sit in
the same band as genuine matches. Cross-encoder logits separate three of four
decisively, making **a negative rerank score a usable abstention threshold**. The
survivor (`adv-cost`, "how many dollars did it cost to train") scores +0.96 because
the paper *does* discuss training cost, just in FLOPs rather than currency — a
near-miss rather than a failure of the signal.

## Known remaining failures

Identical across every reranked configuration, so neither reranking nor fusion
addresses them:

| case | MRR | gold → got | diagnosis |
|---|---|---|---|
| `d-model` | 0.25 | [3] → [5, 6, 6, 3, 5] | "dimensionality of the model's embeddings" matches many chunks discussing dimensions; the defining one (`d_model = 512`) is not lexically distinctive |
| `compound` | 0.33 | [9] → [8, 8, 9, 8, 9] | compound question — parameter count and dropout rate live in different places; no single chunk answers both |

Both are **query-side** problems, not ranking problems. They are the case for query
rewriting and decomposition, not for a better reranker.

## Caveats on these numbers

- **24 cases is small.** Each case is worth ~4 points of hit rate, so single-case
  changes move the metric visibly. Treat differences under ~0.05 as noise.
- **Page-level relevance is coarse.** It cannot distinguish a precise passage from a
  vague one on the same page. Chosen because it survives re-chunking, where
  chunk-level labels would need relabelling after every re-index.
- **One document.** Nothing here tests cross-document retrieval, where a wrong-document
  hit is a distinct failure mode that does not exist in a single-source corpus.
- **The labels were drafted by the same process that built the system**, and
  spot-checked rather than independently authored.
