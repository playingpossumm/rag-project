# Retrieval evaluation results

Corpus: `attention_is_all_you_need.pdf`, 15 pages → **64 chunks** (240 tokens, 40 overlap).
Golden set: **24 answerable + 16 adversarial** cases, labelled at page level, each
answerable case carrying a verified answer string.
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

## Abstention — knowing when the corpus cannot answer

Nearest-neighbour search has no concept of "nothing here": it always returns `k`
results, so a question about a topic the corpus has never seen still yields five
chunks at scores indistinguishable from real matches. Handing those to a generator
invites confabulation, and citations make confabulation look authoritative.

Cosine similarity structurally cannot fix this — every vector has some cosine to
every other. Cross-encoder logits can, because the model is asked whether a passage
*answers* a query rather than whether it resembles one.

Calibrated against 24 answerable and **16 unanswerable** cases:

| population | n | min | median | max |
|---|---|---|---|---|
| answerable | 24 | **+1.88** | +4.78 | +9.24 |
| unanswerable | 16 | −11.06 | −8.10 | **+3.95** |

Adversarial cases are grouped by *why* they are unanswerable, because the groups
behave very differently:

| kind | n | worst (highest) score | caught at threshold 0 |
|---|---|---|---|
| absent topic | 5 | −4.53 | **5 / 5** |
| absent metadata | 4 | −5.34 | **4 / 4** |
| near-miss | 7 | +3.95 | 5 / 7 |

**All overlap between the populations comes from near-misses** — questions where the
paper discusses the adjacent thing. That is why the adversarial set was expanded from
4 cases to 16 with near-misses deliberately over-represented: an easy adversarial set
would have produced a threshold that collapsed on the first realistic hard question.

### Choosing the threshold — and not taking the sweep's optimum

| threshold | unanswerable caught | answerable wrongly refused | net |
|---|---|---|---|
| −4 | 0.812 | 0.000 | 0.812 |
| **0** | **0.875** | **0.000** | 0.875 |
| +1 | 0.938 | 0.000 | **0.938** |
| +2 | 0.938 | 0.042 | 0.896 |

The sweep's best net separation is **+1**. The shipped threshold is **0**, deliberately:

- The lowest answerable case scores **+1.88**. A threshold of +1 leaves 0.88 of margin
  on a 24-case sample — tuning to the edge of the data rather than to the signal, and
  the first unseen hard-but-answerable question gets refused.
- **False abstention is the costlier error.** Refusing a question the corpus *can*
  answer is worse than returning weak-but-cited context, because the citation lets a
  reader judge for themselves.
- Zero is where the logit's own semantics put the boundary: positive means the passage
  answers the query. A threshold that needs justifying is worse than one that already
  means something.

The two survivors at zero are `adv-decoder-layers` (+3.95 — the paper does give N = 6
for the base model) and `adv-cost` (+0.96 — training cost is given, in FLOPs not
dollars). Both return cited context for a genuinely neighbouring question, which is a
defensible outcome rather than a hallucination.

### Verified end to end

The abstention path runs entirely before any API call, so unlike generation it is
fully verified despite the billing block:

```
$ python src/generate.py "What noise schedule does the diffusion model use?"

The indexed documents do not appear to contain an answer to: '...'
Rather than answer from general knowledge, which would be ungrounded and
uncitable, the system is declining.

Closest match was attention_is_all_you_need.pdf, page 8
(confidence -10.57, below threshold +0.0).
```

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
