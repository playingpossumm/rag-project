# Retrieval evaluation results

**Corpus:** 20 arXiv ML/NLP papers → 2,371 chunks (240 tokens, 40 overlap).
**Golden set:** 23 answerable (9 multi-source) + 10 adversarial, labels *derived*
from answer strings against the current corpus rather than hand-written.
**Reproduce:** `python src/evaluate.py [--per-case] [--candidate-k N]`

> Earlier revisions of this file reported numbers from a single-document,
> 64-chunk corpus. Those are superseded. Several conclusions drawn there
> inverted at 20 documents, which is the most useful thing this project has
> demonstrated — see *Corpus size changes the answer* below.

## Headline

| pipeline | any-hit@5 | MRR | NDCG | src recall |
|---|---|---|---|---|
| dense only, no rerank *(naive RAG)* | 0.696 | 0.580 | 0.598 | 0.686 |
| dense + rerank | 0.783 | 0.761 | 0.751 | 0.792 |
| RRF hybrid + rerank | 0.870 | 0.813 | 0.814 | 0.755 |
| weighted hybrid + rerank | **0.913** | **0.828** | **0.836** | 0.701 |
| **+ diversity cap 2/src** | **0.913** | **0.828** | 0.833 | 0.733 |
| **+ diversity cap 1/src** | **0.913** | **0.828** | 0.813 | **0.804** |

Against naive RAG the full pipeline improves any-hit by **+21.7 points** and MRR
by **+24.8**. On the old 64-chunk corpus that gap was far smaller — an easy
corpus flatters the baseline and hides the value of every later stage.

## The metric that changed the design

`hit@k` and MRR ask *"did we find an answer?"*. The requirement here is *"did we
find every source that answers?"* — and those diverge sharply.

For `bleu-ende` (five papers report 28.4 BLEU), the pipeline returned **all five
passages from one paper**: MRR **1.000**, source recall **0.200**. Ranking
metrics score that a perfect result, because by their definition it is.

**Source recall** measures the fraction of answering documents actually
returned, normalised by `min(|gold|, k)` so a run is not penalised for a ceiling
the caller's own `k` imposes. It exposed a failure that was invisible to every
other metric in the harness.

### The fix: a per-document cap, not MMR

MMR diversifies on embedding distance, conflating two different kinds of
redundancy — similar wording vs. same source — and needs a `lambda` tuned per
corpus. Here the unit of redundancy is known exactly: the document. Capping it
directly gives one interpretable knob and cannot be defeated by two documents
phrasing a fact differently.

The cap only *skips*, never reorders, which is why **any-hit and MRR are
identical at every setting** — the top result never moves. The cost is NDCG,
where displaced same-document passages were genuinely relevant.

## Corpus size changes the answer

Every conclusion below was measured twice: once at 64 chunks (1 document) and
once at 2,371 (20 documents). Three inverted.

| finding | at 64 chunks | at 2,371 chunks |
|---|---|---|
| Does hybrid fusion help? | No — all modes tied at 1.000 | **Yes** — dense 0.826 → weighted 1.000 pool any-hit |
| RRF vs weighted | RRF (never worse, no tuning) | **Weighted** measurably better |
| Abstention threshold | Clean separation, +1.88 vs −4.53 | Collapses; populations overlap |

At 64 chunks a 20-candidate pool was **31% of the entire corpus**, so recall was
trivially perfect for any first stage. At 2,371 chunks it is **0.84%** — a real
selectivity regime. Defaults chosen on the small corpus would have silently
degraded as the corpus grew.

**The lesson generalises past this project:** a conclusion measured on a toy
corpus may not merely be imprecise, it may be backwards.

## Fusion vs. breadth — a real tension

| config | any-hit | src recall |
|---|---|---|
| dense + rerank | 0.783 | **0.792** |
| weighted + rerank | **0.913** | 0.701 |

Fusion improves finding *an* answer but hurts finding *all* sources: it
concentrates hits in the best-matching document. Neither config dominates.
Adding the diversity cap to weighted fusion resolves it — 0.913 any-hit *and*
0.804 source recall — rather than forcing a choice between them.

## Context expansion (parent-document retrieval)

Expansion changes returned text, never ranking, so hit/MRR/NDCG are identical by
construction. Measured by **context recall**: does the returned text actually
contain the answer?

| mode | context recall | tokens/query | blocks |
|---|---|---|---|
| none (240-token chunks) | 0.826 | 1,127 | 5.0 |
| window ±1 chunk | — | — | — |
| **page** | *(see note)* | *(see note)* | *(see note)* |

Window expansion was **strictly dominated** on the previous corpus — same token
cost as page expansion, no recall gain — and was dropped as a default.

## Known limitations

- **Cross-document confusion is real and unfixed.** "What optimizer was used to
  train the Transformer?" returns Vision Transformer first; the correct paper's
  relevant page is not in the top 5. The word "Transformer" does not separate
  *the* Transformer paper from the architecture family.
- **Abstention needs recalibration.** The threshold was tuned on a corpus where
  off-topic meant *wildly* off-topic. With 20 adjacent ML papers, near-misses
  are the normal case.
- **Derived labels can be too narrow.** `dropout-rate` scores 0.000 partly
  because its answer string appears in only 2 papers while many more legitimately
  discuss dropout. Deriving labels removes drift but does not remove judgement.
- **23 cases is small.** Treat differences under ~0.05 as noise.
- **Labels were authored by the same process that built the system.** Mitigated
  by deriving them from the corpus, not eliminated.
