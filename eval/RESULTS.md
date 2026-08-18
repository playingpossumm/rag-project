# Retrieval evaluation results

> **Generated from [`results.json`](results.json), written by `src/evaluate.py`.**
> If this file disagrees with that one, this file is stale.

**Corpus:** 20 arXiv ML/NLP papers -> 2,768 chunks (210 tokens, 40 overlap).
**Golden set:** 66 answerable (30 multi-source) + 18 adversarial, labels *derived*
from answer strings against the current corpus rather than hand-written.
**Reproduce:** `python src/evaluate.py`

## Headline

| pipeline | any-hit@5 | MRR | NDCG | src recall |
|---|---|---|---|---|
| dense only, no rerank *(naive RAG)* | 0.788 | 0.601 | 0.645 | 0.704 |
| + cross-encoder rerank | 0.788 | 0.710 | 0.715 | 0.711 |
| + RRF hybrid fusion | 0.864 | 0.757 | 0.766 | 0.742 |
| **+ diversity cap 2/src** *(shipped)* | 0.848 | 0.751 | 0.765 | 0.773 |
| + diversity cap 1/src | 0.818 | 0.739 | 0.751 | 0.816 |

Against naive RAG the shipped pipeline improves any-hit by **+6.0 points** and
MRR by **+15.0**.

## The diversity cap is a trade, not a free win

An earlier 23-case golden set showed the cap costing nothing. With 66 cases it
plainly does:

| cap | any-hit | src recall | trade |
|---|---|---|---|
| none | 0.864 | 0.742 | — |
| 2/src | 0.848 | 0.773 | +3.1 src recall for −1.6 any-hit |
| 1/src | 0.818 | 0.816 | +7.4 src recall for −4.6 any-hit |

When a question is answered by only one document, capping that document pushes a
relevant passage out for an irrelevant one. 2/src ships: most of the breadth for
a third of the cost.

## Context expansion

| mode | context recall | tokens/query | recall per 1k tokens |
|---|---|---|---|
| none (chunks) | 0.742 | 997 | 0.744 |
| **window ±1** *(shipped)* | 0.833 | 2,371 | 0.351 |
| page | 0.848 | 4,828 | 0.176 |

**Reversal.** On the 23-case set window expansion measured no better than raw
chunks and was documented as "strictly dominated". It is not: it recovers 91 of
the 106 points page gains, for less than half the context. Window is now the
default; page remains right where a citation must point at a complete unit.

## Abstention

| population | n | min | median | max |
|---|---|---|---|---|
| answerable | 66 | -4.02 | +4.93 | +9.58 |
| unanswerable | 18 | -9.52 | -2.33 | +2.76 |

The distributions overlap. Shipped threshold is **0.0** — catching 72% of
unanswerable questions while falsely refusing 1 of 66 answerable ones. Third
recalibration of this constant; it is a property of the data, not the model.

## Caveats

- **66 answerable cases.** Each is worth ~1.5 points; treat differences under
  ~0.03 as noise. The previous 23-case set had 4.3-point resolution and produced
  three false conclusions.
- **20 ML papers.** Deliberately similar, which is the hard case, but not
  contracts or spreadsheets. Behaviour on those is untested.
- **Cross-document confusion is unsolved.** Title prefixing was tried, measured,
  and reverted.
- **Labels were authored by the same process that built the system**, mitigated
  by deriving them from the corpus rather than writing them by hand.
