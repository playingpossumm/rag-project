# Retrieval evaluation results

> **Figures copied from [`results.json`](results.json), written by
> `src/evaluate.py`.** If this file disagrees with that one, this file is stale.
> It was stale: it claimed 20 papers and 2,768 chunks long after the corpus
> reached 36 and 5,459, because `results.json` declared a
> `corpus` key and never filled it. It fills it now.

**Corpus:** 36 arXiv ML/NLP papers -> 5,459 chunks (210 tokens, 40 overlap).
**Golden set:** 66 answerable (30 multi-source) + 18 adversarial, labels *derived*
from answer strings against the current corpus rather than hand-written.
**Reproduce:** `python src/evaluate.py`

## Headline

| pipeline | any-hit@5 | MRR | NDCG | src recall |
|---|---|---|---|---|
| dense only, no rerank *(naive RAG)* | 0.788 | 0.601 | 0.645 | 0.704 |
| + cross-encoder rerank | 0.788 | 0.710 | 0.715 | 0.711 |
| + RRF hybrid fusion | 0.864 | 0.757 | 0.766 | 0.742 |
| **+ diversity cap 2/src** *(shipped)* | 0.848 | 0.754 | 0.769 | 0.773 |
| + diversity cap 1/src | 0.818 | 0.742 | 0.753 | 0.817 |

Against naive RAG the shipped pipeline improves any-hit by **+6.0 points** and
MRR by **+15.3**.

## The diversity cap is a trade, not a free win

An earlier 23-case golden set showed the cap costing nothing. With 66 cases it
plainly does:

| cap | any-hit | src recall | trade |
|---|---|---|---|
| none | 0.864 | 0.742 | — |
| 2/src | 0.848 | 0.773 | +3.1 src recall for −1.6 any-hit |
| 1/src | 0.818 | 0.817 | +7.5 src recall for −4.6 any-hit |

When a question is answered by only one document, capping that document pushes a
relevant passage out for an irrelevant one. 2/src ships: most of the breadth for
a third of the cost.

## Which questions fail, and whether they are always the same ones

The system misses roughly 15% of answerable questions, and that has been
described as cross-document confusion. That is a claim about *which* questions
fail, and nothing checked it until now. `src/failure_overlap.py` does, over the
six configurations `src/per_case.py` can emit.

Repeating one configuration proves nothing — retrieval is deterministic, so the
same settings return the same passages every time. The variation has to come
from changing the pipeline.

| configuration | missed | refused wrongly | fail rate | adversarial answered |
|---|---|---|---|---|
| default (rrf + rerank + cap 2) | 10 | 1 | 16.7% | 5 |
| no rerank | 12 | 0 | 18.2% | 18 † |
| dense only | 14 | 2 | 24.2% | 4 |
| weighted fusion | 10 | 1 | 16.7% | 5 |
| cap 1/src | 12 | 1 | 19.7% | 5 |
| no cap | 9 | 1 | 15.2% | 5 |

† Not a gate failure. Without reranking there is no calibrated score, so the
gate never runs and all 18 pass by construction — an absent gate, not a failing
one. Excluded from the adversarial comparison below for that reason.

**The failures are not the same set.** 18 of 66 answerable cases (27%) fail
under at least one configuration, but only **7 (10.6%) fail under all six**:

- **7 structural** — no fusion, reranking or capping reaches them. Four are
  labelled `cross-doc`, one `fact`, two `multi`, so cross-document confusion
  describes most of the hard core but not all of it.
- **11 contested** — ranking work moves them, so a change can be scored on
  them. Four are rescued by exactly one configuration, which is within the
  noise this corpus supports and should not be read as that setting being
  better.

So "15% cross-document confusion" is really a ~11% floor plus a shifting
margin, and the floor is the part worth optimising against. It is a fixture:
the same seven cases, reproducible from `src/per_case.py` under any settings.

### A hypothesis this refuted

Three of the seven return the **right document at rank 1** and are scored as
misses because the gold page differs — `adam-bias` returns `adam_optimizer.pdf`
pages 2–3 against gold pages 5, 8, 9. Page-level labelling is known to be
coarse, and labels were derived from answer strings rather than independently
authored, so the obvious reading is that the labels are incomplete and
retrieval found a better passage than the label credits.

That reading is wrong. Context recall — does the returned *text* contain the
labelled answer string — is **0.000 for all seven**. The answer was genuinely
not in what came back. These are real retrieval misses, and the coarse-labelling
caveat does not excuse them.

### The abstention gate is stable

Over the five gated configurations, **4 of 18 adversarial cases are answered
under every one**, and one more (`adv-license-terms`, +0.17) sits close enough
to the threshold to flip. The gate does not become better or worse as ranking
changes, because it reads the top rerank score and ranking configuration barely
moves that score across zero. Those four are a threshold question, not a
retrieval one.

    .venv\Scripts\python.exe src\per_case.py --emit runs\default.json
    .venv\Scripts\python.exe src\per_case.py --no-rerank --emit runs\no-rerank.json
    .venv\Scripts\python.exe src\failure_overlap.py runs\*.json

## The abstention threshold: what each setting costs, by name

`evaluate.py` has reported "best net separation at threshold **+2**" on every
run since the sweep was written, while the shipped threshold is **0.0**. The
project's answer has been that false abstention is the costlier error — a
policy, stated against an aggregate. `src/calibrate_threshold.py` makes it
concrete, and the aggregate turns out to have been the misleading half.

| threshold | refuses | of adversarial | wrongly refuses | of answerable |
|---|---|---|---|---|
| −1 | 12 | 67% | 1 | 1.5% |
| **0.0** *(shipped)* | **13** | **72%** | **1** | **1.5%** |
| +1 | 15 | 83% | 4 | 6.1% |
| +2 | 17 | 94% | 5 | 7.6% |
| +3 | 18 | 100% | 10 | 15.2% |

**In cases, moving 0 → +2 trades 4 more caught for 4 more lost.** Exactly
break-even before any cost weighting — not the clear win `net` reports.

### Why the sweep and the counts disagree

`net = caught − false_abstain` is Youden's J. It is a legitimate statistic and
the wrong one for this decision, because it subtracts two **rates** over
populations of very different size: 18 adversarial against 66 answerable. That
values one adversarial case at **3.7 answerable ones**, and users experience
cases rather than rates. `evaluate.py` now prints a counts column beside the
rates so the two readings can be compared instead of one hiding the other.

Under counts, no threshold above 0.0 wins on its own terms:

| vs 0.0 | more caught | more lost | break-even weight |
|---|---|---|---|
| +1 | 2 | 3 | 0.7× |
| +2 | 4 | 4 | 1.0× |
| +3 | 5 | 9 | 0.6× |

Read: 0.0 wins whenever a wrongly refused question costs more than that
multiple of an unanswerable one slipping through. At +1 and +3 it wins even if
the two cost the same.

### What +1 would actually cost

Not "6.1% of answerable questions" — these three:

| case | top score | question |
|---|---|---|
| `cot-gsm8k` | +0.66 | Which grade-school math benchmark is used to evaluate reasoning? |
| `gpt3-fewshot` | +0.97 | How does task performance change with the number of examples in the prompt? |
| `resnet-shortcut` | +0.16 | What connections allow gradients to flow through very deep networks? |

**Conclusion: 0.0 stays**, and now for a measured reason rather than a stated
preference. The four adversarial cases that slip through under every gated
configuration are not fixable by moving the threshold without losing answerable
questions one for one; they need a different signal, not a different cut point.

### The gate costs different things on different paths

One number serves two cost structures, and the aggregate hides it:

- `api.ask()` — the flag is **advisory**. Passages and citations are returned
  either way, so a false abstention mislabels a good answer.
- the UI — the answer is **withheld** and replaced with "The corpus does not
  contain this", so a false abstention loses an answer the corpus has.

The second is where the cost asymmetry bites, and it is the path a person uses.

## Query expansion (PRF) — built, measured, left off

Pseudo-relevance feedback, RM3-style: run the query, take the top ten results as
if relevant, add their highest-IDF terms, re-run. No model and no network.

| stage | any-hit | MRR | src recall |
|---|---|---|---|
| candidate pool, rrf | 0.924 | 0.717 | 0.836 |
| candidate pool, rrf + PRF | **0.939** | 0.723 | 0.821 |
| end to end, default | 0.848 | 0.754 | 0.773 |
| end to end, default + PRF | 0.848 | 0.754 | 0.767 |

It does what it claims at the stage it targets — **+1.5 points of pool any-hit** —
and the reranker then erases all of it: end-to-end any-hit and MRR are identical
to three decimals, and source recall is slightly worse. This is the same
mechanism already documented for fusion: the cross-encoder re-scores every
candidate from scratch, so a better *ordering* entering the pool is discarded,
and only a better *membership* survives. PRF changes both, and the membership
gain is too small to show.

Available as `query_expansion="prf"`. Default `"none"`.

## Late interaction (ColBERT-style) — built, measured, left off

MaxSim over token embeddings, applied to the candidate pool as a reranker.

| reranker | any-hit | MRR | NDCG | src recall | s/query |
|---|---|---|---|---|---|
| none (fusion order) | 0.818 | 0.701 | 0.724 | 0.777 | 0.04 |
| **cross-encoder** *(shipped)* | **0.848** | **0.754** | **0.769** | 0.773 | 1.11 |
| late interaction | 0.833 | 0.713 | 0.740 | 0.752 | 1.04 |

Strictly dominated: it lands between no reranking and the cross-encoder on every
ranking metric while costing **the same** (1.04s against 1.11s). There is no
operating point at which it is the right choice here.

**One caveat, and it is a real one.** This scores with `all-MiniLM-L6-v2`'s token
embeddings, not a ColBERT checkpoint. Real ColBERT models are trained for MaxSim
with a projection layer, so this measures *late interaction using a bi-encoder's
tokens*, which is not the same claim as "ColBERT is worse". A fair test needs
`colbertv2.0` and is the obvious follow-up.

## Chunk size is bounded by the encoder, not by preference

`all-MiniLM-L6-v2` has `max_seq_length = 256`. A 500-token chunk loses its last
**244 tokens — 49% of every chunk** — silently, before the encoder sees it. That
is the defect this project already found and fixed once, when chunks were sized
in *words*.

`build_index()` refuses to write such an index, so the setting cannot be applied
by accident. Size and overlap are configurable via `RAG_CHUNK_SIZE` and
`RAG_CHUNK_OVERLAP` for measuring alternatives within the ceiling; going beyond
it is a request for a different embedding model, not a different setting.

## Scaling the corpus 20 -> 36 documents

The corpus grew from 20 arXiv papers (2,768 passages) to 36 (5,459) by adding
sixteen recent cs.CL preprints. Recent, not foundational: arXiv was queried
newest-first, so the additions are topically adjacent to the questions without
answering any of them. That is the realistic growth pattern and the harder test
-- more near-misses for the retriever to reject.

| pipeline | hit rate | MRR | NDCG | src recall |
|---|---|---|---|---|
| 20 documents *(shipped config)* | 0.848 | 0.754 | 0.769 | 0.773 |
| 36 documents *(shipped config)* | 0.848 | 0.749 | 0.761 | 0.738 |
| change | **0.000** | −0.005 | −0.008 | **−0.035** |

**Ranking quality did not move.** Hit rate is identical, MRR and NDCG shift by
less than one case (66 cases, so one case is worth ~1.5 points). Nothing here
justifies changing a default.

### Source recall fell for a measurement reason, not a retrieval one

Gold locations are derived from answer strings, so adding documents that
*contain* an answer string adds them to that question's gold set. Eleven of 66
questions gained gold sources, and the mean source-recall divisor -- which is
`min(|gold|, k)` -- rose from **2.12 to 2.30**.

That divisor moving 8.5% is enough to account for the drop on its own. Returning
two of two answering documents scores 1.000; returning the same two documents
when five now answer scores 0.400, with retrieval having done nothing
differently. The metric got harder, the system did not get worse.

### Abstention looks much worse, and it is the labels

| | 20 documents | 36 documents |
|---|---|---|
| unanswerable median | −2.33 | **+0.09** |
| unanswerable max | +2.76 | **+6.85** |
| caught at threshold 0 | 13 / 18 | **9 / 18** |

Five adversarial cases crossed the threshold, and four of them did so because
the corpus now genuinely answers them:

| case | 20 docs | 36 docs | what it now hits |
|---|---|---|---|
| `adv-moe-routing` | −8.36 | **+6.85** | a compute-efficient scaling paper |
| `adv-rlhf` | −4.30 | **+4.22** | a multi-agent orchestration paper |
| `adv-quantize-4bit` | −2.98 | **+2.26** | a small-model architecture paper |
| `adv-mamba` | −1.12 | **+0.70** | the same convolution-attention hybrid |
| `adv-salary` | −4.14 | **+0.09** | a wine benchmark — a genuine false positive |

The gate is doing its job. The labels went stale, which is the failure the
golden set was designed around and warns about in its own docstring: *"when the
corpus grew from 1 document to 20, twelve of sixteen adversarial cases silently
became answerable."* It happened again at 20 -> 36.

**Nothing was adjusted in the retrieval system, because nothing in it regressed.**
The five stale labels are the open item.

### The audit tool missed all of it

`audit_golden_set.py` exists to catch exactly this. Matching hardcoded subject
terms, it flagged two cases -- neither of which crossed the threshold -- and
missed all five that did. It could not have caught them: a paper discusses
expert load balancing without ever writing "mixture of experts".

It now also asks the retriever. If the reranker scores an unanswerable question
above the threshold, either the label is stale or the gate is broken, and both
need a human. On the current corpus that check flags nine cases and reports that
**eight were invisible to the term list**.

## A second corpus: what transfers and what does not

A corpus of 45 Wikipedia ornithology documents was built to test whether any of
this is a property of the system or of the ML papers it was tuned on. It is
deliberately mixed-format -- 12 `.docx`, 12 `.pptx`, 11 `.xlsx`, 10 `.pdf` --
because the loaders for three of those had never run on a real file.

| shipped pipeline | ML papers (36) | ornithology (45) |
|---|---|---|
| hit rate | 0.848 | 0.808 |
| MRR | 0.749 | **0.570** |
| NDCG | 0.761 | 0.635 |
| source recall | 0.738 | **0.762** |

Ranking is materially worse and source recall is slightly better. Both point the
same way: encyclopaedia prose states a fact in one place and in ordinary words,
so the right passage is found but sits lower, and the answer is spread across
fewer documents that are easier to cover.

### The abstention threshold does not transfer at all

This is the finding that matters, and it invalidates a default rather than
adjusting one.

| | answerable median | wrongly refused at 0.0 |
|---|---|---|
| ML papers | +4.93 | 1 of 66 — **1.5%** |
| ornithology | +1.82 | 10 of 26 — **38.5%** |

The shipped threshold refuses **more than a third** of the questions the bird
corpus can answer. On its own sweep the best point is **−6**, six full points
away from the shipped value. The cross-encoder scores encyclopaedia prose far
lower than it scores academic papers, so a cut point calibrated on one corpus
says very little about another.

`abstain.py` had already guessed this in a comment — *"it is a property of the
data, not of the model"* — written after the third recalibration. It is now
measured rather than suspected, and the constant is overridable with
`RAG_ABSTAIN_THRESHOLD`. **A corpus nobody has calibrated is running on a number
derived from somebody else's documents.**

### Hybrid fusion may not transfer either

| candidate pool @20 | ML papers | ornithology |
|---|---|---|
| dense only | 0.909 | **0.962** |
| + BM25, fused by RRF | **0.924** | 0.885 |

On the ML corpus adding the lexical retriever is the single largest gain in the
system. On the bird corpus it appears to *cost* 7.7 points of pool any-hit --
the opposite direction. The plausible reason is that questions about birds and
the text answering them share ordinary vocabulary, so BM25 promotes passages
that merely repeat common words.

**Treated as a lead, not a result.** 26 answerable cases means one case is worth
3.8 points, so this is a two-case difference and inside the noise this corpus can
resolve. It is recorded because it points the opposite way to a settled default,
and that is worth re-testing on a larger bird golden set rather than acting on
now.

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

## Cross-document confusion — diagnosed, not solved

`src/diagnose_crossdoc.py` classifies where the top result comes from:

| outcome | cases | share |
|---|---|---|
| correct | 45 | 68.2% |
| right document, wrong location | 8 | 12.1% |
| **wrong document first** | 13 | 19.7% |

Reading the returned passages, 3 of those 13 are not system failures: `warmup`
returns T5 saying *"we use an 'inverse square root' learning rate schedule"* —
exactly what was asked, but the derived label credits only the paper containing
the string `warmup_steps`. **Derived labels resist drift but are narrow: a
document that answers in different words scores as wrong.** Real confusion is
~10/66 (15%).

### The obvious fix does not work

The remaining failures share a mechanism — the retriever matches the *topic* and
ignores the *constraint that distinguishes the answer*:

| case | distinguishing clause | what it returned |
|---|---|---|
| `ln-stats` | "across features **rather than examples**" | Batch Norm — the excluded concept |
| `sbert-speed` | "faster than a **cross-encoder**" | FAISS: "8.5× faster than prior GPU state of the art" |
| `gpt3-params` | "largest **autoregressive** model" | BERT — not autoregressive |

Reading query and passage jointly is what a cross-encoder is *for*, so the
hypothesis was that the 6-layer MiniLM is too small. Measured on the failing
cases:

| reranker | params | fixed | ms/query |
|---|---|---|---|
| MiniLM-L6 (current) | 22M | 0/10 | 1,226 |
| MiniLM-L12 | 33M | 2/10 | 2,536 |
| BGE-reranker-base | 278M | **2/10** | **10,052** |

**An 8× larger model fixes nothing beyond a 1.5× model.** Capacity is not the
bottleneck. These clauses are *contrastive* — they define the answer by what it
excludes — and negation is a known transformer weakness that scale does not
resolve.

The reranker was left at MiniLM-L6: +3 points sits at the noise floor for 66
cases, and costs 2× latency.

**Conclusion: this is a query-side problem.** The fix is decomposing the
constraint out of the question before retrieval, which needs an LLM — deferred
rather than guessed at.

## Tables are a strength here, not a weakness

Table-aware retrieval was scoped as the next piece of work on the assumption that
tables chunk badly. Measured first:

| | n | any-hit | MRR | src recall |
|---|---|---|---|---|
| **table-answered** | 21 | **0.952** | **0.821** | 0.726 |
| prose-answered | 45 | 0.800 | 0.723 | 0.796 |

Questions whose answer sits in a table are **15 points easier**, not harder. And
91% of table chunks retain their markdown header separator, so the feared
orphaned-rows problem (numbers with no column names) affects ~0.4% of the corpus.

The reason is straightforward once measured: a results table is lexically dense
and distinctive — `MaxSim`, `152`, `28.4`, `WSJ 23 F1` — which is precisely what
BM25 and a cross-encoder latch onto. Prose is diffuse by comparison.

**No work done, because there was no problem to solve.**

### Why this must not be generalised

This corpus's tables are small academic results tables. A financial spreadsheet
is a different object:

| academic results table | financial spreadsheet |
|---|---|
| tens of rows | thousands |
| distinctive terms per cell | repeated numbers, few unique tokens |
| meaning in the caption | meaning in row/column *position* |
| read as a unit | queried by intersection ("Q3, EMEA") |

Nothing measured here says anything about the second column. The `.xlsx` loader
already serialises rows as `Column: value` pairs for exactly that reason, but that
choice is **unvalidated** — no spreadsheet has been evaluated. Testing on real
financial data is the only way to know, and it is likely to produce a different
answer.
