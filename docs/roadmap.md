# Roadmap

What is worth doing next, what has been ruled out, and the mistakes that cost
the most time. `HANDOFF.md` carries the reasoning behind every decision named
here; this is the short list.

Current as of 2026-08-28.

---

## Where the system stands

Three corpora, each measured separately, each with its own tuning. Run
`python src/serve.py` and open <http://127.0.0.1:8000/quality> for the live
figures — the ones below are a snapshot and the page is generated.

| | documents | passages | any-hit@5 |
|---|---|---|---|
| ML & NLP papers | 36 | 5,459 | 0.851 |
| Ornithology | 45 | 864 | 0.846 |
| Quantitative finance | 35 | 6,184 | 0.886 |

**324 checks**, none of which need a network or an API key:

```bash
python src/test_metrics.py        # the scoring functions, hand-computed
python src/test_trace.py          # the trace and the serving path agree
python src/test_routes.py         # every HTTP route answers (starts its own server)
python src/test_generate_local.py # generation, over a real socket
node  ui/test-answer-mark.mjs     # which words of a passage are set bold
```

**Four guards**, each exiting non-zero rather than printing a warning nobody
reads. They exist because a generated file has gone stale silently three times:

```bash
python src/check_freshness.py     # golden set -> per_case -> analytics -> front page
python src/check_golden.py        # do the labels still describe the corpus?
python src/check_docs.py          # do the documents match the measurements?
python src/build_results_doc.py --check
```

One thing is deliberately outside the count: `python src/smoke_routes.py` asks
every route of an already-running server. `src/test_routes.py` starts its own,
which is why that one *is* counted.

---

## What is worth doing next

**1. Ship the dense ensemble properly, or decide not to.** Fusing a second
embedder is measured and enabled on the quantitative-finance corpus, where it
buys +0.036 MRR, +0.022 NDCG and +0.028 source recall for no loss of any-hit.
It is *not* enabled on the other two, and the reason is the interesting part:
on pool recall it looked weakly dominant everywhere, and measured end to end the
ML corpus got **worse on all four metrics** from a strictly better candidate
pool. What is left is deciding whether a second vector per chunk — roughly
doubling the index and adding an encode pass to ingest — is worth one corpus's
gain. `src/sweep_ensemble.py` reproduces the measurement.

**2. Query decomposition, for seven of the twelve structural failures.**
Splitting "normalisation across features *rather than examples*" into topic and
constraint is what an LLM does well. This is no longer blocked: `generate_local.py`
speaks to a local Ollama, so a small instruct model can be measured against
`eval/hard_cases.json`, which exists for exactly this and has been verified
stable across a threshold change, a blend change and a corpus that grew. If a
local 7B moves two of the seven, that is the finding — a better model can be
swapped in later.

**3. The other five structural failures need no LLM.** They *describe* a term
and ask for its name — "which small group of feathers helps prevent a stall at
low speed" — so the answer word is absent from the question. One of the five was
already recovered by changing the embedder alone, which is what motivated the
ensemble work above.

**4. The hero diagram.** All seven stages are drawn as the same sheet-of-cells,
which is wrong in one specific place: dense retrieval and BM25 have identical
geometry in the spec (`layers: 4, cells: [5,4], w: 6.5`) and are opposites in
nature — continuous similarity against sparse term hits. Drawing them the same
hides the reason for running both. Also wanted: a *gate* at the diversity cap
showing passages blocked, and more motion inside the arrays. The
index-as-a-field is right and should stay.

## What has been ruled out, with numbers

Kept because a refuted experiment is worth as much as a shipped one. All of
these are in `docs/engineering-log.md` with the measurements.

- **A better cross-encoder.** Three compared. MiniLM-L12 ranks better on the ML
  papers and worse on birds; BGE-reranker-base ranks worst and separates best.
  Neither is shippable.
- **int8 quantisation of the reranker.** 1.5× faster with near-identical
  rankings — and every score shifts by a systematic −0.22, so the gate silently
  starts refusing answerable questions. Rank-preserving is not enough when a
  threshold reads the score as an absolute.
- **Pseudo-relevance feedback.** Recovers 0 of 12 structural cases, because the
  feedback documents do not contain the missing word either.
- **Prefixing chunks with their document title.** Built, measured, reverted:
  no ranking gain and source recall −0.051.
- **Larger candidate pools.** 28 candidates scores *worse* than 20 on all three
  corpora while the pool ceiling rises on all three.

## Things that will bite you

- **A corpus setting does not transfer.** Five have now been measured
  per-corpus rather than global: the abstention threshold, the rerank blend, the
  candidate pool size, the choice of embedder, and whether fusing a second
  embedder helps at all. The 0.0 threshold that costs the ML papers one question
  costs the bird corpus ten.
- **Pool recall is a ceiling, not a proxy.** Three separate experiments have now
  improved what the first stage retrieves and made the finished pipeline worse.
  Measure end to end before shipping anything that changes the candidate pool.
- **When a corpus scores badly, read the failing questions first.** The
  quantitative-finance corpus looked like a retrieval weakness at 0.543 and was
  entirely a golden-set problem — textbook definitions asked of research papers.
  0.886 after rewriting the questions, retrieval untouched.
- **A measurement that returns zero deserves as much suspicion as a surprise**,
  and so does one that returns 100%. Two zeros here came from comparing a dict
  locator against its serialised string; a check reporting "67 of 67 cases
  ambiguous" came from subtracting a key that no case has.
- **Render the interface and look at it.** Roughly twenty real defects came from
  screenshotting the page rather than from reading the diff. Check a laptop
  height (~660px) and a phone, not only a wide desktop window.
- **Run the thing.** Every defect found in the last five working sessions was in
  code that had tests *around* it and nothing *running* it — a route that
  ignored its corpus argument, a JSON field of the wrong type that dropped the
  connection, indexing one corpus deleting another's cache, and a module that
  had never been invoked at all.

## Still genuinely blocked

Nothing is blocked on API credit any more — `RAG_GENERATOR=ollama` runs the
whole generation path against a local model. What remains unmeasured is answer
*quality*: every number in this project measures retrieval, and no evaluation of
generated prose exists. That needs a rubric and a judge, and the judge is the
hard part.
