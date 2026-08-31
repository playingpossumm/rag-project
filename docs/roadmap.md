# Roadmap

Current as of 2026-08-31. `HANDOFF.md` carries the reasoning behind every
decision named here.

---

## Current state

There are three corpora, each measured separately and each with its own
tuning. Running `python src/serve.py` and opening
<http://127.0.0.1:8000/quality> gives the live figures, of which the table below
is a snapshot.

| | documents | passages | any-hit@5 |
|---|---|---|---|
| ML & NLP papers | 36 | 5,459 | 0.851 |
| Ornithology | 45 | 864 | 0.846 |
| Quantitative finance | 35 | 6,184 | 0.886 |

**406 checks**, none of which need a network or an API key:

```bash
python src/test_metrics.py        # the scoring functions, hand-computed
python src/test_trace.py          # the trace and the serving path agree
python src/test_routes.py         # every HTTP route answers (starts its own server)
python src/test_generate_local.py # generation, over a real socket
python src/test_evaluate_answers.py  # the answer-quality judge, on answers it has scored wrongly
node  ui/test-answer-mark.mjs     # which words of a passage are set bold
```

**Six guards**, each exiting non-zero rather than printing a warning nobody
reads. They exist because a generated file has gone stale silently three times,
and because a link on `/about` named a branch this repository does not have:

```bash
python src/check_freshness.py     # golden set -> per_case -> analytics -> front page
python src/check_golden.py        # do the labels still describe the corpus?
python src/check_docs.py          # do the documents match the measurements?
python src/check_links.py         # do the links the interface serves go anywhere?
python src/build_results_doc.py --check
python src/build_corpus_manifest.py --check   # does the document list match the indexes?
```

One thing is deliberately outside the count: `python src/smoke_routes.py` asks
every route of an already-running server. `src/test_routes.py` starts its own,
which is why that one *is* counted.

---

## Worth doing next

**1. The dense ensemble is decided, and stays where it is.** Measured end to
end on all three corpora on 2026-08-31, in both directions. It is worse on
every metric on the ML papers, a one-question any-hit gain against an MRR loss
on birds, and earned on quant, which already ships it and loses 0.036 MRR and
0.028 source recall without it. Three answers on three corpora: the fifth
setting here that does not transfer. Nothing changed, and the indexes built for
the experiment were deleted. The reasoning is in `docs/engineering-log.md`
under that date, and the part worth carrying is that the ML candidate pool did
not move at all, so only the ordering handed to the reranker changed.

The engineering cost that made this worth deciding rather than defaulting is
unchanged: a second embedder is a second vector per chunk, so the index roughly
doubles and ingest gains an encode pass. `src/sweep_ensemble.py` reproduces the
pool measurement and `src/evaluate.py` the end-to-end one.

**2. Query decomposition is measured, and it does not work.** Splitting the
question into topic and constraint was this project's named fix for the seven
structural cases for months, on an argument rather than a measurement. Measured
on 2026-08-31 with a local llama3.2 over all 67 answerable cases:
decomposition recovers **none** of the seven and costs 0.045 any-hit at 21
times the latency. Rewriting recovers one and loses three questions net, gaining
two cases and breaking five that worked.

The fixture said the opposite. Against the seven alone, rewriting recovered one
and lost nothing, because every case it breaks is outside the fixture. That is
the trap `src/sweep_decompose.py` warns about in its own docstring, which is
why it scores the whole set by default.

A stronger model may do better, and the harness is in place to find out:
`RAG_GENERATOR=ollama python src/sweep_decompose.py --corpus llm`. What is no
longer available is citing decomposition as the answer without running it.

**3. The other five structural failures need no LLM.** They *describe* a term
and ask for its name, as in "which small group of feathers helps prevent a
stall at low speed", so the answer word is absent from the question. One of the five was
already recovered by changing the embedder alone, which is what motivated the
ensemble work above.

**4. The hero diagram.** All seven stages are drawn as the same sheet-of-cells,
which is wrong in one specific place: dense retrieval and BM25 have identical
geometry in the spec (`layers: 4, cells: [5,4], w: 6.5`) and are opposites in
nature: continuous similarity against sparse term hits. Drawing them the same
hides the reason for running both. Also wanted: a *gate* at the diversity cap
showing passages blocked, and more motion inside the arrays. The
index-as-a-field is right and should stay.

## Ruled out, with numbers

These are kept because a refuted experiment is worth as much as a shipped one,
and all of them are in `docs/engineering-log.md` with the measurements.

- **A better cross-encoder.** Of the three compared, MiniLM-L12 ranks better on
  the ML papers and worse on birds, while BGE-reranker-base ranks worst and
  separates best, so neither is shippable.
- **int8 quantisation of the reranker.** 1.5× faster with near-identical
  rankings, and every score shifts by a systematic −0.22, so the gate silently
  starts refusing answerable questions. Rank-preserving is not enough when a
  threshold reads the score as an absolute.
- **Query rewriting and decomposition.** Decomposition recovers 0 of 7 and
  rewriting recovers 1 while breaking 5 that worked, for a net loss of three
  questions and 0.045 any-hit. Measured over all 67 cases; the fixture alone
  said rewriting was a pure win, because everything it breaks lies outside the
  fixture.
- **Dropping a paper's title block from the candidate pool.** Costs 0.015
  any-hit on the ML papers and 0.029 on quant. The whole of that loss is two
  questions, and both are genuine, because a chunk that opens with a title block
  continues into the abstract and both of these abstracts answer the question
  outright. Dropping the first chunk of a paper drops its abstract, so the
  filter destroys information rather than withdrawing a scoring artefact. It is
  off and it stays off. An audit that first reported those two as false credits
  was wrong for the reason recorded under 2026-08-31 in the log.
- **Pseudo-relevance feedback.** Recovers 0 of 12 structural cases, because the
  feedback documents do not contain the missing word either.
- **Prefixing chunks with their document title.** Built, measured and then
  reverted, because it gained no ranking and cost 0.051 of source recall.
- **Larger candidate pools.** 28 candidates scores *worse* than 20 on all three
  corpora while the pool ceiling rises on all three.

## Traps

- **A corpus setting does not transfer.** Five have now been measured
  per-corpus rather than global: the abstention threshold, the rerank blend, the
  candidate pool size, the choice of embedder, and whether fusing a second
  embedder helps at all. Four of those are configured per corpus in
  `corpora.json`; the embedder was measured per corpus and shipped the same
  everywhere, so `HANDOFF.md` counts four settings and this counts five
  measurements. The 0.0 threshold that costs the ML papers one question costs
  the bird corpus ten.
- **Pool recall is a ceiling, not a proxy.** Four separate experiments have now
  improved what the first stage retrieves and made the finished pipeline worse.
  The fourth is the sharpest: fusing a second embedder leaves the ML papers'
  pool recall *identical* at 0.910 and costs 0.030 any-hit, so the same twenty
  passages arrived at the reranker in a different order and it did worse with
  them. A pool is an ordering, not a set. Measure end to end before shipping
  anything that touches the candidate pool.
- **When a corpus scores badly, read the failing questions first.** The
  quantitative-finance corpus looked like a retrieval weakness at 0.543 and was
  entirely a golden-set problem: textbook definitions asked of research papers.
  0.886 after rewriting the questions, retrieval untouched.
- **A measurement that returns zero deserves as much suspicion as a surprise**,
  and so does one that returns 100%. Two zeros here came from comparing a dict
  locator against its serialised string; a check reporting "67 of 67 cases
  ambiguous" came from subtracting a key that no case has.
- **Render the interface and look at it.** Roughly twenty real defects came from
  screenshotting the page rather than from reading the diff. Check a laptop
  height (~660px) and a phone, not only a wide desktop window.
- **Run the thing.** Every defect found in the last five working sessions was in
  code that had tests *around* it and nothing *running* it: a route that
  ignored its corpus argument, a JSON field of the wrong type that dropped the
  connection, indexing one corpus deleting another's cache, and a module that
  had never been invoked at all.

## Blocked

Nothing is blocked on API credit any more. `RAG_GENERATOR=ollama` runs the
whole generation path against a local model, and `src/evaluate_answers.py`
scores the prose that comes out of it: invented citations, correctness against
the labelled answer string, refusal on adversarial questions, and a lexical
groundedness proxy. There is no LLM judge, deliberately, because a judge model
is a second system whose own failures are invisible and would make this
project's claim that every number is reproducible from the repository false.

Three limits on that measurement, none of which is a missing rubric:

- **The generator is a 3B local model**, so the numbers describe llama3.2 on
  this corpus and not the pipeline's ceiling. A stronger model would need
  either credit or a larger local one.
- **Correctness is a substring test**, so it cannot credit a right answer in
  other words. It is a floor rather than a rate, and the cases it rejects are
  reported in `unmatched` to be read rather than silently scored as wrong.
- **The sample is small.** Answer quality costs about a minute per question on
  a CPU, which is why it samples by default rather than running all 157 cases.

`src/test_evaluate_answers.py` holds the judge to answers it has already got
wrong, which is how both of its first-run defects were found: a plain refusal
scored as an answer, and a correct paraphrase scored as wrong with nothing
saying so.
