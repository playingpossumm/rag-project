# Roadmap

Current as of 2026-09-06. `HANDOFF.md` carries the reasoning behind every
decision named here.

---

## Current state

There are three corpora, each measured separately and each with its own
tuning. Running `python src/serve.py` and opening
<http://127.0.0.1:8000/quality> gives the live figures, of which the table below
is a snapshot.

| | documents | passages | any-hit@5 |
|---|---|---|---|
| ML & NLP papers | 36 | 5,459 | 0.910 |
| Ornithology | 45 | 864 | 0.880 |
| Quantitative finance | 35 | 6,184 | 0.939 |

**605 checks**, none of which need a network or an API key:

```bash
python src/test_metrics.py        # the scoring functions, hand-computed
python src/test_trace.py          # the trace and the serving path agree
python src/test_routes.py         # every HTTP route answers (starts its own server)
python src/test_generate_local.py # generation, over a real socket
python src/test_evaluate_answers.py  # the answer-quality judge, on answers it has scored wrongly
python src/test_links.py          # the link checker, against the links that broke
node  ui/test-answer-mark.mjs     # which words of a passage are set bold
node  ui/test-passages.mjs        # whether a reader can see the passage that answers
```

**Seven guards**, each exiting non-zero rather than printing a warning nobody
reads. They exist because a generated file has gone stale silently three times,
and because a link on `/about` named a branch this repository does not have.
The seventh reads the recorded payloads in `static-demo/`, which is gitignored,
so on a clone without a recording it exits 2 and says it did not run rather
than passing:

```bash
python src/check_freshness.py     # golden set -> per_case -> analytics -> front page
python src/check_golden.py        # do the labels still describe the corpus?
python src/check_docs.py          # do the documents match the measurements?
python src/check_links.py         # do the links name this repository, and resolve in it?
python src/build_results_doc.py --check
python src/build_corpus_manifest.py --check   # does the document list match the indexes?
python src/audit_page_credit.py   # do the hand-read verdicts still cover every page-only credit?
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

This item said until 2026-09-06 that a stronger model might do better and that
the harness was in place to find out. It was run on 2026-09-05 with
llama3.1:8b, a model three times the size of the llama3.2 3B the earlier runs
used, on all three corpora, and the larger model does not do better.
Decomposition, in `eval/decompose-sweep-8b.json`, recovers 0 of the 3
structural cases on the papers and takes any-hit from 0.910 to 0.895, recovers
1 of 2 on the birds while taking any-hit from 0.880 to 0.840, at 44 to 56
seconds a query across the three corpora, and takes quantitative finance
from 0.939 to 0.909. Rewriting on the same run reaches
0.866, 0.880 and 0.849 against the same controls. Retrieving on an invented
answer, in `eval/hyde-sweep-8b.json`, reaches 0.716 and 0.806 on the papers
against the 3B model's 0.806 and 0.851, gains 2 questions on the birds at
0.920 while losing 1 and letting 3 more adversarial questions through the
gate, and falls to 0.848 and 0.818 on quantitative finance. The larger model
writes a longer and more assured invention, not a more accurate one, and the
entry in `docs/engineering-log.md` under 2026-09-05 has every figure. What is
no longer available is citing decomposition, or a larger model behind it, as
the answer.

**3. The frontier is 23 questions with one cause.** Until 2026-09-03 this item
said the frontier was 5 questions, and that was too narrow. It counted only the
questions that fail under every configuration, which `src/failure_overlap.py`
returns from 6 configurations on each corpus, and treated the misses on the
shipped configuration as retrieval already counted elsewhere and the questions
credited on the locator alone as a scoring detail. All three groups are the
same failure. On 23 questions across the three corpora the chunk that answers
exists in the corpus and is in no retrieved chunk, and `src/answer_distance.py`
lists them. They are the 11 questions the shipped configuration scores as
misses, among them the 5 that fail everywhere, `gpt3-params`,
`roberta-nsp-drop` and `wmt14` on the papers and `bird-hollow-bones` and
`bird-precocial` on the birds, and 12 questions scored as hits because the page
returned holds the answer in a chunk that was not returned.
`src/audit_page_credit.py` reads every hit whose excerpt in the recorded
payload lacks the answer string, 14 by that comparison, records that 9 do not
answer, 2 answer in other words and 3 are arguable, and fails if the population
moves. Its 14 and the 12 above differ because the audit compares against the
excerpt the page shows and `answer_visible` against the retrieved chunk, and on
`pos-enc-fn` and `qf-degeneracy` the chunk holds the answer past the end of the
excerpt. Until 2026-09-05 this item, following the log entry of 2026-09-03 it
was written from, gave the 23 as the misses and the 9 that do not answer, and
those groups sum to 20.

The passage that answers does not contain the words of the question, which is
why the search missed it, so every mechanism that ranks, places or marks by
question vocabulary moves away from it, and giving that mechanism more room to
look moves it further. `wmt14` is the sharpest case, returning
English-to-French passages for a question about English-to-German, so the
single word that decides the answer is the one ignored. `src/answer_distance.py`
measures where the answering chunk sits. On 10 of the 23 it is 1 chunk from a
retrieved chunk, on 3 it is 2 chunks away, on 8 it is 3 to 37 chunks off in a
retrieved document, and on 2 it is in another document. `src/sweep_neighbours.py`
and `src/sweep_stride.py` reach the neighbouring chunk by 3 arrangements.
Neighbours in the pool and the finer stride each gain 1 question, the same one,
`bird-fledging`, and window scoring is worse on every metric but the gate,
because the reranker reads the answering chunk against the same question and
reaches the conclusion the first stage did. `src/hyde_trigger.py` measures the
11 scored misses, not the locator-only credits, and shows that no cheap signal
separates them from the questions that work. What is left is supplying the
missing vocabulary, which only a model that already knows the answer or a
person writing a different question can do, and neither is a retrieval result.

The measurement moved with the framing. `src/evaluate.py` reports
`answer_visible` beside `hit_rate`, because a hit credits the locator, and a
page can hold the answer in a paragraph that was not returned. On the shipped
configuration the answer is in a retrieved chunk on 0.821 of the papers'
questions, 0.760 of the birds' and 0.848 of quant's, against any-hit of 0.910,
0.880 and 0.939. The excerpt the page shows is capped at 720 characters, raised
from 420 on 2026-09-03, which took the share of answers on the page across the
three corpora from 0.656 to 0.808 with retrieval untouched. At 420 the page
was cutting off answers the system had retrieved on 21 questions, and the 4
that had been read as "the passage does not answer" all had the answer in the
passage past the edge of what was shown.

**4. The hero diagram.** All seven stages are drawn as the same sheet-of-cells,
which is wrong in one specific place: dense retrieval and BM25 have identical
geometry in the spec (`layers: 4, cells: [5,4], w: 6.5`) and are opposites in
nature: continuous similarity against sparse term hits. Drawing them the same
hides the reason for running both. Also wanted: a *gate* at the diversity cap
showing passages blocked, and more motion inside the arrays. The
index-as-a-field is right and should stay.

## Ruled out, with numbers

These are kept because a refuted experiment is worth as much as a shipped one,
and all of them are in `docs/engineering-log.md` with the measurements. The
first 4 were measured on 2026-09-03 against the 23 questions of item 3, and the
3 that put the neighbouring chunk in front of the ranker fail for the reason
given there.

- **Neighbours in the candidate pool.** Pulling each candidate's adjacent
  chunks into the pool before reranking, so the cross-encoder can promote the
  chunk next door on its own merits, gains 1 question, `bird-fledging`, which
  takes answer shown on the birds from 0.760 to 0.800. Any-hit is unchanged on
  all three corpora, MRR falls from 0.784 to 0.772 on the papers and from 0.826
  to 0.811 on quant, 1 more adversarial case slips the gate on the papers, 8
  to 9, and latency is 2.5 times what it was. The answering chunk was in the
  pool and the cross-encoder read it and did not promote it. The measurement is
  `src/sweep_neighbours.py`, run on 2026-09-03.
- **Scoring a chunk on the window around it.** Ranking each candidate on the
  chunk before, the chunk itself and the chunk after, while still returning the
  chunk, takes any-hit from 0.910 to 0.881 on the papers, from 0.880 to 0.840
  on the birds and from 0.939 to 0.818 on quant. It cuts adversarial slips on
  the papers from 8 to 5 because a wider passage dilutes every match and lowers
  every score, which is a threshold shift rather than an improvement, and it
  costs 7 answerable questions on the papers to buy it. `src/sweep_neighbours.py`
  holds this arm as well, measured on 2026-09-03.
- **A finer stride.** Chunking the bird corpus at an overlap of 105 tokens
  rather than 40 adds 32% more chunks and, with the pool scaled to match,
  takes any-hit from 0.880 to 0.840 and answer shown from 0.760 to 0.720. It
  recovers the same question the neighbours arm recovers. The scratch index
  was deleted and `src/sweep_stride.py` rebuilds it. The distance measurement
  first named a larger chunk rather than a shorter stride as the indicated
  experiment, on the reasoning that overlap cannot extend a chunk forwards,
  which was true and beside the point, because the answers 1 chunk away were
  already inside the next chunk and what is missing is a chunk holding both the
  words that find the passage and the words that answer.
- **Showing each passage with its neighbours.** The pipeline can attach the
  adjacent chunks to each result and the page does not use it. Turning it on
  and rendering it as `completed()` does takes the share of answers on the page
  from 0.808 to 0.832 at 591 to 836 characters a passage, and a symmetric
  lead-out reaches 0.840 at 1,073 characters, so 4 questions cost nearly twice
  the text on every answer. Showing the raw expanded window reaches 0.880 and
  multiplies the text on screen by 3.2. Choosing the 720-character excerpt over
  the expanded window rather than over the chunk scores 0.792 against 0.808,
  because with more room `_brief` places the window by where the question's
  words fall and moves away from the chunk that was ranked. The page shows
  0.808 against 0.816 in the retrieved chunks, so the display is at its
  ceiling and none of these arrangements shipped.
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
- **Retrieving on a hypothetical answer.** Generating the passage from the
  model's own memory and searching on that recovers three of the five remaining
  failures, which nothing else has done, and costs 9 questions on the ML papers
  and 4 on quant. It also triples the adversarial cases slipping the gate on
  two corpora, because a hypothetical answer reads just as confident for a
  question the documents cannot answer. `src/sweep_hyde.py`, measured
  2026-09-01, off by default. Running it **only** when the pipeline is already
  below its abstention threshold, so that it cannot disturb a question that
  works, was measured on 2026-09-03 and does not fire at all: every one of the
  11 questions the three corpora get wrong scores above its own corpus's
  threshold, the highest at +6.36. Neither does any cheaper trigger, since
  coverage, rerank margin and retriever agreement all put the failing questions
  inside the range of the working ones. `src/hyde_trigger.py`.
- **A candidate pool of 28 or 40.** 28 scores 0.895 on the ML papers against
  0.910 at 16. 40 recovers `gpt3-params` and loses `lora-frozen`.
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
  0.886 after rewriting the questions, retrieval untouched, and 0.939 after
  two questions the corpus does not answer moved to the adversarial half on
  2026-09-01.
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

One thing is blocked on a machine. The Dockerfile's `HF_HOME` ordering was
fixed on 2026-09-07 and the image has not been built since, because Docker
is not on the development laptop and that laptop is a managed work device
on which installing it was started and then stopped the same day. The build
and its two-command check wait for a personal machine; they are under
"Check the build" in `docs/deploying.md`.

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
  Since 2026-09-05 a second figure, `correct_loose`, is reported beside it and
  credits an answer carrying 80% of the labelled string's content words in any
  order; on the full run it moves the papers from 37 to 43 of 67 and finance
  from 12 to 16 of 33, and the birds not at all.
- **The sample was small, and is not any more.** Answer quality costs about a
  minute per question on a CPU, so the script samples by default, and until
  2026-09-05 every published figure came from 15 answers per corpus. The full
  run of all 157 took 3.9 hours and is what `eval/answer-quality.json` and the
  analytics page now carry. It found two invented citations where the sample
  had found none.

`src/test_evaluate_answers.py` holds the judge to answers it has already got
wrong, which is how both of its first-run defects were found: a plain refusal
scored as an answer, and a correct paraphrase scored as wrong with nothing
saying so.
