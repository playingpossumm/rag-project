# retrieval visualized


A retrieval-augmented generation pipeline in Python, with no framework. Hybrid
dense and BM25 retrieval, reciprocal rank fusion, cross-encoder reranking, a
per-document diversity cap, and a calibrated abstention threshold. Every stage
emits a trace, and the bundled web interface draws it.

The [recorded demo](https://rag-retrieval-visualized.vercel.app) holds all 157
evaluation questions across the three corpora, answered in advance and written
to files. No model runs behind that page, so it answers those 157 and nothing
else, and asking your own question means cloning this repository.

Every default here was chosen by measuring the alternatives on a labelled
question set. `docs/engineering-log.md` records each experiment, including the
ones that were reverted.

```
                       any-hit@5    MRR    NDCG   source recall
  naive RAG                0.821  0.607   0.657           0.678
  + cross-encoder rerank   0.836  0.700   0.720           0.712
  + hybrid BM25 fusion     0.925  0.788   0.808           0.745
  + diversity cap (2/src)  0.910  0.784   0.804           0.785
```

36 arXiv ML/NLP papers, 5,459 passages, and 84 labelled cases: 67 answerable
and 17 adversarial. Reproduce with `python src/evaluate.py`, which writes
[`eval/results.json`](eval/results.json); `python src/build_results_doc.py`
regenerates the tables in [`eval/RESULTS.md`](eval/RESULTS.md) from it, and
`--check` fails when they have drifted.

In the last row the diversity cap trades hit rate for source recall rather than
adding one for free, which an earlier and smaller golden set had reported as
free before the larger set contradicted it. Finding 6 covers how that happened.

## Corpora

The system ships with three sets of documents, chosen to be unlike each other.

| set | documents | passages | formats | refuses below |
|---|---|---|---|---|
| ML & NLP papers | 36 | 5,459 | PDF | 0.0 |
| Ornithology | 45 | 864 | DOCX, PPTX, XLSX, PDF | −5.5 |
| Quantitative finance | 35 | 6,184 | PDF | −4.0 |

Every document in each set is listed with its filename, title, format and
passage count in [`docs/corpus-manifest.md`](docs/corpus-manifest.md), generated from the
indexes themselves. The files are not in this repository;
[`ATTRIBUTION.md`](ATTRIBUTION.md) says why and how to rebuild each set.

The abstention threshold is the score below which the system declines to
answer. It was a module constant for most of this project's life, carrying a
comment that guessed it was "a property of the data, not of the model", and
measuring it across all three sets confirmed the guess. The size of the effect
is what matters.

```
                        wrongly refused at 0.0
  ML & NLP papers        1 of 67    ( 1.5%)
  ornithology            9 of 25    (36.0%)
  quantitative finance   9 of 33    (27.3%)
```

The same number that served the original corpus for its whole life throws away
more than a third of the answers retrieval had already found on the birds and
more than a quarter on the finance papers. This table read 10 of 26 and 19 of 35
until 2026-09-06, figures measured on 2026-08-21 against golden sets that were
rewritten on 2026-08-26 and reclassified on 2026-09-01; the finance figure in
particular counted textbook questions the papers never answer, which
`corpora.json` had already recorded as superseded while this page kept quoting
it. `check_docs.py` now holds these three lines to the measurement.
Each set now carries its own calibrated threshold in
[`corpora.json`](corpora.json), beside the documents it was derived from. A set
with no calibration says so rather than silently borrowing another's.

One question shows what that means in practice. "Which ratio of body mass to
wing area governs flight performance?" scores −0.99, which the ML threshold
refuses and the ornithology threshold answers, and either outcome is correct for
the corpus being asked.

---

## Findings

The evaluation harness was built early, before most of the pipeline, so every
change after it had to be measured against a labelled question set before it
could ship. The six findings below came out of that and would otherwise have
gone unnoticed.

### 1. Chunk truncation

The embedding model caps input at **256 tokens** and truncates the rest with no
error. Chunks were sized in *words* (500), so the median chunk ran to 626 tokens
and **15 of 22 lost roughly 60% of their content** before being embedded. That
text was stored, returned and citable. It could never influence whether its
own chunk was retrieved.

Fixed by chunking on the tokenizer, plus an assertion that refuses to build an
index if any chunk exceeds the ceiling. → [`src/ingest.py`](src/ingest.py)

### 2. Corpus size, and three conclusions that reversed

On one document, hybrid search looked useless, because dense, RRF and weighted
fusion all tied at a perfect score. A 20-candidate pool was 31% of that corpus,
so recall was perfect for any method and the tie meant nothing.

At 20 documents the pool is **0.84%**, and hybrid fusion is worth **+7.6 points of
hit rate**. Three separate conclusions inverted when the corpus grew: whether
fusion helps, which fusion to use, and where the abstention threshold belongs.

A conclusion measured on a corpus that small is not merely imprecise, it can
point the wrong way.

### 3. Ranking metrics against source coverage

For a question five papers answer, the system returned **all five passages from
one paper**. MRR scored it **1.000**; source recall scored it **0.200**.

Ranking metrics call that perfect and by their own definition it is, but the
definition is the wrong one when the goal is to compile every relevant source. A
per-document cap lifts source recall from 0.742 to 0.773.
→ [`src/diversify.py`](src/diversify.py)

### 4. A bug in the evaluation tool

NDCG is normalised and cannot exceed 1.0. Its first run printed **1.373**.
Page-level relevance let several returned chunks share one gold page, so achieved
DCG summed over all of them while the ideal allowed only one.

It was caught only because the output violated a bound the metric is known to
have, and a metric with no known bounds would have shipped wrong and stayed
wrong.

### 5. A feature built, measured and deleted

Chunks carry no document identity, so passages from similar papers look alike,
and prefixing each chunk with its document title should fix that. It was
reported as a small improvement and the report was wrong, because the prefix
arrived alongside a chunk-size change and the two could not be attributed
separately.

Re-measured with one variable *(on the 23-case golden set; superseded by
finding 6, but the direction held under review)*:

```
             any-hit    MRR   NDCG   source recall
  prefix       0.913  0.848  0.855           0.779
  no prefix    0.913  0.848  0.852           0.830
```

No ranking benefit; source recall five points worse. Every chunk in a document
received the *same* prefix, making them more similar to each other and clustering
retrieval harder onto one document, which is the opposite of the goal.
Reverted.
→ [commit `5a82da0`](../../commit/5a82da0)

### 6. Three settled conclusions that were noise

The golden set began at 23 answerable cases, mostly drawn from one paper. Growing
it to **67 cases covering all 36 documents**, a strictly harder test, reversed
three findings that had already been written up as results:

| claim, on 23 cases | on the full set |
|---|---|
| The diversity cap is **free** — no loss to ranking | It **trades**: +4.0 source recall for −1.5 any-hit |
| Window expansion is **strictly dominated** by page expansion | Window reaches 0.833 context recall for **half the tokens** (2,371 vs 4,828) |
| Reranking improves hit rate | It improves **MRR only** (0.601→0.710); any-hit is flat at 0.788 |

The abstention threshold moved too, for the third time. A gap that looked clean
between the two score distributions closed once there were enough cases to see
it, and the constant went 0.0 → 1.5 → back to 0.0.

None of these were bugs. Each was a real measurement, correctly performed, on a
sample too small to support the conclusion drawn from it, and with 23 cases each
one is worth 4.3 points, so differences that looked decisive sat inside the
noise. A test set trusted further than it deserves fails this way, and it is
much harder to notice than a crash.

---

## Architecture

```
  documents ──> parse ──> chunk ──> embed ──> FAISS index
   pdf/docx     +OCR     210 tok    MiniLM        │
   pptx/xlsx   fallback  overlap                  │
                                                  ▼
  question ──> dense retrieval ─┐            candidate pool
           └─> BM25 retrieval ──┴─> fuse ──>  (k per corpus:
                                               16, 20 or 16)
                                                  │
                                    cross-encoder rerank
                                                  │
                                    per-document diversity cap
                                                  │
                                    expand to window +/-1
                                                  ▼
                                   passages + citations
                                   (+ optional generation)
```

Fusion decides what the candidate pool contains and reranking decides its
order, so the two stages do different jobs and are measured separately.
Improving one is invisible in the other's metrics, which is why the harness
reports both.

| Module | Responsibility |
|---|---|
| [`loaders.py`](src/loaders.py) | Per-format parsing with format-appropriate citation locators |
| [`ocr.py`](src/ocr.py) | Per-page OCR fallback for scanned documents |
| [`ingest.py`](src/ingest.py) | Token-bounded chunking, embedding, index build |
| [`retrieve.py`](src/retrieve.py) | Shortlist → rerank → diversify pipeline |
| [`hybrid.py`](src/hybrid.py) | BM25 and rank fusion (RRF / weighted) |
| [`rerank.py`](src/rerank.py) | Cross-encoder second stage |
| [`diversify.py`](src/diversify.py) | Per-document cap for multi-source answers |
| [`abstain.py`](src/abstain.py) | Calibrated confidence gate |
| [`api.py`](src/api.py) | `ask()` — the single entry point |
| [`evaluate.py`](src/evaluate.py) | hit rate, MRR, NDCG, source recall, context recall |
| [`per_case.py`](src/per_case.py) | Per-case outcomes for all 84 golden-set questions, as JSON |
| [`failure_overlap.py`](src/failure_overlap.py) | Which cases fail under *every* configuration, and which move |
| [`hard_cases.py`](src/hard_cases.py) | Runs only the cases nothing currently gets right — seconds, not minutes |
| [`calibrate_threshold.py`](src/calibrate_threshold.py) | Names the questions each abstention threshold would cost |
| [`metadata_filter.py`](src/metadata_filter.py) | Scoped retrieval — constrains the search, not the results |
| [`query_expansion.py`](src/query_expansion.py) | Pseudo-relevance feedback (measured; off by default) |
| [`query_rewrite.py`](src/query_rewrite.py) | LLM rewrite/decomposition — measured 2026-08-31, off by default |
| [`pipeline_trace.py`](src/pipeline_trace.py) | Re-runs retrieval keeping every intermediate ranking |

Two further modules, `late_interaction.py` (ColBERT-style MaxSim reranking) and
`web_fallback.py` (web search when the corpus declines), were built, never wired
into `ask()`, and moved to `archive/` on 2026-09-06; until then this table
listed them as if they were stages of the pipeline.

Three interfaces (Python, HTTP and CLI) are thin shells over one `ask()`
function, so behaviour cannot drift between them.

---

## Design decisions

**Citations are locators, not page numbers.** A page number means nothing for a
spreadsheet. PDFs cite pages, PowerPoint cites slides, Excel cites sheet and row
range, Word cites *sections*. `.docx` pagination is computed by the
renderer and shifts with fonts and margins, so any page number would be wrong on
the reader's copy.

**A per-document cap, not MMR.** MMR diversifies on embedding distance, which
conflates two kinds of redundancy, similar wording and the same source, and it
needs a `lambda` tuned per corpus. Here the unit of redundancy is known exactly
and it is the document.

**RRF over weighted fusion by default.** A cosine similarity and a BM25 score are
not commensurable, so weighted fusion needs per-query normalisation, which is
*relative*. A query where every candidate is mediocre still yields a top
score of 1.0. RRF uses only rank, so it is scale-free with nothing to tune.

**Golden set labels are derived, not written.** Each case declares a distinctive
answer string; every location containing it *becomes* gold. Labels therefore
cannot drift from the corpus. When the corpus grew from 1 to 20 documents, 12 of
16 hand-written adversarial cases had silently become answerable, and nothing
errored. → [`src/build_golden_set.py`](src/build_golden_set.py)

**Retrieval-only by default.** Returning source passages verbatim costs nothing
and cannot hallucinate. For contracts, the exact wording *is* the answer.
Generation is an optional layer, not a dependency.

---

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/activate     # Windows
pip install -r requirements.txt

python src/fetch_corpus.py     # 36 arXiv papers (or drop your own in data/)
python src/ingest.py           # build the index
python src/serve.py            # then open http://127.0.0.1:8000
```

The other two sets are built by topic rather than by a list of paper IDs. A
wrong ID downloads a real paper under a confidently wrong filename, so the
fetcher queries the arXiv and Wikipedia APIs instead:

```bash
python src/fetch_topic.py birds --into data-birds       # writes .docx/.pptx/.xlsx/.pdf
RAG_STORE_DIR=store-birds RAG_DATA_DIR=data-birds python src/ingest.py
python src/per_case.py --golden eval/golden-birds.json   # -> eval/per_case-birds.json
python src/calibrate_threshold.py --per-case eval/per_case-birds.json
```

Until 2026-09-06 this block gave `--topic` and `--out` to a script that takes
the topic as a positional argument and `--into` for the directory, named
`make_documents.py` as a step when `fetch_topic.py` already calls it and it has
no entry point of its own, and gave `calibrate_threshold.py` a `--golden` flag
it does not have. Three of its four lines failed on the first run. It was found
by an audit on 2026-09-06, and every command above has been run as written.

**Measuring.** Every number in this README and on the analytics page comes
from these, and nothing is transcribed by hand:

```bash
python src/evaluate.py         # reproduce every number in this README
python src/per_case.py         # per-case outcomes -> eval/per_case.json
python src/build_analytics.py  # what the pages plot -> eval/analytics.json
python src/hard_cases.py       # only the cases nothing gets right, in ~20s
```

**Seven guards, each exiting non-zero rather than printing a warning nobody
reads.** They exist because a generated file has gone stale silently three
times, once the front page spent four days offering questions that had been
deleted from the golden set for being unanswerable, and once a link on `/about`
named a branch this repository does not have. The seventh reads the recorded
payloads in `static-demo/`, which is gitignored, so on a clone without a
recording it exits 2 and says it did not run rather than passing:

```bash
python src/check_freshness.py     # golden set -> per_case -> analytics -> front page
python src/check_golden.py        # do the labels still describe the corpus?
python src/check_docs.py          # do the documents match the measurements?
python src/check_links.py         # do the links the interface serves go anywhere?
python src/build_results_doc.py --check
python src/build_corpus_manifest.py --check   # does the document list match the indexes?
python src/audit_page_credit.py   # do the hand-read verdicts still cover every page-only credit?
```

**Experiments**, kept because a refuted one is worth as much as a shipped one:

```bash
python src/sweep_fusion.py            # every fusion, every corpus
python src/sweep_candidates.py        # what reranking fewer candidates costs
python src/sweep_ensemble.py          # two embedders fused
python src/compare_rerankers.py       # would a different cross-encoder help?
python src/profile_query.py           # where the time in one query goes
```

**Tests.** All of them run without a network and without an API key:

```bash
python src/test_metrics.py        # the scoring functions, hand-computed
python src/test_trace.py          # the trace and the serving path agree
python src/test_loaders.py        # .docx/.pptx/.xlsx/.pdf round-trips
python src/test_golden.py         # the label audit catches each known fault
python src/test_freshness.py      # the staleness check catches each known fault
python src/test_routes.py         # every HTTP route answers (starts its own server)
python src/test_generate_local.py # generation over a real socket
node  ui/test-answer-mark.mjs     # which words of a passage are set bold
```

**Writing prose from the passages is optional and off by default.** What the
interface shows is the retrieved text, verbatim. To have a model write the
answer instead, point it at a local one. No API key, no account, and no data
leaves the machine:

```bash
ollama serve && ollama pull llama3.2
RAG_GENERATOR=ollama python src/serve.py
```

`RAG_GENERATOR=anthropic` (the default) uses `claude-opus-5` and needs
`ANTHROPIC_API_KEY` in a `.env`. Both go through one `synthesize(question,
chunks)` contract, so the rest of the system does not know which is running.

There is no authentication, and [`SECURITY.md`](SECURITY.md) says what that
means before you point this at documents that are not yours.

The server binds to 127.0.0.1 on purpose and makes no external request at
query time: the documents may be private, and the interface's fonts are
bundled rather than pulled from a CDN for the same reason. The one exception
is a first run with an empty Hugging Face cache, which downloads the models
once; after that `serve.py` sets `HF_HUB_OFFLINE=1` itself so the hub is not
asked again.

Set `RAG_DATA_DIR` to point at any folder, including a Google Drive for Desktop
mount. The interface does the same thing without an environment variable,
through a Local folder tab that takes a pasted path, reports what it would index
and what it would skip, and then indexes it. A browser cannot read a filesystem
path out of a file picker, so pasting a path is the only form of folder picker
available here.

---

## The interface

`python src/serve.py`, then <http://127.0.0.1:8000>.

There are three pages. `/` asks a question, `/about` says why the project
exists and what every term means, and `/quality` carries every measurement
behind it.

Pick a set of documents, ask a question, and the answer arrives with the passage
it came from, cited to the page, slide, or spreadsheet rows. A passage pulled
out of a spreadsheet is shown as the table it came from rather than as a
paragraph with the column headers left in the middle of the sentence.

Underneath the answer is the retrieval result, with the score given in the units
it is actually in, which is a cross-encoder logit running roughly −11 to +11
rather than a probability. The same panel shows which documents were drawn on,
how much document text was read, and how long it took.

Below that the search is drawn as a schematic of seven stages, each a block of
sheets whose depth tracks how many candidates are still in play, so the fall
from 5,459 passages to five is the shape of the picture rather than a number
written under it. Colour carries one meaning only, which is that a passage is in
your answer.

Every answer states what it was run with, and the four retrieval settings can be
changed from there; changing one re-runs the same question and shows what moved.
Each carries the measurement that justifies its default.

---

## Limitations

**Every number here is a property of this corpus, not of the code.** Three tuned
parameters had to be re-derived when the corpus changed, and one conclusion
reversed outright. Pointing this at different documents means re-running the
harness.

- **Five questions fail under every pipeline configuration**, and six
  approaches have now been measured against them without one shipping. They are
  `gpt3-params`, `roberta-nsp-drop` and `wmt14` on the papers, and
  `bird-hollow-bones` and `bird-precocial` on the birds; the finance corpus has
  none. Four of the five never reach the reranker at all, because RRF rewards
  agreement between the two retrievers and on these the two disagree sharply:
  BM25 ranks `wmt14`'s answer 11th and the embedder ranks it 138th. Refuted
  against them on the corrected labels: a deeper candidate pool, every rerank
  blend, three cross-encoders, scoring the best window inside a chunk, and
  retrieving on a hypothetical answer. The last of those recovers three of the
  five and loses nine other questions on the papers, which is the closest
  anything has come and still not close. `docs/engineering-log.md` under
  2026-09-01 has the per-stage diagnosis and every measurement.
- **That number was twelve until the twelve were read one at a time.** Four were
  cases the pipeline answers and the label scored wrong, because a derived label
  marks every passage containing its answer string rather than every passage
  that answers. `t5-text2text` is the clearest: "unified text-to-text" is part
  of the T5 paper's title, so four of its six gold passages were other papers'
  bibliographies, while the passage reading "we cast all of the tasks we consider
  into a text-to-text format" was not gold and is what comes back. Three more
  were questions the documents do not answer, where the term appears only as a
  factor name or in a list of anatomical features. The labels were corrected on
  2026-09-01 and `src/failure_overlap.py`, re-derived from six configurations
  per corpus, independently returns the same five. The reading and every number
  that moved with it are in `docs/engineering-log.md` under that date.
- **The reranker is the weakest stage and off-the-shelf options are exhausted.**
  Three cross-encoders were compared and int8 quantisation measured; the
  candidates that are faster are worse, and the one that separates best is
  dramatically slower and ranks worse. What remains is a fine-tune on 157
  labelled cases, which is probably too few.
- **Answer quality is measured on every case, all 157, with a 3B local
  model.** `src/evaluate_answers.py` scores the generated prose without an LLM
  judge, on the argument that a judge model is a second system whose own
  failures are invisible. Until 2026-09-05 it had been run on a 45-answer
  sample, 15 per corpus; the full run took 3.9 hours on a CPU.

  | | ML papers | Ornithology | Quant |
  |---|---|---|---|
  | invented citations | 0 in 84 | 1 in 32 | 1 in 41 |
  | refused when it should | 11/17 | 7/7 | 6/8 |
  | refused when it should not | 4/67 | 10/25 | 4/33 |
  | contains the labelled answer | 37/67 | 9/25 | 12/33 |
  | contains it, allowing other words | 43/67 | 9/25 | 16/33 |
  | groundedness (proxy) | 0.705 | 0.475 | 0.606 |

  **Two invented citations in 157 answers**, one on each of the smaller
  corpora and none on the papers. The 45-answer sample had reported none, and
  a fabricated citation is the failure that matters most, since it is worse
  than no answer. Correctness is reported twice, because the strict figure is
  a substring test and a right answer in other words counts against it. The
  looser figure credits an answer that carries 80% of the labelled string's
  content words in any order, and it is an upper bound of the same kind that
  the strict figure is a floor. The 67 answers the strict test rejects are
  listed in `unmatched` to be read rather than scored. Groundedness is lexical
  overlap, reported because it is cheap and directional, not as a verdict.

  The judge itself was wrong four times before these numbers settled, each one
  found by reading the answers rather than the code: a plain refusal scored as
  an answer, mathematics scored as fabricated citations, a hedge-then-answer
  scored as a refusal, and a model describing the corpus instead of answering
  scored as answering. `src/test_evaluate_answers.py` holds it to 79 checks,
  most of them real answers this repository has already scored wrongly.

  The generator is brittle at this size: changing one word of the prompt from
  "Context:" to "Excerpts:" is the difference between a citation with no prose
  and a correct answer. The bird corpus is where it is weakest, refusing 10 of
  the 25 questions its documents answer and grounding under half of its
  wording in the passages it was given.
  → [`eval/answer-quality.json`](eval/answer-quality.json)

  **A larger model was tried and did not help.** llama3.1:8b, the same family
  at roughly 2.7x the parameters, was compared over the 43 cases both models
  answered. It gets 16 of 28 answerable questions right against 14, which is two
  questions on a sample where finding 6 above says two questions are noise, and
  every other measure moves the other way. It refuses 12 of 15 adversarial
  questions against 13, emits one invented citation and two malformed ones where
  the 3B emits none, scores 0.476 on the groundedness proxy against 0.588, and
  takes 5,665 seconds against 98. An earlier partial run over 11 answers on one
  corpus reported the opposite direction on two of those three counts, which is
  what a sample that size is worth.
  → [`eval/answer-quality-8b.json`](eval/answer-quality-8b.json)
- **157 cases across three corpora is still small.** On the 25 answerable bird
  questions each is worth 4.0 points, so a one-question difference looks like a
  result and is not. Treat small differences as noise: a 23-case set
  earlier in this project produced three false conclusions (finding 6), and
  assume these are hiding others.
- **Labels were authored by the same process that built the system.** Mitigated by
  deriving them from the corpus and by auditing them structurally
  (`src/check_golden.py`), not eliminated.
- **One measured instance of that, and the count it came to.** The labels mark
  correct pages rather than passages, and a paper's title block sits on page 1,
  so page 1 is a gold page for 43 of the 100 answerable cases in the two corpora
  that have title pages, with the answer string inside the title itself for 17
  of them. `bn-covariate` asks what normalizing layer inputs addresses and its
  paper is titled "... by Reducing Internal Covariate Shift", which the scoring
  would accept. `src/audit_title_credit.py` measures how much of that reach is
  collected and finds **two hits satisfied only by a chunk that opens with a
  title block, and both of those chunks contain the answer**, because what
  follows a title block is the abstract. False credits: **zero**. The reach is
  real and the cost today is nothing, which is a distinction the figures above
  depend on and no aggregate can show.
- **Every corpus needs its own tuning.** Five settings have now been measured as
  per-corpus rather than global: the abstention threshold, the rerank blend,
  the candidate pool size, the choice of embedder, and whether fusing a second
  embedder helps at all. Four of the five are configured per corpus in
  `corpora.json`; the choice of embedder was measured per corpus and shipped
  the same everywhere, which is why `HANDOFF.md` counts four and this counts
  five. The last of those was settled end to end on
  2026-08-31: worse on every metric on the ML papers, a one-question trade on
  the birds, and earned on quantitative finance, which is the only corpus that
  ships it. Pointing this at your own documents means re-running the
  harness, not just re-indexing.

---

## Further reading

- [`eval/RESULTS.md`](eval/RESULTS.md) — full measurements and methodology
- [`eval/golden_set.json`](eval/golden_set.json) — 84 labelled cases
