# retrieval visualized


A retrieval-augmented generation pipeline in Python, with no framework. Hybrid
dense and BM25 retrieval, reciprocal rank fusion, cross-encoder reranking, a
per-document diversity cap, and a calibrated abstention threshold. Every stage
emits a trace, and the bundled web interface draws it.

**[Recorded demo](https://rag-retrieval-visualized.vercel.app)**. All 157
evaluation questions across three corpora, answered in advance and written to
files. No model runs behind that page, so it answers only those 157. Clone this
repository to ask your own.

Every default here was chosen by measuring the alternatives on a labelled
question set. `docs/engineering-log.md` records each experiment, including the
ones that were reverted.

```
                       any-hit@5    MRR    NDCG   source recall
  naive RAG                0.791  0.581   0.629           0.664
  + cross-encoder rerank   0.806  0.666   0.688           0.688
  + hybrid BM25 fusion     0.866  0.743   0.758           0.721
  + diversity cap (2/src)  0.851  0.739   0.755           0.761
```

36 arXiv ML/NLP papers, 5,459 passages, and 84 labelled cases: 67 answerable
and 17 adversarial. Reproduce with `python src/evaluate.py`, which writes
[`eval/results.json`](eval/results.json); `python src/build_results_doc.py`
regenerates the tables in [`eval/RESULTS.md`](eval/RESULTS.md) from it, and
`--check` fails when they have drifted.

Note the last row: the diversity cap **trades** hit rate for source recall rather
than adding one for free. An earlier, smaller golden set said it was free. It was
wrong. See finding 6.

## Three sets of documents, and one threshold per set

The system ships with three corpora, deliberately unlike each other:

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
answer. It was a module constant for most of this project's life, with a comment
guessing it was "a property of the data, not of the model". Measured across all
three, that guess is right, and the size of it is the finding:

```
                        wrongly refused at 0.0
  ML & NLP papers        1 of 66    ( 1.5%)
  ornithology           10 of 26    (38.5%)
  quantitative finance  19 of 35    (54.3%)
```

The same number that served the original corpus for its whole life throws away
more than a third of the answers retrieval had already found on the other two.
Each set now carries its own calibrated threshold in
[`corpora.json`](corpora.json), beside the documents it was derived from. A set
with no calibration says so rather than silently borrowing another's.

One question makes it concrete. *"Which ratio of body mass to wing area governs
flight performance?"* scores **−0.99**. That is refused under the ML
threshold and answered under the ornithology one, and it is correct either way
for the corpus being asked.

---

## What the measurements changed

Most RAG tutorials produce a pipeline and stop. There is no way to tell whether
any individual piece helps, so techniques accumulate on faith. This project
inverts that: an evaluation harness came early, and **every subsequent change had
to earn its place against a number.**

That discipline produced findings that would otherwise have shipped silently.

### 1. Most of every chunk was invisible to search

The embedding model caps input at **256 tokens** and truncates the rest with no
error. Chunks were sized in *words* (500), so the median chunk ran to 626 tokens
and **15 of 22 lost roughly 60% of their content** before being embedded. That
text was stored, returned and citable. It could never influence whether its
own chunk was retrieved.

Fixed by chunking on the tokenizer, plus an assertion that refuses to build an
index if any chunk exceeds the ceiling. → [`src/ingest.py`](src/ingest.py)

### 2. A conclusion measured on a small corpus was backwards

On one document, hybrid search looked useless. Dense, RRF and weighted fusion
all tied at a perfect score. But a 20-candidate pool was **31% of that corpus**,
so recall was trivially perfect for any method.

At 20 documents the pool is **0.84%**, and hybrid fusion is worth **+7.6 points of
hit rate**. Three separate conclusions inverted when the corpus grew: whether
fusion helps, which fusion to use, and where the abstention threshold belongs.

**A conclusion measured on a toy corpus may not merely be imprecise. It may be
backwards.**

### 3. The metrics reported success on questions the system could not answer

For a question five papers answer, the system returned **all five passages from
one paper**. MRR scored it **1.000**; source recall scored it **0.200**.

Ranking metrics call that perfect, and by their definition it is. That is the
wrong definition when the goal is to compile every relevant source. A per-document cap
lifts source recall from 0.742 to 0.773. → [`src/diversify.py`](src/diversify.py)

### 4. The evaluation tool had a bug in it

NDCG is normalised and cannot exceed 1.0. Its first run printed **1.373**.
Page-level relevance let several returned chunks share one gold page, so achieved
DCG summed over all of them while the ideal allowed only one.

Caught only because the output violated a bound the metric is known to have. **A
metric with no known bounds would have shipped wrong and stayed wrong.**

### 5. A feature was built, measured, and deleted

Chunks carry no document identity, so passages from similar papers look alike.
Prefixing each chunk with its document title should fix that, and it was reported
as a small improvement. **That report was wrong.** The prefix arrived alongside a
chunk-size change, and the two could not be attributed separately.

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
→ [commit `d4ea4ab`](../../commit/d4ea4ab)

### 6. Three "settled" conclusions were noise from too small a test set

The golden set began at 23 answerable cases, mostly drawn from one paper. Growing
it to **67 cases covering all 36 documents**, a strictly harder test, reversed
three findings that had already been written up as results:

| claim, on 23 cases | on the full set |
|---|---|
| The diversity cap is **free** — no loss to ranking | It **trades**: +3.1 source recall for −1.6 any-hit |
| Window expansion is **strictly dominated** by page expansion | Window reaches 0.833 context recall for **half the tokens** (2,371 vs 4,828) |
| Reranking improves hit rate | It improves **MRR only** (0.601→0.710); any-hit is flat at 0.788 |

The abstention threshold moved too, for the third time. A gap that looked clean
between the two score distributions closed once there were enough cases to see
it, and the constant went 0.0 → 1.5 → back to 0.0.

**None of these were bugs.** Every one was a real measurement, correctly
performed, on a sample too small to support the conclusion drawn from it. With 23
cases each one is worth 4.3 points, so differences that looked decisive were
inside the noise. That is the failure mode a test set produces when it is trusted
more than it deserves, and it is far harder to notice than a crash.

---

## Architecture

```
  documents ──> parse ──> chunk ──> embed ──> FAISS index
   pdf/docx     +OCR     240 tok    MiniLM        │
   pptx/xlsx   fallback  overlap                  │
                                                  ▼
  question ──> dense retrieval ─┐            candidate pool
           └─> BM25 retrieval ──┴─> fuse ──>    (k=20)
                                                  │
                                    cross-encoder rerank
                                                  │
                                    per-document diversity cap
                                                  │
                                    expand to full page
                                                  ▼
                                   passages + citations
                                   (+ optional generation)
```

Two stages with different jobs, measured separately: **fusion decides what the
candidate pool contains; reranking decides its order.** Improving one is invisible
in the other's metrics, so the harness reports both.

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
| [`late_interaction.py`](src/late_interaction.py) | ColBERT-style MaxSim reranking (measured; off by default) |
| [`query_rewrite.py`](src/query_rewrite.py) | LLM rewrite/decomposition — written, never run |
| [`web_fallback.py`](src/web_fallback.py) | Optional web search when the corpus declines — off |
| [`pipeline_trace.py`](src/pipeline_trace.py) | Re-runs retrieval keeping every intermediate ranking |

Three interfaces (Python, HTTP and CLI) are thin shells over one `ask()`
function, so behaviour cannot drift between them.

---

## Design decisions

**Citations are locators, not page numbers.** A page number means nothing for a
spreadsheet. PDFs cite pages, PowerPoint cites slides, Excel cites sheet and row
range, Word cites *sections*. `.docx` pagination is computed by the
renderer and shifts with fonts and margins, so any page number would be wrong on
the reader's copy.

**A per-document cap, not MMR.** MMR diversifies on embedding distance, conflating
two kinds of redundancy, similar wording and same source, and needs a
`lambda` tuned per corpus. Here the unit of redundancy is known exactly: the document.

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
python src/fetch_topic.py --topic birds --out data-birds
python src/make_documents.py --src data-birds   # .docx/.pptx/.xlsx/.pdf
RAG_STORE_DIR=store-birds RAG_DATA_DIR=data-birds python src/ingest.py
python src/calibrate_threshold.py --golden eval/golden-birds.json
```

**Measuring.** Every number in this README and on the analytics page comes
from these, and nothing is transcribed by hand:

```bash
python src/evaluate.py         # reproduce every number in this README
python src/per_case.py         # per-case outcomes -> eval/per_case.json
python src/build_analytics.py  # what the pages plot -> eval/analytics.json
python src/hard_cases.py       # only the cases nothing gets right, in ~20s
```

**Six guards, each exiting non-zero rather than printing a warning nobody
reads.** They exist because a generated file has gone stale silently three
times, once the front page spent four days offering questions that had been
deleted from the golden set for being unanswerable, and once a link on `/about`
named a branch this repository does not have:

```bash
python src/check_freshness.py     # golden set -> per_case -> analytics -> front page
python src/check_golden.py        # do the labels still describe the corpus?
python src/check_docs.py          # do the documents match the measurements?
python src/check_links.py         # do the links the interface serves go anywhere?
python src/build_results_doc.py --check
python src/build_corpus_manifest.py --check   # does the document list match the indexes?
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

The server binds to 127.0.0.1 on purpose and makes no external request: the
documents may be private, and the interface's fonts are bundled rather than
pulled from a CDN for the same reason.

Set `RAG_DATA_DIR` to point at any folder, including a Google Drive for
Desktop mount. The UI does the same thing without an environment variable: the **Local
folder** tab takes a pasted path, reports what it would index and what it would
skip, and then indexes it. A browser cannot read a filesystem path out of a file
picker, so pasting a path is the only version of a folder picker available
here.

---

## The interface

`python src/serve.py`, then <http://127.0.0.1:8000>.

Three pages: **`/`** to ask, **`/about`** for why the project exists and what
every term means, **`/quality`** for every measurement behind it.

Pick a set of documents, ask a question, and the answer arrives with the passage
it came from, cited to the page, slide, or spreadsheet rows. A passage pulled
out of a spreadsheet is shown as the table it came from rather than as a
paragraph with the column headers left in the middle of the sentence.

Underneath it, the retrieval result: the score, read in the units it is actually
in. That is a cross-encoder logit running roughly −11 to +11, not a
probability. The panel also shows which documents were drawn on, how much document text was read, and
how long it took.

Below that the search is drawn as a schematic: seven stages, each a block of
sheets whose depth tracks how many candidates are still in play, so the funnel
from 5,459 passages down to five is the shape of the picture rather than a
number written under it. Colour means one thing: this passage is in your
answer.

Every answer states what it was run with, and the four retrieval settings can be
changed from there; changing one re-runs the same question and shows what moved.
Each carries the measurement that justifies its default.

---

## Limitations

**Every number here is a property of this corpus, not of the code.** Three tuned
parameters had to be re-derived when the corpus changed, and one conclusion
reversed outright. Pointing this at different documents means re-running the
harness.

- **Twelve questions fail under every pipeline configuration**, and they are two
  different problems. Seven, on the ML papers, ask about an attribute many
  papers share. The topic matches a dozen documents and the clause that picks
  one out is ignored. The other five *describe* a term and ask for its name
  ("which small group of feathers helps prevent a stall at low speed"), so the
  answer word is absent from the question. The first wants query decomposition;
  the second is vocabulary mismatch, and one of the five was recovered by
  changing the embedder alone.
- **The reranker is the weakest stage and off-the-shelf options are exhausted.**
  Three cross-encoders were compared and int8 quantisation measured; the
  candidates that are faster are worse, and the one that separates best is
  dramatically slower and ranks worse. What remains is a fine-tune on 157
  labelled cases, which is probably too few.
- **Answer quality is measured, but on nine cases and one small model.**
  `src/evaluate_answers.py` scores the generated prose without an LLM judge:
  zero invented citations across nine answers, 2 of 3 adversarial questions
  refused, and no answerable question refused wrongly. Correctness is 3 of 6,
  which is a floor rather than a rate, since it is a substring test and cannot
  credit a correct paraphrase. Groundedness is a lexical-overlap proxy at
  0.602, not a verdict. The generator is a 3B local model and is brittle:
  changing one word of the prompt from "Context:" to "Excerpts:" is the
  difference between a citation with no prose and a correct answer. Nine cases
  on one corpus is too few to conclude much. → [`eval/answer-quality.json`](eval/answer-quality.json)
- **157 cases across three corpora is still small.** On the 26-case bird set each
  answerable question is worth ~3.8 points, so a one-question difference looks
  like a result and is not. Treat small differences as noise: a 23-case set
  earlier in this project produced three false conclusions (finding 6), and
  assume these are hiding others.
- **Labels were authored by the same process that built the system.** Mitigated by
  deriving them from the corpus and by auditing them structurally
  (`src/check_golden.py`), not eliminated.
- **Every corpus needs its own tuning.** Five settings have now been measured as
  per-corpus rather than global: the abstention threshold, the rerank blend,
  the candidate pool size, the choice of embedder, and whether fusing a second
  embedder helps at all. Pointing this at your own documents means re-running the
  harness, not just re-indexing.

---

## Further reading

- [`eval/RESULTS.md`](eval/RESULTS.md) — full measurements and methodology
- [`eval/golden_set.json`](eval/golden_set.json) — 84 labelled cases
