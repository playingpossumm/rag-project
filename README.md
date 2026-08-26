# retrieval visualized

A retrieval-augmented generation system you can watch work. Ask a question of a
set of documents and the interface draws the whole search — what dense retrieval
and BM25 each found, what fusion and the cross-encoder did to the ranking, what
the diversity cap cut, and which passages the answer actually stands on, cited
down to the page, slide or spreadsheet row.

**The point of this repository is not that it implements RAG. It is that every
design decision in it was measured — and several conclusions that had already
been written up as results turned out to be wrong.**

```
                       any-hit@5    MRR    NDCG   source recall
  naive RAG                0.791  0.581   0.629           0.664
  + cross-encoder rerank   0.791  0.688   0.698           0.700
  + hybrid BM25 fusion     0.866  0.741   0.754           0.724
  + diversity cap (2/src)  0.851  0.738   0.754           0.760
```

36 arXiv ML/NLP papers, 5,459 passages, 84 labelled cases — 67 answerable and
17 adversarial. Reproduce with `python src/evaluate.py`, which writes
[`eval/results.json`](eval/results.json); `python src/build_results_doc.py`
regenerates the tables in [`eval/RESULTS.md`](eval/RESULTS.md) from it, and
`--check` fails when they have drifted.

Note the last row: the diversity cap **trades** hit rate for source recall rather
than adding one for free. An earlier, smaller golden set said it was free. It was
wrong — see finding 6.

## Three sets of documents, and a threshold that does not transfer

The system ships with three corpora, deliberately unlike each other:

| set | documents | passages | formats | refuses below |
|---|---|---|---|---|
| ML & NLP papers | 36 | 5,459 | PDF | 0.0 |
| Ornithology | 45 | 900 | DOCX, PPTX, XLSX, PDF | −3.0 |
| Quantitative finance | 35 | 6,184 | PDF | −4.0 |

The abstention threshold — the score below which the system declines to answer
— had been a module constant for most of this project's life, with a comment
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
flight performance?"* scores **−0.99** — refused under the ML threshold,
answered under the ornithology one, and correct either way for the corpus being
asked.

---

## Why this exists

Most RAG tutorials produce a pipeline and stop. There is no way to tell whether
any individual piece helps, so techniques accumulate on faith. This project
inverts that: an evaluation harness came early, and **every subsequent change had
to earn its place against a number.**

That discipline produced findings that would otherwise have shipped silently.

### 1. Most of every chunk was invisible to search

The embedding model caps input at **256 tokens** and truncates the rest with no
error. Chunks were sized in *words* (500), so the median chunk ran to 626 tokens
and **15 of 22 lost roughly 60% of their content** before being embedded. That
text was stored, returned, and citable — but could never influence whether its own
chunk was retrieved.

Fixed by chunking on the tokenizer, plus an assertion that refuses to build an
index if any chunk exceeds the ceiling. → [`src/ingest.py`](src/ingest.py)

### 2. A conclusion measured on a small corpus was backwards

On one document, hybrid search looked useless — dense, RRF, and weighted fusion
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

Ranking metrics call that perfect — by their definition it is. It is the wrong
definition when the goal is to compile every relevant source. A per-document cap
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
as a small improvement — **that report was wrong.** The prefix arrived alongside a
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
retrieval harder onto one document — the opposite of the goal. Reverted.
→ [commit `d4ea4ab`](../../commit/d4ea4ab)

### 6. Three "settled" conclusions were noise from too small a test set

The golden set began at 23 answerable cases, mostly drawn from one paper. Growing
it to **67 cases covering all 36 documents** — a strictly harder test — reversed
three findings that had already been written up as results:

| claim, on 23 cases | on the full set |
|---|---|
| The diversity cap is **free** — no loss to ranking | It **trades**: +3.1 source recall for −1.6 any-hit |
| Window expansion is **strictly dominated** by page expansion | Window reaches 0.833 context recall for **half the tokens** (2,371 vs 4,828) |
| Reranking improves hit rate | It improves **MRR only** (0.601→0.710); any-hit is flat at 0.788 |

The abstention threshold moved too — for the third time. A gap that looked clean
between the two score distributions closed once there were enough cases to see
it, and the constant went 0.0 → 1.5 → back to 0.0.

**None of these were bugs.** Every one was a real measurement, correctly
performed, on a sample too small to support the conclusion drawn from it. With 23
cases each one is worth 4.3 points, so differences that looked decisive were
inside the noise. That is the failure mode a test set produces when it is trusted
more than it deserves — and it is far harder to notice than a crash.

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

Three interfaces — Python, HTTP, CLI — are all thin shells over one `ask()`
function, so behaviour cannot drift between them.

---

## Design decisions worth defending

**Citations are locators, not page numbers.** A page number means nothing for a
spreadsheet. PDFs cite pages, PowerPoint cites slides, Excel cites sheet and row
range, Word cites *sections* — because `.docx` pagination is computed by the
renderer and shifts with fonts and margins, so any page number would be wrong on
the reader's copy.

**A per-document cap, not MMR.** MMR diversifies on embedding distance, conflating
two kinds of redundancy — similar wording and same source — and needs a `lambda`
tuned per corpus. Here the unit of redundancy is known exactly: the document.

**RRF over weighted fusion by default.** A cosine similarity and a BM25 score are
not commensurable, so weighted fusion needs per-query normalisation, which is
*relative* — a query where every candidate is mediocre still yields a top score of
1.0. RRF uses only rank, so it is scale-free with nothing to tune.

**Golden set labels are derived, not written.** Each case declares a distinctive
answer string; every location containing it *becomes* gold. Labels therefore
cannot drift from the corpus. When the corpus grew from 1 to 20 documents, 12 of
16 hand-written adversarial cases had silently become answerable — and nothing
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

The other two sets are built by topic rather than by a list of paper IDs — a
wrong ID downloads a real paper under a confidently wrong filename, so the
fetcher queries the arXiv and Wikipedia APIs instead:

```bash
python src/fetch_topic.py --topic birds --out data-birds
python src/make_documents.py --src data-birds   # .docx/.pptx/.xlsx/.pdf
RAG_STORE_DIR=store-birds RAG_DATA_DIR=data-birds python src/ingest.py
python src/calibrate_threshold.py --golden eval/golden-birds.json
```

```bash
python src/evaluate.py         # reproduce every number in this README
python src/per_case.py         # per-case outcomes -> eval/per_case.json
python src/build_analytics.py  # what the pages plot -> eval/analytics.json
python src/check_freshness.py  # is any of the above stale? exit 1 if so
python src/hard_cases.py       # only the cases nothing gets right, in ~20s
python src/test_metrics.py     # the scoring functions, hand-computed (ms)
python src/test_trace.py       # the trace and the serving path still agree
python src/test_freshness.py   # the staleness check catches each known failure
python src/serve.py            # local HTTP API and UI on :8000
python src/test_loaders.py     # .docx/.pptx/.xlsx/.pdf round-trips
```

The server binds to 127.0.0.1 on purpose and makes no external request: the
documents may be private, and the interface's fonts are bundled rather than
pulled from a CDN for the same reason.

Set `RAG_DATA_DIR` to point at any folder — including a Google Drive for Desktop
mount. The UI does the same thing without an environment variable: the **Local
folder** tab takes a pasted path, reports what it would index and what it would
skip, and then indexes it. A browser cannot read a filesystem path out of a file
picker, so pasting it is not a lesser version of a folder picker — it is the
only version there is.

---

## The interface

`python src/serve.py`, then <http://127.0.0.1:8000>.

Pick a set of documents, ask a question, and the answer arrives with the passage
it came from, cited to the page, slide, or spreadsheet rows. A passage pulled
out of a spreadsheet is shown as the table it came from rather than as a
paragraph with the column headers left in the middle of the sentence.

Underneath it, the retrieval result: the score, read in the units it is actually
in — a cross-encoder logit running roughly −11 to +11, not a probability —
along with which documents were drawn on, how much document text was read, and
how long it took.

Below that the search is drawn as a schematic: seven stages, each a block of
sheets whose depth tracks how many candidates are still in play, so the funnel
from 5,459 passages down to five is the shape of the picture rather than a
number written under it. Colour means one thing only — this passage is in your
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

- **Cross-document confusion is unsolved.** "What optimizer was used to train the
  Transformer?" returns Vision Transformer. Title prefixing was tried and failed.
- **Native Google Docs cannot be read.** A `.gdoc` is a URL pointer with no
  content; ingestion says so explicitly rather than indexing emptiness.
- **Generation has never been executed** — the code path exists but requires API
  credit. Retrieval-only is fully functional.
- **No permissions model.** Fine for one user; not for a team.
- **The corpus is 20 ML papers.** Deliberately similar, which is the hard case,
  but not contracts or spreadsheets — behaviour on those is untested.
- **84 cases is still small.** Each answerable case is worth ~1.5 points, so
  treat differences under ~0.03 as noise. The 23-case set had a 4.3-point
  resolution and produced three false conclusions (finding 6); assume this one
  is hiding others.
- **Labels were authored by the same process that built the system.** Mitigated by
  deriving them from the corpus, not eliminated.

---

## Further reading

- [`eval/RESULTS.md`](eval/RESULTS.md) — full measurements and methodology
- [`eval/golden_set.json`](eval/golden_set.json) — 84 labelled cases
