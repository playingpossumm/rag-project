# rag-project

A document question-answering system that retrieves passages from your own files
and cites exactly where each came from — page, slide, or spreadsheet row.

**The point of this repository is not that it implements retrieval-augmented
generation. It is that every design decision in it was measured, and three of
them turned out to be wrong.**

```
                       any-hit@5    MRR    NDCG   source recall
  naive RAG                0.739  0.612   0.643           0.733
  + cross-encoder rerank   0.783  0.739   0.730           0.746
  + hybrid BM25 fusion     0.913  0.848   0.846           0.789
  + diversity cap          0.913  0.848   0.852           0.830
```

20 documents, 2,768 chunks, 84 labelled evaluation cases. Reproduce with
`python src/evaluate.py`.

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

At 20 documents the pool is **0.84%**, and hybrid fusion is worth **+13 points of
hit rate**. Three separate conclusions inverted when the corpus grew: whether
fusion helps, which fusion to use, and where the abstention threshold belongs.

**A conclusion measured on a toy corpus may not merely be imprecise. It may be
backwards.**

### 3. The metrics reported success on questions the system could not answer

For a question five papers answer, the system returned **all five passages from
one paper**. MRR scored it **1.000**; source recall scored it **0.200**.

Ranking metrics call that perfect — by their definition it is. It is the wrong
definition when the goal is to compile every relevant source. A per-document cap
lifted source recall to 0.830 with **no loss to ranking**, because the cap only
skips; it never reorders. → [`src/diversify.py`](src/diversify.py)

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

Re-measured with one variable:

```
             any-hit    MRR   NDCG   source recall
  prefix       0.913  0.848  0.855           0.779
  no prefix    0.913  0.848  0.852           0.830
```

No ranking benefit; source recall five points worse. Every chunk in a document
received the *same* prefix, making them more similar to each other and clustering
retrieval harder onto one document — the opposite of the goal. Reverted.
→ [commit `d4ea4ab`](../../commit/d4ea4ab)

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

python src/fetch_corpus.py     # 20 arXiv papers (or drop your own in data/)
python src/ingest.py           # build the index
python src/cli.py "What is late interaction in a retrieval model?"
```

```bash
python src/evaluate.py         # reproduce every number in this README
python src/serve.py            # local HTTP API on :8000
```

Set `RAG_DATA_DIR` to point at any folder — including a Google Drive for Desktop
mount.

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
- **84 cases is still small.** Treat differences under ~0.03 as noise.
- **Labels were authored by the same process that built the system.** Mitigated by
  deriving them from the corpus, not eliminated.

---

## Further reading

- [`eval/RESULTS.md`](eval/RESULTS.md) — full measurements and methodology
- [`eval/golden_set.json`](eval/golden_set.json) — 84 labelled cases
