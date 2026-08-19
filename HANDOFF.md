# Handoff

Written 2026-08-19 so a new session can pick this up cold. Everything here is
measured or verifiable from the repo; where something is unverified it says so.

---

## 1. What this is, and why it exists

A retrieval-augmented generation (RAG) system built by hand, in Python, from
parts — not a wrapper around a framework.

**The owner's stated goal is learning and portfolio**, not shipping a product
today. That matters for how you weigh trade-offs: a working measurement that
overturns a plan is worth more here than a feature that ships. They have also
explicitly said they intend to keep iterating and covering edge cases rather
than stopping at "good enough", and that is their call, already reaffirmed once.

**The strategic thesis** (settled, do not relitigate): this will not beat
NotebookLM or Glean on answer quality — better models, more compute, teams. It
competes on what closed consumer products structurally cannot offer:

| Question a user has | Closed product | Here |
|---|---|---|
| Why did it return *that*? | no answer | every stage inspectable, with scores |
| Did my change help? | no numbers exist | 84 labelled cases, 4 configurations |
| What if it doesn't know? | answers anyway | refuses below a calibrated threshold |
| Sources disagree? | blended into one voice | conflict surfaced, judgement left to user |
| Embed it in my tool? | no, it's a destination | yes — a library + HTTP wrapper |

---

## 2. Current measured state

Corpus: **20 documents, 2,768 chunks** — all PDFs (see §7, this is the biggest gap).
Golden set: **66 answerable + 18 adversarial = 84 cases**.

End-to-end at k=5 (from `eval/results.json`, regenerate with `src/evaluate.py`):

| pipeline | hit rate | MRR | NDCG | source recall |
|---|---|---|---|---|
| dense, no rerank (naive RAG) | 0.788 | 0.601 | 0.645 | 0.704 |
| + cross-encoder rerank | 0.788 | 0.710 | 0.715 | 0.711 |
| + hybrid fusion (RRF) | 0.864 | 0.757 | 0.766 | 0.742 |
| **+ diversity cap 2/src (default)** | **0.848** | **0.751** | **0.765** | **0.773** |
| + diversity cap 1/src | 0.818 | 0.739 | 0.751 | 0.816 |

Context expansion: none 0.742 recall @ 997 tok/query · **window±1 0.833 @ 2371 (default)** ·
page 0.848 @ 4828. Page buys 1.5 points of recall for 2× the tokens; window is the default.

Abstention separation: answerable median **+4.93**, unanswerable median **−2.33**,
threshold **0.0**. The distributions overlap (answerable min −4.02), so the
threshold is a deliberate trade, not a clean split.

Indexing: full cold build **432 s** · re-index nothing changed **0.76 s** ·
add 1 document to 20 **18.8 s**.

---

## 3. Architecture

```
data/*.pdf
  └─ ingest.build_index()          ← emits structured progress events
       ├─ loaders.load_document()  ← pdf / docx / pptx / xlsx dispatch
       │    └─ ocr.py              ← per-page RapidOCR fallback for scans
       ├─ corpus_health            ← FAIL/WARN verdict; FAIL is excluded from the index
       ├─ parse_cache              ← keyed on file bytes + PARSER_VERSION + chunk salt
       ├─ chunk_text()             ← token-offset windows, sliced from the original string
       └─ embedding_cache          ← keyed on chunk text + model name
            └─ vector_store/index.faiss + metadata.json

query
  └─ api.ask()
       └─ retrieve.retrieve()
            ├─ shortlist  = dense (FAISS IndexFlatIP) ⊕ BM25, fused by RRF
            ├─ rerank     = cross-encoder/ms-marco-MiniLM-L-6-v2 over the whole pool
            ├─ diversify  = at most 2 passages per document
            └─ expand     = window ±1 chunk
       └─ abstain gate → confident / declined

serve.py    GET /  (UI) · POST /ask · POST /api/trace · GET|POST /api/index/*
pipeline_trace.py  re-runs retrieval keeping every intermediate ranking
```

Defaults, all justified by measurement in `eval/RESULTS.md`:
`CHUNK_SIZE_TOKENS=210`, `CHUNK_OVERLAP_TOKENS=40`, `TOP_K=5`, `CANDIDATE_K=20`,
`DEFAULT_FUSION="rrf"`, `RRF_K=60`, `DEFAULT_MAX_PER_SOURCE=2`,
`ABSTAIN_THRESHOLD=0.0`, `DEFAULT_EXPANSION="window"`, `PARSER_VERSION=3`,
`USE_TITLE_PREFIX=0` (off — see §4).

---

## 4. The discipline that defines this project

**Measure before building.** Four times a measurement overturned a plan that
sounded obviously right. Three would have produced correct, well-engineered code
that improved nothing; one was actively harmful and had already been written up
as a success.

| Sounded right | Measurement said | Outcome |
|---|---|---|
| Prefix each chunk with its document title | no ranking gain, source recall **−0.051**; the earlier "win" came from changing two variables at once | built, measured, **reverted** (`USE_TITLE_PREFIX=0`) |
| The reranker misses constraint words because it's too small | an 8× model (BGE 278M) fixed the *same* 2/10 cases at 10 s/query | upgrade **rejected** |
| Tables retrieve badly, need special handling | tables measured **easier** than prose (0.952 vs 0.800 any-hit), 91% keep headers | **no code written** |
| Re-indexing is slow because the index is rebuilt | rebuilding the FAISS index is **0.01 s = 0%** of runtime | task **redefined** — caching + lazy loading instead |

**Corollary that keeps biting: render it and look at it.** Screenshotting the UI
with Playwright caught four defects invisible in source, three of them in code
written and reviewed in the same session. Do not treat visual verification as a
final polish step.

Other hard-won corrections worth not repeating:
- NDCG once read **1.373** — the eval tool's own IDCG bug. Fixed and unit-checked.
- Embeddings were silently truncated at 256 tokens (median 370 tokens lost per
  chunk) because chunk size was specified in *words*. Now token-based, with an assertion.
- README numbers went stale against a grown golden set; the owner caught it. Re-running
  eval surfaced three more conclusion reversals.
- `src/trace.py` would have shadowed the stdlib `trace` module for every dependency
  (`src/` is first on `sys.path`). Renamed `pipeline_trace.py`.

---

## 5. UI state

`ui/index.html` — self-contained, no CDN, both themes. Served by `serve.py` at `/`.
Two views: **Inspect a question** (verdict + bump chart + per-stage tables) and
**Corpus & indexing** (stat tiles, progress meter, per-document status).

**The chart palette is load-bearing.** Document colours are the first three slots
of a documented categorical palette, validated *all-pairs* (lines can sit
anywhere) against this UI's own surfaces: light CVD ΔE 9.2 / normal 24.0; dark
9.4 / 20.9 — both pass. Only three: no ordering of more clears the all-pairs
floor. A fourth document folds to grey and is identified by a direct label.
The previous code generated hues by cycling HSL and failed at ΔE 1.6 — two
documents the same colour for a deuteranopic reader. **Do not "improve" this by
adding a fourth hue without re-running the validator.**

Design direction for the redesign lives in **`docs/ui-brief.md`** — read it before
touching the UI. Short version: editorial × laboratory instrument, "elegant and
complex yet easy to understand and follow", Stripe docs / Observable as
reference, serif headings + sans body + mono data, layered density, system
theme, prominent explained refusals. One thing deliberately left undecided: the
**owner wants to see several ambient/generative options side by side** before choosing.

---

## 6. Environment

- Windows 11. Project at `C:\Users\owner\Desktop\rag-project`.
- venv at `.venv` — **always use `.venv\Scripts\python.exe`**, not bare `python`.
- Run the server: `.venv\Scripts\python.exe src\serve.py` → http://127.0.0.1:8000/
- Playwright + Chromium installed in that venv. Use it; see §4.
- `node` v24 available (the dataviz palette validator and architecture-map need it).
- Console is cp1252 — scripts printing corpus text must
  `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` or they crash.
- Very long heredocs fail with `ENAMETOOLONG`; use the Write tool for large files.
- `RAG_DATA_DIR` env var repoints the corpus directory (works for a synced Drive folder).

**Secrets:** `.env` holds an Anthropic API key. It is gitignored (verified). Never
echo, commit, or include it in any artifact. **The account has zero credit**, so all
generation calls fail with `invalid_request_error`. Retrieval-only works and is the
deliberate default.

**Git identity is repo-local**: `playingpossumm <the owner's address, removed 2026-09-07>`. The owner
explicitly did not want their work email on this repo — do not change it.

**Commit style**: explain *why*, not just what; record the measurement that drove
the change, including measurements that refuted the original plan.

**Skills installed** (`~/.claude/skills/`): `frontend-design`, `design-anti-slop`,
`algorithmic-art`, `hyperframes-animation`, `webapp-testing`, `architecture-map`.
Installed as copies, not symlinks — Windows blocks symlinks without Developer Mode,
so `git pull` will not update them. Sources are in `~/.claude/external-skills/`
(note: the `hyperframes` clone there is 1.4 GB and can be deleted; the installed
skill is a standalone copy).

---

## 7. What is unfinished — stated plainly

1. **The corpus is 20 PDFs and nothing else.** `load_docx`, `load_pptx`,
   `load_xlsx` and the entire OCR path are written and have **never run on real
   files**. The spreadsheet path is the least trustworthy: rows are serialised to
   `Column: value` text on an assumption with no evidence behind it. This is the
   largest unknown in the system and the cheapest to close — it needs the owner's
   documents in `data/`.
2. **Cross-document confusion, ~15% of questions.** The retriever matches the
   topic and ignores the constraint that distinguishes the answer ("normalisation
   across features *rather than examples*" still returns Batch Normalization).
   Mechanism understood; the obvious fix is ruled out by measurement (§4). The
   real fix is query decomposition, which needs an LLM → needs credit.
3. **Generation is unverified end to end.** `generate.py` targets `claude-opus-5`
   and has never completed a real call. Code written, evidence absent.
4. **Folder linking** — deferred by the owner. Constraint already established: a
   browser cannot read a filesystem path from a file picker, so it must be a
   pasted path or drag-and-drop that copies into `data/`.
5. **Google Drive native docs** — `.gdoc`/`.gsheet`/`.gslides` are pointers, not
   files. Needs the Drive API export path in `loaders.py`. Uploaded PDFs/Office
   files already work today via `RAG_DATA_DIR`.
6. **No permissions model.** Fine for a local single-user tool; would matter if
   this ever served more than one person.
7. **The isometric architecture map is requested but not built.** The owner asked
   for this directly — "I want to implement that kind of visual thing of the
   architecture map" — and the `architecture-map` skill was installed for it.
   Nothing has been built yet. It is a *separate* piece of work from the UI
   redesign: it maps the repo's own structure (27 modules across ingestion,
   caching, retrieval, serving and evaluation), not the retrieval interface.
   Strong portfolio value for the same reason the Inspector has it — it shows the
   system rather than claiming things about it. The skill's own rule matches this
   project's discipline: *prose, groups and flows are authored; counts, coverage
   and geometry are measured* — so give it `README.md` and `HANDOFF.md` for the
   authored half rather than letting it invent one.

Ranked by value: **(1) is worth more than everything else combined**, and only the
owner can unblock it. **(7) is the next most valuable** — it is explicitly wanted,
fully unblocked, and needs nothing from anyone.

---

## 8. Published artifacts — update, never re-publish

Two pages are published to the owner's Claude account. Their **source is in this
repo**; the published copies live in the cloud and are not carried by any session.

| Artifact | Source file | URL |
|---|---|---|
| Anatomy of a Retrieval Pipeline (Phases 1–3) | `docs/phase-1-field-notes.html` | https://claude.ai/code/artifact/an artifact id, removed 2026-09-07 |
| Retrieval System Status | `docs/project-status.html` | https://claude.ai/code/artifact/an artifact id, removed 2026-09-07 |

**To update one, pass its URL.** Publishing the source file without the `url`
creates a *second, separate* artifact instead of updating the existing one, and
the owner's existing link silently goes stale. This is the single easiest way to
break something here, and nothing in the file itself warns you — which is why the
URLs are recorded here.

Both carry a banner scoping Phase 1–2 figures as historical. Phase 3 reflects the
system as it stands. If Phase 4 is added, extend that banner rather than letting it
disclaim current work.

---

## 9. Where to read more

| File | What it holds |
|---|---|
| `README.md` | project overview, current numbers |
| `eval/RESULTS.md` | every measurement and the default it justifies |
| `eval/results.json` | machine-readable, regenerated by `src/evaluate.py` |
| `docs/ui-brief.md` | the UI design interview and direction |
| `docs/phase-1-field-notes.html` | mechanism-level explanation, Phases 1–3 (published artifact) |
| `git log` | why each decision was made, including the reversals |
