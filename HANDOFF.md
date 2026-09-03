# The complete technical account

`README.md` is the short version of this document and the `/about` page in the
running app is the plain-language one.

Started 2026-08-19, current as of 2026-08-31. **Everything here is measured or
verifiable from the repository, and where something is unverified it says so.**
Numbers in §2 and §3 are checked against the generated measurements by
`python src/check_docs.py`, so this document cannot quietly drift from the
system it describes.

---

## 1. Purpose

This is a retrieval-augmented generation (RAG) system built by hand, in Python,
from parts rather than as a wrapper around a framework.

**The owner's stated goal is learning and portfolio**, not shipping a product
today, which sets how trade-offs are weighed, because a measurement that
overturns a plan is worth more here than a feature that ships. They have also
said they intend to keep iterating and covering edge cases rather than stopping
at "good enough", and that is their call, already reaffirmed once.

**The strategic thesis is settled and should not be relitigated.** This will not
beat NotebookLM or Glean on answer quality, since those have larger models and
more compute behind them, and it competes instead on what closed consumer
products cannot offer.

| Question a user has | Closed product | Here |
|---|---|---|
| Why did it return *that*? | no answer | every stage inspectable, with scores |
| Did my change help? | no numbers exist | 84 labelled cases, 4 configurations |
| What if it doesn't know? | answers anyway | refuses below a calibrated threshold |
| Sources disagree? | blended into one voice | conflict surfaced, judgement left to user |
| Embed it in my tool? | no, it's a destination | yes — a library + HTTP wrapper |

---

## 2. Current measured state

There are three corpora, each with its own index, golden set, abstention
threshold and rerank blend, because **nothing about a corpus transfers to
another one**. That is the most reused finding in this project, and §4 covers
it.

| | ML & NLP papers | Ornithology | Quant finance |
|---|---|---|---|
| documents | 36 PDF | 45 mixed | 35 PDF |
| passages | 5,459 | 864 | 6,184 |
| cases | 67 + 17 adv | 25 + 7 adv | 33 + 8 adv |
| any-hit@5 | **0.910** | **0.880** | **0.939** |
| MRR | 0.784 | 0.638 | 0.826 |
| source recall | 0.785 | 0.792 | 0.816 |
| abstention threshold | 0.0 | −5.5 | −4.0 |
| rerank blend | 0.0 | 0.20 | 0.0 |
| answerable median | +5.25 | +1.21 | +2.81 |

Shipped config throughout: RRF fusion, cross-encoder rerank, diversity cap
2/source, window±1 context expansion.

**Those are three tuned pipelines, not one pipeline on three corpora.** Four
settings are per-corpus, so the table above answers "what does this system do on
this corpus" and not "does this approach generalise". The second question needs
one configuration everywhere, and `src/uniform_baseline.py` measures it:

| under the untuned defaults | any-hit | MRR | NDCG | src recall |
|---|---|---|---|---|
| ML & NLP papers | 0.910 | 0.783 | 0.802 | 0.784 |
| Ornithology | 0.840 | 0.593 | 0.660 | 0.792 |
| Quantitative finance | 0.939 | 0.758 | 0.794 | 0.786 |

**Quote this table for the generalisation claim and the one above for the
per-corpus one.** Across three corpora sharing no format, subject or provenance,
one untuned configuration spans 0.099 of any-hit. The corpus the pipeline was
developed on gains +0.000 any-hit from its own tuning, so the tuning is not
holding up three special cases.

Reproduce: `python src/evaluate.py` (add `--golden eval/golden-birds.json` and
the matching `RAG_STORE_DIR`/`RAG_DATA_DIR` for the others). It prints the
rerank blend it is using and takes it from `corpora.json`, so the harness
measures what the server serves.

`eval/RESULTS.md` tables are **generated** from `eval/results.json` by
`src/build_results_doc.py`; `--check` fails when they have drifted. The prose
around them is not generated.

The other generated chain, golden set -> `per_case*.json` -> `analytics.json`
-> the questions the front page offers, is checked by
`python src/check_freshness.py`, which exits 1 when any link is stale and prints
the commands that rebuild it in order. `src/serve.py` runs it on startup and
warns rather than blocking. See §7.

ML pipeline ladder, each row adding one stage:

| pipeline | any-hit | MRR | NDCG | src recall |
|---|---|---|---|---|
| dense, no rerank *(naive RAG)* | 0.821 | 0.607 | 0.657 | 0.678 |
| + cross-encoder rerank | 0.836 | 0.700 | 0.720 | 0.712 |
| + RRF hybrid fusion | 0.925 | 0.788 | 0.808 | 0.745 |
| **+ diversity cap 2/src** *(shipped)* | **0.910** | 0.784 | 0.804 | **0.785** |
| + diversity cap 1/src | 0.851 | 0.760 | 0.776 | 0.816 |

Context expansion (ML): none 0.821 recall @ 989 tok · **window±1 0.881 @ 2,338
(default)** · page 0.910 @ 4,646. Page buys 2.9 points for twice the tokens.

Fusion is corpus-dependent too, and the obvious reading of that was wrong.
On the **candidate pool**, RRF beats dense-only on ML and quant and loses on
birds, at dense 1.000 against RRF 0.920.
That reading said the bird corpus wanted more dense weighting. Measured **end to
end at the served configuration** (`src/sweep_fusion.py`, re-run 2026-09-01
against the corrected golden set) it reverses:

| birds | pool any-hit | shipped-pipeline any-hit |
|---|---|---|
| dense only | **1.000** | **0.840** |
| RRF *(shipped)* | 0.920 | **0.880** |
| weighted a=0.5 | 0.960 | 0.920 |

Dense now reaches every answerable question in the pool and still produces the
worst finished answer of the five.

The first stage that finds the answer most often produces the worst final
answer. A pool is not just a set of passages, it is an *ordering* handed to the
cross-encoder, and dense hands over one the reranker cannot exploit. That is
the same weakness the rerank blend exists to hedge against. **Quote the end-to-end row,
never the pool row, when arguing about fusion.**

`weighted a=0.5` does win a question on birds (`bird-dialects`, 0.880 → 0.920)
and costs MRR (0.638 → 0.611) and NDCG. That is the same trade the rerank blend
was judged on, and this corpus's own note already settles how to read it: on 25
cases an any-hit gain of one question is thinner evidence than MRR, so **not
shipped**. On quant, `weighted a=0.7` is weakly dominant: MRR 0.714 → 0.727,
NDCG +0.006, source recall +0.006, any-hit unchanged. No case changes hands, so
it buys a fraction of a rank position at the cost of recalibrating that
corpus's threshold. Also **not shipped**, and recorded rather than forgotten.

Query latency, measured per stage by `src/profile_query.py` (warm, median):
**reranking is 92–95% of a query**, about 1,000 ms of 1,100. Embedding, FAISS,
BM25, fusion, the diversity cap and context expansion together are ~80 ms. Two
consequences worth carrying: any latency work that is not about the
cross-encoder is rounding error, and the interface's single "search, scoring and
reranking" number was hiding which of the three it was.

`src/sweep_candidates.py` measures quality against milliseconds. **28 candidates
scores worse than 20 on all three corpora while the pool ceiling rises on all
three.** The reranker's precision degrades faster than the first stage's
recall improves. 16 candidates gives *identical* any-hit on all three for ~22% less
time, at −2.6 MRR on birds; a real option, not shipped, because it regresses a
corpus that was not the problem.

Indexing caches live in the store being written, one set per corpus. Until
2026-08-27 they were shared and `prune()` deleted whatever did not belong to
the corpus in hand, so indexing one corpus made the next re-index of the
others a full re-parse. The timings below assume the caches survive.

Indexing: full cold build **432 s** · re-index nothing changed **0.76 s** ·
add 1 document to 20 **18.8 s**. Re-ingest reuses embeddings by content hash, so
a change that does not alter chunk text costs a re-index and no compute.

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
pipeline_trace.py  re-runs retrieval keeping every intermediate ranking,
                   under any option combination the settings panel can ask for
```

Defaults, all justified by measurement in `eval/RESULTS.md`:
`CHUNK_SIZE_TOKENS=210`, `CHUNK_OVERLAP_TOKENS=40`, `TOP_K=5`, `CANDIDATE_K=20`,
`DEFAULT_FUSION="rrf"`, `RRF_K=60`, `DEFAULT_MAX_PER_SOURCE=2`,
`ABSTAIN_THRESHOLD=0.0`, `DEFAULT_EXPANSION="window"`, `PARSER_VERSION=3`,
`USE_TITLE_PREFIX=0` (off; see §4).

---

## 4. Method

**Measure before building.** Four times a measurement overturned a plan that
sounded obviously right. Three would have produced correct, well-engineered code
that improved nothing; one was actively harmful and had already been written up
as a success.

| Sounded right | Measurement said | Outcome |
|---|---|---|
| The harness says the abstention threshold should be +2 | `net` is a difference of rates over unequal populations; in cases, +2 trades 4 caught for 4 lost | threshold **unchanged at 0.0**, and the harness now prints counts too |
| Prefix each chunk with its document title | no ranking gain, source recall **−0.051**; the earlier "win" came from changing two variables at once | built, measured, **reverted** (`USE_TITLE_PREFIX=0`) |
| The reranker misses constraint words because it's too small | an 8× model (BGE 278M) fixed the *same* 2/10 cases at 10 s/query | upgrade **rejected** |
| Tables retrieve badly, need special handling | tables measured **easier** than prose (0.952 vs 0.800 any-hit), 91% keep headers | **no code written** |
| Re-indexing is slow because the index is rebuilt | rebuilding the FAISS index is **0.01 s = 0%** of runtime | task **redefined** — caching + lazy loading instead |
| A stronger cross-encoder will fix the descriptive questions | ranking and gate separation move in **opposite** directions across models; L12 ranks better on ML, worse on birds | swap **rejected**, all three corpora measured |
| Let the strong model do only the gate, at 1/20 the cost | BGE scored AUC 0.968 ranking *and* gating, 0.827 gating MiniLM's pick — the gain was self-consistency | idea **refuted by its own harness** |
| Query decomposition will reach the seven structural cases | recovers **0 of 7** and costs 0.045 any-hit at 21× the latency; rewriting recovers 1 and breaks 5 | named fix **ruled out**, nothing shipped |
| Fusing a second embedder is weakly dominant everywhere | true of pool recall; end to end it is worse on ML, a one-question trade on birds, earned only on quant | **not shipped** beyond the one corpus |
| A larger generator will answer better | over 43 matched cases llama3.1:8b gains 2 of 28 correct, refuses one adversarial question fewer, invents its first citation, and takes 58x as long | **no change**, run finished 2026-09-01 |

**The generalisation, added 2026-08-27 after five in a row.** Every defect
found in this repo's last five working sessions was in code that had tests
*around* it and nothing *running* it: `/ask` ignored its corpus argument, a
wrong-typed JSON field dropped the connection, indexing one corpus deleted
another's parse cache, six routes were served and documented nowhere, and
`generate.py` had never been executed at all. In each case the module was read,
reviewed and described correctly in the documents, and never invoked.

So the rule is not to write more tests but to **run the thing**. A test that
imports a module and asserts on its output found all five, and no amount of
reading found any of them. The three claimed-but-absent tests in this repo's
history are the same failure one step earlier, because a check that was
performed and not committed is indistinguishable, six days later, from one that
never happened.

**The same rule applied to the interface is to render it and look at it.**
Screenshotting the UI
with Playwright caught four defects invisible in source, three of them in code
written and reviewed in the same session. Do not treat visual verification as a
final polish step.

Other hard-won corrections worth not repeating:
- NDCG once read **1.373**, from the eval tool's own IDCG bug. Fixed, and as of
  2026-08-21 actually unit-checked: `src/test_metrics.py` holds the four
  hand-computed cases the field notes have always said were verified but which
  were never committed. The regression test has teeth: the pre-fix arithmetic
  returns 2.131 where the assertion demands ≤ 1.0. Second claimed-but-absent
  test found in this project, after `test_trace_matches_pipeline`.
- Embeddings were silently truncated at 256 tokens (median 370 tokens lost per
  chunk) because chunk size was specified in *words*. Now token-based, with an assertion.
- README numbers went stale against a grown golden set, and were caught by hand rather
  than by anything automatic, which is why `check_docs.py` now exists. Re-running
  eval surfaced three more conclusion reversals.
- `src/trace.py` would have shadowed the stdlib `trace` module for every dependency
  (`src/` is first on `sys.path`). Renamed `pipeline_trace.py`.

---

## 5. UI state

### The front page, rewritten 2026-08-26

`retrieval visualized/`. Header carries **About**, **Analytics** and
**Source**. The name is a **home button** that clears the thread. A third link
to `/archive`, the pre-rebuild front page served live beside the current one,
was removed on 2026-08-29. The rebuild it existed to compare against is
finished, and a visitor has no use for a frozen copy of the page they are
already reading. The snapshot stays in `archive/`.

Flow: headline → what RAG is → the pipeline diagram → **pick a document set**
(dropdown, each with an isometric mark, counts and formats) → ask. Clicking the
field opens the example questions, grouped by what each exercises: *answered in
one place*, *spread over several documents*, *similar documents to tell apart*,
*not in these documents*. The third read *decoys that look right* until
2026-09-02, which labelled eight answerable questions per corpus as traps. Those come from `eval/analytics.json`. **Regenerate it
after any golden-set change, or the page offers questions the corpus cannot
answer.**

An answer shows the passage at normal weight with only the **answering words**
bold, its citation with the document's real title, a plain-language reading of
the score, and **Also found**, which lists passages from *other* documents and
is open by default. That last one exists because the top passage is sometimes wrong in a
specific way: asked for a bird's fused collarbone, the reranker prefers the
pygostyle passage and puts the furcula second.

The diagram: upright isometric panels, left to right, each stage a block whose
**depth is how much survives**, from six sheets at the index down to one at
the answer. Projection is two numbers, not one angle: `SPREAD` 0.68 (how wide the
ground axes fan) and `RISE` 0.15 (camera height; 0 is eye level, 0.5 is a true
isometric). Conflating them is why several attempts went wrong.

**Lessons that cost time, so they are written down:**

- Verify at a laptop viewport (~660px of usable height) and a phone, not only
  at 1400px+. A label collision at 1080px survived several rounds because I
  only ever looked at wide windows.
- **Never hide content to keep something above the fold.** A rule hiding the
  hero blurb under 745px of height fired on an ordinary laptop and the page lost
  its own explanation. The page scrolls; the constraint was mine, not the
  user's.
- Screenshot the rendered page. It has caught roughly twenty defects that
  reading the diff did not.


Rewritten in the session of 2026-08-19/20. `ui/index.html` is now an **app**, not
the two-tab inspector this section used to describe.

### Direction, 2026-08-20

The owner's current instruction, and it supersedes the ambition recorded below:
**keep the UI clean and simple, and put the effort into the technicals.**

The UI went through four visual passes in one session and landed somewhere the
owner was not happy with. Read that as a signal about sequencing rather than
taste, because the interface was being designed ahead of the capabilities it is
meant to expose, so each pass was decorating a demo instead of surfacing a
tool.

What that means for the work after it:

- **Do not start another whole-page visual pass.** Simplify what is there.
- **The isometric city is the most elaborate thing on the page and is a fair
  candidate for simplification or removal.** It is genuinely good at explaining
  the mechanism, but it cost most of a session and the page works without it.
  Ask before deleting; do not expand it unasked.
- Prefer plain, legible, conventional components over anything bespoke.
- Technical work below (§7) now outranks anything visual.

### Scope

One page, served at `/`. A folio bar (source tabs + corpus readout), a search
field, then three panels that appear once you trace: **the pipeline** (the city),
**the answer**, and a two-up of **settings** and **stages**.

The settings panel **re-runs retrieval** as of 2026-08-21, and the folder tab
takes a pasted path. Both are described under §7, which now records them as done
rather than outstanding.

| File | What it is |
|---|---|
| `ui/index.html` | the app |
| `ui/pipeline-map.js` | the city — `buildCity` (corpus only), `buildRun` (one query), `drawScene`, `captions` |
| `ui/fonts/*.woff2` | Inter + DM Mono, latin subsets, 78 KB, served from `/fonts/` |
| `ui/versions.html` + `ui/versions/` | **temporary.** Every version of the hero diagram, drawn side by side from one trace |

**`/versions` is deployed, linked from nothing, and meant to be deleted.** It
draws all four versions the hero diagram has been through, each rendered by the
code from the commit that produced it, from one trace and one palette, so the
comparison is about the drawing and nothing else. It exists so a direction can
be chosen by looking rather than by remembering.

Delete `ui/versions.html`, `ui/versions/` and the two data files under it once
that choice is made. `src/check_links.py` lists `/versions` as a file page
rather than a route, so removing it means removing that exception too. That is
deliberate, since the checker fails and names the page rather than letting a
dead one sit quietly.
*(`ui/pipeline.html`, `ui/ambient.html` and `ui/ambient-fields.js` were deleted on
2026-08-21; see "Ambient fields" below. They are in git if wanted back.)*

### Design direction, superseding docs/ui-brief.md

The brief was written before a reference design was chosen
(`JearDesuss/compute-debt-obligations`, a near-black editorial explainer). Where
the two disagree, this list wins and the brief is marked stale:

- **Light only.** The dark palette was deleted, not disabled. The owner
  considered going dark to match the reference and chose to stay light.
- **All sans.** Inter for everything, DM Mono for every label and number. The
  brief's "serif headings" is dead.
- **Tracked mono micro-labels** on every panel, axis and field. This is the
  single device that makes the screen read as an instrument.
- **Typographic rows, not cards.**
- **No CDN, ever.** Fonts are bundled precisely because the corpus may be
  private and a page about private documents should not call a font host.
- **The answer is the largest element on the page**, at 30px against 10px
  labels.

Motion follows the `emil-design-eng` skill: custom `cubic-bezier(.23,1,.32,1)`
rather than the weak built-in eases, `scale(.97)` press feedback, 60 ms staggered
panel entry, specific transition properties, reduced-motion honoured.

### The city

Stages are places, and **the city is always fully drawn.** Every block, tower
and road exists before a question is asked; a query changes lighting and
nothing else. This is load-bearing and was arrived at the hard way: an earlier
draft grew the roads as results arrived, and the verdict on it was that it
"pops out", because structure appearing as a consequence of the question is
backwards.

- The index is one block per document, sized by that document's **real** chunk
  count from `/api/chunks`. An even split looked right and was wrong.
- Dense and BM25 are two towers on two forked roads, because they run **at the
  same moment**. The old bump chart drew them as column 1 and column 2, which
  taught a sequence that does not exist.
- Every tower is ten floors and **a floor is a rank**, floor one on top, so
  reranking visibly reshuffles the building. That is the bump chart's logic,
  kept.
- The replay is a **replay**, and says so: a trace finishes in 30–120 ms, so an
  8-second playback is labelled `NNms · replay NNN×`. Animating during the wait
  would be motion pretending to be progress.

### Still load-bearing

**The chart palette.** Document colours are the first three slots of a documented
categorical palette, validated *all-pairs* (lines can sit anywhere): light CVD
ΔE 9.2 / normal 24.0. Only three: no ordering of more clears the floor, so a
fourth document folds to grey with a direct label. The previous code cycled HSL
and failed at ΔE 1.6, which gave two documents the same colour for a
deuteranopic reader.
**Do not add a fourth hue without re-running the validator.** Note the dark-mode
validation is now moot; if anyone reintroduces dark, it must be re-run.

### Ambient fields, decided and then orphaned

The owner asked to see generative options side by side, saw four at `/ambient`,
and chose **Lattice for the masthead, Fringe for the idle backdrop, at 0.6
intensity**. That decision was implemented and then lost in the app rewrite.

**Resolved 2026-08-21 by deleting** `ui/ambient.html`, `ui/ambient-fields.js`
and `ui/pipeline.html`, along with their routes. Both options were open; the
current direction is to dial the UI back, and wiring an animated generative
field into the masthead is the opposite of that. The work is in git
(`f2ab5de`, `f1ea450`) if the decision is ever revisited.

---

## 5b. The architecture map, removed 2026-08-28

`docs/architecture.html` was a self-contained isometric map of the codebase,
generated from measured module sizes and hand-authored prose per building. It
is gone, along with `architecture/`, the three build scripts, `package.json`,
`tsconfig.json` and 43 MB of `node_modules`.

**Three reasons, in the order that decided it.**

**It was stale and said so itself.** `node scripts/architecture-sync.mjs --check`
printed "measured.generated.ts is stale". Its coverage file described
twenty-nine modules; `src/` holds seventy-five. Fixing it did not mean re-running
the sync. The measurements regenerate, but every unclaimed module appears as
drift until somebody writes prose for it, so the real cost was authoring
forty-six buildings' worth of description that this document already covers in
words.

**Nothing linked to it.** No `href` anywhere in `ui/` pointed at
`/~/architecture`. It was reachable only by typing the URL, which means it was
serving no reader.

**It contradicted the thing the project claims to be.** The README's first
paragraph is that this is Python from parts with no framework; the map dragged
in React 19, TypeScript and esbuild for one page. `package.json`'s own
description conceded the point: "the project itself is Python; this exists
because the map ships as a bundled standalone page."

**What replaced it, and it is better.** The `/about` page explains the pipeline
in plain language with a glossary, the front page draws the live pipeline from
real per-document chunk counts, and this document is the structural account. All
three are current by construction; the map was current only when somebody
remembered to rebuild it.

It is in git history if it is ever wanted back.

## 6. Environment

- Developed on Windows 11, which is why the paths in these documents use
  backslashes. Nothing in the code is Windows-specific and the test suite has
  no platform dependencies, but it has not been run on Linux or macOS. Treat
  that as untested rather than as supported.
- venv at `.venv`. On Windows use `.venv\Scripts\python.exe` rather than bare
  `python`, so the tooling runs against the pinned dependencies rather than
  whatever is on PATH.
- Run the server: `.venv\Scripts\python.exe src\serve.py` → http://127.0.0.1:8000/
- Routes: checked against the dispatch in `serve.py` by `src/check_docs.py`,
  both ways, because the gap that mattered was a route the code
  served and no document mentioned. Nobody tests what nobody has written down.

  **Pages:** `/` the app · `/about` why the project exists, a guide to the
  pipeline and a glossary · `/quality` the retrieval-quality dashboard.

  The archive route, which served the pre-rebuild front page beside the current
  one, was removed on 2026-08-29: the rebuild it existed to compare against is
  finished. The snapshot stays in `archive/`.

  **Assets:** `/pipeline-map.js` the diagram · `/answer-mark.js` the logic
  that decides which words of a passage are set bold ·
  `/fonts/*`.

  **Asking:** `POST /api/chat` is what the interface calls for every question:
  answer and trace in one round trip, so the two cannot disagree.
  `POST /ask` is the library wrapper, and `POST /api/trace` the pipeline
  inspector, which takes options. All three follow the selected corpus;
  `/ask` did not until 2026-08-27, and see §7.

  **Corpus:** `/api/corpus` the active one · `/api/corpora` all of them, for
  the document-set picker · `POST /api/corpus/select` switches which is served ·
  `/api/chunks` · `/api/index/inspect` · `POST /api/index/start` ·
  `/api/index/status`.

  **Measurements:** `/api/eval` · `/api/analytics`, everything the analytics
  page plots, generated by `src/build_analytics.py`.

  **`/health`**.
- Playwright + Chromium installed in that venv. Use it; see §4.
- `node` v24 available (the dataviz palette validator and architecture-map need it).
- Console is cp1252, so scripts printing corpus text must
  `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` or they crash.
- Very long heredocs fail with `ENAMETOOLONG`; use the Write tool for large files.
- `RAG_DATA_DIR` env var repoints the corpus directory (works for a synced Drive folder).

**Secrets:** `.env` holds an Anthropic API key. It is gitignored (verified). Never
echo, commit, or include it in any artifact. **The account has zero credit**, so all
generation calls fail with `invalid_request_error`. Retrieval-only works and is the
deliberate default.

**Licence: MIT** (`LICENSE`), added 2026-08-28 for the public release.

**`data/`, `store-birds/` and `store-quant/` are untracked** as of 2026-08-28:
78 MB of third-party papers, and the same text again inside `metadata.json`.
They remain in the *history* deliberately, because this repository's prose
cites
**fourteen** of its own commit SHAs. `HANDOFF.md` §5 cites `f2ab5de` and
`f1ea450`, and `docs/angle-sweep.html` stamps one on every entry. A rewrite
would dangle them silently, because a dead SHA in a sentence reads exactly like
a live one. (`serve.py` embedded `d1f8c71` in the archive banner
until that route was removed on 2026-08-29; the count `check_docs.py` reports
is the live one.) `src/check_docs.py` now counts and resolves them rather than
leaving that as a number in a paragraph. See `docs/engineering-log.md`,
2026-08-28. Rebuild the corpora with
`src/fetch_corpus.py`, `src/fetch_topic.py` and `src/ingest.py`.

**Git identity is repo-local**: `playingpossumm <the owner's address, removed 2026-09-07>`. The owner
explicitly did not want their work email on this repo. Do not change it.

**Commit style**: explain *why*, not just what; record the measurement that drove
the change, including measurements that refuted the original plan.

**Skills installed** (`~/.claude/skills/`): `frontend-design`, `design-anti-slop`,
`algorithmic-art`, `hyperframes-animation`, `webapp-testing`, `architecture-map`,
`dataviz`, plus all eleven from `emilkowalski/skills`. The relevant ones there
are `emil-design-eng`, which is the UI-polish philosophy the current motion
follows, along with `animate`, `review-animations`, `apple-design` and
`prototype`. The last of those builds several different versions behind a
picker, which is the right tool when a design direction is being guessed at
rather than known.

Installed as **copies, not symlinks.** Windows blocks symlinks without
Developer Mode, so `git pull` in a skills repo will not update them. Re-copy
instead.

---

## 7. Unfinished

### Open problems, added 2026-08-26

**The cross-encoder is the weakest stage, and only partly addressed.** On the
bird corpus the shipped candidate pool contains the answer 92.0% of the
time and the finished pipeline returns it 88.0% of the time, and a dense-only
pool reaches 100% while its pipeline returns the least of the five fusions. Reranking used to discard the
first stage's ordering outright; it now keeps 20% of it on that corpus, which
recovered a question and four points of MRR. What remains is the model itself:
`ms-marco-MiniLM-L-6-v2` is weak on questions that *describe* a term rather
than naming it ("the burst of collective singing at first light").
`src/compare_rerankers.py` exists for trying another. **Use
`src/sweep_blend.py` for anything touching the blend.** It runs every corpus,
because the trap is tuning to one. That trap was walked into directly: 0.35 was
shipped globally, then measured worse than doing nothing on all three.

**The adversarial half was read, 2026-08-27, and the labels held.** The ML gate
lets 8 of 17 adversarial cases through, and the plausible story was that the
corpus had grown into them. It has not: every top passage is topically adjacent
and answers nothing: a driving paper's "reward signal" for the RLHF question,
and a training-details appendix with no seed in it for the seed question. They are
genuine gate failures. Relatedly, `audit_golden_set.py` reported 67 of 67
answerable cases as ambiguous because it subtracted `case["source"]`, a key no
case has; corrected, the figure is **0 of 67**.

**A page-level label credits a title page.** Found 2026-08-31. `bn-covariate`
asks what normalizing layer inputs addresses, wants the string "internal
covariate shift", and lists `batch_normalization.pdf` page 1 among its gold
pages. The paper's title is "Batch Normalization: Accelerating Deep Network
Training by Reducing Internal Covariate Shift", and it sits on page 1. So the
title block is a gold page and contains the answer string, and the measurement
scores it as a correct answer although it answers nothing. **43 of the 100
answerable cases in the two corpora that have title pages have a title block on
a gold page, and 17 have the answer string in the title itself.** The birds are
Wikipedia articles and have no title pages, which is why they are outside that
denominator and inside the one below. This is why filtering title blocks out of
retrieval measures worse.

**What it actually costs, measured the same day: nothing.** Reaching a label
is not collecting it. `src/audit_title_credit.py` runs the shipped
configuration over every answerable case and finds two hits satisfied only by a
chunk flagged as front matter, `bn-covariate` and `qf-whale-attack`, and both
of those chunks contain the answer string, so both credits are correct. The
count of false credits is **zero**.

**The first version of that audit said two, and was wrong.** It read
`front_matter.is_front_matter`, which judges the first 600 characters of a
chunk, as a verdict on a chunk of about 1,000, and what follows a title block
is the abstract. The wrong figure reached five documents and the deployed site
before the chunks were read in full. Recorded here because the same error, in
the same week, is what put a title page on screen in the first place: judging a
passage by how it opens. Both entries are in `docs/engineering-log.md` under
2026-08-31.

**Deriving labels from answer strings has a hole.** It proves the string is
present, not that the passage answers the question. Sixteen quant cases asked
textbook definitions of research papers that use the term once in passing,
such as "volatility clustering" inside a list of stylized facts. The label looked valid
and the question was unanswerable. Fixing the questions moved that corpus from
0.543 to 0.886 any-hit with retrieval untouched. **When a corpus scores badly,
read the failing questions before concluding anything about retrieval.**

**Generated files went stale silently three times, and that is now checked.**
`python src/check_freshness.py` covers the chain `golden set -> per_case ->
analytics -> the front page`, the one that had no test. It works two ways because they fail
differently. Each generated file records a digest of the files it was built
from (exact, so it catches rewording one question in place, which
changes no count), and case ids, question strings, corpus sizes and the
per-corpus threshold and blend are compared directly (weaker, but it works on
files written before provenance existed and names the question rather than a
hash). `src/test_freshness.py` stages each known failure on a synthetic corpus
and asserts the check reports it: 30 checks, hermetic, running in
milliseconds.

**And it found a live instance of the trap it was written for.** `per_case.py`
imported `ABSTAIN_THRESHOLD`, the ML papers' 0.0, and scored every corpus
against it, and never set a rerank blend at all. `evaluate.py` did the same with
the threshold. So:

| | claimed | serves |
|---|---|---|
| birds, answerable wrongly refused | 10 of 26 | **4 of 26** |
| quant, answerable wrongly refused | 11 of 35 | **3 of 35** |
| birds, per-case any-hit | 0.808 | **0.846** |
| birds, per-case MRR | 0.570 | **0.614** |
| birds/quant `shipped_threshold` in results | +0.0 | **-3.0 / -4.0** *(birds since recalibrated, below)* |

The bird rows moved because the blend moved: the file was measuring a reranker
weighted differently from the one behind the answer on screen. Both scripts now
resolve threshold, blend *and index* from `corpora.json` via the golden set, so
a corpus can no longer be scored against another's index by forgetting
`RAG_STORE_DIR`. Re-running everything left every retrieval figure in
`results*.json` **identical** to three decimals, which is the evidence the
resolution change measured nothing new. Only the abstention block moved.

`per_case.py` and `evaluate.py` now agree to three decimals on all three
corpora from separate code paths; before the fix birds read 0.808 against 0.846.

One more thing fell out: `answerable_median` in `evaluate.py` was
`sorted(scores)[len(scores) // 2]`, the upper middle value rather than the
median. On the 26 even-sized bird cases that reported **+1.21** where
`analytics.json`, which uses `statistics.median`, said **+1.15**: one quantity
reported as two values in two documents. Now `statistics.median` in both. §2 updated.

**Then the bird threshold, which the correction above made worth re-deriving.**
`-3.0` was calibrated against confidences measured at rerank blend 0.00 on a
corpus that ships 0.20, so it was a gate tuned to a pipeline the server does
not run. On the re-measured scores `-3.0` is **dominated**: every threshold in
`(-6.48, -4.55]` catches the same five of six adversarial cases and wrongly
refuses **three** of twenty-six rather than four. Shipped **-5.5**, the middle
of that interval, which leaves 0.95 of margin before it refuses an answerable
question and 0.98 before it stops catching an adversarial one. An edge of the
interval would fit the threshold to a single case.

Retrieval is untouched, with any-hit at 0.846 and MRR at 0.614. Only the gate
moved, and
it recovered `bird-incubation`, whose answer the pipeline had already retrieved
into the top five.

Checked on all three corpora before shipping, because the trap is tuning to one:
the ML papers' 0.0 and quant's -4.0 are both already on the frontier, where
every lowering costs catches. Only birds was dominated.

`src/calibrate_threshold.py` could not have found this. Its grid was hardcoded
to `[-4 .. +4]`, which is where the ML papers' scores live and nowhere near the
bird corpus's, so the tool used to calibrate that corpus could not display the
region being calibrated. The grid is derived from the scores now, and it reports
the interval a threshold sits in rather than only a grid point, because -4.6 and
-5.9 do the same thing today and only one of them survives the corpus growing.

**What this does not fix:** `bird-dawn-chorus`
(-10.68) and `bird-imprinting` (-8.28) are questions whose answer the pipeline
retrieved into the top five and whose passage the cross-encoder then scored
below three of the six adversarial cases. No threshold recovers those without
losing catches. That is the reranker's calibration, not the gate's, and it is
the next item.

The original entry, kept because the history is the argument:
`results.json` was shared by every corpus so the last run owned it;
`RESULTS.md` claimed to be copied from it and was typed by hand; `per_case.json`
had the same shared-path bug and had not been regenerated for four days, so the
interface was offering questions that had just been deleted as unanswerable. Only
`RESULTS.md` had a `--check`, and the chain `per_case -> analytics -> the
interface` had no freshness test at all, which is what made it the most
valuable small thing left and what the entry above closes.

**No longer blocked, and both were measured.** The API key still has no
credit, and neither path needed it in the end. Generated prose runs against a
local Ollama and is scored by `src/evaluate_answers.py` on all three corpora.
Query decomposition runs the same way and was scored on 2026-08-31: it recovers
**none** of the seven structural cases and costs 0.045 any-hit at 21 times the
latency, so it is measured and rejected rather than pending. The interface
still shows the retrieved passage verbatim, which is a choice now rather than a
limitation.

**How to unblock each of them without Anthropic credit**, written down
2026-08-27 so the next attempt can act rather than re-derive:

1. **Generation, end to end. Built 2026-08-27.** `src/generate_local.py`
   speaks the same `synthesize(question, chunks)` contract to a local **Ollama**,
   selected by `RAG_GENERATOR=ollama`, and `generate.synthesize_with_backend()`
   is what `api.ask` and `serve.chat` now call so neither has to branch.
   `src/test_generate_local.py` stands up a real HTTP server implementing
   Ollama's two endpoints and drives the module over a socket. That is 24
   checks, and the first in this repo where the generation path executes end
   to end.

   **Still outstanding:** no real model has answered. Install Ollama, `ollama
   pull llama3.2`, set `RAG_GENERATOR=ollama`, and `ask(generate=True)` writes
   prose for the first time. The transport is proven; the model is not.

2. **Query decomposition, for the seven ML cases.** Splitting "normalisation
   across features rather than examples" into topic and constraint is what an
   LLM does well, and a local instruct model does it well enough to *measure*.
   Score it against `eval/hard_cases.json`, which exists precisely for this and
   has been verified stable across a threshold change, a blend change and a
   corpus that grew. If a 7B local model moves 2 of the 7, that is the finding;
   the shipped implementation can still call a better model later.

3. **The five description cases. Measured 2026-08-27, and the answer is
   yes.** `src/sweep_ensemble.py` fuses the shipped embedder with a second one
   by RRF. Fused with `bge-small-en-v1.5` it is **weakly dominant**: +0.015 pool
   recall on the ML papers, +0.038 on birds and level on quant. It is the
   first retrieval change here that is better or equal on all three rather
   than a trade. Bird pool recall reaches **1.000** and `bird-alula` leaves the
   structural fixture.

   **Not shipped, for engineering reasons rather than evidence:** a second
   embedder is a second vector per chunk, so the index roughly doubles, ingest
   gains an encode pass, and `retrieve()` grows a second dense arm to fuse
   before the existing dense+BM25 fusion. That is a change to the shape of the
   store and wants its own session with its own re-index and re-calibration.
   Two cautions live in the log: `bge-small` **alone** beats the fusion on the
   ML papers (0.910 vs 0.881), and the `multi-qa` pairing costs quant 0.057, so
   "fuse two dense models" is not a general improvement.

4. **The reranker.** Exhausted among off-the-shelf options:
   `compare_rerankers.py` covers three and `sweep_quantized.py` covers int8. The
   remaining honest move is a **fine-tune** on this project's own labelled data,
   which is 157 cases and probably too few, or accepting the ceiling and saying
   so. Not credit-blocked; data-blocked.


1. **Closed 2026-08-21. The spreadsheet assumption was wrong.** The loaders
   have now run on real files. `load_xlsx` took row 1 as the header; a sheet whose
   first row is a *title* made every row read `Q3 sales report: 41200`. The
   title was repeated as the column name, the real headers were demoted to
   data, every value was unlabelled, and nothing errored. Fixed by finding the header rather than
   assuming it. `load_docx` and `load_pptx` were correct as written, tables and
   speaker notes included. `src/test_loaders.py` writes a real file per format and
   reads it back: 15 checks, verified to fail 3 when the old assumption is put
   back.

   The corpus is now **36 documents / 5,459 passages**. Two further topic corpora
   are built by `src/fetch_topic.py`: `data-birds/` (45 documents, mixed
   `.docx`/`.pptx`/`.xlsx`/`.pdf`, from Wikipedia) and `data-quant/` (18 arXiv
   q-fin papers). Both are gitignored and rebuildable. `RAG_STORE_DIR` now moves
   the index with `RAG_DATA_DIR`, so indexing a second corpus no longer overwrites
   the first one's store.

   **Closed since, and this paragraph was stale until 2026-08-27.** The OCR path
   has seen a scanned document since 2026-08-23: `src/test_ocr.py` builds one by
   rendering a page to an image and putting it on a fresh page, then asserts the
   text is unextractable before OCR and recovered after: 8 checks, including
   that a page with no text yields no invented text. Both new corpora have
   golden sets and are measured: 26 + 6 bird cases, 35 + 6 quant.
2. **Two structural failure modes, ~10% and ~6%.** Measured on all three
   corpora 2026-08-27, six configurations each: 7 of 67 ML cases (10.4%), 3 of
   26 bird cases (11.5%) and 2 of 35 quant cases (5.7%) fail under *every*
   configuration. **That floor is relative to the first stage, not absolute.**
   All six configurations shared one embedder, and a different embedder reaches
   `bird-alula` immediately; see the embedder comparison in the log. The ~11%
   floor reproduces between ML and birds and does not hold on quant, so do not
   quote it as universal.

   Reading them together gave **two problems**, and reading them one at a
   time on 2026-09-01 gave **three**, of which only one is retrieval. The
   earlier account is left below the corrected one because it was quoted for
   five days and the correction is the more useful half.

   - **Five are genuine retrieval failures, and they are all that remain.**
     `wmt14` returns English-to-French passages for an English-to-German
     question, `gpt3-params` asks for the largest autoregressive model and
     matches every discussion of model size, `roberta-nsp-drop` never returns
     `roberta.pdf` at all, `bird-precocial` returns passages about hatching
     that never use the word, and `bird-hollow-bones` returns the right
     document's Overview and Axial skeleton sections and misses its Skeletal
     system section. Nothing currently reaches these.
   - **Four were answered and scored wrong, and were corrected 2026-09-01**,
     because a derived label marks every passage containing the answer string
     rather than every passage that answers. `adam-bias` returned "we therefore
     divide by this term to correct the initialization bias" one page from its
     gold pages, and `t5-text2text` had four of its six gold passages in other
     papers' bibliographies, since "unified text-to-text" is part of the T5
     paper's title.
   - **Three were questions the documents do not answer**, and moved to the
     adversarial half the same day. `bird-alula`, `qf-mean-reversion` and
     `qf-momentum` name terms the corpora use without explaining, as a factor
     name or in a list of anatomical features. The gate already refused all
     three at their shipped thresholds.

   `src/failure_overlap.py`, re-derived from six configurations on each corpus
   after the corrections, returns exactly the five above, which is the harness
   agreeing with the reading rather than repeating it.

   **Six approaches have been measured against those five and none ships.**
   Four of the five never reach the reranker, because RRF rewards agreement and
   the two retrievers disagree sharply on them: BM25 ranks `wmt14`'s answer
   11th and the embedder ranks it 138th, so it loses to chunks both rank in the
   middle. Only `bird-hollow-bones` reaches the pool and is dropped afterwards,
   and not by the diversity cap, which leaves the top five identical. Refuted
   on the corrected labels: a deeper pool, every rerank blend, three
   cross-encoders, window-level scoring, and retrieving on a hypothetical
   answer (`src/sweep_hyde.py`), which recovers three of the five and costs
   nine questions on the papers while tripling the adversarial cases that slip
   the gate. Do not reach for any of these again without reading
   `docs/engineering-log.md` under 2026-09-01 first.

   **Gating that last one on the abstention threshold does not work either**,
   and the reason matters beyond it. Measured 2026-09-03 by
   `src/hyde_trigger.py`: all eleven questions the three corpora get wrong score
   ABOVE their own corpus's threshold, `lora-latency` at +6.36 against 0.0 and
   `bird-hollow-bones` at +5.03 against -5.5, so a fallback fired on abstention
   would run on four working questions and twenty-two correctly-refused
   adversarial ones and on none of its targets. The system is not uncertain when
   it is wrong. The gate separates answerable questions from unanswerable ones
   and is not a wrongness detector, which rules out every design that treats a
   low score as a signal of error.

   The account this replaces, written 2026-08-27: the seven on the ML papers
   were cross-document confusion, where the topic matches a dozen documents
   and the distinguishing clause is ignored, and the five on birds and quant
   were vocabulary mismatch, where the question describes a term and asks its
   name. Query decomposition was the named fix for the first and **was
   measured on 2026-08-31 and recovers none of the seven**.
   `src/query_expansion.py` (RM3 pseudo-relevance feedback) was **measured
   against the second on 2026-08-27 and recovers 0 of 12**, with no question
   changing hands on any corpus, because feedback terms are harvested from the
   top results of the original query and the missing word is not in those
   passages either. A stopword leak found in the same module was fixed and
   measured **worse** on two corpora of three, and reverted.

   Fixtures: `eval/hard_cases.json`, `eval/hard_cases-birds.json`,
   `eval/hard_cases-quant.json`.

   **Cross-document confusion: a ~11% floor, not a flat 15%.** The retriever
   matches the topic and ignores the constraint that distinguishes the answer
   ("normalisation across features *rather than examples*" still returns Batch
   Normalization). Mechanism understood, and now both candidate fixes are ruled
   out by measurement: title prefixing and a larger reranker in §4, and query
   decomposition on 2026-08-31, which recovered none of the seven and cost the
   corpus three questions. Rewriting recovered one and broke five. No fix for
   this failure mode is currently known.

   Sharpened 2026-08-21 by `src/failure_overlap.py` across six configurations
   (see `eval/RESULTS.md`). **Re-measured 2026-08-27** across the same six
   configurations: **21 of 67** answerable cases fail under *some*
   configuration, and **7 fail under all of them**. They are the **same seven
   ids** as on 2026-08-21, unchanged through a threshold recalibration, a
   rerank-blend change, a corrected label and a corpus a case larger. The
   earlier "18 of 66" was measured before the golden set grew.

   `adam-bias`, `dropout-rate`, `gpt3-fewshot`, `gpt3-params`,
   `roberta-nsp-drop`, `t5-text2text`, `wmt14`. Reproducible, unreachable by any
   fusion/rerank/cap change, and the thing query decomposition was scored
   against on 2026-08-31, which is how that fix came to be ruled out. Regenerate with `src/failure_overlap.py --emit-fixture
   eval/hard_cases.json` over six `src/per_case.py` runs. The other eleven move with
   ranking and should not be counted as the same problem.

   A tempting explanation was tested and refuted: three of the seven return the
   right document at rank 1 and miss on page, which looks like incomplete
   labels rather than bad retrieval. Context recall is 0.000 for all seven, so
   the answer text was not returned at all. They are real misses.
3. **Generation is unverified against a real model.** `generate.py` targets
   `claude-opus-5` and has never completed a real call. As of **2026-08-27 the
   path itself does execute**: `generate_local.py` speaks the same contract to a
   local Ollama and `test_generate_local.py` drives it over a real socket
   against a fake one, so prompt assembly, the citation format, the refusal
   branch and the empty-response branch have all now actually run. What has
   never happened is a real language model answering. Three defects were also
   fixed by reading the code against the data it receives (§5b), and everything
   up to the network boundary is exercised with a stubbed client. One judgement call left open deliberately: the current
   API guidance is to pass server-side `fallbacks` on `claude-opus-5` calls so a
   safety decline reroutes rather than stopping. It is **not** added here. It
   is an unverifiable beta parameter on a path that has never run, and a
   refusal on grounded Q&A over ML papers is close to hypothetical. `synthesize()` raises a
   named error on `stop_reason == "refusal"` instead. Worth adding the moment
   there is credit to test it with.
4. **Folder linking. Done 2026-08-21**, as a pasted path; see item 7. The
   constraint recorded here was right and is why the design is what it is: a
   browser cannot read a filesystem path from a file picker.
5. **Google Drive native docs.** `.gdoc`/`.gsheet`/`.gslides` are pointers,
   not files. Needs the Drive API export path in `loaders.py`. Uploaded PDFs/Office
   files already work today via `RAG_DATA_DIR`.
6. **No permissions model.** Fine for a local single-user tool; would matter if
   this ever served more than one person.
7. **Drive intake is interface-only; local folders now work.** The app takes a
   pasted folder path, checks it (`POST /api/index/inspect` reports what would
   be indexed and what would be skipped, and refuses a folder with nothing
   indexable in it), then indexes it. Drive is still a badged tab: native
   `.gdoc`/`.gsheet` files are pointers and need the export API. Files that are
   genuinely files in a synced Drive folder already work. Point the folder
   intake at the synced directory.

8. **The settings controls re-run retrieval.** Done 2026-08-21. `/api/trace`
   takes `rerank`, `fusion`, `max_per_source` and `expansion`; the panel runs
   the question twice, once at defaults and once at the current settings, then
   reports what changed between them for that one question. The aggregate
   numbers beside each option still come from `eval/results.json` and still
   describe 84 cases; the two answer different questions on purpose.

   Two honesty constraints fell out of building it, both load-bearing:

   - With reranking off there is **no verdict**. The abstention threshold is
     calibrated on cross-encoder scores; RRF scores are bounded near 0.03 and
     always positive, so testing one against 0.0 would pass every query
     including the ones that should be refused. `verdict.confident` is `null`
     and the UI has a third state for it.
   - `retrieve()` takes a **different path** when reranking is off. It
     shortlists `k` rather than `candidate_k` and skips the diversity cap
     entirely. `pipeline_trace` now mirrors that rather than drawing a
     twenty-candidate pool and a cap the serving path never ran.
     `src/test_trace.py` asserts the two agree across the option matrix
     (80 checks). That test is what the module's docstring had claimed existed
     since it was written; it did not.

9. **Two of the three unbuilt journey pieces are still unbuilt:** the pool
   visibly growing as documents are added, and Act 1. The third is unblocked:
   `src/per_case.py` writes every case's outcome to `eval/per_case.json`
   (84 rows: retrieved passages, per-passage relevance, confidence, and one of
   `found` / `missed` / `refused` / `refused_wrongly` / `answered_anyway`).

   It closed a real gap on its first run, and the gap is now closed properly.
   `eval/results.json` measured the diversity cap only on the **weighted**
   fusion branch, so the row every document quoted as the default was
   `weighted + cap 2`, while `DEFAULT_FUSION` is `"rrf"`, which is what
   `api.ask()` and the app serve. The served configuration had never appeared
   in the aggregate table at all.

   **Corrected 2026-08-21.** `evaluate.py`'s `finals` list now runs the cap on
   RRF and orders the rows so the indent means what it looks like. Five numbers
   moved, none by more than 0.004:

   | | any-hit | MRR | NDCG | src recall |
   |---|---|---|---|---|
   | cap 2/src, was | 0.848 | 0.751 | 0.765 | 0.773 |
   | cap 2/src, now | 0.848 | **0.754** | **0.769** | 0.773 |
   | cap 1/src, was | 0.818 | 0.739 | 0.751 | 0.816 |
   | cap 1/src, now | 0.818 | **0.742** | **0.753** | **0.817** |

   So no conclusion drawn from that table was ever wrong, which is exactly why
   it survived. What makes the correction worth trusting: `per_case.py` had
   already measured the served configuration independently, and the harness now
   reproduces its figures to three decimals from a separate code path.

Ranked by value: **(1) is worth more than everything else combined**, and it is
unblocked by pointing the system at more documents rather than by writing
code. The folder intake in (7) is the mechanism, so files no longer need to be
copied into `data/`. (8) is done.

**The cross-encoder question is now answered, and the answer is no.** Measured
on all three corpora 2026-08-27: MiniLM-L12 and BGE-reranker-base both trade a
gain on one corpus for a loss on another, and the two halves of the reranker's
job, ordering and scoring, move in opposite directions as the model grows.
Neither is shippable. The mechanism behind the remaining failures is questions
that *describe* a term rather than naming it, and a cross-encoder of any size
reads the same words. Query decomposition was the fix that addressed it on
paper, and measured on 2026-08-31 it reaches none of them. Full numbers in
`docs/engineering-log.md`.

Test counts, as of 2026-09-03: **456 checks plus the route suite**: 80 trace,
66 answer-quality judge, 20 excerpt, 18 links, 33 loaders, 33 freshness, 25
serve, 24 local generation, 22 metrics, 19 generate, 17 golden-set audit, 17
analytics, 13 api, 10 reranker cache, 9 ingest cache and 8 OCR, which is 414,
plus 33 answer-highlight checks under `node ui/test-answer-mark.mjs` and 9
passage-selection checks under `node ui/test-passages.mjs`.

These are counted by running the suites. They were wrong until 2026-08-30,
when the total read 384 because the list still said 28 loaders, a figure five
behind since `_unhash` added five cases, and the total had been computed from
the stale item. Since 2026-08-31 `python src/check_docs.py --tests` runs every
suite and compares its total against the sentence above, which takes two to
three minutes and is why it is opt-in rather than part of the default run.

The 60 on the judge are worth their own sentence, because that module is the
only measurement here whose input is prose, and prose is where a string test
goes wrong quietly. Most of those checks are answers this repository has
already scored wrongly, kept verbatim.

Six guards now cover the things that have gone wrong silently before, and all
six exit non-zero rather than printing a warning nobody reads. The count in
this sentence disagreed with the list under it until 2026-08-30, which is the
failure `check_docs.py` exists to stop, one level up:

```
python src/check_freshness.py     # golden set -> per_case -> analytics -> front page
python src/check_golden.py        # do the labels still describe the corpus?
python src/check_docs.py          # do the documents match the measurements?
python src/check_links.py         # do the links the interface serves go anywhere?
python src/build_results_doc.py --check
python src/build_corpus_manifest.py --check   # does the document list match the indexes?
```

---

## 8. Published artifacts

Their **source is in this repo**; the published copies live in the cloud and are
not carried by any session.

| Artifact | Source file | URL | Updatable from `the owner's address, removed 2026-09-07`? |
|---|---|---|---|
| **Understanding This Retrieval System** (the complete guide) | `docs/understanding-rag.html` | https://claude.ai/code/artifact/an artifact id, removed 2026-09-07 | **yes** |
| Anatomy of a Retrieval Pipeline (Phases 1–3) | `docs/phase-1-field-notes.html` | https://claude.ai/code/artifact/an artifact id, removed 2026-09-07 | **yes** |
| ” (earlier copy, other account) | ” | https://claude.ai/code/artifact/an artifact id, removed 2026-09-07 | no — see below |
| Retrieval System Status | `docs/project-status.html` | https://claude.ai/code/artifact/an artifact id, removed 2026-09-07 | **yes — use this one** |
| ” (earlier copy, other account) | ” | https://claude.ai/code/artifact/an artifact id, removed 2026-09-07 | no — tested 2026-08-21, same error |

**To update one, pass its URL.** Publishing the source file without the `url`
creates a *second, separate* artifact instead of updating the existing one, and
the existing link silently goes stale. This is the single easiest way to
break something here, and nothing in the file itself warns you, which is why
the URLs are recorded here.

### The `8c62ba9a` and `0fa6a672` URLs, which this account does not own

Discovered 2026-08-21 while updating the field notes. `action: "list"` on
`the owner's address, removed 2026-09-07` returns eight artifacts and **neither of those UUIDs is
among them**, so they were published from somewhere else. That produces a
deadlock rather than a clean error, and it is worth recording so the next
session does not spend the same time on it:

- Publishing to that URL is refused until the session has read the live version.
- Reading it is refused: *"served to you as a public (non-member) reader, and
  reading public artifacts that way is not enabled yet."* Making the artifact
  public does **not** lift this; it was already being served that way.

So the only route to that URL is `force: true`, which overwrites the live copy
without seeing it. The owner was asked and chose to **keep both** rather than
force. The 2026-08-21 corrections (§5b, §7) are therefore live on
`c8fef8f2` and **absent from `8c62ba9a`**, which still shows MRR 0.751, the
"bump chart" module row, and "indexing progress is emitted but unread".

If access to the original account turns up, publishing `phase-1-field-notes.html`
from there with the `8c62ba9a` URL updates it in place with no force and no risk,
and the two copies converge.

Both carry a banner scoping Phase 1–2 figures as historical. Phase 3 reflects the
system as it stands. If Phase 4 is added, extend that banner rather than letting it
disclaim current work.

---

## 9. Further reading

| File | What it holds |
|---|---|
| `README.md` | project overview, current numbers |
| `eval/RESULTS.md` | every measurement and the default it justifies |
| `eval/results.json` | machine-readable, regenerated by `src/evaluate.py` |
| `docs/roadmap.md` | what is worth doing next, what has been ruled out, and why |
| `docs/ui-brief.md` | the UI design interview and direction (superseded in part) |
| `eval/per_case.json` | every golden-set case's outcome, written by `src/per_case.py` |
| `src/test_trace.py` | asserts the trace and the serving path agree under every option |
| `src/test_api.py` | asserts `ask()` answers from the corpus it was handed, not a cached one |
| `src/smoke_routes.py` | asks every dispatched route, against a running server |
| `src/test_routes.py` | the same checks, against a server it starts itself |
| `src/test_ingest_cache.py` | indexing one corpus must not evict another's caches |
| `src/test_generate.py` | everything `generate.py` does short of the HTTP request |
| `src/generate_local.py` | the same contract against a local Ollama, so the path can run |
| `src/test_generate_local.py` | that path over a real socket, against a fake Ollama |
| `src/sweep_ensemble.py` | two embedders fused; the first change to move the fixture |
| `src/sweep_decompose.py` | rewriting and decomposing the question, scored against the cases nothing else reaches |
| `src/sweep_hyde.py` | retrieving on a generated hypothetical answer, scored on every case and on the gate |
| `src/hyde_trigger.py` | whether any signal can trigger that fallback selectively. None can |
| `src/audit_page_credit.py` | how often a hit credits the right page while showing the reader no answer |
| `src/query_rewrite.py` | the rewrite and decompose strategies themselves, against a local model or the API |
| `src/test_serve.py` | request handling, the offered questions, and the folder intake |
| `docs/engineering-log.md` | every attempt in full, including the refuted ones — the why behind §4's table |
| `src/check_freshness.py` | is the front page still offering the questions the harness measured? |
| `src/check_golden.py` | does each golden set still describe the corpus it scores? |
| `src/check_links.py` | do the interface's links resolve, including the repository and branch a GitHub link names? |
| `src/test_evaluate_answers.py` | the answer-quality judge, against answers it has already scored wrongly |
| `src/check_docs.py` | do the numbers written in HANDOFF §2 and the README match the measurements? |
| `src/sweep_fusion.py` | every fusion, every corpus, at the configuration served |
| `src/profile_query.py` | where the time in one query goes, stage by stage |
| `src/sweep_candidates.py` | what reranking fewer candidates costs, in quality and in ms |
| `src/compare_rerankers.py` | ranking **and** gate separation for a candidate reranker, on every corpus |
| `ui/answer-mark.js` | which words of a passage are set bold, and the bounds on that |
| `src/test_freshness.py` | stages each known way that chain has gone stale and asserts it is caught |
| `docs/phase-1-field-notes.html` | mechanism-level explanation, Phases 1–3 (published artifact) |
| `src/evaluate_answers.py` | the generated prose, not the passages: invented citations, refusals, groundedness |
| `src/build_ensemble_index.py` | adds a second dense index to a store without touching the first |
| `src/uniform_baseline.py` | all three corpora under one untuned configuration — the fairness check |
| `src/cli.py` | the third shell over `api.ask`, so library, HTTP and CLI cannot drift |
| `src/hard_cases.py` | only the cases nothing gets right, in about twenty seconds |
| `src/dump_top_passages.py` | every golden question's top passage, as the page receives it |
| `src/verify_candidates.py` | answer strings checked before they are allowed into a golden set |
| `src/check_answers.py` | answer strings checked against the corpus **as parsed**, not as the PDF renders |
| `src/label_multisource.py` | proposes every document that answers a case, not one |
| `src/diagnose_crossdoc.py` | cross-document confusion measured instead of argued from one example |
| `src/eval_probe.py` | single-stage retrieval beside retrieval + reranking, before there were labels |
| `src/build_guide_figures.py` | the data figures in `docs/understanding-rag.html`, drawn from the measurements |
| `src/record_static.py` | records every answer to files, for a deployment with no models behind it |
| `docs/deploying.md` | how the recorded build is made and where it is hosted |
| `docs/angle-sweep.html` | every version of the front page in order, the wrong turns included |
| `docs/corpus-manifest.md` | every document in every corpus — the answer to "what files is this?" |
| `src/build_corpus_manifest.py` | writes that list from the indexes; `--check` fails when it has drifted |
| `src/retitle.py` | recomputes stored titles in a built index without re-embedding it |
| `src/test_analytics.py` | the analytics the quality page plots, against the per-case file |
| `src/test_links.py` | the link checker, staged against both links that actually broke |
| `src/test_excerpt.py` | which 260 characters of a passage the interface shows, against the passage it got wrong |
| `src/front_matter.py` | a paper's title block, detected; a refuted experiment, off by default |
| `src/sweep_front_matter.py` | what dropping title blocks costs end to end, on every corpus |
| `src/audit_title_credit.py` | whether a hit satisfied by a title-block chunk is a false credit; on this corpus, none are |
| `git log` | why each decision was made, including the reversals |
