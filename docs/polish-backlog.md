# Polish backlog

The 22 findings the repository audit of 2026-09-05 rated below the bar for
fixing that week. The 3 it rated dire were fixed the same day and the 30 it
rated should were fixed on 2026-09-06 and 2026-09-07, both recorded in
`docs/engineering-log.md`. These 22 are what was left.

**This file exists because the list was nearly lost.** It lived only in a
workflow result inside one session, that session was compacted, and for 7 days
the only record was a sentence in the log saying the items "are in the audit's
report". They were recovered on 2026-09-14 from the workflow journal on disk,
which is luck rather than a filing system. A backlog that is not in the
repository is not a backlog.

**Status was re-probed against the tree on 2026-09-14**, at commit `8606c7e`,
because the intervening work touched most of these files. 18 of the 22 were
still live. The file and line references come from the audit, so read them as
pointers rather than coordinates.

**Re-probed again on 2026-09-17**, after the interface rebuild of 2026-09-15
and the spreadsheet rendering change. 1 item is closed, 3 are smaller, and the
other 18 reproduce unchanged. Each status line below says which. The probe was
mechanical, a grep for the thing the finding names, so a line saying an item
reproduces means the code it points at is still there and not that it was read
again in full. Item 18 was re-checked against a recording taken the same day.

Nothing here is a defect in a published number. The audit's own summary of this
tier was "items of polish".

## Serving and runtime

**1. The agreement flag hard-codes 2 arms.** `src/pipeline_trace.py:402`
compares against a literal 2 while `arm_count` is computed at :385, so a third
retrieval arm would leave the flag wrong. Use `len(c["contrib"]) == arm_count`.
Still live.

**2. `chat()` takes the lock twice and mutates shared metadata.** It resolves
chunks in a second acquisition after the trace, and uses `setdefault` on the
dicts in `RES.metadata`, which every other request shares. Resolve inside the
same `with RES.lock:` block and copy instead, `{**RES.metadata[cid],
"chunk_id": cid}`. Still live.

**3. Heavy ML imports run at module import.** `from sentence_transformers
import SentenceTransformer` at `api.py:23` and the faiss import in
`retrieve.py` cost about a minute on suites that never touch a model. Move them
inside `_resources()` and `load_index`/`search`. `evaluate.py` was given this
treatment on 2026-09-06 and these two were not. Still live.

**4. Small HTTP hygiene gaps.** `do_POST` does not strip a query string the way
`do_GET` does. `version_string()` is not overridden, so every response carries
`Server: BaseHTTP/0.6 Python/3.12.10`. The image runs as root, with no `USER`
line after the `COPY`s. All 3 still live. The 413 connection close and the
Dockerfile's 40 s comment, also in this finding, were fixed on 2026-09-07.

## Guards and tests

**5. Both node sweeps skip silently on a fresh clone.** `ui/test-answer-mark.mjs`
and `ui/test-passages.mjs` print a note and pass when `eval/top-passages.json`
is absent, which it is in any clone, so the documented counts describe a run
nobody else can reproduce. Exit 2 with `(sweep skipped: fixture missing)`, or
document the fixture-less counts beside the full ones. Still live.

**6. `check_docs.py` has no test.** It is the guard the other guards lean on,
and it has twice carried a pattern that silently matched nothing. Stage it
against a synthetic HANDOFF, README and results.json the way `test_links.py`
and `test_golden.py` do. Still live, and the most valuable item here.

**7. `check_links` reads only `href` attributes under `ui/`.** Markdown links
in README and `docs/*.md` and the standalone `docs/*.html` pages are outside
it. Extend `hrefs()` to those, or reword the README line to say "the
interface's links". Partly addressed on 2026-09-07, when bare filenames such as
`tokens.css` stopped being treated as external.

## Data and packaging

**8. Evaluation files carry no provenance.** 10 of 36 `eval/*.json` lack
`generated_by` and 27 lack an `inputs` digest, so `check_freshness` cannot
reach them. Have the writers in `sweep_*.py`, `compare_*.py`,
`profile_query.py` and `dump_top_passages.py` emit both, and add a generic
digest loop over `eval/*.json`. Two parts of this finding are done: the
absolute path in `results.json` became relative on 2026-09-07, and 3 writers
were stamped on 2026-09-06.

**9. No `.gitattributes`, and `.gitignore` is narrower than its siblings on
secrets.** 122 files are `i/lf w/crlf` only because of a machine-wide
`core.autocrlf`. Add `* text=auto eol=lf` and `*.woff2 binary`, and add
`.env.*`, `*.key` and `*.pem` to `.gitignore`. Still live, and it is the reason
every commit in this repository prints a CRLF warning.

**10. `numpy` and `torch` are imported directly and absent from
`requirements.txt`.** `pip install -r` does not reproduce the Docker
environment; the Dockerfile installs torch unpinned from the CPU index. Pin
both. Still live.

**11. The static demo sends only `Cache-Control`.** Add a `/(.*)` rule in
`record_static.py` for `X-Content-Type-Options: nosniff`, `Referrer-Policy`,
`X-Frame-Options: DENY` and a `default-src 'self'` CSP. Still live, and it is
the one item here with any security content, on a page that is now public.

## Interface

**14. Dead CSS, hardcoded colours and unused JS symbols.** `.picker`, `.menu`,
`.sub2`, `.row2`, `.warn2`, `.oldlink`, `.legend`, `.fignum`, `.pills`,
`.genrow` in `index.html` and `.callout` in `about.html`; `"#f0685f"` twice in
`pipeline-map.js` and once in `versions.html` where `ink.critical` exists;
`easeOut`, a third argument to `slot()` that is ignored, and unused `variant`
and `sig` parameters. One correction to the audit: `.wrapviz` was in its dead
list and is now load-bearing, added on 2026-09-07 to make the diagram scroll on
a phone.

Smaller on 2026-09-17. Of the 10 selectors, 4 are still declared once and
never used: `.sub2`, `.warn2`, `.fignum`, `.pills`. The interface rebuild of
2026-09-15 either removed or started using `.picker`, `.menu`, `.row2`,
`.oldlink`, `.legend` and `.genrow`, and `.callout` in `about.html` is now
referenced. `#f0685f` is still written out 3 times.

**15. `quality.html` resets all motion and reads analytics unguarded.** The
blanket `prefers-reduced-motion` rule should be scoped the way `index.html`
does it, and the boot should sit in a `try/catch` that writes to `#readout` so
a partial `analytics.json` says which field is missing.

**Closed on 2026-09-17.** The rewrite of the analytics page on 2026-09-15 did
both: the blanket rule is gone and the boot is guarded.

**16. Accessibility gaps.** `about.html` has no `:focus-visible` rule at all.
The `quality.html` tiles need a role and `aria-describedby` pointing at `#tip`.
The listbox roles should either move focus with the arrow keys or be dropped.
Still live. The answer tabs got full keyboard support on 2026-09-07; these did
not.

Smaller on 2026-09-17. The audit also asked for `scope="row"` on the
spreadsheet cells in `index.html`. There are no spreadsheet cells any more: a
row out of a spreadsheet is quoted as prose like every other answer, so that
clause is closed by deletion rather than by a fix.

**17. `versions.html` is a self-declared temporary page.** Its own comment says
to delete it. Either do that, with `ui/versions/` and its 2 data files, or
promote it and make it import the shared tokens instead of its own copy. This
is the open `/versions` decision.

## Documents

**12. Two tracked pages load Google Fonts, and the README has no licence
line.** `docs/angle-sweep.html` and `docs/understanding-rag.html` pull from
`fonts.googleapis.com`, which is the thing `ui/` bundles fonts to avoid. Point
them at `ui/fonts` or say why they differ. Add a Licence section to the README
pointing at `LICENSE` and `ATTRIBUTION.md`. Both still live, and the licence
line matters more now the repository is public.

**13. Three unused imports.** `urlparse` in `check_links.py`, `re` in
`label_multisource.py`, `is_relevant` in `sweep_neighbours.py`. 2 of the 3 still live on 2026-09-17:
`is_relevant` in `sweep_neighbours.py` no longer reads as unused. The other two
are unchanged, and `urlparse` was explicitly left alone on 2026-09-07 as out of
that chain's scope.

**19. HANDOFF section 5 describes interface pieces it says were removed.** The
Direction 2026-08-20, Scope, The city and Ambient fields subsections name
deleted files. Collapse them into a short history note with the log dates.
Still live.

**20. README and roadmap list different subsets of the suites.** Neither names
`test_serve`, `test_api`, `test_excerpt`, `test_generate`, `test_ingest_cache`,
`test_ocr`, `test_rerank` or `test_analytics`. Point both at
`python src/check_docs.py --tests`. The roadmap also has 2 ruled-out bullets
for one experiment. Still live.

**21. The engineering log is out of date order.** 8 of 65 entries break
descending order as of 2026-09-17, unchanged in count. Either move them or say at the top that the log is grouped by
topic. The same finding objects to the headings being sentences rather than
labels, which `docs/writing-style.md` asks for everywhere else. Still live, and
this file's own headings follow the style rule.

**22. Two README phrases rate the result rather than state it.** "dramatically
slower" should be the measured factor. The other half, "the result that matters
most", is gone.

## Measured behaviour

**18. The showcase question about the fused collarbone answers with the wrong
bone.** Re-checked on 2026-09-17 against a recording taken the same day, the
third time, and it is unchanged to three decimals. For "What is the fused collarbone of a bird called?" the lead
passage is `bird_anatomy.docx` at **+1.869**, and it is about the pygostyle,
which is fused caudal vertebrae at the other end of the bird. The answer sits
second at **+0.325** in `origin_of_birds.pptx`, which says in as many words
that birds "had clavicles (collar bones) fused to form a bone called the
furcula". The gate passes it as confident because the gate reads the
cross-encoder's score, and the cross-encoder preferred the wrong passage.

This is the only item in this file a visitor can see, and it is offered on the
front page as a suggested question. It is also a clean instance of the
project's own headline finding: the answering chunk does not repeat the
question's words, `fused` appears in both, and the ranker went with the wrong
one. Either record it in `eval/answer-quality` as a known miss, or choose a
different chip, or leave it and point at it, which is the most honest of the
three and costs nothing.
