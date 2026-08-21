# Next session — start here

Written 2026-08-20, rewritten 2026-08-21 after the four technical items in the
previous version were built. Paste the block below into a new session, or just
read it. It is a pointer, not a substitute: `HANDOFF.md` is the cold-start
document and carries the reasoning behind every decision named here.

---

```
Project: C:\Users\owner\Desktop\rag-project

Read HANDOFF.md FIRST — it's the cold-start doc, current as of 2026-08-21.
docs/ui-brief.md is superseded in part; where it and HANDOFF §5 disagree, §5 wins.

DIRECTION: dial the UI back. Keep it clean and simple. Focus on the technicals.

The UI went through four visual passes in one session and landed somewhere I
wasn't happy with. The lesson is sequencing, not taste — the interface was being
designed ahead of the capabilities it's meant to expose. So: don't start another
whole-page visual pass. Simplify what's there. Prefer plain conventional
components over anything bespoke. The isometric "city" is the most elaborate
thing on the page and is a fair candidate for simplifying or removing — it does
explain the mechanism well, so ask me before deleting it, but don't expand it.

Run the server:  .venv\Scripts\python.exe src\serve.py  →  http://127.0.0.1:8000/

Constraints:
- Don't touch the retrieval code unless the task is explicitly about retrieval.
  src/pipeline_trace.py is now an exception in one direction only: it takes
  pipeline options, and src/test_trace.py asserts it still agrees with
  retrieve() across all of them. Change one and run that test.
- Chart palette is validated all-pairs for colourblind separation — no fourth
  hue without re-running the dataviz validator (HANDOFF §5).
- No CDN. Fonts bundled in ui/fonts/ deliberately: the corpus may be private and
  the page must make no external request.
- Light theme only.

WHAT IS NOW WORTH DOING, in priority order:

1. Point it at real documents. This is worth more than everything else
   combined and only you can do it: the corpus is 20 PDFs, and load_docx,
   load_pptx, load_xlsx and the whole OCR path have never run on a real file.
   The mechanism now exists — the Local folder tab takes a pasted path, checks
   it, and indexes it. The .xlsx path is the least trustworthy: rows are
   serialised to "Column: value" on an assumption with no evidence behind it.
   Point it at a folder with a spreadsheet and a deck in it, ask something only
   they can answer, and see what breaks.

2. Use the per-case data. eval/per_case.json now has all 84 outcomes. The
   obvious question it can answer and nothing else can: is the ~15% cross-
   document confusion the SAME fifteen percent from run to run, or does it move?
   If it is stable, those cases are a fixture worth optimising against.

3. Generation, if there is ever credit. src/generate.py is fixed and exercised
   to the network boundary but has still never completed a real call. When there
   is credit: run it once, and consider adding server-side `fallbacks` (see
   HANDOFF §7 item 3 for why it is deliberately not there yet).

Skills: webapp-testing (Playwright is in .venv — screenshot and LOOK at every
change; it has caught ~15 real defects here that reading the diff did not),
dataviz, emil-design-eng. Don't reach for the design skills unless asked.
```

---

## Why the prompt says what it says

**Why "don't start another visual pass" is still first.** The instruction is easy
to drift from — a fresh session sees an unfinished-looking page and starts
redesigning. It is stated before anything else for that reason.

**Why the corpus moved from item 4 to item 1.** In the previous version it sat
last because only the owner could supply documents and an agent could not act on
it. The folder intake removed that blocker: the mechanism now exists, so the
remaining step is genuinely just pointing it somewhere, and it is back where its
value says it belongs.

**Why the mislabel that was item 2 is gone.** evaluate.py measured the
diversity cap on the weighted fusion branch while the app serves RRF, so the row
documented as the default described a configuration this system has never run.
Corrected 2026-08-21: five numbers moved, none by more than 0.004, and no
conclusion changed. Recorded here because the size of the correction is the
point — it changed nothing, which is why it survived, and this repo has been
bitten by exactly that shape of error before.

**Why the design skills are named and then withheld.** They are installed and
they are good, which is exactly why a session will reach for them by default.

**Why Playwright is singled out.** Rendering the page has caught roughly fifteen
real defects in this repo that reading the diff did not — labels drawn at 6px
inside a scaled canvas, a bounding box that measured every block as zero-width,
a `requestAnimationFrame` timestamp arriving before its own start time. The
2026-08-21 session added four more, all invisible in the diff: score bars scaled
to a cross-encoder range collapsing to stubs on the fusion scale, "+0.03" for
every document at two decimal places, a latency row reporting 38ms against 35ms
as though it meant something, and the map captioning "5 kept, 1 cut" beside a
panel reading "nothing displaced". It is the single highest-yield habit here and
the easiest one to skip.
