# Next session — start here

Written 2026-08-20. Paste the block below into a new session, or just read it.
It is a pointer, not a substitute: `HANDOFF.md` is the cold-start document and
carries the reasoning behind every decision named here.

---

```
Project: C:\Users\owner\Desktop\rag-project

Read HANDOFF.md FIRST — it's the cold-start doc, current as of 2026-08-20.
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
- Don't touch src/pipeline_trace.py or the retrieval code unless the task is
  explicitly about retrieval.
- Chart palette is validated all-pairs for colourblind separation — no fourth
  hue without re-running the dataviz validator (HANDOFF §5).
- No CDN. Fonts bundled in ui/fonts/ deliberately: the corpus may be private and
  the page must make no external request.
- Light theme only.

TECHNICAL WORK, in priority order:

1. Live settings comparison. The settings panel shows real measured numbers from
   eval/results.json, but toggling re-runs nothing. Needs an endpoint taking
   pipeline options (rerank on/off, fusion, cap, expansion) and returning a fresh
   trace. This is the strongest differentiator the project has — a closed product
   structurally cannot offer it — and my own brief called it out as such.

2. Fix src/api.py:152 — `from generate_answer import synthesize`. No such module;
   it's generate.py and has no `synthesize`. Lazy import inside the optional
   generation branch, so it has never raised (no API credit). Decide what that
   function should be, or delete the branch.

3. Per-case eval data. evaluate.py already computes per-case outcomes behind
   --per-case but only prints them. A small script writing them to JSON unlocks
   real per-case analysis (and the "swarm" idea, if it ever comes back).

4. Folder intake. The app shows local-folder and Drive tabs badged "not wired".
   A pasted path is the cheap honest version — RAG_DATA_DIR already works
   server-side. This is what makes the tool usable on real documents, and
   HANDOFF §7 item 1 says the corpus being 20 PDFs and nothing else is worth
   more than everything else combined.

UI HOUSEKEEPING (small, do while you're in there):
- ui/ambient.html + ui/ambient-fields.js are orphaned — index.html no longer
  references them. Delete both, or wire the lattice back into the folio bar.
  Serving them unused is the worst option.
- ui/pipeline.html is a superseded prototype sharing the app's renderer. Delete
  it or keep it as a bare harness, but don't evolve both.

Skills: webapp-testing (Playwright is in .venv — screenshot and LOOK at every
change; it has caught ~15 real defects here that reading the diff did not),
dataviz, emil-design-eng. Don't reach for the design skills unless asked.

State: everything committed, working tree clean, architecture drift zero.
```

---

## Why the prompt says what it says

**Why "don't start another visual pass" is first.** The instruction is easy to
drift from — a fresh session sees an unfinished-looking page and starts
redesigning. It is stated before anything else for that reason.

**Why the corpus is under item 4 rather than item 1.** Pointing this at real
documents is worth more than every other item combined (`HANDOFF.md` §7), but
only the owner can supply the documents. Folder intake is the part an agent can
actually build, so that is what the list asks for.

**Why the design skills are named and then withheld.** They are installed and
they are good, which is exactly why a session will reach for them by default.

**Why Playwright is singled out.** Rendering the page has caught roughly fifteen
real defects in this repo that reading the diff did not — labels drawn at 6px
inside a scaled canvas, a bounding box that measured every block as zero-width,
a `requestAnimationFrame` timestamp arriving before its own start time. It is the
single highest-yield habit here and the easiest one to skip.
