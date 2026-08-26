# Next session — start here

Rewritten 2026-08-26. It is a pointer, not a substitute: `HANDOFF.md` is the
cold-start document and carries the reasoning behind every decision named here.

---

```
Project: C:\Users\owner\Desktop\rag-project

Read HANDOFF.md FIRST. §2 is the measured state, §5 is the UI, §7 opens with
what is unfinished and why.

STATE: three corpora, all measured, all green.
  ML papers    36 docs  5,459 passages   any-hit 0.851
  Ornithology  45 docs    864 passages   any-hit 0.846
  Quant        35 docs  6,184 passages   any-hit 0.886
  138 checks passing: 22 metrics, 28 loaders, 80 trace, 8 OCR.
  Working tree clean, everything pushed.

RUN IT:  python src/serve.py      then http://127.0.0.1:8000/
TEST IT: python src/test_metrics.py test_loaders.py test_trace.py test_ocr.py
```

## The three things worth doing next

**1. A freshness check for the eval chain.** The highest-value small job. A
generated file has gone stale silently three times now: `results.json` shared
between corpora so the last run owned it, `RESULTS.md` claiming to be copied
from it while being typed by hand, and `per_case.json` four days out of date —
which meant the front page was offering questions I had just deleted as
unanswerable. Only `RESULTS.md` has `--check`. The chain
`per_case → analytics → the interface` has none.

**2. A better cross-encoder.** The bird candidate pool contains the answer 96.2%
of the time; the pipeline returns it 84.6%. Blending 20% of the first stage back
in recovered part of that, but the rest is `ms-marco-MiniLM-L-6-v2` being weak
on questions that *describe* a term rather than naming it.
`src/compare_rerankers.py` exists for this. **Run `src/sweep_blend.py` for
anything touching the blend — every corpus, every time.**

**3. The hero diagram.** Discussed, not built. All seven stages are drawn as the
same sheet-of-cells, which is wrong in one specific place: dense retrieval and
BM25 are identical in the spec (`layers: 4, cells: [5,4], w: 6.5`) and are
opposites in nature — continuous similarity against sparse term hits. Drawing
them the same hides the reason for running both. The owner also asked for a
*gate* at the diversity cap showing particles blocked, and more animation inside
the arrays. The index-as-a-field is right and should stay.

## Things that will bite you

- **A corpus setting does not transfer.** Threshold, rerank blend, and probably
  fusion are all per-corpus in `corpora.json`. The same 0.0 threshold that costs
  the ML papers one question costs the bird corpus ten.
- **When a corpus scores badly, read the failing questions first.** Quant looked
  like a retrieval weakness at 0.543 and was entirely a golden-set problem —
  textbook definitions asked of research papers. 0.886 after rewriting the
  questions, retrieval untouched.
- **A measurement that returns zero deserves as much suspicion as a surprise.**
  Two of mine returned zero because I compared a dict locator against its
  serialised string.
- **Screenshot the UI.** Roughly twenty real defects came from rendering the
  page, not from reading the diff. Check a laptop height (~660px) and a phone.
- **Escaping in heredocs.** `\n` inside a bash heredoc becomes a real newline and
  breaks Python and JS string literals. Use the Write tool for patch scripts.

## Blocked

The API key has no credit, so generated prose and query decomposition have never
run. Everything the interface shows is the retrieved passage verbatim. The
`generate` path is written and wired; it has never executed successfully.
