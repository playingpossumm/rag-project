"""When the answer is not on the page, where is it? Measured, not guessed.

Three problems that were tracked separately and are one problem. The eleven
scored misses, the five permanent failures, and the nine questions where gold
credits the page while the passage returned does not answer all have the same
shape: the chunk holding the answer exists in the corpus and was not returned.
Twenty-three questions across the three corpora, once the display defects fixed
on 2026-09-03 are out of the way.

The useful question is not which ranking lever to try next -- every lever in
this repository has been tried against these and none ships -- but how FAR the
answering chunk was from one that came back.

    10  one chunk away in a document that was retrieved
     3  two chunks away
     8  further away in a retrieved document, 3 to 37 chunks
     2  in another document entirely

**So 13 of 23 sit next door.** That is a statement about chunk geometry rather
than about ranking, and it is why the levers keep failing: the retriever is not
choosing the wrong document or the wrong page, it is choosing a neighbouring
piece of the right page.

**Display cannot fix it, which was measured before concluding it.** The
pipeline can attach neighbouring chunks to each result and the demo was not
using it -- `api.ask` defaults expansion to "window" and
`pipeline_trace.trace_pipeline` defaults it to "none". Turning it on and
showing the raw expanded window puts the answer on the page for 9 more
questions and multiplies the text shown by 3.2, which is not the trade this
interface makes. Showing what `completed()` in ui/index.html would actually
render is far less: it locates the excerpt inside the window and returns a
slice starting up to 300 characters earlier, so it adds a lead-in and never
text after the chunk.

    shipped, the chunk alone                 0.808   591 characters a passage
    expansion on, as completed() renders     0.832   836
    the same with a symmetric lead-out       0.840  1073
    a 600-character lead-out                 0.840  1275

Four questions for nearly twice the text on every answer, and the last row buys
nothing over the one above it. Not shipped. A third arrangement, choosing the
excerpt over the expanded window rather than over the chunk, is *worse* than
what ships, 0.792 against 0.808: `_brief` places its window by where the
question's words fall, and given more room to place it, it moves away from the
chunk that was actually ranked. The same root cause as the marking limitation
recorded on 2026-09-03 -- a question asking what something is called does not
contain the word that finds it.

**The display is already at its ceiling.** 0.808 of answers are on the page and
0.816 are in the retrieved chunks at all, so the excerpt is showing 99% of what
retrieval hands it. Everything left needs retrieval to return a different
chunk.

**Where a shorter stride would and would not reach**, in characters from the
retrieved chunk's nearer edge to where the answer begins:

     3 of 13 within 160 characters, which is today's 40-token overlap
     4 of 13 within 320
     7 of 13 within 480
     8 of 13 within 640

Five sit further than 640 characters out, up to 1,979, and no plausible chunk
geometry reaches those. So the indicated experiment is chunk size rather than
overlap, and it is worth stating the counter-argument with it: this pipeline's
precision rests on ranking small chunks, `retrieve()` says so where expansion
is applied last and deliberately, and a larger chunk trades that away. It also
costs a re-ingest of three corpora, a re-derivation of three golden sets, and
every published number. Not attempted here.

    .venv\\Scripts\\python.exe src\\answer_distance.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluate import answer_normalize  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
STORE = {"birds": "store-birds", "llm": "vector_store", "quant": "store-quant"}
GOLDEN = {"birds": "eval/golden-birds.json", "llm": "eval/golden_set.json",
          "quant": "eval/golden-quant.json"}
NEAR = 2


def unanswered(corpus: str):
    """Recorded questions whose answer is in no retrieved chunk, with the
    corpus metadata so distance can be measured."""
    meta = json.loads((ROOT / STORE[corpus] / "metadata.json")
                      .read_text(encoding="utf-8"))
    cases = {c["question"]: c for c in json.loads(
        (ROOT / GOLDEN[corpus]).read_text(encoding="utf-8"))["cases"]}
    out = []
    recorded = ROOT / "static-demo" / corpus
    if not recorded.is_dir():
        return meta, out
    for f in sorted(recorded.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(d, dict) or "query" not in d:
            continue
        case = cases.get(d["query"])
        if not case or case.get("unanswerable") or not case.get("answer_contains"):
            continue
        items = d["stages"][-1]["items"]
        if any("chunk_id" not in i for i in items):
            continue
        needle = answer_normalize(case["answer_contains"])
        if any(needle in answer_normalize(meta[i["chunk_id"]]["text"])
               for i in items):
            continue
        out.append((case["id"], needle, items))
    return meta, out


def main() -> int:
    distances, reaches, total = {}, [], 0
    for corpus in GOLDEN:
        meta, cases = unanswered(corpus)
        total += len(cases)
        for cid, needle, items in cases:
            best_gap, best_reach = None, None
            for j, chunk in enumerate(meta):
                text = answer_normalize(chunk["text"])
                at = text.find(needle)
                if at < 0:
                    continue
                for it in items:
                    if chunk["source"] != it["source"]:
                        continue
                    gap = j - it["chunk_id"]
                    if best_gap is None or abs(gap) < abs(best_gap):
                        best_gap = gap
                    if abs(gap) > NEAR:
                        continue
                    reach = at if gap > 0 else len(text) - (at + len(needle))
                    reach += max(0, abs(gap) - 1) * len(text)
                    if best_reach is None or reach < best_reach:
                        best_reach = reach
            key = ("in another document entirely" if best_gap is None
                   else f"{abs(best_gap)} chunk(s) away")
            distances.setdefault(key, []).append(cid)
            if best_reach is not None:
                reaches.append((cid, best_reach))

    print(f"\n  {total} questions whose answer is in no retrieved chunk\n")
    for key in sorted(distances, key=lambda k: (not k[0].isdigit(), k)):
        ids = distances[key]
        print(f"    {len(ids):>3}  {key}")
        print(f"         {', '.join(sorted(ids))}")

    near = [r for r in reaches]
    print(f"\n  of those, {len(near)} sit within {NEAR} chunks. Characters from "
          f"the retrieved\n  chunk's nearer edge to where the answer begins:\n")
    for cid, reach in sorted(near, key=lambda r: r[1]):
        print(f"    {cid:<24}{reach:>7}")
    for cut, label in ((160, "40 tokens, today's overlap"), (320, "80 tokens"),
                       (480, "120 tokens"), (640, "160 tokens")):
        n = sum(1 for _c, r in near if r <= cut)
        print(f"\n  within {cut:>4} characters ({label}): {n} of {len(near)}")
    print("\n  A shorter stride reaches the top of that list and not the "
          "bottom. See the\n  module docstring for what was measured and "
          "refuted on the display side.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
