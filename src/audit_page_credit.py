"""Does a hit mean the reader got the answer, or only the right page?

Across the three corpora, 11 questions score as misses and 14 score a hit while
no returned passage carries the answer string. A first reading on 2026-09-03
put that second figure at 33, and the correction is recorded below. The gap is
in what `is_relevant` compares.

Gold is derived by `build_golden_set.derive_gold`: a case declares an answer
string, every chunk containing it is found, and its LOCATOR is recorded, which
is a source, a kind and a page. `is_relevant` then asks whether a result's
locator is in that set. A page is longer than a chunk, so a page carrying the
answer in one of its chunks marks EVERY chunk of that page relevant, and a
result from the right page scores a hit whether or not the answer is anywhere
the reader can see it.

That is not automatically wrong, and this file exists because the obvious
reading of it is the one this project has already got wrong once. On
2026-08-31 an audit called two credits false because their chunks opened on a
title block, and reading them in full showed both continue into an abstract
that answers outright. A passage can answer without carrying one exact
phrasing, so the only way to settle this is to read the passages.

All were read rather than counted, and the population was corrected twice
before the counts below meant anything.

**The first reading was wrong about the cause, and the correction is the more
useful finding.** It was taken off the DISPLAYED EXCERPT, which
`pipeline_trace._brief` capped at 420 characters, so it reported 33 cases and
blamed all of them on the locator. 21 of the 33 had the answer in the
retrieved chunk and lost it to that cap: the system found the answer and the
page cut it off, which is a display defect and not a scoring one. The cap is
720 from 2026-09-03 and answer-shown across the three corpora went 0.656 to
0.808. What remains here is the population no display change can reach, because
the answer is not in the retrieved chunk at all:

     9  the passage does not answer. The reader is shown a page that holds the
        answer somewhere and a paragraph of it that does not.
     2  the passage answers in different words, so the credit is right and the
        string test is what is too strict.
     3  arguable, recorded as arguable rather than pushed to either side.

So 9 of 125 answerable questions are scored as found while the reader gets
nothing, not the 21 first reported. any-hit@5 is not measuring the wrong thing.
It measures whether retrieval reached a location that answers, and it does that
correctly. It is simply not the question a reader asks, and the two were being
reported as though they were the same. `evaluate.py` now reports both.

**Two measurement artefacts, both found by reading and both corrected here.**
The corpus is converted from PDF and keeps markdown emphasis, so the
batch-normalization paper renders "_internal_ _covariate_ _shift,_" and an
exact-string test called it absent when a reader plainly sees it; the display
path already strips these in `answer-mark.js`. En dashes do the same to
"mean-variance". Both normalisations are applied below, and between them they
moved two cases out of the population before anything was read.

    .venv\\Scripts\\python.exe src\\audit_page_credit.py

Exit 0 means every case credited on the locator alone has a verdict above.
Exit 1 means that population moved, so a verdict is missing or stale. Exit 2
means the guard did not run, because the recorded payloads it reads live in
`static-demo/`, which is gitignored, and `src/record_static.py --all` has not
produced them for every corpus on this machine. A recording for some corpora
and not others is also exit 2. Until 2026-09-05 the script audited whichever
corpora were recorded, so with quant absent it reported the 4 quant verdicts
above as stale and exited 1 for a reason that was not drift.

This is the seventh guard listed in README.md. `build_corpus_manifest.py
--check` depends on a gitignored input in the same way, since it reads the
built index and returns 2 when there is none, and `check_golden.py` skips a
corpus whose index is missing. A draft of this paragraph on 2026-09-05 called
this the only guard of the seven that needs a file the repository does not
carry, which was wrong.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluate import gold_keys  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent

GOLDEN = {"birds": "eval/golden-birds.json", "llm": "eval/golden_set.json",
          "quant": "eval/golden-quant.json"}

DASH = re.compile("[‐-―−]")
# Emphasis markers, heading hashes, and the short bracketed markers the PDF
# converter leaves behind. `answer-mark.js` strips the same things before the
# reader sees a passage, so a test that does not is measuring a different text.
MARKUP = re.compile(r"[_*`#]+|\[[^\]]{0,3}\]")


def norm(text) -> str:
    return re.sub(r"\s+", " ",
                  MARKUP.sub("", DASH.sub("-", str(text)))).strip().lower()


# Read individually on 2026-09-03, and re-sorted the same day after the
# population was corrected. The first reading of these was taken off the
# DISPLAYED EXCERPT, which was capped at 420 characters, so it conflated two
# causes and blamed both on the locator. 21 of the 33 had the
# answer in the retrieved chunk and lost it to that cap; raising the cap to 720
# in pipeline_trace._brief fixed those. What is left below is the population
# that survives it: the answer is not in the retrieved chunk at all, so no
# display change can reach them.
#
# The reason matters more than the verdict. A later reader who disagrees with
# one of these should be able to see what was in front of the passage.
DOES_NOT_ANSWER = {
    "bird-flyway": "names migration routes, never the word flyway",
    "bird-imprinting": "mallard clutch timing, unrelated to imprinting",
    "bird-fledging": "chick development by week, never names the stage",
    "cot-scale": "asks why scale helps and gives no scale",
    "faiss-ivf": "describes searching partitions without naming the index",
    "lora-frozen": "describes full fine-tuning, the opposite arrangement",
    "qf-deep-learning": "probabilistic estimation, names no method family",
    "qf-market-making": "defines lots and ticks, not the activity",
    "pos-enc-fn": "explains why the function was chosen, never gives it",
}
ANSWERS_IN_OTHER_WORDS = {
    "w2v-cbow": "'the CBOW architecture predicts the current word'",
    "qf-order-flow": "names the weighted volume imbalance signal",
}
ARGUABLE = {
    "cot-prompt": "says equation-only prompting helps, not the full effect",
    "resnet-degradation": "discusses deep plain nets without saying saturated",
    "qf-degeneracy": "the right paper, and the name is in its title only",
}
READ = {**DOES_NOT_ANSWER, **ANSWERS_IN_OTHER_WORDS, **ARGUABLE}


def locator_triple(item: dict) -> tuple:
    """The recorded payload flattens the locator to "page 4"."""
    kind, _, value = str(item["locator"]).partition(" ")
    return (item["source"], kind, value)


RECORD_CMD = ".venv\\Scripts\\python.exe src\\record_static.py --all"


def not_run(reason: str) -> int:
    """Exit 2, distinct from 0 and 1, so a caller cannot read it as a pass.

    The verdicts above are only checked against a recording, and a missing
    recording says nothing about whether they still describe their cases.
    """
    print(f"  NOT RUN  {reason}\n"
          f"  The credits were not audited. Produce the recording first with\n"
          f"      {RECORD_CMD}")
    return 2


def main() -> int:
    rows = []
    missing = [c for c in GOLDEN if not (ROOT / "static-demo" / c).is_dir()]
    if missing:
        # A partial recording is treated the same as none. Until 2026-09-05
        # this loop audited whichever corpora were recorded and reported the
        # absent corpus's verdicts as stale with exit 1, which is the wrong
        # answer to a different question.
        return not_run(f"static-demo/ holds no recording for "
                       f"{', '.join(missing)}.")
    for corpus, gpath in GOLDEN.items():
        cases = {c["question"]: c for c in json.loads(
            (ROOT / gpath).read_text(encoding="utf-8"))["cases"]}
        recorded = ROOT / "static-demo" / corpus
        for f in sorted(recorded.glob("*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            if not isinstance(d, dict) or "query" not in d:
                continue
            case = cases.get(d["query"])
            if not case or case.get("unanswerable"):
                continue
            want = case.get("answer_contains")
            if not want:
                continue
            items = d["stages"][-1]["items"]
            gold = gold_keys(case)
            rows.append({
                "id": case["id"], "corpus": corpus,
                "scored": any(locator_triple(i) in gold for i in items),
                "visible": any(norm(want) in norm(i.get("text", ""))
                               for i in items),
            })

    if not rows:
        return not_run("static-demo/ holds no payload that matches an "
                       "answerable golden case.")
    both = [r for r in rows if r["scored"] and r["visible"]]
    page_only = [r for r in rows if r["scored"] and not r["visible"]]
    missed = [r for r in rows if not r["scored"]]

    print(f"\n  {len(rows)} answerable questions carrying an answer string\n")
    print(f"    {len(both):>4}  scored a hit, and a returned passage holds the "
          f"answer")
    print(f"    {len(page_only):>4}  scored a hit on the LOCATOR alone")
    print(f"    {len(missed):>4}  scored a miss")

    ids = {r["id"] for r in page_only}
    unread = sorted(ids - READ.keys())
    stale = sorted(READ.keys() - ids)

    print(f"\n  of the {len(ids)} credited on the locator alone, as read on "
          f"2026-09-03:")
    for label, group in (("do not answer", DOES_NOT_ANSWER),
                         ("answer in other words", ANSWERS_IN_OTHER_WORDS),
                         ("arguable", ARGUABLE)):
        live = sorted(set(group) & ids)
        print(f"\n    {len(live)} {label}")
        for cid in live:
            print(f"      {cid:<24}{group[cid]}")

    problems = []
    if unread:
        problems.append(f"credited on the locator alone and never read: "
                        f"{', '.join(unread)}")
    if stale:
        problems.append(f"read, but no longer credited on the locator alone, "
                        f"so the verdict is stale: {', '.join(stale)}")
    print()
    for p in problems:
        print(f"  DRIFTED  {p}")
    if problems:
        print("\n  The population moved. Read the new cases before quoting any "
              "count\n  from this file, because the counts above describe the "
              "cases that were read.")
        return 1
    print(f"  Every case credited on the locator alone has been read. "
          f"{len(DOES_NOT_ANSWER)} of\n  {len(rows)} answerable questions are "
          f"scored as found while the reader gets nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
