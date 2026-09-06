"""How many of the reported hits rest on a paper's title block?

**The hole.** The golden set labels pages, not passages. A paper's title block
sits on page 1, and page 1 is a gold page for 43 of the 100 answerable cases in
the two corpora that have title pages. For 17 of those the answer string is
inside the title as well, because a paper is usually titled after the thing it
is about: `bn-covariate` asks what normalizing layer inputs addresses, wants the
string "internal covariate shift", and its paper is titled "... by Reducing
Internal Covariate Shift".

So both halves of the labelling scheme will score a title block as a correct
answer, and neither can see that it answers nothing. That is why dropping title
blocks from retrieval measures *worse*; `src/front_matter.py` carries the
numbers.

**What that costs is a different question from how far it reaches**, and it is
the one that decides whether the published figures mean anything. A label that
*would* credit a title block costs nothing unless the pipeline actually returns
one, and returning one costs nothing if the chunk answers anyway.

**That second clause is the whole difficulty, and the first version of this
script got it wrong.** `front_matter.is_front_matter` decides from the head of
a chunk: does it open with a heading and carry an email or an affiliation. That
is a statement about the first 600 characters. These chunks run to about 1,000,
and a paper's first chunk is the title block *and then the abstract*. An
abstract is among the most answer-dense text in a paper.

So a chunk being flagged says nothing on its own. This asks instead whether the
satisfying chunk contains the answer:

  - **structural**: the case counts toward any-hit@5 and every returned chunk
    satisfying its page label is flagged front matter.
  - **answered anyway**: of those, the ones whose text contains the case's
    answer string. These are correct credits and the flag was misleading.
  - **unanswered**: the residual, which are the only candidates for a false
    credit and are few enough to read.

**What it found, re-run 2026-09-01 against the corrected golden sets.** Two of
114 hits are structural, `bn-covariate` and `qf-whale-attack`, and **both contain the answer
string**, read and confirmed: "We refer to this phenomenon as internal
covariate shift, and address the problem by normalizing layer inputs", and "by
introducing certain detectability threshold, joining the attack can lead to
strictly less reward for whales". **Zero false credits.**

Which settles the title-block filter against itself. Those two hits are the
entire loss it measured, to three decimals on any-hit and MRR, and both are
genuine. Dropping a paper's first chunk drops its abstract. That is not a
scoring artefact, it is information.

    .venv\\Scripts\\python.exe src\\audit_title_credit.py

Read-only. Nothing is rewritten, and no setting is changed.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import corpora  # noqa: E402
import rerank as rr  # noqa: E402
from evaluate import gold_keys, is_relevant, load_cases, normalize  # noqa: E402
from front_matter import is_front_matter  # noqa: E402
from hybrid import build_bm25  # noqa: E402
from retrieve import (DEFAULT_FUSION, EMBEDDING_MODEL, TOP_K,  # noqa: E402
                      load_ensemble, load_index, retrieve)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def title_block(result: dict) -> bool:
    return is_front_matter(result.get("text", ""), result.get("locator"))


def audit(name: str, cfg: dict, model) -> dict:
    answerable, _ = load_cases(Path(cfg["golden"]))
    index, metadata = load_index(cfg["store"])
    bm25 = build_bm25(metadata)
    ensemble = load_ensemble(cfg["store"])

    out = {"cases": len(answerable), "hits": 0, "structural": [],
           "answered_anyway": [], "unanswered": [],
           "recalled": 0, "false_recall": [], "showing": []}

    for case in answerable:
        rr.clear_cache()
        results = retrieve(case["question"], index, metadata, model, k=TOP_K,
                           candidate_k=cfg["candidate_k"], use_reranker=True,
                           fusion=DEFAULT_FUSION, bm25=bm25, max_per_source=2,
                           # Passed per call; set on the rerank module until
                           # 2026-09-07, which left a blend behind for the
                           # next importer.
                           rerank_blend=cfg["rerank_blend"], ensemble=ensemble)
        if results and title_block(results[0]):
            out["showing"].append(case["id"])

        answer = case.get("answer_contains")
        needle = normalize(answer) if answer else ""

        gold = gold_keys(case)
        relevant = [r for r in results if is_relevant(r, gold)]
        if relevant:
            out["hits"] += 1
            if all(title_block(r) for r in relevant):
                out["structural"].append(case["id"])
                # Flagged is not the same as empty. The chunk carries the
                # abstract after the title block, and an abstract answers.
                if needle and any(needle in normalize(r["text"])
                                  for r in relevant):
                    out["answered_anyway"].append(case["id"])
                else:
                    out["unanswered"].append(case["id"])

        if answer:
            carrying = [r for r in results if needle in normalize(r["text"])]
            if carrying:
                out["recalled"] += 1
                # Same correction: the string being inside a flagged chunk is
                # not evidence against it. It is evidence the abstract said it.
                if all(title_block(r) for r in carrying):
                    out["false_recall"].append(case["id"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", help="one corpus by name; default is all of them")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)

    reg = corpora.registry()
    total = {"hits": 0, "structural": 0, "answered_anyway": 0,
             "unanswered": 0, "recalled": 0, "false_recall": 0}
    for name, cfg in reg.items():
        if args.corpus and name != args.corpus:
            continue
        if not cfg["indexed"] or not cfg["golden"]:
            continue
        r = audit(name, cfg, model)
        total["hits"] += r["hits"]
        for key in ("structural", "answered_anyway", "unanswered"):
            total[key] += len(r[key])
        total["recalled"] += r["recalled"]
        total["false_recall"] += len(r["false_recall"])

        def names(ids):
            return "  " + ", ".join(ids) if ids else ""

        print(f"\n{cfg['label']}  ({r['cases']} answerable)")
        print(f"  hits at 5                          {r['hits']}")
        print(f"  ... satisfied only by a flagged chunk  "
              f"{len(r['structural'])}{names(r['structural'])}")
        print(f"      of those, containing the answer    "
              f"{len(r['answered_anyway'])}{names(r['answered_anyway'])}")
        print(f"      of those, not                      "
              f"{len(r['unanswered'])}{names(r['unanswered'])}")
        print(f"  answer string found                {r['recalled']}")
        print(f"  ... only inside a flagged chunk    {len(r['false_recall'])}"
              f"{names(r['false_recall'])}")
        print(f"  showing a flagged chunk at rank 1  {len(r['showing'])}"
              f"{names(r['showing'])}")

    print(f"\nacross the corpora audited: {total['structural']} of "
          f"{total['hits']} hits are satisfied only by a chunk flagged as front "
          f"matter,\nand {total['answered_anyway']} of those contain the answer "
          f"string. False credits: {total['unanswered']}.")
    print("A chunk being flagged is a statement about its first 600 characters. "
          "The chunk\nis about a thousand, and what follows a title block is "
          "the abstract.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
