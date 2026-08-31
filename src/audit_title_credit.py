"""How many of the reported hits rest on a paper's title block?

**The hole.** The golden set labels pages, not passages. A paper's title block
sits on page 1, and page 1 is a gold page for 43 of the 102 answerable cases in
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
one. This runs the shipped configuration over every answerable case and counts
the credits that are actually being collected:

  - a **false hit**: the case counts toward any-hit@5, and every returned chunk
    that satisfies its page label is a title block.
  - a **false context recall**: the answer string was found in the returned
    text, and only inside a title block.

Both are upper bounds on the damage, not estimates of it. A false hit here is
one the golden set awards and a reader would not.

**What it found on 2026-08-31, at the shipped configuration.** Two of 110 hits,
`bn-covariate` and `qf-whale-attack`, and three of 97 answer-string credits.
Those two hits are the entire loss the title-block filter measured, to three
decimals on both any-hit and MRR: 57/67 is 0.851 and 56/67 is 0.836, 31/35 is
0.886 and 30/35 is 0.857. So the labelling hole reaches 43 cases and costs two.

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
    rr.RERANK_BLEND = cfg["rerank_blend"]

    out = {"cases": len(answerable), "hits": 0, "false_hits": [],
           "recalled": 0, "false_recall": [], "showing": []}

    for case in answerable:
        rr.clear_cache()
        results = retrieve(case["question"], index, metadata, model, k=TOP_K,
                           candidate_k=cfg["candidate_k"], use_reranker=True,
                           fusion=DEFAULT_FUSION, bm25=bm25, max_per_source=2,
                           ensemble=ensemble)
        if results and title_block(results[0]):
            out["showing"].append(case["id"])

        gold = gold_keys(case)
        relevant = [r for r in results if is_relevant(r, gold)]
        if relevant:
            out["hits"] += 1
            if all(title_block(r) for r in relevant):
                out["false_hits"].append(case["id"])

        answer = case.get("answer_contains")
        if answer:
            needle = normalize(answer)
            carrying = [r for r in results if needle in normalize(r["text"])]
            if carrying:
                out["recalled"] += 1
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
    total = {"hits": 0, "false_hits": 0, "recalled": 0, "false_recall": 0}
    for name, cfg in reg.items():
        if args.corpus and name != args.corpus:
            continue
        if not cfg["indexed"] or not cfg["golden"]:
            continue
        r = audit(name, cfg, model)
        total["hits"] += r["hits"]
        total["false_hits"] += len(r["false_hits"])
        total["recalled"] += r["recalled"]
        total["false_recall"] += len(r["false_recall"])

        print(f"\n{cfg['label']}  ({r['cases']} answerable)")
        print(f"  hits at 5                        {r['hits']}")
        print(f"  ... resting only on a title block {len(r['false_hits'])}"
              f"{'  ' + ', '.join(r['false_hits']) if r['false_hits'] else ''}")
        print(f"  answer string found              {r['recalled']}")
        print(f"  ... only inside a title block    {len(r['false_recall'])}"
              f"{'  ' + ', '.join(r['false_recall']) if r['false_recall'] else ''}")
        print(f"  showing a title block at rank 1  {len(r['showing'])}"
              f"{'  ' + ', '.join(r['showing']) if r['showing'] else ''}")

    print(f"\nacross the corpora audited: {total['false_hits']} of "
          f"{total['hits']} hits and {total['false_recall']} of "
          f"{total['recalled']} answer-string credits rest on a title block")
    print("Reaching a label is not the same as collecting it. The count above "
          "is what the\nlabelling hole actually costs; front_matter.py carries "
          "how far it could reach.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
