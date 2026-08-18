"""Measure cross-document confusion instead of arguing from one example.

The known failure is anecdotal: "what optimizer was used to train the
Transformer" returns Vision Transformer. One case is not a problem statement --
it could be a quirk of that phrasing rather than a systematic weakness, and a
fix designed against a single anecdote usually solves the anecdote.

This classifies every answerable case by WHERE the failure happens, because the
three modes need completely different fixes:

  wrong-doc-first   the top result is from a document that does not answer at
                    all. The retriever picked the wrong source.
  right-doc-wrong-loc  the top result is from a correct document but the wrong
                    place in it. That is a within-document ranking problem, not
                    cross-document confusion.
  correct           top result is a gold location.

Only the first is the problem being investigated. Reporting them together would
overstate it.
"""
import sys
from collections import Counter

from sentence_transformers import SentenceTransformer

from evaluate import gold_keys, gold_sources, load_cases
from hybrid import build_bm25
from retrieve import EMBEDDING_MODEL, load_index, retrieve

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    answerable, _ = load_cases()
    index, metadata = load_index()
    model = SentenceTransformer(EMBEDDING_MODEL)
    bm25 = build_bm25(metadata)

    tally = Counter()
    offenders = Counter()   # documents that wrongly win rank 1
    failures = []

    for case in answerable:
        results = retrieve(case["question"], index, metadata, model,
                           k=5, bm25=bm25, use_reranker=True, expansion="none")
        if not results:
            tally["empty"] += 1
            continue

        gold = gold_keys(case)
        sources = gold_sources(case)
        top = results[0]
        top_key = (top["source"], top["locator"]["kind"], str(top["locator"]["value"]))

        if top_key in gold:
            verdict = "correct"
        elif top["source"] in sources:
            verdict = "right-doc-wrong-loc"
        else:
            verdict = "wrong-doc-first"
            offenders[top["source"]] += 1
            # Where did the correct document end up, if anywhere?
            rank_of_gold = next(
                (i for i, r in enumerate(results, 1) if r["source"] in sources), None)
            failures.append((case, top["source"], rank_of_gold))

        tally[verdict] += 1
        tally[f"{verdict} [{case.get('kind', '-')}]"] += 1

    n = len(answerable)
    print(f"{n} answerable cases\n")
    print("WHERE THE TOP RESULT COMES FROM")
    for verdict in ("correct", "right-doc-wrong-loc", "wrong-doc-first"):
        c = tally[verdict]
        print(f"  {verdict:<22} {c:>3}  ({c / n:.1%})")
        for kind in ("fact", "multi", "cross-doc"):
            k = tally[f"{verdict} [{kind}]"]
            if k:
                print(f"      {kind:<12} {k}")

    print(f"\nDOCUMENTS THAT WRONGLY WIN RANK 1 ({sum(offenders.values())} total)")
    for src, count in offenders.most_common(8):
        print(f"  {count:>2}x  {src}")

    print("\nWRONG-DOC CASES (where did a correct document land?)")
    for case, wrong, rank in failures:
        where = f"rank {rank}" if rank else "NOT IN TOP 5"
        print(f"  {case['id']:<18} {case.get('kind','-'):<10} got {wrong[:24]:<24} correct doc: {where}")
        print(f"      q: {case['question'][:88]}")

    recoverable = sum(1 for _c, _w, r in failures if r)
    if failures:
        print(f"\n{recoverable}/{len(failures)} wrong-doc cases still have a correct "
              f"document somewhere in the top 5 -- those are RANKING failures, "
              f"recoverable by reordering. The rest are RETRIEVAL failures.")


if __name__ == "__main__":
    main()
