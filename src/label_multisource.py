"""Propose candidate answering documents for each golden-set question.

The corpus holds many similar documents, and the target behaviour is to compile
every relevant source rather than pick one. That makes single-source gold labels
wrong: a question answered by three papers should count all three, and returning
all three is success, not ambiguity.

Candidates are proposed with BM25 only -- deliberately not the dense retriever or
the reranker, which are the components under evaluation. Using them to generate
their own answer key would bake their blind spots into the labels: any document
they systematically fail to retrieve would never enter the gold set, and the
metric would then certify that failure as correct.

Output is a review sheet, not a label file. Every candidate still needs a human
judgement about whether the text genuinely answers the question.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from hybrid import bm25_search, build_bm25

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
STORE = ROOT / "vector_store" / "metadata.json"
GOLDEN = ROOT / "eval" / "golden_set.json"


def main():
    metadata = json.loads(STORE.read_text(encoding="utf-8"))
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    bm25 = build_bm25(metadata)

    cases = [c for c in golden["cases"] if not c.get("unanswerable")]
    print(f"{len(cases)} answerable cases against {len(metadata)} chunks\n")

    for case in cases:
        hits = bm25_search(case["question"], bm25, metadata, k=40)

        # Best-scoring chunk per source, so each document is judged once.
        best = {}
        for h in hits:
            cur = best.get(h["source"])
            if cur is None or h["score"] > cur["score"]:
                best[h["source"]] = h
        ranked = sorted(best.values(), key=lambda h: h["score"], reverse=True)[:6]

        current = case.get("source", "?")
        print(f"### {case['id']}  --  {case['question']}")
        print(f"    currently labelled: {current} p{case['pages']}")
        for h in ranked:
            mark = "*" if h["source"] == current else " "
            snippet = " ".join(h["text"].split())[:96]
            print(f"  {mark} {h['score']:6.2f}  {h['source'][:26]:<26} p{h['page']:<3} {snippet}")
        print()


if __name__ == "__main__":
    main()
