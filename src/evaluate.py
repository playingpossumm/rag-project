"""Retrieval evaluation harness.

Scores a retrieval configuration against a labelled golden set, so changes can
be shown to help rather than argued to help.

Relevance is judged at PAGE level, not chunk level: a chunk counts as relevant
if it came from a page the golden set lists for that question. Page-level
labelling is coarse -- it cannot tell a precise passage from a vague one on the
same page -- but it is the granularity a human can actually label reliably, and
it stays stable when chunking parameters change. Chunk-level labels would have
to be redone after every re-index, which is exactly when you most want a
comparable metric.
"""
import argparse
import json
import math
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

from retrieve import CANDIDATE_K, EMBEDDING_MODEL, TOP_K, load_index, retrieve

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GOLDEN_SET = Path(__file__).parent.parent / "eval" / "golden_set.json"


def load_cases(path: Path = GOLDEN_SET) -> tuple[list[dict], list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data["cases"]
    answerable = [c for c in cases if not c.get("unanswerable")]
    adversarial = [c for c in cases if c.get("unanswerable")]
    return answerable, adversarial


def hit_rate(results: list[dict], gold: set[int]) -> float:
    """1.0 if any relevant page appears anywhere in the returned set."""
    return 1.0 if any(r["page"] in gold for r in results) else 0.0


def reciprocal_rank(results: list[dict], gold: set[int]) -> float:
    """1/rank of the first relevant result; 0 if none. Rewards ranking it first."""
    for i, r in enumerate(results, start=1):
        if r["page"] in gold:
            return 1.0 / i
    return 0.0


def ndcg(results: list[dict], gold: set[int]) -> float:
    """Normalized discounted cumulative gain.

    Unlike MRR this credits every relevant result, not just the first, while
    discounting each by log2 of its rank so later hits count for less.

    The ideal ranking is defined as the hits we actually found, moved to the top
    -- not one slot per gold page. Because relevance is judged per page, several
    returned chunks can share a single gold page, so normalizing by len(gold)
    lets DCG exceed IDCG and produces scores above 1.0.
    """
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, r in enumerate(results, start=1)
        if r["page"] in gold
    )
    n_relevant = sum(1 for r in results if r["page"] in gold)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_relevant + 1))
    return dcg / idcg if idcg else 0.0


def evaluate(cases, adversarial, index, metadata, model, k, use_reranker):
    per_case, scores = [], {"hit_rate": 0.0, "mrr": 0.0, "ndcg": 0.0}

    for case in cases:
        results = retrieve(case["question"], index, metadata, model,
                           k=k, use_reranker=use_reranker)
        gold = set(case["pages"])
        m = {
            "hit_rate": hit_rate(results, gold),
            "mrr": reciprocal_rank(results, gold),
            "ndcg": ndcg(results, gold),
        }
        for key in scores:
            scores[key] += m[key]
        per_case.append({
            "id": case["id"],
            "difficulty": case.get("difficulty", "-"),
            "gold": sorted(gold),
            "got": [r["page"] for r in results],
            **m,
        })

    n = len(cases) or 1
    summary = {key: value / n for key, value in scores.items()}

    # Adversarial cases have no relevant page, so recall metrics are undefined.
    # What we can observe is the top score: a system that separates answerable
    # from unanswerable questions should be measurably less confident here.
    score_key = "rerank_score" if use_reranker else "score"
    adv_top = []
    for case in adversarial:
        results = retrieve(case["question"], index, metadata, model,
                           k=k, use_reranker=use_reranker)
        adv_top.append(results[0][score_key] if results else float("nan"))

    return summary, per_case, adv_top


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=TOP_K)
    parser.add_argument("--per-case", action="store_true", help="show every case")
    args = parser.parse_args()

    answerable, adversarial = load_cases()
    index, metadata = load_index()
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"corpus: {index.ntotal} chunks | cases: {len(answerable)} answerable, "
          f"{len(adversarial)} adversarial | k={args.k}\n")

    runs = {}
    for label, use_rr in (("baseline", False), ("reranked", True)):
        summary, per_case, adv = evaluate(
            answerable, adversarial, index, metadata, model, args.k, use_rr
        )
        runs[label] = (summary, per_case, adv)

    print(f"{'config':<12}{'hit@k':>9}{'MRR':>9}{'NDCG':>9}")
    for label, (summary, _, _) in runs.items():
        print(f"{label:<12}{summary['hit_rate']:>9.3f}{summary['mrr']:>9.3f}{summary['ndcg']:>9.3f}")

    base, rr = runs["baseline"][0], runs["reranked"][0]
    print(f"{'delta':<12}"
          f"{rr['hit_rate'] - base['hit_rate']:>+9.3f}"
          f"{rr['mrr'] - base['mrr']:>+9.3f}"
          f"{rr['ndcg'] - base['ndcg']:>+9.3f}")

    if args.per_case:
        for label, (_, per_case, _) in runs.items():
            print(f"\n--- {label} ---")
            print(f"{'case':<14}{'diff':<8}{'MRR':>6}  gold -> got")
            for c in sorted(per_case, key=lambda x: x["mrr"]):
                flag = "  " if c["mrr"] else " x"
                print(f"{c['id']:<14}{c['difficulty']:<8}{c['mrr']:>6.2f}{flag} "
                      f"{c['gold']} -> {c['got']}")

    print("\nadversarial top-1 score (lower is better -- these have no answer):")
    for label, (_, _, adv) in runs.items():
        joined = ", ".join(f"{s:+.2f}" for s in adv)
        print(f"  {label:<10} {joined}")


if __name__ == "__main__":
    main()
