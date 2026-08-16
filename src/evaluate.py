"""Retrieval evaluation harness.

Scores retrieval configurations against a labelled golden set, so changes can be
shown to help rather than argued to help.

Relevance is judged at PAGE level, not chunk level: a chunk counts as relevant
if it came from a page the golden set lists for that question. Page-level
labelling is coarse -- it cannot tell a precise passage from a vague one on the
same page -- but it is the granularity a human can label reliably, and it stays
stable when chunking parameters change. Chunk-level labels would have to be
redone after every re-index, which is exactly when a comparable metric matters
most.

The harness reports two layers, because the two pipeline stages do different
jobs and improving one does not show up in the other's numbers:

  candidate recall@N  -- did the first stage put a relevant chunk in the pool
                         at all? Fusion is judged here; the reranker cannot
                         recover what was never retrieved.
  final hit/MRR/NDCG  -- did the finished pipeline rank it near the top?
                         Reranking is judged here.
"""
import argparse
import json
import math
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

from hybrid import build_bm25
from retrieve import CANDIDATE_K, EMBEDDING_MODEL, TOP_K, load_index, retrieve, shortlist

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GOLDEN_SET = Path(__file__).parent.parent / "eval" / "golden_set.json"


def load_cases(path: Path = GOLDEN_SET) -> tuple[list[dict], list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data["cases"]
    return (
        [c for c in cases if not c.get("unanswerable")],
        [c for c in cases if c.get("unanswerable")],
    )


def hit_rate(results, gold) -> float:
    """1.0 if any relevant page appears anywhere in the returned set."""
    return 1.0 if any(r["page"] in gold for r in results) else 0.0


def reciprocal_rank(results, gold) -> float:
    """1/rank of the first relevant result; 0 if none. Rewards ranking it first."""
    for i, r in enumerate(results, start=1):
        if r["page"] in gold:
            return 1.0 / i
    return 0.0


def ndcg(results, gold) -> float:
    """Normalized discounted cumulative gain.

    The ideal ranking is the hits actually found, moved to the top -- not one
    slot per gold page. Because relevance is judged per page, several returned
    chunks can share a gold page, so normalizing by len(gold) would let DCG
    exceed IDCG and produce scores above 1.0.
    """
    dcg = sum(1.0 / math.log2(i + 1)
              for i, r in enumerate(results, start=1) if r["page"] in gold)
    n_relevant = sum(1 for r in results if r["page"] in gold)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_relevant + 1))
    return dcg / idcg if idcg else 0.0


def score_run(cases, retrieve_fn) -> tuple[dict, list[dict]]:
    totals = {"hit_rate": 0.0, "mrr": 0.0, "ndcg": 0.0}
    per_case = []
    for case in cases:
        results = retrieve_fn(case["question"])
        gold = set(case["pages"])
        m = {
            "hit_rate": hit_rate(results, gold),
            "mrr": reciprocal_rank(results, gold),
            "ndcg": ndcg(results, gold),
        }
        for key in totals:
            totals[key] += m[key]
        per_case.append({"id": case["id"], "difficulty": case.get("difficulty", "-"),
                         "gold": sorted(gold), "got": [r["page"] for r in results], **m})
    n = len(cases) or 1
    return {k: v / n for k, v in totals.items()}, per_case


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--candidate-k", type=int, default=CANDIDATE_K)
    ap.add_argument("--per-case", action="store_true")
    args = ap.parse_args()

    answerable, adversarial = load_cases()
    index, metadata = load_index()
    model = SentenceTransformer(EMBEDDING_MODEL)
    bm25 = build_bm25(metadata)

    print(f"corpus: {index.ntotal} chunks | cases: {len(answerable)} answerable, "
          f"{len(adversarial)} adversarial | k={args.k}, candidates={args.candidate_k}\n")

    # ---- Layer 1: candidate pool quality (no reranking) -------------------
    # This is where fusion is judged. Sparse-only is included as a reference
    # point, not a candidate configuration.
    pools = [
        ("dense",           dict(fusion="none")),
        ("rrf",             dict(fusion="rrf")),
        ("weighted a=0.3",  dict(fusion="weighted", alpha=0.3)),
        ("weighted a=0.5",  dict(fusion="weighted", alpha=0.5)),
        ("weighted a=0.7",  dict(fusion="weighted", alpha=0.7)),
    ]

    print(f"CANDIDATE POOL @ {args.candidate_k}   (can the reranker even see the answer?)")
    print(f"{'first stage':<18}{'recall':>9}{'MRR':>9}")
    pool_rows = {}
    for label, cfg in pools:
        summary, _ = score_run(answerable, lambda q, c=cfg: shortlist(
            q, index, metadata, model, k=args.candidate_k, bm25=bm25, **c))
        pool_rows[label] = summary
        print(f"{label:<18}{summary['hit_rate']:>9.3f}{summary['mrr']:>9.3f}")

    # ---- Layer 2: end-to-end, with reranking ------------------------------
    finals = [
        ("dense, no rerank", dict(fusion="none", use_reranker=False)),
        ("dense + rerank",   dict(fusion="none", use_reranker=True)),
        ("rrf + rerank",     dict(fusion="rrf", use_reranker=True)),
        ("weighted + rerank", dict(fusion="weighted", alpha=0.5, use_reranker=True)),
    ]

    print(f"\nEND TO END @ {args.k}")
    print(f"{'pipeline':<20}{'hit@k':>9}{'MRR':>9}{'NDCG':>9}")
    runs = {}
    for label, cfg in finals:
        summary, per_case = score_run(answerable, lambda q, c=cfg: retrieve(
            q, index, metadata, model, k=args.k,
            candidate_k=args.candidate_k, bm25=bm25, **c))
        runs[label] = (summary, per_case)
        print(f"{label:<20}{summary['hit_rate']:>9.3f}{summary['mrr']:>9.3f}{summary['ndcg']:>9.3f}")

    if args.per_case:
        for label, (_, per_case) in runs.items():
            print(f"\n--- {label} ---")
            for c in sorted(per_case, key=lambda x: x["mrr"]):
                if c["mrr"] < 1.0:
                    print(f"  {c['id']:<14}{c['difficulty']:<8}{c['mrr']:>5.2f}  "
                          f"{c['gold']} -> {c['got']}")

    # ---- Abstention signal ------------------------------------------------
    print("\nADVERSARIAL top-1 score (no correct answer exists; lower is better)")
    for label, cfg in (("dense + rerank", dict(fusion="none")),
                       ("rrf + rerank", dict(fusion="rrf"))):
        tops = []
        for case in adversarial:
            res = retrieve(case["question"], index, metadata, model, k=args.k,
                           candidate_k=args.candidate_k, bm25=bm25,
                           use_reranker=True, **cfg)
            tops.append(res[0]["rerank_score"] if res else float("nan"))
        print(f"  {label:<18}" + ", ".join(f"{s:+.2f}" for s in tops))


if __name__ == "__main__":
    main()
