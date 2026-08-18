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
import re
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


def gold_keys(case: dict) -> set:
    """Relevant locations as (source, kind, value) triples.

    Keyed on source as well as locator because page number alone stopped
    identifying a location once the corpus held more than one document -- every
    paper has a page 5. Matching on page alone would score a hit on the wrong
    document as correct, which is the most common real-world RAG failure.
    """
    return {
        (entry["source"], "page", str(p))
        for entry in case.get("gold", [])
        for p in entry["pages"]
    }


def gold_sources(case: dict) -> set:
    return {entry["source"] for entry in case.get("gold", [])}


def is_relevant(result: dict, gold: set) -> bool:
    loc = result["locator"]
    return (result["source"], loc["kind"], str(loc["value"])) in gold


def source_recall(results, sources: set, k: int) -> float:
    """Fraction of the documents that answer the question which were returned.

    This is the metric that matches "compile every relevant source" rather than
    "find one". hit@k cannot express it: a run that returns one of four
    answering papers scores a perfect 1.000 on hit@k while missing three
    quarters of the answer.

    Normalised by min(|gold|, k) because k results cannot represent more than k
    documents -- scoring against the raw count would penalise a run for a
    ceiling imposed by the caller's own k rather than by retrieval quality.
    """
    if not sources:
        return float("nan")
    found = {r["source"] for r in results} & sources
    return len(found) / min(len(sources), k)


def hit_rate(results, gold) -> float:
    """1.0 if any relevant location appears anywhere in the returned set."""
    return 1.0 if any(is_relevant(r, gold) for r in results) else 0.0


def reciprocal_rank(results, gold) -> float:
    """1/rank of the first relevant result; 0 if none. Rewards ranking it first."""
    for i, r in enumerate(results, start=1):
        if is_relevant(r, gold):
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
              for i, r in enumerate(results, start=1) if is_relevant(r, gold))
    n_relevant = sum(1 for r in results if is_relevant(r, gold))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_relevant + 1))
    return dcg / idcg if idcg else 0.0


def normalize(text: str) -> str:
    """Collapse whitespace and lowercase, so a phrase spanning a line break matches."""
    return re.sub(r"\s+", " ", text).strip().lower()


def context_recall(results, answer: str) -> float:
    """Does the text actually returned contain the answer?

    This is the metric ranking cannot express. hit@k asks whether a chunk from
    the right *page* was returned; a chunk can satisfy that while being cut
    before the sentence carrying the answer. Context recall asks the question
    that decides whether a model could answer at all.
    """
    if not answer:
        return float("nan")
    needle = normalize(answer)
    return 1.0 if any(needle in normalize(r["text"]) for r in results) else 0.0


def context_tokens(results, tokenizer) -> int:
    """Total tokens handed to the generator -- the cost side of expansion."""
    return sum(len(tokenizer.encode(r["text"], add_special_tokens=False)) for r in results)


def score_run(cases, retrieve_fn, k: int = 5) -> tuple[dict, list[dict]]:
    totals = {"hit_rate": 0.0, "mrr": 0.0, "ndcg": 0.0, "src_recall": 0.0}
    per_case = []
    for case in cases:
        results = retrieve_fn(case["question"])
        gold = gold_keys(case)
        m = {
            "hit_rate": hit_rate(results, gold),
            "mrr": reciprocal_rank(results, gold),
            "ndcg": ndcg(results, gold),
            "src_recall": source_recall(results, gold_sources(case), k),
        }
        for key in totals:
            totals[key] += m[key]
        per_case.append({
            "id": case["id"], "kind": case.get("kind", "-"),
            "gold_sources": sorted(gold_sources(case)),
            "got": [f"{r['source'][:14]}:{r['locator']['value']}" for r in results],
            **m,
        })
    n = len(cases) or 1
    return {name: total / n for name, total in totals.items()}, per_case


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--candidate-k", type=int, default=CANDIDATE_K)
    ap.add_argument("--per-case", action="store_true")
    ap.add_argument("--emit", type=Path, default=Path(__file__).parent.parent / "eval" / "results.json",
                    help="write machine-readable results here (default: eval/results.json)")
    args = ap.parse_args()

    # Every reported figure is collected here and written out at the end.
    # Documentation quoting these numbers has drifted before -- the README once
    # claimed "84 evaluation cases" directly above figures measured on 23 -- so
    # there is now one machine-written source rather than four hand-maintained
    # copies. If a document disagrees with results.json, the document is wrong.
    emitted: dict = {"corpus": {}, "candidate_pool": {}, "end_to_end": {},
                     "expansion": {}, "abstention": {}}

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
    print(f"{'first stage':<18}{'any-hit':>9}{'MRR':>9}{'src recall':>12}")
    for label, cfg in pools:
        summary, _ = score_run(answerable, lambda q, c=cfg: shortlist(
            q, index, metadata, model, k=args.candidate_k, bm25=bm25, **c),
            k=args.candidate_k)
        emitted["candidate_pool"][label] = {
            k2: round(v, 3) for k2, v in summary.items()}
        print(f"{label:<18}{summary['hit_rate']:>9.3f}{summary['mrr']:>9.3f}"
              f"{summary['src_recall']:>12.3f}")

    # ---- Layer 2: end-to-end, with reranking ------------------------------
    # max_per_source=None disables the diversity cap, so its effect is isolated:
    # the two "weighted + rerank" rows differ only in that one parameter.
    finals = [
        ("dense, no rerank", dict(fusion="none", use_reranker=False)),
        ("dense + rerank",   dict(fusion="none", use_reranker=True, max_per_source=None)),
        ("rrf + rerank",     dict(fusion="rrf", use_reranker=True, max_per_source=None)),
        ("weighted + rerank", dict(fusion="weighted", alpha=0.5, use_reranker=True,
                                   max_per_source=None)),
        ("  + diversity 2/src", dict(fusion="weighted", alpha=0.5, use_reranker=True,
                                     max_per_source=2)),
        ("  + diversity 1/src", dict(fusion="weighted", alpha=0.5, use_reranker=True,
                                     max_per_source=1)),
    ]

    print(f"\nEND TO END @ {args.k}")
    print(f"{'pipeline':<20}{'any-hit':>9}{'MRR':>9}{'NDCG':>9}{'src recall':>12}")
    runs = {}
    for label, cfg in finals:
        summary, per_case = score_run(answerable, lambda q, c=cfg: retrieve(
            q, index, metadata, model, k=args.k,
            candidate_k=args.candidate_k, bm25=bm25, **c), k=args.k)
        runs[label] = (summary, per_case)
        emitted["end_to_end"][label.strip()] = {
            k2: round(v, 3) for k2, v in summary.items()}
        print(f"{label:<20}{summary['hit_rate']:>9.3f}{summary['mrr']:>9.3f}"
              f"{summary['ndcg']:>9.3f}{summary['src_recall']:>12.3f}")

    if args.per_case:
        for label, (_, per_case) in runs.items():
            print(f"\n--- {label} ---")
            for c in sorted(per_case, key=lambda x: (x["src_recall"], x["mrr"])):
                if c["mrr"] < 1.0 or c["src_recall"] < 1.0:
                    gold = ", ".join(g[:16] for g in c["gold_sources"])
                    print(f"  {c['id']:<14}{c['kind']:<10}"
                          f"mrr={c['mrr']:.2f} src={c['src_recall']:.2f}")
                    print(f"      gold: {gold}")
                    print(f"      got : {', '.join(c['got'])}")

    # ---- Layer 3: context expansion ---------------------------------------
    # Expansion cannot change ranking, so hit/MRR/NDCG are identical by
    # construction and reporting them here would be noise. What changes is
    # whether the returned text contains the answer, and what that costs.
    print(f"\nCONTEXT EXPANSION @ {args.k}   (ranking is unchanged by construction)")
    print(f"{'mode':<16}{'ctx recall':>12}{'tokens/query':>14}{'blocks':>9}")
    tok = model.tokenizer
    with_answers = [c for c in answerable if c.get("answer_contains")]
    for label, cfg in (("none (chunks)", dict(expansion="none")),
                       ("window +/-1", dict(expansion="window", window=1)),
                       ("page", dict(expansion="page"))):
        hits, toks, blocks = 0.0, 0, 0
        for case in with_answers:
            res = retrieve(case["question"], index, metadata, model, k=args.k,
                           candidate_k=args.candidate_k, bm25=bm25,
                           use_reranker=True, **cfg)
            hits += context_recall(res, case["answer_contains"])
            toks += context_tokens(res, tok)
            blocks += len(res)
        n = len(with_answers)
        emitted["expansion"][label] = {
            "context_recall": round(hits / n, 3),
            "tokens_per_query": round(toks / n),
            "blocks_per_query": round(blocks / n, 1),
        }
        print(f"{label:<16}{hits / n:>12.3f}{toks / n:>14.0f}{blocks / n:>9.1f}")

    # ---- Abstention calibration -------------------------------------------
    # Collect top-1 confidence for both populations, then sweep the threshold.
    # A threshold picked by eyeballing a handful of scores is a guess; this is
    # the tradeoff curve it should be picked from.
    def top1(case):
        res = retrieve(case["question"], index, metadata, model, k=args.k,
                       candidate_k=args.candidate_k, bm25=bm25, use_reranker=True)
        return res[0]["rerank_score"] if res else float("-inf")

    ans_scores = [top1(c) for c in answerable]
    adv_scores = [top1(c) for c in adversarial]

    print("\nABSTENTION CALIBRATION")
    print(f"  answerable   n={len(ans_scores):<3} min {min(ans_scores):+.2f}  "
          f"median {sorted(ans_scores)[len(ans_scores) // 2]:+.2f}  max {max(ans_scores):+.2f}")
    print(f"  unanswerable n={len(adv_scores):<3} min {min(adv_scores):+.2f}  "
          f"median {sorted(adv_scores)[len(adv_scores) // 2]:+.2f}  max {max(adv_scores):+.2f}")

    by_kind = {}
    for case, score in zip(adversarial, adv_scores):
        by_kind.setdefault(case.get("adversarial_kind", "?"), []).append(score)
    print("  unanswerable by kind:")
    for kind, scores in sorted(by_kind.items()):
        print(f"    {kind:<11} n={len(scores):<3} max {max(scores):+.2f}  "
              f"({', '.join(f'{s:+.1f}' for s in sorted(scores, reverse=True))})")

    print(f"\n  {'threshold':>10}{'caught':>9}{'false abstain':>15}{'net':>8}")
    best = None
    for t in [-10, -8, -6, -5, -4, -3, -2, -1, 0, 1, 2]:
        caught = sum(1 for s in adv_scores if s < t) / len(adv_scores)
        false_ab = sum(1 for s in ans_scores if s < t) / len(ans_scores)
        net = caught - false_ab
        flag = ""
        if best is None or net > best[1]:
            best, flag = (t, net), ""
        print(f"  {t:>10}{caught:>9.3f}{false_ab:>15.3f}{net:>8.3f}{flag}")
    print(f"\n  best net separation at threshold {best[0]} (net {best[1]:.3f})")
    print("  note: false abstention is the costlier error -- refusing a question the")
    print("        corpus CAN answer is worse than answering a weak one with citations.")


    if args.emit:
        args.emit.parent.mkdir(exist_ok=True)
        emitted["generated_by"] = "src/evaluate.py"
        args.emit.write_text(json.dumps(emitted, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.emit}")


if __name__ == "__main__":
    main()
