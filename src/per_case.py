"""Write the per-case outcome of every golden-set question to JSON.

`evaluate.py --per-case` already computes this and prints it. Printing is enough
to read a run and useless for anything that wants to work with one: which cases
a change fixed and which it broke, whether the 15% cross-document confusion is
the same fifteen percent from run to run, or drawing all 84 cases at once.

A separate script rather than a flag on the harness, deliberately. evaluate.py's
job is to produce the aggregate numbers every document quotes, and its output
file is the one source those documents are checked against. Growing it a second
output shape gives that one job two reasons to change.

It reuses evaluate.py's metric functions rather than reimplementing them, so a
per-case number here and an aggregate number there cannot disagree -- the
aggregates in this file's `totals` are the mean of its own rows, which is a
check on both.

    .venv\\Scripts\\python.exe src\\per_case.py             # serving defaults
    .venv\\Scripts\\python.exe src\\per_case.py --no-rerank
"""
import argparse
import json
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

from abstain import ABSTAIN_THRESHOLD
from evaluate import (context_recall, gold_keys, gold_sources, hit_rate,
                      is_relevant, load_cases, ndcg, reciprocal_rank,
                      source_recall)
from hybrid import build_bm25
from retrieve import (CANDIDATE_K, DEFAULT_FUSION, EMBEDDING_MODEL, TOP_K,
                      load_index, retrieve)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(__file__).parent.parent / "eval" / "per_case.json"


def outcome(case: dict, results: list[dict], gold: set, confident: bool) -> str:
    """One word for what happened, for filtering and for colouring a chart.

    Answerable and unanswerable cases are scored on different questions --
    "was the answer found" against "was the refusal correct" -- so they get
    different vocabularies rather than a shared right/wrong that would mean two
    things at once.
    """
    if case.get("unanswerable"):
        return "refused" if not confident else "answered_anyway"
    if not confident:
        return "refused_wrongly"
    return "found" if any(is_relevant(r, gold) for r in results) else "missed"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--candidate-k", type=int, default=CANDIDATE_K)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--fusion", default=DEFAULT_FUSION,
                    choices=["none", "rrf", "weighted"])
    ap.add_argument("--max-per-source", type=int, default=2,
                    help="0 disables the diversity cap")
    ap.add_argument("--emit", type=Path, default=OUT)
    args = ap.parse_args()

    opts = dict(use_reranker=not args.no_rerank, fusion=args.fusion,
                max_per_source=args.max_per_source or None)

    answerable, adversarial = load_cases()
    index, metadata = load_index()
    model = SentenceTransformer(EMBEDDING_MODEL)
    bm25 = build_bm25(metadata)

    print(f"{len(answerable)} answerable + {len(adversarial)} adversarial "
          f"at k={args.k}, {opts}")

    rows = []
    for case in answerable + adversarial:
        results = retrieve(case["question"], index, metadata, model, k=args.k,
                           candidate_k=args.candidate_k, bm25=bm25, **opts)
        gold = gold_keys(case)
        sources = gold_sources(case)

        # Read before expansion would have diluted it, which is why this script
        # does not expand: the gate scores the chunk the reranker actually saw.
        # It is also why `confidence` is absent rather than 0 with the reranker
        # off -- there is no calibrated score on that path, and 0 would compare
        # as "at the threshold" in anything that read this file.
        top = results[0].get("rerank_score") if results and not args.no_rerank else None
        confident = (top >= ABSTAIN_THRESHOLD) if top is not None else True

        row = {
            "id": case["id"],
            "kind": case.get("kind", case.get("adversarial_kind", "-")),
            "question": case["question"],
            "unanswerable": bool(case.get("unanswerable")),
            "gold_sources": sorted(sources),
            "confidence": round(float(top), 3) if top is not None else None,
            "abstained": not confident,
            "outcome": outcome(case, results, gold, confident),
            "results": [
                {"rank": i,
                 "source": r["source"],
                 "locator": f"{r['locator']['kind']} {r['locator']['value']}",
                 "score": round(float(r.get("rerank_score", r.get("score", 0.0))), 4),
                 "relevant": is_relevant(r, gold)}
                for i, r in enumerate(results, 1)
            ],
        }
        if not case.get("unanswerable"):
            row |= {
                "hit_rate": hit_rate(results, gold),
                "mrr": round(reciprocal_rank(results, gold), 4),
                "ndcg": round(ndcg(results, gold), 4),
                "src_recall": round(source_recall(results, sources, args.k), 4),
            }
            if case.get("answer_contains"):
                row["context_recall"] = context_recall(results, case["answer_contains"])
        rows.append(row)

    # Totals are rounded to 4dp for readability; the ROWS are authoritative.
    # This matters when comparing against eval/results.json, which rounds to 3:
    # 56 hits in 66 cases is 0.848484..., stored here as 0.8485, and re-rounding
    # that to 3dp gives 0.849 against the harness's 0.848 -- a disagreement
    # invented by rounding twice, not a difference in what was measured.
    # Recompute from "cases" rather than re-rounding these.
    ans_rows = [r for r in rows if not r["unanswerable"]]
    adv_rows = [r for r in rows if r["unanswerable"]]
    mean = lambda key, src: round(sum(r[key] for r in src) / len(src), 4) if src else None

    payload = {
        "generated_by": "src/per_case.py",
        "options": {"k": args.k, "candidate_k": args.candidate_k, **opts},
        "threshold": ABSTAIN_THRESHOLD,
        "corpus": {"chunks": len(metadata),
                   "documents": len({c["source"] for c in metadata})},
        "totals": {
            "answerable": {
                "n": len(ans_rows),
                **{k: mean(k, ans_rows) for k in ("hit_rate", "mrr", "ndcg", "src_recall")},
                "refused_wrongly": sum(r["outcome"] == "refused_wrongly" for r in ans_rows),
                "missed": sum(r["outcome"] == "missed" for r in ans_rows),
            },
            "adversarial": {
                "n": len(adv_rows),
                "refused": sum(r["outcome"] == "refused" for r in adv_rows),
                "answered_anyway": sum(r["outcome"] == "answered_anyway" for r in adv_rows),
            },
        },
        "cases": rows,
    }

    args.emit.parent.mkdir(exist_ok=True)
    args.emit.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    t = payload["totals"]
    print(f"\nanswerable   hit {t['answerable']['hit_rate']:.3f}  "
          f"mrr {t['answerable']['mrr']:.3f}  ndcg {t['answerable']['ndcg']:.3f}  "
          f"src {t['answerable']['src_recall']:.3f}")
    print(f"             {t['answerable']['missed']} missed, "
          f"{t['answerable']['refused_wrongly']} refused wrongly")
    print(f"adversarial  {t['adversarial']['refused']}/{t['adversarial']['n']} refused, "
          f"{t['adversarial']['answered_anyway']} answered anyway")
    print(f"\nwrote {args.emit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
