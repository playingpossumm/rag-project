"""Does pseudo-relevance feedback reach the questions that describe a term?

Measured on 2026-08-27, the twelve cases that fail under every pipeline
configuration split into two kinds (see `docs/engineering-log.md`). Seven, all
on the ML corpus, ask about an attribute many papers share. The other five, on
birds and quant, **describe a term and ask for its name** -- "which small group
of feathers helps prevent a stall at low speed" for the alula, "which effect
describes past winners continuing to outperform" for momentum. The answer word
is absent from the question, so BM25 has nothing to match and the cross-encoder
has little to read.

Query decomposition is the planned fix for the first kind and needs an LLM.
`src/query_expansion.py` is the classical answer to the second kind and needs
nothing: run the query, assume the top results are relevant, harvest the terms
that distinguish them, re-run. It has been in the repo unmeasured.

**The hypothesis has a visible way to fail, and it is worth stating before the
run.** Feedback terms come from the top results of the *original* query. If the
answer word is absent from those, expansion cannot invent it -- it will sharpen
a query that was already pointed at the wrong passages. Recall that expansion is
given to the sparse side only, precisely because it distorts the dense side.

So this reports three things per corpus: the aggregate, which says whether
turning it on is safe; the structural fixture, which says whether it reaches the
cases it was considered for; and which individual questions change hands.

    .venv\\Scripts\\python.exe src\\sweep_query_expansion.py
    .venv\\Scripts\\python.exe src\\sweep_query_expansion.py --corpus birds
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import corpora  # noqa: E402
import rerank as rr  # noqa: E402
from evaluate import (gold_keys, gold_sources, hit_rate, load_cases, ndcg,  # noqa: E402
                      reciprocal_rank, source_recall)
from hybrid import build_bm25  # noqa: E402
from retrieve import (CANDIDATE_K, DEFAULT_FUSION, EMBEDDING_MODEL, TOP_K,  # noqa: E402
                      load_index, retrieve)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
OUT = ROOT / "eval" / "query-expansion-sweep.json"
MODES = ["none", "prf"]


def fixture_ids(cfg: dict) -> set:
    """The structural cases for this corpus, if a fixture has been written."""
    from check_freshness import artefact_suffix
    p = ROOT / "eval" / f"hard_cases{artefact_suffix(cfg['golden'])}.json"
    if not p.exists():
        return set()
    data = json.loads(p.read_text(encoding="utf-8"))
    return {c["id"] if isinstance(c, dict) else c
            for c in data.get("structural", [])}


def score(cases, index, metadata, model, bm25, mode, k, cand):
    totals = {"hit_rate": 0.0, "mrr": 0.0, "ndcg": 0.0, "src_recall": 0.0}
    per_case = {}
    for case in cases:
        results = retrieve(case["question"], index, metadata, model, k=k,
                           candidate_k=cand, use_reranker=True,
                           fusion=DEFAULT_FUSION, bm25=bm25, max_per_source=2,
                           query_expansion=mode)
        gold = gold_keys(case)
        m = {"hit_rate": hit_rate(results, gold),
             "mrr": reciprocal_rank(results, gold),
             "ndcg": ndcg(results, gold),
             "src_recall": source_recall(results, gold_sources(case), k)}
        for key in totals:
            totals[key] += m[key]
        per_case[case["id"]] = m["hit_rate"] == 1.0
    n = max(len(cases), 1)
    return {key: round(v / n, 4) for key, v in totals.items()}, per_case


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", action="append")
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--candidate-k", type=int, default=CANDIDATE_K)
    ap.add_argument("--emit", type=Path, default=OUT)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)

    reg = corpora.registry()
    wanted = args.corpus or list(reg)
    report = {}

    for name in wanted:
        cfg = reg.get(name)
        if not cfg or not cfg["indexed"] or not cfg["golden"]:
            continue
        answerable, _ = load_cases(cfg["golden"])
        index, metadata = load_index(cfg["store"])
        bm25 = build_bm25(metadata)
        rr.RERANK_BLEND = cfg["rerank_blend"]
        structural = fixture_ids(cfg)

        print(f"\n  {cfg['label']}  ({len(answerable)} answerable, "
              f"{len(structural)} structural)")
        print(f"    {'expansion':<12}{'any-hit':>10}{'MRR':>9}{'NDCG':>9}"
              f"{'src':>8}{'structural hit':>16}")

        rows, hits = {}, {}
        for mode in MODES:
            rr.clear_cache()
            agg, per = score(answerable, index, metadata, model, bm25, mode,
                             args.k, args.candidate_k)
            struck = sum(1 for i in structural if per.get(i))
            agg["structural_hit"] = f"{struck}/{len(structural)}" if structural else "-"
            rows[mode], hits[mode] = agg, per
            mark = "  <- shipped" if mode == "none" else ""
            print(f"    {mode:<12}{agg['hit_rate']:>10.3f}{agg['mrr']:>9.3f}"
                  f"{agg['ndcg']:>9.3f}{agg['src_recall']:>8.3f}"
                  f"{agg['structural_hit']:>16}{mark}")

        base, prf = hits["none"], hits["prf"]
        won = sorted(i for i in base if prf.get(i) and not base[i])
        lost = sorted(i for i in base if base[i] and not prf.get(i))
        if won or lost:
            print(f"      won:  {won or '[]'}")
            print(f"      lost: {lost or '[]'}")
            if structural:
                s_won = [i for i in won if i in structural]
                print(f"      of the structural fixture, expansion recovers: "
                      f"{s_won or 'none'}")
        else:
            print("      no question changes hands")
        report[name] = {"label": cfg["label"], "n": len(answerable),
                        "structural": sorted(structural),
                        "modes": rows, "won": won, "lost": lost}

    if not report:
        print("nothing to sweep")
        return 1

    args.emit.parent.mkdir(exist_ok=True)
    args.emit.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    try:
        where = args.emit.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        where = str(args.emit)
    print(f"\n  wrote {where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
