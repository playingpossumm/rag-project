"""What does reranking fewer candidates cost, and what does it buy back?

`src/profile_query.py` settled where the time goes: reranking is **92-95%** of a
query on all three corpora, around 1,000 ms of roughly 1,100. Everything else
together -- embedding, FAISS, BM25, fusion, the diversity cap, context expansion
-- is about 80 ms. So there is exactly one stage worth optimising and two ways
to make it cheaper.

The first was measured and rejected. Threads are already at the machine's best
(10 of 12 cores, and fewer is slower), and padding waste inside the batch is
3% on the ML corpus and 17% on birds, so length bucketing could buy at most a
sixth of one stage on one corpus in exchange for reordering logic in the hot
path. `src/compare_rerankers.py` had already ruled out a faster *model*: the
candidates that are quicker are worse, and the one that separates best is ten
times slower.

That leaves `CANDIDATE_K`. Cost is linear in it -- twenty pairs at ~48 ms each --
and its current value of 20 was chosen for recall headroom with no latency term
in the decision, because nothing had measured the latency.

So this sweeps it, on every corpus, at the shipped configuration, and reports
quality and measured milliseconds side by side. The reranker cannot recover a
passage the first stage never retrieved, so the honest ceiling is reported too:
how often the answer is in the pool at all at each size.

    .venv\\Scripts\\python.exe src\\sweep_candidates.py
    .venv\\Scripts\\python.exe src\\sweep_candidates.py --corpus birds
"""
import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import corpora  # noqa: E402
import rerank as rr  # noqa: E402
from diversify import diversify  # noqa: E402
from evaluate import (gold_keys, gold_sources, hit_rate, is_relevant,  # noqa: E402
                      load_cases, ndcg, reciprocal_rank, source_recall)
from hybrid import build_bm25  # noqa: E402
from retrieve import (DEFAULT_FUSION, EMBEDDING_MODEL, TOP_K, load_index,  # noqa: E402
                      shortlist)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
OUT = ROOT / "eval" / "candidate-sweep.json"
SIZES = [6, 8, 10, 12, 16, 20, 28]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", action="append")
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    ap.add_argument("--emit", type=Path, default=OUT)
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
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
        rr.load_reranker()

        print(f"\n  {cfg['label']}  ({len(answerable)} answerable, "
              f"blend {cfg['rerank_blend']:.2f})")
        print(f"    {'cand':>5}{'pool ceiling':>14}{'any-hit':>10}{'MRR':>8}"
              f"{'NDCG':>8}{'src':>7}{'rerank ms':>11}")

        rows = {}
        for cand in sizes:
            rr.clear_cache()
            totals = {"hit_rate": 0.0, "mrr": 0.0, "ndcg": 0.0, "src_recall": 0.0}
            ceiling = 0.0
            laps = []
            for case in answerable:
                gold = gold_keys(case)
                pool = shortlist(case["question"], index, metadata, model,
                                 k=cand, fusion=DEFAULT_FUSION, bm25=bm25)
                # The ceiling: the reranker can only reorder what it is given,
                # so a pool without the answer is a loss no ranking can undo.
                ceiling += 1.0 if any(is_relevant(c, gold) for c in pool) else 0.0

                started = time.perf_counter()
                ranked = rr.rerank(case["question"], pool, k=len(pool))
                laps.append((time.perf_counter() - started) * 1000)

                results = diversify(ranked, k=args.k, max_per_source=2)
                totals["hit_rate"] += hit_rate(results, gold)
                totals["mrr"] += reciprocal_rank(results, gold)
                totals["ndcg"] += ndcg(results, gold)
                totals["src_recall"] += source_recall(
                    results, gold_sources(case), args.k)

            n = max(len(answerable), 1)
            row = {k: round(v / n, 4) for k, v in totals.items()}
            row["pool_ceiling"] = round(ceiling / n, 4)
            row["rerank_ms"] = round(st.median(laps), 1)
            rows[cand] = row
            mark = "  <- shipped" if cand == 20 else ""
            print(f"    {cand:>5}{row['pool_ceiling']:>14.3f}"
                  f"{row['hit_rate']:>10.3f}{row['mrr']:>8.3f}{row['ndcg']:>8.3f}"
                  f"{row['src_recall']:>7.3f}{row['rerank_ms']:>11.0f}{mark}")

        base = rows.get(20)
        if base:
            print(f"\n    against 20 candidates:")
            for cand, row in rows.items():
                if cand == 20:
                    continue
                dh = (row["hit_rate"] - base["hit_rate"]) * 100
                dm = (row["mrr"] - base["mrr"]) * 100
                saved = base["rerank_ms"] - row["rerank_ms"]
                print(f"      {cand:>3}: any-hit {dh:+5.1f}pts  MRR {dm:+5.1f}pts  "
                      f"{saved:+6.0f} ms "
                      f"({saved / max(base['rerank_ms'], 1):+.0%})")
        report[name] = {"label": cfg["label"], "n": len(answerable),
                        "blend": cfg["rerank_blend"],
                        "sizes": {str(c): r for c, r in rows.items()}}

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
