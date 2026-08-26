"""Which fusion does each corpus want, measured at the configuration served?

The abstention threshold does not transfer between corpora. Neither does the
rerank blend. `HANDOFF.md` has said "and probably fusion" since the second
corpus was built, on the strength of one number: the bird candidate pool holds
the answer 96.2% of the time under dense retrieval alone and 88.5% under RRF, so
fusing costs that corpus two questions before reranking even starts.

That was never acted on, for a stated reason -- 26 cases is too few to move a
global default -- and the reason is sound about a *global* default and silent
about a per-corpus one, which is what the threshold and the blend already are.

What was missing was the measurement at the served configuration. `evaluate.py`
compares fusions *before* the diversity cap and applies the cap only to RRF, so
the table every document quotes has no row for "weighted, capped" at all. The
comparison that decides this has never been run.

So: every fusion, every corpus, at k=5 from 20 candidates with the corpus's own
rerank blend and the 2-per-source cap -- the pipeline `api.ask()` runs. A fusion
is only worth setting for one corpus if the whole table is on the table.

    .venv\\Scripts\\python.exe src\\sweep_fusion.py
    .venv\\Scripts\\python.exe src\\sweep_fusion.py --corpus birds
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
from retrieve import (CANDIDATE_K, EMBEDDING_MODEL, TOP_K, load_index,  # noqa: E402
                      retrieve)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
OUT = ROOT / "eval" / "fusion-sweep.json"

# `alpha` only means anything to the weighted branch; dense and RRF ignore it.
# Spelled out as pairs rather than a nested loop so the printed table has one
# row per thing that can be configured, which is what a decision needs.
FUSIONS = [
    ("dense", "none", 0.5),
    ("rrf", "rrf", 0.5),
    ("weighted a=0.3", "weighted", 0.3),
    ("weighted a=0.5", "weighted", 0.5),
    ("weighted a=0.7", "weighted", 0.7),
]


def score(cases, index, metadata, model, bm25, fusion, alpha, k, cand):
    totals = {"hit_rate": 0.0, "mrr": 0.0, "ndcg": 0.0, "src_recall": 0.0}
    missed = []
    for case in cases:
        results = retrieve(case["question"], index, metadata, model, k=k,
                           candidate_k=cand, use_reranker=True, fusion=fusion,
                           alpha=alpha, bm25=bm25, max_per_source=2)
        gold = gold_keys(case)
        m = {"hit_rate": hit_rate(results, gold),
             "mrr": reciprocal_rank(results, gold),
             "ndcg": ndcg(results, gold),
             "src_recall": source_recall(results, gold_sources(case), k)}
        for key in totals:
            totals[key] += m[key]
        if m["hit_rate"] == 0.0:
            missed.append(case["id"])
    n = max(len(cases), 1)
    return {key: round(total / n, 4) for key, total in totals.items()} | {
        "missed": sorted(missed)}


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
    out = {}

    for name in wanted:
        cfg = reg.get(name)
        if not cfg:
            print(f"  unknown corpus {name!r}; have {sorted(reg)}")
            return 2
        if not cfg["indexed"] or not cfg["golden"] or not cfg["golden"].exists():
            print(f"  {name}: no index or golden set, skipped")
            continue

        answerable, _ = load_cases(cfg["golden"])
        index, metadata = load_index(cfg["store"])
        bm25 = build_bm25(metadata)
        # The corpus's own blend, so this measures a change of fusion and not a
        # change of fusion plus a change of everything else.
        rr.RERANK_BLEND = cfg["rerank_blend"]

        print(f"\n  {cfg['label']}  ({len(answerable)} answerable, "
              f"blend {cfg['rerank_blend']:.2f})")
        print(f"    {'fusion':<16}{'any-hit':>10}{'MRR':>9}{'NDCG':>9}"
              f"{'src recall':>12}")
        rows = {}
        shipped = None
        for label, fusion, alpha in FUSIONS:
            rr.clear_cache()
            r = score(answerable, index, metadata, model, bm25, fusion, alpha,
                      args.k, args.candidate_k)
            rows[label] = r
            if fusion == "rrf":
                shipped = r
            mark = "  <- shipped" if fusion == "rrf" else ""
            print(f"    {label:<16}{r['hit_rate']:>10.3f}{r['mrr']:>9.3f}"
                  f"{r['ndcg']:>9.3f}{r['src_recall']:>12.3f}{mark}")

        # Which questions a change would actually win or lose. A delta of 0.038
        # on 26 cases is one question, and the only way to judge one question is
        # to read it.
        if shipped is not None:
            base = set(shipped["missed"])
            for label, r in rows.items():
                if label == "rrf":
                    continue
                fixed = base - set(r["missed"])
                broke = set(r["missed"]) - base
                if fixed or broke:
                    print(f"      vs {label}: "
                          f"+{sorted(fixed) if fixed else '[]'} "
                          f"-{sorted(broke) if broke else '[]'}")
        out[name] = {"label": cfg["label"], "blend": cfg["rerank_blend"],
                     "n": len(answerable), "fusions": rows}

    if not out:
        print("nothing to sweep")
        return 1

    args.emit.parent.mkdir(exist_ok=True)
    args.emit.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    try:
        where = args.emit.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        where = str(args.emit)
    print(f"\n  wrote {where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
