"""One configuration on all three corpora, against each corpus's own tuning.

This project's claim is that the pipeline works across unlike document sets:
arXiv PDFs, Wikipedia pages in four file formats, and quantitative-finance
papers. The numbers usually quoted for that -- 0.851, 0.846, 0.886 -- are each
produced under that corpus's own settings, because four things are now tuned per
corpus: the abstention threshold, the rerank blend, the candidate pool size, and
whether a second embedder is fused.

**So those three numbers are three tuned pipelines, not one pipeline on three
corpora, and quoting them as evidence of generalisation overstates the case.**
Both questions are worth answering and they are different questions:

    does the pipeline generalise?     one configuration, three corpora
    what does each corpus need?       each corpus's own configuration

This measures the first, which nothing here had measured. The ranking metrics
are threshold-independent -- any-hit, MRR, NDCG and source recall are computed
from the ordering alone -- so a single shared configuration is directly
comparable across corpora in a way the gate never can be. The gate is reported
separately and per corpus, because a threshold is a property of a corpus's score
distribution and cannot be made uniform without being meaningless.

The uniform configuration is the *untuned* one: the defaults a new corpus gets
before anybody measures anything. That is the honest baseline, because it is
what the system does for a document set it has never seen.

    .venv\\Scripts\\python.exe src\\uniform_baseline.py
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
from retrieve import (CANDIDATE_K, DEFAULT_FUSION, EMBEDDING_MODEL,  # noqa: E402
                      TOP_K, load_ensemble, load_index, retrieve)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
OUT = ROOT / "eval" / "uniform-baseline.json"

# The defaults a corpus gets before anyone tunes it. Deliberately the module
# constants rather than a hand-picked "fair" middle: a compromise chosen after
# seeing the results would be tuning with extra steps.
UNIFORM = {"candidate_k": CANDIDATE_K, "blend": 0.0, "fusion": DEFAULT_FUSION,
           "max_per_source": 2, "ensemble": False}


def score(cases, index, metadata, model, bm25, *, k, candidate_k, blend,
          fusion, max_per_source, ensemble):
    totals = {"hit_rate": 0.0, "mrr": 0.0, "ndcg": 0.0, "src_recall": 0.0}
    rr.clear_cache()
    for case in cases:
        results = retrieve(case["question"], index, metadata, model, k=k,
                           candidate_k=candidate_k, use_reranker=True,
                           fusion=fusion, bm25=bm25,
                           # Passed per call; set on the rerank module until
                           # 2026-09-07.
                           max_per_source=max_per_source, rerank_blend=blend,
                           ensemble=ensemble)
        gold = gold_keys(case)
        totals["hit_rate"] += hit_rate(results, gold)
        totals["mrr"] += reciprocal_rank(results, gold)
        totals["ndcg"] += ndcg(results, gold)
        totals["src_recall"] += source_recall(results, gold_sources(case), k)
    n = max(len(cases), 1)
    return {key: round(v / n, 4) for key, v in totals.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--emit", type=Path, default=OUT)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)

    report = {}
    print(f"\n  uniform configuration: {UNIFORM['candidate_k']} candidates, "
          f"blend {UNIFORM['blend']:.2f}, {UNIFORM['fusion']} fusion, "
          f"cap {UNIFORM['max_per_source']}/source, no ensemble\n")
    print(f"  {'corpus':<22}{'any-hit':>9}{'MRR':>8}{'NDCG':>8}{'src':>8}   configuration")

    for name, cfg in corpora.registry().items():
        if not cfg["indexed"] or not cfg["golden"] or not cfg["golden"].exists():
            continue
        answerable, _ = load_cases(cfg["golden"])
        index, metadata = load_index(cfg["store"])
        bm25 = build_bm25(metadata)
        ensemble = load_ensemble(cfg["store"]) if cfg.get("ensemble_model") else None

        uniform = score(answerable, index, metadata, model, bm25, k=args.k,
                        **{**UNIFORM, "ensemble": None})
        tuned = score(answerable, index, metadata, model, bm25, k=args.k,
                      candidate_k=cfg["candidate_k"], blend=cfg["rerank_blend"],
                      fusion=UNIFORM["fusion"], max_per_source=2,
                      ensemble=ensemble)

        print(f"  {cfg['label']:<22}{uniform['hit_rate']:>9.3f}"
              f"{uniform['mrr']:>8.3f}{uniform['ndcg']:>8.3f}"
              f"{uniform['src_recall']:>8.3f}   uniform")
        print(f"  {'':<22}{tuned['hit_rate']:>9.3f}{tuned['mrr']:>8.3f}"
              f"{tuned['ndcg']:>8.3f}{tuned['src_recall']:>8.3f}   tuned "
              f"({cfg['candidate_k']} cand, blend {cfg['rerank_blend']:.2f}"
              f"{', +ensemble' if ensemble else ''})")
        deltas = {m: round(tuned[m] - uniform[m], 4) for m in uniform}
        print(f"  {'':<22}{deltas['hit_rate']:>+9.3f}{deltas['mrr']:>+8.3f}"
              f"{deltas['ndcg']:>+8.3f}{deltas['src_recall']:>+8.3f}   "
              f"what tuning buys\n")
        report[name] = {"label": cfg["label"], "n": len(answerable),
                        "uniform": uniform, "tuned": tuned, "delta": deltas}

    if not report:
        print("nothing to measure")
        return 1

    spread = {m: round(max(r["uniform"][m] for r in report.values())
                       - min(r["uniform"][m] for r in report.values()), 4)
              for m in ("hit_rate", "mrr", "ndcg", "src_recall")}
    print(f"  Spread across corpora under ONE configuration: "
          f"any-hit {spread['hit_rate']:.3f}, MRR {spread['mrr']:.3f}, "
          f"NDCG {spread['ndcg']:.3f}, src {spread['src_recall']:.3f}")
    print("  That spread is the generalisation claim. The tuned rows are a "
          "different claim\n  and should not be quoted for this one.")

    args.emit.write_text(json.dumps(
        {"generated_by": "src/uniform_baseline.py", "uniform_config": UNIFORM,
         "spread_under_uniform": spread, "corpora": report}, indent=1) + "\n",
        encoding="utf-8")
    print(f"\n  wrote {args.emit.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
