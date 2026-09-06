"""Measure the rerank blend weight across every corpus.

The cross-encoder used to replace the first stage's ordering outright. That
discards evidence: when a question describes a term rather than naming it, the
cross-encoder has little to work with and its ordering approaches arbitrary,
while the first stage may already have the answer at rank 1.

The obvious risk in fixing that is tuning the weight to the corpus that
prompted it. So this sweeps every weight over every corpus and prints them
side by side. A weight is only worth shipping if it does not hurt the two
corpora that were never the problem.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import corpora  # noqa: E402
import rerank as rr  # noqa: E402
from evaluate import (gold_keys, hit_rate, load_cases, ndcg,  # noqa: E402
                      reciprocal_rank, source_recall)
from hybrid import build_bm25  # noqa: E402
from retrieve import EMBEDDING_MODEL, load_index, retrieve  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent

# Read from corpora.json rather than listed here. This was the fourth copy of
# the corpus table in the repo, and the one build_analytics.py kept had already
# demonstrated the failure: a corpus added to corpora.json and not to the copy
# is skipped in silence, and a sweep that quietly measures two of three corpora
# is exactly the trap this script exists to prevent.
def configured():
    return [(cfg["label"], cfg["store"], cfg["golden"])
            for cfg in corpora.registry().values()
            if cfg["indexed"] and cfg["golden"] and cfg["golden"].exists()]


def score(cases, index, metadata, model, bm25, blend, k, cand):
    # The harness's own hit_rate and reciprocal_rank. This file carried its own
    # arithmetic for both until 2026-09-06, keyed on a locator triple that
    # defaulted a missing kind to "page" and fell back to a "page" field no
    # result carries; on real results the two agreed, so the figures in
    # eval/blend-sweep.json are unchanged by the import. The blend goes on the
    # call rather than on the rerank module, which is the same figure by a
    # route that cannot leak into the next importer.
    hits = mrr = nd = src = 0.0
    for c in cases:
        gold = gold_keys(c)
        gold_src = {g["source"] for g in c.get("gold", [])}
        res = retrieve(c["question"], index, metadata, model, k=k,
                       candidate_k=cand, use_reranker=True, fusion="rrf",
                       bm25=bm25, max_per_source=2, rerank_blend=blend)
        hits += hit_rate(res, gold)
        mrr += reciprocal_rank(res, gold)
        nd += ndcg(res, gold)
        src += source_recall(res, gold_src, k)
    n = max(len(cases), 1)
    return hits / n, mrr / n, nd / n, src / n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="0,0.2,0.35,0.5,0.7")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--candidate-k", type=int, default=20)
    args = ap.parse_args()
    weights = [float(w) for w in args.weights.split(",")]

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)

    out = {}
    for label, store, golden in configured():
        index, metadata = load_index(store)
        bm25 = build_bm25(metadata)
        answerable, _ = load_cases(golden)
        print(f"\n  {label}  ({len(answerable)} answerable)")
        print(f"    {'blend':>6}{'any-hit':>10}{'MRR':>9}{'NDCG':>9}{'src recall':>12}")
        rows = {}
        for w in weights:
            rr.clear_cache()
            h, m, n, s = score(answerable, index, metadata, model, bm25,
                               w, args.k, args.candidate_k)
            rows[w] = {"hit_rate": h, "mrr": m, "ndcg": n, "src_recall": s}
            star = "  <-- current" if abs(w - 0.0) < 1e-9 else ""
            print(f"    {w:>6.2f}{h:>10.3f}{m:>9.3f}{n:>9.3f}{s:>12.3f}{star}")
        out[label] = rows

    dest = ROOT / "eval" / "blend-sweep.json"
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\n  wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
