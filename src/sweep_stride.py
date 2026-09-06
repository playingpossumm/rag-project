"""Does a finer chunk stride put the answer in the chunk that gets ranked?

`src/answer_distance.py` found the answer sitting one chunk away on 13 of the
23 questions that return no passage containing it: the question's words are in
chunk N and the answer is in N+1, so the first stage ranks N and the reader
never sees N+1. Coverage is not the problem, since N+1 exists and holds the
answer. The problem is that no single chunk holds BOTH the words that find it
and the words that answer.

A shorter stride makes one. Chunks are 210 tokens with 40 of overlap, so the
stride is 170; halving the stride puts a chunk boundary between every pair of
today's, and a passage spanning a boundary falls wholly inside one of the new
chunks. Chunk SIZE is deliberately unchanged, because this pipeline's precision
rests on ranking small chunks -- `retrieve()` applies expansion last for that
reason -- and a larger chunk trades that away to reach the same passages.

The cost is chunk count, which is index size, ingest time and reranking work,
and it is not free in quality either: more chunks per document means a fixed
candidate pool covers less of the corpus, so the pool is reported at its
shipped size and at one scaled to match.

Build the index this compares against first, into a scratch store so the
shipped one is untouched:

    set RAG_CHUNK_OVERLAP=105
    set RAG_DATA_DIR=data-birds
    set RAG_STORE_DIR=store-birds-stride
    .venv\\Scripts\\python.exe src\\ingest.py

then

    .venv\\Scripts\\python.exe src\\sweep_stride.py --corpus birds ^
        --store store-birds-stride

The golden set is NOT rebuilt and does not need to be. Gold is recorded as a
locator -- a source, a kind and a page -- and chunking does not move which page
a sentence is on, so the same labels score both indexes. That is worth stating
because it is the reason this experiment is affordable at all.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import corpora  # noqa: E402
import rerank as rr  # noqa: E402
from evaluate import (answer_normalize, gold_keys, gold_sources,  # noqa: E402
                      hit_rate, load_cases, ndcg, reciprocal_rank,
                      source_recall)
from hybrid import build_bm25  # noqa: E402
from retrieve import (DEFAULT_FUSION, EMBEDDING_MODEL, TOP_K,  # noqa: E402
                      load_ensemble, load_index, retrieve)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent


def score(store, cfg, candidate_k, answerable, adversarial, model, label):
    index, metadata = load_index(store)
    bm25 = build_bm25(metadata)
    ensemble = load_ensemble(store)

    totals = {"hit": 0.0, "mrr": 0.0, "ndcg": 0.0, "src": 0.0}
    visible, visible_n, slips, seconds = 0, 0, 0, 0.0
    found = set()
    for case in answerable:
        rr.clear_cache()
        t0 = time.perf_counter()
        res = retrieve(case["question"], index, metadata, model, k=TOP_K,
                       candidate_k=candidate_k, use_reranker=True,
                       fusion=DEFAULT_FUSION, bm25=bm25, max_per_source=2,
                       ensemble=ensemble, rerank_blend=cfg["rerank_blend"])
        seconds += time.perf_counter() - t0
        gold = gold_keys(case)
        totals["hit"] += hit_rate(res, gold)
        totals["mrr"] += reciprocal_rank(res, gold)
        totals["ndcg"] += ndcg(res, gold)
        totals["src"] += source_recall(res, gold_sources(case), TOP_K)
        want = case.get("answer_contains")
        if want:
            visible_n += 1
            if any(answer_normalize(want) in answer_normalize(r["text"])
                   for r in res):
                visible += 1
                found.add(case["id"])
    for case in adversarial:
        rr.clear_cache()
        res = retrieve(case["question"], index, metadata, model, k=TOP_K,
                       candidate_k=candidate_k, use_reranker=True,
                       fusion=DEFAULT_FUSION, bm25=bm25, max_per_source=2,
                       ensemble=ensemble, rerank_blend=cfg["rerank_blend"])
        if res and res[0]["rerank_score"] >= cfg["threshold"]:
            slips += 1

    n = len(answerable) or 1
    return {"label": label, "chunks": len(metadata), "candidate_k": candidate_k,
            "any_hit": round(totals["hit"] / n, 4),
            "mrr": round(totals["mrr"] / n, 4),
            "ndcg": round(totals["ndcg"] / n, 4),
            "src_recall": round(totals["src"] / n, 4),
            "answer_visible": round(visible / max(visible_n, 1), 4),
            "slips": slips, "seconds_per_query": round(seconds / n, 2),
            "found": sorted(found)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--store", required=True,
                    help="the scratch index built at the finer stride")
    ap.add_argument("--emit", type=Path,
                    default=ROOT / "eval" / "stride-sweep.json")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    cfg = corpora.registry()[args.corpus]
    answerable, adversarial = load_cases(Path(cfg["golden"]))
    shipped_k = cfg["candidate_k"]

    base = score(cfg["store"], cfg, shipped_k, answerable, adversarial,
                 model, "shipped stride")
    fine = score(Path(args.store), cfg, shipped_k, answerable, adversarial,
                 model, "finer stride")
    # The pool covers proportionally less of a corpus that has more chunks in
    # it, so a like-for-like comparison also needs the pool scaled by the same
    # ratio. Reporting only the first row would credit or blame the stride for
    # something the pool did.
    ratio = fine["chunks"] / max(base["chunks"], 1)
    scaled = score(Path(args.store), cfg, max(1, round(shipped_k * ratio)),
                   answerable, adversarial, model, "finer, pool scaled")

    rows = [base, fine, scaled]
    print(f"\n  {cfg['label']}  ({len(answerable)} answerable, "
          f"{len(adversarial)} adversarial)")
    print(f"    {'arm':<20}{'chunks':>8}{'pool':>6}{'any-hit':>9}{'MRR':>8}"
          f"{'NDCG':>8}{'src':>7}{'answer shown':>14}{'slips':>7}{'sec/q':>8}")
    for r in rows:
        print(f"    {r['label']:<20}{r['chunks']:>8}{r['candidate_k']:>6}"
              f"{r['any_hit']:>9.3f}{r['mrr']:>8.3f}{r['ndcg']:>8.3f}"
              f"{r['src_recall']:>7.3f}{r['answer_visible']:>14.3f}"
              f"{r['slips']:>7}{r['seconds_per_query']:>8.2f}")
    ref = set(base.pop("found"))
    for r in rows[1:]:
        got = set(r.pop("found"))
        gained, lost = sorted(got - ref), sorted(ref - got)
        r["gained"], r["lost"] = gained, lost
        if gained:
            print(f"      {r['label']} gained: {', '.join(gained)}")
        if lost:
            print(f"      {r['label']} lost:   {', '.join(lost)}")

    args.emit.write_text(json.dumps(
        {"generated_by": "src/sweep_stride.py", "corpus": args.corpus,
         "arms": rows}, indent=1) + "\n", encoding="utf-8")
    print(f"\n  wrote {args.emit.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
