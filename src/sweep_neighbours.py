"""Can the pipeline reach the answer that sits one chunk away?

`src/answer_distance.py` found that 13 of the 23 questions whose answer is not
in any returned chunk have it in a chunk NEXT DOOR, in a document that was
retrieved. The question's words are in chunk N and the answer is in N+1, so the
first stage ranks N and the reader never sees N+1. Two arrangements aim at that
directly, and neither needs the corpus rebuilt:

    neighbours   pull each candidate's immediate neighbours into the pool
                 before reranking, so the cross-encoder can score N+1 against
                 the question and promote it on its own merits.
    window       score each candidate on N-1 + N + N+1 while still RETURNING
                 N, so a chunk is ranked by the passage it sits in.

Both change ranking, unlike context expansion, which is why every column is
reported and not only the one being aimed at. `sweep_hyde.py` records what
happens when a change is judged on the metric it was built to move: retrieving
on a hypothetical answer recovers three of the five permanent failures and
costs nine other questions.

The adversarial half matters as much. A wider pool and a wider scoring context
both give the gate more chances to find something that looks answerable in a
corpus that cannot answer, and that cost appears in no hit-rate column.

    .venv\\Scripts\\python.exe src\\sweep_neighbours.py
    .venv\\Scripts\\python.exe src\\sweep_neighbours.py --corpus birds
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import corpora  # noqa: E402
import rerank as rr  # noqa: E402
from diversify import diversify  # noqa: E402
from evaluate import (answer_normalize, gold_keys, gold_sources,  # noqa: E402
                      hit_rate, is_relevant, load_cases, ndcg,
                      reciprocal_rank, source_recall)
from hybrid import build_bm25  # noqa: E402
from retrieve import (DEFAULT_FUSION, EMBEDDING_MODEL, TOP_K,  # noqa: E402
                      load_ensemble, load_index, shortlist)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
ARMS = ("control", "neighbours", "window")


def neighbours_of(candidates, metadata):
    """Each candidate's immediate neighbours, in document order, deduplicated.

    Appended after the originals rather than interleaved, so a neighbour starts
    with a worse first-stage rank than anything the first stage actually chose.
    That matters on the corpus that ships a non-zero rerank blend, where the
    first-stage ordering still carries weight.
    """
    seen = {c["chunk_id"] for c in candidates}
    extra = []
    for c in candidates:
        for step in (-1, 1):
            j = c["chunk_id"] + step
            if j < 0 or j >= len(metadata) or j in seen:
                continue
            # Consecutive ids cross document boundaries; a neighbour in another
            # document is not a neighbour.
            if metadata[j]["source"] != c["source"]:
                continue
            seen.add(j)
            extra.append({**metadata[j], "chunk_id": j, "score": 0.0})
    return extra


def windowed_text(chunk_id, metadata, source):
    parts = []
    for j in (chunk_id - 1, chunk_id, chunk_id + 1):
        if 0 <= j < len(metadata) and metadata[j]["source"] == source:
            parts.append(metadata[j]["text"])
    return " ".join(parts)


def run_arm(arm, question, index, metadata, model, cfg, bm25, ensemble):
    from rerank import rerank

    pool = shortlist(question, index, metadata, model, k=cfg["candidate_k"],
                     fusion=DEFAULT_FUSION, bm25=bm25, ensemble=ensemble)
    if arm == "neighbours":
        pool = pool + neighbours_of(pool, metadata)

    if arm == "window":
        # Scored on the passage it sits in, returned as itself. The scores are
        # taken from a parallel list whose text is the window, then attached to
        # the real candidates, so nothing downstream sees the widened text.
        widened = [{**c, "text": windowed_text(c["chunk_id"], metadata,
                                               c["source"])} for c in pool]
        scored = rerank(question, widened, k=len(widened),
                        blend=cfg["rerank_blend"])
        order = {c["chunk_id"]: i for i, c in enumerate(scored)}
        by_id = {c["chunk_id"]: c for c in pool}
        ranked = [{**by_id[c["chunk_id"]],
                   "rerank_score": c["rerank_score"],
                   "retrieval_score": c.get("retrieval_score")}
                  for c in scored if c["chunk_id"] in by_id]
    else:
        ranked = rerank(question, pool, k=len(pool), blend=cfg["rerank_blend"])

    return diversify(ranked, k=TOP_K, max_per_source=2)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--emit", type=Path,
                    default=ROOT / "eval" / "neighbour-sweep.json")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    reg = corpora.registry()
    names = ([args.corpus] if args.corpus
             else [n for n, c in reg.items() if c["indexed"] and c["golden"]])
    report = {"generated_by": "src/sweep_neighbours.py", "corpora": {}}

    for name in names:
        cfg = reg[name]
        index, metadata = load_index(cfg["store"])
        bm25 = build_bm25(metadata)
        ensemble = load_ensemble(cfg["store"])
        answerable, adversarial = load_cases(Path(cfg["golden"]))
        threshold = cfg["threshold"]

        rows = {}
        for arm in ARMS:
            totals = {"hit": 0.0, "mrr": 0.0, "ndcg": 0.0, "src": 0.0}
            visible, visible_n, slips, seconds = 0, 0, [], 0.0
            found = set()
            for case in answerable:
                rr.clear_cache()
                t0 = time.perf_counter()
                res = run_arm(arm, case["question"], index, metadata, model,
                              cfg, bm25, ensemble)
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
                res = run_arm(arm, case["question"], index, metadata, model,
                              cfg, bm25, ensemble)
                if res and res[0]["rerank_score"] >= threshold:
                    slips.append(case["id"])

            n = len(answerable) or 1
            rows[arm] = {
                "any_hit": round(totals["hit"] / n, 4),
                "mrr": round(totals["mrr"] / n, 4),
                "ndcg": round(totals["ndcg"] / n, 4),
                "src_recall": round(totals["src"] / n, 4),
                "answer_visible": round(visible / max(visible_n, 1), 4),
                "slips": len(slips),
                "seconds_per_query": round(seconds / n, 2),
                "_found": sorted(found),
            }

        base = set(rows["control"]["_found"])
        print(f"\n  {cfg['label']}  ({len(answerable)} answerable, "
              f"{len(adversarial)} adversarial)")
        print(f"    {'arm':<12}{'any-hit':>9}{'MRR':>8}{'NDCG':>8}{'src':>7}"
              f"{'answer shown':>14}{'slips':>7}{'sec/q':>8}")
        for arm in ARMS:
            r = rows[arm]
            print(f"    {arm:<12}{r['any_hit']:>9.3f}{r['mrr']:>8.3f}"
                  f"{r['ndcg']:>8.3f}{r['src_recall']:>7.3f}"
                  f"{r['answer_visible']:>14.3f}{r['slips']:>7}"
                  f"{r['seconds_per_query']:>8.2f}")
            got = set(r.pop("_found"))
            if arm != "control":
                gained, lost = sorted(got - base), sorted(base - got)
                r["gained"], r["lost"] = gained, lost
                if gained:
                    print(f"        gained: {', '.join(gained)}")
                if lost:
                    print(f"        lost:   {', '.join(lost)}")
        report["corpora"][name] = {"label": cfg["label"], "arms": rows}

    args.emit.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"\n  wrote {args.emit.relative_to(ROOT)}")
    print("  Read the gained and lost lists, not the answer-shown column "
          "alone. A change\n  that gains four and loses four is not neutral, "
          "it is a different pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
