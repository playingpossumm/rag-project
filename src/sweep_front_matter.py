"""The cost of dropping title blocks from the candidate pool.

`src/front_matter.py` describes the defect it was written for, which is that a
paper's title block outscores the paragraph that answers, because a title is the
densest statement of what a document is about and a cross-encoder reads that
density as relevance. 56 chunks across the three corpora match, about one per
PDF.

That makes it look free, and this project has been wrong about free before.
Four separate experiments here improved what the first stage retrieves and made
the finished answer worse, so the filter is measured end to end at the served
configuration, on the whole answerable set, before its default changes.

It was measured and it lost, scoring 0.851 to 0.836 any-hit on the ML papers and
0.886 to 0.857 on quant.

**Read that loss with `src/audit_title_credit.py` before believing it.** Two
questions account for all of it, and both are answered by the chunk the filter
removes, because a chunk that opens with a title block continues into the
abstract. The aggregate this script prints is the right number, and on its own
it does not say whether the questions behind it were lost fairly.

Also reported: how many questions had a title block in their top five at all.
A change that fixes a visible defect on three questions and moves no metric is
still worth shipping, and the only way to say so honestly is to count the
questions it touches rather than to read the aggregate and guess.

    .venv\\Scripts\\python.exe src\\sweep_front_matter.py
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import corpora  # noqa: E402
import rerank as rr  # noqa: E402
from evaluate import (gold_keys, gold_sources, hit_rate, load_cases,  # noqa: E402
                      ndcg, reciprocal_rank, source_recall)
from front_matter import is_front_matter  # noqa: E402
from hybrid import build_bm25  # noqa: E402
from retrieve import (DEFAULT_FUSION, EMBEDDING_MODEL, TOP_K,  # noqa: E402
                      load_ensemble, load_index, retrieve)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
OUT = ROOT / "eval" / "front-matter-sweep.json"


def score(cases, index, metadata, model, bm25, ensemble, cfg, drop):
    totals = {"hit_rate": 0.0, "mrr": 0.0, "ndcg": 0.0, "src_recall": 0.0}
    per_case, touched = {}, []
    for case in cases:
        results = retrieve(case["question"], index, metadata, model, k=TOP_K,
                           candidate_k=cfg["candidate_k"], use_reranker=True,
                           fusion=DEFAULT_FUSION, bm25=bm25, max_per_source=2,
                           ensemble=ensemble, drop_front_matter=drop,
                           rerank_blend=cfg["rerank_blend"])
        gold = gold_keys(case)
        m = {"hit_rate": hit_rate(results, gold),
             "mrr": reciprocal_rank(results, gold),
             "ndcg": ndcg(results, gold),
             "src_recall": source_recall(results, gold_sources(case), TOP_K)}
        for key in totals:
            totals[key] += m[key]
        per_case[case["id"]] = m["hit_rate"] == 1.0
        if any(is_front_matter(r.get("text", ""), r.get("locator"))
               for r in results):
            touched.append(case["id"])
    n = max(len(cases), 1)
    return ({k: round(v / n, 4) for k, v in totals.items()}, per_case, touched)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", action="append")
    ap.add_argument("--emit", type=Path, default=OUT)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)

    reg = corpora.registry()
    report = {}
    for name in (args.corpus or list(reg)):
        cfg = reg.get(name)
        if not cfg or not cfg["indexed"] or not cfg["golden"]:
            continue
        answerable, _ = load_cases(cfg["golden"])
        index, metadata = load_index(cfg["store"])
        bm25 = build_bm25(metadata)
        ensemble = load_ensemble(cfg["store"])

        n_fm = sum(1 for c in metadata
                   if is_front_matter(c.get("text", ""), c.get("locator")))
        print(f"\n  {cfg['label']}  ({len(answerable)} answerable, "
              f"{n_fm} title-block chunks of {len(metadata)})")
        print(f"    {'filter':<8}{'any-hit':>9}{'MRR':>8}{'NDCG':>8}{'src':>8}"
              f"{'questions showing a title block':>34}")

        rows, pers = {}, {}
        for drop in (False, True):
            rr.clear_cache()
            agg, per, touched = score(answerable, index, metadata, model, bm25,
                                      ensemble, cfg, drop)
            label = "on" if drop else "off"
            rows[label] = {**agg, "showing_front_matter": touched}
            pers[label] = per
            print(f"    {label:<8}{agg['hit_rate']:>9.3f}{agg['mrr']:>8.3f}"
                  f"{agg['ndcg']:>8.3f}{agg['src_recall']:>8.3f}"
                  f"{len(touched):>34}")

        gained = sorted(i for i in pers["on"] if pers["on"][i] and not pers["off"][i])
        lost = sorted(i for i in pers["on"] if not pers["on"][i] and pers["off"][i])
        print(f"    gained {gained or 'nothing'}, lost {lost or 'nothing'}")
        print(f"    questions no longer showing a title block: "
              f"{sorted(set(rows['off']['showing_front_matter']) - set(rows['on']['showing_front_matter']))}")
        report[name] = {"label": cfg["label"], "front_matter_chunks": n_fm,
                        "chunks": len(metadata), "modes": rows,
                        "gained": gained, "lost": lost}

    if not report:
        print("nothing measured")
        return 1
    args.emit.write_text(json.dumps(
        {"generated_by": "src/sweep_front_matter.py",
         "note": "End to end at the served configuration, on every answerable "
                 "case. `showing_front_matter` counts questions whose top five "
                 "contained a title block, which is the defect this filter "
                 "exists for and is not visible in the aggregate metrics.",
         "corpora": report}, indent=1) + "\n", encoding="utf-8")
    print(f"\n  wrote {args.emit.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
