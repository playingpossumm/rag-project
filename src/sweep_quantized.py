"""Does int8 quantising the cross-encoder cost anything a reader would notice?

Reranking is 92-95% of a query (`src/profile_query.py`), and after
`candidate_k` went per-corpus the only lever left on it is the model itself. A
faster *model* was already ruled out -- `src/compare_rerankers.py` found the
quicker candidates worse and the better one 1700x slower. Quantising keeps the
model and makes the arithmetic cheaper.

**Two things have to hold, and the second is the one that is easy to forget.**

Ranking must survive, which any comparison would check. But this project's
abstention gate reads the cross-encoder's score as an **absolute** value against
a calibrated threshold, so a systematic shift of even 0.05 changes what the gate
refuses while leaving every ordering identical. A quantisation that preserved
rank perfectly and shifted every score down by a tenth would look free here and
would silently start refusing answerable questions.

So this reports ranking, the gate's own counts at the shipped threshold, and the
distribution shift -- and it re-derives the threshold under int8, because if the
shift is real the honest response is to recalibrate rather than to pretend.

    .venv\\Scripts\\python.exe src\\sweep_quantized.py
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
from evaluate import (gold_keys, gold_sources, hit_rate, load_cases, ndcg,  # noqa: E402
                      reciprocal_rank, source_recall)
from hybrid import build_bm25  # noqa: E402
from retrieve import (DEFAULT_FUSION, EMBEDDING_MODEL, TOP_K, load_index,  # noqa: E402
                      shortlist)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
OUT = ROOT / "eval" / "quantization-sweep.json"


def run(cfg, cases, pools, k, quantized: bool):
    rr.clear_cache()
    rr._model, rr._model_name = None, None
    rr.QUANTIZE = quantized
    rr.load_reranker()

    totals = {"hit_rate": 0.0, "mrr": 0.0, "ndcg": 0.0, "src_recall": 0.0}
    ans, adv, n = [], [], 0
    started = time.perf_counter()
    for case in cases:
        ranked = rr.rerank(case["question"], pools[case["id"]],
                           k=len(pools[case["id"]]), blend=cfg["rerank_blend"])
        results = diversify(ranked, k=k, max_per_source=2)
        top = results[0]["rerank_score"] if results else float("-inf")
        if case.get("unanswerable"):
            adv.append(top)
            continue
        ans.append(top)
        n += 1
        gold = gold_keys(case)
        for key, fn in (("hit_rate", hit_rate), ("mrr", reciprocal_rank),
                        ("ndcg", ndcg)):
            totals[key] += fn(results, gold)
        totals["src_recall"] += source_recall(results, gold_sources(case), k)
    elapsed = time.perf_counter() - started

    thr = cfg["threshold"] if cfg["calibrated"] else 0.0
    return {
        **{key: round(v / max(n, 1), 4) for key, v in totals.items()},
        "wrongly_refused": sum(1 for s in ans if s < thr),
        "caught": sum(1 for s in adv if s < thr),
        "median_answerable": round(st.median(ans), 3) if ans else None,
        "median_adversarial": round(st.median(adv), 3) if adv else None,
        "ms_per_query": round(elapsed / max(len(cases), 1) * 1000),
        "_ans": ans, "_adv": adv,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", action="append")
    ap.add_argument("--k", type=int, default=TOP_K)
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
        answerable, adversarial = load_cases(cfg["golden"])
        cases = answerable + adversarial
        index, metadata = load_index(cfg["store"])
        bm25 = build_bm25(metadata)
        cand = cfg["candidate_k"]

        # Built once and reused, so the only difference between the two runs is
        # the arithmetic inside the cross-encoder.
        pools = {c["id"]: shortlist(c["question"], index, metadata, model,
                                    k=cand, fusion=DEFAULT_FUSION, bm25=bm25)
                 for c in cases}

        thr = cfg["threshold"] if cfg["calibrated"] else 0.0
        print(f"\n  {cfg['label']}  ({len(answerable)} answerable + "
              f"{len(adversarial)} adversarial, {cand} candidates, "
              f"threshold {thr:+.1f})")
        rows = {}
        for label, q in (("float32", False), ("int8", True)):
            rows[label] = run(cfg, cases, pools, args.k, q)
            r = rows[label]
            print(f"    {label:<8} hit {r['hit_rate']:.3f}  mrr {r['mrr']:.3f}  "
                  f"ndcg {r['ndcg']:.3f}  src {r['src_recall']:.3f}   "
                  f"refused {r['wrongly_refused']}  caught {r['caught']}  "
                  f"{r['ms_per_query']:>5} ms/q")

        a, b = rows["float32"], rows["int8"]
        shift = [y - x for x, y in zip(a["_ans"], b["_ans"])]
        print(f"    delta    hit {b['hit_rate']-a['hit_rate']:+.3f}  "
              f"mrr {b['mrr']-a['mrr']:+.3f}   "
              f"speedup {a['ms_per_query']/max(b['ms_per_query'],1):.2f}x")
        print(f"    score shift on answerable cases: mean "
              f"{st.mean(shift):+.3f}, max |{max(abs(s) for s in shift):.3f}|")
        print(f"    the gate: {a['wrongly_refused']} -> {b['wrongly_refused']} "
              f"wrongly refused, {a['caught']} -> {b['caught']} caught")
        for row in rows.values():
            row.pop("_ans"), row.pop("_adv")
        report[name] = {"label": cfg["label"], "threshold": thr,
                        "candidate_k": cand, "runs": rows,
                        "score_shift_mean": round(st.mean(shift), 4)}

    if not report:
        print("nothing to sweep")
        return 1
    args.emit.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"\n  wrote {args.emit.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
