"""Two embedders, fused. Does the union beat either one alone?

`src/compare_embedders.py` found that no single first stage dominates: on birds
`multi-qa-MiniLM-L6-cos-v1` ties the shipped `all-MiniLM-L6-v2` at 0.962 pool
recall while finding a **different set** -- it wins `bird-alula` and loses
`bird-precocial`. On quant `bge-small-en-v1.5` scores lower overall and still
wins `qf-cvar-interval`.

Two retrievers with the same score and different mistakes is exactly the
condition under which fusing helps, and it is the same argument that already
justifies fusing dense with BM25 in this pipeline. So this measures it.

**It has a specific way to fail, worth stating before the run.** Dense and BM25
fuse well because they are *unlike* -- one scores meaning, the other exact
terms. Two dense models trained on overlapping data are far more correlated, so
the union may be mostly the intersection, and RRF's damping can push a passage
that one model ranked first down below one that both ranked seventh. A fused
recall that lands between the two inputs rather than above them is the result
that means "correlated", and it is a real possibility rather than a hedge.

Pool recall only, and no re-index: each model embeds the corpus in memory once,
the two rankings are fused per question, and what is measured is whether the
answer reaches the reranker at all -- the ceiling everything downstream works
under.

    .venv\\Scripts\\python.exe src\\sweep_ensemble.py --corpus birds
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402

import corpora  # noqa: E402
from evaluate import gold_keys, is_relevant, load_cases  # noqa: E402
from hybrid import RRF_K  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
OUT = ROOT / "eval" / "ensemble-sweep.json"

SHIPPED = {"name": "sentence-transformers/all-MiniLM-L6-v2", "prefix": ""}
PARTNERS = [
    {"name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1", "prefix": ""},
    {"name": "BAAI/bge-small-en-v1.5",
     "prefix": "Represent this sentence for searching relevant passages: "},
]


def fixture_ids(cfg) -> set:
    from check_freshness import artefact_suffix
    p = ROOT / "eval" / f"hard_cases{artefact_suffix(cfg['golden'])}.json"
    if not p.exists():
        return set()
    d = json.loads(p.read_text(encoding="utf-8"))
    return {c["id"] if isinstance(c, dict) else c for c in d.get("structural", [])}


def rankings(spec, texts, questions):
    """Every chunk index, best first, for each question."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(spec["name"])
    docs = model.encode(texts, convert_to_numpy=True, batch_size=64,
                        show_progress_bar=False, normalize_embeddings=True)
    qs = model.encode([spec["prefix"] + q for q in questions],
                      convert_to_numpy=True, normalize_embeddings=True)
    return [np.argsort(-(docs @ qv)) for qv in qs]


def rrf(order_a, order_b, k, depth):
    """Reciprocal-rank fusion of two orderings, best first.

    Only the top `depth` of each is considered. Fusing full corpus-length
    rankings would let a passage that neither model ranked anywhere near the top
    accumulate a score from two mediocre positions, which is not what either
    retriever was claiming.
    """
    scores = {}
    for order in (order_a, order_b):
        for rank, idx in enumerate(order[:depth], start=1):
            scores[int(idx)] = scores.get(int(idx), 0.0) + 1.0 / (RRF_K + rank)
    return [i for i, _ in sorted(scores.items(), key=lambda kv: -kv[1])][:k]


def recall(pools, cases, metadata):
    hits, per_case = 0, {}
    for case, pool in zip(cases, pools):
        gold = gold_keys(case)
        found = any(is_relevant(metadata[i], gold) for i in pool)
        hits += found
        per_case[case["id"]] = bool(found)
    return round(hits / max(len(cases), 1), 4), per_case


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", action="append")
    ap.add_argument("--depth", type=int, default=None,
                    help="how deep into each ranking to fuse; default 3x the pool")
    ap.add_argument("--emit", type=Path, default=OUT)
    args = ap.parse_args()

    reg = corpora.registry()
    report = {}
    for name in (args.corpus or list(reg)):
        cfg = reg.get(name)
        if not cfg or not cfg["indexed"] or not cfg["golden"]:
            continue
        answerable, _ = load_cases(cfg["golden"])
        metadata = json.loads(
            (cfg["store"] / "metadata.json").read_text(encoding="utf-8"))
        texts = [c.get("embed_text", c["text"]) for c in metadata]
        questions = [c["question"] for c in answerable]
        k = cfg["candidate_k"]
        depth = args.depth or k * 3
        structural = fixture_ids(cfg)

        print(f"\n  {cfg['label']}  ({len(answerable)} answerable, "
              f"{len(metadata):,} chunks, pool {k}, fusing top {depth})")

        base_orders = rankings(SHIPPED, texts, questions)
        base_r, base_cases = recall([o[:k] for o in base_orders], answerable, metadata)
        got = sum(1 for i in structural if base_cases.get(i))
        print(f"    {'all-MiniLM-L6-v2 (shipped)':<42}{base_r:>8.3f}"
              f"{f'{got}/{len(structural)}':>12}")

        rows = {"shipped": {"pool_recall": base_r, "cases": base_cases}}
        for spec in PARTNERS:
            short = spec["name"].split("/")[-1]
            try:
                other_orders = rankings(spec, texts, questions)
            except Exception as exc:                            # noqa: BLE001
                print(f"    {short:<42}  unavailable: {exc}")
                continue
            other_r, other_cases = recall([o[:k] for o in other_orders],
                                          answerable, metadata)
            fused = [rrf(a, b, k, depth) for a, b in zip(base_orders, other_orders)]
            fused_r, fused_cases = recall(fused, answerable, metadata)

            g_o = sum(1 for i in structural if other_cases.get(i))
            g_f = sum(1 for i in structural if fused_cases.get(i))
            print(f"    {short:<42}{other_r:>8.3f}{f'{g_o}/{len(structural)}':>12}")
            print(f"    {'  fused with the shipped model':<42}{fused_r:>8.3f}"
                  f"{f'{g_f}/{len(structural)}':>12}"
                  f"   {fused_r - base_r:+.3f} vs shipped")

            won = sorted(i for i, ok in fused_cases.items() if ok and not base_cases.get(i))
            lost = sorted(i for i, ok in base_cases.items() if ok and not fused_cases.get(i))
            print(f"        fusion wins {won or '[]'}  loses {lost or '[]'}")
            rows[short] = {"pool_recall": other_r, "cases": other_cases}
            rows[f"fused:{short}"] = {"pool_recall": fused_r, "cases": fused_cases,
                                      "won": won, "lost": lost}

        report[name] = {"label": cfg["label"], "pool": k, "depth": depth,
                        "structural": sorted(structural), "models": rows}

    if not report:
        print("nothing to sweep")
        return 1
    args.emit.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"\n  wrote {args.emit.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
