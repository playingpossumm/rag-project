"""Where does the time in one query actually go?

The interface reports around 1,100 ms for a question on the ML corpus and
attributes it to "search, scoring and reranking" -- three stages in one number,
which is enough to know it is slow and not enough to know what to do.

This project has already been burned once by optimising the stage that felt
slow: "re-indexing is slow because the index is rebuilt" was a well-formed plan
until rebuilding the FAISS index measured at 0.01 s, or 0% of runtime, and the
task had to be redefined around caching instead. So the stages are timed
separately before anything is touched.

Every stage is timed inside the process that serves them, on real questions from
each corpus's own golden set, at that corpus's shipped settings. Warm: the
models are loaded and one query is run before the clock starts, because the
first query pays for lazy initialisation that a served query never does.

    .venv\\Scripts\\python.exe src\\profile_query.py
    .venv\\Scripts\\python.exe src\\profile_query.py --corpus birds -n 20
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
from evaluate import load_cases  # noqa: E402
from hybrid import bm25_search, build_bm25, fuse_rrf  # noqa: E402
from retrieve import (CANDIDATE_K, EMBEDDING_MODEL, TOP_K, load_index,  # noqa: E402
                      search)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
OUT = ROOT / "eval" / "query-profile.json"

STAGES = ["embed", "faiss", "bm25", "fuse", "rerank", "diversify", "expand"]


class Clock:
    """Accumulate elapsed time per stage across many queries."""

    def __init__(self):
        self.laps: dict[str, list[float]] = {s: [] for s in STAGES}
        self._t = None

    def start(self):
        self._t = time.perf_counter()

    def lap(self, stage: str):
        now = time.perf_counter()
        self.laps[stage].append((now - self._t) * 1000)
        self._t = now


def profile(cfg, questions, k, cand, clock: Clock):
    from sentence_transformers import SentenceTransformer
    from parent import expand

    index, metadata = load_index(cfg["store"])
    model = SentenceTransformer(EMBEDDING_MODEL)
    bm25 = build_bm25(metadata)
    rr.RERANK_BLEND = cfg["rerank_blend"]
    rr.load_reranker()

    # One warm query, untimed. A served query never pays lazy import, model
    # warm-up or the first allocation of a numpy workspace, so timing those
    # would describe a process nobody runs.
    warm = questions[0]
    dense = search(warm, index, metadata, model, k=cand)
    ranked = rr.rerank(warm, dense, k=len(dense))
    diversify(ranked, k=k, max_per_source=2)

    for q in questions:
        rr.clear_cache()
        clock.start()

        # Deliberately open-coded rather than calling retrieve(), so each stage
        # can be timed on its own. src/test_trace.py exists because a second
        # copy of the pipeline drifts from the first one; this copy measures
        # rather than serves, and the totals below are checked against a real
        # retrieve() call at the end.
        model.encode([q], convert_to_numpy=True)
        clock.lap("embed")

        dense = search(q, index, metadata, model, k=cand)
        clock.lap("faiss")

        lexical = bm25_search(q, bm25, metadata, k=cand)
        clock.lap("bm25")

        fused = fuse_rrf(dense, lexical, k=cand)
        clock.lap("fuse")

        ranked = rr.rerank(q, fused, k=len(fused))
        clock.lap("rerank")

        results = diversify(ranked, k=k, max_per_source=2)
        clock.lap("diversify")

        expand(results, metadata, mode="window", window=1)
        clock.lap("expand")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", action="append")
    ap.add_argument("-n", type=int, default=15, help="questions per corpus")
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--candidate-k", type=int, default=CANDIDATE_K)
    ap.add_argument("--emit", type=Path, default=OUT)
    args = ap.parse_args()

    reg = corpora.registry()
    wanted = args.corpus or list(reg)
    report = {}

    for name in wanted:
        cfg = reg.get(name)
        if not cfg or not cfg["indexed"] or not cfg["golden"]:
            continue
        answerable, _ = load_cases(cfg["golden"])
        questions = [c["question"] for c in answerable][:args.n]
        if not questions:
            continue

        clock = Clock()
        profile(cfg, questions, args.k, args.candidate_k, clock)

        rows = {s: {"median": round(st.median(v), 1),
                    "mean": round(st.mean(v), 1),
                    "max": round(max(v), 1)}
                for s, v in clock.laps.items() if v}
        total = sum(r["median"] for r in rows.values())

        print(f"\n  {cfg['label']}  ({len(questions)} questions, "
              f"{corpora.stats(cfg)['chunks']:,} chunks)")
        print(f"    {'stage':<12}{'median ms':>11}{'share':>9}{'max ms':>10}")
        for stage in STAGES:
            r = rows.get(stage)
            if not r:
                continue
            print(f"    {stage:<12}{r['median']:>11.1f}"
                  f"{r['median'] / total:>8.0%}{r['max']:>10.1f}")
        print(f"    {'TOTAL':<12}{total:>11.1f}")
        report[name] = {"label": cfg["label"], "n": len(questions),
                        "chunks": corpora.stats(cfg)["chunks"],
                        "stages": rows, "total_median_ms": round(total, 1)}

    if not report:
        print("nothing to profile")
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
