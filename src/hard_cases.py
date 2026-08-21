"""Run only the cases retrieval cannot currently get right. Seconds, not minutes.

`evaluate.py` scores eleven configurations over 84 cases and takes minutes.
That is the right tool for "did this change the numbers I publish" and the wrong
one for "did this change anything at all", which is the question you ask twenty
times while trying an idea.

This runs the fixture `failure_overlap.py` derives: the answerable cases that
fail under *every* pipeline configuration, plus the adversarial ones the gate
lets through under every gated configuration. Those are the cases no amount of
fusion, reranking or capping reaches -- so if an idea does not move one of them,
it has not addressed the problem, and that verdict is available in seconds.

**It is a tripwire, not a benchmark.** Seven cases cannot tell you a change is
good; the golden set does that. What it can tell you, cheaply and immediately,
is that a change is not doing what you hoped -- which is the answer most ideas
deserve, and the one worth getting fast.

    .venv\\Scripts\\python.exe src\\failure_overlap.py runs\\*.json --emit-fixture eval\\hard_cases.json
    .venv\\Scripts\\python.exe src\\hard_cases.py
    .venv\\Scripts\\python.exe src\\hard_cases.py --no-rerank
"""
import argparse
import json
import sys
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer

from abstain import ABSTAIN_THRESHOLD
from evaluate import context_recall, gold_keys, gold_sources, is_relevant, load_cases
from hybrid import build_bm25
from retrieve import CANDIDATE_K, DEFAULT_FUSION, EMBEDDING_MODEL, TOP_K, load_index, retrieve

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FIXTURE = Path(__file__).parent.parent / "eval" / "hard_cases.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", type=Path, default=FIXTURE)
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--candidate-k", type=int, default=CANDIDATE_K)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--fusion", default=DEFAULT_FUSION,
                    choices=["none", "rrf", "weighted"])
    ap.add_argument("--max-per-source", type=int, default=2)
    args = ap.parse_args()

    if not args.fixture.exists():
        print(f"no fixture at {args.fixture} -- generate it with:\n"
              f"  src/failure_overlap.py runs/*.json --emit-fixture {args.fixture}")
        return 2

    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    want = {c["id"] for c in fixture["structural"]}
    slips = {c["id"] for c in fixture["slips_the_gate"]}

    answerable, adversarial = load_cases()
    cases = {c["id"]: c for c in answerable + adversarial}
    missing = (want | slips) - set(cases)
    if missing:
        print(f"fixture references cases not in the golden set: {sorted(missing)}")
        return 2

    opts = dict(use_reranker=not args.no_rerank, fusion=args.fusion,
                max_per_source=args.max_per_source or None)

    t0 = time.perf_counter()
    index, metadata = load_index()
    model = SentenceTransformer(EMBEDDING_MODEL)
    bm25 = build_bm25(metadata)
    load_s = time.perf_counter() - t0

    print(f"{len(want)} structural + {len(slips)} gate-slips, {opts}\n")

    t1 = time.perf_counter()
    fixed, still = [], []
    for cid in sorted(want):
        case = cases[cid]
        results = retrieve(case["question"], index, metadata, model, k=args.k,
                           candidate_k=args.candidate_k, bm25=bm25, **opts)
        gold = gold_keys(case)
        hit = any(is_relevant(r, gold) for r in results)
        # Context recall is reported beside the hit because the two disagreeing
        # is the interesting case: text containing the answer that the page
        # labels do not credit, or a labelled page whose text does not carry it.
        ctx = (context_recall(results, case["answer_contains"])
               if case.get("answer_contains") else float("nan"))
        (fixed if hit else still).append(cid)
        src = source_recall_str(results, gold_sources(case), args.k)
        print(f"  {'FIXED' if hit else '  .  '}  {cid:<18}"
              f"hit {int(hit)}  ctx {ctx:.0f}  src {src}  "
              f"-> {', '.join(r['source'][:18] for r in results[:3])}")

    print()
    for cid in sorted(slips):
        case = cases[cid]
        results = retrieve(case["question"], index, metadata, model, k=args.k,
                           candidate_k=args.candidate_k, bm25=bm25, **opts)
        top = results[0].get("rerank_score") if results and not args.no_rerank else None
        if top is None:
            print(f"  {'  ?  '}  {cid:<18}no calibrated score on this path")
            continue
        refused = top < ABSTAIN_THRESHOLD
        print(f"  {'REFUSED' if refused else '  .  '}  {cid:<18}"
              f"top {top:+6.2f} against {ABSTAIN_THRESHOLD:+.1f}")

    run_s = time.perf_counter() - t1
    print(f"\n{len(fixed)} of {len(want)} structural cases now hit"
          f"{': ' + ', '.join(fixed) if fixed else ''}")
    print(f"{load_s:.1f}s loading, {run_s:.1f}s running "
          f"({run_s / max(len(want) + len(slips), 1):.2f}s per case)")
    return 0


def source_recall_str(results, sources, k) -> str:
    if not sources:
        return "  -"
    found = {r["source"] for r in results} & sources
    return f"{len(found)}/{min(len(sources), k)}"


if __name__ == "__main__":
    raise SystemExit(main())
