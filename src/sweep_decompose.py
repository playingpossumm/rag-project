"""Does rewriting or decomposing the question reach the structural cases?

**The claim this tests.** Seven ML cases fail under every combination of
fusion, reranking and diversity cap. `docs/roadmap.md` has said for a while
that query decomposition is the right fix for them, on an argument rather than
a measurement: the question names a topic and a constraint, retrieval matches
the topic, and the constraint that picks the right document is averaged away.
Splitting the question is what an LLM does well.

That argument has never been scored. It could not be while it needed credit;
`query_rewrite.py` now speaks to a local Ollama, so it can be.

**Three modes, at the served configuration.**

    none        the question as asked, which is what ships
    rewrite     one question in, one clearer question out
    decompose   one question in, several out, retrieved separately and fused
                by RRF

**What is measured, and why it is not pool recall.** End to end, through
reranking and the diversity cap, on the whole answerable set rather than on the
seven alone. Two reasons, both learned here the expensive way. Three separate
experiments in this project improved what the first stage retrieves and made
the finished pipeline worse, so a candidate pool is a ceiling and not a proxy.
And a change measured only on the cases it was built for cannot show what it
costs the cases it was not: the seven are 10% of this corpus, and a rewrite
that recovers two of them while losing four elsewhere is a bad trade that looks
like a good one.

**Cost is reported too.** Every mode calls a model once per question before
retrieval can start, which is a second or more on a local CPU against about
1,100ms for the whole query today. A recovered question that triples latency is
a finding, not a fix, and the number belongs beside the quality figure rather
than in a footnote.

    ollama serve
    set RAG_GENERATOR=ollama
    .venv\\Scripts\\python.exe src\\sweep_decompose.py --corpus llm
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import corpora  # noqa: E402
import rerank as rr  # noqa: E402
from evaluate import (gold_keys, gold_sources, hit_rate, load_cases,  # noqa: E402
                      ndcg, reciprocal_rank, source_recall)
from hybrid import build_bm25  # noqa: E402
from retrieve import (CANDIDATE_K, DEFAULT_FUSION, EMBEDDING_MODEL,  # noqa: E402
                      TOP_K, load_index, retrieve)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
OUT = ROOT / "eval" / "decompose-sweep.json"
MODES = ("none", "rewrite", "decompose")


def run_one(question, mode, index, metadata, model, bm25, k, cand, blend):
    """One question under one mode, returning the results and what it cost.

    `blend` is the corpus's rerank blend, passed on every call rather than
    set on the rerank module (which this file did until 2026-09-06), so the
    setting cannot outlive the corpus it belongs to. `retrieve_decomposed`
    forwards it to `retrieve` with the other keywords.
    """
    import query_rewrite as qr

    started = time.perf_counter()
    if mode == "rewrite":
        asked = qr.rewrite(question)
        results = retrieve(asked, index, metadata, model, k=k,
                           candidate_k=cand, use_reranker=True,
                           fusion=DEFAULT_FUSION, bm25=bm25, max_per_source=2,
                           rerank_blend=blend)
        return results, time.perf_counter() - started, [asked]
    if mode == "decompose":
        subs = qr.decompose(question)
        results = qr.retrieve_decomposed(
            question, index, metadata, model, k=k, bm25=bm25,
            candidate_k=cand, use_reranker=True, fusion=DEFAULT_FUSION,
            max_per_source=2, rerank_blend=blend)
        return results, time.perf_counter() - started, subs
    results = retrieve(question, index, metadata, model, k=k, candidate_k=cand,
                       use_reranker=True, fusion=DEFAULT_FUSION, bm25=bm25,
                       max_per_source=2, rerank_blend=blend)
    return results, time.perf_counter() - started, [question]


def score(cases, index, metadata, model, bm25, mode, k, cand, blend,
          verbose=False):
    totals = {"hit_rate": 0.0, "mrr": 0.0, "ndcg": 0.0, "src_recall": 0.0}
    per_case, asked, seconds = {}, {}, 0.0
    for case in cases:
        results, took, queries = run_one(case["question"], mode, index,
                                         metadata, model, bm25, k, cand, blend)
        seconds += took
        gold = gold_keys(case)
        m = {"hit_rate": hit_rate(results, gold),
             "mrr": reciprocal_rank(results, gold),
             "ndcg": ndcg(results, gold),
             "src_recall": source_recall(results, gold_sources(case), k)}
        for key in totals:
            totals[key] += m[key]
        per_case[case["id"]] = m["hit_rate"] == 1.0
        if mode != "none":
            asked[case["id"]] = queries
        if verbose:
            print(f"      {case['id']:<20} {'hit' if per_case[case['id']] else 'miss'}")
    n = max(len(cases), 1)
    agg = {key: round(v / n, 4) for key, v in totals.items()}
    agg["seconds_per_query"] = round(seconds / n, 2)
    return agg, per_case, asked


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", action="append")
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--candidate-k", type=int, default=CANDIDATE_K)
    ap.add_argument("--structural-only", action="store_true",
                    help="score only the fixture. Fast, and it cannot show "
                         "what a mode costs the cases it was not built for.")
    ap.add_argument("--emit", type=Path, default=OUT)
    args = ap.parse_args()

    import query_rewrite as qr
    if not qr.available():
        print("no model reachable. Set RAG_GENERATOR=ollama with `ollama "
              "serve` running, or configure ANTHROPIC_API_KEY.")
        return 2

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)

    reg = corpora.registry()
    report = {}
    for name in (args.corpus or list(reg)):
        cfg = reg.get(name)
        if not cfg or not cfg["indexed"] or not cfg["golden"]:
            continue
        answerable, _ = load_cases(cfg["golden"])
        structural = fixture_ids(cfg)
        if args.structural_only:
            answerable = [c for c in answerable if c["id"] in structural]
        if not answerable:
            continue

        index, metadata = load_index(cfg["store"])
        bm25 = build_bm25(metadata)

        print(f"\n  {cfg['label']}  ({len(answerable)} cases, "
              f"{len(structural)} structural, "
              f"{os.environ.get('RAG_OLLAMA_MODEL', 'llama3.2')})")
        print(f"    {'mode':<11}{'any-hit':>9}{'MRR':>8}{'NDCG':>8}{'src':>8}"
              f"{'structural':>12}{'s/query':>9}")

        rows, recovered = {}, {}
        base_per = None
        for mode in MODES:
            rr.clear_cache()
            agg, per, asked = score(answerable, index, metadata, model, bm25,
                                    mode, args.k, args.candidate_k,
                                    cfg["rerank_blend"])
            if mode == "none":
                base_per = per
            struck = sum(1 for i in structural if per.get(i))
            rows[mode] = {**agg, "structural_hit": struck,
                          "structural_total": len(structural),
                          "rewritten": asked}
            # Which individual cases changed hands, in both directions. An
            # aggregate that moves by one question hides whether it is the same
            # question.
            if base_per is not None and mode != "none":
                recovered[mode] = {
                    "gained": sorted(i for i in per
                                     if per[i] and not base_per.get(i)),
                    "lost": sorted(i for i in per
                                   if not per[i] and base_per.get(i)),
                }
            print(f"    {mode:<11}{agg['hit_rate']:>9.3f}{agg['mrr']:>8.3f}"
                  f"{agg['ndcg']:>8.3f}{agg['src_recall']:>8.3f}"
                  f"{struck:>8} / {len(structural):<3}"
                  f"{agg['seconds_per_query']:>9.2f}")

        for mode, moved in recovered.items():
            if moved["gained"] or moved["lost"]:
                print(f"    {mode}: gained {moved['gained'] or 'nothing'}, "
                      f"lost {moved['lost'] or 'nothing'}")
            else:
                print(f"    {mode}: no case changed hands")
        report[name] = {"label": cfg["label"], "modes": rows,
                        "changed": recovered,
                        "model": os.environ.get("RAG_OLLAMA_MODEL", "llama3.2")}

    if not report:
        print("nothing measured")
        return 1
    # The scope has to come from what was run. The note used to be a constant
    # claiming "the whole answerable set", so a --structural-only file
    # described itself as a full one and its any-hit column, correct for the
    # fixture and absurd for the corpus, read as the corpus figure.
    scope = "structural fixture only" if args.structural_only else "all answerable cases"
    note = ("End to end at the served configuration, scored over "
            + scope + ". A pool is a ceiling and not a proxy here.")
    if args.structural_only:
        note += (" These are the cases that fail under every pipeline "
                 "configuration, so the baseline any-hit is 0.000 by "
                 "construction and is not this corpus's hit rate. Re-run "
                 "without --structural-only to see what a mode costs the "
                 "cases it was not built for.")
    else:
        note += (" A mode measured only on the cases it was built for cannot "
                 "show what it costs the rest, which is why this is the whole "
                 "set.")
    args.emit.write_text(json.dumps(
        {"generated_by": "src/sweep_decompose.py",
         "scope": scope,
         "structural_only": bool(args.structural_only),
         "note": note,
         "corpora": report}, indent=1) + "\n", encoding="utf-8")
    print(f"\n  wrote {args.emit.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
