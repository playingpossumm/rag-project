"""Would a different cross-encoder be worth its cost, on every corpus at once?

The reranker is the weakest stage. On the bird corpus the candidate pool holds
the answer 96.2% of the time and the finished pipeline returns it 84.6%, and the
failures share a shape: the question DESCRIBES a term rather than naming it --
"the burst of collective singing at first light" -- so there is little lexical
purchase and `ms-marco-MiniLM-L-6-v2` has little to work with.

This measures a swap rather than assuming one. Two things about how, because the
obvious version of both is wrong.

**It measures two different failures, not one.** Reranking can fail by ordering
badly, and it can fail by SCORING badly -- and only the first shows up in
any-hit. On the bird corpus today, three of the seven failures are questions
whose answer the pipeline put in the top five and whose passage the
cross-encoder then scored below three of the six adversarial cases, so the gate
refused an answer that had already been found. A model that fixed the ordering
and left the scores would move any-hit and change nothing a reader sees.

**Scores from two models are not comparable, so a shared threshold is
meaningless.** Cross-encoder outputs are raw logits on each model's own scale;
holding -5.5 fixed while swapping the model measures the scale, not the
separation. So the gate half is reported threshold-free, as the probability that
a random answerable question outscores a random adversarial one -- AUC, which is
the same quantity the dot plot on the analytics page draws, without needing a
cut point to state it.

Candidate pools are built once per corpus and reused across models, so every
difference reported is the reranker's doing and not the first stage's.

    .venv\\Scripts\\python.exe src\\compare_rerankers.py                    # every corpus
    .venv\\Scripts\\python.exe src\\compare_rerankers.py --corpus birds     # just one
    .venv\\Scripts\\python.exe src\\compare_rerankers.py --models a,b       # by name
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
from evaluate import (gold_keys, gold_sources, hit_rate, load_cases, ndcg,  # noqa: E402
                      reciprocal_rank, source_recall)
from hybrid import build_bm25  # noqa: E402
from retrieve import (CANDIDATE_K, DEFAULT_FUSION, EMBEDDING_MODEL, TOP_K,  # noqa: E402
                      load_index, shortlist)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
OUT = ROOT / "eval" / "reranker-comparison.json"

# Everything already on disk, cheapest first. BGE-base is 278M against MiniLM-L6's
# 22M and was measured once before on the ML papers, where it fixed the same two
# of ten cases at 10 s/query and the upgrade was rejected. It is here because
# that was one corpus, and the corpus with the headroom is a different one.
MODELS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",      # shipped
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "BAAI/bge-reranker-base",
    # Split candidates: rank with one model, gate with another. The first run
    # of this harness found the two jobs want different models -- BGE ranked
    # worse than the shipped MiniLM and separated answerable from adversarial
    # far better -- and they have very different costs. Ranking scores twenty
    # candidates; the gate scores one.
    "cross-encoder/ms-marco-MiniLM-L-6-v2|BAAI/bge-reranker-base",
    "cross-encoder/ms-marco-MiniLM-L-6-v2|cross-encoder/ms-marco-MiniLM-L-12-v2",
]


def split_spec(spec: str) -> tuple[str, str]:
    """`rank|gate`, or one name used for both."""
    rank, _, gate = spec.partition("|")
    return rank, (gate or rank)


def auc(pos: list[float], neg: list[float]) -> float:
    """P(a random answerable outscores a random adversarial), ties at half.

    Threshold-free on purpose: it is the one number about the gate that can be
    compared between two models whose scores are on different scales. 1.0 is a
    clean separation, 0.5 is a coin flip, and the shipped model does not manage
    either.
    """
    if not pos or not neg:
        return float("nan")
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def unreachable(pos: list[float], neg: list[float]) -> int:
    """Answerable questions no threshold can save without losing a catch.

    An answerable case scoring below the k-th adversarial cannot be recovered
    without letting k adversarial cases through. Counting those at k=3 gives a
    number that says how much of the gate's failure belongs to the model rather
    than to the cut point.

    Read it DOWN a column, never across one. The bar is the third-highest
    adversarial score, so a corpus with seventeen adversarial cases sets a
    higher bar than one with six, and the ML papers' 9 is not worse than the
    bird corpus's 2 -- the two numbers are answering different questions. Within
    one corpus, where the bar is fixed, it compares models exactly.
    """
    if len(neg) < 3:
        return sum(1 for p in pos if p < max(neg)) if neg else 0
    third = sorted(neg, reverse=True)[2]
    return sum(1 for p in pos if p < third)


def evaluate_model(spec: str, corpus: dict, pools: dict, cases: list[dict],
                   k: int, blend: float) -> dict:
    """Rank every corpus case with one model and score both halves.

    `spec` may name two models: one to order the pool and one to score the
    passage the gate reads. When they differ the second is asked for exactly one
    score per query, which is what makes an expensive model affordable here.
    """
    name, gate_name = split_spec(spec)
    rr.clear_cache()
    # Load before the clock starts. A 278M model takes seconds to construct and
    # that is paid once per process, not once per query -- charging it to the
    # first query would make the biggest model look worse than it serves.
    rr.load_reranker(name)
    gate_model = None
    if gate_name != name:
        from sentence_transformers import CrossEncoder
        gate_model = CrossEncoder(gate_name)
    started = time.perf_counter()

    ranking = {"hit_rate": 0.0, "mrr": 0.0, "ndcg": 0.0, "src_recall": 0.0}
    answerable_top1: list[float] = []
    adversarial_top1: list[float] = []
    per_case = {}
    n_answerable = 0

    for case in cases:
        pool = pools[case["id"]]
        # The serving path exactly: rerank the whole pool, then diversify, then
        # cut to k. Selecting k first would leave the diversity cap nothing to
        # choose between, which is why retrieve.py does it in this order.
        ranked = rr.rerank(case["question"], pool, k=len(pool),
                           model_name=name, blend=blend)
        results = diversify(ranked, k=k, max_per_source=2)

        # What the gate reads: the cross-encoder's score for whatever passage
        # the blended ordering put first -- not the best score in the pool.
        top1 = results[0]["rerank_score"] if results else float("-inf")
        if gate_model is not None and results:
            # One pair, on the passage the ranking already chose. Note this is
            # NOT the same quantity as the gate model's own AUC above: there it
            # scored the passage it would itself have ranked first, here it
            # scores someone else's pick. Assuming the two are equal is the
            # mistake this row exists to test.
            top1 = float(gate_model.predict([(case["question"],
                                              results[0]["text"])])[0])

        if case.get("unanswerable"):
            adversarial_top1.append(top1)
            continue

        n_answerable += 1
        answerable_top1.append(top1)
        gold = gold_keys(case)
        m = {"hit_rate": hit_rate(results, gold),
             "mrr": reciprocal_rank(results, gold),
             "ndcg": ndcg(results, gold),
             "src_recall": source_recall(results, gold_sources(case), k)}
        for key in ranking:
            ranking[key] += m[key]
        per_case[case["id"]] = {"hit": m["hit_rate"] == 1.0,
                                "confidence": round(top1, 3)}

    elapsed = time.perf_counter() - started
    n = max(n_answerable, 1)
    return {
        "model": spec,
        "rank_model": name,
        "gate_model": gate_name,
        **{key: round(total / n, 4) for key, total in ranking.items()},
        "auc": round(auc(answerable_top1, adversarial_top1), 4),
        "unreachable": unreachable(answerable_top1, adversarial_top1),
        "ms_per_query": round(elapsed / max(len(cases), 1) * 1000),
        "seconds": round(elapsed, 1),
        "cases": per_case,
    }


def run_corpus(name: str, cfg: dict, models: list[str], k: int,
               candidate_k: int) -> dict | None:
    from sentence_transformers import SentenceTransformer

    answerable, adversarial = load_cases(cfg["golden"])
    cases = answerable + adversarial
    index, metadata = load_index(cfg["store"])
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    bm25 = build_bm25(metadata)
    blend = cfg["rerank_blend"]

    print(f"\n{'=' * 74}\n{cfg['label']}  --  {len(answerable)} answerable + "
          f"{len(adversarial)} adversarial, blend {blend:.2f}")

    # Once, and reused by every model. Any difference below is the reranker's.
    pools = {c["id"]: shortlist(c["question"], index, metadata, embedder,
                                k=candidate_k, fusion=DEFAULT_FUSION, bm25=bm25)
             for c in cases}

    print(f"  {'model':<52}{'any-hit':>9}{'MRR':>8}{'AUC':>8}"
          f"{'unrec':>7}{'ms/q':>8}")
    rows = []
    for spec in models:
        try:
            row = evaluate_model(spec, cfg, pools, cases, k, blend)
        except Exception as exc:                              # noqa: BLE001
            print(f"  {spec:<38}  unavailable: {exc}")
            continue
        rows.append(row)
        rank_m, gate_m = split_spec(spec)
        short = (rank_m.split("/")[-1] if rank_m == gate_m
                 else f"{rank_m.split('/')[-1]} + {gate_m.split('/')[-1]} gate")
        mark = "  <- shipped" if spec == rr.RERANK_MODEL else ""
        print(f"  {short:<52}{row['hit_rate']:>9.3f}{row['mrr']:>8.3f}"
              f"{row['auc']:>8.3f}{row['unreachable']:>7}"
              f"{row['ms_per_query']:>8}{mark}")

    return {"corpus": name, "label": cfg["label"], "blend": blend,
            "n_answerable": len(answerable), "n_adversarial": len(adversarial),
            "models": rows}


def verdict(results: list[dict]) -> None:
    """Say plainly whether anything here is worth shipping, on every corpus.

    A model is only worth the swap if it does not hurt the corpora that were
    never the problem. That rule is not a general principle -- it is what this
    project learned by shipping a rerank blend of 0.35 tuned on one corpus and
    then measuring it worse than doing nothing on all three.
    """
    print(f"\n{'=' * 74}\nAgainst the shipped model, per corpus\n")
    baseline = rr.RERANK_MODEL
    others = [m for m in {r["model"] for res in results for r in res["models"]}
              if m != baseline]

    for model in sorted(others):
        print(f"  {model}")
        verdicts = []
        for res in results:
            rows = {r["model"]: r for r in res["models"]}
            if model not in rows or baseline not in rows:
                continue
            a, b = rows[baseline], rows[model]
            d_hit = b["hit_rate"] - a["hit_rate"]
            d_auc = b["auc"] - a["auc"]
            d_unr = b["unreachable"] - a["unreachable"]
            slower = b["ms_per_query"] / max(a["ms_per_query"], 1)
            verdicts.append(d_hit >= 0 and d_auc >= 0)
            print(f"    {res['label']:<24} any-hit {d_hit:+.3f}   AUC {d_auc:+.3f}"
                  f"   unrecoverable {d_unr:+d}   {slower:.1f}x slower")
        if verdicts and all(verdicts):
            print("    -> better or equal on every corpus; the cost is the question\n")
        else:
            print("    -> worse on at least one corpus; not shippable as a default\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", action="append",
                    help="limit to one corpus (repeatable); default is all indexed")
    ap.add_argument("--models", help="comma-separated model names")
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--candidate-k", type=int, default=CANDIDATE_K)
    ap.add_argument("--emit", type=Path, default=OUT)
    args = ap.parse_args()

    models = args.models.split(",") if args.models else MODELS
    reg = corpora.registry()
    wanted = args.corpus or list(reg)

    results = []
    for name in wanted:
        cfg = reg.get(name)
        if not cfg:
            print(f"  unknown corpus {name!r}; have {sorted(reg)}")
            return 2
        if not cfg["indexed"] or not cfg["golden"] or not cfg["golden"].exists():
            print(f"  {name}: no index or no golden set, skipped")
            continue
        res = run_corpus(name, cfg, models, args.k, args.candidate_k)
        if res:
            results.append(res)

    if not results:
        print("nothing to compare")
        return 1

    verdict(results)
    args.emit.parent.mkdir(exist_ok=True)
    args.emit.write_text(json.dumps(
        {"generated_by": "src/compare_rerankers.py",
         "shipped": rr.RERANK_MODEL,
         "k": args.k, "candidate_k": args.candidate_k,
         "corpora": results}, indent=1) + "\n", encoding="utf-8")
    # A path given on the command line may be relative or outside the repo,
    # and relative_to() raises rather than falling back -- which crashed the
    # run AFTER it had written its results, turning a finished measurement into
    # what looked like a failed one.
    try:
        where = args.emit.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        where = str(args.emit)
    print(f"wrote {where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
