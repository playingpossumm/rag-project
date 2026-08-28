"""Write the per-case outcome of every golden-set question to JSON.

`evaluate.py --per-case` already computes this and prints it. Printing is enough
to read a run and useless for anything that wants to work with one: which cases
a change fixed and which it broke, whether the 15% cross-document confusion is
the same fifteen percent from run to run, or drawing all 84 cases at once.

A separate script rather than a flag on the harness, deliberately. evaluate.py's
job is to produce the aggregate numbers every document quotes, and its output
file is the one source those documents are checked against. Growing it a second
output shape gives that one job two reasons to change.

It reuses evaluate.py's metric functions rather than reimplementing them, so a
per-case number here and an aggregate number there cannot disagree -- the
aggregates in this file's `totals` are the mean of its own rows, which is a
check on both.

The corpus's own settings come from corpora.json, resolved from the golden set,
because for most of this script's life they did not. It imported the module
constant ABSTAIN_THRESHOLD -- the ML papers' 0.0 -- and scored every corpus
against it, and never set a rerank blend at all. On the bird corpus, which ships
-3.0 and 0.20, that produced a file claiming ten answerable questions were
wrongly refused where the served configuration refuses four, and confidences
from a reranker weighted differently from the one behind the answer on screen.
Nothing errored: a threshold is a number and every number was present. This is
the same trap `results.json` fell into by sharing one path across corpora, and
`src/check_freshness.py` now fails when the file and corpora.json disagree.

    .venv\\Scripts\\python.exe src\\per_case.py             # serving defaults
    .venv\\Scripts\\python.exe src\\per_case.py --no-rerank
"""
import argparse
import json
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

import corpora
import rerank as _rerank
from abstain import ABSTAIN_THRESHOLD
from check_freshness import artefact_suffix, stamp
from evaluate import (GOLDEN_SET, context_recall, gold_keys, gold_sources, hit_rate,
                      is_relevant, load_cases, ndcg, reciprocal_rank,
                      source_recall)
from hybrid import build_bm25
from retrieve import (CANDIDATE_K, DEFAULT_FUSION, EMBEDDING_MODEL, TOP_K,
                      load_ensemble, load_index, retrieve)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(__file__).parent.parent / "eval" / "per_case.json"


def corpus_for(golden: Path) -> dict | None:
    """The corpus this golden set belongs to, by filename.

    A corpus and its questions travel together, and so do its index, its
    threshold and its rerank blend. Matching on the filename is the same rule
    evaluate.py uses; matching on the full path would fail the moment either
    side is passed a relative path.
    """
    for cfg in corpora.registry().values():
        if cfg["golden"] and Path(cfg["golden"]).name == golden.name:
            return cfg
    return None


def outcome(case: dict, results: list[dict], gold: set, confident: bool) -> str:
    """One word for what happened, for filtering and for colouring a chart.

    Answerable and unanswerable cases are scored on different questions --
    "was the answer found" against "was the refusal correct" -- so they get
    different vocabularies rather than a shared right/wrong that would mean two
    things at once.
    """
    if case.get("unanswerable"):
        return "refused" if not confident else "answered_anyway"
    if not confident:
        return "refused_wrongly"
    return "found" if any(is_relevant(r, gold) for r in results) else "missed"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=TOP_K)
    # None means "whatever this corpus ships", resolved below. Defaulting to
    # the module constant would measure 20 for corpora that serve 16, which is
    # the trap this file already fell into for the threshold and the blend.
    ap.add_argument("--candidate-k", type=int, default=None)
    # A corpus and its questions travel together, like the corpus and its
    # index. Scoring one corpus against another corpus's golden set produces
    # numbers that look fine and mean nothing.
    ap.add_argument("--golden", type=Path, default=GOLDEN_SET,
                    help="golden set to score against")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-ensemble", action="store_true",
                    help="ignore this corpus's second dense index, to measure "
                         "what it is worth")
    ap.add_argument("--fusion", default=DEFAULT_FUSION,
                    choices=["none", "rrf", "weighted"])
    ap.add_argument("--max-per-source", type=int, default=2,
                    help="0 disables the diversity cap")
    ap.add_argument("--emit", type=Path, default=None,
                    help="default: derived from --golden, so each corpus keeps "
                         "its own file rather than the last run owning one")
    args = ap.parse_args()

    # Same trap results.json had: one hardcoded path meant whichever corpus ran
    # last owned it, and eval/analytics.json -- which is what the interface
    # offers as example questions -- is built from these files.
    if args.emit is None:
        args.emit = OUT.parent / f"per_case{artefact_suffix(args.golden)}.json"

    # Everything that does not transfer between corpora, read from the one file
    # that records it per corpus. Falling back to the module constants is only
    # for a golden set corpora.json does not know about, and it says so.
    cfg = corpus_for(args.golden)
    threshold = ABSTAIN_THRESHOLD
    blend = _rerank.RERANK_BLEND
    store = None
    if cfg:
        threshold = cfg["threshold"] if cfg["calibrated"] else 0.0
        blend = cfg["rerank_blend"]
        if args.candidate_k is None:
            args.candidate_k = cfg["candidate_k"]
        store = cfg["store"]
        _rerank.RERANK_BLEND = blend
        print(f"corpus {cfg['name']} ({cfg['label']}): threshold {threshold:+.1f}, "
              f"rerank blend {blend:.2f}, candidates {args.candidate_k}, "
              f"index {store.name}")
    else:
        print(f"{args.golden.name} is not in corpora.json -- using the module "
              f"defaults: threshold {threshold:+.1f}, rerank blend {blend:.2f}")
    if args.candidate_k is None:
        args.candidate_k = CANDIDATE_K

    opts = dict(use_reranker=not args.no_rerank, fusion=args.fusion,
                max_per_source=args.max_per_source or None)

    answerable, adversarial = load_cases(args.golden)
    # The store comes from the corpus, not from RAG_STORE_DIR, so a golden set
    # can no longer be scored against another corpus's index by forgetting an
    # environment variable.
    index, metadata = load_index(store)
    # The second dense retriever, if this corpus has one. Measured here because
    # the server runs it: a harness that skipped it would report a pipeline
    # nobody serves, which is the trap the threshold and the blend both fell
    # into before they were resolved per corpus.
    ensemble = None if args.no_ensemble else load_ensemble(store)
    if ensemble:
        print(f"second dense retriever: {ensemble[1]}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    bm25 = build_bm25(metadata)

    print(f"{len(answerable)} answerable + {len(adversarial)} adversarial "
          f"at k={args.k}, {opts}")

    rows = []
    for case in answerable + adversarial:
        results = retrieve(case["question"], index, metadata, model, k=args.k,
                           candidate_k=args.candidate_k, bm25=bm25,
                           ensemble=ensemble, **opts)
        gold = gold_keys(case)
        sources = gold_sources(case)

        # Read before expansion would have diluted it, which is why this script
        # does not expand: the gate scores the chunk the reranker actually saw.
        # It is also why `confidence` is absent rather than 0 with the reranker
        # off -- there is no calibrated score on that path, and 0 would compare
        # as "at the threshold" in anything that read this file.
        top = results[0].get("rerank_score") if results and not args.no_rerank else None
        confident = (top >= threshold) if top is not None else True

        row = {
            "id": case["id"],
            "kind": case.get("kind", case.get("adversarial_kind", "-")),
            "question": case["question"],
            "unanswerable": bool(case.get("unanswerable")),
            "gold_sources": sorted(sources),
            "confidence": round(float(top), 3) if top is not None else None,
            "abstained": not confident,
            "outcome": outcome(case, results, gold, confident),
            "results": [
                {"rank": i,
                 "source": r["source"],
                 "locator": f"{r['locator']['kind']} {r['locator']['value']}",
                 "score": round(float(r.get("rerank_score", r.get("score", 0.0))), 4),
                 "relevant": is_relevant(r, gold)}
                for i, r in enumerate(results, 1)
            ],
        }
        if not case.get("unanswerable"):
            row |= {
                "hit_rate": hit_rate(results, gold),
                "mrr": round(reciprocal_rank(results, gold), 4),
                "ndcg": round(ndcg(results, gold), 4),
                "src_recall": round(source_recall(results, sources, args.k), 4),
            }
            if case.get("answer_contains"):
                row["context_recall"] = context_recall(results, case["answer_contains"])
        rows.append(row)

    # Totals are rounded to 4dp for readability; the ROWS are authoritative.
    # This matters when comparing against eval/results.json, which rounds to 3:
    # 56 hits in 66 cases is 0.848484..., stored here as 0.8485, and re-rounding
    # that to 3dp gives 0.849 against the harness's 0.848 -- a disagreement
    # invented by rounding twice, not a difference in what was measured.
    # Recompute from "cases" rather than re-rounding these.
    ans_rows = [r for r in rows if not r["unanswerable"]]
    adv_rows = [r for r in rows if r["unanswerable"]]
    mean = lambda key, src: round(sum(r[key] for r in src) / len(src), 4) if src else None

    # What this file was built from, so a later change to either input is
    # detectable rather than silent. src/check_freshness.py reads this.
    inputs = {"golden": stamp(args.golden, cases=len(answerable) + len(adversarial))}
    if store:
        inputs["index"] = stamp(store / "metadata.json", chunks=len(metadata))

    payload = {
        "generated_by": "src/per_case.py",
        "options": {"k": args.k, "candidate_k": args.candidate_k,
                    "rerank_blend": blend,
                    "ensemble": ensemble[1] if ensemble else None, **opts},
        "threshold": threshold,
        "corpus": {"name": cfg["name"] if cfg else None,
                   "chunks": len(metadata),
                   "documents": len({c["source"] for c in metadata})},
        "inputs": inputs,
        "totals": {
            "answerable": {
                "n": len(ans_rows),
                **{k: mean(k, ans_rows) for k in ("hit_rate", "mrr", "ndcg", "src_recall")},
                "refused_wrongly": sum(r["outcome"] == "refused_wrongly" for r in ans_rows),
                "missed": sum(r["outcome"] == "missed" for r in ans_rows),
            },
            "adversarial": {
                "n": len(adv_rows),
                "refused": sum(r["outcome"] == "refused" for r in adv_rows),
                "answered_anyway": sum(r["outcome"] == "answered_anyway" for r in adv_rows),
            },
        },
        "cases": rows,
    }

    args.emit.parent.mkdir(exist_ok=True)
    args.emit.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    t = payload["totals"]
    print(f"\nanswerable   hit {t['answerable']['hit_rate']:.3f}  "
          f"mrr {t['answerable']['mrr']:.3f}  ndcg {t['answerable']['ndcg']:.3f}  "
          f"src {t['answerable']['src_recall']:.3f}")
    print(f"             {t['answerable']['missed']} missed, "
          f"{t['answerable']['refused_wrongly']} refused wrongly")
    print(f"adversarial  {t['adversarial']['refused']}/{t['adversarial']['n']} refused, "
          f"{t['adversarial']['answered_anyway']} answered anyway")
    print(f"\nwrote {args.emit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
