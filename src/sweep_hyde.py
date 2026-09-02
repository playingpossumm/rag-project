"""Does retrieving on a hypothetical answer help, measured on the whole set?

Four of the five cases nothing reaches share a shape: the word that would find
the passage is absent from the question. That is why rewriting and
decomposition failed against them, since rewriting a question that lacks a word
produces another question that lacks it. A hypothetical answer is different in
kind: asked to answer from memory, a model writes a passage containing the
term, and the search runs on that. The corpus still supplies the citation and
the model supplies only the vocabulary to find it with.

**On the five failures alone it recovers two**, `wmt14` and `bird-hollow-bones`,
and the mechanism is visible: it hits when the hypothetical contains the answer
term and misses when it does not. That is a fixture result and this project has
been burned by fixture results before. `src/sweep_decompose.py` records the
case: rewriting recovered one of the seven it was aimed at and lost three
questions net over the full set, because every case it broke lay outside the
fixture. So this scores every answerable case on every corpus.

Three modes:

    none            the shipped pipeline, as the control
    hypothetical    retrieve on the generated passage alone
    both            retrieve on the question and the passage concatenated

The adversarial half matters as much as the answerable half here. A
hypothetical answer is written confidently whether or not the corpus holds the
answer, so it can push an unanswerable question over the abstention threshold,
and that cost would be invisible in any hit-rate column.

    set RAG_GENERATOR=ollama
    .venv\\Scripts\\python.exe src\\sweep_hyde.py
    .venv\\Scripts\\python.exe src\\sweep_hyde.py --corpus birds --limit 10
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import corpora  # noqa: E402
import query_rewrite as qw  # noqa: E402
import rerank as rr  # noqa: E402
from evaluate import gold_keys, is_relevant, load_cases  # noqa: E402
from hybrid import build_bm25  # noqa: E402
from retrieve import (DEFAULT_FUSION, EMBEDDING_MODEL, TOP_K,  # noqa: E402
                      load_ensemble, load_index, retrieve)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent

# Asks for a passage rather than an answer, because the text is going to be
# embedded and compared against passages. "Name the specific term" is the whole
# point: a hedged answer contains none of the words that would find anything.
HYDE_SYSTEM = (
    "Answer the question in two or three sentences, as if writing a passage "
    "from a reference document. Name the specific term, model, dataset or "
    "quantity the question is asking about. Do not hedge, do not say you are "
    "unsure, and do not mention that you are an AI. Write only the passage."
)

MODES = ("none", "hypothetical", "both")


def hypothetical(question: str) -> str | None:
    out = qw._ask(HYDE_SYSTEM, question)
    return " ".join(out.split()) if out else None


def run(cfg, cases, adversarial, limit):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    index, metadata = load_index(cfg["store"])
    bm25 = build_bm25(metadata)
    ensemble = load_ensemble(cfg["store"])
    rr.RERANK_BLEND = cfg["rerank_blend"]
    threshold = cfg["threshold"]

    answerable = cases[:limit] if limit else cases
    adv = adversarial[:limit] if limit else adversarial
    out = {m: {"hits": [], "misses": [], "slips": [], "seconds": 0.0}
           for m in MODES}
    generated = {}

    for case in answerable + adv:
        q = case["question"]
        if q not in generated:
            t0 = time.perf_counter()
            generated[q] = (hypothetical(q), time.perf_counter() - t0)
        hypo, gen_s = generated[q]

        for mode in MODES:
            if mode == "none":
                query, cost = q, 0.0
            elif hypo is None:
                continue
            else:
                query = hypo if mode == "hypothetical" else f"{q} {hypo}"
                cost = gen_s

            t0 = time.perf_counter()
            rr.clear_cache()
            res = retrieve(query, index, metadata, model, k=TOP_K,
                           candidate_k=cfg["candidate_k"], use_reranker=True,
                           fusion=DEFAULT_FUSION, bm25=bm25, max_per_source=2,
                           ensemble=ensemble)
            out[mode]["seconds"] += cost + time.perf_counter() - t0

            if case in adv:
                # The gate reads the reranker's score, not the fused one that
                # `score` carries, and reading the wrong field reported every
                # adversarial case as slipping under the control too.
                top = res[0].get("rerank_score") if res else None
                if top is not None and top >= threshold:
                    out[mode]["slips"].append(case["id"])
                continue

            gold = gold_keys(case)
            if any(is_relevant(r, gold) for r in res):
                out[mode]["hits"].append(case["id"])
            else:
                out[mode]["misses"].append(case["id"])

    return out, len(answerable), len(adv)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="first N answerable and N adversarial, for a smoke run")
    ap.add_argument("--emit", type=Path, default=ROOT / "eval" / "hyde-sweep.json")
    args = ap.parse_args()

    os.environ.setdefault("RAG_GENERATOR", "ollama")
    if not qw.available():
        print("no generator available -- set RAG_GENERATOR=ollama and start it")
        return 2

    reg = corpora.registry()
    names = [args.corpus] if args.corpus else [n for n, c in reg.items()
                                               if c["indexed"] and c["golden"]]
    report = {"generated_by": "src/sweep_hyde.py",
              "generator": qw._backend(),
              "model": os.environ.get("RAG_OLLAMA_MODEL", "llama3.2"),
              "corpora": {}}

    for name in names:
        cfg = reg[name]
        cases, adversarial = load_cases(Path(cfg["golden"]))
        res, n_ans, n_adv = run(cfg, cases, adversarial, args.limit)
        base = set(res["none"]["hits"])

        print(f"\n  {cfg['label']}  ({n_ans} answerable, {n_adv} adversarial)")
        print(f"    {'mode':<14}{'any-hit':>9}{'gained':>8}{'lost':>6}"
              f"{'slips':>7}{'sec/q':>8}")
        rows = {}
        for mode in MODES:
            r = res[mode]
            if not r["hits"] and not r["misses"]:
                continue
            hits = set(r["hits"])
            rate = len(hits) / n_ans if n_ans else 0.0
            gained = sorted(hits - base)
            lost = sorted(base - hits)
            rows[mode] = {"any_hit": round(rate, 4),
                          "gained": gained, "lost": lost,
                          "slips": sorted(r["slips"]),
                          "seconds_per_query": round(
                              r["seconds"] / max(n_ans + n_adv, 1), 2)}
            print(f"    {mode:<14}{rate:>9.3f}{len(gained):>8}{len(lost):>6}"
                  f"{len(r['slips']):>7}{rows[mode]['seconds_per_query']:>8.2f}")
            if gained:
                print(f"        gained: {', '.join(gained)}")
            if lost:
                print(f"        lost:   {', '.join(lost)}")
        report["corpora"][name] = {"label": cfg["label"], "n_answerable": n_ans,
                                   "n_adversarial": n_adv, "modes": rows}

    args.emit.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"\n  wrote {args.emit.relative_to(ROOT)}")
    print("  Read the gained and lost lists, not the any-hit column alone: a "
          "mode that\n  gains two and loses two is not neutral, it is two "
          "different pipelines.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
