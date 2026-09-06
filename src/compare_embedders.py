"""Can a different first stage find a term the question never names?

The twelve cases that fail under every pipeline configuration split in two (see
`docs/engineering-log.md`, 2026-08-27). Seven, all on the ML corpus, are
cross-document confusion and want query decomposition, which wants an LLM. The
other five **describe a term and ask for its name** -- "which small group of
feathers helps prevent a stall at low speed" for the alula -- and the answer word
is absent from the question.

Reranking cannot fix that: `compare_rerankers.py` found nothing better, and
`sweep_query_expansion.py` found pseudo-relevance feedback recovers 0 of 12
because the feedback documents do not contain the missing word either.

**The stage nobody has varied is the first one.** `all-MiniLM-L6-v2` is a
general-purpose *symmetric* model -- trained to place two similar sentences near
each other. Description-to-term is *asymmetric*: a long question and a short
passage that share almost no vocabulary. Models trained for asymmetric retrieval
exist and are free to try.

This measures pool recall only -- can the first stage put the answer in front of
the reranker at all -- because that is the ceiling everything downstream works
under, and it needs no re-index: each model embeds the corpus in memory once.

    .venv\\Scripts\\python.exe src\\compare_embedders.py --corpus birds
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402

import corpora  # noqa: E402
from evaluate import fixture_ids, gold_keys, is_relevant, load_cases  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
OUT = ROOT / "eval" / "embedder-comparison.json"

# `query_prefix` is not decoration: the BGE and E5 families are trained with an
# instruction on the query side and score materially worse without it, so
# omitting it would measure the prefix rather than the model.
MODELS = [
    {"name": "sentence-transformers/all-MiniLM-L6-v2", "query_prefix": "",
     "note": "shipped"},
    {"name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1", "query_prefix": "",
     "note": "same size, trained for question-to-passage rather than "
             "sentence-to-sentence"},
    {"name": "BAAI/bge-small-en-v1.5",
     "query_prefix": "Represent this sentence for searching relevant passages: ",
     "note": "33M, asymmetric, instruction-prefixed queries"},
]


def evaluate_model(spec, metadata, cases, k):
    from sentence_transformers import SentenceTransformer

    started = time.perf_counter()
    model = SentenceTransformer(spec["name"])
    texts = [c.get("embed_text", c["text"]) for c in metadata]
    doc_vecs = model.encode(texts, convert_to_numpy=True, batch_size=64,
                            show_progress_bar=False, normalize_embeddings=True)
    q_vecs = model.encode([spec["query_prefix"] + c["question"] for c in cases],
                          convert_to_numpy=True, normalize_embeddings=True)
    encode_s = time.perf_counter() - started

    hits, per_case = 0, {}
    for case, qv in zip(cases, q_vecs):
        order = np.argsort(-(doc_vecs @ qv))[:k]
        pool = [metadata[i] for i in order]
        gold = gold_keys(case)
        found = any(is_relevant(c, gold) for c in pool)
        hits += found
        per_case[case["id"]] = bool(found)
    return {"pool_recall": round(hits / max(len(cases), 1), 4),
            "encode_seconds": round(encode_s, 1), "cases": per_case}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", action="append")
    ap.add_argument("--k", type=int, default=None, help="pool size; default is "
                                                        "the corpus's candidate_k")
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
        k = args.k or cfg["candidate_k"]
        structural = fixture_ids(cfg)

        print(f"\n  {cfg['label']}  ({len(answerable)} answerable, "
              f"{len(metadata):,} chunks, pool of {k})")
        print(f"    {'model':<44}{'pool recall':>13}{'structural':>12}{'encode':>9}")
        rows = {}
        for spec in MODELS:
            try:
                r = evaluate_model(spec, metadata, answerable, k)
            except Exception as exc:                            # noqa: BLE001
                print(f"    {spec['name'].split('/')[-1]:<44}  unavailable: {exc}")
                continue
            got = sum(1 for i in structural if r["cases"].get(i))
            r["structural_found"] = f"{got}/{len(structural)}"
            rows[spec["name"]] = r
            mark = "  <- shipped" if spec["note"] == "shipped" else ""
            print(f"    {spec['name'].split('/')[-1]:<44}{r['pool_recall']:>13.3f}"
                  f"{r['structural_found']:>12}{r['encode_seconds']:>8.0f}s{mark}")

        base = rows.get(MODELS[0]["name"])
        if base:
            for spec in MODELS[1:]:
                r = rows.get(spec["name"])
                if not r:
                    continue
                won = sorted(i for i, ok in r["cases"].items()
                             if ok and not base["cases"].get(i))
                lost = sorted(i for i, ok in base["cases"].items()
                              if ok and not r["cases"].get(i))
                print(f"      vs {spec['name'].split('/')[-1]}: "
                      f"+{won or '[]'} -{lost or '[]'}")
        report[name] = {"label": cfg["label"], "pool": k,
                        "structural": sorted(structural), "models": rows}

    if not report:
        print("nothing to compare")
        return 1
    args.emit.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"\n  wrote {args.emit.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
