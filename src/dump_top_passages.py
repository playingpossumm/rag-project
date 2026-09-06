"""Every golden question's top passage, as the page would receive it.

The answer highlight is the one part of the interface a reader examines word by
word, and it had no test and no measurement -- there was no way to ask "how much
of the passage does it set bold, across every question we have?" without asking
it by hand, one screenshot at a time.

This writes the input half. `ui/test-answer-mark.mjs` reads it and runs the real
`markAnswer` over all of it, so the highlight is measured on the passages it
actually receives rather than on examples chosen to suit it.

The file is an object with `inputs` and `rows`. Until 2026-09-06 it was a bare
list, which left no way to tell which golden sets or indexes the passages came
from; on that day the list disagreed with the golden sets on 4 of its 157 rows
(1 reworded question, 3 cases moved between answerable and adversarial) and
nothing had reported it. `check_freshness.py` reads the digests now.

    .venv\\Scripts\\python.exe src\\dump_top_passages.py    ->  eval/top-passages.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import corpora  # noqa: E402
from check_freshness import stamp  # noqa: E402
from evaluate import load_cases  # noqa: E402
from hybrid import build_bm25  # noqa: E402
from retrieve import (CANDIDATE_K, DEFAULT_FUSION, EMBEDDING_MODEL, TOP_K,  # noqa: E402
                      load_index, retrieve)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
OUT = ROOT / "eval" / "top-passages.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emit", type=Path, default=OUT)
    ap.add_argument("--expansion", default="window",
                    help="the page receives expanded passages; match it")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    rows = []
    inputs = {}
    for name, cfg in corpora.registry().items():
        if not cfg["indexed"] or not cfg["golden"] or not cfg["golden"].exists():
            continue
        answerable, adversarial = load_cases(cfg["golden"])
        index, metadata = load_index(cfg["store"])
        bm25 = build_bm25(metadata)
        inputs[name] = {
            "golden": stamp(cfg["golden"], cases=len(answerable) + len(adversarial)),
            "index": stamp(cfg["store"] / "metadata.json", chunks=int(index.ntotal)),
        }
        print(f"  {name}: {len(answerable) + len(adversarial)} questions")

        for case in answerable + adversarial:
            # The corpus's blend is passed on the call rather than set on the
            # rerank module, so this script cannot leave one corpus's setting
            # behind for the next.
            results = retrieve(case["question"], index, metadata, model,
                               k=TOP_K, candidate_k=CANDIDATE_K, bm25=bm25,
                               use_reranker=True, fusion=DEFAULT_FUSION,
                               max_per_source=2, expansion=args.expansion,
                               rerank_blend=cfg["rerank_blend"])
            if not results:
                continue
            # The top passage and the two the page offers under "Also found",
            # because those are marked by the same function and are the ones
            # most likely to be off-topic enough to mark badly.
            seen = {results[0]["source"]}
            others = []
            for r in results[1:]:
                if r["source"] in seen:
                    continue
                seen.add(r["source"])
                others.append(r["text"])
                if len(others) == 2:
                    break
            rows.append({"corpus": name, "id": case["id"],
                         "question": case["question"],
                         "unanswerable": bool(case.get("unanswerable")),
                         "text": results[0]["text"], "also": others})

    args.emit.parent.mkdir(exist_ok=True)
    args.emit.write_text(json.dumps(
        {"generated_by": "src/dump_top_passages.py",
         "expansion": args.expansion,
         "inputs": {"per_corpus": inputs},
         "rows": rows}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.emit.relative_to(ROOT).as_posix()}: {len(rows)} questions, "
          f"{sum(1 + len(r['also']) for r in rows)} passages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
