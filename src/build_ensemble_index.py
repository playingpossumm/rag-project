"""Add a second dense index to a store, without touching the first.

`src/sweep_ensemble.py` measured that fusing the shipped embedder with
`bge-small-en-v1.5` is weakly dominant across all three corpora -- +0.015 pool
recall on the ML papers, +0.038 on birds, level on quant -- and that it pulls
`bird-alula` out of the structural fixture, taking bird pool recall to 1.000.

Serving that needs a second vector per chunk. This writes one **beside** the
existing index rather than rebuilding it:

    <store>/index-ensemble.faiss      the second model's vectors
    <store>/ensemble.json             which model wrote them, and over what

Additive on purpose. The shipped index, its metadata and its caches are not
read for writing and not modified, so the change is reversible by deleting two
files, and a corpus whose `corpora.json` entry names no ensemble model keeps
behaving exactly as it did. Row order matches `metadata.json` exactly, which is
what lets the two indexes share one metadata list.

    .venv\\Scripts\\python.exe src\\build_ensemble_index.py --corpus birds
    .venv\\Scripts\\python.exe src\\build_ensemble_index.py            # every corpus
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import faiss  # noqa: E402
import numpy as np  # noqa: E402

import corpora  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

INDEX_FILE = "index-ensemble.faiss"
MANIFEST = "ensemble.json"


# What each model wants on the query side. This belongs to the model, not to
# the corpus: BGE retrieval models are trained with the query prefixed and land
# in a different part of the space without it.
#
# It used to be read only from corpora.json, so `--model` changed the model and
# left the prefix behind. Overriding the model on a corpus that had never named
# one built an index with an empty prefix, and nothing failed: right size, right
# row count, well-formed manifest, quietly worse retrieval.
MODEL_PREFIX = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "intfloat/e5-small-v2": "query: ",
    "intfloat/e5-base-v2": "query: ",
}


def prefix_for(model_name: str, cfg: dict) -> str:
    """The query prefix for this model, unless the corpus names its own.

    corpora.json still wins when it says something, so a corpus can override.
    What it can no longer do is supply an empty prefix by saying nothing for a
    model that needs one.
    """
    # `corpora.registry()` normalises a missing key to "", not None, so an
    # `is not None` test here never falls through and the model default is
    # never reached. Testing truthiness instead means a corpus cannot ask for
    # an empty prefix on a model that has a default; nothing wants that today,
    # and a corpus that does can name a single space.
    named = cfg.get("ensemble_prefix")
    if named:
        return named
    return MODEL_PREFIX.get(model_name, "")


def build(cfg: dict, model_name: str, prefix: str, batch: int = 64) -> dict:
    from sentence_transformers import SentenceTransformer

    store = cfg["store"]
    metadata = json.loads((store / "metadata.json").read_text(encoding="utf-8"))
    texts = [c.get("embed_text", c["text"]) for c in metadata]

    started = time.perf_counter()
    model = SentenceTransformer(model_name)
    vectors = model.encode(texts, convert_to_numpy=True, batch_size=batch,
                           show_progress_bar=False, normalize_embeddings=True)
    vectors = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(store / INDEX_FILE))

    manifest = {
        "generated_by": "src/build_ensemble_index.py",
        "model": model_name,
        # The query-side instruction, stored rather than assumed: the BGE and
        # E5 families are trained with one and score materially worse without
        # it, so a reader of this file must not have to know which family the
        # model belongs to.
        "query_prefix": prefix,
        "chunks": len(metadata),
        "dim": int(vectors.shape[1]),
        "seconds": round(time.perf_counter() - started, 1),
    }
    (store / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n",
                                  encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", action="append")
    ap.add_argument("--model", default=None,
                    help="override the model named in corpora.json")
    args = ap.parse_args()

    reg = corpora.registry()
    built = 0
    for name in (args.corpus or list(reg)):
        cfg = reg.get(name)
        if not cfg:
            print(f"  unknown corpus {name!r}")
            return 2
        if not cfg["indexed"]:
            print(f"  {name}: not indexed, skipped")
            continue
        model_name = args.model or cfg.get("ensemble_model")
        if not model_name:
            print(f"  {name}: no ensemble_model in corpora.json, skipped")
            continue

        print(f"  {cfg['label']}: embedding with {model_name}...")
        prefix = prefix_for(model_name, cfg)
        if prefix:
            print(f"    query prefix: {prefix!r}")
        m = build(cfg, model_name, prefix)
        print(f"    {m['chunks']:,} chunks, {m['dim']} dimensions, "
              f"{m['seconds']}s -> {cfg['store'].name}/{INDEX_FILE}")
        built += 1

    if not built:
        print("nothing built -- name an ensemble_model in corpora.json")
        return 1
    print(f"\nbuilt {built} ensemble index(es). Retrieval uses them only for "
          f"corpora that name one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
