from pathlib import Path
import json
import os
import sys

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from diversify import DEFAULT_MAX_PER_SOURCE, diversify

# Windows consoles default to cp1252, which cannot encode the mathematical
# notation common in technical PDFs; printing a retrieved chunk would crash.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Must agree with ingest.STORE_DIR, and does so by reading the same variable.
# Serving one corpus from an index built over another produces answers with
# citations pointing at documents that were never searched.
STORE_DIR = Path(os.environ.get("RAG_STORE_DIR",
                                Path(__file__).parent.parent / "vector_store")).expanduser()
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

# The reranker only reorders what the first stage hands it, so a relevant chunk
# missing from the candidate set can never be recovered. Over-retrieving here
# trades a little reranking latency for recall headroom.
CANDIDATE_K = 20

# RRF measured never worse than dense-only and increasingly better as the
# candidate pool gets selective (see eval/RESULTS.md). On this 64-chunk corpus
# k=20 is 31% of everything, so recall is perfect either way and the two look
# equivalent -- that equivalence is an artifact of a small corpus, not a
# property of the method. Defaulting to RRF is the choice that stays correct as
# the corpus grows.
DEFAULT_FUSION = "rrf"


def load_index(store_dir=None):
    """Read an index. `store_dir` lets one process hold several corpora.

    The env var still sets the default, so every script keeps working unchanged;
    only a caller that genuinely serves more than one corpus needs to pass a
    path.
    """
    store = Path(store_dir) if store_dir else STORE_DIR
    index = faiss.read_index(str(store / "index.faiss"))
    with open(store / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def search(query: str, index, metadata, model, k: int = TOP_K,
           allowed_ids: list[int] | None = None) -> list[dict]:
    query_vec = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)

    # A selector makes FAISS skip excluded rows during the scan, so the result
    # is the true top k WITHIN the subset. Retrieving normally and filtering
    # afterwards would return the survivors of the top k overall, which is
    # empty whenever the unfiltered leaders all sit outside the scope.
    if allowed_ids is None:
        scores, ids = index.search(query_vec.astype(np.float32), k)
    else:
        if not allowed_ids:
            return []
        from metadata_filter import selector

        params = faiss.SearchParameters()
        params.sel = selector(allowed_ids)
        scores, ids = index.search(query_vec.astype(np.float32),
                                   min(k, len(allowed_ids)), params=params)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        chunk = metadata[idx]
        # chunk_id is the row position shared by the FAISS index and the
        # metadata sidecar. Carrying it explicitly lets rankings from different
        # retrievers be fused by identity rather than by comparing text.
        results.append({**chunk, "chunk_id": int(idx), "score": float(score)})
    return results


def shortlist(
    query: str,
    index,
    metadata,
    model,
    k: int,
    fusion: str = DEFAULT_FUSION,
    bm25=None,
    alpha: float = 0.5,
    allowed_ids: list[int] | None = None,
    query_expansion: str = "none",
) -> list[dict]:
    """First stage: produce the candidate pool the reranker will reorder.

    fusion="none" is dense-only; "rrf" and "weighted" add BM25 and fuse. Both
    retrievers are asked for k candidates each, so fusion sees the same depth
    per retriever rather than splitting one budget between them.
    """
    dense = search(query, index, metadata, model, k=k, allowed_ids=allowed_ids)
    if fusion == "none":
        return dense

    from hybrid import bm25_search, build_bm25, fuse_rrf, fuse_weighted

    # Callers that run many queries should build this once and pass it in; the
    # eval harness does. Building per call is only acceptable for one-shot use.
    if bm25 is None:
        bm25 = build_bm25(metadata)

    # Expansion is given to the sparse retriever only: appending a bag of
    # keywords to a sentence moves its embedding somewhere that is not a
    # question, so it helps lexical matching and distorts semantic matching.
    terms = None
    if query_expansion == "prf":
        from query_expansion import expand

        terms = expand(query, bm25, metadata)
    elif query_expansion != "none":
        raise ValueError(f"unknown query expansion: {query_expansion!r}")

    sparse = bm25_search(query, bm25, metadata, k=k, allowed_ids=allowed_ids,
                         terms=terms)
    if fusion == "rrf":
        return fuse_rrf(dense, sparse, k=k)
    if fusion == "weighted":
        return fuse_weighted(dense, sparse, k=k, alpha=alpha)
    raise ValueError(f"unknown fusion strategy: {fusion!r}")


def retrieve(
    query: str,
    index,
    metadata,
    model,
    k: int = TOP_K,
    candidate_k: int = CANDIDATE_K,
    use_reranker: bool = True,
    fusion: str = DEFAULT_FUSION,
    bm25=None,
    alpha: float = 0.5,
    expansion: str = "none",
    window: int = 1,
    max_per_source: int | None = DEFAULT_MAX_PER_SOURCE,
    allowed_ids: list[int] | None = None,
    query_expansion: str = "none",
) -> list[dict]:
    """Full retrieval pipeline: shortlist, rerank, then expand context.

    Each stage is an independent knob so they can be measured separately --
    fusion widens what the candidate pool contains, reranking reorders it, and
    expansion changes how much text each survivor carries. Note that expansion
    cannot move any ranking metric; it is measured by context recall instead.
    """
    if use_reranker:
        from rerank import rerank  # lazy so the baseline path stays light

        candidates = shortlist(query, index, metadata, model, k=candidate_k,
                               fusion=fusion, bm25=bm25, alpha=alpha,
                               allowed_ids=allowed_ids,
                               query_expansion=query_expansion)
        # Rerank the whole pool, then select k. Selecting first would give the
        # diversity step nothing to choose between.
        ranked = rerank(query, candidates, k=len(candidates))
        results = (diversify(ranked, k=k, max_per_source=max_per_source)
                   if max_per_source else ranked[:k])
    else:
        results = shortlist(query, index, metadata, model, k=k,
                            fusion=fusion, bm25=bm25, alpha=alpha,
                            allowed_ids=allowed_ids,
                            query_expansion=query_expansion)

    # Expansion runs last, deliberately. Ranking on small chunks is what keeps
    # precision high; growing them any earlier would feed the reranker diluted
    # text and undo the gain.
    if expansion != "none":
        from parent import expand

        results = expand(results, metadata, mode=expansion, window=window)
    return results


def main():
    query = " ".join(sys.argv[1:]) or input("Ask a question: ")

    index, metadata = load_index()
    model = SentenceTransformer(EMBEDDING_MODEL)

    results = retrieve(query, index, metadata, model)

    print(f"\nTop {len(results)} matches for: {query!r}\n")
    for rank, r in enumerate(results, start=1):
        score = r.get("rerank_score", r.get("score"))
        loc = r["locator"]
        print(f"[{rank}] {r['source']}, {loc['kind']} {loc['value']} (score: {score:+.3f})")
        print(r["text"][:300].replace("\n", " ") + "...")
        print()


if __name__ == "__main__":
    main()
