from pathlib import Path
import json
import sys

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Windows consoles default to cp1252, which cannot encode the mathematical
# notation common in technical PDFs; printing a retrieved chunk would crash.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STORE_DIR = Path(__file__).parent.parent / "vector_store"
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


def load_index():
    index = faiss.read_index(str(STORE_DIR / "index.faiss"))
    with open(STORE_DIR / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def search(query: str, index, metadata, model, k: int = TOP_K) -> list[dict]:
    query_vec = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    scores, ids = index.search(query_vec.astype(np.float32), k)

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
) -> list[dict]:
    """First stage: produce the candidate pool the reranker will reorder.

    fusion="none" is dense-only; "rrf" and "weighted" add BM25 and fuse. Both
    retrievers are asked for k candidates each, so fusion sees the same depth
    per retriever rather than splitting one budget between them.
    """
    dense = search(query, index, metadata, model, k=k)
    if fusion == "none":
        return dense

    from hybrid import bm25_search, build_bm25, fuse_rrf, fuse_weighted

    # Callers that run many queries should build this once and pass it in; the
    # eval harness does. Building per call is only acceptable for one-shot use.
    if bm25 is None:
        bm25 = build_bm25(metadata)

    sparse = bm25_search(query, bm25, metadata, k=k)
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
) -> list[dict]:
    """Full retrieval pipeline: shortlist, then optionally rerank.

    The two stages are independent knobs so they can be measured separately --
    fusion widens what the candidate pool contains, reranking reorders it.
    """
    if not use_reranker:
        return shortlist(query, index, metadata, model, k=k,
                         fusion=fusion, bm25=bm25, alpha=alpha)

    from rerank import rerank  # imported lazily so the baseline path stays light

    candidates = shortlist(query, index, metadata, model, k=candidate_k,
                           fusion=fusion, bm25=bm25, alpha=alpha)
    return rerank(query, candidates, k=k)


def main():
    query = " ".join(sys.argv[1:]) or input("Ask a question: ")

    index, metadata = load_index()
    model = SentenceTransformer(EMBEDDING_MODEL)

    results = retrieve(query, index, metadata, model)

    print(f"\nTop {len(results)} matches for: {query!r}\n")
    for rank, r in enumerate(results, start=1):
        score = r.get("rerank_score", r.get("score"))
        print(f"[{rank}] {r['source']}, page {r['page']} (score: {score:+.3f})")
        print(r["text"][:300].replace("\n", " ") + "...")
        print()


if __name__ == "__main__":
    main()
