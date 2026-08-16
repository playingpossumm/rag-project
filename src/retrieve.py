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
        results.append({**chunk, "score": float(score)})
    return results


def retrieve(
    query: str,
    index,
    metadata,
    model,
    k: int = TOP_K,
    candidate_k: int = CANDIDATE_K,
    use_reranker: bool = True,
) -> list[dict]:
    """Full retrieval pipeline: bi-encoder shortlist, then cross-encoder rerank.

    With use_reranker=False this is the single-stage baseline, which is what the
    reranked pipeline is measured against.
    """
    if not use_reranker:
        return search(query, index, metadata, model, k=k)

    from rerank import rerank  # imported lazily so the baseline path stays light

    candidates = search(query, index, metadata, model, k=candidate_k)
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
