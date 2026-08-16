from pathlib import Path
import json
import sys

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

STORE_DIR = Path(__file__).parent.parent / "vector_store"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5


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


def main():
    query = " ".join(sys.argv[1:]) or input("Ask a question: ")

    index, metadata = load_index()
    model = SentenceTransformer(EMBEDDING_MODEL)

    results = search(query, index, metadata, model)

    print(f"\nTop {len(results)} matches for: {query!r}\n")
    for rank, r in enumerate(results, start=1):
        print(f"[{rank}] {r['source']}, page {r['page']} (score: {r['score']:.3f})")
        print(r["text"][:300].replace("\n", " ") + "...")
        print()


if __name__ == "__main__":
    main()
