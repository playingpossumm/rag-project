"""Ad-hoc retrieval probe: run several queries and print compact top-k results.

Not the Phase 3 eval harness -- this is a quick sanity check for comparing
retrieval behaviour before and after a chunking change.
"""
import sys

from sentence_transformers import SentenceTransformer

from retrieve import EMBEDDING_MODEL, load_index, search

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

QUERIES = [
    "What is multi-head attention?",
    "What optimizer and learning rate schedule was used for training?",
    "How does self-attention complexity compare to recurrent layers?",
    "What is positional encoding and why is it needed?",
    "How many parameters and what dropout rate did the base model use?",
]


def main():
    index, metadata = load_index()
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"index: {index.ntotal} vectors\n")

    for q in QUERIES:
        results = search(q, index, metadata, model, k=5)
        print(f"Q: {q}")
        for rank, r in enumerate(results, start=1):
            snippet = " ".join(r["text"].split())[:88]
            print(f"   {rank}. p{r['page']:<3} {r['score']:.3f}  {snippet}")
        print()


if __name__ == "__main__":
    main()
