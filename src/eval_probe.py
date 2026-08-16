"""Side-by-side probe: single-stage retrieval vs. retrieval + reranking.

Not the Phase 3 eval harness -- there are no labelled relevance judgements
here, so this reports observable ranking changes, not a quality metric. Its job
is to make the effect of a change visible before committing to it.
"""
import sys
import time

from sentence_transformers import SentenceTransformer

from rerank import load_reranker
from retrieve import CANDIDATE_K, EMBEDDING_MODEL, TOP_K, load_index, retrieve

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

QUERIES = [
    "What is multi-head attention?",
    "What optimizer and learning rate schedule was used for training?",
    "How does self-attention complexity compare to recurrent layers?",
    "What is positional encoding and why is it needed?",
    "How many parameters and what dropout rate did the base model use?",
    "Why is the dot product scaled by the square root of the key dimension?",
    "What BLEU score did the big model achieve on English-to-German?",
]


def fmt(results: list[dict], scored_by: str) -> list[str]:
    lines = []
    for rank, r in enumerate(results, start=1):
        snippet = " ".join(r["text"].split())[:64]
        lines.append(f"{rank}. p{r['page']:<3} {r[scored_by]:+.3f}  {snippet}")
    return lines


def main():
    index, metadata = load_index()
    model = SentenceTransformer(EMBEDDING_MODEL)
    load_reranker()  # warm up so timings exclude the model load

    print(f"index: {index.ntotal} vectors | candidates: {CANDIDATE_K} -> top {TOP_K}\n")

    moved = 0
    for q in QUERIES:
        t0 = time.perf_counter()
        base = retrieve(q, index, metadata, model, use_reranker=False)
        t_base = time.perf_counter() - t0

        t0 = time.perf_counter()
        rr = retrieve(q, index, metadata, model, use_reranker=True)
        t_rr = time.perf_counter() - t0

        changed = base[0]["text"] != rr[0]["text"]
        moved += changed

        print(f"Q: {q}")
        print(f"   baseline  ({t_base * 1000:5.0f} ms)")
        for line in fmt(base, "score"):
            print(f"     {line}")
        print(f"   reranked  ({t_rr * 1000:5.0f} ms){'   <-- rank 1 changed' if changed else ''}")
        for line in fmt(rr, "rerank_score"):
            print(f"     {line}")
        print()

    print(f"rank-1 changed on {moved}/{len(QUERIES)} queries")


if __name__ == "__main__":
    main()
