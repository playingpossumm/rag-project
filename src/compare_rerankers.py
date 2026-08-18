"""Test whether a stronger reranker honours the distinguishing clause in a query.

Diagnosis from src/diagnose_crossdoc.py: ~15% of questions return a top result
from a document that does not answer them, and the failures share a mechanism --
the retriever matches the TOPIC and ignores the CONSTRAINT that distinguishes the
answer. "How are normalization statistics computed across features rather than
examples" returns Batch Normalization, which is the concept the clause rules out.

Reading query and passage jointly is exactly what a cross-encoder is for, so the
hypothesis is that the current one (6-layer MiniLM) is too small to represent
"rather than examples" as a constraint rather than as more topic words.

This measures that directly on the cases that fail, rather than swapping the
model and hoping the aggregate moves.
"""
import sys
import time

from sentence_transformers import CrossEncoder, SentenceTransformer

from evaluate import gold_sources, load_cases
from hybrid import build_bm25
from retrieve import EMBEDDING_MODEL, load_index, shortlist

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODELS = [
    ("MiniLM-L6  (current)", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    ("MiniLM-L12", "cross-encoder/ms-marco-MiniLM-L-12-v2"),
]

# The cases where the top result came from a non-answering document, minus the
# three judged to be label narrowness rather than system error.
FAILING = ["dropout-rate", "wmt14", "colbert-msmarco", "gpt3-params",
           "gpt3-fewshot", "ln-stats", "roberta-nsp-drop", "sbert-speed",
           "seq2seq-reverse", "vit-jft"]


def main():
    answerable, _ = load_cases()
    cases = {c["id"]: c for c in answerable}
    index, metadata = load_index()
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    bm25 = build_bm25(metadata)

    # Candidate pools are identical across rerankers, so any difference is the
    # reranker's doing and not the first stage's.
    pools = {}
    for case in answerable:
        pools[case["id"]] = shortlist(case["question"], index, metadata, embedder,
                                      k=20, fusion="rrf", bm25=bm25)

    for label, name in MODELS:
        print(f"\n=== {label} ===")
        try:
            model = CrossEncoder(name)
        except Exception as exc:  # noqa: BLE001
            print(f"  unavailable: {exc}")
            continue

        started = time.perf_counter()
        top1_correct = fixed = 0
        for case in answerable:
            pool = pools[case["id"]]
            scores = model.predict([(case["question"], c["text"]) for c in pool])
            best = max(zip(scores, pool), key=lambda p: p[0])[1]
            hit = best["source"] in gold_sources(case)
            top1_correct += hit
            if case["id"] in FAILING and hit:
                fixed += 1
        elapsed = time.perf_counter() - started

        n = len(answerable)
        print(f"  top-1 from an answering document: {top1_correct}/{n} ({top1_correct/n:.1%})")
        print(f"  previously-failing cases fixed:   {fixed}/{len(FAILING)}")
        print(f"  time: {elapsed:.0f}s for {n} queries ({elapsed/n*1000:.0f} ms/query)")


if __name__ == "__main__":
    main()
