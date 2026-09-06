"""ColBERT-style late interaction, as a reranker over the candidate pool.

A bi-encoder compresses a passage into one vector before the question exists, so
everything the passage says is averaged together. A cross-encoder reads query
and passage jointly and is far more accurate, but computes a single opaque
score. Late interaction sits between them: keep a vector **per token**, and score
by MaxSim -- for each query token, its best match anywhere in the passage, summed.

    score(q, d) = sum over query tokens t of ( max over passage tokens s of cos(t, s) )

That is exactly what ColBERT computes, and it is the reason ColBERT is cited for
retrieving on rare tokens: a model number contributes its own similarity instead
of being averaged into 384 dimensions of surrounding prose.

**Where this departs from ColBERT proper, and why.** ColBERT precomputes token
embeddings for the whole corpus and searches them directly. Here that would be
2,768 chunks x ~210 tokens x 384 dims x 4 bytes ~= 890 MB of index for a 12 MB
corpus, to accelerate a stage that currently scans 2,768 vectors in single-digit
milliseconds. The cost is real and the benefit is nil at this scale, so token
embeddings are computed on demand for the shortlist only -- the same bargain the
cross-encoder makes, and for the same reason.

So this is late interaction as a **reranking** stage, not as a first-stage
retriever. It is genuinely MaxSim over token embeddings; it is not a drop-in
ColBERT index, and calling it one would be the kind of claim this project keeps
finding in its own documentation.
"""
import numpy as np

# The encoder's own ceiling. Chunks are built to sit under it (210 tokens plus
# a title prefix), so this only ever bites on an expanded passage.
MAX_TOKENS = 256


def _token_embeddings(texts: list[str], model) -> list[np.ndarray]:
    """Per-token vectors, L2-normalised so a dot product is a cosine.

    Normalising here rather than at scoring time means MaxSim is a plain matrix
    multiply, and it matches how the rest of the pipeline treats vectors -- the
    FAISS index stores unit vectors for exactly the same reason.
    """
    out = model.encode(texts, output_value="token_embeddings",
                       convert_to_numpy=False, show_progress_bar=False)
    if not isinstance(out, list):
        out = [out]
    normed = []
    for t in out:
        v = t.detach().cpu().numpy().astype(np.float32)[:MAX_TOKENS]
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        # A padding or all-zero token would divide by zero and poison the whole
        # matrix with NaN, which scores as neither high nor low but breaks sort.
        np.maximum(norms, 1e-12, out=norms)
        normed.append(v / norms)
    return normed


def maxsim(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> float:
    """Sum over query tokens of the best-matching passage token."""
    if query_vecs.size == 0 or doc_vecs.size == 0:
        return 0.0
    return float((query_vecs @ doc_vecs.T).max(axis=1).sum())


def rerank_late(query: str, candidates: list[dict], model,
                k: int | None = None, normalise: bool = True) -> list[dict]:
    """Re-score `candidates` by MaxSim and return them best first.

    `normalise` divides by the query's token count. Raw MaxSim grows with query
    length, so an unnormalised score is comparable between passages for ONE
    query and meaningless between queries -- which is fine for ranking and wrong
    for anything that compares a score to a threshold, as the abstention gate
    does. Dividing makes it a mean per query token and bounded by 1.
    """
    if not candidates:
        return []

    q = _token_embeddings([query], model)[0]
    docs = _token_embeddings([c["text"] for c in candidates], model)

    denom = max(len(q), 1) if normalise else 1
    scored = [
        {**c, "late_score": maxsim(q, d) / denom}
        for c, d in zip(candidates, docs)
    ]
    scored.sort(key=lambda c: c["late_score"], reverse=True)
    return scored[:k] if k else scored
