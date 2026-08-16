"""Second-stage reranking with a cross-encoder.

The first-stage bi-encoder (retrieve.py) compares pre-computed vectors, so the
query and a chunk never actually meet -- each was summarised into a single
point independently. That makes it fast enough to scan the whole corpus, but it
scores topical *aboutness* rather than whether a passage answers the question,
which is why short keyword-dense fragments outrank longer precise ones.

A cross-encoder encodes query and chunk *together* in one forward pass, letting
every query token attend to every chunk token. Far more accurate, far too slow
to run over a whole corpus -- so it runs only over the candidates the
bi-encoder already shortlisted.
"""
from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = None


def load_reranker(model_name: str = RERANK_MODEL) -> CrossEncoder:
    """Load and cache the cross-encoder (it is reused across queries)."""
    global _model
    if _model is None:
        _model = CrossEncoder(model_name)
    return _model


def rerank(query: str, candidates: list[dict], k: int, model=None) -> list[dict]:
    """Re-score candidates against the query jointly, returning the best k.

    Each candidate keeps its original bi-encoder `score` under `retrieval_score`
    so the two stages can be compared, and gains a `rerank_score`. Cross-encoder
    scores are raw logits -- ordering is meaningful, absolute values are not, and
    they are not comparable to cosine similarities.
    """
    if not candidates:
        return []

    model = model or load_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    ranked = [
        {**c, "retrieval_score": c.get("score"), "rerank_score": float(s)}
        for c, s in zip(candidates, scores)
    ]
    ranked.sort(key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:k]
