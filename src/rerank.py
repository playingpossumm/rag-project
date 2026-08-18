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


# Scores are a pure function of (query, chunk text, model), so they cache. This
# matters most for evaluation: the harness runs six pipeline configurations over
# the same questions, and configurations that differ only in fusion or diversity
# hand the reranker largely the SAME candidates. Without caching, each one pays
# for scoring pairs an earlier configuration already scored.
#
# Process-local rather than on disk: within an eval run the reuse is high, while
# across runs the queries usually change, so persisting it would mostly store
# entries that are never read again.
_score_cache: dict[tuple[str, str], float] = {}


def cache_stats() -> dict:
    return {"entries": len(_score_cache)}


def clear_cache() -> None:
    _score_cache.clear()


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

    # Score only the pairs not already known, then reassemble in input order.
    todo = [(i, c) for i, c in enumerate(candidates)
            if (query, c["text"]) not in _score_cache]
    if todo:
        fresh = model.predict([(query, c["text"]) for _i, c in todo])
        for (_i, c), score in zip(todo, fresh):
            _score_cache[(query, c["text"])] = float(score)

    ranked = [
        {**c, "retrieval_score": c.get("score"),
         "rerank_score": _score_cache[(query, c["text"])]}
        for c in candidates
    ]
    ranked.sort(key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:k]
