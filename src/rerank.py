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


# How much the first stage still counts once the cross-encoder has spoken.
#
# At 0.0 the cross-encoder's ordering replaces the first stage's outright, which
# is what this did for its whole life. That discards real evidence. Retrieval
# and reranking disagree in a specific way: the bi-encoder scores topical
# aboutness and the cross-encoder scores whether a passage answers, and when the
# question describes a term rather than naming it -- "the burst of collective
# singing at first light" -- the cross-encoder has little to work with and its
# ordering is close to arbitrary. Asked that, fusion ranked the right passage
# FIRST and reranking moved it to seventh.
#
# Blending is on RANKS, not scores. Cross-encoder outputs are raw logits and
# fusion outputs are RRF values; neither is calibrated and they are not on a
# common scale, so combining the numbers would be meaningless. Combining the
# orderings is the same reciprocal-rank trick already used to fuse dense and
# BM25, applied one stage later.
# 0.0 by default: the cross-encoder decides alone, which is what every corpus
# but one measured best with. Set per corpus in corpora.json, the same way the
# abstention threshold is, and for the same reason -- it does not transfer.
RERANK_BLEND = 0.0
RRF_K = 60


def rerank(query: str, candidates: list[dict], k: int, model=None,
           blend: float | None = None) -> list[dict]:
    """Re-score candidates against the query jointly, returning the best k.

    Each candidate keeps its original bi-encoder `score` under `retrieval_score`
    so the two stages can be compared, and gains a `rerank_score`. Cross-encoder
    scores are raw logits -- ordering is meaningful, absolute values are not, and
    they are not comparable to cosine similarities.

    `blend` is how much weight the first stage's ordering keeps; 0.0 reproduces
    the old behaviour of letting the cross-encoder decide alone.
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

    w = RERANK_BLEND if blend is None else blend
    ranked = [
        {**c, "retrieval_score": c.get("score"),
         "rerank_score": _score_cache[(query, c["text"])],
         "first_stage_rank": i + 1}
        for i, c in enumerate(candidates)
    ]

    if w <= 0:
        ranked.sort(key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:k]

    # Rank by the cross-encoder, then fuse that ordering with the one the
    # candidates arrived in.
    by_ce = sorted(ranked, key=lambda c: c["rerank_score"], reverse=True)
    ce_rank = {id(c): i + 1 for i, c in enumerate(by_ce)}
    for c in ranked:
        c["blended_score"] = (
            (1 - w) / (RRF_K + ce_rank[id(c)])
            + w / (RRF_K + c["first_stage_rank"])
        )
    ranked.sort(key=lambda c: c["blended_score"], reverse=True)
    return ranked[:k]
