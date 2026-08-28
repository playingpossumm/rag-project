"""Sparse (BM25) retrieval and dense/sparse fusion.

The dense retriever matches on meaning, which is what makes it robust to
paraphrase -- but it is correspondingly weak on tokens that carry meaning by
identity rather than by semantics: model numbers (P100), dataset names
(WMT 2014), symbols (d_model), acronyms. Those get averaged into a 384-dim
vector alongside everything else in the chunk and largely wash out.

BM25 has the opposite profile. It scores exact term overlap weighted by how
rare each term is across the corpus, so a rare token is a strong signal and a
paraphrase is invisible. Running both and fusing their rankings covers each
one's blind spot.
"""
import re

from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[a-z0-9]+")
RRF_K = 60  # damping constant from the original reciprocal-rank-fusion paper


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens.

    Deliberately crude: no stemming, no stopword list. Stemming would collapse
    distinctions this corpus depends on, and BM25's IDF weighting already
    discounts common words without needing a stoplist.
    """
    return TOKEN_RE.findall(text.lower())


def build_bm25(metadata: list[dict]) -> BM25Okapi:
    """Build the sparse index. Cheap enough to rebuild per process.

    Indexes `embed_text` -- the title-prefixed form -- so both retrievers see
    the same content. Lexical matching benefits most directly: a query naming a
    paper can now match its chunks on the title terms, which the passage text
    alone frequently never contains.
    """
    return BM25Okapi([tokenize(c.get("embed_text", c["text"])) for c in metadata])


def bm25_search(query: str, bm25: BM25Okapi, metadata: list[dict], k: int,
                allowed_ids: list[int] | None = None,
                terms: list[str] | None = None) -> list[dict]:
    # `terms` lets a caller hand in an already-expanded bag (see
    # query_expansion.py) without this module knowing how it was built.
    scores = bm25.get_scores(terms if terms is not None else tokenize(query))
    # BM25 scores the whole corpus regardless, so scoping is just a narrower
    # argmax -- exact, and no cheaper to approximate.
    pool = range(len(scores)) if allowed_ids is None else allowed_ids
    ranked = sorted(pool, key=lambda i: scores[i], reverse=True)[:k]
    return [
        {**metadata[i], "chunk_id": int(i), "score": float(scores[i])}
        for i in ranked
        if scores[i] > 0  # BM25 gives 0 when no query term occurs at all
    ]


def fuse_rrf_many(lists: list[list[dict]], k: int,
                  rrf_k: int = RRF_K) -> list[dict]:
    """Reciprocal rank fusion over any number of retrievers.

    Uses only *rank*, never the raw score, which is what makes it scale-free --
    cosine similarities and BM25 scores live on incomparable scales, so any
    method that adds them needs per-query normalization and inherits that
    normalization's failure modes. RRF sidesteps the problem entirely and has no
    weight to tune.

    Generalised to N lists on 2026-08-27 for the second dense retriever. Note
    what adding an arm does to the arithmetic: a passage found by two of three
    retrievers now scores below one found by all three, where before "found by
    both" was the ceiling. That is the intended behaviour and it is also why
    adding an arm is not free -- `sweep_ensemble.py` measured a case on the ML
    corpus where fusing a better model with a worse one landed between them.
    """
    fused: dict[int, dict] = {}
    for results in lists:
        for rank, r in enumerate(results, start=1):
            entry = fused.setdefault(r["chunk_id"], {**r, "fusion_score": 0.0})
            entry["fusion_score"] += 1.0 / (rrf_k + rank)

    out = sorted(fused.values(), key=lambda c: c["fusion_score"], reverse=True)
    return out[:k]


def fuse_rrf(dense: list[dict], sparse: list[dict], k: int,
             rrf_k: int = RRF_K) -> list[dict]:
    """Two-retriever RRF. Kept as the name every caller already uses."""
    return fuse_rrf_many([dense, sparse], k, rrf_k)


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def fuse_weighted(dense: list[dict], sparse: list[dict], k: int, alpha: float = 0.5) -> list[dict]:
    """Weighted score fusion: alpha * dense + (1 - alpha) * sparse.

    Scores are min-max normalized per query first, since the two retrievers'
    raw scales are unrelated. alpha=1.0 is dense-only, alpha=0.0 sparse-only.

    Normalization is per-query and therefore relative: a query where every
    candidate is mediocre still produces a 1.0 top score, so these values say
    nothing about absolute confidence. RRF avoids this; weighted fusion buys
    tunability at the cost of that distortion.
    """
    fused: dict[int, dict] = {}
    for results, weight in ((dense, alpha), (sparse, 1.0 - alpha)):
        normalized = _minmax([r["score"] for r in results])
        for r, norm in zip(results, normalized):
            entry = fused.setdefault(r["chunk_id"], {**r, "fusion_score": 0.0})
            entry["fusion_score"] += weight * norm

    out = sorted(fused.values(), key=lambda c: c["fusion_score"], reverse=True)
    return out[:k]
