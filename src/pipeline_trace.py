"""Record what each retrieval stage did, so the pipeline can be inspected.

`retrieve()` returns five passages. Everything that produced them -- which
retriever found what, how fusion reconciled two disagreeing rankings, what the
reranker promoted or buried, which passages the diversity cap displaced -- is
discarded on the way out. That is fine for serving and useless for understanding,
and "why did it return THAT" is the question actually worth answering, both when
debugging a bad result and when explaining the system to someone.

This re-runs the same stages and keeps the intermediate rankings. It deliberately
does not modify retrieve(): a serving path that carries debug state is a serving
path that eventually ships debug state, and the two have genuinely different
requirements -- one wants to be fast and forget, this one wants to be slow and
remember.

The cost of that choice is duplication: the stage order here must match
retrieve(). test_trace_matches_pipeline() asserts the two agree on the final
result, which is what makes the duplication safe to keep.
"""
from dataclasses import dataclass, field

from diversify import DEFAULT_MAX_PER_SOURCE
from hybrid import RRF_K, bm25_search, tokenize
from retrieve import CANDIDATE_K, TOP_K, search


@dataclass
class Stage:
    """One step of the pipeline, with the ranking it produced."""
    name: str
    title: str
    detail: str
    items: list[dict] = field(default_factory=list)
    note: str = ""


def _brief(chunk: dict, limit: int = 260) -> str:
    return " ".join(chunk["text"].split())[:limit]


def _identity(chunk: dict) -> dict:
    """The fields the UI needs to display and correlate a chunk across stages."""
    loc = chunk.get("locator", {})
    return {
        "chunk_id": chunk["chunk_id"],
        "source": chunk["source"],
        "locator": f"{loc.get('kind', '?')} {loc.get('value', '?')}",
        "text": _brief(chunk),
    }


def trace_pipeline(
    query: str,
    index,
    metadata,
    model,
    bm25=None,
    k: int = TOP_K,
    candidate_k: int = CANDIDATE_K,
    max_per_source: int | None = DEFAULT_MAX_PER_SOURCE,
) -> dict:
    """Run retrieval, recording every intermediate ranking.

    Returns stages in pipeline order plus a `verdict` summarising the outcome.
    """
    from abstain import ABSTAIN_THRESHOLD
    from hybrid import build_bm25
    from rerank import rerank

    if bm25 is None:
        bm25 = build_bm25(metadata)

    stages: list[Stage] = []

    # ---- stage 1 & 2: the two retrievers, run independently ----------------
    dense = search(query, index, metadata, model, k=candidate_k)
    sparse = bm25_search(query, bm25, metadata, k=candidate_k)

    dense_rank = {c["chunk_id"]: i for i, c in enumerate(dense, 1)}
    sparse_rank = {c["chunk_id"]: i for i, c in enumerate(sparse, 1)}

    stages.append(Stage(
        "dense", "Dense retrieval",
        "Embeds the query and finds the nearest chunk vectors by cosine "
        "similarity. Matches meaning, so it survives paraphrase -- and misses "
        "rare tokens that carry meaning by identity, like model numbers.",
        [{**_identity(c), "rank": i, "score": round(c["score"], 4),
          "also_found_by": sparse_rank.get(c["chunk_id"])}
         for i, c in enumerate(dense, 1)],
        note=f"all-MiniLM-L6-v2, {index.ntotal} chunks searched",
    ))

    query_terms = tokenize(query)
    stages.append(Stage(
        "sparse", "Sparse retrieval (BM25)",
        "Scores exact term overlap, weighting rare terms far above common ones. "
        "The exact inverse profile: strong on identifiers and acronyms, blind to "
        "paraphrase. Returns nothing when no query term appears at all.",
        [{**_identity(c), "rank": i, "score": round(c["score"], 3),
          "also_found_by": dense_rank.get(c["chunk_id"])}
         for i, c in enumerate(sparse, 1)],
        note=f"{len(query_terms)} query terms: {' '.join(query_terms[:12])}",
    ))

    # ---- stage 3: fusion ---------------------------------------------------
    # Recomputed here rather than calling fuse_rrf, so each contribution can be
    # attributed to the retriever it came from. The arithmetic is identical.
    fused: dict[int, dict] = {}
    for results, which in ((dense, "dense"), (sparse, "sparse")):
        for rank, r in enumerate(results, start=1):
            entry = fused.setdefault(r["chunk_id"], {**r, "fusion_score": 0.0,
                                                   "contrib": {}})
            entry["fusion_score"] += 1.0 / (RRF_K + rank)
            entry["contrib"][which] = rank

    fused_list = sorted(fused.values(), key=lambda c: c["fusion_score"],
                        reverse=True)[:candidate_k]

    both = sum(1 for c in fused_list if len(c["contrib"]) == 2)
    stages.append(Stage(
        "fused", "Reciprocal rank fusion",
        "Combines the two rankings using only position, never raw score: each "
        "retriever contributes 1/(60+rank). Cosine similarity and BM25 live on "
        "incomparable scales, so anything that adds them needs normalisation and "
        "inherits its failure modes. RRF has no weight to tune.",
        [{**_identity(c), "rank": i, "score": round(c["fusion_score"], 5),
          "dense_rank": c["contrib"].get("dense"),
          "sparse_rank": c["contrib"].get("sparse"),
          "agreement": len(c["contrib"]) == 2}
         for i, c in enumerate(fused_list, 1)],
        note=f"{len(fused_list)} candidates, {both} found by both retrievers",
    ))

    # ---- stage 4: reranking ------------------------------------------------
    fused_rank = {c["chunk_id"]: i for i, c in enumerate(fused_list, 1)}
    ranked = rerank(query, fused_list, k=len(fused_list))

    moves = []
    for i, c in enumerate(ranked, 1):
        was = fused_rank[c["chunk_id"]]
        moves.append({**_identity(c), "rank": i,
                      "score": round(float(c["rerank_score"]), 3),
                      "was": was, "delta": was - i})

    biggest = max(moves, key=lambda m: abs(m["delta"])) if moves else None
    stages.append(Stage(
        "reranked", "Cross-encoder reranking",
        "Re-scores every candidate by reading the query and the passage together "
        "in one pass, so it can weigh how they relate rather than comparing two "
        "independently-made vectors. Far more accurate, and impossible to "
        "precompute -- which is why it only runs on the shortlist.",
        moves,
        note=(f"largest move: {biggest['source']} {biggest['delta']:+d} places"
              if biggest and biggest["delta"] else "ranking largely unchanged"),
    ))

    # ---- stage 5: diversity cap -------------------------------------------
    counts: dict[str, int] = {}
    selected, displaced = [], []
    cap = max_per_source or 0
    for m in moves:
        if cap and counts.get(m["source"], 0) >= cap:
            displaced.append(m)
            continue
        counts[m["source"]] = counts.get(m["source"], 0) + 1
        selected.append(m)
        if len(selected) == k:
            break

    filled = (selected + displaced)[:k] if len(selected) < k else selected
    final = sorted(filled, key=lambda m: m["score"], reverse=True)

    stages.append(Stage(
        "selected", "Per-document diversity cap",
        f"Takes at most {cap} passages from any one document. Rerankers score "
        "each passage independently, so a single strong document takes every "
        "slot -- which ranking metrics score as perfect, and which is wrong when "
        "the goal is to compile every relevant source rather than find one.",
        [{**m, "rank": i} for i, m in enumerate(final, 1)],
        note=(f"{len(displaced)} passage(s) displaced by the cap"
              if displaced else "cap not reached; nothing displaced"),
    ))

    # ---- the gate ----------------------------------------------------------
    top = final[0]["score"] if final else float("-inf")
    confident = top >= ABSTAIN_THRESHOLD
    sources = sorted({m["source"] for m in final})

    return {
        "query": query,
        "stages": [s.__dict__ for s in stages],
        "verdict": {
            "confident": confident,
            "confidence": round(top, 3) if final else None,
            "threshold": ABSTAIN_THRESHOLD,
            "documents": sources,
            "multi_document": len(sources) > 1,
            "explanation": (
                f"Top passage scores {top:+.2f} against a threshold of "
                f"{ABSTAIN_THRESHOLD:+.2f}. "
                + ("Answering." if confident else
                   "Below threshold -- the corpus likely does not contain this, "
                   "so the honest response is to say so rather than return the "
                   "closest topical match.")
            ) if final else "No candidates retrieved.",
        },
    }
