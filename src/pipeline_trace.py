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
retrieve(). test_trace.py asserts the two agree on the final result under every
option combination, which is what makes the duplication safe to keep.
"""
from dataclasses import dataclass, field

from diversify import DEFAULT_MAX_PER_SOURCE
# _minmax is imported rather than reimplemented: this module already duplicates
# the stage order, and a second copy of the normalisation would be a second
# thing to keep in step for no gain.
from hybrid import RRF_K, _minmax, bm25_search, tokenize
from retrieve import CANDIDATE_K, DEFAULT_FUSION, TOP_K, search


@dataclass
class Stage:
    """One step of the pipeline, with the ranking it produced.

    `skipped` carries the reason an option switched this stage off. The stage is
    still emitted, empty -- see the note on stage order in trace_pipeline().
    """
    name: str
    title: str
    detail: str
    items: list[dict] = field(default_factory=list)
    note: str = ""
    skipped: str = ""


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
    use_reranker: bool = True,
    fusion: str = DEFAULT_FUSION,
    alpha: float = 0.5,
    expansion: str = "none",
    threshold: float | None = None,
    # How much of the first stage's ordering survives reranking, per corpus.
    rerank_blend: float | None = None,
) -> dict:
    """Run retrieval, recording every intermediate ranking.

    Returns stages in pipeline order plus a `verdict` summarising the outcome.

    The option arguments mirror retrieve()'s, so one query can be re-run under a
    different configuration and the two traces compared. **Five stages are
    always emitted, in the same order, even when an option switches one off**: a
    stage that did not run reports `skipped` with the reason instead of
    vanishing. Comparing two runs then means comparing stage 3 with stage 3
    rather than realigning two lists of different lengths, and the map can
    darken a tower rather than have a building disappear between runs -- which
    would read as the city being rebuilt by the question, the thing its whole
    layout exists to deny.
    """
    from abstain import ABSTAIN_THRESHOLD
    from hybrid import build_bm25
    from rerank import rerank

    # The gate's cut point belongs to the CORPUS, not to this module. Measured
    # across three of them, the same 0.0 wrongly refuses 1.5% of answerable
    # questions on ML papers, 38.5% on ornithology and 54.3% on quantitative
    # finance. A caller serving more than one corpus must pass the value that
    # corpus was calibrated at; the module constant is only a default for
    # callers that have one.
    if threshold is None:
        threshold = ABSTAIN_THRESHOLD

    if fusion not in ("none", "rrf", "weighted"):
        raise ValueError(f"unknown fusion strategy: {fusion!r}")
    if expansion not in ("none", "window", "page"):
        raise ValueError(f"unknown expansion mode: {expansion!r}")

    stages: list[Stage] = []

    # retrieve() takes a DIFFERENT path when reranking is off: it shortlists k
    # rather than candidate_k, and skips the diversity cap entirely, because
    # both exist to feed and then trim a reranked pool. Mirroring that here is
    # the whole point -- a trace showing a 20-candidate pool and a cap under
    # settings the serving path would run without either is a picture of a
    # pipeline that does not exist, which is worse than no picture.
    pool_k = candidate_k if use_reranker else k
    cap = (max_per_source or 0) if use_reranker else 0

    # ---- stage 1 & 2: the two retrievers, run independently ----------------
    dense = search(query, index, metadata, model, k=pool_k)

    # fusion="none" is dense-only, so BM25 genuinely does not run. Building it
    # anyway to report a ranking nothing consumed would make the trace show work
    # the pipeline did not do.
    if fusion == "none":
        sparse = []
    else:
        if bm25 is None:
            bm25 = build_bm25(metadata)
        sparse = bm25_search(query, bm25, metadata, k=pool_k)

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
        note=(f"{len(query_terms)} query terms: {' '.join(query_terms[:12])}"
              if sparse else "not run"),
        skipped="" if fusion != "none" else
                "Fusion is off, so only the dense retriever runs and the lexical "
                "half of the pipeline contributes nothing.",
    ))

    # ---- stage 3: fusion ---------------------------------------------------
    # Recomputed here rather than calling fuse_rrf/fuse_weighted, so each
    # contribution can be attributed to the retriever it came from. The
    # arithmetic is identical to hybrid.py's.
    fused: dict[int, dict] = {}
    if fusion == "weighted":
        for results, which, weight in ((dense, "dense", alpha),
                                       (sparse, "sparse", 1.0 - alpha)):
            normalized = _minmax([r["score"] for r in results])
            for rank, (r, norm) in enumerate(zip(results, normalized), start=1):
                entry = fused.setdefault(r["chunk_id"], {**r, "fusion_score": 0.0,
                                                        "contrib": {}})
                entry["fusion_score"] += weight * norm
                entry["contrib"][which] = rank
    else:
        # RRF, and dense-only: dense-only is RRF over one list, which preserves
        # dense's order exactly, so the same arithmetic covers both and the
        # stage stays comparable across runs instead of switching score scale.
        for results, which in ((dense, "dense"), (sparse, "sparse")):
            for rank, r in enumerate(results, start=1):
                entry = fused.setdefault(r["chunk_id"], {**r, "fusion_score": 0.0,
                                                        "contrib": {}})
                entry["fusion_score"] += 1.0 / (RRF_K + rank)
                entry["contrib"][which] = rank

    fused_list = sorted(fused.values(), key=lambda c: c["fusion_score"],
                        reverse=True)[:pool_k]
    pool = {c["chunk_id"]: c for c in fused_list}

    both = sum(1 for c in fused_list if len(c["contrib"]) == 2)
    stages.append(Stage(
        "fused", {"none": "Fusion (off -- dense only)",
                  "rrf": "Reciprocal rank fusion",
                  "weighted": f"Weighted fusion (alpha {alpha:g})"}[fusion],
        "Combines the two rankings using only position, never raw score: each "
        "retriever contributes 1/(60+rank). Cosine similarity and BM25 live on "
        "incomparable scales, so anything that adds them needs normalisation and "
        "inherits its failure modes. RRF has no weight to tune."
        if fusion != "weighted" else
        "Min-max normalises each retriever's scores per query, then adds them "
        "weighted by alpha. Tunable, at the cost of a normalisation that is "
        "relative: a query where everything is mediocre still produces a 1.0.",
        [{**_identity(c), "rank": i, "score": round(c["fusion_score"], 5),
          "dense_rank": c["contrib"].get("dense"),
          "sparse_rank": c["contrib"].get("sparse"),
          "agreement": len(c["contrib"]) == 2}
         for i, c in enumerate(fused_list, 1)],
        note=(f"{len(fused_list)} candidates, {both} found by both retrievers"
              if fusion != "none" else
              f"{len(fused_list)} candidates, dense order unchanged"),
        skipped="" if fusion != "none" else
                "Nothing to fuse: with one retriever the candidate pool is the "
                "dense ranking as it stands.",
    ))

    # ---- stage 4: reranking ------------------------------------------------
    fused_rank = {c["chunk_id"]: i for i, c in enumerate(fused_list, 1)}
    if use_reranker:
        ranked = rerank(query, fused_list, k=len(fused_list),
                        blend=rerank_blend)
        moves = [{**_identity(c), "rank": i,
                  "score": round(float(c["rerank_score"]), 3),
                  "was": fused_rank[c["chunk_id"]],
                  "delta": fused_rank[c["chunk_id"]] - i}
                 for i, c in enumerate(ranked, 1)]
    else:
        # The candidate pool goes to selection in the order the first stage left
        # it. Scores stay on the fusion scale -- substituting a rerank score
        # would be inventing the number this run deliberately did not compute.
        moves = [{**_identity(c), "rank": i,
                  "score": round(float(c["fusion_score"]), 5),
                  "was": i, "delta": 0}
                 for i, c in enumerate(fused_list, 1)]

    biggest = max(moves, key=lambda m: abs(m["delta"])) if moves else None
    stages.append(Stage(
        "reranked", "Cross-encoder reranking" if use_reranker
                    else "Cross-encoder reranking (off)",
        "Re-scores every candidate by reading the query and the passage together "
        "in one pass, so it can weigh how they relate rather than comparing two "
        "independently-made vectors. Far more accurate, and impossible to "
        "precompute -- which is why it only runs on the shortlist.",
        moves if use_reranker else [],
        note=((f"largest move: {biggest['source']} {biggest['delta']:+d} places"
               if biggest and biggest["delta"] else "ranking largely unchanged")
              if use_reranker else "not run"),
        skipped="" if use_reranker else
                "Reranking is off, so the first stage's ranking goes straight to "
                "selection. This is the naive-RAG baseline: MRR 0.601 against "
                "0.710 with the cross-encoder, on the 84-case set.",
    ))

    # ---- stage 5: diversity cap -------------------------------------------
    counts: dict[str, int] = {}
    selected, displaced = [], []
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

    # ---- context expansion -------------------------------------------------
    # Not a sixth stage: expansion changes how much text each survivor carries
    # and cannot move a ranking, so giving it a rank column would invite reading
    # it as one. It is reported on the passages it grew.
    context = {"mode": expansion, "blocks": len(final), "chars": 0, "merged": 0}
    if final:
        from parent import expand

        grown = expand([pool[m["chunk_id"]] for m in final], metadata,
                       mode=expansion) if expansion != "none" else []
        by_id = {g["chunk_id"]: g for g in grown}
        for m in final:
            g = by_id.get(m["chunk_id"])
            if g:
                m["context"] = {"text": " ".join(g["text"].split()),
                                "merged": g.get("n_chunks_merged", 1)}
        context["blocks"] = len(grown) if expansion != "none" else len(final)
        context["merged"] = len(final) - context["blocks"]
        context["chars"] = sum(len(g["text"]) for g in grown) if grown else \
            sum(len(pool[m["chunk_id"]]["text"]) for m in final)

    stages.append(Stage(
        "selected", "Per-document diversity cap" if cap else "Selection (no cap)",
        (f"Takes at most {cap} passages from any one document. Rerankers score "
         "each passage independently, so a single strong document takes every "
         "slot -- which ranking metrics score as perfect, and which is wrong when "
         "the goal is to compile every relevant source rather than find one.")
        if cap else
        "The cap is off, so selection is the top k of the ranking as it stands. "
        "Measured at +0.031 source recall for -0.016 any-hit when set to 2: a "
        "real trade, not a free win.",
        [{**m, "rank": i} for i, m in enumerate(final, 1)],
        note=((f"{len(displaced)} passage(s) displaced by the cap"
               if displaced else "cap not reached; nothing displaced")
              if cap else f"top {len(final)}, no cap applied"),
        skipped="" if cap or use_reranker else
                "The cap trims a reranked pool, so retrieve() does not apply it "
                "on the no-rerank path at all -- the shortlist of k is the "
                "result. Setting a cap here changes nothing until reranking is "
                "back on.",
    ))

    # ---- the gate ----------------------------------------------------------
    sources = sorted({m["source"] for m in final})
    top = final[0]["score"] if final else float("-inf")

    # The threshold is calibrated on cross-encoder scores (abstain.py). Fusion
    # scores are a different quantity on a different scale -- RRF is bounded
    # near 0.03 and always positive, so testing it against 0.0 would report
    # every query as confident, including the ones the corpus cannot answer.
    # A gate that cannot fail is worse than no gate, so with reranking off this
    # says it has no verdict rather than manufacturing one.
    gated = use_reranker and bool(final)
    confident = (top >= threshold) if gated else None

    if not final:
        explanation = "No candidates retrieved."
    elif gated:
        explanation = (
            f"Top passage scores {top:+.2f} against a threshold of "
            f"{threshold:+.2f}. "
            + ("Answering." if confident else
               "Below threshold -- the corpus likely does not contain this, "
               "so the honest response is to say so rather than return the "
               "closest topical match.")
        )
    else:
        explanation = (
            "No gate. The abstention threshold is calibrated on cross-encoder "
            "scores, and with reranking off none were computed -- the top score "
            f"here is a fusion score ({top:.5f}), which is a different quantity "
            "on a different scale. Comparing it to the threshold would pass "
            "every query, including the ones that should be refused."
        )

    return {
        "query": query,
        "options": {
            "k": k, "candidate_k": candidate_k, "use_reranker": use_reranker,
            "fusion": fusion, "alpha": alpha, "max_per_source": max_per_source,
            "expansion": expansion,
        },
        "stages": [s.__dict__ for s in stages],
        "context": context,
        # The cap's own count, rather than leaving every consumer to infer one.
        # The map inferred "candidates in the top 6 that did not survive", which
        # is a different quantity: at k=5 it reads 1 for every query, including
        # the ones where the cap displaced nothing at all.
        "selection": {"cap": cap, "kept": len(final), "displaced": len(displaced)},
        "verdict": {
            "confident": confident,
            "gated": gated,
            "confidence": round(top, 5 if not use_reranker else 3) if final else None,
            "threshold": threshold,
            "documents": sources,
            "multi_document": len(sources) > 1,
            "explanation": explanation,
        },
    }
