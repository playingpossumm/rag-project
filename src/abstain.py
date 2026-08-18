"""Confidence gating: decide whether the corpus can answer at all.

Dense retrieval has no notion of "nothing here". Nearest-neighbour search always
returns k results -- ask a corpus about a topic it has never heard of and it
still hands back its five least-unrelated chunks, at scores indistinguishable
from a genuine match. A generator handed that context is being invited to
confabulate, and citations make the confabulation look authoritative.

The cross-encoder scores differently. Because it reads query and passage
together and asks "does this answer that", it can express genuine non-relevance
as a strongly negative logit -- a signal cosine similarity structurally cannot
produce, since every vector has some cosine to every other.

The threshold here is calibrated in evaluate.py against labelled answerable and
unanswerable cases rather than chosen by intuition. See eval/RESULTS.md.
"""

# Calibrated on 24 answerable + 16 unanswerable cases (`python src/evaluate.py`).
#
#   answerable    min +1.88   median +4.78   max  +9.24
#   unanswerable  min -11.06  median -8.10   max  +3.95
#
# Zero is chosen deliberately over the sweep's arithmetic optimum of +1, which
# scored marginally better on this sample. Two reasons:
#
#   1. The lowest answerable case sits at +1.88. A threshold of +1 leaves 0.88
#      of margin on a 24-case sample -- that is tuning to the edge of the data
#      rather than to the signal, and the first unseen hard-but-answerable
#      question would be refused.
#   2. Zero is where the logit's own semantics put the boundary: positive means
#      the passage answers the query, negative means it does not. A threshold
#      that needs justifying is worse than one that is already meaningful.
#
# At zero the gate catches 5/5 "absent topic" and 4/4 "absent metadata" cases --
# every question about something the corpus has never heard of. The two that get
# through are both near-misses where the paper genuinely discusses the adjacent
# material (decoder depth, training cost in FLOPs rather than dollars). Returning
# cited context there is defensible: the citation lets a reader see for
# themselves that the cited page answers a neighbouring question, which is a far
# better failure than a confident answer about diffusion models.
ABSTAIN_THRESHOLD = 0.0


def top_score(results: list[dict]) -> float:
    """Confidence of the best result, or -inf when nothing was retrieved."""
    if not results:
        return float("-inf")
    # Only rerank_score is calibrated for this; cosine cannot separate
    # answerable from unanswerable (measured -- see RESULTS.md).
    if "rerank_score" not in results[0]:
        raise ValueError(
            "abstention requires reranked results; cosine scores are not separable"
        )
    return results[0]["rerank_score"]


def should_abstain(results: list[dict], threshold: float = ABSTAIN_THRESHOLD) -> bool:
    """True when the best candidate is too weak to answer from."""
    return top_score(results) < threshold


def abstention_message(query: str) -> str:
    return (
        f"The indexed documents do not appear to contain an answer to: {query!r}\n"
        "Rather than answer from general knowledge, which would be ungrounded and "
        "uncitable, the system is declining. Try rephrasing, or check whether the "
        "relevant document has been ingested."
    )
