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

# RECALIBRATED for the 20-document corpus. The previous value (0.0) was tuned on
# a single-paper corpus where anything off-topic was WILDLY off-topic; that
# separation does not survive a corpus of adjacent ML papers, where almost any
# ML question finds plausibly-related material. This is the clearest evidence in
# the project that a threshold is a property of a corpus, not of a model --
# re-run `python src/evaluate.py` after any substantial corpus change.
#
# Measured on 23 answerable + 10 unanswerable cases:
#
#   answerable    min -2.92   median +5.05   max +8.75
#   unanswerable  min -8.96   median -3.12   max +1.09
#
# The distributions overlap, but not messily. Sorted, there is a clean gap:
#
#   highest unanswerable      +1.09   (adv-seed)
#   ---- threshold sits here ----
#   second-lowest answerable  +2.21   (p100)
#
# 1.5 sits in that gap. It flags 10/10 unanswerable questions and exactly one
# answerable case: `adam` at -2.92, which asks "which optimizer, and with what
# beta values" -- a question four papers answer differently. Low confidence
# there is arguably correct rather than a mistake: the system genuinely cannot
# tell which paper was meant.
#
# Note what this threshold does and does not do. `api.ask` NEVER withholds
# passages; the flag is advisory and the citations are always returned, so a
# reader can judge for themselves. Only `generate.py` refuses outright, because
# spending money to synthesise prose from a weak retrieval is the one case where
# proceeding has a real cost.
ABSTAIN_THRESHOLD = 1.5


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
