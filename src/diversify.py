"""Spread results across documents instead of stacking them on the best one.

Rerankers score each (query, passage) pair independently, so when one document
matches well its passages take every slot. Measured on this corpus: for "which
model achieved 28.4 BLEU", five papers report that figure and all five returned
passages came from one of them -- MRR 1.000, source recall 0.200. Ranking
metrics call that a perfect result, because by their definition it is.

That is the wrong definition when the goal is to compile every relevant source
rather than find one. This applies a per-document cap during selection: walk the
ranked list, take a passage if its document is not already at the cap, and keep
the skipped ones as fallback so k results are still returned when few documents
qualify.

A cap, not MMR. Maximal Marginal Relevance diversifies on embedding distance,
which mixes two different notions of redundancy -- similar wording and same
source -- and needs a lambda tuned per corpus. Here the unit of redundancy is
known exactly: the document. Capping it directly is predictable, has one
interpretable knob, and cannot be defeated by two documents that happen to
phrase the same fact differently.
"""

# Measured at k=5 on the 20-document corpus (weighted fusion + reranking):
#
#   cap    any-hit    MRR    NDCG   src recall
#   none     0.913  0.828   0.836        0.701
#   2        0.913  0.828   0.833        0.733
#   1        0.913  0.828   0.813        0.804
#
# any-hit and MRR are untouched at every cap -- the top result never moves,
# because the cap only skips, never reorders. The cost is NDCG: relevant
# same-document passages get displaced to make room for other sources.
#
# 2 is the default rather than 1 despite 1 scoring higher on source recall,
# because a single passage per document cannot corroborate itself -- a claim
# and the sentence qualifying it usually sit together. Callers who want maximum
# breadth should pass 1 explicitly; with page expansion enabled that gives five
# distinct documents at full-page context, which is the strongest "compile every
# source" configuration measured here.
DEFAULT_MAX_PER_SOURCE = 2


def diversify(results: list[dict], k: int,
              max_per_source: int = DEFAULT_MAX_PER_SOURCE) -> list[dict]:
    """Select k results, allowing at most `max_per_source` from any document.

    Relative order is preserved -- this only skips, never reorders -- so the
    best passage of each document still outranks its weaker ones, and a
    genuinely dominant source still leads.
    """
    if max_per_source <= 0 or not results:
        return results[:k]

    counts: dict[str, int] = {}
    chosen, overflow = [], []

    for r in results:
        source = r["source"]
        if counts.get(source, 0) < max_per_source:
            counts[source] = counts.get(source, 0) + 1
            chosen.append(r)
        else:
            overflow.append(r)
        if len(chosen) == k:
            return chosen

    # Not enough distinct documents to fill k under the cap. Falling back to the
    # highest-scoring rejects is better than returning fewer results: the cap
    # exists to encourage breadth, not to withhold relevant material when the
    # corpus genuinely has only one source for the question.
    #
    # Re-sorted by score because appending overflow at the end produces a list
    # that is not in score order -- a reader going top-down would see a +3.87
    # sitting below a -5.09 and reasonably conclude the ranking is broken. The
    # cap decides WHICH passages are returned; score decides the order they are
    # read in.
    filled = (chosen + overflow)[:k]
    return sorted(filled, key=_score, reverse=True)


def _score(result: dict) -> float:
    return float(result.get("rerank_score", result.get("score", 0.0)))


def source_spread(results: list[dict]) -> dict:
    """How many documents the returned set actually covers -- for reporting."""
    counts: dict[str, int] = {}
    for r in results:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
    return counts
