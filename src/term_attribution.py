"""Which words in the question did the lexical retrieving, and how much.

BM25 scores a passage as a SUM over the query terms, so the contribution of
each term is exact rather than estimated -- `get_scores([a, b])` equals
`get_scores([a]) + get_scores([b])` term by term. `check_additivity` asserts
that on real data rather than trusting the claim, because the moment it stops
holding the heatmap becomes a picture of nothing.

This explains ONE HALF of retrieval and the page that draws it has to say so.
Dense retrieval has no per-term story at all: the question becomes a single
384-dimensional vector and the individual words stop existing. That asymmetry
is not a gap in this module, it is the actual difference between the two
retrievers and the reason the system runs both.

Read-only. It borrows the BM25 index the server already built and calls nothing
in the pipeline, so it cannot change what retrieval returns.
"""
from hybrid import tokenize

# Words that are in every document and therefore attribute nothing. BM25's IDF
# already drives their contribution towards zero; dropping them keeps the
# heatmap's rows down to the terms a reader would actually ask about.
STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for",
    "from", "how", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "to", "was", "what", "when", "where", "which", "who", "why", "with",
}


def attribute(question: str, bm25, chunk_ids: list[int],
              max_terms: int = 10) -> dict:
    """Per-term BM25 contribution to each of `chunk_ids`.

    Returns the terms that actually earned something, strongest first, with the
    matrix of contributions. A term every passage ignores is dropped -- an
    all-zero row on a heatmap reads as a bug rather than as "this word did
    nothing", which is what it means.
    """
    terms, seen = [], set()
    for t in tokenize(question):
        if t in STOP or t in seen:
            continue
        seen.add(t)
        terms.append(t)
    if not terms or not chunk_ids:
        return {"terms": [], "matrix": [], "totals": []}

    rows = []
    for t in terms:
        scores = bm25.get_scores([t])
        rows.append([round(float(scores[i]), 4) for i in chunk_ids])

    keep = [i for i, r in enumerate(rows) if any(v > 0 for v in r)]
    keep.sort(key=lambda i: -sum(rows[i]))
    keep = keep[:max_terms]

    # The score the pipeline ACTUALLY used, over the untouched token bag. The
    # attributed rows will not sum to it: stopwords are dropped from the heatmap
    # but BM25 still scored them, and on a naturally-worded question that
    # remainder is not small -- about a fifth of the score for "how does the
    # diversity cap change source recall". Reporting the shortfall is the
    # difference between a decomposition and a plausible-looking picture.
    full = bm25.get_scores(tokenize(question))
    shown = [round(sum(rows[i][c] for i in keep), 4) for c in range(len(chunk_ids))]
    actual = [round(float(full[i]), 4) for i in chunk_ids]

    return {
        "terms": [terms[i] for i in keep],
        "matrix": [rows[i] for i in keep],
        "totals": shown,
        "actual": actual,
        # What the words this heatmap does not draw were worth, per passage.
        "unattributed": [round(x - y, 4) for x, y in zip(actual, shown)],
        "dropped": [terms[i] for i in range(len(terms))
                    if i not in keep and all(v == 0 for v in rows[i])],
    }


def check_additivity(question: str, bm25, chunk_ids: list[int],
                     tol: float = 1e-6) -> None:
    """Assert the decomposition is exact. Raises rather than warning."""
    terms = [t for t in dict.fromkeys(tokenize(question))]
    if not terms or not chunk_ids:
        return
    whole = bm25.get_scores(terms)
    parts = [bm25.get_scores([t]) for t in terms]
    for i in chunk_ids:
        got = sum(p[i] for p in parts)
        if abs(got - whole[i]) > tol:
            raise AssertionError(
                f"BM25 is not additive over query terms for chunk {i}: "
                f"sum of parts {got!r} vs whole {whole[i]!r}. "
                f"Per-term attribution is not valid for this scorer.")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import corpora
    from hybrid import build_bm25
    from retrieve import load_index

    cfg = corpora.get(corpora.active_name())
    _, metadata = load_index(cfg["store"])
    bm25 = build_bm25(metadata)
    q = " ".join(sys.argv[1:]) or "How does the diversity cap change source recall?"

    ids = list(range(min(8, len(metadata))))
    check_additivity(q, bm25, ids)
    print(f"additivity holds on {len(ids)} chunks -- decomposition is exact")

    from hybrid import bm25_search
    hits = bm25_search(q, bm25, metadata, k=6)
    ids = [h["chunk_id"] for h in hits]
    check_additivity(q, bm25, ids)
    a = attribute(q, bm25, ids)
    print(f"\n{q}\n")
    w = max((len(t) for t in a["terms"]), default=4)
    print(" " * (w + 2) + "".join(f"{h['chunk_id']:>8}" for h in hits))
    for t, row in zip(a["terms"], a["matrix"]):
        print(f"  {t:<{w}}" + "".join(f"{v:>8.2f}" for v in row))
    print(f"  {'shown':<{w}}" + "".join(f"{v:>8.2f}" for v in a["totals"]))
    print(f"  {'stopwd':<{w}}" + "".join(f"{v:>8.2f}" for v in a["unattributed"]))
    print(f"  {'ACTUAL':<{w}}" + "".join(f"{v:>8.2f}" for v in a["actual"]))
    print(f"\n  bm25_search said: {[round(h['score'], 2) for h in hits]}")
    if a["dropped"]:
        print(f"  matched nothing: {', '.join(a['dropped'])}")
