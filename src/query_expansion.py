"""Query expansion by pseudo-relevance feedback. No model, no network, no cost.

The dense retriever already handles paraphrase, so the recall gap this can close
is the lexical one: a question that names a thing differently from the corpus.
BM25 scores exact overlap, so "how fast does it train" shares no term with "step
time was 1.0 seconds" and scores zero.

RM3-style feedback is the classical answer and needs nothing but the index that
already exists. Run the query, assume the top few results are relevant, harvest
the terms that distinguish them, and re-run with those terms appended. The
assumption is doing the work and it is often wrong -- which is why this is
measured before it is defaulted on, and why it is off by default until the
measurement says otherwise.

Terms are scored by feedback frequency times IDF, so a term must be both common
among the top results and rare across the corpus. Frequency alone selects
"the"; IDF alone selects whatever appears once in a footnote.

Weighting is by repetition. BM25Okapi takes a bag of query terms and sums each
term's contribution, so a term listed twice counts twice -- crude next to a
proper weighted query, and exactly equivalent for integer weights.

Applied to the SPARSE side only. Dense retrieval embeds the query as a sentence,
and appending a bag of keywords to a sentence moves its vector somewhere that is
not a question -- expansion helps lexical matching and distorts semantic
matching, so it is given only to the retriever it helps.
"""
from collections import Counter

from hybrid import tokenize

# Ten feedback documents and ten terms are the values the RM3 literature
# converges on. They are exposed rather than baked in because this corpus is
# small enough that the top ten may already be half the relevant material.
FEEDBACK_DOCS = 10
FEEDBACK_TERMS = 10
ORIGINAL_WEIGHT = 2  # the query's own terms still dominate the expansion's


def prf_terms(query: str, bm25, metadata: list[dict],
              top_docs: int = FEEDBACK_DOCS,
              n_terms: int = FEEDBACK_TERMS) -> list[str]:
    """Terms to add, best first. Never includes a term already in the query."""
    query_terms = set(tokenize(query))
    scores = bm25.get_scores(list(query_terms))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_docs]
    if not top or scores[top[0]] <= 0:
        # No lexical match at all, so there is no feedback to take: every
        # candidate term would come from documents chosen at random.
        return []

    # Document frequency WITHIN the feedback set, not raw term frequency: a term
    # repeated forty times in one passage should not outrank one appearing in
    # eight of the ten.
    freq = Counter()
    for i in top:
        chunk = metadata[i]
        freq.update(set(tokenize(chunk.get("embed_text", chunk["text"]))))

    idf = getattr(bm25, "idf", {})
    scored = [
        (n * idf.get(term, 0.0), term)
        for term, n in freq.items()
        if term not in query_terms and len(term) > 2 and not term.isdigit()
    ]
    scored.sort(reverse=True)
    return [term for weight, term in scored[:n_terms] if weight > 0]


def expand(query: str, bm25, metadata: list[dict], **kw) -> list[str]:
    """The token bag to hand BM25: the original query, weighted, plus feedback."""
    base = tokenize(query)
    added = prf_terms(query, bm25, metadata, **kw)
    return base * ORIGINAL_WEIGHT + added


def explain(query: str, bm25, metadata: list[dict], **kw) -> str:
    added = prf_terms(query, bm25, metadata, **kw)
    return f"+{len(added)}: {' '.join(added)}" if added else "no expansion"
