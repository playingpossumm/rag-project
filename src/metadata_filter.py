"""Restrict retrieval to a subset of the corpus, exactly rather than approximately.

"Search only the 2023 contracts", "answer from this one paper", "exclude the
drafts". The requirement is scoping, and the naive implementation -- retrieve
normally, then drop what does not match -- is wrong in a way that is easy to
miss: if the top 20 are all from documents the filter excludes, filtering
afterwards returns nothing, while the corpus plainly contains an answer. The
filter has to constrain the search, not the results.

Both retrievers support that exactly, for different reasons.

  dense   FAISS takes an IDSelector, so the scan skips excluded rows instead of
          scoring and discarding them. Exact: the top k within the subset, not
          the survivors of the top k overall.
  sparse  BM25 scores every chunk anyway, so the ranking is simply taken over
          the allowed ids. Also exact.

Note what a filter does NOT do: it cannot rescue a question the subset cannot
answer. Scoping to one document and asking something it does not discuss
returns that document's least-bad passages, and the abstention gate is what
should catch it -- the gate reads the reranker's score, which knows nothing
about how the pool was constrained.

    from metadata_filter import matching_ids
    ids = matching_ids(metadata, {"sources": ["gpt3.pdf"]})
    retrieve(q, index, metadata, model, allowed_ids=ids)
"""
import re

# Every key a spec may carry. Unknown keys are rejected rather than ignored:
# a typo in a filter silently returning the whole corpus is the failure mode
# that makes a scoped search quietly unscoped.
KEYS = {"sources", "exclude_sources", "title_contains", "locator_kind",
        "locator_range", "text_contains"}


def _locator_number(chunk: dict):
    """The numeric part of a locator, or None if it has none.

    Locators are {"kind": "page", "value": 5} for PDFs but a slide number, a
    sheet name or a row range elsewhere, so a range filter has to cope with
    values that are not numbers at all rather than assuming pages.
    """
    value = chunk.get("locator", {}).get("value")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    m = re.search(r"\d+", str(value or ""))
    return int(m.group()) if m else None


def matching_ids(metadata: list[dict], spec: dict | None) -> list[int] | None:
    """Chunk ids satisfying every clause in `spec`. None means "no filter".

    Returning None rather than "all ids" is deliberate: it lets the retrievers
    skip building a selector entirely on the common unfiltered path, so scoping
    costs nothing when nobody asked for it.
    """
    if not spec:
        return None

    unknown = set(spec) - KEYS
    if unknown:
        raise ValueError(f"unknown filter key(s): {sorted(unknown)}; "
                         f"valid keys are {sorted(KEYS)}")

    sources = set(spec.get("sources") or ())
    excluded = set(spec.get("exclude_sources") or ())
    title_sub = (spec.get("title_contains") or "").lower()
    kind = spec.get("locator_kind")
    text_sub = (spec.get("text_contains") or "").lower()

    lo = hi = None
    if spec.get("locator_range"):
        lo, hi = spec["locator_range"]

    out = []
    for i, chunk in enumerate(metadata):
        if sources and chunk["source"] not in sources:
            continue
        if excluded and chunk["source"] in excluded:
            continue
        if title_sub and title_sub not in str(chunk.get("title", "")).lower():
            continue
        if kind and chunk.get("locator", {}).get("kind") != kind:
            continue
        if lo is not None:
            n = _locator_number(chunk)
            if n is None or not (lo <= n <= hi):
                continue
        if text_sub and text_sub not in chunk["text"].lower():
            continue
        out.append(i)
    return out


def describe(metadata: list[dict], ids: list[int] | None) -> str:
    """One line saying what a filter actually selected, for logs and the UI."""
    if ids is None:
        return f"no filter, {len(metadata)} chunks"
    if not ids:
        return "filter matched nothing"
    docs = {metadata[i]["source"] for i in ids}
    return (f"{len(ids)} of {len(metadata)} chunks, "
            f"{len(docs)} of {len({c['source'] for c in metadata})} documents")


def selector(ids: list[int] | None):
    """A FAISS IDSelector for `ids`, or None when unfiltered.

    Built here rather than in retrieve() so the faiss import stays out of the
    unfiltered path, and so the numpy dtype -- int64, which faiss requires and
    does not coerce -- is specified in exactly one place.
    """
    if ids is None:
        return None
    import faiss
    import numpy as np

    return faiss.IDSelectorBatch(np.asarray(ids, dtype="int64"))
