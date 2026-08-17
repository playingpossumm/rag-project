"""Parent-document retrieval: rank on small chunks, return larger context.

Chunk size sits under two opposing pressures. Small chunks rank better --
a 240-token passage averages into a vector that stays specific, so retrieval
precision is high. But a small chunk is often a poor thing to *answer* from:
it can be cut mid-argument, missing the sentence that defines its own subject.

Parent retrieval separates the two. Ranking still happens on small chunks, so
precision is unchanged; but each surviving hit is then expanded to the larger
unit it came from before the text is handed to the model. The retriever gets
specificity and the reader gets context.

Two expansion modes:

    page   -- expand to the whole page. Natural here because chunking is
              already page-scoped and citations are page-level, so the context
              a claim is grounded in is exactly the page cited.
    window -- expand to the chunk plus N neighbours either side, staying inside
              the page. Finer-grained; useful when pages are long.
"""
from collections import defaultdict


def _page_key(chunk: dict) -> tuple:
    return (chunk["source"], chunk["page"])


def expand_to_pages(results: list[dict], metadata: list[dict]) -> list[dict]:
    """Replace each hit with the full text of its page, deduplicated.

    Several hits often share a page. Emitting that page once, at the position of
    its best-ranked hit, avoids sending the same text to the model repeatedly --
    which would waste context and over-weight that page in the model's reading.
    """
    pages = defaultdict(list)
    for chunk in metadata:
        pages[_page_key(chunk)].append(chunk)

    seen, out = set(), []
    for r in results:
        key = _page_key(r)
        if key in seen:
            continue
        seen.add(key)

        members = pages[key]
        out.append({
            **r,
            "text": _stitch(members),
            "expanded_from": r["text"],
            "expansion": "page",
            "n_chunks_merged": len(members),
        })
    return out


def expand_to_window(results: list[dict], metadata: list[dict], window: int = 1) -> list[dict]:
    """Expand each hit to itself plus `window` neighbouring chunks on each side.

    Neighbours are taken by chunk_id, but clipped to the same page: chunk ids are
    sequential across the whole corpus, so id+1 can belong to the next document
    entirely. Crossing that boundary would attach text to a citation that does
    not support it.
    """
    seen_windows, out = set(), []
    for r in results:
        cid = r["chunk_id"]
        lo, hi = cid - window, cid + window
        members = [
            c for i, c in enumerate(metadata)
            if lo <= i <= hi and _page_key(c) == _page_key(r)
        ]
        span = (members[0]["source"], members[0]["page"],
                metadata.index(members[0]), metadata.index(members[-1]))
        if span in seen_windows:
            continue
        seen_windows.add(span)

        out.append({
            **r,
            "text": _stitch(members),
            "expanded_from": r["text"],
            "expansion": f"window±{window}",
            "n_chunks_merged": len(members),
        })
    return out


def _stitch(chunks: list[dict]) -> str:
    """Join chunks back into continuous text.

    Consecutive chunks overlap by 40 tokens, so a naive join duplicates that
    span. Rather than reconstruct exact token boundaries, drop the longest
    suffix/prefix repeat between neighbours -- cheap, and correct for the
    overlap sizes we use.
    """
    if not chunks:
        return ""
    text = chunks[0]["text"]
    for nxt in chunks[1:]:
        text = _join_without_repeat(text, nxt["text"])
    return text


def _join_without_repeat(left: str, right: str, max_probe: int = 400) -> str:
    probe = min(len(left), len(right), max_probe)
    for size in range(probe, 20, -1):
        if left[-size:] == right[:size]:
            return left + right[size:]
    return left + "\n\n" + right


def expand(results: list[dict], metadata: list[dict],
           mode: str = "page", window: int = 1) -> list[dict]:
    if mode == "none":
        return results
    if mode == "page":
        return expand_to_pages(results, metadata)
    if mode == "window":
        return expand_to_window(results, metadata, window=window)
    raise ValueError(f"unknown expansion mode: {mode!r}")
