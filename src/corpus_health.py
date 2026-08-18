"""Detect documents that will index badly -- loudly, before they poison the index.

The failure this exists to prevent is silent. A scanned PDF has no text layer, so
the extractor returns an empty string rather than an error. The page is chunked
into nothing, embedded into nothing, and simply never retrieved. Nothing in the
pipeline reports a problem: evaluation still produces confident numbers, they are
just computed over a corpus with holes in it.

An unsupported file type fails the same way. `ingest.py` globs *.pdf, so a .docx
dropped into data/ is skipped without comment.

Both are cheap to detect and expensive to discover late, so ingestion refuses to
proceed quietly past either.
"""
from pathlib import Path

# A digital page of prose runs to hundreds of words. Under this, a page is
# almost certainly a scan, a full-page figure, or an extraction failure.
MIN_WORDS_PER_PAGE = 20

# If most of a document's pages are near-empty, the document itself is the
# problem -- typically a scan needing OCR -- rather than a few figure pages.
SUSPECT_PAGE_FRACTION = 0.5

SUPPORTED = {".pdf"}
KNOWN_UNSUPPORTED = {
    ".docx": "Word -- needs python-docx",
    ".doc": "legacy Word -- needs conversion first",
    ".pptx": "PowerPoint -- needs python-pptx",
    ".xlsx": "Excel -- needs openpyxl, and tabular data needs its own chunking",
    ".xls": "legacy Excel -- needs conversion first",
    ".csv": "tabular -- needs its own chunking strategy",
    ".txt": "plain text -- easy to add, but has no page numbers for citations",
    ".md": "markdown -- easy to add, but has no page numbers for citations",
    ".html": "web page -- needs an HTML loader, and has no page numbers",
}


def page_report(pages: list[tuple[int, str]]) -> dict:
    """Summarise text yield for one document's extracted pages."""
    counts = [(num, len(text.split())) for num, text in pages]
    empty = [num for num, n in counts if n < MIN_WORDS_PER_PAGE]
    total_words = sum(n for _, n in counts)
    return {
        "pages": len(counts),
        "words": total_words,
        "empty_pages": empty,
        "empty_fraction": len(empty) / len(counts) if counts else 1.0,
        "median_words": sorted(n for _, n in counts)[len(counts) // 2] if counts else 0,
    }


def verdict(report: dict) -> tuple[str, str]:
    """Classify a document as ok / warn / fail, with a reason."""
    if report["pages"] == 0:
        return "FAIL", "no pages extracted at all"
    if report["empty_fraction"] >= SUSPECT_PAGE_FRACTION:
        pct = round(report["empty_fraction"] * 100)
        return "FAIL", (
            f"{pct}% of pages yielded almost no text -- probably a scanned "
            f"document with no text layer. It needs OCR; indexing it now would "
            f"add mostly-empty chunks that can never be retrieved."
        )
    if report["empty_pages"]:
        return "WARN", (
            f"{len(report['empty_pages'])} low-text page(s): "
            f"{report['empty_pages'][:8]} -- likely full-page figures or scans"
        )
    return "ok", ""


def scan_unsupported(data_dir: Path) -> list[tuple[Path, str]]:
    """Files present in the corpus directory that ingestion would silently skip."""
    out = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() in SUPPORTED:
            continue
        reason = KNOWN_UNSUPPORTED.get(path.suffix.lower(), "unrecognised file type")
        out.append((path, reason))
    return out
