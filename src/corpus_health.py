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

SUPPORTED = {".pdf", ".docx", ".pptx", ".xlsx"}
KNOWN_UNSUPPORTED = {
    ".doc": "legacy Word -- save as .docx first",
    ".xls": "legacy Excel -- save as .xlsx first",
    ".ppt": "legacy PowerPoint -- save as .pptx first",
    ".csv": "tabular -- loader not written yet; would reuse the .xlsx row strategy",
    ".txt": "plain text -- easy to add; has no natural locator, so citations would be by block",
    ".md": "markdown -- easy to add; would cite by heading like .docx",
    ".html": "web page -- needs an HTML loader; would cite by heading",
}


def page_report(units: list[dict]) -> dict:
    """Summarise text yield across one document's extracted units.

    A "unit" is whatever the format's natural citable division is -- a PDF page,
    a slide, a Word section, a block of spreadsheet rows.
    """
    counts = [(u["locator"], len(u["text"].split())) for u in units]
    empty = [loc for loc, n in counts if n < MIN_WORDS_PER_PAGE]
    return {
        "pages": len(counts),
        "words": sum(n for _, n in counts),
        "empty_pages": [loc.get("value") for loc in empty],
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
