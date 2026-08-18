"""Document loaders with format-appropriate citation locators.

A citation has to name a place a human can actually turn to. "Page 5" does that
for a PDF and means nothing for a spreadsheet, so each format contributes its own
locator kind rather than being forced into a page number it does not have:

    PDF          page 5
    PowerPoint   slide 12
    Excel        sheet "Q3 Revenue", rows 40-80
    Word         section "3.2 Termination"

Word is the awkward one. A .docx has no fixed pages -- pagination is computed by
the renderer and shifts with fonts, margins, and edits -- so any page number we
invented would be wrong for the reader's copy. Headings are stable, meaningful to
a reader, and survive editing, so sections are used instead.

Every loader returns the same shape:

    [{"locator": {"kind": ..., "value": ...}, "text": ...}, ...]
"""
from pathlib import Path

# Excel sheets are split into row blocks so a citation points at a region of a
# large sheet rather than the whole thing.
XLSX_ROWS_PER_BLOCK = 40

# Below this word count a PDF page is treated as having no usable text layer and
# is sent to OCR. Kept in step with corpus_health.MIN_WORDS_PER_PAGE so the
# guard and the rescue agree on what "empty" means -- otherwise a page could be
# rejected as blank without OCR ever being attempted on it.
OCR_TRIGGER_WORDS = 20

# Phrases that mark front matter rather than a title. Publishers, preprint
# servers and document management systems all stamp headers onto page one, and
# those stamps sit above the real title as often as not.
BOILERPLATE_MARKERS = (
    "permission", "copyright", "all rights reserved", "licence", "license",
    "proceedings of", "published as", "preprint", "under review",
    "confidential", "draft", "do not distribute",
)


def extract_title(units: list[dict], path: Path) -> str:
    """Best available human title for a document.

    Chunks carry no document identity of their own, which is what makes
    cross-document confusion possible: a passage from one paper that never names
    itself is indistinguishable from a passage in another paper about the same
    subject. Attaching a title gives every chunk that identity.

    Prefers a real heading over the filename, because filenames are frequently
    uninformative in practice ("final_v3.pdf", "Doc1.docx") while the first
    heading usually is the title. Falls back to a cleaned-up filename.
    """
    candidates = []
    for unit in units[:1]:  # the title, if anywhere, is on the first unit
        for line in unit["text"].splitlines():
            stripped = line.strip()
            if not stripped.startswith("#"):
                continue
            depth = len(stripped) - len(stripped.lstrip("#"))
            text = stripped.lstrip("#").strip()
            if not (8 <= len(text) <= 160) or text[0].isdigit():
                continue
            if any(marker in text.lower() for marker in BOILERPLATE_MARKERS):
                continue
            candidates.append((depth, text))

    if candidates:
        # Shallowest heading wins, not the first. Front matter frequently puts
        # a deeper-level copyright or permission notice above the actual title
        # -- the Attention paper leads with "### Provided proper attribution...
        # Google hereby grants permission", with "## Attention Is All You Need"
        # below it. Taking the first heading picks the notice.
        return min(candidates, key=lambda c: c[0])[1]
    return path.stem.replace("_", " ").replace("-", " ").strip()


def locator_label(locator: dict) -> str:
    """Human-readable citation fragment, e.g. 'page 5' or 'sheet Q3, rows 1-40'."""
    kind, value = locator["kind"], locator["value"]
    if kind in ("page", "slide"):
        return f"{kind} {value}"
    if kind == "sheet":
        rows = locator.get("rows")
        return f"sheet {value}" + (f", rows {rows}" if rows else "")
    if kind == "section":
        return f"section {value!r}"
    return f"{kind} {value}"


# --------------------------------------------------------------------------- PDF

def load_pdf(path: Path, use_ocr: bool = True) -> list[dict]:
    """Extract a PDF page by page, falling back to OCR only where needed.

    OCR runs per page rather than per document because the common real case is
    mixed: a born-digital report with scanned exhibits appended, or a contract
    whose signature pages were scanned back in. OCR output is worse than a real
    text layer, so running it across pages that already extracted cleanly would
    actively degrade them -- as well as being far slower.
    """
    import pymupdf4llm

    pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    units = [
        {"locator": {"kind": "page", "value": i + 1}, "text": p["text"]}
        for i, p in enumerate(pages)
    ]

    if not use_ocr:
        return units

    needs_ocr = [
        u["locator"]["value"] for u in units
        if len(u["text"].split()) < OCR_TRIGGER_WORDS
    ]
    if not needs_ocr:
        return units

    import ocr
    if not ocr.available():
        print(f"    {path.name}: {len(needs_ocr)} page(s) have no text layer and "
              f"OCR is not installed -- they will be empty in the index")
        return units

    print(f"    {path.name}: OCR on {len(needs_ocr)} page(s) with no text layer...")
    recovered = ocr.ocr_pdf_pages(path, needs_ocr)
    by_page = {u["locator"]["value"]: u for u in units}
    gained = 0
    for page_num, text in recovered.items():
        if len(text.split()) > len(by_page[page_num]["text"].split()):
            by_page[page_num]["text"] = text
            gained += len(text.split())
    print(f"    {path.name}: OCR recovered {gained} words")
    return units


# -------------------------------------------------------------------------- DOCX

def load_docx(path: Path) -> list[dict]:
    """Split a Word document at its headings.

    Falls back to fixed-size paragraph blocks when a document has no heading
    styles at all, which is common in documents converted from other formats.
    """
    import docx

    document = docx.Document(str(path))
    sections, current, heading = [], [], None

    def flush():
        if current:
            sections.append({
                "locator": {"kind": "section", "value": heading or "(untitled)"},
                "text": "\n\n".join(current),
            })

    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        if para.style.name.startswith("Heading"):
            flush()
            current, heading = [], text
            continue
        current.append(text)
    flush()

    # Tables carry meaning that paragraph iteration skips entirely.
    for i, table in enumerate(document.tables, start=1):
        rows = [
            " | ".join(cell.text.strip() for cell in row.cells)
            for row in table.rows
        ]
        if any(r.strip(" |") for r in rows):
            sections.append({
                "locator": {"kind": "table", "value": i},
                "text": "\n".join(rows),
            })

    if not sections:
        paras = [p.text.strip() for p in document.paragraphs if p.text.strip()]
        sections = [
            {"locator": {"kind": "part", "value": i + 1},
             "text": "\n\n".join(paras[i:i + 30])}
            for i in range(0, len(paras), 30)
        ]
    return sections


# -------------------------------------------------------------------------- PPTX

def load_pptx(path: Path) -> list[dict]:
    """One unit per slide, including speaker notes.

    Notes routinely hold the substance behind a slide's bullet fragments, so
    dropping them loses the most answerable content in the deck.
    """
    from pptx import Presentation

    prs = Presentation(str(path))
    units = []
    for i, slide in enumerate(prs.slides, start=1):
        parts = [
            shape.text.strip()
            for shape in slide.shapes
            if getattr(shape, "has_text_frame", False) and shape.text.strip()
        ]
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                parts.append(f"[speaker notes] {notes}")
        if parts:
            units.append({
                "locator": {"kind": "slide", "value": i},
                "text": "\n\n".join(parts),
            })
    return units


# -------------------------------------------------------------------------- XLSX

def load_xlsx(path: Path) -> list[dict]:
    """Serialise sheets as labelled rows rather than raw CSV.

    A bare CSV dump loses the header once it is chunked, leaving rows of bare
    numbers that no embedding can interpret. Repeating the column name with each
    value ("Revenue: 4.2M") keeps every row independently meaningful, which is
    what retrieval needs -- at the cost of verbosity.
    """
    from openpyxl import load_workbook

    wb = load_workbook(str(path), data_only=True, read_only=True)
    units = []

    for sheet in wb.worksheets:
        rows = [
            [("" if c is None else str(c).strip()) for c in row]
            for row in sheet.iter_rows(values_only=True)
        ]
        rows = [r for r in rows if any(cell for cell in r)]
        if not rows:
            continue

        header, body = rows[0], rows[1:]
        if not body:  # header-only sheet
            units.append({
                "locator": {"kind": "sheet", "value": sheet.title},
                "text": " | ".join(header),
            })
            continue

        for start in range(0, len(body), XLSX_ROWS_PER_BLOCK):
            block = body[start:start + XLSX_ROWS_PER_BLOCK]
            lines = []
            for row in block:
                pairs = [
                    f"{head}: {cell}"
                    for head, cell in zip(header, row)
                    if cell and head
                ]
                if pairs:
                    lines.append("; ".join(pairs))
            if lines:
                first, last = start + 2, start + len(block) + 1  # 1-indexed, +header
                units.append({
                    "locator": {"kind": "sheet", "value": sheet.title,
                                "rows": f"{first}-{last}"},
                    "text": "\n".join(lines),
                })
    wb.close()
    return units


# ---------------------------------------------------------------------- registry

LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".pptx": load_pptx,
    ".xlsx": load_xlsx,
}


def load_document(path: Path) -> list[dict]:
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"no loader for {path.suffix} ({path.name})")
    return loader(path)
