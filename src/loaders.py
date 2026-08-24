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
import re
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


# A continuation is plain prose directly under the heading: no marker of its
# own, not an author block, not a code fence. Anything with structure is a new
# element, not the rest of the title.
_TITLE_STOP = ("#", "*", "`", "|", ">", "-", "[", "!")

# Section names, so a title is never glued to the heading that follows it.
_SECTIONS = {
    "abstract", "introduction", "background", "related work", "prior work",
    "method", "methods", "methodology", "approach", "experiments", "experiment",
    "results", "evaluation", "discussion", "conclusion", "conclusions",
    "references", "bibliography", "appendix", "appendices", "keywords",
    "acknowledgements", "acknowledgments", "contents", "summary", "overview",
    "preprint", "a preprint", "notation", "preliminaries", "limitations",
}

# Words a title cannot end on. If a heading stops here it was cut, not finished.
_DANGLING = {
    "for", "of", "in", "on", "with", "and", "or", "the", "a", "an", "to",
    "from", "by", "via", "using", "under", "over", "at", "as", "into",
    "through", "across", "between", "toward", "towards", "per", "against",
    "without", "within", "beyond", "during",
}


def _looks_like_names(head: str) -> bool:
    """Is this heading an author line rather than the rest of a title?

    Converters promote an author line to a heading exactly as readily as they
    promote half a title, so "## Priya Ramanathan, Devesh Iyer" arrives looking like a
    continuation. A comma-separated run of title-case pairs is people; a title
    fragment at that length almost always carries a lowercase word or is set in
    caps.
    """
    parts = [q.strip() for q in re.split(r",| and ", head) if q.strip()]
    if len(parts) < 2:
        return False
    for part in parts:
        toks = part.split()
        if not (1 <= len(toks) <= 4):
            return False
        for t in toks:
            w = t.strip(".")
            if not w or not w[0].isupper() or w.isupper():
                return False      # lowercase connective, or set in caps
    return True


def _followed_by_email(following: list[str], start: int, span: int = 3) -> bool:
    """An email within a line or two below marks the heading as an author block.

    Affiliation words are no good for this -- "Department of Computer Science"
    sits below the SECOND HALF OF A TITLE in one of these papers, so keying on
    it would reject a real continuation. An address is more specific.
    """
    for raw in following[start + 1:start + 1 + span]:
        if "@" in raw:
            return True
    return False


def _looks_cut(text: str) -> bool:
    last = re.sub(r"[^A-Za-z-]", "", text.split()[-1] if text.split() else "")
    return last.lower() in _DANGLING


def _rejoin_wrapped(text: str, following: list[str], depth: int = 0) -> str:
    """Reattach title lines that were split from their heading.

    Two ways a cover-page title comes apart. It wraps, and only the first line
    keeps its marker -- the rest is a plain line directly beneath. Or the
    converter promotes each line to its own heading, and the halves end up as
    separate headings with blank lines between them.

    The second case is the dangerous one, because the thing directly after a
    title is usually a section heading, which must NOT be joined. Two
    independent conditions guard it, and one of them has to hold:

      the title ends on a word a title cannot end on -- "...ESTIMATOR FOR";

      or the title was ALREADY seen wrapping onto a plain line, and the next
      heading sits at the same depth with only blanks between. The earlier wrap
      is the evidence: it shows this converter is breaking the title up, which
      a document with a normal title and a normal first section never does.

    Without that second clause requiring a prior wrap, "Some Paper Title"
    followed by "## Model Architecture" joins into nonsense.
    """
    if text.endswith((".", "?", "!")):
        return text

    seen_blank = False
    joined_plain = False
    for idx, raw in enumerate(following):
        cand = raw.strip()
        if not cand:
            seen_blank = True
            continue

        if cand.startswith("#"):
            d = len(cand) - len(cand.lstrip("#"))
            head = cand.lstrip("#").strip()
            if (not head
                    or len(head.split()) > 5
                    or head.lower() in _SECTIONS
                    or head[0].isdigit()
                    or len(text) + len(head) > 200
                    or _looks_like_names(head)
                    or _followed_by_email(following, idx)):
                break
            if not (_looks_cut(text) or (d == depth and seen_blank and joined_plain)):
                break
            return f"{text} {head}".strip()

        if seen_blank:
            break                        # prose after a blank is the body
        if cand.startswith(_TITLE_STOP) or cand[0].isdigit():
            break
        if "@" in cand or len(cand) > 120:
            break
        text = f"{text} {cand}".strip()
        joined_plain = True
        if text.endswith((".", "?", "!")):
            break
    return text


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
        lines = unit["text"].splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.startswith("#"):
                continue
            depth = len(stripped) - len(stripped.lstrip("#"))
            text = stripped.lstrip("#").strip()

            # A long title on a PDF cover wraps, and only the first line keeps
            # the heading marker -- the rest is a plain line underneath it. The
            # heading alone then reads as a sentence cut in half: "OenoBench: A
            # Wine-Domain Benchmark for". Pull the continuation back on.
            # Three lines of lookahead: a long paper title routinely wraps to
            # three on a cover page. The join stops at the first blank or
            # structured line regardless, so a larger window costs nothing.
            text = _rejoin_wrapped(text, lines[i + 1:i + 7], depth)

            if not (8 <= len(text) <= 200) or text[0].isdigit():
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

        # The header is FOUND, not assumed to be row 1. Real spreadsheets open
        # with a title spanning the sheet, and taking that as the header made
        # every row read "Q3 sales report: 41200" -- the title repeated as a
        # column name, the actual column names demoted to data, and the values
        # unlabelled. Measured on a generated workbook whose first row is a
        # title, which is the common case, not an exotic one.
        #
        # A title row is one with a single filled cell; a header row is the
        # first with at least two. Falls back to row 1 when nothing qualifies,
        # which is the old behaviour and correct for a single-column sheet.
        head_at = next((i for i, r in enumerate(rows)
                        if sum(1 for c in r if c) >= 2), 0)
        header, body = rows[head_at], rows[head_at + 1:]
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
                # 1-indexed, and offset past whatever was skipped above the header
                first = head_at + start + 2
                last = head_at + start + len(block) + 1
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
