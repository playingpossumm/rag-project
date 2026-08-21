"""Write real .docx, .pptx and .xlsx documents from source prose.

The point is not the files. It is that `load_docx`, `load_pptx` and `load_xlsx`
have never run on anything -- HANDOFF section 7 calls that the largest unknown in
the system, and the spreadsheet path the least trustworthy of the three, because
rows are flattened into `Column: value` text on an assumption with no evidence
behind it.

So these are written to be **awkward in the ways real documents are**, not
convenient:

  docx   headings at several levels, ordinary paragraphs, a table with a header
         row, and a bulleted list -- because a loader that only reads
         `document.paragraphs` silently drops every table in the corpus.
  pptx   a title slide, bullet slides, a table slide, and speaker notes --
         because notes are where the actual sentences live in most decks, and a
         loader reading only shapes gets titles and fragments.
  xlsx   several sheets, a header row, mixed text and numbers, and one sheet
         whose first row is a title rather than headers -- which is what
         spreadsheets in the wild look like and what breaks naive header
         detection.

Nothing here is tailored to what the loaders happen to support. Files that fail
to parse are the finding, not a bug in this script.
"""
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def sections(text: str) -> list[tuple[str, list[str]]]:
    """Wikipedia plain-text extracts mark sections with == Heading ==."""
    out, head, buf = [], "Overview", []
    for line in text.splitlines():
        m = re.match(r"^\s*(={2,6})\s*(.+?)\s*\1\s*$", line)
        if m:
            if buf:
                out.append((head, [p for p in buf if p.strip()]))
            head, buf = m.group(2), []
        elif line.strip():
            buf.append(line.strip())
    if buf:
        out.append((head, [p for p in buf if p.strip()]))
    # Reference and navigation sections carry no prose worth retrieving.
    drop = {"references", "external links", "see also", "further reading",
            "notes", "bibliography", "sources", "citations"}
    return [(h, ps) for h, ps in out if h.strip().lower() not in drop and ps]


def _slug(text: str, limit: int = 58) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:limit].rstrip("_")


# ------------------------------------------------------------------ docx --
def write_docx(art: dict, dest: Path) -> Path:
    from docx import Document

    doc = Document()
    doc.add_heading(art["title"], level=0)
    secs = sections(art["text"])

    for i, (head, paras) in enumerate(secs[:9]):
        doc.add_heading(head, level=1 if i % 3 else 2)
        for para in paras[:6]:
            doc.add_paragraph(para)
        # A table partway through, so a loader that reads only paragraphs
        # demonstrably loses content rather than merely being suspected of it.
        if i == 1:
            t = doc.add_table(rows=1, cols=2)
            t.style = "Table Grid"
            hdr = t.rows[0].cells
            hdr[0].text, hdr[1].text = "Aspect", "Detail"
            for para in (paras[:3] or ["(no body)"]):
                row = t.add_row().cells
                row[0].text = head
                row[1].text = para[:220]
        if i == 3:
            for para in paras[:4]:
                doc.add_paragraph(para[:180], style="List Bullet")

    doc.save(dest)
    return dest


# ------------------------------------------------------------------ pptx --
def write_pptx(art: dict, dest: Path) -> Path:
    from pptx import Presentation
    from pptx.util import Inches, Pt

    prs = Presentation()
    title_slide = prs.slides.add_slide(prs.slide_layouts[0])
    title_slide.shapes.title.text = art["title"]
    title_slide.placeholders[1].text = "Reference notes"

    secs = sections(art["text"])
    for head, paras in secs[:8]:
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = head[:90]
        body = slide.placeholders[1].text_frame
        body.text = paras[0][:190]
        for para in paras[1:4]:
            p = body.add_paragraph()
            p.text = para[:190]
            p.level = 1
        # The full prose goes in the notes, which is where it lives in real
        # decks -- slides carry fragments.
        slide.notes_slide.notes_text_frame.text = " ".join(paras[:3])[:1800]

    if secs:
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        slide.shapes.title.text = "Summary"
        rows = min(5, len(secs))
        table = slide.shapes.add_table(rows + 1, 2, Inches(0.6), Inches(1.6),
                                       Inches(8.8), Inches(0.8 * rows)).table
        table.cell(0, 0).text = "Section"
        table.cell(0, 1).text = "First line"
        for r, (head, paras) in enumerate(secs[:rows], start=1):
            table.cell(r, 0).text = head[:40]
            table.cell(r, 1).text = paras[0][:110]

    prs.save(dest)
    return dest


# ------------------------------------------------------------------ xlsx --
def write_xlsx(art: dict, dest: Path) -> Path:
    from openpyxl import Workbook

    wb = Workbook()
    secs = sections(art["text"])

    ws = wb.active
    ws.title = "Sections"
    ws.append(["Section", "Words", "Opening line"])
    for head, paras in secs[:20]:
        ws.append([head, sum(len(p.split()) for p in paras), paras[0][:240]])

    # A sheet whose first row is a title rather than headers. Spreadsheets in
    # the wild look like this constantly, and it is what naive header detection
    # gets wrong -- every row then reads "Bird: <text>" with no field names.
    ws2 = wb.create_sheet("Notes")
    ws2.append([f"{art['title']} — working notes"])
    ws2.append([])
    ws2.append(["Topic", "Detail", "Length"])
    for head, paras in secs[:14]:
        for para in paras[:2]:
            ws2.append([head, para[:300], len(para)])

    ws3 = wb.create_sheet("Facts")
    ws3.append(["Attribute", "Value"])
    ws3.append(["Title", art["title"]])
    ws3.append(["Sections", len(secs)])
    ws3.append(["Total words", sum(len(p.split()) for _, ps in secs for p in ps)])

    wb.save(dest)
    return dest


# ------------------------------------------------------------------- pdf --
def write_pdf(art: dict, dest: Path) -> Path:
    import pymupdf

    secs = sections(art["text"])
    lines = [art["title"], ""]
    for head, paras in secs[:10]:
        lines.append(head)
        lines.append("")
        for para in paras[:6]:
            # Wrap by hand rather than relying on the textbox to reflow: it
            # returns only whether the text fit, not where it stopped, and
            # asking the page for its own document to probe with does not work
            # (page.parent is not live after the page is made).
            words, line = para.split(), ""
            for w in words:
                if len(line) + len(w) + 1 > 95:
                    lines.append(line)
                    line = w
                else:
                    line = f"{line} {w}".strip()
            if line:
                lines.append(line)
            lines.append("")

    PER_PAGE = 46
    doc = pymupdf.open()
    for i in range(0, max(len(lines), 1), PER_PAGE):
        page = doc.new_page(width=595, height=842)
        page.insert_text((56, 70), "\n".join(lines[i:i + PER_PAGE]),
                         fontsize=10.5, fontname="helv", lineheight=1.35)
    doc.save(dest)
    doc.close()
    return dest


WRITERS = [(".docx", write_docx), (".pptx", write_pptx),
           (".xlsx", write_xlsx), (".pdf", write_pdf)]


def write_mixed(art: dict, into: Path, index: int) -> Path | None:
    """Rotate through the formats so the corpus is genuinely mixed."""
    ext, writer = WRITERS[index % len(WRITERS)]
    dest = into / f"{_slug(art['title'])}{ext}"
    if dest.exists():
        return None
    try:
        return writer(art, dest)
    except Exception as exc:  # noqa: BLE001 - a format that cannot be written
        print(f"  FAILED to write {dest.name}: {type(exc).__name__}: {exc}")
        return None
