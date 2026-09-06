"""The OCR fallback, against PDFs with no text layer.

Until now this path had never seen a scanned document. It was written, it was
reasoned about, and it was never once run on the thing it exists for -- which
means the only evidence it worked was that it did not crash on documents that
never triggered it.

Fixtures are built here rather than committed. A scanned page is made by
rendering a normal PDF page to an image and putting that image on a fresh page:
what comes out has pixels where the text was and no text layer at all, which is
exactly what a scanner produces.

    python src/test_ocr.py

Exit code is the contract: 0 every check passed, 1 a check failed, 2 the
checks did not run because RapidOCR is not installed. The loader degrades to
"these pages will be empty" without RapidOCR, which is a supported
configuration and not a broken one, so the skip is not a failure. It is not a
pass either. Until 2026-09-06 the skip path printed a message and returned
None, so the process exited 0 with no N/N line and a machine without RapidOCR
reported this suite as green while running none of it. `check_docs.py --tests`
reads exit 2 as NOT RUN, distinct from a failed suite.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import ocr  # noqa: E402
from loaders import OCR_TRIGGER_WORDS, load_pdf  # noqa: E402

PASS = FAIL = 0

# How many checks a full run makes. Printed on the skip path as "0/8", so a
# reader of check_docs --tests sees how many checks the document counts for a
# suite that did not run, and asserted against PASS + FAIL at the end of a full
# run so adding a check here without raising this number fails loudly.
N_CHECKS = 8

BODY = [
    "Quarterly Field Report",
    "The syrinx is the vocal organ of birds, located at the base",
    "of the trachea. Measured across 41 specimens, mean mass was",
    "3.82 grams with a standard deviation of 0.44 grams.",
]
# Long enough to clear OCR_TRIGGER_WORDS on its own. The first version of this
# fixture was one short line, which fell UNDER the trigger -- so the "a good
# page is left alone" check passed while OCR was in fact running on it. The
# check only means anything if the page never qualifies for OCR.
DIGITAL_LINES = [
    "This page was never scanned and carries a real text layer throughout its",
    "body. It exists to prove that optical recognition is not applied to pages",
    "that already extracted cleanly, because doing so would replace accurate",
    "characters with recognised ones and quietly degrade every readable page in",
    "a document to rescue the few that need help. The count of words here is",
    "comfortably above the threshold that triggers the fallback.",
]
DIGITAL = DIGITAL_LINES[0]


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f"\n       {detail}" if detail else ""))


def _typeset(doc, lines, size=15):
    page = doc.new_page()
    y = 90
    for line in lines:
        page.insert_text((64, y), line, fontsize=size)
        y += size * 1.9
    return page


def make_pdfs(tmp: Path):
    """A fully scanned PDF, and a mixed one: page 1 digital, page 2 scanned."""
    import pymupdf

    # A normal PDF, then the same page flattened to an image.
    src = pymupdf.open()
    page = _typeset(src, BODY)
    pix = page.get_pixmap(dpi=190)

    scanned = pymupdf.open()
    sp = scanned.new_page(width=page.rect.width, height=page.rect.height)
    sp.insert_image(sp.rect, pixmap=pix)
    scanned_path = tmp / "scanned.pdf"
    scanned.save(str(scanned_path))
    scanned.close()

    mixed = pymupdf.open()
    _typeset(mixed, DIGITAL_LINES)
    mp = mixed.new_page(width=page.rect.width, height=page.rect.height)
    mp.insert_image(mp.rect, pixmap=pix)
    mixed_path = tmp / "mixed.pdf"
    mixed.save(str(mixed_path))
    mixed.close()

    return scanned_path, mixed_path


def words(text: str) -> set[str]:
    return {w.strip(".,;:()").lower() for w in text.split()}


def main() -> int:
    global FAIL
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not ocr.available():
        print("  NOT RUN  RapidOCR is not installed, so none of these checks ran.\n"
              "  The loader treats that as a supported configuration: pages with\n"
              "  no text layer stay empty and it says so. Install with:\n"
              "      pip install rapidocr_onnxruntime\n"
              f"\n  0/{N_CHECKS} OCR checks ran -- RapidOCR not installed")
        return 2

    # pymupdf keeps its file handles open on Windows, so the directory
    # cannot always be removed; that is cleanup, not a test result.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        tmp = Path(td)
        scanned, mixed = make_pdfs(tmp)

        # ---- the fixture really has no text layer ------------------------
        raw = load_pdf(scanned, use_ocr=False)
        n_raw = len(raw[0]["text"].split())
        check("a scanned page has no extractable text",
              n_raw < OCR_TRIGGER_WORDS,
              f"got {n_raw} words without OCR; the fixture is not scanned")

        # ---- OCR recovers it ----------------------------------------------
        got = load_pdf(scanned, use_ocr=True)
        text = got[0]["text"]
        found = words(text)
        wanted = ["syrinx", "vocal", "organ", "birds", "trachea", "specimens"]
        hits = [w for w in wanted if w in found]
        check("OCR recovers the body text of a scanned page",
              len(hits) >= 5,
              f"recovered {len(hits)}/{len(wanted)} key words: {hits}\n"
              f"       text was: {text[:200]!r}")

        # Numbers are what OCR is worst at and what a retrieval answer most
        # often turns on, so they are checked separately rather than folded in.
        check("OCR recovers the figures, not just the prose",
              "3.82" in text or "3,82" in text,
              f"expected 3.82 somewhere in: {text[:200]!r}")

        # ---- a good page is left alone --------------------------------------
        # The design claim is that OCR fills gaps rather than replacing pages.
        # If it ran everywhere it would quietly degrade every clean page.
        # It must not even qualify: if page 1 is under the trigger, the check
        # below passes whether or not OCR left it alone.
        pre = load_pdf(mixed, use_ocr=False)
        check("the digital page is above the OCR trigger",
              len(pre[0]["text"].split()) >= OCR_TRIGGER_WORDS,
              f"page 1 has {len(pre[0]['text'].split())} words, "
              f"trigger is {OCR_TRIGGER_WORDS} -- fixture too short to prove anything")

        both = load_pdf(mixed, use_ocr=True)
        check("the mixed document keeps both pages", len(both) == 2,
              f"got {len(both)} pages")
        digital = both[0]["text"]
        check("a page with a real text layer is returned untouched",
              DIGITAL in digital,
              f"page 1 came back as: {digital[:160]!r}")
        check("the scanned page of a mixed document is recovered",
              "syrinx" in words(both[1]["text"]),
              f"page 2 came back as: {both[1]['text'][:160]!r}")

        # ---- the confidence floor is doing something -------------------------
        blank = pdf_of_noise(tmp)
        empty = load_pdf(blank, use_ocr=True)
        check("a page with no text yields no invented text",
              len(empty[0]["text"].split()) < OCR_TRIGGER_WORDS,
              f"invented: {empty[0]['text'][:160]!r}")

    if PASS + FAIL != N_CHECKS:
        # A check was added or removed without moving N_CHECKS, so the skip
        # path would announce the wrong count. Reported as a failure, because
        # a suite that misstates its own size is what check_docs holds the
        # documents to.
        print(f"  FAIL N_CHECKS is {N_CHECKS} and the run made {PASS + FAIL} checks")
        FAIL += 1

    print(f"\n  {PASS}/{PASS + FAIL} OCR checks passed")
    return 1 if FAIL else 0


def pdf_of_noise(tmp: Path) -> Path:
    """A page with marks on it but no writing -- rules, boxes, a scan artefact."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    for i in range(9):
        y = 80 + i * 34
        page.draw_line(pymupdf.Point(60, y), pymupdf.Point(520, y), width=1.4)
    page.draw_rect(pymupdf.Rect(60, 380, 240, 470), width=1.4)
    pix = page.get_pixmap(dpi=190)

    out = pymupdf.open()
    op = out.new_page(width=page.rect.width, height=page.rect.height)
    op.insert_image(op.rect, pixmap=pix)
    path = tmp / "noise.pdf"
    out.save(str(path))
    out.close()
    doc.close()
    return path


if __name__ == "__main__":
    raise SystemExit(main())
