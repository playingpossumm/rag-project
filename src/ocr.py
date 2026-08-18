"""OCR fallback for pages with no extractable text layer.

Applied per page, not per document. A scanned PDF is the obvious case, but the
common real one is mixed: a born-digital report with a few scanned exhibits
appended, or a contract whose signature pages were scanned back in. Running OCR
across an entire document to rescue three pages would be slow and would also
*degrade* the good pages, since OCR output is worse than a real text layer.

So extraction stays the primary path and OCR only fills the gaps it leaves.

RapidOCR is used rather than Tesseract because it installs from pip with no
system binary and no administrator rights, which keeps the project portable --
`pip install -r requirements.txt` remains the whole setup.
"""
import sys

# Rendering resolution. 72 dpi is the PDF's native scale and too coarse for
# reliable character recognition; 200 is the usual floor for body text. Higher
# improves small print but costs time and memory quadratically.
OCR_DPI = 200

# RapidOCR reports per-detection confidence. Below this the "text" is usually
# noise from lines, stamps, or scan artefacts -- keeping it would inject
# plausible-looking garbage into the index, which is worse than a gap.
MIN_DETECTION_CONFIDENCE = 0.5

_engine = None


def available() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def _get_engine():
    """Load the OCR models once; they are expensive to initialise."""
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _engine = RapidOCR()
    return _engine


def ocr_page(page, dpi: int = OCR_DPI) -> str:
    """Render one PDF page to an image and recognise its text.

    `page` is a pymupdf Page. Returns recognised text, or "" when nothing
    passes the confidence floor.
    """
    import numpy as np
    import pymupdf

    zoom = dpi / 72.0
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )
    if pixmap.n == 4:  # drop alpha; the recogniser expects RGB
        image = image[:, :, :3]

    result, _elapsed = _get_engine()(image)
    if not result:
        return ""

    lines = [
        text for _box, text, score in result
        if score >= MIN_DETECTION_CONFIDENCE and text.strip()
    ]
    return "\n".join(lines)


def ocr_pdf_pages(path, page_numbers: list[int], dpi: int = OCR_DPI) -> dict[int, str]:
    """OCR specific 1-indexed pages of a PDF. Returns {page_number: text}."""
    import pymupdf

    out = {}
    doc = pymupdf.open(str(path))
    try:
        for num in page_numbers:
            if 1 <= num <= doc.page_count:
                out[num] = ocr_page(doc[num - 1], dpi=dpi)
    finally:
        doc.close()
    return out


def main():
    """CLI: OCR a single PDF's low-text pages, for spot-checking quality."""
    from pathlib import Path
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print("usage: python src/ocr.py <file.pdf> [page ...]")
        return
    path = Path(sys.argv[1])
    pages = [int(a) for a in sys.argv[2:]] or [1]

    for num, text in ocr_pdf_pages(path, pages).items():
        words = len(text.split())
        print(f"--- page {num}: {words} words recognised ---")
        print(text[:600] or "(nothing recognised)")
        print()


if __name__ == "__main__":
    main()
