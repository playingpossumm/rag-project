"""Round-trip every supported format: write a real file, read it back.

Until 2026-08-21 `load_docx`, `load_pptx` and `load_xlsx` had never run on a
single file. HANDOFF section 7 called that the largest unknown in the system and
the spreadsheet path the least trustworthy of the three, "because rows are
serialised to `Column: value` on an assumption with no evidence behind it".

That assumption was wrong, and this is the test that found it. A sheet whose
first row is a TITLE rather than headers -- which is what spreadsheets in the
wild look like -- made every row read

    Q3 sales report: 41200

with the title repeated as a column name, the real column names demoted to the
first data row, and the values unlabelled. Nothing errored.

The files are written here rather than committed, so the test carries no
fixtures and cannot go stale against them. It builds each format with the
structure that actually breaks loaders -- a table in the middle of a Word
document, speaker notes behind slide fragments, a title row above a header row --
and asserts the content comes back.

    .venv\\Scripts\\python.exe src\\test_loaders.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

from loaders import load_document
from make_documents import write_docx, write_pptx, write_xlsx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ART = {
    "title": "Test Subject",
    "text": "\n".join([
        "An opening paragraph that establishes the subject in ordinary prose.",
        "A second paragraph carrying a distinctive phrase: MARKER_OVERVIEW.",
        "== Anatomy ==",
        "Anatomy discussion with the phrase MARKER_ANATOMY inside it.",
        "More anatomy prose so the section has enough body to survive chunking.",
        "A third anatomy line to give the table something to quote.",
        "== Behaviour ==",
        "Behavioural notes containing MARKER_BEHAVIOUR for retrieval.",
        "Supporting behavioural detail in a second paragraph.",
        "== Distribution ==",
        "Distribution prose with MARKER_DISTRIBUTION as its distinctive token.",
        "Second distribution paragraph.",
        "== References ==",
        "This section must be dropped by the section splitter.",
    ]),
}

CHECKS = []


def check(name, got, want=True):
    CHECKS.append((name, got, want, got == want))


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="rag-loaders-"))
    try:
        # ---------------------------------------------------------- docx --
        docx_path = write_docx(ART, tmp / "subject.docx")
        units = load_document(docx_path)
        kinds = {u["locator"]["kind"] for u in units}
        text = " ".join(u["text"] for u in units)
        check("docx / produced units", len(units) > 0)
        check("docx / splits at headings", "section" in kinds)
        check("docx / prose recovered", "MARKER_BEHAVIOUR" in text)
        # A loader reading only document.paragraphs loses every table silently.
        check("docx / tables are read too", "table" in kinds)
        check("docx / table header recovered", "Aspect" in text)
        check("docx / references section dropped",
              "must be dropped" not in text)

        # ---------------------------------------------------------- pptx --
        pptx_path = write_pptx(ART, tmp / "subject.pptx")
        units = load_document(pptx_path)
        text = " ".join(u["text"] for u in units)
        check("pptx / produced units", len(units) > 0)
        check("pptx / one unit per slide",
              {u["locator"]["kind"] for u in units} == {"slide"})
        # Notes hold the sentences; slides hold fragments. Losing notes loses
        # the most answerable content in the deck.
        check("pptx / speaker notes recovered", "MARKER_ANATOMY" in text)

        # ---------------------------------------------------------- xlsx --
        xlsx_path = write_xlsx(ART, tmp / "subject.xlsx")
        units = load_document(xlsx_path)
        check("xlsx / produced units", len(units) > 0)

        notes = [u for u in units if u["locator"].get("value") == "Notes"]
        check("xlsx / title-first sheet still parsed", len(notes) > 0)
        if notes:
            body = " ".join(u["text"] for u in notes)
            # THE REGRESSION. Before the fix the header was assumed to be row 1,
            # so the sheet title became the column name for every single row.
            check("xlsx / real column names used as labels", "Topic:" in body)
            check("xlsx / sheet title NOT used as a column name",
                  "working notes:" not in body)
            check("xlsx / row numbers skip past the title row",
                  notes[0]["locator"].get("rows", "").startswith("3"))

        # a sheet that genuinely starts with headers must still work
        sect = [u for u in units if u["locator"].get("value") == "Sections"]
        if sect:
            check("xlsx / header-first sheet unaffected",
                  "Section:" in " ".join(u["text"] for u in sect))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    width = max(len(n) for n, *_ in CHECKS)
    failed = 0
    for name, got, want, ok in CHECKS:
        if not ok:
            failed += 1
            print(f"FAIL  {name:<{width}}  got {got!r}, want {want!r}")
    print(f"{len(CHECKS) - failed}/{len(CHECKS)} loader checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
