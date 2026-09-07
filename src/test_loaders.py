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

from loaders import _unhash, load_document
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
        # ---- titles split by the PDF converter ---------------------------
        # A cover-page title comes apart two ways: it wraps and only the first
        # line keeps its heading marker, or each line is promoted to its own
        # heading. Both left the index holding half a title. The joining is
        # guarded, and the guards are what these check -- gluing a title to the
        # section heading below it would be worse than the truncation.
        from loaders import _rejoin_wrapped

        JOIN = [
            ("wrapped onto a plain line",
             "OenoBench: A Wine-Domain Benchmark for",
             [" Knowledge-Grounded Evaluation of Large Language Models", ""],
             "Knowledge-Grounded"),
            ("split across two headings, title ends on a dangling word",
             "A GENERIC NONPARAMETRIC VALUE-AT-RISK ESTIMATOR FOR",
             ["", "### HIGH DIMENSIONS", "", "A PREPRINT"],
             "HIGH DIMENSIONS"),
            ("wrapped, then split across a second heading",
             "Multi-Agent Orchestration with the Common-Sense",
             [" Reasoning Capabilities of LLMs for Autonomous", "", "", "## Driving"],
             "Driving"),
        ]
        for name, head, rest, want in JOIN:
            check(f"title / {name}", want in _rejoin_wrapped(head, rest, 2))

        KEEP = [
            ("the section heading below a finished title",
             "Attention Is All You Need", ["", "### Abstract", ""]),
            # Without requiring evidence of an earlier wrap this one joins into
            # "Some Paper Title Model Architecture".
            ("a same-depth section heading with no prior wrap",
             "Some Paper Title", ["", "## Model Architecture", ""]),
            ("an author block",
             "Some Paper Title", ["**Ada Lovelace**", "Institute"]),
            ("an email line",
             "Some Paper Title", ["ada@example.org"]),
            ("anything at all, after a full stop",
             "A Complete Title.", ["", "## Anything", ""]),
        ]
        for name, head, rest in KEEP:
            check(f"title / does not absorb {name}",
                  _rejoin_wrapped(head, rest, 2) == head)

        # An author line is promoted to a heading as readily as half a title,
        # so it arrives looking exactly like a continuation. This one shipped
        # past the earlier guards -- a plain wrap came first, and the authors
        # sat at the same depth -- and was caught only by dry-running before
        # writing. Two signals stop it: a comma-separated run of title-case
        # pairs, and an email directly below.
        # The names and addresses here were the real ones from the paper
        # until 2026-09-07 and are invented now. This repository became
        # public and 2 academics' addresses were not this fixture's to
        # publish; the shape the check reads, a comma-separated run of
        # title-case pairs with an email under it, is unchanged.
        AUTHORS = [" Generative Models Must Preserve", "", "",
                   "## Priya Ramanathan, Devesh Iyer", "", "```",
                   "   p.ramanathan@example.edu, d.iyer@example.edu"]
        got = _rejoin_wrapped(
            "Simulating Stress Laws under Extremal Dependence: Characterizing What",
            AUTHORS, 2)
        check("title / still joins the wrapped half", "Must Preserve" in got)
        check("title / does not absorb an author heading", "Priya" not in got)

        # A title ending on a dangling word earns a longer continuation than
        # one whose only evidence is position. Seven words was being refused by
        # a flat cap of five, which left the last truncated title in the index.
        got = _rejoin_wrapped(
            "Communicating Credit Risk with Large Language Models: Evaluation of",
            ["", "", "## Explanations from Standard and Alternative Data-Based Models",
             "", "Sahab Zandi[a], Noah Kostesku[b]"], 2)
        check("title / a dangling word earns a longer continuation",
              got.endswith("Data-Based Models"))
        check("title / the weak path keeps the tight cap",
              _rejoin_wrapped("Some Paper Title",
                              ["", "## Model Architecture Design Choices Here", ""], 2)
              == "Some Paper Title")

        # ...while a real second half that happens to have an affiliation below
        # it is still joined. Keying the rejection on affiliation words rather
        # than an address would have broken this one.
        got = _rejoin_wrapped(
            "Multi-Agent Orchestration with the Common-Sense",
            [" Reasoning Capabilities of LLMs for Autonomous", "", "", "## Driving",
             "", "", "**Mehdi Azarafza**", "Department of Computer Science"], 2)
        check("title / joins a half whose affiliation sits below it",
              got.endswith("Autonomous Driving"))

        # A heading run stranded mid-line. Both real cases are mid-word, which
        # is why the default is to close the gap rather than leave a space.
        check("title / a mid-word heading run closes up",
              _unhash("LORA: LOW-RANK ADAPTATION OF LARGE LAN### GUAGE MODELS")
              == "LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS")
        check("title / and the other one in this corpus",
              _unhash("FORMALTCS: BENCHMARKING END-TO-END FRON### TIER FORMAL "
                      "THEORETICAL COMPUTER SCIENCE")
              == "FORMALTCS: BENCHMARKING END-TO-END FRONTIER FORMAL "
                 "THEORETICAL COMPUTER SCIENCE")
        check("title / a run between two words leaves one space",
              _unhash("Attention Is All ### You Need")
              == "Attention Is All You Need")
        check("title / a title with no run is untouched",
              _unhash("Deep Residual Learning for Image Recognition")
              == "Deep Residual Learning for Image Recognition")
        # A single hash is not a stranded heading marker -- C# is a language,
        # and "#1" is an ordinal. Only a run of two or more is one.
        check("title / a lone hash is left alone",
              _unhash("Benchmarking C# Compilers") == "Benchmarking C# Compilers")

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
