"""Do the numbers written in the documents still match the ones measured?

This repo's most-repeated failure is a hand-typed number outliving the
measurement it came from. The README once claimed "84 evaluation cases" directly
above figures measured on 23. `eval/RESULTS.md` said 20 papers and 2,768 chunks
long after the corpus reached 36 and 5,459, under a header promising that if the
two disagreed the document was stale -- which was true, and which nobody noticed
because nothing checked it.

`build_results_doc.py --check` closed that for RESULTS.md by generating its
tables. `check_freshness.py` closed the generated chain feeding the interface.
Neither covers **HANDOFF.md §2 and the README**, which are prose with tables in
them -- the two documents a new session and a visitor actually read first.

Generating them is the wrong fix: the surrounding argument is judgement and
cannot come from a JSON file, and a generated §2 would lose the reasons each
number matters. So the numbers stay hand-written and are *checked* instead. Each
row below names where its truth lives, and a row this cannot find is a failure
rather than a skip -- a checker that quietly matches nothing is worse than none,
because it reports success.

    .venv\\Scripts\\python.exe src\\check_docs.py

Exit code is the contract: 0 the documents agree with the measurements, 1 they
have drifted, 2 a table this expects to find is gone.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import corpora  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
SHIPPED_ROW = "+ diversity 2/src"       # the configuration api.ask() serves

# HANDOFF §2's table, in the column order it is written in.
COLUMN_ORDER = ["llm", "birds", "quant"]


def num(text: str) -> float | None:
    """The first number in a cell, tolerating the ways they are written.

    Bold markers, thousands separators, a signed value, and the typographic
    minus the documents use in prose -- U+2212, which is not the hyphen Python
    parses. That last one is why this exists rather than a float() call: the
    bird threshold is written -5.5 with a character float() rejects.
    """
    cleaned = (text.replace("**", "").replace(",", "")
               .replace("−", "-").replace("–", "-").strip())
    m = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
    return float(m.group()) if m else None


def parse_table(doc: str, first_cell: str) -> dict[str, list[str]] | None:
    """A markdown table, keyed by the label in each row's first cell.

    Located by a row that must exist rather than by position, so inserting a
    paragraph above it does not silently move the check onto another table.
    """
    rows: dict[str, list[str]] = {}
    started = False
    for line in doc.splitlines():
        if not line.startswith("|"):
            if started:
                break
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not cells:
            continue
        if cells[0].startswith(first_cell):
            started = True
        if started:
            rows[cells[0]] = cells[1:]
    return rows or None


def load_results(cfg: dict) -> dict | None:
    from check_freshness import artefact
    p = artefact("results", cfg["golden"])
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def truth() -> dict[str, dict]:
    """Every quantity the documents quote, from where it is measured."""
    out = {}
    for name, cfg in corpora.registry().items():
        if not cfg["indexed"] or not cfg["golden"]:
            continue
        res = load_results(cfg)
        if not res:
            continue
        stats = corpora.stats(cfg)
        e2e = res["end_to_end"].get(SHIPPED_ROW, {})
        ab = res.get("abstention", {})
        out[name] = {
            "label": cfg["label"],
            "documents": stats.get("documents"),
            "passages": stats.get("chunks"),
            "answerable": res["corpus"]["answerable_cases"],
            "adversarial": res["corpus"]["adversarial_cases"],
            "any-hit@5": e2e.get("hit_rate"),
            "MRR": e2e.get("mrr"),
            "NDCG": e2e.get("ndcg"),
            "source recall": e2e.get("src_recall"),
            "abstention threshold": cfg["threshold"] if cfg["calibrated"] else 0.0,
            "rerank blend": cfg["rerank_blend"],
            "answerable median": ab.get("answerable_median"),
            "ladder": res["end_to_end"],
        }
    return out


# Which §2 row reads which measured quantity. Rows carrying two numbers -- the
# case counts -- are handled separately below.
HANDOFF_ROWS = {
    "documents": "documents",
    "passages": "passages",
    "any-hit@5": "any-hit@5",
    "MRR": "MRR",
    "source recall": "source recall",
    "abstention threshold": "abstention threshold",
    "rerank blend": "rerank blend",
    "answerable median": "answerable median",
}

# The README's ladder block, which is fenced rather than a table: each line is
# a label and four numbers, and the label maps to a row of results.json.
README_LADDER = {
    "naive RAG": "dense, no rerank",
    "+ cross-encoder rerank": "dense + rerank",
    "+ hybrid BM25 fusion": "rrf + rerank",
    "+ diversity cap (2/src)": SHIPPED_ROW,
}
LADDER_METRICS = ["hit_rate", "mrr", "ndcg", "src_recall"]


def written_places(cell: str) -> int:
    """How many decimals the document itself wrote.

    Compare at the precision the sentence claims, not at the precision the
    measurement happens to carry. "+4.93" against 4.934403419494629 is a
    document rounding correctly, and reporting it as drift would train a reader
    to ignore this checker -- which is how a check stops being read.
    """
    m = re.search(r"\.(\d+)", cell.replace(",", ""))
    return min(len(m.group(1)), 4) if m else 0


def close(a, b, places=3) -> bool:
    if a is None or b is None:
        return False
    return round(float(a), places) == round(float(b), places)


def check_handoff(measured: dict, problems: list[str], notes: list[str]) -> bool:
    doc = (ROOT / "HANDOFF.md").read_text(encoding="utf-8")
    table = parse_table(doc, "documents")
    if not table:
        problems.append("HANDOFF.md: the §2 measured-state table is gone -- "
                        "this checker cannot find a row starting `| documents`")
        return False

    cols = [c for c in COLUMN_ORDER if c in measured]
    for row_label, key in HANDOFF_ROWS.items():
        cells = table.get(row_label)
        if cells is None:
            problems.append(f"HANDOFF.md §2: no row {row_label!r}")
            continue
        for i, name in enumerate(cols):
            if i >= len(cells):
                problems.append(f"HANDOFF.md §2: row {row_label!r} has no column "
                                f"for {measured[name]['label']}")
                continue
            written, real = num(cells[i]), measured[name][key]
            places = written_places(cells[i])
            if not close(written, real, places):
                problems.append(
                    f"HANDOFF.md §2, {row_label} / {measured[name]['label']}: "
                    f"says {cells[i].strip()}, measured {real}")
        notes.append(f"  §2 {row_label}")

    cases = table.get("cases")
    if cases is None:
        problems.append("HANDOFF.md §2: no row 'cases'")
    else:
        for i, name in enumerate(cols):
            if i >= len(cases):
                continue
            found = [int(x) for x in re.findall(r"\d+", cases[i])]
            want = [measured[name]["answerable"], measured[name]["adversarial"]]
            if found[:2] != want:
                problems.append(
                    f"HANDOFF.md §2, cases / {measured[name]['label']}: says "
                    f"{cases[i].strip()}, measured {want[0]} + {want[1]} adv")
        notes.append("  §2 cases")
    return True


def check_readme(measured: dict, problems: list[str], notes: list[str]) -> bool:
    doc = (ROOT / "README.md").read_text(encoding="utf-8")
    ml = measured.get("llm")
    if not ml:
        problems.append("README.md: the ML corpus has no results.json to check against")
        return False

    seen = 0
    for line in doc.splitlines():
        stripped = line.strip()
        for label, row in README_LADDER.items():
            if not stripped.startswith(label):
                continue
            seen += 1
            found = re.findall(r"\d+\.\d+", stripped)
            real = ml["ladder"].get(row)
            if real is None:
                problems.append(f"README.md ladder: results.json has no row {row!r}")
                continue
            if len(found) < 4:
                problems.append(f"README.md ladder, {label!r}: expected four "
                                f"numbers, found {len(found)}")
                continue
            for value, metric in zip(found, LADDER_METRICS):
                if not close(float(value), real[metric], written_places(value)):
                    problems.append(
                        f"README.md ladder, {label} / {metric}: says {value}, "
                        f"measured {real[metric]:.3f}")
            notes.append(f"  README ladder {label}")
    if seen != len(README_LADDER):
        problems.append(f"README.md: found {seen} of {len(README_LADDER)} ladder "
                        f"rows -- the block this checks has moved or been renamed")

    # The corpus table: documents, passages and the threshold, per corpus.
    table = parse_table(doc, "| set")
    if table is None:
        table = parse_table(doc, "set")
    if not table:
        problems.append("README.md: the three-corpora table is gone")
        return False
    labels = {measured[n]["label"]: n for n in measured}
    matched = 0
    for row_label, cells in table.items():
        name = labels.get(row_label)
        if not name or len(cells) < 4:
            continue
        matched += 1
        m = measured[name]
        for cell, key in ((cells[0], "documents"),
                          (cells[1], "passages"),
                          (cells[3], "abstention threshold")):
            if not close(num(cell), m[key], written_places(cell)):
                problems.append(f"README.md corpora table, {row_label} / {key}: "
                                f"says {cell.strip()}, measured {m[key]}")
    if matched != len(measured):
        problems.append(f"README.md corpora table: matched {matched} of "
                        f"{len(measured)} configured corpora by label")
    notes.append("  README corpora table")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    measured = truth()
    if not measured:
        print("no results*.json to check the documents against -- run "
              "src/evaluate.py first")
        return 2

    problems: list[str] = []
    notes: list[str] = []
    ok = check_handoff(measured, problems, notes)
    ok = check_readme(measured, problems, notes) and ok

    if not args.quiet and notes:
        print("checked:")
        for n in notes:
            print(n)
    print()
    for p in problems:
        print(f"  DRIFTED  {p}")
    if not ok:
        print("\na table this checker expects is missing, so it checked less "
              "than it claims to.\nFix the anchor rather than deleting the check.")
        return 2
    if problems:
        print(f"\n{len(problems)} number(s) in the documents no longer match the "
              f"measurements.\nThe measurement is the truth; edit the document.")
        return 1
    print("every number quoted in HANDOFF.md §2 and README.md matches the "
          "measurement it came from")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
