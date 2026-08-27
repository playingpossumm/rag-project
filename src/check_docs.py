"""Do the numbers written in the documents still match the ones measured?

This repo's most-repeated failure is a hand-typed number outliving the
measurement it came from. The README once claimed "84 evaluation cases" directly
above figures measured on 23. `eval/RESULTS.md` said 20 papers and 2,768 chunks
long after the corpus reached 36 and 5,459, under a header promising that if the
two disagreed the document was stale -- which was true, and which nobody noticed
because nothing checked it.

`build_results_doc.py --check` closed that for RESULTS.md by generating its
tables. `check_freshness.py` closed the generated chain feeding the interface.
Neither covers **HANDOFF.md §2 and §3 or the README**, which are prose with
tables in them -- the two documents a new session and a visitor actually read
first. §3's list of defaults is checked against the constants themselves, not
just for internal consistency: "Defaults, all justified by measurement" is a
claim about the code, and changing TOP_K in retrieve.py would otherwise leave
the document quietly describing a system that no longer exists.

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

# HANDOFF §2's copy of the ladder, which spells the rows differently from the
# README's and carries one more of them. Keys are the row label with emphasis
# and parentheticals stripped.
HANDOFF_LADDER = {
    "dense, no rerank": "dense, no rerank",
    "+ cross-encoder rerank": "dense + rerank",
    "+ RRF hybrid fusion": "rrf + rerank",
    "+ diversity cap 2/src": SHIPPED_ROW,
    "+ diversity cap 1/src": "+ diversity 1/src",
}


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

    # HANDOFF's own copy of the pipeline ladder. The README has one too, and a
    # table duplicated across two documents is a table that agrees until one of
    # them is edited.
    ladder = parse_table(doc, "| pipeline") or parse_table(doc, "pipeline")
    if not ladder:
        problems.append("HANDOFF.md §2: the pipeline ladder table is gone")
    else:
        ml = measured.get("llm")
        seen = 0
        for row_label, cells in ladder.items():
            key = HANDOFF_LADDER.get(re.sub(r"[*_]|\(.*?\)", "", row_label).strip())
            if not key or not ml:
                continue
            real = ml["ladder"].get(key)
            if real is None:
                problems.append(f"HANDOFF.md §2 ladder: results.json has no row "
                                f"{key!r}")
                continue
            seen += 1
            for cell, metric in zip(cells, LADDER_METRICS):
                if not close(num(cell), real[metric], written_places(cell)):
                    problems.append(
                        f"HANDOFF.md §2 ladder, {row_label} / {metric}: says "
                        f"{cell.strip()}, measured {real[metric]:.3f}")
        if seen != len(HANDOFF_LADDER):
            problems.append(f"HANDOFF.md §2 ladder: matched {seen} of "
                            f"{len(HANDOFF_LADDER)} rows -- the table has been "
                            f"renamed or reordered")
        notes.append("  §2 pipeline ladder")

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


# Which module owns each default HANDOFF §3 quotes. A name here that the
# document stops mentioning is reported, and so is one the document mentions
# that no module defines -- both are ways the paragraph drifts from the code.
DEFAULT_HOMES = {
    "CHUNK_SIZE_TOKENS": "ingest",
    "CHUNK_OVERLAP_TOKENS": "ingest",
    "USE_TITLE_PREFIX": "ingest",
    "TOP_K": "retrieve",
    "CANDIDATE_K": "retrieve",
    "DEFAULT_FUSION": "retrieve",
    "RRF_K": "hybrid",
    "DEFAULT_MAX_PER_SOURCE": "diversify",
    "ABSTAIN_THRESHOLD": "abstain",
    "DEFAULT_EXPANSION": "api",
    "PARSER_VERSION": "parse_cache",
}

# Constants an environment variable can override. Reading one that has been
# overridden and calling the document wrong would be this checker lying, so
# they are skipped with a note when the variable is set.
ENV_OVERRIDES = {
    "CHUNK_SIZE_TOKENS": "RAG_CHUNK_SIZE",
    "CHUNK_OVERLAP_TOKENS": "RAG_CHUNK_OVERLAP",
    "USE_TITLE_PREFIX": "RAG_TITLE_PREFIX",
    "ABSTAIN_THRESHOLD": "RAG_ABSTAIN_THRESHOLD",
}


def check_defaults(problems: list[str], notes: list[str]) -> bool:
    """HANDOFF §3's list of defaults, against the modules that define them.

    The paragraph opens "Defaults, all justified by measurement in
    eval/RESULTS.md", which makes each of these a claim about the code. Nothing
    checked them, so changing TOP_K in retrieve.py would have left the document
    quietly describing a system that no longer exists -- the same failure as a
    stale metric, in a place nobody thinks to look because it reads as prose.
    """
    import importlib
    import os

    doc = (ROOT / "HANDOFF.md").read_text(encoding="utf-8")
    m = re.search(r"Defaults, all justified by measurement.*?\n\n", doc, re.S)
    if not m:
        problems.append("HANDOFF.md §3: the defaults paragraph is gone -- this "
                        "checker looks for 'Defaults, all justified by measurement'")
        return False

    written = dict(re.findall(r"`([A-Z_]+)=(\"[a-z]+\"|[-\d.]+)`", m.group()))
    if not written:
        problems.append("HANDOFF.md §3: the defaults paragraph names no "
                        "`NAME=VALUE` pairs")
        return False

    for name, home in DEFAULT_HOMES.items():
        if name not in written:
            problems.append(f"HANDOFF.md §3: no longer documents {name}, which "
                            f"{home}.py still defines as a default")
    for name in written:
        if name not in DEFAULT_HOMES:
            problems.append(f"HANDOFF.md §3 documents {name}, which this checker "
                            f"does not know where to find -- add it to DEFAULT_HOMES")

    for name, raw in written.items():
        home = DEFAULT_HOMES.get(name)
        if not home:
            continue
        env = ENV_OVERRIDES.get(name)
        if env and os.environ.get(env) is not None:
            notes.append(f"  §3 {name} (skipped: {env} is set)")
            continue
        try:
            actual = getattr(importlib.import_module(home), name)
        except (ImportError, AttributeError) as exc:
            problems.append(f"HANDOFF.md §3 documents {name}={raw}, but "
                            f"{home}.py does not define it ({exc})")
            continue

        want = raw.strip('"')
        if isinstance(actual, bool):
            # Written as 0/1 in the document, held as a bool in the code.
            ok = bool(actual) == (want not in ("0", "false", "False"))
        elif isinstance(actual, (int, float)):
            ok = float(want) == float(actual)
        else:
            ok = want == str(actual)
        if not ok:
            problems.append(f"HANDOFF.md §3 says {name}={raw}, {home}.py has "
                            f"{name}={actual!r}")
        notes.append(f"  §3 {name}")

    # RRF_K is defined in two modules. They agree today; nothing makes them.
    # Fusion damping and the rerank blend's damping are the same constant by
    # intention, and by copy in practice.
    try:
        import hybrid
        import rerank
        if hybrid.RRF_K != rerank.RRF_K:
            problems.append(
                f"RRF_K is defined twice and the copies disagree: "
                f"hybrid.py has {hybrid.RRF_K}, rerank.py has {rerank.RRF_K}. "
                f"Fusion and the rerank blend would damp differently.")
        notes.append("  RRF_K agrees between hybrid.py and rerank.py")
    except ImportError:
        pass
    return True


# Number words, because the notes are written as prose and count in words as
# often as in digits. Only what actually appears in them -- inventing a general
# parser would be more code than the thing it checks.
WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "twenty-five": 25, "twenty-six": 26, "thirty": 30, "thirty-five": 35,
    "sixty-six": 66, "sixty-seven": 67,
}


def as_count(token: str):
    token = token.lower().strip()
    if token.isdigit():
        return int(token)
    return WORDS.get(token)


def check_notes(measured: dict, problems: list[str], notes: list[str]) -> bool:
    """corpora.json's per-corpus notes carry measured claims in prose.

    Each note argues why that corpus ships the threshold it does, in sentences
    like "at 0.0 the gate wrongly refuses 1 of 66 answerable questions and
    catches 9 of 18 adversarial". Those are measurements, written by hand, next
    to the setting they justify -- the most persuasive place in the repo for a
    number to be wrong, because it is read as the reason for a decision.

    It had drifted: the ML note said 1 of 66 and 9 of 18 while the golden set
    holds 67 answerable and 17 adversarial.

    Only claims of the form "refuses N of M" / "catches N of M" are checked, and
    every one that is checked is listed, so the coverage is visible rather than
    assumed. A note may argue anything else it likes in prose.
    """
    import corpora as _c

    raw = json.loads((ROOT / "corpora.json").read_text(encoding="utf-8"))
    reg = _c.registry()
    checked = 0

    for name, entry in raw.items():
        note = entry.get("note") or ""
        cfg = reg.get(name)
        if not note or not cfg or name not in measured:
            continue
        m = measured[name]
        n_ans, n_adv = m["answerable"], m["adversarial"]

        for verb, count, total in re.findall(
                r"(refuse[sd]?|catch(?:es|)|caught)\s+"
                r"(?:the\s+same\s+|every\s+one\s+of\s+the\s+|the\s+)?"
                r"([\w-]+)\s+of\s+(?:the\s+)?([\w-]+)", note):
            got_total = as_count(total)
            if got_total is None:          # "of the corpus", "of them" -- prose
                continue
            checked += 1
            # Only the DENOMINATOR is compared. A note may quote a superseded
            # numerator on purpose -- the quant note reports that "the earlier
            # note claimed 0.0 refused nineteen of thirty-five", which is an
            # accurate statement about a wrong number, and flagging it would
            # be the check misreading history as drift. The corpus size in that
            # sentence is still checkable and still right.
            want_total = n_adv if verb.startswith("catch") else n_ans
            if got_total != want_total:
                problems.append(
                    f"corpora.json, {name}: the note says {verb} "
                    f"{count} of {total}, but the golden set holds {want_total} "
                    f"{'adversarial' if verb.startswith('catch') else 'answerable'} cases")
            notes.append(f"  corpora.json {name}: {verb} … of {total}")

    if not checked:
        problems.append("corpora.json: no note makes a checkable "
                        "'refuses N of M' claim -- either they were rewritten "
                        "or this check has stopped matching them")
        return False
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
    ok = check_defaults(problems, notes) and ok
    ok = check_notes(measured, problems, notes) and ok

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
          "measurement it came from,\nevery default §3 lists matches the "
          "constant that defines it, and the corpus\nnotes agree with the "
          "golden sets they argue about")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
