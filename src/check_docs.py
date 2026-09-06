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
have drifted, 2 this checked less than it claims to -- a table it expects to
find is gone, or under `--tests` a suite exited 2 to say it did not run.

A suite that did not run is neither passed nor failed. `test_ocr.py` exits 2
without RapidOCR, and this reads the `0/N` on its last line as the number of
checks the suite declares, holds the sentence to that number, counts it into
the total so the arithmetic is still checked, and exits 2 because the checks
themselves were not seen to pass. Until 2026-09-06 that suite exited 0 on the
skip path with no count line, and `--tests` reported it as "no count on its
last line", a failure it was not.
"""
import argparse
import json
import re
import subprocess
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


def served_routes() -> set[str]:
    """Every route serve.py dispatches on, read from its syntax tree.

    The dispatch is a chain of `route == "..."`, `route in (...)` and
    `route.startswith("...")`. A regex over that picks up whichever quoted
    strings sit nearby; the tree says which ones are actually compared against
    the request path.
    """
    import ast

    tree = ast.parse((ROOT / "src" / "serve.py").read_text(encoding="utf-8"))
    found: set[str] = set()

    def literals(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return [node.value]
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            out = []
            for elt in node.elts:
                out += literals(elt)
            return out
        return []

    # `route in QUALITY_FILES`, a module-level table. Until 2026-09-07 a
    # comparator that was a name rather than a literal contributed nothing,
    # so the three routes served through that table were invisible here
    # and undocumented in HANDOFF §6 without this check noticing.
    tables: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            value = node.value
            if isinstance(value, ast.Dict):
                tables[node.targets[0].id] = [
                    v for key in value.keys if key is not None for v in literals(key)]
            else:
                tables[node.targets[0].id] = literals(value)

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) \
                and node.left.id == "route":
            for comp in node.comparators:
                if isinstance(comp, ast.Name) and comp.id in tables:
                    found.update(tables[comp.id])
                else:
                    found.update(literals(comp))
        # `route.startswith("/fonts/")` -- a prefix route, recorded with its
        # wildcard so it reads the way the document writes it.
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "startswith" \
                and isinstance(node.func.value, ast.Name) \
                and node.func.value.id == "route":
            for arg in node.args:
                found.update(v.rstrip("/") + "/*" for v in literals(arg))

    # serve.py rstrips the trailing slash before dispatching, so the app route
    # is compared against the empty string. It is "/" everywhere a person
    # writes it down.
    if "" in found:
        found = (found - {""}) | {"/"}
    return {r for r in found if r.startswith("/")}


def check_commits(problems: list[str], notes: list[str]) -> bool:
    """Every commit these documents name still has to exist.

    The decision not to rewrite this repository's history rests on the fact that
    its own prose cites its own SHAs -- HANDOFF §5, the engineering log, and
    every entry in the rendered changelog. A rewrite would break all of them,
    and it would
    break them silently: a dangling SHA in a sentence reads exactly like a live
    one.

    So the risk that argued against the rewrite is now checked instead of
    described. It also catches the ordinary version of the same mistake --
    quoting a SHA from a branch that was never merged, or mistyping one.
    """
    import subprocess

    root = Path(__file__).resolve().parent.parent
    try:
        listed = subprocess.run(["git", "ls-files", "*.md", "*.py", "*.html"],
                                cwd=root, capture_output=True, text=True,
                                check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        notes.append("  commit references (no git here -- not checked)")
        return True

    # Seven to eight lowercase hex characters, standing alone. Longer runs are
    # digests, not abbreviated SHAs, and this project has plenty of those.
    pattern = re.compile(r"(?<![0-9a-zA-Z])[0-9a-f]{7,8}(?![0-9a-zA-Z])")
    # A UUID's first block looks exactly like a short SHA, and §8 writes the
    # published artifacts' ids bare and in backticks the same way it writes
    # commits. What separates them is that the full uuid is in the same file:
    # if `c8fef8f2` also appears as `c8fef8f2-2060-...`, it is an artifact.
    uuid_head = re.compile(r"\b([0-9a-f]{8})-[0-9a-f]{4}-")
    cited: dict[str, str] = {}
    for name in listed:
        path = root / name
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        ids = set(uuid_head.findall(text))
        for sha in pattern.findall(text):
            if sha in ids:
                continue
            cited.setdefault(sha, name)

    # Only the ones git recognises as a commit are claims about this history.
    # A hex run that happens to be seven characters long is not a reference,
    # and asking git is cheaper than trying to tell them apart by eye.
    live, dead = 0, []
    for sha, where in sorted(cited.items()):
        found = subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"],
                               cwd=root, capture_output=True)
        if found.returncode == 0:
            live += 1
        elif re.search(rf"`{sha}`|commit `?{sha}", (root / where).read_text(
                encoding="utf-8", errors="ignore")):
            # Backticked or introduced by the word "commit": prose meant it as
            # a reference, and it does not resolve.
            dead.append((sha, where))

    for sha, where in dead:
        problems.append(f"{where} cites commit {sha}, which this repository "
                        f"does not contain -- a rewritten history or a typo, "
                        f"and both read identically in a sentence")
    notes.append(f"  {live} commit reference(s) in prose, all resolving")
    return True


def check_routes(problems: list[str], notes: list[str]) -> bool:
    """HANDOFF §6's route list against the dispatch itself.

    Both directions matter, and the second is the one that bit. A documented
    route that does not exist is a broken promise a reader finds immediately. A
    route that exists and is documented nowhere is a surface nobody thinks to
    test -- which is how `POST /ask` served the wrong corpus for its whole life
    while the interface, which calls `/api/chat`, looked correct.
    """
    doc = (ROOT / "HANDOFF.md").read_text(encoding="utf-8")
    m = re.search(r"- Routes:(.*?)\n- ", doc, re.S)
    if not m:
        problems.append("HANDOFF.md §6: the route list is gone -- this checker "
                        "looks for a line starting '- Routes:'")
        return False

    documented = {r.rstrip("`.,·") for r in re.findall(r"`(/[^`]*)`", m.group(1))}
    # The architecture sketch in §3 names the two POST routes as well, and a
    # route documented in either place is documented.
    documented |= {r.rstrip("`.,·") for r in re.findall(r"(?:POST|GET\|POST) (/\S+)", doc)}
    served = served_routes()

    # A documented wildcard covers everything under it: "/api/index/*" is how
    # the architecture sketch writes the three index routes, and expanding it
    # into three lines would make the document worse to read. It only ever
    # EXCUSES an undocumented route -- it must not remove a served one, or the
    # explicitly documented members start reporting as missing.
    prefixes = tuple(r[:-1] for r in documented if r.endswith("/*"))
    covered = {r for r in served if prefixes and r.startswith(prefixes)}

    for route in sorted(served - documented - covered):
        problems.append(f"HANDOFF.md §6: serve.py serves {route} and no "
                        f"document lists it -- an undocumented route is one "
                        f"nobody thinks to test")
    for route in sorted(documented - served):
        if route.endswith("/*"):
            continue          # a prefix route; its members were folded in above
        if route.rstrip("/*") in {r.rstrip("/*") for r in served}:
            continue
        problems.append(f"HANDOFF.md §6 lists {route}, which serve.py does not "
                        f"dispatch on")
    wild = sum(1 for r in documented if r.endswith("/*"))
    notes.append(f"  §6 routes ({len(served)} served, "
                 f"{len(documented) - wild} documented"
                 + (f" + {wild} wildcard" if wild else "") + ")")
    return True


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


# --------------------------------------------------------------- test counts
# Which prose word in HANDOFF's "Test counts" sentence means which suite.
# Written out rather than derived: a checker that guesses would stop covering a
# suite the day somebody rewords the sentence, which is the failure it exists
# to catch, one level up.
SUITE_LABELS = {
    "test_trace": "trace",
    "test_excerpt": "excerpt",
    "test_links": "links",
    "test_evaluate_answers": "answer-quality judge",
    "test_loaders": "loaders",
    "test_freshness": "freshness",
    "test_serve": "serve",
    "test_generate_local": "local generation",
    "test_metrics": "metrics",
    "test_generate": "generate",
    "test_golden": "golden-set audit",
    "test_analytics": "analytics",
    "test_api": "api",
    "test_rerank": "reranker cache",
    "test_ingest_cache": "ingest cache",
    "test_ocr": "OCR",
}
# Counted in the same sentence and run by node, not python.
NODE_SUITES = (("ui/test-answer-mark.mjs", "answer-highlight"),
               ("ui/test-passages.mjs", "passage-selection"))
# Documented as "plus the route suite" and deliberately outside the total,
# because it starts its own server.
UNCOUNTED = {"test_routes"}

COUNT_LINE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\b")


def label_pattern(label: str) -> str:
    """Match "24 local generation" even where the label wraps across a line.

    These documents are hard-wrapped, so a two-word label is routinely split by
    a newline. Escaping the label whole requires a literal space and silently
    matches nothing, which is how this checker first reported that the sentence
    gave no number for `local generation` while the sentence plainly did.
    """
    return (r"(\d+)\s+"
            + r"\s+".join(re.escape(w) for w in label.split())
            + r"\b")


def run_suite(cmd: list[str], cwd: Path) -> tuple[str, int | None, str]:
    """Run one suite and read the count off its last line.

    Returns (status, count, note). Status is "ok" with the number of checks
    that passed, "failed" with count None, because a count from a failing suite
    is not a number anyone should compare against, or "not run" for a suite
    that exited 2 to say its checks did not execute -- test_ocr.py does so
    without RapidOCR. For "not run" the count is the N of the `0/N` line the
    suite printed, which is how many checks it declares, or None if it printed
    no such line.
    """
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           timeout=900, encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError) as exc:        # noqa: BLE001
        return "failed", None, f"could not run: {exc}"
    tail = [ln for ln in (r.stdout or "").splitlines() if ln.strip()]
    last = tail[-1] if tail else ""
    m = COUNT_LINE.match(last)
    if r.returncode == 2:
        declared = int(m.group(2)) if m else None
        return "not run", declared, f"NOT RUN ({last.strip()[:60]})"
    if r.returncode != 0:
        return "failed", None, f"FAILED ({last.strip()[:60]})"
    if not m:
        return "failed", None, f"no count on its last line: {last.strip()[:60]!r}"
    passed, total = int(m.group(1)), int(m.group(2))
    if passed != total:
        return "failed", None, f"{passed} of {total} passed"
    return "ok", passed, ""


# A per-suite count written beside the suite's name in the README's prose,
# such as "`src/test_evaluate_answers.py` holds it to 79 checks", rather than
# in HANDOFF's test-count sentence. Held to the run count the same way. Until
# 2026-09-06 nothing read these, so a suite could grow and the README would
# keep quoting the size it had when the paragraph was written.
SUITE_COUNT_IN_PROSE = re.compile(
    r"`src/(test_[a-z_]+)\.py`[^`]{0,120}?\b(\d+) checks\b")


def suite_counts_in_prose(text: str) -> list[tuple[str, int]]:
    """Every (module, count) the prose states beside a suite's file name."""
    return [(mod, int(n)) for mod, n in SUITE_COUNT_IN_PROSE.findall(one_line(text))]


def check_tests(problems: list[str], notes: list[str]) -> bool:
    """Every count in HANDOFF's test-count sentence, against the suites.

    Returns False when a suite did not run, so the run exits 2. Its declared
    count is still held to the sentence and still summed into the total, and
    the suite is reported as NOT RUN rather than as passed or failed. The
    node suites are held exactly as the python ones are. Until 2026-09-06 a
    node suite that failed, or whose file was gone, was appended to the notes
    and the run still exited 0.
    """
    handoff = (ROOT / "HANDOFF.md").read_text(encoding="utf-8")
    sentence = re.search(r"Test counts,[^\n]*(?:\n(?!\n)[^\n]*)*", handoff)
    if not sentence:
        print("  the 'Test counts' sentence is gone from HANDOFF.md -- this "
              "checker looks for a line starting 'Test counts,'")
        return False
    text = sentence.group(0)

    py = sorted(p.stem for p in (ROOT / "src").glob("test_*.py"))
    undocumented = [m for m in py
                    if m not in SUITE_LABELS and m not in UNCOUNTED]
    for m in undocumented:
        problems.append(f"src/{m}.py exists and the test-count sentence does "
                        f"not mention it")

    def hold(label: str, count: int, verb: str) -> None:
        """The sentence's number for `label` against `count`."""
        m = re.search(label_pattern(label), text)
        if not m:
            problems.append(f"the sentence gives no number for '{label}', "
                            f"which {verb} {count}")
        elif int(m.group(1)) != count:
            problems.append(f"'{label}' is written as {m.group(1)} and {verb} "
                            f"{count}")
        else:
            notes.append(f"    {'ok   ' if verb == 'runs' else 'NOT RUN'} "
                         f"{label:<22} {count}")

    subtotal = 0
    ran: dict[str, int] = {}
    not_run: list[str] = []
    for mod, label in SUITE_LABELS.items():
        path = ROOT / "src" / f"{mod}.py"
        if not path.exists():
            problems.append(f"the sentence counts '{label}' and "
                            f"src/{mod}.py does not exist")
            continue
        status, count, why = run_suite([sys.executable, str(path)], ROOT)
        if status == "not run":
            not_run.append(f"src/{mod}.py {why}")
            if count is None:
                problems.append(f"src/{mod}.py {why} and printed no 0/N line, "
                                f"so the sentence's number for '{label}' and "
                                f"the total cannot be checked")
                continue
            subtotal += count
            hold(label, count, "declares")
            continue
        if status == "failed":
            problems.append(f"src/{mod}.py {why}")
            continue
        ran[mod] = count
        subtotal += count
        hold(label, count, "runs")

    node_total = 0
    for node_path, node_label in NODE_SUITES:
        status, node_count, why = run_suite(["node", str(ROOT / node_path)], ROOT)
        if status != "ok":
            problems.append(f"{node_path} {why}")
            continue
        hold(node_label, node_count, "runs")
        node_total += node_count

    # README.md and HANDOFF.md state a few suite sizes in their prose, beside
    # the file name. Only suites that ran are compared; a declared count from
    # a suite that did not run is already reported above. HANDOFF.md joined
    # the scan on 2026-09-07, when three of its counts had gone stale by
    # 38, 18 and 1 checks, and the window after the file name widened from
    # 80 to 120 characters to reach the count in its longest sentence.
    for doc in ("README.md", "HANDOFF.md"):
        for mod, said in suite_counts_in_prose(
                (ROOT / doc).read_text(encoding="utf-8")):
            count = ran.get(mod)
            if count is None:
                notes.append(f"    note  {doc} says src/{mod}.py has {said} "
                             f"checks; that suite did not run here")
            elif said != count:
                problems.append(f"{doc} says src/{mod}.py has {said} checks and "
                                f"it runs {count}")
            else:
                notes.append(f"    ok    {doc} src/{mod}.py     {count}")

    # The two totals the sentence states: the python subtotal, and the headline
    # that adds the node suites to it. Summed rather than taken from the last
    # suite run, which is what a second node suite turned into a silent
    # undercount on 2026-09-03.
    stated_sub = re.search(r"which is (\d+)", text)
    if stated_sub and int(stated_sub.group(1)) != subtotal:
        problems.append(f"the sentence says the suites sum to "
                        f"{stated_sub.group(1)} and they sum to {subtotal}")
    total = subtotal + node_total
    for doc, pat in (("HANDOFF.md", r"\*\*(\d+) checks plus the route suite\*\*"),
                     ("docs/roadmap.md", r"\*\*(\d+) checks\*\*")):
        body = (ROOT / doc).read_text(encoding="utf-8")
        m = re.search(pat, body)
        if not m:
            problems.append(f"{doc} no longer states a test total where this "
                            f"checker looks for one")
        elif int(m.group(1)) != total:
            problems.append(f"{doc} says {m.group(1)} checks and {total} run")
        else:
            notes.append(f"    ok    {doc:<22} total {total}")

    for line in not_run:
        print(f"  NOT RUN  {line} -- its checks were not seen to pass, so this "
              f"run does not pass")
    return not not_run


# ------------------------------------------------------- the running tallies
# Two claims in this repository are counters: how many settings have turned out
# to need measuring per corpus, and how many experiments have improved what the
# first stage retrieves and made the finished pipeline worse. Both go up, and
# both are quoted as a number word in several documents at once. That is the
# shape that drifts the first time one document is edited alone, and it did:
# roadmap.md called the dense ensemble "the sixth setting here that does not
# transfer" nine lines above its own list of five.
#
# The list after the colon is the authority. The word before it, and every
# ordinal claim elsewhere, has to agree with the list.
#
# docs/engineering-log.md is deliberately not read. Its entries are dated and
# say "the fourth setting in a row" because that is what it was on the day;
# forcing those to today's total would be this checker rewriting history rather
# than catching drift. Same reasoning as the superseded numerator in
# check_notes.
TALLY_DOCS = ("README.md", "docs/roadmap.md", "HANDOFF.md", "ui/about.html",
              # A module argues from the same counter, and is where it
              # drifted: sweep_front_matter.py said three long after it
              # was four.
              "src/sweep_front_matter.py")

ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
            "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}

# "Five settings have now been measured as per-corpus rather than global: the
# abstention threshold, the rerank blend, ... and whether fusing a second
# embedder helps at all."
ENUMERATED = re.compile(
    r"\b(\w+)\b[^.:]{0,40}?(?:have|has) now been measured\s+(?:as\s+)?"
    r"per-corpus[^:]*:(?P<items>[^.]+)\.")

# "the fifth setting here that does not transfer"
ORDINAL_CLAIM = re.compile(
    r"\bthe\s+(\w+)\s+setting\b[^.]{0,90}?(?:does|do) not transfer")

# "Four separate experiments have now improved ...", "Four separate
# experiments here improved ...". Up to three words of hedge between the
# noun and the verb, because the sentence is written three ways in three
# places and a pattern that matched only one of them would quietly cover
# one document.
POOL_CLAIM = re.compile(
    r"\b(\w+)\s+separate\s+experiments\b(?:\s+\w+){0,3}?\s+improved\b")

# "Four settings are per-corpus". A different claim from the one above and
# easy to read as contradicting it: this counts what corpora.json configures
# per corpus, while the list counts what has been *measured* per corpus,
# which is one more because the choice of embedder was measured per corpus
# and shipped the same everywhere. This one is checkable against the file.
CONFIGURED_CLAIM = re.compile(r"\b(\w+)\s+settings\s+are\s+per-corpus\b")

# corpora.json keys that say where a corpus lives rather than how it is
# tuned. Everything else is a setting, and the two ensemble keys are one
# setting in two fields: the second embedder and the query prefix it needs.
WHERE_IT_LIVES = frozenset(("data", "golden", "label", "store", "note"))


def configured_settings() -> set:
    raw = json.loads((ROOT / "corpora.json").read_text(encoding="utf-8"))
    found = set()
    for entry in raw.values():
        for key in entry:
            if key in WHERE_IT_LIVES:
                continue
            found.add("ensemble" if key.startswith("ensemble") else key)
    return found


def one_line(text: str) -> str:
    """Documents wrap, so a claim spans lines. Match against one long line."""
    return " ".join(text.split())


# The About page's kept ladder: five rows under "Kept" in ui/about.html, each
# carrying its figure in a <span class="num">. Until 2026-09-05 nothing read
# them. Three were typed on 2026-08-30 from the results file as it then stood,
# the results file moved on 2026-09-01, only the fourth row was updated, and the
# page's own ladder disagreed with itself -- the BM25 row ending at 0.866 and
# the diversity row beginning at 0.925 -- on the live site for four days, while
# check_readme held the README's copy of the same figures the whole time.
#
# Keys are the row's <h4> title. Values are the results rows the "from" and
# "to" figures come from. The fifth kept row, the excerpt window, is not here on
# purpose: its figure is answer-on-the-page from the excerpt sweep, which
# results.json does not carry (results.json's answer_visible is answer-in-the-
# returned-chunk, a different quantity), so there is nothing to hold it to.
ABOUT_LADDER = {
    "Dense retrieval": (None, "dense, no rerank"),
    "Cross-encoder reranking": ("dense, no rerank", "dense + rerank"),
    "BM25 beside dense retrieval": ("dense + rerank", "rrf + rerank"),
    "Diversity cap": ("rrf + rerank", SHIPPED_ROW),
}


def check_about_ladder(measured: dict, problems: list[str], notes: list[str]) -> bool:
    doc = (ROOT / "ui" / "about.html").read_text(encoding="utf-8")
    ml = measured.get("llm")
    if not ml:
        problems.append("ui/about.html: the ML corpus has no results.json to check against")
        return False

    seen = 0
    pattern = re.compile(
        r'<li class="kept">\s*<h4>(.*?)<span class="verdict">.*?'
        r'<span class="num">(.*?)</span>', re.S)
    for m in pattern.finditer(doc):
        title = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        rows = ABOUT_LADDER.get(title)
        if rows is None:
            continue
        seen += 1
        found = re.findall(r"\d\.\d+", re.sub(r"<[^>]+>", "", m.group(2)))
        from_row, to_row = rows
        # Every row carries any-hit; the diversity row also carries source
        # recall, from the same two results rows, after the any-hit pair.
        want = [(r, "hit_rate") for r in (from_row, to_row) if r]
        if title == "Diversity cap":
            want += [(from_row, "src_recall"), (to_row, "src_recall")]
        if len(found) < len(want):
            problems.append(f"ui/about.html ladder, {title!r}: expected "
                            f"{len(want)} numbers, found {len(found)}")
            continue
        for value, (row, metric) in zip(found, want):
            real = ml["ladder"].get(row, {}).get(metric)
            if real is None:
                problems.append(f"ui/about.html ladder: results.json has no "
                                f"{row!r} / {metric}")
                continue
            if not close(float(value), real, written_places(value)):
                problems.append(f"ui/about.html ladder, {title} / {row} "
                                f"{metric}: says {value}, measured {real:.3f}")
        notes.append(f"  about.html ladder {title}")
    if seen != len(ABOUT_LADDER):
        problems.append(f"ui/about.html: found {seen} of {len(ABOUT_LADDER)} "
                        f"kept-ladder rows -- the block this checks has moved or "
                        f"been renamed")
    return True


# The README's threshold table: three fenced lines, each "<label>  N of M (P%)",
# under the heading "wrongly refused at 0.0". Until 2026-09-06 nothing read it,
# and it quoted 10 of 26 and 19 of 35 for sixteen days after the golden sets it
# described had been rewritten -- while this checker held the README's ladder
# and corpora table, two blocks away, to the same results files. Labels map to
# corpora.json labels case-insensitively by their first word.
THRESHOLD_TABLE_HEADING = "wrongly refused at 0.0"


def check_readme_threshold_table(measured: dict, problems: list[str],
                                 notes: list[str]) -> bool:
    doc = (ROOT / "README.md").read_text(encoding="utf-8")
    at = doc.find(THRESHOLD_TABLE_HEADING)
    if at < 0:
        problems.append("README.md: the threshold table under 'wrongly refused at "
                        "0.0' is gone or renamed")
        return False
    block = doc[at:doc.find("```", at)]
    by_first_word = {m["label"].split()[0].lower(): (n, m) for n, m in measured.items()}
    seen = 0
    for line in block.splitlines()[1:]:
        m = re.match(r"\s*([A-Za-z&][A-Za-z& ]*?)\s+(\d+)\s+of\s+(\d+)", line)
        if not m:
            continue
        label, refused, total = m.group(1).strip(), int(m.group(2)), int(m.group(3))
        hit = by_first_word.get(label.split()[0].lower())
        if not hit:
            problems.append(f"README.md threshold table: {label!r} matches no corpus")
            continue
        name, rec = hit
        res = load_results(corpora.registry()[name])
        want_total = rec["answerable"]
        want_refused = None
        for row in (res or {}).get("abstention", {}).get("sweep", []):
            if abs(float(row.get("threshold", row.get("t", 99))) - 0.0) < 1e-9:
                want_refused = row.get("false_abstain", row.get("refused"))
        seen += 1
        if total != want_total:
            problems.append(f"README.md threshold table, {label}: says of {total}, "
                            f"the set holds {want_total} answerable cases")
        if want_refused is not None and refused != want_refused:
            problems.append(f"README.md threshold table, {label}: says {refused} "
                            f"refused at 0.0, measured {want_refused}")
        notes.append(f"  README threshold table {label}")
    if seen != len(measured):
        problems.append(f"README.md threshold table: matched {seen} of "
                        f"{len(measured)} corpora")
    return True


# The guard tally. "Seven guards now cover the things that have gone wrong
# silently before" in HANDOFF §7, "**Seven guards, each exiting non-zero" in the
# README and roadmap, each followed by a fenced block listing one `python src/`
# command per guard. HANDOFF's own paragraph records that the word disagreed
# with the list under it until 2026-08-30 and calls that "the failure
# check_docs.py exists to stop, one level up" -- and until 2026-09-06 nothing
# in this file read that paragraph. Only a number word or digits before
# "guards" is a claim. The first version of this pattern matched any word
# there and reported one it could not read as a problem, so a later sentence
# starting "These guards" would have failed check_docs for saying nothing.
GUARD_TALLY_DOCS = ("README.md", "docs/roadmap.md", "HANDOFF.md")
GUARD_CLAIM = re.compile(
    r"(?im)^\**(\d+|" + "|".join(re.escape(w) for w in WORDS) + r") guards\b")
GUARD_COMMAND = re.compile(r"(?m)^\s*python src/\S+")


def guard_tally(text: str) -> list[tuple[str, int, int | None]]:
    """Every "<Number> guards" paragraph, with the count of `python src/` lines
    in the fenced block directly beneath it.

    Returns (word, said, counted) per paragraph. `counted` is None when no
    fenced block opens where the paragraph ends, so a paragraph that has lost
    its list is reported rather than matched against some later block.
    """
    out = []
    for m in GUARD_CLAIM.finditer(text):
        said = as_count(m.group(1))
        end = text.find("\n\n", m.end())
        if end < 0:
            out.append((m.group(1), said, None))
            continue
        after = text[end:].lstrip("\n")
        if not after.startswith("```"):
            out.append((m.group(1), said, None))
            continue
        body_start = after.find("\n") + 1
        body_end = after.find("\n```", body_start)
        body = after[body_start:body_end if body_end >= 0 else None]
        out.append((m.group(1), said, len(GUARD_COMMAND.findall(body))))
    return out


def check_guard_tally(rel: str, text: str, problems: list[str],
                      notes: list[str]) -> int:
    """One document's guard paragraphs against the lists beneath them.
    Returns how many paragraphs it found."""
    found = guard_tally(text)
    for word, said, counted in found:
        if counted is None:
            problems.append(f"{rel}: says {word} guards and no fenced block of "
                            f"commands follows the paragraph")
        elif said != counted:
            problems.append(f"{rel}: says {word} guards and the block beneath "
                            f"lists {counted} `python src/` commands")
        else:
            notes.append(f"  {rel}: {counted} guards listed, as the sentence says")
    return len(found)


def check_tallies(problems: list[str], notes: list[str]) -> bool:
    """The number word, against the list it introduces and its other copies."""
    listed: dict[str, int] = {}
    pools: dict[str, int] = {}
    ordinals: list[tuple[str, int]] = []

    guard_paragraphs = 0
    for rel in GUARD_TALLY_DOCS:
        path = ROOT / rel
        if not path.exists():
            problems.append(f"{rel}: missing, so its guard paragraph went unchecked")
            continue
        found = check_guard_tally(rel, path.read_text(encoding="utf-8"),
                                  problems, notes)
        if not found:
            # Each of these documents carries the paragraph. One that stopped
            # matching, re-wrapped so the number word was no longer at a
            # line start, say, dropped out silently until 2026-09-07, and
            # the check failed only once all three had.
            problems.append(f"{rel}: no paragraph starting \"<Number> guards\" "
                            f"-- either it was rewritten or the pattern "
                            f"stopped matching it")
        guard_paragraphs += found
    if not guard_paragraphs:
        problems.append("no document has a paragraph starting \"<Number> "
                        "guards\" -- either the sentences were rewritten or "
                        "this check has stopped matching them")
        return False

    for rel in TALLY_DOCS:
        path = ROOT / rel
        if not path.exists():
            problems.append(f"{rel}: missing, so the running tallies went "
                            f"unchecked")
            return False
        text = one_line(path.read_text(encoding="utf-8"))

        for m in ENUMERATED.finditer(text):
            said = as_count(m.group(1))
            # Split on commas alone. Every such list here is written with a
            # serial comma, so the count is the comma count plus one, and a
            # list that stops using one should fail loudly rather than be
            # guessed at by a cleverer parser.
            items = [i for i in m.group("items").split(",") if i.strip()]
            if said is None:
                problems.append(f"{rel}: \"{m.group(1)} ... have now been "
                                f"measured per-corpus\" does not start with a "
                                f"number this checker can read")
                continue
            if said != len(items):
                problems.append(
                    f"{rel}: says {m.group(1)} settings are measured "
                    f"per-corpus and then lists {len(items)}")
            listed[rel] = len(items)
            notes.append(f"  {rel}: {len(items)} settings listed as per-corpus")

        for m in ORDINAL_CLAIM.finditer(text):
            got = ORDINALS.get(m.group(1).lower())
            if got is not None:
                ordinals.append((rel, got))

        for m in CONFIGURED_CLAIM.finditer(text):
            said = as_count(m.group(1))
            tuned = configured_settings()
            if said is not None and said != len(tuned):
                problems.append(
                    f"{rel}: says {m.group(1)} settings are per-corpus, but "
                    f"corpora.json configures {len(tuned)}: "
                    + ", ".join(sorted(tuned)))
            elif said is not None:
                notes.append(f"  {rel}: {said} settings per-corpus, as "
                             f"corpora.json configures")

        for m in POOL_CLAIM.finditer(text):
            got = as_count(m.group(1))
            if got is not None:
                pools[rel] = got
                notes.append(f"  {rel}: {got} experiments raised pool recall "
                             f"and lowered the end-to-end score")

    if not listed:
        problems.append("no document lists the settings that are measured "
                        "per corpus -- either the sentence was rewritten or "
                        "this check has stopped matching it")
        return False

    total = max(listed.values())
    if len(set(listed.values())) > 1:
        problems.append("the documents disagree on how many settings are "
                        "measured per corpus: "
                        + ", ".join(f"{k} says {v}" for k, v in listed.items()))

    for rel, got in ordinals:
        if got != total:
            problems.append(
                f"{rel}: calls something \"the {[k for k, v in ORDINALS.items() if v == got][0]} "
                f"setting that does not transfer\" while the list holds {total}")

    # Only agreement is checkable here. Nothing in the repository counts the
    # experiments that raised pool recall and lowered the end-to-end score, so
    # this cannot say the number is right -- only that the documents have not
    # drifted apart, which is the failure that actually happens.
    if len(set(pools.values())) > 1:
        problems.append("the documents disagree on how many experiments raised "
                        "pool recall and lowered the end-to-end score: "
                        + ", ".join(f"{k} says {v}" for k, v in pools.items()))
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--tests", action="store_true",
                    help="also run every test suite and check the counts the "
                         "documents quote. Two to three minutes, which is why "
                         "it is not part of the default run.")
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
    ok = check_about_ladder(measured, problems, notes) and ok
    ok = check_readme_threshold_table(measured, problems, notes) and ok
    ok = check_defaults(problems, notes) and ok
    ok = check_notes(measured, problems, notes) and ok
    ok = check_routes(problems, notes) and ok
    ok = check_commits(problems, notes) and ok
    ok = check_tallies(problems, notes) and ok
    if args.tests:
        print("running every suite to check the counts; this takes a few "
              "minutes")
        ok = check_tests(problems, notes) and ok

    if not args.quiet and notes:
        print("checked:")
        for n in notes:
            print(n)
    print()
    for p in problems:
        print(f"  DRIFTED  {p}")
    if not ok:
        print("\na table this checker expects is missing, or a suite did not "
              "run, so it checked less\nthan it claims to. Fix the anchor, or "
              "install what the suite needs, rather than deleting the check.")
        return 2
    if problems:
        print(f"\n{len(problems)} number(s) in the documents no longer match the "
              f"measurements.\nThe measurement is the truth; edit the document.")
        return 1
    print("every number quoted in HANDOFF.md §2 and README.md matches the "
          "measurement it came from,\nevery default §3 lists matches the "
          "constant that defines it, the corpus notes agree with the\n"
          "golden sets they argue about, and the running tallies agree with "
          "the lists they count")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
