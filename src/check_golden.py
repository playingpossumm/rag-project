"""Is every golden set still a valid description of the corpus it scores?

`evaluate.py` will happily score a case whose gold document was renamed, whose
page numbers moved when the parser changed, or whose `answer_contains` string is
nowhere near the passage the label points at. It produces a number either way.
That number is then the thing every document in this repo quotes.

`src/audit_golden_set.py` already asks the *semantic* question -- has the corpus
grown into an adversarial case's subject, has an answerable question become
ambiguous -- but it is ML-only, with a hand-written table of which words would
make each adversarial case answerable. This asks the *structural* question, and
needs no per-case knowledge, so it runs on every corpus including ones that do
not exist yet:

  1. ids unique, questions non-empty and not duplicated
  2. every answerable case has at least one gold entry; no adversarial case has
     one
  3. every gold `source` is a document that is actually in the index
  4. every gold locator resolves to a chunk that exists under that source
  5. `answer_contains`, where present, actually appears in the text at one of
     the locators the label points at
  6. `kind` is one the interface knows how to group

Number 5 is the one worth the trouble. HANDOFF §7 records that deriving labels
from answer strings "proves the string is present, not that the passage answers
the question" -- true, and a weaker claim than the labels make. This checks the
weaker claim at least holds: sixteen quant cases once carried a label whose
string was present somewhere in the corpus while the passage the label named did
not contain it at all.

Number 6 is quiet and expensive. `build_analytics.py` buckets example questions
by `kind` and offers three of them; a case with an unrecognised kind is never
offered to anyone and nothing says so.

    .venv\\Scripts\\python.exe src\\check_golden.py            # every corpus
    .venv\\Scripts\\python.exe src\\check_golden.py --corpus birds

Exit code is the contract: 0 every configured golden set was checked and is
valid, 1 problems found, 2 a configured corpus was not checked because it has
no index built or no golden set, so the run says nothing about it. The same
contract as `audit_page_credit.py`, which also exits 2 when a recording exists
for some corpora and not others. Until 2026-09-06 an unindexed corpus was
skipped with a one-line note and the run still printed "all 0 golden sets are
structurally valid" and exited 0 on a fresh clone, having checked nothing.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import corpora  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent

# The kinds the interface knows how to group. `build_analytics.KIND_LABEL` maps
# the first three; the adversarial ones are recognised by `unanswerable` rather
# than by name, but they are listed so a typo in one is still caught.
ANSWERABLE_KINDS = {"fact", "multi", "cross-doc"}
ADVERSARIAL_KINDS = {"absent", "near-miss", "metadata"}


# One rule, defined in evaluate.py. This file carried its own copy until
# 2026-09-07, the same rule with a str() around the argument that no caller
# needed, and a rule that lives in two files is one that drifts.
from evaluate import normalize  # noqa: E402


def locator_key(loc: dict) -> tuple:
    return (str(loc.get("kind", "page")), str(loc.get("value", "")))


def check_corpus(name: str, cfg: dict) -> list[str]:
    """Every structural complaint about one golden set, as plain sentences."""
    problems: list[str] = []
    gold_path = cfg["golden"]
    data = json.loads(gold_path.read_text(encoding="utf-8"))
    cases = data.get("cases", [])

    meta = json.loads((cfg["store"] / "metadata.json").read_text(encoding="utf-8"))
    sources = {c["source"] for c in meta}
    # Which locators exist under each source, and the text at each, so a label
    # can be checked against the passage it actually names rather than against
    # the corpus as a whole.
    by_place: dict[tuple, list[str]] = {}
    for chunk in meta:
        by_place.setdefault((chunk["source"], *locator_key(chunk["locator"])),
                            []).append(chunk["text"])

    # Whole-document text, for the ambiguity note below.
    by_source: dict[str, list[str]] = {}
    for chunk in meta:
        by_source.setdefault(chunk["source"], []).append(chunk["text"])

    ids = Counter(c.get("id") for c in cases)
    questions = Counter(normalize(c.get("question", "")) for c in cases)
    for cid, n in ids.items():
        if n > 1:
            problems.append(f"id {cid!r} appears {n} times")
    for q, n in questions.items():
        if n > 1 and q:
            problems.append(f"{n} cases ask the same question: {q[:60]!r}")

    for case in cases:
        cid = case.get("id", "(no id)")
        if not str(case.get("question", "")).strip():
            problems.append(f"{cid}: empty question")

        unanswerable = bool(case.get("unanswerable"))
        gold = case.get("gold") or []

        if unanswerable:
            if gold:
                problems.append(f"{cid}: adversarial but carries {len(gold)} "
                                f"gold entr{'y' if len(gold) == 1 else 'ies'}")
            kind = case.get("adversarial_kind")
            if kind and kind not in ADVERSARIAL_KINDS:
                problems.append(f"{cid}: adversarial_kind {kind!r} is not one of "
                                f"{sorted(ADVERSARIAL_KINDS)}")
            continue

        if not gold:
            problems.append(f"{cid}: answerable but has no gold entries, so it "
                            f"can never be scored as a hit")
        kind = case.get("kind")
        if kind not in ANSWERABLE_KINDS:
            problems.append(f"{cid}: kind {kind!r} is not one the interface "
                            f"groups ({sorted(ANSWERABLE_KINDS)}), so this case "
                            f"is never offered as an example")

        wanted = case.get("answer_contains")
        found_here = False
        for entry in gold:
            src = entry.get("source")
            if src not in sources:
                problems.append(f"{cid}: gold names {src!r}, which is not in "
                                f"the index")
                continue
            pages = entry.get("pages") or []
            if not pages:
                problems.append(f"{cid}: gold entry for {src} lists no locators")
            # A single entry listing both a number and a name under one kind is
            # the shape of a locator that belongs to a different kind -- a table
            # number filed under "section". bird-hollow-bones carried exactly
            # that, and (docx, section, "1") matched nothing while looking like
            # a label. Caught here as well as by the resolve check below,
            # because a stray number CAN resolve by coincidence.
            if len({type(p).__name__ for p in pages}) > 1:
                problems.append(
                    f"{cid}: gold entry for {src} mixes numbered and named "
                    f"locators under kind {entry.get('kind', 'page')!r}: "
                    f"{pages} -- one of them probably belongs to another kind")
            for page in pages:
                key = (src, str(entry.get("kind", "page")), str(page))
                texts = by_place.get(key)
                if texts is None:
                    problems.append(
                        f"{cid}: gold points at {src} "
                        f"{entry.get('kind', 'page')} {page!r}, which no chunk "
                        f"has -- the parser or the chunking moved under it")
                    continue
                if wanted and any(normalize(wanted) in normalize(t) for t in texts):
                    found_here = True

        # The weaker claim the labels rest on: the answer string is at least
        # present where the label says it is. Only checked when every locator
        # resolved, or the message would blame the label for a missing chunk.
        if wanted and not found_here and not any(
                p.startswith(f"{cid}: gold") for p in problems):
            problems.append(
                f"{cid}: answer_contains {wanted[:40]!r} appears at none of the "
                f"{len(gold)} place(s) the gold points at")

    # Ambiguity, reported as a note rather than a problem: a second document
    # carrying the answer string is a judgement about whether the question
    # still identifies one source, not a structural error. audit_golden_set.py
    # asked this for the ML corpus only, and asked it wrongly -- it subtracted
    # case["source"], a key no case has, so every case matched its own gold
    # document and it reported 67 of 67. Corrected and generalised here.
    for case in cases:
        wanted = case.get("answer_contains")
        if case.get("unanswerable") or not wanted:
            continue
        needle = normalize(wanted)
        gold_sources = {e["source"] for e in case.get("gold") or []}
        elsewhere = sorted(
            src for src, texts in by_source.items()
            if src not in gold_sources and any(needle in normalize(t) for t in texts))
        if elsewhere:
            problems.append(
                f"NOTE {case['id']}: {wanted[:30]!r} also appears in "
                f"{len(elsewhere)} document(s) the label does not name "
                f"({', '.join(elsewhere[:3])}) -- the question may no longer "
                f"identify one source")

    return problems


def shown(path: Path | None) -> str:
    """A path as the documents write it, relative to the repo where it can be."""
    if path is None:
        return "(none configured)"
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", action="append")
    args = ap.parse_args()

    reg = corpora.registry()
    wanted = args.corpus or list(reg)
    total = 0
    checked = 0
    not_run: list[str] = []

    for name in wanted:
        cfg = reg.get(name)
        if not cfg:
            print(f"  unknown corpus {name!r}; have {sorted(reg)}")
            return 2
        # A corpus that cannot be checked is reported in the same voice as a
        # problem, because a reader scanning for trouble reads "skipped" as
        # "fine". Nothing about its golden set has been established.
        if not cfg["indexed"]:
            print(f"\n{cfg['label']}\n  NOT RUN  no index is built under "
                  f"{shown(cfg['store'])}, so the golden set was not checked. "
                  f"Build one with\n           python src/ingest.py "
                  f"(RAG_DATA_DIR={shown(cfg['data'])} "
                  f"RAG_STORE_DIR={shown(cfg['store'])})")
            not_run.append(name)
            continue
        if not cfg["golden"] or not cfg["golden"].exists():
            print(f"\n{cfg['label']}\n  NOT RUN  no golden set at "
                  f"{shown(cfg['golden'])}, so there is nothing to check "
                  f"against the index")
            not_run.append(name)
            continue

        found = check_corpus(name, cfg)
        # A note is information, not a failure. Mixing them would make the exit
        # code mean "something to read" instead of "something is wrong".
        problems = [p for p in found if not p.startswith("NOTE ")]
        notes = [p[5:] for p in found if p.startswith("NOTE ")]
        checked += 1
        total += len(problems)
        n = len(json.loads(cfg["golden"].read_text(encoding="utf-8"))["cases"])
        print(f"\n{cfg['label']}  ({n} cases, "
              f"{cfg['golden'].relative_to(ROOT).as_posix()})")
        if not problems:
            print("  every case names a document in the index, every locator "
                  "resolves,\n  and every answer string is where its label says "
                  "it is")
        for p in problems:
            print(f"  PROBLEM  {p}")
        for n in notes:
            print(f"  note     {n}")

    print()
    skipped = (f" {len(not_run)} configured corpus(es) NOT RUN: "
               f"{', '.join(not_run)}." if not_run else "")
    if total:
        print(f"{total} problem(s) across {checked} corpora.{skipped} A golden "
              f"set that does not describe\nits corpus produces numbers that "
              f"look fine and measure something else.")
        return 1
    if not checked:
        print(f"no golden set could be checked because no index is built.{skipped}\n"
              f"Nothing here has been established; build an index and run this "
              f"again.")
        return 2
    if not_run:
        print(f"{checked} golden set(s) structurally valid against their "
              f"indexes.{skipped}\nThis run says nothing about those, so it "
              f"does not pass.")
        return 2
    print(f"all {checked} golden sets are structurally valid against their indexes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
