"""Does the golden-set audit catch each way a label can go quietly wrong?

`check_golden.py` found a real broken label on its first run -- a table number
filed under `kind: "section"`, which matched no chunk and contributed nothing
while looking exactly like a label. The value of a check like that is entirely
in whether it fires, so each failure it claims to catch is staged here on a
synthetic corpus and asserted.

Hermetic: a two-document index and a five-case golden set written into a
temporary directory. No models, no real corpus, milliseconds -- and no
dependence on the state of the real eval artefacts, so this passes or fails on
the code rather than on whatever the golden sets happen to say today.

    .venv\\Scripts\\python.exe src\\test_golden.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import check_golden as cg

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHECKS: list[tuple[str, object, object, bool]] = []


def check(name: str, got, want) -> None:
    CHECKS.append((name, got, want, got == want))


# A corpus of two documents: a PDF with pages and a DOCX with named sections and
# a numbered table, because the mixed locator kinds are where labels go wrong.
CHUNKS = [
    {"source": "birds.pdf", "locator": {"kind": "page", "value": 1},
     "text": "Birds have hollow bones which reduce weight for flight."},
    {"source": "birds.pdf", "locator": {"kind": "page", "value": 2},
     "text": "The syrinx is the vocal organ at the base of the trachea."},
    {"source": "anatomy.docx", "locator": {"kind": "section", "value": "Skeleton"},
     "text": "The skeleton is pneumatised, with hollow bones throughout."},
    {"source": "anatomy.docx", "locator": {"kind": "table", "value": 1},
     "text": "Bone | hollow bones | pneumatic"},
]

GOLD = {
    "cases": [
        {"id": "q-hollow", "question": "What reduces a bird's weight?",
         "kind": "fact", "answer_contains": "hollow bones",
         "gold": [{"source": "birds.pdf", "kind": "page", "pages": [1]}]},
        {"id": "q-syrinx", "question": "What organ produces song?",
         "kind": "fact", "answer_contains": "syrinx",
         "gold": [{"source": "birds.pdf", "kind": "page", "pages": [2]}]},
        {"id": "q-section", "question": "How is the skeleton described?",
         "kind": "multi", "answer_contains": "pneumatised",
         "gold": [{"source": "anatomy.docx", "kind": "section",
                   "pages": ["Skeleton"]}]},
        {"id": "adv-one", "question": "What is a transformer?",
         "unanswerable": True, "adversarial_kind": "absent", "gold": []},
        {"id": "adv-two", "question": "Who wrote this document?",
         "unanswerable": True, "adversarial_kind": "metadata", "gold": []},
    ]
}


def scenario(mutate=None) -> list[str]:
    """Write a corpus and a golden set, mutate the golden set, run the audit."""
    tmp = Path(tempfile.mkdtemp(prefix="golden-"))
    try:
        store = tmp / "store"
        store.mkdir()
        (store / "metadata.json").write_text(json.dumps(CHUNKS), encoding="utf-8")
        gold_path = tmp / "golden-t.json"
        g = json.loads(json.dumps(GOLD))
        if mutate:
            mutate(g)
        gold_path.write_text(json.dumps(g), encoding="utf-8")
        cfg = {"name": "t", "label": "Test corpus", "store": store,
               "golden": gold_path, "indexed": True}
        return cg.check_corpus("t", cfg)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def says(problems, fragment) -> bool:
    return any(fragment in p for p in problems)


def faults(found) -> list[str]:
    """Only the structural problems. A NOTE is information, not a failure.

    The ambiguity note is a judgement -- a second document carrying the answer
    string may mean the question no longer identifies one source, or may mean
    the corpus repeats itself -- so it must not decide the exit code.
    """
    return [p for p in found if not p.startswith("NOTE ")]


# A golden set that describes its corpus must be silent. Stated first, because
# every assertion below is worthless if the audit complains about everything.
check("a valid golden set reports no structural problem", faults(scenario()), [])

# Ambiguity, reported and not counted. "hollow bones" is in birds.pdf (which
# q-hollow names) and in anatomy.docx (which it does not), so the note fires.
# audit_golden_set.py asked this question for the ML corpus and asked it
# wrongly, subtracting a key no case has, so it reported 67 of 67 answerable
# cases as ambiguous -- the same information as reporting none of them.
check("a second document carrying the answer string is noted",
      says(scenario(), "also appears in 1 document(s)"), True)
check("and a case whose label names both documents is not",
      says(scenario(lambda g: g["cases"][0]["gold"].append(
          {"source": "anatomy.docx", "kind": "section", "pages": ["Skeleton"]})),
           "also appears"), False)

# The real defect, reproduced: a table number filed under kind "section".
check("a locator that no chunk has",
      says(scenario(lambda g: g["cases"][2]["gold"][0].update(
          pages=["Skeleton", 1])), "which no chunk has"), True)
check("and the mixed-kind shape is named before it is resolved",
      says(scenario(lambda g: g["cases"][2]["gold"][0].update(
          pages=["Skeleton", 1])), "mixes numbered and named locators"), True)

# A renamed or removed document. The label survives the rename and silently
# stops matching anything.
check("a gold document that is not in the index",
      says(scenario(lambda g: g["cases"][0]["gold"][0].update(
          source="renamed.pdf")), "which is not in the index"), True)

# The weaker claim the derived labels rest on: the answer string is at least
# present at the place the label names.
check("an answer string that is not where the label points",
      says(scenario(lambda g: g["cases"][0]["gold"][0].update(pages=[2])),
           "appears at none of the"), True)
check("and a label pointing at a page that does carry it is accepted",
      faults(scenario(lambda g: g["cases"][0]["gold"][0].update(pages=[1]))), [])

# Structure.
check("a duplicated id",
      says(scenario(lambda g: g["cases"].append(dict(g["cases"][0]))),
           "appears 2 times"), True)
check("two cases asking the same question",
      says(scenario(lambda g: g["cases"][1].update(
          question=g["cases"][0]["question"])), "ask the same question"), True)
check("an answerable case with no gold at all",
      says(scenario(lambda g: g["cases"][0].update(gold=[])),
           "can never be scored as a hit"), True)
check("an adversarial case carrying gold",
      says(scenario(lambda g: g["cases"][3].update(
          gold=[{"source": "birds.pdf", "kind": "page", "pages": [1]}])),
           "adversarial but carries"), True)
check("an empty question",
      says(scenario(lambda g: g["cases"][0].update(question="   ")),
           "empty question"), True)
check("a gold entry with no locators",
      says(scenario(lambda g: g["cases"][0]["gold"][0].update(pages=[])),
           "lists no locators"), True)

# The quiet one: a kind the interface cannot group is never offered to anyone.
check("an answerable kind the interface does not group",
      says(scenario(lambda g: g["cases"][0].update(kind="figure")),
           "is never offered as an example"), True)
check("an adversarial kind that is not one of the three",
      says(scenario(lambda g: g["cases"][3].update(adversarial_kind="tricky")),
           "is not one of"), True)

# Matching follows evaluate.py's rule, so a phrase broken across a line break in
# a parsed document still counts. A stricter comparison here would report
# working labels as broken.
check("an answer string split across a line break still matches",
      faults(scenario(lambda g: g.update(cases=[{
          "id": "q-wrap", "question": "What reduces weight?", "kind": "fact",
          "answer_contains": "hollow bones",
          "gold": [{"source": "birds.pdf", "kind": "page", "pages": [1]}]}]))),
      [])


def main() -> int:
    width = max(len(n) for n, *_ in CHECKS)
    failed = 0
    for name, got, want, ok in CHECKS:
        if not ok:
            failed += 1
            print(f"FAIL  {name:<{width}}  got {got!r}, want {want!r}")
    print(f"{len(CHECKS) - failed}/{len(CHECKS)} golden-set audit checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
