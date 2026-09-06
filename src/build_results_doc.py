"""Write the figures in eval/RESULTS.md from eval/results.json.

RESULTS.md has always said its numbers were copied from results.json. Nothing
copied them. They were typed, and they went stale exactly the way typed numbers
do -- the document claimed 20 papers and 2,768 chunks long after the corpus had
reached 36 and 5,459, and the header saying "if this file disagrees with that
one, this file is stale" was true without anyone noticing.

So the tables are generated and the prose is not. Everything between a
`<!-- generated:NAME -->` and `<!-- /generated:NAME -->` marker is replaced from
results.json; everything outside them is left alone, because the analysis around
the numbers is judgement and cannot be derived from a JSON file.

The corpus identity is checked before anything is written. results.json used to
be shared by every corpus, so whichever evaluation ran last owned it -- running
the bird set left it describing 45 Wikipedia documents while this document
described 36 arXiv papers. Each corpus writes its own file now, and this refuses
to fill an ML document from a bird run.

`--check` also reads the prose. The generated blocks cannot drift, but the
sentences around them quote the same case counts by hand, and on 2026-09-06
the prose said 66 answerable, 18 adversarial and 7 universal failures against
generated blocks that said 67 and 17 and a hard-cases fixture that held 3,
while `--check` diffed only the blocks and reported the document current. So
every "N answerable", "N adversarial", "N structural" and "N ... fail under
all" outside the markers is held to the results file and to
eval/hard_cases.json, with 3 exemptions. A count that equals another
configured corpus's total is accepted when the paragraph names that corpus,
because the document compares corpora and "25 answerable cases" in a
paragraph about the bird set is not a claim about this one. A count in a
sentence that carries an absolute date and says what the set "held", how it
"stood", what it "carried", or what was so "until", "before", "earlier" or
"previously" is reported as a note and not checked, because those sentences
record the earlier count beside the corrected one on purpose. A count
introduced by a verb such as "misses" or "refuses" is a subset of the set and
is reported as a note. The first version of the first 2 exemptions was
wider, and a reviewer showed on 2026-09-07 that "The set then holds 66
answerable cases" passed as history on the word "then" alone and that "8
adversarial" passed in the ML document because the quant set counts 8, so the
date and the corpus name are now required.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).parent.parent

# The prose counts this holds to the results file, each as the pattern, the
# key in the results' corpus block and the word used in the report. The
# lookbehind keeps "3.9 answerable ones" from reading as 9 answerable, which
# the first version of this pattern did.
PROSE_COUNTS = (
    (re.compile(r"(?<![\d.])(\d+) answerable\b"), "answerable_cases", "answerable"),
    (re.compile(r"(?<![\d.])(\d+) adversarial\b"), "adversarial_cases", "adversarial"),
)
# The universal-failure count, written 3 ways in the same section: "**7
# structural.**", "**7 (10.6%) fail under all six**" and "the cases that fail
# under all six are **3 of 67**". Number words are read for the first form
# because the prose also says "seven".
UNIVERSAL = re.compile(
    r"\b(?P<a>\w+)\*{0,2}\s+structural\b"
    r"|\b(?P<b>\d+)\b[^\n]{0,25}?\bfail under all\b"
    r"|\bfail under all \w+ are \*{0,2}(?P<c>\d+)\b")
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}

# A sentence that says what the set "held", how it "stood", or what was so
# "until" or "before" a date is a record of an earlier count, kept on purpose
# because the writing style asks for the mistake to be recorded beside the
# correction, so "measured on 2026-08-27, when the set held 66 answerable and
# 18 adversarial cases" is not drift. Such a sentence is reported as a note
# and its counts are not held to the current file. The sentence must also
# carry an absolute date, which the writing style requires of any record of
# an earlier state. Until 2026-09-07 the word alone was enough and the list
# held "then", so "The set then holds 66 answerable cases" passed as history.
# The weakness that remains is a dated history sentence that also states a
# current count, which is not checked either.
HISTORY = re.compile(r"\b(?:held|stood|carried|earlier|previous(?:ly)?|until|before)\b")
DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
SENTENCE_END = re.compile(r"[.!?:](?:\s|$)")

# A count introduced by one of these verbs, as in "misses 6 answerable cases"
# or "wrongly refuses 1", is a subset and not the population, and is skipped
# with a note. "1 of 67 answerable" is still checked, because the number this
# reads there is the 67 after "of". A verb missing from this set makes a
# subset count read as a population claim and report drift against the
# total, which is the loud failure rather than the quiet one.
SUBSET_VERBS = frozenset((
    "miss", "misses", "missed", "refuse", "refuses", "refused", "refusing",
    "lose", "loses", "lost", "catch", "catches", "caught", "answer", "answers",
    "answered", "rescue", "rescues", "rescued", "recover", "recovers",
    "recovered", "fail", "fails", "failed", "cost", "costs", "trade", "trades",
    "gain", "gains", "reach", "reaches", "reached"))
PREVIOUS_WORD = re.compile(r"(\w+)[ \t*]+$")


def sentence_around(text: str, pos: int) -> str:
    starts = [m.end() for m in re.finditer(r"[.!?:]\s+|\n\n", text[:pos])]
    start = starts[-1] if starts else 0
    m = SENTENCE_END.search(text, pos)
    return text[start:m.end() if m else len(text)]


def as_count(token: str) -> int | None:
    token = token.lower()
    return int(token) if token.isdigit() else WORDS.get(token)


def hard_cases_for(results: Path) -> Path:
    """results-birds.json -> hard_cases-birds.json, in the same directory."""
    return results.with_name("hard_cases" + results.stem[len("results"):] + ".json")


def other_corpus_counts(this: str) -> dict[str, dict[int, list[str]]]:
    """Case totals of every other configured corpus with a results file, as
    {key: {total: [words that name the corpus]}}. The words are the corpus
    name with a trailing "s" dropped, so "bird" covers "birds", "bird set"
    and "bird corpus", and every word of its label, so "ornithology" and
    "finance" count too."""
    out: dict[str, dict[int, list[str]]] = {"answerable_cases": {},
                                            "adversarial_cases": {}}
    try:
        import corpora
        from check_freshness import artefact
    except ImportError:
        return out
    for name, cfg in corpora.registry().items():
        if name == this or not cfg["golden"]:
            continue
        p = artefact("results", cfg["golden"])
        if not p.exists():
            continue
        c = json.loads(p.read_text(encoding="utf-8")).get("corpus", {})
        words = [name.lower().rstrip("s")] + [
            w.lower() for w in re.findall(r"[A-Za-z]+", cfg.get("label", ""))]
        for key in out:
            if key in c:
                out[key].setdefault(int(c[key]), []).extend(words)
    return out


def paragraph_around(text: str, pos: int) -> str:
    start = text.rfind("\n\n", 0, pos)
    end = text.find("\n\n", pos)
    return text[start + 2 if start >= 0 else 0:end if end >= 0 else len(text)]


def names_other_corpus(text: str, pos: int, words: list[str]) -> bool:
    """Whether the paragraph around `pos` uses one of the words that name
    the corpus whose total the count equals. Until 2026-09-07 the total
    alone was enough, so "8 adversarial" anywhere in the ML document passed
    because the quant set counts 8."""
    para = paragraph_around(text, pos).lower()
    return any(re.search(r"\b" + re.escape(w), para) for w in words)


def without_blocks(doc: str) -> str:
    """The prose alone, with each generated block replaced by as many newlines
    as it held so line numbers in the report still point into the file."""
    return re.sub(r"<!-- generated:(\w+) -->.*?<!-- /generated:\1 -->",
                  lambda m: "\n" * m.group(0).count("\n"), doc, flags=re.S)


def check_prose(doc: str, r: dict, hard_cases: Path) -> tuple[list[str], list[str]]:
    """Every hand-written case count outside the markers, against the files.
    Returns (problems, notes); the notes list every count read as history or
    as a subset."""
    prose = without_blocks(doc)
    c = r["corpus"]
    others = other_corpus_counts(c.get("name", ""))
    problems: list[str] = []
    notes: list[str] = []

    def line_of(pos: int) -> int:
        return prose.count("\n", 0, pos) + 1

    def history(pos: int, n: int, label: str) -> bool:
        sentence = sentence_around(prose, pos)
        if HISTORY.search(sentence) and DATE.search(sentence):
            notes.append(f"line {line_of(pos)}: {n} {label}, read as a dated "
                         f"record of an earlier set and not checked")
            return True
        return False

    def other_corpus(pos: int, n: int, key: str) -> bool:
        return names_other_corpus(prose, pos, others[key].get(n, []))

    def subset(pos: int, n: int, label: str) -> bool:
        before = PREVIOUS_WORD.search(prose, 0, pos)
        if before and before.group(1).lower() in SUBSET_VERBS:
            notes.append(f"line {line_of(pos)}: \"{before.group(1)} {n} "
                         f"{label}\" counts a subset, not the set, and is not "
                         f"checked")
            return True
        return False

    for pat, key, label in PROSE_COUNTS:
        for m in pat.finditer(prose):
            n = int(m.group(1))
            if (n == c[key] or other_corpus(m.start(), n, key)
                    or history(m.start(), n, label)
                    or subset(m.start(), n, label)):
                continue
            problems.append(
                f"line {line_of(m.start())}: says {n} {label}; the results file "
                f"counts {c[key]} and no other configured corpus named in the "
                f"paragraph counts {n}")

    claims = []
    for m in UNIVERSAL.finditer(prose):
        n = as_count(m.group("a") or m.group("b") or m.group("c"))
        if n is not None:
            claims.append((n, m.start()))
    if claims and not hard_cases.exists():
        problems.append(f"the prose counts universal failures and "
                        f"{hard_cases.name} is not there to check it against")
    elif claims:
        want = len(json.loads(hard_cases.read_text(encoding="utf-8"))
                   .get("structural", []))
        for n, pos in claims:
            if n != want and not history(pos, n, "universal failures"):
                problems.append(f"line {line_of(pos)}: says {n} cases fail under "
                                f"every configuration; {hard_cases.name} lists "
                                f"{want}")
    return problems, notes


def table(rows: dict, cols: list[tuple[str, str]], first: str) -> str:
    """A markdown table from an ordered dict of {label: {metric: value}}."""
    head = f"| {first} | " + " | ".join(h for h, _ in cols) + " |"
    rule = "|" + "---|" * (len(cols) + 1)
    out = [head, rule]
    for label, vals in rows.items():
        cells = []
        for _, key in cols:
            v = vals.get(key)
            cells.append("" if v is None
                         else f"{v:,}" if isinstance(v, int)
                         else f"{v:.3f}")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def blocks(r: dict) -> dict[str, str]:
    c = r["corpus"]
    ab = r.get("abstention", {})

    corpus = (
        f"**Corpus:** {c['documents']} documents -> {c['chunks']:,} chunks "
        f"(210 tokens, 40 overlap).\n"
        f"**Golden set:** {c['answerable_cases']} answerable + "
        f"{c['adversarial_cases']} adversarial, labels *derived* from answer "
        f"strings against the current corpus rather than hand-written.\n"
        f"**Retrieved:** k={c['k']} from {c['candidate_k']} candidates.\n"
        f"**Reproduce:** `python src/evaluate.py`"
    )

    metrics = [("any-hit@5", "hit_rate"), ("MRR", "mrr"),
               ("NDCG", "ndcg"), ("src recall", "src_recall")]
    # "answer shown" is reported for the end-to-end pipeline only, and beside
    # any-hit rather than instead of it. any-hit asks whether retrieval reached
    # a LOCATION that answers; gold is derived per locator, so a page holding
    # the answer in one chunk marks every chunk of that page relevant. This
    # asks whether the answer is in the text the reader is handed. Measured
    # 2026-09-03, the two differ on 14 of 125 answerable questions. The
    # candidate pool keeps the old columns, because a pool is not shown to
    # anyone.
    end_to_end = table(r["end_to_end"],
                       metrics + [("answer shown", "answer_visible")],
                       "pipeline")
    pool = table(r["candidate_pool"], metrics[:2] + metrics[3:], "first stage")
    exp = table(r["expansion"],
                [("context recall", "context_recall"),
                 ("tokens/query", "tokens_per_query"),
                 ("blocks/query", "blocks_per_query")], "expansion")

    sweep = ""
    if ab.get("sweep"):
        lines = ["| threshold | adversarial caught | answerable wrongly refused |",
                 "|---|---|---|"]
        for row in ab["sweep"]:
            lines.append(f"| {row['threshold']:+d} | "
                         f"{row['caught']} / {ab.get('n_adversarial', '?')} | "
                         f"{row['false_abstain']} / {ab.get('n_answerable', '?')} |")
        sweep = "\n".join(lines)
        shipped = ab.get("shipped_threshold")
        if shipped is not None:
            sweep += (f"\n\nShipped threshold: **{shipped:+.1f}**. "
                      f"Answerable questions score a median of "
                      f"{ab.get('answerable_median', 0):+.2f}.")

    # The cap's cost, read straight off the same rows. This was a hand-typed
    # table and drifted: it still said 0.818 for 1/src after the golden set
    # changed and the real figure became 0.776.
    e = r["end_to_end"]
    base = e.get("rrf + rerank", {})
    cap = ""
    if base:
        rows = ["| cap | any-hit | src recall | trade vs no cap |", "|---|---|---|---|",
                f"| none | {base['hit_rate']:.3f} | {base['src_recall']:.3f} | — |"]
        for label, key in (("2/src", "+ diversity 2/src"), ("1/src", "+ diversity 1/src")):
            v = e.get(key)
            if not v:
                continue
            dh = (v["hit_rate"] - base["hit_rate"]) * 100
            ds = (v["src_recall"] - base["src_recall"]) * 100
            rows.append(f"| {label} | {v['hit_rate']:.3f} | {v['src_recall']:.3f} | "
                        f"{ds:+.1f} src recall for {dh:+.1f} any-hit |")
        cap = "\n".join(rows)

    return {"corpus": corpus, "end-to-end": end_to_end, "diversity-cap": cap,
            "candidate-pool": pool, "expansion": exp, "abstention": sweep}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=ROOT / "eval" / "results.json")
    ap.add_argument("--doc", type=Path, default=ROOT / "eval" / "RESULTS.md")
    ap.add_argument("--expect-documents", type=int, default=None,
                    help="refuse to write unless results.json has this many "
                         "documents; guards against filling a document from "
                         "another corpus's run")
    ap.add_argument("--check", action="store_true",
                    help="report whether the document is current, write nothing")
    args = ap.parse_args()

    r = json.loads(args.results.read_text(encoding="utf-8"))
    if not r.get("corpus"):
        print(f"{args.results} has no corpus block -- re-run src/evaluate.py")
        return 1
    if args.expect_documents and r["corpus"]["documents"] != args.expect_documents:
        print(f"REFUSING: {args.results} describes "
              f"{r['corpus']['documents']} documents, expected "
              f"{args.expect_documents}. Wrong corpus's results.")
        return 1

    doc = args.doc.read_text(encoding="utf-8")
    made = blocks(r)
    written, missing = 0, []
    for name, body in made.items():
        # Match the markers themselves and rebuild the newlines, rather than
        # requiring one on each side of the body. An empty block has a single
        # newline between the two markers, not two, so a pattern demanding both
        # matched nothing on the first run -- every marker reported missing
        # while all of them sat in the file.
        pat = re.compile(
            rf"<!-- generated:{name} -->.*?<!-- /generated:{name} -->", re.S)
        if not pat.search(doc):
            missing.append(name)
            continue
        new = pat.sub(
            lambda _: f"<!-- generated:{name} -->\n{body}\n<!-- /generated:{name} -->",
            doc)
        if new != doc:
            written += 1
        doc = new

    if missing:
        print("no marker for: " + ", ".join(missing))
    if args.check:
        current = doc == args.doc.read_text(encoding="utf-8")
        print(f"{args.doc.name} generated blocks are current" if current
              else f"{args.doc.name} is STALE -- run src/build_results_doc.py")
        prose, notes = check_prose(args.doc.read_text(encoding="utf-8"), r,
                                   hard_cases_for(args.results))
        for n in notes:
            print(f"  note     {n}")
        for p in prose:
            print(f"  DRIFTED  {p}")
        if prose:
            print(f"{len(prose)} count(s) in the prose disagree with the files "
                  f"the generated blocks are built from. Edit the prose.")
        else:
            print("the prose's case counts agree with the results file and "
                  "the hard-cases fixture")
        return 0 if current and not prose else 1

    args.doc.write_text(doc, encoding="utf-8")
    print(f"{args.doc}: {written} block(s) rewritten from "
          f"{r['corpus']['documents']} documents / {r['corpus']['chunks']:,} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
