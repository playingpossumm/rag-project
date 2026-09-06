"""Measure the generated answer, not just the passages that fed it.

Every number in this project measures **retrieval**: did the right passage come
back, and how high. Nothing measured the prose, because until 2026-08-27 no
prose had ever been generated. `generate_local.py` changed that, so this is the
missing half.

**No LLM judge.** A judge model is the usual answer and it is a bad first move
here: it introduces a second system whose failures are invisible, it cannot be
audited, and it would make this project's central claim -- every number is
reproducible from the repository -- false. Three of the four measures below are
objective, and the fourth is a stated proxy rather than a verdict.

What is measured, in descending order of how much it matters:

**1. Invented citations.** The canonical RAG failure and exactly checkable. Every
`[source, page]` the answer emits must correspond to a passage that was actually
supplied. A citation to a document the model was never given is a fabrication
wearing the costume of evidence, and it is worse than no answer at all, because
a citation is what a reader checks *instead of* the source.

**2. Correctness.** The golden sets already carry `answer_contains` -- the string
a correct answer must include, written when the case was labelled and used for
context recall. So correctness needs no judge: does the answer contain it?
Beside it, since 2026-09-04, `correct_loose` accepts an answer that holds at
least 80% of the label's content words in any order, so a right answer in
other words is counted as well as listed. It is the looser of the two and is
reported as such; the strict figure is unchanged.

**3. Refusal.** On adversarial questions the corpus cannot answer, does the model
say so? A system that confabulates fluently here is worse than one that
retrieves badly, because the retrieval failure is visible and this is not.

**4. Groundedness (a proxy, and labelled as one).** What fraction of the answer's
content words appear in the passages it was given. High means it is paraphrasing
what it read; low means it is drawing on what it already knew. It cannot
distinguish a correct claim from parametric memory from a hallucinated one, and
it is reported because it is cheap and directional -- not as a verdict.

Generation on a local CPU model costs seconds per question, so this samples by
default. `--all` runs the whole golden set and takes a while.

    ollama serve && ollama pull llama3.2
    .venv\\Scripts\\python.exe src\\evaluate_answers.py --corpus birds
    .venv\\Scripts\\python.exe src\\evaluate_answers.py --all
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import corpora  # noqa: E402
from evaluate import load_cases, normalize  # noqa: E402
from hybrid import build_bm25  # noqa: E402
from retrieve import (EMBEDDING_MODEL, TOP_K, load_ensemble, load_index,  # noqa: E402
                      retrieve)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
OUT = ROOT / "eval" / "answer-quality.json"

# Words that carry no evidence either way. Kept short deliberately: a long
# stop-list would flatter the groundedness number by removing exactly the words
# a model reaches for when it has nothing to say.
STOP = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "for",
    "with", "as", "by", "at", "from", "is", "are", "was", "were", "be", "been",
    "it", "its", "this", "that", "these", "those", "which", "who", "what",
    "not", "no", "there", "their", "they", "them", "he", "she", "we", "you",
    "i", "can", "will", "would", "should", "may", "might", "do", "does", "did",
    "have", "has", "had", "than", "then", "so", "such", "also", "into", "used",
    "using", "use", "according", "text", "based", "provided", "context",
}

# Everything inside a pair of square brackets. The whole content, not just the
# name: whether a bracket group is a citation is decided partly by the locator
# after the comma, so a pattern that consumed the locator could not be asked
# about it. Per generate.SYSTEM_PROMPT the format is [source, page X].
CITATION = re.compile(r"\[([^\]]+)\]")

# What a bracket group has to look like before it counts as a citation at all.
#
# Without this the measure counted mathematics. On the ML papers it reported 20
# invented citations across 12 answers and every one was notation the model had
# copied out of the passages: [Q], [K], [3], and the learning-rate schedule
# arriving as [min(], [num], [warmup], [steps]. A measure that leads with "a
# fabricated citation is worse than no answer" cannot fire on a model that
# fabricated nothing, least of all on the most technical corpus.
EXTENSIONS = (".pdf", ".docx", ".pptx", ".xlsx", ".md", ".txt", ".htm", ".html")
# How much of a filename a bracket group has to carry before a substring match
# counts as naming that document.
MIN_STEM_MATCH = 4

# A citation whose name is only digits is a reference number lifted from the
# passage, not a document. "[4, page 4]" and "[36, page not specified]" are
# real examples from llama3.1:8b on the ML papers. They name nothing, so they
# cannot point a reader at a document that does not exist; they point nowhere
# at all. Counted as malformed rather than invented, because the case for
# leading with invented citations is that a reader checks the citation instead
# of the source, and this kind cannot be checked in the first place.
BARE_NUMBER = re.compile(r"^[\d\s.,;:\-\[\]()]+$")
LOCATOR = re.compile(r"\b(?:page|slide|sheet|section|para|paragraph)\b", re.I)


def citation_shaped(raw: str, supplied_stems: set[str]) -> bool:
    """Does this bracket group claim to name a document?

    A citation names a file or carries a locator. Notation does neither, and
    the difference has to be decided on shape, because by the time a citation
    is wrong there is nothing else left to decide it on.
    """
    low = raw.lower()
    if any(ext in low for ext in EXTENSIONS):
        return True
    if LOCATOR.search(low):
        return True
    lead = low.split(",")[0].strip()
    if not lead:
        return False
    # Substring either way, because models truncate long filenames, but only
    # once the token is long enough for the match to mean anything. A single
    # character is inside almost every filename: "[t]" matched
    # multi_level_market_making_with_reinforcement_learning and turned one
    # formula into five citations.
    return any(lead == st or (len(lead) >= MIN_STEM_MATCH
                              and (lead in st or st in lead))
               for st in supplied_stems)

# How a refusal is worded. Matched loosely because the instruction is "say so
# plainly" rather than a fixed form, and a model that invents its own wording
# for "I cannot answer this" is still refusing.
#
# This began as a list of fixed phrases and missed a refusal on its first real
# run: "I don't have any information about proof of stake protocols" was scored
# as an answer, because the list held "no information" and nothing that matched
# "don't have any information". A miss here is expensive in one direction only:
# it reports a model that refused correctly as one that confabulated, which is
# the worst thing this harness can say about an answer.
#
# Split in two, because the second group is only a refusal when it is talking
# about the source material. "Air sacs are not found in mammals" is a claim
# about biology and was scored as a refusal by an earlier version of this list.
# Verbs a refusal turns on: saying, containing, answering.
_SAY = (r"contain|provide|include|specify|mention|say|discuss|state|answer|"
        r"address|describe|indicate|list|cover|detail|report|give")
_SAID = (r"contained|provided|included|specified|mentioned|said|discussed|"
         r"stated|answered|addressed|described|indicated|listed|covered|"
         r"detailed|reported|given|available|present|found")
# "not *explicitly* stated": an adverb between the negation and the verb.
_ADV = r"(?:\w+ly\s+)?"
_NT = r"n[o\u2019']?t"

# Refusals that need no support from context.
REFUSAL_PLAIN = tuple(re.compile(p) for p in (
    rf"\b(?:can|could|would)\s?{_NT}\s+(?:\w+\s+){{0,3}}"
    rf"(?:answer|determine|tell|say)\b",
    r"\bcannot\s+be\s+answered\b",
    rf"\bunable\s+to\s+(?:\w+\s+){{0,3}}(?:answer|determine|find|say)\b",
    rf"\bdo(?:es)?\s?{_NT}\s+have\s+(?:any\s+|enough\s+|sufficient\s+)?"
    r"(?:information|details|data|mention)\b",
    r"\bno\s+(?:information|mention|reference|indication)\b",
    r"\bnot\s+enough\s+information\b",
    # "does not provide information on X" is an absence statement wherever
    # it sits, so it needs no source noun beside it. Without this, a refusal
    # that went on to describe what the corpus *does* hold read as an
    # answer, because the noun naming the corpus was 111 characters from the
    # negation and the scope window is 60.
    rf"\b(?:do|does|did)\s?{_NT}\s+{_ADV}"
    r"(?:provide|contain|include|offer|give)\s+(?:any\s+)?"
    r"(?:information|details|data)\b",
))

# The same negations, but they have to be about the documents to count. Without
# the scope, "air sacs are not found in mammals" is a refusal.
REFUSAL_SCOPED = tuple(re.compile(p) for p in (
    rf"\b(?:is|are|was|were|do|does|did)\s?{_NT}\s+{_ADV}(?:{_SAY})\b",
    rf"\bnot\s+{_ADV}(?:{_SAID})\b",
    r"\bno\s+(?:details|record)\b",
))

# What a model calls the material it was handed.
SOURCE_NOUN = re.compile(
    r"excerpt|passage|context|document|text|source|material|corpus|"
    r"provided|supplied|given|above|here|these|record")

# How far either side of the negation a source noun still scopes it. One
# clause, roughly: long enough for "not mentioned anywhere in the provided
# excerpts", short enough that a later unrelated sentence cannot rescue it.
SCOPE = 60


# A sentence, roughly. Newlines matter as much as full stops here, because
# these answers arrive as numbered lists as often as as prose.
SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")

# How many content words a sentence needs before it counts as a claim rather
# than a connective. Low, because the alternative error is worse: treating a
# short real answer as no answer at all.
CLAIM_WORDS = 5

# A sentence whose subject is the source material talks about the documents
# rather than about the question. "The provided excerpts are about the
# classification of bird orders" is what a model says *instead of* answering,
# and counting it as an answer turned two refusals into confabulations.
#
# Deliberately anchored at the start of the sentence: naming a document in the
# middle of a real claim is ordinary, and only the subject position means the
# sentence is about the corpus.
META = re.compile(
    r"^\W*(?:the|these|those|its|their)?\s*"
    r"(?:provided|supplied|given|available|relevant)?\s*"
    r"(?:excerpts?|passages?|contexts?|documents?|texts?|sources?|materials?|"
    r"corpus|information)\b", re.I)


def declines(sentence: str) -> bool:
    """Does this one sentence decline to answer?"""
    low = normalize(sentence)
    if any(p.search(low) for p in REFUSAL_PLAIN):
        return True
    for p in REFUSAL_SCOPED:
        m = p.search(low)
        if m and SOURCE_NOUN.search(low[max(0, m.start() - SCOPE):m.end() + SCOPE]):
            return True
    return False


def is_refusal(answer: str) -> bool:
    """Did the answer decline, taken as a whole?

    A negation somewhere is not a refusal. "It is not stated. However, it does
    mention X" answers the question, and so does an answer that covers two of
    three points and says the third is absent. The test is whether any claim
    survives once the declining sentences are set aside.
    """
    parts = [p for p in SENTENCE.split(answer or "") if p.strip()]
    if not parts:
        return False
    declining = [p for p in parts if declines(p)]
    if not declining:
        return False
    answering = [p for p in parts
                 if p not in declining
                 and not META.match(p.strip())
                 and len(words(p)) >= CLAIM_WORDS]
    return not answering


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9][a-z0-9\-']+", normalize(text))
            if w not in STOP and len(w) > 2}


def stem(word: str) -> str:
    """Enough inflection-stripping to match "keeled" against "keel".

    Not a real stemmer, and it does not need to be: it decides only whether an
    answer the string test rejected is worth a human reading. It refuses to
    strip when the remainder would be a stump, so "wing" stays "wing" rather
    than becoming "w".
    """
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[:-len(suffix)]
    return word


# The looser correctness test. `correct` is containment of the labelled
# string, so "a sternum that is keeled" fails it against "keeled sternum".
# `correct_loose` asks instead whether the label's content words are in the
# answer, in any order, and passes at this share of them. A content word is an
# alphanumeric run of at least LOOSE_MIN_LEN characters that is not in STOP,
# taken after the same normalisation the strict test uses. It is a word test
# and not a judge: "dialect" still misses "dialects", and an answer that names
# the right words in the wrong relation passes.
LOOSE_SHARE = 0.8
LOOSE_MIN_LEN = 3


def loose_tokens(text: str) -> set[str]:
    """The content words of `text` as the loose correctness test sees them."""
    return {w for w in re.findall(r"[a-z0-9]+", normalize(text))
            if len(w) >= LOOSE_MIN_LEN and w not in STOP}


def matches_loosely(wanted: str, answer: str) -> bool:
    """Does the answer hold at least LOOSE_SHARE of the label's content words?

    A label with no content words cannot pass, rather than passing on nothing.
    "28.4" is one: the point splits it into two runs of digits and both are
    under LOOSE_MIN_LEN. Stopwords never carry it either, because they are
    removed before the share is taken.
    """
    want = loose_tokens(wanted)
    if not want:
        return False
    return len(want & loose_tokens(answer)) / len(want) >= LOOSE_SHARE


def cited_sources(answer: str, supplied_stems: set[str] | None = None) -> list[str]:
    """The bracket groups that claim to name a document, as source strings.

    `supplied_stems` lets a bare document name count even without an extension
    or a locator. Omitted, only shape decides.
    """
    stems = supplied_stems or set()
    out = []
    for raw in CITATION.findall(answer):
        if not citation_shaped(raw, stems):
            continue
        name = raw.split(",")[0].strip()
        if name:
            out.append(name)
    return out


def judge(answer: str, case: dict, passages: list[dict]) -> dict:
    """Four measures over one answer. No model is consulted."""
    supplied = {p["source"] for p in passages}
    supplied_stems = {Path(s).stem.lower() for s in supplied}

    cites = cited_sources(answer, supplied_stems)
    invented, malformed = [], []
    for c in cites:
        if BARE_NUMBER.match(c):
            malformed.append(c)
            continue
        cited = Path(c).stem.lower()
        # A citation counts as supplied if it names a document that was given,
        # allowing for the model dropping or mangling the extension. Substring
        # both ways, because models truncate long filenames.
        if not any(cited == s or (len(cited) >= MIN_STEM_MATCH
                                  and (cited in s or s in cited))
                   for s in supplied_stems):
            invented.append(c)

    answer_words = words(answer)
    passage_words = set()
    for p in passages:
        passage_words |= words(p["text"])
    grounded = (len(answer_words & passage_words) / len(answer_words)
                if answer_words else 0.0)

    refused = is_refusal(answer)

    # `correct` is containment of the labelled string and nothing looser. On
    # the first real run it called `bird-keel` wrong for answering "the keel on
    # their breastbone" where the label says "keeled sternum": the right answer
    # in the wrong words. Loosening this test would trade that false negative
    # for false positives and inflate the number, so it is unchanged and the
    # rejects are surfaced in `unmatched`. `near` says which of them share the
    # label's content words and are therefore worth reading first.
    #
    # Surfacing alone left the right-in-other-words answers uncounted, which
    # was the gap. Since 2026-09-04 `correct_loose`, below, counts them as a
    # second measure labelled as the looser of the two, beside the strict
    # figure rather than in place of it.
    wanted = case.get("answer_contains")
    correct, correct_loose, near = None, None, None
    if wanted and not case.get("unanswerable"):
        correct = normalize(wanted) in normalize(answer)
        # A strict hit is a loose hit. Left to the word test alone, "28.4"
        # (no run of 3 characters) and a label word that appears inside a
        # longer word would pass the strict test and fail the looser one, and
        # a measure reported as the looser of the two cannot reject what the
        # strict one accepts.
        correct_loose = correct or matches_loosely(wanted, answer)
        if not correct:
            want_stems = {stem(w) for w in words(wanted)}
            have_stems = {stem(w) for w in words(answer)}
            near = bool(want_stems) and bool(want_stems & have_stems)

    return {"citations": len(cites), "invented": invented,
            "malformed": malformed,
            "grounded": round(grounded, 3), "refused": refused,
            "correct": correct, "correct_loose": correct_loose, "near": near,
            "chars": len(answer)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", action="append")
    ap.add_argument("--limit", type=int, default=8,
                    help="answerable cases per corpus (default 8; local "
                         "generation costs seconds each)")
    ap.add_argument("--adversarial", type=int, default=4,
                    help="unanswerable cases per corpus")
    ap.add_argument("--all", action="store_true", help="every case, no sampling")
    ap.add_argument("--rescore", action="store_true",
                    help="re-judge the answers already in --emit instead of "
                         "generating new ones. Retrieval re-runs (the judge "
                         "needs the passages); the model is never called.")
    ap.add_argument("--emit", type=Path, default=OUT)
    args = ap.parse_args()

    import generate_local

    stored: dict = {}
    if args.rescore:
        if not args.emit.exists():
            print(f"--rescore needs {args.emit}, and it does not exist")
            return 2
        stored = json.loads(args.emit.read_text(encoding="utf-8"))["corpora"]
        print(f"  rescoring {sum(len(c['cases']) for c in stored.values())} "
              f"stored answers. The model is not called.")
    elif not generate_local.available():
        print(f"no Ollama at {generate_local.HOST}. Start it with "
              f"`ollama serve` and `ollama pull {generate_local.MODEL}`.")
        return 2

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)

    reg = corpora.registry()
    report = {}
    for name in (args.corpus or list(reg)):
        cfg = reg.get(name)
        if not cfg or not cfg["indexed"] or not cfg["golden"]:
            continue
        answerable, adversarial = load_cases(cfg["golden"])
        if args.rescore:
            # Exactly the cases the stored file holds, in its order, so a
            # rescore compares like with like rather than resampling.
            if name not in stored:
                continue
            want = [r["id"] for r in stored[name]["cases"]]
            answers = {r["id"]: r["answer"] for r in stored[name]["cases"]}
            by_id = {c["id"]: c for c in answerable + adversarial}
            cases = [by_id[i] for i in want if i in by_id]
            if len(cases) != len(want):
                print(f"    {len(want) - len(cases)} stored case(s) are no "
                      f"longer in the golden set and were dropped")
        else:
            if not args.all:
                answerable = answerable[:args.limit]
                adversarial = adversarial[:args.adversarial]
            cases = answerable + adversarial

        index, metadata = load_index(cfg["store"])
        bm25 = build_bm25(metadata)
        ensemble = load_ensemble(cfg["store"]) if cfg.get("ensemble_model") else None

        # The model that wrote the answers, which on a rescore is the one
        # recorded in the file rather than whatever the environment names now.
        # Printing the current one made a rescore of llama3.1:8b answers
        # announce llama3.2.
        naming = (stored.get(name, {}).get("model") or generate_local.MODEL
                  if args.rescore else generate_local.MODEL)
        shown = ("rescoring stored answers" if args.rescore
                 else f"{len(answerable)} answerable + {len(adversarial)} adversarial")
        print(f"\n  {cfg['label']}  ({shown}, {naming})")
        rows, started = [], time.perf_counter()
        for case in cases:
            results = retrieve(case["question"], index, metadata, model,
                               k=TOP_K, candidate_k=cfg["candidate_k"],
                               use_reranker=True, bm25=bm25,
                               # Passed per call; set on the rerank module
                               # until 2026-09-07.
                               max_per_source=2, rerank_blend=cfg["rerank_blend"],
                               ensemble=ensemble)
            if args.rescore:
                if not results:
                    print(f"    {case['id']}: retrieval returns nothing now, "
                          f"not rescored")
                    continue
                answer = answers[case["id"]]
            else:
                try:
                    answer = generate_local.synthesize(case["question"], results)
                except Exception as exc:                        # noqa: BLE001
                    print(f"    {case['id']}: generation failed: {exc}")
                    continue
            row = {"id": case["id"], "unanswerable": bool(case.get("unanswerable")),
                   **judge(answer, case, results), "answer": answer,
                   # The documents this answer was actually given. Kept so a
                   # change to the judge can be re-scored against answers that
                   # already exist rather than regenerating them.
                   "sources": sorted({p["source"] for p in results})}
            rows.append(row)
            flag = ("INVENTED" if row["invented"] else
                    "refused" if row["refused"] else
                    "correct" if row["correct"] else
                    "" if row["correct"] is None else
                    "loose only" if row["correct_loose"] else "wrong")
            print(f"    {case['id']:<22} grounded {row['grounded']:.2f}  "
                  f"{row['citations']} cites  {flag}")

        # A corpus that produced no answers is not a corpus that scored zero.
        # The 8B run lost Ollama partway and wrote two corpora as "0 of 0
        # correct, groundedness 0.000", which reads in a table exactly like a
        # measurement and means the opposite of one.
        if not rows:
            print(f"    no answers for {name}; not recorded")
            continue
        ans = [r for r in rows if not r["unanswerable"]]
        adv = [r for r in rows if r["unanswerable"]]
        unmatched = [r["id"] for r in ans if r["correct"] is False]
        summary = {
            "n": len(rows),
            "invented_citations": sum(len(r["invented"]) for r in rows),
            "answers_with_invented": sum(1 for r in rows if r["invented"]),
            # A citation that is only a number: the paper's own reference
            # marker, copied out of the passage. It names no document, so it
            # cannot mislead a reader towards one; it is uncheckable instead.
            "malformed_citations": sum(len(r.get("malformed", [])) for r in rows),
            "correct": sum(1 for r in ans if r["correct"]),
            # The looser test beside it: LOOSE_SHARE of the label's content
            # words present, in any order. Never below `correct`.
            "correct_loose": sum(1 for r in ans if r["correct_loose"]),
            "n_answerable": len(ans),
            # Every answerable case the string test rejected. Some are wrong
            # answers and some are right answers in other words, and this
            # harness does not claim to tell them apart. Read them.
            "unmatched": unmatched,
            "unmatched_sharing_words": [r["id"] for r in ans if r.get("near")],
            "refused_wrongly": sum(1 for r in ans if r["refused"]),
            "refused_rightly": sum(1 for r in adv if r["refused"]),
            "n_adversarial": len(adv),
            "grounded_mean": round(sum(r["grounded"] for r in rows)
                                   / max(len(rows), 1), 3),
            "seconds": round(time.perf_counter() - started, 1),
        }
        print(f"    ---")
        print(f"    invented citations   {summary['invented_citations']} "
              f"(in {summary['answers_with_invented']} of {summary['n']} answers)")
        if summary["malformed_citations"]:
            print(f"    malformed citations  {summary['malformed_citations']}"
                  f"   (a bare reference number, naming no document)")
        # Both figures on one line, so a reader sees the strict count and the
        # looser one together. A first version printed the loose figure on a
        # row of its own under the strict one, which separated the two figures
        # this line exists to set beside each other. With two-digit counts
        # the line is 155 characters and wraps on a narrow terminal.
        print(f"    correct              {summary['correct']}/{summary['n_answerable']}"
              f"   (contains the labelled answer string)   "
              f"loose {summary['correct_loose']}/{summary['n_answerable']}"
              f"   (the looser of the two: {LOOSE_SHARE:.0%} of the label's "
              f"content words, any order)")
        if unmatched:
            print(f"    unmatched            {len(unmatched)} to read by hand: "
                  f"{', '.join(unmatched)}")
        print(f"    refused when it should  {summary['refused_rightly']}/"
              f"{summary['n_adversarial']}   and when it should not: "
              f"{summary['refused_wrongly']}/{summary['n_answerable']}")
        print(f"    groundedness (proxy) {summary['grounded_mean']:.3f}")
        # What was asked for, beside what came back. A run that loses its
        # model partway produces a smaller n and nothing else that says so,
        # and the next reader takes the smaller n for the intended sample.
        # On a rescore the original request has to be carried forward, or a
        # truncated run relabels itself complete the moment it is re-judged.
        wanted = (stored[name].get("requested", len(stored[name]["cases"]))
                  if args.rescore else len(cases))
        report[name] = {"label": cfg["label"], "model": naming,
                        "requested": wanted,
                        "produced": len(rows),
                        "complete": len(rows) == wanted,
                        "summary": summary, "cases": rows}
        if len(rows) != wanted:
            print(f"    PARTIAL: {wanted} case(s) asked for, {len(rows)} "
                  f"answered. The rest failed to generate.")
        # After each corpus, so an interrupted run keeps what it has measured.
        write(args.emit, report)

    if not report:
        print("nothing evaluated")
        return 1
    print(f"\n  wrote {args.emit.name}")
    return 0


def write(dest: Path, report: dict) -> None:
    """The results so far, in the shape the finished file has.

    A corpus carries `requested`, `produced` and `complete`, so a run that lost
    its model partway is legible as a truncated run rather than as a smaller
    sample.
    """
    dest.write_text(json.dumps(
        {"generated_by": "src/evaluate_answers.py",
         "note": "No LLM judge. Citations and correctness are objective; "
                 "groundedness is a lexical proxy and not a verdict. "
                 "`correct` is containment of the labelled answer string, so a "
                 "right answer in other words counts against it; the cases it "
                 "rejected are listed in `unmatched` to be read rather than "
                 "scored. `correct_loose` is the looser test beside it, true "
                 f"when {LOOSE_SHARE:.0%} of the label's content words appear "
                 "in the answer in any order. It credits a paraphrase at the "
                 "cost of accepting an answer that names the right words in "
                 "the wrong relation.",
         "corpora": report}, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
