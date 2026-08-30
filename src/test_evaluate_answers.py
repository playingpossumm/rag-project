"""The answer-quality judge, against answers it has already got wrong.

`evaluate_answers.py` is the only measurement in this project whose input is
prose, and prose is where a string test goes wrong quietly. Its first real run
produced two such errors and nothing caught them, because nothing imported the
module: the refusal set missed a plain refusal, and a right answer in other
words was scored as wrong without saying so.

So the cases below are mostly real. Every one marked `# real` is text a local
llama3.2 actually produced on this repository's corpora on 2026-08-29, copied
out of eval/answer-quality.json, rather than an example written to pass.

Hermetic: no Ollama, no index, no model. `judge()` is a pure function of an
answer, a case and the passages that were supplied, which is what makes it
testable at all.

    .venv\\Scripts\\python.exe src\\test_evaluate_answers.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluate_answers import cited_sources, judge, words  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PASSAGES = [
    {"source": "feather.pdf", "text":
     "Down feathers are fluffy because they lack barbicels, so the barbules "
     "float free of each other, allowing the down to trap air and provide "
     "excellent thermal insulation."},
    {"source": "bird_anatomy.docx", "text":
     "Air sacs make airflow through the lungs one-directional, so fresh air "
     "passes over the gas-exchange surface on both inhalation and exhalation."},
]

fails: list[str] = []
checks = 0


def check(name, got, want):
    global checks
    checks += 1
    if got != want:
        fails.append(f"{name}: got {got!r}, want {want!r}")


def case(cid, wanted=None, unanswerable=False):
    return {"id": cid, "answer_contains": wanted, "unanswerable": unanswerable}


# ---------------------------------------------------------------- refusals
# Every string here is a refusal. The harness reporting one of these as an
# answer means reporting a model that behaved correctly as one that
# confabulated, which is the most damaging thing it can get wrong.
REFUSALS = [
    # real: adv-bird-consensus, and the exact miss that prompted this file.
    "I don't have any information about proof of stake protocols or consensus "
    "mechanisms from the provided excerpts.",
    # real: adv-bird-transformer
    "There is no mention of the transformer architecture or its attention "
    "mechanism in the provided excerpts.",
    # real: adv-bird-insulin
    "This question cannot be answered based on the provided context excerpts.",
    "The excerpts do not contain information about this.",
    "The passages don't mention anything about tax law.",
    "I do not have enough information to answer this question.",
    "I am unable to answer this from the documents provided.",
    "That detail is not specified in these documents.",
    "The context does not provide the training seed.",
    "No information about the licence appears in these excerpts.",
]

for text in REFUSALS:
    r = judge(text, case("adv", unanswerable=True), PASSAGES)
    check(f"refusal detected: {text[:44]!r}", r["refused"], True)

# And the other direction. A confident wrong answer must not be scored as a
# refusal just because it contains a negation; that would hide the failure the
# adversarial half exists to catch.
NOT_REFUSALS = [
    # real: bird-syrinx, an ordinary confident answer.
    "The organ that birds use to produce song is called the syrinx, which is a "
    "bony structure located at the base of the trachea.",
    # A wrong answer that negates, but answers.
    "Down feathers do not have barbicels, which is why they trap air.",
    "Air sacs are not found in mammals, only in birds.",
]

for text in NOT_REFUSALS:
    r = judge(text, case("q", unanswerable=True), PASSAGES)
    check(f"not a refusal: {text[:44]!r}", r["refused"], False)

# ------------------------------------------------------------ invented cites
supplied = judge("The syrinx sits at the base of the trachea. [feather.pdf, page 2]",
                 case("q"), PASSAGES)
check("a supplied citation is not invented", supplied["invented"], [])
check("the citation is counted", supplied["citations"], 1)

fabricated = judge("Birds sing with a syrinx. [ornithology_handbook.pdf, page 44]",
                   case("q"), PASSAGES)
check("a citation to an unsupplied document is invented",
      fabricated["invented"], ["ornithology_handbook.pdf"])

# The model dropping or mangling the extension is not a fabrication.
loose = judge("Air sacs do it. [bird_anatomy, section Respiration]",
              case("q"), PASSAGES)
check("a citation missing its extension is still supplied", loose["invented"], [])

# real: bird-syrinx cited two documents in one bracket pair.
check("two sources in one bracket parse as one citation each",
      cited_sources("[bird_vocalization.pdf, page 2; bird_vocalization.docx, "
                    "section Anatomy and physiology]"),
      ["bird_vocalization.pdf"])

check("no citation at all", cited_sources("Brood parasitism."), [])

# --------------------------------------------------------------- correctness
hit = judge("The interlocking structures are barbules.",
            case("q", "barbules"), PASSAGES)
check("the labelled string present is correct", hit["correct"], True)
check("a correct answer needs no adjudication", hit["near"], None)

# real: bird-barbules. Genuinely the wrong structure, not a wording difference.
wrong = judge("The interlocking structures that hold the vane of a feather "
              "together are barbicels. [feather.pdf, page 2]",
              case("q", "barbules"), PASSAGES)
check("a wrong answer is not correct", wrong["correct"], False)

# real: bird-keel. The right answer in the wrong words. `correct` is allowed to
# say False -- it is a containment test and it is honest about that -- but the
# case has to be surfaced rather than silently counted against the model.
keel = judge("The keel on their breastbone anchors the muscles needed for wing "
             "movement. [flightless_bird.pdf, page 2]",
             case("q", "keeled sternum"), PASSAGES)
check("a paraphrase does not contain the labelled string", keel["correct"], False)
check("a paraphrase sharing the label's words is flagged to be read",
      keel["near"], True)

# A wrong answer that shares no word with the label is not worth reading first.
check("an unrelated wrong answer is not flagged as near",
      judge("Birds have hollow bones.", case("q", "keeled sternum"),
            PASSAGES)["near"], False)

# An adversarial case carries no answer string, so correctness does not apply.
check("correctness is not scored on an unanswerable case",
      judge("Not in these documents.", case("adv", unanswerable=True),
            PASSAGES)["correct"], None)

# -------------------------------------------------------------- groundedness
grounded = judge("Down feathers lack barbicels so the barbules trap air.",
                 case("q"), PASSAGES)
check("a paraphrase of the passages scores high", grounded["grounded"] > 0.8, True)

ungrounded = judge("Photosynthesis converts sunlight into chemical energy "
                   "inside chloroplasts.", case("q"), PASSAGES)
check("prose about something else scores low", ungrounded["grounded"] < 0.2, True)

check("an empty answer does not divide by zero",
      judge("", case("q"), PASSAGES)["grounded"], 0.0)

# The stop list must not swallow the words that carry the evidence.
check("content words survive the stop list", "barbules" in words("the barbules"), True)
check("stopwords are removed", "the" in words("the barbules"), False)

print(f"\n{checks - len(fails)}/{checks} answer-quality judge checks passed")
for f in fails:
    print(f"  FAIL  {f}")
raise SystemExit(1 if fails else 0)
