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
    # real, all four: the ML corpus reported 1 of 5 adversarial questions
    # refused when all five were. An adverb between the negation and the verb,
    # a verb the list did not hold, and a phrase between the modal and its
    # object were enough to lose them.
    "The noise schedule used by the denoising diffusion model is not "
    "explicitly stated in the provided excerpts.",
    "The text does not explicitly state how the reward model is trained for "
    "RLHF.",
    "The question about the cost to train the model is not answered in the "
    "provided excerpts.",
    "I cannot provide an answer based on the provided excerpts.",
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

# An answer that hedges and then answers has answered. Broadening the patterns
# to catch the four misses above turned five of these into "wrongly refused",
# which is the same error pointing the other way. All four are real.
HEDGE_THEN_ANSWER = [
    # real: moe-routing, and the labelled answer is in the second sentence.
    "The text does not explicitly state the type of load balancing loss used "
    "for expert routing. However, it does mention that the "
    "auxiliary-loss-free load balancing strategy is adopted, and that a "
    "sequence-level load balancing loss is applied.",
    # real: bleu-ende
    "The model that achieved 28.4 BLEU on WMT 2014 English-to-German is not "
    "explicitly stated in the provided excerpts. However, it is mentioned "
    "that the Transformer model achieves 28.4 BLEU on that translation task.",
    # real: qf-markowitz
    "The text does not explicitly mention the name of the individual whose "
    "portfolio selection theory underpins modern allocation methods. However, "
    "it does mention Markowitz (1952) as the one who established the "
    "mean-variance framework.",
    # real: attn-uses, a partial answer. It answers the first of three points
    # and says the excerpts do not carry the third.
    "According to [attention_is_all_you_need.pdf, page 3], multi-head "
    "attention is used in an encoder-decoder Transformer in three ways:\n"
    "1. In encoder-decoder attention layers, the queries come from the "
    "previous decoder layer, and the memory keys and values come from the "
    "output of the encoder.\n"
    "The provided excerpts do not mention the third way.",
]

for text in HEDGE_THEN_ANSWER:
    r = judge(text, case("q"), PASSAGES)
    check(f"hedge then answer is not a refusal: {text[:40]!r}",
          r["refused"], False)

# But a refusal that goes on to describe what the documents *do* hold is still
# a refusal: it is saying what it does not have, not answering. All three real.
DECLINE_THEN_DESCRIBE = [
    # adv-bird-insulin
    "This question cannot be answered based on the provided context excerpts. "
    "The context only discusses the physiology and adaptations of various bird "
    "species, such as the emperor penguin, but does not provide information on "
    "human physiology or the regulation of blood glucose by insulin.",
    # adv-bird-consensus
    "I don't have any information about proof of stake protocols or consensus "
    "mechanisms from the provided excerpts. The provided excerpts are about "
    "the classification of bird orders, the origin of birds, and the "
    "adaptations of the emperor penguin.",
    # adv-qf-syrinx
    "I don't have enough information to answer this question. The provided "
    "excerpts appear to be related to machine learning and natural language "
    "processing, but they do not mention birds or the production of song.",
]

for text in DECLINE_THEN_DESCRIBE:
    check(f"declining then describing the corpus is a refusal: {text[:36]!r}",
          judge(text, case("adv", unanswerable=True), PASSAGES)["refused"], True)

# And the confabulations that look almost identical but do answer. Both real,
# and both are genuine gate failures that must not be excused as refusals.
CONFABULATED = [
    # adv-rlhf: answers with adjacent material about a different model
    "The text does not explicitly state how the reward model is trained for "
    "RLHF. However, it does mention that the initial RL model is trained and "
    "evaluated without LLM guidance, and then its outputs are analyzed to "
    "identify potential improvements.",
    # adv-qf-broker-fee: invents a commission from an unrelated fee term
    "The text does not explicitly state the commission charged by the broker. "
    "However, it does mention that the CPM charges a fee proportional to the "
    "size of the trade, and that one minus that fee is the net percentage of "
    "the trade after trading costs.",
]

for text in CONFABULATED:
    check(f"answering from adjacent material is not a refusal: {text[:36]!r}",
          judge(text, case("adv", unanswerable=True), PASSAGES)["refused"], False)

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

# ------------------------------------------------- brackets that are not cites
# The ML papers are full of notation, and the model copies it out of the
# passages it was given. Before the shape rule these counted as citations, and
# because they name no supplied document they counted as *invented* ones: 20
# fabrications reported across 12 answers, none of them real. All of these are
# real strings from that run.
NOTATION = [
    # real: mha-def, attention notation
    "head_i = Attention(QW_i^[Q], KW_i^[K], VW_i^[V])",
    # real: pos-enc-fn, the d_model subscript
    "PE(pos, 2i) = sin(pos / 10000^(2i/d_[model]))",
    # real: warmup, the learning-rate schedule verbatim
    "lrate = d_[model]^(-0.5) * min(step_num^(-0.5), step_num * "
    "warmup_[steps]^(-1.5))",
    # real: scale-why, a reference number rather than a document
    "This follows the argument in [3].",
    "The gate fires when score > [threshold].",
]

for text in NOTATION:
    check(f"notation is not a citation: {text[:40]!r}",
          cited_sources(text, {"attention", "t5", "bert"}), [])
    r = judge(text, case("q"), PASSAGES)
    check(f"notation invents nothing: {text[:40]!r}", r["invented"], [])

# A citation still has to be caught when it wears the format it was asked for.
check("a fabricated document with an extension is still invented",
      judge("Birds sing. [ornithology_handbook.pdf, page 44]",
            case("q"), PASSAGES)["invented"], ["ornithology_handbook.pdf"])
check("a fabricated document with a locator is still invented",
      judge("Birds sing. [handbook of birds, page 44]",
            case("q"), PASSAGES)["invented"], ["handbook of birds"])
# A bare supplied name, with neither extension nor locator, is still a citation.
check("a bare supplied name counts as a citation",
      cited_sources("As shown in [feather].", {"feather.pdf", "bird_anatomy.docx"}),
      ["feather"])

# real: qf-order-flow. One real citation and a formula, against a long
# filename. Every single character in that formula is a substring of the
# filename, so before the length floor this counted as six citations to a paper
# the answer cites once.
LONG = "multi_level_market_making_with_reinforcement_learning"
FORMULA = ("_I t_ = e[-][B][(][t][+/t][)] _I-/t + B_ "
           f"[{LONG}.pdf, page 11]")
check("notation beside a long filename is one citation, not six",
      cited_sources(FORMULA, {LONG}), [f"{LONG}.pdf"])

# The floor must not break a genuinely short filename citing itself.
check("a short filename still matches itself exactly",
      cited_sources("As shown in [t5].", {"t5", "bert"}), ["t5"])

# And a short token that is not a supplied name is still not a citation.
check("a one-character bracket group is never a citation",
      cited_sources("The value [t] rises.", {LONG}), [])

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
