"""The excerpt window, against the passage it was getting wrong.

`pipeline_trace._brief` chooses which 260 characters of a passage the interface
shows. It used to take the head, which is right whenever a passage opens with
its substance and wrong in the one place it matters most: the first chunk of a
paper is its title, its authors and their email addresses, so a topical
question got an excerpt made entirely of metadata.

The case that found it is real. Asked "What problem does normalizing layer
inputs address?", the interface showed the batchnorm title block cut off at
"complicated by", with the answering clause immediately past the edge.

Hermetic: no index, no model, no server. `_brief` is a pure function of a chunk
and a query.

    .venv\\Scripts\\python.exe src\\test_excerpt.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline_trace import _brief  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

fails: list[str] = []
checks = 0


def check(name, ok, detail=""):
    global checks
    checks += 1
    if not ok:
        fails.append(f"{name}{'  ' + str(detail) if detail else ''}")


# real: batch_normalization.pdf page 1, the chunk the owner reported.
BATCHNORM = (
    "## Batch Normalization: Accelerating Deep Network Training by Reducing "
    "Internal Covariate Shift #### Sergey Ioffe Google Inc., "
    "sioffe@google.com #### Christian Szegedy Google Inc., "
    "szegedy@google.com ### Abstract Training Deep Neural Networks is "
    "complicated by the fact that the distribution of each layer's inputs "
    "changes during training, as the parameters of the previous layers "
    "change. We refer to this phenomenon as internal covariate shift, and "
    "address the problem by normalizing layer inputs."
)
Q = "What problem does normalizing layer inputs address?"

out = _brief({"text": BATCHNORM}, Q)
check("the excerpt no longer opens with the title",
      not out.startswith("## Batch Normalization"), out[:60])
check("it does not show the authors' email addresses",
      "@google.com" not in out, out[:60])
# Deliberately not "does the phrase 'internal covariate shift' appear": it
# appears in the title, so that test passes on the very excerpt this fix
# exists to stop showing. The same weakness is why the golden label credits
# the title block as a correct answer. Ask for the sentence instead.
check("it reaches the sentence that answers, not the title that names it",
      "we refer to this phenomenon" in out.lower()
      or "address the problem by normalizing" in out.lower(), out[:80])
check("it stays within the limit", len(out) <= 260, len(out))

# A passage that opens with its substance must be left alone: its first
# sentence wins the same test, so the head is still the head.
OPENS_WELL = (
    "Air sacs make airflow through the lungs one-directional, so fresh air "
    "passes over the gas-exchange surface on both inhalation and exhalation. "
    "This is unlike the tidal flow of a mammalian lung, where inhaled and "
    "exhaled air mix in the same passages and the exchange surface never sees "
    "a continuous stream of fresh air at all times during the cycle."
)
head = _brief({"text": OPENS_WELL}, "What makes airflow one-directional?")
check("a passage that opens with its answer is unchanged",
      head.startswith("Air sacs make airflow"), head[:50])

# Shorter than the limit: returned whole, never windowed.
short = "Down feathers trap air and provide insulation."
check("a passage under the limit comes back whole",
      _brief({"text": short}, Q) == short)

# No query, or a query of nothing but stopwords: fall back to the head rather
# than picking an arbitrary window.
check("no query falls back to the head",
      _brief({"text": BATCHNORM}, "").startswith("## Batch Normalization"))
check("a query of only stopwords falls back to the head",
      _brief({"text": BATCHNORM}, "what is the of").startswith("## Batch"))

# A query sharing no word with the passage must not silently window on noise.
check("a query with no overlap falls back to the head",
      _brief({"text": BATCHNORM}, "peregrine falcon nostril tubercles")
      .startswith("## Batch Normalization"))

# Whitespace is collapsed, as the old implementation did, because the chunk
# carries the PDF's line breaks and the interface renders one line.
check("newlines are collapsed",
      "\n" not in _brief({"text": "a\nb\nc " + "x" * 400}, "b"))

# Ties go to the earlier window: with two equally good sentences a reader
# expects the one nearer the start.
TIE = ("The alula prevents a stall at low speed. " + "Filler sentence. " * 12
       + "The alula prevents a stall at low speed.")
tie = _brief({"text": TIE}, "Which structure prevents a stall at low speed?")
check("ties go to the earlier sentence", tie.startswith("The alula"), tie[:40])

# ---- reported 2026-09-01, by reading the page ------------------------------

# real: lora.pdf page 2. Half the budget went on citation apparatus, so the
# excerpt displayed 122 characters of prose out of 260 once ui/answer-mark.js
# had stripped what the window had already paid for.
LORA = (
    "often introduce inference latency (Houlsby et al., 2019; Rebuffi et al., "
    "2017) by extending model depth or reduce the model's usable sequence "
    "length (Li & Liang, 2021; Lester et al., 2021; Hambardzumyan et al., "
    "2020; Liu et al., 2021) (Section 3). More importantly, these method often "
    "fail to match the fine-tuning baselines, posing a trade-off between "
    "efficiency and model quality. We take inspiration from Li et al. (2018a); "
    "Aghajanyan et al. (2020) which show that the learned over-parametrized "
    "models in fact reside on a low intrinsic dimension."
)
out = _brief({"text": LORA}, "Does the adaptation method add inference latency?")
check("an author-year citation is not shown", "et al., 2019" not in out, out[:70])
check("a parenthetical pointer to a section is not shown",
      "(Section 3)" not in out, out[:70])
check("the sentence itself survives the stripping",
      "introduce inference latency by extending model depth" in out, out[:70])
check("no space is left in front of the punctuation",
      " ." not in out and " ," not in out, out[:70])

# An ordinary parenthetical is not apparatus and stays.
KEEP = ("Birds have many bones that are hollow (pneumatized) with criss-crossing "
        "struts or trusses for structural strength. " + "Filler sentence. " * 20)
check("a parenthetical that is not a citation is kept",
      "(pneumatized)" in _brief({"text": KEEP}, "which bones are hollow"))

# real: t5.pdf page 9. "Appendix D." is a sentence end, and treating the single
# capital as an abbreviation glued two sentences together.
T5 = (
    "examples is shown in Figure 1. We provide full examples of preprocessed "
    "inputs for every task we studied in Appendix D. Our text-to-text "
    "framework follows previous work that casts multiple NLP tasks into a "
    "common format: McCann et al. propose the \u201cNatural Language "
    "Decathlon\u201d, a benchmark that uses a consistent question-answering "
    "format for a suite of ten NLP tasks. Radford et al. show that a language "
    "model can perform some tasks in a zero-shot setting given a natural "
    "language prompt, and Keskar et al. cast several tasks into span "
    "extraction over the input. The real chunk is 953 characters, so the "
    "window has to choose a start, which is the behaviour under test here."
)
out = _brief({"text": T5}, "How are all NLP tasks cast into a single format?")
check("the window opens at a sentence rather than mid-clause",
      not out.startswith("examples is shown"), out[:60])

# An excerpt that stops mid-word reads as a truncation rather than a statement.
LONG = ("First sentence that does not answer anything at all here. "
        + "The alula is a small group of feathers on the leading edge of the "
          "wing that prevents a stall at low speed by keeping airflow attached. "
        + "Filler sentence about something else entirely. " * 12)
out = _brief({"text": LONG}, "which feathers prevent a stall at low speed")
check("the excerpt ends at a sentence boundary",
      out.rstrip().endswith((".", "!", "?")), out[-60:])
check("and it still carries the answer",
      "prevents a stall at low speed" in out, out[:80])

# The limit is a budget, not a target: a passage under it comes back whole and
# is not padded or cut to a boundary that does not exist.
SHORT_WHOLE = "Down feathers trap air and provide insulation for the bird."
check("a passage under the limit is returned whole",
      _brief({"text": SHORT_WHOLE}, "what do down feathers do") == SHORT_WHOLE)

print(f"\n{checks - len(fails)}/{checks} excerpt checks passed")
for f in fails:
    print(f"  FAIL  {f}")
raise SystemExit(1 if fails else 0)
