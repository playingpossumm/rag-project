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

print(f"\n{checks - len(fails)}/{checks} excerpt checks passed")
for f in fails:
    print(f"  FAIL  {f}")
raise SystemExit(1 if fails else 0)
