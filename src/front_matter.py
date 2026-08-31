"""A paper's title block is metadata, not an answer, and dropping it is worse.

A refuted experiment, kept because the reasoning is worth more than the
filter. Off by default; the verdict and its numbers are below the argument.

**The defect this exists for.** Asked "What problem does normalizing layer
inputs address?", the pipeline returned the batchnorm paper's first chunk at
rank 1:

    ## Batch Normalization: Accelerating Deep Network Training by
     Reducing Internal Covariate Shift
    #### Sergey Ioffe Google Inc., sioffe@google.com
    #### Christian Szegedy Google Inc., szegedy@google.com

Nothing in it answers the question, and the interface had nothing to set bold,
because the question's words appear nowhere in it. The passage that does answer,
"the covariate shift problem can be reduced by fixing the mean and the variance
of the summed inputs within each layer", sat at rank 4.

**Why it wins.** A title is the densest possible statement of what a paper is
about, so it carries more of a topical question's vocabulary per token than any
paragraph in the document. A cross-encoder asked "does this passage answer this
question" reads that density as relevance. The title scored 4.545 and the real
answer 3.53.

This is a different failure from the ones already recorded here. It is not a
wrong document, and not a ranking that a better reranker fixes: the right
document came back first, and the part of it that came back is the part that
names the paper.

**Scope, and its precision, both read rather than assumed.** 56 chunks across
the three corpora match: 30 of 5,459 on the ML papers, 26 of 6,184 on quant, and
none at all on the birds, which are Wikipedia articles and have no title pages.
That is roughly one per PDF, which is what a title block should be, and all 56
were read. One is a false positive, a page-1 figure in
`break_it_down_pass_it_on_cross_task_skill_transfer.pdf` listing interface
labels, which is document content. A second, the reproduction-permission notice
on `attention_is_all_you_need.pdf`, is not a title block either, although it
answers nothing.

**Measured, refuted, and off by default. Read this before turning it on.**
Everything above is the argument, and the argument is not the result. End to
end on the shipped configuration:

    ML papers   any-hit 0.851 -> 0.836,  MRR 0.739 -> 0.724
    Quant       any-hit 0.886 -> 0.857,  MRR 0.779 -> 0.764
    Birds       unchanged, having no title pages to drop

**And the reason it loses is worth more than the filter.** An earlier draft of
this docstring claimed no golden case names a title block as its gold passage.
That is true only because this golden set has no passage-level labels at all.
At the level the labels are written and scored, the page, **43 of the 102
answerable cases in the two corpora that have title pages have a title block on
one of their gold pages, and 17 of those have the answer string inside the title
itself**. `bn-covariate` wants
"internal covariate shift" from a paper titled "... by Reducing Internal
Covariate Shift", on page 1. So the measurement scores the title block as a
correct answer, and removing it removes a hit.

**The loss is two questions, and both are genuine.**
`src/audit_title_credit.py` finds two hits satisfied only by a chunk this module
flags, `bn-covariate` and `qf-whale-attack`, and they are the whole of the loss
above to three decimals on both any-hit and MRR. Read in full, both chunks
answer the question outright, because `is_front_matter` judges the first 600
characters and these chunks run to about 1,000, and what follows a title block
is the abstract. The batchnorm chunk continues into "We refer to this phenomenon
as internal covariate shift, and address the problem by normalizing layer
inputs".

So this filter drops abstracts, which is why it loses. That is information
destroyed rather than a scoring artefact withdrawn, and it settles the question:
the filter is off and it stays off.

The first audit reported those two hits as false credits, and that reading
reached five documents and the deployed site before the chunks were read whole.
It took the flag on a chunk's opening for a verdict on the chunk, which is the
same error that put a title block on screen in the first place. The defect this
module was written for was a presentation defect and not a retrieval one:
`pipeline_trace._brief` was showing the head of a passage, the answer sat 260
characters further in, and windowing the excerpt on the answering sentence
fixed the reported case without removing anything.

`src/sweep_front_matter.py` reproduces the table above and
`src/audit_title_credit.py` the count.

**Deliberately narrow.** All three conditions must hold: the first page, the
chunk opening with a heading the PDF converter produced, and an email address
or an affiliation inside its opening. A page-1 chunk of ordinary prose does not
match. Neither does a later page citing an institution, nor an abstract that
happens to start a page. The cost of a false positive here is a real passage
silently removed from every search, so the test errs toward missing front
matter rather than toward catching prose.
"""
import re

EMAIL = re.compile(r"[\w.\-]+@[\w.\-]+\.\w+")

# Institutions that appear on a title block. Matched only inside the opening of
# a first-page chunk, never over a whole document, so a paper discussing Google
# on page 9 is unaffected.
AFFILIATION = re.compile(
    r"\b(?:Google|University|Universit|Institute|Research|Labs?|Corporation|"
    r"Microsoft|OpenAI|DeepMind|Facebook|Meta AI|Stanford|Berkeley|MIT|"
    r"Carnegie|Allen Institute|Amazon|NVIDIA|Anthropic|Inc\.)\b")

# How much of the chunk counts as its opening.
HEAD = 600


def _is_first_page(locator) -> bool:
    """True for page one, in either shape a locator takes.

    `metadata.json` holds {"kind": "page", "value": 1}; the recorded payloads
    and the trace hold the rendered string. Both reach this function.
    """
    if isinstance(locator, dict):
        return (locator.get("kind") == "page"
                and str(locator.get("value")).strip() == "1")
    return str(locator or "").strip().lower() in ("page 1", "p. 1", "1")


def is_front_matter(text: str, locator=None) -> bool:
    """Does this chunk name the paper rather than say something about it?"""
    if not text or not _is_first_page(locator):
        return False
    head = text[:HEAD]
    if not head.lstrip().startswith("#"):
        return False
    return bool(EMAIL.search(head) or AFFILIATION.search(head))


def drop(candidates: list[dict]) -> list[dict]:
    """Remove title blocks from a candidate list.

    Returns the list unchanged when every candidate is front matter, which
    cannot happen with a real corpus and would otherwise turn a narrow filter
    into an empty result set.
    """
    kept = [c for c in candidates
            if not is_front_matter(c.get("text", ""), c.get("locator"))]
    return kept or candidates
