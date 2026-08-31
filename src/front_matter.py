"""A paper's title block is metadata, not an answer.

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

**Scope, measured before writing any of this.** 55 chunks across the three
corpora carry it: 29 of 5,459 on the ML papers, 26 of 6,184 on quant, and none
at all on the birds, which are Wikipedia articles and have no title pages.
Roughly one per PDF, which is what a title block should be. **No golden case
names one as its gold passage**, on any corpus, so removing them from
consideration cannot cost a labelled answer.

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
