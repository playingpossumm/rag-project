"""Optional generation layer: write retrieved passages into a cited answer.

Retrieval-only is the default everywhere else in this project, and deliberately
so -- returning the source text verbatim costs nothing and cannot hallucinate.
This module is the layer on top, for callers who want prose. It is optional in
the strict sense: nothing imports it unless `generate=True` is passed.

`synthesize()` is the library entry point and `main()` is a thin CLI over the
same function, so `api.ask(generate=True)` and `python src/generate.py "..."`
cannot answer differently.

**This path has never completed a real call** -- the account has no credit -- so
treat the code as written-but-unverified. Three defects were found by reading it
against the rest of the codebase rather than by running it, and are fixed here:

  * `api.py` imported `synthesize` from a module named `generate_answer`, which
    has never existed. The import is lazy and inside the optional branch, so it
    never raised.
  * `build_context()` read `chunk["page"]`. Chunks carry `locator`
    ({"kind": "page", "value": 5}); there is no `page` key and has not been one
    since locators replaced it. Every call would have died on KeyError.
  * `max_tokens=1024` was set when a model's output budget held only its answer.
    Thinking is on by default on Claude Opus 5 and is billed and counted against
    the same ceiling, so a hard question could have spent the whole budget
    thinking and returned a truncated answer or none at all.

Which is the argument for the rest of the project's discipline, restated: code
that has never run is not code that works, and reviewing it against the data it
actually receives is cheaper than finding out later.
"""
import sys

import anthropic
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from abstain import ABSTAIN_THRESHOLD, abstention_message, should_abstain
from parent import expand
from retrieve import EMBEDDING_MODEL, load_index, retrieve

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

CLAUDE_MODEL = "claude-opus-5"

# Generous rather than tight. max_tokens is a ceiling, not a spend -- only
# tokens actually generated are billed -- and on this model thinking counts
# against the same ceiling, so a low value risks truncating the answer to save
# nothing. Effort is what actually controls cost here, and it is set to "low":
# the reasoning has already been done by the retriever, and this step is
# grounded rewriting, not analysis.
MAX_TOKENS = 16000
EFFORT = "low"

SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided context excerpts. "
    "For every claim, cite the source and page number in the format [source, page X]. "
    "If the context does not contain the answer, say so plainly instead of guessing."
)


def cite(chunk: dict) -> str:
    """Human-readable citation for a chunk: 'paper.pdf, page 5'.

    Reads `locator` rather than a page number, because the locator's kind
    varies by format -- pages for PDFs, slides for decks, sheets for
    spreadsheets -- and a citation that says "page 3" of a slide deck sends the
    reader somewhere that does not exist.
    """
    loc = chunk.get("locator", {})
    return f"{chunk['source']}, {loc.get('kind', 'location')} {loc.get('value', '?')}"


def build_context(chunks: list[dict]) -> str:
    parts = [f"[{cite(c)}]\n{c['text']}" for c in chunks]
    return "\n\n---\n\n".join(parts)


def synthesize(question: str, chunks: list[dict]) -> str:
    """Write the retrieved passages into one cited answer.

    Takes the passages a caller has already retrieved rather than retrieving its
    own, so `ask(generate=True)` generates from exactly the text it reports
    having used. Fetching them again here would let the citations shown to the
    reader and the text the model actually read drift apart -- the one failure
    a citation is supposed to make impossible.

    Raises RuntimeError if the model declines or returns no text, rather than
    returning an empty string that a caller would render as a blank answer.
    """
    if not chunks:
        raise ValueError("synthesize() needs at least one passage")

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        output_config={"effort": EFFORT},
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Context:\n\n{build_context(chunks)}\n\nQuestion: {question}",
        }],
    )

    # Checked before reading content: a refusal returns HTTP 200 with no text
    # block, and `next(...)` over an empty generator raises StopIteration, which
    # in a caller reads as a bug in this code rather than a decision by the
    # model.
    if response.stop_reason == "refusal":
        detail = getattr(response.stop_details, "explanation", None) or "no explanation given"
        raise RuntimeError(f"the model declined to answer: {detail}")

    text = "\n".join(b.text for b in response.content if b.type == "text").strip()
    if not text:
        raise RuntimeError(
            f"no text in the response (stop_reason={response.stop_reason!r}). "
            "If this is 'max_tokens', raise MAX_TOKENS."
        )
    return text


def main():
    query = " ".join(sys.argv[1:]) or input("Ask a question: ")

    index, metadata = load_index()
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    # Retrieve WITHOUT expansion first: the abstention gate reads the reranker's
    # confidence on the small chunk it actually scored. Expanding to a whole page
    # would dilute that judgement with surrounding text the reranker never saw.
    ranked = retrieve(query, index, metadata, embed_model)

    if should_abstain(ranked):
        print(f"\n{abstention_message(query)}")
        best = ranked[0] if ranked else None
        if best:
            print(f"\nClosest match was {cite(best)} "
                  f"(confidence {best['rerank_score']:+.2f}, "
                  f"below threshold {ABSTAIN_THRESHOLD:+.1f}).")
        return

    # expansion="page": rank on small chunks, then hand the model whole pages.
    # Measured to lift context recall 0.958 -> 1.000 at ~2.4x the token cost --
    # worth it here, since a citation the model cannot substantiate from the
    # text it was given is the failure this whole pipeline exists to prevent.
    chunks = expand(ranked, metadata, mode="page")

    print(f"\n{synthesize(query, chunks)}\n")
    print("Sources consulted:")
    for c in chunks:
        print(f"  - {cite(c)}")


if __name__ == "__main__":
    main()
