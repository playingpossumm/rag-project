"""Programmatic entry point: one function, structured output.

Two decisions shape this module.

**Retrieval-only by default.** Returning the source passages verbatim costs
nothing, cannot hallucinate, and for contracts or policies is often better than
a paraphrase -- the exact wording is the thing that matters. Generation is an
optional layer on top, not a dependency, so the system is fully usable with no
API key and no spend.

**Structured, not printed.** Results are returned as data so a caller can render
them, store them, or feed them onward. The HTTP wrapper and the CLI are both
thin shells over `ask()`; none of the logic lives in either.

Sources are grouped by document, because the target behaviour is to compile
every relevant source rather than pick one. When several documents answer the
same question the caller is told so explicitly -- that is a prompt to judge,
not a defect to hide.
"""
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from abstain import ABSTAIN_THRESHOLD
from hybrid import build_bm25
from parent import expand
from retrieve import CANDIDATE_K, EMBEDDING_MODEL, TOP_K, load_index, retrieve

# Shares the calibrated threshold with the abstention gate (see abstain.py for
# the measurement). Safe to use the same value here because this flag is
# ADVISORY: `ask` always returns its passages and citations regardless, so a
# low-confidence result is labelled rather than withheld. Refusing an answerable
# question is the costlier error, and nothing is refused on this path.
DEFAULT_MIN_CONFIDENCE = ABSTAIN_THRESHOLD

# Measured on the 84-case golden set (context recall / tokens per query):
#
#   none (chunks)   0.742 @   997      window +/-1  0.833 @ 2371
#   page            0.848 @ 4828
#
# Window is the better trade: it recovers 91 of the 106 points page gains over
# raw chunks, for less than half the context. Page was the earlier default
# because on a 23-case set window measured no better than raw chunks at all --
# one of three conclusions that reversed when the test set grew.
#
# Page remains available and is the right choice where a citation must point at
# a complete unit a reader can verify in one place -- a contract clause read
# half-in and half-out of context is worse than useless.
#
# The numbers above were measured for a GENERATOR reading the context.
# pipeline_trace.DISPLAY_EXPANSION is "none" because the trace feeds a reader
# instead, and that case was measured separately on 2026-09-03 as the page
# renders it, a lead-in only. Neighbours moved answer-on-page from 0.808 to
# 0.832 for 591 to 836 characters a passage, a symmetric lead-out reached 0.840
# at 1073, and a 600-character lead-out gained nothing further, so display kept
# raw chunks. The two defaults disagree on purpose, and test_trace.py holds
# both so that neither can drift into the other unnoticed.
DEFAULT_EXPANSION = "window"


@dataclass
class Passage:
    source: str
    locator: dict          # {"kind": "page", "value": 5} -- see note in locators
    score: float
    text: str


@dataclass
class Answer:
    question: str
    mode: str                       # "retrieval" | "generated"
    confident: bool
    confidence: float               # top reranker score; higher is better
    documents: list[str]            # distinct documents represented
    passages: list[Passage] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    answer: str | None = None       # populated only in "generated" mode

    def to_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=1)
def _resources():
    """Load index, embedder and BM25 once per process.

    Cached because every one of these is expensive to build and none of them
    change between queries; a caller serving many questions would otherwise pay
    the cost per request.
    """
    index, metadata = load_index()
    model = SentenceTransformer(EMBEDDING_MODEL)
    bm25 = build_bm25(metadata)
    return index, metadata, model, bm25


def ask(
    question: str,
    k: int = TOP_K,
    candidate_k: int = CANDIDATE_K,
    expansion: str = DEFAULT_EXPANSION,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    generate: bool = False,
    resources=None,
    rerank_blend: float | None = None,
) -> Answer:
    """Answer a question from the indexed corpus.

    With generate=False (the default) this returns the retrieved passages and
    their citations, and makes no network call of any kind.

    `resources` is `(index, metadata, model, bm25)` for a caller that already
    holds a corpus, and `rerank_blend` is that corpus's blend. Both exist for
    the server, which can switch corpora while this module's own cache cannot:
    `_resources()` is an lru_cache over the process, so a server that called
    this function served whichever corpus loaded first no matter what the
    interface said was selected -- the ML papers, under Ornithology's name and
    Ornithology's document count. Omit both and the behaviour is unchanged,
    because the library entry point should still be able to load a corpus by
    itself.
    """
    index, metadata, model, bm25 = resources or _resources()

    # Ranking happens on small chunks; confidence is read here, before any
    # expansion, so it reflects the text the reranker actually scored.
    ranked = retrieve(question, index, metadata, model,
                      k=k, candidate_k=candidate_k, bm25=bm25, use_reranker=True,
                      rerank_blend=rerank_blend)

    confidence = ranked[0]["rerank_score"] if ranked else float("-inf")
    confident = confidence >= min_confidence

    notes = []
    if not confident:
        notes.append(
            "Low confidence: nothing in the indexed documents clearly answers this. "
            "The passages below are the closest matches, not necessarily an answer."
        )

    results = expand(ranked, metadata, mode=expansion) if ranked else []
    passages = [
        Passage(source=r["source"],
                locator=r["locator"],
                score=round(float(r.get("rerank_score", r.get("score", 0.0))), 3),
                text=r["text"])
        for r in results
    ]
    documents = list(dict.fromkeys(p.source for p in passages))

    # Only meaningful when something relevant was actually found. Announcing
    # that "5 documents contain relevant material" directly after reporting that
    # nothing answers the question contradicts itself, and would train a reader
    # to ignore the note in exactly the cases it matters.
    if confident and len(documents) > 1:
        notes.append(
            f"{len(documents)} different documents contain relevant material "
            f"({', '.join(documents)}). They may not agree -- read the passages "
            f"and judge which applies."
        )

    result = Answer(
        question=question,
        mode="retrieval",
        confident=confident,
        confidence=round(float(confidence), 3),
        documents=documents,
        passages=passages,
        notes=notes,
    )

    if generate:
        # Lazy: importing generate.py pulls in the anthropic client and reads
        # .env, and the whole point of retrieval-only being the default is that
        # a caller without an API key never touches any of that.
        #
        # This read `from generate_answer import synthesize` until 2026-08-21.
        # No module of that name has ever existed. Being both lazy and inside
        # the optional branch, it could only fail on a path that has never run
        # for want of API credit -- so the code was wrong for as long as it was
        # unexercised, and looked fine.
        from generate import synthesize_with_backend

        # Whichever backend RAG_GENERATOR names. The default is unchanged;
        # setting it to "ollama" runs the same contract against a local model,
        # which is how this path can execute at all without API credit.
        result.answer = synthesize_with_backend(question, results)
        result.mode = "generated"

    return result
