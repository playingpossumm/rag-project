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

from hybrid import build_bm25
from parent import expand
from retrieve import CANDIDATE_K, EMBEDDING_MODEL, TOP_K, load_index, retrieve

# Lower than the abstention gate's own default: refusing an answerable question
# is the costlier error here, and confidence is returned on every response so a
# caller can apply a stricter bar without the pipeline hiding results from them.
DEFAULT_MIN_CONFIDENCE = -2.0


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
    expansion: str = "page",
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    generate: bool = False,
) -> Answer:
    """Answer a question from the indexed corpus.

    With generate=False (the default) this returns the retrieved passages and
    their citations, and makes no network call of any kind.
    """
    index, metadata, model, bm25 = _resources()

    # Ranking happens on small chunks; confidence is read here, before any
    # expansion, so it reflects the text the reranker actually scored.
    ranked = retrieve(question, index, metadata, model,
                      k=k, candidate_k=candidate_k, bm25=bm25, use_reranker=True)

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
        from generate_answer import synthesize  # imported lazily: needs an API key
        result.answer = synthesize(question, results)
        result.mode = "generated"

    return result
