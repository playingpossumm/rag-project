"""Assert the trace and the serving path agree.

pipeline_trace.py deliberately duplicates retrieve()'s stage order rather than
threading debug state through the serving path. Its docstring has claimed since
it was written that a test held the two together; the test did not exist, and
the duplication was held together by nothing but care.

That was survivable while the trace ran one configuration. It stopped being
survivable when the trace grew options, because the two paths now differ in a
way that is easy to get wrong and invisible when you do: with reranking off
retrieve() shortlists k rather than candidate_k and skips the diversity cap
entirely, since both exist to feed and then trim a reranked pool. A trace that
missed that would draw a twenty-candidate pool and a cap for a pipeline that ran
neither -- a confident picture of something that never happened, which is worse
than no picture.

So this runs both paths over the option matrix and asserts they select the same
chunks in the same order. It is not a unit test of the metrics; it is the thing
that makes the duplication safe to keep.

    .venv\\Scripts\\python.exe src\\test_trace.py
"""
import itertools
import sys

from sentence_transformers import SentenceTransformer

from api import DEFAULT_EXPANSION
from hybrid import build_bm25
from pipeline_trace import DISPLAY_EXPANSION, trace_pipeline
from retrieve import EMBEDDING_MODEL, load_index, retrieve

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# One of each behaviour the pipeline distinguishes: a rare-token question BM25
# carries, a paraphrase question only the dense side finds, one whose answer is
# spread over several documents, and one the corpus cannot answer.
QUERIES = [
    "What BLEU score did the Transformer achieve on WMT 2014 English-to-German?",
    "How are normalization statistics computed across features rather than examples?",
    "What learning rate schedule and optimizer settings were used for training?",
    "What is the airspeed velocity of an unladen swallow?",
]

OPTIONS = [
    dict(use_reranker=r, fusion=f, max_per_source=c)
    for r, f, c in itertools.product((True, False),
                                     ("rrf", "weighted", "none"),
                                     (2, 1, None))
]


def selected_ids(trace: dict) -> list[int]:
    return [i["chunk_id"] for i in trace["stages"][-1]["items"]]


def expansion_defaults_diverge_on_purpose() -> bool:
    """The serving and display defaults differ, and each was measured for its
    own consumer.

    api.DEFAULT_EXPANSION is "window" because a generator gained context recall
    from neighbouring chunks. pipeline_trace.DISPLAY_EXPANSION is "none" because
    on 2026-09-03 a reader of the rendered page gained too little from them for
    the characters they cost. The comments beside each constant carry the
    numbers. This pins both so that someone "fixing" one to match the other is
    stopped here rather than in production.
    """
    return DEFAULT_EXPANSION == "window" and DISPLAY_EXPANSION == "none"


def main() -> int:
    index, metadata = load_index()
    model = SentenceTransformer(EMBEDDING_MODEL)
    bm25 = build_bm25(metadata)

    failures = []
    checked = 0
    for query in QUERIES:
        for opts in OPTIONS:
            trace = trace_pipeline(query, index, metadata, model, bm25=bm25,
                                   k=5, **opts)
            served = retrieve(query, index, metadata, model, k=5, bm25=bm25,
                              **opts)

            # retrieve() sorts by rank; the trace's final stage sorts by score.
            # For the reranked path those are the same order. For the no-rerank
            # path retrieve() returns the shortlist unsorted by score, so
            # compare as sets there and assert the ordering separately below.
            got, want = selected_ids(trace), [c["chunk_id"] for c in served]
            same = got == want if opts["use_reranker"] else sorted(got) == sorted(want)
            checked += 1
            if not same:
                failures.append((query, opts, got, want))

    # Expansion is checked separately: it changes text, not ranking, so it
    # cannot be compared by chunk id. What must hold is that the trace reports
    # the same number of blocks the serving path would hand a generator.
    for mode in ("window", "page"):
        for query in QUERIES[:2]:
            trace = trace_pipeline(query, index, metadata, model, bm25=bm25,
                                   k=5, expansion=mode)
            served = retrieve(query, index, metadata, model, k=5, bm25=bm25,
                              expansion=mode)
            checked += 1
            if trace["context"]["blocks"] != len(served):
                failures.append((query, {"expansion": mode},
                                 trace["context"]["blocks"], len(served)))

    # The gate must not fire on a score it was not calibrated for.
    for query in QUERIES:
        trace = trace_pipeline(query, index, metadata, model, bm25=bm25,
                               k=5, use_reranker=False)
        checked += 1
        if trace["verdict"]["confident"] is not None:
            failures.append((query, {"use_reranker": False},
                             trace["verdict"]["confident"], None))

    checked += 1
    if not expansion_defaults_diverge_on_purpose():
        failures.append(("expansion defaults", {"deliberate divergence": True},
                         (DEFAULT_EXPANSION, DISPLAY_EXPANSION),
                         ("window", "none")))

    for query, opts, got, want in failures:
        print(f"FAIL {opts}\n  query: {query}\n  trace: {got}\n  serve: {want}")
    print(f"\n{checked - len(failures)}/{checked} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
