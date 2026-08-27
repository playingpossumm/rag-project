"""Does `ask()` answer from the corpus it was handed?

`api.ask()` loads its own index through an `lru_cache` over the process. That is
right for the library entry point and wrong for a server, which can switch
corpora while a process-wide cache cannot. `POST /ask` called it bare, so it
served whichever corpus loaded first: select Ornithology in the interface, ask a
bird question, and the server reported "Ornithology, 45 documents, threshold
-5.5" and answered out of `t5.pdf`.

Nothing caught it because nothing exercised it. The interface talks to
`/api/chat`, which was corpus-aware all along; `/ask` is the documented HTTP API
that HANDOFF §6 lists and the README calls the wrapper around the library, and
no page in the repo calls it.

Stubbed retrieval, not a real corpus: two model loads and a FAISS index would
take half a minute to assert something that is entirely about which arguments
reach which call. The stub records what it was given, which is the whole
question.

    .venv\\Scripts\\python.exe src\\test_api.py
"""
import sys

import api

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHECKS: list[tuple[str, object, object, bool]] = []


def check(name: str, got, want) -> None:
    CHECKS.append((name, got, want, got == want))


# A chunk as the pipeline hands it on: chunk_id included, because context
# expansion looks up neighbours by it and a stub without one exercises a
# shorter path than the real caller takes.
PASSAGE = {
    "source": "birds.pdf", "locator": {"kind": "page", "value": 1},
    "text": "Incubation is the period of sitting on eggs before hatching.",
    "rerank_score": -4.55, "score": 0.4, "chunk_id": 0,
}

CALLS: list[dict] = []


def fake_retrieve(question, index, metadata, model, **kw):
    """Record the corpus it was handed, and answer from it."""
    CALLS.append({"index": index, "metadata": metadata, "model": model, **kw})
    return [dict(PASSAGE, text=f"{index}: {PASSAGE['text']}")]


api.retrieve = fake_retrieve

# A corpus the caller holds, distinguishable from anything api.py could load.
HELD = ("bird-index",
        [dict(PASSAGE, chunk_id=0)],          # metadata, for window expansion
        "bird-model", "bird-bm25")


def last() -> dict:
    return CALLS[-1]


# The defect, reproduced: without `resources`, ask() would reach for its own
# cache. With them, it must use exactly what it was given and never consult it.
answer = api.ask("What is the period of sitting on eggs before hatching called?",
                 resources=HELD, min_confidence=-5.5)
check("the index it was handed is the index it searched", last()["index"], "bird-index")
check("and the BM25 too", last()["bm25"], "bird-bm25")
check("and the embedder", last()["model"], "bird-model")
# With expansion on -- the default -- the text is rebuilt from the metadata,
# so this asserts the EXPANSION read the held corpus too, not just the search.
check("the text handed back is rebuilt from the metadata it was given",
      answer.passages[0].text, PASSAGE["text"])
check("and the citation names that corpus's document",
      answer.passages[0].source, "birds.pdf")
# With expansion off, the stub's own marker survives, which pins down that the
# passage came from the search rather than from anywhere else.
check("with expansion off, the searched index is visible in the passage",
      api.ask("q", resources=HELD, expansion="none",
              min_confidence=-5.5).passages[0].text.startswith("bird-index:"),
      True)

# The other half of what a corpus carries. /ask ignored this entirely, so the
# bird corpus was served at blend 0.00 while shipping 0.20.
api.ask("q", resources=HELD, rerank_blend=0.2)
check("the corpus's rerank blend is forwarded to retrieval",
      last()["rerank_blend"], 0.2)
api.ask("q", resources=HELD)
check("and omitting it forwards None, which means the module default",
      last()["rerank_blend"], None)

# The gate. serve.py hardcoded -2.0 here rather than the corpus's calibrated
# cut, so /ask answered under a looser gate than the interface did.
confident = api.ask("q", resources=HELD, min_confidence=-5.5).confident
refused = api.ask("q", resources=HELD, min_confidence=-3.0).confident
check("a passage above the cut is confident", confident, True)
check("the same passage below a higher cut is not", refused, False)
check("the confidence reported is the reranker's score, not the fused one",
      api.ask("q", resources=HELD, min_confidence=-5.5).confidence, -4.55)

# The library path must be unchanged: no resources means load your own.
seen = {}


def fake_resources():
    seen["called"] = True
    return HELD


api._resources = fake_resources
CALLS.clear()
api.ask("q", min_confidence=-5.5)
check("with no resources given, ask() still loads its own", seen.get("called"), True)
check("and searches what it loaded", last()["index"], "bird-index")


def main() -> int:
    width = max(len(n) for n, *_ in CHECKS)
    failed = 0
    for name, got, want, ok in CHECKS:
        if not ok:
            failed += 1
            print(f"FAIL  {name:<{width}}  got {got!r}, want {want!r}")
    print(f"{len(CHECKS) - failed}/{len(CHECKS)} api checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
