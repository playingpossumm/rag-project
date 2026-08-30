"""Rewrite and decompose ambiguous questions before retrieving. Needs an LLM.

This is the measured answer to the one failure the retriever cannot reach on its
own. `failure_overlap.py` isolated seven questions that fail under *every*
combination of fusion, reranking and diversity cap, and four of them share a
shape: the query names a topic and a constraint, retrieval matches the topic,
and the constraint -- the part that decides which document is right -- is
averaged away.

    "normalisation across features RATHER THAN examples"  -> Batch Normalization
    "the LARGEST autoregressive model's parameters"       -> whichever paper
                                                             mentions parameters

No ranking change fixes this, and that is not a guess: an 8x larger reranker
fixed the same 2 of 10 cases at 10 s/query, and title prefixing measured worse.
The information the retriever needs is not in the ranking, it is in the
*question*, and getting it out means understanding the question. That needs a
model.

Two strategies, because the seven cases are not all one shape:

  rewrite     one query in, one better query out. Resolves pronouns, expands
              acronyms, makes an implicit constraint explicit.
  decompose   one query in, several out, retrieved separately and fused. For
              questions that are really two questions -- "parameter count and
              dropout rate" lives in two places, so no single passage answers
              it and no ranking of single passages ever will.

**This has never run.** The account has no credit, so everything here is written
against the API contract and unverified in exactly the way `generate.py` was --
which is how three defects got into that file. It is built to fail safely: every
entry point returns the original query unchanged if anything goes wrong, so
turning this on can degrade to today's behaviour but cannot break retrieval.
Score it against the seven with `src/hard_cases.py` the moment there is credit.
"""
import json
import os
import sys

from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

CLAUDE_MODEL = "claude-opus-5"

# Small ceiling and minimal effort deliberately: this is a short structural
# transformation of one sentence, not analysis. The expensive thinking belongs
# in the answer, and this call sits on the latency path of every query.
MAX_TOKENS = 2000
EFFORT = "low"

REWRITE_SYSTEM = (
    "You rewrite search queries for a retrieval system over technical papers. "
    "Return ONE rewritten query and nothing else -- no preamble, no explanation.\n"
    "Make implicit constraints explicit, expand ambiguous acronyms, and keep "
    "every distinguishing term from the original: the words that rule an answer "
    "OUT are the ones the retriever most often loses. Never invent specifics "
    "the question does not contain. If the query is already unambiguous, return "
    "it unchanged."
)

DECOMPOSE_SYSTEM = (
    "You split search queries into independently answerable sub-queries for a "
    "retrieval system over technical papers.\n"
    "Return a JSON array of strings and nothing else. Return ONE element if the "
    "question asks for a single fact. Return several only when the question asks "
    "for facts that would live in different places -- and then make each element "
    "a complete, standalone question carrying enough context to be retrieved on "
    "its own."
)


def _client():
    """None when no key is configured, so callers degrade instead of raising."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    import anthropic

    return anthropic.Anthropic()


def _ask_ollama(system: str, user: str) -> str | None:
    """The same question, to a model running on this machine.

    Deterministic, because this path gets measured: a sampled rewrite would
    make two runs of the same question disagree for reasons that have nothing
    to do with retrieval.
    """
    import json as _json
    import urllib.error
    import urllib.request

    host = os.environ.get("RAG_OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    model = os.environ.get("RAG_OLLAMA_MODEL", "llama3.2")
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "options": {"temperature": 0.0, "seed": 0, "num_predict": MAX_TOKENS},
    }
    req = urllib.request.Request(
        f"{host}/api/chat", method="POST",
        data=_json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            body = _json.loads(r.read().decode("utf-8"))
    except Exception as exc:                                    # noqa: BLE001
        # Same contract as the Anthropic path: a rewrite is an optimisation,
        # so losing it costs the improvement and never the query.
        print(f"  query rewrite unavailable ({type(exc).__name__}); "
              f"using the original query", file=sys.stderr)
        return None
    return ((body.get("message") or {}).get("content") or "").strip() or None


def _backend() -> str:
    """Which model answers. Shares RAG_GENERATOR with the generation path, so
    one variable turns both on."""
    return os.environ.get("RAG_GENERATOR", "").strip().lower()


def available() -> bool:
    """Is a model reachable? Reports state; never gates."""
    if _backend() in ("ollama", "local"):
        import urllib.request
        host = os.environ.get("RAG_OLLAMA_HOST",
                              "http://localhost:11434").rstrip("/")
        try:
            urllib.request.urlopen(f"{host}/api/tags", timeout=3).read()
            return True
        except Exception:                                       # noqa: BLE001
            return False
    return _client() is not None


def _ask(system: str, user: str) -> str | None:
    if _backend() in ("ollama", "local"):
        return _ask_ollama(system, user)
    client = _client()
    if client is None:
        return None
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            output_config={"effort": EFFORT},
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return next((b.text for b in response.content if b.type == "text"), None)
    except Exception as exc:  # noqa: BLE001
        # A rewrite is an optimisation. Losing it should cost the improvement,
        # never the query -- so no API failure is allowed to reach the caller.
        print(f"  query rewrite unavailable ({type(exc).__name__}); "
              f"using the original query", file=sys.stderr)
        return None


def rewrite(query: str) -> str:
    """A clearer version of `query`, or `query` itself if unavailable."""
    out = _ask(REWRITE_SYSTEM, query)
    if not out:
        return query
    out = out.strip().strip('"')
    # A rewrite that returns a paragraph has misunderstood the task; a rewrite
    # that returns nothing has failed. Both fall back rather than propagate.
    if not out or len(out) > 4 * len(query) + 200:
        return query
    return out


def decompose(query: str) -> list[str]:
    """Sub-queries to retrieve separately. Always at least [query]."""
    out = _ask(DECOMPOSE_SYSTEM, query)
    if not out:
        return [query]
    try:
        parts = json.loads(out[out.index("["):out.rindex("]") + 1])
    except (ValueError, json.JSONDecodeError):
        return [query]
    parts = [p.strip() for p in parts
             if isinstance(p, str) and p.strip()]
    return parts or [query]


def retrieve_decomposed(query: str, index, metadata, model, k: int = 5,
                        bm25=None, rrf_k: int = 60, **kw) -> list[dict]:
    """Decompose, retrieve each sub-query, and fuse the rankings by RRF.

    RRF rather than concatenation because the sub-queries return overlapping
    passages on incomparable score scales -- the same reason fusion uses it for
    dense and sparse. A passage found by two sub-queries should outrank one
    found strongly by a single one, which is precisely what a compound question
    needs.
    """
    from retrieve import retrieve

    subs = decompose(query)
    if len(subs) == 1:
        return retrieve(subs[0], index, metadata, model, k=k, bm25=bm25, **kw)

    fused: dict[int, dict] = {}
    for sub in subs:
        ranked = retrieve(sub, index, metadata, model, k=k, bm25=bm25, **kw)
        for rank, r in enumerate(ranked, start=1):
            entry = fused.setdefault(r["chunk_id"], {**r, "fusion_score": 0.0,
                                                    "found_by": []})
            entry["fusion_score"] += 1.0 / (rrf_k + rank)
            entry["found_by"].append(sub)

    out = sorted(fused.values(), key=lambda c: c["fusion_score"], reverse=True)
    return out[:k]


def main() -> int:
    query = " ".join(sys.argv[1:]) or input("Query: ")
    if _client() is None:
        print("No ANTHROPIC_API_KEY set -- both strategies return the input "
              "unchanged, which is the designed fallback.\n")
    print(f"  original   {query}")
    print(f"  rewritten  {rewrite(query)}")
    print(f"  decomposed {decompose(query)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
