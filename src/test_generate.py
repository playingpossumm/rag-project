"""Everything `generate.py` does up to the network boundary.

HANDOFF §5b says: "What is now tested is everything up to the network boundary
-- `src/test_trace.py`'s sibling check in the session log stubbed the client and
confirmed the model is handed exactly the passages the answer cites."

No such test existed. The check was run in a session and never committed, which
makes this the **third** claimed-but-absent test in this repo, after
`test_trace_matches_pipeline` and the four hand-computed NDCG cases. Both of
those were found the same way and both were found to matter.

It matters more here than usual. `generate.py` has never completed a real call --
the account has no credit -- so *nothing* has executed this module end to end.
Three defects were already found in it by reading it against the data it
receives, and code that has never run is not code that works. Stubbing the
client tests the only part that can be tested, which is all of it except the
HTTP request.

    .venv\\Scripts\\python.exe src\\test_generate.py
"""
import sys
import types

import generate

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHECKS: list[tuple[str, object, object, bool]] = []


def check(name: str, got, want) -> None:
    CHECKS.append((name, got, want, got == want))


# ---------------------------------------------------------------- stubs ----
class Block:
    def __init__(self, text, kind="text"):
        self.text, self.type = text, kind


class Response:
    def __init__(self, blocks, stop_reason="end_turn", explanation=None):
        self.content = blocks
        self.stop_reason = stop_reason
        self.stop_details = types.SimpleNamespace(explanation=explanation)


class StubClient:
    """Records the call rather than making it."""

    def __init__(self, response):
        self.response = response
        self.calls = []
        self.messages = types.SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def install(response):
    client = StubClient(response)
    generate.anthropic = types.SimpleNamespace(Anthropic=lambda *a, **kw: client)
    return client


# Chunks in the shape retrieve.py really returns: a locator, not a page number.
CHUNKS = [
    {"source": "attention.pdf", "locator": {"kind": "page", "value": 3},
     "text": "The model uses scaled dot-product attention."},
    {"source": "deck.pptx", "locator": {"kind": "slide", "value": 7},
     "text": "Multi-head attention runs several projections in parallel."},
]


# ---- the citation format, which is where the module's worst bug lived ----
# build_context() read chunk["page"]. Chunks carry `locator` and have since
# locators replaced page numbers, so every call would have died on KeyError --
# in a module nothing had ever run.
check("a PDF chunk cites its page",
      generate.cite(CHUNKS[0]), "attention.pdf, page 3")
check("a slide cites a slide, not a page",
      generate.cite(CHUNKS[1]), "deck.pptx, slide 7")
check("a chunk with no locator does not raise",
      generate.cite({"source": "x.pdf"}), "x.pdf, location ?")

context = generate.build_context(CHUNKS)
check("the context carries every passage's text",
      all(c["text"] in context for c in CHUNKS), True)
check("and every passage's citation",
      all(generate.cite(c) in context for c in CHUNKS), True)

# ---- the claim HANDOFF makes: the model reads exactly what is cited ----
client = install(Response([Block("Attention weights queries against keys.")]))
answer = generate.synthesize("How does attention work?", CHUNKS)
sent = client.calls[0]
prompt = sent["messages"][0]["content"]

check("synthesize returns the model's text", answer,
      "Attention weights queries against keys.")
check("the model is handed exactly the passages it will cite",
      all(c["text"] in prompt for c in CHUNKS), True)
check("and nothing it was not given",
      "hollow bones" in prompt, False)
check("the question reaches the model", "How does attention work?" in prompt, True)

# ---- the settings the module argues for in its own comments ----
check("the documented model is the one called",
      sent["model"], generate.CLAUDE_MODEL)
check("max_tokens is the ceiling the comment justifies, not the old 1024",
      sent["max_tokens"], 16000)
check("effort is what controls cost here",
      sent["output_config"], {"effort": "low"})
check("the system prompt forbids answering outside the context",
      "ONLY the provided context" in sent["system"], True)

# ---- refusal, which returns HTTP 200 and no text ----
# Checked before the content is read: next() over an empty generator raises
# StopIteration, which in a caller reads as a bug here rather than a decision
# by the model.
install(Response([], stop_reason="refusal", explanation="not answerable safely"))
try:
    generate.synthesize("Something refused", CHUNKS)
    refusal = "no error raised"
except RuntimeError as exc:
    refusal = "declined" if "declined" in str(exc) else str(exc)
except StopIteration:
    refusal = "StopIteration -- the refusal check ran too late"
check("a refusal raises a named error, not StopIteration", refusal, "declined")
install(Response([], stop_reason="refusal", explanation="not answerable safely"))
try:
    generate.synthesize("Something refused", CHUNKS)
    detail = ""
except RuntimeError as exc:
    detail = str(exc)
check("the refusal explanation is passed through",
      "not answerable safely" in detail, True)

# ---- an empty response is a failure, not a blank answer ----
install(Response([], stop_reason="max_tokens"))
try:
    generate.synthesize("Something long", CHUNKS)
    empty = "no error raised"
except RuntimeError as exc:
    empty = str(exc)
check("an empty response raises rather than returning ''",
      "no text in the response" in empty, True)
check("and the error names the stop_reason, which is the thing to act on",
      "max_tokens" in empty, True)

# ---- no passages is the caller's error, and a different one ----
try:
    generate.synthesize("A question", [])
    nothing = "no error raised"
except ValueError:
    nothing = "ValueError"
except Exception as exc:                                        # noqa: BLE001
    nothing = type(exc).__name__
check("generating from no passages is a ValueError, not a request",
      nothing, "ValueError")

# Only text blocks are read, so a thinking block cannot leak into the answer.
install(Response([Block("internal reasoning", kind="thinking"),
                  Block("The answer.")]))
check("a non-text block is not part of the answer",
      generate.synthesize("q", CHUNKS), "The answer.")


def main() -> int:
    width = max(len(n) for n, *_ in CHECKS)
    failed = 0
    for name, got, want, ok in CHECKS:
        if not ok:
            failed += 1
            print(f"FAIL  {name:<{width}}  got {got!r}, want {want!r}")
    print(f"{len(CHECKS) - failed}/{len(CHECKS)} generate checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
