"""Generation against a local model, so the path can actually run.

`generate.py` targets `claude-opus-5` and has never completed a real call: the
account has no credit. Three defects were found in it by reading it against the
data it receives, and `src/test_generate.py` now covers everything up to the
network boundary -- but the boundary itself, the part where a request goes out
and prose comes back, has never executed.

The contract it implements is not Anthropic-specific. Passages in, cited prose
out, a named error when the model declines, a named error when nothing comes
back. So this speaks the same contract to a local **Ollama** server, selected by
`RAG_GENERATOR=ollama`, and the whole path becomes runnable on a machine with no
API credit at all.

**What this is and is not.** It is not a claim that a 7B local model writes
answers as well as Claude does; it will not. It is a way to make the *code* stop
being code that has never run -- prompt assembly, the citation format, the
refusal branch, the empty-response branch, and `ask(generate=True)` rendering
prose in the interface. The model differs; the failure modes exercised are the
real ones.

Ollama is not required to be installed for the tests: `src/test_generate_local.py`
stands up a real HTTP server implementing the two endpoints this uses, so the
request genuinely crosses a socket.

    ollama serve && ollama pull llama3.2
    set RAG_GENERATOR=ollama
    .venv\\Scripts\\python.exe src\\serve.py

Configuration, all with defaults that work for a stock Ollama:

    RAG_OLLAMA_HOST    http://localhost:11434
    RAG_OLLAMA_MODEL   llama3.2
"""
import json
import os
import urllib.error
import urllib.request

from generate import SYSTEM_PROMPT, build_context

HOST = os.environ.get("RAG_OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MODEL = os.environ.get("RAG_OLLAMA_MODEL", "llama3.2")

# Long, because a local model on CPU is slow and a timeout that fires mid-answer
# looks exactly like a broken server to the caller. The interface reports
# generation as a state rather than throwing, so a slow answer degrades to
# "unavailable" with the reason attached rather than to an error page.
TIMEOUT = float(os.environ.get("RAG_OLLAMA_TIMEOUT", "180"))


class LocalModelUnavailable(RuntimeError):
    """Ollama is not running, or the model is not pulled.

    Its own class because this is the one failure a reader can act on -- the
    fix is two shell commands -- and it should not read like the model
    declining to answer.
    """


def available() -> bool:
    """Is there an Ollama to talk to? Used to report state, never to gate."""
    try:
        urllib.request.urlopen(f"{HOST}/api/tags", timeout=3).read()
        return True
    except Exception:                                           # noqa: BLE001
        return False


def synthesize(question: str, chunks: list[dict]) -> str:
    """Write the retrieved passages into one cited answer, locally.

    Deliberately the same signature and the same raising behaviour as
    `generate.synthesize`, so `api.ask` and `serve.chat` can call either without
    knowing which. Same reason it takes passages rather than retrieving its own:
    generating from a second retrieval would let the citations shown and the
    text read drift apart.
    """
    if not chunks:
        raise ValueError("synthesize() needs at least one passage")

    # The user message opens "Excerpts:" rather than "Context:", and the
    # difference is not cosmetic. Measured 2026-08-28 on llama3.2 at
    # temperature 0, same passages, same system prompt, one word changed:
    #
    #   "Context:"   ->  "[36, page 1]"
    #   "Excerpts:"  ->  "According to the text, the label smoothing value
    #                     used during training is (epsilon)ls = 0.1 [36]."
    #
    # Reproducible in both directions on demand. A 3B model appears to read
    # "Context" as the name of a block to be cited and "Excerpts" as material
    # to be read, and collapses to emitting a citation and nothing else. The
    # Anthropic path keeps its own wording and is unaffected: this brittleness
    # belongs to small local models, and so does the workaround.

    payload = {
        "model": MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"Excerpts:\n\n{build_context(chunks)}\n\n"
                        f"Question: {question}"},
        ],
        # Deterministic, because this path is measured. A sampled answer would
        # make two runs of the same question disagree for reasons that have
        # nothing to do with retrieval, which is the one thing this project
        # measures carefully.
        "options": {
            "temperature": 0.0,
            # Ollama defaults to a 4096-token window and TRUNCATES silently
            # past it. Five passages plus the citation instruction can exceed
            # that, and a truncated prompt looks exactly like a model ignoring
            # its context -- which is what the first real run of this path
            # produced: an answer consisting of "[36, page 1]" and nothing else.
            "num_ctx": int(os.environ.get("RAG_OLLAMA_NUM_CTX", "8192")),
        },
    }

    request = urllib.request.Request(
        f"{HOST}/api/chat", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:200]
        if exc.code == 404:
            raise LocalModelUnavailable(
                f"Ollama has no model named {MODEL!r} -- run "
                f"`ollama pull {MODEL}`") from exc
        raise RuntimeError(f"Ollama returned {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LocalModelUnavailable(
            f"no Ollama at {HOST} -- run `ollama serve` "
            f"({exc.reason})") from exc

    text = (body.get("message") or {}).get("content", "").strip()
    if not text:
        # Mirrors generate.py: an empty answer is a failure to report, not a
        # blank answer to render. `done_reason` is Ollama's stop_reason.
        raise RuntimeError(
            f"no text in the response (done_reason="
            f"{body.get('done_reason')!r}). If this is 'length', raise "
            f"RAG_OLLAMA_NUM_PREDICT.")
    return text


def as_chunks(passages) -> list[dict]:
    """`ask()`'s public passages as the dicts `synthesize` reads.

    `ask` returns `Passage` dataclasses to its callers and passes raw retrieval
    dicts to the generator internally, so the two shapes are not
    interchangeable -- `synthesize` calls `.get("locator")` and a dataclass has
    no `.get`. The first version of this function did not exist and this CLI
    wrote `result["passages"]`, which fails twice over: `Answer` is not
    subscriptable either.
    """
    return [{"source": p.source, "locator": p.locator, "text": p.text}
            for p in passages]


def main():
    """Ask the local model one question over the shipped corpus."""
    import sys

    from api import ask

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    question = " ".join(sys.argv[1:]) or "What is attention?"
    if not available():
        print(f"no Ollama at {HOST}. Start it with `ollama serve` and "
              f"`ollama pull {MODEL}`.")
        return 2

    result = ask(question, k=5)
    print(f"\n{question}\n")
    print(synthesize(question, as_chunks(result.passages)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
