"""Optional web search when the corpus cannot answer. OFF, and loudly so.

The gate already knows when retrieval has failed: below the calibrated
threshold the honest response is "the corpus does not contain this". This module
is the alternative answer to that moment -- go and look outside.

**It contradicts a constraint this project has held throughout, and the
contradiction is the point of this docstring rather than something to bury.**
The UI bundles its own fonts specifically so that a page about private documents
makes no external request; the server binds to loopback so a corpus of contracts
is not reachable by accident. Sending a user's question to a third party is a
larger disclosure than either of those guarded against: the question itself
often carries the sensitive part ("what is our termination clause with X").

So the design is opt-in at every level and cannot be enabled by accident:

  * `enabled()` is False unless RAG_WEB_FALLBACK=1 is explicitly set.
  * With no provider configured it is False regardless.
  * It runs only after the gate has already declined -- never as a first resort,
    never in parallel, so a confident local answer never leaves the machine.
  * Results are returned as clearly-marked WEB passages with their URL as the
    locator, never blended into corpus citations. A user must be able to see
    which claims came from their documents and which did not, because those
    have completely different trust.

**No provider is wired.** `search()` raises unless one is registered, because a
plausible-looking stub that silently returns nothing is how a feature gets
believed in without ever working -- which this repo has now found three times in
its own documentation. Register one explicitly:

    from web_fallback import register
    register(lambda q, n: [{"title": ..., "url": ..., "snippet": ...}, ...])

The account's MCP tooling can supply one (Monid exposes search endpoints), and
those calls are billed per run, which is the other reason this is not switched
on by default.
"""
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_PROVIDER = None


def register(fn) -> None:
    """Install the search backend. `fn(query, n) -> list[{title,url,snippet}]`."""
    global _PROVIDER
    _PROVIDER = fn


def configured() -> bool:
    return _PROVIDER is not None


def enabled() -> bool:
    """Both switches, because either alone is an accident waiting to happen."""
    return os.environ.get("RAG_WEB_FALLBACK") == "1" and configured()


def search(query: str, n: int = 5) -> list[dict]:
    if _PROVIDER is None:
        raise RuntimeError(
            "no web search provider registered -- call web_fallback.register(fn). "
            "This raises rather than returning [] so an unconfigured fallback "
            "cannot be mistaken for one that found nothing.")
    return _PROVIDER(query, n)


def as_passages(results: list[dict]) -> list[dict]:
    """Shape web results like corpus passages, marked so they cannot be confused.

    `origin: "web"` and a url-kind locator, so any renderer that groups by
    source or prints a citation shows these as external without needing to know
    this module exists. The score is None rather than 0.0: these were never
    scored by the reranker, and 0.0 would compare as "at the threshold" in
    anything that reads it.
    """
    return [
        {
            "source": r.get("url") or r.get("title") or "web result",
            "title": r.get("title") or "",
            "locator": {"kind": "url", "value": r.get("url") or ""},
            "text": r.get("snippet") or "",
            "score": None,
            "origin": "web",
        }
        for r in results
    ]


def maybe_fallback(query: str, confident: bool, n: int = 5) -> list[dict]:
    """Web passages when the corpus declined and every switch is on, else [].

    The `confident` flag is passed in rather than recomputed so this module
    never needs to know how the gate works -- it only needs to know it fired.
    """
    if confident or not enabled():
        return []
    try:
        return as_passages(search(query, n))
    except Exception as exc:  # noqa: BLE001
        print(f"  web fallback failed ({type(exc).__name__}: {exc})", file=sys.stderr)
        return []


def status() -> str:
    if not configured():
        return "web fallback: no provider registered (off)"
    if os.environ.get("RAG_WEB_FALLBACK") != "1":
        return "web fallback: provider registered, RAG_WEB_FALLBACK not set (off)"
    return "web fallback: ON -- questions will be sent to a third party"
