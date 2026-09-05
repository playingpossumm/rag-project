"""The request-handling logic in serve.py, without starting a server.

`serve.py` is the largest module in the project and no test imported it. What
exercised it was `src/smoke_routes.py`, which needs a running server and is
therefore not in the counted suite -- so the code between a request arriving and
retrieval starting had no automated coverage at all.

The three things covered here are the three that have actually gone wrong:

  `_text()`         a JSON field of the wrong type used to drop the connection
                    with an unhandled AttributeError. Added 2026-08-27; until
                    now only the live smoke test proved it.
  `examples_for()`  the questions the front page offers. This is the end of the
                    chain that went stale and spent four days suggesting
                    questions deleted from the golden set as unanswerable.
  `inspect_folder()` indexing REPLACES the vector store, so a pasted path that
                    quietly resolves to nothing would destroy a working index
                    and report success.

Importing serve costs about 25 seconds because it pulls in the embedding stack
at module level. It does not start a server, load an index or touch a model.

    .venv\\Scripts\\python.exe src\\test_serve.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import serve

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHECKS: list[tuple[str, object, object, bool]] = []


def check(name: str, got, want) -> None:
    CHECKS.append((name, got, want, got == want))


class FakeHandler:
    """Enough of a handler for _text: it only ever calls self._send."""

    def __init__(self):
        self.sent = []

    def _send(self, status, body):
        self.sent.append((status, body))


def text_field(payload, field):
    h = FakeHandler()
    value = serve.Handler._text(h, payload, field)
    return value, h.sent


# ---------------------------------------------------------- _text() --------
value, sent = text_field({"question": "  what is a syrinx  "}, "question")
check("a string field comes back stripped", value, "what is a syrinx")
check("and nothing is sent", sent, [])

value, sent = text_field({}, "question")
check("a missing field reads as empty rather than None", value, "")
check("and still nothing is sent", sent, [])

# The defect: JSON carries types, and .strip() on a dict raised AttributeError
# outside the try that catches everything else, so the socket closed.
for bad, kind in (({"not": "text"}, "dict"), ([1, 2], "list"), (7, "int")):
    value, sent = text_field({"question": bad}, "question")
    check(f"a {kind} field returns None instead of raising", value, None)
    check(f"and a {kind} field answers 400", sent and sent[0][0], 400)
    check(f"and the 400 names the field and the type it got",
          sent and "question" in sent[0][1]["error"] and kind in sent[0][1]["error"],
          True)

# ----------------------------------------------------- examples_for() ------
tmp = Path(tempfile.mkdtemp(prefix="serve-"))
real_eval = serve.EVAL_DIR
try:
    serve.EVAL_DIR = tmp
    (tmp / "analytics.json").write_text(json.dumps({"corpora": [
        {"name": "birds", "examples": [
            {"q": "What organ produces song?", "label": "answered in one place"}]},
        {"name": "empty", "examples": []},
    ]}), encoding="utf-8")

    check("a corpus with questions gets its own",
          [e["q"] for e in serve.examples_for("birds")],
          ["What organ produces song?"])

    # The failure this guards: switching corpus used to keep offering the ML
    # papers' questions, so a bird corpus suggested asking about BLEU on WMT.
    check("a corpus analytics.json does not know falls back",
          serve.examples_for("nosuch"), serve.EXAMPLES)
    check("a corpus with an empty list falls back rather than offering nothing",
          serve.examples_for("empty"), serve.EXAMPLES)
    check("no corpus at all falls back", serve.examples_for(None), serve.EXAMPLES)

    (tmp / "analytics.json").write_text("{ this is not json", encoding="utf-8")
    check("a corrupt analytics.json falls back instead of raising",
          serve.examples_for("birds"), serve.EXAMPLES)

    (tmp / "analytics.json").unlink()
    check("a missing analytics.json falls back",
          serve.examples_for("birds"), serve.EXAMPLES)
finally:
    serve.EVAL_DIR = real_eval
    shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------- inspect_folder() -----
# Indexing REPLACES the vector store. A path that quietly resolves to an empty
# directory would destroy a working index and report success, so each of these
# must raise with a message meant for a person.
tmp = Path(tempfile.mkdtemp(prefix="folder-"))
try:
    def refuses(raw):
        try:
            serve.inspect_folder(raw)
            return "accepted"
        except ValueError as exc:
            return str(exc)
        except Exception as exc:                                # noqa: BLE001
            return f"{type(exc).__name__}: {exc}"

    check("an empty string is refused", "no folder given" in refuses(""), True)
    check("a path that does not exist is refused",
          "no such folder" in refuses(str(tmp / "nope")), True)

    a_file = tmp / "notes.pdf"
    a_file.write_bytes(b"%PDF-1.4 not really")
    check("a file is refused, and says it is a file",
          "not a folder" in refuses(str(a_file)), True)

    empty = tmp / "empty"
    empty.mkdir()
    check("a folder with nothing indexable is refused",
          "no indexable" in refuses(str(empty)).lower()
          or "holds no indexable" in refuses(str(empty)), True)

    # Quoted paths are what a Windows right-click "copy as path" produces, and
    # the field takes a pasted string.
    good = tmp / "docs"
    good.mkdir()
    (good / "paper.pdf").write_bytes(b"%PDF-1.4 not really")
    folder, found, skipped = serve.inspect_folder(f'"{good}"')
    check("a quoted path is accepted", folder, good)
    check("and the indexable file is found", found, ["paper.pdf"])
finally:
    shutil.rmtree(tmp, ignore_errors=True)


# ---- the warm-up runs through the loaded corpus, and never fails startup ---
# Until 2026-09-06 main() called ask("warmup") bare, which opens retrieve's
# default store rather than RES's: a second index locally, and in the Docker
# image, which bakes in the ornithology corpus only, a missing directory and a
# server that died before it bound the port.
_ask, _res = serve.ask, serve.RES
try:
    calls = []

    class _Loaded:
        index, metadata, model, bm25 = "I", "M", "E", "B"

        def rerank_blend(self):
            return 0.2

    serve.RES = _Loaded()
    serve.ask = lambda q, **kw: calls.append(kw)
    ran = serve.warm_up()
    check("the warm-up runs through the loaded corpus, not a second index",
          calls[0].get("resources") if calls else None, ("I", "M", "E", "B"))
    check("and carries that corpus's rerank blend",
          calls[0].get("rerank_blend") if calls else None, 0.2)
    check("and reports that it ran", ran, True)

    def _missing(q, **kw):
        raise FileNotFoundError("vector_store/index.faiss")

    serve.ask = _missing
    check("a warm-up that cannot run does not stop the server coming up",
          serve.warm_up(), False)
finally:
    serve.ask, serve.RES = _ask, _res


def main() -> int:
    width = max(len(n) for n, *_ in CHECKS)
    failed = 0
    for name, got, want, ok in CHECKS:
        if not ok:
            failed += 1
            print(f"FAIL  {name:<{width}}  got {got!r}, want {want!r}")
    print(f"{len(CHECKS) - failed}/{len(CHECKS)} serve checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
