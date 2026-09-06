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

The sections added on 2026-09-06 run whole routes without a socket, through a
Handler subclass whose constructor takes a path and a body instead of a
connection, with RES and the pipeline replaced by stubs that record what they
were handed. They cover the audit's findings of that date: a body that is not
a JSON object dropped the connection; /api/trace and /api/chat ran different
pipelines; /ask held the lock across generation; a reindex rebuilt the wrong
store and was open to any visitor on a public server; error bodies, /api/corpus
and inspect_folder echoed absolute paths; /api/eval served the ML papers'
figures under every corpus; and startup contacted huggingface.co.

Each of those sections is wrapped so that a missing attribute on an older
serve.py records a failed check rather than stopping the file, which is what
lets the suite be run against the code before the fix to prove the checks bite.

Importing serve costs about 25 seconds because it pulls in the embedding stack
at module level. It does not start a server, load an index or touch a model.

    .venv\\Scripts\\python.exe src\\test_serve.py
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

import serve
from api import DEFAULT_EXPANSION, Answer

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


# --------------------------------------------------------- warm_up() ------
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


# ----------------------------------------------- routes without a socket ---
# Handler's own __init__ is BaseHTTPRequestHandler's, which reads from a
# connection and dispatches. This one takes the request as arguments and
# captures what the route sends, so a route runs end to end in-process.
class Request(serve.Handler):
    def __init__(self, method: str, path: str, body: bytes = b"",
                 headers: dict | None = None):
        self.path = path
        self.headers = {"Content-Length": str(len(body)), **(headers or {})}
        self.rfile = io.BytesIO(body)
        self.sent: list[tuple] = []
        (self.do_POST if method == "POST" else self.do_GET)()

    def _send(self, status, payload):
        self.sent.append((status, payload))

    def _raw(self, status, body, content_type):
        self.sent.append((status, body))

    @property
    def status(self):
        return self.sent[0][0] if self.sent else None

    @property
    def body(self):
        return self.sent[0][1] if self.sent else None


class Lock:
    """A lock that records whether it is held, for asserting what runs inside it."""

    def __init__(self):
        self.held = False

    def __enter__(self):
        self.held = True

    def __exit__(self, *_):
        self.held = False


class FakeRes:
    """RES as the routes see it, with paths that must never reach a response."""
    index, model, bm25, ensemble = "I", "E", "B", "ENS"
    metadata = [{"source": "owl.pdf", "locator": {"kind": "page", "value": 1},
                 "text": "The syrinx produces song."}]
    corpus = {"name": "birds", "label": "Ornithology",
              "data": "C:/private/data-birds", "store": "C:/private/store-birds",
              "golden": str(serve.EVAL_DIR / "golden-birds.json")}

    def __init__(self):
        self.lock = Lock()
        self.reloaded = False

    def threshold(self):
        return -5.5

    def rerank_blend(self):
        return 0.2

    def candidate_k(self):
        return 16

    def stats(self):
        return {"chunks": 1, "documents": 1, "sources": ["owl.pdf"], "titles": {}}

    def reload(self):
        self.reloaded = True


@contextlib.contextmanager
def env(**values):
    """Set environment variables for a block; None removes one."""
    saved = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def section(name: str):
    """Turn an exception in a block into one failed check, not a crash.

    The point is running this file against serve.py before a fix: an
    attribute the fix introduced is then missing, and the section it belongs
    to should fail visibly and let the rest of the file run.
    """
    try:
        yield
    except Exception as exc:                                    # noqa: BLE001
        check(f"{name} runs without raising", f"{type(exc).__name__}: {exc}",
              "no exception")


@contextlib.contextmanager
def fake_module(name: str, module):
    real = sys.modules.get(name)
    sys.modules[name] = module
    try:
        yield
    finally:
        if real is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = real


PUBLIC = dict(RAG_PUBLIC="1", RAG_ALLOW_GENERATE=None, RAG_ALLOW_REINDEX=None)
PRIVATE = dict(RAG_PUBLIC=None, RAG_ALLOW_GENERATE=None, RAG_ALLOW_REINDEX=None)

# --------------------------------------------- _body(): malformed shapes ---
# Every caller reads None from _body() as "a 4xx was already sent". Until
# 2026-09-06 three shapes produced None, or an exception, with nothing sent,
# so the client waited on a connection the server had silently given up on.
# A refusal sent before the body is read must also close the connection,
# because on HTTP/1.1 keep-alive the unread body would be parsed as the next
# request; until 2026-09-07 it did not.
with section("malformed bodies"), env(**PRIVATE):
    r = Request("POST", "/ask", b'{"question": "x"}', {"Content-Length": "abc"})
    check("a Content-Length that is not a number answers 400", r.status, 400)
    check("and says what was wrong with it",
          "Content-Length" in (r.body or {}).get("error", ""), True)
    check("and closes the connection, since the body was never read",
          getattr(r, "close_connection", False), True)
    r = Request("POST", "/ask", b"{}", {"Content-Length": str(serve.MAX_BODY + 1)})
    check("a body over MAX_BODY answers 413 and closes the connection",
          (r.status, getattr(r, "close_connection", False)), (413, True))

    r = Request("POST", "/ask", b"null")
    check("a body of null answers 400 rather than nothing", r.status, 400)
    check("and names the shape it wanted",
          (r.body or {}).get("error"), "body must be a JSON object")

    r = Request("POST", "/ask", b'["question"]')
    check("an array body answers 400 rather than failing on .get()", r.status, 400)

    r = Request("POST", "/api/chat", b"1")
    check("a bare number answers 400 on /api/chat too", r.status, 400)

# --------------------------- /api/trace and /api/chat share one pipeline ---
# Until 2026-09-06 /api/trace built its own call without the corpus's second
# dense retriever or its candidate pool, so on the quant corpus the inspector
# and the chat surface answered the same question from different searches.
with section("shared trace"), env(**PRIVATE):
    import pipeline_trace

    _real_tp, _real_attr, _res = pipeline_trace.trace_pipeline, serve.attribution_for, serve.RES
    seen: list[dict] = []

    def fake_trace(question, index, metadata, model, **kw):
        seen.append(kw)
        return {"stages": [{"name": "reranked",
                            "items": [{"chunk_id": 0, "score": 1.0}]}]}

    try:
        pipeline_trace.trace_pipeline = fake_trace
        serve.attribution_for = lambda q, t: {}
        serve.RES = FakeRes()
        r = Request("POST", "/api/trace", b'{"question": "what sings"}')
        check("/api/trace answers 200 through the shared helper", r.status, 200)
        trace_kw = seen[-1] if seen else {}
        check("/api/trace searches the corpus's second dense index",
              trace_kw.get("ensemble"), "ENS")
        check("and pools the corpus's candidate_k, not the module default",
              trace_kw.get("candidate_k"), 16)
        check("and applies the corpus's threshold and blend",
              (trace_kw.get("threshold"), trace_kw.get("rerank_blend")), (-5.5, 0.2))
        check("and puts the structured locator back on each traced item",
              r.body["stages"][0]["items"][0].get("loc"), {"kind": "page", "value": 1})

        r = Request("POST", "/api/chat", b'{"question": "what sings"}')
        check("/api/chat answers 200", r.status, 200)
        check("/api/chat and /api/trace hand the pipeline the same arguments",
              seen[-1] if len(seen) >= 2 else None, trace_kw)
        check("and /api/chat names the corpus it searched", r.body.get("corpus"), "birds")

        r = Request("POST", "/api/trace", b'{"question": "x", "fusion": "magic"}')
        check("an unknown option is a 400 that names it",
              (r.status, "fusion" in r.body.get("error", "")), (400, True))
    finally:
        pipeline_trace.trace_pipeline, serve.attribution_for, serve.RES = _real_tp, _real_attr, _res

# -------------------------------------------------- /ask generation gate ---
# Until 2026-09-06 /ask passed generate= into ask() inside RES.lock, so one
# visitor's model call stalled every other request for as long as the model
# took, and any visitor to a public server could start that call.
with section("/ask generation"):
    _ask, _gen, _res = serve.ask, serve.generate_answer, serve.RES
    asked: list[dict] = []
    generated: list[dict] = []

    def fake_ask(question, **kw):
        asked.append({**kw, "lock_held": serve.RES.lock.held})
        return Answer(question=question, mode="retrieval", confident=True,
                      confidence=-4.5, documents=["owl.pdf"])

    def fake_generate(answer):
        generated.append({"lock_held": serve.RES.lock.held})
        answer.answer, answer.mode = "prose", "generated"
        return answer

    try:
        serve.ask, serve.generate_answer, serve.RES = fake_ask, fake_generate, FakeRes()
        with env(**PRIVATE):
            r = Request("POST", "/ask", b'{"question": "what sings", "generate": true}')
            check("/ask answers 200 with generation on a private server", r.status, 200)
            check("retrieval ran under the lock", asked[-1]["lock_held"], True)
            check("generation ran after the lock was released",
                  generated[-1]["lock_held"] if generated else None, False)
            check("and the generated prose is in the response",
                  (r.body.get("mode"), r.body.get("answer")), ("generated", "prose"))
            check("retrieval reads the corpus's calibrated threshold",
                  asked[-1].get("min_confidence"), -5.5)
            check("and the corpus's rerank blend", asked[-1].get("rerank_blend"), 0.2)
            check("and the shipped expansion default, not a restated one",
                  asked[-1].get("expansion"), DEFAULT_EXPANSION)

            Request("POST", "/ask", b'{"question": "x", "expansion": "none"}')
            check("an explicit expansion is forwarded", asked[-1].get("expansion"), "none")

        generated.clear()
        with env(**PUBLIC):
            r = Request("POST", "/ask", b'{"question": "what sings", "generate": true}')
            check("a public server without RAG_ALLOW_GENERATE does not generate",
                  generated, [])
            check("but still answers 200 with the passages", r.status, 200)
            check("and says why there is no prose",
                  any("RAG_ALLOW_GENERATE" in n for n in r.body.get("notes", [])), True)

        with env(**{**PUBLIC, "RAG_ALLOW_GENERATE": "1"}):
            Request("POST", "/ask", b'{"question": "what sings", "generate": true}')
            check("RAG_ALLOW_GENERATE=1 turns generation back on", len(generated), 1)
    finally:
        serve.ask, serve.generate_answer, serve.RES = _ask, _gen, _res

# ------------------------------------------- reindex target and gate ---
# Until 2026-09-06 a reindex passed no store_dir, so it rebuilt ingest's
# default vector_store/ and then reloaded the active corpus's untouched store.
with section("reindex target"):
    _res = serve.RES
    built: list[tuple] = []
    ingest = types.ModuleType("ingest")
    ingest.DATA_DIR, ingest.STORE_DIR = Path("data"), Path("vector_store")

    def build_index(data_dir, store_dir=None, progress=None):
        built.append((str(data_dir), str(store_dir)))
        return {"chunks": 1}

    ingest.build_index = build_index
    try:
        serve.RES = FakeRes()
        with fake_module("ingest", ingest):
            run = serve.IndexRun()
            run._run(None)
        check("a reindex with no path reads the active corpus's documents",
              built[-1][0] if built else None, "C:/private/data-birds")
        check("and writes the active corpus's store, not ingest's default",
              built[-1][1] if built else None, "C:/private/store-birds")
        check("and reloads the server's copy afterwards",
              (run.state, serve.RES.reloaded), ("done", True))
    finally:
        serve.RES = _res

with section("reindex gate"):
    with env(**PRIVATE):
        check("a private server allows a reindex", serve.reindex_allowed(), True)
    with env(**PUBLIC):
        check("a public server refuses one by default", serve.reindex_allowed(), False)
        # The body is not valid JSON on purpose: the refusal must come before
        # the body is read, so a visitor cannot make the server parse anything.
        r = Request("POST", "/api/index/start", b"not json")
        check("/api/index/start answers 403 when public", r.status, 403)
        r = Request("POST", "/api/index/inspect", b'{"path": "C:/"}')
        check("/api/index/inspect answers 403 when public", r.status, 403)
        check("and the 403 names the variable that enables it",
              "RAG_ALLOW_REINDEX" in r.body.get("error", ""), True)
    with env(**{**PUBLIC, "RAG_ALLOW_REINDEX": "1"}):
        check("RAG_ALLOW_REINDEX=1 turns it back on", serve.reindex_allowed(), True)

# ---------------------------------------------------- path disclosure ---
# Until 2026-09-06 error bodies carried the exception text verbatim,
# /api/corpus reported ingest's default data directory as an absolute path,
# and inspect_folder's messages echoed the resolved path. Until 2026-09-07
# the attribution block and the generation report inside a 200 /api/chat
# body still carried the exception text on a public server.
with section("path disclosure"):
    _res, _info = serve.RES, serve.corpus_info
    try:
        serve.RES, serve.corpus_info = FakeRes(), lambda: {}
        boom = RuntimeError("C:/private/store-birds/index.faiss is missing")
        with env(**PUBLIC):
            h = Request("GET", "/health")
            h.sent.clear()
            serve.Handler._fail(h, boom)
            check("a public 500 does not carry the exception text",
                  (h.status, "private" in json.dumps(h.body)), (500, False))
            r = Request("GET", "/api/corpus")
            check("/api/corpus reports the corpus label when public",
                  r.body.get("data_dir"), "Ornithology")
        with env(**PRIVATE):
            h = Request("GET", "/health")
            h.sent.clear()
            serve.Handler._fail(h, boom)
            check("a private 500 names the exception for the operator",
                  "RuntimeError" in h.body.get("error", ""), True)
            r = Request("GET", "/api/corpus")
            check("/api/corpus reports the active corpus's folder when private",
                  r.body.get("data_dir"), "C:/private/data-birds")
    finally:
        serve.RES, serve.corpus_info = _res, _info

with section("chat body disclosure"):
    import pipeline_trace
    _real_tp, _real_attr, _res = pipeline_trace.trace_pipeline, serve.attribution_for, serve.RES

    def fake_trace(question, index, metadata, model, **kw):
        return {"stages": [{"name": "reranked",
                            "items": [{"chunk_id": 0, "locator": "page 1"}]}]}

    class NoAttribution(types.ModuleType):
        def attribute(self, question, bm25, ids):
            raise RuntimeError("C:/private/store-birds/bm25.pkl is unreadable")

    class NoGenerator(types.ModuleType):
        def synthesize_with_backend(self, question, chunks):
            raise ConnectionError("http://127.0.0.1:11434/api/generate refused")

    try:
        pipeline_trace.trace_pipeline = fake_trace
        serve.RES = FakeRes()
        with env(**{**PUBLIC, "RAG_ALLOW_GENERATE": "1"}), \
                fake_module("term_attribution", NoAttribution("term_attribution")), \
                fake_module("generate", NoGenerator("generate")), \
                contextlib.redirect_stderr(io.StringIO()) as log:
            r = Request("POST", "/api/chat", b'{"question": "what sings", "generate": true}')
            check("a public /api/chat with a failed attribution answers 200", r.status, 200)
            check("and its attribution error does not carry the path",
                  "private" in json.dumps(r.body.get("attribution")), False)
            check("and its generation reason does not carry the generator's address",
                  (r.body.get("generation", {}).get("state"),
                   "11434" in json.dumps(r.body.get("generation"))), ("unavailable", False))
            check("and both details went to the server log",
                  ("bm25.pkl" in log.getvalue(), "11434" in log.getvalue()), (True, True))
        with env(**PRIVATE), \
                fake_module("term_attribution", NoAttribution("term_attribution")), \
                fake_module("generate", NoGenerator("generate")):
            r = Request("POST", "/api/chat", b'{"question": "what sings", "generate": true}')
            check("a private /api/chat keeps the generation reason verbatim",
                  "11434" in json.dumps(r.body.get("generation")), True)
            check("and the attribution error verbatim",
                  "bm25.pkl" in json.dumps(r.body.get("attribution")), True)
    finally:
        pipeline_trace.trace_pipeline, serve.attribution_for, serve.RES = _real_tp, _real_attr, _res

with section("inspect_folder wording"):
    raw = "~/no-such-folder-2026-09-06"
    try:
        serve.inspect_folder(raw)
        message = "accepted"
    except ValueError as exc:
        message = str(exc)
    check("inspect_folder repeats the string it was given",
          raw in message, True)
    check("and not the home directory it expanded to",
          str(Path.home()) in message, False)

# --------------------------------------------- /api/eval per corpus ---
# Until 2026-09-06 /api/eval read eval/results.json whatever corpus was
# active, so the quality view showed the ML papers' figures under the bird
# corpus, and the recorded static demo repeated that on every corpus.
with section("eval per corpus"):
    check("the bird corpus reads eval/results-birds.json",
          serve.eval_file({"golden": "eval/golden-birds.json"}).name, "results-birds.json")
    check("the ML corpus, whose files came first, reads eval/results.json",
          serve.eval_file({"golden": str(serve.EVAL_DIR / "golden_set.json")}).name,
          "results.json")
    check("no corpus at all reads eval/results.json", serve.eval_file(None).name,
          "results.json")
    _res = serve.RES
    try:
        serve.RES = FakeRes()
        r = Request("GET", "/api/eval")
        served = json.loads(r.body) if r.status == 200 else {}
        check("/api/eval serves the active corpus's own results file",
              served.get("corpus", {}).get("name"), "birds")
    finally:
        serve.RES = _res

# ------------------------------------------- quality files per corpus ---
# /api/per-case and /api/hard-cases named the ML papers' files until
# 2026-09-07 and served them under every corpus, and /api/threshold served
# a calibration of the ML gate that nothing fetched.
with section("quality files per corpus"):
    check("the bird corpus reads eval/per_case-birds.json",
          serve.quality_file("/api/per-case", {"golden": "eval/golden-birds.json"}).name,
          "per_case-birds.json")
    check("and eval/hard_cases-birds.json",
          serve.quality_file("/api/hard-cases", {"golden": "eval/golden-birds.json"}).name,
          "hard_cases-birds.json")
    check("the ML corpus, whose files came first, reads the unsuffixed file",
          serve.quality_file("/api/per-case", None).name, "per_case.json")
    check("/api/threshold is no longer a route", Request("GET", "/api/threshold").status, 404)
    _res = serve.RES
    try:
        serve.RES = FakeRes()
        r = Request("GET", "/api/per-case")
        served = json.loads(r.body) if r.status == 200 else {}
        check("/api/per-case serves the active corpus's own rows",
              served.get("corpus", {}).get("name"), "birds")
    finally:
        serve.RES = _res

# ------------------------------------------------- shared page assets ---
# The four pages share tokens.css, base.css and common.js since 2026-09-06
# and index.html imports passages.js. Until 2026-09-07 the dispatch served
# none of the four, so the live pages loaded unstyled and the front page's
# module could not start; the recorded build serves a directory and never
# noticed.
with section("shared page assets"):
    for _route, _head in (("/passages.js", b"export"), ("/common.js", b""),
                          ("/tokens.css", b":root"), ("/base.css", b"")):
        r = Request("GET", _route)
        check(f"{_route} is served from ui/",
              (r.status, isinstance(r.body, bytes) and _head in r.body), (200, True))

# ------------------------------------------------------- offline hub ---
# Loading a cached model without HF_HUB_OFFLINE sends a HEAD request to
# huggingface.co, and on a network that drops it startup waits on the hub's
# timeout. A fresh machine with no cache must still be allowed to download.
with section("offline hub"):
    tmp = Path(tempfile.mkdtemp(prefix="hub-"))
    try:
        with env(HF_HUB_OFFLINE=None):
            check("an empty cache leaves the variable unset so a first run can download",
                  (serve.set_offline_if_cached(tmp), os.environ.get("HF_HUB_OFFLINE")),
                  (False, None))
            (tmp / "models--sentence-transformers--all-MiniLM-L6-v2").mkdir()
            check("a cache holding a model turns the hub offline",
                  (serve.set_offline_if_cached(tmp), os.environ.get("HF_HUB_OFFLINE")),
                  (True, "1"))
        with env(HF_HUB_OFFLINE="0"):
            check("an operator's own setting is left alone",
                  (serve.set_offline_if_cached(tmp), os.environ.get("HF_HUB_OFFLINE")),
                  (False, "0"))
        with env(HF_HUB_CACHE=str(tmp)):
            check("HF_HUB_CACHE names the directory the hub reads",
                  serve.hf_cache_dir(), tmp)
        # The container sets HF_HOME before the build-time download, so the
        # models it reads at run time are under HF_HOME/hub.
        with env(HF_HUB_CACHE=None, HF_HOME=str(tmp)):
            check("HF_HOME names the parent of the hub directory",
                  serve.hf_cache_dir(), tmp / "hub")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

# --------------------------------------------------- /health defaults ---
with section("health defaults"), env(**PUBLIC):
    r = Request("GET", "/health")
    check("/health states the shipped expansion default",
          r.body.get("defaults", {}).get("expansion"), DEFAULT_EXPANSION)
    check("and reports what a public server has turned off",
          (r.body.get("public"), r.body.get("generate_allowed"),
           r.body.get("reindex_allowed")), (True, False, False))


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
