"""Local HTTP server: the JSON API, plus a UI for inspecting retrieval.

Two audiences, one process. `POST /ask` is the machine-facing endpoint another
tool embeds. `GET /` is the Retrieval Inspector -- the same pipeline, but showing
every intermediate ranking instead of only the five passages that survived.

The inspector exists because "why did it return that?" is the question that
actually matters, and no amount of prose about reciprocal rank fusion explains it
as well as watching a passage climb from rank 14 to rank 1. It is also the honest
answer to what this offers over a closed product: not better answers, but visible
reasoning and a number you can regress against.

Uses the standard library rather than FastAPI deliberately. This is a thin
adapter -- parse JSON, call one function, serialise the result -- and adding a
web framework plus an ASGI server for that would be two more dependencies to
install, pin, and carry between machines, for no behaviour the caller can see.
If this ever needs auth, concurrency, or a schema contract, FastAPI is the right
upgrade and the swap is confined to this file.

Binds to 127.0.0.1 by default. The corpus may contain contracts and internal
documents, and a service that answers questions about them should not be
reachable from the network by accident; exposing it has to be a deliberate act,
not the default.

    python src/serve.py                     # then open http://127.0.0.1:8000
    curl -s localhost:8000/ask -d '{"question": "What is late interaction?"}'
"""
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from api import ask

HOST, PORT = "127.0.0.1", 8000
MAX_BODY = 64 * 1024  # a question is small; refuse anything that clearly is not
UI_FILE = Path(__file__).parent.parent / "ui" / "index.html"
ARCH_FILE = Path(__file__).parent.parent / "docs" / "architecture.html"
EVAL_DIR = Path(__file__).parent.parent / "eval"
EVAL_FILE = EVAL_DIR / "results.json"

# The quality view reads whatever the harness has written. Each is optional:
# a file that has not been generated yet makes its panel say so rather than
# making the page fail to load, because "not measured" and "measured as zero"
# must not look the same.
QUALITY_FILES = {
    "/api/eval": "results.json",
    "/api/per-case": "per_case.json",
    "/api/threshold": "threshold.json",
    "/api/hard-cases": "hard_cases.json",
}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class Resources:
    """Index, embedder and BM25, loaded once and shared across requests.

    Loading these per request would cost ~20s each time. The lock serialises
    tracing rather than the whole server: SentenceTransformer and the reranker
    are not documented as thread-safe, and a ThreadingHTTPServer will happily
    call them concurrently the moment two browser tabs are open.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self._loaded = False
        self.corpus = None          # the active corpus config, see corpora.py

    def load(self, name: str | None = None):
        if self._loaded and (name is None or name == self.corpus["name"]):
            return
        from sentence_transformers import SentenceTransformer

        import corpora
        from hybrid import build_bm25
        from retrieve import EMBEDDING_MODEL, load_index

        cfg = corpora.get(name or corpora.active_name())
        if not cfg["indexed"]:
            raise ValueError(f"corpus {cfg['name']!r} has no index; "
                             f"run ingest.py with RAG_STORE_DIR={cfg['store']}")

        # The embedder is the same model for every corpus and costs ~20s to
        # load, so it is kept across a switch. The index, its metadata and the
        # BM25 built from that metadata all belong to one corpus and are
        # replaced together -- a half-switched state would serve one corpus's
        # passages under another's citations.
        self.index, self.metadata = load_index(cfg["store"])
        if not hasattr(self, "model"):
            self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.bm25 = build_bm25(self.metadata)
        self.corpus = cfg
        self._loaded = True

    def switch(self, name: str):
        with self.lock:
            self._loaded = False
            self.load(name)

    def threshold(self) -> float:
        """The active corpus's calibrated cut point, or 0.0 if nobody set one."""
        return (self.corpus or {}).get("threshold") or 0.0

    def reload(self):
        """Re-read the index after a rebuild, so serving matches what is on disk."""
        with self.lock:
            name = self.corpus["name"] if self.corpus else None
            self._loaded = False
            self.load(name)

    def stats(self) -> dict:
        docs = sorted({c["source"] for c in self.metadata})
        return {"chunks": len(self.metadata), "documents": len(docs), "sources": docs}


RES = Resources()


class IndexRun:
    """A single indexing run, driven in the background so the UI stays live.

    build_index() already emits structured events for every stage; nothing has
    ever consumed them. This collects them into a snapshot the browser can poll,
    which is why the CLI and the UI cannot drift apart -- there is one event
    source, not a second code path written to match the first.

    Polling rather than a websocket or SSE: the whole point of the stdlib server
    is that it stays a thin adapter, and a run emits on the order of one event
    per document. Poll pressure is not the bottleneck; parsing is.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.state = "idle"      # idle | running | done | error
        self.events: list[dict] = []
        self.files: dict[str, dict] = {}
        self.summary: dict | None = None
        self.error: str | None = None
        self.started: float | None = None

    def _on_event(self, event: dict):
        with self.lock:
            self.events.append(event)
            stage = event.get("stage")
            # Per-document rows, keyed by name so a re-emitted document updates
            # in place rather than appending a duplicate row.
            if stage == "parse":
                self.files[event["document"]] = {
                    "document": event["document"], "status": event["status"],
                    "detail": event.get("detail") or "", "units": event.get("units"),
                    "chunks": event.get("chunks"), "index": event.get("current"),
                }
            elif stage == "skip_unsupported":
                self.files[event["document"]] = {
                    "document": event["document"], "status": "SKIP",
                    "detail": event.get("reason") or "unsupported format",
                    "units": None, "chunks": None, "index": None,
                }

    def start(self, data_dir=None) -> bool:
        """Begin a run. Returns False if one is already in flight."""
        with self.lock:
            if self.state == "running":
                return False
            self.reset()
            self.state = "running"
            self.started = time.time()
        threading.Thread(target=self._run, args=(data_dir,), daemon=True).start()
        return True

    def _run(self, data_dir):
        try:
            from ingest import DATA_DIR, build_index

            summary = build_index(data_dir or DATA_DIR, progress=self._on_event)
            # Retrieval holds the OLD index in memory. Leaving it would serve
            # answers from a corpus that no longer matches what the UI reports.
            RES.reload()
            with self.lock:
                self.summary, self.state = summary, "done"
        except Exception as exc:  # noqa: BLE001 - surface it in the UI
            with self.lock:
                self.error = f"{type(exc).__name__}: {exc}"
                self.state = "error"

    def snapshot(self) -> dict:
        with self.lock:
            totals = {}
            for e in self.events:
                if e.get("stage") in ("scan", "guard", "embed", "cache"):
                    totals[e["stage"]] = {k: v for k, v in e.items() if k != "stage"}
            return {
                "state": self.state,
                "elapsed": round(time.time() - self.started, 1) if self.started else 0.0,
                "files": list(self.files.values()),
                "totals": totals,
                "summary": self.summary,
                "error": self.error,
            }


INDEX_RUN = IndexRun()

# Chosen to demonstrate the three behaviours worth seeing: a fact only lexical
# matching finds reliably, a question whose distinguishing clause the ranking
# must honour, and one the corpus cannot answer at all.
# Fallback only. The real suggestions come from each corpus's own golden set
# via eval/analytics.json -- see examples_for(). This list is what a fresh clone
# with no eval run yet has to offer, and it is about the ML papers because that
# is the corpus a fresh clone ships with.
EXAMPLES = [
    {"label": "a specific figure",
     "q": "What BLEU score did the Transformer achieve on WMT 2014 English-to-German?"},
    {"label": "a constraint that must be honoured",
     "q": "How are normalization statistics computed across features rather than examples?"},
    {"label": "spread across documents",
     "q": "What learning rate schedule and optimizer settings were used for training?"},
    {"label": "not in the corpus",
     "q": "What is the airspeed velocity of an unladen swallow?"},
]


def examples_for(name: str | None) -> list[dict]:
    """Suggested questions for one corpus, from questions already scored.

    Every suggestion is drawn from that corpus's golden set, so it is a question
    the harness has run and whose outcome is known -- including the adversarial
    ones, which are meant to be refused. Inventing examples risks offering one
    that happens to fail, which reads as a broken system rather than a
    deliberate demonstration.

    This was a real defect, not a hypothetical: the list was global and written
    for the ML papers, so switching to the bird corpus still suggested asking
    about BLEU scores on WMT 2014.
    """
    f = EVAL_DIR / "analytics.json"
    if not name or not f.exists():
        return EXAMPLES
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return EXAMPLES
    for c in data.get("corpora", []):
        if c.get("name") == name and c.get("examples"):
            return c["examples"]
    return EXAMPLES


def inspect_folder(raw: str) -> tuple[Path, list[str], list[tuple[str, str]]]:
    """Resolve a pasted folder path and report what indexing it would find.

    A browser cannot hand a filesystem path to a server -- a file picker returns
    file contents, never a location -- so a pasted path is the honest form of
    "point this at my documents", and RAG_DATA_DIR already proves the server
    side works. What is added here is the checking a pasted string needs: it may
    be a typo, a file, a folder full of .doc, or empty.

    Raises ValueError with a message meant to be shown to a person. That matters
    more here than anywhere else in this file: indexing REPLACES the vector
    store, so a path that quietly resolves to an empty directory would destroy a
    working index and report success.
    """
    from corpus_health import SUPPORTED, scan_unsupported

    if not raw:
        raise ValueError("no folder given")

    folder = Path(raw.strip().strip('"')).expanduser()
    if not folder.exists():
        raise ValueError(f"no such folder: {folder}")
    if not folder.is_dir():
        raise ValueError(f"that is a file, not a folder: {folder}")

    try:
        supported = sorted(f.name for f in folder.iterdir()
                           if f.is_file() and f.suffix.lower() in SUPPORTED)
        skipped = [(path.name, reason) for path, reason in scan_unsupported(folder)]
    except PermissionError as exc:
        raise ValueError(f"cannot read that folder: {exc}") from exc

    if not supported:
        formats = ", ".join(sorted(SUPPORTED))
        raise ValueError(
            f"{folder} holds no indexable documents ({formats}). "
            + (f"It does hold {len(skipped)} file(s) in other formats."
               if skipped else "It appears to be empty.")
        )
    return folder, supported, skipped


def corpus_info() -> dict:
    """What the UI shows about the corpus it is talking to."""
    import corpora

    if not RES.corpus:
        return {}
    return corpora.describe(RES.corpus)


def trace_options(payload: dict) -> dict:
    """Read pipeline options off a /api/trace request.

    This is what turns the inspector into a bench: the same question, re-run
    under a different configuration, so "did that change help?" can be answered
    on the query in front of you rather than only in aggregate. Aggregate
    numbers come from eval/results.json and describe 84 cases; this describes
    the one you asked.

    Every option is validated here rather than passed through, because these
    arrive from a browser and an unknown fusion strategy should be a 400 that
    names the mistake, not a 500 from three frames deeper.
    """
    opts: dict = {}

    if "rerank" in payload:
        opts["use_reranker"] = bool(payload["rerank"])

    if "fusion" in payload:
        fusion = str(payload["fusion"])
        if fusion not in ("none", "rrf", "weighted"):
            raise ValueError(f"fusion must be none|rrf|weighted, got {fusion!r}")
        opts["fusion"] = fusion

    if "expansion" in payload:
        expansion = str(payload["expansion"])
        if expansion not in ("none", "window", "page"):
            raise ValueError(f"expansion must be none|window|page, got {expansion!r}")
        opts["expansion"] = expansion

    if "max_per_source" in payload:
        cap = payload["max_per_source"]
        # null is "no cap", which is a real setting and distinct from absent.
        opts["max_per_source"] = None if cap is None else int(cap)

    return opts


def chat(question: str, payload: dict) -> dict:
    """One question, one round trip: the answer AND how it was reached.

    The chat surface shows its own working -- which retriever found what, what
    the reranker moved, what the cap dropped -- so it needs the trace and the
    answer together. Two calls would re-run retrieval twice for one question and
    could disagree, which is the exact failure the citation contract exists to
    prevent.

    Generation is reported, never thrown. `generation.state` is one of:

      "off"        the caller did not ask for prose
      "ok"         a real answer came back
      "unavailable" the call was attempted and failed -- no credit, no key, a
                   refusal. The passages are still returned and still correct,
                   so the surface degrades to retrieval-only rather than to an
                   error page. The reason is passed through verbatim because
                   "you have no credit" and "the model declined" need different
                   actions from the reader.
    """
    from pipeline_trace import trace_pipeline

    with RES.lock:
        trace = trace_pipeline(question, RES.index, RES.metadata, RES.model,
                               bm25=RES.bm25, k=int(payload.get("k", 5)),
                               threshold=RES.threshold(),
                               **trace_options(payload))

    selected = trace["stages"][-1]["items"]
    generation = {"state": "off", "text": None, "reason": None}

    if payload.get("generate"):
        try:
            from generate import synthesize

            # Generate from the passages the trace reports, not a second
            # retrieval, so the citations shown and the text read are the same.
            with RES.lock:
                chunks = [RES.metadata[i["chunk_id"]] for i in selected]
                for c, i in zip(chunks, selected):
                    c.setdefault("chunk_id", i["chunk_id"])
            generation = {"state": "ok",
                          "text": synthesize(question, chunks),
                          "reason": None}
        except Exception as exc:  # noqa: BLE001 - reported, never raised
            generation = {"state": "unavailable", "text": None,
                          "reason": f"{type(exc).__name__}: {exc}"}

    return {**trace, "generation": generation}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, status: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self._raw(status, body, "application/json; charset=utf-8")

    def _raw(self, status: int, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict | None:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self._send(413, {"error": "request body too large"})
            return None
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            self._send(400, {"error": f"invalid JSON: {exc}"})
            return None

    def do_GET(self):
        route = self.path.split("?")[0].rstrip("/")

        if route == "":
            if not UI_FILE.exists():
                self._send(404, {"error": "ui/index.html is missing"})
                return
            self._raw(200, UI_FILE.read_bytes(), "text/html; charset=utf-8")

        elif route == "/~/architecture":
            # A committed artifact, served rather than generated: it is built by
            # `npm run architecture:build` and opens straight from docs/ with no
            # server at all. The route exists so it is reachable from the running
            # tool as well.
            if not ARCH_FILE.exists():
                self._send(404, {"error": "docs/architecture.html is missing -- "
                                          "run npm run architecture:build"})
                return
            self._raw(200, ARCH_FILE.read_bytes(), "text/html; charset=utf-8")

        elif route == "/pipeline-map.js":
            f = UI_FILE.parent / "pipeline-map.js"
            if not f.exists():
                self._send(404, {"error": "ui/pipeline-map.js is missing"})
                return
            self._raw(200, f.read_bytes(), "text/javascript; charset=utf-8")

        elif route.startswith("/fonts/"):
            # Inter and DM Mono, served from the repo rather than a CDN. The
            # corpus this tool indexes may be private, and a page about private
            # documents should not announce itself to a font host on load --
            # which is also why the whole UI has no external requests at all.
            name = route.rsplit("/", 1)[-1]
            f = UI_FILE.parent / "fonts" / name
            if name not in {p.name for p in (UI_FILE.parent / "fonts").glob("*.woff2")}:
                self._send(404, {"error": f"no font {name}"})
                return
            self._raw(200, f.read_bytes(), "font/woff2")

        elif route == "/quality":
            f = UI_FILE.parent / "quality.html"
            if not f.exists():
                self._send(404, {"error": "ui/quality.html is missing"})
                return
            self._raw(200, f.read_bytes(), "text/html; charset=utf-8")

        elif route in QUALITY_FILES and route != "/api/eval":
            f = EVAL_DIR / QUALITY_FILES[route]
            if not f.exists():
                self._send(404, {"error": f"eval/{f.name} has not been generated -- "
                                          f"see README for the command"})
                return
            self._raw(200, f.read_bytes(), "application/json; charset=utf-8")

        elif route == "/api/eval":
            # Static passthrough of what evaluate.py wrote. The UI shows measured
            # numbers rather than restating them, so a stale README cannot make
            # the interface lie; if the file is absent the view says so.
            if not EVAL_FILE.exists():
                self._send(404, {"error": "eval/results.json is missing -- run python src/evaluate.py"})
                return
            self._raw(200, EVAL_FILE.read_bytes(), "application/json; charset=utf-8")

        elif route == "/api/analytics":
            # Everything the analytics page plots, for every corpus at once --
            # the page compares them, so it cannot be scoped to the active one.
            # Regenerate with src/build_analytics.py after any eval run.
            f = EVAL_DIR / "analytics.json"
            if not f.exists():
                self._send(404, {"error": "eval/analytics.json is missing -- "
                                          "run python src/build_analytics.py"})
                return
            self._raw(200, f.read_bytes(), "application/json; charset=utf-8")

        elif route == "/api/chunks":
            # Which document each chunk belongs to, so the UI can draw the index
            # as one block per document at its true size. Dividing the total
            # evenly across twenty documents was close enough to look right and
            # wrong in a way nobody would ever catch, which is the worst kind.
            #
            # An earlier version of this returned each chunk's cosine similarity
            # to the question too, to place chunks by distance from it. That
            # layout was dropped in favour of city blocks, so the similarity
            # work went with it rather than being left to rot -- and with the
            # question gone this is a plain GET.
            try:
                with RES.lock:
                    sources = sorted({c["source"] for c in RES.metadata})
                    order = {s: i for i, s in enumerate(sources)}
                    doc = [order[c["source"]] for c in RES.metadata]
                self._send(200, {"n": len(doc), "sources": sources, "doc": doc})
            except Exception as exc:  # noqa: BLE001 - report rather than drop
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

        elif route == "/api/corpus":
            # data_dir is reported because the answer to "which documents is
            # this?" stopped being obvious the moment a folder could be pasted
            # in. A page that can be pointed anywhere has to say where it points.
            from ingest import DATA_DIR

            active = RES.corpus["name"] if RES.corpus else None
            self._send(200, {**RES.stats(), "examples": examples_for(active),
                             "data_dir": str(DATA_DIR),
                             "active": corpus_info()})

        elif route == "/api/corpora":
            import corpora
            self._send(200, {
                "active": RES.corpus["name"] if RES.corpus else None,
                "corpora": [corpora.describe(c)
                            for c in corpora.registry().values()],
            })

        elif route == "/api/index/status":
            self._send(200, INDEX_RUN.snapshot())

        elif route == "/health":
            self._send(200, {
                "status": "ok",
                "ui": "GET /",
                "usage": "POST /ask with {\"question\": \"...\"}",
                "options": {
                    "k": "number of passages (default 5)",
                    "expansion": "'page' | 'window' | 'none'",
                    "min_confidence": "float; lower returns more, less certain results",
                    "generate": "false by default -- true requires an API key and costs money",
                },
            })
        else:
            self._send(404, {"error": f"no route {self.path}"})

    def do_POST(self):
        route = self.path.rstrip("/")

        if route == "/api/corpus/select":
            payload = self._body()
            if payload is None:
                return
            name = (payload.get("name") or "").strip()
            try:
                RES.switch(name)
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
                return
            except Exception as exc:  # noqa: BLE001
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._send(200, {**RES.stats(), "active": corpus_info()})
            return

        if route in ("/api/index/start", "/api/index/inspect"):
            payload = self._body()
            if payload is None:
                return
            # An absent path on /start means the configured corpus directory,
            # which is how the CLI and RAG_DATA_DIR have always worked. On
            # /inspect it means nothing, so it is an error rather than a 200
            # reporting zero documents -- which would read as "your folder is
            # empty" for a request that never named a folder.
            raw = (payload.get("path") or "").strip()
            try:
                folder, supported, skipped = (
                    inspect_folder(raw) if raw or route.endswith("inspect")
                    else (None, None, None))
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
                return

            # Look before indexing. The two routes share one resolver so what
            # the preview reports and what the run reads cannot differ.
            if route == "/api/index/inspect":
                self._send(200, {
                    "folder": str(folder) if folder else None,
                    "documents": supported if supported is not None else [],
                    "skipped": [{"document": n, "reason": r} for n, r in (skipped or [])],
                })
                return

            if INDEX_RUN.start(folder):
                self._send(202, {"state": "running",
                                 "folder": str(folder) if folder else None,
                                 "documents": len(supported) if supported else None})
            else:
                self._send(409, {"error": "an indexing run is already in progress"})
            return

        if route == "/api/chat":
            payload = self._body()
            if payload is None:
                return
            question = (payload.get("question") or "").strip()
            if not question:
                self._send(400, {"error": "field 'question' is required"})
                return
            try:
                self._send(200, chat(question, payload))
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
            return

        if route not in ("/ask", "/api/trace"):
            self._send(404, {"error": f"no route {self.path}"})
            return

        payload = self._body()
        if payload is None:
            return

        question = (payload.get("question") or "").strip()
        if not question:
            self._send(400, {"error": "field 'question' is required"})
            return

        try:
            if route == "/api/trace":
                from pipeline_trace import trace_pipeline  # not 'trace': shadows a stdlib module

                with RES.lock:
                    result = trace_pipeline(question, RES.index, RES.metadata,
                                            RES.model, bm25=RES.bm25,
                                            k=int(payload.get("k", 5)),
                                            threshold=RES.threshold(),
                                            **trace_options(payload))
                self._send(200, result)
            else:
                answer = ask(
                    question,
                    k=int(payload.get("k", 5)),
                    expansion=payload.get("expansion", "page"),
                    min_confidence=float(payload.get("min_confidence", -2.0)),
                    generate=bool(payload.get("generate", False)),
                )
                self._send(200, answer.to_dict())
        except ValueError as exc:
            # A rejected option is the caller's mistake, not the server's, and
            # saying so is the difference between fixing it and guessing.
            self._send(400, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - report rather than drop the connection
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt, *args):
        # Default logging writes to stderr with a noisy prefix; keep it terse.
        sys.stderr.write(f"  {self.address_string()} {fmt % args}\n")


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else HOST
    port = int(sys.argv[2]) if len(sys.argv) > 2 else PORT

    # Warm the index, embedder and BM25 before accepting traffic, so the first
    # real request is not the one that pays ~30s of model loading.
    print("Loading index and models...")
    RES.load()
    ask("warmup", k=1)
    stats = RES.stats()
    print(f"Ready on http://{host}:{port}")
    print(f"  inspector  http://{host}:{port}/")
    print(f"  quality    http://{host}:{port}/quality   (retrieval quality dashboard)")
    print(f"  map        http://{host}:{port}/~/architecture")
    print(f"  api        POST /ask, POST /api/trace")
    print(f"  corpus     {stats['chunks']} chunks from {stats['documents']} documents")
    if host not in ("127.0.0.1", "localhost"):
        print("  WARNING: bound to a non-loopback address -- this exposes your "
              "document contents to the network.")

    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
