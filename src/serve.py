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
AMBIENT_FILE = Path(__file__).parent.parent / "ui" / "ambient.html"
EVAL_FILE = Path(__file__).parent.parent / "eval" / "results.json"

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

    def load(self):
        if self._loaded:
            return
        from sentence_transformers import SentenceTransformer

        from hybrid import build_bm25
        from retrieve import EMBEDDING_MODEL, load_index

        self.index, self.metadata = load_index()
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.bm25 = build_bm25(self.metadata)
        self._loaded = True

    def reload(self):
        """Re-read the index after a rebuild, so serving matches what is on disk."""
        with self.lock:
            self._loaded = False
            self.load()

    def corpus(self) -> dict:
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

        elif route == "/ambient":
            if not AMBIENT_FILE.exists():
                self._send(404, {"error": "ui/ambient.html is missing"})
                return
            self._raw(200, AMBIENT_FILE.read_bytes(), "text/html; charset=utf-8")

        elif route == "/api/eval":
            # Static passthrough of what evaluate.py wrote. The UI shows measured
            # numbers rather than restating them, so a stale README cannot make
            # the interface lie; if the file is absent the view says so.
            if not EVAL_FILE.exists():
                self._send(404, {"error": "eval/results.json is missing -- run python src/evaluate.py"})
                return
            self._raw(200, EVAL_FILE.read_bytes(), "application/json; charset=utf-8")

        elif route == "/api/corpus":
            self._send(200, {**RES.corpus(), "examples": EXAMPLES})

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

        if route == "/api/index/start":
            if INDEX_RUN.start():
                self._send(202, {"state": "running"})
            else:
                self._send(409, {"error": "an indexing run is already in progress"})
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
                                            k=int(payload.get("k", 5)))
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
    stats = RES.corpus()
    print(f"Ready on http://{host}:{port}")
    print(f"  inspector  http://{host}:{port}/")
    print(f"  ambient    http://{host}:{port}/ambient   (generative options, side by side)")
    print(f"  api        POST /ask, POST /api/trace")
    print(f"  corpus     {stats['chunks']} chunks from {stats['documents']} documents")
    if host not in ("127.0.0.1", "localhost"):
        print("  WARNING: bound to a non-loopback address -- this exposes your "
              "document contents to the network.")

    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
