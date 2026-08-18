"""Local HTTP wrapper around `api.ask`, so other tools can call the pipeline.

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

    python src/serve.py
    curl -s localhost:8000/ask -d '{"question": "What is late interaction?"}'
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from api import ask

HOST, PORT = "127.0.0.1", 8000
MAX_BODY = 64 * 1024  # a question is small; refuse anything that clearly is not

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, status: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("", "/health"):
            self._send(200, {
                "status": "ok",
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
        if self.path.rstrip("/") != "/ask":
            self._send(404, {"error": f"no route {self.path}"})
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self._send(413, {"error": "request body too large"})
            return

        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            self._send(400, {"error": f"invalid JSON: {exc}"})
            return

        question = (payload.get("question") or "").strip()
        if not question:
            self._send(400, {"error": "field 'question' is required"})
            return

        try:
            answer = ask(
                question,
                k=int(payload.get("k", 5)),
                expansion=payload.get("expansion", "page"),
                min_confidence=float(payload.get("min_confidence", -2.0)),
                generate=bool(payload.get("generate", False)),
            )
        except Exception as exc:  # noqa: BLE001 - report rather than drop the connection
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
            return

        self._send(200, answer.to_dict())

    def log_message(self, fmt, *args):
        # Default logging writes to stderr with a noisy prefix; keep it terse.
        sys.stderr.write(f"  {self.address_string()} {fmt % args}\n")


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else HOST
    port = int(sys.argv[2]) if len(sys.argv) > 2 else PORT

    # Warm the index, embedder and BM25 before accepting traffic, so the first
    # real request is not the one that pays ~30s of model loading.
    print("Loading index and models...")
    ask("warmup", k=1)
    print(f"Ready on http://{host}:{port}  (POST /ask)")
    if host not in ("127.0.0.1", "localhost"):
        print("  WARNING: bound to a non-loopback address -- this exposes your "
              "document contents to the network.")

    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
