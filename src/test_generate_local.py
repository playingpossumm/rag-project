"""The local generation path, over a real socket.

`src/test_generate.py` covers `generate.py` by stubbing the Anthropic client, so
everything except the HTTP request is exercised. That request has still never
happened -- the account has no credit -- which leaves the one part of the path
that involves a network permanently unrun.

`generate_local.py` speaks to a local Ollama, and Ollama is a plain HTTP server.
So this test **stands one up**: a real `ThreadingHTTPServer` implementing the two
endpoints the module uses, on an ephemeral port. The request is built, sent over
a socket, answered, parsed, and the answer returned. Nothing is monkeypatched
except the host to point at it.

That makes this the first test in the repo where the generation path actually
executes end to end. The model is a fake -- it echoes what it was asked, which
is precisely what lets the test assert that the passages reaching the model are
the ones the answer will cite -- but the transport, the JSON shapes, the error
branches and the timeouts are real.

    .venv\\Scripts\\python.exe src\\test_generate_local.py
"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import generate_local

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHECKS: list[tuple[str, object, object, bool]] = []


def check(name: str, got, want) -> None:
    CHECKS.append((name, got, want, got == want))


CHUNKS = [
    {"source": "attention.pdf", "locator": {"kind": "page", "value": 3},
     "text": "The model uses scaled dot-product attention."},
    {"source": "deck.pptx", "locator": {"kind": "slide", "value": 7},
     "text": "Multi-head attention runs several projections in parallel."},
]

# What the fake Ollama should do next. Set per test, read by the handler.
BEHAVIOUR = {"mode": "echo"}
RECEIVED: list[dict] = []


class FakeOllama(BaseHTTPRequestHandler):
    """The two endpoints generate_local.py uses, and nothing else."""

    def log_message(self, *args):
        pass                       # a test does not need an access log

    def do_GET(self):
        if self.path == "/api/tags":
            self._json(200, {"models": [{"name": "llama3.2"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        RECEIVED.append(body)
        mode = BEHAVIOUR["mode"]
        if mode == "model-missing":
            self._json(404, {"error": f"model '{body['model']}' not found"})
        elif mode == "server-error":
            self._json(500, {"error": "something went wrong"})
        elif mode == "empty":
            self._json(200, {"message": {"role": "assistant", "content": "  "},
                             "done_reason": "length"})
        else:
            # Echo the prompt back, so the test can assert on what the model
            # was actually handed rather than on what it was supposed to be.
            prompt = body["messages"][-1]["content"]
            self._json(200, {"message": {"role": "assistant",
                                         "content": f"ANSWER<<{prompt}>>"},
                             "done_reason": "stop"})

    def _json(self, status, payload):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main() -> int:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    generate_local.HOST = f"http://127.0.0.1:{port}"
    generate_local.MODEL = "llama3.2"

    try:
        check("a reachable Ollama reports available", generate_local.available(), True)

        # ---- the happy path, over a real socket --------------------------
        BEHAVIOUR["mode"] = "echo"
        RECEIVED.clear()
        answer = generate_local.synthesize("How does attention work?", CHUNKS)
        sent = RECEIVED[0]
        prompt = sent["messages"][-1]["content"]

        check("an answer comes back over HTTP", answer.startswith("ANSWER<<"), True)
        check("the model is handed every passage it will cite",
              all(c["text"] in prompt for c in CHUNKS), True)
        check("and nothing it was not given", "hollow bones" in prompt, False)
        check("the question reaches the model",
              "How does attention work?" in prompt, True)

        # The citation format is generate.py's, reused rather than reimplemented,
        # so a slide is cited as a slide here too.
        check("a PDF passage is cited by page", "attention.pdf, page 3" in prompt, True)
        check("a deck passage is cited by slide", "deck.pptx, slide 7" in prompt, True)

        check("the system prompt forbids answering outside the context",
              "ONLY the provided context" in sent["messages"][0]["content"], True)
        check("the configured model is the one asked for", sent["model"], "llama3.2")
        check("streaming is off, because the caller wants one string",
              sent["stream"], False)
        check("sampling is deterministic, because this path gets measured",
              sent["options"]["temperature"], 0.0)

        # ---- an empty answer is a failure, not a blank ---------------------
        BEHAVIOUR["mode"] = "empty"
        try:
            generate_local.synthesize("q", CHUNKS)
            empty = "no error raised"
        except RuntimeError as exc:
            empty = str(exc)
        check("an empty response raises rather than returning ''",
              "no text in the response" in empty, True)
        check("and names done_reason, which is the thing to act on",
              "length" in empty, True)

        # ---- the two failures a reader can actually fix --------------------
        BEHAVIOUR["mode"] = "model-missing"
        try:
            generate_local.synthesize("q", CHUNKS)
            missing = "no error raised"
        except generate_local.LocalModelUnavailable as exc:
            missing = str(exc)
        except Exception as exc:                                # noqa: BLE001
            missing = f"{type(exc).__name__}: {exc}"
        check("a model that is not pulled says so, in its own error class",
              "ollama pull" in missing, True)

        BEHAVIOUR["mode"] = "server-error"
        try:
            generate_local.synthesize("q", CHUNKS)
            broken = "no error raised"
        except generate_local.LocalModelUnavailable as exc:
            broken = f"wrongly classed as unavailable: {exc}"
        except RuntimeError as exc:
            broken = str(exc)
        check("a 500 is a RuntimeError, not 'go install something'",
              broken.startswith("Ollama returned 500"), True)

        check("no passages is the caller's error and a different one",
              _raises_value_error(), True)

        # ---- the selector, because a module nothing calls does not work ----
        # generate_local.py was written, tested, and reachable from nothing:
        # api.ask and serve.chat both said "from generate import synthesize"
        # outright. That is the fifth time this session a module has been
        # correct and unreachable, so the wiring gets its own checks.
        import os

        import generate

        BEHAVIOUR["mode"] = "echo"
        RECEIVED.clear()
        was = os.environ.get("RAG_GENERATOR")
        try:
            os.environ.pop("RAG_GENERATOR", None)
            check("the default backend is unchanged", generate.backend(), "anthropic")

            os.environ["RAG_GENERATOR"] = "ollama"
            check("the env var selects the local one", generate.backend(), "ollama")
            answer = generate.synthesize_with_backend("routed?", CHUNKS)
            check("and synthesize_with_backend really reaches it",
                  answer.startswith("ANSWER<<"), True)
            check("over the same socket, with the same passages",
                  "routed?" in RECEIVED[-1]["messages"][-1]["content"], True)

            os.environ["RAG_GENERATOR"] = "local"
            check("'local' is accepted as a synonym",
                  generate.synthesize_with_backend("q", CHUNKS).startswith("ANSWER<<"),
                  True)

            os.environ["RAG_GENERATOR"] = "gpt-9"
            try:
                generate.synthesize_with_backend("q", CHUNKS)
                unknown = "no error raised"
            except ValueError as exc:
                unknown = str(exc)
            check("an unknown backend names itself rather than falling back",
                  "gpt-9" in unknown, True)
        finally:
            os.environ.pop("RAG_GENERATOR", None)
            if was is not None:
                os.environ["RAG_GENERATOR"] = was

        # ---- nothing listening is the common case on a fresh machine -------
        # Pointed at a port nothing is bound to, rather than shutting the fake
        # one down: shutdown() stops serve_forever but leaves the socket bound,
        # so the connection is accepted and never answered -- a hang, not a
        # refusal, and a different error than the one being asserted.
        generate_local.HOST = "http://127.0.0.1:1"
        check("a server that is not there reports unavailable",
              generate_local.available(), False)
        try:
            generate_local.synthesize("q", CHUNKS)
            gone = "no error raised"
        except generate_local.LocalModelUnavailable as exc:
            gone = str(exc)
        except Exception as exc:                                # noqa: BLE001
            gone = f"{type(exc).__name__}: {exc}"
        check("and asking it says how to start one", "ollama serve" in gone, True)
    finally:
        # shutdown() first: server_close() while a thread is still inside
        # serve_forever leaves it reading from a closed socket, which prints a
        # traceback from a daemon thread after the results and looks like a
        # failure that is not one.
        httpd.shutdown()
        httpd.server_close()

    width = max(len(n) for n, *_ in CHECKS)
    failed = 0
    for name, got, want, ok in CHECKS:
        if not ok:
            failed += 1
            print(f"FAIL  {name:<{width}}  got {got!r}, want {want!r}")
    print(f"{len(CHECKS) - failed}/{len(CHECKS)} local generation checks passed")
    return 1 if failed else 0


def _raises_value_error() -> bool:
    try:
        generate_local.synthesize("q", [])
    except ValueError:
        return True
    except Exception:                                           # noqa: BLE001
        return False
    return False


if __name__ == "__main__":
    raise SystemExit(main())
