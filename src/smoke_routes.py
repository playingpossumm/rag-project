"""Hit every route the server dispatches on, and see what comes back.

`check_docs.py` proves each route is written down. That is not the same as
proving it answers -- `POST /ask` was written down, dispatched, and served the
wrong corpus for its entire life, and what found it was a request.

The route list is read out of `serve.py` rather than typed here, so a route
added tomorrow is smoke-tested tomorrow without anyone remembering to add it.
Typing the list would reproduce exactly the gap this exists to close.

What counts as a pass is deliberately weak: not a 5xx, and a body that parses as
whatever the content type claims. A route can return the wrong corpus with a
perfectly good 200, so this is a floor, not a verdict -- it catches the route
that crashes, the route that 404s because a rename missed it, and the route that
promises JSON and returns a stack trace.

Needs a server: `.venv\\Scripts\\python.exe src\\serve.py` in another terminal.

    .venv\\Scripts\\python.exe src\\smoke_routes.py
    .venv\\Scripts\\python.exe src\\smoke_routes.py --base http://127.0.0.1:8001
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from check_docs import served_routes  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# What each route needs to be asked properly. Anything not named here is a GET
# with no body, which is the right default for a page or a read-only endpoint.
POST_BODIES = {
    "/ask": {"question": "What is attention?"},
    "/api/chat": {"question": "What is attention?"},
    "/api/trace": {"question": "What is attention?"},
    "/api/corpus/select": None,          # filled in below with the active corpus
    # Absolute, resolved from this file rather than the working directory: run
    # from src/ a relative "data" is src/data, and the route correctly answers
    # "no such folder" to a probe that is simply wrong.
    "/api/index/inspect": {"path": str(Path(__file__).parent.parent / "data")},
    "/api/index/start": "SKIP",          # replaces the vector store; never smoke-tested
}

# Routes that cannot be exercised blind, with the reason. Recorded rather than
# quietly dropped: a skipped route is a gap in this file's coverage and should
# read as one.
SKIP = {
    "/api/index/start": "rebuilds the index in place -- destructive, not a smoke test",
    "/fonts/*": "a prefix, exercised below through one real font file",
    "/archive/pipeline-map.js": "exercised, but only meaningful with /archive",
}


def request(base: str, route: str, body=None, timeout=180):
    url = base + route
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read()
    except Exception as exc:                                    # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}", b""


def smoke(base: str) -> list[tuple[str, str]]:
    """Ask every dispatched route and return what did not answer cleanly.

    Returns rather than exits, so `src/test_routes.py` can start a server on an
    ephemeral port and call this directly. The CLI below keeps its own exit
    codes; the checking lives here once.
    """
    args = argparse.Namespace(base=base)
    # The active corpus, so /api/corpus/select is asked for something real
    # rather than for a name that would correctly 400.
    _, _, body = request(args.base, "/api/corpus")
    try:
        info = json.loads(body)
        # /api/corpus reports the corpus itself; the name may sit at the top
        # level or under "active" depending on the shape, and taking whichever
        # is present without checking its TYPE is how the first run of this
        # script posted a dict as a corpus name.
        active = info.get("name")
        if not isinstance(active, str):
            nested = info.get("active")
            active = nested.get("name") if isinstance(nested, dict) else None
    except Exception:                                           # noqa: BLE001
        active = None
    if not isinstance(active, str):
        active = None
    POST_BODIES["/api/corpus/select"] = {"name": active} if active else {"name": "llm"}

    routes = sorted(served_routes())
    print(f"{len(routes)} routes from serve.py's dispatch, against {args.base}\n")
    print(f"  {'route':<28}{'method':>7}{'status':>8}  content-type / problem")

    failures = []
    for route in routes:
        if route in SKIP:
            print(f"  {route:<28}{'-':>7}{'skip':>8}  {SKIP[route]}")
            continue
        target = route
        if route.endswith("/*"):
            print(f"  {route:<28}{'-':>7}{'skip':>8}  prefix route")
            continue

        body = POST_BODIES.get(route)
        method = "POST" if route in POST_BODIES else "GET"
        status, ctype, payload = request(args.base, target, body)

        problem = ""
        if status is None:
            problem = ctype                       # the exception text
        elif status >= 500:
            problem = f"server error: {payload[:120]!r}"
        elif status >= 400:
            problem = f"{status}: {payload[:120]!r}"
        elif "json" in ctype:
            try:
                json.loads(payload)
            except ValueError:
                problem = "claims JSON, does not parse"
        elif not payload:
            problem = "empty body"

        if problem:
            failures.append((route, problem))
        print(f"  {route:<28}{method:>7}{str(status):>8}  {problem or ctype}")

    # The prefix route, through a file that really exists.
    fonts = sorted((Path(__file__).parent.parent / "ui" / "fonts").glob("*.woff2"))
    if fonts:
        status, ctype, payload = request(args.base, f"/fonts/{fonts[0].name}")
        ok = status == 200 and len(payload) > 1000
        print(f"  {'/fonts/' + fonts[0].name:<28}{'GET':>7}{str(status):>8}  "
              f"{ctype if ok else 'FONT DID NOT SERVE'}")
        if not ok:
            failures.append((f"/fonts/{fonts[0].name}", "did not serve"))

    # Every POST route, asked with a field of the wrong TYPE. JSON carries
    # types and a caller can send an object where a string belongs; the answer
    # should be a 400 saying so. Until 2026-08-27 it was an unhandled
    # AttributeError and a closed socket, which a caller cannot tell from the
    # server having died -- found by this script sending exactly that by
    # accident.
    print()
    print(f"  {'POST route':<28}{'wrong-typed field':>20}  answer")
    for route, body in sorted(POST_BODIES.items()):
        if route in SKIP or not isinstance(body, dict) or not body:
            continue
        field = next(iter(body))
        status, ctype, payload = request(args.base, route, {field: {"not": "text"}})
        ok = status == 400
        detail = (json.loads(payload).get("error", "")[:58]
                  if ok and "json" in ctype else ctype)
        print(f"  {route:<28}{field:>20}  {status} {detail}")
        if not ok:
            failures.append((route, f"a {field} of the wrong type gave {status}, "
                                    f"not 400"))

    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    cli = ap.parse_args()

    status, _, _ = request(cli.base, "/health", timeout=5)
    if status is None:
        print(f"no server at {cli.base} -- start one with "
              f".venv\\Scripts\\python.exe src\\serve.py, or run "
              f"src\\test_routes.py which starts its own")
        return 2

    failures = smoke(cli.base)
    print()
    if failures:
        print(f"{len(failures)} route(s) did not answer cleanly:")
        for route, problem in failures:
            print(f"  {route}: {problem}")
        return 1
    print("every dispatched route answers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
