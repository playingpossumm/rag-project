"""Ask every route, against a server this test starts itself.

`src/smoke_routes.py` has asked every dispatched route since 2026-08-27 and
found two real defects doing it. It needed a server someone had already started,
so it was not in the counted suite -- which meant the only coverage of the
request path existed when somebody remembered to run it. That is the same shape
as the three claimed-but-absent tests in this repo's history: a check that
happens sometimes is not a check.

So this starts one. Port 0, so it cannot collide with a server the developer is
already running on 8000, and a daemon thread so a failure here cannot leave a
process behind. The checking logic is imported from `smoke_routes` rather than
copied, because two copies of "what counts as answering" is how they drift.

The slowest test in the suite by a distance: loading the index and the models
costs roughly half a minute. That is the price of exercising the real server
rather than a mock of it, and the two defects it has already caught -- a route
that answered from the wrong corpus, and a wrong-typed field that dropped the
connection -- were both invisible to anything short of a request.

    .venv\\Scripts\\python.exe src\\test_routes.py
"""
import sys
import threading
from http.server import ThreadingHTTPServer

import serve
import smoke_routes

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    print("loading the index and models (this is the slow part)...")
    serve.RES.load()

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    print(f"serving on {base}\n")

    try:
        failures = smoke_routes.smoke(base)
    finally:
        httpd.shutdown()
        httpd.server_close()

    if failures:
        print(f"\n{len(failures)} route check(s) failed:")
        for route, problem in failures:
            print(f"  {route}: {problem}")
        return 1
    print("\nall route checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
