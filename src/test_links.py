"""Does the link checker catch the ways a link here has actually broken?

A check that never fails is indistinguishable from no check, and this one
passed for as long as it existed while four links on the site pointed at a
repository that does not exist. It matched each URL against a pattern named
`SELF_REPO`, treated the match as proof the link pointed at this repository,
and then verified the branch and the file against the local working tree, which
agreed. Everything about the link was checked except the part that was wrong.

Both defects reproduced here are real:

- `/about` linked to `.../blob/main/docs/corpus-manifest.md` while this
  repository's only branch is `master`. That is `branch that does not exist`,
  and it is why the guard was written on 2026-08-30.
- Every Source link named `ArdellAlfatih/rag-project` while the remote is
  `playingpossumm/rag-project`. That is `owner that is not this repository`,
  found on 2026-09-01, and it is why the guard was extended.

The rest are the failures the checker claims to cover, staged so that the claim
is committed rather than performed once and remembered.

Hermetic: every scenario builds its own `ui/` in a temporary directory, and
`git` and the route list are answered by stubs, so this passes or fails on the
checker rather than on the state of this repository or of its remote. No
network, no models, milliseconds.

    .venv\\Scripts\\python.exe src\\test_links.py
"""
import contextlib
import io
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import check_links as cl  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHECKS: list[tuple[str, object, object, bool]] = []


def check(name: str, got, want) -> None:
    CHECKS.append((name, got, want, got == want))


ROUTES = {"/", "/about", "/quality", "/api/chat"}
REMOTE = "https://github.com/owner/proj.git"


class Result:
    def __init__(self, code: int, out: str):
        self.code, self.out = code, out

    @property
    def broken(self) -> bool:
        return self.code != 0

    def says(self, *fragments: str) -> bool:
        return all(f in self.out for f in fragments)


def scenario(hrefs, *, files=("docs/corpus-manifest.md",), branches=("master",),
             remote=REMOTE, ids=("corpora",)):
    """Run the checker over a synthetic ui/ and a stubbed git.

    `hrefs` is what the interface links to, `files` and `branches` are what the
    working tree and the repository hold, and `remote` is what `origin` points
    at. Every one of those is a thing the checker consults, so staging a failure
    means changing one of them and leaving the rest correct.
    """
    tmp = Path(tempfile.mkdtemp(prefix="links-"))
    saved = (cl.ROOT, cl.UI, cl.git, cl.served_routes, sys.argv)
    try:
        ui = tmp / "ui"
        ui.mkdir()
        body = "".join(f'<a href="{h}">x</a>' for h in hrefs)
        marks = "".join(f'<div id="{i}"></div>' for i in ids)
        (ui / "index.html").write_text(body, encoding="utf-8")
        # The pages an anchor can point into have to exist as files, because
        # that is how the checker reads their ids.
        for page in ("about", "quality"):
            (ui / f"{page}.html").write_text(marks, encoding="utf-8")
        for rel in files:
            f = tmp / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("x", encoding="utf-8")

        def git(*args):
            if args[:2] == ("remote", "get-url"):
                return remote
            if args[0] == "rev-parse":
                ref = args[-1].split("^")[0]
                return "abc123" if ref in branches else None
            if args[0] == "for-each-ref":
                return "\n".join(branches)
            return None

        cl.ROOT, cl.UI, cl.git = tmp, ui, git
        cl.served_routes = lambda: set(ROUTES)
        sys.argv = ["check_links.py"]

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cl.main()
        return Result(code, buf.getvalue())
    finally:
        cl.ROOT, cl.UI, cl.git, cl.served_routes, sys.argv = saved
        shutil.rmtree(tmp, ignore_errors=True)


GOOD = "https://github.com/owner/proj"
GOOD_FILE = f"{GOOD}/blob/master/docs/corpus-manifest.md"

# ---- the repository a link names -------------------------------------------
# 2026-09-01. The repository name matches and the owner does not, which is a
# link that cannot resolve however healthy the branch and the path look.
wrong_owner = scenario(["https://github.com/someone-else/proj"])
check("owner / a different owner of the same repository name is reported",
      wrong_owner.broken, True)
check("owner / the message names both, so the fix is in the output",
      wrong_owner.says("someone-else/proj", "owner/proj"), True)

# The failure that hid it: a link whose branch and path are checkable against
# the working tree, and which points somewhere else entirely.
hidden = scenario([f"https://github.com/someone-else/proj/blob/master/docs/corpus-manifest.md"])
check("owner / a wrong owner is caught even when branch and path are valid here",
      hidden.broken, True)

check("owner / the right repository passes",
      scenario([GOOD, GOOD_FILE]).broken, False)

check("owner / a .git suffix on the remote is not part of the name",
      scenario([GOOD], remote="git@github.com:owner/proj.git").broken, False)

# A link to somebody else's repository is not this repository's business
# offline: its branches and files are not in this working tree.
other = scenario(["https://github.com/psf/requests/blob/main/README.md"])
check("owner / another repository is left to --http, not guessed at",
      (other.broken, other.says("another repository")), (False, True))

no_remote = scenario([GOOD])
check("owner / a clone with no remote says so rather than failing",
      (scenario([GOOD], remote=None).broken, no_remote.broken), (False, False))

# ---- the branch a link names -----------------------------------------------
# 2026-08-30. /about linked to blob/main/... and the only branch is master.
main_branch = scenario([f"{GOOD}/blob/main/docs/corpus-manifest.md"])
check("branch / a branch that does not exist is reported",
      main_branch.broken, True)
check("branch / the message lists the branches that do exist",
      main_branch.says("master"), True)

# ---- the path a link names -------------------------------------------------
check("path / a file that is not in the working tree is reported",
      scenario([f"{GOOD}/blob/master/docs/gone.md"]).broken, True)
check("path / a percent-encoded path is decoded before it is looked up",
      scenario([f"{GOOD}/blob/master/docs/a%20b.md"],
               files=("docs/a b.md",)).broken, False)

# ---- internal routes and anchors -------------------------------------------
check("route / a path serve.py does not dispatch is reported",
      scenario(["/nowhere"]).broken, True)
check("route / a page served as a file rather than a route is allowed",
      scenario(["/versions"]).broken, False)
check("anchor / an id no element carries is reported",
      scenario(["/about#missing"]).broken, True)
check("anchor / an id that exists passes",
      scenario(["/about#corpora"]).broken, False)

# An anchor that silently lands at the top of the page is worse than an error,
# which is the whole reason anchors are checked at all.
check("anchor / the message names the id, not just the page",
      scenario(["/about#missing"]).says("#missing"), True)

# ---- the checker's own failure mode ----------------------------------------
# Finding nothing to check is not the same as finding nothing wrong.
check("empty / a ui/ with no hrefs is a failure, not a pass",
      scenario([]).broken, True)

# ---- exit codes are the contract -------------------------------------------
check("exit / clean is 0 and broken is 1",
      (scenario([GOOD, "/about#corpora"]).code, scenario(["/nowhere"]).code),
      (0, 1))


def main() -> int:
    width = max(len(n) for n, *_ in CHECKS)
    failed = 0
    for name, got, want, ok in CHECKS:
        if not ok:
            failed += 1
            print(f"FAIL  {name:<{width}}  got {got!r}, want {want!r}")
    print(f"{len(CHECKS) - failed}/{len(CHECKS)} link checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
