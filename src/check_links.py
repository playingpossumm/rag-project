"""Do the links the interface serves actually go anywhere?

The fifth guard, and it exists because of a link that would have failed twice
over. `/about` pointed at

    https://github.com/ArdellAlfatih/rag-project/blob/main/docs/corpus-manifest.md

and this repository's only branch is `master`. The repository is also still
private, so the link 404s today for a different reason, and making it public
would have fixed the visible symptom while leaving the branch wrong. One
failure hiding behind another is exactly the shape this project keeps finding,
so the branch half is now checked offline, where it is decidable.

Three things are checked, all without a network:

**Internal links** must be routes `serve.py` actually dispatches. The route
list is read out of the dispatch by `check_docs.served_routes()` rather than
repeated here, because a list typed in two places is a list that disagrees in
one of them.

**Anchors** (`/about#corpora`) must name an `id` that exists in the target
page. An anchor to a missing id does not error; it silently lands at the top
of the page, which is worse than an error because it looks like it worked.

**Links into this repository** must name this repository, then a branch that
exists and a file that exists. All three are decidable from the working tree and
from `git remote get-url origin`, so none of them needs the repository to be
public or the network to be up.

The repository half was added on 2026-09-01, after every Source link on the site
was found to name `ArdellAlfatih/rag-project` while the remote is
`playingpossumm/rag-project`. The checker had matched the URL against a pattern
named for this repository, taken the match as proof it was this repository, and
then checked the branch and the file against the local working tree, which
agreed. Four links 404 and the guard passed. A link is only checked against the
working tree when the owner and the repository name both match the remote.

`--http` additionally asks the network about every external link, which
requires the repository to be public to pass and is therefore not part of the
default run.

    .venv\\Scripts\\python.exe src\\check_links.py
    .venv\\Scripts\\python.exe src\\check_links.py --http
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).parent))

from check_docs import served_routes  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
UI = ROOT / "ui"

HREF = re.compile(r'href="([^"]+)"')
SELF_REPO = re.compile(
    r"^https://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)"
    r"(?:/(?:blob|tree)/(?P<ref>[^/]+)/(?P<path>.+))?/?$")

# Pages that are not routes: the temporary diagram comparison and anything else
# served only as a file. Kept explicit so removing one is a decision.
FILE_PAGES = {"/versions"}


def hrefs() -> dict[str, set[str]]:
    """Every href in the interface, mapped to the files that carry it."""
    out: dict[str, set[str]] = {}
    for f in sorted([*UI.glob("*.html"), *UI.glob("*.js")]):
        for url in HREF.findall(f.read_text(encoding="utf-8")):
            out.setdefault(url, set()).add(f.name)
    return out


def page_ids(page: str) -> set[str] | None:
    """The ids in a page, so an anchor can be checked against them."""
    f = UI / f"{page.strip('/') or 'index'}.html"
    if not f.exists():
        return None
    return set(re.findall(r'id="([^"]+)"', f.read_text(encoding="utf-8")))


def remote_slug() -> tuple[str, str] | None:
    """The owner and repository name of `origin`, or None when there is no remote.

    A link that names a different repository is not checkable against this
    working tree, and checking it anyway is how four broken links passed.
    """
    url = git("remote", "get-url", "origin")
    if not url:
        return None
    m = re.search(r"github\.com[:/]([^/]+)/(.+?)(?:\.git)?/?$", url.strip())
    return (m.group(1), m.group(2)) if m else None


def git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                           text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--http", action="store_true",
                    help="also ask the network about external links")
    args = ap.parse_args()

    problems: list[str] = []
    notes: list[str] = []
    links = hrefs()
    if not links:
        print("no hrefs found in ui/ -- this checker looks for href=\"...\" "
              "and found none, which is a failure rather than a pass")
        return 1

    slug = remote_slug()
    routes = served_routes()
    # serve.py compares the request path with the trailing slash stripped, so
    # the route the documents call "/" is the empty string in the dispatch.
    routes = {r or "/" for r in routes}

    for url, files in sorted(links.items()):
        where = ", ".join(sorted(files))

        if url.startswith("#"):
            continue

        if url.startswith("/"):
            path, _, frag = url.partition("#")
            path = path or "/"
            if path not in routes and path not in FILE_PAGES:
                problems.append(f"{url} ({where}) is not a route serve.py dispatches")
            else:
                notes.append(f"{url} -> served")
            if frag:
                ids = page_ids(path)
                if ids is None:
                    problems.append(f"{url} ({where}) targets a page that is "
                                    f"not a file in ui/")
                elif frag not in ids:
                    problems.append(f'{url} ({where}) targets #{frag}, and no '
                                    f'element in {path} has that id')
                else:
                    notes.append(f"{url} -> id present")
            continue

        m = SELF_REPO.match(url)
        if not m:
            notes.append(f"{url} -> external, not checked offline")
            continue

        # Does it name this repository? Everything below reads the local
        # working tree, so answering that first is what makes the rest mean
        # anything.
        owner, repo = m.group("owner"), m.group("repo").removesuffix(".git")
        if slug is None:
            notes.append(f"{url} -> no git remote, repository not checked")
        elif (owner, repo) != slug:
            if repo == slug[1]:
                problems.append(
                    f"{url} ({where}) names '{owner}/{repo}', and this "
                    f"repository's origin is '{slug[0]}/{slug[1]}'. Every link "
                    f"built on that owner points at a repository that is not "
                    f"this one")
            else:
                notes.append(f"{url} -> another repository, not checked offline")
            continue

        # A link into this repository. The branch and the path are both
        # decidable here, whether or not the repository is public.
        ref, path = m.group("ref"), m.group("path")
        if ref:
            if git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}") is None:
                branches = git("for-each-ref", "--format=%(refname:short)",
                               "refs/heads") or "?"
                problems.append(
                    f"{url} ({where}) names branch '{ref}', which does not "
                    f"exist. Branches here: {branches.replace(chr(10), ', ')}")
            elif not (ROOT / unquote(path)).exists():
                problems.append(f"{url} ({where}) names '{path}', which is not "
                                f"a file in this repository")
            else:
                notes.append(f"{url} -> branch '{ref}' and path both exist")
        else:
            notes.append(f"{url} -> repository root")

    if args.http:
        import urllib.error
        import urllib.request
        for url in sorted(u for u in links if u.startswith("http")):
            req = urllib.request.Request(url, method="HEAD",
                                         headers={"User-Agent": "check_links"})
            try:
                with urllib.request.urlopen(req, timeout=15) as r:
                    code = r.status
            except urllib.error.HTTPError as e:
                code = e.code
            except Exception as e:                                # noqa: BLE001
                problems.append(f"{url} could not be reached: {e}")
                continue
            if code >= 400:
                problems.append(f"{url} answers HTTP {code}")
            else:
                notes.append(f"{url} -> HTTP {code}")

    for n in notes:
        print(f"  {n}")
    if problems:
        print(f"\n{len(problems)} broken link(s):")
        for p in problems:
            print(f"  {p}")
        return 1
    print(f"\nevery link the interface serves resolves: "
          f"{len(links)} distinct targets across ui/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
