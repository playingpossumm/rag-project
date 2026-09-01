"""Build a topic corpus: real PDFs from arXiv, real prose from Wikipedia.

Three corpora, so retrieval can be compared across subject matter rather than
tuned on one:

    llm     machine learning and NLP -- the existing corpus, extended
    quant   quantitative finance -- arXiv q-fin
    birds   ornithology -- Wikipedia, which is where the writing on birds is

**Papers are found by querying arXiv, never by hard-coded identifiers.** An id
recalled from memory is a coin flip, and a wrong one downloads a real paper with
a confidently wrong filename -- exactly the kind of error that survives review
because the file opens fine. The API returns the title, so the name on disk is
the name the paper actually has.

Birds is deliberately not arXiv. Ornithology barely appears there, and a corpus
padded with tangentially avian machine-learning papers would test nothing except
whether this script can count. Wikipedia is where the subject is actually
written down.

    .venv\\Scripts\\python.exe src\\fetch_topic.py llm   --into data --add 16
    .venv\\Scripts\\python.exe src\\fetch_topic.py quant --into data-quant --add 35
    .venv\\Scripts\\python.exe src\\fetch_topic.py birds --into data-birds --add 35
"""
import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
# Wikipedia asks for a descriptive agent and throttles hard without one. It
# served ten articles and then refused thirty in a row, which is rate limiting
# rather than a bad request -- hence the backoff in fetch().
UA = ("rag-project-educational/1.0 "
      "(https://github.com/playingpossumm/rag-project; personal learning project)")
ATOM = "{http://www.w3.org/2005/Atom}"

# Searches, not identifiers. Each is a real arXiv query string.
ARXIV = {
    "llm": "cat:cs.CL AND (abs:\"language model\" OR abs:\"transformer\" OR "
           "abs:\"retrieval\" OR abs:\"attention\")",
    "quant": "cat:q-fin.ST OR cat:q-fin.PM OR cat:q-fin.RM OR cat:q-fin.TR",
}

# Wikipedia articles for the bird corpus. Chosen to be topically adjacent --
# species, anatomy, behaviour, taxonomy -- for the same reason the ML corpus is
# all transformers: a corpus of unrelated subjects makes retrieval look good for
# the wrong reason, because telling physics from poetry is easy.
BIRDS = [
    "Bird", "Bird anatomy", "Bird flight", "Bird migration", "Bird vocalization",
    "Bird nest", "Bird egg", "Feather", "Beak", "Bird intelligence",
    "Passerine", "Corvidae", "Common raven", "Eurasian magpie", "New Caledonian crow",
    "Accipitridae", "Peregrine falcon", "Golden eagle", "Barn owl", "Strigiformes",
    "Anatidae", "Mallard", "Mute swan", "Charadriiformes", "Arctic tern",
    "Hummingbird", "Ruby-throated hummingbird", "Woodpecker", "Parrot", "Budgerigar",
    "Penguin", "Emperor penguin", "Ostrich", "Flightless bird", "Archaeopteryx",
    "Origin of birds", "Bird conservation", "Ornithology", "Birdwatching",
    "Dawn chorus (birds)",
]


def slug(text: str, limit: int = 58) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:limit].rstrip("_")


def fetch(url: str, timeout: int = 45, tries: int = 4) -> bytes:
    """GET with backoff. A throttled host answers the next request, not this one."""
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (429, 403, 503):
                raise
            time.sleep(2 ** attempt * 1.5)
        except Exception as exc:  # noqa: BLE001 - transient network
            last = exc
            time.sleep(2 ** attempt)
    raise last


# --------------------------------------------------------------- arXiv ----
def arxiv_search(query: str, want: int, skip: set[str]) -> list[tuple[str, str]]:
    """(pdf_url, title) for `want` papers, newest first, skipping known stems."""
    out, start = [], 0
    while len(out) < want and start < want * 6:
        url = ("http://export.arxiv.org/api/query?"
               + urllib.parse.urlencode({
                   "search_query": query, "start": start, "max_results": 40,
                   "sortBy": "submittedDate", "sortOrder": "descending"}))
        feed = ET.fromstring(fetch(url))
        entries = feed.findall(f"{ATOM}entry")
        if not entries:
            break
        for e in entries:
            title = " ".join(e.find(f"{ATOM}title").text.split())
            stem = slug(title)
            if stem in skip:
                continue
            pdf = next((l.get("href") for l in e.findall(f"{ATOM}link")
                        if l.get("title") == "pdf"), None)
            if not pdf:
                continue
            skip.add(stem)
            out.append((pdf, title))
            if len(out) >= want:
                break
        start += 40
        time.sleep(3)          # arXiv asks for one request every three seconds
    return out


# ----------------------------------------------------------- Wikipedia ----
def wiki_article(title: str) -> dict | None:
    """Plain-text extract plus section structure, via the REST API."""
    url = ("https://en.wikipedia.org/w/api.php?"
           + urllib.parse.urlencode({
               "action": "query", "prop": "extracts", "explaintext": 1,
               "titles": title, "format": "json", "redirects": 1}))
    pages = json.loads(fetch(url))["query"]["pages"]
    page = next(iter(pages.values()))
    text = page.get("extract") or ""
    if len(text) < 1200:
        return None
    return {"title": page.get("title", title), "text": text}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("topic", choices=["llm", "quant", "birds"])
    ap.add_argument("--into", required=True, help="directory to write into")
    ap.add_argument("--add", type=int, default=35, help="how many documents to add")
    args = ap.parse_args()

    into = (ROOT / args.into) if not Path(args.into).is_absolute() else Path(args.into)
    into.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in into.iterdir() if p.is_file()}
    print(f"{args.topic} -> {into}  ({len(have)} already there, adding {args.add})\n")

    written = 0
    if args.topic in ARXIV:
        for pdf_url, title in arxiv_search(ARXIV[args.topic], args.add, set(have)):
            dest = into / f"{slug(title)}.pdf"
            if dest.exists():
                continue
            try:
                dest.write_bytes(fetch(pdf_url))
            except Exception as exc:  # noqa: BLE001
                print(f"  skip  {title[:58]}  ({type(exc).__name__})")
                continue
            written += 1
            print(f"  {dest.stat().st_size // 1024:>5} KB  {dest.name}")
            time.sleep(3)
    else:
        from make_documents import write_mixed

        for title in BIRDS:
            if written >= args.add:
                break
            try:
                art = wiki_article(title)
            except Exception as exc:  # noqa: BLE001
                print(f"  skip  {title}  ({type(exc).__name__})")
                continue
            if not art:
                print(f"  skip  {title}  (too short to be worth indexing)")
                continue
            dest = write_mixed(art, into, written)
            if dest:
                written += 1
                print(f"  {dest.stat().st_size // 1024:>5} KB  {dest.name}")
            time.sleep(1.2)

    print(f"\nwrote {written} documents · {len(list(into.iterdir()))} total in {into}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
