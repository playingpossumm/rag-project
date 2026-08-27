"""Does indexing one corpus leave the other corpora's caches alone?

It did not. `ParseCache()` and `embed_with_cache()` both defaulted to a module
constant pointing at `vector_store/` -- the ML corpus's store -- whatever was
being indexed. Content-addressed caches can be shared safely, so that alone was
untidy rather than wrong.

`prune()` made it wrong. It deletes every entry whose key is not in the set it
is given, and that set holds only the documents of the corpus in hand. So
indexing the birds evicted the parse cache for the ML papers and for quant, and
their next re-index re-parsed every document from scratch -- against HANDOFF §2's
"re-index nothing changed **0.76 s**", which holds only if no other corpus has
been indexed in between. This repo has three.

Measured before the fix: `vector_store/parse_cache` held 90 entries, all 45 bird
documents under two key forms, and nothing for the 36 ML documents or the 35
quant ones.

Real files and the real `build_index`, because the defect is entirely about
which directory each cache chooses -- a stub would have to decide that itself
and would therefore assert its own opinion. Two tiny corpora of plain text, so
the embedder has almost nothing to do.

    .venv\\Scripts\\python.exe src\\test_ingest_cache.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHECKS: list[tuple[str, object, object, bool]] = []


def check(name: str, got, want) -> None:
    CHECKS.append((name, got, want, got == want))


def entries(store: Path) -> set[str]:
    cache = store / "parse_cache"
    return {p.name for p in cache.glob("*.json")} if cache.exists() else set()


def main() -> int:
    import ingest

    tmp = Path(tempfile.mkdtemp(prefix="ingest-cache-"))
    try:
        # .txt is not in SUPPORTED, so the documents have to be a format the
        # loaders read. docx is the cheapest to write.
        from docx import Document

        def write_docx(path: Path, subject: str):
            """A document long enough to survive chunking.

            The first fixture here was a heading and one sentence, and
            build_index stopped after parsing: headings are not indexed as
            passages, so the document produced no chunks and there was nothing
            to embed. The stages emitted were scan, parse, done -- no cache
            event at all -- which read as four mysterious assertion failures
            rather than as "your test corpus is empty".
            """
            doc = Document()
            doc.add_heading("Overview", level=1)
            for i in range(12):
                doc.add_paragraph(
                    f"Paragraph {i}. {subject} This sentence exists to give the "
                    f"chunker something to cut, and repeats the subject so that "
                    f"retrieval over this corpus would have a term to match. "
                    f"Anatomy, flight, physiology, and the structures involved "
                    f"are described here in enough words to make a passage.")
            doc.save(path)

        a_dir, b_dir = tmp / "corpus-a", tmp / "corpus-b"
        a_dir.mkdir(parents=True)
        b_dir.mkdir(parents=True)
        write_docx(a_dir / "alpha.docx", "Alpha discusses hollow bones.")
        write_docx(b_dir / "beta.docx", "Beta discusses the syrinx.")

        a_store, b_store = tmp / "store-a", tmp / "store-b"

        ingest.build_index(a_dir, a_store, progress=lambda e: None)
        after_a = entries(a_store)
        check("indexing a corpus writes a parse cache into its own store",
              len(after_a) > 0, True)

        ingest.build_index(b_dir, b_store, progress=lambda e: None)
        check("and the second corpus writes into its own store, not the first's",
              len(entries(b_store)) > 0, True)

        # The defect: prune() runs over whatever directory the cache is in, and
        # deletes everything that is not the corpus in hand.
        check("indexing the second corpus leaves the first's cache untouched",
              entries(a_store), after_a)

        # The embedding cache travels with the store too, or deleting one
        # corpus's store would silently orphan vectors belonging to another.
        for store in (a_store, b_store):
            check(f"{store.name} holds its own embedding cache",
                  (store / "embedding_cache_keys.json").exists(), True)

        # What the caches are for: a re-index that changes nothing re-parses
        # nothing. This is the claim HANDOFF quotes as 0.76 s.
        before = entries(a_store)
        hits = {}

        def record(event):
            if event.get("stage") == "cache":
                hits.update(event)

        ingest.build_index(a_dir, a_store, progress=record)
        check("re-indexing an unchanged corpus parses nothing again",
              hits.get("parse_misses"), 0)
        check("and embeds nothing again", hits.get("misses", 0), 0)
        check("and its cache is still there afterwards", entries(a_store), before)

        # And a changed document is the one thing that must miss.
        write_docx(a_dir / "alpha.docx", "Alpha now discusses the furcula.")
        hits.clear()
        ingest.build_index(a_dir, a_store, progress=record)
        check("a changed document is re-parsed", hits.get("parse_misses", 0) > 0, True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    width = max(len(n) for n, *_ in CHECKS)
    failed = 0
    for name, got, want, ok in CHECKS:
        if not ok:
            failed += 1
            print(f"FAIL  {name:<{width}}  got {got!r}, want {want!r}")
    print(f"{len(CHECKS) - failed}/{len(CHECKS)} ingest cache checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
