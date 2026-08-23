"""Recompute document titles in a built index, without rebuilding it.

Titles are extracted at ingest and stored on every chunk. When the extraction
improves, the stored titles do not -- they stay whatever they were when the
index was built. Re-running ingest would fix them and re-embed 5,459 passages
to change a string, which is an hour of GPU for a display bug.

This reads each source document again, recomputes its title with the current
extractor, and rewrites metadata.json in place. Vectors, the FAISS index and
chunk_ids are untouched: chunk_id is the row position in metadata and the row
order never changes here.

    python src/retitle.py --data data --store vector_store
    python src/retitle.py --data data-birds --store store-birds --dry-run

A dry run prints what would change and writes nothing.
"""
import argparse
import json
import sys
from pathlib import Path

from loaders import extract_title, load_document

ROOT = Path(__file__).parent.parent


def recompute(data_dir: Path, store_dir: Path, dry_run: bool = False) -> int:
    meta_path = store_dir / "metadata.json"
    if not meta_path.exists():
        raise SystemExit(f"no index at {store_dir}")

    chunks = json.loads(meta_path.read_text(encoding="utf-8"))
    sources = sorted({c["source"] for c in chunks})

    fresh: dict[str, str] = {}
    for n, src in enumerate(sources, 1):
        path = data_dir / src
        if not path.exists():
            print(f"  [{n:>3}/{len(sources)}] {src} -- source missing, kept")
            continue
        try:
            units = load_document(path)
        except Exception as exc:  # noqa: BLE001 -- one bad file must not stop the rest
            print(f"  [{n:>3}/{len(sources)}] {src} -- {type(exc).__name__}, kept")
            continue
        fresh[src] = extract_title(units, path)

    # Report only what actually moved. A run that changes nothing should say so
    # rather than printing 36 unchanged lines.
    before = {c["source"]: c.get("title") for c in chunks}
    changed = {s: t for s, t in fresh.items() if t != before.get(s)}

    print(f"\n  {len(sources)} documents, {len(changed)} titles changed")
    for src, new in sorted(changed.items()):
        print(f"    {src}")
        print(f"      was: {before.get(src)!r}")
        print(f"      now: {new!r}")

    if not changed or dry_run:
        if dry_run and changed:
            print("\n  dry run -- nothing written")
        return len(changed)

    for c in chunks:
        if c["source"] in changed:
            c["title"] = changed[c["source"]]
    meta_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")

    # stats.json caches counts derived from this file; it is keyed on mtime, so
    # touching metadata.json alone would leave a cache that looks current.
    stats = store_dir / "stats.json"
    if stats.exists():
        stats.unlink()

    print(f"\n  wrote {meta_path}")
    return len(changed)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = args.data if args.data.is_absolute() else ROOT / args.data
    store = args.store if args.store.is_absolute() else ROOT / args.store
    recompute(data, store, args.dry_run)


if __name__ == "__main__":
    main()
