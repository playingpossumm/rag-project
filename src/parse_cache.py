"""Cache of parsed document units, keyed on file content.

The embedding cache removed the embedding cost from re-indexing, which exposed
parsing as the new bottleneck: 300 of the remaining 432 seconds. That is the
cost that actually hurts, because it is paid per document on every run even
when nothing about the document changed.

Keyed on a hash of the file's bytes rather than its path or modification time:

  * A renamed or moved file is recognised as the same content, so reorganising
    a folder does not trigger a full re-parse.
  * A file touched but not edited (which sync clients do constantly, and Google
    Drive especially) does not invalidate anything.
  * An edited file gets a new hash and is re-parsed, with no staleness window.

The cost is reading each file to hash it. That is milliseconds against seconds
of parsing, and for scanned documents where OCR runs it is nothing at all.

One JSON file per document rather than a single combined store: parsed output
is large (a 67-page paper is ~400KB of text), and a combined file would have to
be fully read and rewritten to add one document.
"""
import hashlib
import json
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "vector_store" / "parse_cache"

# Bumped when the parsing logic changes in a way that alters its output.
# Without this, a loader improvement would be invisible to every document
# already in the cache -- the old output would be served indefinitely, and the
# bug you just fixed would appear to persist.
PARSER_VERSION = 2


def file_key(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    digest.update(f"\x00v{PARSER_VERSION}".encode())
    return digest.hexdigest()[:32]


class ParseCache:
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.dir = cache_dir
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, path: Path) -> list[dict] | None:
        entry = self._path(file_key(path))
        if not entry.exists():
            self.misses += 1
            return None
        try:
            units = json.loads(entry.read_text(encoding="utf-8"))
            self.hits += 1
            return units
        except (json.JSONDecodeError, OSError):
            # A corrupt entry is discarded rather than trusted. Parsing again is
            # cheap; serving half a document silently is not.
            entry.unlink(missing_ok=True)
            self.misses += 1
            return None

    def put(self, path: Path, units: list[dict]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        target = self._path(file_key(path))
        # Write to a temporary file and replace, so an interrupted run cannot
        # leave a truncated entry that later reads as a valid short document.
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(units, ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)

    def prune(self, keep_keys: set[str]) -> int:
        """Delete entries for documents no longer in the corpus."""
        if not self.dir.exists():
            return 0
        removed = 0
        for entry in self.dir.glob("*.json"):
            if entry.stem not in keep_keys:
                entry.unlink(missing_ok=True)
                removed += 1
        return removed

    def stats(self) -> dict:
        size = sum(f.stat().st_size for f in self.dir.glob("*.json")) if self.dir.exists() else 0
        return {
            "parse_hits": self.hits,
            "parse_misses": self.misses,
            "parse_cache_mb": round(size / 1_000_000, 1),
        }
