"""Content-addressed cache of chunk embeddings.

Re-indexing currently re-embeds everything. On this corpus that is ~2,400 chunks
and about six minutes; adding a twenty-first document to twenty repeats the work
for all twenty. Since the corpus is expected to grow document by document, that
cost lands on every single addition.

Embeddings are a pure function of (text, model), so they cache perfectly. Keying
on a hash of both means:

  * unchanged chunks are never re-embedded, no matter which document they are in;
  * an edited document only re-embeds the chunks that actually changed, because
    chunk boundaries are stable outside the edit;
  * changing the embedding model invalidates the cache automatically rather than
    silently mixing vectors from two different spaces, which would be
    catastrophic and completely invisible -- distances between them are
    computable but meaningless.

Storage is one .npy matrix plus a JSON key->row map, rather than a per-key file
or a dict-of-arrays .npz. Thousands of tiny files are slow on Windows, and .npz
with thousands of keys is slow to open; a single contiguous matrix loads in one
read and slices for free.

This is the same idea as LangChain's CacheBackedEmbeddings and LlamaIndex's
IngestionCache, kept local so it stays inspectable and has no extra dependency.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

# The default is the ML corpus's store, which is where this lived when there
# was only one corpus. Callers that know their corpus pass its store instead --
# ingest.build_index() does -- so each corpus's derived files sit together, the
# same rule corpora.py states for the index and the golden set.
CACHE_DIR = Path(__file__).parent.parent / "vector_store"
KEYS_FILE = CACHE_DIR / "embedding_cache_keys.json"
VECS_FILE = CACHE_DIR / "embedding_cache_vectors.npy"


def key_for(text: str, model_name: str) -> str:
    """Stable identifier for one (text, model) pair.

    The model name is part of the key, not stored alongside it, so vectors from
    different models can never collide on the same key.
    """
    digest = hashlib.sha256(f"{model_name}\x00{text}".encode("utf-8")).hexdigest()
    return digest[:32]


class EmbeddingCache:
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.keys_file = cache_dir / KEYS_FILE.name
        self.vecs_file = cache_dir / VECS_FILE.name
        self.keys: dict[str, int] = {}
        self.vectors: np.ndarray | None = None
        self._load()

    def _load(self) -> None:
        if self.keys_file.exists() and self.vecs_file.exists():
            try:
                self.keys = json.loads(self.keys_file.read_text(encoding="utf-8"))
                self.vectors = np.load(self.vecs_file)
                # A truncated write, an interrupted run, or a hand-edited file
                # would desynchronise the two. Distrust rather than repair: a
                # wrong cache silently returns another chunk's vector, which is
                # far worse than paying to rebuild it.
                if len(self.keys) != len(self.vectors):
                    self.keys, self.vectors = {}, None
            except (json.JSONDecodeError, ValueError, OSError):
                self.keys, self.vectors = {}, None

    def get_many(self, texts: list[str], model_name: str) -> tuple[list[int], list[str]]:
        """Split texts into (indices already cached, texts still to embed)."""
        hits, misses = [], []
        for i, text in enumerate(texts):
            if key_for(text, model_name) in self.keys:
                hits.append(i)
            else:
                misses.append(i)
        return hits, misses

    def lookup(self, text: str, model_name: str) -> np.ndarray | None:
        row = self.keys.get(key_for(text, model_name))
        if row is None or self.vectors is None:
            return None
        return self.vectors[row]

    def add_many(self, texts: list[str], vectors: np.ndarray, model_name: str) -> None:
        if len(texts) == 0:
            return
        new_rows = []
        start = 0 if self.vectors is None else len(self.vectors)
        for offset, text in enumerate(texts):
            k = key_for(text, model_name)
            if k not in self.keys:
                self.keys[k] = start + len(new_rows)
                new_rows.append(vectors[offset])
        if not new_rows:
            return
        stacked = np.vstack(new_rows).astype(np.float32)
        self.vectors = stacked if self.vectors is None else np.vstack([self.vectors, stacked])

    def save(self) -> None:
        if self.vectors is None:
            return
        self.vecs_file.parent.mkdir(exist_ok=True)
        # Vectors first. If the process dies between the two writes, a keys file
        # referencing rows that do not exist is the dangerous ordering; extra
        # unreferenced vectors are merely wasted space.
        np.save(self.vecs_file, self.vectors)
        self.keys_file.write_text(json.dumps(self.keys), encoding="utf-8")

    def stats(self) -> dict:
        return {
            "entries": len(self.keys),
            "megabytes": round(self.vectors.nbytes / 1_000_000, 1) if self.vectors is not None else 0.0,
        }


def embed_with_cache(texts: list[str], model, model_name: str,
                     cache: EmbeddingCache | None = None,
                     progress=None) -> tuple[np.ndarray, dict]:
    """Embed `texts`, computing only what is not already cached.

    Returns (vectors in the original order, stats about the cache hit rate).
    """
    cache = cache or EmbeddingCache()
    _, misses = cache.get_many(texts, model_name)

    if misses:
        # Resolved only here: `model` may be a lazy handle, and when every text
        # is already cached the model is never needed at all.
        model = model.get() if hasattr(model, "get") else model
        if progress:
            progress({"stage": "embed", "chunks": len(misses),
                      "cached": len(texts) - len(misses), "total": len(texts)})
        missing_texts = [texts[i] for i in misses]
        fresh = model.encode(missing_texts, show_progress_bar=True, convert_to_numpy=True)
        cache.add_many(missing_texts, fresh, model_name)
        cache.save()

    dim = cache.vectors.shape[1]
    out = np.empty((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        out[i] = cache.lookup(text, model_name)

    return out, {
        "requested": len(texts),
        "computed": len(misses),
        "reused": len(texts) - len(misses),
        **cache.stats(),
    }
