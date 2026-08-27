"""Which corpora exist, and what travels with each one.

A corpus is not just a folder of documents. It is a folder, the index built from
it, the questions it is scored against, and **its own abstention threshold** --
and all four have to move together or the system quietly serves one corpus's
answers under another's assumptions.

The threshold is the part that surprised this project. It was a module constant
for most of the project's life, recalibrated three times, with a comment
guessing it was "a property of the data, not of the model". Measured across two
corpora on 2026-08-21 that guess is confirmed and the size of it is the point:

                        answerable median   wrongly refused at 0.0
    ML papers   (36)            +4.93        1 of 66   ( 1.5%)
    ornithology (45)            +1.82       10 of 26   (38.5%)

The same number refuses 1.5% of answerable questions on one corpus and 38.5% on
another. So it is registered here, per corpus, next to the documents it was
derived from -- not set once in a constant that the next corpus inherits by
accident.

A corpus with `"threshold": null` has not been calibrated. That is recorded
honestly rather than defaulted silently, because a borrowed threshold is exactly
the failure this file exists to prevent; callers get 0.0 and a flag saying it is
a guess.
"""
import json
import os
from pathlib import Path

# The fallback for a corpus that does not name its own, kept in one place so
# this module and retrieve.py cannot disagree about what "default" means.
CANDIDATE_K = 20

ROOT = Path(__file__).parent.parent
CONFIG = ROOT / "corpora.json"

# Used when corpora.json is absent, so a fresh clone with one index still works.
FALLBACK = {
    "default": {"label": "Corpus", "data": "data", "store": "vector_store",
                "golden": "eval/golden_set.json", "threshold": 0.0},
}


def _resolve(p: str | None) -> Path | None:
    if not p:
        return None
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def registry() -> dict[str, dict]:
    """Every configured corpus, whether or not it has been indexed yet."""
    raw = json.loads(CONFIG.read_text(encoding="utf-8")) if CONFIG.exists() else FALLBACK
    out = {}
    for name, cfg in raw.items():
        store = _resolve(cfg.get("store"))
        out[name] = {
            "name": name,
            "label": cfg.get("label", name),
            "data": _resolve(cfg.get("data")),
            "store": store,
            "golden": _resolve(cfg.get("golden")),
            # A threshold of null means nobody has calibrated this corpus. The
            # distinction between "0.0 because it was measured" and "0.0 because
            # nothing better was available" is the whole point of this module.
            "threshold": cfg.get("threshold"),
            "calibrated": cfg.get("threshold") is not None,
            # How much of the first stage's ordering survives reranking. Like
            # the threshold, it was measured per corpus and does not transfer:
            # 0.2 is worth a question and four points of MRR on the bird set
            # and costs a question on each of the other two.
            "rerank_blend": float(cfg.get("rerank_blend") or 0.0),
            # How many candidates the reranker sees. Measured per corpus for
            # the same reason as the two above: 16 is better or equal on every
            # metric for the ML papers and quant and costs the bird corpus 2.6
            # points of MRR, so as one global number it reads as a trade and
            # per corpus it is not one. Reranking is 92-95% of query latency,
            # so this is also the only latency dial that matters.
            "candidate_k": int(cfg.get("candidate_k") or CANDIDATE_K),
            "indexed": bool(store and (store / "metadata.json").exists()),
        }
    return out


def available() -> dict[str, dict]:
    """Only the corpora that actually have an index on disk."""
    return {n: c for n, c in registry().items() if c["indexed"]}


def active_name() -> str:
    """The corpus to serve. RAG_CORPUS wins; otherwise the first indexed one."""
    want = os.environ.get("RAG_CORPUS")
    reg = registry()
    if want and want in reg:
        return want
    ready = available()
    return next(iter(ready), next(iter(reg)))


def get(name: str) -> dict:
    reg = registry()
    if name not in reg:
        raise ValueError(f"unknown corpus {name!r}; have {sorted(reg)}")
    return reg[name]


# metadata.json is 9 MB on the largest corpus, so counting documents by parsing
# it is not something to do while drawing a menu. The counts are derived once and
# cached beside the index; the cache is rebuilt whenever metadata.json is newer,
# so a re-ingest cannot leave a stale document count on screen.
def stats(cfg: dict) -> dict:
    """How big a corpus is, and which file formats it is made of."""
    if not cfg["indexed"]:
        return {}
    meta = cfg["store"] / "metadata.json"
    cache = cfg["store"] / "stats.json"
    if cache.exists() and cache.stat().st_mtime >= meta.stat().st_mtime:
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass  # a truncated cache is rebuilt, not fatal

    chunks = json.loads(meta.read_text(encoding="utf-8"))
    sources = {c["source"] for c in chunks}
    formats: dict[str, int] = {}
    for src in sources:
        ext = src.rsplit(".", 1)[-1].lower() if "." in src else "other"
        formats[ext] = formats.get(ext, 0) + 1
    out = {
        "documents": len(sources),
        "chunks": len(chunks),
        # Sorted by count so the dominant format reads first, then by name so the
        # order does not change between runs on a tie.
        "formats": dict(sorted(formats.items(), key=lambda kv: (-kv[1], kv[0]))),
    }
    try:
        cache.write_text(json.dumps(out, indent=2), encoding="utf-8")
    except OSError:
        pass  # a read-only store still gets correct numbers, just not cached
    return out


def describe(cfg: dict) -> dict:
    """The shape the UI needs: paths flattened, nothing that leaks a filesystem."""
    return {
        "name": cfg["name"],
        "label": cfg["label"],
        "indexed": cfg["indexed"],
        "calibrated": cfg["calibrated"],
        "threshold": cfg["threshold"] if cfg["calibrated"] else 0.0,
        **stats(cfg),
    }
