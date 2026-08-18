from pathlib import Path
import json
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from corpus_health import SUPPORTED, page_report, scan_unsupported, verdict
from embedding_cache import embed_with_cache
from loaders import extract_title, load_document

DATA_DIR = Path(__file__).parent.parent / "data"
STORE_DIR = Path(__file__).parent.parent / "vector_store"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunk size is measured in TOKENS, not words, because the encoder's input
# ceiling (max_seq_length) is a token count. Sizing in words let chunks run to
# ~1200 tokens, and everything past token 256 was silently dropped before the
# encoder saw it -- that text was stored and retrievable, but invisible to
# search.
#
# 210 rather than 240: each chunk is embedded with a title prefix (see
# embedding_text), which consumes part of the same 256-token budget. The
# remaining headroom covers the prefix plus the [CLS]/[SEP] tokens added at
# encode time.
CHUNK_SIZE_TOKENS = 210
CHUNK_OVERLAP_TOKENS = 40

# Bounds the embedding prefix. Long enough for a real paper title,
# short enough that it cannot crowd out the passage it labels.
MAX_TITLE_CHARS = 90


def extract_units(path: Path) -> list[dict]:
    """Format-appropriate citable units: PDF pages, slides, sections, row blocks."""
    return load_document(path)


def chunk_text(text: str, tokenizer, size: int, overlap: int) -> list[str]:
    """Split text into overlapping windows of `size` tokens.

    Windows are cut by token count but sliced out of the original string via
    character offsets, so the returned text is byte-identical to the source
    (Markdown structure included) rather than a lossy detokenization.
    """
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    if not offsets:
        return []

    chunks = []
    start = 0
    while start < len(offsets):
        window = offsets[start:start + size]
        piece = text[window[0][0]:window[-1][1]].strip()
        if piece:
            chunks.append(piece)
        if start + size >= len(offsets):
            break
        start += size - overlap
    return chunks


def embedding_text(title: str, locator: dict, chunk: str) -> str:
    """Text handed to the encoder: the chunk, prefixed with where it came from.

    A chunk mid-way through a paper rarely names the paper, so its vector
    encodes only local subject matter. Every transformer paper then looks alike
    -- which is exactly the observed failure: "what optimizer was used to train
    the Transformer" returned Vision Transformer, because nothing in the
    Attention paper's chunks says which Transformer they belong to.

    The prefix is deliberately short, and bounded. It has to shift the vector
    enough to carry document identity without dominating a 210-token chunk, and
    it spends part of the same 256-token budget -- an unbounded title from a
    long-winded document would push chunks over the encoder's ceiling and
    reintroduce the silent-truncation bug this project already fixed once.
    """
    short = title if len(title) <= MAX_TITLE_CHARS else title[:MAX_TITLE_CHARS].rstrip() + "..."
    return f"{short} ({locator['kind']} {locator['value']}). {chunk}"


def build_chunks(path: Path, tokenizer) -> tuple[list[dict], dict, tuple]:
    """Chunk one document, and report on whether it extracted usably.

    Returns (chunks, health_report, verdict). A document that extracted to
    almost nothing is reported rather than silently contributing empty chunks.
    Chunking stays inside a unit, so no chunk ever spans two citable locations --
    which is what keeps each chunk's citation unambiguous.
    """
    units = extract_units(path)
    report = page_report(units)
    status, reason = verdict(report)
    title = extract_title(units, path)

    chunks = []
    for unit in units:
        pieces = chunk_text(unit["text"], tokenizer, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS)
        for piece in pieces:
            chunks.append({
                "source": path.name,
                "title": title,
                "locator": unit["locator"],
                "text": piece,
                # What actually gets embedded. Kept separate from `text` so the
                # citation shows the passage as written, while the vector
                # carries the document identity the passage itself omits.
                "embed_text": embedding_text(title, unit["locator"], piece),
            })
    return chunks, report, (status, reason)


def _print_progress(event: dict) -> None:
    """Default reporter: one line per event, flushed immediately.

    Flushing matters. Indexing is the slowest thing this system does, and
    buffered output makes a long run indistinguishable from a hung one -- the
    caller has no way to tell whether to wait or intervene.
    """
    stage = event["stage"]
    if stage == "scan":
        print(f"Found {event['documents']} document(s) to index", flush=True)
    elif stage == "skip_unsupported":
        print(f"  SKIP  {event['document']:<40} {event['reason']}", flush=True)
    elif stage == "parse":
        pos = f"[{event['current']}/{event['total']}]"
        status = event["status"]
        label = f"{event['document'][:34]:<34}"
        if status == "FAIL":
            print(f"  {pos} FAIL  {label} {event['detail']}", flush=True)
        elif status == "WARN":
            print(f"  {pos} warn  {label} {event['detail']}", flush=True)
        else:
            print(f"  {pos} ok    {label} {event['units']:>3} units, "
                  f"{event['chunks']:>4} chunks", flush=True)
    elif stage == "embed":
        print(f"Embedding {event['chunks']} chunks "
              f"(max {event['max_tokens']}/{event['limit']} tokens)...", flush=True)
    elif stage == "cache":
        reused, computed = event["reused"], event["computed"]
        pct = 100 * reused / max(event["requested"], 1)
        print(f"  cache: {reused} reused ({pct:.0f}%), {computed} computed, "
              f"{event['entries']} entries / {event['megabytes']} MB", flush=True)
    elif stage == "done":
        print(f"Indexed {event['chunks']} chunks from {event['indexed']} document(s) "
              f"in {event['seconds']:.0f}s", flush=True)
        if event["skipped"]:
            print(f"SKIPPED {len(event['skipped'])}: {', '.join(event['skipped'])}",
                  flush=True)


def build_index(data_dir: Path = DATA_DIR, store_dir: Path = STORE_DIR,
                progress=None) -> dict:
    """Index every supported document in `data_dir`, reporting progress.

    `progress` receives structured events rather than formatted text, so a UI
    can drive a progress bar from the same source the CLI prints from -- there
    is no second code path to keep in step.
    """
    emit = progress or _print_progress
    started = time.perf_counter()

    docs = sorted(p for p in data_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in SUPPORTED)
    emit({"stage": "scan", "documents": len(docs)})
    if not docs:
        return {"chunks": 0, "indexed": 0, "skipped": [], "seconds": 0.0}

    # Loaded before chunking: the tokenizer defines the chunk boundaries.
    model = SentenceTransformer(EMBEDDING_MODEL)
    limit = model.max_seq_length

    # Files we would otherwise skip in silence.
    for path, reason in scan_unsupported(data_dir):
        emit({"stage": "skip_unsupported", "document": path.name, "reason": reason})

    all_chunks, skipped = [], []
    for i, doc_path in enumerate(docs, start=1):
        chunks, report, (status, reason) = build_chunks(doc_path, model.tokenizer)
        emit({
            "stage": "parse", "current": i, "total": len(docs),
            "document": doc_path.name, "status": status, "detail": reason,
            "units": report["pages"], "chunks": len(chunks),
        })
        if status == "FAIL":
            # Indexing this would add chunks that can never be retrieved, and
            # would quietly lower every metric with no visible cause.
            skipped.append(doc_path.name)
            continue
        all_chunks.extend(chunks)

    if not all_chunks:
        summary = {"chunks": 0, "indexed": 0, "skipped": skipped,
                   "seconds": time.perf_counter() - started}
        emit({"stage": "done", **summary})
        return summary

    texts = [c["embed_text"] for c in all_chunks]

    # Fail loudly rather than truncate silently: anything over the encoder's
    # ceiling would be dropped mid-chunk with no error at encode time.
    lengths = [len(model.tokenizer.encode(t)) for t in texts]
    over = [n for n in lengths if n > limit]
    if over:
        raise ValueError(
            f"{len(over)} chunk(s) exceed max_seq_length={limit} "
            f"(largest {max(over)}). Lower CHUNK_SIZE_TOKENS."
        )
    emit({"stage": "embed", "chunks": len(texts),
          "max_tokens": max(lengths), "limit": limit})
    # Only chunks whose text is new to the cache are actually encoded, so adding
    # one document to an existing corpus costs one document's worth of work.
    embeddings, cache_stats = embed_with_cache(texts, model, EMBEDDING_MODEL)
    emit({"stage": "cache", **cache_stats})
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    store_dir.mkdir(exist_ok=True)
    faiss.write_index(index, str(store_dir / "index.faiss"))
    with open(store_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    summary = {
        "chunks": len(all_chunks),
        "indexed": len(docs) - len(skipped),
        "skipped": skipped,
        "seconds": time.perf_counter() - started,
    }
    emit({"stage": "done", **summary})
    return summary


def main():
    build_index()


if __name__ == "__main__":
    main()
