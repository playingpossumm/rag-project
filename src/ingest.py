from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from corpus_health import SUPPORTED, page_report, scan_unsupported, verdict
from loaders import load_document

DATA_DIR = Path(__file__).parent.parent / "data"
STORE_DIR = Path(__file__).parent.parent / "vector_store"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunk size is measured in TOKENS, not words, because the encoder's input
# ceiling (max_seq_length) is a token count. Sizing in words let chunks run to
# ~1200 tokens, and everything past token 256 was silently dropped before the
# encoder saw it -- that text was stored and retrievable, but invisible to
# search. 240 leaves headroom under the 256 limit for the [CLS]/[SEP] tokens
# the tokenizer adds at encode time.
CHUNK_SIZE_TOKENS = 240
CHUNK_OVERLAP_TOKENS = 40


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

    chunks = []
    for unit in units:
        pieces = chunk_text(unit["text"], tokenizer, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS)
        for piece in pieces:
            chunks.append({
                "source": path.name,
                "locator": unit["locator"],
                "text": piece,
            })
    return chunks, report, (status, reason)


def main():
    docs = sorted(p for p in DATA_DIR.iterdir()
                  if p.is_file() and p.suffix.lower() in SUPPORTED)
    if not docs:
        print(f"No supported documents found in {DATA_DIR}")
        return

    # Loaded before chunking: the tokenizer defines the chunk boundaries.
    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    limit = model.max_seq_length

    # Files we would otherwise skip in silence.
    unsupported = scan_unsupported(DATA_DIR)
    if unsupported:
        print(f"\n{len(unsupported)} file(s) in data/ will NOT be indexed:")
        for path, reason in unsupported:
            print(f"  SKIP  {path.name:<40} {reason}")
        print()

    all_chunks, skipped = [], []
    for doc_path in docs:
        chunks, report, (status, reason) = build_chunks(doc_path, model.tokenizer)
        label = f"{doc_path.name[:34]:<34}"

        if status == "FAIL":
            # Indexing this would add chunks that can never be retrieved, and
            # would quietly lower every metric with no visible cause.
            print(f"  FAIL  {label} {reason}")
            skipped.append(doc_path.name)
            continue
        if status == "WARN":
            print(f"  warn  {label} {reason}")
        else:
            print(f"  ok    {label} {report['pages']:>3} pages, "
                  f"{report['median_words']:>4} median words/page")
        all_chunks.extend(chunks)

    indexed = len(docs) - len(skipped)
    print(f"\nBuilt {len(all_chunks)} chunks from {indexed} document(s)")
    if skipped:
        print(f"SKIPPED {len(skipped)}: {', '.join(skipped)}")
        print("These need OCR before they can contribute anything to retrieval.")
    if not all_chunks:
        print("Nothing to index -- aborting rather than writing an empty store.")
        return

    texts = [c["text"] for c in all_chunks]

    # Fail loudly rather than truncate silently: anything over the encoder's
    # ceiling would be dropped mid-chunk with no error at encode time.
    lengths = [len(model.tokenizer.encode(t)) for t in texts]
    over = [n for n in lengths if n > limit]
    if over:
        raise ValueError(
            f"{len(over)} chunk(s) exceed max_seq_length={limit} "
            f"(largest {max(over)}). Lower CHUNK_SIZE_TOKENS."
        )
    print(f"Token budget OK: max {max(lengths)}/{limit}, median {sorted(lengths)[len(lengths) // 2]}")

    print("Embedding chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    STORE_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(STORE_DIR / "index.faiss"))
    with open(STORE_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved index and metadata to {STORE_DIR}")


if __name__ == "__main__":
    main()
