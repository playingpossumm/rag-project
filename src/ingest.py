from pathlib import Path
import json

import faiss
import numpy as np
import pymupdf4llm
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent.parent / "data"
STORE_DIR = Path(__file__).parent.parent / "vector_store"
CHUNK_SIZE_WORDS = 500
CHUNK_OVERLAP_WORDS = 50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    page_chunks = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    return [(i + 1, chunk["text"]) for i, chunk in enumerate(page_chunks)]


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        start += size - overlap
    return chunks


def build_chunks(pdf_path: Path) -> list[dict]:
    chunks = []
    for page_num, page_text in extract_pages(pdf_path):
        for piece in chunk_text(page_text, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS):
            chunks.append({
                "source": pdf_path.name,
                "page": page_num,
                "text": piece,
            })
    return chunks


def main():
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {DATA_DIR}")
        return

    all_chunks = []
    for pdf_path in pdf_files:
        print(f"Reading {pdf_path.name}...")
        all_chunks.extend(build_chunks(pdf_path))

    print(f"Built {len(all_chunks)} chunks from {len(pdf_files)} PDF(s)")

    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["text"] for c in all_chunks]
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
