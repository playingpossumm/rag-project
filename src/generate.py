import sys

import anthropic
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from retrieve import EMBEDDING_MODEL, load_index, search

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

CLAUDE_MODEL = "claude-opus-5"

SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided context excerpts. "
    "For every claim, cite the source and page number in the format [source, page X]. "
    "If the context does not contain the answer, say so plainly instead of guessing."
)


def build_context(chunks: list[dict]) -> str:
    parts = [f"[{c['source']}, page {c['page']}]\n{c['text']}" for c in chunks]
    return "\n\n---\n\n".join(parts)


def main():
    query = " ".join(sys.argv[1:]) or input("Ask a question: ")

    index, metadata = load_index()
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    chunks = search(query, index, metadata, embed_model)

    context = build_context(chunks)
    client = anthropic.Anthropic()

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        output_config={"effort": "low"},
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Context:\n\n{context}\n\nQuestion: {query}",
        }],
    )

    answer = next(b.text for b in response.content if b.type == "text")
    print(f"\n{answer}\n")
    print("Sources consulted:")
    for c in chunks:
        print(f"  - {c['source']}, page {c['page']}")


if __name__ == "__main__":
    main()
