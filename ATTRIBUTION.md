# Attribution

## The demo corpus

The public demo serves an **ornithology** corpus built from English Wikipedia
articles, converted into four file formats (`.docx`, `.pptx`, `.xlsx`, `.pdf`)
so the system's format handling is exercised by something other than PDFs.

Those articles are published under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), which permits
redistribution with attribution and share-alike. Each passage the interface
shows names the file it came from, and every file is named after the Wikipedia
article it was built from — `peregrine_falcon.pdf` is
<https://en.wikipedia.org/wiki/Peregrine_falcon>, and so on.

`src/fetch_topic.py` records how each file was produced.

## Corpora that are *not* published

Two other corpora exist in development and are deliberately absent from both
this repository and the deployed demo:

| corpus | source | why it is not published |
|---|---|---|
| ML & NLP papers | 36 arXiv papers | arXiv submissions carry varied licences; many permit reading but not redistribution |
| Quantitative finance | 35 arXiv q-fin papers | the same |

They were removed from git history on 2026-08-28 for this reason, not for size.
Both are rebuildable locally with `python src/fetch_corpus.py` and
`python src/fetch_topic.py`, which download from the original sources rather
than from here.

This distinction matters more for a retrieval system than for most software: a
RAG interface exists to show you passage text verbatim, so publishing one
publishes its corpus, whatever the repository contains.

## Models

Neither model is redistributed here; both are downloaded from Hugging Face at
build time.

| | model | licence |
|---|---|---|
| bi-encoder | `sentence-transformers/all-MiniLM-L6-v2` | Apache-2.0 |
| cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Apache-2.0 |
| second embedder (quant corpus only) | `BAAI/bge-small-en-v1.5` | MIT |
| local generation (optional) | whatever Ollama serves, default `llama3.2` | Meta Llama 3.2 Community Licence |

## This project

MIT, see `LICENSE`.
