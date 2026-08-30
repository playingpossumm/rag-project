# Attribution

**Every document in all three sets is listed in
[`docs/corpus-manifest.md`](docs/corpus-manifest.md)**, with filename, title,
format and passage count, generated from the built indexes. The demo cites its sources
by filename, and a citation you cannot look up is a dead end; a bibliography is
metadata, not redistribution.

## Ornithology — English Wikipedia

Built from English Wikipedia articles, converted into four file formats
(`.docx`, `.pptx`, `.xlsx`, `.pdf`) so the system's format handling is
exercised by something other than PDFs.

Those articles are published under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), which permits
redistribution with attribution and share-alike. Each passage the interface
shows names the file it came from, and every file is named after the Wikipedia
article it was built from. `peregrine_falcon.pdf` is
<https://en.wikipedia.org/wiki/Peregrine_falcon>, and so on.

`src/fetch_topic.py` records how each file was produced.

## ML & NLP papers, and quantitative finance — arXiv

36 arXiv cs.CL/cs.LG papers and 35 q-fin papers. **Neither set ships with
this repository.** Both were untracked on 2026-08-28, so a clone gives you the
code and not the corpus. `src/fetch_corpus.py` and
`src/fetch_topic.py` rebuild them by downloading from arXiv, so anyone running
this locally gets the papers from the people who published them.

**They are still in the git history, and that is a deliberate choice rather than
an oversight.** Removing them means rewriting history, which changes every
commit SHA; this repository cites fourteen of its own SHAs in prose, and one of
them is printed on screen by the server. All fourteen would dangle, and nothing
checks that a commit named in a document still resolves. Weighed against that,
arXiv's submission licence grants a non-exclusive right to distribute, many of
these papers are CC-BY, and a research repository carrying the papers it was
measured against is ordinary practice. The full reasoning is in
`docs/engineering-log.md` under 2026-08-28.

### What the recorded demo publishes, and why

The [recorded demo](https://rag-retrieval-visualized.vercel.app) answers all 157
evaluation questions, which means it carries the passages those answers stand
on. Measured on the current build:

| corpus | distinct passages shown | characters | documents |
|---|---|---|---|
| Ornithology | 549 | 141k | 43 |
| ML & NLP papers | 1,493 | 385k | 36 |
| Quantitative finance | 1,211 | 312k | 35 |

For the arXiv sets that is roughly 40 non-contiguous excerpts per paper,
averaging 257 characters each. That is about a quarter of a typical paper's
body text, in retrieval-rank order, with no figures, tables or reference lists.
Every excerpt is labelled with its source file and page.

The demo publishes **quotation with attribution**, not the papers. arXiv's
terms allow anyone to read and quote a submission, whatever licence its author
chose. They do not allow republishing one. A page showing forty short
fragments out of order, each captioned with where it came from, is not a copy
of a paper. It is a page of search results, which is what it is.

An earlier version of this file said both arXiv corpora were absent from the
demo as well as from the repository. That was the intention when it was written
and it stopped being true when the demo grew from ten bird questions to all 157;
the sentence is corrected here rather than quietly deleted, because a licensing
claim that was wrong for a while is worth leaving a record of.

**If you are an author of one of these papers and would rather not be in the
demo, open an issue and it comes out.** Rebuilding without a corpus is one
command: `python src/record_static.py --corpus birds --all`.

This distinction matters more for a retrieval system than for most software: a
RAG interface exists to show you passage text verbatim, so deploying one
publishes some of its corpus whatever the repository contains. That is worth
deciding deliberately rather than discovering.

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
