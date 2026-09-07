# Security

## What this is

A demonstration and portfolio project, not a product. There is no release, no
versioning and no support commitment. Read what follows before running it
anywhere a stranger can reach.

## The server has no authentication

`src/serve.py` answers every request it receives. It has no users, no sessions
and no rate limit. Its defence is where it listens: `127.0.0.1` by default, so
a fresh clone serves nobody but the machine it runs on.

`RAG_HOST=0.0.0.0` and `RAG_PUBLIC=1` are the deliberate switches, and the
second one changes the startup line from a warning into a statement of what is
exposed. With `RAG_PUBLIC=1` set, the server also refuses the routes that
rebuild an index, declines to write prose with a generator, and reports a
failed call without the exception text, each until the matching
`RAG_ALLOW_` variable says otherwise. Those are guard rails on a public
instance, not a substitute for putting one behind something that authenticates.

## A retrieval system publishes its corpus

This is the constraint that matters most here and it is easy to miss. An
interface whose purpose is to show passage text verbatim will show the passages
it retrieves, to whoever is asking. Deploying one publishes some of its corpus
whatever the repository contains. `ATTRIBUTION.md` measures how much this
project's own recorded demo publishes and argues why that is quotation rather
than republication. Make the same decision deliberately before pointing this at
documents that are not yours to show.

## Keys

`RAG_GENERATOR=anthropic` reads `ANTHROPIC_API_KEY` from a `.env` file that
`.gitignore` has excluded since the first commit, and no key has ever been
committed. `RAG_GENERATOR=ollama` needs no key at all and is what every
measurement in this repository was produced with.

## Reporting something

Use GitHub's private vulnerability reporting on this repository for anything
that should not be public while it is unfixed. For everything else, including a
request to take a document out of the demo, open an issue.
