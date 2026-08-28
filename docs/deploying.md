# Deploying the demo

## Why not Vercel

Vercel was the first idea and it does not work, for reasons of arithmetic rather
than taste. This process holds a FAISS index, a bi-encoder and a cross-encoder in
memory and answers from them:

| | this system | Vercel Python function |
|---|---|---|
| dependencies | **1,470 MB** (torch alone is 514 MB) | 250 MB limit |
| cold start | **~30 s** to load two models | 10 s timeout (Hobby) |
| memory between requests | required — the index stays loaded | none; every request starts cold |

Serverless is the wrong shape for this, not a tight fit. A function that reloads
half a gigabyte of torch to answer one question in a second would be slower and
more expensive than the laptop it was developed on.

**The same is true of Netlify Functions, Cloudflare Workers and Lambda at its
default limits.** What this needs is an ordinary long-lived process.

## What does work

| platform | why | notes |
|---|---|---|
| **Hugging Face Spaces** | built for exactly this: ML models, a persistent container, free CPU tier with enough RAM | Docker SDK, listens on **7860** — which is why the `Dockerfile` defaults there |
| **Fly.io** | small always-on VM, scales to zero if you accept a cold start | `fly launch` reads the Dockerfile; give it 2 GB |
| **Render / Railway** | the same shape, one-click from a repository | free tiers sleep, so the first visitor after idle waits ~40 s |

Hugging Face Spaces is the recommendation: the free CPU tier has the RAM, the
cold start is acceptable, and the audience for a project like this is already
there.

### If you want it on Vercel anyway

One honest option remains: **a static demo**. Every question the front page
offers has a recorded trace — that is what `eval/analytics.json` and
`eval/per_case*.json` already hold — so a build step could render those to
static pages. You would lose live querying and keep the thing the project is
actually about, which is showing the retrieval process rather than the answer.
Nothing here does that yet.

## Before you deploy

The repository contains **no documents and no index** — they were removed from
git history because publishing them would redistribute other people's work (see
`ATTRIBUTION.md`). So the corpus has to be built once, locally, before the
image can be built:

```bash
python src/fetch_topic.py --topic birds        # Wikipedia -> data-birds/
RAG_STORE_DIR=store-birds RAG_DATA_DIR=data-birds python src/ingest.py
python src/check_golden.py --corpus birds      # labels still describe the corpus
```

That produces `data-birds/` and `store-birds/`, which the `Dockerfile` copies in.
Building the index inside the image instead would work, but it downloads from
Wikipedia at build time and makes the build non-reproducible.

**Only the ornithology corpus is deployable.** It is Wikipedia under CC BY-SA
and redistributable with attribution. The two arXiv corpora are not, and a RAG
interface publishes its corpus by design — it exists to show passage text
verbatim.

## Build and run

```bash
docker build -t rag-demo .
docker run -p 7860:7860 rag-demo
# then http://localhost:7860/
```

### Configuration

| variable | default | what it does |
|---|---|---|
| `PORT` | 7860 | the port to bind; container platforms set this |
| `RAG_HOST` | 127.0.0.1 | set to `0.0.0.0` in a container |
| `RAG_PUBLIC` | unset | `1` acknowledges that binding publicly is deliberate, and changes the startup warning to a statement of what is exposed |
| `RAG_CORPUS` | first indexed | which corpus to serve |
| `RAG_GENERATOR` | `anthropic` | `ollama` writes prose with a local model instead |

`RAG_PUBLIC` exists because the server refuses to be quietly public. Bound to
anything but loopback without it, it prints a warning; with it, it prints what a
visitor can read — passage text, document names, and the full trace of every
query. That is the correct list to think about before deploying a corpus.

## What a visitor can do

Everything, and that is intentional. There is no write path: no upload, no
indexing trigger from the browser, no authentication because there is nothing to
authenticate. `POST /api/index/start` rebuilds the index and **is reachable**, so
if you deploy this publicly and care, put it behind your platform's auth or
remove the route. For a read-only demo over Wikipedia articles, the exposure is
the corpus itself, which is already public.

## Cost

Free tiers are sufficient. The container is ~2 GB with models baked in, holds
~700 MB resident, and answers in about 100 ms warm and 900 ms cold — the front
page's own questions are pre-warmed at startup, so the first click is fast.
