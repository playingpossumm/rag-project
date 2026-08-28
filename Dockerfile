# A container for the public demo.
#
# Not Vercel, and the reason is arithmetic rather than preference: this process
# holds a FAISS index, a bi-encoder and a cross-encoder in memory. torch alone
# is 514 MB and the full dependency set is about 1.5 GB, against a 250 MB limit
# for a Python serverless function. Loading the models costs ~30 s, against a
# 10 s function timeout on Vercel's Hobby tier. And serverless keeps no memory
# between invocations, so every request would reload half a gigabyte of torch to
# answer one question in a second.
#
# What this needs is an ordinary long-lived process with ~2 GB of RAM. Hugging
# Face Spaces (Docker SDK, free CPU tier) is the closest fit and listens on
# 7860; Fly.io, Render and Railway all work the same way.
#
#   docker build -t rag-demo .
#   docker run -p 7860:7860 rag-demo
#
# The image bakes in the ornithology corpus only. That is a licensing decision,
# not a size one: those documents are Wikipedia articles under CC BY-SA and can
# be redistributed with attribution (see ATTRIBUTION.md). The arXiv papers in
# the other two corpora cannot be, which is why they were removed from git
# history and are not shipped here either.

FROM python:3.12-slim

# libgomp is faiss's OpenMP runtime; without it `import faiss` fails at load
# with an error that names no missing package.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The CPU wheel, explicitly. The default torch wheel carries CUDA and is several
# gigabytes, which on a free tier is the difference between deploying and not.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Both models, downloaded at build time rather than on first request. A cold
# start that also downloads 100 MB of weights looks like a hang to whoever
# opened the page.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY src/ ./src/
COPY ui/ ./ui/
COPY eval/ ./eval/
COPY corpora.json ATTRIBUTION.md LICENSE ./

# The corpus and its index. Built here rather than committed, so the repository
# stays free of document text: see docs/deploying.md for fetching it first.
COPY data-birds/ ./data-birds/
COPY store-birds/ ./store-birds/

ENV RAG_HOST=0.0.0.0 \
    PORT=7860 \
    RAG_CORPUS=birds \
    RAG_PUBLIC=1 \
    HF_HOME=/app/.cache

EXPOSE 7860

# No healthcheck on the HTTP port for the first 40 s: the server loads models
# before it binds, and a platform that probes too early will restart it in a
# loop and never let it finish.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4)"

CMD ["python", "src/serve.py"]
