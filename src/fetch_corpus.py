"""Fetch a multi-document corpus of ML/NLP papers from arXiv.

Deliberately topically adjacent rather than unrelated. A corpus of papers from
different fields would make retrieval and abstention look good for the wrong
reason -- telling physics from poetry is easy. Papers that all discuss
transformers, attention, and training procedures force the retriever to
distinguish between genuinely similar sources, which is the realistic case.

Several are about the machinery this project is built on (FAISS, Sentence-BERT,
ColBERT, DPR, RAG), which makes for unusually good test queries.
"""
import sys
import time
import urllib.request
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"

# (arXiv id, filename stem) -- Attention is already present and omitted.
PAPERS = [
    ("1810.04805", "bert"),
    ("2005.14165", "gpt3"),
    ("1907.11692", "roberta"),
    ("1910.10683", "t5"),
    ("1412.6980",  "adam_optimizer"),
    ("1502.03167", "batch_normalization"),
    ("1607.06450", "layer_normalization"),
    ("1512.03385", "resnet"),
    ("1409.3215",  "seq2seq"),
    ("1409.0473",  "bahdanau_attention"),
    ("1301.3781",  "word2vec"),
    ("2010.11929", "vision_transformer"),
    ("2005.11401", "rag"),
    ("2004.04906", "dense_passage_retrieval"),
    ("2004.12832", "colbert"),
    ("1908.10084", "sentence_bert"),
    ("1702.08734", "faiss_billion_scale"),
    ("2106.09685", "lora"),
    ("2201.11903", "chain_of_thought"),
]

UA = "rag-project-educational/1.0 (personal learning project)"


def main():
    DATA.mkdir(exist_ok=True)
    got, skipped, failed = 0, 0, []

    for arxiv_id, stem in PAPERS:
        dest = DATA / f"{stem}.pdf"
        if dest.exists():
            print(f"  skip   {stem} (already present)")
            skipped += 1
            continue

        url = f"https://arxiv.org/pdf/{arxiv_id}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            if not data.startswith(b"%PDF"):
                raise ValueError("response was not a PDF")
            dest.write_bytes(data)
            print(f"  ok     {stem:<26} {len(data) / 1_000_000:5.1f} MB")
            got += 1
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  FAIL   {stem:<26} {exc}")
            failed.append(stem)

        time.sleep(1.5)  # be polite to arXiv

    print(f"\ndownloaded {got}, skipped {skipped}, failed {len(failed)}")
    if failed:
        print("failed:", ", ".join(failed))
    total = sorted(DATA.glob("*.pdf"))
    size = sum(p.stat().st_size for p in total) / 1_000_000
    print(f"corpus now: {len(total)} PDFs, {size:.1f} MB")


if __name__ == "__main__":
    main()
