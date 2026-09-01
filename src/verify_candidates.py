"""Check candidate answer strings before they enter the golden set.

A derived-label golden set is only as good as its answer strings. A string that
matches nothing makes a case unusable; one that matches half the corpus inflates
the gold set and quietly makes the metric easier. Both are invisible once the
case is in the file, so candidates are checked first.
"""
import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STORE = Path(__file__).parent.parent / "vector_store" / "metadata.json"

# (id, question, answer_string, kind)
CANDIDATES = [
    # ---- adam ----
    ("adam-bias", "How does the optimizer correct for bias in its moment estimates?", "correct the initialization bias", "fact"),
    ("adam-adamax", "What variant based on the infinity norm is proposed?", "AdaMax", "fact"),
    # ---- bahdanau ----
    ("bahdanau-fixed", "What bottleneck does encoding a sentence into a fixed-length vector create?", "fixed-length vector", "fact"),
    ("bahdanau-align", "How does the model learn to align and translate jointly?", "align and translate", "fact"),
    # ---- batch norm ----
    ("bn-covariate", "What problem does normalizing layer inputs address?", "internal covariate shift", "fact"),
    ("bn-minibatch", "How are normalization statistics computed during training?", "mini-batch", "multi"),
    # ---- bert ----
    ("bert-nsp", "What sentence-level pre-training task is used alongside masked prediction?", "next sentence prediction", "cross-doc"),
    ("bert-wordpiece", "What subword tokenization scheme is used?", "WordPiece", "multi"),
    ("bert-glue", "What benchmark suite is used to evaluate language understanding?", "GLUE", "multi"),
    # ---- chain of thought ----
    ("cot-gsm8k", "Which grade-school math benchmark is used to evaluate reasoning?", "GSM8K", "cross-doc"),
    ("cot-scale", "At what model scale does chain-of-thought reasoning emerge?", "540B", "cross-doc"),
    # ---- colbert ----
    ("colbert-maxsim", "What operator computes the similarity between query and document embeddings?", "MaxSim", "fact"),
    ("colbert-msmarco", "Which passage ranking dataset is used for evaluation?", "MS MARCO", "multi"),
    # ---- DPR ----
    ("dpr-bm25", "How does dense retrieval compare with BM25 on top-20 accuracy?", "BM25", "multi"),
    ("dpr-nq", "Which open-domain question answering dataset is used?", "Natural Questions", "multi"),
    # ---- FAISS ----
    ("faiss-kselect", "How is top-k selection implemented efficiently on GPU?", "k-selection", "fact"),
    ("faiss-ivf", "What index structure partitions vectors into cells for search?", "inverted file", "fact"),
    # ---- GPT-3 ----
    ("gpt3-params", "How many parameters does the largest autoregressive model have?", "175 billion", "cross-doc"),
    ("gpt3-icl", "What is in-context learning without gradient updates?", "in-context learning", "cross-doc"),
    ("gpt3-fewshot", "How does task performance change with the number of examples in the prompt?", "as a function of the number of in-context examples", "fact"),
    # ---- layer norm ----
    ("ln-vs-bn", "Why is layer normalization preferred over batch normalization for recurrent networks?", "recurrent neural networks", "multi"),
    ("ln-stats", "How are normalization statistics computed across features rather than examples?", "layer normalization", "cross-doc"),
    # ---- LoRA ----
    ("lora-latency", "Does the adaptation method add inference latency?", "no additional inference latency", "fact"),
    ("lora-frozen", "Are the pretrained weights updated during adaptation?", "frozen", "multi"),
    # ---- RAG ----
    ("rag-parametric", "How are parametric and non-parametric memory combined?", "non-parametric", "cross-doc"),
    ("rag-marginalize", "How are retrieved documents marginalized during generation?", "marginaliz", "cross-doc"),
    # ---- ResNet ----
    ("resnet-degradation", "What happens to accuracy when plain networks get deeper?", "degradation", "fact"),
    ("resnet-shortcut", "What connections allow gradients to flow through very deep networks?", "shortcut connections", "fact"),
    ("resnet-152", "How deep is the deepest network evaluated on ImageNet?", "152", "fact"),
    # ---- RoBERTa ----
    ("roberta-dynamic", "What masking strategy is applied differently at each epoch?", "dynamic masking", "fact"),
    ("roberta-nsp-drop", "What pre-training objective was found unnecessary and removed?", "NSP", "cross-doc"),
    # ---- Sentence-BERT ----
    ("sbert-cosine", "How is sentence similarity computed from the derived embeddings?", "cosine-similarity", "multi"),
    ("sbert-speed", "How much faster is the approach than a cross-encoder for similarity search?", "65 hours", "fact"),
    # ---- seq2seq ----
    ("seq2seq-reverse", "What input transformation improved translation performance markedly?", "reversing", "fact"),
    ("seq2seq-lstm", "What recurrent architecture maps sequences to a fixed-dimensional vector?", "LSTM", "multi"),
    # ---- T5 ----
    ("t5-c4", "What cleaned web-scraped corpus is used for pre-training?", "Colossal Clean Crawled Corpus", "fact"),
    ("t5-text2text", "How are all NLP tasks cast into a single format?", "cast all of the tasks", "fact"),
    ("t5-span", "What corruption objective masks contiguous spans of tokens?", "span", "multi"),
    # ---- ViT ----
    ("vit-jft", "Which large private dataset is used for pre-training?", "JFT", "fact"),
    ("vit-inductive", "What inductive biases do convolutional networks have that transformers lack?", "inductive bias", "cross-doc"),
    # ---- word2vec ----
    ("w2v-cbow", "What architecture predicts a word from its surrounding context?", "continuous bag-of-words", "fact"),
    ("w2v-skipgram", "What architecture predicts surrounding words from a target word?", "Skip-gram", "fact"),
    ("w2v-analogy", "How are syntactic and semantic regularities measured in word vectors?", "syntactic", "multi"),
]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def main():
    chunks = json.loads(STORE.read_text(encoding="utf-8"))
    ok = usable = 0
    problems = []

    for cid, _q, answer, kind in CANDIDATES:
        needle = norm(answer)
        found = {}
        for c in chunks:
            if needle in norm(c["text"]):
                found.setdefault(c["source"], set()).add(c["locator"]["value"])
        n_src = len(found)
        n_loc = sum(len(v) for v in found.values())

        if n_src == 0:
            flag, note = "MISS ", "matches nothing"
            problems.append((cid, note))
        elif n_loc > 30:
            flag, note = "BROAD", f"{n_loc} locations -- too generic"
            problems.append((cid, note))
        else:
            flag, note = "ok   ", ""
            usable += 1
        ok += 1
        srcs = ", ".join(sorted(s.replace(".pdf", "")[:16] for s in found)[:3])
        print(f"  {flag} {cid:<18} {kind:<9} {n_src:>2} src / {n_loc:>3} loc  {srcs}{'  ' + note if note else ''}")

    print(f"\n{usable}/{ok} usable")
    if problems:
        print("needs a different string:")
        for cid, note in problems:
            print(f"  {cid}: {note}")


if __name__ == "__main__":
    main()
