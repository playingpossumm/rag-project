"""Generate the golden set by deriving gold locations from answer strings.

Hand-labelling page numbers does not survive a corpus change. When the corpus
grew from 1 document to 20, twelve of sixteen adversarial cases silently became
answerable and eight answerable cases became ambiguous -- and nothing errored,
so the harness kept reporting confident numbers against labels that no longer
described reality.

So labels are derived instead of asserted. Each case declares a distinctive
answer string; every (source, locator) in the corpus containing that string
becomes gold. Three properties follow:

  * Labels cannot drift from the corpus -- they are recomputed from it.
  * Multi-source falls out naturally. If four papers state an answer, all four
    are gold, which matches the requirement to compile every relevant source
    rather than pick one.
  * Adding documents cannot silently invalidate a label; it just changes how
    many gold locations a question has.

The cost is that answer strings must be distinctive enough not to match
incidentally. `check` reports how many locations each matched so an over-broad
string is visible rather than quietly inflating the gold set.
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Gold is derived by substring match, and the match has to use the rule the
# harness scores with or a case can be derived here and missed there. The one
# definition lives in evaluate.py, and this file carried its own copy until
# 2026-09-06.
from evaluate import normalize as norm  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
# Follows RAG_STORE_DIR, so a golden set is always derived from the corpus that
# is actually indexed. Deriving ML labels from a bird index would produce a file
# full of "matches nothing" and look like the questions were wrong.
STORE = Path(os.environ.get("RAG_STORE_DIR",
                            ROOT / "vector_store")).expanduser() / "metadata.json"
OUT = ROOT / "eval" / "golden_set.json"

# Questions are phrased as a user would actually ask them -- no document named,
# because in real use you do not know which document holds the answer. Where a
# bare question would be hopelessly ambiguous across 20 similar papers, it is
# made discriminating by CONTENT (a distinctive number, term, or condition)
# rather than by naming the source, which would leak the answer's location.
CASES = [
    # -- single-source by nature: the defining passage lives in one paper -----
    ("mha-def", "What is multi-head attention and why use several heads instead of one?",
     "jointly attend to information from different representation subspaces", "fact"),
    ("scale-why", "Why is the dot product scaled by the square root of the key dimension?",
     "extremely small gradients", "fact"),
    ("warmup", "Which learning rate schedule increases linearly for the first steps then decays by inverse square root?",
     "warmup_steps", "fact"),
    ("pos-enc-fn", "What sinusoidal function is used to compute positional encodings?",
     "where pos is the position", "fact"),
    ("pos-enc-why", "Why does a model with no recurrence and no convolution need positional encodings?",
     "no recurrence and no convolution", "fact"),
    # Was adversarial, on the grounds that no mixture-of-experts paper was in
    # the corpus. One arrived when the corpus grew from 20 documents to 36, and
    # it discusses exactly this. Verified by reading the retrieved passage, not
    # by trusting the score: it scores +6.85 and it genuinely answers.
    ("moe-routing", "What load balancing loss is used for expert routing?",
     "auxiliary-loss-free load balancing", "fact"),
    ("path-length", "How does the maximum path length between positions compare for self-attention and recurrent layers?",
     "Maximum Path Length", "fact"),
    ("bleu-ende", "Which model achieved 28.4 BLEU on WMT 2014 English-to-German?",
     "28.4", "fact"),
    ("attn-uses", "In which three different ways is multi-head attention used in an encoder-decoder Transformer?",
     "three different ways", "fact"),
    ("label-smooth", "What label smoothing value was used during training?",
     "label smoothing of value", "fact"),
    ("p100", "What GPU hardware was used for training, and how many?",
     "8 NVIDIA P100 GPUs", "fact"),

    # -- naturally multi-source: several papers legitimately answer ----------
    ("adam", "Which optimizer was used, and with what beta values?",
     "β1 = 0.9", "multi"),
    ("relu", "Which activation function is applied between the two linear layers of a feed-forward block?",
     "ReLU", "multi"),
    # The gold here was correct and the question was not: many papers state a
    # dropout rate, so a passage saying "we use a dropout probability of 0.1
    # everywhere" answered it and scored zero. Made discriminating by
    # content rather than by naming the paper, which is the rule above.
    # Corrected 2026-09-01.
    ("dropout-rate", "Which dropout rate was used for the English-to-French model instead of 0.3?",
     "Pdrop = 0.1", "fact"),
    ("residual", "How are residual connections and layer normalization applied around each sub-layer?",
     "residual connection", "multi"),
    ("wmt14", "Which translation dataset was used for the English-to-German experiments?",
     "WMT 2014 English-German", "multi"),

    # -- cross-document: the point is that similar papers must be told apart --
    ("colbert-late", "What is late interaction in a retrieval model?",
     "late interaction", "cross-doc"),
    ("dpr-dual", "How is a dual-encoder retriever trained with in-batch negatives?",
     "in-batch negatives", "cross-doc"),
    ("lora-rank", "How does low-rank adaptation reduce the number of trainable parameters?",
     "low-rank", "cross-doc"),
    ("bert-mlm", "What is the masked language model pre-training objective?",
     "masked language model", "cross-doc"),
    ("cot-prompt", "How does prompting a model with intermediate reasoning steps affect arithmetic accuracy?",
     "a series of intermediate reasoning steps", "cross-doc"),
    ("sbert-siamese", "How are siamese network structures used to derive sentence embeddings?",
     "siamese", "cross-doc"),
    ("vit-patches", "How is an image split into patches and fed to a Transformer?",
     "patches", "cross-doc"),
    ("faiss-pq", "How does product quantization compress vectors for similarity search?",
     "product quantization", "cross-doc"),

    # ---- Expanded coverage: every document in the corpus represented -----
    # Written after coverage analysis showed the original set was heavily
    # weighted toward one paper. Answer strings were verified against the
    # corpus first (src/verify_candidates.py); four were rejected as too
    # generic -- "NSP" alone matched 14 documents because it is a substring
    # inside other words, and "span" matched inside "spanning".

    # ---- adam ----
    # "bias-correction" marked pages 5, 8 and 9, which discuss the effect of
    # the correction terms and never say how the correction works. Page 3
    # does: "we therefore divide by this term to correct the initialization
    # bias". Corrected 2026-09-01.
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
    ("bert-glue", "What benchmark suite is used to evaluate language understanding?", "GLUE benchmark", "multi"),
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
    # "zero-shot, one-shot" marked pages naming the three settings, which do
    # not say how performance changes with the count. Page 24 does, and is
    # the only page that does. Corrected 2026-09-01, and no longer cross-doc
    # because the old label's second document was a results table.
    ("gpt3-fewshot", "How does task performance change with the number of examples in the prompt?", "as a function of the number of in-context examples", "fact"),
    # ---- layer norm ----
    ("ln-vs-bn", "Why is layer normalization preferred over batch normalization for recurrent networks?", "recurrent neural networks", "multi"),
    ("ln-stats", "How are normalization statistics computed across features rather than examples?", "layer normalization", "cross-doc"),
    # ---- LoRA ----
    ("lora-latency", "Does the adaptation method add inference latency?", "no additional inference latency", "fact"),
    # "frozen" alone matched 37 locations across 8 documents -- every paper
    # that freezes anything. The gold set was inflated to the point where the
    # question could be "hit" by retrieving almost any adaptation paper.
    ("lora-frozen", "Are the pretrained weights updated during adaptation?",
     "keeping the pre-trained weights frozen", "multi"),
    # ---- RAG ----
    ("rag-parametric", "How are parametric and non-parametric memory combined?", "non-parametric", "cross-doc"),
    ("rag-marginalize", "How are retrieved documents marginalized during generation?", "marginaliz", "cross-doc"),
    # ---- ResNet ----
    # "degradation" matched 26 locations across 12 documents, most of them
    # using the word in an unrelated sense.
    ("resnet-degradation", "What happens to accuracy when plain networks get deeper?",
     "accuracy gets saturated", "fact"),
    ("resnet-shortcut", "What connections allow gradients to flow through very deep networks?", "shortcut connections", "fact"),
    ("resnet-152", "How deep is the deepest network evaluated on ImageNet?", "152", "fact"),
    # ---- RoBERTa ----
    ("roberta-dynamic", "What masking strategy is applied differently at each epoch?", "dynamic masking", "fact"),
    ("roberta-nsp-drop", "What pre-training objective was found unnecessary and removed?", "NSP loss", "cross-doc"),
    # ---- Sentence-BERT ----
    ("sbert-cosine", "How is sentence similarity computed from the derived embeddings?", "cosine-similarity", "multi"),
    ("sbert-speed", "How much faster is the approach than a cross-encoder for similarity search?", "65 hours", "fact"),
    # ---- seq2seq ----
    ("seq2seq-reverse", "What input transformation improved translation performance markedly?", "reversing", "fact"),
    ("seq2seq-lstm", "What recurrent architecture maps sequences to a fixed-dimensional vector?", "deep LSTM", "fact"),
    # ---- T5 ----
    ("t5-c4", "What cleaned web-scraped corpus is used for pre-training?", "Colossal Clean Crawled Corpus", "fact"),
    # "unified text-to-text" is part of the T5 paper's title, so four of the
    # six chunks carrying it were reference-list entries in colbert, DPR,
    # gpt3 and rag. The case was only cross-doc because papers cite T5.
    # "cast all of the tasks" marks pages 8 and 9 of t5.pdf, which is where
    # the paper explains it. Corrected 2026-09-01.
    ("t5-text2text", "How are all NLP tasks cast into a single format?", "cast all of the tasks", "fact"),
    ("t5-span", "What corruption objective masks contiguous spans of tokens?", "corrupted spans", "multi"),
    # ---- ViT ----
    ("vit-jft", "Which large private dataset is used for pre-training?", "JFT", "fact"),
    ("vit-inductive", "What inductive biases do convolutional networks have that transformers lack?", "inductive bias", "cross-doc"),
    # ---- word2vec ----
    ("w2v-cbow", "What architecture predicts a word from its surrounding context?", "continuous bag-of-words", "fact"),
    ("w2v-skipgram", "What architecture predicts surrounding words from a target word?", "Skip-gram", "fact"),
    ("w2v-analogy", "How are syntactic and semantic regularities measured in word vectors?", "syntactic", "multi"),
]

# Unanswerable cases, re-authored for a corpus of 20 ML/NLP papers. The previous
# set was written against a single paper, where anything off-topic was WILDLY
# off-topic; here almost any ML question finds plausibly-related material, so
# near-misses are the normal case rather than the hard one.
ADVERSARIAL = [
    ("adv-diffusion", "What noise schedule does the denoising diffusion model use?",
     "absent", "no diffusion paper in the corpus"),
    ("adv-rlhf", "How is the reward model trained for RLHF?",
     "absent", "no InstructGPT/RLHF paper in the corpus"),
    ("adv-mamba", "How does the state space model handle long sequences without attention?",
     "absent", "no state-space/Mamba paper in the corpus"),
    ("adv-price", "How many dollars did it cost to train the model?",
     "near-miss", "compute cost is reported in FLOPs and GPU-hours, never in currency"),
    ("adv-energy", "How many kilowatt-hours did training consume?",
     "near-miss", "training cost is reported, but never as energy"),
    ("adv-carbon", "What was the carbon footprint of training?",
     "near-miss", "adjacent to compute cost, never stated"),
    ("adv-seed", "What random seed was used for the reported runs?",
     "near-miss", "hyperparameters are listed exhaustively; seeds are not"),
    ("adv-review", "What did the peer reviewers say about this work?",
     "metadata", "review commentary is not part of any document"),
    ("adv-salary", "What were the authors paid for this research?",
     "metadata", "no compensation information exists in any paper"),
    ("adv-retract", "Has this paper been retracted?",
     "metadata", "publication status is not recorded in the documents"),

    # Near-misses dominate deliberately. With 20 adjacent ML papers, almost any
    # ML question finds plausibly-related material, so the threshold is decided
    # entirely by these -- an adversarial set of obvious absurdities would
    # produce a threshold that collapses on the first realistic hard question.
    ("adv-flash", "How does flash attention reduce memory reads and writes?",
     "absent", "FlashAttention postdates every paper here"),
    ("adv-quantize-4bit", "How is the model quantized to 4-bit precision?",
     "absent", "quantization to 4 bits is never discussed"),
    ("adv-inference-cost", "What is the cost per thousand inference requests?",
     "near-miss", "training cost is reported; inference cost never is"),
    ("adv-latency-ms", "What is the end-to-end latency in milliseconds per query?",
     "near-miss", "throughput appears, but not per-query latency"),
    ("adv-context-window", "What is the maximum context length in tokens?",
     "near-miss", "sequence lengths appear; no paper frames it as a context window"),
    ("adv-license-terms", "Under what licence were the model weights released?",
     "metadata", "release terms are not stated in any paper"),
    ("adv-compute-budget", "How many GPU-hours were budgeted before training began?",
     "near-miss", "compute used is reported after the fact, never as a prior budget"),
]


def derive_gold(chunks, answer: str) -> list[dict]:
    """Every (source, locator) whose text contains the answer string."""
    needle = norm(answer)
    # Keyed on (source, kind), not on source. A document can hold chunks of two
    # locator kinds at once: bird_anatomy.docx carries the answer in a named
    # section and in a numbered table. Keying on the source alone took the kind
    # from whichever chunk matched first and filed every later value under it,
    # which produced a gold entry for "section 1" that no chunk has and lost
    # "table 1", which one does. Found 2026-09-01, by rebuilding the set.
    found: dict[tuple[str, str], set] = {}
    for c in chunks:
        if needle in norm(c["text"]):
            loc = c["locator"]
            found.setdefault((c["source"], loc.get("kind", "page")), set()).add(loc["value"])
    # Locator values are not all one type once a corpus holds more than PDFs:
    # a page is an int, a Word section and a spreadsheet sheet are strings. Any
    # question whose gold spans both crashed on sorted(), which an all-PDF
    # corpus could never reveal. Ints first in numeric order, then strings
    # alphabetically -- the ordering is cosmetic, the type safety is not.
    # The KIND travels with the value. A location is (source, kind, value), and
    # a gold set that recorded only the value silently assumed "page" -- which
    # is true of a PDF corpus and false of every .docx, .pptx and .xlsx, whose
    # locators are sections, slides and sheets. Every non-PDF hit would have
    # scored as a miss.
    return [
        {"source": src, "kind": kind,
         "pages": sorted(vals, key=lambda v: (isinstance(v, str), v))}
        for (src, kind), vals in sorted(found.items())
    ]


def load_cases(path: Path | None):
    """Authored questions, from a JSON file or the built-in ML list.

    A corpus needs its own questions -- there is no generic set, because a
    question is only useful if its answer is in the documents. The JSON form is
    the same four fields as the built-in tuples, so neither can drift into a
    shape the other cannot read.
    """
    if path is None:
        return CASES, ADVERSARIAL
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = [(c["id"], c["question"], c["answer_contains"], c.get("kind", "fact"))
             for c in data.get("cases", [])]
    adv = [(c["id"], c["question"], c.get("adversarial_kind", "absent"),
            c.get("why", "")) for c in data.get("adversarial", [])]
    return cases, adv


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=Path, default=None,
                    help="authored questions as JSON; omit for the built-in ML set")
    ap.add_argument("--emit", type=Path,
                    default=ROOT / "eval" / "golden_set.json")
    ap.add_argument("--label", default="arXiv ML/NLP papers")
    args = ap.parse_args()

    CASE_LIST, ADV_LIST = load_cases(args.cases)

    chunks = json.loads(STORE.read_text(encoding="utf-8"))
    sources = sorted({c["source"] for c in chunks})
    print(f"corpus: {len(sources)} documents, {len(chunks)} chunks\n")

    cases, problems = [], []
    for cid, question, answer, kind in CASE_LIST:
        gold = derive_gold(chunks, answer)
        n_src = len(gold)
        n_loc = sum(len(g["pages"]) for g in gold)

        if n_src == 0:
            problems.append((cid, "answer string matches NOTHING -- case unusable"))
            flag = "MISS"
        elif n_loc > 25:
            problems.append((cid, f"matches {n_loc} locations -- string too generic"))
            flag = "BROAD"
        else:
            flag = "ok"

        print(f"  {flag:<5} {cid:<16} {kind:<9} {n_src} source(s), {n_loc} location(s)"
              f"{'  ' + ', '.join(g['source'][:18] for g in gold[:4]) if gold else ''}")

        cases.append({
            "id": cid, "question": question, "kind": kind,
            "answer_contains": answer, "gold": gold,
            "multi_source": n_src > 1,
        })

    for cid, question, kind, why in ADV_LIST:
        cases.append({
            "id": cid, "question": question, "unanswerable": True,
            "adversarial_kind": kind, "why": why, "gold": [],
        })

    payload = {
        "corpus": f"{len(sources)} {args.label}",
        "generated_by": "src/build_golden_set.py",
        "note": (
            "Gold locations are DERIVED from answer strings against the current "
            "corpus, not hand-labelled, so labels cannot drift out of step with "
            "the documents. Multi-source falls out naturally: every location "
            "containing the answer is gold, matching the requirement to compile "
            "all relevant sources rather than pick one. Regenerate after any "
            "corpus change by re-running the generator."
        ),
        "cases": cases,
    }
    args.emit.parent.mkdir(parents=True, exist_ok=True)
    args.emit.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")

    answerable = [c for c in cases if not c.get("unanswerable")]
    multi = [c for c in answerable if c["multi_source"]]
    print(f"\nwrote {args.emit}: {len(answerable)} answerable "
          f"({len(multi)} multi-source), {len(ADV_LIST)} adversarial")
    if problems:
        print("\nPROBLEMS -- these need a better answer string:")
        for cid, why in problems:
            print(f"  {cid}: {why}")


if __name__ == "__main__":
    main()
