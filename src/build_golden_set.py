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
import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
STORE = ROOT / "vector_store" / "metadata.json"
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
    ("dropout-rate", "What dropout rate was applied for regularization?",
     "dropout rate", "multi"),
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
]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def derive_gold(chunks, answer: str) -> list[dict]:
    """Every (source, locator) whose text contains the answer string."""
    needle = norm(answer)
    found = {}
    for c in chunks:
        if needle in norm(c["text"]):
            loc = c["locator"]
            found.setdefault(c["source"], set()).add(loc["value"])
    return [
        {"source": src, "pages": sorted(vals)}
        for src, vals in sorted(found.items())
    ]


def main():
    chunks = json.loads(STORE.read_text(encoding="utf-8"))
    sources = sorted({c["source"] for c in chunks})
    print(f"corpus: {len(sources)} documents, {len(chunks)} chunks\n")

    cases, problems = [], []
    for cid, question, answer, kind in CASES:
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

    for cid, question, kind, why in ADVERSARIAL:
        cases.append({
            "id": cid, "question": question, "unanswerable": True,
            "adversarial_kind": kind, "why": why, "gold": [],
        })

    payload = {
        "corpus": f"{len(sources)} arXiv ML/NLP papers",
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
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    answerable = [c for c in cases if not c.get("unanswerable")]
    multi = [c for c in answerable if c["multi_source"]]
    print(f"\nwrote {OUT.relative_to(ROOT)}: {len(answerable)} answerable "
          f"({len(multi)} multi-source), {len(ADVERSARIAL)} adversarial")
    if problems:
        print("\nPROBLEMS -- these need a better answer string:")
        for cid, why in problems:
            print(f"  {cid}: {why}")


if __name__ == "__main__":
    main()
