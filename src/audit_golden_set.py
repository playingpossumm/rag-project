"""Check whether the golden set is still valid for the current corpus.

A golden set is a function of the corpus it was written against. Adding
documents can silently turn an "unanswerable" case answerable, and can make a
question that had one correct answer ambiguous across several papers. Both
failures are invisible -- the harness keeps producing numbers, they are just
measuring something other than what the labels claim.

This audits both directions:

  1. Adversarial cases whose subject now appears in the corpus.
  2. Answerable cases whose question no longer identifies one document,
     because the same fact is discussed in several papers.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
STORE = ROOT / "vector_store" / "metadata.json"
GOLDEN = ROOT / "eval" / "golden_set.json"

# Subject terms that would make each adversarial case answerable if present.
ADVERSARIAL_SUBJECT = {
    "adv-bert": ["bert"],
    "adv-gpt3-params": ["gpt-3", "gpt3"],
    "adv-finetune": ["fine-tuning", "fine-tune"],
    "adv-diffusion": ["diffusion model", "noise schedule"],
    "adv-rlhf": ["reinforcement learning from human feedback", "rlhf"],
    "adv-review": ["peer review", "reviewer"],
    "adv-funding": ["grant", "funded by"],
    "adv-license": ["licence", "license"],
    "adv-authors": ["openai"],
    "adv-cost": ["dollars", "usd", "$"],
    "adv-inference-time": ["inference time", "inference latency"],
    "adv-energy": ["kilowatt", "kwh", "energy consumption"],
    "adv-decoder-layers": ["decoder layers"],
    "adv-val-split": ["validation split", "held out"],
    "adv-seed": ["random seed"],
    "adv-human-eval": ["human evaluation", "human evaluators"],
}


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).lower()


ABSTAIN = 0.0


def audit_by_retrieval(golden: dict) -> list[tuple[str, float, str]]:
    """Adversarial cases the pipeline now answers. Empty if the index is absent.

    Loading the models costs ~20s, so this degrades to the term check rather
    than making the audit unrunnable on a machine that has not indexed yet.
    """
    try:
        from sentence_transformers import SentenceTransformer

        from abstain import ABSTAIN_THRESHOLD
        from hybrid import build_bm25
        from retrieve import EMBEDDING_MODEL, load_index, retrieve
    except Exception:  # noqa: BLE001
        return []

    try:
        index, metadata = load_index()
    except Exception:  # noqa: BLE001 - no index built yet
        return []

    model = SentenceTransformer(EMBEDDING_MODEL)
    bm25 = build_bm25(metadata)
    out = []
    for case in golden["cases"]:
        if not case.get("unanswerable"):
            continue
        hits = retrieve(case["question"], index, metadata, model, k=5, bm25=bm25)
        if not hits:
            continue
        top = float(hits[0].get("rerank_score", 0.0))
        if top >= ABSTAIN_THRESHOLD:
            out.append((case["id"], top, hits[0]["source"]))
    return out


def main():
    chunks = json.loads(STORE.read_text(encoding="utf-8"))
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))

    sources = sorted({c["source"] for c in chunks})
    print(f"corpus: {len(sources)} documents, {len(chunks)} chunks\n")

    by_source = {}
    for c in chunks:
        by_source.setdefault(c["source"], []).append(norm(c["text"]))
    by_source = {s: " ".join(t) for s, t in by_source.items()}

    # ---- 1a. ask the RETRIEVER, not a word list --------------------------
    # The term list below caught 2 of 18 when the corpus grew 20 -> 36, and
    # missed all 5 cases that actually became answerable. It could not have
    # caught them: a paper discusses expert load balancing without ever writing
    # "mixture of experts", so a literal term match is looking for the wrong
    # thing.
    #
    # Retrieval is not looking for the wrong thing. If the reranker scores an
    # "unanswerable" question above the abstention threshold, then either the
    # label is stale or the gate is broken -- and both need a human to look.
    # Using the system to audit its own labels is circular, but in the useful
    # direction: it flags exactly where the two disagree.
    scored = audit_by_retrieval(golden)

    # ---- 1b. the subject-term check, kept as a second opinion -------------
    print("ADVERSARIAL CASES -- is the subject now in the corpus?")
    suspect = []
    for case in golden["cases"]:
        if not case.get("unanswerable"):
            continue
        terms = ADVERSARIAL_SUBJECT.get(case["id"], [])
        hits = {
            src for src, text in by_source.items()
            if any(t in text for t in terms)
        }
        kind = case.get("adversarial_kind", "?")
        if hits:
            suspect.append((case["id"], kind, sorted(hits)))
            shown = ", ".join(s.replace(".pdf", "") for s in sorted(hits)[:3])
            more = f" +{len(hits) - 3}" if len(hits) > 3 else ""
            print(f"  SUSPECT {case['id']:<20} {kind:<10} now in: {shown}{more}")
        else:
            print(f"  ok      {case['id']:<20} {kind:<10} subject still absent")

    if scored:
        print("\n  ...and what retrieval says, which is the check that matters:")
        for cid, top, src in scored:
            print(f"  ANSWERED {cid:<19} scores {top:+6.2f} against {ABSTAIN}"
                  f"  top hit: {src[:36]}")
        missed = [c for c, _, _ in scored if c not in {i for i, _, _ in suspect}]
        if missed:
            print(f"\n  {len(missed)} of these were NOT flagged by the term list: "
                  f"{', '.join(missed)}")

    # ---- 2. answerable cases that may now be ambiguous --------------------
    print("\nANSWERABLE CASES -- does the answer string now appear in other documents?")
    ambiguous = []
    for case in golden["cases"]:
        answer = case.get("answer_contains")
        if not answer:
            continue
        needle = norm(answer)
        found_in = {src for src, text in by_source.items() if needle in text}
        # The gold documents, which are exactly where the answer is SUPPOSED to
        # be. This subtracted `case.get("source")` -- a key no case has, since a
        # case carries `gold: [{"source": ...}]` -- so it evaluated to
        # {None} and removed nothing. Every case then matched its own gold
        # document and the audit reported 67 of 67 answerable cases as
        # ambiguous, which is the same information as reporting none of them.
        # A measurement that comes back at 100% deserves the same suspicion as
        # one that comes back at zero.
        others = found_in - {e["source"] for e in case.get("gold", [])}
        if others:
            ambiguous.append((case["id"], sorted(others)))
            shown = ", ".join(s.replace(".pdf", "") for s in sorted(others)[:3])
            more = f" +{len(others) - 3}" if len(others) > 3 else ""
            print(f"  AMBIG   {case['id']:<20} also in: {shown}{more}")

    print(f"\n{'=' * 62}")
    print(f"adversarial cases needing review : {len(suspect)} / "
          f"{sum(1 for c in golden['cases'] if c.get('unanswerable'))}")
    print(f"answerable cases now ambiguous   : {len(ambiguous)} / "
          f"{sum(1 for c in golden['cases'] if c.get('answer_contains'))}")


if __name__ == "__main__":
    main()
