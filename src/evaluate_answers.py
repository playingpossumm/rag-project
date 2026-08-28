"""Measure the generated answer, not just the passages that fed it.

Every number in this project measures **retrieval**: did the right passage come
back, and how high. Nothing measured the prose, because until 2026-08-27 no
prose had ever been generated. `generate_local.py` changed that, so this is the
missing half.

**No LLM judge.** A judge model is the usual answer and it is a bad first move
here: it introduces a second system whose failures are invisible, it cannot be
audited, and it would make this project's central claim -- every number is
reproducible from the repository -- false. Three of the four measures below are
objective, and the fourth is a stated proxy rather than a verdict.

What is measured, in descending order of how much it matters:

**1. Invented citations.** The canonical RAG failure and exactly checkable. Every
`[source, page]` the answer emits must correspond to a passage that was actually
supplied. A citation to a document the model was never given is a fabrication
wearing the costume of evidence, and it is worse than no answer at all, because
a citation is what a reader checks *instead of* the source.

**2. Correctness.** The golden sets already carry `answer_contains` -- the string
a correct answer must include, written when the case was labelled and used for
context recall. So correctness needs no judge: does the answer contain it?

**3. Refusal.** On adversarial questions the corpus cannot answer, does the model
say so? A system that confabulates fluently here is worse than one that
retrieves badly, because the retrieval failure is visible and this is not.

**4. Groundedness (a proxy, and labelled as one).** What fraction of the answer's
content words appear in the passages it was given. High means it is paraphrasing
what it read; low means it is drawing on what it already knew. It cannot
distinguish a correct claim from parametric memory from a hallucinated one, and
it is reported because it is cheap and directional -- not as a verdict.

Generation on a local CPU model costs seconds per question, so this samples by
default. `--all` runs the whole golden set and takes a while.

    ollama serve && ollama pull llama3.2
    .venv\\Scripts\\python.exe src\\evaluate_answers.py --corpus birds
    .venv\\Scripts\\python.exe src\\evaluate_answers.py --all
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import corpora  # noqa: E402
import rerank as rr  # noqa: E402
from evaluate import load_cases, normalize  # noqa: E402
from hybrid import build_bm25  # noqa: E402
from retrieve import (EMBEDDING_MODEL, TOP_K, load_ensemble, load_index,  # noqa: E402
                      retrieve)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
OUT = ROOT / "eval" / "answer-quality.json"

# Words that carry no evidence either way. Kept short deliberately: a long
# stop-list would flatter the groundedness number by removing exactly the words
# a model reaches for when it has nothing to say.
STOP = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "for",
    "with", "as", "by", "at", "from", "is", "are", "was", "were", "be", "been",
    "it", "its", "this", "that", "these", "those", "which", "who", "what",
    "not", "no", "there", "their", "they", "them", "he", "she", "we", "you",
    "i", "can", "will", "would", "should", "may", "might", "do", "does", "did",
    "have", "has", "had", "than", "then", "so", "such", "also", "into", "used",
    "using", "use", "according", "text", "based", "provided", "context",
}

# How a citation is written, per generate.SYSTEM_PROMPT: [source, page X].
CITATION = re.compile(r"\[([^\]]+?)(?:,\s*(?:page|slide|sheet|section)\s*[^\]]*)?\]")

# Phrases a refusal uses. Matched loosely because the instruction is "say so
# plainly" rather than a fixed form, and a model that invents its own wording
# for "I cannot answer this" is still refusing.
REFUSAL = ("does not contain", "doesn't contain", "no information",
           "not contain", "cannot answer", "can't answer", "unable to answer",
           "not provided", "not mentioned", "not available", "does not provide",
           "not enough information", "no mention", "not present", "not found",
           "does not include", "not specify", "does not specify")


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9][a-z0-9\-']+", normalize(text))
            if w not in STOP and len(w) > 2}


def cited_sources(answer: str) -> list[str]:
    """Whatever the answer put in square brackets, as source-ish strings."""
    out = []
    for raw in CITATION.findall(answer):
        name = raw.split(",")[0].strip()
        if name:
            out.append(name)
    return out


def judge(answer: str, case: dict, passages: list[dict]) -> dict:
    """Four measures over one answer. No model is consulted."""
    supplied = {p["source"] for p in passages}
    supplied_stems = {Path(s).stem.lower() for s in supplied}

    cites = cited_sources(answer)
    invented = []
    for c in cites:
        stem = Path(c).stem.lower()
        # A citation counts as supplied if it names a document that was given,
        # allowing for the model dropping or mangling the extension. Substring
        # both ways, because models truncate long filenames.
        if not any(stem == s or stem in s or s in stem for s in supplied_stems):
            invented.append(c)

    answer_words = words(answer)
    passage_words = set()
    for p in passages:
        passage_words |= words(p["text"])
    grounded = (len(answer_words & passage_words) / len(answer_words)
                if answer_words else 0.0)

    lowered = answer.lower()
    refused = any(phrase in lowered for phrase in REFUSAL)

    wanted = case.get("answer_contains")
    correct = None
    if wanted and not case.get("unanswerable"):
        correct = normalize(wanted) in normalize(answer)

    return {"citations": len(cites), "invented": invented,
            "grounded": round(grounded, 3), "refused": refused,
            "correct": correct, "chars": len(answer)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", action="append")
    ap.add_argument("--limit", type=int, default=8,
                    help="answerable cases per corpus (default 8; local "
                         "generation costs seconds each)")
    ap.add_argument("--adversarial", type=int, default=4,
                    help="unanswerable cases per corpus")
    ap.add_argument("--all", action="store_true", help="every case, no sampling")
    ap.add_argument("--emit", type=Path, default=OUT)
    args = ap.parse_args()

    import generate_local

    if not generate_local.available():
        print(f"no Ollama at {generate_local.HOST}. Start it with "
              f"`ollama serve` and `ollama pull {generate_local.MODEL}`.")
        return 2

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)

    reg = corpora.registry()
    report = {}
    for name in (args.corpus or list(reg)):
        cfg = reg.get(name)
        if not cfg or not cfg["indexed"] or not cfg["golden"]:
            continue
        answerable, adversarial = load_cases(cfg["golden"])
        if not args.all:
            answerable = answerable[:args.limit]
            adversarial = adversarial[:args.adversarial]
        cases = answerable + adversarial

        index, metadata = load_index(cfg["store"])
        bm25 = build_bm25(metadata)
        ensemble = load_ensemble(cfg["store"]) if cfg.get("ensemble_model") else None
        rr.RERANK_BLEND = cfg["rerank_blend"]

        print(f"\n  {cfg['label']}  ({len(answerable)} answerable + "
              f"{len(adversarial)} adversarial, {generate_local.MODEL})")
        rows, started = [], time.perf_counter()
        for case in cases:
            results = retrieve(case["question"], index, metadata, model,
                               k=TOP_K, candidate_k=cfg["candidate_k"],
                               use_reranker=True, bm25=bm25,
                               max_per_source=2, ensemble=ensemble)
            try:
                answer = generate_local.synthesize(case["question"], results)
            except Exception as exc:                            # noqa: BLE001
                print(f"    {case['id']}: generation failed: {exc}")
                continue
            row = {"id": case["id"], "unanswerable": bool(case.get("unanswerable")),
                   **judge(answer, case, results), "answer": answer}
            rows.append(row)
            flag = ("INVENTED" if row["invented"] else
                    "refused" if row["refused"] else
                    "correct" if row["correct"] else
                    "" if row["correct"] is None else "wrong")
            print(f"    {case['id']:<22} grounded {row['grounded']:.2f}  "
                  f"{row['citations']} cites  {flag}")

        ans = [r for r in rows if not r["unanswerable"]]
        adv = [r for r in rows if r["unanswerable"]]
        summary = {
            "n": len(rows),
            "invented_citations": sum(len(r["invented"]) for r in rows),
            "answers_with_invented": sum(1 for r in rows if r["invented"]),
            "correct": sum(1 for r in ans if r["correct"]),
            "n_answerable": len(ans),
            "refused_wrongly": sum(1 for r in ans if r["refused"]),
            "refused_rightly": sum(1 for r in adv if r["refused"]),
            "n_adversarial": len(adv),
            "grounded_mean": round(sum(r["grounded"] for r in rows)
                                   / max(len(rows), 1), 3),
            "seconds": round(time.perf_counter() - started, 1),
        }
        print(f"    ---")
        print(f"    invented citations   {summary['invented_citations']} "
              f"(in {summary['answers_with_invented']} of {summary['n']} answers)")
        print(f"    correct              {summary['correct']}/{summary['n_answerable']}"
              f"   (contains the labelled answer string)")
        print(f"    refused when it should  {summary['refused_rightly']}/"
              f"{summary['n_adversarial']}   and when it should not: "
              f"{summary['refused_wrongly']}/{summary['n_answerable']}")
        print(f"    groundedness (proxy) {summary['grounded_mean']:.3f}")
        report[name] = {"label": cfg["label"], "model": generate_local.MODEL,
                        "summary": summary, "cases": rows}

    if not report:
        print("nothing evaluated")
        return 1
    args.emit.write_text(json.dumps(
        {"generated_by": "src/evaluate_answers.py",
         "note": "No LLM judge. Citations and correctness are objective; "
                 "groundedness is a lexical proxy and not a verdict.",
         "corpora": report}, indent=1) + "\n", encoding="utf-8")
    print(f"\n  wrote {args.emit.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
