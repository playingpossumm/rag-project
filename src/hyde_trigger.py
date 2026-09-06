"""Is there any trigger for a hypothetical-answer fallback? Measured: no.

`sweep_hyde.py` measured retrieving on a model-written answer and found it net
negative on two of three corpora -- it recovers a few questions and breaks more,
because a question that already works is moved off the wording that was working.
The obvious repair is to stop running it on questions that already work, so the
proposal was a fallback fired only when the pipeline was already below its
abstention threshold. A question about to be refused cannot be made worse by
searching again, so the losing half of the trade disappears.

This measures that proposal before building it, and refutes it.

**Nothing fires.** Across the three corpora there are eleven answerable
questions where retrieval returns no correct passage, and every one of them
scores ABOVE its corpus's threshold. Not marginally: lora-latency reaches +6.36
and bird-hollow-bones +5.03 against a threshold of -5.5. The fallback would
have run on four questions that already work and on the twenty-two adversarial
questions the gate is catching correctly, which is precisely the population
where it can only do harm, and on none of the questions it was written for.

The reason is worth more than the experiment, and it generalises past HyDE. The
system is not uncertain when it is wrong. The cross-encoder reads a plausible
wrong passage and scores it as highly as a right one, which is what makes these
failures permanent, and it means the gate cannot be used to detect them. The
gate separates answerable questions from unanswerable ones. It says nothing
about whether the passage returned is the right one, and no threshold on it can.

So the second half asks whether any cheaper signal separates the misses:

    coverage    how many of the question's content words appear in the top
                passage. The argument for a hypothetical answer was that the
                word which would find the passage is absent from the question,
                and this is that argument stated as a number.
    margin      top rerank score minus the fifth, since a pool the reranker
                cannot separate is one whose ordering is close to arbitrary.
    agreement   how many of the top five both retrievers ranked. RRF
                correlation dilution is this project's recorded failure mode
                for exactly these cases.

None of the three separates. On every one the failing questions sit inside the
range of the working ones, and a cut loose enough to catch all eleven fires on
90% of every question asked. The clearest single number is that the lowest-
scoring HIT is -10.68 while the lowest-scoring MISS is -1.09, so at the bottom
of the scale a low score predicts a correct answer slightly better than a wrong
one.

Two runs of this script agreed exactly, which is the check that matters for a
result this negative.

    .venv\\Scripts\\python.exe src\\hyde_trigger.py
    .venv\\Scripts\\python.exe src\\hyde_trigger.py --emit eval\\hyde-trigger.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import corpora  # noqa: E402
import rerank as rr  # noqa: E402
from evaluate import gold_keys, is_relevant, load_cases  # noqa: E402
from hybrid import bm25_search, build_bm25  # noqa: E402
from retrieve import (DEFAULT_FUSION, EMBEDDING_MODEL, TOP_K,  # noqa: E402
                      load_ensemble, load_index, retrieve, search)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent

# Enough to stop "what", "which" and "the" counting as content words the
# passage ought to contain. Not a linguistic resource and not meant to be one:
# coverage is being tested as a signal, and a signal that only works with a
# carefully tuned stop list is not the cheap trigger this was looking for.
STOP = set("""a an the of and or in on to for with by is are was were be been
what which who whom whose how why when where does do did can could would should
this that these those it its as at from than then there their they them we you
i not no if but so such other more most some any each""".split())

WORD = re.compile(r"[a-z0-9][a-z0-9-]*")


def content_words(question: str) -> set[str]:
    return {w for w in WORD.findall(question.lower())
            if w not in STOP and len(w) > 2}


def measure(cfg, model):
    """One corpus, every case, with the four signals and the gate verdict."""
    index, metadata = load_index(cfg["store"])
    bm25 = build_bm25(metadata)
    ensemble = load_ensemble(cfg["store"])
    threshold = cfg["threshold"]
    cases, adversarial = load_cases(Path(cfg["golden"]))

    rows = []
    for case in cases + adversarial:
        question = case["question"]
        rr.clear_cache()
        results = retrieve(question, index, metadata, model, k=TOP_K,
                           candidate_k=cfg["candidate_k"], use_reranker=True,
                           fusion=DEFAULT_FUSION, bm25=bm25, max_per_source=2,
                           ensemble=ensemble, rerank_blend=cfg["rerank_blend"])
        if not results:
            continue

        scores = [r["rerank_score"] for r in results]
        terms = content_words(question)
        top_words = set(WORD.findall(results[0]["text"].lower()))

        dense = {r["chunk_id"] for r in
                 search(question, index, metadata, model, k=cfg["candidate_k"])}
        sparse = {r["chunk_id"] for r in
                  bm25_search(question, bm25, metadata, k=cfg["candidate_k"])}
        returned = {r["chunk_id"] for r in results}

        adversarial_case = case in adversarial
        rows.append({
            "id": case["id"],
            "adversarial": adversarial_case,
            # An adversarial case has no gold, so "hit" is meaningless for it
            # and the trigger analysis below counts answerable cases only.
            "hit": (None if adversarial_case else
                    any(is_relevant(r, gold_keys(case)) for r in results)),
            "above_gate": scores[0] >= threshold,
            "top": round(scores[0], 2),
            "coverage": round(len(terms & top_words) / max(len(terms), 1), 2),
            "margin": round(scores[0] - scores[-1], 2),
            "agreement": len(returned & dense & sparse),
        })
    return rows, threshold


def gate_table(rows, threshold, label):
    """Where the gate sits relative to the questions a fallback would rescue."""
    def n(pred):
        return sum(1 for r in rows if pred(r))

    ans = [r for r in rows if not r["adversarial"]]
    adv = [r for r in rows if r["adversarial"]]
    counts = {
        "hit_above": n(lambda r: r["hit"] is True and r["above_gate"]),
        "hit_below": n(lambda r: r["hit"] is True and not r["above_gate"]),
        "miss_above": n(lambda r: r["hit"] is False and r["above_gate"]),
        "miss_below": n(lambda r: r["hit"] is False and not r["above_gate"]),
        "adv_above": n(lambda r: r["adversarial"] and r["above_gate"]),
        "adv_below": n(lambda r: r["adversarial"] and not r["above_gate"]),
    }
    print(f"\n  {label}   threshold {threshold}   "
          f"{len(ans)} answerable, {len(adv)} adversarial")
    print(f"    {'':<8}{'above':>8}{'below':>8}")
    for kind in ("hit", "miss", "adv"):
        print(f"    {kind:<8}{counts[kind + '_above']:>8}"
              f"{counts[kind + '_below']:>8}")
    print(f"    a fallback fired on abstention would reach "
          f"{counts['miss_below']} of "
          f"{counts['miss_below'] + counts['miss_above']} misses, and would "
          f"also fire on\n    {counts['hit_below']} working question(s) and "
          f"{counts['adv_below']} correctly-refused adversarial case(s).")
    return counts


def trigger_table(rows):
    """Does any signal separate the misses from the hits? Answered: no."""
    ans = [r for r in rows if not r["adversarial"]]
    misses = [r for r in ans if r["hit"] is False]
    print(f"\n  {len(ans)} answerable cases, {len(misses)} of them missed\n")
    print(f"  {'signal':<12}{'':>9}{'min':>8}{'p25':>8}{'med':>8}{'max':>8}")

    for signal in ("coverage", "margin", "agreement", "top"):
        for name, subset in (("hits", [r for r in ans if r["hit"]]),
                             ("misses", misses)):
            vals = sorted(r[signal] for r in subset)
            if not vals:
                continue
            q = len(vals) // 4
            print(f"  {signal if name == 'hits' else '':<12}{name:>9}"
                  f"{vals[0]:>8.2f}{vals[q]:>8.2f}{vals[len(vals) // 2]:>8.2f}"
                  f"{vals[-1]:>8.2f}")

    print("\n  misses, in full:")
    print(f"    {'case':<24}{'cov':>7}{'margin':>8}{'agree':>7}{'top':>7}")
    for r in misses:
        print(f"    {r['id']:<24}{r['coverage']:>7.2f}{r['margin']:>8.2f}"
              f"{r['agreement']:>7}{r['top']:>7.2f}")

    # A trigger is only useful if firing on every miss does not also fire on
    # most of the hits. This is that ratio, at the loosest cut that still
    # catches all of them.
    verdicts = {}
    for signal in ("coverage", "margin", "agreement"):
        if not misses:
            continue
        cut = max(r[signal] for r in misses)
        fires = sum(1 for r in ans if r[signal] <= cut)
        verdicts[signal] = {"cut": cut, "fires": fires, "of": len(ans)}
        print(f"\n  {signal}: catching all {len(misses)} misses needs a cut at "
              f"{cut:.2f}, which fires\n  on {fires} of {len(ans)} cases "
              f"({fires / len(ans):.0%}).")
    return verdicts


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--emit", type=Path, default=None,
                    help="write the per-case rows as JSON")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    reg = corpora.registry()
    names = ([args.corpus] if args.corpus
             else [n for n, c in reg.items() if c["indexed"] and c["golden"]])

    report, everything = {}, []
    for name in names:
        cfg = reg[name]
        rows, threshold = measure(cfg, model)
        counts = gate_table(rows, threshold, cfg["label"])
        report[name] = {"label": cfg["label"], "threshold": threshold,
                        "gate": counts, "cases": rows}
        everything.extend(rows)

    print("\n" + "-" * 66)
    print("  All corpora together. A trigger has to work across them, because "
          "a cut\n  fitted to one corpus is the mistake this project has "
          "already recorded\n  three times over.")
    verdicts = trigger_table(everything)

    print("\n  No signal separates. The failing questions sit inside the range "
          "of the\n  working ones on every one of them, so a hypothetical-"
          "answer fallback has\n  no trigger on this corpus, and the gate is "
          "not a wrongness detector.")

    if args.emit:
        args.emit.write_text(
            json.dumps({"generated_by": "src/hyde_trigger.py",
                        "corpora": report, "triggers": verdicts},
                       indent=1) + "\n", encoding="utf-8")
        print(f"\n  wrote {args.emit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
