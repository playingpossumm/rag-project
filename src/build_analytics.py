"""Everything the analytics page plots, derived from the eval artefacts.

The page must never contain a transcribed number. Four numbers in the docs were
wrong once because they were copied from a terminal by hand after the config
they came from had changed, so every figure the browser draws is generated here
from `eval/results*.json` and `eval/per_case*.json` and nothing else.

Run:  python src/build_analytics.py     ->  eval/analytics.json
"""
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import corpora  # noqa: E402
from check_freshness import artefact, stamp  # noqa: E402

ROOT = Path(__file__).parent.parent
OUT = ROOT / "eval" / "analytics.json"
ANSWERS = ROOT / "eval" / "answer-quality.json"

# The order the harness reports configurations in is the order they were added,
# which is not the order they compose in. This is the ladder: each row adds one
# mechanism to the row above it.
LADDER = ["dense, no rerank", "dense + rerank", "rrf + rerank",
          "+ diversity 2/src", "+ diversity 1/src"]
METRICS = [("hit_rate", "Hit rate @5"), ("mrr", "MRR"),
           ("ndcg", "NDCG"), ("src_recall", "Source recall")]


def load(cfg: dict, kind: str):
    """One eval artefact for one corpus, or None if it has not been run.

    Which file belongs to which corpus used to be a hardcoded table here, and a
    corpus added to corpora.json but not to the table was skipped in silence --
    the front page then offered it the fallback questions, which are about the
    ML papers. The name is derived from the golden set now, by the same rule
    evaluate.py and per_case.py use to write it.
    """
    p = artefact(kind, cfg["golden"])
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def sweep(ans: list[float], adv: list[float], lo=-8.0, hi=4.0, step=0.5):
    """What each cut point would cost, in cases rather than rates.

    Rates are the wrong unit here and the harness said so: the two populations
    are different sizes, so subtracting one rate from the other silently values
    one adversarial case at several answerable ones.
    """
    out = []
    t = lo
    while t <= hi + 1e-9:
        out.append({
            "t": round(t, 2),
            "refused": sum(1 for c in ans if c < t),      # answerable, lost
            "caught": sum(1 for c in adv if c < t),        # adversarial, caught
        })
        t += step
    return out


# What each case kind is called on screen. The harness's own vocabulary --
# "fact", "multi", "cross-doc" -- describes how a case was constructed, which is
# the labeller's concern. A reader picking a question wants to know what kind of
# retrieval it will exercise.
KIND_LABEL = {
    "fact": "answered in one place",
    "multi": "spread over several documents",
    # "decoys that look right" until 2026-09-02, which described the passages
    # these questions compete against and read on screen as a label on the
    # QUESTION. A reader picking one was told it was a decoy while the same row
    # said it should be answered. The golden set calls these cross-document
    # because similar papers have to be told apart, which is what the reader
    # is being shown.
    "cross-doc": "similar documents to tell apart",
}
ADVERSARIAL = "not in these documents"

# Roughly how many of each to offer, in the order they should be shown. Weighted
# towards the two kinds that show the system doing something a plain keyword
# search would not.
QUOTA = [("fact", 3), ("multi", 3), ("cross-doc", 2), ("__adv__", 2)]


def examples(cases: list[dict]) -> list[dict]:
    """A few real, already-scored questions per corpus.

    Drawn from the golden set rather than written for the occasion, so every
    suggestion is one the harness has actually run and whose outcome is known --
    including the adversarial ones, which are supposed to be refused. A made-up
    example that happens to fail would look like a broken system.
    """
    buckets: dict[str, list[dict]] = {}
    for c in cases:
        key = "__adv__" if c["unanswerable"] else c.get("kind", "fact")
        buckets.setdefault(key, []).append(c)

    out = []
    for key, n in QUOTA:
        pool = buckets.get(key, [])
        # Shortest first: a suggestion has to be readable at a glance, and the
        # long ones are long because they carry three clauses of qualification.
        pool = sorted(pool, key=lambda c: len(c["question"]))[:n]
        for c in pool:
            out.append({
                "q": c["question"],
                "label": ADVERSARIAL if key == "__adv__" else KIND_LABEL.get(key, key),
                "adversarial": key == "__adv__",
                # Known from the harness, so the UI can say what should happen
                # without pretending to predict it.
                "expect": "should be refused" if key == "__adv__" else "should be answered",
            })
    return out


def answer_quality() -> dict:
    """What `src/evaluate_answers.py` measured, per corpus, if anything.

    Optional on purpose. Answer quality costs about a minute per question on a
    CPU, so the script samples by default and the block may not exist for a
    corpus at all; a missing block means the section is not drawn, rather than
    drawn empty. Since 2026-09-05 the shipped file holds every case, 157
    answers in 3.9 hours, rather than the 45-answer sample it held before.
    """
    if not ANSWERS.exists():
        return {}
    try:
        raw = json.loads(ANSWERS.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:                      # noqa: BLE001
        print(f"  answer-quality.json unreadable, skipped: {exc}")
        return {}
    out = {}
    for name, block in (raw.get("corpora") or {}).items():
        sm = block.get("summary") or {}
        if not sm.get("n"):
            continue
        out[name] = {
            "model": block.get("model"),
            "n": sm["n"],
            "n_answerable": sm.get("n_answerable", 0),
            "n_adversarial": sm.get("n_adversarial", 0),
            "invented_citations": sm.get("invented_citations", 0),
            "answers_with_invented": sm.get("answers_with_invented", 0),
            # A citation that is only a reference number lifted from the
            # passage. Carried so a zero on the page means zero rather than
            # "not read from the file".
            "malformed_citations": sm.get("malformed_citations", 0),
            "correct": sm.get("correct", 0),
            # The strict test is a substring match and a floor. This one credits
            # an answer carrying 80% of the labelled string's content words in
            # any order, and is an upper bound of the same kind. Both are
            # carried so the page can show the pair; either alone misleads.
            "correct_loose": sm.get("correct_loose", sm.get("correct", 0)),
            "unmatched": len(sm.get("unmatched", [])),
            "refused_rightly": sm.get("refused_rightly", 0),
            "refused_wrongly": sm.get("refused_wrongly", 0),
            "grounded_mean": sm.get("grounded_mean"),
        }
    return out


def answer_first(per: dict) -> dict:
    """Of the answerable cases, how many put a gold passage at rank 1.

    `found` counts the cases with a gold passage anywhere in the five, so
    `first / answerable` and `found / answerable` are the two numbers side by
    side and the gap between them is the ordering cost.
    """
    cases = [c for c in per.get("cases", []) if not c.get("unanswerable")]
    first = found = 0
    for c in cases:
        ranks = [r["rank"] for r in c.get("results", []) if r.get("relevant")]
        if not ranks:
            continue
        found += 1
        if min(ranks) == 1:
            first += 1
    return {"answerable": len(cases), "found": found, "first": first,
            "rate": round(first / len(cases), 4) if cases else 0.0}


def build():
    data = {"generated_by": "src/build_analytics.py", "corpora": [],
            # Which files each entry was built from. This is what makes a
            # rebuilt per_case.json that nobody followed with a rebuild here
            # detectable: every count still agrees and every number is old.
            # src/check_freshness.py compares these against what is on disk.
            "inputs": {"corpora_json": stamp(corpora.CONFIG)["digest"],
                       # Answer quality is sampled, so a re-score can move every
                       # figure without moving any count. Only a digest catches
                       # that, and check_freshness compares this one.
                       "answers": (stamp(ANSWERS)["digest"]
                                   if ANSWERS.exists() else None),
                       "per_corpus": {}}}
    reg = corpora.registry()
    answers = answer_quality()

    for name, cfg in reg.items():
        if not cfg["indexed"] or not cfg["golden"]:
            continue
        res = load(cfg, "results")
        per = load(cfg, "per_case")
        if not res or not per:
            print(f"  {name}: no eval artefacts, skipped")
            continue
        data["inputs"]["per_corpus"][name] = {
            "golden": stamp(cfg["golden"]),
            "per_case": stamp(artefact("per_case", cfg["golden"])),
            "results": stamp(artefact("results", cfg["golden"])),
        }

        cases = per["cases"]
        ans = [c["confidence"] for c in cases if not c["unanswerable"]]
        adv = [c["confidence"] for c in cases if c["unanswerable"]]
        gold = [len(c["gold_sources"]) for c in cases if not c["unanswerable"]]
        stats = corpora.stats(cfg)
        thr = cfg["threshold"] if cfg["calibrated"] else 0.0

        entry = {
            "name": name,
            "label": cfg["label"],
            "documents": stats.get("documents"),
            "chunks": stats.get("chunks"),
            "formats": stats.get("formats", {}),
            "threshold": thr,
            "calibrated": cfg["calibrated"],
            "n_answerable": len(ans),
            "n_adversarial": len(adv),

            # How often the FIRST passage is the one that answers, which is
            # what a reader experiences and what no published figure here
            # carried. Every other number is @5, and MRR is an average of
            # reciprocals that does not read as "is the top result right".
            # Counted from the per-case rankings rather than re-derived, so it
            # cannot disagree with the rows the site draws.
            "answer_first": answer_first(per),

            # The ladder, and the pool it draws candidates from.
            "ladder": [{"config": k, **res["end_to_end"][k]}
                       for k in LADDER if k in res["end_to_end"]],
            "pool": [{"config": k, **v} for k, v in res["candidate_pool"].items()],

            # Abstention. `dist` is every score, because the overlap between the
            # two populations is the reason a single threshold cannot serve all
            # three corpora, and a summary statistic hides an overlap.
            "dist": {"answerable": [round(c, 3) for c in sorted(ans)],
                     "adversarial": [round(c, 3) for c in sorted(adv)]},
            "median_answerable": round(st.median(ans), 3),
            "median_adversarial": round(st.median(adv), 3),
            "sweep": sweep(ans, adv),
            "at_zero": {"refused": sum(1 for c in ans if c < 0.0),
                        "caught": sum(1 for c in adv if c < 0.0)},
            "at_calibrated": {"refused": sum(1 for c in ans if c < thr),
                              "caught": sum(1 for c in adv if c < thr)},

            # Why source recall is not comparable across corpora as a raw number:
            # it is normalised by min(|gold|, 5), and the corpora do not agree on
            # how many documents answer a question.
            "gold_mean": round(st.mean(gold), 2),
            "gold_divisor": round(st.mean([min(g, 5) for g in gold]), 2),

            # Suggested questions for this corpus. The UI used to offer one
            # global list written for the ML papers, so switching to the bird
            # corpus still suggested asking about BLEU scores on WMT 2014.
            "examples": examples(cases),
        }
        # The only measurement here produced by a language model rather than by
        # arithmetic over rankings, and the only one that would move under a
        # different generator. Absent when that corpus has none.
        if name in answers:
            entry["answers"] = answers[name]
        data["corpora"].append(entry)
        print(f"  {name}: {len(ans)} answerable, {len(adv)} adversarial, "
              f"{len(entry['ladder'])} ladder rows"
              + (f", {entry['answers']['n']} answers scored"
                 if "answers" in entry else ""))

    OUT.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB)")
    return data


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    build()
