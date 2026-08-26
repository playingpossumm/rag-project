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
    "fact": "one figure in one place",
    "multi": "spread over several documents",
    "cross-doc": "decoys that look right",
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


def build():
    data = {"generated_by": "src/build_analytics.py", "corpora": [],
            # Which files each entry was built from. This is what makes a
            # rebuilt per_case.json that nobody followed with a rebuild here
            # detectable: every count still agrees and every number is old.
            # src/check_freshness.py compares these against what is on disk.
            "inputs": {"corpora_json": stamp(corpora.CONFIG)["digest"],
                       "per_corpus": {}}}
    reg = corpora.registry()

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
        data["corpora"].append(entry)
        print(f"  {name}: {len(ans)} answerable, {len(adv)} adversarial, "
              f"{len(entry['ladder'])} ladder rows")

    OUT.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB)")
    return data


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    build()
