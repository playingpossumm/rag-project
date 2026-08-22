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

ROOT = Path(__file__).parent.parent
OUT = ROOT / "eval" / "analytics.json"

# Which eval artefacts belong to which corpus. The ML corpus was the first and
# its files are unsuffixed, which is why this is a table rather than a pattern.
SUFFIX = {"llm": "", "birds": "-birds", "quant": "-quant"}

# The order the harness reports configurations in is the order they were added,
# which is not the order they compose in. This is the ladder: each row adds one
# mechanism to the row above it.
LADDER = ["dense, no rerank", "dense + rerank", "rrf + rerank",
          "+ diversity 2/src", "+ diversity 1/src"]
METRICS = [("hit_rate", "Hit rate @5"), ("mrr", "MRR"),
           ("ndcg", "NDCG"), ("src_recall", "Source recall")]


def load(name: str, stem: str):
    p = ROOT / "eval" / f"{stem}{SUFFIX[name]}.json"
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


def build():
    data = {"generated_by": "src/build_analytics.py", "corpora": []}
    reg = corpora.registry()

    for name, cfg in reg.items():
        if name not in SUFFIX or not cfg["indexed"]:
            continue
        res = load(name, "results")
        per = load(name, "per_case")
        if not res or not per:
            print(f"  {name}: no eval artefacts, skipped")
            continue

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
