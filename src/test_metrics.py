"""Hand-computed checks on the scoring functions every published number rests on.

The field notes say NDCG was "verified against four hand-computed cases -- all
hits at top returning exactly 1.0, hits demoted returning 0.571, no hits
returning 0.0". Those cases were computed. They were never committed, so
nothing re-ran them, and this is the second claimed-but-absent test in the repo
after test_trace_matches_pipeline.

It matters more here than it did there. `evaluate.py` is the tool that decides
whether every other change helped, its NDCG once printed **1.373** -- impossible
for a normalised metric -- and that bug was caught only because a human noticed
the output violated a bound. A metric with no known bound would have shipped
wrong and stayed wrong. These assertions are that bound, written down.

Pure arithmetic, no corpus and no models, so it runs in milliseconds:

    .venv\\Scripts\\python.exe src\\test_metrics.py
"""
import math
import sys

from evaluate import (context_recall, hit_rate, ndcg, normalize,
                      reciprocal_rank, source_recall)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Discount at rank i is 1/log2(i+1): rank 1 = 1.0, rank 2 = 0.6309, rank 3 = 0.5,
# rank 4 = 0.4307. Spelled out because every expected value below is derived
# from these by hand, and a reader should be able to check the arithmetic
# without trusting the code that produced it.
D = [1.0 / math.log2(i + 1) for i in range(1, 8)]


def res(source: str, page: int, text: str = "") -> dict:
    return {"source": source, "locator": {"kind": "page", "value": page}, "text": text}


def gold_of(*pairs) -> set:
    return {(src, "page", str(p)) for src, p in pairs}


CHECKS: list[tuple] = []


def check(name, got, want, tol=5e-4):
    ok = (abs(got - want) <= tol) if isinstance(want, float) else got == want
    CHECKS.append((name, got, want, ok))


# ---------------------------------------------------------------- NDCG ----
GOLD = gold_of(("a.pdf", 1), ("a.pdf", 2))

# 1. every hit already at the top -- the ideal ordering, by definition 1.0
check("ndcg / all hits at top",
      ndcg([res("a.pdf", 1), res("a.pdf", 2), res("b.pdf", 9)], GOLD), 1.0)

# 2. the same two hits pushed to ranks 3 and 4
#    DCG  = 1/log2(4) + 1/log2(5) = 0.5000 + 0.4307 = 0.9307
#    IDCG = 1/log2(2) + 1/log2(3) = 1.0000 + 0.6309 = 1.6309
#    NDCG = 0.9307 / 1.6309 = 0.571
check("ndcg / hits demoted to 3 and 4",
      ndcg([res("b.pdf", 8), res("b.pdf", 9), res("a.pdf", 1), res("a.pdf", 2)], GOLD),
      (D[2] + D[3]) / (D[0] + D[1]))
check("ndcg / that value is the documented 0.571",
      round((D[2] + D[3]) / (D[0] + D[1]), 3), 0.571)

# 3. nothing relevant returned
check("ndcg / no hits", ndcg([res("b.pdf", 8), res("b.pdf", 9)], GOLD), 0.0)

# 4. THE REGRESSION. Relevance is judged per page, so several returned chunks
#    can share one gold page. Normalising by len(gold) let DCG exceed IDCG and
#    printed 1.373. The ideal is the hits actually found, moved to the top, so
#    three chunks from one gold page must still score exactly 1.0 -- never more.
one_page = gold_of(("a.pdf", 1))
crowded = ndcg([res("a.pdf", 1), res("a.pdf", 1), res("a.pdf", 1)], one_page)
check("ndcg / three chunks share one gold page", crowded, 1.0)
check("ndcg / never exceeds 1.0", crowded <= 1.0, True)

# ------------------------------------------------------------ hit rate ----
check("hit_rate / relevant present", hit_rate([res("b.pdf", 9), res("a.pdf", 2)], GOLD), 1.0)
check("hit_rate / nothing relevant", hit_rate([res("b.pdf", 9)], GOLD), 0.0)
check("hit_rate / empty result set", hit_rate([], GOLD), 0.0)
# Position-blind by design: rank 1 and rank 5 score the same. Asserted so the
# saturation the harness documents cannot be quietly "fixed" into something else.
check("hit_rate / ignores position",
      hit_rate([res("b.pdf", 7), res("b.pdf", 8), res("b.pdf", 9), res("a.pdf", 1)], GOLD),
      hit_rate([res("a.pdf", 1)], GOLD))

# --------------------------------------------------------------- MRR -----
check("mrr / first result relevant", reciprocal_rank([res("a.pdf", 1)], GOLD), 1.0)
check("mrr / third result relevant",
      reciprocal_rank([res("b.pdf", 8), res("b.pdf", 9), res("a.pdf", 1)], GOLD), 1 / 3)
check("mrr / none relevant", reciprocal_rank([res("b.pdf", 9)], GOLD), 0.0)
# Only the FIRST hit counts -- one perfect result scores like five.
check("mrr / blind to everything after the first hit",
      reciprocal_rank([res("a.pdf", 1), res("a.pdf", 2)], GOLD),
      reciprocal_rank([res("a.pdf", 1), res("b.pdf", 9)], GOLD))

# ------------------------------------------------------ source recall ----
srcs = {"a.pdf", "b.pdf", "c.pdf", "d.pdf"}
check("src_recall / two of four documents, k=5",
      source_recall([res("a.pdf", 1), res("b.pdf", 1)], srcs, k=5), 0.5)
# Normalised by min(|gold|, k): k results cannot represent more than k documents,
# so a caller's small k must not be scored as a retrieval failure.
check("src_recall / k below the gold count is not penalised",
      source_recall([res("a.pdf", 1), res("b.pdf", 1)], srcs, k=2), 1.0)
check("src_recall / duplicates from one document count once",
      source_recall([res("a.pdf", 1), res("a.pdf", 2), res("a.pdf", 3)], srcs, k=5), 0.25)
check("src_recall / no gold sources is not-a-number",
      math.isnan(source_recall([res("a.pdf", 1)], set(), k=5)), True)

# ----------------------------------------------------- context recall ----
check("context_recall / phrase present",
      context_recall([res("a.pdf", 1, "the model achieves 28.4 BLEU on WMT")], "28.4 BLEU"), 1.0)
check("context_recall / phrase absent",
      context_recall([res("a.pdf", 1, "unrelated text")], "28.4 BLEU"), 0.0)
# The documented reason normalize() exists: a phrase split across a line break
# in the parsed PDF must still match.
check("context_recall / phrase spanning a line break",
      context_recall([res("a.pdf", 1, "the model achieves 28.4\n   BLEU on WMT")], "28.4 BLEU"), 1.0)
check("normalize / collapses whitespace and case",
      normalize("  The  MODEL\n\tachieves "), "the model achieves")


def main() -> int:
    width = max(len(n) for n, *_ in CHECKS)
    failed = 0
    for name, got, want, ok in CHECKS:
        if not ok:
            failed += 1
            print(f"FAIL  {name:<{width}}  got {got!r}, want {want!r}")
    print(f"{len(CHECKS) - failed}/{len(CHECKS)} metric checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
