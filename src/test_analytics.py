"""Which questions the front page offers, and in what order.

`check_freshness.py` proves the offered questions still exist in the golden set.
`test_serve.py` proves the interface falls back sanely when it cannot read them.
Neither says anything about **which** questions get chosen, and that choice is
`build_analytics.examples()` — a per-kind quota over the golden set that no test
asserted.

It decides what a visitor sees first, and it has a quiet failure mode the audit
already warns about from the other side: a case whose `kind` is not one of the
three the quota asks for is never offered to anyone, and nothing says so. This
file pins the quota, the order, the shortest-first rule and that silent drop, so
`check_golden.py`'s warning and this module's behaviour cannot drift apart.

Hermetic: a synthetic case list, no corpus, no models, milliseconds.

    .venv\\Scripts\\python.exe src\\test_analytics.py
"""
import sys

from build_analytics import ADVERSARIAL, KIND_LABEL, QUOTA, examples

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHECKS: list[tuple[str, object, object, bool]] = []


def check(name: str, got, want) -> None:
    CHECKS.append((name, got, want, got == want))


def case(cid, kind, question, unanswerable=False):
    return {"id": cid, "kind": kind, "question": question,
            "unanswerable": unanswerable}


# Five of each answerable kind against quotas of 3/3/2, and three adversarial
# against a quota of 2, so every bucket is over-subscribed and the quota has to
# do something. Lengths are deliberately out of order.
CASES = (
    [case(f"fact{i}", "fact", "f" * (30 - i) + "?") for i in range(5)]
    + [case(f"multi{i}", "multi", "m" * (40 - i) + "?") for i in range(5)]
    + [case(f"cross{i}", "cross-doc", "c" * (50 - i) + "?") for i in range(5)]
    + [case(f"adv{i}", "absent", "a" * (25 - i) + "?", unanswerable=True)
       for i in range(3)]
)

out = examples(CASES)
by_label = {}
for e in out:
    by_label.setdefault(e["label"], []).append(e["q"])

quota = dict(QUOTA)
check("the quota is the one this test was written against",
      QUOTA, [("fact", 3), ("multi", 3), ("cross-doc", 2), ("__adv__", 2)])
check("ten questions offered in total", len(out), 10)

check("three of the one-figure kind", len(by_label[KIND_LABEL["fact"]]), 3)
check("three spread over several documents", len(by_label[KIND_LABEL["multi"]]), 3)
check("two decoys", len(by_label[KIND_LABEL["cross-doc"]]), 2)
check("two that are not in the documents", len(by_label[ADVERSARIAL]), 2)

# Shortest first, because a suggestion has to be readable at a glance and the
# long ones are long because they carry three clauses of qualification.
facts = by_label[KIND_LABEL["fact"]]
check("the shortest of a kind is offered first",
      facts, sorted(facts, key=len))
check("and the longest of an over-subscribed kind is dropped",
      max(len(q) for q in facts) < 30, True)

# The order the quota lists, not the order the golden set happens to be in.
check("the kinds appear in the quota's order",
      [e["label"] for e in out][:1] + [e["label"] for e in out][-1:],
      [KIND_LABEL["fact"], ADVERSARIAL])

# What the interface tells the reader to expect. The adversarial ones are
# offered ON PURPOSE -- a system that refuses them is working -- so the label
# has to say that or a refusal reads as a failure.
adv = [e for e in out if e["adversarial"]]
check("every adversarial example is flagged", len(adv), 2)
check("and says it should be refused",
      {e["expect"] for e in adv}, {"should be refused"})
check("while the others say they should be answered",
      {e["expect"] for e in out if not e["adversarial"]}, {"should be answered"})

# The quiet one: a kind the quota does not ask for is never offered, and
# check_golden.py reports exactly this from the other direction.
with_unknown = CASES + [case("odd1", "figure", "What does figure 3 show?")]
check("a case whose kind the quota does not name is never offered",
      any(e["q"] == "What does figure 3 show?" for e in examples(with_unknown)),
      False)

# A corpus too small to fill a bucket must not borrow from another one.
thin = [case("f1", "fact", "short?"), case("a1", "absent", "absent?", True)]
thin_out = examples(thin)
check("a bucket with fewer cases than its quota offers what it has",
      len(thin_out), 2)
check("and does not pad from another kind",
      sorted(e["label"] for e in thin_out),
      sorted([KIND_LABEL["fact"], ADVERSARIAL]))
check("an empty case list offers nothing rather than raising",
      examples([]), [])


def main() -> int:
    width = max(len(n) for n, *_ in CHECKS)
    failed = 0
    for name, got, want, ok in CHECKS:
        if not ok:
            failed += 1
            print(f"FAIL  {name:<{width}}  got {got!r}, want {want!r}")
    print(f"{len(CHECKS) - failed}/{len(CHECKS)} analytics checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
