"""Name the questions each abstention threshold costs, not just the rate.

`evaluate.py` sweeps the threshold and prints a net-separation column, and on
every run it reports that the best net separation is at **+2** while the shipped
threshold is **0.0**. That gap has stood unexplained-in-numbers since the sweep
was written: the project's answer is that false abstention is the costlier
error, which is a policy, and the sweep's `net` column silently assumes the two
errors cost the same.

This makes the trade concrete in two ways the aggregate cannot.

**It names the cases.** Moving the threshold does not cost "6.1% of answerable
questions", it costs four specific questions, and whether that is acceptable is
a judgement about those four.

**It reports the implied cost ratio.** `net = caught - false_abstain` ranks
thresholds only under equal costs. Weighting a false abstention w times a
missed refusal, the shipped threshold wins whenever w exceeds the break-even
printed below -- which converts "false abstention is costlier" from an assertion
into a number that can be agreed or disagreed with.

One caveat the aggregate hides entirely, and it matters more than either: the
gate costs different things on different paths.

  api.ask()   the flag is ADVISORY. Passages and citations are returned either
              way, so a false abstention mislabels a good answer -- annoying.
  the UI      the answer is withheld and replaced with "The corpus does not
              contain this" -- a false abstention loses an answer the corpus
              has.

One number is serving two cost structures. Read the table with the path in mind.

    .venv\\Scripts\\python.exe src\\calibrate_threshold.py
    .venv\\Scripts\\python.exe src\\calibrate_threshold.py --emit eval/threshold.json
"""
import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_IN = Path(__file__).parent.parent / "eval" / "per_case.json"
STEPS = [-4, -3, -2, -1, 0, 1, 2, 3, 4]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-case", type=Path, default=DEFAULT_IN)
    ap.add_argument("--emit", type=Path)
    args = ap.parse_args()

    data = json.loads(args.per_case.read_text(encoding="utf-8"))
    shipped = data["threshold"]
    cases = data["cases"]

    # Every case carries the top rerank score the gate reads. A case with no
    # score (the no-rerank path) cannot be calibrated and is excluded rather
    # than counted as passing, which is what treating None as 0 would do.
    answerable = [c for c in cases
                  if not c["unanswerable"] and c["confidence"] is not None]
    adversarial = [c for c in cases
                   if c["unanswerable"] and c["confidence"] is not None]
    if not answerable or not adversarial:
        print("per_case.json has no calibrated scores -- rerun without --no-rerank")
        return 2

    print(f"{len(answerable)} answerable, {len(adversarial)} adversarial "
          f"(shipped threshold {shipped:+.1f})\n")
    print(f"{'threshold':>10}{'refuses':>9}{'of adversarial':>16}"
          f"{'wrongly refuses':>17}{'of answerable':>15}")

    rows = []
    for t in STEPS:
        caught = [c for c in adversarial if c["confidence"] < t]
        lost = [c for c in answerable if c["confidence"] < t]
        rows.append({"threshold": t,
                     "caught": [c["id"] for c in caught],
                     "wrongly_refused": [c["id"] for c in lost]})
        mark = "  <- shipped" if t == shipped else ""
        print(f"{t:>10}{len(caught):>9}{len(caught) / len(adversarial):>15.0%}"
              f"{len(lost):>17}{len(lost) / len(answerable):>14.1%}{mark}")

    # ---- what each step upward actually costs, by name ---------------------
    print(f"\n{'-' * 72}\nWhat raising the threshold costs, question by question\n")
    prev = set()
    for row in rows:
        lost_now = set(row["wrongly_refused"])
        new = lost_now - prev
        prev = lost_now
        if not new or row["threshold"] < shipped:
            continue
        print(f"  at {row['threshold']:+d}, these answerable questions start being refused:")
        for cid in sorted(new):
            c = next(x for x in answerable if x["id"] == cid)
            print(f"    {cid:<18}{c['confidence']:+6.2f}  {c['question'][:66]}")
        print()

    # ---- what LOWERING would recover ---------------------------------------
    # This tool assumed the shipped threshold was a floor and only ever swept
    # upward, which is true of the corpus it was written against and false in
    # general. On a corpus the reranker scores lower, the shipped value is too
    # HIGH and the interesting move is downward -- and the tool could not say so.
    base0 = next(r for r in rows if r["threshold"] == shipped)
    better = [r for r in rows if r["threshold"] < shipped
              and len(r["wrongly_refused"]) < len(base0["wrongly_refused"])]
    if better:
        print(f"\n{'-' * 72}\nWhat LOWERING the threshold would recover\n")
        print(f"  {'threshold':>10}{'still caught':>15}{'wrongly refused':>18}{'recovered':>12}")
        for r in sorted(better, key=lambda r: -r["threshold"]):
            rec = len(base0["wrongly_refused"]) - len(r["wrongly_refused"])
            lost = len(base0["caught"]) - len(r["caught"])
            note = "  same catching" if lost == 0 else f"  costs {lost} caught"
            print(f"  {r['threshold']:>+10}{len(r['caught']):>15}"
                  f"{len(r['wrongly_refused']):>18}{rec:>+12}{note}")
        free = [r for r in better if len(r["caught"]) == len(base0["caught"])]
        if free:
            best = max(free, key=lambda r: -r["threshold"])
            names = sorted(set(base0["wrongly_refused"]) - set(best["wrongly_refused"]))
            print(f"\n  {best['threshold']:+d} catches exactly as many and recovers "
                  f"{len(names)} answerable question(s) for nothing:")
            for cid in names:
                c = next(x for x in answerable if x["id"] == cid)
                print(f"    {cid:<18}{c['confidence']:+6.2f}  {c['question'][:62]}")

    # ---- the implied cost ratio --------------------------------------------
    base = next(r for r in rows if r["threshold"] == shipped)
    print(f"{'-' * 72}\nThe cost ratio the shipped threshold implies\n")
    print("  `net = caught - wrongly_refused` ranks thresholds only if the two errors")
    print("  cost the same. Weighting a wrong refusal w times a missed refusal:\n")
    print(f"  {'vs':>6}{'more caught':>14}{'more lost':>12}{'break-even w':>15}")
    for row in rows:
        t = row["threshold"]
        if t <= shipped:
            continue
        dc = len(row["caught"]) - len(base["caught"])
        dl = len(row["wrongly_refused"]) - len(base["wrongly_refused"])
        w = (dc / dl) if dl else float("inf")
        shown = "never wins" if dl == 0 else f"{w:.1f}x"
        print(f"  {t:>+6}{dc:>14}{dl:>12}{shown:>15}")
    print(f"\n  Read: {shipped:+.1f} beats a higher threshold whenever a wrongly refused")
    print("  question is worth more than that multiple of an unanswerable one slipping")
    print("  through. The project's stated position -- false abstention is the costlier")
    print("  error -- is the claim that it is.")

    if args.emit:
        args.emit.parent.mkdir(exist_ok=True)
        args.emit.write_text(json.dumps(
            {"generated_by": "src/calibrate_threshold.py",
             "shipped_threshold": shipped,
             "n_answerable": len(answerable), "n_adversarial": len(adversarial),
             "sweep": rows}, indent=1) + "\n", encoding="utf-8")
        print(f"\nwrote {args.emit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
