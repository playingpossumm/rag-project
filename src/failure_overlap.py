"""Are the questions retrieval gets wrong the SAME questions under every setting?

The system misses roughly 15% of answerable questions, and the cause has been
described as cross-document confusion: the retriever matches the topic and
ignores the constraint that distinguishes the answer. That is a claim about
*which* questions fail, and until now nothing checked it.

Repeating one configuration proves nothing -- retrieval is deterministic, so the
same settings return the same passages every time. The variation has to come
from changing the pipeline. So this compares the failure sets across
configurations and splits them:

    structural   failed under EVERY configuration. No amount of fusion,
                 reranking or capping reaches these. They are query-side
                 problems, and they are the fixture worth optimising against.
    contested    failed under some and not others. Ranking work moves these,
                 so a change can be scored on them.

The distinction matters because it says where effort should go. A structural
case will not respond to a better reranker; a contested one might.

    .venv\\Scripts\\python.exe src\\per_case.py --emit runs\\default.json
    .venv\\Scripts\\python.exe src\\failure_overlap.py runs\\*.json
"""
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# An answerable case counts as a failure if nothing relevant came back, or if
# the gate refused a question the corpus can answer. Both are "the user did not
# get their answer", which is the outcome being characterised.
ANSWERABLE_FAILURE = {"missed", "refused_wrongly"}


def load(paths):
    runs = {}
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        name = Path(p).stem.replace("per_case.", "")
        runs[name] = data
    return runs


def main() -> int:
    args = sys.argv[1:]
    fixture_out = None
    if "--emit-fixture" in args:
        i = args.index("--emit-fixture")
        fixture_out = Path(args[i + 1])
        del args[i:i + 2]
    paths = args
    if not paths:
        print(__doc__)
        return 2

    runs = load(paths)
    names = list(runs)

    # by_case[id] = {config: outcome}
    by_case: dict[str, dict] = {}
    meta: dict[str, dict] = {}
    for name, data in runs.items():
        for c in data["cases"]:
            by_case.setdefault(c["id"], {})[name] = c["outcome"]
            meta.setdefault(c["id"], c)

    answerable = [i for i in by_case if not meta[i]["unanswerable"]]
    adversarial = [i for i in by_case if meta[i]["unanswerable"]]

    print(f"{len(answerable)} answerable + {len(adversarial)} adversarial, "
          f"{len(names)} configurations\n")
    print(f"{'configuration':<12}{'missed':>8}{'refused':>9}{'fail %':>9}"
          f"{'adversarial answered':>23}")
    for name in names:
        d = runs[name]["totals"]
        miss, ref = d["answerable"]["missed"], d["answerable"]["refused_wrongly"]
        n = d["answerable"]["n"]
        print(f"{name:<12}{miss:>8}{ref:>9}{(miss + ref) / n * 100:>8.1f}%"
              f"{d['adversarial']['answered_anyway']:>23}")

    def failures(case_id):
        return {n for n, o in by_case[case_id].items() if o in ANSWERABLE_FAILURE}

    ever = [i for i in answerable if failures(i)]
    always = [i for i in ever if len(failures(i)) == len(names)]
    sometimes = [i for i in ever if i not in always]

    print(f"\n{'-' * 68}")
    print(f"{len(ever)} of {len(answerable)} answerable cases fail under at least one "
          f"configuration ({len(ever) / len(answerable) * 100:.0f}%)")
    print(f"  {len(always):>3} STRUCTURAL -- fail under all {len(names)}: no ranking change reaches them")
    print(f"  {len(sometimes):>3} CONTESTED  -- fail under some: ranking work moves these")

    print(f"\n{'-' * 68}\nSTRUCTURAL  (the fixture worth optimising against)\n")
    for i in sorted(always, key=lambda x: meta[x]["kind"]):
        c = meta[i]
        print(f"  {i:<16}{c['kind']:<10}gold: {', '.join(s[:26] for s in c['gold_sources'])}")
        print(f"  {'':<26}q: {c['question'][:88]}")
        got = ", ".join(f"{r['source'][:20]}:{r['locator'].split()[-1]}"
                        for r in c["results"][:4])
        print(f"  {'':<26}got: {got}\n")

    if sometimes:
        print(f"{'-' * 68}\nCONTESTED  (which settings rescue them)\n")
        for i in sorted(sometimes, key=lambda x: len(failures(x)), reverse=True):
            ok = [n for n in names if n not in failures(i)]
            print(f"  {i:<16}{meta[i]['kind']:<10}"
                  f"fails under {len(failures(i))}/{len(names)}; works under: {', '.join(ok)}")

    # ---- adversarial, gated configurations only -----------------------------
    # A configuration without reranking has no calibrated score, so the gate
    # does not run and every adversarial case is "answered" by construction.
    # Counting those alongside gated runs would report the absence of a gate as
    # a gate failing, and would put every case in the "answered under some
    # configuration" bucket for a reason that says nothing about the threshold.
    gated = [n for n in names
             if runs[n]["options"].get("use_reranker", True)]
    ungated = [n for n in names if n not in gated]

    print(f"\n{'-' * 68}\nADVERSARIAL -- answered when they should have been refused\n")
    if ungated:
        print(f"  excluding {', '.join(ungated)}: no reranking means no calibrated")
        print(f"  score, so the gate never runs and all {len(adversarial)} are answered by")
        print(f"  construction -- an absent gate, not a failing one.\n")
    if not gated:
        print("  no gated configuration supplied")
        return 0

    adv_ever = {i: {n for n in gated if by_case[i][n] == "answered_anyway"}
                for i in adversarial}
    adv_always = [i for i, f in adv_ever.items() if len(f) == len(gated)]
    adv_some = [i for i, f in adv_ever.items() if f and i not in adv_always]
    print(f"  over {len(gated)} gated configurations: {len(adv_always)} answered under "
          f"every one, {len(adv_some)} under some")
    for i in adv_always:
        print(f"    always  {i:<18}{meta[i]['confidence']:+.2f}  {meta[i]['question'][:62]}")
    for i in adv_some:
        print(f"    partly  {i:<18}{meta[i]['confidence']:+.2f}  {meta[i]['question'][:62]}")

    # ---- the fixture --------------------------------------------------------
    # Written out so a change can be scored against these seven in seconds
    # instead of a full harness run. Derived, never hand-listed: the set is
    # whatever currently fails everywhere, so it shrinks when something is
    # actually fixed rather than being a stale list someone has to remember to
    # prune.
    if fixture_out:
        fixture_out.parent.mkdir(exist_ok=True)
        fixture_out.write_text(json.dumps({
            "generated_by": "src/failure_overlap.py --emit-fixture",
            "configurations": names,
            "note": "Answerable cases that fail under every configuration listed above, "
                    "and adversarial cases answered under every gated one. Run with "
                    "src/hard_cases.py.",
            "structural": [
                {"id": i, "kind": meta[i]["kind"], "question": meta[i]["question"],
                 "gold_sources": meta[i]["gold_sources"]}
                for i in sorted(always)
            ],
            "slips_the_gate": [
                {"id": i, "question": meta[i]["question"],
                 "confidence": meta[i]["confidence"]}
                for i in sorted(adv_always)
            ],
        }, indent=1) + "\n", encoding="utf-8")
        print(f"\nwrote fixture: {fixture_out} "
              f"({len(always)} structural, {len(adv_always)} slipping the gate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
