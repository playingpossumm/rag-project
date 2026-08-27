"""Is what the interface offers still what the harness measured?

Three times a generated file has gone stale without anything erroring.
`results.json` was shared by every corpus, so whichever evaluation ran last
owned it. `RESULTS.md` said its numbers were copied from that file and they were
typed by hand. `per_case.json` had the same shared-path bug and sat four days
out of date, which meant the front page was offering questions that had already
been deleted from the golden set as unanswerable -- the system inviting you to
ask something it had itself concluded it could not answer.

Only `RESULTS.md` grew a `--check`. This is the same idea for the other chain:

    golden set  ->  per_case*.json  ->  analytics.json  ->  the front page

Two kinds of check, deliberately, because they fail differently:

**Provenance.** Each generated file records a digest of the files it was built
from. A digest that no longer matches means the input changed after the output
was written -- exact, and it catches an edit that changes no count at all, such
as rewording one question in place.

**Semantics.** Case ids, question strings, corpus sizes and the per-corpus
threshold and rerank blend are compared directly. This is the weaker check and
the more useful one: it works on files written before provenance existed, it
says which question is missing rather than that a hash moved, and it is what
catches the interface offering a question the corpus no longer has.

Config drift is checked here too, and found a live instance on the first run:
`per_case.py` scored every corpus at the ML papers' 0.0 threshold and at rerank
blend 0.0, while the bird corpus ships -3.0 and 0.20 and quant ships -4.0. The
file claimed ten bird questions were wrongly refused; the served configuration
wrongly refuses four. Nothing errored, because a threshold is just a number.

    .venv\\Scripts\\python.exe src\\check_freshness.py          # report, exit 1 if stale
    .venv\\Scripts\\python.exe src\\check_freshness.py --quiet   # only the problems

Exit code is the contract: 0 current, 1 stale, 2 the chain is broken (a file the
chain needs is missing entirely).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import corpora  # noqa: E402

ROOT = Path(__file__).parent.parent
EVAL = ROOT / "eval"
ANALYTICS = EVAL / "analytics.json"

# Enough of a sha256 that a collision is not a thing to think about, short
# enough to sit on one line of a diff. Full hashes made every regeneration a
# wall of noise in `git diff` and nobody read them.
DIGEST_CHARS = 16


def digest(path: Path) -> str:
    """Content fingerprint of one file, or "missing" if it is not there.

    Bytes, not parsed JSON: re-serialising to compare would make the digest
    depend on this module's json settings rather than on the file, and two files
    that differ only in indentation are still two different files on disk.
    """
    if not path.exists():
        return "missing"
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{h[:DIGEST_CHARS]}"


def stamp(path: Path, **extra) -> dict:
    """The provenance record for one input file."""
    rel = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
    return {"path": rel, "digest": digest(path), **extra}


def artefact_suffix(golden: Path) -> str:
    """`golden-birds.json` -> `-birds`, so per_case-birds.json is its output.

    The ML corpus came first and its files are unsuffixed, which is the whole
    reason this rule exists rather than a plain stem. It lived in three copies
    -- evaluate.py, per_case.py and a hardcoded table in build_analytics.py --
    and three copies of a naming rule is how a corpus gets added to one and not
    the others.
    """
    stem = Path(golden).stem
    if stem in ("golden_set", "golden"):
        return ""
    return "-" + stem.split("-", 1)[-1]


def artefact(kind: str, golden: Path) -> Path:
    """Where `kind` (`per_case` / `results`) for this golden set lives."""
    return EVAL / f"{kind}{artefact_suffix(golden)}.json"


def load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


class Report:
    """Findings, grouped by corpus, with the command that fixes each one.

    A stale file is worth reporting alongside the exact invocation that rebuilds
    it, in the order the chain runs. Rebuilding analytics before per_case just
    copies the stale numbers forward, and that is not obvious from a list of
    complaints.
    """

    def __init__(self):
        self.rows: list[tuple[str, str, str, str]] = []   # scope, item, state, detail
        self.fixes: list[str] = []
        self.broken = False

    def ok(self, scope, item, detail=""):
        self.rows.append((scope, item, "ok", detail))

    def stale(self, scope, item, detail, fix=None):
        self.rows.append((scope, item, "STALE", detail))
        if fix and fix not in self.fixes:
            self.fixes.append(fix)

    def missing(self, scope, item, detail, fix=None):
        self.broken = True
        self.rows.append((scope, item, "MISSING", detail))
        if fix and fix not in self.fixes:
            self.fixes.append(fix)

    def note(self, scope, item, detail):
        """Something to know about, not something that is wrong."""
        self.rows.append((scope, item, "note", detail))

    @property
    def problems(self):
        return [r for r in self.rows if r[2] in ("STALE", "MISSING")]


def check_per_case(rep: Report, name: str, cfg: dict, gold: dict, gold_path: Path):
    """per_case*.json against the golden set, the index, and corpora.json."""
    path = artefact("per_case", gold_path)
    fix = (f".venv\\Scripts\\python.exe src\\per_case.py "
           f"--golden {gold_path.relative_to(ROOT).as_posix()}")
    pc = load(path)
    if pc is None:
        rep.missing(name, "per_case", f"{path.name} is missing or unreadable", fix)
        return None

    rel = path.relative_to(ROOT).as_posix()

    # Provenance first, because it is exact. A digest that no longer matches
    # catches an edit that changes no count -- rewording one question in place
    # leaves every semantic check below satisfied.
    inputs = pc.get("inputs")
    if not inputs:
        rep.note(name, "per_case", f"{rel} predates provenance; checked by content only")
    else:
        got = inputs.get("golden", {}).get("digest")
        want = digest(gold_path)
        if got != want:
            rep.stale(name, "per_case",
                      f"golden set has changed since {rel} was written "
                      f"({got} -> {want})", fix)
        store_meta = cfg["store"] / "metadata.json"
        got_i = inputs.get("index", {}).get("digest")
        want_i = digest(store_meta)
        if got_i and got_i != want_i:
            rep.stale(name, "per_case",
                      f"the index has been rebuilt since {rel} was written", fix)

    # Semantics. These work on a file written before provenance existed, and
    # they name the case rather than the hash, which is what a person needs.
    gold_ids = {c["id"]: c for c in gold["cases"]}
    pc_ids = {c["id"]: c for c in pc.get("cases", [])}
    dropped = sorted(set(pc_ids) - set(gold_ids))
    added = sorted(set(gold_ids) - set(pc_ids))
    if dropped:
        rep.stale(name, "per_case",
                  f"scores {len(dropped)} case(s) the golden set no longer has: "
                  f"{', '.join(dropped[:4])}{' ...' if len(dropped) > 4 else ''}", fix)
    if added:
        rep.stale(name, "per_case",
                  f"{len(added)} golden case(s) have never been scored: "
                  f"{', '.join(added[:4])}{' ...' if len(added) > 4 else ''}", fix)

    reworded = [i for i in set(gold_ids) & set(pc_ids)
                if gold_ids[i]["question"] != pc_ids[i]["question"]]
    if reworded:
        rep.stale(name, "per_case",
                  f"{len(reworded)} question(s) reworded since scoring: "
                  f"{', '.join(sorted(reworded)[:4])}", fix)

    flipped = [i for i in set(gold_ids) & set(pc_ids)
               if bool(gold_ids[i].get("unanswerable")) != bool(pc_ids[i]["unanswerable"])]
    if flipped:
        rep.stale(name, "per_case",
                  f"{len(flipped)} case(s) changed between answerable and "
                  f"adversarial: {', '.join(sorted(flipped)[:4])}", fix)

    # The index it was scored against. A re-ingest that adds documents leaves
    # every case id intact and every number wrong.
    stats = corpora.stats(cfg)
    got_c = pc.get("corpus", {})
    if stats and got_c and (got_c.get("chunks") != stats.get("chunks")
                            or got_c.get("documents") != stats.get("documents")):
        rep.stale(name, "per_case",
                  f"scored against {got_c.get('documents')} documents / "
                  f"{got_c.get('chunks'):,} chunks; the index now holds "
                  f"{stats.get('documents')} / {stats.get('chunks'):,}", fix)

    # Config drift. The single most reused finding in this project is that a
    # corpus setting does not transfer, so a file that recorded one corpus's
    # threshold while describing another's questions is measuring a pipeline
    # nobody serves.
    want_thr = cfg["threshold"] if cfg["calibrated"] else 0.0
    if pc.get("threshold") is not None and abs(pc["threshold"] - want_thr) > 1e-9:
        rep.stale(name, "per_case",
                  f"scored at threshold {pc['threshold']:+.1f}; corpora.json "
                  f"ships {want_thr:+.1f} for this corpus", fix)
    got_cand = pc.get("options", {}).get("candidate_k")
    if got_cand is not None and got_cand != cfg.get("candidate_k", got_cand):
        rep.stale(name, "per_case",
                  f"scored over {got_cand} candidates; corpora.json ships "
                  f"{cfg.get('candidate_k')} for this corpus", fix)
    got_blend = pc.get("options", {}).get("rerank_blend")
    if got_blend is not None and abs(got_blend - cfg["rerank_blend"]) > 1e-9:
        rep.stale(name, "per_case",
                  f"scored at rerank blend {got_blend:.2f}; corpora.json ships "
                  f"{cfg['rerank_blend']:.2f}", fix)
    elif got_blend is None and cfg["rerank_blend"]:
        rep.note(name, "per_case",
                 f"{rel} does not record a rerank blend; this corpus ships "
                 f"{cfg['rerank_blend']:.2f}")

    if not [r for r in rep.rows if r[0] == name and r[1] == "per_case"
            and r[2] in ("STALE", "MISSING")]:
        rep.ok(name, "per_case", f"{len(pc_ids)} cases, {rel}")
    return pc


def check_results(rep: Report, name: str, cfg: dict, gold: dict, gold_path: Path):
    """results*.json -- the aggregate half of what analytics.json reads."""
    path = artefact("results", gold_path)
    fix = (f".venv\\Scripts\\python.exe src\\evaluate.py "
           f"--golden {gold_path.relative_to(ROOT).as_posix()}")
    res = load(path)
    if res is None:
        rep.missing(name, "results", f"{path.name} is missing or unreadable", fix)
        return None

    rel = path.relative_to(ROOT).as_posix()

    # The digest half, the same as per_case gets. Every check below this one
    # compares a count, and the edit this is here for changes no count: a
    # corrected locator in one label leaves the case totals, the corpus size and
    # every metric exactly as they were.
    inputs = res.get("inputs")
    if not inputs:
        rep.note(name, "results", f"{rel} predates provenance; checked by content only")
    else:
        got = inputs.get("golden", {}).get("digest")
        want = digest(gold_path)
        if got != want:
            rep.stale(name, "results",
                      f"golden set has changed since {rel} was measured "
                      f"({got} -> {want})", fix)
        got_i = (inputs.get("index") or {}).get("digest")
        want_i = digest(cfg["store"] / "metadata.json")
        if got_i and got_i != want_i:
            rep.stale(name, "results",
                      f"the index has been rebuilt since {rel} was measured", fix)

    c = res.get("corpus") or {}
    n_ans = sum(1 for x in gold["cases"] if not x.get("unanswerable"))
    n_adv = len(gold["cases"]) - n_ans
    if (c.get("answerable_cases"), c.get("adversarial_cases")) != (n_ans, n_adv):
        rep.stale(name, "results",
                  f"measured {c.get('answerable_cases')} answerable + "
                  f"{c.get('adversarial_cases')} adversarial; the golden set now "
                  f"holds {n_ans} + {n_adv}", fix)

    # The golden set it names. results.json records an absolute path, which is
    # the one thing here that cannot be compared across machines, so only the
    # filename is checked -- enough to catch a bird run filling an ML file.
    named = Path(c["golden"]).name if c.get("golden") else None
    if named and named != gold_path.name:
        rep.stale(name, "results",
                  f"was run against {named}, not {gold_path.name}", fix)

    stats = corpora.stats(cfg)
    if stats and (c.get("chunks") != stats.get("chunks")
                  or c.get("documents") != stats.get("documents")):
        rep.stale(name, "results",
                  f"measured {c.get('documents')} documents / "
                  f"{c.get('chunks'):,} chunks; the index now holds "
                  f"{stats.get('documents')} / {stats.get('chunks'):,}", fix)

    # The harness records the settings it ran under, and they are the same two
    # that do not transfer between corpora. It read both from module constants
    # until 2026-08-26, so a bird run wrote `shipped_threshold: 0.0` under a
    # corpus that ships -3.0 and printed a paragraph reasoning about a gate
    # nobody serves.
    want_thr = cfg["threshold"] if cfg["calibrated"] else 0.0
    got_thr = (res.get("abstention") or {}).get("shipped_threshold")
    if got_thr is not None and abs(got_thr - want_thr) > 1e-9:
        rep.stale(name, "results",
                  f"records a shipped threshold of {got_thr:+.1f}; corpora.json "
                  f"ships {want_thr:+.1f}", fix)
    got_cand = c.get("candidate_k")
    if got_cand is not None and got_cand != cfg.get("candidate_k", got_cand):
        rep.stale(name, "results",
                  f"measured over {got_cand} candidates; corpora.json ships "
                  f"{cfg.get('candidate_k')}", fix)
    got_blend = c.get("rerank_blend")
    if got_blend is not None and abs(got_blend - cfg["rerank_blend"]) > 1e-9:
        rep.stale(name, "results",
                  f"was measured at rerank blend {got_blend:.2f}; corpora.json "
                  f"ships {cfg['rerank_blend']:.2f}", fix)

    if not [r for r in rep.rows if r[0] == name and r[1] == "results"
            and r[2] in ("STALE", "MISSING")]:
        rep.ok(name, "results", f"{n_ans} + {n_adv} cases, {path.name}")
    return res


def check_analytics(rep: Report, reg: dict, golds: dict, per_cases: dict):
    """analytics.json, and through it the questions the front page offers.

    This is the end of the chain and the only part a visitor sees. The failure
    it exists to prevent is specific and has happened: the page offered
    questions that had been deleted from the golden set for being unanswerable,
    so the system was inviting people to ask it things it had already concluded
    it could not answer.
    """
    scope = "analytics"
    fix = ".venv\\Scripts\\python.exe src\\build_analytics.py"
    data = load(ANALYTICS)
    if data is None:
        rep.missing(scope, "analytics.json",
                    "eval/analytics.json is missing or unreadable -- the front "
                    "page falls back to four hardcoded ML questions", fix)
        return

    entries = {c["name"]: c for c in data.get("corpora", [])}
    inputs = (data.get("inputs") or {}).get("per_corpus") or {}
    if not data.get("inputs"):
        rep.note(scope, "provenance",
                 "analytics.json predates provenance; checked by content only")

    expected = {n for n, cfg in reg.items() if cfg["indexed"] and n in per_cases}
    for name in sorted(expected - set(entries)):
        rep.stale(scope, name,
                  f"{reg[name]['label']} has eval artefacts but no entry in "
                  f"analytics.json; the front page offers it the fallback "
                  f"questions, which are about the ML papers", fix)
    for name in sorted(set(entries) - set(reg)):
        rep.stale(scope, name,
                  f"analytics.json describes {name!r}, which corpora.json no "
                  f"longer configures", fix)

    for name, entry in sorted(entries.items()):
        cfg = reg.get(name)
        if not cfg:
            continue

        # Provenance: was this entry built from the per_case file that is on
        # disk now? This is what catches a per_case re-run that nobody followed
        # with a rebuild -- every count still agrees, every number is old.
        rec = inputs.get(name, {})
        for kind in ("per_case", "results"):
            got = (rec.get(kind) or {}).get("digest")
            if not got:
                continue
            want = digest(artefact(kind, cfg["golden"]))
            if got != want:
                rep.stale(scope, name,
                          f"built from an older {kind}{artefact_suffix(cfg['golden'])}"
                          f".json ({got} -> {want})", fix)

        gold = golds.get(name)
        if not gold:
            continue
        by_q = {c["question"]: c for c in gold["cases"]}
        n_ans = sum(1 for c in gold["cases"] if not c.get("unanswerable"))
        n_adv = len(gold["cases"]) - n_ans

        if (entry.get("n_answerable"), entry.get("n_adversarial")) != (n_ans, n_adv):
            rep.stale(scope, name,
                      f"reports {entry.get('n_answerable')} + "
                      f"{entry.get('n_adversarial')} cases; the golden set holds "
                      f"{n_ans} + {n_adv}", fix)

        stats = corpora.stats(cfg)
        if stats and (entry.get("chunks") != stats.get("chunks")
                      or entry.get("documents") != stats.get("documents")):
            rep.stale(scope, name,
                      f"draws {entry.get('documents')} documents / "
                      f"{entry.get('chunks'):,} chunks; the index holds "
                      f"{stats.get('documents')} / {stats.get('chunks'):,}", fix)

        want_thr = cfg["threshold"] if cfg["calibrated"] else 0.0
        if entry.get("threshold") is not None and abs(entry["threshold"] - want_thr) > 1e-9:
            rep.stale(scope, name,
                      f"plots the threshold at {entry['threshold']:+.1f}; "
                      f"corpora.json ships {want_thr:+.1f}", fix)

        # The questions themselves, verbatim. Everything above is a count; this
        # is the thing the visitor actually clicks.
        gone = [e["q"] for e in entry.get("examples", []) if e["q"] not in by_q]
        if gone:
            rep.stale(scope, name,
                      f"offers {len(gone)} question(s) the golden set no longer "
                      f"contains, first: {gone[0][:70]!r}", fix)
        mislabelled = [e["q"] for e in entry.get("examples", [])
                       if e["q"] in by_q
                       and bool(e.get("adversarial")) != bool(by_q[e["q"]].get("unanswerable"))]
        if mislabelled:
            rep.stale(scope, name,
                      f"{len(mislabelled)} offered question(s) changed between "
                      f"answerable and adversarial, first: "
                      f"{mislabelled[0][:70]!r}", fix)
        if not entry.get("examples"):
            rep.stale(scope, name, "offers no questions at all", fix)

        if not [r for r in rep.rows if r[0] == scope and r[1] == name
                and r[2] in ("STALE", "MISSING")]:
            rep.ok(scope, name,
                   f"{len(entry.get('examples', []))} questions offered, all in "
                   f"the golden set")


def run() -> Report:
    rep = Report()
    reg = corpora.registry()
    golds: dict[str, dict] = {}
    per_cases: dict[str, dict] = {}

    for name, cfg in reg.items():
        if not cfg["indexed"]:
            rep.note(name, "index", "not indexed; nothing downstream to check")
            continue
        gold_path = cfg["golden"]
        gold = load(gold_path) if gold_path else None
        if not gold or "cases" not in gold:
            rep.missing(name, "golden",
                        f"{gold_path} is missing or has no cases -- this corpus "
                        f"cannot be measured at all")
            continue
        golds[name] = gold
        rep.ok(name, "golden",
               f"{len(gold['cases'])} cases, "
               f"{gold_path.relative_to(ROOT).as_posix()}")
        pc = check_per_case(rep, name, cfg, gold, gold_path)
        if pc:
            per_cases[name] = pc
        check_results(rep, name, cfg, gold, gold_path)

    check_analytics(rep, reg, golds, per_cases)
    return rep


STATE_MARK = {"ok": "  ok  ", "STALE": " STALE", "MISSING": " GONE ", "note": " note "}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quiet", action="store_true",
                    help="print only what is wrong")
    args = ap.parse_args()

    rep = run()
    scopes: list[str] = []
    for scope, *_ in rep.rows:
        if scope not in scopes:
            scopes.append(scope)

    for scope in scopes:
        rows = [r for r in rep.rows if r[0] == scope]
        if args.quiet and not any(r[2] in ("STALE", "MISSING") for r in rows):
            continue
        print(f"\n{scope}")
        for _, item, state, detail in rows:
            if args.quiet and state not in ("STALE", "MISSING"):
                continue
            print(f"  {STATE_MARK[state]}  {item:<12} {detail}")

    problems = rep.problems
    print()
    if not problems:
        print("the chain is current: golden set -> per_case -> analytics -> "
              "the questions the front page offers")
        return 0

    print(f"{len(problems)} problem(s). Rebuild in this order -- analytics reads "
          f"the others, so\nrebuilding it first only copies stale numbers forward:")
    for cmd in rep.fixes:
        print(f"    {cmd}")
    if ".venv\\Scripts\\python.exe src\\build_analytics.py" not in rep.fixes:
        print("    .venv\\Scripts\\python.exe src\\build_analytics.py")
    return 2 if rep.broken else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
