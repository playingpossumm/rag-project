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
                         |                    |
                         v                    v
                  threshold.json      static-demo/ (manifest, recordings)
    golden set + index  ->  top-passages.json

The three branches were added on 2026-09-06. Until then `eval/threshold.json`
was served at `/api/threshold` while naming a case the golden set had deleted
and counting 66 + 18 cases against a set that held 67 + 17; `top-passages.json`
disagreed with the golden sets on 4 of 157 entries; and the static demo's
manifest carried no record of which golden set or index it was recorded from.
None of the three carried a digest, so a file written before that date is
reported UNSTAMPED, which is a problem and not a note: the file cannot be
checked by provenance, and the way to fix that is to rewrite it.

There are 3 kinds of check, and each fails in a different way.

**Provenance.** Each generated file records a digest of the files it was built
from. A digest that no longer matches means the input changed after the output
was written -- exact, and it catches an edit that changes no count at all, such
as rewording one question in place.

**Semantics.** Case ids, question strings, corpus sizes and the per-corpus
threshold and rerank blend are compared directly. This is the weaker check and
the more useful one: it works on files written before provenance existed, it
says which question is missing rather than that a hash moved, and it is what
catches the interface offering a question the corpus no longer has.

**Stamps.** A generated file with no `inputs` block at all. For `per_case` and
`results` this is a note, because both predate provenance and every copy in
the repository has since been rewritten with one. For the three files added on
2026-09-06 it is a problem, because the copies on disk are the unstamped ones
and the point of adding them to the chain is to have them rewritten.

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
ANSWER_QUALITY = EVAL / "answer-quality.json"
TOP_PASSAGES = EVAL / "top-passages.json"
STATIC = ROOT / "static-demo"

# The states that make the exit code nonzero. "note" is not one of them.
PROBLEM_STATES = ("STALE", "MISSING", "UNSTAMPED")

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


def count(value) -> str:
    """A count for a message, or "none" when the file did not carry one, so a
    partial manifest gets a STALE row rather than a TypeError from the
    thousands separator. Until 2026-09-07 every site formatted the raw value."""
    return f"{value:,}" if isinstance(value, int) else "none"


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

    def unstamped(self, scope, item, detail, fix=None):
        """A file with no provenance block, where that is a problem.

        This is kept apart from `stale` because the finding is different even
        though the fix is the same. Nothing has been shown to disagree, and
        nothing can be shown to disagree until a generator that records its
        inputs rewrites the file.
        """
        self.rows.append((scope, item, "UNSTAMPED", detail))
        if fix and fix not in self.fixes:
            self.fixes.append(fix)

    def note(self, scope, item, detail):
        """Something to know about, not something that is wrong."""
        self.rows.append((scope, item, "note", detail))

    @property
    def problems(self):
        return [r for r in self.rows if r[2] in PROBLEM_STATES]

    def clean(self, scope, item) -> bool:
        """Whether nothing has been reported wrong under (scope, item) yet."""
        return not [r for r in self.rows if r[0] == scope and r[1] == item
                    and r[2] in PROBLEM_STATES]

    def commands(self) -> list[str]:
        """The rebuild commands, in the order the files have to be rebuilt.

        The fixes arrive in check order, which is per-corpus files first, then
        analytics, then top-passages, then the static demo. build_analytics.py
        is always included because analytics.json carries the figures the
        others feed. When analytics was found current it is not in `fixes`,
        and until 2026-09-07 it was then appended last, after record_static.py,
        although record_static.py copies eval/analytics.json into static-demo/
        on every run; following that printed order would have left the copy
        behind. It now goes in before the first record_static.py command.
        """
        analytics = ".venv\\Scripts\\python.exe src\\build_analytics.py"
        cmds = list(self.fixes)
        if analytics not in cmds:
            at = next((i for i, c in enumerate(cmds) if "record_static.py" in c),
                      len(cmds))
            cmds.insert(at, analytics)
        return cmds


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
                  f"{count(got_c.get('chunks'))} chunks; the index now holds "
                  f"{stats.get('documents')} / {count(stats.get('chunks'))}", fix)

    # Config drift. The single most reused finding in this project is that a
    # corpus setting does not transfer, so a file that recorded one corpus's
    # threshold while describing another's questions is measuring a pipeline
    # nobody serves.
    want_thr = cfg["threshold"] if cfg["calibrated"] else 0.0
    if pc.get("threshold") is not None and abs(pc["threshold"] - want_thr) > 1e-9:
        rep.stale(name, "per_case",
                  f"scored at threshold {pc['threshold']:+.1f}; corpora.json "
                  f"ships {want_thr:+.1f} for this corpus", fix)
    got_ens = pc.get("options", {}).get("ensemble")
    want_ens = cfg.get("ensemble_model")
    if "ensemble" in pc.get("options", {}) and got_ens != want_ens:
        rep.stale(name, "per_case",
                  f"scored with second retriever {got_ens!r}; corpora.json "
                  f"ships {want_ens!r}", fix)
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

    if rep.clean(name, "per_case"):
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
                  f"{count(c.get('chunks'))} chunks; the index now holds "
                  f"{stats.get('documents')} / {count(stats.get('chunks'))}", fix)

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

    if rep.clean(name, "results"):
        rep.ok(name, "results", f"{n_ans} + {n_adv} cases, {path.name}")
    return res


def check_threshold(rep: Report, name: str, cfg: dict, gold: dict, gold_path: Path):
    """threshold*.json -- the calibration table serve.py offers at /api/threshold.

    Optional, so a corpus without one gets no row. It reads per_case, so its
    provenance is the per_case digest plus whatever per_case recorded about
    the golden set and the index. The live instance this was written against
    was eval/threshold.json, calibrated on 2026-08-21 on 66 + 18 cases, which
    on 2026-09-06 still named adv-moe-routing although the golden set had
    dropped that case on 2026-08-25 (commit 818eded). The first draft of this
    sentence put the gap at 16 days, counted from the calibration date rather
    than from the git history; the dates above replace that figure.
    """
    path = artefact("threshold", gold_path)
    if not path.exists():
        return None
    per_case = artefact("per_case", gold_path)
    fix = (f".venv\\Scripts\\python.exe src\\calibrate_threshold.py "
           f"--per-case {per_case.relative_to(ROOT).as_posix()} "
           f"--emit {path.relative_to(ROOT).as_posix()}")
    data = load(path)
    if data is None:
        rep.missing(name, "threshold", f"{path.name} is unreadable", fix)
        return None
    rel = path.relative_to(ROOT).as_posix()

    inputs = data.get("inputs")
    if not inputs:
        rep.unstamped(name, "threshold",
                      f"{rel} records no inputs, so nothing can say which "
                      f"per_case or golden set it was calibrated from", fix)
    else:
        got = (inputs.get("golden") or {}).get("digest")
        if got and got != digest(gold_path):
            rep.stale(name, "threshold",
                      f"golden set has changed since {rel} was calibrated "
                      f"({got} -> {digest(gold_path)})", fix)
        got_p = (inputs.get("per_case") or {}).get("digest")
        if got_p and got_p != digest(per_case):
            rep.stale(name, "threshold",
                      f"{per_case.name} has been re-scored since {rel} was "
                      f"calibrated", fix)
        got_i = (inputs.get("index") or {}).get("digest")
        if got_i and got_i != digest(cfg["store"] / "metadata.json"):
            rep.stale(name, "threshold",
                      f"the index has been rebuilt since {rel} was calibrated", fix)

    n_ans = sum(1 for x in gold["cases"] if not x.get("unanswerable"))
    n_adv = len(gold["cases"]) - n_ans
    if (data.get("n_answerable"), data.get("n_adversarial")) != (n_ans, n_adv):
        rep.stale(name, "threshold",
                  f"calibrated on {data.get('n_answerable')} answerable + "
                  f"{data.get('n_adversarial')} adversarial; the golden set now "
                  f"holds {n_ans} + {n_adv}", fix)

    gold_ids = {c["id"] for c in gold["cases"]}
    named = set()
    for row in data.get("sweep") or []:
        named |= set(row.get("caught") or []) | set(row.get("wrongly_refused") or [])
    gone = sorted(named - gold_ids)
    if gone:
        rep.stale(name, "threshold",
                  f"names {len(gone)} case(s) the golden set no longer has: "
                  f"{', '.join(gone[:4])}{' ...' if len(gone) > 4 else ''}", fix)

    want_thr = cfg["threshold"] if cfg["calibrated"] else 0.0
    got_thr = data.get("shipped_threshold")
    if got_thr is not None and abs(got_thr - want_thr) > 1e-9:
        rep.stale(name, "threshold",
                  f"records a shipped threshold of {got_thr:+.1f}; corpora.json "
                  f"ships {want_thr:+.1f}", fix)

    if rep.clean(name, "threshold"):
        rep.ok(name, "threshold", f"{n_ans} + {n_adv} cases, {path.name}")
    return data


def check_top_passages(rep: Report, reg: dict, golds: dict):
    """top-passages.json -- the passages the answer-highlight sweep runs over.

    One file for every corpus, written by dump_top_passages.py and read by
    ui/test-answer-mark.mjs. Before 2026-09-06 it was a bare list, so a file
    of that shape is UNSTAMPED and its rows are still checked by content.
    """
    scope = "top-passages"
    fix = ".venv\\Scripts\\python.exe src\\dump_top_passages.py"
    if not TOP_PASSAGES.exists():
        return
    data = load(TOP_PASSAGES)
    if data is None:
        rep.missing(scope, "provenance", "eval/top-passages.json is unreadable", fix)
        return

    if isinstance(data, list):
        rows, inputs = data, {}
        rep.unstamped(scope, "provenance",
                      "eval/top-passages.json is a bare list with no inputs, so "
                      "nothing can say which golden sets or indexes its passages "
                      "came from", fix)
    else:
        rows = data.get("rows") or []
        inputs = (data.get("inputs") or {}).get("per_corpus") or {}
        if not data.get("inputs"):
            rep.unstamped(scope, "provenance",
                          "eval/top-passages.json records no inputs", fix)

    for name in sorted(inputs):
        cfg = reg.get(name)
        if not cfg:
            continue
        got = (inputs[name].get("golden") or {}).get("digest")
        if got and cfg["golden"] and got != digest(cfg["golden"]):
            rep.stale(scope, name,
                      f"golden set has changed since the passages were dumped "
                      f"({got} -> {digest(cfg['golden'])})", fix)
        got_i = (inputs[name].get("index") or {}).get("digest")
        if got_i and got_i != digest(cfg["store"] / "metadata.json"):
            rep.stale(scope, name,
                      "the index has been rebuilt since the passages were dumped", fix)

    by_corpus: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_corpus.setdefault(r.get("corpus"), {})[r.get("id")] = r
    for name in sorted(set(by_corpus) - set(golds), key=str):
        rep.stale(scope, name,
                  f"holds passages for {name!r}, which has no golden set", fix)
    for name, gold in sorted(golds.items()):
        gold_ids = {c["id"]: c for c in gold["cases"]}
        got_ids = by_corpus.get(name, {})
        if not got_ids:
            rep.stale(scope, name, "no passages dumped for this corpus", fix)
            continue
        gone = sorted(set(got_ids) - set(gold_ids))
        if gone:
            rep.stale(scope, name,
                      f"holds {len(gone)} case(s) the golden set no longer has: "
                      f"{', '.join(gone[:4])}{' ...' if len(gone) > 4 else ''}", fix)
        never = sorted(set(gold_ids) - set(got_ids))
        if never:
            rep.stale(scope, name,
                      f"{len(never)} golden case(s) have never been dumped: "
                      f"{', '.join(never[:4])}{' ...' if len(never) > 4 else ''}", fix)
        shared = set(gold_ids) & set(got_ids)
        reworded = sorted(i for i in shared
                          if gold_ids[i]["question"] != got_ids[i].get("question"))
        if reworded:
            rep.stale(scope, name,
                      f"{len(reworded)} question(s) reworded since the dump: "
                      f"{', '.join(reworded[:4])}", fix)
        flipped = sorted(i for i in shared
                         if bool(gold_ids[i].get("unanswerable"))
                         != bool(got_ids[i].get("unanswerable")))
        if flipped:
            rep.stale(scope, name,
                      f"{len(flipped)} case(s) changed between answerable and "
                      f"adversarial since the dump: {', '.join(flipped[:4])}", fix)
        if rep.clean(scope, name):
            rep.ok(scope, name, f"{len(got_ids)} passages, all in the golden set")


def check_static(rep: Report, reg: dict, golds: dict):
    """static-demo/ -- the recorded answers a visitor to the static build gets.

    The manifest names each corpus with the golden set and index it was
    recorded from; each corpus directory holds an index.json naming the
    recorded questions, which are compared to the golden set the way the
    front page's offered questions are. analytics.json there is a byte copy of
    eval/analytics.json, so those two are compared by digest, and so are the
    manifest and the copy of it under site/recorded/. The manifest also
    records the digest of the analytics.json the recording was taken from,
    because the questions a plain (not --all) recording covers are the ones
    analytics.json offers, so an analytics.json rebuilt since then may offer
    questions the recording does not hold. Until 2026-09-07 that digest was
    written and never read.
    """
    scope = "static-demo"
    manifest_path = STATIC / "manifest.json"
    if not manifest_path.exists():
        return
    man = load(manifest_path)
    if man is None:
        rep.missing(scope, "manifest", "static-demo/manifest.json is unreadable",
                    ".venv\\Scripts\\python.exe src\\record_static.py")
        return
    entries = man.get("corpora") or {}

    # Whether the last recording covered whole golden sets or the offered
    # ten, so the fix names the flag that reproduces it.
    whole = all(golds.get(n) and e.get("questions") == len(golds[n]["cases"])
                for n, e in entries.items())
    fix = (".venv\\Scripts\\python.exe src\\record_static.py"
           + (" --all" if whole else ""))
    site_fix = ".venv\\Scripts\\python.exe src\\record_static.py --site-only"

    copy = STATIC / "analytics.json"
    if copy.exists() and ANALYTICS.exists() and digest(copy) != digest(ANALYTICS):
        rep.stale(scope, "analytics",
                  "static-demo/analytics.json is behind eval/analytics.json, so "
                  "the static build's quality page draws the previous figures",
                  site_fix)
    site_copy = STATIC / "site" / "recorded" / "manifest.json"
    if site_copy.exists() and digest(site_copy) != digest(manifest_path):
        rep.stale(scope, "site",
                  "site/recorded/ was built from an older manifest than the one "
                  "in static-demo/", site_fix)
    recorded_from = ((man.get("inputs") or {}).get("analytics") or {}).get("digest")
    if recorded_from and ANALYTICS.exists() and recorded_from != digest(ANALYTICS):
        rep.stale(scope, "manifest",
                  "the recording was taken from an older eval/analytics.json "
                  f"({recorded_from} -> {digest(ANALYTICS)}), so the questions "
                  "it offers may not all have recorded answers", fix)

    for name in sorted(set(entries) - set(reg)):
        rep.stale(scope, name,
                  f"records {name!r}, which corpora.json no longer configures", fix)
    for name in sorted(n for n, c in reg.items()
                       if c["indexed"] and c["golden"] and n not in entries):
        rep.note(scope, name, f"{reg[name]['label']} has not been recorded")

    for name, entry in sorted(entries.items()):
        cfg = reg.get(name)
        if not cfg:
            continue
        inputs = entry.get("inputs")
        if not inputs:
            rep.unstamped(scope, name,
                          f"manifest.json records no inputs for {name}, so nothing "
                          f"can say which golden set or index its "
                          f"{entry.get('questions')} recorded answers came from", fix)
        else:
            got = (inputs.get("golden") or {}).get("digest")
            if got and cfg["golden"] and got != digest(cfg["golden"]):
                rep.stale(scope, name,
                          f"golden set has changed since {name} was recorded "
                          f"({got} -> {digest(cfg['golden'])})", fix)
            got_i = (inputs.get("index") or {}).get("digest")
            if got_i and got_i != digest(cfg["store"] / "metadata.json"):
                rep.stale(scope, name,
                          f"the index has been rebuilt since {name} was recorded", fix)

        stats = corpora.stats(cfg)
        if stats and (entry.get("chunks") != stats.get("chunks")
                      or entry.get("documents") != stats.get("documents")):
            rep.stale(scope, name,
                      f"recorded against {entry.get('documents')} documents / "
                      f"{count(entry.get('chunks'))} chunks; the index now holds "
                      f"{stats.get('documents')} / {count(stats.get('chunks'))}", fix)
        want_thr = cfg["threshold"] if cfg["calibrated"] else 0.0
        if entry.get("threshold") is not None and abs(entry["threshold"] - want_thr) > 1e-9:
            rep.stale(scope, name,
                      f"recorded at threshold {entry['threshold']:+.1f}; "
                      f"corpora.json ships {want_thr:+.1f}", fix)

        gold = golds.get(name)
        if not gold:
            continue
        by_q = {c["question"]: c for c in gold["cases"]}
        recorded = load(STATIC / name / "index.json") or []
        gone = [r["question"] for r in recorded if r.get("question") not in by_q]
        if gone:
            rep.stale(scope, name,
                      f"answers {len(gone)} question(s) the golden set no longer "
                      f"contains, first: {gone[0][:70]!r}", fix)
        mislabelled = [r["question"] for r in recorded
                       if r.get("question") in by_q
                       and bool(r.get("adversarial"))
                       != bool(by_q[r["question"]].get("unanswerable"))]
        if mislabelled:
            rep.stale(scope, name,
                      f"{len(mislabelled)} recorded question(s) changed between "
                      f"answerable and adversarial, first: "
                      f"{mislabelled[0][:70]!r}", fix)
        if rep.clean(scope, name):
            rep.ok(scope, name,
                   f"{len(recorded)} questions recorded, all in the golden set")


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

    # analytics.json carries the answer-quality figures the quality page draws,
    # so re-scoring answers without rebuilding analytics leaves that panel
    # showing the previous run. Sampled measurements can move every figure
    # without moving a count, so this compares the digest.
    if ANSWER_QUALITY.exists():
        recorded = (data.get("inputs") or {}).get("answers")
        current = stamp(ANSWER_QUALITY)["digest"]
        if recorded is None:
            rep.note(scope, "answers",
                     "analytics.json predates the answer-quality stamp; "
                     "rebuild it to cover the quality page's answer panel")
        elif recorded != current:
            rep.stale(scope, "answers",
                      "eval/answer-quality.json has changed since analytics.json "
                      "was built, so /quality still shows the previous answer "
                      "figures", fix)
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
                      f"{count(entry.get('chunks'))} chunks; the index holds "
                      f"{stats.get('documents')} / {count(stats.get('chunks'))}", fix)

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

        if rep.clean(scope, name):
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
        check_threshold(rep, name, cfg, gold, gold_path)

    # analytics before the static demo, which copies from it, so the fix list
    # comes out in the order the files have to be rebuilt. Report.commands()
    # keeps that order when analytics itself is current.
    check_analytics(rep, reg, golds, per_cases)
    check_top_passages(rep, reg, golds)
    check_static(rep, reg, golds)
    return rep


STATE_MARK = {"ok": "    ok   ", "STALE": "  STALE  ", "MISSING": "  GONE   ",
              "UNSTAMPED": "UNSTAMPED", "note": "  note   "}


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
        if args.quiet and not any(r[2] in PROBLEM_STATES for r in rows):
            continue
        print(f"\n{scope}")
        for _, item, state, detail in rows:
            if args.quiet and state not in PROBLEM_STATES:
                continue
            print(f"  {STATE_MARK[state]}  {item:<12} {detail}")

    problems = rep.problems
    print()
    if not problems:
        print("the chain is current: golden set -> per_case -> analytics -> "
              "the questions the front page offers, threshold, top-passages "
              "and the static demo")
        return 0

    n_unstamped = sum(1 for r in problems if r[2] == "UNSTAMPED")
    if n_unstamped:
        print(f"{n_unstamped} UNSTAMPED finding(s): a file with no inputs digest "
              f"cannot be checked by provenance until it is rewritten.")
    print(f"{len(problems)} problem(s). Rebuild in this order -- analytics reads "
          f"the others, so\nrebuilding it first only copies stale numbers forward:")
    for cmd in rep.commands():
        print(f"    {cmd}")
    return 2 if rep.broken else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
