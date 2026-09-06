"""Does the freshness check actually catch the ways this chain has gone stale?

A check that never fails is indistinguishable from no check, and this project
has already shipped one: `pipeline_trace.py`'s docstring claimed a test asserting
it matched the serving path, and no such test existed. So each way the chain
`golden set -> per_case -> analytics -> the front page` is known to have broken
is reproduced here on a synthetic corpus, and the assertion is that
`check_freshness` reports it -- naming the case, not just returning nonzero.

6 of these are not hypotheses. They happened:

- `per_case.json` sat four days out of date while the golden set moved on, and
  the front page went on offering questions that had been deleted for being
  unanswerable. That is `offers a deleted question`.
- Every corpus wrote one shared `results.json`, so the last run owned it and a
  bird run left an ML document describing 45 Wikipedia pages. That is
  `results run against another corpus's golden set`.
- `per_case.py` scored every corpus at the ML papers' 0.0 threshold and at
  rerank blend 0.0 while two corpora ship neither. That is `threshold drift` and
  `blend drift`, and it was live in the repo until 2026-08-26.
- `threshold.json`, served at /api/threshold, was calibrated on 2026-08-21 on
  66 + 18 cases and still named `adv-moe-routing` on 2026-09-06, although the
  golden set had dropped that case on 2026-08-25. That is `names a deleted
  case` and `calibrated on a different case count`. The first draft of this
  line said the case had been gone for 16 days, a figure counted from the
  calibration date rather than taken from the git history.
- `top-passages.json` disagreed with the golden sets on 4 of 157 rows on
  2026-09-06: one question reworded, three cases moved between answerable and
  adversarial. That is `reworded since the dump` and `flipped since the dump`.
- Neither of those two carried an inputs digest, and nor did the static demo's
  manifest, so the disagreements could only be found by reading. That is `UNSTAMPED`.

Hermetic: everything runs against files written into a temporary directory, so
this passes or fails on the code rather than on the state of the real eval
artefacts. No corpus, no models, milliseconds.

    .venv\\Scripts\\python.exe src\\test_freshness.py
"""
import json
import shutil
import tempfile
from pathlib import Path

import check_freshness as cf
from testkit import Suite

suite = Suite("freshness")
check = suite.check


# --------------------------------------------------------------- fixture ----
# One corpus, three questions, two documents. Small enough to read, and every
# quantity the checker compares is present exactly once.
GOLD = {
    "cases": [
        {"id": "q1", "question": "How many primary feathers?", "kind": "fact",
         "gold": [{"source": "a.pdf", "pages": [1]}]},
        {"id": "q2", "question": "What is a syrinx?", "kind": "fact",
         "gold": [{"source": "b.pdf", "pages": [2]}]},
        {"id": "adv1", "question": "What is the airspeed of a swallow?",
         "unanswerable": True},
    ]
}


def write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8")


def scenario(gold=None, pc=None, res=None, an=None, thr=None, tp=None, st=None,
             after=None):
    """Build a whole chain in a temp directory and run the checker over it.

    `pc`, `res`, `an`, `thr`, `tp` and `st` mutate one artefact as it is
    written, which is how a file that was generated wrong gets staged. `gold`
    and `after` run once the whole chain exists, which is how a file that was
    generated right and then overtaken gets staged -- and that ordering is the
    whole point. Mutating the golden set first would produce a chain that
    agrees with itself perfectly, which is a test of nothing: staleness is a
    relationship between files, not a property of one.
    """
    tmp = Path(tempfile.mkdtemp(prefix="freshness-"))
    try:
        (tmp / "eval").mkdir()
        store = tmp / "store-t"
        store.mkdir()

        # A real metadata.json, so corpora.stats() does its real work rather
        # than being stubbed into agreeing with the file it is checking.
        write(store / "metadata.json",
              [{"source": s, "text": f"chunk {i}"}
               for i, s in enumerate(["a.pdf", "a.pdf", "a.pdf", "b.pdf", "b.pdf", "b.pdf"])])

        cf.ROOT, cf.EVAL, cf.ANALYTICS = tmp, tmp / "eval", tmp / "eval" / "analytics.json"
        cf.TOP_PASSAGES = tmp / "eval" / "top-passages.json"
        cf.STATIC = tmp / "static-demo"

        cfg = {"name": "t", "label": "Test corpus", "data": tmp / "data-t",
               "store": store, "golden": tmp / "eval" / "golden-t.json",
               "threshold": -3.0, "calibrated": True, "rerank_blend": 0.2,
               "candidate_k": 20,
               "indexed": True}

        g = json.loads(json.dumps(GOLD))
        write(cfg["golden"], g)

        n_ans = sum(1 for c in g["cases"] if not c.get("unanswerable"))
        n_adv = len(g["cases"]) - n_ans
        stamps = lambda: {"golden": cf.stamp(cfg["golden"], cases=len(g["cases"])),  # noqa: E731
                          "index": cf.stamp(store / "metadata.json", chunks=6)}

        per = {
            "generated_by": "src/per_case.py",
            "options": {"k": 5, "candidate_k": 20, "rerank_blend": 0.2,
                        "use_reranker": True, "fusion": "rrf", "max_per_source": 2},
            "threshold": -3.0,
            "corpus": {"name": "t", "chunks": 6, "documents": 2},
            "inputs": stamps(),
            "cases": [{"id": c["id"], "question": c["question"],
                       "unanswerable": bool(c.get("unanswerable")),
                       "confidence": -1.0, "gold_sources": [],
                       "outcome": "found"} for c in g["cases"]],
        }
        if pc:
            pc(per)
        per_path = cfg["golden"].parent / "per_case-t.json"
        write(per_path, per)

        results = {"corpus": {"chunks": 6, "documents": 2,
                              "answerable_cases": n_ans, "adversarial_cases": n_adv,
                              "k": 5, "candidate_k": 20,
                              "golden": str(cfg["golden"])},
                   "abstention": {"shipped_threshold": -3.0},
                   "inputs": stamps()}
        if res:
            res(results)
        write(cfg["golden"].parent / "results-t.json", results)

        # The calibration table, as calibrate_threshold.py writes it: the
        # per_case digest taken directly, the golden and index digests
        # carried from per_case's own record.
        threshold = {
            "generated_by": "src/calibrate_threshold.py", "corpus": "t",
            "shipped_threshold": -3.0,
            "n_answerable": n_ans, "n_adversarial": n_adv,
            "inputs": {"per_case": cf.stamp(per_path), **(per.get("inputs") or {})},
            "sweep": [{"threshold": -4.0, "caught": [], "wrongly_refused": []},
                      {"threshold": -3.0, "caught": ["adv1"], "wrongly_refused": []},
                      {"threshold": 0.0, "caught": ["adv1"], "wrongly_refused": ["q2"]}],
        }
        if thr:
            thr(threshold)
        write(cfg["golden"].parent / "threshold-t.json", threshold)

        analytics = {
            "generated_by": "src/build_analytics.py",
            "inputs": {"per_corpus": {"t": {
                "golden": cf.stamp(cfg["golden"]),
                "per_case": cf.stamp(cf.artefact("per_case", cfg["golden"])),
                "results": cf.stamp(cf.artefact("results", cfg["golden"])),
            }}},
            "corpora": [{
                "name": "t", "label": "Test corpus", "documents": 2, "chunks": 6,
                "threshold": -3.0, "calibrated": True,
                "n_answerable": n_ans, "n_adversarial": n_adv,
                "examples": [{"q": c["question"], "adversarial": bool(c.get("unanswerable")),
                              "label": "answered in one place"} for c in g["cases"]],
            }],
        }
        if an:
            an(analytics)
        write(cf.ANALYTICS, analytics)

        top = {
            "generated_by": "src/dump_top_passages.py", "expansion": "window",
            "inputs": {"per_corpus": {"t": stamps()}},
            "rows": [{"corpus": "t", "id": c["id"], "question": c["question"],
                      "unanswerable": bool(c.get("unanswerable")),
                      "text": "a passage", "also": []} for c in g["cases"]],
        }
        if tp:
            tp(top)
        write(cf.TOP_PASSAGES, top)

        # The static demo: the manifest, one corpus directory with its
        # index.json, a byte copy of analytics.json, and the site's copy of
        # the manifest.
        (cf.STATIC / "t").mkdir(parents=True)
        (cf.STATIC / "site" / "recorded").mkdir(parents=True)
        static = {
            "manifest": {
                "generated_by": "src/record_static.py", "recorded": "2026-09-06",
                "generation": None,
                "corpora": {"t": {"label": "Test corpus", "questions": len(g["cases"]),
                                  "threshold": -3.0, "documents": 2, "chunks": 6,
                                  "inputs": stamps()}},
                "default": "t",
                "inputs": {"analytics": cf.stamp(cf.ANALYTICS)},
            },
            "index": [{"key": f"k{i}", "question": c["question"], "id": c["id"],
                       "adversarial": bool(c.get("unanswerable")), "confident": True}
                      for i, c in enumerate(g["cases"])],
        }
        if st:
            st(static)
        write(cf.STATIC / "manifest.json", static["manifest"])
        write(cf.STATIC / "t" / "index.json", static["index"])
        shutil.copy2(cf.ANALYTICS, cf.STATIC / "analytics.json")
        shutil.copy2(cf.STATIC / "manifest.json",
                     cf.STATIC / "site" / "recorded" / "manifest.json")

        # The golden set moves on, and nothing downstream is rebuilt. Written
        # last so every digest and every count above describes the chain as it
        # was before the edit.
        if gold:
            gold(g)
            write(cfg["golden"], g)
        if after:
            after(tmp, cfg)

        rep = cf.Report()
        gold_now = json.loads(cfg["golden"].read_text(encoding="utf-8"))
        per_now = cf.check_per_case(rep, "t", cfg, gold_now, cfg["golden"])
        cf.check_results(rep, "t", cfg, gold_now, cfg["golden"])
        cf.check_threshold(rep, "t", cfg, gold_now, cfg["golden"])
        cf.check_analytics(rep, {"t": cfg}, {"t": gold_now},
                           {"t": per_now} if per_now else {})
        cf.check_top_passages(rep, {"t": cfg}, {"t": gold_now})
        cf.check_static(rep, {"t": cfg}, {"t": gold_now})
        return rep
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def problems(rep) -> str:
    """Every complaint as one lowercase string, for substring assertions."""
    return " || ".join(f"{s}/{i}: {d}" for _, i, s, d in rep.rows
                       if s in cf.PROBLEM_STATES).lower()


# ------------------------------------------------------------- the checks ----

# A chain nobody has touched must be silent. Stated first because every
# assertion below is only meaningful if this one holds -- a checker that
# complains about everything catches every regression and is still useless.
check("a current chain reports no problems", problems(scenario()), "")

# ---- provenance: an input moved after the output was written ----------------
# The digest is the only check that catches this. Rewording a question in place
# changes no count, no id and no total; every semantic comparison below still
# passes, and every number in per_case.json is now measured on a question that
# is not being asked.
def reword_after(tmp, cfg):
    g = json.loads(cfg["golden"].read_text(encoding="utf-8"))
    g["cases"][0]["question"] = "How many primary feathers does a swift have?"
    write(cfg["golden"], g)


rep = scenario(after=reword_after)
check("provenance / golden set edited after per_case was written",
      "golden set has changed since eval/per_case-t.json" in problems(rep), True)
# The same edit must also invalidate the harness run. results.json carried no
# provenance until 2026-08-27, so a corrected label that changed no case count,
# no corpus size and no metric left it reported as current -- which is exactly
# the edit the digest half exists for.
check("provenance / and the harness run it invalidates too",
      "golden set has changed since eval/results-t.json" in problems(rep), True)
check("provenance / and the reworded question is named",
      "reworded" in problems(rep), True)
# And the three files added to the chain on 2026-09-06, each of which rests
# on the golden set and none of which recorded that before.
check("provenance / and the calibration table",
      "golden set has changed since eval/threshold-t.json" in problems(rep), True)
check("provenance / and the dumped passages",
      "golden set has changed since the passages were dumped" in problems(rep), True)
check("provenance / and the static demo's recording",
      "golden set has changed since t was recorded" in problems(rep), True)


# per_case re-run, analytics not rebuilt. The counts all still agree -- this is
# exactly the shape of the failure that left the front page four days behind.
def rerun_per_case(tmp, cfg):
    p = cf.artefact("per_case", cfg["golden"])
    d = json.loads(p.read_text(encoding="utf-8"))
    d["cases"][0]["confidence"] = 7.5      # a re-run, same cases, new numbers
    write(p, d)


rep = scenario(after=rerun_per_case)
check("provenance / analytics built from an older per_case",
      "built from an older per_case" in problems(rep), True)
check("provenance / per_case itself is still current",
      any(i == "per_case" and s == "ok" for _, i, s, _ in rep.rows), True)
# The calibration table reads per_case, so a re-score with the same cases and
# new confidences moves every row of the sweep while moving no count.
check("provenance / threshold calibrated from an older per_case",
      "per_case-t.json has been re-scored since" in problems(rep), True)


# The index rebuilt under a scored file. Chunk counts happen to be unchanged
# here, so only the digest can see it -- an ingest that re-chunks the same
# documents moves every offset while moving no total.
def reingest(tmp, cfg):
    write(cfg["store"] / "metadata.json",
          [{"source": s, "text": f"re-chunked {i}"}
           for i, s in enumerate(["a.pdf", "a.pdf", "a.pdf", "b.pdf", "b.pdf", "b.pdf"])])


rep = scenario(after=reingest)
check("provenance / the index was rebuilt after scoring",
      "index has been rebuilt since eval/per_case-t.json" in problems(rep), True)
check("provenance / and after the harness measured it",
      "index has been rebuilt since eval/results-t.json" in problems(rep), True)
check("provenance / and after the passages were dumped",
      "index has been rebuilt since the passages were dumped" in problems(rep), True)
check("provenance / and after the static demo was recorded",
      "index has been rebuilt since t was recorded" in problems(rep), True)

# ---- semantics: these work on files written before provenance existed -------
check("semantics / a file with no provenance block is still checked",
      problems(scenario(pc=lambda d: d.pop("inputs"))), "")
check("semantics / and says it was checked by content only",
      any("predates provenance" in d for *_, d in scenario(pc=lambda d: d.pop("inputs")).rows),
      True)


def drop_from_gold(d):
    d["cases"] = [c for c in d["cases"] if c["id"] != "q2"]


rep = scenario(gold=drop_from_gold, pc=lambda d: d.pop("inputs"),
               an=lambda d: d.pop("inputs"))
check("semantics / per_case scores a case the golden set no longer has",
      "no longer has: q2" in problems(rep), True)
check("semantics / and the deleted question is still on the front page",
      "the golden set no longer contains" in problems(rep), True)


def add_to_gold(d):
    d["cases"].append({"id": "q9", "question": "What is a pygostyle?", "kind": "fact",
                       "gold": [{"source": "b.pdf", "pages": [3]}]})


check("semantics / a new golden case has never been scored",
      "never been scored: q9" in problems(scenario(gold=add_to_gold)), True)


def flip(d):
    d["cases"][1]["unanswerable"] = True


check("semantics / a case moved between answerable and adversarial",
      "changed between answerable and adversarial" in problems(scenario(gold=flip)), True)

# ---- config drift: the trap this project keeps re-learning ------------------
# The live instance, reproduced: per_case scored at the ML papers' 0.0 while
# corpora.json ships -3.0 for this corpus. Ten bird questions were reported
# wrongly refused where the served configuration refuses four.
check("config / scored at another corpus's threshold",
      "corpora.json ships -3.0" in problems(scenario(pc=lambda d: d.update(threshold=0.0))),
      True)
check("config / scored over another pool size",
      "corpora.json ships 20 for this corpus" in problems(
          scenario(pc=lambda d: d["options"].update(candidate_k=16))), True)
check("config / scored at another corpus's rerank blend",
      "rerank blend 0.00" in problems(
          scenario(pc=lambda d: d["options"].update(rerank_blend=0.0))), True)
check("config / analytics plots a threshold corpora.json does not ship",
      "plots the threshold" in problems(
          scenario(an=lambda d: d["corpora"][0].update(threshold=0.0))), True)

# ---- the index grew under everything ---------------------------------------
check("corpus / per_case scored against a smaller index",
      "the index now holds" in problems(
          scenario(pc=lambda d: d["corpus"].update(chunks=3, documents=1))), True)
check("corpus / results measured a different corpus size",
      "the index now holds" in problems(
          scenario(res=lambda d: d["corpus"].update(chunks=3, documents=1))), True)
check("corpus / results run against another corpus's golden set",
      "was run against golden_set.json" in problems(
          scenario(res=lambda d: d["corpus"].update(golden="eval/golden_set.json"))), True)
check("corpus / results measured a different number of cases",
      "the golden set now holds" in problems(
          scenario(res=lambda d: d["corpus"].update(answerable_cases=9))), True)

# ---- the end of the chain: what the front page offers -----------------------
check("interface / a corpus with eval artefacts and no analytics entry",
      "no entry in analytics.json" in problems(
          scenario(an=lambda d: d.update(corpora=[], inputs={"per_corpus": {}}))), True)
check("interface / an offered question that is not in the golden set",
      "the golden set no longer contains" in problems(
          scenario(an=lambda d: d["corpora"][0]["examples"].insert(
              0, {"q": "Which bird has the longest migration?", "adversarial": False}))),
      True)
check("interface / an offered question labelled the wrong way round",
      "changed between answerable and adversarial" in problems(
          scenario(an=lambda d: d["corpora"][0]["examples"][0].update(adversarial=True))),
      True)
check("interface / a corpus offering no questions at all",
      "offers no questions at all" in problems(
          scenario(an=lambda d: d["corpora"][0].update(examples=[]))), True)
check("interface / analytics.json missing entirely",
      "missing or unreadable" in problems(
          scenario(after=lambda tmp, cfg: cf.ANALYTICS.unlink())), True)

# ---- the calibration table served at /api/threshold -------------------------
# The live instance was eval/threshold.json, which on 2026-09-06 had no inputs
# block, counted 66 + 18 against a golden set holding 67 + 17, and named
# adv-moe-routing, a case the golden set had dropped on 2026-08-25.
check("threshold / no inputs block is UNSTAMPED, and a problem",
      "unstamped/threshold" in problems(scenario(thr=lambda d: d.pop("inputs"))), True)
check("threshold / calibrated on a different case count",
      "the golden set now holds" in problems(
          scenario(thr=lambda d: d.update(n_answerable=9))), True)
check("threshold / names a case the golden set no longer has",
      "no longer has: adv-moe-routing" in problems(
          scenario(thr=lambda d: d["sweep"][1]["caught"].append("adv-moe-routing"))), True)
check("threshold / records a shipped threshold corpora.json does not ship",
      "records a shipped threshold of +0.0" in problems(
          scenario(thr=lambda d: d.update(shipped_threshold=0.0))), True)
check("threshold / a corpus without one gets no row",
      [r for r in scenario(after=lambda tmp, cfg: cf.artefact(
          "threshold", cfg["golden"]).unlink()).rows if r[1] == "threshold"], [])

# ---- the passages the answer-highlight sweep runs over ----------------------
# The live instance was a bare list of 157 rows, 4 of which disagreed with the
# golden sets because dropout-rate had been reworded and bird-alula,
# qf-mean-reversion and qf-momentum had moved between answerable and
# adversarial.
def bare_list(tmp, cfg):
    """The pre-2026-09-06 shape: the rows alone, no inputs, no generator."""
    d = json.loads(cf.TOP_PASSAGES.read_text(encoding="utf-8"))
    write(cf.TOP_PASSAGES, d["rows"])


check("top-passages / a bare list is UNSTAMPED, and a problem",
      "unstamped/provenance" in problems(scenario(after=bare_list)), True)
check("top-passages / and a bare list's rows are still checked by content",
      "reworded since the dump" in problems(scenario(
          after=lambda tmp, cfg: (bare_list(tmp, cfg), reword_after(tmp, cfg)))),
      True)
check("top-passages / a reworded question",
      "reworded since the dump: q1" in problems(scenario(after=reword_after)), True)
check("top-passages / a case flipped since the dump",
      "adversarial since the dump: q2" in problems(scenario(gold=flip)), True)
check("top-passages / a golden case never dumped",
      "never been dumped: q9" in problems(scenario(gold=add_to_gold)), True)
check("top-passages / a row the golden set no longer has",
      "no longer has: q2" in problems(scenario(gold=drop_from_gold)), True)
check("top-passages / a corpus with no rows at all",
      "no passages dumped" in problems(
          scenario(tp=lambda d: d.update(rows=[]))), True)

# ---- the static demo --------------------------------------------------------
# The live instance was static-demo/manifest.json, recorded on 2026-09-03,
# which named 157 recorded answers across 3 corpora and said nothing about
# where they came from.
check("static / a manifest entry with no inputs is UNSTAMPED, and a problem",
      "unstamped/t" in problems(
          scenario(st=lambda d: d["manifest"]["corpora"]["t"].pop("inputs"))), True)
check("static / a recorded question the golden set no longer has",
      "the golden set no longer contains" in problems(scenario(gold=drop_from_gold)),
      True)
check("static / a recorded question labelled the wrong way round",
      "recorded question(s) changed between answerable and adversarial" in problems(
          scenario(st=lambda d: d["index"][0].update(adversarial=True))), True)
check("static / recorded against a smaller index",
      "recorded against 1 documents" in problems(
          scenario(st=lambda d: d["manifest"]["corpora"]["t"].update(
              chunks=3, documents=1))), True)
check("static / recorded at a threshold corpora.json does not ship",
      "recorded at threshold +0.0" in problems(
          scenario(st=lambda d: d["manifest"]["corpora"]["t"].update(threshold=0.0))),
      True)


def rebuild_analytics(tmp, cfg):
    d = json.loads(cf.ANALYTICS.read_text(encoding="utf-8"))
    d["corpora"][0]["examples"].pop()
    write(cf.ANALYTICS, d)


check("static / its analytics.json copy is behind eval/analytics.json",
      "static-demo/analytics.json is behind" in problems(
          scenario(after=rebuild_analytics)), True)


def rerecord_without_site(tmp, cfg):
    d = json.loads((cf.STATIC / "manifest.json").read_text(encoding="utf-8"))
    d["recorded"] = "2026-09-07"
    write(cf.STATIC / "manifest.json", d)


check("static / site/recorded/ built from an older manifest",
      "built from an older manifest" in problems(
          scenario(after=rerecord_without_site)), True)


def analytics_rebuilt_and_copied(tmp, cfg):
    """analytics.json rebuilt and copied into static-demo/ without re-recording.

    The byte copy then agrees, so the only trace is the digest the manifest
    recorded when the answers were taken.
    """
    rebuild_analytics(tmp, cfg)
    shutil.copy2(cf.ANALYTICS, cf.STATIC / "analytics.json")


check("static / recorded from an older analytics.json than the current one",
      "recording was taken from an older eval/analytics.json" in problems(
          scenario(after=analytics_rebuilt_and_copied)), True)
check("static / and the manifest's analytics digest is read, not only written",
      "recording was taken from an older" in problems(scenario(
          st=lambda d: d["manifest"]["inputs"].update(
              analytics={"path": "eval/analytics.json", "digest": "sha256:0"}))),
      True)


# Until 2026-09-07 this check removed the whole static-demo directory, which
# exercised check_static's early return and not the note it was named for.
rep = scenario(st=lambda d: d["manifest"]["corpora"].pop("t"))
check("static / a corpus that was never recorded is a note, not a problem",
      (problems(rep),
       [r[2] for r in rep.rows if r[0] == "static-demo" and r[1] == "t"]),
      ("", ["note"]))
check("static / no static-demo directory at all is silent",
      [r for r in scenario(after=lambda tmp, cfg: shutil.rmtree(cf.STATIC)).rows
       if r[0] == "static-demo"], [])
check("static / a corpus corpora.json no longer configures",
      "no longer configures" in problems(scenario(
          st=lambda d: d["manifest"]["corpora"].update(
              u={"label": "Gone", "questions": 1, "inputs": {}}))), True)

# ---- the helpers everything above rests on ----------------------------------
check("digest / the ML corpus's artefacts are unsuffixed",
      cf.artefact_suffix(Path("eval/golden_set.json")), "")
check("digest / a topic corpus keeps its own name",
      cf.artefact_suffix(Path("eval/golden-birds.json")), "-birds")
check("digest / a hyphenated topic keeps everything after the first hyphen",
      cf.artefact_suffix(Path("eval/golden-quant-2026.json")), "-quant-2026")
check("digest / a file that is not there is not an error",
      cf.digest(Path("does-not-exist.json")), "missing")


def digests_differ():
    tmp = Path(tempfile.mkdtemp(prefix="freshness-"))
    try:
        a, b = tmp / "a.json", tmp / "b.json"
        a.write_text('{"x": 1}', encoding="utf-8")
        b.write_text('{"x": 2}', encoding="utf-8")
        same = tmp / "c.json"
        same.write_text('{"x": 1}', encoding="utf-8")
        return cf.digest(a) != cf.digest(b), cf.digest(a) == cf.digest(same)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


check("digest / different bytes, different digest", digests_differ(), (True, True))

# ---- exit codes are the contract -------------------------------------------
check("exit / a stale file is 1, a missing one is 2",
      (scenario(pc=lambda d: d.update(threshold=0.0)).broken,
       scenario(after=lambda tmp, cfg: cf.ANALYTICS.unlink()).broken),
      (False, True))
check("exit / an unstamped file is 1, not 2",
      scenario(thr=lambda d: d.pop("inputs")).broken, False)

# ---- the rebuild commands come out in chain order ---------------------------
# record_static.py copies eval/analytics.json into static-demo/ on every run,
# so build_analytics.py has to be listed before it even when analytics itself
# was found current. Until 2026-09-07 it was appended after.
ANALYTICS_CMD = ".venv\\Scripts\\python.exe src\\build_analytics.py"


def order(cmds, first, second):
    return next(i for i, c in enumerate(cmds) if first in c) < next(
        i for i, c in enumerate(cmds) if second in c)


cmds = scenario(after=rerecord_without_site).commands()
check("order / build_analytics.py precedes record_static.py when analytics is current",
      (ANALYTICS_CMD in cmds, order(cmds, "build_analytics.py", "record_static.py")),
      (True, True))
cmds = scenario(an=lambda d: d["corpora"][0]["examples"][0].update(adversarial=True),
                after=rerecord_without_site).commands()
check("order / and when analytics itself is stale",
      (cmds.count(ANALYTICS_CMD), order(cmds, "build_analytics.py", "record_static.py")),
      (1, True))
cmds = scenario(thr=lambda d: d.pop("inputs")).commands()
check("order / calibrate_threshold.py precedes build_analytics.py",
      order(cmds, "calibrate_threshold.py", "build_analytics.py"), True)
check("order / a current chain still lists build_analytics.py once",
      scenario().commands(), [ANALYTICS_CMD])


if __name__ == "__main__":
    raise SystemExit(suite.report())
