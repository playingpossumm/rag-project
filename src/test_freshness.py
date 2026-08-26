"""Does the freshness check actually catch the ways this chain has gone stale?

A check that never fails is indistinguishable from no check, and this project
has already shipped one: `pipeline_trace.py`'s docstring claimed a test asserting
it matched the serving path, and no such test existed. So each way the chain
`golden set -> per_case -> analytics -> the front page` is known to have broken
is reproduced here on a synthetic corpus, and the assertion is that
`check_freshness` reports it -- naming the case, not just returning nonzero.

Three of these are not hypotheses. They happened:

- `per_case.json` sat four days out of date while the golden set moved on, and
  the front page went on offering questions that had been deleted for being
  unanswerable. That is `offers a deleted question`.
- Every corpus wrote one shared `results.json`, so the last run owned it and a
  bird run left an ML document describing 45 Wikipedia pages. That is
  `results run against another corpus's golden set`.
- `per_case.py` scored every corpus at the ML papers' 0.0 threshold and at
  rerank blend 0.0 while two corpora ship neither. That is `threshold drift` and
  `blend drift`, and it was live in the repo until 2026-08-26.

Hermetic: everything runs against files written into a temporary directory, so
this passes or fails on the code rather than on the state of the real eval
artefacts. No corpus, no models, milliseconds.

    .venv\\Scripts\\python.exe src\\test_freshness.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import check_freshness as cf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHECKS: list[tuple[str, object, object, bool]] = []


def check(name: str, got, want) -> None:
    CHECKS.append((name, got, want, got == want))


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


def scenario(gold=None, pc=None, res=None, an=None, after=None):
    """Build a whole chain in a temp directory and run the checker over it.

    `pc`, `res` and `an` mutate one artefact as it is written, which is how a
    file that was generated wrong gets staged. `gold` and `after` run once the
    whole chain exists, which is how a file that was generated right and then
    overtaken gets staged -- and that ordering is the whole point. Mutating the
    golden set first would produce a chain that agrees with itself perfectly,
    which is a test of nothing: staleness is a relationship between files, not a
    property of one.
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

        cfg = {"name": "t", "label": "Test corpus", "data": tmp / "data-t",
               "store": store, "golden": tmp / "eval" / "golden-t.json",
               "threshold": -3.0, "calibrated": True, "rerank_blend": 0.2,
               "indexed": True}

        g = json.loads(json.dumps(GOLD))
        write(cfg["golden"], g)

        n_ans = sum(1 for c in g["cases"] if not c.get("unanswerable"))
        n_adv = len(g["cases"]) - n_ans

        per = {
            "generated_by": "src/per_case.py",
            "options": {"k": 5, "candidate_k": 20, "rerank_blend": 0.2,
                        "use_reranker": True, "fusion": "rrf", "max_per_source": 2},
            "threshold": -3.0,
            "corpus": {"name": "t", "chunks": 6, "documents": 2},
            "inputs": {"golden": cf.stamp(cfg["golden"], cases=len(g["cases"])),
                       "index": cf.stamp(store / "metadata.json", chunks=6)},
            "cases": [{"id": c["id"], "question": c["question"],
                       "unanswerable": bool(c.get("unanswerable")),
                       "confidence": -1.0, "gold_sources": [],
                       "outcome": "found"} for c in g["cases"]],
        }
        if pc:
            pc(per)
        write(cfg["golden"].parent / "per_case-t.json", per)

        results = {"corpus": {"chunks": 6, "documents": 2,
                              "answerable_cases": n_ans, "adversarial_cases": n_adv,
                              "k": 5, "candidate_k": 20,
                              "golden": str(cfg["golden"])}}
        if res:
            res(results)
        write(cfg["golden"].parent / "results-t.json", results)

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
                              "label": "one figure in one place"} for c in g["cases"]],
            }],
        }
        if an:
            an(analytics)
        write(cf.ANALYTICS, analytics)

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
        cf.check_analytics(rep, {"t": cfg}, {"t": gold_now},
                           {"t": per_now} if per_now else {})
        return rep
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def problems(rep) -> str:
    """Every complaint as one lowercase string, for substring assertions."""
    return " || ".join(f"{i}: {d}" for _, i, s, d in rep.rows
                       if s in ("STALE", "MISSING")).lower()


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
      "golden set has changed" in problems(rep), True)
check("provenance / and the reworded question is named",
      "reworded" in problems(rep), True)


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


# The index rebuilt under a scored file. Chunk counts happen to be unchanged
# here, so only the digest can see it -- an ingest that re-chunks the same
# documents moves every offset while moving no total.
def reingest(tmp, cfg):
    write(cfg["store"] / "metadata.json",
          [{"source": s, "text": f"re-chunked {i}"}
           for i, s in enumerate(["a.pdf", "a.pdf", "a.pdf", "b.pdf", "b.pdf", "b.pdf"])])


check("provenance / the index was rebuilt after scoring",
      "index has been rebuilt" in problems(scenario(after=reingest)), True)

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


def main() -> int:
    width = max(len(n) for n, *_ in CHECKS)
    failed = 0
    for name, got, want, ok in CHECKS:
        if not ok:
            failed += 1
            print(f"FAIL  {name:<{width}}  got {got!r}, want {want!r}")
    print(f"{len(CHECKS) - failed}/{len(CHECKS)} freshness checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
