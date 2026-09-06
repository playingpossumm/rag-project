"""Retrieval evaluation harness.

Scores retrieval configurations against a labelled golden set, so changes can be
shown to help rather than argued to help.

Relevance is judged at PAGE level, not chunk level: a chunk counts as relevant
if it came from a page the golden set lists for that question. Page-level
labelling is coarse -- it cannot tell a precise passage from a vague one on the
same page -- but it is the granularity a human can label reliably, and it stays
stable when chunking parameters change. Chunk-level labels would have to be
redone after every re-index, which is exactly when a comparable metric matters
most.

The harness reports two layers, because the two pipeline stages do different
jobs and improving one does not show up in the other's numbers:

  candidate recall@N  -- did the first stage put a relevant chunk in the pool
                         at all? Fusion is judged here; the reranker cannot
                         recover what was never retrieved.
  final hit/MRR/NDCG  -- did the finished pipeline rank it near the top?
                         Reranking is judged here.
"""
import argparse
import json
import math
import re
import sys
import statistics as st
from pathlib import Path

# The retrieval stack (faiss, sentence_transformers, the reranker) is imported
# inside main(), not here. Until 2026-09-06 it was imported at module level,
# so every script that wanted `normalize` or `gold_keys` from this file paid
# about 10 seconds and needed the models installed. The metric and
# normalisation functions below depend on none of it.

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
GOLDEN_SET = ROOT / "eval" / "golden_set.json"


def load_cases(path: Path = GOLDEN_SET) -> tuple[list[dict], list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data["cases"]
    return (
        [c for c in cases if not c.get("unanswerable")],
        [c for c in cases if c.get("unanswerable")],
    )


def gold_keys(case: dict) -> set:
    """Relevant locations as (source, kind, value) triples.

    Keyed on source as well as locator because page number alone stopped
    identifying a location once the corpus held more than one document -- every
    paper has a page 5. Matching on page alone would score a hit on the wrong
    document as correct, which is the most common real-world RAG failure.
    """
    # `kind` defaults to "page" so a golden set written before locator kinds
    # were recorded still scores correctly -- every corpus that predates this
    # was all-PDF.
    return {
        (entry["source"], entry.get("kind", "page"), str(p))
        for entry in case.get("gold", [])
        for p in entry["pages"]
    }


def gold_sources(case: dict) -> set:
    return {entry["source"] for entry in case.get("gold", [])}


def result_key(result: dict) -> tuple:
    """A returned passage's location as the (source, kind, value) triple
    `gold_keys` produces, so the two can be compared with `in`.

    One definition. `sweep_blend.py` carried its own until 2026-09-06, which
    defaulted a missing kind to "page" and fell back to a "page" field no
    result has carried since locators gained kinds; on real results the two
    agreed, and on a malformed one this raises where that one guessed.
    """
    loc = result["locator"]
    return (result["source"], loc["kind"], str(loc["value"]))


def is_relevant(result: dict, gold: set) -> bool:
    return result_key(result) in gold


def source_recall(results, sources: set, k: int) -> float:
    """Fraction of the documents that answer the question which were returned.

    This is the metric that matches "compile every relevant source" rather than
    "find one". hit@k cannot express it: a run that returns one of four
    answering papers scores a perfect 1.000 on hit@k while missing three
    quarters of the answer.

    Normalised by min(|gold|, k) because k results cannot represent more than k
    documents -- scoring against the raw count would penalise a run for a
    ceiling imposed by the caller's own k rather than by retrieval quality.
    """
    if not sources:
        return float("nan")
    found = {r["source"] for r in results} & sources
    return len(found) / min(len(sources), k)


def hit_rate(results, gold) -> float:
    """1.0 if any relevant location appears anywhere in the returned set."""
    return 1.0 if any(is_relevant(r, gold) for r in results) else 0.0


def reciprocal_rank(results, gold) -> float:
    """1/rank of the first relevant result; 0 if none. Rewards ranking it first."""
    for i, r in enumerate(results, start=1):
        if is_relevant(r, gold):
            return 1.0 / i
    return 0.0


def ndcg(results, gold) -> float:
    """Normalized discounted cumulative gain.

    The ideal ranking is the hits actually found, moved to the top -- not one
    slot per gold page. Because relevance is judged per page, several returned
    chunks can share a gold page, so normalizing by len(gold) would let DCG
    exceed IDCG and produce scores above 1.0.
    """
    dcg = sum(1.0 / math.log2(i + 1)
              for i, r in enumerate(results, start=1) if is_relevant(r, gold))
    n_relevant = sum(1 for r in results if is_relevant(r, gold))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_relevant + 1))
    return dcg / idcg if idcg else 0.0


def normalize(text: str) -> str:
    """Collapse whitespace and lowercase, so a phrase spanning a line break matches."""
    return re.sub(r"\s+", " ", text).strip().lower()


# Emphasis markers, heading hashes and the short bracketed markers the PDF
# conversion leaves behind. Stripping them matters because the corpus keeps
# them: the batch-normalization paper renders "_internal_ _covariate_ _shift,_"
# and a comparison that does not strip them calls the phrase absent from a
# passage where the reader plainly sees it -- the display path already strips
# the same things in ui/answer-mark.js, so a test that does not is measuring a
# text nobody is shown. En dashes do the same to "mean-variance". Found
# 2026-09-03; between them the two overstated the shortfall by two cases.
_MARKUP = re.compile(r"[_*`#]+|\[[^\]]{0,3}\]")
_DASH = re.compile("[‐-―−]")


def answer_normalize(text: str) -> str:
    """`normalize`, plus the conversion artefacts that hide a phrase.

    Separate from `normalize` deliberately. That one is shared with the
    answer-quality judge and the title-credit audit, which compare different
    things, and widening it under them would change numbers they are not about.
    """
    return normalize(_MARKUP.sub("", _DASH.sub("-", str(text))))


def context_recall(results, answer: str) -> float:
    """Does the text actually returned contain the answer?

    This is the metric ranking cannot express. hit@k asks whether a chunk from
    the right *page* was returned; a chunk can satisfy that while being cut
    before the sentence carrying the answer. Context recall asks the question
    that decides whether a model could answer at all.

    Measured across all three corpora on 2026-09-03, the difference is not
    small: 33 of 125 answerable questions score a hit whose returned text does
    not contain the answer, and reading all 33 found 21 where the passage does
    not answer at all. See src/audit_page_credit.py.
    """
    if not answer:
        return float("nan")
    needle = answer_normalize(answer)
    return 1.0 if any(needle in answer_normalize(r["text"]) for r in results) else 0.0


def context_tokens(results, tokenizer) -> int:
    """Total tokens handed to the generator -- the cost side of expansion."""
    return sum(len(tokenizer.encode(r["text"], add_special_tokens=False)) for r in results)


def relative_to_root(path: Path) -> str:
    """`eval/golden_set.json` however the path was given, with forward slashes."""
    try:
        return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def fixture_ids(cfg: dict) -> set:
    """The structural cases for one corpus, from eval/hard_cases*.json.

    Empty when no fixture has been written. Three sweeps and one comparison
    script carried an identical copy of this until 2026-09-06
    (sweep_decompose, sweep_ensemble, sweep_query_expansion and
    compare_embedders); they import it from here now.
    """
    from check_freshness import artefact_suffix
    p = ROOT / "eval" / f"hard_cases{artefact_suffix(cfg['golden'])}.json"
    if not p.exists():
        return set()
    data = json.loads(p.read_text(encoding="utf-8"))
    return {c["id"] if isinstance(c, dict) else c
            for c in data.get("structural", [])}


def score_run(cases, retrieve_fn, k: int = 5) -> tuple[dict, list[dict]]:
    totals = {"hit_rate": 0.0, "mrr": 0.0, "ndcg": 0.0, "src_recall": 0.0}
    # Averaged over the cases that declare an answer string rather than over
    # every case, because a case without one is not evidence either way and
    # folding it in as a zero would report a shortfall that is missing labels.
    visible_sum, visible_n = 0.0, 0
    per_case = []
    for case in cases:
        results = retrieve_fn(case["question"])
        gold = gold_keys(case)
        m = {
            "hit_rate": hit_rate(results, gold),
            "mrr": reciprocal_rank(results, gold),
            "ndcg": ndcg(results, gold),
            "src_recall": source_recall(results, gold_sources(case), k),
        }
        for key in totals:
            totals[key] += m[key]
        visible = context_recall(results, case.get("answer_contains", ""))
        if visible == visible:                          # not nan
            visible_sum += visible
            visible_n += 1
            m["answer_visible"] = visible
        per_case.append({
            "id": case["id"], "kind": case.get("kind", "-"),
            "gold_sources": sorted(gold_sources(case)),
            "got": [f"{r['source'][:14]}:{r['locator']['value']}" for r in results],
            **m,
        })
    n = len(cases) or 1
    summary = {name: total / n for name, total in totals.items()}
    # Reported beside hit_rate, not instead of it. hit_rate asks whether
    # retrieval reached a location that answers and answers that correctly;
    # this asks whether the answer is in the text the reader is handed. They
    # are different questions and were being read as one.
    if visible_n:
        summary["answer_visible"] = visible_sum / visible_n
    return summary, per_case


def main():
    from sentence_transformers import SentenceTransformer

    from abstain import ABSTAIN_THRESHOLD
    from hybrid import build_bm25
    from retrieve import (CANDIDATE_K, EMBEDDING_MODEL, TOP_K, load_ensemble,
                          load_index, retrieve, shortlist)
    import rerank as _rr

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=TOP_K)
    # None means "whatever this corpus ships", resolved once the corpus is
    # known. The module constant would measure a pool the server does not use.
    ap.add_argument("--candidate-k", type=int, default=None)
    # A corpus and its questions travel together, like the corpus and its
    # index. Scoring one corpus against another corpus's golden set produces
    # numbers that look fine and mean nothing.
    ap.add_argument("--golden", type=Path, default=GOLDEN_SET,
                    help="golden set to score against")
    ap.add_argument("--per-case", action="store_true")
    ap.add_argument("--rerank-blend", type=float, default=None,
                    help="how much first-stage ordering survives reranking; "
                         "default: the value corpora.json sets for this golden "
                         "set, so the harness measures what the server serves")
    ap.add_argument("--emit", type=Path, default=None,
                    help="write machine-readable results here "
                         "(default: derived from --golden, so each corpus keeps "
                         "its own file)")
    args = ap.parse_args()

    # Every corpus used to write eval/results.json, so whichever ran last owned
    # it. Running the bird evaluation left results.json describing 45 Wikipedia
    # documents while eval/RESULTS.md, which says its figures are copied from
    # that file, described 36 arXiv papers. Nothing errored; the two just
    # silently disagreed about which corpus they were about.
    # The server reads rerank_blend per corpus from corpora.json. If the
    # harness ignored it the two would disagree about the same pipeline: birds
    # ships at 0.20 and would be measured at 0.00. Match on the golden set.
    #
    # corpora.registry() answers from its built-in table when corpora.json is
    # absent and raises when the file does not parse, which is the right
    # outcome for a measurement run. This caught every exception until
    # 2026-09-06, so a corpora.json with a syntax error was reported as "not
    # in corpora.json" and the run scored at the module defaults for blend,
    # pool and threshold without saying why; the 2026-09-06 version caught
    # FileNotFoundError instead, which the registry never raises.
    corpus_cfg = None
    import corpora as _c
    for _c_cfg in _c.registry().values():
        if _c_cfg["golden"] and Path(_c_cfg["golden"]).name == args.golden.name:
            corpus_cfg = _c_cfg
            break
    # Passed to retrieve() on every call below rather than set on the rerank
    # module, so this file cannot leave a blend behind for the next importer.
    if args.rerank_blend is not None:
        blend = args.rerank_blend
    elif corpus_cfg:
        blend = corpus_cfg["rerank_blend"]
    else:
        blend = _rr.RERANK_BLEND

    # The threshold is the other setting that does not transfer, and it was
    # still being read from the module constant here -- so a bird run recorded
    # `shipped_threshold: 0.0` while corpora.json ships -3.0 for that corpus,
    # and the paragraph printed under it reasoned about a gate nobody serves.
    if args.candidate_k is None:
        args.candidate_k = (corpus_cfg["candidate_k"] if corpus_cfg
                            else CANDIDATE_K)
    shipped = (corpus_cfg["threshold"] if corpus_cfg and corpus_cfg["calibrated"]
               else ABSTAIN_THRESHOLD)
    if corpus_cfg:
        print(f"corpus {corpus_cfg['name']} ({corpus_cfg['label']}): threshold "
              f"{shipped:+.1f}, rerank blend {blend}, "
              f"candidates {args.candidate_k}, index {corpus_cfg['store'].name}")
    else:
        print(f"{args.golden.name} is not in corpora.json -- module defaults: "
              f"threshold {shipped:+.1f}, rerank blend {blend}")

    if args.emit is None:
        stem = args.golden.stem                      # golden_set / golden-birds
        suffix = "" if stem in ("golden_set", "golden") else stem.split("-", 1)[-1]
        name = "results.json" if not suffix else f"results-{suffix}.json"
        args.emit = Path(__file__).parent.parent / "eval" / name

    # Every reported figure is collected here and written out at the end.
    # Documentation quoting these numbers has drifted before -- the README once
    # claimed "84 evaluation cases" directly above figures measured on 23 -- so
    # there is now one machine-written source rather than four hand-maintained
    # copies. If a document disagrees with results.json, the document is wrong.
    emitted: dict = {"corpus": {}, "candidate_pool": {}, "end_to_end": {},
                     "expansion": {}, "abstention": {}}

    answerable, adversarial = load_cases(args.golden)
    # From the corpus rather than from RAG_STORE_DIR, so a golden set cannot be
    # scored against another corpus's index by forgetting an environment
    # variable. The env var still decides when the golden set is not one of
    # the configured corpora.
    index, metadata = load_index(corpus_cfg["store"] if corpus_cfg else None)
    # The second dense retriever, if this corpus has one. Without it this file
    # measured MRR 0.743 on quant while per_case.py and the server produced
    # 0.779 -- two independent paths disagreeing, which is the signal this
    # project treats as a defect rather than as noise.
    ensemble = load_ensemble(corpus_cfg["store"] if corpus_cfg else None)
    if ensemble:
        print(f"second dense retriever: {ensemble[1]}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    bm25 = build_bm25(metadata)

    print(f"corpus: {index.ntotal} chunks | cases: {len(answerable)} answerable, "
          f"{len(adversarial)} adversarial | k={args.k}, candidates={args.candidate_k}\n")

    # "corpus" and "abstention" have been declared and left empty since this
    # file was written, so every document quoting a corpus size was quoting a
    # hand-maintained number -- which is the exact failure results.json exists
    # to prevent. eval/RESULTS.md still said 20 papers and 2,768 chunks long
    # after the corpus reached 36 and 5,459.
    emitted["corpus"] = {
        "ensemble": ensemble[1] if ensemble else None,
        "chunks": int(index.ntotal),
        "documents": len({c["source"] for c in metadata}),
        "answerable_cases": len(answerable),
        "adversarial_cases": len(adversarial),
        "k": args.k,
        "candidate_k": args.candidate_k,
        # Relative to the repository. /api/eval serves this file verbatim, and
        # until 2026-09-07 the ML file carried the absolute path of the golden
        # set, which is a home directory on a public server.
        "golden": relative_to_root(args.golden),
    }

    # ---- Layer 1: candidate pool quality (no reranking) -------------------
    # This is where fusion is judged. Sparse-only is included as a reference
    # point, not a candidate configuration.
    pools = [
        ("dense",           dict(fusion="none")),
        ("rrf",             dict(fusion="rrf")),
        ("weighted a=0.3",  dict(fusion="weighted", alpha=0.3)),
        ("weighted a=0.5",  dict(fusion="weighted", alpha=0.5)),
        ("weighted a=0.7",  dict(fusion="weighted", alpha=0.7)),
    ]

    print(f"CANDIDATE POOL @ {args.candidate_k}   (can the reranker even see the answer?)")
    print(f"{'first stage':<18}{'any-hit':>9}{'MRR':>9}{'src recall':>12}")
    for label, cfg in pools:
        summary, _ = score_run(answerable, lambda q, c=cfg: shortlist(
            q, index, metadata, model, k=args.candidate_k, bm25=bm25, ensemble=ensemble, **c),
            k=args.candidate_k)
        emitted["candidate_pool"][label] = {
            k2: round(v, 3) for k2, v in summary.items()}
        print(f"{label:<18}{summary['hit_rate']:>9.3f}{summary['mrr']:>9.3f}"
              f"{summary['src_recall']:>12.3f}")

    # ---- Layer 2: end-to-end, with reranking ------------------------------
    # max_per_source=None disables the diversity cap, so its effect is isolated:
    # the indented rows differ from "rrf + rerank" in that one parameter only.
    #
    # Those two rows ran on WEIGHTED fusion until 2026-08-21, which meant the
    # row every document quoted as the default described a configuration this
    # system has never served -- DEFAULT_FUSION is "rrf". The cap was still
    # correctly isolated, so no conclusion drawn from it was wrong, and the two
    # branches sit within noise of each other, which is exactly why it survived
    # unnoticed. It was found by src/per_case.py measuring the served
    # configuration and getting a number that was not in this table at all.
    #
    # The ordering matters as much as the fusion: the indent means "the row
    # above, plus one change", so the row above has to be the one the indented
    # rows are actually a change *to*.
    finals = [
        ("dense, no rerank", dict(fusion="none", use_reranker=False)),
        ("dense + rerank",   dict(fusion="none", use_reranker=True, max_per_source=None)),
        ("weighted + rerank", dict(fusion="weighted", alpha=0.5, use_reranker=True,
                                   max_per_source=None)),
        ("rrf + rerank",     dict(fusion="rrf", use_reranker=True, max_per_source=None)),
        ("  + diversity 2/src", dict(fusion="rrf", use_reranker=True, max_per_source=2)),
        ("  + diversity 1/src", dict(fusion="rrf", use_reranker=True, max_per_source=1)),
    ]

    print(f"\nEND TO END @ {args.k}")
    print(f"{'pipeline':<20}{'any-hit':>9}{'MRR':>9}{'NDCG':>9}{'src recall':>12}"
          f"{'answer shown':>14}")
    runs = {}
    for label, cfg in finals:
        summary, per_case = score_run(answerable, lambda q, c=cfg: retrieve(
            q, index, metadata, model, k=args.k,
            candidate_k=args.candidate_k, bm25=bm25, ensemble=ensemble,
            rerank_blend=blend, **c), k=args.k)
        runs[label] = (summary, per_case)
        emitted["end_to_end"][label.strip()] = {
            k2: round(v, 3) for k2, v in summary.items()}
        print(f"{label:<20}{summary['hit_rate']:>9.3f}{summary['mrr']:>9.3f}"
              f"{summary['ndcg']:>9.3f}{summary['src_recall']:>12.3f}"
              f"{summary.get('answer_visible', float('nan')):>14.3f}")

    if args.per_case:
        for label, (_, per_case) in runs.items():
            print(f"\n--- {label} ---")
            for c in sorted(per_case, key=lambda x: (x["src_recall"], x["mrr"])):
                if c["mrr"] < 1.0 or c["src_recall"] < 1.0:
                    gold = ", ".join(g[:16] for g in c["gold_sources"])
                    print(f"  {c['id']:<14}{c['kind']:<10}"
                          f"mrr={c['mrr']:.2f} src={c['src_recall']:.2f}")
                    print(f"      gold: {gold}")
                    print(f"      got : {', '.join(c['got'])}")

    # ---- Layer 3: context expansion ---------------------------------------
    # Expansion cannot change ranking, so hit/MRR/NDCG are identical by
    # construction and reporting them here would be noise. What changes is
    # whether the returned text contains the answer, and what that costs.
    print(f"\nCONTEXT EXPANSION @ {args.k}   (ranking is unchanged by construction)")
    print(f"{'mode':<16}{'ctx recall':>12}{'tokens/query':>14}{'blocks':>9}")
    tok = model.tokenizer
    with_answers = [c for c in answerable if c.get("answer_contains")]
    for label, cfg in (("none (chunks)", dict(expansion="none")),
                       ("window +/-1", dict(expansion="window", window=1)),
                       ("page", dict(expansion="page"))):
        hits, toks, blocks = 0.0, 0, 0
        for case in with_answers:
            res = retrieve(case["question"], index, metadata, model, k=args.k,
                           candidate_k=args.candidate_k, bm25=bm25, ensemble=ensemble,
                           use_reranker=True, rerank_blend=blend, **cfg)
            hits += context_recall(res, case["answer_contains"])
            toks += context_tokens(res, tok)
            blocks += len(res)
        n = len(with_answers)
        emitted["expansion"][label] = {
            "context_recall": round(hits / n, 3),
            "tokens_per_query": round(toks / n),
            "blocks_per_query": round(blocks / n, 1),
        }
        print(f"{label:<16}{hits / n:>12.3f}{toks / n:>14.0f}{blocks / n:>9.1f}")

    # ---- Abstention calibration -------------------------------------------
    # Collect top-1 confidence for both populations, then sweep the threshold.
    # A threshold picked by eyeballing a handful of scores is a guess; this is
    # the tradeoff curve it should be picked from.
    def top1(case):
        res = retrieve(case["question"], index, metadata, model, k=args.k,
                       candidate_k=args.candidate_k, bm25=bm25, ensemble=ensemble,
                       use_reranker=True, rerank_blend=blend)
        return res[0]["rerank_score"] if res else float("-inf")

    ans_scores = [top1(c) for c in answerable]
    adv_scores = [top1(c) for c in adversarial]

    # `sorted(x)[len(x) // 2]` is the upper middle value, not the median, and on
    # an even-sized set the two differ: 26 bird cases reported +1.21 here and
    # +1.15 in eval/analytics.json, which uses statistics.median. One number,
    # two documents, two values -- the same kind of silent disagreement the
    # generated files exist to prevent, and only visible because both were
    # written down.
    print("\nABSTENTION CALIBRATION")
    print(f"  answerable   n={len(ans_scores):<3} min {min(ans_scores):+.2f}  "
          f"median {st.median(ans_scores):+.2f}  max {max(ans_scores):+.2f}")
    print(f"  unanswerable n={len(adv_scores):<3} min {min(adv_scores):+.2f}  "
          f"median {st.median(adv_scores):+.2f}  max {max(adv_scores):+.2f}")

    by_kind = {}
    for case, score in zip(adversarial, adv_scores):
        by_kind.setdefault(case.get("adversarial_kind", "?"), []).append(score)
    print("  unanswerable by kind:")
    for kind, scores in sorted(by_kind.items()):
        print(f"    {kind:<11} n={len(scores):<3} max {max(scores):+.2f}  "
              f"({', '.join(f'{s:+.1f}' for s in sorted(scores, reverse=True))})")

    print(f"\n  {'threshold':>10}{'caught':>9}{'false abstain':>15}{'net':>8}"
          f"{'in cases':>14}")
    best = None
    counts = {}
    for t in [-10, -8, -6, -5, -4, -3, -2, -1, 0, 1, 2]:
        n_caught = sum(1 for s in adv_scores if s < t)
        n_false = sum(1 for s in ans_scores if s < t)
        counts[t] = (n_caught, n_false)
        emitted["abstention"].setdefault("sweep", []).append(
            {"threshold": t, "caught": n_caught, "false_abstain": n_false})
        caught = n_caught / len(adv_scores)
        false_ab = n_false / len(ans_scores)
        net = caught - false_ab
        if best is None or net > best[1]:
            best = (t, net)
        print(f"  {t:>10}{caught:>9.3f}{false_ab:>15.3f}{net:>8.3f}"
              f"{f'{n_caught} / {n_false}':>14}")

    # The `net` column is Youden's J, which is a legitimate statistic and not
    # the one this decision wants. It subtracts two RATES over populations of
    # very different size, so an adversarial case is implicitly worth
    # len(answerable)/len(adversarial) answerable ones -- 3.7x here. Users
    # experience cases, not rates. The counts column above is there so the two
    # readings can be compared, because they disagree: the rate says +2 is a
    # clear win, and in cases it is break-even.
    ratio = len(ans_scores) / len(adv_scores)
    print(f"\n  best net separation at threshold {best[0]} (net {best[1]:.3f})"
          f"  -- but read the caveat")
    print(f"  CAVEAT: `net` subtracts rates over {len(adv_scores)} adversarial and "
          f"{len(ans_scores)} answerable cases,")
    print(f"          so it values one adversarial case at {ratio:.1f} answerable ones.")
    if shipped in counts and best[0] in counts and best[0] != shipped:
        dc = counts[best[0]][0] - counts[shipped][0]
        dl = counts[best[0]][1] - counts[shipped][1]
        print(f"          In cases, moving {shipped:+.0f} -> {best[0]:+d} trades "
              f"{dc} more caught for {dl} more lost.")
    print("          src/calibrate_threshold.py names the questions each step costs.")
    print("  note: false abstention is the costlier error -- refusing a question the")
    print("        corpus CAN answer is worse than answering a weak one with citations,")
    print(f"        which is why the shipped threshold stays at {shipped:+.1f}.")
    emitted["abstention"]["shipped_threshold"] = float(shipped)
    emitted["abstention"]["best_net_threshold"] = best[0]
    emitted["abstention"]["n_answerable"] = len(ans_scores)
    emitted["abstention"]["n_adversarial"] = len(adv_scores)
    emitted["abstention"]["answerable_median"] = st.median(ans_scores)
    emitted["abstention"]["adversarial_median"] = st.median(adv_scores)
    if corpus_cfg:
        emitted["corpus"]["name"] = corpus_cfg["name"]
        emitted["corpus"]["rerank_blend"] = blend

    if args.emit:
        args.emit.parent.mkdir(exist_ok=True)
        emitted["generated_by"] = "src/evaluate.py"
        # What this run was measured against, so a later edit to either input
        # is detectable rather than silent. per_case.py has recorded this since
        # 2026-08-26 and this file did not, which meant a corrected label that
        # changed no count left these figures looking current.
        from check_freshness import stamp as _stamp
        emitted["inputs"] = {
            "golden": _stamp(args.golden,
                             cases=len(answerable) + len(adversarial)),
        }
        if corpus_cfg:
            emitted["inputs"]["index"] = _stamp(
                corpus_cfg["store"] / "metadata.json", chunks=int(index.ntotal))
        args.emit.write_text(json.dumps(emitted, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.emit}")


if __name__ == "__main__":
    main()
