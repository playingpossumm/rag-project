"""Does a hit mean the reader got the answer, or only the right page?

Two counts that ought to be close and are not. Eleven questions score as misses
across the three corpora, and thirty-three score a hit while no passage the
reader is shown contains the answer. The gap is in what `is_relevant` compares.

Gold is derived by `build_golden_set.derive_gold`: a case declares an answer
string, every chunk containing it is found, and its LOCATOR is recorded, which
is a source, a kind and a page. `is_relevant` then asks whether a result's
locator is in that set. A page is longer than a chunk, so a page carrying the
answer in one of its chunks marks EVERY chunk of that page relevant, and a
result from the right page scores a hit whether or not the answer is anywhere
the reader can see it.

That is not automatically wrong, and this file exists because the obvious
reading of it is the one this project has already got wrong once. On
2026-08-31 an audit called two credits false because their chunks opened on a
title block, and reading them in full showed both continue into an abstract
that answers outright. A passage can answer without carrying one exact
phrasing, so the only way to settle this is to read the passages.

All thirty-three were read on 2026-09-03. The verdicts are below, with the
reason for each, and they split three ways rather than one:

    21  the passage does not answer. The reader is shown a page that holds
        the answer somewhere and a paragraph of it that does not.
     7  the passage answers in different words, so the credit is right and the
        string test is what is too strict.
     5  arguable, recorded as arguable rather than pushed to either side.

So roughly one answerable question in six is scored as found while the reader
gets nothing. any-hit@5 is not measuring the wrong thing -- it measures whether
retrieval reached a location that answers, and it does -- but it is not the
question a reader asks, and the two were being reported as though they were.

**Two measurement artefacts, both found by reading and both corrected here.**
The corpus is converted from PDF and keeps markdown emphasis, so the
batch-normalization paper renders "_internal_ _covariate_ _shift,_" and an
exact-string test called it absent when a reader plainly sees it; the display
path already strips these in `answer-mark.js`. En dashes do the same to
"mean-variance". Both normalisations are applied below, and between them they
moved two cases out of the population before anything was read.

    .venv\\Scripts\\python.exe src\\audit_page_credit.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluate import gold_keys  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent

GOLDEN = {"birds": "eval/golden-birds.json", "llm": "eval/golden_set.json",
          "quant": "eval/golden-quant.json"}

DASH = re.compile("[‐-―−]")
# Emphasis markers, heading hashes, and the short bracketed markers the PDF
# converter leaves behind. `answer-mark.js` strips the same things before the
# reader sees a passage, so a test that does not is measuring a different text.
MARKUP = re.compile(r"[_*`#]+|\[[^\]]{0,3}\]")


def norm(text) -> str:
    return re.sub(r"\s+", " ",
                  MARKUP.sub("", DASH.sub("-", str(text)))).strip().lower()


# Read individually on 2026-09-03. The reason matters more than the verdict:
# a later reader disagreeing with one of these should be able to see what was
# in front of the passage when it was judged.
DOES_NOT_ANSWER = {
    "bird-flyway": "names migration routes, never the word flyway",
    "bird-imprinting": "mallard clutch timing, unrelated to imprinting",
    "bird-aspect-ratio": "wing chord measurements, not the ratio asked for",
    "bird-fledging": "chick development by week, never names the stage",
    "bird-melanin": "says colours come from pigments without naming one",
    "attn-uses": "gives h = 8 heads, not the three uses asked for",
    "lora-frozen": "describes full fine-tuning, the opposite arrangement",
    "faiss-ivf": "describes searching partitions without naming the index",
    "t5-c4": "denoising objectives, not the corpus",
    "sbert-speed": "the FAISS paper's 8.5x, a different paper and quantity",
    "cot-gsm8k": "a figure caption naming no benchmark",
    "cot-scale": "asks why scale helps and gives no scale",
    "dpr-nq": "describes open-domain QA and names no dataset",
    "qf-news-taxonomy": "states that layers are built, gives no count",
    "qf-market-making": "defines lots and ticks, not the activity",
    "qf-max-drawdown": "Rachev and Sharpe ratios, not drawdown",
    "qf-market-making-rl": "a title block and keywords",
    "qf-deep-learning": "probabilistic estimation, names no method family",
    "qf-cvar-interval": "convex cost criteria, not the interval",
    "qf-bvar-probabilistic": "forecasting accuracy in general terms",
    "qf-markowitz": "a keywords block that never names Markowitz",
}
ANSWERS_IN_OTHER_WORDS = {
    "bird-mirror": "'mirror self-recognition ... in European magpies'",
    "adam": "a hyperparameter table giving AdamW and beta .9/.999",
    "seq2seq-lstm": "names the LSTM architecture outright",
    "w2v-cbow": "'the CBOW architecture predicts the current word'",
    "t5-span": "'corrupts contiguous, randomly spaced spans of tokens'",
    "qf-transaction-costs": "'blockchain frictions, such as gas fees'",
    "qf-order-flow": "names the weighted volume imbalance signal",
}
ARGUABLE = {
    "cot-prompt": "says equation-only prompting helps, not the full effect",
    "resnet-degradation": "discusses deep plain nets without saying saturated",
    "pos-enc-fn": "explains why the function was chosen, not the function",
    "qf-degeneracy": "the right paper, and the name is in its title only",
    "qf-liquidity-policy": "allocation under uncertainty, not by mispricing",
}
READ = {**DOES_NOT_ANSWER, **ANSWERS_IN_OTHER_WORDS, **ARGUABLE}


def locator_triple(item: dict) -> tuple:
    """The recorded payload flattens the locator to "page 4"."""
    kind, _, value = str(item["locator"]).partition(" ")
    return (item["source"], kind, value)


def main() -> int:
    rows = []
    for corpus, gpath in GOLDEN.items():
        cases = {c["question"]: c for c in json.loads(
            (ROOT / gpath).read_text(encoding="utf-8"))["cases"]}
        recorded = ROOT / "static-demo" / corpus
        if not recorded.is_dir():
            print(f"  {corpus}: no recorded payloads; run src/record_static.py")
            continue
        for f in sorted(recorded.glob("*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            if not isinstance(d, dict) or "query" not in d:
                continue
            case = cases.get(d["query"])
            if not case or case.get("unanswerable"):
                continue
            want = case.get("answer_contains")
            if not want:
                continue
            items = d["stages"][-1]["items"]
            gold = gold_keys(case)
            rows.append({
                "id": case["id"], "corpus": corpus,
                "scored": any(locator_triple(i) in gold for i in items),
                "visible": any(norm(want) in norm(i.get("text", ""))
                               for i in items),
            })

    if not rows:
        return 2
    both = [r for r in rows if r["scored"] and r["visible"]]
    page_only = [r for r in rows if r["scored"] and not r["visible"]]
    missed = [r for r in rows if not r["scored"]]

    print(f"\n  {len(rows)} answerable questions carrying an answer string\n")
    print(f"    {len(both):>4}  scored a hit, and a returned passage holds the "
          f"answer")
    print(f"    {len(page_only):>4}  scored a hit on the LOCATOR alone")
    print(f"    {len(missed):>4}  scored a miss")

    ids = {r["id"] for r in page_only}
    unread = sorted(ids - READ.keys())
    stale = sorted(READ.keys() - ids)

    print(f"\n  of the {len(ids)} credited on the locator alone, as read on "
          f"2026-09-03:")
    for label, group in (("do not answer", DOES_NOT_ANSWER),
                         ("answer in other words", ANSWERS_IN_OTHER_WORDS),
                         ("arguable", ARGUABLE)):
        live = sorted(set(group) & ids)
        print(f"\n    {len(live)} {label}")
        for cid in live:
            print(f"      {cid:<24}{group[cid]}")

    problems = []
    if unread:
        problems.append(f"credited on the locator alone and never read: "
                        f"{', '.join(unread)}")
    if stale:
        problems.append(f"read, but no longer credited on the locator alone, "
                        f"so the verdict is stale: {', '.join(stale)}")
    print()
    for p in problems:
        print(f"  DRIFTED  {p}")
    if problems:
        print("\n  The population moved. Read the new cases before quoting any "
              "count\n  from this file, because the counts above describe the "
              "cases that were read.")
        return 1
    print(f"  Every case credited on the locator alone has been read. "
          f"{len(DOES_NOT_ANSWER)} of\n  {len(rows)} answerable questions are "
          f"scored as found while the reader gets nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
