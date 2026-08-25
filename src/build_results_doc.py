"""Write the figures in eval/RESULTS.md from eval/results.json.

RESULTS.md has always said its numbers were copied from results.json. Nothing
copied them. They were typed, and they went stale exactly the way typed numbers
do -- the document claimed 20 papers and 2,768 chunks long after the corpus had
reached 36 and 5,459, and the header saying "if this file disagrees with that
one, this file is stale" was true without anyone noticing.

So the tables are generated and the prose is not. Everything between a
`<!-- generated:NAME -->` and `<!-- /generated:NAME -->` marker is replaced from
results.json; everything outside them is left alone, because the analysis around
the numbers is judgement and cannot be derived from a JSON file.

The corpus identity is checked before anything is written. results.json used to
be shared by every corpus, so whichever evaluation ran last owned it -- running
the bird set left it describing 45 Wikipedia documents while this document
described 36 arXiv papers. Each corpus writes its own file now, and this refuses
to fill an ML document from a bird run.
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent


def table(rows: dict, cols: list[tuple[str, str]], first: str) -> str:
    """A markdown table from an ordered dict of {label: {metric: value}}."""
    head = f"| {first} | " + " | ".join(h for h, _ in cols) + " |"
    rule = "|" + "---|" * (len(cols) + 1)
    out = [head, rule]
    for label, vals in rows.items():
        cells = []
        for _, key in cols:
            v = vals.get(key)
            cells.append("" if v is None
                         else f"{v:,}" if isinstance(v, int)
                         else f"{v:.3f}")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def blocks(r: dict) -> dict[str, str]:
    c = r["corpus"]
    ab = r.get("abstention", {})

    corpus = (
        f"**Corpus:** {c['documents']} documents -> {c['chunks']:,} chunks "
        f"(210 tokens, 40 overlap).\n"
        f"**Golden set:** {c['answerable_cases']} answerable + "
        f"{c['adversarial_cases']} adversarial, labels *derived* from answer "
        f"strings against the current corpus rather than hand-written.\n"
        f"**Retrieved:** k={c['k']} from {c['candidate_k']} candidates.\n"
        f"**Reproduce:** `python src/evaluate.py`"
    )

    metrics = [("any-hit@5", "hit_rate"), ("MRR", "mrr"),
               ("NDCG", "ndcg"), ("src recall", "src_recall")]
    end_to_end = table(r["end_to_end"], metrics, "pipeline")
    pool = table(r["candidate_pool"], metrics[:2] + metrics[3:], "first stage")
    exp = table(r["expansion"],
                [("context recall", "context_recall"),
                 ("tokens/query", "tokens_per_query"),
                 ("blocks/query", "blocks_per_query")], "expansion")

    sweep = ""
    if ab.get("sweep"):
        lines = ["| threshold | adversarial caught | answerable wrongly refused |",
                 "|---|---|---|"]
        for row in ab["sweep"]:
            lines.append(f"| {row['threshold']:+d} | "
                         f"{row['caught']} / {ab.get('n_adversarial', '?')} | "
                         f"{row['false_abstain']} / {ab.get('n_answerable', '?')} |")
        sweep = "\n".join(lines)
        shipped = ab.get("shipped_threshold")
        if shipped is not None:
            sweep += (f"\n\nShipped threshold: **{shipped:+.1f}**. "
                      f"Answerable questions score a median of "
                      f"{ab.get('answerable_median', 0):+.2f}.")

    # The cap's cost, read straight off the same rows. This was a hand-typed
    # table and drifted: it still said 0.818 for 1/src after the golden set
    # changed and the real figure became 0.776.
    e = r["end_to_end"]
    base = e.get("rrf + rerank", {})
    cap = ""
    if base:
        rows = ["| cap | any-hit | src recall | trade vs no cap |", "|---|---|---|---|",
                f"| none | {base['hit_rate']:.3f} | {base['src_recall']:.3f} | — |"]
        for label, key in (("2/src", "+ diversity 2/src"), ("1/src", "+ diversity 1/src")):
            v = e.get(key)
            if not v:
                continue
            dh = (v["hit_rate"] - base["hit_rate"]) * 100
            ds = (v["src_recall"] - base["src_recall"]) * 100
            rows.append(f"| {label} | {v['hit_rate']:.3f} | {v['src_recall']:.3f} | "
                        f"{ds:+.1f} src recall for {dh:+.1f} any-hit |")
        cap = "\n".join(rows)

    return {"corpus": corpus, "end-to-end": end_to_end, "diversity-cap": cap,
            "candidate-pool": pool, "expansion": exp, "abstention": sweep}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=ROOT / "eval" / "results.json")
    ap.add_argument("--doc", type=Path, default=ROOT / "eval" / "RESULTS.md")
    ap.add_argument("--expect-documents", type=int, default=None,
                    help="refuse to write unless results.json has this many "
                         "documents; guards against filling a document from "
                         "another corpus's run")
    ap.add_argument("--check", action="store_true",
                    help="report whether the document is current, write nothing")
    args = ap.parse_args()

    r = json.loads(args.results.read_text(encoding="utf-8"))
    if not r.get("corpus"):
        print(f"{args.results} has no corpus block -- re-run src/evaluate.py")
        return 1
    if args.expect_documents and r["corpus"]["documents"] != args.expect_documents:
        print(f"REFUSING: {args.results} describes "
              f"{r['corpus']['documents']} documents, expected "
              f"{args.expect_documents}. Wrong corpus's results.")
        return 1

    doc = args.doc.read_text(encoding="utf-8")
    made = blocks(r)
    written, missing = 0, []
    for name, body in made.items():
        # Match the markers themselves and rebuild the newlines, rather than
        # requiring one on each side of the body. An empty block has a single
        # newline between the two markers, not two, so a pattern demanding both
        # matched nothing on the first run -- every marker reported missing
        # while all of them sat in the file.
        pat = re.compile(
            rf"<!-- generated:{name} -->.*?<!-- /generated:{name} -->", re.S)
        if not pat.search(doc):
            missing.append(name)
            continue
        new = pat.sub(
            lambda _: f"<!-- generated:{name} -->\n{body}\n<!-- /generated:{name} -->",
            doc)
        if new != doc:
            written += 1
        doc = new

    if missing:
        print("no marker for: " + ", ".join(missing))
    if args.check:
        current = doc == args.doc.read_text(encoding="utf-8")
        print("RESULTS.md is current" if current
              else "RESULTS.md is STALE -- run src/build_results_doc.py")
        return 0 if current else 1

    args.doc.write_text(doc, encoding="utf-8")
    print(f"{args.doc}: {written} block(s) rewritten from "
          f"{r['corpus']['documents']} documents / {r['corpus']['chunks']:,} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
