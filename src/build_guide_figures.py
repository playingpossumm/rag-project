"""Data figures for docs/understanding-rag.html, drawn from the measurements.

The guide's first five figures are hand-drawn schematics -- how indexing works,
how RRF is computed -- and those are diagrams of a mechanism, so they are
authored once and stay. Everything this file emits is a picture of a NUMBER, and
a picture of a number goes stale the moment the number moves. So they are
generated from eval/analytics.json and re-injected, the same rule the analytics
page follows.

Insertion is by anchor comment. Each figure is written between
    <!-- GEN:name -->  ...  <!-- /GEN:name -->
so re-running replaces the previous drawing rather than appending a second one.

Run:  python src/build_guide_figures.py     ->  edits docs/understanding-rag.html
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "eval" / "analytics.json"
GUIDE = ROOT / "docs" / "understanding-rag.html"

W = 740
HUE = ["#3987e5", "#d95926", "#199e70"]
INK, NEAR, SOFT, MUTED, FAINT = "#f3f3f1", "#dcdedf", "#a8adb1", "#7f858a", "#5d6367"
HAIR, EDGE = "#26282b", "#33363a"
SERIF = "Newsreader, Georgia, serif"
MONO = "JetBrains Mono, ui-monospace, Consolas, monospace"

# Named, because a literal newline escape inside a generated f-string is exactly
# how the last four versions of this file got mangled.
NL = chr(10)


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def figure(num, title, svg, caption):
    return (f'<figure>\n  <div class="fignum">FIG. {num} — {title.upper()}</div>\n'
            f'{svg}\n  <figcaption>{caption}</figcaption>\n</figure>')


def frame(h, body, label):
    return (f'  <svg viewBox="0 0 {W} {h}" role="img" aria-label="{esc(label)}">\n'
            f'    <g font-family="{SERIF}" font-size="13" fill="{NEAR}">\n'
            f'{body}\n    </g>\n  </svg>')


def txt(x, y, s, anchor="start", fill=NEAR, size=13, mono=False, weight=None):
    fam = f' font-family="{MONO}"' if mono else ""
    wt = f' font-weight="{weight}"' if weight else ""
    return (f'      <text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'fill="{fill}" font-size="{size}"{fam}{wt}>{esc(s)}</text>')


def bar(x, y, w, h, fill, opacity=1.0):
    return (f'      <rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 1.2):.1f}" '
            f'height="{h}" rx="3" fill="{fill}" opacity="{opacity}"/>')


# ---------------------------------------------------------------------------
def fig_bm25(C):
    """What the lexical retriever adds -- and, on two corpora, costs."""
    L, R, ROW, top = 172, 92, 22, 30
    plot = W - L - R
    rows = []
    for i, c in enumerate(C):
        pool = {p["config"]: p for p in c["pool"]}
        rows.append((c, pool["dense"]["hit_rate"], pool["rrf"]["hit_rate"], i))
    h = top + len(rows) * ROW * 2 + 26

    b = [txt(L, 18, "candidate pool @20 — any relevant passage found", size=11,
             mono=True, fill=MUTED)]
    for n, (c, dense, rrf, i) in enumerate(rows):
        y = top + n * ROW * 2
        b.append(txt(L - 12, y + 12, c["label"], "end", NEAR, 13))
        for k, v in enumerate((dense, rrf)):
            yy = y + k * 15
            b.append(bar(L, yy + 3, plot * v, 9, HUE[i], 1.0 if k else 0.42))
            b.append(txt(L + plot * v + 7, yy + 11.5, f"{v:.3f}", "start",
                         SOFT, 10.5, mono=True))
        d = rrf - dense
        sign = "+" if d > 0 else "−"
        b.append(txt(W - 4, y + 19, f"{sign}{abs(d):.3f}", "end",
                     "#3fb950" if d > 0 else "#f0685f", 11.5, mono=True))
    b.append(txt(L, h - 6, "pale = dense only  ·  solid = with BM25 fused in",
                 "start", FAINT, 9.5, mono=True))
    return frame(h, "\n".join(x for x in b if x),
                 "Candidate pool hit rate with dense retrieval alone versus "
                 "dense fused with BM25, on three corpora.")


def fig_ladder(C):
    """Each mechanism's contribution, on the corpus the system was tuned on."""
    c = C[0]
    NAMES = {"dense, no rerank": "dense retrieval alone",
             "dense + rerank": "+ cross-encoder rerank",
             "rrf + rerank": "+ BM25, fused by RRF",
             "+ diversity 2/src": "+ diversity cap, 2 per document",
             "+ diversity 1/src": "+ diversity cap, 1 per document"}
    L, R, ROW, top = 226, 96, 27, 40
    plot = W - L - R
    rows = c["ladder"]
    h = top + len(rows) * ROW + 20

    b = [txt(L, 18, "MRR — how high the first relevant passage sits", size=11,
             mono=True, fill=MUTED)]
    for n, r in enumerate(rows):
        y = top + n * ROW
        ship = r["config"] == "+ diversity 2/src"
        b.append(txt(L - 12, y + 12, NAMES.get(r["config"], r["config"]), "end",
                     INK if ship else SOFT, 13))
        b.append(bar(L, y + 3, plot * r["mrr"], 11, HUE[0], 1.0 if ship else 0.5))
        b.append(txt(L + plot * r["mrr"] + 7, y + 12.5, f"{r['mrr']:.3f}", "start",
                     INK if ship else SOFT, 11, mono=True))
        if n:
            d = r["mrr"] - rows[n - 1]["mrr"]
            sign = "+" if d > 0 else "−"
            b.append(txt(W - 4, y + 12.5, f"{sign}{abs(d):.3f}", "end",
                         "#3fb950" if d > 0 else "#f0685f", 11, mono=True))
        if ship:
            b.append(txt(L + plot * r["mrr"] + 58, y + 12.5, "shipped", "start",
                         "#3fb950", 9.5, mono=True))
    return frame(h, "\n".join(b),
                 "Mean reciprocal rank for each configuration, each row adding "
                 "one mechanism to the row above.")


def fig_cap(C):
    """The diversity cap is a trade, and the figure has to show both sides."""
    c = C[0]
    lad = {r["config"]: r for r in c["ladder"]}
    before, after = lad["rrf + rerank"], lad["+ diversity 2/src"]
    pairs = [("Hit rate @5", "any relevant passage in the top five",
              before["hit_rate"], after["hit_rate"]),
             ("Source recall", "every document that answers, returned",
              before["src_recall"], after["src_recall"])]
    L, R, top = 210, 110, 36
    plot = W - L - R
    h = top + len(pairs) * 62 + 14
    b = [txt(L, 18, "before and after the cap, everything else held", size=11,
             mono=True, fill=MUTED)]
    for n, (name, sub, v0, v1) in enumerate(pairs):
        y = top + n * 62
        b.append(txt(L - 12, y + 12, name, "end", INK, 13.5))
        b.append(txt(L - 12, y + 28, sub, "end", FAINT, 10, mono=True))
        for k, v in enumerate((v0, v1)):
            yy = y + k * 17
            b.append(bar(L, yy + 2, plot * v, 11, HUE[0], 0.42 if k == 0 else 1.0))
            b.append(txt(L + plot * v + 7, yy + 11.5, f"{v:.3f}", "start",
                         SOFT if k == 0 else INK, 11, mono=True))
        d = v1 - v0
        sign = "+" if d > 0 else "−"
        b.append(txt(W - 4, y + 20, f"{sign}{abs(d):.3f}", "end",
                     "#3fb950" if d > 0 else "#f0685f", 12, mono=True))
    b.append(txt(L, h - 6, "pale = without the cap  ·  solid = with it",
                 "start", FAINT, 9.5, mono=True))
    return frame(h, "\n".join(b),
                 "Hit rate and source recall, with and without the "
                 "two-passages-per-document cap.")


def fig_thresholds(C):
    """Every scored question on one axis. The overlap is the whole argument."""
    L, R, ROW, top = 168, 30, 74, 30
    plot = W - L - R
    allv = [v for c in C for v in c["dist"]["answerable"] + c["dist"]["adversarial"]]
    lo, hi = min(allv) - 0.6, max(allv) + 0.6
    x = lambda v: L + (v - lo) / (hi - lo) * plot
    h = top + len(C) * ROW + 26

    b = []
    for i, c in enumerate(C):
        y = top + i * ROW
        b.append(txt(L - 14, y + 20, c["label"], "end", INK, 13))
        b.append(txt(L - 14, y + 36, f"{c['n_answerable']} + {c['n_adversarial']} cases",
                     "end", FAINT, 9.5, mono=True))
        b.append(f'      <line x1="{x(0):.1f}" y1="{y + 4}" x2="{x(0):.1f}" '
                 f'y2="{y + 52}" stroke="{FAINT}" stroke-width="1" stroke-dasharray="3 3"/>')
        b.append(f'      <line x1="{x(c["threshold"]):.1f}" y1="{y + 4}" '
                 f'x2="{x(c["threshold"]):.1f}" y2="{y + 52}" stroke="{HUE[i]}" '
                 f'stroke-width="1.5"/>')
        lab = f"{c['threshold']:+.1f}".replace("+0.0", "0.0")
        b.append(txt(x(c["threshold"]), y, lab + " calibrated", "middle", HUE[i],
                     9.5, mono=True))
        if c["threshold"] != 0:
            b.append(txt(x(0), y, "old 0.0", "middle", FAINT, 9.5, mono=True))
        for v in c["dist"]["answerable"]:
            b.append(f'      <circle cx="{x(v):.1f}" cy="{y + 18}" r="3.2" '
                     f'fill="{HUE[i]}" opacity=".85"/>')
        for v in c["dist"]["adversarial"]:
            b.append(f'      <circle cx="{x(v):.1f}" cy="{y + 38}" r="3" fill="none" '
                     f'stroke="{HUE[i]}" stroke-width="1.4" opacity=".9"/>')
    for v in range(int(lo // 2 * 2), int(hi) + 1, 2):
        b.append(txt(x(v), h - 12, f"{v:+d}".replace("+0", "0"), "middle",
                     FAINT, 9.5, mono=True))
    b.append(txt(L, h - 0, "filled = answerable  ·  hollow = adversarial, the "
                 "answer is deliberately absent", "start", FAINT, 9.5, mono=True))
    return frame(h + 4, "\n".join(b),
                 "Cross-encoder confidence for every labelled question on three "
                 "corpora, answerable and adversarial shown separately.")


def fig_divisor(C):
    """Why source recall is not comparable across corpora as a raw number."""
    L, R, ROW, top = 210, 150, 30, 34
    plot = W - L - R
    top_v = max(c["gold_divisor"] for c in C)
    h = top + len(C) * ROW + 16
    b = [txt(L, 18, "mean min(|gold|, 5) — the divisor source recall is scored against",
             size=11, mono=True, fill=MUTED)]
    for i, c in enumerate(C):
        y = top + i * ROW
        b.append(txt(L - 12, y + 13, c["label"], "end", NEAR, 13))
        b.append(bar(L, y + 4, plot * (c["gold_divisor"] / top_v), 11, HUE[i]))
        b.append(txt(L + plot * (c["gold_divisor"] / top_v) + 7, y + 13,
                     f"{c['gold_divisor']:.2f}", "start", INK, 11, mono=True))
        b.append(txt(W - 4, y + 13, f"{c['gold_mean']:.2f} gold docs/question", "end",
                     FAINT, 10, mono=True))
    return frame(h, "\n".join(b),
                 "The average number of documents a question is scored against, "
                 "per corpus.")


def fig_transfer(C):
    """The same pipeline, three corpora, four measures."""
    METRICS = [("hit_rate", "Hit rate @5"), ("mrr", "MRR"),
               ("src_recall", "Source recall")]
    L, GAP, top, BH = 150, 34, 44, 13
    colw = (W - L - 20 - GAP * (len(METRICS) - 1)) / len(METRICS)
    h = top + len(C) * (BH + 6) + 52
    b = []
    for m, (key, name) in enumerate(METRICS):
        x0 = L + m * (colw + GAP)
        b.append(txt(x0, 20, name, "start", INK, 12.5))
        b.append(f'      <line x1="{x0}" y1="{top - 10}" x2="{x0 + colw}" '
                 f'y2="{top - 10}" stroke="{HAIR}"/>')
        for i, c in enumerate(C):
            ship = next(r for r in c["ladder"] if r["config"] == "+ diversity 2/src")
            v = ship[key]
            y = top + i * (BH + 6)
            b.append(bar(x0, y, colw * v, BH, HUE[i]))
            b.append(txt(x0 + colw * v + 6, y + 10.5, f"{v:.3f}", "start",
                         SOFT, 10, mono=True))
            if m == 0:
                b.append(txt(L - 12, y + 10.5, c["label"], "end", NEAR, 12.5))
                b.append(txt(L - 12, y + 22, f"{c['documents']} docs", "end",
                             FAINT, 9, mono=True))
    b.append(txt(L, h - 10, "shipped configuration, scored on each corpus's own "
                 "golden set", "start", FAINT, 9.5, mono=True))
    return frame(h, "\n".join(b),
                 "Hit rate, MRR and source recall for the shipped pipeline on "
                 "each of the three corpora.")


# --- tables ----------------------------------------------------------------
# Chapter 12's two tables were written by hand against two corpora. Generated
# now for the same reason as the figures: a third corpus arrived and the tables
# did not know about it.
def tbl_transfer(C):
    rows = [("hit rate", "hit_rate"), ("MRR", "mrr"),
            ("NDCG", "ndcg"), ("source recall", "src_recall")]
    ship = {c["name"]: next(r for r in c["ladder"] if r["config"] == "+ diversity 2/src")
            for c in C}
    head = "".join(f'<th class="r">{esc(c["label"])} ({c["documents"]})</th>' for c in C)
    out = ['<div class="tw">', "<table>",
           f"  <thead><tr><th>shipped pipeline</th>{head}</tr></thead>", "  <tbody>"]
    for name, key in rows:
        # Bold the best cell per row. On source recall that is the bird corpus,
        # not the one the system was tuned on, which is worth not hiding.
        best = max(ship[c["name"]][key] for c in C)
        cells = ""
        for c in C:
            v = ship[c["name"]][key]
            inner = f"<strong>{v:.3f}</strong>" if v == best else f"{v:.3f}"
            cells += f'<td class="r m">{inner}</td>'
        out.append(f"    <tr><td>{name}</td>{cells}</tr>")
    out += ["  </tbody>", "</table>", "</div>"]
    return NL.join(out)


def tbl_threshold(C):
    rows = []
    for c in C:
        z, k = c["at_zero"], c["at_calibrated"]
        pct = z["refused"] / c["n_answerable"] * 100
        bad = pct > 20
        refused = (f'{z["refused"]} of {c["n_answerable"]} &mdash; {pct:.1f}%')
        if bad:
            refused = f"<strong>{refused}</strong>"
        cut = f'{c["threshold"]:+.1f}'.replace("+0.0", "0.0")
        # Five columns, not six. The answerable median was the sixth and it is
        # already drawn, per case rather than summarised, in the distribution
        # figure back in chapter 8 -- and at six the last column fell off the
        # page and had to be scrolled to.
        rows.append(
            f'      <tr><td>{esc(c["label"])}</td>'
            f'<td class="r m">{refused}</td>'
            f'<td class="r m">{cut}</td>'
            f'<td class="r m">{k["refused"]} of {c["n_answerable"]}</td>'
            f'<td class="r m">{z["caught"]} &rarr; {k["caught"]} of {c["n_adversarial"]}</td>'
            f"</tr>")
    head = ('    <thead><tr><th></th>'
            '<th class="r">refused at 0.0</th><th class="r">calibrated</th>'
            '<th class="r">refused there</th>'
            '<th class="r">adversarial caught</th></tr></thead>')
    return NL.join(['  <div class="tw">', "  <table>", head, "    <tbody>",
                    *rows, "    </tbody>", "  </table>", "  </div>"])


TABLES = {"tbl_transfer": tbl_transfer, "tbl_threshold": tbl_threshold}


# ---------------------------------------------------------------------------
FIGURES = {
    "bm25": (fig_bm25, "06", "What the lexical half is worth, per corpus",
             "On the corpus the system was tuned on, adding BM25 is the single largest "
             "gain in the pipeline. On both corpora built afterwards it <b>costs</b> "
             "pool coverage. One reversal is noise; two pointing the same way is a "
             "pattern, and the plausible mechanism is that questions about birds and "
             "the prose answering them share ordinary vocabulary, so BM25 promotes "
             "passages that merely repeat common words. Hybrid search is a default "
             "worth re-testing per corpus rather than assuming."),
    "ladder": (fig_ladder, "07", "Each mechanism, measured separately",
               "The cross-encoder is the largest single gain in the system and the "
               "most expensive thing in it, which is the trade the architecture is "
               "built around: retrieve cheaply and widely, then re-read a shortlist "
               "properly. Note that the last row is worse than the one above it — the "
               "tighter cap is not shipped."),
    "cap": (fig_cap, "08", "The diversity cap is a trade, not a free win",
            "Capping each document at two passages costs a little ranking and buys "
            "more of the documents that answer the question. Which side of that trade "
            "is right depends on whether you want the best passage or the fullest "
            "answer; this system chose coverage, and the figure is here so the cost is "
            "visible rather than buried."),
    "thresholds": (fig_thresholds, "09", "Why one refusal threshold cannot serve three corpora",
                   "Each corpus was measured separately, and the gap between the two "
                   "populations lands somewhere different every time. The shipped 0.0 "
                   "refuses 1 of 66 answerable questions on the ML papers, 10 of 26 on "
                   "the bird corpus and 19 of 35 on the finance one. Moving each to its "
                   "own cut point recovers those answers while catching <b>exactly as "
                   "many</b> adversarial questions — strictly dominant, and invisible "
                   "until the corpora were scored apart."),
    "divisor": (fig_divisor, "10", "The measurement artefact under source recall",
                "Source recall is normalised by min(|gold|, 5). The finance corpus "
                "averages 4.14 gold documents per question against the ML corpus's "
                "2.67, so it is scored against a divisor 42% larger and must return "
                "more documents in the same five slots to earn the same number. Its "
                "lower score is substantially a property of its labels, not of the "
                "retriever — which is the sort of thing that gets written up as a "
                "regression if nobody checks."),
    "transfer": (fig_transfer, "11", "The same pipeline on three corpora",
                 "Ranking degrades most: encyclopaedia prose states a fact once, in "
                 "ordinary words, so the right passage is found but sits lower. The "
                 "finance corpus is weakest on every measure, and the figure above "
                 "explains part of why before anyone concludes retrieval is broken on "
                 "it."),
}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not DATA.exists():
        raise SystemExit("eval/analytics.json missing -- run src/build_analytics.py")
    C = json.loads(DATA.read_text(encoding="utf-8"))["corpora"]
    html = GUIDE.read_text(encoding="utf-8")

    made = 0

    def inject(name, block):
        """Replace the content between this block's anchors, or say it is missing."""
        nonlocal html
        wrapped = NL.join([f"<!-- GEN:{name} -->", block, f"<!-- /GEN:{name} -->"])
        pat = re.compile(rf"<!-- GEN:{name} -->.*?<!-- /GEN:{name} -->", re.S)
        if not pat.search(html):
            print(f"  {name}: NO ANCHOR -- add <!-- GEN:{name} --><!-- /GEN:{name} -->")
            return False
        html = pat.sub(lambda _: wrapped, html)
        print(f"  {name}: replaced")
        return True

    for name, (fn, num, title, caption) in FIGURES.items():
        made += inject(name, figure(num, title, fn(C), caption))
    for name, fn in TABLES.items():
        inject(name, fn(C))

    # Figures are numbered by where they end up, not by the order this file
    # happens to define them -- a generated figure inserted into chapter 3 sits
    # ahead of a hand-drawn one in chapter 5, and a reader following "see FIG. 07"
    # has no idea about generation order.
    n = 0

    def bump(m):
        nonlocal n
        n += 1
        return f'{m.group(1)}FIG. {n:02d} '

    html, _ = re.subn(r'(<div class="fignum">)FIG\. \d+ ', bump, html)
    print(f"  renumbered {n} figures in document order")

    GUIDE.write_text(html, encoding="utf-8")
    print(f"{made}/{len(FIGURES)} figures written into {GUIDE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
