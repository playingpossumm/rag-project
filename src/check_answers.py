"""Verify candidate answer strings actually occur in the parsed corpus.

Answer strings are only useful as a metric if they match the text as parsed --
not as the PDF renders it. Markdown artifacts and maths notation mean the two
often differ, so every string is checked before it enters the golden set.

The candidates below are answer strings for the Transformer paper, and the
pages are keyed by page number across every document in the index, so the
"also on N other page(s)" count includes every other document.

    .venv\\Scripts\\python.exe src\\check_answers.py
"""
import argparse
import json
import sys
from pathlib import Path

# The same whitespace-collapsing rule the harness scores with, imported rather
# than copied. Five scripts carried their own copy until 2026-09-06.
from evaluate import normalize as norm  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STORE = Path(__file__).parent.parent / "vector_store" / "metadata.json"

# id -> (expected page, distinctive substring that constitutes the answer)
CANDIDATES = {
    "mha-def":      (5, "jointly attend to information from different representation subspaces"),
    "mha-heads":    (5, "employ _h_ = 8 parallel attention layers"),
    "scale-why":    (4, "extremely small gradients"),
    "optimizer":    (7, "Adam optimizer"),
    "lr-schedule":  (7, "warmup_steps"),
    "enc-layers":   (3, "stack of _N_ = 6 identical layers"),
    "d-model":      (3, "dmodel = 512"),
    "dropout":      (8, "_Pdrop_ = 0.1"),
    "complexity":   (6, "O(n[2]"),
    "path-length":  (6, "Maximum Path Length"),
    "pos-enc-fn":   (6, "where pos is the position"),
    "pos-enc-why":  (6, "no recurrence and no convolution"),
    "bleu-ende":    (8, "28.4"),
    "bleu-enfr":    (8, "41.0"),
    "hardware":     (7, "8 NVIDIA P100 GPUs"),
    "train-time":   (7, "12 hours"),
    "dataset":      (7, "WMT 2014 English-German"),
    "label-smooth": (8, "label smoothing of value"),
    "parsing":      (10, "WSJ 23 F1"),
    "ffn-act":      (5, "ReLU"),
    "ffn-dim":      (5, "2048"),
    "attn-uses":    (5, "three different ways"),
    "compound":     (9, "4.92 25.8 65"),
    "residual":     (3, "residual connection"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    chunks = json.loads(STORE.read_text(encoding="utf-8"))
    by_page = {}
    for c in chunks:
        # A chunk carried a `page` field until 2026-08-18, when the Office
        # loaders generalised it to a `locator` of {kind, value}. This script
        # went on reading `page` and raised KeyError on every run from that
        # date until 2026-09-07. Chunks located by something other than a
        # page (a slide, a sheet) have no page to check against and are
        # skipped.
        loc = c.get("locator") or {"kind": "page", "value": c.get("page")}
        if loc.get("kind", "page") != "page":
            continue
        by_page.setdefault(loc.get("value"), []).append(c["text"])
    # Chunk text preserves Markdown line breaks, so a phrase spanning a newline
    # will not match a single-spaced needle. Collapse whitespace on both sides.
    by_page = {p: norm(" ".join(t)) for p, t in by_page.items()}

    ok, bad = 0, []
    for case_id, (page, needle) in CANDIDATES.items():
        page_text = by_page.get(page, "")
        needle_n = norm(needle)
        found_here = needle_n in page_text

        # Also report how many OTHER pages contain it -- a string appearing
        # everywhere is a weak answer marker even if it is technically correct.
        elsewhere = sum(
            1 for p, text in by_page.items()
            if p != page and needle_n in text
        )

        status = "ok " if found_here else "MISS"
        if found_here:
            ok += 1
        else:
            bad.append(case_id)
        note = f"  (also on {elsewhere} other page(s))" if elsewhere else ""
        print(f"{status} {case_id:<14} p{page:<3} {needle!r}{note}")

    print(f"\n{ok}/{len(CANDIDATES)} verified")
    if bad:
        print("needs a different string:", ", ".join(bad))


if __name__ == "__main__":
    main()
