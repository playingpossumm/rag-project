"""Command-line front end: readable by default, JSON when piped somewhere.

Third and last shell over `api.ask` -- library, HTTP, and CLI all call the same
function, so behaviour cannot drift between them.

    python src/cli.py "What is late interaction?"
    python src/cli.py "..." --json | jq '.passages[0].source'
"""
import argparse
import json
import sys

from api import DEFAULT_EXPANSION, ask
from loaders import locator_label

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def render(answer, show_text: bool, width: int = 400) -> None:
    print(f"\n{answer.question}")
    print("=" * min(len(answer.question), 78))

    if not answer.confident:
        print("\n! low confidence -- treat these as closest matches, not an answer")

    for note in answer.notes:
        print(f"\n[note] {note}")

    if answer.answer:
        print(f"\n{answer.answer}\n")

    print(f"\n{len(answer.passages)} passage(s) from {len(answer.documents)} document(s):\n")
    for i, p in enumerate(answer.passages, start=1):
        print(f"[{i}] {p.source} -- {locator_label(p.locator)}   (score {p.score:+.2f})")
        if show_text:
            body = " ".join(p.text.split())
            print(f"    {body[:width]}{'...' if len(body) > width else ''}")
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("question", nargs="+")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("-k", type=int, default=5, help="passages to return")
    # The default is api.DEFAULT_EXPANSION, read rather than restated. Until
    # 2026-09-06 this said "page" while the library shipped "window", so the
    # CLI and the library answered the same question with different context.
    ap.add_argument("--expansion", default=DEFAULT_EXPANSION,
                    choices=["page", "window", "none"],
                    help=f"context around each passage (default {DEFAULT_EXPANSION})")
    ap.add_argument("--min-confidence", type=float, default=-2.0)
    ap.add_argument("--quiet", action="store_true", help="citations only, no passage text")
    ap.add_argument("--generate", action="store_true",
                    help="synthesize a prose answer (requires an API key; costs money)")
    args = ap.parse_args()

    answer = ask(" ".join(args.question), k=args.k, expansion=args.expansion,
                 min_confidence=args.min_confidence, generate=args.generate)

    if args.json:
        print(json.dumps(answer.to_dict(), ensure_ascii=False, indent=2))
    else:
        render(answer, show_text=not args.quiet)


if __name__ == "__main__":
    main()
