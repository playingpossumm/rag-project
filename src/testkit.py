"""The check-and-report harness the test_*.py suites share.

Every suite in src/ has the same shape: a list of (name, got, want) checks,
a loop that prints the failures, and a last line reading "N/N <label> checks
passed". `check_docs.py --tests` runs each suite and reads the count off that
last line, so the format is a contract and not a habit. Until 2026-09-06 each
suite carried its own copy of the loop, 17 of them, and two had already
drifted: `test_ocr.py` counts with a pair of integers and `test_trace.py`
prints "checks passed" with no label, which `check_docs` cannot attribute to a
suite.

    from testkit import Suite

    suite = Suite("freshness")
    suite.check("a current chain reports no problems", problems, "")
    raise SystemExit(suite.report())

`report()` prints one FAIL line per failed check, then the count line, and
returns 1 when anything failed so the process exit code carries the verdict.
The label is the word or words that appear in HANDOFF.md's test-count
sentence, which is how `check_docs` matches a suite to its stated count.
"""
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class Suite:
    """One suite's checks, recorded as they run and reported at the end."""

    def __init__(self, label: str):
        self.label = label
        self.checks: list[tuple[str, object, object, bool]] = []

    def check(self, name: str, got, want) -> bool:
        ok = got == want
        self.checks.append((name, got, want, ok))
        return ok

    @property
    def failed(self) -> list[tuple[str, object, object, bool]]:
        return [c for c in self.checks if not c[3]]

    def report(self) -> int:
        """Print the failures and the count line; 0 if every check passed.

        The count line is the last thing printed and has the shape
        "N/N <label> checks passed", because that is what check_docs parses.
        """
        width = max((len(name) for name, *_ in self.checks), default=0)
        for name, got, want, _ in self.failed:
            print(f"FAIL  {name:<{width}}  got {got!r}, want {want!r}")
        n_failed = len(self.failed)
        print(f"{len(self.checks) - n_failed}/{len(self.checks)} "
              f"{self.label} checks passed")
        return 1 if n_failed else 0
