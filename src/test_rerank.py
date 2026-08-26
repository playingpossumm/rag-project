"""The score cache must not serve one model's scores under another's name.

`_score_cache` was keyed on `(query, chunk text)` while its own comment said the
score is a pure function of "(query, chunk text, model)". Nothing had ever
swapped cross-encoders inside one process, so nothing noticed. The first thing
that would have -- a reranker comparison -- would have reported every candidate
model as scoring exactly like whichever one loaded first, which is not an error
anyone would question: it looks like a null result.

Stubbed models, not real ones. A real pair would take half a minute to load and
would prove the same thing less precisely, since two real cross-encoders differ
for reasons other than the cache. These stubs differ ONLY by name, so a
disagreement can only come from the key.

    .venv\\Scripts\\python.exe src\\test_rerank.py
"""
import sys

import rerank as rr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHECKS: list[tuple[str, object, object, bool]] = []


def check(name: str, got, want) -> None:
    CHECKS.append((name, got, want, got == want))


class StubEncoder:
    """Scores every pair as a constant, so the model is the only variable."""

    def __init__(self, value: float):
        self.value = value
        self.calls = 0

    def predict(self, pairs):
        self.calls += len(pairs)
        return [self.value] * len(pairs)


STUBS = {"model-a": StubEncoder(1.0), "model-b": StubEncoder(-1.0)}


def install_stubs() -> None:
    rr._model, rr._model_name = None, None
    rr.clear_cache()
    rr.CrossEncoder = lambda name, *a, **kw: STUBS[name]


def candidates() -> list[dict]:
    return [{"text": "a passage about feathers", "source": "birds.pdf",
             "locator": {"kind": "page", "value": 1}, "score": 0.5},
            {"text": "a passage about flight", "source": "birds.pdf",
             "locator": {"kind": "page", "value": 2}, "score": 0.4}]


real_encoder = rr.CrossEncoder
install_stubs()

first = rr.rerank("what is a syrinx", candidates(), k=2, model_name="model-a")
second = rr.rerank("what is a syrinx", candidates(), k=2, model_name="model-b")
again = rr.rerank("what is a syrinx", candidates(), k=2, model_name="model-a")

# The check the old key could not pass: same query, same text, different model.
check("a second model is not served the first model's scores",
      second[0]["rerank_score"], -1.0)
check("the first model still scores what it scored",
      first[0]["rerank_score"], 1.0)
check("returning to the first model reuses its cached scores, not the second's",
      again[0]["rerank_score"], 1.0)

# ...and it is genuinely cached, rather than correct because nothing is reused.
check("the first model was asked once for its two pairs, not twice",
      STUBS["model-a"].calls, 2)
check("the second model was asked for its own two pairs",
      STUBS["model-b"].calls, 2)

# Swapping back must swap the loaded model too, not just the key.
check("load_reranker returns the model that was asked for",
      (rr.load_reranker("model-b") is STUBS["model-b"],
       rr.load_reranker("model-a") is STUBS["model-a"]),
      (True, True))

# The blend path reads the cache through a second code path -- the blended
# ordering is chosen by rank, but the score reported is still the raw logit, and
# that lookup was the other place the old key appeared.
install_stubs()
blended = rr.rerank("what is a syrinx", candidates(), k=2,
                    model_name="model-b", blend=0.2)
check("the blended path reports the requested model's score",
      blended[0]["rerank_score"], -1.0)

rr.CrossEncoder = real_encoder
rr._model, rr._model_name = None, None
rr.clear_cache()


def main() -> int:
    width = max(len(n) for n, *_ in CHECKS)
    failed = 0
    for name, got, want, ok in CHECKS:
        if not ok:
            failed += 1
            print(f"FAIL  {name:<{width}}  got {got!r}, want {want!r}")
    print(f"{len(CHECKS) - failed}/{len(CHECKS)} reranker cache checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
