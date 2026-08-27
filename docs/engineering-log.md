# Engineering log

Every attempt, including the ones that were wrong. Started 2026-08-26.

`HANDOFF.md` records where the project **is**; this records how it got there,
and specifically what was tried and refuted. The project's own §4 argues that a
measurement overturning a plan is worth more than a feature that ships — this is
where those measurements live in full, rather than compressed to one row of a
table.

Rules for entries: state the hypothesis before the result, give the number, and
say what was done about it. An entry whose outcome is "no change" is worth as
much as one that ships something, and is more likely to be forgotten.

---

## 2026-08-26 — A freshness check for the chain the front page reads

**Why.** Generated files had gone stale silently three times. The worst was
`per_case.json` sitting four days out of date, which meant the front page was
offering questions that had already been deleted from the golden set as
unanswerable — the system inviting people to ask it things it had itself
concluded it could not answer. Only `RESULTS.md` had a `--check`.

**Built.** `src/check_freshness.py`, covering
`golden set → per_case → analytics → the front page`, two ways:

| kind | what it catches | what it misses |
|---|---|---|
| provenance — a digest of every input recorded in the output | an edit that changes no count, e.g. rewording one question in place | anything about a file written before provenance existed |
| semantics — case ids, question strings, corpus sizes, threshold, blend | the front page offering a deleted question; a corpus that grew | an edit that leaves all of those identical |

Neither alone is enough, which is why both are there. Exit code is the contract;
`serve.py` warns on startup without blocking; `src/test_freshness.py` stages each
known failure on a synthetic corpus and asserts it is reported (30 checks).

**What it found on its first run against the real repo.** The trap the project
names first, in the two scripts that produce every published number.
`per_case.py` imported `ABSTAIN_THRESHOLD` — the ML papers' `0.0` — and scored
every corpus against it, and never set a rerank blend at all. `evaluate.py` did
the same with the threshold.

| | claimed | actually served |
|---|---|---|
| birds, answerable wrongly refused | 10 of 26 | **4 of 26** |
| quant, answerable wrongly refused | 11 of 35 | **3 of 35** |
| birds, per-case any-hit | 0.808 | **0.846** |
| birds, per-case MRR | 0.570 | **0.614** |

Both scripts now resolve threshold, blend *and index* from `corpora.json` via the
golden set. Re-running left every retrieval figure in `results*.json` identical
to three decimals — the evidence that the change measured nothing new, only
corrected which pipeline was being described.

**Smaller things found on the way.**

- `answerable_median` in `evaluate.py` was `sorted(scores)[len(scores) // 2]` —
  the upper middle value, not the median. On the 26 even-sized bird cases it
  reported +1.21 where `analytics.json`, using `statistics.median`, said +1.15.
  One quantity, two documents, two values.
- `build_analytics.py` mapped corpora to eval files with a hardcoded table. A
  corpus added to `corpora.json` and not to that table was skipped in silence,
  and the front page then offered it the ML papers' fallback questions.
- README said the bird corpus holds 900 passages. It holds 864.

---

## 2026-08-26 — The bird gate was calibrated against a pipeline nobody runs

**Hypothesis.** With the confidences re-measured on the served pipeline, the
bird threshold `-3.0` might no longer be right — it was derived at rerank blend
0.00 on a corpus that ships 0.20.

**Measured.** It was dominated. Every threshold in `(-6.48, -4.55]` catches the
same five of six adversarial cases and wrongly refuses **three** of twenty-six
rather than four.

| cut | answerable lost | adversarial caught |
|---|---|---|
| −3.0 *(was)* | 4 | 5 |
| **−5.5 *(now)*** | **3** | 5 |
| −7.0 | 3 | 4 |

**Shipped −5.5**, the middle of the interval — 0.95 of margin before it refuses
an answerable question, 0.98 before it stops catching an adversarial one. An
edge would fit the threshold to a single case. Retrieval untouched: any-hit
0.846, MRR 0.614.

**Checked on all three corpora before shipping**, because the trap is tuning to
the one that prompted the question. ML's `0.0` and quant's `-4.0` are already on
the frontier — every lowering costs catches. Only birds was dominated.

**Tooling flaw this exposed.** `calibrate_threshold.py` had its grid hardcoded to
`[-4 .. +4]`, which is where the ML papers' scores live and nowhere near the bird
corpus's — every question deciding that corpus's threshold sits below −4. The
tool used to calibrate a corpus could not display the region being calibrated.
The grid is derived from the observed scores now, and it reports the *interval* a
threshold sits in rather than a grid point, because nothing changes until a cut
point crosses an actual score.

---

## 2026-08-27 — The score cache claimed to be keyed on the model and was not

**Found while preparing to compare rerankers.** `rerank._score_cache` was keyed
on `(query, chunk text)`. The comment directly above it had always read "scores
are a pure function of (query, chunk text, model)".

Nothing had ever swapped cross-encoders inside one process, so nothing caught
it. The first thing that would have is a reranker comparison — and it would have
reported every candidate model as scoring **exactly** like whichever loaded
first. That is not an error anyone questions; it looks like a null result.

Fixed by keying on the model name and taking the name as the argument rather
than an already-constructed model object, whose identity a cache cannot read.
`src/test_rerank.py` proves it with two stub encoders differing only by name —
7 checks, and verified to fail 2 when the old two-part key is put back, with the
telling failure `got 1.0, want -1.0`: the second model served the first's score
and was never even asked.

---

## 2026-08-27 — A better cross-encoder: what the measurement actually said

**Hypothesis.** `ms-marco-MiniLM-L-6-v2` is weak on questions that *describe* a
term rather than naming it — "the burst of collective singing at first light" —
and the bird corpus has the headroom to show it: the candidate pool holds the
answer 96.2% of the time and the pipeline returns it 84.6%.

**First, the harness had to be rebuilt.** `compare_rerankers.py` was ML-only: it
hardcoded a list of failing ML case ids, called `load_index()` with no store, set
no rerank blend, and measured a single number (top-1 from an answering
document). It could not run on the corpus with the headroom.

Rewritten to measure **two failures, not one**, because reranking fails in two
ways and only one of them shows up in any-hit:

- **ordering** — any-hit, MRR, NDCG, source recall
- **scoring** — the gate reads the cross-encoder's score of the top passage, and
  on the bird corpus three of the seven failures are questions whose answer the
  pipeline *found* and then refused to show

Scores from two models are on different scales, so holding a threshold fixed
across models measures the scale, not the separation. The gate half is therefore
reported threshold-free as **AUC** — P(a random answerable outscores a random
adversarial) — plus `unreachable`, the count of answerable questions scoring
below the third-highest adversarial, which no threshold can save.

**Result on the bird corpus (26 answerable + 6 adversarial):**

| model | any-hit | MRR | AUC | unreachable |
|---|---|---|---|---|
| MiniLM-L6 *(shipped)* | **0.846** | **0.614** | 0.859 | 2 |
| MiniLM-L12 | 0.808 | 0.594 | 0.878 | 1 |
| BGE-reranker-base (278M) | 0.769 | 0.596 | **0.968** | **0** |

**The bigger models rank worse and separate better.** That was not the expected
shape at all, and it is the finding: the two halves of the reranker's job move in
opposite directions as the model grows. A straight swap is not available —
BGE would trade two ranking failures for a perfect gate.

**A latency number that was wrong, and how it was caught.** The first run
reported BGE at 1,576,300 ms/query — 1700× the shipped model, against the
project's own earlier measurement of 10 s/query on the ML papers. A 150×
disagreement with a prior measurement is a reason to distrust the new one. Timed
again with nothing else running: **244 ms/pair against 25 ms/pair, 9.8×**. The
first figure measured a contended machine — several evaluation runs, a server
and a browser were competing — not a model. Quality metrics were unaffected,
being deterministic.

**Hypothesis that followed, and was refuted.** Ranking needs a score for twenty
candidates; the gate needs one. So let the expensive model do only the half it
wins at — rank with L6, gate with BGE, at a twentieth of BGE's cost.

| configuration | any-hit | AUC | unreachable |
|---|---|---|---|
| L6 *(shipped)* | 0.846 | 0.859 | 2 |
| L6 rank + **BGE** gate | 0.846 | **0.827** | 3 |
| L6 rank + **L12** gate | 0.846 | **0.878** | 1 |

**BGE as a gate alone is worse than the shipped model**, despite scoring 0.968
when it also did the ranking. Its separation was partly self-consistency: a model
scores its own pick confidently, and handed someone else's pick it separates
worse than MiniLM does. The AUC of a model ranking-and-gating is not the AUC of
that model gating. Assuming those are the same quantity is the mistake the split
row was written to test, and it was a real one.

**The verdict, on all three corpora.** Nothing here is shippable, and the
reason is the finding.

| against the shipped MiniLM-L6 | ML papers | Ornithology | Quant |
|---|---|---|---|
| **L12 as gate only** — AUC | −0.020 | **+0.019** | −0.009 |
| — unreachable | 9 → 18 | **2 → 1** | 2 → 3 |
| **L12 as full reranker** — any-hit | **+0.015** | −0.038 | +0.000 |
| — MRR | +0.004 | −0.020 | **+0.025** |
| — AUC | −0.023 | **+0.019** | **+0.010** |

The gate swap helps one corpus and hurts two. The full swap trades one ML case
won for one bird case lost, improves quant's MRR, and costs 2× the latency. Both
**rejected**, and they join the two refutations already in HANDOFF §4.

**What the measurement actually established**, which is worth more than the swap
would have been:

1. **Ranking quality and gate separation are not the same axis, and across
   models they move in opposite directions.** L12 ranks better than L6 on the ML
   papers and worse on birds; it separates worse on ML and better on birds.
   There is no ordering of these models by "better".
2. **A model's separation while it ranks is not its separation while it gates.**
   BGE scored AUC 0.968 choosing and scoring its own top passage, and 0.827
   scoring MiniLM's. Part of that 0.968 was self-consistency — a model is
   confident about its own pick. Anyone comparing rerankers on AUC alone would
   have read 0.968 as a reason to swap.
3. **The "it does not transfer" law now covers the model too**, alongside the
   abstention threshold and the rerank blend. That is three independent
   settings, measured separately, all corpus-specific. It is the strongest form
   of the project's own thesis and it was not assumed — each one was found by
   shipping the opposite first.

**So the next move on this stage is not a bigger cross-encoder.** The failure is
specific — questions that describe a term rather than naming it — and a
cross-encoder of any size reads the same words. The fix that addresses the
mechanism is query decomposition, which needs an LLM, which needs credit. That
is now measured rather than asserted.

---

## 2026-08-27 — Fusion per corpus: the pool says one thing, the pipeline another

**Hypothesis.** The reranker turned out to be a dead end, but the stage before it
looked promising. `HANDOFF.md` had said since the second corpus was built that
fusion is probably per-corpus, on the strength of one number: the bird candidate
pool holds the answer 96.2% of the time under dense retrieval and 88.5% under
RRF, so fusing costs that corpus two questions before reranking starts. It was
never acted on because 26 cases is too few to move a *global* default — sound
about a global default, silent about a per-corpus one, which is what the
threshold and the blend already are.

**What was missing was the measurement at the served configuration.**
`evaluate.py` compares fusions *before* the diversity cap and applies the cap
only to RRF, so the table every document quotes has no row for "weighted,
capped" at all. `src/sweep_fusion.py` runs every fusion on every corpus at
k=5 from 20 candidates, with that corpus's own rerank blend and the 2-per-source
cap — the pipeline `api.ask()` actually runs.

**The result reverses the premise.**

| birds | pool any-hit | shipped-pipeline any-hit |
|---|---|---|
| dense only | **0.962** | **0.808** |
| RRF *(shipped)* | 0.885 | **0.846** |
| weighted a=0.5 | 0.923 | **0.885** |

Dense retrieval finds the answer most often and produces the *worst* final
result. A candidate pool is not a set, it is an **ordering handed to the
cross-encoder**, and dense hands over one the reranker cannot exploit — the same
weakness the rerank blend exists to hedge. Reading pool recall as a proxy for
pipeline quality is the error, and this project's own handoff had been making it
for a week.

**Nothing shipped, and both reasons are stated rather than assumed.**

- birds, `weighted a=0.5`: wins one question (`bird-dialects`) and loses MRR
  0.614 → 0.587 and NDCG 0.671 → 0.662. Identical in shape to the trade the
  rerank blend was judged on, and this corpus's own note already settles how to
  read it — on 26 cases, an any-hit gain of one question is thinner evidence
  than MRR. **Rejected by the project's own stated principle**, which is the
  best kind of rejection: the rule existed before the result.
- quant, `weighted a=0.7`: weakly dominant — MRR 0.714 → 0.727, NDCG +0.006,
  source recall +0.006, any-hit unchanged. But the missed set is *identical*, so
  no question changes hands; it buys a fraction of a rank position. Shipping it
  would also require recalibrating that corpus's threshold, because changing
  fusion changes the top passage and therefore every confidence the gate reads.
  Too much machinery for +0.013 MRR on 35 cases. **Rejected on cost.**
- ML: RRF wins outright. No candidate.

The value here is the corrected claim, not a config change. HANDOFF §2 now
carries the end-to-end table and the instruction to quote it rather than the
pool row.

---

## 2026-08-27 — The answer highlight: moved, bounded, and measured

**Why.** The interface sets the answering words bold inside a passage shown at
normal weight. That logic lived inline in a two-thousand-line HTML file, so
nothing could run it without a browser, and nothing ever had. Every claim about
it — "only the answering words", "at most a sentence" — described code no test
had executed.

**Moved** to `ui/answer-mark.js` (with `clean` and `fragment`, the page's own
text tidying, so a test sees exactly the string the page marks), served at
`/answer-mark.js`, and given a contract in `ui/test-answer-mark.mjs`:

- at most one sentence is ever marked, and never the whole passage
- the mark is a contiguous run of at most `MAX_MARK_WORDS` (12)
- a mark carries two distinct question words, or one and a figure, or there is
  no mark
- the passage survives marking character for character

**Two real defects, both found by measuring rather than reading.**

*An unbounded path.* When the chosen sentence contained no question word,
`answerSpan` marked the **entire sentence** — the one case with the least
justification for marking anything. Now it marks nothing.

*A word bound is not a length bound.* Swept over the real top passage for every
question in all three golden sets — 450 passages, 157 questions — one mark
covered **82.7%** of its passage while satisfying "at most twelve words". The
passage was a bibliography entry and the mark was a markdown link: four words,
359 characters. Added `MAX_MARK_CHARS` (180) and applied it to the
short-sentence early return, which had been bypassing every bound below it.

| across 450 real passages | before | after |
|---|---|---|
| median share of passage set bold | 3.6% | 3.6% |
| 90th percentile | 6.2% | 6.2% |
| **maximum** | **82.3%** | **38.8%** |
| marks over the bound | 1 | 0 |
| marks spanning two sentences | 0 | 0 |

The median did not move, which is the point: the fix touched the tail and left
the ordinary case alone.

**One more, found while writing the test.** `splitSentences` treated
"Vaswani et al. (2017) introduced…" as two sentences — terminator, whitespace,
capital. A false boundary is the mirror image of a missed one and much harder to
notice: it does not run the mark into the next sentence, it cuts the answer in
half. It went unseen on a corpus of academic papers, which is exactly where
those abbreviations live.

---

## 2026-08-27 — Stress-testing the golden sets

**Why.** Every number in this repo rests on three golden sets, and nothing had
ever checked that they still *describe* the corpora they score. `evaluate.py`
will happily score a case whose gold document was renamed, whose locators moved
when the parser changed, or whose `answer_contains` string is nowhere near the
passage the label names. It produces a number either way, and that number is
what the documents quote.

`src/audit_golden_set.py` already asked the *semantic* question — has the corpus
grown into an adversarial case's subject, has an answerable question become
ambiguous — but it is ML-only, with a hand-written table of which words would
make each adversarial case answerable. `src/check_golden.py` asks the
*structural* one and needs no per-case knowledge, so it runs on every corpus
including ones that do not exist yet.

**It found a broken label on its first run.** `bird-hollow-bones` carried
`{"kind": "section", "pages": [1, "Skeletal system"]}`. "Skeletal system" is a
real section; the `1` was meant for `table 1`, a different locator kind in the
same document. As written, `(bird_anatomy.docx, section, "1")` matched no chunk
in the corpus and contributed nothing, while looking exactly like a label.

**And it did not change the score, which is worth saying plainly.** The pipeline
returns none of the three places carrying "hollow bones" — it returns slide 4,
`section Axial skeleton`, `section Overview`, slide 2 — so the case misses
either way. A broken label that happens to sit on a genuine miss. Fixing it left
any-hit at 0.846 and MRR at 0.614, and the temptation to present the fix as a
recovered question is exactly the kind of thing this log exists to prevent.

`src/test_golden.py` stages each failure class on a synthetic two-document
corpus and asserts the audit reports it — 15 checks, hermetic, milliseconds.

**A gap in my own work, found by using it.** Fixing that one locator changed no
case count, no corpus size and no metric — and `check_freshness` still reported
`results-birds.json` as current. `per_case.json` had recorded a digest of its
golden set since the day the check was written; `results.json` never had. The
digest half exists precisely for the edit that moves no count, and it was
missing from half the chain. Now stamped by `evaluate.py` and checked, with two
more cases in `test_freshness.py` (30 → 32).

---

## 2026-08-27 — The documents are now checked against the measurements

**The failure this closes** is this repo's most-repeated one. The README once
claimed "84 evaluation cases" directly above figures measured on 23.
`eval/RESULTS.md` said 20 papers and 2,768 chunks long after the corpus reached
36 and 5,459 — under a header promising that if the two disagreed, the document
was stale. It was true, and nobody noticed, because nothing checked it.

`build_results_doc.py --check` closed that for RESULTS.md by *generating* its
tables. `check_freshness.py` closed the generated chain feeding the interface.
Neither covered **HANDOFF.md §2 and the README** — the two documents a new
session and a visitor read first.

Generating them would be the wrong fix: the argument around each number is
judgement and cannot come from a JSON file. So `src/check_docs.py` leaves them
hand-written and checks them — 14 quantities, each naming where its truth lives,
and a table it cannot find is a *failure* rather than a skip, because a checker
that quietly matches nothing reports success.

**Verified to have teeth** by perturbing two numbers and confirming both were
caught, one of them the exact `900 passages` error that was really in the README
earlier the same day.

One subtlety that would have made it useless: compare at the precision the
document *writes*, not the precision the measurement carries. `+4.93` against
`4.934403419494629` is a document rounding correctly, and reporting that as
drift trains a reader to ignore the checker — which is how a check stops being
read.

---

## 2026-08-27 — Optimising: measure first, and the measurement is unambiguous

**The rule this project already learned.** "Re-indexing is slow because the
index is rebuilt" was a well-formed plan until rebuilding the FAISS index
measured at 0.01 s — 0% of runtime — and the task had to be redefined. So
`src/profile_query.py` times each stage separately before anything is touched,
warm, on real questions from each corpus's own golden set.

| stage | ML papers | Ornithology | Quant |
|---|---|---|---|
| embed | 24.7 ms | 25.2 | 24.0 |
| faiss | 21.3 | 19.5 | 20.4 |
| bm25 | 36.5 | 5.5 | 36.0 |
| fuse | 0.1 | 0.1 | 0.1 |
| **rerank** | **1017.0** | **966.5** | **988.2** |
| diversify | 0.0 | 0.0 | 0.0 |
| expand | 8.9 | 2.3 | 9.7 |
| total | 1108.5 | 1019.1 | 1078.4 |

**Reranking is 92–95% of a query.** Everything else together is about 80 ms.
There is exactly one stage worth optimising, and the interface's single
"search, scoring and reranking — 1164 ms" had been hiding which one.

**Two ways to make it cheaper, both measured.**

*Faster execution, same work.* Threads are already at the machine's best — torch
defaults to 10 of 12 cores, and 4, 2 and 1 are all slower. Padding waste inside
the batch is 3% on ML and 17% on birds, so length bucketing could buy at most a
sixth of one stage on one corpus in exchange for reordering logic in the hot
path. A faster *model* was already ruled out: the quicker candidates are worse
and the one that separates best is ten times slower. **Nothing to take here.**

*Less work.* `CANDIDATE_K` is 20 and cost is linear in it. Its value was chosen
for recall headroom with no latency term, because nothing had measured the
latency. `src/sweep_candidates.py` measures quality and milliseconds together.

**The result, against the shipped 20:**

| candidates | ML any-hit / MRR | birds | quant | rerank time |
|---|---|---|---|---|
| 12 | −3.0 / −1.1 | −3.8 / −3.2 | −2.9 / +0.2 | −36 to −42% |
| **16** | **+0.0 / +0.1** | **+0.0 / −2.6** | **+0.0 / +2.9** | **−21 to −25%** |
| 20 *(shipped)* | — | — | — | — |
| 28 | −1.5 / −1.4 | +0.0 / −2.4 | +0.0 / −0.6 | +42 to +49% |

**28 candidates is worse than 20 on all three corpora, while the pool ceiling
rises on all three** — 0.925 → 0.955 on ML, 0.885 → 0.962 on birds,
0.943 → 0.971 on quant. More candidates means the answer is available more
often and returned less often. **The reranker's precision degrades faster than
the first stage's recall improves**, which is the third time today the same
shape has appeared: handing this cross-encoder a better pool makes the pipeline
worse. It is the sharpest statement of "the reranker is the weakest stage" the
project has, and it came from a latency experiment.

**16 is a real option and is not shipped unilaterally.** Any-hit is *identical*
on all three corpora, MRR nets +0.4 points across them (+0.1, −2.6, +2.9), and
it removes roughly 220 ms from a 1,100 ms query. But the −2.6 on birds is a
regression on a corpus that was not the problem, and this project's rule is that
such a change is not shippable as a default. It is a trade for the owner to
make, with the numbers on the table, rather than a decision to slip in under a
latency heading.

**One thing was fixed rather than measured.** `rerank._score_cache` grows for
the life of the process and nothing outside a test had ever called
`clear_cache()`. That was free for its original caller — an evaluation run
scores a few hundred questions and exits — but `serve.py` imports the same
module and does not exit, adding `candidate_k` entries per distinct question
forever, each keyed on a tuple holding the whole chunk text. Bounded at 20,000
entries with insertion-ordered eviction, trimmed *before* the current query is
scored rather than after: the scores are written to the cache and then read back
out of it to build the result, so trimming afterwards would have turned a memory
bound into a KeyError on the serving path. Three checks in `src/test_rerank.py`
cover it, including that a query with more candidates than the whole limit still
returns all of them.

---

## 2026-08-27 — Stress-testing the adversarial half, which nobody had read

**Why.** Every discussion of the abstention gate quotes "catches 9 of 17" on the
ML corpus. Nobody had read the eight it does not catch. The obvious hypothesis
is that the corpus has grown into them — it holds papers that did not exist when
those questions were written — in which case the labels are wrong and the
threshold is being blamed for a golden-set problem. That is exactly what
happened to the quant corpus, which went from 0.543 to 0.886 on a rewritten
golden set with retrieval untouched.

**Read the passages.** Four highest-scoring, all of them:

| case | score | what the top passage actually is |
|---|---|---|
| `adv-rlhf` "how is the reward model trained for RLHF" | +4.22 | a driving orchestrator whose "reward signal penalises lane deviation" |
| `adv-seed` "what random seed was used" | +4.26 | a training-details appendix listing DeepSpeed and LoRA settings, no seed |
| `adv-context-window` "maximum context length" | +2.90 | a candidate grid and BF16 weights |
| `adv-quantize-4bit` "how is the model quantized to 4-bit" | +2.26 | memory footprint and thread scaling |

**All four are correctly labelled.** Every passage is topically adjacent and
answers nothing — "reward", "training details", "context", "memory" are doing
the work. So the eight are **genuine gate failures, not mislabelled cases**, and
the hypothesis that motivated the check is refuted. Worth recording precisely
because the plausible story was the wrong one and only reading the passages
settled it.

**A check that fired on everything, and therefore on nothing.**
`audit_golden_set.py` reported **67 of 67** answerable cases as "now ambiguous".
The cause:

```python
others = found_in - {case.get("source")}      # cases have gold[].source
```

A case carries `gold: [{"source": ...}]` and no top-level `source`, so this
subtracted `{None}`, removed nothing, and matched every case against its own
gold document — which is the definition of a *correct* label. Fixed to subtract
the gold sources, the real figure is **0 of 67**: every answer string appears
only in the documents its label names.

That number had been sitting in the output as a wall of false positives for as
long as the check existed, hiding a clean result. It is the same trap
`HANDOFF.md` already records once — "a measurement returning zero deserves as
much suspicion as a surprise" — with the sign flipped. 100% deserves it too.

**And the term list does not predict the gate.** Seven of the eight cases
retrieval answers anyway were not flagged by the hand-written subject-term
table, which the script's own output already says: "...and what retrieval says,
which is the check that matters." A hand-maintained list of what *would* make a
case answerable is a guess; running the retriever is a measurement.

---

## 2026-08-27 — The documented defaults are now checked against the code

HANDOFF §3 lists eleven constants under "Defaults, all justified by measurement
in eval/RESULTS.md". That opening makes each of them a claim about the code, and
nothing checked it — change `TOP_K` in `retrieve.py` and the document goes on
describing a system that no longer exists, in a place nobody thinks to look
because it reads as prose rather than as a number.

`check_docs.py` now imports the module that owns each constant and compares.
Twenty-six quantities in total across §2, §3 and the README. Verified to have
teeth by setting `TOP_K = 7` and confirming it was caught.

It also found something the list itself hides: **`RRF_K` is defined twice**, in
`hybrid.py` and `rerank.py`. They agree at 60 today and nothing makes them —
fusion damping and the rerank blend's damping are the same constant by intention
and a copy in practice. The checker asserts the two agree and reports what would
break if they stopped: "fusion and the rerank blend would damp differently".

Four constants are environment-overridable. When the variable is set the check
says so and skips, rather than reading an overridden value and calling the
document wrong — which would be the checker lying.

**And the ambiguity question, asked of all three corpora.** With
`audit_golden_set.py`'s version corrected, the check needs no per-case knowledge,
so it moved into `check_golden.py` where every corpus gets it — as a *note*, not
a failure, since a second document carrying the answer string is a judgement
rather than a structural error. Mixing the two would make the exit code mean
"something to read" instead of "something is wrong".

Across all 157 cases in three corpora: **no ambiguity at all**. Every answer
string appears only in the documents its label names. Two tests assert the note
fires on a deliberately ambiguous synthetic case, because a check reporting zero
on real data is exactly where you have to demonstrate it can report anything
else.

---

## 2026-08-27 — The seven-case fixture is still the same seven

**Why re-measure.** HANDOFF §7 claims "18 of 66 answerable cases fail under some
configuration, but only 7 fail under all of them", and calls those seven a
fixture worth optimising against. Two reasons to distrust it today: the corpus
has **67** answerable cases, not 66, so the figure predates the golden set it
describes; and this session changed the abstention threshold, the rerank blend
resolution and one label. A "structural" classification that moves when the
config moves was never structural.

**Six `per_case.py` configurations** — default, no-rerank, dense-only, weighted
fusion, no cap, cap 1/src — fed to `src/failure_overlap.py`.

| | 2026-08-21 | 2026-08-27 |
|---|---|---|
| answerable cases | 66 | **67** |
| fail under *some* configuration | 18 | **21** |
| fail under *all* | **7** | **7** |

**And they are the same seven ids**: `adam-bias`, `dropout-rate`,
`gpt3-fewshot`, `gpt3-params`, `roberta-nsp-drop`, `t5-text2text`, `wmt14`.
Not seven again by coincidence — set-identical, with nothing entering or
leaving. The classification survived everything this session changed, which is
the strongest evidence yet that it is a property of the questions rather than of
the settings, and that scoring query decomposition against it will mean
something.

**A second reading the run gives for free.** Of the eight adversarial cases the
gate lets through on the ML corpus, **seven are answered under every gated
configuration** and only `adv-diffusion` (+0.43, the weakest of them) moves.
Those seven are the gate's own fixture, and they are exactly the cases whose
passages were read earlier today and found to be correctly labelled. Two
independent routes — reading the text, and varying the pipeline — agreeing that
these are gate failures rather than label failures.

`norerank` is excluded from that count and the tool says why: with no reranking
there is no calibrated score, so the gate never runs and all 17 adversarial
cases are answered by construction. An absent gate, not a failing one.

**One incidental number.** `nocap` has the lowest failure rate of the six at
14.9%, against the shipped 16.4% — the diversity cap costing a case, which is
the trade §2 already documents as deliberate (it buys source recall). Consistent
rather than new, and worth noting that it reproduced.

---

## 2026-08-27 — Every corpus gets a fixture, and there turn out to be two failure modes

**Why.** The structural-versus-contested split had only ever been computed for
the ML corpus, so "stress-test all the golden sets" was two-thirds unmet. Six
configurations each for birds and quant — default, no-rerank, dense-only,
weighted fusion, no cap, cap 1/src — through `src/failure_overlap.py`.

| | ML papers | Ornithology | Quant |
|---|---|---|---|
| answerable | 67 | 26 | 35 |
| fail under *some* configuration | 21 (31%) | 11 (42%) | 12 (34%) |
| **structural** — fail under all six | **7 (10.4%)** | **3 (11.5%)** | **2 (5.7%)** |
| lowest-failing configuration | nocap 14.9% | **weighted 19.2%** | default/nocap/weighted 14.3% |

The "~11% floor" HANDOFF §7 records for the ML corpus reproduces almost exactly
on birds — 10.4% against 11.5% — on a corpus of Wikipedia pages in four file
formats rather than arXiv PDFs. Quant is **5.7%**, about half, so the floor is
not universal and should not be quoted as if it were.

**Then read the twelve structural questions together, and they split in two.**

*ML — all seven ask about an attribute many papers share:*
"How does the optimizer correct for bias in its moment estimates", "what dropout
rate was applied", "how many parameters does the largest autoregressive model
have", "which translation dataset". The topic matches a dozen documents and the
clause that picks one out is ignored. This is the cross-document confusion the
project already names.

*Birds and quant — all five describe a term and ask for its name:*
"Which small group of feathers helps prevent a stall at low speed" (alula),
"which skeletal adaptation reduces a bird's weight" (hollow bones), "what term
describes chicks able to move and feed themselves soon after hatching"
(precocial), "which behaviour describes prices being pulled back toward a
long-run level" (mean reversion), "which effect describes past winners
continuing to outperform" (momentum). **The answer word never appears in the
question at all.**

**These are different problems and they want different fixes.** Query
decomposition — the planned fix, blocked on credit — splits a question into its
constraints, which is exactly right for "normalisation across features *rather
than examples*". It does nothing obvious for "which small group of feathers
prevents a stall", where nothing needs splitting and the target vocabulary is
simply absent. That is a vocabulary-mismatch problem, and `src/query_expansion.py`
already exists in the repo unmeasured against it.

So the roadmap item "the ~11% floor needs query decomposition" was **one plan for
two problems**, and it covers seven of the twelve cases rather than all of them.
Recorded rather than acted on: measuring query expansion against the five is the
next thing this suggests, and it is not blocked on credit.

**Three confirmations that came free.**

- `bird-hollow-bones`, whose broken locator was fixed earlier today, is
  **structural** — it fails under all six configurations. That independently
  confirms what was said at the time: the broken label was not masking a hit.
- `bird-dawn-chorus` and `bird-imprinting` are contested and work **only under
  `norerank`**. With reranking off there is no calibrated score and no gate, so
  "works" there means the first stage retrieved the answer and the cross-encoder's
  score is what refused it. Third independent route to the same conclusion.
- `weighted` fusion has the lowest failure rate on birds (19.2% against the
  shipped 23.1%), matching `sweep_fusion.py`'s finding from the other direction.

Fixtures written to `eval/hard_cases-birds.json` and `eval/hard_cases-quant.json`.

---

## 2026-08-27 — Query expansion against the description-style cases: refuted twice

**The hypothesis, from the previous entry.** The twelve structural cases split
in two, and the five on birds and quant describe a term without naming it —
vocabulary mismatch rather than cross-document confusion. Query decomposition
needs credit; `src/query_expansion.py` implements RM3 pseudo-relevance feedback,
needs nothing, and had been sitting in the repo unmeasured. It is the classical
answer to exactly this failure.

**The way it could fail was written down before the run**, in the harness's own
docstring: feedback terms come from the top results of the *original* query, so
if the answer word is absent from those, expansion cannot invent it — it can
only sharpen a query already pointed at the wrong passages.

**That is what happened.**

| | any-hit | MRR | structural recovered |
|---|---|---|---|
| ML, none *(shipped)* | 0.851 | 0.738 | — |
| ML, prf | 0.851 | 0.744 | **0 of 7** |
| birds, none | 0.846 | 0.614 | — |
| birds, prf | 0.846 | 0.616 | **0 of 3** |
| quant, none | 0.886 | 0.714 | — |
| quant, prf | 0.886 | 0.731 | **0 of 2** |

Zero of twelve, and **not one question changes hands on any corpus**. The gains
are MRR and NDCG only: reordering inside a result set that is already the same
set.

**The mechanism was checked rather than assumed.** For each of the five,
`prf_terms` was asked what it would add:

```
bird-alula        adds: ratio, aspect, hoverers, kestrels, hovering, ...   'alula'     absent
bird-precocial    adds: days, nutritional, independence, enhanced, ...     'precocial' absent
qf-mean-reversion adds: the, and, certify, need, segments, our, ...        'reversion' absent
qf-momentum       adds: attempt, algorithm, picks, instruments, ...        'momentum'  absent
```

The target word is in none of them. No term-selection rule can help, because the
word is not in the material being harvested.

**A second defect fell out, and fixing it made things worse.** Look at the
`qf-mean-reversion` row: `the`, `and`, `our`. The module's docstring says
"a term must be both common among the top results and rare across the corpus.
Frequency alone selects 'the'" — and 'the' is being selected. The arithmetic:
feedback frequency is capped at the number of feedback documents, so it spans
one order of magnitude, while BM25's idf for a common word is small but not
small enough. On quant, `the` scores 1.788 against `momentum` at 5.341 — a
factor of three that ten-in-ten beats easily.

Thresholding on mean idf was rejected by measurement first: the distribution is
skewed towards rare terms (mean 7.15, max 8.32 on quant), so "above the mean"
would also discard `momentum` at 5.341 and `precocial` at 5.253. A document
frequency cut separates them properly — `the` is in ~17% of quant chunks,
`momentum` in ~0.5% — so terms appearing in more than 5% of the corpus were
excluded, with the threshold derived from BM25's own idf formula so it means the
same thing on 900 chunks and 6,000.

It worked, in the sense that `the`, `and` and `while` disappeared from the
feedback terms. Then it was measured:

| | before the fix | after the fix |
|---|---|---|
| ML any-hit | 0.851 | **0.836** *(loses `lora-frozen`)* |
| ML MRR / src recall | 0.744 / 0.762 | 0.741 / 0.745 |
| birds MRR | 0.616 | 0.623 |
| quant MRR / NDCG | 0.731 / 0.758 | 0.726 / 0.749 |

**Worse on two corpora of three, and it costs a question on the largest.**
The stopwords were doing less harm than what replaced them: they contribute
almost nothing to a BM25 score, but they occupy slots in a fixed-size term list,
and freeing those slots admitted more mid-frequency terms that pulled the query
further from its intent. **Reverted**, alongside `USE_TITLE_PREFIX` and the
rerank swaps.

**Two things to carry forward.** PRF is not the fix for the description-style
cases and is not worth turning on for the MRR — it is off by default and stays
off. And an obvious-looking defect ("it selects 'the'!") was doing less damage
than its obvious-looking fix, which is the entire argument for measuring before
believing, applied to a change that took four lines.

---

## 2026-08-27 — The most persuasive place for a wrong number

**Where nothing was looking.** `corpora.json` carries a `note` per corpus,
arguing why that corpus ships the threshold it does — "at 0.0 the gate wrongly
refuses 1 of 66 answerable questions (1.5%) and catches 9 of 18 adversarial".
Those are measurements, hand-written, sitting next to the setting they justify.
That is the most persuasive place in the repo for a number to be wrong, because
it is not read as a number: it is read as the reason for a decision.

**It had drifted.** The ML note said **1 of 66** and **9 of 18**. The golden set
holds **67** answerable and **17** adversarial. The counts moved when the set
grew and the sentence explaining the shipped threshold was never revisited —
which is exactly the failure mode of every other stale number found today,
except that this one argues for a config value rather than describing an
outcome.

Corrected, and `check_docs.py` now parses the notes for `refuses N of M` /
`catches N of M` claims and compares against the golden sets. Seven claims
across the three corpora, each one listed in the output so the coverage is
visible rather than assumed. Verified to have teeth by reintroducing the exact
stale figure and confirming it was caught.

**One design decision worth recording, because the obvious version is wrong.**
The quant note deliberately quotes a superseded figure: "The earlier note
claimed 0.0 refused nineteen of thirty-five; that was measured against a golden
set asking textbook definitional questions this corpus never answers." That
sentence is an accurate statement *about a wrong number*, and a checker matching
numerators would flag it as drift — misreading history as error, and training a
reader to ignore the checker.

So only the **denominator** is compared. The corpus size in that sentence is
still 35 and still checkable; the numerator is the note's own business. The
check reads the part of the claim that cannot be deliberately historical.

---

## 2026-08-27 — `POST /ask` answered from the wrong corpus, and the UI hid it

**Found by rendering the app**, which this project's §4 says to do and which had
not been done since `rerank.py` changed. The page itself was clean at every
viewport with no console errors, and the highlight was inside its bounds — 8, 12
and 3 words. Then a probe of the HTTP API selected the bird corpus, asked a bird
question, and got back `t5.pdf`.

**The defect.** `/api/trace` and `/api/chat` both take the live resources from
`RES` — index, metadata, BM25, the corpus's threshold, its rerank blend.
`POST /ask` called `api.ask()` bare. That function loads its own index through
an `lru_cache` over the process, which is right for a library entry point and
wrong for a server that can switch corpora. So `/ask` served:

- whichever corpus loaded first, **whatever the interface said was selected**
- at rerank blend 0.00, whatever the corpus ships (birds ship 0.20)
- against a hardcoded `min_confidence=-2.0`, not the calibrated threshold

The server would report "Ornithology, 45 documents, threshold -5.5" and answer
out of the ML papers, with citations pointing at documents that were never
searched. `retrieve.py`'s own comment names this exact failure as the reason the
store and the index must move together.

**Why it survived.** The interface talks to `/api/chat`, which was corpus-aware
all along. `/ask` is the documented HTTP API — HANDOFF §6 lists it, the README
calls it the wrapper around the library — and **nothing in the repo calls it**.
A route can be in every document and exercised by nothing.

**The fix** gives `ask()` optional `resources` and `rerank_blend`, and has
`serve.py` pass `RES`'s. Omitting both leaves the library path byte-identical,
because loading a corpus by itself is what the library entry point is for.

**Verified end to end, and it confirmed two other things at once.** With the ML
corpus selected the bird question is declined at -11.11 out of ML papers, which
is correct. With Ornithology selected it is **answered at -4.55 out of
`peregrine_falcon.pdf`** — the same confidence the harness measured for
`bird-incubation`, answered under the new -5.5 where the old -3.0 refused. One
call confirming the corpus switch, the threshold recalibration, and that the
server and the harness agree.

`src/test_api.py` stubs retrieval and asserts the resources handed in are the
ones searched, that the blend is forwarded, and that the gate uses the value it
was given — 13 checks, and it fails against the unfixed code.

---

## 2026-08-27 — Six routes nobody had written down

**Generalising the previous finding.** `POST /ask` served the wrong corpus for
its whole life because nothing exercised it. The general shape of that is not
"one route was broken", it is **a route the code serves and no document
mentions** — nobody tests what nobody has written down.

So `check_docs.py` now reads the dispatch out of `serve.py` with `ast` and
compares it against HANDOFF §6 in both directions. A regex was the obvious tool
and the wrong one: the dispatch is a chain of `route == "..."`, `route in (...)`
and `route.startswith(...)`, and a regex over that collects whichever quoted
strings happen to sit nearby.

**Six routes were served and documented nowhere:**

| route | what it is |
|---|---|
| `/api/analytics` | everything the analytics page plots |
| `/api/corpora` | the corpus list behind the document-set picker |
| `/quality` | the retrieval-quality dashboard |
| `/archive`, `/archive/pipeline-map.js` | the pre-rebuild front page, served live beside the current one |
| `/answer-mark.js` | the highlight module, added earlier today |

None of them broken — but `/ask` was not broken *visibly* either. `/api/chat`,
the endpoint the interface calls for every question, was itself listed nowhere
until this morning's `/ask` entry mentioned it in passing.

§6's route list is now grouped by what each route is for — pages, assets,
asking, corpus, measurements — rather than being a flat run of paths, since a
list nobody can read is a list nobody checks either.

**Three of the checker's first complaints were its own bugs**, and they are worth
recording because each is a way this kind of check quietly lies:

- `/` reported as documented-but-not-served. `serve.py` rstrips the trailing
  slash and compares against `""`, so the route the document calls `/` appears
  in the tree as the empty string.
- `/api/index/inspect` and its siblings reported as not served, because a
  documented wildcard `/api/index/*` was being used to *remove* served routes
  rather than to excuse undocumented ones. A wildcard must only ever forgive.
- `/api/index/inspect\`` — a trailing backtick captured into the route name.

**And one of its complaints was real, immediately.** Rewriting §6 changed the
heading from `- Routes:` to `- Routes.`, and the checker said "the route list is
gone -- this checker looks for a line starting '- Routes:'" rather than matching
nothing and reporting success. That is the behaviour every check here is
supposed to have, caught in the act.

Verified in both directions by misspelling one documented route, which produces
two findings from one typo: the real route undocumented, and the misspelling
undispatched.

---

## 2026-08-27 — Asking every route, and a field of the wrong type

**The next step after writing the routes down** is asking them. `check_docs.py`
proves a route is documented and dispatched; neither proves it answers, and
`POST /ask` was documented, dispatched, and wrong for its whole life.

`src/smoke_routes.py` reads the route list **out of `serve.py`'s dispatch**
rather than repeating it, so a route added tomorrow is exercised tomorrow
without anyone remembering. Typing the list here would reproduce the exact gap
it exists to close. The bar is deliberately low — not a 5xx, and a body that
parses as whatever the content type claims — because a route can return the
wrong corpus with a perfectly good 200. It is a floor, not a verdict.

All 21 routes answer. Two things came back that were not 200, and only one of
them was the server's fault.

**Mine:** `/api/index/inspect` returned a correct 400 because the probe pointed
it at the repo root, which holds no indexable documents. The route was right and
the request was wrong. Pointed at `data/` it answers.

**The server's:** `/api/corpus/select` **closed the connection without
responding**.

```
name = (payload.get("name") or "").strip()
                                   ^^^^^ 'dict' object has no attribute 'strip'
```

The probe sent `{"name": {...}}` — its own bug, from reading the corpus name out
of a response without checking its type. But JSON carries types and any caller
can send an object where a string belongs, and the answer should be a sentence
saying so. The route has `except ValueError` and a bare `except Exception`
around the work; the `.strip()` runs *before* the try, so the wrong-typed field
escaped both and dropped the socket. A dropped connection is indistinguishable
from the server having died, which is a worse thing to tell a caller than "that
field must be text".

**The same line shape was at four routes** — `/api/corpus/select`,
`/api/index/inspect`, `/api/chat`, and the `/ask` + `/api/trace` pair. All four
now go through one helper that answers 400 with the field name and the type it
got. The smoke test asks every POST route with a wrong-typed field and requires
a 400, so the fix is guarded by the thing that found it:

```
  /api/chat            question  400 field 'question' must be text, not dict
  /api/corpus/select       name  400 field 'name' must be text, not dict
  /api/index/inspect       path  400 field 'path' must be text, not dict
  /api/trace           question  400 field 'question' must be text, not dict
  /ask                 question  400 field 'question' must be text, not dict
```

**Not part of the counted suite**, because it needs a running server. That is a
real gap and is stated rather than papered over: 230 checks run without one, and
this is the twenty-second route's worth of coverage that only exists when
somebody runs it.

---

## 2026-08-27 — Indexing one corpus deleted the parse cache of the other two

**Fourth defect in four iterations of exercising rather than reading**, and the
largest. Indexing is the biggest subsystem nothing had ever run in a test, and
HANDOFF §2 quotes three timings that depend entirely on caches: full cold build
432 s, **re-index nothing changed 0.76 s**, add one document 18.8 s.

`ParseCache()` and `embed_with_cache()` were both called with no directory, so
both defaulted to a module constant pointing at `vector_store/` — the ML
corpus's store — whatever corpus was being indexed. Both caches are
content-addressed, so sharing them is safe on its own. `prune()` is what made it
unsafe:

```python
parse_cache.prune({file_key(p) for p in docs} | {file_key(p, chunk_salt()) for p in docs})
```

It deletes every entry whose key is not in that set, and the set holds only the
documents of the corpus in hand. **So indexing the birds deleted the parse cache
for the ML papers and for quant.**

**Measured before touching anything**, which is what turned a reading of the
code into a finding: `vector_store/parse_cache` held 90 entries — 45 bird
documents under two key forms — and **nothing at all** for the 36 ML documents
or the 35 quant ones. The last ingest run in this repo's history was the bird
corpus, and it had taken both other corpora's caches with it. Re-indexing the ML
papers today would have re-parsed all 36 PDFs, and the documented 0.76 s holds
only if no other corpus has been indexed in between. This repo has three.

**Fixed** by rooting both caches in the store being written, which is the rule
`corpora.py` already states for the index, the golden set and the threshold: a
corpus and everything derived from it travel together. For the ML papers
`store_dir` *is* `vector_store`, so its caches keep working untouched; birds and
quant start empty caches in their own stores and pay one re-parse each on their
next ingest — which they were paying every time under the old behaviour.

`src/test_ingest_cache.py` runs the real `build_index` over two throwaway
corpora, because the defect is entirely about which directory each cache
chooses, and a stub would have to decide that itself and would therefore assert
its own opinion. Nine checks. Against the old code five of them fail, including
the one that matters most: *"re-indexing an unchanged corpus parses nothing
again — got 2, want 0"*, which is the documented claim breaking in the test.

**One detour worth recording.** The first fixture was a heading and one
sentence, and `build_index` stopped after parsing — headings are not indexed as
passages, so the document produced no chunks and there was nothing to embed. The
stages emitted were `scan, parse, done`, no cache event at all, which surfaced
as four mysterious assertion failures rather than as "your test corpus is
empty". The fixture is twelve paragraphs now, and the docstring says why.

---

## 2026-08-27 — The third test this repo said it had and did not

**HANDOFF §5b, before today:** "What is now tested is everything up to the
network boundary — `src/test_trace.py`'s sibling check in the session log
stubbed the client and confirmed the model is handed exactly the passages the
answer cites."

`grep -l generate src/test_*.py` returns nothing. **No test imported
`generate.py`.** The check was run in a session and never committed.

That is the **third** claimed-but-absent test here, after
`test_trace_matches_pipeline` and the four hand-computed NDCG cases. All three
were found the same way — reading a confident sentence and then looking — and
all three mattered. A check that was performed and not committed is
indistinguishable, six days later, from one that never happened.

**It matters more in this module than the others.** `generate.py` has never
completed a real call, because the account has no credit. Nothing had executed
it end to end, ever. Three defects were already found in it by reading it
against the data it receives, including `build_context()` reading
`chunk["page"]` when chunks carry `locator` — which would have killed *every*
call with a KeyError.

`src/test_generate.py` stubs the client and tests all of it except the HTTP
request: 19 checks covering the citation format per locator kind, that the model
is handed exactly the passages it will cite and nothing else, the settings the
module's own comments argue for (`max_tokens` 16000 not the old 1024, effort
low), a refusal raising a named error rather than `StopIteration`, an empty
response raising rather than returning a blank answer, and a thinking block not
leaking into the text.

**Teeth, demonstrated:** put `chunk["page"]` back and the suite dies on the
first assertion with `KeyError: 'page'` — exactly how every real call would have
died. Put `max_tokens = 1024` back and that assertion fails too.

**And the rule this belongs to**, now written into §4. Every defect found in the
last five sessions was in code that had tests *around* it and nothing *running*
it: `/ask` ignoring its corpus argument, a wrong-typed field dropping the
connection, indexing one corpus deleting another's cache, six undocumented
routes, and a module that had never been invoked. Each was read, reviewed and
described correctly — and never called. The rule is not "write more tests", it
is **run the thing**.

---

## 2026-08-27 — Checking my own claim, which was too broad

Last entry ended with "every module in `src/` that can be executed without
credit has something executing it". §4 now says to check rather than assert, so
the claim got the treatment it asks for: an import graph from the ten test files,
transitively closed, against every module in `src/`.

**22 modules reachable. 32 never imported by any test.**

Most of the 32 are one-off scripts run by hand and recently — the sweeps,
`profile_query`, `fetch_topic`, `retitle`. "Nothing runs them" is false for
those; they are run, just not automatically. But the list also held **`serve.py`,
the largest module in the project**, and the claim was wrong.

What exercised `serve.py` was `smoke_routes.py`, which needs a running server and
is deliberately not in the counted suite. So everything between a request
arriving and retrieval starting had no automated coverage at all.

`src/test_serve.py` covers the three parts that have actually gone wrong, without
starting a server:

- **`_text()`** — the wrong-typed-field validation added earlier today. Until now
  only the live smoke test proved it, which meant it was proved only when
  somebody remembered to run a server. Reverting the fix now kills the suite with
  the same `AttributeError: 'dict' object has no attribute 'strip'` that dropped
  the connection.
- **`examples_for()`** — the questions the front page offers, which is the end of
  the chain that went stale and spent four days suggesting questions deleted from
  the golden set. Covered: a known corpus gets its own, and an unknown corpus, an
  empty list, a corrupt `analytics.json` and a missing one all fall back rather
  than raising or offering nothing.
- **`inspect_folder()`** — indexing REPLACES the vector store, so a pasted path
  that quietly resolves to an empty directory would destroy a working index and
  report success. Covered: empty string, missing path, a file rather than a
  folder, a folder with nothing indexable, and a Windows "copy as path" string
  with the quotes still on it.

25 checks. Importing `serve` costs about 25 seconds because it pulls the
embedding stack in at module level; it starts no server and loads no index.

**Still not covered, and stated rather than implied:** `build_analytics.examples()`
picks *which* questions reach the front page by a quota per case kind, and no
test asserts that quota. It is the same chain, one step up.

---

## 2026-08-27 — "The website is very laggy", measured rather than assumed

**The first measurement moved the target.** The obvious suspect was the
cross-encoder, since `profile_query.py` had already put it at 92-95% of a query.
But timing the endpoint the interface actually calls said something else:

```
  /api/chat   1,347-1,605 ms      (first time a question is asked)
  /ask               61-70 ms      same questions, same server
```

Twenty times apart on the same warm process. Timing the two halves of
`chat()` in-process found `trace_pipeline` at 88-100 ms and attribution at
90-98 ms — about 190 ms, nowhere near 1,400. The gap was the **rerank cache**:
`/ask` had been asked those questions before and `/api/chat` had not. Re-measured
warm, `/api/chat` is **113 ms**.

So the real shape is: **a cold question costs ~1,400 ms and a repeat costs
~113 ms**, and the lag a visitor feels is entirely the cold path.

**What shipped, and why it is not a quality change.** The questions the front
page offers are known before anyone asks them — they come from
`eval/analytics.json`. `serve.py` now scores them in a background thread once
the index is loaded, warming exactly the cache entries the same call would have
computed. Clicking an offered question went from ~1,400 ms to **102-178 ms**.
A question nobody pre-warmed is 931 ms, down from ~1,400 because `candidate_k`
went per-corpus earlier the same day.

It runs in a thread and swallows its own errors on purpose: the server is ready
to answer immediately, and a visitor typing their own question should not queue
behind a warm-up for questions they did not ask.

**And what was refuted.** Dynamic int8 quantisation of the cross-encoder looked
excellent on a 16-pair sample: **1.53x faster, max score delta 0.031** on a scale
running -11 to +11, top-1 unchanged. On the golden sets it fails, and it fails in
the way the sweep's own docstring predicted before it ran:

| | ranking | the gate | speed |
|---|---|---|---|
| ML papers | identical | **1 -> 3** wrongly refused, **9 -> 8** caught | 1.71x |
| Ornithology | +0.004 MRR | unchanged | **0.99x** |
| Quant | **-0.029** any-hit, **-0.055** MRR | 3 -> 4 wrongly refused | 1.58x |

Every corpus shifts by a systematic **-0.22** in the mean. Ordering survives
almost perfectly — which is all a normal rerank comparison would have checked —
and this project's abstention gate reads the score as an **absolute** against a
calibrated threshold. A change that preserves every ranking and moves every
score down by a fifth of a point silently starts refusing answerable questions.
That is what happened on the ML corpus, which lost two answerable questions and
one adversarial catch while its hit rate and MRR did not move at all.

Recalibrating the thresholds under int8 would recover the ML gate. It would not
recover quant's ranking, and birds gets no speedup at all, so the option is
rejected rather than patched. Kept behind `RAG_RERANK_INT8=1` and
`src/sweep_quantized.py` so the finding is reproducible.

**The lever that remains unmeasured** is ONNX export, which is a genuine
possibility and needs `optimum` and `onnxruntime` installed. It is not on this
machine and adding a dependency to chase latency the pre-warm has already
removed from the common path is the wrong trade today.

---

## 2026-08-27 — Smaller things

- `compare_rerankers.py` crashed **after** writing its results, on
  `Path.relative_to` with a relative `--emit`. A finished measurement looked
  like a failed run. Guarded.
- A Playwright screenshot script died on `UnicodeEncodeError` printing a marked
  run containing `ϵ`. The console is cp1252; `sys.stdout.reconfigure` is
  required in anything that prints corpus text, and the handoff says so.
- The first bird-corpus screenshot of this session asked a bird question against
  the ML corpus. It refused, at −10.86, and said so plainly. Not a defect — the
  gate doing its job, caught on camera.

---
