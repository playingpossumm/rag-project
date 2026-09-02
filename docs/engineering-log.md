# Engineering log

Started 2026-08-26. Every attempt is here, including the ones that were wrong.

`HANDOFF.md` records where the project is and this records how it got there,
and in particular what was tried and refuted. Section 4 of that document argues
that a measurement overturning a plan is worth more than a feature that ships,
and these are those measurements in full rather than compressed to a row of a
table.

An entry states the hypothesis before the result, gives the number, and says
what was done about it. An entry whose outcome is "no change" is worth as much
as one that ships something and is more likely to be forgotten.

---

## 2026-08-26 — A freshness check for the chain the front page reads

**Why.** Generated files had gone stale silently three times. The worst was
`per_case.json` sitting four days out of date, which meant the front page was
offering questions that had already been deleted from the golden set as
unanswerable, so the system was inviting people to ask it things it had
itself concluded it could not answer. Only `RESULTS.md` had a `--check`.

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
`per_case.py` imported `ABSTAIN_THRESHOLD`, the ML papers' `0.0`, and scored
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
to three decimals. That is the evidence that the change measured nothing new
and only corrected which pipeline was being described.

**Smaller things found on the way.**

- `answerable_median` in `evaluate.py` was `sorted(scores)[len(scores) // 2]`,
  the upper middle value rather than the median. On the 26 even-sized bird cases it
  reported +1.21 where `analytics.json`, using `statistics.median`, said +1.15.
  One quantity, two documents, two values.
- `build_analytics.py` mapped corpora to eval files with a hardcoded table. A
  corpus added to `corpora.json` and not to that table was skipped in silence,
  and the front page then offered it the ML papers' fallback questions.
- README said the bird corpus holds 900 passages. It holds 864.

---

## 2026-08-26 — The bird gate, calibrated against a pipeline nobody runs

**Hypothesis.** With the confidences re-measured on the served pipeline, the
bird threshold `-3.0` might no longer be right, having been derived at rerank
blend 0.00 on a corpus that ships 0.20.

**Measured.** It was dominated. Every threshold in `(-6.48, -4.55]` catches the
same five of six adversarial cases and wrongly refuses **three** of twenty-six
rather than four.

| cut | answerable lost | adversarial caught |
|---|---|---|
| −3.0 *(was)* | 4 | 5 |
| **−5.5 *(now)*** | **3** | 5 |
| −7.0 | 3 | 4 |

**Shipped −5.5**, the middle of the interval, which leaves 0.95 of margin
before it refuses an answerable question and 0.98 before it stops catching an
adversarial one. An edge would fit the threshold to a single case. Retrieval
untouched: any-hit 0.846, MRR 0.614.

**Checked on all three corpora before shipping**, because the trap is tuning to
the one that prompted the question. ML's `0.0` and quant's `-4.0` are already on
the frontier, where every lowering costs catches. Only birds was dominated.

**Tooling flaw this exposed.** `calibrate_threshold.py` had its grid hardcoded to
`[-4 .. +4]`, which is where the ML papers' scores live and nowhere near the bird
corpus's, since every question deciding that corpus's threshold sits below
−4. The
tool used to calibrate a corpus could not display the region being calibrated.
The grid is derived from the observed scores now, and it reports the *interval* a
threshold sits in rather than a grid point, because nothing changes until a cut
point crosses an actual score.

---

## 2026-08-27 — A score cache keyed on the model in name only

**Found while preparing to compare rerankers.** `rerank._score_cache` was keyed
on `(query, chunk text)`. The comment directly above it had always read "scores
are a pure function of (query, chunk text, model)".

Nothing had ever swapped cross-encoders inside one process, so nothing caught
it. The first thing that would have is a reranker comparison, and it would
have reported every candidate model as scoring **exactly** like whichever
loaded first. That is not an error anyone questions; it looks like a null result.

Fixed by keying on the model name and taking the name as the argument rather
than an already-constructed model object, whose identity a cache cannot read.
`src/test_rerank.py` proves it with two stub encoders differing only by name:
7 checks, verified to fail 2 when the old two-part key is put back, with the
telling failure `got 1.0, want -1.0`: the second model served the first's score
and was never even asked.

---

## 2026-08-27 — A better cross-encoder, and what the measurement said

**Hypothesis.** `ms-marco-MiniLM-L-6-v2` is weak on questions that *describe* a
term rather than naming it, as in "the burst of collective singing at first
light", and the bird corpus has the headroom to show it: the candidate pool holds the
answer 96.2% of the time and the pipeline returns it 84.6%.

**First, the harness had to be rebuilt.** `compare_rerankers.py` was ML-only: it
hardcoded a list of failing ML case ids, called `load_index()` with no store, set
no rerank blend, and measured a single number (top-1 from an answering
document). It could not run on the corpus with the headroom.

Rewritten to measure **two failures, not one**, because reranking fails in two
ways and only one of them shows up in any-hit:

- **ordering:** any-hit, MRR, NDCG, source recall
- **scoring:** the gate reads the cross-encoder's score of the top passage, and
  on the bird corpus three of the seven failures are questions whose answer the
  pipeline *found* and then refused to show

Scores from two models are on different scales, so holding a threshold fixed
across models measures the scale, not the separation. The gate half is therefore
reported threshold-free as **AUC**, the probability that a random answerable
question outscores a random adversarial one. It is reported alongside
`unreachable`, the count of answerable questions scoring
below the third-highest adversarial, which no threshold can save.

**Result on the bird corpus (26 answerable + 6 adversarial):**

| model | any-hit | MRR | AUC | unreachable |
|---|---|---|---|---|
| MiniLM-L6 *(shipped)* | **0.846** | **0.614** | 0.859 | 2 |
| MiniLM-L12 | 0.808 | 0.594 | 0.878 | 1 |
| BGE-reranker-base (278M) | 0.769 | 0.596 | **0.968** | **0** |

**The bigger models rank worse and separate better.** That was not the expected
shape at all, and it is the finding: the two halves of the reranker's job move in
opposite directions as the model grows. A straight swap is not available: BGE
would trade two ranking failures for a perfect gate.

**A latency number that was wrong, and how it was caught.** The first run
reported BGE at 1,576,300 ms/query, or 1700× the shipped model, against the
project's own earlier measurement of 10 s/query on the ML papers. A 150×
disagreement with a prior measurement is a reason to distrust the new one.
Timed again with nothing else running: **244 ms/pair against 25 ms/pair,
9.8×**. The first figure measured a contended machine rather than a model:
several evaluation runs, a server and a browser were competing for it. Quality
metrics were unaffected, being deterministic.

**Hypothesis that followed, and was refuted.** Ranking needs a score for twenty
candidates; the gate needs one. So let the expensive model do only the half it
wins at, ranking with L6 and gating with BGE, at a twentieth of BGE's cost.

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
   scoring MiniLM's. Part of that 0.968 was self-consistency, because a model
   is confident about its own pick. Anyone comparing rerankers on AUC alone would
   have read 0.968 as a reason to swap.
3. **The "it does not transfer" law now covers the model too**, alongside the
   abstention threshold and the rerank blend. That is three independent
   settings, measured separately, all corpus-specific. It is the strongest form
   of the project's own thesis and it was not assumed. Each one was found by
   shipping the opposite first.

**So the next move on this stage is not a bigger cross-encoder.** The failure is
specific, being questions that describe a term rather than naming it, and a
cross-encoder of any size reads the same words. The fix that addresses the
mechanism is query decomposition, which needs an LLM, which needs credit. That
is now measured rather than asserted.

---

## 2026-08-27 — Fusion per corpus, where the pool and the pipeline disagree

**Hypothesis.** The reranker turned out to be a dead end, but the stage before it
looked promising. `HANDOFF.md` had said since the second corpus was built that
fusion is probably per-corpus, on the strength of one number: the bird candidate
pool holds the answer 96.2% of the time under dense retrieval and 88.5% under
RRF, so fusing costs that corpus two questions before reranking starts. It was
never acted on because 26 cases is too few to move a *global* default. That was
sound
about a global default, silent about a per-corpus one, which is what the
threshold and the blend already are.

**What was missing was the measurement at the served configuration.**
`evaluate.py` compares fusions *before* the diversity cap and applies the cap
only to RRF, so the table every document quotes has no row for "weighted,
capped" at all. `src/sweep_fusion.py` runs every fusion on every corpus at
k=5 from 20 candidates, with that corpus's own rerank blend and the 2-per-source
cap, which is the pipeline `api.ask()` actually runs.

**The result reverses the premise.**

| birds | pool any-hit | shipped-pipeline any-hit |
|---|---|---|
| dense only | **0.962** | **0.808** |
| RRF *(shipped)* | 0.885 | **0.846** |
| weighted a=0.5 | 0.923 | **0.885** |

Dense retrieval finds the answer most often and produces the *worst* final
result. A candidate pool is not a set, it is an **ordering handed to the
cross-encoder**, and dense hands over one the reranker cannot exploit. That is
the same
weakness the rerank blend exists to hedge. Reading pool recall as a proxy for
pipeline quality is the error, and this project's own handoff had been making it
for a week.

**Nothing shipped, and both reasons are stated rather than assumed.**

- birds, `weighted a=0.5`: wins one question (`bird-dialects`) and loses MRR
  0.614 → 0.587 and NDCG 0.671 → 0.662. Identical in shape to the trade the
  rerank blend was judged on, and this corpus's own note already settles how to
  read it: on 26 cases, an any-hit gain of one question is thinner evidence
  than MRR. **Rejected by the project's own stated principle**, which is the
  best kind of rejection: the rule existed before the result.
- quant, `weighted a=0.7`: weakly dominant, at MRR 0.714 → 0.727, NDCG +0.006,
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

## 2026-08-27 — The answer highlight, moved, bounded and measured

**Why.** The interface sets the answering words bold inside a passage shown at
normal weight. That logic lived inline in a two-thousand-line HTML file, so
nothing could run it without a browser, and nothing ever had. Every claim about
it, "only the answering words" and "at most a sentence", described code no test
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
`answerSpan` marked the **entire sentence**, which is the one case with the least
justification for marking anything. Now it marks nothing.

*A word bound is not a length bound.* Swept over the real top passage for every
question in all three golden sets, which is 450 passages over 157 questions:
one mark
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
"Vaswani et al. (2017) introduced…" as two sentences, since terminator,
whitespace,
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

`src/audit_golden_set.py` already asked the *semantic* question, which is
whether the corpus has grown into an adversarial case's subject or an
answerable question has become ambiguous. It is ML-only, with a hand-written
table of which words would make each adversarial case answerable.
`src/check_golden.py` asks the
*structural* one and needs no per-case knowledge, so it runs on every corpus
including ones that do not exist yet.

**It found a broken label on its first run.** `bird-hollow-bones` carried
`{"kind": "section", "pages": [1, "Skeletal system"]}`. "Skeletal system" is a
real section; the `1` was meant for `table 1`, a different locator kind in the
same document. As written, `(bird_anatomy.docx, section, "1")` matched no chunk
in the corpus and contributed nothing, while looking exactly like a label.

**And it did not change the score, which is worth saying plainly.** The pipeline
returns none of the three places carrying "hollow bones". It returns slide 4,
`section Axial skeleton`, `section Overview` and slide 2, so the case misses
either way. The broken label sits on a genuine miss. Fixing it left
any-hit at 0.846 and MRR at 0.614, and the temptation to present the fix as a
recovered question is exactly the kind of thing this log exists to prevent.

`src/test_golden.py` stages each failure class on a synthetic two-document
corpus and asserts the audit reports it: 15 checks, hermetic, running in
milliseconds.

**A gap in my own work, found by using it.** Fixing that one locator changed no
case count, no corpus size and no metric, and `check_freshness` still reported
`results-birds.json` as current. `per_case.json` had recorded a digest of its
golden set since the day the check was written; `results.json` never had. The
digest half exists precisely for the edit that moves no count, and it was
missing from half the chain. Now stamped by `evaluate.py` and checked, with two
more cases in `test_freshness.py` (30 → 32).

---

## 2026-08-27 — Documents checked against the measurements

**The failure this closes** is this repo's most-repeated one. The README once
claimed "84 evaluation cases" directly above figures measured on 23.
`eval/RESULTS.md` said 20 papers and 2,768 chunks long after the corpus reached
36 and 5,459, under a header promising that if the two disagreed the document
was stale. It was true, and nobody noticed, because nothing checked it.

`build_results_doc.py --check` closed that for RESULTS.md by *generating* its
tables. `check_freshness.py` closed the generated chain feeding the interface.
Neither covered **HANDOFF.md §2 and the README**, which are the two documents
a new session and a visitor read first.

Generating them would be the wrong fix: the argument around each number is
judgement and cannot come from a JSON file. So `src/check_docs.py` leaves them
hand-written and checks them: 14 quantities, each naming where its truth
lives. A table it cannot find is a *failure* rather than a skip, because a
checker that quietly matches nothing reports success.

**Verified to have teeth** by perturbing two numbers and confirming both were
caught, one of them the exact `900 passages` error that was really in the README
earlier the same day.

One subtlety that would have made it useless: compare at the precision the
document *writes*, not the precision the measurement carries. `+4.93` against
`4.934403419494629` is a document rounding correctly, and reporting that as
drift trains a reader to ignore the checker, which is how a check stops being
read.

---

## 2026-08-27 — Optimising, and an unambiguous measurement

**The rule this project already learned.** "Re-indexing is slow because the
index is rebuilt" was a well-formed plan until rebuilding the FAISS index
measured at 0.01 s, which is 0% of runtime, and the task had to be redefined.
So
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
"search, scoring and reranking, 1164 ms" had been hiding which one.

**Two ways to make it cheaper, both measured.**

*Faster execution, same work.* Threads are already at the machine's best:
torch defaults to 10 of 12 cores, and 4, 2 and 1 are all slower. Padding waste inside
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
rises on all three**, at 0.925 → 0.955 on ML, 0.885 → 0.962 on birds and
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
such a change is not shippable as a default. It is a trade for the author to
make, with the numbers on the table, rather than a decision to slip in under a
latency heading.

**One thing was fixed rather than measured.** `rerank._score_cache` grows for
the life of the process and nothing outside a test had ever called
`clear_cache()`. That was free for its original caller, since an evaluation
run scores a few hundred questions and exits, but `serve.py` imports the same
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
is that the corpus has grown into them, since it holds papers that did not
exist when those questions were written. In that case the labels are wrong and
the threshold is being blamed for a golden-set problem. That is exactly what
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
answers nothing: "reward", "training details", "context" and "memory" are
doing the work. So the eight are **genuine gate failures, not mislabelled cases**, and
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
gold document, which is the definition of a *correct* label. Fixed to subtract
the gold sources, the real figure is **0 of 67**: every answer string appears
only in the documents its label names.

That number had been sitting in the output as a wall of false positives for as
long as the check existed, hiding a clean result. It is the same trap
`HANDOFF.md` already records once, "a measurement returning zero deserves as
much suspicion as a surprise", with the sign flipped. 100% deserves it too.

**And the term list does not predict the gate.** Seven of the eight cases
retrieval answers anyway were not flagged by the hand-written subject-term
table, which the script's own output already says: "...and what retrieval says,
which is the check that matters." A hand-maintained list of what *would* make a
case answerable is a guess; running the retriever is a measurement.

---

## 2026-08-27 — Documented defaults checked against the code

HANDOFF §3 lists eleven constants under "Defaults, all justified by measurement
in eval/RESULTS.md". That opening makes each of them a claim about the code, and
nothing checked it. Change `TOP_K` in `retrieve.py` and the document goes on
describing a system that no longer exists, in a place nobody thinks to look
because it reads as prose rather than as a number.

`check_docs.py` now imports the module that owns each constant and compares.
Twenty-six quantities in total across §2, §3 and the README. Verified to have
teeth by setting `TOP_K = 7` and confirming it was caught.

It also found something the list itself hides: **`RRF_K` is defined twice**, in
`hybrid.py` and `rerank.py`. They agree at 60 today and nothing makes them
agree: fusion damping and the rerank blend's damping are the same constant by
intention and a copy in practice. The checker asserts the two agree and reports
what would break if they stopped: "fusion and the rerank blend would damp
differently".

Four constants are environment-overridable. When the variable is set the check
says so and skips, rather than reading an overridden value and calling the
document wrong, which would be the checker lying.

**And the ambiguity question, asked of all three corpora.** With
`audit_golden_set.py`'s version corrected, the check needs no per-case knowledge,
so it moved into `check_golden.py` where every corpus gets it. It reports a
*note* rather than a failure, since a second document carrying the answer string
is a judgement rather than a structural error. Mixing the two would make the
exit code mean "something to read" instead of "something is wrong".

Across all 157 cases in three corpora: **no ambiguity at all**. Every answer
string appears only in the documents its label names. Two tests assert the note
fires on a deliberately ambiguous synthetic case, because a check reporting zero
on real data is exactly where you have to demonstrate it can report anything
else.

---

## 2026-08-27 — The seven-case fixture, unchanged

**Why re-measure.** HANDOFF §7 claims "18 of 66 answerable cases fail under some
configuration, but only 7 fail under all of them", and calls those seven a
fixture worth optimising against. Two reasons to distrust it today: the corpus
has **67** answerable cases, not 66, so the figure predates the golden set it
describes; and this session changed the abstention threshold, the rerank blend
resolution and one label. A "structural" classification that moves when the
config moves was never structural.

**Six `per_case.py` configurations**, being default, no-rerank, dense-only,
weighted fusion, no cap and cap 1/src, fed to `src/failure_overlap.py`.

| | 2026-08-21 | 2026-08-27 |
|---|---|---|
| answerable cases | 66 | **67** |
| fail under *some* configuration | 18 | **21** |
| fail under *all* | **7** | **7** |

**And they are the same seven ids**: `adam-bias`, `dropout-rate`,
`gpt3-fewshot`, `gpt3-params`, `roberta-nsp-drop`, `t5-text2text`, `wmt14`.
Not seven again by coincidence: the sets are identical, with nothing entering
or leaving. The classification survived everything this session changed, which is
the strongest evidence yet that it is a property of the questions rather than of
the settings, and that scoring query decomposition against it will mean
something.

**A second reading the run gives for free.** Of the eight adversarial cases the
gate lets through on the ML corpus, **seven are answered under every gated
configuration** and only `adv-diffusion` (+0.43, the weakest of them) moves.
Those seven are the gate's own fixture, and they are exactly the cases whose
passages were read earlier today and found to be correctly labelled. Two
independent routes, reading the text and varying the pipeline, agree that
these are gate failures rather than label failures.

`norerank` is excluded from that count and the tool says why: with no reranking
there is no calibrated score, so the gate never runs and all 17 adversarial
cases are answered by construction. An absent gate, not a failing one.

**One incidental number.** `nocap` has the lowest failure rate of the six at
14.9%, against the shipped 16.4%. That is the diversity cap costing a case,
which is the trade §2 already documents as deliberate (it buys source recall).
Consistent rather than new, and worth noting that it reproduced.

---

## 2026-08-27 — A fixture for every corpus, and two failure modes

**Why.** The structural-versus-contested split had only ever been computed for
the ML corpus, so "stress-test all the golden sets" was two-thirds unmet. Six
configurations each for birds and quant, being default, no-rerank,
dense-only, weighted fusion, no cap and cap 1/src, through
`src/failure_overlap.py`.

| | ML papers | Ornithology | Quant |
|---|---|---|---|
| answerable | 67 | 26 | 35 |
| fail under *some* configuration | 21 (31%) | 11 (42%) | 12 (34%) |
| **structural** — fail under all six | **7 (10.4%)** | **3 (11.5%)** | **2 (5.7%)** |
| lowest-failing configuration | nocap 14.9% | **weighted 19.2%** | default/nocap/weighted 14.3% |

The "~11% floor" HANDOFF §7 records for the ML corpus reproduces almost exactly
on birds, at 10.4% against 11.5%, on a corpus of Wikipedia pages in four file
formats rather than arXiv PDFs. Quant is **5.7%**, about half, so the floor is
not universal and should not be quoted as if it were.

**Then read the twelve structural questions together, and they split in two.**

*ML, all seven, ask about an attribute many papers share:*
"How does the optimizer correct for bias in its moment estimates", "what dropout
rate was applied", "how many parameters does the largest autoregressive model
have", "which translation dataset". The topic matches a dozen documents and the
clause that picks one out is ignored. This is the cross-document confusion the
project already names.

*Birds and quant, all five, describe a term and ask for its name:*
"Which small group of feathers helps prevent a stall at low speed" (alula),
"which skeletal adaptation reduces a bird's weight" (hollow bones), "what term
describes chicks able to move and feed themselves soon after hatching"
(precocial), "which behaviour describes prices being pulled back toward a
long-run level" (mean reversion), "which effect describes past winners
continuing to outperform" (momentum). **The answer word never appears in the
question at all.**

**These are different problems and they want different fixes.** Query
decomposition, the planned fix and the one blocked on credit, splits a question
into its constraints, which is exactly right for "normalisation across features
*rather than examples*". It does nothing obvious for "which small group of
feathers prevents a stall", where nothing needs splitting and the target
vocabulary is simply absent. That is a vocabulary-mismatch problem, and
`src/query_expansion.py` already exists in the repo unmeasured against it.

So the roadmap item "the ~11% floor needs query decomposition" was **one plan for
two problems**, and it covers seven of the twelve cases rather than all of them.
Recorded rather than acted on: measuring query expansion against the five is the
next thing this suggests, and it is not blocked on credit.

**Three confirmations that came free.**

- `bird-hollow-bones`, whose broken locator was fixed earlier today, is
  **structural**, failing under all six configurations. That independently
  confirms what was said at the time: the broken label was not masking a hit.
- `bird-dawn-chorus` and `bird-imprinting` are contested and work **only under
  `norerank`**. With reranking off there is no calibrated score and no gate, so
  "works" there means the first stage retrieved the answer and the cross-encoder's
  score is what refused it. Third independent route to the same conclusion.
- `weighted` fusion has the lowest failure rate on birds (19.2% against the
  shipped 23.1%), matching `sweep_fusion.py`'s finding from the other direction.

Fixtures written to `eval/hard_cases-birds.json` and `eval/hard_cases-quant.json`.

---

## 2026-08-27 — Query expansion against the description-style cases, refuted twice

**The hypothesis, from the previous entry.** The twelve structural cases split
in two, and the five on birds and quant describe a term without naming it,
which is vocabulary mismatch rather than cross-document confusion. Query decomposition
needs credit; `src/query_expansion.py` implements RM3 pseudo-relevance feedback,
needs nothing, and had been sitting in the repo unmeasured. It is the classical
answer to exactly this failure.

**The way it could fail was written down before the run**, in the harness's own
docstring: feedback terms come from the top results of the *original* query, so
if the answer word is absent from those, expansion cannot invent it. It can
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
Frequency alone selects 'the'", and 'the' is being selected. The arithmetic:
feedback frequency is capped at the number of feedback documents, so it spans
one order of magnitude, while BM25's idf for a common word is small but not
small enough. On quant, `the` scores 1.788 against `momentum` at 5.341, a
factor of three that ten-in-ten beats easily.

Thresholding on mean idf was rejected by measurement first: the distribution is
skewed towards rare terms (mean 7.15, max 8.32 on quant), so "above the mean"
would also discard `momentum` at 5.341 and `precocial` at 5.253. A document
frequency cut separates them properly, since `the` is in ~17% of quant chunks
and `momentum` in ~0.5%, so terms appearing in more than 5% of the corpus were
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
cases and is not worth turning on for the MRR. It is off by default and stays
off. And an obvious-looking defect ("it selects 'the'!") was doing less damage
than its obvious-looking fix, which is the entire argument for measuring before
believing, applied to a change that took four lines.

---

## 2026-08-27 — The most persuasive place for a wrong number

**Where nothing was looking.** `corpora.json` carries a `note` per corpus,
arguing why that corpus ships the threshold it does: "at 0.0 the gate wrongly
refuses 1 of 66 answerable questions (1.5%) and catches 9 of 18 adversarial".
Those are measurements, hand-written, sitting next to the setting they justify.
That is the most persuasive place in the repo for a number to be wrong, because
it is not read as a number: it is read as the reason for a decision.

**It had drifted.** The ML note said **1 of 66** and **9 of 18**. The golden set
holds **67** answerable and **17** adversarial. The counts moved when the set
grew and the sentence explaining the shipped threshold was never revisited,
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
numerators would flag it as drift, misreading history as error and training a
reader to ignore the checker.

So only the **denominator** is compared. The corpus size in that sentence is
still 35 and still checkable; the numerator is the note's own business. The
check reads the part of the claim that cannot be deliberately historical.

---

## 2026-08-27 — `POST /ask`, answering from the wrong corpus

**Found by rendering the app**, which this project's §4 says to do and which had
not been done since `rerank.py` changed. The page itself was clean at every
viewport with no console errors, and the highlight was inside its bounds at 8,
12 and 3 words. Then a probe of the HTTP API selected the bird corpus, asked a bird
question, and got back `t5.pdf`.

**The defect.** `/api/trace` and `/api/chat` both take the live resources from
`RES`: the index, the metadata, BM25, the corpus's threshold and its rerank
blend.
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
all along. `/ask` is the documented HTTP API, listed in HANDOFF §6 and called
the wrapper around the library by the README, and **nothing in the repo calls
it**.
A route can be in every document and exercised by nothing.

**The fix** gives `ask()` optional `resources` and `rerank_blend`, and has
`serve.py` pass `RES`'s. Omitting both leaves the library path byte-identical,
because loading a corpus by itself is what the library entry point is for.

**Verified end to end, and it confirmed two other things at once.** With the ML
corpus selected the bird question is declined at -11.11 out of ML papers, which
is correct. With Ornithology selected it is **answered at -4.55 out of
`peregrine_falcon.pdf`**, which is the same confidence the harness measured for
`bird-incubation`, answered under the new -5.5 where the old -3.0 refused. One
call confirming the corpus switch, the threshold recalibration, and that the
server and the harness agree.

`src/test_api.py` stubs retrieval and asserts the resources handed in are the
ones searched, that the blend is forwarded, and that the gate uses the value it
was given: 13 checks, and it fails against the unfixed code.

---

## 2026-08-27 — Six routes nobody had written down

**Generalising the previous finding.** `POST /ask` served the wrong corpus for
its whole life because nothing exercised it. The general shape of that is not
"one route was broken", it is **a route the code serves and no document
mentions**. Nobody tests what nobody has written down.

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

None of them were broken, but `/ask` was not broken *visibly* either. `/api/chat`,
the endpoint the interface calls for every question, was itself listed nowhere
until this morning's `/ask` entry mentioned it in passing.

§6's route list is now grouped by what each route is for, under pages,
assets, asking, corpus and measurements, rather than being a flat run of paths.
A list nobody can read is a list nobody checks either.

**Three of the checker's first complaints were its own bugs**, and they are worth
recording because each is a way this kind of check quietly lies:

- `/` reported as documented-but-not-served. `serve.py` rstrips the trailing
  slash and compares against `""`, so the route the document calls `/` appears
  in the tree as the empty string.
- `/api/index/inspect` and its siblings reported as not served, because a
  documented wildcard `/api/index/*` was being used to *remove* served routes
  rather than to excuse undocumented ones. A wildcard must only ever forgive.
- `/api/index/inspect\``, from a trailing backtick captured into the route
  name.

**And one of its complaints was real, immediately.** Rewriting §6 changed the
heading from `- Routes:` to `- Routes.`, and the checker said "the route list is
gone: this checker looks for a line starting '- Routes:'" rather than matching
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
it exists to close. The bar is deliberately low, requiring only that the
response is not a 5xx and that the body parses as whatever the content type
claims, because a route can return the wrong corpus with a perfectly good 200.
It is a floor, not a verdict.

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

The probe sent `{"name": {...}}`, which was its own bug, from reading the
corpus name out of a response without checking its type. But JSON carries types
and any caller can send an object where a string belongs, and the answer should
be a sentence saying so. The route has `except ValueError` and a bare
`except Exception` around the work; the `.strip()` call runs *before* the try, so the
wrong-typed field escaped both and dropped the socket. A dropped connection is
indistinguishable from the server having died, which is a worse thing to tell a
caller than "that field must be text".

**The same line shape was at four routes**: `/api/corpus/select`,
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

## 2026-08-27 — Indexing one corpus, deleting another's parse cache

**Fourth defect in four iterations of exercising rather than reading**, and the
largest. Indexing is the biggest subsystem nothing had ever run in a test, and
HANDOFF §2 quotes three timings that depend entirely on caches: full cold build
432 s, **re-index nothing changed 0.76 s**, add one document 18.8 s.

`ParseCache()` and `embed_with_cache()` were both called with no directory, so
both defaulted to a module constant pointing at `vector_store/`, the ML
corpus's store, whatever corpus was being indexed. Both caches are
content-addressed, so sharing them is safe on its own. `prune()` is what made it
unsafe:

```python
parse_cache.prune({file_key(p) for p in docs} | {file_key(p, chunk_salt()) for p in docs})
```

It deletes every entry whose key is not in that set, and the set holds only the
documents of the corpus in hand. **So indexing the birds deleted the parse cache
for the ML papers and for quant.**

**Measured before touching anything**, which is what turned a reading of the
code into a finding: `vector_store/parse_cache` held 90 entries, being 45 bird
documents under two key forms, and **nothing at all** for the 36 ML documents
or the 35 quant ones. The last ingest run in this repo's history was the bird
corpus, and it had taken both other corpora's caches with it. Re-indexing the ML
papers today would have re-parsed all 36 PDFs, and the documented 0.76 s holds
only if no other corpus has been indexed in between. This repo has three.

**Fixed** by rooting both caches in the store being written, which is the rule
`corpora.py` already states for the index, the golden set and the threshold: a
corpus and everything derived from it travel together. For the ML papers
`store_dir` *is* `vector_store`, so its caches keep working untouched; birds and
quant start empty caches in their own stores and pay one re-parse each on their
next ingest, which they were paying every time under the old behaviour.

`src/test_ingest_cache.py` runs the real `build_index` over two throwaway
corpora, because the defect is entirely about which directory each cache
chooses, and a stub would have to decide that itself and would therefore assert
its own opinion. Nine checks. Against the old code five of them fail, including
the one that matters most: *"re-indexing an unchanged corpus parses nothing
again: got 2, want 0"*, which is the documented claim breaking in the test.

**One detour worth recording.** The first fixture was a heading and one
sentence, and `build_index` stopped after parsing. Headings are not indexed as
passages, so the document produced no chunks and there was nothing to embed. The
stages emitted were `scan, parse, done`, no cache event at all, which surfaced
as four mysterious assertion failures rather than as "your test corpus is
empty". The fixture is twelve paragraphs now, and the docstring says why.

---

## 2026-08-27 — The third test this repo said it had and did not

**HANDOFF §5b, before today:** "What is now tested is everything up to the
network boundary. `src/test_trace.py`'s sibling check in the session log
stubbed the client and confirmed the model is handed exactly the passages the
answer cites."

`grep -l generate src/test_*.py` returns nothing. **No test imported
`generate.py`.** The check was run in a session and never committed.

That is the **third** claimed-but-absent test here, after
`test_trace_matches_pipeline` and the four hand-computed NDCG cases. All three
were found the same way, by reading a confident sentence and then looking, and
all three mattered. A check that was performed and not committed is
indistinguishable, six days later, from one that never happened.

**It matters more in this module than the others.** `generate.py` has never
completed a real call, because the account has no credit. Nothing had executed
it end to end, ever. Three defects were already found in it by reading it
against the data it receives, including `build_context()` reading
`chunk["page"]` when chunks carry `locator`, which would have killed *every*
call with a KeyError.

`src/test_generate.py` stubs the client and tests all of it except the HTTP
request: 19 checks covering the citation format per locator kind, that the model
is handed exactly the passages it will cite and nothing else, the settings the
module's own comments argue for (`max_tokens` 16000 not the old 1024, effort
low), a refusal raising a named error rather than `StopIteration`, an empty
response raising rather than returning a blank answer, and a thinking block not
leaking into the text.

**Teeth, demonstrated:** put `chunk["page"]` back and the suite dies on the
first assertion with `KeyError: 'page'`, which is exactly how every real call
would have died. Put `max_tokens = 1024` back and that assertion fails too.

**And the rule this belongs to**, now written into §4. Every defect found in the
last five sessions was in code that had tests *around* it and nothing *running*
it: `/ask` ignoring its corpus argument, a wrong-typed field dropping the
connection, indexing one corpus deleting another's cache, six undocumented
routes, and a module that had never been invoked. Each was read, reviewed and
described correctly, and never called. The rule is not "write more tests"; it
is **run the thing**.

---

## 2026-08-27 — Checking my own claim, which was too broad

Last entry ended with "every module in `src/` that can be executed without
credit has something executing it". §4 now says to check rather than assert, so
the claim got the treatment it asks for: an import graph from the ten test files,
transitively closed, against every module in `src/`.

**22 modules reachable. 32 never imported by any test.**

Most of the 32 are one-off scripts run by hand and recently: the sweeps,
`profile_query`, `fetch_topic` and `retitle`. "Nothing runs them" is false for
those; they are run, just not automatically. But the list also held **`serve.py`,
the largest module in the project**, and the claim was wrong.

What exercised `serve.py` was `smoke_routes.py`, which needs a running server and
is deliberately not in the counted suite. So everything between a request
arriving and retrieval starting had no automated coverage at all.

`src/test_serve.py` covers the three parts that have actually gone wrong, without
starting a server:

- **`_text()`**, the wrong-typed-field validation added earlier today. Until now
  only the live smoke test proved it, which meant it was proved only when
  somebody remembered to run a server. Reverting the fix now kills the suite with
  the same `AttributeError: 'dict' object has no attribute 'strip'` that dropped
  the connection.
- **`examples_for()`**, the questions the front page offers, which is the end of
  the chain that went stale and spent four days suggesting questions deleted from
  the golden set. Covered: a known corpus gets its own, and an unknown corpus, an
  empty list, a corrupt `analytics.json` and a missing one all fall back rather
  than raising or offering nothing.
- **`inspect_folder()`**. Indexing *replaces* the vector store, so a pasted path
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

Twenty times apart on the same warm process. Timing the two halves of `chat()`
in-process found `trace_pipeline` at 88-100 ms and attribution at 90-98 ms, or
about 190 ms together, nowhere near 1,400. The gap was the **rerank cache**:
`/ask` had been asked those questions before and `/api/chat` had not.
Re-measured warm, `/api/chat` is **113 ms**.

So the real shape is: **a cold question costs ~1,400 ms and a repeat costs
~113 ms**, and the lag a visitor feels is entirely the cold path.

**What shipped, and why it is not a quality change.** The questions the front
page offers are known before anyone asks them, since they come from
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
almost perfectly, which is all a normal rerank comparison would have checked,
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

## 2026-08-27 — The structural fixture, relative to the first stage

**The stage nobody had varied.** Of the twelve cases that fail under every
pipeline configuration, five describe a term and ask for its name, as in
"which small group of feathers helps prevent a stall at low speed" for the
alula, and the answer word is absent from the question. Reranking cannot fix that
(`compare_rerankers.py` found nothing better) and pseudo-relevance feedback
cannot either (`sweep_query_expansion.py`: 0 of 12, because the feedback
documents do not contain the missing word). Every one of those experiments
varied a stage *after* retrieval.

`all-MiniLM-L6-v2` is a general-purpose **symmetric** model, trained to place two
similar sentences near each other. Description-to-term is **asymmetric**: a long
question and a short passage sharing almost no vocabulary. So
`src/compare_embedders.py` measures pool recall under models trained for that,
without re-indexing, since each embeds the corpus in memory once.

| Ornithology, pool of 20 | pool recall | structural found |
|---|---|---|
| all-MiniLM-L6-v2 *(shipped)* | 0.962 | 2/3 |
| multi-qa-MiniLM-L6-cos-v1 | 0.962 | 2/3 |
| bge-small-en-v1.5 | 0.923 | 1/3 |

| Quantitative finance, pool of 16 | pool recall | structural found |
|---|---|---|
| all-MiniLM-L6-v2 *(shipped)* | **0.914** | 1/2 |
| multi-qa-MiniLM-L6-cos-v1 | 0.857 | 1/2 |
| bge-small-en-v1.5 | 0.886 | 0/2 |

**Nothing dominates, and the aggregate hides the interesting part.** On birds
`multi-qa` ties the shipped model at 0.962 while finding a *different* set:
it **wins `bird-alula`** and loses `bird-precocial`. On quant it is worse
outright. The fourth setting in a row that does not transfer.

On quant the pattern repeats with different names: `bge-small` scores below the
shipped model overall and still **wins `qf-cvar-interval`**, losing
`qf-momentum` and `qf-sparse-params`. Three models, two corpora, and in every
pairing each finds something the others cannot.

One detail worth keeping: `qf-momentum` is structural and the shipped embedder
DOES put it in the pool. The pipeline loses it after retrieval, which makes it a
reranker failure wearing a retrieval failure's label.

**But `bird-alula` is one of the twelve.** It was called structural because no
fusion, rerank or cap change reached it, and a different first stage reaches
it immediately. So the fixture is **not a property of the questions**, as the
2026-08-27 entry claimed on the strength of six configurations. It is a property
of the questions *given this embedder*. Six configurations that all shared one
first stage could not have discovered that, and the word "structural" was
carrying more weight than the measurement supported.

The honest restatement: seven ML cases are structural with respect to
retrieval-as-a-whole and want query decomposition; of the five description
cases, at least one is reachable by changing the embedder alone.

**What this suggests, and it is measurable without credit:** the two models find
*different* cases at the same aggregate. That is the exact condition under which
fusing them helps. It is the same argument that already justifies fusing dense
with BM25, applied to two dense retrievers. A dense ensemble is the next experiment,
and unlike everything else on the structural list it needs no LLM.

---

## 2026-08-27 — Two embedders fused, and the first thing to move the fixture

**The prediction, made before the run**, was that two dense models might be too
correlated to fuse usefully: dense and BM25 fuse well because one scores meaning
and the other exact terms, while two dense models trained on overlapping data
may produce a union that is mostly the intersection. That happened once and the
rest of the time it did not.

Pool recall, and the structural cases each pool contains:

| | ML papers | Ornithology | Quant |
|---|---|---|---|
| all-MiniLM-L6-v2 *(shipped)* | 0.866 · 3/7 | 0.962 · 2/3 | **0.914** · 1/2 |
| multi-qa-MiniLM-L6-cos-v1 | 0.866 · 3/7 | 0.962 · 2/3 | 0.857 · 1/2 |
| bge-small-en-v1.5 | **0.910** · 3/7 | 0.923 · 1/3 | 0.886 · 0/2 |
| fused: shipped + multi-qa | 0.881 · **4/7** | **1.000** · **3/3** | 0.857 ✗ |
| **fused: shipped + bge** | **0.881** · 3/7 | **1.000** · **3/3** | **0.914** · 1/2 |

**Fusing the shipped model with `bge-small` is weakly dominant**, at +0.015 on
the ML papers, +0.038 on birds and exactly level on quant. That is the first
retrieval change in this project to be better-or-equal on all three corpora
rather than a trade.

**And it moves the fixture.** `bird-alula`, which asks "which small group of
feathers helps prevent a stall at low speed", was one of the twelve cases that
fail under every pipeline configuration. Fused, it is in the pool. Bird pool recall
reaches **1.000**: every answerable question in that corpus now reaches the
reranker. With the `multi-qa` pairing the ML fixture moves too, `gpt3-params`
and `cot-prompt` both entering the pool, but that pairing costs quant 0.057, so
it is the wrong one to ship.

**Two things the aggregate hides, and both matter.**

`bge-small` **alone** is the best single first stage on the ML papers by a
distance, at 0.910 against the shipped 0.866, and worse on both other corpora.
The fifth setting in a row that does not transfer, and a reminder that "which
embedder" is a per-corpus question exactly like the threshold, the blend, the
fusion and the candidate pool.

On the ML papers the fusion (0.881) is **worse than `bge-small` alone** (0.910).
That is the correlation dilution predicted in the sweep's docstring, caught in
the act: RRF damping can push a passage one model ranked first below one that
both ranked seventh. So "fuse two dense models" is not a general improvement; it
is a thing that happened to help two of these three corpora.

**Not shipped, and the reason is engineering rather than evidence.** A second
embedder means a second vector per chunk: the index roughly doubles, ingest
gains a second encode pass over the whole corpus, and `retrieve()` grows a
second dense arm to fuse before the existing dense+BM25 fusion. That is a real
change to the shape of the store, not a value in `corpora.json`, and it wants
its own session with its own re-index and re-calibration. The measurement is
here, reproducible with `src/sweep_ensemble.py`, and it says the work is worth
doing.

---

## 2026-08-27 — The generation path, run for the first time

`generate.py` targets `claude-opus-5` and has never completed a call: no credit.
`src/test_generate.py` covers everything up to the network boundary by stubbing
the client, which leaves the one part that involves a network permanently unrun.

The contract is not Anthropic-specific: passages in, cited prose out, a named
error when the model declines and a named error when nothing comes back. So
`src/generate_local.py` speaks it to a local **Ollama**, selected by
`RAG_GENERATOR=ollama`, with the same `synthesize(question, chunks)` signature
and the same raising behaviour, so `api.ask` and `serve.chat` can call either
without knowing which.

**Ollama is not installed on this machine, and the test does not need it to be.**
`src/test_generate_local.py` stands up a real `ThreadingHTTPServer` implementing
the two endpoints the module uses, on an ephemeral port, and points the module
at it. The request is built, sent over a socket, answered, parsed, and returned.
Nothing is monkeypatched except the host.

That makes it the first test here in which the generation path actually
executes end to end. The model is a fake that echoes its prompt, which is
exactly what lets the test assert that the passages reaching the model are the
ones the answer will cite, and the transport, the JSON shapes and every error
branch are real. Eighteen checks: the happy path, the citation format per
locator kind, deterministic sampling because this path gets measured, an empty
answer raising rather than rendering blank, a model that is not pulled saying
`ollama pull`, a 500 that is *not* misreported as "go install something", and
nothing listening at all saying `ollama serve`.

**What it is not:** a claim that a small local model writes answers as well as
Claude. It will not. It is the difference between code that has never run and
code that has, on the one path in this repo where that distinction was still
outstanding.

One detour worth recording: the first teardown called `server_close()` while a
thread was still inside `serve_forever()`, which printed a traceback from a
daemon thread *after* the results and read like a failure that was not one. And
the "nothing listening" check first pointed at the shut-down server rather than a
dead port. `shutdown()` stops serving but leaves the socket bound, so the
connection is accepted and never answered, which is a hang rather than the
refusal being asserted.

---

## 2026-08-28 — Going public under MIT, with the history kept

**The licence is MIT.** "Public" without one legally means nobody may reuse
anything, which for a project whose stated purpose is to be read and learned
from is the opposite of the intent. Apache-2.0 was considered and rejected: its
patent grant and NOTICE requirements are overhead here with no corresponding
benefit.

**The source documents are untracked but NOT purged from history**, and the
second half of that was the real decision.

`data/` is 36 arXiv PDFs and the two topic stores hold the same text again,
verbatim, inside `metadata.json`, for 78 MB of other people's work. Adding them to
`.gitignore` does not untrack them, so all 44 files were still in HEAD; they are
out now, and rebuildable with `fetch_corpus.py`, `fetch_topic.py` and
`ingest.py`.

Removing them from the *history* is a different operation, and the argument for
it is sound: a file in the history of a public repository is published whether
or not it is in HEAD, so every clone still downloads all 78 MB and the PDFs stay
retrievable from earlier commits. `git-filter-repo` was installed and the rewrite
prepared, with a backup branch, a tag and a copy of the files.

**It was not run, and the deciding reason is specific rather than general.**
This project's documents cite its own commit SHAs. `HANDOFF.md` §5 points at
`f2ab5de` and `f1ea450` for the deleted ambient-field work, and `serve.py`
embeds `d1f8c71` in the banner it serves over the archived front page, where a
visitor reads that SHA on screen. All three resolve today. A rewrite changes
every SHA in the repository, so all three would become dangling references, and
they would fail silently: nothing in the test suite or the four guards checks
that a commit named in prose still exists.

Weighed against that, the redistribution concern is real but modest. arXiv's
submission licence grants a non-exclusive right to distribute, many of these
papers are CC-BY, and a research-tooling repository carrying the papers it was
measured against is ordinary practice. 67 MB is unremarkable for GitHub.

So: history preserved, HEAD clean, and the reasoning written down rather than
left as an unexplained absence. If the redistribution question is ever revisited,
the rewrite is one command, and the three SHA references have to be fixed in
the same change, which is the part that is easy to miss.

---

## 2026-08-28 — Three tuned pipelines, not one pipeline on three corpora

**The challenge, and it was right.** Four things are tuned per corpus now: the
abstention threshold, the rerank blend, the candidate pool size, and whether a
second embedder is fused. The numbers this project quotes for working across
unlike document sets, being 0.851, 0.846 and 0.886, are each produced under
that corpus's own settings. Quoting them as evidence that the *pipeline* generalises
overstates the case, and it had been true since well before the ensemble.

Two different questions were being answered with one set of numbers:

    does the pipeline generalise?    one configuration, three corpora
    what does each corpus need?      each corpus's own configuration

`src/uniform_baseline.py` measures the first, which nothing had. The ranking
metrics are threshold-independent, since any-hit, MRR, NDCG and source recall
come from the ordering alone, so a single shared configuration is directly
comparable in a way the gate never can be. The uniform configuration is the
*untuned* one, the defaults a corpus gets before anybody measures anything,
because a compromise chosen after seeing the results would be tuning with extra
steps.

| under one untuned configuration | any-hit | MRR | NDCG | src | what tuning adds |
|---|---|---|---|---|---|
| ML & NLP papers | 0.851 | 0.738 | 0.754 | 0.760 | **+0.000 / +0.001** |
| Ornithology | 0.808 | 0.570 | 0.635 | 0.761 | +0.038 / +0.044 |
| Quantitative finance | 0.886 | 0.714 | 0.749 | 0.741 | +0.000 / **+0.064** |

**The answer strengthens the claim rather than weakening it.** Across three
corpora that share no format, subject or provenance, being arXiv PDFs,
Wikipedia pages in four file formats and quantitative-finance papers, one
untuned configuration spans **0.808 to 0.886** on any-hit. A spread of 0.078.

And the corpus the pipeline was *developed on* gains **nothing** from its own
tuning: +0.000 any-hit, +0.001 MRR. So the tuning is not propping up three
special cases. It is worth 4 points of any-hit on birds and 6.4 points of MRR on
quant, on top of a system that already works untuned.

**What this changes in how the numbers are quoted.** The tuned figures are
correct and remain the right ones for "what does this system do on this corpus".
They are the wrong ones for "does this approach generalise", and §2 now carries
both tables rather than one. `eval/uniform-baseline.json` is regenerated by the
script and holds the spread.

**A caveat that belongs with it:** MRR spans 0.570 to 0.738 under the uniform
configuration, a spread of 0.168 and more than twice the any-hit spread. Finding
the answer generalises better than ranking it first does, which is consistent
with everything else measured here about the cross-encoder being the weakest
stage.

---

## 2026-08-29 — An interface slow on any machine but this one

**The report was "the website is very laggy" and the first three measurements
disagreed with it.** Headless Chromium on the development desktop: 240 ms to
first paint, 60 fps on the hero, 667 ms for a question. The deployed build was
no worse: four cold loads settled between 276 and 494 ms. Nothing to fix.

That is the wrong instrument. A machine that holds 60 fps whatever the page
does cannot tell a cheap frame from an expensive one; it reports the frame
budget, not the frame. Throttling the CPU 6x through CDP, as a proxy for a
mid-range laptop, made the difference visible immediately:

    idle hero                20 fps
    three answers on screen  11 fps, worst frame 283 ms
    long tasks               129, totalling 14.9 s

**Four causes, all of them work whose result could not change.**

1. `canvas.width = ...` reallocates and clears the backing store whether or not
   the value changed. Both draw loops assigned it every frame, so a 2400x1400
   bitmap was being thrown away and rebuilt sixty times a second to paint the
   size it already was, preceded by a `getBoundingClientRect` that forces
   layout to obtain the number it was about to ignore. Now behind a
   `ResizeObserver`, which reports the only event that matters.

2. Every answered turn started a seven-second replay and none of them stopped.
   Three questions in a row meant three full-canvas animations at once, two of
   them scrolled off the screen. An `IntersectionObserver` plays a drawing when
   it is on screen and stops it when it leaves, showing the finished state
   rather than a frozen half-drawn one.

3. `captions(run)`, which builds six objects and their prose, was rebuilt
   inside every frame of the replay, and `cap.innerHTML` was rewritten every frame to show
   the same six captions. Roughly 2,500 strings and 420 HTML reparses per
   answer, for six changes.

4. The index plate draws a mark per sampled passage on each of its six sheets.
   A CPU profile put over half the time in raster and named `mark`, `stroke`
   and `save` beneath it, and that block is identical in every frame, since
   the sheets do not move and only the front one lights up. It is now painted
   once into an offscreen canvas and stamped, keyed on the run so a different
   answer gets a different layer. Only while the plate is still forming does it
   draw directly: the first version keyed the cache on the fade too, which
   allocated a full-size canvas per frame of the fade and was slower than what
   it replaced.

The bounds probe at the top of `drawScene` was also memoised. It is a pure
function of a layout that was already memoised next door, and it ran per frame.

**After, at the same 6x:**

    idle hero                60 fps   (was 20)
    three answers on screen  30 fps   (was 11), worst frame 167 ms (was 283)
    first answer             1,702 ms (was 2,990)
    long tasks               32 totalling 3.3 s   (was 129 / 14.9 s)

At 4x, a more ordinary laptop, everything is 60 fps and ten long tasks remain.

**The drawing is unchanged, and that was checked rather than assumed.** Both
canvases were screenshotted before and after under `prefers-reduced-motion`,
which removes the pulse and the replay and leaves a deterministic frame: zero
pixels differ past a threshold of 8, on a maximum channel delta of 3.

The lesson is the one this project keeps relearning in new clothes. The rule
has been "run the thing"; the amendment is that running it on the machine that
built it is not running it. Every number in the first three measurements was
correct and all of them were about the wrong computer.

---

## 2026-08-30 — Four defects in the answer-quality judge

**Why look.** Answer quality had been measured once: nine answers, one corpus,
2026-08-29. Extending it to three corpora meant reading the nine that existed
against the labels they had been scored on, which is the only reason any of
what follows was found. Nothing here came from reading the code.

**Every one of the four made the model look worse than it was**, which is the
direction that flatters a project: a harness that reports failures nobody has
to explain away is a harness nobody checks.

**1. A refusal scored as an answer.** `adv-bird-consensus` replied "I don't
have any information about proof of stake protocols or consensus mechanisms
from the provided excerpts". The refusal set held the fixed phrase "no
information" and nothing matching "don't have any information", so the corpus
reported 2 of 3 adversarial questions refused where the truth was 3 of 3.

The list was replaced with patterns and immediately made the opposite error:
"air sacs are not found in mammals" is a claim about biology, and "not found"
matched. So the patterns split in two. The ones whose verbs also occur in
ordinary claims have to land near a word naming the source material; the
unambiguous ones do not. Against eight real answers the old list is wrong on
five, in both directions, and the patterns on none.

**2. Mathematics scored as fabricated citations.** The ML papers reported
**20 invented citations across 12 answers**, and not one was a citation:

    mha-def     [Q]  [K]  [j]
    pos-enc-fn  [2]  [model]
    warmup      [min(]  [num]  [warmup]  [steps]

The regex matched any bracket group, and these are papers full of notation the
model copies out of the passages it is given. This is the worst wrong number
here: `invented_citations` is the measure the harness leads with, on the
argument that a fabricated citation is worse than no answer, and it was firing
hardest on the most technical corpus, at a model that had fabricated nothing.

A bracket group now has to be citation-shaped before it counts: an extension, a
locator, or a supplied filename. The cost is stated rather than hidden, because
it is real: a fabrication written in some other form, "[Smith et al. 2019]", is
no longer counted, since nothing distinguishes it from "[Q]" by shape alone.

**3. A single character matched every long filename.** With the shape rule in
and 44 checks passing, the volume-imbalance formula still reported five
citations. `[t]` is a substring of
`multi_level_market_making_with_reinforcement_learning`, and the rule allowed a
substring match because models truncate long filenames. Substring matching now
needs four characters. Exact equality still counts at any length, so a corpus
holding `t5.pdf` can still cite it.

**4. A hedge scored as a refusal.** Broadening the patterns in (1) fixed the
misses and reported seven answerable questions as wrongly refused, five of
which had been answered:

    "The text does not explicitly state the type of load balancing loss used.
     However, it does mention that the auxiliary-loss-free load balancing
     strategy is adopted..."                                   moe-routing

    "According to [attention...pdf, page 3], multi-head attention is used in
     the following three ways: 1. ...  The provided excerpts do not mention
     the third way."                                           attn-uses

Refusal is a property of the answer, not of a sentence in it. The test is now
whether any claim survives once the declining sentences are set aside, so a
hedge followed by an answer has answered, and so has a partial answer that says
which part is missing.

That produced one more error in the other direction, and it is the interesting
one. A model that declines and then describes the corpus instead of answering

    "I don't have enough information to answer this question. The provided
     excerpts appear to be related to machine learning and natural language
     processing."                                              adv-qf-syrinx

has a second sentence that declines nothing and carries plenty of content
words, so it counted as a surviving claim. A sentence whose subject is the
source material is therefore never counted as an answer. It is a claim about
the documents, not about the question.

**Results, after all four, hand-adjudicated case by case:**

| | ML papers | Ornithology | Quant |
|---|---|---|---|
| invented citations | **0** | **0** | **0** |
| refused correctly | 4/5 | 5/5 | 4/5 |
| refused wrongly | 0/10 | 1/10 | 1/10 |
| contains the labelled answer | 6/10 | 6/10 | 4/10 |
| groundedness (proxy) | 0.678 | 0.525 | 0.566 |

Zero invented citations across 45 answers is the result that matters, and it is
the one the broken measure was hiding. Correctness is a floor rather than a
rate: it is a substring test, so `bird-keel` answering "the keel on their
breastbone" against a label reading "keeled sternum" counts as wrong. The test
was left exactly as strict, because loosening it trades a false negative for
false positives and inflates the number, and the 14 answers it rejects are
listed under `unmatched` to be read instead.

**Two things built because of this, not planned before it.**

`--rescore` re-judges answers that already exist. Twice today a defect
invalidated a finished run and both times the answers were fine, and
regenerating an hour of identical prose to re-apply a corrected string test is
the wrong shape of work. It re-runs retrieval, which the judge needs for the
passages, and never calls the model: a minute instead of the better part of an
hour. Every number above was produced by it.

`src/test_evaluate_answers.py`, 60 checks, hermetic. Most of the strings in it
are answers this repository has already scored wrongly, kept verbatim rather
than written to pass. Each of the four defects has a check that fails without
its fix. The suite is also the reason (3) and the second half of (4) were
found: both were introduced by a fix and caught by writing the test for it.

**The lesson, and it is not "write more tests".** Every one of these was found
by reading the harness's own output against the thing it was measuring. The
module had been read, reviewed and described correctly, and it had never been
checked against a single real answer. This is the same finding as 2026-08-27,
in the one place left where the input is prose: a string test over prose does
not fail loudly, it reports a number.

---

## 2026-08-30 — A link that would have failed twice

`/about` linked to `docs/corpus-manifest.md` under `blob/main/`, and this
repository's only branch is `master`. The repository is also still private, so
the link 404s today for an entirely different reason.

That is the shape worth recording. Making the repository public fixes the
visible symptom and leaves the branch wrong, and the second failure looks
exactly like the first from outside: a 404. One failure hiding behind another,
where fixing the one you can see hides the one you cannot.

The branch half is decidable offline, so `src/check_links.py` decides it
offline. It reads the internal routes out of `serve.py`'s dispatch through
`check_docs.served_routes()` rather than repeating them, checks that an anchor
names an `id` that exists in the target page, and checks that a link into this
repository names a branch and a file that exist. `--http` asks the network
about external links and is not part of the default run, because it cannot pass
while the repository is private.

Verified by reintroducing both faults. An anchor is worth checking for the same
reason as the branch: `/about#nosuchid` does not error, it lands silently at
the top of the page, which is worse than an error because it looks like it
worked.

**The guard lists disagreed with each other**, and that was found while adding
the sixth. `HANDOFF.md` said "three checks" above four commands, the roadmap
said "four guards" and omitted `build_corpus_manifest --check`, the README said
five. All three now name the same six.

---

## 2026-08-30 — Three panes, and a bar that could not draw

All three reported by the owner, all three confirmed by rendering the panes
rather than by reading the diff.

**The Stages pane had no visualisation.** Each stage carried a bar whose width
was a percentage of the passages it held, and every stage drew the same stub.
The bar was a flex item with a percentage width inside a `float:right` that
shrink-wraps to its content, so the percentage resolved against a box the bar
was itself sizing. Rebuilt with a fixed-width track and a proportional fill: it
now measures 102, 102, 102, 102 and 32 pixels across the five stages of a
16-candidate run, so the narrowing is a shape rather than five numbers.

**The Word-matches heatmap introduced orange and green with nothing naming
them.** Hue is the document, which the rest of the site establishes, but this
pane's only key was a blue ramp, so the first orange column arrived
unexplained. It now carries a document key with swatch, name and column count,
and the intensity ramp is neutral grey, because it encodes strength and not
identity and a blue ramp beside coloured cells read as a fourth document.

**The rank chart's markers were ovals because the geometry was dishonest.** It
was drawn in a 100-unit viewBox with `preserveAspectRatio="none"`, stretched
across about 1,030 pixels, so the horizontal scale ran roughly ten times the
vertical. Both axes scale a stroke, which is why the end markers were ellipses
and a steep line drew several times the weight of a shallow one. Drawn in pixel
units taken from the element instead, redrawn from a `ResizeObserver` rather
than per frame.

Worth separating the two halves of that. The distortion was a bug; what it hid
was that the chart had no axes. Once a circle was a circle it was obvious that
rank was unlabelled and the stage labels were distributed by `space-between`
rather than sitting under their columns. Looking at it then showed two more:
the tick list appended the deepest rank unconditionally and drew 15 and 16 a
few pixels apart, and the "removed here" ring was drawn for the grey lines too,
stacking ten of them into a vertical chain at each column.

**And a stale file the same day.** `record_static.py --site-only` returns
before the line that copies `eval/analytics.json` into the build, so every
cheap page rebuild shipped the current interface against whatever measurements
the last full recording carried. It surfaced as a panel that would not appear,
which is the lucky version. The unlucky version is a number, because a stale
number looks exactly like a current one. Copying a JSON file needs no models,
so it now happens on both paths.

---

## 2026-08-31 — The dense ensemble, decided on all three corpora

**The open question.** Roadmap item 1 has read "ship the dense ensemble
properly, or decide not to" since the sweep measured it. That sweep measured
**pool recall**, where fusing the shipped embedder with `bge-small-en-v1.5`
looked weakly dominant: +0.015 on the ML papers, +0.038 on birds, level on
quant. It was enabled on quant and left off the other two, and the decision was
deferred rather than taken.

**Measured end to end, in both directions.** The two missing indexes were
built, both corpora were scored at the served configuration, and quant, which
ships the ensemble, was scored with it removed. The baseline for the first two
is the shipped `results*.json`, which is the ensemble-off measurement.

| shipped configuration | any-hit | MRR | source recall |
|---|---|---|---|
| ML papers, off → on | 0.851 → **0.821** | 0.739 → 0.721 | 0.761 → 0.743 |
| Ornithology, off → on | 0.846 → **0.885** | 0.613 → 0.597 | 0.762 → 0.762 |
| Quant, on → off | 0.886 → 0.886 | 0.779 → **0.743** | 0.769 → **0.741** |

**Three different answers on three corpora.** This is the fifth setting
measured as per-corpus rather than global, after the abstention threshold, the
rerank blend, the candidate pool size and the choice of embedder, and the
documents have listed it as the fifth since the pool sweep. What is new is not
the classification but the evidence: until now it rested on pool recall, which
cannot see the thing that actually decided it.

On the ML papers it is worse on every metric. On birds it wins a question of
any-hit and loses MRR, which is precisely the trade `weighted a=0.5` was judged
on and rejected on this corpus: on 26 cases a one-question gain in any-hit is
thinner evidence than a fall in MRR. On quant it is earned, and removing it
costs 0.036 MRR and 0.028 source recall for no change in any-hit.

**Shipped: unchanged.** Quant keeps it, the other two do not get it, and the
two indexes built for the experiment were deleted.

**The part worth keeping is why the pool was wrong.** On the ML papers the
candidate pool did not move at all: RRF pool recall is 0.910 with the ensemble
and 0.910 without it. The same twenty passages arrive at the reranker in a
different order, and the reranker does worse with that order. This is the
fourth time in this project that a better or equal first stage has produced a
worse finished answer, and the clearest instance yet, because here the pool is
not merely a ceiling that failed to help. It is *identical*, and only the
ordering changed.

That also explains the sweep. Pool recall cannot see an ordering change inside
the pool, so a measurement taken there was structurally incapable of predicting
this result in either direction.

---

## 2026-08-31 — An override that changed the model and not the prefix

The ensemble indexes for this experiment were built with `--model
BAAI/bge-small-en-v1.5` on two corpora whose `corpora.json` entries name no
ensemble model. The builder took the model from the flag and the query prefix
from the corpus, so both indexes were written with an empty prefix.

BGE retrieval models are trained with the query side carrying "Represent this
sentence for searching relevant passages: " and land in a different region of
the space without it. Nothing failed: the file is the right size, the row count
matches `metadata.json`, the manifest is well formed, and retrieval is quietly
worse. The first measurement off those indexes read 0.851 → 0.821 any-hit, and
that number described a broken index rather than the ensemble.

The prefix belongs to the model, so the model carries it now, and
`corpora.json` still wins when it says something.

**The fix then did not fire, and that is the more useful half.**
`corpora.registry()` normalises an absent key to an empty string rather than
`None`, so an `is not None` test always matched and the model default was never
reached. The rebuild produced two more prefix-less indexes and reported
success. The fix had been verified by reading the code rather than by reading
what it produced, which is this project's own recurring failure appearing
inside a fix for a different instance of it. One line, reading the manifest the
builder had just written, is what found it.

Rebuilt correctly, the ML figure was **unchanged at 0.821**. The prefix was a
real defect and not the cause of the result, which is worth stating plainly:
finding a bug in the instrument does not mean the reading was wrong.

---

## 2026-08-31 — A bigger generator, on a sample too small to settle it

**Why.** Every answer-quality figure describes llama3.2, a 3B model, so the
numbers describe that model on this corpus rather than the pipeline's ceiling.
llama3.1:8b is the same family at roughly 2.7x the parameters, which makes size
the only variable that changes.

**The run did not finish.** Ollama stopped partway, so the ML corpus produced
11 of its 15 answers and the other two corpora produced none. Everything below
is the 11 cases both models answered, which is 7 answerable and 4 adversarial.
That is small enough that one question is fourteen points, and it is reported
because the shape of the result is interesting, not because the numbers are
settled.

| ML papers, 11 shared cases | llama3.2 (3B) | llama3.1 (8B) |
|---|---|---|
| contains the labelled answer | 4/7 | 4/7 |
| refused correctly | 3/4 | **4/4** |
| refused wrongly | 0/7 | 0/7 |
| invented citations | 0 | 0 |
| malformed citations | 0 | **2** |
| groundedness (proxy) | 0.729 | 0.553 |
| median answer length | 255 chars | 139 |

**Correctness did not move.** Same 4 of 7, with one case swapped in each
direction: the 8B loses `moe-routing` and gains `warmup`. On this sample 2.7x
the parameters bought nothing on the measure that matters most, which is worth
knowing before anyone reaches for a larger model as the obvious next step.

**The gate improved**, 3 of 4 adversarial to 4 of 4. One question, so it is a
direction and not a result.

**Citation discipline got worse, and that is the surprise.** The 8B emitted the
papers' own bibliography numbers as citations, `[4, page 4]` and
`[36, page not specified]`, where the 3B emitted none across 45 answers. One of
those answers is *only* the reference number, with no prose at all, which is
the same collapse the 3B shows when the prompt says "Context:" instead of
"Excerpts:". A larger model did not remove that brittleness; it found a
different way into it.

That also forced a distinction in the judge. A bare number names no document,
so it cannot point a reader at one that does not exist, and calling it an
invented citation overstates it. Invented and malformed are now counted apart:
the first is a fabricated document, the second is uncheckable.

**Shorter and less grounded.** Median answer length halves and lexical overlap
with the supplied passages falls from 0.729 to 0.553. On a grounded-answering
task, drawing less of the answer from the passages is not obviously an
improvement, and the correctness column says it did not buy one here.

**What would settle it** is the finished run on all three corpora, which needs
Ollama up for about an hour. The harness, the model and `--rescore` are all in
place, so it is one command.

---

## 2026-08-31 — Query decomposition, measured at last, and refuted

**The claim.** Seven ML cases fail under every combination of fusion, reranking
and diversity cap, and the roadmap has said since they were isolated that query
decomposition is the right fix: the question names a topic and a constraint,
retrieval matches the topic, and the constraint that picks the right document is
averaged away. Splitting the question is what an LLM does well. That was an
argument, never a measurement, and it was blocked on credit until
`query_rewrite.py` learned to speak to a local Ollama.

**Two modes against the shipped pipeline, on all 67 answerable cases:**

| mode | any-hit | MRR | NDCG | src recall | structural | s/query |
|---|---|---|---|---|---|---|
| none *(shipped)* | **0.851** | **0.738** | **0.754** | 0.760 | 0/7 | **1.30** |
| rewrite | 0.806 | 0.686 | 0.715 | **0.765** | **1/7** | 7.35 |
| decompose | 0.806 | 0.622 | 0.666 | 0.759 | 0/7 | 27.56 |

**Decomposition recovers none of the seven** and costs 0.045 any-hit at 21
times the latency. The claim it was supposed to support does not survive its
first measurement.

**Rewriting recovers one of the seven and loses three questions net.** It gains
`adam-bias` and `bert-wordpiece` and loses `cot-prompt`, `lora-frozen`,
`mha-def`, `sbert-speed` and `seq2seq-reverse`. Five for two.

**The part worth keeping is that the fixture said the opposite.** Run against
the seven alone, rewrite recovered one and lost nothing, which reads as a pure
win. It is not: the cases it breaks are all outside the fixture, so a harness
pointed only at the fixture cannot see them. The module's own docstring named
this before either run, which is why the sweep defaults to the whole set and
`--structural-only` carries a warning:

> a change measured only on the cases it was built for cannot show what it
> costs the cases it was not: the seven are 10% of this corpus, and a rewrite
> that recovers two of them while losing four elsewhere is a bad trade that
> looks like a good one.

The real trade was two recovered against five lost. Predicting the shape of a
trap and then walking into a slightly worse version of it is the ordinary way
this goes, and the reason the prediction was written into the code rather than
into a plan.

**Why rewriting breaks working questions.** The rewrite is a paraphrase, and a
paraphrase of a question that already retrieves well moves it off the wording
that was working. `mha-def` asks about multi-head attention in terms the paper
uses; a rewrite that says the same thing differently retrieves worse. The seven
structural cases are the ones where the original wording is the problem, so
they are exactly the minority the change helps.

**Nothing shipped.** Both modes stay off, `query_rewrite.py` keeps its
degrade-to-the-original behaviour, and `eval/decompose-sweep.json` holds the
numbers. The seven remain unreached, and the honest position is that this
project has now ruled out the fix it had been naming for them.

---

## 2026-08-31 — The answer on screen, which was the title page

**Reported by looking.** The owner asked "What problem does normalizing layer
inputs address?" and got the batchnorm paper's title, its two authors, their
two email addresses, and eight words of abstract stopping at "complicated by".
Nothing bold, because none of the question's words were in it.

**Three separate things were wrong, and the first two are not what they look
like.**

**1. Retrieval was right.** The chunk returned at rank 1 is 973 characters and
contains "We refer to this phenomenon as internal covariate shift, and address
the problem by normalizing layer inputs". The answer was in the passage the
system chose.

**2. The excerpt was wrong.** `pipeline_trace._brief` showed the first 260
characters of a passage, always. On a median chunk of 696 characters the head
reaches the substance; on the first chunk of a paper the head is the title, the
authors and their email addresses, because that is what the first 260
characters of a paper are. The reader was shown the 260 characters immediately
before the answer.

The window now opens at the sentence with the most question words in it, scored
the way `ui/answer-mark.js` already scores sentences to decide what to embolden.
A passage that opens with its answer is unaffected, because its first sentence
wins the same test.

**3. Filtering title blocks was the obvious fix and it is wrong.** Before
finding the excerpt bug, the title block looked like the problem, so
`src/front_matter.py` was written to drop it from the candidate pool: 55 chunks
across the three corpora, about one per PDF, and no golden case names one.
Measured end to end it is worse on both corpora that have title pages:

| | any-hit | MRR | questions showing a title block |
|---|---|---|---|
| ML papers, filter off | **0.851** | **0.739** | 5 |
| ML papers, filter on | 0.836 | 0.724 | **0** |
| Quant, filter off | **0.886** | **0.779** | 2 |
| Quant, filter on | 0.857 | 0.764 | **0** |
| Ornithology, either | 0.846 | 0.614 | 0 |

Kept, default off, as a refuted experiment. The birds are unaffected because
Wikipedia articles have no title pages.

**And the reason it measures worse is the interesting part.** `bn-covariate`
carries `answer_contains: "internal covariate shift"` and gold pages that
include `batch_normalization.pdf` page 1. The title reads "Batch
Normalization: Accelerating Deep Network Training by Reducing Internal
Covariate Shift". So the page is a gold page **and** the answer string is
present, and both halves of the labelling scheme score the title block as a
correct answer. Removing it removes a hit.

**43 of the 102 answerable cases in the two corpora that have title pages have
a title block on one of their gold pages, and 17 of those have the answer string
inside the title itself.** The bird corpus is Wikipedia and has no title pages,
so it is outside that count. The measurement
cannot distinguish "returned the answer" from "returned the title of the paper
that contains the answer", and on those 17 a passage-level label would not fix
it either.

That is the same hole `HANDOFF.md` already records one level finer, where a
derived label proves the string is present rather than that the passage answers.
Here it is the page that is proved, not the passage.

**What was actually shipped, and how it was checked.** The excerpt change only.
It moves no retrieval metric, by construction, so it was verified by reading:
a heuristic pass over all 157 recorded answers flagged 31 whose top passage
looked like metadata rather than an answer, and 5 after. Of the 5, three are
prose that happens to cite several papers, and two are genuinely weak
retrievals scoring -1.09 and -1.15, near the floor. One of those two still
shows a title block, correctly: the question's words really are in that paper's
title and nowhere better in the chunk.

**The lesson, and it is not a new one here.** Rendering the interface and
reading it found this; no metric could have. The first fix that suggested
itself would have made the system worse while making the defect invisible, and
the measurement that stopped it is only trustworthy because the labels were
read afterwards to find out why it said no.

---

## 2026-08-31 — The title-block filter, settled

Dropping a paper's title block from the candidate pool scored 0.851 to 0.836
any-hit on the ML papers and 0.886 to 0.857 on quant, and the entry above
recorded it as refuted on that arithmetic. The reason offered was that the
labels credit a title block as a correct answer, since a title block sits on
page 1 and page 1 is a gold page for 43 of the 102 answerable cases in the two
corpora that have title pages.

`src/audit_title_credit.py` was written to find out how much of that reach is
actually collected, because a label that would credit a title block costs
nothing unless the pipeline returns one. It found two hits satisfied only by a
chunk flagged as front matter, `bn-covariate` and `qf-whale-attack`, and those
two account for the whole of the filter's loss to three decimals on both
any-hit and MRR, since 57/67 is 0.851 and 56/67 is 0.836, and 31/35 is 0.886
and 30/35 is 0.857.

**The first conclusion drawn from that was wrong, and the error is the useful
half.** The script reported the two hits as false credits, which was written
into five documents and deployed. `front_matter.is_front_matter` decides from
the head of a chunk, asking whether it opens with a heading and carries an
email address or an affiliation, so it is a statement about the first 600
characters. These chunks run to about 1,000 characters, and what follows a
title block is the abstract. Treating the flag as a verdict on the whole chunk
is the same mistake `_brief` made in the entry above, which took the head of a
passage for the passage.

Read whole, both chunks answer outright. The batchnorm chunk contains "We refer
to this phenomenon as internal covariate shift, and address the problem by
normalizing layer inputs", and the blockchain chunk contains "by introducing
certain detectability threshold, joining the attack can lead to strictly less
reward for whales", which is the label's answer string word for word. Both
credits are correct, and the corrected count of false credits is **zero**.

So the filter is settled against itself, and the original verdict stands for a
better reason than it was given. The two hits it removes are genuine, and it
removes them because a paper's first chunk carries the abstract, which is among
the most answer-dense text in the document. Dropping it destroys information
rather than withdrawing a scoring artefact. Reading what the filter does to the
five questions that touch it says the same thing: on `bn-covariate` the answer
moves from rank 1 to outside the top five entirely, and the passage that
replaces it discusses what normalizing an input can do to a layer's
representation without naming the problem the question asks about.

The filter is off, it stays off, and `src/front_matter.py` and
`src/sweep_front_matter.py` carry the reasoning where someone reaching for it
would look. What survives is the labelling weakness itself, which is real,
reaches 43 cases, and today costs nothing.

**Two smaller corrections found while checking.** The detector flags 56 chunks
and not the 55 recorded, 30 on the ML papers rather than 29, and one of the 56
is a genuine false positive: a page-1 figure in
`break_it_down_pass_it_on_cross_task_skill_transfer.pdf` listing interface
labels, which is document content rather than front matter. A second, the
reproduction-permission notice on `attention_is_all_you_need.pdf`, is not a
title block either, although it answers nothing.

---

## 2026-09-01 — A bigger generator, finished

The run recorded on 2026-08-31 stopped partway and covered 11 answers on the ML
papers alone. It reported that llama3.1:8b scored the same 4 of 7, refused one
adversarial question more than llama3.2, and cited worse. That entry said seven
answerable cases settle nothing, and finishing the run proved the point, because
two of the three claims were wrong in direction.

Completed across all three corpora on 2026-09-01, matched on the 43 cases both
models answered. The 8B run produced 13 of 15 on the ML papers and the two it
did not produce are excluded from both sides, since crediting a model on a case
the other never saw is the comparison this project keeps warning about.

| | llama3.2 (3B) | llama3.1 (8B) |
|---|---|---|
| correct | 14 of 28 | **16 of 28** |
| adversarial refused | **13 of 15** | 12 of 15 |
| invented citations | **0** | 1 |
| malformed citations | **0** | 2 |
| groundedness, lexical proxy | **0.588** | 0.476 |
| generation time, all three corpora | **98 s** | 5,665 s |

**Two questions better, and everything else worse.** The 8B model wins four
cases and loses two, for a net of two out of 28, which on a sample this size is
inside the noise that finding 6 of the README was written about. Against that it
misses one adversarial question the 3B refuses, emits the first invented
citation either model has produced, emits two malformed ones, draws 0.112 less
of its wording from the passages it was given, and takes 58 times as long.

**No change, and now on evidence rather than on a partial run.** The shipped
generator stays llama3.2. What the finished run adds beyond the verdict is that
the earlier one, on seven answerable cases, got the direction wrong on both the
correctness and the refusal comparison while sounding precise about each.

---

## 2026-09-01 — Four links to a repository that is not this one

Every Source link the interface serves named
`github.com/ArdellAlfatih/rag-project`. The remote is
`github.com/playingpossumm/rag-project`. The two are different repositories and
the first does not exist, so all four links 404 for a visitor, and have since
they were written.

**`check_links.py` passed throughout, and the reason is the interesting part.**
It matched each URL against a pattern named `SELF_REPO`, treated the match as
proof that the link pointed at this repository, and then checked the branch and
the file path against the local working tree. Both agreed, because `master` and
`docs/corpus-manifest.md` do exist here. The checker verified everything about
the link except the only part that was wrong.

That is the same shape as the entry the guard was written for, where a link
named a branch this repository does not have while the repository was private,
so one failure hid behind another. Here the hiding is done by the checker
itself: a name chosen for a pattern, `SELF_REPO`, was read as a fact about the
URL it matched.

**The fix is offline, which is what makes it a guard.** `git remote get-url
origin` gives the owner and repository name, and a link is now compared against
them before anything else is checked. When the repository name matches and the
owner does not, that is reported, since a link built on the wrong owner cannot
resolve. When both differ the link is another repository's and is left to
`--http`, which needs the network and is therefore not part of the default run.
Staged against the old links the guard reports both and exits non-zero.

The six links now name `playingpossumm/rag-project`, which is where the code
is. They still 404 for a visitor, because that repository is private, and that
is a decision for the owner rather than a defect in the interface. What has
changed is that the link names the right repository, so making it public is now
sufficient.

**And then the check was committed, which found one more defect.**
`src/test_links.py` stages both failures this guard exists for, along with the
route, anchor, branch and path cases it claims to cover, in a synthetic `ui/`
with `git` and the route list answered by stubs. 18 checks, hermetic, running
in milliseconds. Run against the version of the checker from before the fix, 4
of the 18 fail and 14 pass, which is the right split, because only the
repository half is new.

The fourth failure is not one that was staged deliberately. The old checker
treated any `github.com` URL as a link into this repository, so a link to
somebody else's repository had its branch checked against this working tree and
was reported broken for naming a branch that exists in that repository and not
in this one. There is no such link in `ui/` today, so the bug had never fired,
and the test says so before one is added rather than after.

The rule this satisfies is the project's own: a check performed and not
committed is indistinguishable, six days later, from one that never happened.

---

## 2026-09-01 — The twelve that fail everywhere, read one at a time

The documents have described these as two problems since 2026-08-27. Seven on
the ML papers ask for an attribute many papers share, so the topic matches a
dozen documents and the distinguishing clause is ignored, and five on birds and
quant describe a term and ask for its name, so the answer word is absent from
the question. That account was written once and quoted since, and no entry
records reading the twelve individually.

Read individually they are three problems, and only five are retrieval
failures.

**Four are cases the system answers and the label scores wrong.** On
`adam-bias`, "how does the optimizer correct for bias in its moment estimates",
rank 2 is `adam_optimizer.pdf` page 3, which reads "in algorithm 1 we therefore
divide by this term to correct the initialization bias". That is the answer, in
the right document, one page from the gold pages 5, 8 and 9, which are where
the hyphenated string "bias-correction" happens to appear. On `gpt3-fewshot`,
rank 2 is `gpt3.pdf` page 24, "in-context learning curves, which show task
performance as a function of the number of in-context examples", against gold
pages 7, 17 and 40. On `dropout-rate` rank 1 states a dropout rate, in a paper
the label does not list, and the question names no paper while many state one.

**`t5-text2text` is the sharpest of the four, because its gold is other
people's bibliographies.** The case asks how all NLP tasks are cast into a
single format and wants the string "unified text-to-text". Six chunks carry
that string, and four of them are reference-list entries in `colbert.pdf` page
10, `dense_passage_retrieval.pdf` page 11, `gpt3.pdf` page 73 and `rag.pdf`
page 14, each carrying the line "Colin Raffel, Noam Shazeer, Adam Roberts,
Katherine Lee, Sharan Narang, Michael Matena". The string is part of the T5
paper's own title, so citing T5 makes a page gold. Meanwhile `t5.pdf` page 8,
which says "we cast all of the tasks we consider into a text-to-text format",
is not gold and is what the pipeline returns.

**Five are genuine retrieval failures.** `wmt14` asks which dataset was used for
the English-to-German experiments and returns `bahdanau_attention.pdf` on
English-to-French and `seq2seq.pdf` on "the WMT'14 English to French dataset",
so the one word that decides the answer is the one ignored. `gpt3-params` asks
for the largest autoregressive model's parameter count and matches every
discussion of model size in every paper. `roberta-nsp-drop` asks which
pre-training objective was found unnecessary and removed, and `roberta.pdf`
does not appear in the top five at all. `bird-precocial` is answered in the
corpus by "the ducklings are precocial and fully capable of swimming as soon as
they hatch" and retrieval returns passages about hatching that never use the
word. `bird-hollow-bones` returns the right document twice, its Overview and
Axial skeleton sections, and misses the Skeletal system section that says
"birds have many bones that are hollow (pneumatized)".

**Three are questions the documents do not answer.** `bird-alula` asks which
small group of feathers prevents a stall at low speed. The word appears twice
in the corpus, in "the development of an enlarged, keeled sternum and the
alula" and "the presence of a pygostyle for tail feathers, and an alula on the
wing". Neither says the alula is a group of feathers, and neither mentions
stalling. `qf-mean-reversion` asks about prices pulled back toward a long-run
level, and all five occurrences of the term are about temperature in a
weather-derivatives paper, one of them a reference entry on page 43.
`qf-momentum` asks which effect describes past winners continuing to
outperform, and its 30 occurrences across seven documents are factor names such
as "12-month momentum" and bibliography entries. A derived label marks every
chunk containing the string, which cannot distinguish a passage that explains a
term from one that uses it, so these three look answerable and are not.

**A detector for reference lists was written, calibrated and abandoned.** The
six chunks carrying "unified text-to-text" separate cleanly on citation
furniture counted per 1,000 characters, with the four bibliography entries
scoring 13.2 to 25.6 and the two prose passages 0.0 and 1.0. Applied to the
corpora at a threshold of 8, inside that gap, it flags 14.7% of the ML papers
and 11.9% of quant, and reading the chunks nearest the line shows it is wrong
in both directions. It flags `rag.pdf` page 4, which is prose that cites
heavily, and two quant chunks that are mathematics, because a converted PDF
writes subscripts as `[24]` and the pattern cannot tell that from a citation
number. It misses author-year bibliographies that carry no brackets and no
arXiv identifier. Six examples separated and the corpus does not, which is the
same mistake the answer-quality judge made on 2026-08-30 when it counted
`[min(]` as an invented citation. No detector is shipped and no number from it
is quoted anywhere but here.

**What this changes.** The research frontier is five questions rather than
twelve. Four would pass under labels that marked the passages answering the
question rather than the passages containing a string, and three belong with
the adversarial cases, since the documents do not answer them. Relabelling
would raise the published figures, so nothing was changed here and the decision
is recorded for the owner rather than taken.

**And the rule it repeats.** This is the fourth time in this repository that a
number describing a weakness turned out to describe the labels: the quant
corpus at 0.543, the title block scored as a correct answer, the two hits
called false credits, and now the twelve. Each was found by reading the cases
and none by looking at an aggregate.

---

## 2026-09-01 — The labels corrected, and every number that moved with them

The entry above read the twelve cases that fail under every configuration and
found four the pipeline answers while the label scores them wrong, and three
the documents do not answer at all. It recorded them and changed nothing,
because correcting them raises the published figures. The owner then asked for
the technicals fixed to the fullest extent, so they are corrected here.

**What changed in the labels.** Four answer strings, each replaced with one that
appears where the question is answered, and each candidate checked against the
corpus and read before use:

| case | was | is |
|---|---|---|
| `adam-bias` | "bias-correction", on pages discussing the effect of the terms | "correct the initialization bias", on the page that says how |
| `gpt3-fewshot` | "zero-shot, one-shot", on pages naming the three settings | "as a function of the number of in-context examples" |
| `t5-text2text` | "unified text-to-text", four of six chunks bibliographies | "cast all of the tasks", on t5.pdf pages 8 and 9 |
| `dropout-rate` | an ambiguous question many papers answer | "Which dropout rate was used for the English-to-French model instead of 0.3?" |

`dropout-rate` is the odd one: its gold was correct and its question was not.
Many papers state a dropout rate, so a passage reading "we use a dropout
probability of 0.1 everywhere" answered it and scored zero. It is now made
discriminating by content, which is the rule `build_golden_set.py` states at the
top of its own case list and this question had never followed.

`bird-alula`, `qf-mean-reversion` and `qf-momentum` moved to the adversarial
half as near-miss cases, the same class as `adv-price`, where compute cost is
reported in FLOPs and never in currency. **The abstention gate agrees**: at
their shipped thresholds all three score below the cut, at -7.28, -7.21 and
-8.61, so the system already declined to answer them and the golden set was
calling that a failure.

**A bug in the label derivation, found by rebuilding.** `derive_gold` keyed its
accumulator on the source alone, took the locator kind from whichever matching
chunk came first, and filed every later value under that kind. `bird_anatomy.docx`
carries "hollow bones" in a section named "Skeletal system" and in table 1, so
the rebuilt gold read `{"kind": "section", "pages": [1, "Skeletal system"]}`.
Section 1 does not exist and table 1, which does, became unmatchable.
`check_golden.py` reported both the moment the set was rebuilt, which is the
guard doing exactly what it was written for. The comment three lines below the
bug already said "the KIND travels with the value"; the code keyed on source and
did not.

**What the numbers did.** Every figure below is the shipped configuration, and
no line of retrieval code changed:

| | any-hit | MRR | NDCG | source recall |
|---|---|---|---|---|
| ML & NLP papers | 0.851 → **0.910** | 0.739 → 0.784 | 0.755 → 0.804 | 0.761 → 0.785 |
| Ornithology | 0.846 → **0.880** | 0.613 → 0.638 | 0.671 → 0.698 | 0.762 → 0.792 |
| Quantitative finance | 0.886 → **0.939** | 0.779 → 0.826 | 0.786 → 0.833 | 0.769 → 0.816 |

**And every point of that reconciles, which is the only reason to trust it.** The
ML corpus went from 57 hits of 67 to 61 of 67, and the four it gained are the
four relabelled cases; 61/67 is 0.910 and 57/67 is 0.851. The other two corpora
kept their numerators exactly: birds was 22 of 26 and is 22 of 25, quant was 31
of 35 and is 31 of 33. Nothing was retrieved that was not retrieved before. The
bird and quant rises are entirely denominator, which is to say they are the
score no longer counting questions the documents cannot answer as retrieval
failures.

**Read the corpora table as a different measurement from the one it replaces.**
0.910 is not an improvement on 0.851. It is what 0.851 was measuring once four
labels stopped disagreeing with their own passages.

**What had to be re-run.** `evaluate.py` on three corpora, `per_case.py` on
three, `build_analytics.py`, `build_results_doc.py`, `uniform_baseline.py` and
`sweep_fusion.py` for the birds, and then 55 numbers across `HANDOFF.md`, the
README, `docs/roadmap.md`, `corpora.json` and the site. `check_docs.py` named
every one of the 55 and confirmed the rewrite, which is the whole argument for
having built it: a labelling change touches numbers in five files and no one
remembers all of them.

**The corpus notes were recounted rather than patched.** Each argues for a
threshold from counts over its own golden set, so the numerators move with the
denominators: the birds note now reads six of seven adversarial caught and two
of twenty-five wrongly refused, and quant reads seven of eight and one of
thirty-three. Figures in those notes that were measured against the earlier sets
are now marked with the date they were taken, because a superseded measurement
is a record and an unmarked one is a mistake.

**What did not change.** The five genuine retrieval failures are still failures,
and nothing here was tuned to make them pass. `src/failure_overlap.py`,
re-derived afterwards from six configurations on each corpus, returns exactly
the five the reading identified: `gpt3-params`, `roberta-nsp-drop` and `wmt14`
on the papers, `bird-hollow-bones` and `bird-precocial` on the birds, and none
on quant. That is the harness agreeing with the reading rather than repeating
it, and it is the check that the four corrections did not quietly make a
genuine failure pass. The ML gate still lets 8 of 17
adversarial cases through. The thresholds, blends, candidate pools and the
ensemble are all where they were.

---

## 2026-09-01 — Five failures, diagnosed by stage, and four levers refuted

The five that survive every configuration were treated as one problem. Asking
where each loses the answer splits them, and the split decides what could
possibly help.

**Where the answer is lost.** For each case, the first gold chunk's rank in the
dense ranking, in BM25, in the pool the reranker is handed, and in the final
five:

| case | dense | BM25 | shipped pool | final |
|---|---|---|---|---|
| `wmt14` | 138 | **11** | absent | missed |
| `gpt3-params` | 23 | 16 | absent | missed |
| `roberta-nsp-drop` | 35 | 167 | absent | missed |
| `bird-precocial` | 12 | 55 | absent | missed |
| `bird-hollow-bones` | 7 | 20 | **8** | missed |

**Four of the five never reach the reranker**, and the reason is visible in
`wmt14`: BM25 ranks the answer 11th and dense ranks it 138th, and RRF rewards
agreement, so a chunk one retriever likes and the other does not loses to
chunks both rank in the middle. That is the same correlation dilution recorded
when a second embedder was fused. Only `bird-hollow-bones` is in the pool and
dropped afterwards, and the diversity cap is not what drops it: capped and
uncapped return an identical top five.

**Four levers, re-measured on the corrected labels, and all four refuted.**
Every one of these had been ruled out before against labels now known to be
wrong, so none of the earlier rulings could be quoted.

1. **A deeper candidate pool.** 28 candidates scores 0.895 on the ML papers
   against 0.910 at 16, and buys nothing on the other two while their pool
   ceilings reach 1.000. A fifth instance of pool recall being a ceiling and
   not a proxy. At 40 the pool does recover `gpt3-params`, and loses
   `lora-frozen`, for 60 of 67 either way.
2. **The rerank blend.** The shipped values are already the best available:
   0.00 on the papers, 0.20 on birds, 0.00 on quant.
3. **Another cross-encoder.** `bge-reranker-base` costs 0.045 any-hit on the
   papers and 0.080 on birds. `ms-marco-MiniLM-L-12-v2` gains 0.015 on the
   papers and loses 0.040 on birds. Two gate-only combinations move nothing.
4. **Scoring the best window inside a chunk rather than the whole chunk**, the
   same idea that fixed the excerpt on screen. It raises the gold chunk's score
   on `bird-hollow-bones` from -1.13 to -0.74 and moves it no places.

**And one that came closer than anything before it.** Four of the five ask for
a word the question does not contain, which is exactly why rewriting and
decomposition failed: rewriting a question that lacks a word produces another
question that lacks it. A hypothetical answer is different in kind. Asked to
write the passage from memory, a model produces text containing the term, and
the search runs on that. `src/sweep_hyde.py` measures it on every case.

On the five it recovers three, `wmt14`, `roberta-nsp-drop` and
`bird-hollow-bones`, which nothing else has moved. The mechanism is exactly
visible: it hits when the generated passage contains the answer term and misses
when it does not. llama3.2 writes "WMT 2014 English-German" and "hollow bones"
correctly, and for the three it misses it writes about autoregressive models of
order p, says masked language modelling was the objective removed from BERT,
and invents "neonatal mobility" for precocial chicks.

Measured on every case it is much worse:

| | control | hypothetical | question and hypothetical |
|---|---|---|---|
| ML & NLP papers | **0.910** | 0.806, +2 -9 | 0.851, +3 -7 |
| Ornithology | 0.880 | **0.920**, +2 -1 | **0.920**, +2 -1 |
| Quantitative finance | **0.939** | 0.788, -5 | 0.818, -4 |

It also does the thing the sweep was written to watch for. A hypothetical
answer is written just as confidently for a question the corpus cannot answer,
so adversarial cases slip the gate more often: 8 of 17 becomes 11 on the
papers, 1 of 7 becomes 3 on birds, and 1 of 8 becomes 6 on quant. And it costs
about 14 seconds a query against 1.2.

**Not shipped, including on the birds.** The bird corpus gains a question net,
and triples the adversarial cases that slip the gate to get it. Refusing a
question the corpus cannot answer is the property this project has spent the
most measurement defending, and one question of any-hit on 25 cases is thinner
evidence than three of seven slipping.

**What is left, stated plainly.** Every lever in this repository has now been
measured against these five on correct labels, and none of them ships. Three of
the five are reachable by a language model that knows the answer already, which
is a statement about the model and not about the retrieval. `gpt3-params` is
reachable by a pool of 40 at the cost of another question. `bird-precocial` is
reached by nothing.

---

## 2026-08-27 — Smaller things

- `compare_rerankers.py` crashed **after** writing its results, on
  `Path.relative_to` with a relative `--emit`. A finished measurement looked
  like a failed run. Guarded.
- A Playwright screenshot script died on `UnicodeEncodeError` printing a marked
  run containing `ϵ`. The console is cp1252; `sys.stdout.reconfigure` is
  required in anything that prints corpus text, and the handoff says so.
- The first bird-corpus screenshot of this session asked a bird question against
  the ML corpus. It refused, at −10.86, and said so plainly. That is the gate
  doing its job rather than a defect, caught on camera.

---
