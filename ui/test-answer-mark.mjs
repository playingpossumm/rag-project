/* Does the answer highlight keep its promises, on every passage we have?

   The highlight is the one part of the interface a reader examines word by
   word, and until 2026-08-27 it lived inline in a two-thousand-line HTML file
   where nothing could run it without a browser. Nothing did. Every claim about
   it -- "only the answering words", "at most a sentence" -- was a claim about
   code no test had ever executed.

   Two halves, and the second is the one that matters:

   **The contract**, on hand-written cases where the right answer is obvious.
   **The sweep**, over eval/top-passages.json: the real top passage for every
   question in all three golden sets, marked by the real function. A highlight
   can satisfy every unit test and still set half of a real passage bold,
   because real passages are tables, headings, and sentences that never end.

   Run:  node ui/test-answer-mark.mjs
   The sweep needs:  .venv\Scripts\python.exe src\dump_top_passages.py
*/
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { clean, esc, fragment, markAnswer, splitSentences,
         MAX_MARK_WORDS, MAX_MARK_CHARS, MAX_MARK_SHARE } from "./answer-mark.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const CHECKS = [];
const check = (name, got, want) =>
  CHECKS.push({ name, got, want, ok: JSON.stringify(got) === JSON.stringify(want) });

const marked = html => [...html.matchAll(/<mark>([\s\S]*?)<\/mark>/g)].map(m => m[1]);
const stripped = html => html.replace(/<\/?mark>/g, "");
const wordsIn = s => (s.trim().match(/\S+/g) || []).length;

/* ---------------------------------------------------------- the contract -- */

// Nothing may be lost or duplicated. This is the invariant that makes every
// other check meaningful: whatever the highlight decides, the passage a reader
// sees is the passage that was retrieved.
const passage = "The clavicles are fused to form a bone called the furcula. "
  + "Birds also have a pygostyle, which supports the tail feathers.";
const q = "What is the fused collarbone of a bird called?";
check("the passage survives marking, character for character",
      stripped(markAnswer(passage, q)), esc(passage));

check("exactly one run is marked", marked(markAnswer(passage, q)).length, 1);

check("the mark is at most MAX_MARK_WORDS words",
      wordsIn(marked(markAnswer(passage, q))[0]) <= MAX_MARK_WORDS, true);

// The promise in the section heading: a highlight points into a sentence, it
// does not span two. A mark crossing a boundary reads as a paragraph set bold.
const sentences = splitSentences(passage);
check("the mark falls inside a single sentence",
      sentences.some(s => esc(s).includes(marked(markAnswer(passage, q))[0])), true);

// A long sentence with a figure anchors on the figure, not on the whole clause.
const figure = "For all of our experiments we employed label smoothing of value "
  + "0.1, which hurts perplexity as the model learns to be more unsure, but "
  + "improves accuracy and BLEU score on the development set.";
const figMark = marked(markAnswer(figure, "What label smoothing value was used?"))[0];
check("a question about a figure marks the figure", figMark.includes("0.1"), true);
check("and does not mark the whole sentence",
      wordsIn(figMark) <= MAX_MARK_WORDS, true);

// The rule that stops the highlight pointing at nothing. Asked what a bird's
// fused collarbone is called, the marked words were once "this fused structure"
// -- one query word and a pronoun standing in for the noun being asked about.
check("a run carrying one query word and no figure is not marked",
      markAnswer("This fused structure is discussed at length in the "
                 + "literature and has been for a considerable time.",
                 "What is the fused collarbone of a bird called?").includes("<mark>"),
      false);

// The path that used to set an unbounded run bold: a chosen sentence with no
// question word in it at all. Marking there points at nothing, at length.
check("a sentence with no question word is left unmarked",
      markAnswer("Something entirely unrelated appears here at some length, "
                 + "going on for rather more than twelve words in total.",
                 "syrinx furcula pygostyle").includes("<mark>"),
      false);

// Escaping, inside the mark and outside it. A passage carrying < or & is
// ordinary in a corpus of papers.
const nasty = 'The furcula & the "wishbone" <b>are</b> the same fused clavicle bone.';
const out = markAnswer(nasty, "What is the fused clavicle called?");
check("no raw angle bracket survives outside the mark tags",
      /<(?!\/?mark>)/.test(out), false);
check("and the escaped text still round-trips", stripped(out), esc(nasty));

// Passages that are not prose at all. A chunk is 210 tokens sliced from a PDF,
// so it is regularly a table row, a heading, or a sentence with no end.
check("a passage with no sentence terminator is still bounded",
      wordsIn(marked(markAnswer(
        "syrinx vocal organ located at the base of the trachea unique to birds "
        + "producing song without vocal cords across many families worldwide",
        "What is the syrinx and where is it located?"))[0] || "") <= MAX_MARK_WORDS,
      true);

check("an empty passage does not throw", markAnswer("", q), "");
check("a question of only stop words marks nothing",
      markAnswer(passage, "what is the of").includes("<mark>"), false);

// Abbreviations are not sentence ends. A false boundary cuts the answer in
// half rather than running past it, which is why it went unnoticed.
check("et al. does not end a sentence",
      splitSentences("Vaswani et al. (2017) introduced it. It uses attention.").length, 2);
check("Fig. does not end a sentence",
      splitSentences("See Fig. 3 for the curve. Training took 12 hours.").length, 2);
check("a real boundary still splits",
      splitSentences("The furcula is fused. The pygostyle is not.").length, 2);

/* A colon ends the clause. Marking past it picks up whatever it introduces,
   which on an academic passage is a citation. */
{
  const passage = "Our text-to-text framework follows previous work that casts "
    + "multiple NLP tasks into a common format: McCann et al. propose the "
    + "Natural Language Decathlon, a benchmark that uses a consistent "
    + "question-answering format for a suite of ten NLP tasks.";
  const run = marked(markAnswer(passage, "How are all NLP tasks cast into a single format?"))[0] || "";
  check("a mark does not run past a colon", /[:;]/.test(run.slice(0, -1)), false);
  check("and it does not end on a citation name", /et al\.?$/.test(run.trim()), false);
}


/* A colon introduces a definition as often as it introduces a citation, and
   the definition is the half worth marking. Cutting at every colon threw away
   the query words after it, left one behind, and failed the two-word check, so
   the passage showed NO mark at all -- two bird answers lost their highlight
   this way. Reported by reading the bird corpus, 2026-09-03. */
{
  // The recorded passage in full. A trimmed version made the answering
  // sentence 70% of the passage, which MAX_MARK_SHARE refuses by design, so
  // the check failed against correct code. Here it is 28%, as on the page.
  const passage = "Eggs and nests The chicks of passerines are altricial: "
    + "blind, featherless, and helpless when hatched from their eggs. Hence, "
    + "the chicks require extensive parental care. Most passerines lay colored "
    + "eggs, in Clutches vary considerably in size: some larger passerines of "
    + "Australia such as lyrebirds and scrub-robins lay only a single egg, most "
    + "smaller passerines in warmer climates lay between two a The family "
    + "Viduidae do";
  const q = "What term describes chicks that hatch helpless and wholly dependent on their parents?";
  const runs = marked(markAnswer(passage, q));
  check("a definitional colon does not suppress the mark entirely", runs.length > 0, true);
  check("and the mark reaches past the colon to the words that define it",
        /helpless|hatched/i.test(runs[0] || ""), true);
}

/* ---- measured 2026-09-05, by ui/audit-marks.mjs ------------------------- */

/* The span started 1 word before the first question word and ran 6 words
   past it, so where the answer FOLLOWS the question words the mark stopped
   short. Of the 30 audit misses under that code, 19 marks sat in the sentence
   that contains the answer, and in 15 of those the answer began at or after
   the point where the mark stopped (11 wholly after it, 4 overlapping its
   end). An earlier version of this comment gave 16, a count made by reading
   the list rather than by locating each mark in its sentence. The span now
   runs on to the end of its phrase. In each passage below the sentence that
   answers is the recorded one; the sentences around it are written for the
   test, so that the answering sentence stays under half the passage and the
   scorer has a choice to make. */
{
  const passage = "Our main results are as follows. We show that the CVaR "
    + "threshold can be restricted to a common compact interval and establish "
    + "existence of a primal optimizer. The proof uses a compactness argument "
    + "on the feasible set of thresholds together with a continuity bound on "
    + "the objective, and neither step depends on the choice of confidence level.";
  const run = marked(markAnswer(passage, "Can the CVaR threshold be restricted to a bounded range?"))[0] || "";
  check("the mark runs on to the end of the answering phrase",
        run.includes("common compact interval"), true);
  check("and does not end on a stopword", /\b(and|the|of|to|a)$/i.test(run), false);
}
{
  const passage = "Bayesian methods have a long history in macroeconomic "
    + "forecasting. BVAR, when complemented with machine learning, enables "
    + "probabilistic forecasts instead of singlepoint estimates, while AI-driven "
    + "sentiment analysis improves real-time responsiveness to economic signals. "
    + "Both directions are pursued in the sections that follow, with the data "
    + "described first.";
  const run = marked(markAnswer(passage, "What does combining BVAR with machine learning make possible?"))[0] || "";
  check("a run-on reaches an answer 5 words past the last question word",
        run.includes("probabilistic forecasts instead of singlepoint"), true);
}
{
  const passage = "Several ratios summarise the risk side of a strategy, and "
    + "each answers a different question about the path of returns. The Calmar "
    + "ratio compares the selected return measure with the worst peak-to-trough "
    + "decline, i.e., maximum drawdown (MDD), capturing path-dependent downside "
    + "risk. The Sortino ratio instead penalises only downside deviation.";
  const run = marked(markAnswer(passage, "Which measure reports the largest peak-to-trough decline of a strategy?"))[0] || "";
  check("a run-on crosses an \"i.e.,\" to reach the term it introduces",
        run.includes("maximum drawdown"), true);
}

/* A question of the form "what is X called" asks for the one word it does not
   contain, so the sentence that answers often carries a single question word,
   and the two-word rule refused it: asked what the period of sitting on eggs
   is called, "helping with the incubation of the eggs" matched only on "eggs"
   and nothing was marked. For a definitional question the clause around the
   one word is now marked, subject to the guards tested after it. The two-word
   rule itself stays, and the "this fused structure" check above still holds.

   The passage is the recorded lead passage for birds/bird-incubation without
   its last 2 sentences. On the full passage the scorer picks "After hatching,
   the chicks (called "eyases") are covered with creamy-white down", because
   that sentence carries 2 question words, "hatching" and "called", so the
   audit still classifies the recorded case as a MISS. The clause rule fixes
   the span, not the choice of sentence. */
{
  const passage = "They are incubated for 29 to 33 days, mainly by the female, "
    + "with the male also helping with the incubation of the eggs during the "
    + "day, but only the female incubating them at night. The average number of "
    + "young found in nests is 2.5, and the average number that fledge is about "
    + "1.5, due to the occasional production of infertile eggs and various "
    + "natural losses of nestlings.";
  const q = "What is the period of sitting on eggs before hatching called?";
  const run = marked(markAnswer(passage, q))[0] || "";
  check("a definitional question marks the clause around its one question word",
        run.includes("incubation"), true);
  check("and that clause does not open on a pronoun",
        /^(this|that|these|those|it|its|they|them|such)\b/i.test(run), false);
}

/* The guards on that exception. Each was measured on 2026-09-05 by disabling
   it and sweeping the 450 real passages. The marker guard stopped 1 mark,
   "also called the preen gland." for the dawn-chorus question; the 5-word
   floor stopped 1, "brown-blotched eggs." for the incubation question; the
   whole-word check stopped 2, "naked, unable to lift their head and totally
   helpless" for the precocial question ("able" inside "unable") and the
   snipes' "display flight" clause for the dawn-chorus question ("light"
   inside "flight"); and the participle guard stopped 2 for the flyway
   question, "so the established Linnean system is followed here." and a
   clause about brent geese "migrating between the Taymyr Peninsula". The
   passages here are written to carry the same shapes; they are not the
   recorded text. */
check("the question's own marker word is not a pointer",
  markAnswer("The gland sits at the base of the tail, also called the preen "
             + "gland. It secretes an oil that birds spread with the bill across "
             + "the feathers while preening, which keeps them flexible and dry.",
             "What is the burst of collective singing at first light called?").includes("<mark>"),
  false);
check("a fragment carrying one question word is not marked",
  markAnswer("The nest is a scrape on a ledge, lined with a few stems, holding "
             + "three or four cream, brown-blotched eggs. The female does most of "
             + "the work and the male brings food to the ledge through the day.",
             "What is the period of sitting on eggs before hatching called?").includes("<mark>"),
  false);
check("a substring match does not count as the one question word",
  markAnswer("The drumming of woodpeckers and the winnowing of snipes' wings in "
             + "display flight are mechanical sounds rather than song. Both carry "
             + "over long distances in open country and serve the same purpose.",
             "What is the burst of collective singing at first light called?").includes("<mark>"),
  false);
check("a participle is not the one question word",
  markAnswer("Some recent sources apply the phylogenetic taxon Spheniscidae to "
             + "what is here referred to as Spheniscinae, and the two arrangements "
             + "differ only in rank, so the established Linnean system is followed "
             + "here. The relationships of the subfamilies to each other remain "
             + "unresolved, and the arrangement below is provisional.",
             "What is the established route a migrating population follows called?").includes("<mark>"),
  false);

/* ---- reported 2026-09-05, by review of the run-on ----------------------- */

/* The first run-on trimmed trailing stopwords by their letters alone and had
   no rule for what it ran into. Over the 450 sweep passages it raised marks
   with an unbalanced bracket from 4 to 8, marks ending on an operator from 2
   to 5 and marks carrying a URL from 1 to 2. In each passage below the
   sentence that carries the defect is recorded in eval/top-passages.json and
   the sentences around it are written. */
{
  // qf-mean-variance#2. "at)." closes a bracket, so it stays whatever its
  // letters say; the first trim stripped it and left "(α =" open. The second
  // sentence is written without the recorded "covariance", which carries
  // "variance" as a substring and would win the sentence for itself.
  const passage = "Conversely, for Dirac-3 and Gurobi, the policy network acts as a "
    + "signal generator whose continuous action vector represents an expected "
    + "return forecast (α = at). This expected return signal, along with the "
    + "shrinkage matrix Σ t and the preceding binary allocation vector x prev, "
    + "is assembled into a sparse Quadratic Unconstrained Binary Optimization "
    + "(QUBO) matrix Q.";
  const run = marked(markAnswer(passage, "Which framework trades expected return against portfolio variance?"))[0] || "";
  check("a stopword that closes a bracket is kept", run.includes("(α = at)"), true);
}
{
  // qf-max-drawdown#1. The run-on reached "(β2 =" and stopped on it.
  const passage = "Results Overview Exhibit 2 reports peak metrics per pipeline "
    + "across the β1 sweep (β2 = 0 throughout). The results reveal "
    + "differentiated performance profiles that vary by metric and "
    + "configuration. No single pipeline dominates on all metrics simultaneously.";
  const run = marked(markAnswer(passage, "Which measure reports the largest peak-to-trough decline of a strategy?"))[0] || "";
  check("a mark does not end on a bracket it opens or on an operator",
        /[(\[=]$/.test(run.trim()), false);
  check("and it leaves no bracket open",
        run.split("(").length > run.split(")").length, false);
}
{
  // adv-latency-ms#0. The chunk ends in a footnote URL and a rule of dashes.
  const passage = "For the mostly CPU-based retrieval experiments in §4.3 and the "
    + "indexing experiments in §4.5, we use another server with the same CPU "
    + "and system memory specifications but which has four Titan V GPUs "
    + "attached, each with 12 GiBs of memory. Across all experiments, only one "
    + "GPU is dedicated per query for 5htps://github.com/huggingface/transformers -----";
  const run = marked(markAnswer(passage, "What is the end-to-end latency in milliseconds per query?"))[0] || "";
  check("a run-on stops before a URL", /:\/\/|-----/.test(run), false);
}
{
  // vit-inductive#0. A footnote marker fused to the next sentence's first word.
  const passage = "When trained on mid-sized datasets such as ImageNet without strong "
    + "regularization, these models yield modest accuracies of a few percentage "
    + "points below ResNets of comparable size. This seemingly discouraging "
    + "outcome may be expected: Transformers lack some of the inductive biases "
    + "1Fine-tuning code and pre-trained models are available at github.";
  const run = marked(markAnswer(passage, "What inductive biases do convolutional networks have that transformers lack?"))[0] || "";
  check("a run-on stops before a fused footnote marker", /\d[A-Z][a-z]/.test(run), false);
  check("and still reaches the phrase it was after", run.includes("inductive biases"), true);
}
{
  // roberta-dynamic#1. The 12-word cap cut the span before its last question
  // word, and with the trim's floor beyond the end nothing was trimmed.
  const passage = "C.2 Ablation for Different Masking Procedures In Section 3.1, we "
    + "mention that BERT uses a mixed strategy for masking the target tokens "
    + "when pre-training with the masked language model (MLM) objective. We "
    + "compare the procedures on the development sets in Table 8.";
  const run = marked(markAnswer(passage, "What masking strategy is applied differently at each epoch?"))[0] || "";
  check("a span cut by the word cap is still trimmed of a trailing stopword",
        /\b(a|the|of|to|and)$/i.test(run), false);
}

/* -------------------------------------------------------------- the sweep -- */

/* ---- reported 2026-09-01, by reading the page -------------------------- */

/* A citation is apparatus for a reader with a bibliography, and this reader
   has none. Numeric markers were already stripped and author-year ones were
   not, so "(Houlsby et al., 2019; Rebuffi et al., 2017)" sat inside a marked
   clause. */
check("an author-year citation is stripped",
  clean("introduce inference latency (Houlsby et al., 2019; Rebuffi et al., 2017) by extending depth"),
  "introduce inference latency by extending depth");
check("a citation list with an ampersand is stripped",
  clean("usable sequence length (Li & Liang, 2021; Lester et al., 2021) overall"),
  "usable sequence length overall");
check("a pointer to another part of the paper is stripped",
  clean("posing a trade-off (Section 3). More importantly"),
  "posing a trade-off. More importantly");

/* And the mirror image: a parenthetical that explains something is part of the
   sentence, not apparatus, and a rule that removed it would be worse than the
   defect it fixes. */
check("an explanatory parenthetical is kept",
  clean("bones that are hollow (pneumatized) with struts"),
  "bones that are hollow (pneumatized) with struts");
check("a parenthetical with no year and no et al is kept",
  clean("adapts deep LMs (in particular, BERT) for retrieval"),
  "adapts deep LMs (in particular, BERT) for retrieval");

/* "Appendix D." ends a sentence. The abbreviation rule read the single capital
   as an initial, glued two sentences into one, and the mark then landed on a
   clause about previous work rather than on the sentence being asked about. */
check("a labelled single capital ends a sentence",
  splitSentences("for every task we studied in Appendix D. Our text-to-text framework follows previous work.").length,
  2);
check("an initial in a name still does not end a sentence",
  splitSentences("Introduced by Vaswani, A. Barret and others in 2017.").length,
  1);

/* A sentence that names the thing asked about beats one that merely contains
   the same words apart. Asked what late interaction is, the mark landed on
   "a novel ranking model that adapts deep LMs ... for efficient retrieval"
   and left "ColBERT introduces a late interaction architecture" unmarked. */
{
  const passage = "To tackle this, we present ColBERT, a novel ranking model that "
    + "adapts deep LMs (in particular, BERT) for efficient retrieval. ColBERT "
    + "introduces a late interaction architecture that independently encodes "
    + "the query and the document using BERT and then employs a cheap yet "
    + "powerful interaction step that models their fine-grained similarity.";
  // Truncated at 260, which is where the excerpt used to stop and where the
  // defect appeared: with the second sentence cut short the scorer preferred
  // the first, which merely contains "retrieval" and "model". A complete
  // sentence is chosen correctly by either version, so testing the whole
  // passage would pass against the code this check exists to catch.
  const out = markAnswer(passage.slice(0, 260),
                         "What is late interaction in a retrieval model?");
  const run = marked(out)[0] || "";
  check("the mark lands on the sentence naming the phrase asked about",
    run.includes("late interaction"), true);
}

/* A mark is a pointer into a passage, so it cannot be the passage. On a
   passage that is one short sentence every absolute bound was satisfied and
   the mark still covered all of it. */
{
  const passage = "Though models like GPT-3 consume significant resources during training";
  const out = markAnswer(passage, "How many resources does GPT-3 consume during training?");
  const run = marked(out)[0] || "";
  check("a mark never covers more than half its passage",
    run.length <= passage.length * MAX_MARK_SHARE, true);
}

const dump = join(HERE, "..", "eval", "top-passages.json");
if (!existsSync(dump)) {
  console.log("  (no eval/top-passages.json -- run src/dump_top_passages.py "
    + "for the sweep over real passages)");
} else {
  const rows = JSON.parse(readFileSync(dump, "utf8"));
  let n = 0, unmarked = 0, overLong = 0, crossed = 0, lost = 0;
  const shares = [];
  const worst = [];

  for (const row of rows) {
    for (const raw of [row.text, ...(row.also || [])]) {
      // fragment() is what the page hands markAnswer: the extractor's markdown
      // stripped, reference markers dropped, ellipses added where the chunk was
      // cut mid-sentence. Sweeping the raw chunk instead would measure a string
      // no reader ever sees.
      const text = fragment(raw);
      n += 1;
      const html = markAnswer(text, row.question);
      if (stripped(html) !== esc(text)) lost += 1;
      const runs = marked(html);
      if (!runs.length) { unmarked += 1; continue; }
      const words = wordsIn(runs[0]);
      if (runs[0].length > MAX_MARK_CHARS) {
        overLong += 1;
        worst.push({ id: row.id, words, run: runs[0].slice(0, 60), chars: runs[0].length });
      } else if (words > MAX_MARK_WORDS) {
        overLong += 1;
        worst.push({ id: row.id, words, run: runs[0].slice(0, 60) });
      }
      const parts = splitSentences(text).map(esc);
      if (!parts.some(p => p.includes(runs[0]))) {
        crossed += 1;
        worst.push({ id: row.id, words, run: runs[0].slice(0, 60), crossed: true });
      }
      shares.push(runs[0].length / Math.max(esc(text).length, 1));
    }
  }

  shares.sort((a, b) => a - b);
  const pct = p => shares.length
    ? (shares[Math.min(shares.length - 1, Math.floor(p * shares.length))] * 100).toFixed(1)
    : "n/a";

  console.log(`\n  swept ${n} real passages from ${rows.length} questions`);
  console.log(`    marked           ${n - unmarked}  (${unmarked} left unmarked)`);
  console.log(`    share of passage set bold: median ${pct(0.5)}%, `
    + `90th ${pct(0.9)}%, max ${pct(0.999)}%`);
  for (const w of worst.slice(0, 5)) {
    console.log(`    ${w.crossed ? "CROSSED" : "TOO LONG"} ${w.id}: `
      + `${w.words} words -- ${JSON.stringify(w.run)}`);
  }

  check("no real passage loses or duplicates text", lost, 0);
  check("no real mark exceeds the word or character bound", overLong, 0);
  // The bound that the 82.7% case satisfied and still failed the reader: a
  // mark covering most of its passage is not pointing at anything.
  check("no real mark covers more than half its passage",
        shares.filter(x => x > 0.5).length, 0);
  check("no real mark spans more than one sentence", crossed, 0);
}

/* --------------------------------------------------------------- report --- */

const width = Math.max(...CHECKS.map(c => c.name.length));
let failed = 0;
for (const c of CHECKS) {
  if (!c.ok) {
    failed += 1;
    console.log(`FAIL  ${c.name.padEnd(width)}  got ${JSON.stringify(c.got)}, `
      + `want ${JSON.stringify(c.want)}`);
  }
}
console.log(`${CHECKS.length - failed}/${CHECKS.length} answer-highlight checks passed`);
process.exit(failed ? 1 : 0);
