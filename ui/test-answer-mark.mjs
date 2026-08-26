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

import { esc, fragment, markAnswer, splitSentences,
         MAX_MARK_WORDS, MAX_MARK_CHARS } from "./answer-mark.js";

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

/* -------------------------------------------------------------- the sweep -- */

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
