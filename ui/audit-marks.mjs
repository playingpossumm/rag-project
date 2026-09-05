/* Where the answer highlight lands, measured against the golden answers.

   Until 2026-09-05 the only measurement of the highlight was whether a mark
   existed and how much of the passage it covered (ui/test-answer-mark.mjs).
   Nothing checked that the marked words were the answer. This script does,
   for every recorded confident question in static-demo, by comparing the
   marked text against the golden set's answer_contains.

   A case is eligible when the lead passage contains the answer string, after
   the same normalisation src/evaluate.py applies in answer_normalize. Passages
   that do not contain the answer cannot be scored, since no mark could be
   right. Eligible cases fall into 3 classes.

     HIT   a mark exists and its text contains the answer
     MISS  a mark exists and its text does not contain the answer
     NONE  no mark

   Run:  node ui/audit-marks.mjs
   The exit code is always 0, because the script measures the highlight
   rather than gating on it. */
import { readdirSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { clean, markAnswer } from "./answer-mark.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");

/* src/evaluate.py answer_normalize, transcribed. The dash class is U+2010 to
   U+2015 (hyphen, non-breaking hyphen, figure dash, en dash, em dash,
   horizontal bar) plus U+2212 (minus sign), which is the range the Python
   source writes as a literal character class. */
const DASH = /[‐-―−]/g;
const MARKUP = /[_*`#]+|\[[^\]]{0,3}\]/g;
export const answerNormalize = s => String(s)
  .replace(DASH, "-")
  .replace(MARKUP, "")
  .replace(/\s+/g, " ")
  .trim()
  .toLowerCase();

const CORPORA = ["birds", "llm", "quant"];
const GOLDEN = ["golden-birds.json", "golden_set.json", "golden-quant.json"];

const golden = new Map();
for (const name of GOLDEN) {
  const g = JSON.parse(readFileSync(join(ROOT, "eval", name), "utf8"));
  for (const c of g.cases || []) {
    if (c.unanswerable || !c.answer_contains) continue;
    golden.set(c.question, { id: c.id, answer: c.answer_contains });
  }
}

const markedText = html => {
  const m = html.match(/<mark>([\s\S]*?)<\/mark>/);
  return m ? m[1] : null;
};
const unescape = s => String(s)
  .replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&amp;/g, "&");

const tally = {};
const problems = [];
for (const corpus of CORPORA) {
  const dir = join(ROOT, "static-demo", corpus);
  const t = { confident: 0, eligible: 0, HIT: 0, MISS: 0, NONE: 0 };
  tally[corpus] = t;
  for (const file of readdirSync(dir)) {
    if (!file.endsWith(".json")) continue;
    let rec;
    try { rec = JSON.parse(readFileSync(join(dir, file), "utf8")); } catch { continue; }
    if (!rec || typeof rec !== "object" || Array.isArray(rec) || !("query" in rec)) continue;
    if (!rec.verdict || rec.verdict.confident !== true) continue;
    t.confident += 1;
    const stages = rec.stages || [];
    const lead = stages.length && stages[stages.length - 1].items
      ? stages[stages.length - 1].items[0] : null;
    if (!lead || !lead.text) continue;
    const gold = golden.get(rec.query);
    if (!gold) continue;
    const text = clean(lead.text);
    const needle = answerNormalize(gold.answer);
    if (!answerNormalize(text).includes(needle)) continue;
    t.eligible += 1;
    const run = markedText(markAnswer(text, rec.query));
    let cls;
    if (run === null) cls = "NONE";
    else if (answerNormalize(unescape(run)).includes(needle)) cls = "HIT";
    else cls = "MISS";
    t[cls] += 1;
    if (cls !== "HIT") {
      problems.push({ corpus, id: gold.id, cls, answer: gold.answer,
                      run: run === null ? "-" : unescape(run) });
    }
  }
}

const pct = (a, b) => (b ? (100 * a / b).toFixed(1) : "n/a");
const line = (label, t) =>
  `  ${label.padEnd(8)} confident ${String(t.confident).padStart(3)}  `
  + `eligible ${String(t.eligible).padStart(3)}  `
  + `HIT ${String(t.HIT).padStart(3)} (${pct(t.HIT, t.eligible).padStart(5)}%)  `
  + `MISS ${String(t.MISS).padStart(3)} (${pct(t.MISS, t.eligible).padStart(5)}%)  `
  + `NONE ${String(t.NONE).padStart(3)} (${pct(t.NONE, t.eligible).padStart(5)}%)`;

const total = { confident: 0, eligible: 0, HIT: 0, MISS: 0, NONE: 0 };
console.log("answer highlight audit -- does the mark contain answer_contains?");
for (const corpus of CORPORA) {
  console.log(line(corpus, tally[corpus]));
  for (const k of Object.keys(total)) total[k] += tally[corpus][k];
}
console.log(line("overall", total));

if (problems.length) {
  console.log("\n  marks that do not contain the answer:");
  for (const p of problems) {
    console.log(`    ${p.cls.padEnd(4)} ${p.corpus}/${p.id}`);
    console.log(`         answer  ${JSON.stringify(p.answer)}`);
    console.log(`         marked  ${JSON.stringify(p.run)}`);
  }
}
process.exit(0);
