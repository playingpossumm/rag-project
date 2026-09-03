/* Can a reader see the passage that answers?

   The page leads with the top result and lists the rest under a disclosure.
   Until 2026-09-03 that list showed one passage per document and at most two,
   which together hid the answering passage on 15 of the 79 recorded questions
   where retrieval had returned it. Nothing asserted otherwise, because the
   rule lived inside a 2,400-line HTML file.

   Two halves, and the second is the one that matters:

   **The contract**, on hand-written sets where the right order is obvious.
   **The sweep**, over every recorded answer in all three corpora, asking the
   only question a reader cares about: when retrieval found the answer, is it
   on the page at all. A selection rule can satisfy every unit test and still
   hide the answer, because real result sets are five passages of one document
   as often as they are five documents.

   Run:  node ui/test-passages.mjs
*/
import { readFileSync, readdirSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { alsoFoundOrder, alsoFoundHint } from "./passages.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");
const CHECKS = [];
const check = (name, got, want) =>
  CHECKS.push({ name, got, want, ok: JSON.stringify(got) === JSON.stringify(want) });

const p = (source, id) => ({ source, id, text: `${id} of ${source}` });
const ids = list => list.map(x => x.id);

/* ---------------------------------------------------------- the contract -- */

{
  const sel = [p("a.pdf", 1), p("b.pdf", 2), p("c.pdf", 3), p("d.pdf", 4)];
  check("the lead is never listed again below itself",
        ids(alsoFoundOrder(sel, sel[0])), [2, 3, 4]);
}

{
  // One per document first, which is what the section has always been for,
  // and then what that pass set aside, in the order retrieval ranked it.
  const sel = [p("a.pdf", 1), p("b.pdf", 2), p("b.pdf", 3), p("c.pdf", 4)];
  check("a new document outranks a second passage of one already listed",
        ids(alsoFoundOrder(sel, sel[0])), [2, 4, 3]);
}

{
  // qf-social-disclosure's shape. diversify() backfills from what its own cap
  // pushed aside rather than return fewer than k, so every result can be one
  // document -- and the answer was at rank 4 of five.
  const sel = [p("only.pdf", 1), p("only.pdf", 2), p("only.pdf", 3),
               p("only.pdf", 4), p("only.pdf", 5)];
  const others = alsoFoundOrder(sel, sel[0]);
  check("five passages of one document still list all four of the others",
        ids(others), [2, 3, 4, 5]);
  check("and the wording does not call them other documents",
        alsoFoundHint(others, sel[0]),
        "the other 4 passages that came back, all from this document");
}

{
  const sel = [p("a.pdf", 1), p("b.pdf", 2), p("a.pdf", 3)];
  const others = alsoFoundOrder(sel, sel[0]);
  check("a second passage of the lead's own document is listed, not dropped",
        ids(others), [2, 3]);
  check("and it is not counted as another document",
        alsoFoundHint(others, sel[0]),
        "the other 2 passages that came back, from 1 other document");
}

{
  // `top` is an element of `sel`, not a copy of one -- index.html passes
  // sel[0] itself. The first draft of this check passed a second object with
  // the same fields, which the identity test correctly did not treat as the
  // lead, so it failed against code that is right.
  const sel = [p("a.pdf", 1)];
  check("a lone result lists nothing below it",
        alsoFoundOrder(sel, sel[0]), []);
}
check("an empty set does not throw", alsoFoundOrder([], undefined), []);

/* -------------------------------------------------------------- the sweep -- */

const GOLDEN = { birds: "eval/golden-birds.json", llm: "eval/golden_set.json",
                 quant: "eval/golden-quant.json" };
const DASH = /[‐-―−]/g;
const norm = t => String(t).replace(DASH, "-").replace(/\s+/g, " ")
  .trim().toLowerCase();

let onPage = 0, hidden = 0, asLead = 0;
const missing = [];
for (const [corpus, gpath] of Object.entries(GOLDEN)) {
  const dir = join(ROOT, "static-demo", corpus);
  if (!existsSync(dir) || !existsSync(join(ROOT, gpath))) continue;
  const gold = {};
  for (const c of JSON.parse(readFileSync(join(ROOT, gpath), "utf8")).cases)
    gold[c.question] = c;

  for (const f of readdirSync(dir)) {
    let d;
    try { d = JSON.parse(readFileSync(join(dir, f), "utf8")); } catch { continue; }
    if (!d || typeof d !== "object" || !d.query || !d.verdict?.confident) continue;
    const c = gold[d.query];
    if (!c?.answer_contains) continue;

    const items = d.stages.at(-1).items;
    const at = items.findIndex(it => norm(it.text || "").includes(norm(c.answer_contains)));
    if (at < 0) continue;                  // retrieval did not return it at all

    if (at === 0) { asLead++; onPage++; continue; }
    if (alsoFoundOrder(items, items[0]).includes(items[at])) onPage++;
    else { hidden++; missing.push(c.id); }
  }
}

if (onPage + hidden) {
  console.log(`  swept ${onPage + hidden} recorded answers whose passage `
    + `retrieval returned`);
  console.log(`    shown as the answer   ${asLead}`);
  console.log(`    listed below it       ${onPage - asLead}`);
  console.log(`    hidden                ${hidden}`);
  // The whole point of the rule. A passage retrieval found and the reader
  // cannot reach is indistinguishable, to them, from one never retrieved.
  check("no answer that retrieval returned is hidden from the reader",
        missing, []);
} else {
  console.log("  sweep skipped: no recorded answers found");
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
console.log(`${CHECKS.length - failed}/${CHECKS.length} passage-selection checks passed`);
process.exit(failed ? 1 : 0);
