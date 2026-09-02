/* Which words of a passage are the answer, and therefore set bold.

   Extracted from index.html on 2026-08-27. It is pure text in, marked-up text
   out -- no DOM, no network, no state -- and it is the one part of the page a
   reader examines word by word, so it is the part most worth having a test for.
   Inline in a two-thousand-line HTML file, nothing could run it without a
   browser, and nothing did.

   The contract, which src/../ui/test-answer-mark.mjs enforces:

     - at most ONE sentence is ever marked, and never the whole passage
     - the mark is a contiguous run of at most MAX_MARK_WORDS words
     - a mark has to carry at least two distinct question words, or one and a
       figure, or there is no mark at all
     - the text outside the mark comes back escaped and unchanged
*/
/* The parser emits Markdown, so a passage carries the extractor's own marks:
   <br> where it found a line break, ### where it inferred a heading, ** around
   what looked bold. None of that is in the document -- it is how the text was
   recovered -- so stripping it removes the extractor's marks, never the
   document's words. */
export const clean = s => String(s)
  .replace(/<br\s*\/?>/gi, " ")
  .replace(/^#{1,6}\s+/gm, "")
  .replace(/#{2,6}\s+/g, "")
  .replace(/\*\*(.+?)\*\*/g, "$1")
  .replace(/(^|\s)[*_]{1,2}(?=\S)|(?<=\S)[*_]{1,2}(?=\s|$)/g, "$1")
  // Numeric reference markers -- [36], [9, 12], [4-7] -- are apparatus for a
  // reader following a bibliography, and this reader has no bibliography. They
  // survive into the middle of a quoted sentence and read as debris.
  .replace(/\s*\[\s*\d+(?:\s*[,;–-]\s*\d+)*\s*\]/g, "")
  // Author-year citations are the same apparatus in a different notation, and
  // they survive into the middle of a marked clause where a numeric marker
  // would not: "introduce inference latency (Houlsby et al., 2019; Rebuffiet
  // al., 2017) by extending model depth". A parenthetical counts as a citation
  // when it carries a four-digit year or an "et al.", which leaves ordinary
  // parentheticals alone -- "(in particular, BERT)" and "(pneumatized)" have
  // neither.
  .replace(/\s*\((?=[^)]*(?:\b(?:19|20)\d{2}[a-z]?\b|\bet\s+al\b))[^)]{0,200}\)/g, "")
  // A trailing pointer to another part of the same paper is apparatus too.
  .replace(/\s*\((?:see\s+)?(?:Section|Sec\.|Figure|Fig\.|Table|Tab\.|Appendix|Eq\.|Equation)\s*[\d.A-Z]+\s*\)/gi, "")
  .replace(/\s+([,.;:])/g, "$1")
  .replace(/\s+/g, " ").trim();

export const fragment = s => {
  const t = clean(s);
  if (!t) return t;
  return (/^[A-Z0-9"“(\[]/.test(t) ? "" : "… ") + t + (/[.!?"”)\]]$/.test(t) ? "" : " …");
};

export const esc = s => String(s).replace(/[&<>"]/g, c =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/* The passage is context; one sentence in it is usually the answer. Setting
   the whole block bold made all of it shout and none of it stand out, which is
   the opposite of pointing at the answer.

   The sentence is chosen the way the retrieval chose the passage: overlap with
   the words of the question, rare words counting for more than common ones.
   It is a highlight, not a claim -- the surrounding sentences stay readable at
   normal weight, so a wrong pick costs nothing but a misplaced emphasis. */
export const STOP = new Set(("a an and are as at be by do does for from has have how in is it its of on or "
  + "that the this to was were what when where which who why with your you").split(" "));

/* Questions that turn on a figure. For these the highlight is anchored on the
   number, because that is the thing being asked for -- everything else in the
   sentence is scaffolding around it. */
// The quantity word can sit anywhere in the question, not right after "what".
// Requiring "what value" missed "what label smoothing VALUE was used", which is
// the exact shape these questions take, and the highlight then spanned the
// sentence instead of landing on 0.1.
/* The hard bound on a highlight, in words.

   The passage is context and the mark is a pointer into it. Twelve words is
   about a clause -- long enough to read as a statement, short enough that the
   eye lands on it rather than reading it as a second paragraph. Past that the
   mark stops distinguishing anything, which is the failure the whole highlight
   exists to avoid: everything bold is the same as nothing bold. */
export const MAX_MARK_WORDS = 12;

/* And a bound in characters, because a word count is not a length.

   Swept over every golden question's real top passage, one mark covered 82.7%
   of its passage while satisfying "at most twelve words". The passage was a
   bibliography entry and the mark was a markdown link: four words, 359
   characters. Twelve ordinary words is about seventy characters, so this is
   loose enough never to touch prose and tight enough that a URL cannot pass as
   a clause. */
export const MAX_MARK_CHARS = 180;

/* And a bound relative to the passage. The two above are absolute, and a mark
   can satisfy both while still covering all of a short passage: "Though models
   like GPT-3 consume significant resources during training, they can" is
   eleven words and 82 characters, and it was the entire passage. */
export const MAX_MARK_SHARE = 0.5;

export const QUANT = new RegExp(
  "\\b(how many|how much|how large|how long|how deep|how wide|"
  + "value|rate|size|number|probability|dimension|percentage|fraction|"
  + "count|length|depth|width|score)\\b",
  "i");

/* Sentence boundaries. A terminator has to be followed by whitespace, or the
   decimal point in "0.1" ends the sentence. A section number also starts a new
   sentence -- "...applied in the model. 3.1.2 Training As described in..." was
   coming through as ONE sentence, because 3 is not a capital letter, and the
   highlight then ran across the boundary into the next section.

   Abbreviations are the mirror image of that bug: "et al. (2017) showed" and
   "Fig. 3 plots" are terminator-then-capital and are not sentence ends. A false
   boundary is not as visible as a missed one -- it does not run the mark into
   the next sentence, it cuts the answer in half -- so it went unnoticed on a
   corpus of academic papers, which is where these abbreviations all live. */
const ABBREV = /\b(?:[A-Z]|al|e\.g|i\.e|cf|vs|etc|approx|Fig|Figs|Eq|Eqs|Tab|Ref|Refs|Sec|Ch|No|pp|Dr|Prof|St|Mr|Ms|Mrs)\.$/;

/* A single capital letter before a full stop is an initial in "Vaswani, A."
   and a label in "in Appendix D.", and only the first is an abbreviation. The
   difference is the word in front of it, so the labelling words are named and
   everything else keeps the initial rule. Without this, "for every task we
   studied in Appendix D. Our text-to-text framework follows previous work"
   was one sentence, and the mark landed on the clause about previous work. */
const LABELLED = /\b(?:Appendix|Section|Figure|Fig|Table|Tab|Part|Chapter|Ch|Volume|Vol|Model|Case|Step|Phase|Class|Type|Level|Group|Panel)\s+[A-Z]\.$/;

export function splitSentences(text) {
  const rough = text.split(/(?<=[.!?])(?=\s+(?:[A-Z"“(]|\d+(?:\.\d+)*\s+[A-Z]))/);
  const out = [];
  for (const piece of rough) {
    // Re-join a piece whose predecessor ended in an abbreviation, so
    // "Vaswani et al." and "(2017) introduced the Transformer." stay one
    // sentence rather than two half-sentences the highlight has to choose
    // between.
    const prev = out.length ? out[out.length - 1].trimEnd() : "";
    if (out.length && ABBREV.test(prev) && !LABELLED.test(prev)) {
      out[out.length - 1] += piece;
    } else {
      out.push(piece);
    }
  }
  return out.filter(x => x.trim());
}

/* The tightest run of words in a sentence that actually answers.

   For a question about a figure this anchors on the number nearest a query
   term and takes the few words in front of it, so "we employed label smoothing
   of value 0.1" is marked rather than the whole sentence it sits in. For
   everything else it spans the query terms themselves. Capped at twelve words:
   past that it stops being a highlight and becomes a second paragraph. */
export function answerSpan(sentence, want, quantitative) {
  const toks = sentence.split(/(\s+)/);
  const words = [];
  toks.forEach((t, i) => { if (t.trim()) words.push({ i, t, low: t.toLowerCase() }); });
  // A short sentence is the answer, and marking all of it is right -- but
  // "short" has to mean short in characters too. A bibliography entry is five
  // whitespace-separated words and four hundred characters, and this early
  // return handed the whole thing back as a highlight, bypassing every bound
  // below it. That was the one mark in 450 real passages that covered most of
  // its passage.
  if (words.length <= MAX_MARK_WORDS && sentence.length <= MAX_MARK_CHARS) {
    return [0, sentence.length];
  }


  const hits = [];
  words.forEach((w, k) => { if (want.some(x => w.low.includes(x))) hits.push(k); });

  let lo = -1, hi = -1;
  if (quantitative) {
    const figs = [];
    words.forEach((w, k) => { if (/\d/.test(w.t)) figs.push(k); });
    if (figs.length) {
      const near = f => (hits.length ? Math.min(...hits.map(h => Math.abs(h - f))) : 0);
      const anchor = figs.reduce((bestF, f) => (near(f) < near(bestF) ? f : bestF), figs[0]);
      lo = Math.max(0, anchor - 7);
      // Run on a little past the figure to the end of its phrase, so a span
      // reads as a clause rather than stopping mid-thought: "h = 8 parallel
      // attention layers, or heads" instead of "h = 8 parallel".
      hi = anchor;
      for (let k = anchor + 1; k <= Math.min(words.length - 1, anchor + 4); k++) {
        hi = k;
        if (/[.,;:)]$/.test(words[k].t)) break;
      }
    }
  }
  if (lo < 0) {
    // No question word in the sentence at all. This used to mark the WHOLE
    // sentence, which is the one path that could set an unbounded run bold --
    // and it did it in the case with the least justification for marking
    // anything, since nothing in the sentence matched what was asked. It is
    // the same rule as the two-distinct-words check below, applied earlier:
    // when there is nothing to point at, point at nothing.
    if (!hits.length) return null;
    lo = Math.max(0, hits[0] - 1);
    hi = Math.min(words.length - 1, Math.max(hits[hits.length - 1], hits[0] + 6));
  }
  if (hi - lo >= MAX_MARK_WORDS) hi = lo + MAX_MARK_WORDS - 1;

  // A colon or a semicolon ends the clause, and a mark that runs past one
  // picks up whatever it introduces: "into a common format: McCann et al."
  // marked the clause and then two words of the citation after it.
  for (let k = lo; k < hi; k++) {
    if (/[:;]$/.test(words[k].t)) { hi = k; break; }
  }

  // Trim from the end to fit the character bound, whole words only. A span
  // that no longer carries two query words after trimming fails the check
  // below and is not marked at all, which is the right outcome for the case
  // this bound exists for: a citation whose "words" are URLs.
  while (hi > lo &&
         toks.slice(words[lo].i, words[hi].i + 1).join("").length > MAX_MARK_CHARS) {
    hi -= 1;
  }
  if (toks.slice(words[lo].i, words[hi].i + 1).join("").length > MAX_MARK_CHARS) {
    return null;                      // a single token longer than the bound
  }

  // A span has to carry something. Asked what a bird's fused collarbone is
  // called, the marked words were "this fused structure" -- one query word and
  // a pronoun standing in for the noun the question was about. Marking that
  // points at nothing. Two distinct query words, or one and a figure.
  const span = words.slice(lo, hi + 1);
  const carried = new Set();
  let figure = false;
  for (const w of span) {
    for (const x of want) if (w.low.includes(x)) carried.add(x);
    if (/\d/.test(w.t)) figure = true;
  }
  if (carried.size < 2 && !(carried.size === 1 && figure)) return null;

  const startTok = words[lo].i;
  const endTok = words[hi].i;
  const start = toks.slice(0, startTok).join("").length;
  let end = start + toks.slice(startTok, endTok + 1).join("").length;
  // A mark ending on the punctuation that closed its clause reads as though it
  // were cut off. The words are the pointer; the colon belongs to the sentence.
  while (end > start && /[:;,\s]/.test(sentence[end - 1])) end -= 1;
  return [start, end];
}

export function markAnswer(text, question) {
  const qWords = String(question).toLowerCase().match(/[a-z0-9]{3,}/g) || [];
  const want = [...new Set(qWords)].filter(w => !STOP.has(w));
  if (!want.length) return esc(text);

  const parts = splitSentences(text);
  if (!parts.length) return esc(text);

  // Rarer words are worth more: a term in one sentence out of six says more
  // about which sentence answers than one that appears in all of them.
  const spread = {};
  for (const w of want) {
    spread[w] = parts.filter(seg => seg.toLowerCase().includes(w)).length || parts.length;
  }
  // Adjacent pairs of content words from the question, in the order asked.
  // "What is late interaction in a retrieval model?" asks about a thing called
  // "late interaction", and a sentence naming it answers in a way that a
  // sentence merely containing "retrieval" and "model" does not. Scoring words
  // one at a time cannot tell those apart, and it picked the wrong sentence by
  // 0.1789 to 0.1728.
  const phrases = [];
  for (let i = 0; i + 1 < want.length; i++) {
    const pair = `${want[i]} ${want[i + 1]}`;
    if (String(question).toLowerCase().includes(pair)) phrases.push(pair);
  }

  let best = -1, top = 0;
  parts.forEach((seg, i) => {
    const low = seg.toLowerCase();
    let sc = 0;
    for (const w of want) if (low.includes(w)) sc += 1 / spread[w];
    // A phrase is worth the words it contains again, so naming the thing asked
    // about doubles that part of the score rather than swamping everything.
    for (const p of phrases) {
      if (low.includes(p)) for (const w of p.split(" ")) sc += 1 / spread[w];
    }
    sc /= Math.sqrt(Math.max(seg.trim().length, 40));
    if (sc > top) { top = sc; best = i; }
  });
  if (best < 0) return esc(text);

  // Then the answering WORDS inside that sentence. A whole sentence set bold
  // is most of the passage shouting; the reader still has to find the figure
  // inside it, which is the job the highlight was meant to do.
  const span = answerSpan(parts[best], want, QUANT.test(question));
  if (!span) return esc(text);            // nothing worth pointing at
  const [a, b] = span;

  // A mark is a pointer into a passage, so it has to be smaller than the
  // passage. On a passage that is one short sentence the span above is the
  // whole of it, and marking everything points at nothing. Half is the bound
  // the suite checks, and it is checked here so the rule lives with the code
  // rather than only in the test.
  if (b - a > text.length * MAX_MARK_SHARE) return esc(text);
  return parts.map((seg, i) => {
    if (i !== best) return esc(seg);
    return esc(seg.slice(0, a)) + "<mark>" + esc(seg.slice(a, b)) + "</mark>"
         + esc(seg.slice(b));
  }).join("");
}
