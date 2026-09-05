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
       figure, or there is no mark at all; the one exception is a definitional
       question ("what is X called"), where a clause of 5 or more words around
       a single whole-word hit may be marked when it does not open on a pronoun
       or a demonstrative, the hit is neither a participle nor one of the
       question's own marker words, and the clause carries a content word the
       question lacks
     - the text outside the mark comes back escaped and unchanged

   Whether the mark lands ON the answer is measured separately, by
   ui/audit-marks.mjs, against the golden answer_contains strings. The suite
   here checks the shape of the mark; the audit checks its aim.
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
/* Function words, which count for nothing in the overlap. "than", "rather",
   "before" and "each" were added on 2026-09-05, after the sweep marked "lower
   than the emperor penguin's average body temperature of 39 °C" for a question
   about insulating feathers, scored on "than" and a figure, and "For several
   days before they are ready to leave the nest" for the incubation question,
   scored on "before". Measured on their own against the rest of this file as
   of 2026-09-05, the 4 words left the audit at 38 of 68, moved 12 of the 450
   sweep marks (4 gained, 3 lost, 5 changed) and brought the 90th-percentile
   share from 6.7% to 6.5%. The one loss of aim was sbert-speed#0, where "8.5×
   faster than prior GPU" gave way to a sentence chosen on the phrase
   "similarity search" once "than" stopped counting; and with "before" gone
   the incubation passage's pick moved to " The chicks are altricial, hatching
   nearly naked with closed eyes.", which the short-sentence rule in
   answerSpan marks whole without asking what it carries. */
export const STOP = new Set(("a an and are as at be by do does for from has have how in is it its of on or "
  + "that the this to was were what when where which who why with your you "
  + "than rather before each").split(" "));

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
   everything else it spans the query terms and runs on to the end of the
   phrase they sit in. Capped at twelve words: past that it stops being a
   highlight and becomes a second paragraph. */

/* Questions that ask for a name. "What is X called", "what term describes",
   "known as": the answer is the one word the question does not contain, so
   the sentence that answers often carries a single question word, and the
   two-word rule below refused it. Asked what the period of sitting on eggs
   is called, the passage said "helping with the incubation of the eggs";
   only "eggs" matched, and nothing was marked.

   The lookbehind keeps "term" inside "long-term" from counting, since a hyphen
   is a word boundary to \b. All 11 golden questions that match are
   definitional, so this changes no measured case. */
export const DEFINITIONAL = /(?<![a-z-])(?:called|term|termed|known as|name for|referred to as)\b/i;
/* The marker words themselves, which are not pointers. "also called the
   preen gland" carried one question word, "called", and was marked for a
   question about the dawn chorus.

   They still count when the sentence is chosen. Dropping them from the
   question's words was measured on 2026-09-05: bird-incubation went from a
   MISS to a NONE in the audit, the sweep gained "Melanin is often involved in
   the absorption of light" for the dawn-chorus question, and bird-furcula#2
   moved off "a three-pronged bone called the intermaxillary", because "called"
   is what picks out a sentence that names something. */
const DEF_MARKERS = new Set("called term termed known name referred".split(" "));
// A span that opens on one of these refers back to the previous sentence,
// and the thing it refers to is what the reader wanted marked.
const ANAPHORIC = new Set("this that these those it its they them such".split(" "));
// A clause shorter than this carrying one question word is a list item or a
// table cell. "brown-blotched eggs." was marked for the incubation question.
const MIN_RELAXED_WORDS = 5;
/* A definitional question asks for the name of a thing, so the one question
   word a clause is accepted on has to be a thing, and a word ending in -ed or
   -ing is a participle more often than a noun. Without this the clause path
   marked "so the established Linnean system is followed here." and "brent
   geese Branta bernicla bernicla migrating between the Taymyr Peninsula" for
   the flyway question, anchored on "established" and "migrating", neither of
   which is what the question is about. Nouns in -ing ("training",
   "embedding") cannot anchor a one-word clause either, which costs no mark
   the sweep had. */
const PARTICIPLE = /(?:ed|ing)$/;
/* Words of context in front of the first question word. A width of 2 was
   measured on 2026-09-05 against the sweep of 450 real passages: it lost 5
   marks, because the extra word pushed the last question word past the
   12-word cap, and opened 2, one on a section number ("3.4 Multi-objective
   Reward Function Inspired by") and one on a count ("14 neck vertebrae—humans
   have only seven"). In the audit it turned 2 misses into hits and 1 hit into
   a miss. */
const LEAD_WORDS = 1;
/* How far past the last question word a span may run to finish its phrase.
   The quantitative path runs 4; 5 is the smallest count that reaches
   "singlepoint" in "enables probabilistic forecasts instead of singlepoint
   estimates", which was the answer the audit found cut off, and 4 fails that
   check. */
const RUN_ON_WORDS = 5;

export function answerSpan(sentence, want, quantitative, definitional = false) {
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

  // What a range carries, which decides whether it may run on, the colon cut
  // below, and whether the span is worth marking at all.
  const carriedIn = (a, b) => {
    const got = new Set();
    let fig = false;
    for (const w of words.slice(a, b + 1)) {
      for (const x of want) if (w.low.includes(x)) got.add(x);
      if (/\d/.test(w.t)) fig = true;
    }
    return { n: got.size, fig, got };
  };
  const strong = r => r.n >= 2 || (r.n === 1 && r.fig);

  /* Run on from `from` for up to `n` words, so a span reads as a clause
     rather than stopping mid-thought: "h = 8 parallel attention layers, or
     heads" instead of "h = 8 parallel". It stops ON the word that closes the
     phrase -- one ending in a full stop, a comma, a colon, a semicolon or a
     closing bracket, the same convention as the colon cut below -- and BEFORE
     a word that is not prose: a URL, or a footnote marker fused to the first
     word of the next sentence ("1Fine-tuning"). The first run-on (2026-09-05)
     had no BEFORE rule and ran "only one GPU is dedicated per query" into
     "for 5htps://github.com/huggingface/transformers -----" and "Transformers
     lack some of the inductive biases" into "1Fine-tuning code and
     pre-trained". Stopping before a colon instead was measured against
     stopping on it and differed in 2 of 450 sweep marks: it dropped
     "component" from "clearly resolved shifted component" and dropped the
     extractor's label from "array of se Section", and the phrase mattered
     more than the label. */
  const runOn = (from, n) => {
    let end = from;
    for (let k = from + 1; k <= Math.min(words.length - 1, from + n); k++) {
      const t = words[k].t;
      if (/:\/\/|^www\./.test(t) || /^\d+[A-Z][a-z]/.test(t)) break;
      end = k;
      if (/[.,;:)]$/.test(t)) break;
    }
    return end;
  };

  /* Trim the end of a span back to the last word that belongs to it. A span
     may not end on a function word, on a symbol ("=", "-----", "…"), on a
     bracket it opens and does not close ("(β2", "[Hou"), or on a one- or
     two-letter lowercase fragment ("se", "mt"), and it may not leave a
     bracket open. A word that closes a bracket ends the trimming whatever
     else it is. The first version of this trim (2026-09-05) judged "at)." by
     its letters alone, stripped it as a stopword, and left "an expected
     return forecast (α =" where "an expected return forecast (α = at)." had
     been; over the 450 sweep passages it raised marks with an unbalanced
     bracket from 4 to 8 and marks ending on an operator from 2 to 5, while
     cutting marks ending on a stopword from 51 to 29. Nothing at or below
     `floor` -- the last question word, or the figure -- is trimmed. */
  const tidy = (a, b, floor) => {
    // The 12-word cap can cut a span short of its last question word, and
    // then the floor is the last question word still inside the span. With
    // the floor beyond the end nothing was trimmed, and "we mention that BERT
    // uses a" kept its "a".
    if (floor > b) floor = hits.filter(h => h <= b).reduce((m, h) => Math.max(m, h), a);
    const core = k => words[k].low.replace(/[^a-z]/g, "");
    const closes = k => /[)\]]/.test(words[k].t);
    const junk = k => /^[^a-zA-Z0-9]*$/.test(words[k].t)
      || STOP.has(core(k))
      || (/^[(\[]/.test(words[k].t) && !closes(k))
      || /^[a-z]{1,2}[^a-zA-Z0-9]*$/.test(words[k].t);
    for (let pass = 0; pass < 2; pass++) {
      while (b > floor && !closes(b) && junk(b)) b -= 1;
      let depth = 0, opened = -1;
      for (let k = a; k <= b; k++) {
        for (const c of words[k].t) {
          if (c === "(" || c === "[") { if (depth === 0) opened = k; depth += 1; }
          else if ((c === ")" || c === "]") && depth > 0) depth -= 1;
        }
      }
      if (depth === 0 || opened <= floor) break;
      b = opened - 1;
    }
    return b;
  };

  let lo = -1, hi = -1, floor = -1;
  if (quantitative) {
    const figs = [];
    words.forEach((w, k) => { if (/\d/.test(w.t)) figs.push(k); });
    if (figs.length) {
      const near = f => (hits.length ? Math.min(...hits.map(h => Math.abs(h - f))) : 0);
      const anchor = figs.reduce((bestF, f) => (near(f) < near(bestF) ? f : bestF), figs[0]);
      lo = Math.max(0, anchor - 7);
      hi = runOn(anchor, 4);
      floor = anchor;
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
    const first = hits[0], last = hits[hits.length - 1];
    floor = last;
    const distinct = new Set();
    for (const w of words) for (const x of want) if (w.low.includes(x)) distinct.add(x);
    if (distinct.size === 1 && definitional) {
      /* A single question word, and a question that asks for a name. There
         is no second word to span to, so the clause around the one word is
         the unit. It grows back to the punctuation before the word and
         forward to the punctuation after it, then shrinks to the word bound
         from whichever end is farther from the hit. On "with the male also helping with the
         incubation of the eggs during the day," that keeps "incubation".

         This clause rule was first tried (2026-09-05) for EVERY span, not only
         this case, and it cost 6 answers in the audit of 68: growing back to
         the clause start and trimming from the far end pulled "interval" off
         "restricted to a common compact interval", and a comma stopped the
         growth at "machine learning," before "probabilistic forecasts". The
         span between two question words is a better anchor than a clause
         boundary, so the clause is used only when there is one word. */
      const CLAUSE = /[,;:]$/;
      lo = first;
      while (lo > 0 && !CLAUSE.test(words[lo - 1].t)) lo -= 1;
      hi = last;
      while (hi < words.length - 1 && !CLAUSE.test(words[hi].t)) hi += 1;
      while (hi - lo + 1 > MAX_MARK_WORDS) {
        if (first - lo >= hi - last) lo += 1; else hi -= 1;
      }
    } else {
      lo = Math.max(0, first - LEAD_WORDS);
      hi = Math.min(words.length - 1, Math.max(last, first + 6));
      /* Run on to the end of the phrase, as the quantitative path above
         does. The answer to "what does X enable" or "can X be restricted"
         follows the question words. Of the 30 audit misses under the code
         before this run-on (ui/audit-marks.mjs, 2026-09-05), 19 marks sat in
         the sentence that contains the answer, and in 15 of those the answer
         began at or after the point where the mark stopped; "The CVaR
         threshold can be restricted to a" and "BVAR, when complemented with
         machine learning, enables" were 2 of them.

         Only a span that already qualifies may run on. The run-on finishes a
         phrase; it is not allowed to be what qualifies the span, which the
         first version permitted: "index prices the premium at a mean t of
         +1.14 with standard" carried one question word, "prices", and was
         accepted on the figure the run-on had reached. */
      if (strong(carriedIn(lo, hi))) hi = runOn(hi, RUN_ON_WORDS);
    }
  }
  if (hi - lo >= MAX_MARK_WORDS) hi = lo + MAX_MARK_WORDS - 1;
  hi = tidy(lo, hi, floor);

  /* The exception to the two-word rule, for definitional questions only. A
     span carrying one question word is accepted when the word is matched
     whole rather than as a substring ("light" inside "flight" and "able"
     inside "unable" each produced a mark in the sweep before this check),
     the word is not one of the question's own markers and not a participle,
     the span is a clause rather than a fragment, it does not open on a
     pronoun, and it carries at least one content word of 4 or more letters
     that the question lacks. "This fused structure" fails on the pronoun;
     "the male also helping with the incubation of the eggs during the day"
     passes on "incubation". */
  const novel = (a, b) => words.slice(a, b + 1).some(w => {
    const core = w.low.replace(/[^a-z]/g, "");
    return core.length >= 4 && !STOP.has(core) && !want.some(x => core.includes(x) || x.includes(core));
  });
  const opensPlain = a => {
    const w0 = words[a].low.replace(/[^a-z]/g, "");
    const w1 = words[a + 1] ? words[a + 1].low.replace(/[^a-z]/g, "") : "";
    return !ANAPHORIC.has(w0) && !(w0 === "the" && w1 === "same");
  };
  const wholeWord = (a, b, x) => words.slice(a, b + 1)
    .some(w => w.low.split(/[^a-z0-9]+/).includes(x));
  // The colon cut below uses `strong` and not `enough`. With the relaxed
  // rule it cut "The chicks of passerines are altricial: blind, featherless,
  // and helpless" back to "altricial", because the front half carried one
  // question word and the question had "term" in it, and the definition the
  // colon introduced was lost again.
  const enough = (r, a, b) => strong(r)
    || (definitional && r.n === 1 && !DEF_MARKERS.has([...r.got][0])
        && !PARTICIPLE.test([...r.got][0])
        && wholeWord(a, b, [...r.got][0])
        && b - a + 1 >= MIN_RELAXED_WORDS && opensPlain(a) && novel(a, b));

  // A colon or a semicolon ends the clause, and a mark that runs past one
  // picks up whatever it introduces: "into a common format: McCann et al."
  // marked the clause and then two words of the citation after it.
  //
  // But a colon also introduces a DEFINITION, and there the half after it is
  // the half worth marking. Cutting blindly, added 2026-09-02 for the citation
  // above, broke that: "The chicks of passerines are altricial: blind,
  // featherless, and helpless when hatched" was cut to end at "altricial:",
  // which threw away "helpless" and "hatched" and left one query word, so the
  // span failed the check below and the passage showed NO mark at all. Two
  // answers in the bird corpus lost their highlight to it. The cut is applied
  // only when the span survives it, since it exists to trim debris off a
  // pointer rather than to destroy the pointer.
  for (let k = lo; k < hi; k++) {
    if (/[:;]$/.test(words[k].t)) {
      if (strong(carriedIn(lo, k))) hi = k;
      break;
    }
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
  // points at nothing. Two distinct query words, or one and a figure. The
  // definitional exception, and its guards, are in `enough` above.
  if (!enough(carriedIn(lo, hi), lo, hi)) return null;

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
  const span = answerSpan(parts[best], want, QUANT.test(question), DEFINITIONAL.test(question));
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
