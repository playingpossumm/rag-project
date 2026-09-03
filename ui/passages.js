/* Which of the returned passages appear beneath the answer, and in what order.
 *
 * Split out of index.html so it can be tested. The rule it holds decided
 * whether a reader could see the answer at all on 15 of 79 recorded questions,
 * which is too much to leave inside a 2,400-line page with nothing asserting
 * it.
 *
 * The page shows the top passage in full and then a disclosure listing the
 * rest. That list used to apply two filters: one passage per document, and at
 * most two of them. Each is defensible alone -- three passages of one paper is
 * repetition, and an unbounded list is not a summary -- and together they hid
 * the passage that answers on 15 of the 79 recorded questions whose answer
 * retrieval had in fact returned.
 *
 * Twelve of those fifteen were dropped for sitting in the SAME document as the
 * lead, which is the case the per-document filter was least meant to cover.
 * The reader is not being spared a repetitive second source there; they are
 * being denied the paragraph of this document that answers while the page
 * leads with a paragraph of it that does not. Measured over the recorded set:
 *
 *   one per document, at most two          64 of 79 visible   (shipped until
 *                                                              2026-09-03)
 *   one per document, at most three        65 of 79
 *   plus one from the lead's document      76 of 79
 *   the rest of the set, one per document
 *   first and then the remainder by rank   79 of 79
 *
 * The third row is what says the per-document filter was the cause rather than
 * the limit, since raising the limit alone moves one question. The reason even
 * the fourth row is not enough is `qf-social-disclosure`, which returns five
 * passages of ONE document: diversify() backfills from what its own cap pushed
 * aside rather than return fewer than k, so a single same-document slot goes to
 * rank 2 and the answer at rank 4 stays hidden.
 *
 * So the whole of the rest is shown, ordered one per document first, which is
 * what the section has always been for, then whatever that pass set aside, in
 * rank order. That is the shape diversify() itself uses. It is bounded at four
 * because k is five, and the disclosure opens on request, so completeness costs
 * the reader nothing until they ask for it.
 *
 * This never reorders the answer. The lead is passed in and excluded, and
 * everything here sits below it.
 */

/* `sel` is the returned set and `top` the passage shown as the answer. Returns
 * the others, in the order they should be listed. */
export function alsoFoundOrder(sel, top) {
  const seen = new Set([top && top.source]);
  const first = [], rest = [];
  for (const it of sel || []) {
    if (it === top) continue;
    if (seen.has(it.source)) { rest.push(it); continue; }
    seen.add(it.source);
    first.push(it);
  }
  return [...first, ...rest];
}

/* How to describe that list. The wording it replaces called every entry a
 * document, which a second passage of the lead's own document is not. */
export function alsoFoundHint(others, top) {
  const docs = new Set(others.map(o => o.source)
    .filter(s => s !== (top && top.source))).size;
  const n = `the other ${others.length} passage${others.length > 1 ? "s" : ""}`;
  return docs
    ? `${n} that came back, from ${docs} other document${docs > 1 ? "s" : ""}`
    : `${n} that came back, all from this document`;
}
