// Helpers the pages used to copy. Until 2026-09-06 index.html defined $, nf
// and readInk, quality.html defined $, nf and esc, and versions.html carried
// a second readInk written out by hand, so a change to one had to be made in
// two or three places. esc is the one answer-mark.js already exports, passed
// through rather than written again.

export const $ = s => document.querySelector(s);

export const nf = n => Number(n).toLocaleString();

export { esc } from "./answer-mark.js";

/* The ink the pipeline drawing paints with, read from the page's tokens so
   the canvas and the page cannot disagree. `other` is --wire, not a mid grey:
   everything that did NOT reach the answer is drawn as near-invisible
   wireframe, so hue in the picture means exactly one thing. */
export const readInk = () => {
  const cs = getComputedStyle(document.body);
  const v = n => cs.getPropertyValue(n).trim();
  return { ink: v("--ink"), muted: v("--mute"), faint: v("--wire"), line: v("--hair"),
    lineSoft: v("--hair"), panel: v("--ground"), sunk: v("--card"), rule: v("--hair"),
    s1: v("--s1"), s2: v("--s2"), s3: v("--s3"),
    other: v("--wire"), good: v("--good"), critical: v("--bad") };
};
