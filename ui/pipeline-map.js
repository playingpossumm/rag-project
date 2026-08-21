/* The retrieval pipeline, drawn as a technical schematic.
 *
 * Third attempt. The first drew a city of towers, which is a generic isometric
 * illustration that says nothing true about retrieval. The second drew stacked
 * plates joined by coloured ribbons, which was true but loud: every candidate
 * was tinted and the connectors read as spaghetti.
 *
 * This one follows the reference language directly -- thin monochrome
 * wireframe, a grid mesh on each plane, leader lines out to labels, and a great
 * deal of empty space. Colour appears on exactly the passages that reached the
 * answer, and nowhere else.
 *
 * Two facts about retrieval constrain the layout and are not negotiable:
 *
 *   Stages read LEFT TO RIGHT, because that is how a process is read.
 *
 *   Dense and BM25 share a step and are separated by HEIGHT. They run at the
 *   same moment; placing one after the other would draw a sequence that does
 *   not exist.
 *
 * Draw order is strictly left to right -- plate, then the links leaving it,
 * then the next plate. Drawing all links in one pass first meant a connector
 * existed and then a plane landed on top of it, which looked like a bug
 * because it was one.
 */

export const shortDoc = s => String(s).replace(/\.(pdf|docx|pptx|xlsx)$/i, "");

const KX = Math.cos(Math.PI / 6);
const KY = Math.sin(Math.PI / 6);
const project = (u, v, z) => ({ x: (u - v) * KX, y: (u + v) * KY - z });

const FLOW = 21;
const at = (step, lift) => ({ u: step * FLOW, v: -step * FLOW, z: lift });

/* label side: 1 above, -1 below. Leader lines run vertically out of the plate
   to the text, the way the reference annotates its diagrams. */
const PLATES = [
  { id: "corpus",   n: "01", ...at(0, 0),   w: 13, grid: 8, lead: 1,
    label: "Index",              term: c => `${c.total.toLocaleString()} passages · ${c.docs} documents` },
  { id: "dense",    n: "02", ...at(1, 15),  w: 6.5, grid: 4, lead: 1,
    label: "Dense retrieval",    term: () => "embedding similarity" },
  { id: "sparse",   n: "03", ...at(1, -15), w: 6.5, grid: 4, lead: -1,
    label: "BM25",               term: () => "lexical match" },
  { id: "fused",    n: "04", ...at(2, 0),   w: 7.5, grid: 4, lead: 1,
    label: "Rank fusion",        term: () => "both rankings combined" },
  { id: "reranked", n: "05", ...at(3, 0),   w: 7.5, grid: 4, lead: -1,
    label: "Cross-encoder",      term: () => "query and passage scored together" },
  { id: "selected", n: "06", ...at(4, 0),   w: 6,  grid: 3, lead: 1,
    label: "Diversity cap",      term: () => "max 2 per document" },
  { id: "answer",   n: "07", ...at(5, 0),   w: 5,  grid: 2, lead: -1,
    label: "Answer",             term: () => "cited passages" },
];

const SLOTS = 4;

function palette(sources, ink) {
  const slots = [ink.s1, ink.s2, ink.s3];
  const m = new Map();
  sources.forEach((s, i) => m.set(s, i < 3 ? slots[i] : ink.other));
  return src => m.get(src) || ink.other;
}

function slot(plate, i, n) {
  const cols = Math.min(SLOTS, Math.max(1, n));
  const rows = Math.max(1, Math.ceil(n / cols));
  const cx = i % cols, cy = Math.floor(i / cols);
  const su = (plate.w * 1.35) / (cols + 1);
  const sv = (plate.w * 1.35) / (rows + 1);
  return { u: plate.u - plate.w * 0.68 + su * (cx + 1),
           v: plate.v - plate.w * 0.68 + sv * (cy + 1) };
}

/* ------------------------------------------------------------- the index */
export function buildCity(chunks) {
  const sources = chunks.sources || [];
  const doc = chunks.doc || [];
  const total = chunks.n || doc.length;

  // A mark per chunk is an unreadable smear at this size, so the field is
  // sampled to a fixed budget. The count printed beside it stays exact.
  const BUDGET = 260;
  const step = Math.max(1, Math.floor(total / BUDGET));
  const ids = [];
  for (let i = 0; i < total; i += step) ids.push(i);

  // A jittered lattice, deterministic so the drawing never changes between
  // renders. A hash scatter clumped and read as noise; an index is ordered.
  const side = Math.ceil(Math.sqrt(ids.length));
  const span = 22;
  const marks = ids.map((id, n) => {
    const gx = n % side, gy = Math.floor(n / side);
    const j = k => ((Math.sin(k * 127.1 + n * 311.7) * 43758.5453) % 1 + 1) % 1 - 0.5;
    return { id, doc: doc[id] ?? 0,
             u: -span / 2 + (span * (gx + 0.5)) / side + j(1) * (span / side) * 0.5,
             v: -span / 2 + (span * (gy + 0.5)) / side + j(2) * (span / side) * 0.5 };
  });
  return { total, docs: sources.length, sources, marks,
           chunkAt: new Map(marks.map(m => [m.id, m])), step };
}

/* ---------------------------------------------------------------- a run */
export function buildRun(trace, city, ink) {
  if (!trace || !city) return null;
  const byName = new Map(trace.stages.map(s => [s.name, s]));
  const final = byName.get("selected")?.items || [];
  const colour = palette([...new Set(final.map(i => i.source))], ink);
  const survivors = new Set(final.map(i => i.chunk_id));

  const placed = new Map();
  for (const plate of PLATES) {
    const st = byName.get(plate.id);
    if (!st) continue;
    const items = st.items.slice(0, SLOTS * 3);
    placed.set(plate.id, items.map((it, i) => ({
      ...it, ...slot(plate, i, items.length), z: plate.z,
      lives: survivors.has(it.chunk_id),
      colour: survivors.has(it.chunk_id) ? colour(it.source) : null,
    })));
  }
  const sel = placed.get("selected") || [];
  const ap = PLATES.find(p => p.id === "answer");
  placed.set("answer", sel.map((m, i) => ({ ...m, ...slot(ap, i, sel.length), z: ap.z })));

  const lit = new Map();
  for (const name of ["dense", "sparse"]) {
    for (const it of byName.get(name)?.items || []) {
      const near = city.chunkAt.get(Math.round(it.chunk_id / city.step) * city.step);
      if (!near) continue;
      lit.set(near.id, { colour: survivors.has(it.chunk_id) ? colour(it.source) : null });
    }
  }

  // Links, grouped by the plate they arrive at, so the draw loop can emit them
  // in flow order instead of all at once.
  const byTarget = new Map();
  const chain = [["dense", "fused"], ["sparse", "fused"], ["fused", "reranked"],
                 ["reranked", "selected"], ["selected", "answer"]];
  for (const [a, b] of chain) {
    const idx = new Map((placed.get(a) || []).map(m => [m.chunk_id, m]));
    for (const m of placed.get(b) || []) {
      const src = idx.get(m.chunk_id);
      if (!src) continue;
      if (!byTarget.has(b)) byTarget.set(b, []);
      byTarget.get(b).push({ a: src, b: m, lives: m.lives, colour: m.colour });
    }
  }

  const rer = placed.get("reranked") || [];
  const biggest = rer.reduce((best, m) =>
    (Math.abs(m.delta ?? 0) > Math.abs(best?.delta ?? 0) ? m : best), null);

  return {
    colour, placed, byTarget, lit, city,
    verdict: trace.verdict, confident: trace.verdict?.confident !== false,
    counts: {
      dense: byName.get("dense")?.items.length || 0,
      sparse: byName.get("sparse")?.items.length || 0,
      fused: byName.get("fused")?.items.length || 0,
      reranked: byName.get("reranked")?.items.length || 0,
      selected: final.length,
      both: (byName.get("fused")?.items || []).filter(i => i.agreement).length,
      biggest,
    },
    skipped: new Set(PLATES.map(p => p.id).filter(id => byName.get(id)?.skipped)),
  };
}

/* -------------------------------------------------------------- drawing */
const corners = w => [{ u: -w, v: -w }, { u: w, v: -w }, { u: w, v: w }, { u: -w, v: w }];

function plane(ctx, T, plate, ink, a, dim) {
  const pts = corners(plate.w).map(c => T(plate.u + c.u, plate.v + c.v, plate.z));
  ctx.save();
  ctx.globalAlpha = a;

  // grid mesh first, so the outline sits cleanly on top of it
  ctx.strokeStyle = ink.other;
  ctx.globalAlpha = a * (dim ? 0.16 : 0.3);
  ctx.lineWidth = 0.6;
  for (let i = 1; i < plate.grid; i++) {
    const t = -plate.w + (2 * plate.w * i) / plate.grid;
    const p1 = T(plate.u + t, plate.v - plate.w, plate.z);
    const p2 = T(plate.u + t, plate.v + plate.w, plate.z);
    const q1 = T(plate.u - plate.w, plate.v + t, plate.z);
    const q2 = T(plate.u + plate.w, plate.v + t, plate.z);
    ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(q1.x, q1.y); ctx.lineTo(q2.x, q2.y); ctx.stroke();
  }

  ctx.globalAlpha = a * (dim ? 0.35 : 1);
  ctx.strokeStyle = ink.faint;
  ctx.lineWidth = 0.9;
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < 4; i++) ctx.lineTo(pts[i].x, pts[i].y);
  ctx.closePath();
  ctx.stroke();
  ctx.restore();
  return pts;
}

function mark(ctx, p, r, fill, stroke, a) {
  ctx.save(); ctx.globalAlpha = a;
  ctx.beginPath();
  ctx.moveTo(p.x, p.y - r); ctx.lineTo(p.x + r * 1.7, p.y);
  ctx.lineTo(p.x, p.y + r); ctx.lineTo(p.x - r * 1.7, p.y);
  ctx.closePath();
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 0.9; ctx.stroke(); }
  ctx.restore();
}

/* A leader line out of the plate to its label -- the reference's annotation
   device, and the reason labels never collide with the next plate. */
function annotate(ctx, T, plate, city, run, ink, a, dim) {
  const up = plate.lead === 1;
  const from = T(plate.u, plate.v, plate.z + (up ? plate.w : -plate.w));
  const to = T(plate.u, plate.v, plate.z + (up ? plate.w + 13 : -plate.w - 13));

  ctx.save();
  ctx.globalAlpha = a * 0.5;
  ctx.strokeStyle = ink.other; ctx.lineWidth = 0.8;
  ctx.beginPath(); ctx.moveTo(from.x, from.y); ctx.lineTo(to.x, to.y); ctx.stroke();

  ctx.globalAlpha = a;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  // Always reads number, name, detail from the top down, whichever side it is
  // on. Drawing outward from the plate had reversed it above the line, so the
  // detail sat over the name and the number underneath both.
  const rows = [
    { text: plate.n, font: "500 9px 'DM Mono', ui-monospace, monospace",
      fill: ink.other },
    { text: plate.label, font: "600 12px Inter, system-ui, sans-serif",
      fill: dim ? ink.faint : ink.ink },
    { text: String(dim ? "not run" : (run ? runSub(plate.id, run) : plate.term(city))).toUpperCase(),
      font: "500 9px 'DM Mono', ui-monospace, monospace", fill: ink.muted },
  ];
  const LH = 14;
  // Above: the last row sits nearest the plate, so the stack grows upward.
  const first = up ? to.y - (rows.length - 1) * LH - 4 : to.y + 6;
  rows.forEach((r, i) => {
    ctx.fillStyle = r.fill; ctx.font = r.font;
    ctx.fillText(r.text, to.x, first + i * LH);
  });
  ctx.restore();
}

export function drawScene(ctx, city, run, ink, W, H, progress = 1) {
  ctx.clearRect(0, 0, W, H);
  if (!city) return;

  const probe = [];
  for (const p of PLATES) {
    for (const c of corners(p.w)) probe.push(project(p.u + c.u, p.v + c.v, p.z));
    // the annotation stack, so a label is never clipped by the canvas edge
    probe.push(project(p.u, p.v, p.z + (p.lead === 1 ? p.w + 42 : -p.w - 42)));
  }
  const xs = probe.map(p => p.x), ys = probe.map(p => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);

  const pad = 16;
  const s = Math.min((W - pad * 2) / (maxX - minX), (H - pad * 2) / (maxY - minY));
  const ox = W / 2 - ((minX + maxX) / 2) * s;
  const oy = H / 2 - ((minY + maxY) / 2) * s;
  const T = (u, v, z) => { const p = project(u, v, z); return { x: ox + p.x * s, y: oy + p.y * s }; };

  const reveal = i => {
    if (progress >= 1) return 1;
    const span = 1 / PLATES.length;
    return Math.max(0, Math.min(1, (progress - i * span) / span));
  };

  ctx.lineJoin = "round"; ctx.lineCap = "round";

  // Strict flow order: a plate, then the links that leave it. Nothing is ever
  // drawn on top of a connector that was already on screen.
  PLATES.forEach((plate, i) => {
    const a = reveal(i);
    if (a <= 0) return;
    const dim = run?.skipped.has(plate.id);

    // links arriving here, drawn before this plate so the plate reads as the
    // thing they land on
    for (const l of run?.byTarget.get(plate.id) || []) {
      const la = a * (l.lives ? 0.9 : 0.22);
      const p1 = T(l.a.u, l.a.v, l.a.z + 0.5);
      const p2 = T(l.b.u, l.b.v, l.b.z + 0.5);
      ctx.save();
      ctx.globalAlpha = la;
      ctx.strokeStyle = l.lives ? l.colour : ink.other;
      ctx.lineWidth = l.lives ? 1.1 : 0.6;
      if (!l.lives) ctx.setLineDash([1.5, 3]);
      ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
      ctx.restore();
    }

    plane(ctx, T, plate, ink, a, dim);

    if (plate.id === "corpus") {
      for (const m of city.marks) {
        const hit = run?.lit.get(m.id);
        const p = T(m.u, m.v, 0.3);
        if (hit) mark(ctx, p, 2.2, hit.colour || ink.faint, null, a);
        else mark(ctx, p, 1.1, null, ink.other, a * 0.4);
      }
    } else {
      for (const m of run?.placed.get(plate.id) || []) {
        const p = T(m.u, m.v, plate.z + 0.5);
        if (m.lives) mark(ctx, p, 3.4, m.colour, null, a);
        else mark(ctx, p, 2.2, null, ink.other, a * 0.4);
      }
    }

    annotate(ctx, T, plate, city, run, ink, a, dim);
  });
}

function runSub(id, run) {
  const c = run.counts;
  switch (id) {
    case "corpus": return `${run.city.total.toLocaleString()} passages`;
    case "dense": return `${c.dense} candidates`;
    case "sparse": return c.sparse ? `${c.sparse} candidates` : "no term matched";
    case "fused": return `${c.both} found by both`;
    case "reranked": return c.biggest && c.biggest.delta
      ? `largest move ${c.biggest.delta > 0 ? "+" : ""}${c.biggest.delta}`
      : `${c.reranked} rescored`;
    case "selected": return `${c.selected} kept`;
    case "answer": return `${c.selected} passages`;
    default: return "";
  }
}

/* ------------------------------------------------------------- captions
   Neutral and factual. Each says what the stage did and what it cost or
   found -- no second sentence explaining why the design is good. */
export function captions(run) {
  if (!run) return [];
  const c = run.counts;
  const big = c.biggest && c.biggest.delta ? c.biggest : null;
  return [
    { title: "Index",
      text: `${run.city.total.toLocaleString()} passages across ${run.city.docs} documents, embedded and indexed ahead of the query.` },
    { title: "Dense retrieval",
      text: `Embedding similarity against the whole index. ${c.dense} candidates.` },
    { title: "BM25",
      text: `Lexical match over the same index, run in parallel. ${c.sparse} candidates. It catches rare tokens embeddings average away.` },
    { title: "Rank fusion",
      text: `Both rankings combined by position rather than score. ${c.both} passages appear in both lists.` },
    { title: "Cross-encoder",
      text: big
        ? `Query and passage scored together rather than compared as separate vectors. Largest move: ${shortDoc(big.source)}, ${big.delta > 0 ? "+" : ""}${big.delta} places.`
        : `Query and passage scored together rather than compared as separate vectors. Order largely unchanged.` },
    { title: "Diversity cap",
      text: `At most two passages per document. ${c.selected} kept.` },
    { title: run.confident ? "Answer" : "Below threshold",
      text: run.confident
        ? `Top score clears the abstention threshold. ${c.selected} passages cited.`
        : `Top score falls below the abstention threshold, so nothing is returned as an answer.` },
  ];
}
