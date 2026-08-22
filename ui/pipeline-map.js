/* The retrieval pipeline, drawn as a technical schematic.
 *
 * Fourth attempt, and the first three are worth recording because each failed
 * for a different reason.
 *
 *   1. A city of towers. A generic isometric illustration that says nothing
 *      true about retrieval -- the buildings had no referent.
 *   2. Stacked plates joined by coloured ribbons. True, but loud: every
 *      candidate was tinted and the connectors read as spaghetti.
 *   3. Exploded plates, monochrome, colour only on survivors. Right language,
 *      wrong motion -- every connector was drawn at full length and then the
 *      plate it pointed at faded in on top of it. Reported as "the lines are
 *      already there, then the square pops up", which is an accurate bug
 *      report: the drawing showed an effect arriving before its cause.
 *
 * This one animates TRAVEL rather than opacity. A passage leaves the plate it
 * was found on, the connector is drawn only as far as the passage has actually
 * got, and the next plate begins to exist as the first passage reaches it.
 * Nothing is ever on screen before the thing that produced it.
 *
 * Three facts about retrieval constrain the layout and are not negotiable:
 *
 *   Stages read LEFT TO RIGHT, because that is how a process is read.
 *
 *   Dense and BM25 share a step and are separated by HEIGHT, not by order.
 *   They run at the same moment. Timing is keyed on `step`, not on position in
 *   the PLATES array, so the two of them animate together -- the previous
 *   version revealed dense before BM25 and drew a sequence that does not exist.
 *
 *   Colour means "this passage is in your answer". It is never decoration.
 */

export const shortDoc = s => String(s).replace(/\.(pdf|docx|pptx|xlsx)$/i, "");

const KX = Math.cos(Math.PI / 6);
const KY = Math.sin(Math.PI / 6);
const project = (u, v, z) => ({ x: (u - v) * KX, y: (u + v) * KY - z });

const FLOW = 21;
const at = (step, lift) => ({ step, u: step * FLOW, v: -step * FLOW, z: lift });

/* label side: 1 above, -1 below. Leader lines run vertically out of the plate
   to the text, the way the reference annotates its diagrams. */
/* `cells` is the matrix each stage is drawn as: [columns, rows], sized to the
   number of candidates that stage actually carries. Every plate is a grid of
   real cells rather than a mesh with marks scattered on it, so a stage reads
   the way a layer of activations reads -- occupied cells bright, empty cells
   present but dark. The index keeps its own treatment because 5,459 passages
   is a field, not a matrix. */
const PLATES = [
  { id: "corpus",   n: "01", ...at(0, 0),   w: 13,  cells: null, lead: 1,
    label: "Index",              term: c => `${c.total.toLocaleString()} passages · ${c.docs} documents` },
  { id: "dense",    n: "02", ...at(1, 15),  w: 6.5, cells: [5, 4], lead: 1,
    label: "Dense retrieval",    term: () => "embedding similarity" },
  { id: "sparse",   n: "03", ...at(1, -15), w: 6.5, cells: [5, 4], lead: -1,
    label: "BM25",               term: () => "lexical match" },
  { id: "fused",    n: "04", ...at(2, 0),   w: 7.5, cells: [5, 4], lead: 1,
    label: "Rank fusion",        term: () => "both rankings combined" },
  { id: "reranked", n: "05", ...at(3, 0),   w: 7.5, cells: [5, 4], lead: -1,
    label: "Cross-encoder",      term: () => "query and passage scored together" },
  { id: "selected", n: "06", ...at(4, 0),   w: 6,   cells: [5, 1], lead: 1,
    label: "Diversity cap",      term: () => "max 2 per document" },
  { id: "answer",   n: "07", ...at(5, 0),   w: 5,   cells: [5, 1], lead: -1,
    label: "Answer",             term: () => "cited passages" },
];

const STEPS = 6;                    // steps 0..5, not plates -- see header
const SPAN = 1 / STEPS;
const arrive = step => step * SPAN; // when passages land on that step

// A passage spends most of a span in flight, and its destination plate starts
// forming slightly before it lands so there is something to land on.
const FLIGHT = 0.88;
const FORM_LEAD = 0.22;
const FORM_HOLD = 0.34;

const clamp01 = t => (t < 0 ? 0 : t > 1 ? 1 : t);
const easeOut = t => 1 - Math.pow(1 - t, 3);
const easeInOut = t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const lerp = (a, b, t) => a + (b - a) * t;

/* Cell `i` of a plate's matrix, in reading order: left to right, top to bottom,
   which is also rank order. Returns the centre and the half-extents, so the
   caller can draw the cell as an isometric quad rather than a point. */
function cellAt(plate, i) {
  const [cols, rows] = plate.cells || [1, 1];
  const cw = (plate.w * 2) / cols, ch = (plate.w * 2) / rows;
  const cx = i % cols, cy = Math.floor(i / cols) % rows;
  const GAP = 0.16;                       // fraction of a cell left as gutter
  return {
    u: plate.u - plate.w + cw * (cx + 0.5),
    v: plate.v - plate.w + ch * (cy + 0.5),
    du: (cw / 2) * (1 - GAP),
    dv: (ch / 2) * (1 - GAP),
  };
}

function drawCell(ctx, T, c, z, fill, stroke, a, lw = 0.8) {
  const p = [T(c.u - c.du, c.v - c.dv, z), T(c.u + c.du, c.v - c.dv, z),
             T(c.u + c.du, c.v + c.dv, z), T(c.u - c.du, c.v + c.dv, z)];
  ctx.save();
  ctx.globalAlpha = a;
  ctx.beginPath();
  ctx.moveTo(p[0].x, p[0].y);
  for (let i = 1; i < 4; i++) ctx.lineTo(p[i].x, p[i].y);
  ctx.closePath();
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke(); }
  ctx.restore();
}

function palette(sources, ink) {
  const slots = [ink.s1, ink.s2, ink.s3];
  const m = new Map();
  sources.forEach((s, i) => m.set(s, i < 3 ? slots[i] : ink.other));
  return src => m.get(src) || ink.other;
}

/* A candidate's home on its stage. Position IS rank -- first cell, top left --
   so the same passage visibly moves between plates when the cross-encoder
   reorders it, which is the one thing the reranker stage has to show. */
function slot(plate, i) {
  return cellAt(plate, i);
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
    // As many candidates as this stage's matrix has cells. Taking more would
    // silently wrap them onto occupied cells; taking a fixed number would leave
    // the wider stages looking half-empty when they were full.
    const [cols, rows] = plate.cells || [1, 1];
    const items = st.items.slice(0, cols * rows);
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
      lit.set(near.id, { colour: survivors.has(it.chunk_id) ? colour(it.source) : null,
                         chunk_id: it.chunk_id });
    }
  }

  // Links, grouped by the plate they arrive at, so the draw loop can emit them
  // in flow order instead of all at once.
  const byTarget = new Map();
  const push = (b, link) => {
    if (!byTarget.has(b)) byTarget.set(b, []);
    byTarget.get(b).push(link);
  };

  // The index feeds both retrievers. Only survivors are drawn out of the index
  // field -- one line per candidate would be 40 lines out of a speckled plane
  // and would bury the plane it starts from.
  const litAt = new Map();
  for (const [id, hit] of lit) {
    const m = city.chunkAt.get(id);
    if (m) litAt.set(hit.chunk_id, { u: m.u, v: m.v, z: 0.3 });
  }
  for (const name of ["dense", "sparse"]) {
    for (const m of placed.get(name) || []) {
      const src = litAt.get(m.chunk_id);
      if (src && m.lives) push(name, { a: src, b: m, lives: true, colour: m.colour });
    }
  }

  const chain = [["dense", "fused"], ["sparse", "fused"], ["fused", "reranked"],
                 ["reranked", "selected"], ["selected", "answer"]];
  for (const [a, b] of chain) {
    const idx = new Map((placed.get(a) || []).map(m => [m.chunk_id, m]));
    for (const m of placed.get(b) || []) {
      const src = idx.get(m.chunk_id);
      if (!src) continue;
      push(b, { a: src, b: m, lives: m.lives, colour: m.colour });
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

/* A plate assembles rather than fades: the outline draws itself corner to
   corner, then the mesh fills in behind it. `f` is 0..1 formation. */
function plane(ctx, T, plate, ink, a, dim, f = 1) {
  const pts = corners(plate.w).map(c => T(plate.u + c.u, plate.v + c.v, plate.z));
  ctx.save();

  // The empty matrix. Every slot the stage can hold is drawn, so a stage that
  // filled four of twenty looks different from one that filled twenty -- with a
  // mesh they looked identical. Cells arrive after the outline closes: an empty
  // frame reads as a plate, loose cells read as debris.
  const cellF = clamp01((f - 0.4) / 0.6);
  if (cellF > 0 && plate.cells) {
    const [cols, rows] = plate.cells;
    // Staggered by column so the matrix fills the way the flow runs, left to
    // right, rather than materialising all at once.
    for (let i = 0; i < cols * rows; i++) {
      const col = i % cols;
      const s = clamp01((cellF - (col / cols) * 0.35) / 0.65);
      if (s <= 0) continue;
      drawCell(ctx, T, cellAt(plate, i), plate.z,
               null, ink.other, a * s * (dim ? 0.14 : 0.26), 0.6);
    }
  }

  // outline, drawn as a growing path around the four edges
  ctx.globalAlpha = a * (dim ? 0.35 : 1);
  ctx.strokeStyle = ink.faint;
  ctx.lineWidth = 0.9;
  const drawn = clamp01(f / 0.5) * 4;
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let e = 0; e < 4; e++) {
    const seg = clamp01(drawn - e);
    if (seg <= 0) break;
    const p = pts[e], q = pts[(e + 1) % 4];
    ctx.lineTo(lerp(p.x, q.x, seg), lerp(p.y, q.y, seg));
  }
  ctx.stroke();

  // Corner vertex marks. The reference pins every plane at its corners, and
  // they are what makes the drawing read as a measured projection rather than
  // a floating quadrilateral.
  if (f > 0.55) {
    const va = a * clamp01((f - 0.55) / 0.45) * (dim ? 0.4 : 1);
    ctx.globalAlpha = va;
    ctx.fillStyle = ink.faint;
    for (const p of pts) ctx.fillRect(p.x - 1.4, p.y - 1.4, 2.8, 2.8);
  }
  ctx.restore();
  return pts;
}

/* Lifted plates get a dotted drop line and a ghost of their own footprint on
   the base plane. Without it the height is ambiguous -- dense and BM25 look
   like they are further along the flow rather than above and below it. */
function dropline(ctx, T, plate, ink, a, f) {
  if (!plate.z || f < 0.6) return;
  const k = a * clamp01((f - 0.6) / 0.4) * 0.45;
  ctx.save();
  ctx.globalAlpha = k;
  ctx.strokeStyle = ink.other;
  ctx.lineWidth = 0.7;
  ctx.setLineDash([1.5, 3]);

  const top = T(plate.u, plate.v, plate.z);
  const foot = T(plate.u, plate.v, 0);
  ctx.beginPath(); ctx.moveTo(top.x, top.y); ctx.lineTo(foot.x, foot.y); ctx.stroke();

  const g = corners(plate.w * 0.62).map(c => T(plate.u + c.u, plate.v + c.v, 0));
  ctx.globalAlpha = k * 0.6;
  ctx.beginPath();
  ctx.moveTo(g[0].x, g[0].y);
  for (let i = 1; i < 4; i++) ctx.lineTo(g[i].x, g[i].y);
  ctx.closePath(); ctx.stroke();
  ctx.restore();
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

/* A passage in flight. The connector exists only as far as the passage has
   travelled, and the passage itself rides the head of it. This is the whole
   correction over the previous version. */
function inflight(ctx, T, link, ink, a, t) {
  if (t <= 0) return;
  const p1 = T(link.a.u, link.a.v, link.a.z + 0.5);
  const p2 = T(link.b.u, link.b.v, link.b.z + 0.5);
  const e = easeInOut(clamp01(t));
  const hx = lerp(p1.x, p2.x, e), hy = lerp(p1.y, p2.y, e);

  ctx.save();
  ctx.globalAlpha = a * (link.lives ? 0.9 : 0.22);
  ctx.strokeStyle = link.lives ? link.colour : ink.other;
  ctx.lineWidth = link.lives ? 1.1 : 0.6;
  if (!link.lives) ctx.setLineDash([1.5, 3]);
  ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(hx, hy); ctx.stroke();
  ctx.restore();

  // The head is only worth drawing while it is actually moving; once it lands,
  // the mark on the destination plate is the same passage and drawing both
  // doubles it.
  if (e < 0.995 && link.lives) {
    mark(ctx, { x: hx, y: hy }, 3.0, link.colour, null, a);
  }
}

/* How far the leader runs out of the plate, in world units, and how much
   vertical room the three rows of text need, in pixels. The fit reads both --
   keeping them here is what stops the solver's idea of a label and the drawn
   label from disagreeing, which is how the BM25 caption came to be clipped. */
const LEADER = 13;
const LABEL_PX = 52;

/* A leader line out of the plate to its label -- the reference's annotation
   device, and the reason labels never collide with the next plate. */
function annotate(ctx, T, plate, city, run, ink, a, dim) {
  const up = plate.lead === 1;
  const from = T(plate.u, plate.v, plate.z + (up ? plate.w : -plate.w));
  const to = T(plate.u, plate.v, plate.z + (up ? plate.w + LEADER : -plate.w - LEADER));

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

/* The route itself, dotted, drawn under everything. Without it the seven
   plates read as seven separate objects rather than one pipeline -- which is
   the single thing the drawing exists to communicate. A schematic shows the
   pipe whether or not anything is flowing through it, so this is drawn during
   a run too, behind the passages.

   Ends are pulled back to each plate's edge so a leg starts outside the plane
   it leaves rather than out of its middle. */
const LEGS = [["corpus", "dense"], ["corpus", "sparse"], ["dense", "fused"],
              ["sparse", "fused"], ["fused", "reranked"],
              ["reranked", "selected"], ["selected", "answer"]];

function spine(ctx, T, ink, formation) {
  const P = id => PLATES.find(x => x.id === id);
  ctx.save();
  ctx.strokeStyle = ink.other;
  ctx.lineWidth = 0.7;
  ctx.setLineDash([2, 4]);
  for (const [x, y] of LEGS) {
    const A = P(x), B = P(y);
    // A leg is only as present as the plate it arrives at.
    const a = Math.min(formation(A.step), formation(B.step));
    if (a <= 0.01) continue;
    const p = T(A.u, A.v, A.z), q = T(B.u, B.v, B.z);
    const L = Math.hypot(q.x - p.x, q.y - p.y) || 1;
    const eA = T(A.u + A.w, A.v, A.z), eB = T(B.u - B.w, B.v, B.z);
    const t0 = Math.min(0.45, Math.hypot(eA.x - p.x, eA.y - p.y) / L);
    const t1 = 1 - Math.min(0.45, Math.hypot(eB.x - q.x, eB.y - q.y) / L);
    ctx.globalAlpha = a * 0.34;
    ctx.beginPath();
    ctx.moveTo(lerp(p.x, q.x, t0), lerp(p.y, q.y, t0));
    ctx.lineTo(lerp(p.x, q.x, t1), lerp(p.y, q.y, t1));
    ctx.stroke();
  }
  ctx.restore();
}

/* An idle drawing is still a drawing of a process, so something has to move or
   the isometric reads as a diagram of a dead system. One faint pulse walks the
   spine on a long cycle -- slow enough to notice only once, which is the point.
   `clock` is milliseconds; pass null to hold it still. */
function pulse(ctx, T, ink, clock) {
  if (clock == null) return;
  const CYCLE = 7200;
  const t = (clock % CYCLE) / CYCLE;
  const head = t * STEPS;
  ctx.save();
  for (let s = 0; s < STEPS; s++) {
    const d = head - s;
    if (d < 0 || d > 1.4) continue;
    const a = Math.sin(clamp01(d / 1.4) * Math.PI) * 0.5;
    const e = clamp01(d);
    const q1 = T(s * FLOW, -s * FLOW, 0);
    const q2 = T((s + 1) * FLOW, -(s + 1) * FLOW, 0);
    ctx.globalAlpha = a;
    ctx.fillStyle = ink.s1;
    ctx.beginPath();
    ctx.arc(lerp(q1.x, q2.x, e), lerp(q1.y, q2.y, e), 1.9, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

export function drawScene(ctx, city, run, ink, W, H, progress = 1, clock = null) {
  ctx.clearRect(0, 0, W, H);
  if (!city) return;

  // An annotation is two things in two different units, and fitting them as
  // one was the bug. The leader line is LEADER world units long and scales
  // with the drawing; the three rows of text are a fixed pixel height and do
  // not. The old probe reserved 42 world units for both -- against a leader
  // that is actually 13 -- so roughly half the vertical budget was reserved
  // for ink that is never drawn, the solver shrank everything to fit it, and
  // the slack showed up as an empty band above the plates.
  const probe = [];
  for (const p of PLATES) {
    for (const c of corners(p.w)) probe.push(project(p.u + c.u, p.v + c.v, p.z));
    const tip = p.w + LEADER;
    probe.push(project(p.u, p.v, p.z + (p.lead === 1 ? tip : -tip)));
  }
  const xs = probe.map(p => p.x), ys = probe.map(p => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);

  const padX = 18;
  const padY = LABEL_PX;              // the text stack only -- see above
  const s = Math.min((W - padX * 2) / (maxX - minX),
                     (H - padY * 2) / (maxY - minY));
  const ox = W / 2 - ((minX + maxX) / 2) * s;
  const oy = H / 2 - ((minY + maxY) / 2) * s;
  const T = (u, v, z) => { const p = project(u, v, z); return { x: ox + p.x * s, y: oy + p.y * s }; };

  // Timing is keyed on step, not on array position, so dense and BM25 -- which
  // run at the same moment -- form at the same moment.
  const formation = step => {
    if (progress >= 1) return 1;
    const t0 = arrive(step) - FORM_LEAD * SPAN;
    return easeOut(clamp01((progress - t0) / ((FORM_LEAD + FORM_HOLD) * SPAN)));
  };
  const flight = step => {
    if (progress >= 1) return 1;
    const t1 = arrive(step);
    return clamp01((progress - (t1 - FLIGHT * SPAN)) / (FLIGHT * SPAN));
  };

  ctx.lineJoin = "round"; ctx.lineCap = "round";

  spine(ctx, T, ink, formation);
  if (progress >= 1) pulse(ctx, T, ink, clock);

  // Strict flow order: the passages arriving at a plate, then the plate they
  // land on. Nothing is ever drawn on top of a connector already on screen.
  PLATES.forEach(plate => {
    const f = formation(plate.step);
    if (f <= 0) return;
    const dim = run?.skipped.has(plate.id);
    const ft = flight(plate.step);

    for (const l of run?.byTarget.get(plate.id) || []) {
      inflight(ctx, T, l, ink, 1, ft);
    }

    dropline(ctx, T, plate, ink, 1, f);
    plane(ctx, T, plate, ink, 1, dim, f);

    // Marks settle after the plate has formed under them.
    const ma = clamp01((f - 0.6) / 0.4);
    if (ma > 0) {
      if (plate.id === "corpus") {
        for (const m of city.marks) {
          const hit = run?.lit.get(m.id);
          const p = T(m.u, m.v, 0.3);
          if (hit) mark(ctx, p, 2.2, hit.colour || ink.faint, null, ma);
          else mark(ctx, p, 1.1, null, ink.other, ma * 0.4);
        }
      } else if (ft >= 0.995) {
        // Occupied cells, filled. Brightness is RANK -- one unit that means the
        // same thing on every plate, so a passage getting brighter between two
        // stages is a real promotion. The scores underneath are in four
        // different units and would not compare.
        const items = run?.placed.get(plate.id) || [];
        for (const m of items) {
          const k = 1 - ((m.rank - 1) / Math.max(items.length, 1)) * 0.7;
          drawCell(ctx, T, m, plate.z, m.lives ? m.colour : ink.other, null,
                   ma * (m.lives ? k : k * 0.4));
        }
      }
    }

    annotate(ctx, T, plate, city, run, ink, f, dim);
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
