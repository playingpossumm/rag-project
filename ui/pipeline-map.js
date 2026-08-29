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

/* Two numbers, not one angle.
 *
 * A single isometric angle drives the horizontal spread and the camera height
 * together, which is why turning it up to get more depth also tipped the whole
 * scene over and looked down on it from above. They are separate here:
 *
 *   SPREAD -- how much the two ground axes fan out horizontally. Lower is
 *             narrower: it compresses the plates sideways without touching
 *             their height, which is the only knob that makes the panels
 *             thinner rather than differently angled.
 *   RISE   -- how far above the ground plane the camera sits. This is the one
 *             that reads as "viewing angle". Near zero is eye level; a true
 *             isometric is 0.5; the old 60-degree setting was 0.87, which is
 *             looking down on the drawing from most of the way above it.
 *
 * Height (z) always projects 1:1 to screen y, so lowering RISE flattens the
 * ground plane without collapsing the vertical separation between dense
 * retrieval and BM25.
 */
const SPREAD = 0.68;
const RISE = 0.15;
const KX = SPREAD;
const KY = RISE;
const project = (u, v, z) => ({ x: (u - v) * KX, y: (u + v) * KY - z });

/* Step spacing in world units, and it is NOT held constant on screen.
   The gap between stages is 2 * FLOW * SPREAD, so narrowing the plates narrows
   the gap by the same factor and the composition stays proportional. Pinning
   the on-screen gap instead -- which is what I did when SPREAD dropped to 0.68
   -- shrinks the plates while leaving them just as far apart, so the drawing
   turns into small plates adrift in a wide frame and the fit can only scale it
   to the width it does not need. */
const FLOW = 34;
const at = (step, lift) => ({ step, u: step * FLOW, v: -step * FLOW, z: lift });

/* label side: 1 above, -1 below. Leader lines run vertically out of the plate
   to the text, the way the reference annotates its diagrams. */
/* `layers` is how many sheets deep a stage is drawn, and it is not decoration:
   it tracks how much is still in play. The index is six sheets of dense field
   because 5,459 passages is a lot of paper; retrieval cuts that to twenty and
   the block thins; by the answer it is a single sheet of five rows. The funnel
   is the shape of the drawing, not something written underneath it.

   Every sheet carries its stage's matrix. Empty frames behind a filled face
   read as a picture frame around the real thing; filled sheets read as a
   volume of data, which is what a stage actually is.

   `cells` is the matrix each stage is drawn as: [columns, rows], sized to the
   number of candidates that stage actually carries. Every plate is a grid of
   real cells rather than a mesh with marks scattered on it, so a stage reads
   the way a layer of activations reads -- occupied cells bright, empty cells
   present but dark. The index keeps its own treatment because 5,459 passages
   is a field, not a matrix. */
const SPEC = [
  { id: "corpus", layers: 6,   n: "01", step: 0, lift: 0,   w: 13,  cells: null,   lead: 1,
    label: "Index",              term: c => `${c.total.toLocaleString()} passages · ${c.docs} documents` },
  { id: "dense", layers: 4,    n: "02", step: 1, lift: 15,  w: 6.5, cells: [5, 4], lead: 1, shape: "field",
    label: "Dense retrieval",    term: () => "embedding similarity" },
  { id: "sparse", layers: 4,   n: "03", step: 1, lift: -15, w: 6.5, cells: [5, 4], lead: -1, shape: "bars",
    label: "BM25",               term: () => "lexical match" },
  { id: "fused", layers: 3,    n: "04", step: 2, lift: 0,   w: 7.5, cells: [5, 4], lead: 1, shape: "merge",
    label: "Rank fusion",        term: () => "both rankings combined" },
  { id: "reranked", layers: 3, n: "05", step: 3, lift: 0,   w: 7.5, cells: [5, 4], lead: -1, shape: "sort",
    label: "Cross-encoder",      term: () => "scored as a pair" },
  { id: "selected", layers: 2, n: "06", step: 4, lift: 0,   w: 6,   cells: [1, 5], lead: 1, shape: "gate",
    label: "Diversity cap",      term: () => "max 2 per document" },
  // Wider than the stages before it, and a single row rather than a column:
  // five balls in a column at this scale overlap into one blob, and the
  // answer is the one place on the drawing where the COUNT has to read.
  { id: "answer", layers: 1,   n: "07", step: 5, lift: 0,   w: 7,   cells: [5, 1], lead: -1,
    label: "Answer",             term: () => "cited passages" },
];

/* A plate is a quad in world space spanned by two vectors, and that is the
 * only thing that separates the two looks:
 *
 *   flat    -- spans the two ground axes. The plate lies down, like a floor
 *              tile, which is what this drawing has always done.
 *   upright -- spans one ground axis and the vertical. The plate stands up,
 *              like a panel, and a row of them reads as a stack of layers with
 *              data passing through.
 *
 * The FLOW IS UNCHANGED in both: left to right, dense and BM25 separated by
 * height at the same step. Only the plane the matrix is drawn in rotates.
 *
 * Both basis vectors are one world unit and project to one screen unit, so a
 * plate with equal half-extents reads as a square in either orientation.
 */
const BASIS = {
  flat:    { a: { u: 1, v: 0, z: 0 }, b: { u: 0, v: 1, z: 0 },
             n: { u: 0, v: 0, z: 1 } },
  upright: { a: { u: 0, v: -1, z: 0 }, b: { u: 0, v: 0, z: 1 },
             n: { u: 1, v: 0, z: 0 } },
};

// How far apart the panels within one stage sit, along the plate's normal.
const LAYER_GAP = 2.4;

const LAYOUT = {};

function platesFor(variant = "flat") {
  if (LAYOUT[variant]) return LAYOUT[variant];
  const basis = BASIS[variant] || BASIS.flat;
  LAYOUT[variant] = SPEC.map(d => ({
    ...d,
    basis,
    h: d.w,                          // square in the plate's own plane
    u: d.step * FLOW,
    v: -d.step * FLOW,
    z: d.lift,
    // A stage is a block of panels, not one panel. The stack is centred on the
    // plate position so the flow line still runs through the middle of it, and
    // `front` is the face nearest the viewer -- the one the matrix is drawn on.
    front: ((d.layers - 1) / 2) * LAYER_GAP,
  }));
  return LAYOUT[variant];
}

/* A point on a plate, in world space. `s` runs along the plate's first axis
   and `t` along its second, both measured from the centre. Everything that
   draws on a plate goes through here, so rotating the plate rotates the
   matrix, the cells and the corners together. */
function ptAt(plate, s, t, k = plate.front) {
  const B = plate.basis;
  return {
    u: plate.u + s * B.a.u + t * B.b.u + k * B.n.u,
    v: plate.v + s * B.a.v + t * B.b.v + k * B.n.v,
    z: plate.z + s * B.a.z + t * B.b.z + k * B.n.z,
  };
}

/* The offset of layer `i`, counted from the back of the stack. */
const layerAt = (plate, i) => plate.front - i * LAYER_GAP;

const plateCorners = (plate, k = plate.front) => [[-1, -1], [1, -1], [1, 1], [-1, 1]]
  .map(([i, j]) => ptAt(plate, i * plate.w, j * plate.h, k));

/* Where a plate's leader line ends. The fit and the label drawing both read
   this, so they cannot disagree about how much room an annotation needs --
   the disagreement that clipped the BM25 caption. */
function leaderTip(plate) {
  return { u: plate.u, v: plate.v,
           z: plate.z + plate.lead * (plate.w + LEADER) };
}

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

/* Cell `i` of a plate's matrix, in reading order: left to right, top to
   bottom, which is also rank order. Coordinates are in the plate's own plane,
   so the same code lays out a floor tile and an upright panel. */
function cellAt(plate, i, k = plate.front) {
  const [cols, rows] = plate.cells || [1, 1];
  const cw = (plate.w * 2) / cols, ch = (plate.h * 2) / rows;
  const cx = i % cols, cy = Math.floor(i / cols) % rows;
  const GAP = 0.16;                       // fraction of a cell left as gutter
  const s = -plate.w + cw * (cx + 0.5);
  const t = plate.h - ch * (cy + 0.5);    // first row at the top
  return { ...ptAt(plate, s, t, k), s, t, k,
           ds: (cw / 2) * (1 - GAP), dt: (ch / 2) * (1 - GAP) };
}

function drawCell(ctx, T, plate, c, fill, stroke, a, lw = 0.8) {
  const q = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([i, j]) => {
    const w = ptAt(plate, c.s + i * c.ds, c.t + j * c.dt, c.k ?? plate.front);
    return T(w.u, w.v, w.z);
  });
  ctx.save();
  ctx.globalAlpha = a;
  ctx.beginPath();
  ctx.moveTo(q[0].x, q[0].y);
  for (let i = 1; i < 4; i++) ctx.lineTo(q[i].x, q[i].y);
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
export function buildRun(trace, city, ink, variant = "flat") {
  if (!trace || !city) return null;
  const byName = new Map(trace.stages.map(s => [s.name, s]));
  const final = byName.get("selected")?.items || [];
  const colour = palette([...new Set(final.map(i => i.source))], ink);
  const survivors = new Set(final.map(i => i.chunk_id));

  const placed = new Map();
  for (const plate of platesFor(variant)) {
    const st = byName.get(plate.id);
    if (!st) continue;
    // As many candidates as this stage's matrix has cells. Taking more would
    // silently wrap them onto occupied cells; taking a fixed number would leave
    // the wider stages looking half-empty when they were full.
    const [cols, rows] = plate.cells || [1, 1];
    const items = st.items.slice(0, cols * rows);
    // No `z: plate.z` here. slot() already returns the cell's own height, and
    // overwriting it sent every cable to the plate's centre plane while the
    // square it was flying to sat on its layer -- the line and the lit cell
    // disagreeing by exactly the layer offset.
    placed.set(plate.id, items.map((it, i) => ({
      ...it, ...slot(plate, i, items.length),
      lives: survivors.has(it.chunk_id),
      colour: survivors.has(it.chunk_id) ? colour(it.source) : null,
    })));
  }
  const sel = placed.get("selected") || [];
  const ap = platesFor(variant).find(p => p.id === "answer");
  placed.set("answer", sel.map((m, i) => ({ ...m, ...slot(ap, i, sel.length) })));

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

  // The index feeds both retrievers, and every candidate is drawn leaving it.
  // Drawing only the survivors made the widest leg of the pipeline the
  // thinnest part of the drawing: three lines where 864 passages become 20,
  // against twenty lines on legs that change nothing about the count. The
  // non-survivors are near-invisible, so the plane underneath still reads.
  const litAt = new Map();
  for (const [id, hit] of lit) {
    const m = city.chunkAt.get(id);
    if (m) litAt.set(hit.chunk_id, { u: m.u, v: m.v, z: 0.3 });
  }
  for (const name of ["dense", "sparse"]) {
    for (const m of placed.get(name) || []) {
      const src = litAt.get(m.chunk_id);
      if (src) push(name, { a: src, b: m, lives: m.lives, colour: m.colour });
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

  // The passages the cap threw out. Ranked highly enough by the cross-encoder
  // to be in contention, and then stopped because their document already had
  // two -- which is the entire mechanism, and the drawing had no way to show
  // it. They travel to the cap's face and stop against it.
  const kept = new Set((placed.get("selected") || []).map(m => m.chunk_id));
  const capPlate = platesFor(variant).find(p => p.id === "selected");
  const rejected = (placed.get("reranked") || []).filter(m => !kept.has(m.chunk_id));
  // Five, fanned across the plate's face. Thirteen lines converging on one
  // point is a red hatch, not thirteen stopped passages, and the count that
  // matters is already on the label under the plate.
  const shown = rejected.slice(0, 5);
  shown.forEach((m, i) => {
    const spread = (i - (shown.length - 1) / 2) * (capPlate.w * 0.42);
    push("selected", {
      a: m,
      b: { u: capPlate.u + spread * 0.5, v: capPlate.v - spread * 0.5,
           z: capPlate.z },
      lives: false, colour: null, stopped: true,
    });
  });

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
    skipped: new Set(SPEC.map(p => p.id).filter(id => byName.get(id)?.skipped)),
  };
}

/* -------------------------------------------------------------- drawing */
const corners = w => [{ u: -w, v: -w }, { u: w, v: -w }, { u: w, v: w }, { u: -w, v: w }];

/* A plate assembles rather than fades: the outline draws itself corner to
   corner, then the mesh fills in behind it. `f` is 0..1 formation. */
function plane(ctx, T, plate, ink, a, dim, f = 1) {
  const pts = plateCorners(plate).map(c => T(c.u, c.v, c.z));
  ctx.save();

  // The panels behind the front face. A stage is a block of them rather than a
  // single sheet -- which is what a layer of a network looks like, and it also
  // gives the drawing depth that a lone quad at a shallow angle does not have.
  // They are frames only: the matrix belongs to the face you are looking at,
  // and repeating it on every panel would read as five stages, not one.
  const layers = plate.layers || 1;
  const back = clamp01((f - 0.25) / 0.5);
  const cellsIn = clamp01((f - 0.4) / 0.6);

  // The sheets behind the face, drawn back to front so nearer ones overlap.
  // Each carries its stage's matrix: an empty frame behind a filled face reads
  // as a picture frame around the real thing, where a filled sheet reads as a
  // volume of data, which is what a stage is.
  for (let i = layers - 1; i >= 1; i--) {
    const k = layerAt(plate, i);
    // Only a little dimmer with depth. Fading them out turned the block into a
    // ghost of itself; a layer of a network is not less real for being behind
    // the one in front of it, and the stack has to read as one solid object.
    const depth = 1 - (i / layers) * 0.22;
    // Only the stages that ARE a matrix carry one on their back sheets. A
    // scatter or a set of bars drawn over three lattices is still a lattice,
    // which is why the first attempt at this changed nothing on screen.
    if (cellsIn > 0 && plate.cells && !plate.shape) {
      const [cols, rows] = plate.cells;
      for (let n = 0; n < cols * rows; n++) {
        drawCell(ctx, T, plate, cellAt(plate, n, k),
                 null, ink.other, a * cellsIn * depth * (dim ? 0.16 : 0.34), 0.55);
      }
    }
    const q = plateCorners(plate, k).map(c => T(c.u, c.v, c.z));
    ctx.globalAlpha = a * back * depth * (dim ? 0.26 : 0.5);
    ctx.strokeStyle = ink.other;
    ctx.lineWidth = 0.7;
    ctx.beginPath();
    ctx.moveTo(q[0].x, q[0].y);
    for (let e = 1; e < 4; e++) ctx.lineTo(q[e].x, q[e].y);
    ctx.closePath();
    ctx.stroke();
  }

  // The four edges joining the back sheet to the front one. Without them the
  // sheets are separate quads that happen to line up; with them the stage is a
  // single extruded block, which is what makes the depth read at a glance.
  if (layers > 1) {
    const bk = plateCorners(plate, layerAt(plate, layers - 1))
      .map(c => T(c.u, c.v, c.z));
    ctx.globalAlpha = a * back * (dim ? 0.2 : 0.42);
    ctx.strokeStyle = ink.other;
    ctx.lineWidth = 0.7;
    for (let e = 0; e < 4; e++) {
      ctx.beginPath();
      ctx.moveTo(bk[e].x, bk[e].y);
      ctx.lineTo(pts[e].x, pts[e].y);
      ctx.stroke();
    }
  }

  // The empty stage, drawn as the operation it performs. Every one of these
  // used to be the same lattice of cells, which said "a stage happened here"
  // and nothing about which stage. `cells` still governs where a passage sits;
  // only the empty structure changes.
  const cellF = clamp01((f - 0.4) / 0.6);
  if (cellF > 0 && plate.cells) {
    const [cols, rows] = plate.cells;
    const base = a * (dim ? 0.14 : 0.26);

    if (plate.shape === "field") {
      // Dense retrieval compares position in a continuous space. A lattice
      // implies discrete slots it does not have, so this is a scatter --
      // deterministic, so it never shimmers between frames.
      ctx.globalAlpha = base * cellF * 2.4;
      ctx.fillStyle = ink.other;
      for (let n = 0; n < 70; n++) {
        const j = k => ((Math.sin(n * 12.9898 + k * 78.233) * 43758.5) % 1 + 1) % 1;
        const w = ptAt(plate, (j(1) * 2 - 1) * plate.w * 0.92,
                       (j(2) * 2 - 1) * plate.h * 0.92, plate.front);
        const p = T(w.u, w.v, w.z);
        ctx.beginPath(); ctx.arc(p.x, p.y, 1.1, 0, Math.PI * 2); ctx.fill();
      }
      ctx.globalAlpha = 1;
    } else if (plate.shape === "bars") {
      // BM25 matches whole words, so its unit is a band across the plate
      // rather than a point on it. Drawn as rows, which is also the shape of
      // the word-match grid in the Detail panel.
      for (let r = 0; r < rows; r++) {
        const t = plate.h - ((plate.h * 2) / rows) * (r + 0.5);
        const A = ptAt(plate, -plate.w * 0.88, t, plate.front);
        const B = ptAt(plate, plate.w * 0.88, t, plate.front);
        const p = T(A.u, A.v, A.z), q = T(B.u, B.v, B.z);
        ctx.globalAlpha = base * cellF * 2.2;
        ctx.strokeStyle = ink.other; ctx.lineWidth = 2.4;
        ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke();
      }
      ctx.globalAlpha = 1;
    } else if (plate.shape === "gate") {
      // The cap is the one stage whose job is refusing, so its face is a
      // grate: bars across it, with gaps only where something gets through.
      for (let r = 0; r < rows; r++) {
        const t = plate.h - ((plate.h * 2) / rows) * (r + 0.5);
        const A = ptAt(plate, -plate.w * 0.95, t, plate.front + 0.35);
        const B = ptAt(plate, plate.w * 0.95, t, plate.front + 0.35);
        const p = T(A.u, A.v, A.z), q = T(B.u, B.v, B.z);
        ctx.globalAlpha = a * (dim ? 0.3 : 0.85);
        ctx.strokeStyle = ink.faint; ctx.lineWidth = 2.2;
        ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke();
      }
      ctx.globalAlpha = 1;
    } else {
      for (let i = 0; i < cols * rows; i++) {
        const col = i % cols;
        const st = clamp01((cellF - (col / cols) * 0.35) / 0.65);
        if (st <= 0) continue;
        drawCell(ctx, T, plate, cellAt(plate, i), null, ink.other, base * st, 0.6);
      }
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

  const g = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
    .map(([i, j]) => ptAt(plate, i * plate.w * 0.62, j * plate.h * 0.62))
    .map(c => T(c.u, c.v, 0));
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

  // A blocked passage stops short of the plate it was heading for, and the
  // last thing drawn on its path is the thing that stopped it.
  const STOPPED = "#f0685f";
  const wall = link.stopped ? 0.72 : 1;
  const held = Math.min(e, wall);
  const bx = lerp(p1.x, p2.x, held), by = lerp(p1.y, p2.y, held);

  ctx.save();
  // 0.18 made forty lines out of the index add up to nothing, so the widest
  // leg of the pipeline read as the emptiest. The funnel is the point of the
  // drawing: forty leave the index, twenty cross fusion and the reranker, five
  // reach the answer, and that has to be visible without reading a label.
  ctx.globalAlpha = a * (link.lives ? 0.85 : link.stopped ? 0.42 : 0.34);
  ctx.strokeStyle = link.lives ? link.colour : link.stopped ? STOPPED : ink.other;
  ctx.lineWidth = link.lives ? 1.2 : 0.75;
  if (!link.lives) ctx.setLineDash([1.5, 3]);
  ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(bx, by); ctx.stroke();
  ctx.restore();

  if (link.stopped) {
    // A short bar across the path, at the point it did not get past.
    if (e >= wall) {
      const dx = p2.x - p1.x, dy = p2.y - p1.y;
      const len = Math.hypot(dx, dy) || 1;
      const nx = -dy / len * 3.4, ny = dx / len * 3.4;
      ctx.save();
      ctx.globalAlpha = a * 0.7;
      ctx.strokeStyle = STOPPED; ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(bx + nx, by + ny); ctx.lineTo(bx - nx, by - ny);
      ctx.stroke();
      ctx.restore();
    }
    return;
  }

  // The head, while it is moving: a ball, which is the only round solid on the
  // drawing and therefore the only thing that reads as travelling. It used to
  // be the same 3-unit mark the plates are covered in.
  if (e < 0.995 && link.lives) {
    ctx.save();
    ctx.globalAlpha = a * 0.22;
    ctx.beginPath(); ctx.arc(hx, hy, 6.4, 0, Math.PI * 2);
    ctx.fillStyle = link.colour; ctx.fill();
    ctx.globalAlpha = a;
    ctx.beginPath(); ctx.arc(hx, hy, 3.4, 0, Math.PI * 2);
    ctx.fillStyle = link.colour; ctx.fill();
    ctx.restore();
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
function annotate(ctx, T, plate, city, run, ink, a, dim, room, W) {
  const up = plate.lead === 1;
  const tip = leaderTip(plate);
  const from = T(plate.u, plate.v, plate.z + plate.lead * plate.w);
  const to = T(tip.u, tip.v, tip.z);

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
  // A label is text at a fixed pixel size on a drawing that scales, so below
  // some canvas width the labels of neighbouring stages run into each other --
  // "Index" landing on top of "EMBEDDING SIMILARITY". `room` is the on-screen
  // gap between stages, so each label is measured against the space it actually
  // has and gives way rather than overlapping its neighbour.
  const shown = [];
  for (const r of rows) {
    ctx.font = r.font;
    if (ctx.measureText(r.text).width <= room) { shown.push(r); continue; }

    if (r === rows[1]) {
      // The name is the one row worth shrinking to keep: a stage with no name
      // is not a stage. Two steps down, then it is truncated rather than
      // allowed to collide.
      let size = 12;
      while (size > 9) {
        size -= 1;
        ctx.font = `600 ${size}px Inter, system-ui, sans-serif`;
        if (ctx.measureText(r.text).width <= room) break;
      }
      let text = r.text;
      while (text.length > 4 && ctx.measureText(text + "…").width > room) {
        text = text.slice(0, -1);
      }
      shown.push({ ...r, font: ctx.font,
                   text: text === r.text ? text : text.trimEnd() + "…" });
    }
    // The number and the detail line are dropped. The detail is the first to
    // go because it is the longest and the least load-bearing -- the drawing
    // still says which stage this is without it.
  }

  const LH = 14;
  // Above: the last row sits nearest the plate, so the stack grows away from
  // it and still reads number, name, detail from the top down.
  const first = up ? to.y - (shown.length - 1) * LH - 4 : to.y + 6;
  shown.forEach((r, i) => {
    ctx.fillStyle = r.fill; ctx.font = r.font;
    // Nudged back inside the canvas rather than centred off the edge of it.
    // The first and last stages are half a label wider than the drawing they
    // sit on, and the reserve at the top of drawScene is a constant while what
    // has to fit is measured text -- so the index's "5,459 passages · 36
    // documents" was being clipped to "459 passages · 36 documents", a wrong
    // number that reads like a right one. The leader line still points at the
    // plate, so a few pixels of offset costs nothing; a clipped digit costs
    // the reader a fact.
    const half = ctx.measureText(r.text).width / 2 + 2;
    const x = W ? Math.min(Math.max(to.x, half), W - half) : to.x;
    ctx.fillText(r.text, x, first + i * LH);
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

function spine(ctx, T, ink, formation, variant) {
  const P = id => platesFor(variant).find(x => x.id === id);
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

/* The index plate's sheets, rendered once and stamped from there.

   The plate carries a mark per sampled passage on each of its sheets -- 260 a
   sheet, six sheets -- and that one block was the most expensive thing the
   page drew, in a profile where over half the time was raster. It is also the
   only part of the drawing that is identical in every frame: the sheets do not
   move, and only the front one ever lights up.

   Which passages are lit does change between answers, so the run is part of
   the key rather than assumed away. On the hero nothing is lit and the key
   never changes, so the layer is painted exactly once.

   Stamped in place rather than composited over the top, so the draw order is
   untouched: the sheets still sit above the plate they lie on and below the
   annotation that labels it.

   Two entries, because two canvases can be live at once -- the hero and an
   answer being replayed -- and a single slot would have them evicting each
   other every frame, which is slower than not caching at all. */
const SHEETS = new Map();
const SHEET_CAP = 2;

/* Identity for objects that have no id of their own. Two different runs must
   not share a cached layer, and the same run across frames must. */
const IDS = new WeakMap();
let nextId = 0;
function idOf(o) {
  if (!o) return "-";
  let v = IDS.get(o);
  if (v === undefined) IDS.set(o, (v = ++nextId));
  return v;
}

function sheetLayer(W, H, dpr, key, paint) {
  const hit = SHEETS.get(key);
  if (hit && hit.w === W && hit.h === H && hit.dpr === dpr) return hit.cv;

  const cv = document.createElement("canvas");
  cv.width = Math.round(W * dpr);
  cv.height = Math.round(H * dpr);
  const c = cv.getContext("2d");
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.lineJoin = "round";
  c.lineCap = "round";
  paint(c);

  SHEETS.delete(key);
  SHEETS.set(key, { cv, w: W, h: H, dpr });
  while (SHEETS.size > SHEET_CAP) SHEETS.delete(SHEETS.keys().next().value);
  return cv;
}

// The device ratio the caller already put into the transform. Reading it back
// rather than asking devicePixelRatio again keeps the layer at exactly the
// resolution of the canvas it will be stamped onto -- guessing would resample
// it, and this plate is nothing but one-pixel marks.
function sheetDpr(ctx) {
  const t = typeof ctx.getTransform === "function" ? ctx.getTransform() : null;
  return t && t.a ? t.a : Math.min(devicePixelRatio || 1, 2);
}

/* The extent of the drawing in projected units, memoised per variant. It is a
   pure function of the layout, and the layout is itself memoised, so computing
   it per frame was work whose answer could not change. */
const BOUNDS = {};
function boundsFor(variant, PL) {
  if (BOUNDS[variant]) return BOUNDS[variant];
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  const see = (p) => {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  };
  for (const p of PL) {
    // Front face AND the deepest sheet. A stage is drawn as a block, and
    // measuring only the face it presents leaves the back of the block outside
    // the box the fit solves for -- which is why the index was clipped by the
    // left edge of the frame.
    for (const c of plateCorners(p)) see(project(c.u, c.v, c.z));
    for (const c of plateCorners(p, layerAt(p, (p.layers || 1) - 1)))
      see(project(c.u, c.v, c.z));
    const t = leaderTip(p);
    see(project(t.u, t.v, t.z));
  }
  return (BOUNDS[variant] = { minX, maxX, minY, maxY });
}

export function drawScene(ctx, city, run, ink, W, H, progress = 1, clock = null,
                          variant = "flat") {
  ctx.clearRect(0, 0, W, H);
  if (!city) return;

  // An annotation is two things in two different units, and fitting them as
  // one was the bug. The leader line is LEADER world units long and scales
  // with the drawing; the three rows of text are a fixed pixel height and do
  // not. The old probe reserved 42 world units for both -- against a leader
  // that is actually 13 -- so roughly half the vertical budget was reserved
  // for ink that is never drawn, the solver shrank everything to fit it, and
  // the slack showed up as an empty band above the plates.
  const PL = platesFor(variant);
  const { minX, maxX, minY, maxY } = boundsFor(variant, PL);

  // Labels are centred on their plate, so the first and last ones hang off the
  // ends of the drawing by roughly half their own width. Eighteen pixels was
  // enough when the fit still reserved world space for labels; it is not now,
  // and "5,459 PASSAGES · 36 DOCUMENTS" was losing its first word.
  // 18, not the 122 this carried for months. That reservation was for the
  // column layout, where labels ran off the sides of the plates; the row layout
  // stacks them above and below and needs almost nothing here. Left behind when
  // the column was reverted, it was quietly throwing away a quarter of the
  // canvas width -- and width is what this drawing is limited by, so it was
  // scaling the whole thing down by a quarter.
  const padX = 18;
  const padY = LABEL_PX;              // the text stack only -- see above
  const s = Math.min((W - padX * 2) / (maxX - minX),
                     (H - padY * 2) / (maxY - minY));
  const ox = W / 2 - ((minX + maxX) / 2) * s;
  const oy = H / 2 - ((minY + maxY) / 2) * s;
  const T = (u, v, z) => { const p = project(u, v, z); return { x: ox + p.x * s, y: oy + p.y * s }; };

  // Every stage is drawn in full, from the first frame, always.
  //
  // These plates used to assemble in sequence, which meant that half way
  // through a replay the drawing showed one stage and six empty spaces -- a
  // picture of a pipeline that does not exist. The structure is not what
  // happens when you ask a question; it was built before you asked. What
  // happens is that passages travel through it, and that is what `flight`
  // below animates.
  //
  // Kept as a function rather than deleted because the plate loop reads it per
  // stage and a future stage may want to fade for a reason that is real.
  const formation = () => 1;
  const flight = step => {
    if (progress >= 1) return 1;
    const t1 = arrive(step);
    return clamp01((progress - (t1 - FLIGHT * SPAN)) / (FLIGHT * SPAN));
  };

  ctx.lineJoin = "round"; ctx.lineCap = "round";

  spine(ctx, T, ink, formation, variant);
  if (progress >= 1) pulse(ctx, T, ink, clock);

  // Strict flow order: the passages arriving at a plate, then the plate they
  // land on. Nothing is ever drawn on top of a connector already on screen.
  PL.forEach(plate => {
    const f = formation();
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
        // Every sheet of the index carries the field, not just the front
        // one. 5,459 passages is the one quantity on this drawing that is
        // genuinely large, and a stack of dense sheets is what that looks
        // like. Only the front sheet lights up: a passage is one passage, and
        // repeating its colour six times would claim six.
        //
        const sheets = plate.layers || 1;
        const paintSheets = (c, alpha) => {
          for (let i = sheets - 1; i >= 0; i--) {
            const k = layerAt(plate, i);
            const depth = i === 0 ? 1 : 1 - (i / sheets) * 0.25;
            for (const m of city.marks) {
              const w = ptAt(plate, m.u, m.v, k);
              const p = T(w.u, w.v, w.z);
              const hit = i === 0 ? run?.lit.get(m.id) : null;
              if (hit) mark(c, p, 2.2, hit.colour || ink.faint, null, alpha);
              else mark(c, p, 1.1, null, ink.other, alpha * 0.62 * depth);
            }
          }
        };
        // Cached only once the plate has finished forming. While `ma` is ramping
        // it is a different picture every frame, and keying the cache on it
        // allocated a full-size canvas per frame -- slower than drawing
        // straight to the target, which is what the ramp does instead.
        //
        // Everything the resting picture depends on is in the key: the geometry
        // (fixed by the size and the variant), the two inks, and the run that
        // decides what is lit.
        if (ma < 1) {
          paintSheets(ctx, ma);
        } else {
          const key = [variant, idOf(city), idOf(run), ink.other, ink.faint]
            .join("|");
          ctx.drawImage(
            sheetLayer(W, H, sheetDpr(ctx), key, (c) => paintSheets(c, 1)),
            0, 0, W, H);
        }
      } else if (plate.id === "answer" && ft >= 0.995) {
        // The end of the pipeline, drawn as the only balls that come to rest.
        // Five cells here were indistinguishable from five cells on the five
        // plates before them, so the drawing ended without saying so.
        for (const m of run?.placed.get("answer") || []) {
          const p = T(m.u, m.v, m.z + 0.6);
          ctx.save();
          ctx.globalAlpha = ma * 0.20;
          ctx.beginPath(); ctx.arc(p.x, p.y, 6.6, 0, Math.PI * 2);
          ctx.fillStyle = m.colour || ink.faint; ctx.fill();
          ctx.globalAlpha = ma;
          ctx.beginPath(); ctx.arc(p.x, p.y, 3.2, 0, Math.PI * 2);
          ctx.fillStyle = m.colour || ink.faint; ctx.fill();
          ctx.restore();
        }
      } else if (ft >= 0.995) {
        // Occupied cells, filled. Brightness is RANK -- one unit that means the
        // same thing on every plate, so a passage getting brighter between two
        // stages is a real promotion. The scores underneath are in four
        // different units and would not compare.
        const items = run?.placed.get(plate.id) || [];
        for (const m of items) {
          const k = 1 - ((m.rank - 1) / Math.max(items.length, 1)) * 0.7;
          drawCell(ctx, T, plate, m, m.lives ? m.colour : ink.other, null,
                   ma * (m.lives ? k : k * 0.4));
        }
      }
    }

    // The horizontal room a stage's label has is the gap to the next stage.
    // Dense and BM25 share a step and are separated vertically, so they each
    // get the full gap rather than half of it.
    annotate(ctx, T, plate, city, run, ink, f, dim,
             2 * FLOW * SPREAD * s * 0.98, W);
  });
}

function runSub(id, run) {
  const c = run.counts;
  switch (id) {
    case "corpus": return `${run.city.total.toLocaleString()} passages`;
    case "dense": return `${c.dense} candidates`;
    case "sparse": return c.sparse ? `${c.sparse} candidates` : "no term matched";
    case "fused": return `${c.both} in both lists`;
    // "largest move +15" reads as a score, not a change of position. The plate
    // label says what happened; the caption below carries which passage moved.
    case "reranked": return `${c.reranked} rescored`;
    case "selected": return `${c.selected} kept`;
    case "answer": return `${c.selected} passages`;
    default: return "";
  }
}

/* ------------------------------------------------------------- captions
   Each caption says three things: what the stage did, what came out of it, and
   how the next set is chosen. A count on its own -- "20 candidates" -- tells
   you the size of something without telling you what it is or why it shrank.

   Counts are real, taken from the run. Mechanism is described without quoting
   configuration numbers that a setting could change underneath the sentence. */
export function captions(run) {
  if (!run) return [];
  const c = run.counts;
  const big = c.biggest && c.biggest.delta ? c.biggest : null;
  return [
    { title: "Index",
      text: `${run.city.docs} documents were split into ${run.city.total.toLocaleString()} `
          + `overlapping passages. They overlap so that a sentence crossing a `
          + `boundary survives whole in one of them. Each was turned into a `
          + `vector once, before any question was asked, and every stage after `
          + `this one narrows the set.` },

    { title: "Dense retrieval",
      text: `The question becomes a vector the same way the passages did, and `
          + `the ${c.dense} nearest by cosine similarity are kept. This finds `
          + `passages that mean the same thing as the question even when they share `
          + `none of its words.` },

    { title: "BM25",
      text: c.sparse
        ? `The same index searched a second way, by word overlap, weighting rare `
          + `words far above common ones. ${c.sparse} candidates. It runs alongside `
          + `dense retrieval rather than after it, and catches exact terms that `
          + `an embedding averages away, such as a species name or a symbol.`
        : `No word in the question appears in the index, so lexical search returned `
          + `nothing and the result rests on dense retrieval alone.` },

    { title: "Rank fusion",
      text: `The two lists are merged by POSITION rather than score, because a `
          + `cosine similarity and a BM25 weight are different units and cannot be `
          + `compared. A passage both methods rank highly rises above one `
          + `only a single method liked. ${c.both} of them appeared in both lists.` },

    { title: "Cross-encoder",
      text: `Every stage so far compared the question and the passage separately. `
          + `Here a second model reads them together, one pair at a time. It is `
          + `slower, and much better at telling a passage that mentions the subject `
          + `from one that answers the question. `
          + (big
              ? `Biggest correction: ${shortDoc(big.source)} moved `
                + `${big.delta > 0 ? "up" : "down"} ${Math.abs(big.delta)} places.`
              : `The order barely changed, which means the first two stages already `
                + `had it roughly right.`) },

    { title: "Diversity cap",
      text: `At most two passages from any one document, so a single thorough `
          + `document cannot fill every slot and hide a second source that also `
          + `answers. ${c.selected} kept.` },

    { title: run.confident ? "Answer" : "Below threshold",
      text: run.confident
        ? `The ${c.selected} passages above are what the answer is drawn from and `
          + `what gets cited. Nothing outside them reached the answer.`
        : `The best passage scored below this set's cut-off. That is the score `
          + `under which passages from these documents usually turn out not to `
          + `hold the answer, so nothing is returned. The candidates are still `
          + `listed, so you can see what was considered and judge for yourself.` },
  ];
}
