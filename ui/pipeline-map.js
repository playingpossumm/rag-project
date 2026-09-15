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
  { id: "dense", layers: 1,    n: "02", step: 1, lift: 15,  w: 6.5, cells: [5, 4], lead: 1, shape: "field",
    label: "Dense retrieval",    term: () => "embedding similarity" },
  { id: "sparse", layers: 1,   n: "03", step: 1, lift: -15, w: 6.5, cells: [1, 10], lead: -1, shape: "bars",
    label: "BM25",               term: () => "lexical match" },
  { id: "fused", layers: 2,    n: "04", step: 2, lift: 0,   w: 7.5, cells: [5, 4], lead: 1, shape: "merge",
    label: "Rank fusion",        term: () => "both rankings combined" },
  { id: "reranked", layers: 1, n: "05", step: 3, lift: 0,   w: 7.5, cells: [5, 4], lead: -1, shape: "sort",
    label: "Cross-encoder",      term: () => "scored as a pair" },
  // Eight rows, not five. Five passages get through, and the rows below them
  // are where the cap shows what it turned away; with exactly five there was
  // nowhere to draw a refusal and the gate had nothing to show.
  { id: "selected", layers: 1, n: "06", step: 4, lift: 0,   w: 6,   cells: [1, 8], lead: 1, shape: "gate",
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

/* A home cell for a passage, chosen from its id rather than its rank.

   Rank order put the passages that went on to the answer in the first row of
   every stage and left the rows below them empty, so every stage looked
   top-heavy and all of them looked alike. Hashing the id spreads a stage's
   passages over its whole matrix instead, and rank moves to how large and how
   strong a passage is drawn and to when it settles.

   Passages are placed in id order, not rank order, so the same set of passages
   always lands in the same cells however a stage ranks them. `reuse` lets the
   cross-encoder keep exactly the cells fusion gave its passages, so a
   reordering shows as a change of size in place rather than a shuffle.
   Collisions step through the matrix by a stride that shares no factor with its
   size, which visits every cell before repeating and does not pile collisions
   into a run of neighbours the way stepping by one does. */
const hashId = id => {
  const s = String(id);
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
};
const gcd = (x, y) => (y ? gcd(y, x % y) : x);

function spreadIndex(items, slots, reuse = null) {
  const out = new Map();
  const taken = new Set();
  if (reuse) {
    for (const it of items) {
      const k = reuse.get(it.chunk_id);
      if (k != null && k < slots && !taken.has(k)) {
        taken.add(k);
        out.set(it.chunk_id, k);
      }
    }
  }
  let stride = 7;
  while (gcd(stride, slots) !== 1) stride++;
  const rest = items.filter(it => !out.has(it.chunk_id))
    .sort((x, y) => hashId(x.chunk_id) - hashId(y.chunk_id));
  for (const it of rest) {
    if (out.has(it.chunk_id)) continue;      // a repeated id keeps its first cell
    let k = hashId(it.chunk_id) % slots;
    for (let n = 0; n < slots && taken.has(k); n++) k = (k + stride) % slots;
    taken.add(k);
    out.set(it.chunk_id, k);
  }
  return out;
}

/* A dense-retrieval passage's point. The field is a region rather than a rack,
   so each dot is moved to a fixed offset inside its cell and the field reads as
   a scatter. The offset is applied to the placement itself, not only to the
   drawing, so the line into a dot ends on the dot. */
function spreadHome(plate, idx, id) {
  const c = cellAt(plate, idx);
  if (plate.shape !== "field") return c;
  const h = hashId(id);
  const s = c.s + (((h & 0xffff) / 0xffff) - 0.5) * c.ds * 1.1;
  const t = c.t + ((((h >>> 16) & 0xffff) / 0xffff) - 0.5) * c.dt * 1.1;
  return { ...c, ...ptAt(plate, s, t, c.k), s, t };
}

/* A BM25 passage's bar. Its line arrives at the bar's left end, which is where
   the bar grows from, so the score is seen extending out of the point the
   passage was delivered to. */
function barHome(plate, row) {
  const c = cellAt(plate, row);
  const s = -plate.w * 0.86;
  return { ...c, ...ptAt(plate, s, c.t, c.k), s };
}

/* The plane the flow runs through and height, for the two stages drawn as
   vessels. A step along the flow is (+1, -1, 0) in u and v, so a move of `f`
   along it projects to 2 * f * SPREAD across the screen and nothing up or
   down, while height projects straight up. A shape drawn in this plane reaches
   the screen without the shear a plate's own plane gives it, which is why the
   funnel and the bowl are built here. */
const fz = (plate, f, z) => ({ u: plate.u + f, v: plate.v - f, z: plate.z + z });

// The funnel, in world units. Its mouth is about as tall as its taper is long,
// which is the proportion that makes the shape read as a funnel.
const FUN = { mouthF: -4.3, throatF: 7.8, spoutF: 11.4, mouthZ: 11.2, throatZ: 1.9 };

function funnelSeat(plate, i, n) {
  const k = n > 1 ? i / (n - 1) : 0.5;
  return fz(plate, FUN.mouthF + (FUN.spoutF - FUN.mouthF) * (0.34 + k * 0.62), 0);
}

// Where a refused passage's line meets the lower wall, and where it comes to
// rest after it falls clear.
function funnelWall(plate, j, m) {
  const k = m > 1 ? j / (m - 1) : 0.5;
  const f = FUN.mouthF + (FUN.throatF - FUN.mouthF) * (0.2 + k * 0.5);
  const along = (f - FUN.mouthF) / (FUN.throatF - FUN.mouthF);
  const z = -(FUN.mouthZ + (FUN.throatZ - FUN.mouthZ) * along) + 0.8;
  return { ...fz(plate, f, z), f, zRest: z - 6.5 };
}

// The bowl: a parabola in the same plane, its rim above the flow line and its
// bottom below it, so the lines from the cap dip into it.
const BOWL = { half: 5.0, rimZ: 2.0, bottomZ: -6.0 };
const bowlZ = f => BOWL.bottomZ + (BOWL.rimZ - BOWL.bottomZ) * (f / BOWL.half) ** 2;
const BOWL_AT = [0, -2.6, 2.6, -4.1, 4.1];

function bowlSeat(plate, i) {
  const f = BOWL_AT[i] ?? Math.max(-BOWL.half * 0.9,
    Math.min(BOWL.half * 0.9, (i % 2 ? 1 : -1) * 1.3 * i));
  return fz(plate, f, bowlZ(f) + 0.9);
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

/* A demonstration run for the hero, where no question has been asked.

   The hero drew with `run = null` and `progress = 1` until 2026-09-15, which
   made it a still photograph of an empty pipeline: seven plates, dashed
   connectors, nothing moving. The page's whole claim is that you can watch
   retrieval happen, and the first thing a visitor saw was the one view where
   nothing happens.

   What moves here is generic. The passages are positions in the real index and
   carry the real document colours, but no query chose them, so the run is
   marked `demo` and every stage keeps its idle caption rather than printing a
   count nobody measured. See the note beside `runSub`.

   `seed` varies the run between loops, so the hero is not the same eight
   seconds repeating exactly. It is a hash, not Math.random: the drawing has to
   be reproducible frame to frame within one loop, and the file forbids
   randomness at draw time. */
export function demoTrace(city, seed = 0) {
  if (!city || !city.marks?.length) return null;
  const rnd = n => ((Math.sin((seed + 1) * 9781.17 + n * 127.1) * 43758.5453) % 1 + 1) % 1;
  const srcs = city.sources?.length ? city.sources : ["a", "b", "c"];
  const uniq = xs => { const seen = new Set();
    return xs.filter(c => !seen.has(c.chunk_id) && seen.add(c.chunk_id)); };

  // Candidates drawn from across the whole index rather than one corner, so
  // the lines out of stage 01 fan the way a real search's do.
  const pick = (n, salt) => {
    const out = [];
    const seen = new Set();
    for (let i = 0; out.length < n && i < n * 6; i++) {
      const m = city.marks[Math.floor(rnd(salt + i) * city.marks.length)];
      if (!m || seen.has(m.id)) continue;
      seen.add(m.id);
      out.push({ chunk_id: m.id });
    }
    return out;
  };

  // Scores fall with rank the way a retriever's do, so sizes and bar lengths
  // have something to encode. They are shaped like scores and are not
  // measurements, which is one more reason this run prints no counts.
  const dense = pick(16, 10).map((c, i) => ({ ...c, rank: i + 1,
    score: Math.round((0.71 - i * 0.021) * 1000) / 1000 }));
  // BM25 finds some of the same passages and some of its own, which is the
  // reason the two are fused at all.
  const sparse = uniq([...dense.slice(0, 6), ...pick(10, 400)])
    .map((c, i) => ({ ...c, rank: i + 1,
      score: Math.max(0.6, Math.round((15.2 - i * (0.7 + rnd(900 + i) * 0.5)) * 100) / 100) }));
  const inDense = new Set(dense.map(c => c.chunk_id));
  const inSparse = new Set(sparse.map(c => c.chunk_id));
  const fusedSet = uniq([...dense.slice(0, 8), ...sparse.slice(6, 10)])
    .map((c, i) => ({ ...c, rank: i + 1,
      agreement: inDense.has(c.chunk_id) && inSparse.has(c.chunk_id) }));
  // The reranker moves things. `was` is where fusion had a passage and `rank`
  // is where reading the pair put it, and the gap is what stage 05 shows.
  const reranked = fusedSet
    .map((c, i) => ({ c, k: rnd(700 + i) }))
    .sort((x, y) => x.k - y.k)
    .map(({ c }, i) => {
      const was = fusedSet.findIndex(f => f.chunk_id === c.chunk_id) + 1;
      return { ...c, rank: i + 1, was, delta: was - (i + 1),
               score: Math.round((6.2 - i * 0.8) * 100) / 100 };
    });

  /* Which document each passage came from, decided once per passage so it has
     the same colour at every stage. The reranker's top three share a document,
     which is the situation the cap exists for, one thorough document filling
     every slot, and it means the funnel refuses at least one passage on every
     pass. With documents assigned at random, 9 of the first 12 passes refused
     nothing, so the stage whose point is that more goes in than comes out
     usually showed no difference. The rest spread over three more documents. */
  const off = Math.floor(rnd(77) * srcs.length);
  const doc = j => srcs[(off + j) % srcs.length];
  const sourceOf = new Map();
  reranked.forEach((c, i) => sourceOf.set(c.chunk_id,
    i < 3 ? doc(0) : doc(1 + Math.floor(rnd(1300 + i) * 3))));
  const withSource = xs => xs.map(c => ({ ...c,
    source: sourceOf.get(c.chunk_id) ?? doc(Math.floor(rnd(1700 + c.chunk_id) * 4)) }));

  // Two per document, the cap the real pipeline ships, applied in reranked
  // order the way src/diversify.py applies it.
  const rr = withSource(reranked);
  const perDoc = new Map();
  const selected = [];
  for (const c of rr) {
    const n = perDoc.get(c.source) || 0;
    if (n >= 2) continue;
    perDoc.set(c.source, n + 1);
    selected.push({ ...c, rank: selected.length + 1 });
    if (selected.length === 5) break;
  }

  return { demo: true, stages: [
    { name: "dense", items: withSource(dense) },
    { name: "sparse", items: withSource(sparse) },
    { name: "fused", items: withSource(fusedSet) },
    { name: "reranked", items: rr },
    { name: "selected", items: selected },
  ] };
}

/* ---------------------------------------------------------------- a run */
export function buildRun(trace, city, ink, variant = "flat") {
  if (!trace || !city) return null;
  const byName = new Map(trace.stages.map(s => [s.name, s]));
  const final = byName.get("selected")?.items || [];
  const colour = palette([...new Set(final.map(i => i.source))], ink);
  const survivors = new Set(final.map(i => i.chunk_id));

  const placed = new Map();
  // The cells fusion gave its passages, so the cross-encoder can keep them.
  const fusedCells = new Map();
  for (const plate of platesFor(variant)) {
    const st = byName.get(plate.id);
    if (!st) continue;
    // As many candidates as this stage's matrix has cells. Taking more would
    // silently wrap them onto occupied cells; taking a fixed number would leave
    // the wider stages looking half-empty when they were full.
    const [cols, rows] = plate.cells || [1, 1];
    const items = st.items.slice(0, cols * rows);
    // No `z: plate.z` here. The home functions already return the cell's own
    // height, and overwriting it sent every cable to the plate's centre plane
    // while the square it was flying to sat on its layer -- the line and the
    // lit cell disagreeing by exactly the layer offset.
    const n = items.length;
    let homes;
    if (plate.id === "selected") {
      homes = items.map((it, i) => funnelSeat(plate, i, n));
    } else if (plate.shape === "bars") {
      const rowOf = spreadIndex(items, rows);
      homes = items.map(it => barHome(plate, rowOf.get(it.chunk_id)));
    } else {
      const idx = spreadIndex(items, cols * rows,
                              plate.id === "reranked" ? fusedCells : null);
      if (plate.id === "fused") for (const [id, k] of idx) fusedCells.set(id, k);
      homes = items.map(it => spreadHome(plate, idx.get(it.chunk_id), it.chunk_id));
    }
    placed.set(plate.id, items.map((it, i) => ({
      ...it, ...homes[i],
      // Where the passage stands in its stage's ranking, 0 for the best. Size,
      // strength and settle order are drawn from this, not from position.
      frac: n > 1 ? clamp01((Number.isFinite(it.rank) ? it.rank - 1 : i) / (n - 1)) : 0,
      lives: survivors.has(it.chunk_id),
      colour: survivors.has(it.chunk_id) ? colour(it.source) : null,
    })));
  }
  const sel = placed.get("selected") || [];
  const ap = platesFor(variant).find(p => p.id === "answer");
  placed.set("answer", sel.map((m, i) => ({ ...m, ...bowlSeat(ap, i) })));

  /* Which mark on the index a candidate belongs to. ONE answer, used by both
     the lighting and the departure lines.

     These were two mappings. `lit` keyed a Map by the sampled mark, so several
     candidates rounding to the same mark overwrote each other and fewer
     squares lit than there were candidates; `litAt` keyed by chunk_id and then
     scattered each line off its mark by up to 1.4 world units so two nearby
     candidates would not overdraw. The result was lines leaving from points
     where nothing was lit, next to lit squares nothing left from -- which is
     exactly what it looked like.

     The scatter is gone. A mark carrying more than one candidate is drawn
     bigger and brighter instead, which is the honest reading: at 5,459
     passages sampled to 260 marks each mark stands for about twenty
     passages, so a collision means that region of the index really did
     supply more than one candidate. */
  const markOf = id => city.chunkAt.get(Math.round(id / city.step) * city.step);

  const lit = new Map();
  for (const name of ["dense", "sparse"]) {
    for (const it of byName.get(name)?.items || []) {
      const near = markOf(it.chunk_id);
      if (!near) continue;
      const hue = survivors.has(it.chunk_id) ? colour(it.source) : null;
      const prev = lit.get(near.id);
      // A surviving candidate's colour wins over a non-survivor's absence of
      // one, whichever order they arrive in.
      lit.set(near.id, { colour: prev?.colour || hue,
                         n: (prev?.n || 0) + 1,
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
  // One departure point per candidate, not one per sampled mark. `lit` is
  // keyed by mark and several candidates round to the same one, so building
  // this from `lit` kept the last candidate at each mark and dropped the rest.
  // A surviving passage then had no line out of the index unless it happened
  // to be the last one written to its mark.
  const corpus = platesFor(variant).find(p => p.id === "corpus");
  const litAt = new Map();
  for (const name of ["dense", "sparse"]) {
    for (const it of byName.get(name)?.items || []) {
      if (litAt.has(it.chunk_id)) continue;
      const m = markOf(it.chunk_id);
      if (!m) continue;
      /* The mark's centre IN WORLD SPACE.

         This is the whole of the "lines and lit squares do not match" bug, and
         removing the jitter on 2026-09-14 did not touch it. A mark's `u` and
         `v` are PLATE-LOCAL: the index's marks are drawn at
         `ptAt(plate, m.u, m.v, k)`, which maps them across the plate's two
         basis vectors into world space. This map stored the plate-local pair
         unconverted, and `inflight` then fed it straight to `T(u, v, z)` as if
         it were already world -- so every departure point was a plate-local
         number read as a world coordinate, landing somewhere with no relation
         to the square it was supposed to leave.

         `layerAt(corpus, 0)` is the front sheet, which is the only one that
         lights up, so the line starts on the face the reader sees rather than
         somewhere inside the stack. */
      const w = ptAt(corpus, m.u, m.v, layerAt(corpus, 0));
      litAt.set(it.chunk_id, { u: w.u, v: w.v, z: w.z });
    }
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

  // The passages the cap turned away. A passage counts as refused when the
  // cross-encoder ranked it above the last passage the cap kept and it was left
  // out anyway, so the cap passed over it to keep one ranked lower. Everything
  // ranked below that point was never reached, and drawing it as refused would
  // show a rejection that did not happen; until 2026-09-15 every candidate
  // outside the five was drawn as one. At most five are drawn, because more
  // than that meeting one wall is a red smear and the label states the rule.
  const kept = new Set((placed.get("selected") || []).map(m => m.chunk_id));
  const capPlate = platesFor(variant).find(p => p.id === "selected");
  const rerRank = new Map((placed.get("reranked") || [])
    .map((m, i) => [m.chunk_id, Number.isFinite(m.rank) ? m.rank : i + 1]));
  const lastKept = Math.max(0, ...[...kept].map(id => rerRank.get(id) || 0));
  const capRefused = (placed.get("reranked") || [])
    .filter(m => !kept.has(m.chunk_id) && (rerRank.get(m.chunk_id) || 0) < lastKept)
    .slice(0, 5);
  capRefused.forEach((m, j) => {
    push("selected", {
      a: m, b: funnelWall(capPlate, j, capRefused.length),
      lives: false, colour: null, stopped: true,
    });
  });

  const rer = placed.get("reranked") || [];
  const biggest = rer.reduce((best, m) =>
    (Math.abs(m.delta ?? 0) > Math.abs(best?.delta ?? 0) ? m : best), null);

  return {
    colour, placed, byTarget, lit, city,
    // Carried through from the trace so `annotate` knows not to print counts
    // for a run nobody asked for. See demoTrace above.
    demo: !!trace.demo,
    // The passages the cap passed over, drawn by capFunnel on the lower
    // wall where their lines end.
    capRefused,
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
  const alpha = a * (dim ? 0.35 : 1);

  // The index is a stack, because 5,459 passages is a volume and not a page.
  // Outlines only: filling every sheet with a lattice is what made this a
  // solid box that no treatment drawn on top of it could compete with.
  const layers = plate.layers || 1;
  for (let i = layers - 1; i >= 1; i--) {
    const q = plateCorners(plate, layerAt(plate, i)).map(c => T(c.u, c.v, c.z));
    ctx.globalAlpha = alpha * 0.4 * (1 - (i / layers) * 0.45);
    ctx.strokeStyle = ink.other;
    ctx.lineWidth = 0.7;
    ctx.beginPath();
    ctx.moveTo(q[0].x, q[0].y);
    for (let e = 1; e < 4; e++) ctx.lineTo(q[e].x, q[e].y);
    ctx.closePath();
    ctx.stroke();
  }

  // The face. Dense retrieval's is dashed, because it has no slots to hold
  // anything: it is a region, not a rack.
  const pts = plateCorners(plate).map(c => T(c.u, c.v, c.z));
  ctx.globalAlpha = alpha * (plate.shape === "field" ? 0.75 : 0.85);
  ctx.strokeStyle = ink.faint;
  ctx.lineWidth = 0.9;
  if (plate.shape === "field") ctx.setLineDash([2, 3]);
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let e = 1; e < 4; e++) ctx.lineTo(pts[e].x, pts[e].y);
  ctx.closePath();
  ctx.stroke();
  ctx.setLineDash([]);

  // The empty slots, for the stages that have slots at all. Kept faint: this
  // is the tray, not what is on it.
  if (plate.cells && !plate.shape) {
    const [cols, rows] = plate.cells;
    for (let i = 0; i < cols * rows; i++) {
      drawCell(ctx, T, plate, cellAt(plate, i), null, ink.other,
               alpha * 0.22, 0.6);
    }
  }
  if (plate.shape === "bars" || plate.shape === "gate") {
    const [, rows] = plate.cells;
    for (let r = 0; r < rows; r++) {
      const t = plate.h - ((plate.h * 2) / rows) * (r + 0.5);
      const A = ptAt(plate, -plate.w * 0.86, t, plate.front);
      const B = ptAt(plate, plate.w * 0.86, t, plate.front);
      const p1 = T(A.u, A.v, A.z), p2 = T(B.u, B.v, B.z);
      ctx.globalAlpha = alpha * 0.3;
      ctx.strokeStyle = ink.other;
      ctx.lineWidth = 0.8;
      ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;
}

/* What a stage holds, drawn as the thing that stage does. Every stage used to
   draw cells, which is why seven different operations looked like one
   operation repeated seven times.

   A passage's position no longer says its rank. Until 2026-09-15 every stage
   filled its matrix in rank order from the top left, so the passages that went
   on to the answer were always packed into the first row with empty rows
   beneath them, and every stage looked top-heavy in the same way. A passage's
   cell now comes from its id (see spreadIndex). Rank decides how large and how
   strong it is drawn, and when it settles.

   `ft` is the stage's own flight, 0 while its passages are still travelling and
   1 once they have landed. Contents settle over the tail of the flight, which is
   where `paced` in index.html slows the replay, so each stage does its work
   while there is time to watch it. Every settle window closes by ft = 1, so the
   finished frame, which is all a reader with reduced motion sees, shows every
   stage complete. */
function stageContents(ctx, T, plate, run, city, ink, a, variant, ft = 1) {
  const items = run?.placed.get(plate.id) || [];
  if (!items.length) return;
  const local = clamp01((ft - 0.55) / 0.45);
  const ease = t => 1 - Math.pow(1 - t, 3);
  const shade = m => (m.lives ? m.colour : ink.other);
  // Rank 1 is drawn at full size and strength, the last passage at 45% of the
  // size and half the strength.
  const size = frac => 1 - frac * 0.55;
  const strength = (m, frac) => (m.lives ? 1 : 0.55) * (1 - frac * 0.5);
  // Rank 1 settles first and the last passage half a window after it.
  const settle = frac => clamp01((local - frac * 0.5) / 0.5);
  const at = (s, t, k = plate.front) => {
    const w = ptAt(plate, s, t, k);
    return T(w.u, w.v, w.z);
  };
  // `inflight` lifts every point a line lands on by 0.5, so contents take the
  // same lift and a line ends on the thing it delivers.
  const home = m => T(m.u, m.v, m.z + 0.5);
  const ring = (p, r, colour, alpha, lw = 1) => {
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = colour;
    ctx.lineWidth = lw;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  };

  if (plate.shape === "field") {
    /* Dense retrieval. The question becomes a vector in the same space as the
       passages, drawn as a ring widening from the centre, and the passages
       nearest it are found in rank order, each flashing as it is found. The
       dots do not sit at the passages' real coordinates, which run to 384
       dimensions for the embedder this project uses; the order they are found
       in is real. */
    const c = at(0, 0);
    const corner = at(plate.w, plate.h);
    const reach = Math.hypot(corner.x - c.x, corner.y - c.y);
    if (local > 0 && local < 1) {
      ring(c, reach * (0.08 + 0.95 * ease(local)), ink.faint, a * 0.45 * (1 - local));
    }
    for (const m of items) {
      const t = settle(m.frac);
      if (t <= 0) continue;
      const p = home(m);
      const r = (m.lives ? 3.8 : 2.6) * size(m.frac);
      if (t < 1) ring(p, r + 7 * t, shade(m), a * 0.5 * (1 - t));
      ctx.save();
      ctx.globalAlpha = a * strength(m, m.frac) * t;
      ctx.fillStyle = shade(m);
      ctx.beginPath();
      ctx.arc(p.x, p.y, r * (0.4 + 0.6 * ease(t)), 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
    return;
  }

  if (plate.shape === "bars") {
    /* BM25. Each passage is a bar whose length is its term-match score against
       the best score in the list, and the bars grow to those lengths in rank
       order. A trace that carries no scores falls back to rank. The gap left
       between bars is what stops ten of them filling in as one block, which is
       what this stage rendered as when every row was drawn full height. */
    const [, rows] = plate.cells;
    const scored = items.map(m => (Number.isFinite(m.score) && m.score > 0 ? m.score : null));
    const best = Math.max(0, ...scored.filter(x => x != null));
    const half = (plate.h / rows) * 0.62;
    const span = plate.w * 0.86 * 2;
    items.forEach((m, i) => {
      const t = settle(m.frac);
      if (t <= 0) return;
      const share = best > 0 && scored[i] != null ? scored[i] / best : size(m.frac);
      const len = span * share * ease(t);
      const q = [at(m.s, m.t - half, m.k), at(m.s + len, m.t - half, m.k),
                 at(m.s + len, m.t + half, m.k), at(m.s, m.t + half, m.k)];
      ctx.save();
      ctx.globalAlpha = a * strength(m, m.frac);
      ctx.fillStyle = shade(m);
      ctx.beginPath();
      ctx.moveTo(q[0].x, q[0].y);
      for (let j = 1; j < 4; j++) ctx.lineTo(q[j].x, q[j].y);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    });
    return;
  }

  if (plate.shape === "merge" || plate.shape === "sort") {
    const sort = plate.shape === "sort";
    const n = items.length;
    /* The cross-encoder reads the question and each passage together, one pair
       at a time, which is what makes it slower than comparing two vectors and
       better at it. It is drawn as a scan crossing the plate. Each passage keeps
       the cell fusion gave it, and as the scan reaches it the passage changes
       size from the rank fusion put it at to the rank the cross-encoder gives
       it, so a passage that grows was promoted and one that shrinks was pushed
       down. */
    const scan = sort ? clamp01((local - 0.12) / 0.88) : 1;
    const scanS = -plate.w + plate.w * 2 * scan;
    if (sort && scan > 0 && scan < 1) {
      const A = at(scanS, plate.h), B = at(scanS, -plate.h);
      ctx.save();
      ctx.globalAlpha = a * 0.6;
      ctx.strokeStyle = ink.faint;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(A.x, A.y);
      ctx.lineTo(B.x, B.y);
      ctx.stroke();
      ctx.restore();
    }
    for (const m of items) {
      let frac = m.frac, shown, reached = 1;
      if (sort) {
        shown = clamp01(local / 0.12);
        if (shown <= 0) continue;
        reached = scan >= 1 ? 1 : clamp01((scanS - (m.s - m.ds)) / (2 * m.ds));
        const was = Number.isFinite(m.was) ? m.was : (m.rank ?? 1);
        const before = n > 1 ? clamp01((was - 1) / (n - 1)) : 0;
        frac = before + (m.frac - before) * ease(reached);
      } else {
        shown = settle(m.frac);
        if (shown <= 0) continue;
      }
      const sz = size(frac) * (sort ? 1 : 0.4 + 0.6 * ease(shown));
      drawCell(ctx, T, plate, { ...m, ds: m.ds * sz, dt: m.dt * sz }, shade(m), null,
               a * strength(m, frac) * shown);
      // Fusion. A passage both retrievers found rings as it lands, because
      // reciprocal rank fusion adds a share for every list a passage appears
      // in, so appearing in both is what lifts it.
      if (!sort && m.agreement && shown < 1) {
        ring(home(m), 4 + 10 * shown, m.lives ? m.colour : ink.faint, a * 0.7 * (1 - shown), 1.1);
      }
      // Cross-encoder. A passage it moves up flashes as the scan passes it.
      if (sort && reached > 0 && reached < 1 && Number.isFinite(m.was) && m.rank < m.was) {
        ring(home(m), 5 + 6 * reached, m.lives ? m.colour : ink.faint, a * 0.8 * (1 - reached), 1.2);
      }
    }
    return;
  }

  // A stage without a treatment of its own keeps the plain rack.
  for (const m of items) {
    const t = settle(m.frac);
    if (t <= 0) continue;
    drawCell(ctx, T, plate, m, shade(m), null, a * strength(m, m.frac) * t);
  }
}

/* The diversity cap, as a funnel.

   It was an eight-row rack: five filled cells for the passages that got through
   and three red outlines for the ones that did not, which is a list with rows
   crossed out rather than a funnel, and the one stage whose job is visibly
   different looked the same as the stages around it.

   Everything here is placed in WORLD space, in the plane the flow runs through
   and height (see `fz`). Two earlier versions drew the vessel in screen space
   around the plate's centre. It looked right, but `inflight` draws the lines
   into and out of this stage between world points, so passages travelled along
   lines to one place and were drawn arriving in another, the same disagreement
   between a line and the thing it delivers that the index had. With seats in
   world space, a line into the funnel ends on a passage in it and a line out of
   it leaves from one. */
function capFunnel(ctx, T, plate, run, ink, a, ft, variant = "flat") {
  const P = (f, z) => { const w = fz(plate, f, z); return T(w.u, w.v, w.z); };
  const items = run?.placed.get("selected") || [];
  const refused = run?.capRefused || [];
  const local = clamp01((ft - 0.45) / 0.55);
  const ease = t => 1 - Math.pow(1 - t, 3);

  ctx.save();
  ctx.lineJoin = "round"; ctx.lineCap = "round";
  ctx.strokeStyle = ink.other;
  ctx.globalAlpha = a * 0.85;
  ctx.lineWidth = 1.3;
  for (const side of [1, -1]) {
    const m = P(FUN.mouthF, side * FUN.mouthZ);
    const t = P(FUN.throatF, side * FUN.throatZ);
    const s = P(FUN.spoutF, side * FUN.throatZ);
    ctx.beginPath(); ctx.moveTo(m.x, m.y); ctx.lineTo(t.x, t.y); ctx.lineTo(s.x, s.y); ctx.stroke();
  }
  ctx.restore();

  // What got through, at the seats their lines end on.
  items.forEach((m, i) => {
    const t0 = items.length > 1 ? (i / (items.length - 1)) * 0.5 : 0;
    const t = clamp01((local - t0) / 0.5);
    if (t <= 0) return;
    const p = T(m.u, m.v, m.z + 0.5);
    ctx.save();
    ctx.fillStyle = m.colour || ink.faint;
    ctx.globalAlpha = a * 0.22 * t;
    ctx.beginPath(); ctx.arc(p.x, p.y, 6.2, 0, Math.PI * 2); ctx.fill();
    ctx.globalAlpha = a * t;
    ctx.beginPath(); ctx.arc(p.x, p.y, 3.1 * (0.5 + 0.5 * ease(t)), 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  });

  /* A refused passage meets the lower wall where its line ends, slides off and
     falls clear, and stays where it falls, dimmed, so the finished frame still
     shows that more went in than came out. Each window closes by ft = 1, so a
     reader with reduced motion sees every refusal already at rest. */
  refused.forEach((m, j) => {
    const k = refused.length > 1 ? j / (refused.length - 1) : 0;
    const t = clamp01((local - (0.4 + 0.2 * k)) / 0.4);
    if (t <= 0) return;
    const hit = funnelWall(plate, j, refused.length);
    const rest = fz(plate, hit.f + 0.9, hit.zRest);
    const p0 = T(hit.u, hit.v, hit.z + 0.5);
    const p1 = T(rest.u, rest.v, rest.z + 0.5);
    const fall = ease(clamp01((t - 0.35) / 0.65));
    ctx.save();
    ctx.globalAlpha = a * 0.9 * Math.min(1, t * 2) * (1 - fall * 0.45);
    ctx.strokeStyle = "#f0685f";
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.arc(p0.x + (p1.x - p0.x) * fall, p0.y + (p1.y - p0.y) * fall, 3.0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  });
}

/* The answer, as a bowl the passages come to rest in.

   Placed in world space for the same reason as the funnel: the lines from the
   cap end on the seats `bowlSeat` gives the passages, so a passage is drawn
   where its line delivers it. The seats sit below the flow line, so those lines
   dip into the bowl. Rank 1, the passage the answer quotes, sits at the bottom
   and is drawn largest, and the rest sit up the sides. */
function answerBasin(ctx, T, plate, run, ink, a, ft) {
  const items = run?.placed.get("answer") || [];
  const local = clamp01((ft - 0.45) / 0.55);
  const ease = t => 1 - Math.pow(1 - t, 3);

  ctx.save();
  ctx.strokeStyle = ink.other;
  ctx.globalAlpha = a * 0.8;
  ctx.lineWidth = 1.3;
  ctx.lineCap = "round";
  ctx.beginPath();
  for (let i = 0; i <= 36; i++) {
    const f = -BOWL.half + (2 * BOWL.half * i) / 36;
    const w = fz(plate, f, bowlZ(f));
    const p = T(w.u, w.v, w.z);
    if (i) ctx.lineTo(p.x, p.y); else ctx.moveTo(p.x, p.y);
  }
  ctx.stroke();
  ctx.restore();

  items.forEach((m, i) => {
    const t0 = items.length > 1 ? (i / (items.length - 1)) * 0.5 : 0;
    const t = clamp01((local - t0) / 0.5);
    if (t <= 0) return;
    const p = T(m.u, m.v, m.z + 0.5);
    const bounce = t > 0.8 ? Math.sin(((t - 0.8) / 0.2) * Math.PI) * 2.5 : 0;
    const cited = i === 0;
    ctx.save();
    ctx.fillStyle = m.colour || ink.faint;
    ctx.globalAlpha = a * 0.22 * t;
    ctx.beginPath(); ctx.arc(p.x, p.y - bounce, cited ? 11 : 8, 0, Math.PI * 2); ctx.fill();
    ctx.globalAlpha = a * t;
    ctx.beginPath();
    ctx.arc(p.x, p.y - bounce, (cited ? 5.6 : 3.8) * (0.5 + 0.5 * ease(t)), 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  });
}

/* Lifted plates get a dotted drop line and a ghost of their own footprint on
   the base plane. Without it the height is ambiguous -- dense and BM25 look
   like they are further along the flow rather than above and below it. */
function dropline(ctx, T, plate, ink, a, f) {
  if (!plate.z || f < 0.6) return;
  const k = a * clamp01((f - 0.6) / 0.4) * 0.26;
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
  // A surviving passage is drawn onto the cell it lands in. One that does not
  // survive stops short: twenty of them arriving at one small plate hatch it
  // into a solid block, which is what BM25 kept rendering as. They still carry
  // the volume of the leg they are on, which is the reason they are drawn.
  // A refused passage travels all the way to the funnel's lower wall, which
  // is what stops it, and capFunnel draws it sliding off and falling. A
  // passage that simply did not survive still stops short of its stage.
  const wall = link.stopped ? 1 : link.lives ? 1 : 0.66;
  const held = Math.min(e, wall);
  const bx = lerp(p1.x, p2.x, held), by = lerp(p1.y, p2.y, held);

  ctx.save();
  // 0.18 made forty lines out of the index add up to nothing, so the widest
  // leg of the pipeline read as the emptiest. The funnel is the point of the
  // drawing: forty leave the index, twenty cross fusion and the reranker, five
  // reach the answer, and that has to be visible without reading a label.
  ctx.globalAlpha = a * (link.lives ? 0.85 : link.stopped ? 0.42 : 0.22);
  ctx.strokeStyle = link.lives ? link.colour : link.stopped ? STOPPED : ink.other;
  ctx.lineWidth = link.lives ? 1.2 : 0.75;
  if (!link.lives) ctx.setLineDash([1.5, 3]);
  ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(bx, by); ctx.stroke();
  ctx.restore();

  // A refused passage fades out on its approach. The refusal itself is drawn
  // on the cap's face, where a reader can see which slots were shut; a second
  // mark hanging in mid-air stated the same thing in a place with no gate.
  if (link.stopped) return;

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
    /* A demonstration run animates the mechanism and states no counts.
       `run.demo` is set by the hero, whose passages are generic rather than the
       result of a query. The motion is a drawing; a caption reading
       "16 CANDIDATES" under it would be a measurement, and this project does
       not print a number it did not measure. The demo therefore keeps the idle
       terms -- EMBEDDING SIMILARITY, LEXICAL MATCH -- and the index keeps its
       real 5,459 passages, which is a fact about the corpus and not about any
       query. */
    { text: String(dim ? "not run"
                   : (run && !run.demo ? runSub(plate.id, run) : plate.term(city))).toUpperCase(),
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
    // The two vessels reach past their plates, the bowl past the answer's
    // right edge in particular, and the fit has to leave room for them or
    // the last stage is cut by the frame.
    if (p.id === "answer") {
      for (const f of [-BOWL.half, BOWL.half]) {
        const w = fz(p, f, BOWL.rimZ); see(project(w.u, w.v, w.z));
      }
      const w = fz(p, 0, BOWL.bottomZ); see(project(w.u, w.v, w.z));
    }
    if (p.id === "selected") {
      for (const z of [-FUN.mouthZ, FUN.mouthZ]) {
        const w = fz(p, FUN.mouthF, z); see(project(w.u, w.v, w.z));
      }
      const w = fz(p, FUN.spoutF, 0); see(project(w.u, w.v, w.z));
    }
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

    /* Dense and BM25 share a step, so their legs left the index in the same
       instant and forty lines fanning out at once read as one search with a
       wide spray rather than two searches running side by side. BM25's wave
       is held back by a tenth of the leg so the two arrive as two.

       This is a drawing device, not a claim about timing: the two retrievers
       run concurrently, and the stage labels and the two plate heights are
       what say so. The lag is small enough to separate the waves and too
       small to read as "BM25 runs after dense". */
    const LAG = { sparse: 0.12 };
    const lag = LAG[plate.id] || 0;
    const ftl = lag ? clamp01((ft - lag) / (1 - lag)) : ft;
    for (const l of run?.byTarget.get(plate.id) || []) {
      inflight(ctx, T, l, ink, 1, ftl);
    }

    dropline(ctx, T, plate, ink, 1, f);
    // Two stages draw their own vessel instead of a rack. The answer is a
    // basin and the cap is a funnel, and both read as neither when a square
    // grid is drawn behind them.
    if (plate.id !== "answer" && plate.id !== "selected") {
      plane(ctx, T, plate, ink, 1, dim, f);
    }

    // Marks settle after the plate has formed under them.
    const ma = clamp01((f - 0.6) / 0.4);
    if (ma > 0) {
      if (plate.id === "corpus") {
        // Every sheet of the index carries the field, not just the front one.
        // 5,459 passages is the one quantity on this drawing that is genuinely
        // large, and a stack of dense sheets is what that looks like. Only the
        // front sheet lights up: a passage is one passage, and repeating its
        // colour six times would claim six.
        const sheets = plate.layers || 1;
        const paintSheets = (c, alpha) => {
          for (let i = sheets - 1; i >= 0; i--) {
            const k = layerAt(plate, i);
            const depth = i === 0 ? 1 : 1 - (i / sheets) * 0.25;
            for (const m of city.marks) {
              const w = ptAt(plate, m.u, m.v, k);
              const p = T(w.u, w.v, w.z);
              const hit = i === 0 ? run?.lit.get(m.id) : null;
              // Size carries how many candidates came from this region, now
              // that they are no longer scattered apart to show it. One
              // candidate is 2.2; it grows with the count and stops at 3.4,
              // past which a mark starts reading as a plate of its own.
              if (hit) mark(c, p, Math.min(3.4, 2.2 + (hit.n - 1) * 0.5),
                            hit.colour || ink.faint, null, alpha);
              else mark(c, p, 1.1, null, ink.other, alpha * 0.62 * depth);
            }
          }
        };
        if (ma < 1) {
          paintSheets(ctx, ma);
        } else {
          const key = [variant, idOf(city), idOf(run), ink.other, ink.faint]
            .join("|");
          ctx.drawImage(
            sheetLayer(W, H, sheetDpr(ctx), key, (c) => paintSheets(c, 1)),
            0, 0, W, H);
        }
      } else if (plate.id === "answer") {
        answerBasin(ctx, T, plate, run, ink, ma, ft);
      } else if (plate.id === "selected") {
        capFunnel(ctx, T, plate, run, ink, ma, ft, variant);
      } else {
        stageContents(ctx, T, plate, run, city, ink, ma, variant, ft);
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
      text: `${run.city.docs} documents, split into `
          + `${run.city.total.toLocaleString()} overlapping passages. The `
          + `overlap keeps a sentence that crosses a boundary intact in one of `
          + `them. Each passage was converted to a vector before any question `
          + `was asked. Every stage after this one reduces the set.` },

    { title: "Dense retrieval",
      text: `The question is converted to a vector by the same model that `
          + `converted the passages. The ${c.dense} nearest by cosine `
          + `similarity are kept. This retrieves passages with the same meaning `
          + `as the question even when they share none of its words.` },

    { title: "BM25",
      text: c.sparse
        ? `A second search over the same index, scoring word overlap and `
          + `weighting rare words above common ones. ${c.sparse} candidates. It `
          + `runs at the same time as dense retrieval, not after it, and it `
          + `retrieves exact terms that a vector does not preserve, such as a `
          + `species name or a symbol.`
        : `No word in the question appears in the index, so lexical search returned `
          + `nothing and the result rests on dense retrieval alone.` },

    { title: "Rank fusion",
      text: `The two lists are merged on position rather than score. A cosine `
          + `similarity and a BM25 weight are different units and cannot be `
          + `compared directly. A passage ranked highly by both searches ends up `
          + `above one ranked highly by a single search. ${c.both} passages `
          + `appeared in both lists.` },

    { title: "Cross-encoder",
      text: `The earlier stages scored the question and each passage `
          + `separately, then compared the two scores. This model reads the pair `
          + `together. It is slower, and more accurate at separating a passage `
          + `that mentions the subject from one that answers the question. `
          + (big
              ? `Largest change: ${shortDoc(big.source)} moved `
                + `${big.delta > 0 ? "up" : "down"} ${Math.abs(big.delta)} places.`
              : `The order was almost unchanged on this question.`) },

    { title: "Diversity cap",
      text: `A limit of two passages per document. Without it, one long `
          + `document can occupy every slot and a second document that also `
          + `answers is never returned. ${c.selected} passages kept.` },

    { title: run.confident ? "Answer" : "Below threshold",
      text: run.confident
        ? `The answer is drawn from these ${c.selected} passages and cites `
          + `them. No other text was used.`
        : `The best passage scored below this corpus's threshold, the point `
          + `under which passages from these documents usually do not contain `
          + `the answer. No answer is returned. The candidates remain listed `
          + `below.` },
  ];
}
