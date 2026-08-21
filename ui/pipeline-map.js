/* The pipeline as exploded isometric plates.
 *
 * This replaces a version that drew the stages as buildings on a ground plane.
 * The metaphor was the problem: a "city" of towers is a generic isometric
 * illustration, it says nothing true about retrieval, and the floors-as-ranks
 * idea was only legible once someone explained it.
 *
 * Plates say the true thing instead. Each stage is a sheet; every candidate
 * keeps a position on that sheet; and a passage's line between sheets shows
 * where it moved. Watching a line climb from the back of one plate to the front
 * of the next IS the reranker earning its keep -- which was the whole point of
 * the bump chart this project started with, restored, but with the one thing
 * the bump chart got wrong fixed:
 *
 *   DENSE AND BM25 SIT AT THE SAME ALTITUDE, side by side. They run at the same
 *   moment. Drawing them as column 1 and column 2 taught a sequence that does
 *   not exist, and stacking them here would repeat it.
 *
 * The corpus is the ground plate and is always drawn in full, before any
 * question -- structure exists before the query, and a diagram that grows as
 * results arrive implies the opposite.
 */

export const shortDoc = s => String(s).replace(/\.(pdf|docx|pptx|xlsx)$/i, "");

/* ---------------------------------------------------------- projection */
const KX = Math.cos(Math.PI / 6);     // 0.866
const KY = Math.sin(Math.PI / 6);     // 0.5

const project = (u, v, z) => ({ x: (u - v) * KX, y: (u + v) * KY - z });

/* ------------------------------------------------------------- the plan
 * The pipeline reads LEFT TO RIGHT, the way a process is read. Each stage steps
 * along the flow axis (u up, v down by the same amount), which in isometric
 * moves right on screen without changing height.
 *
 * "By meaning" and "By keyword" sit at the SAME step and are separated by
 * height instead. They run at the same moment; putting one after the other
 * would teach a sequence that does not exist.
 *
 * Plate width shrinks along the flow, so the narrowing is something you can
 * see rather than something a caption claims.
 *
 * Stages are numbered because they genuinely are a sequence -- 01 happens
 * before 02 -- not because numbering looks technical.
 */
const FLOW = 20;
const at = (step, lift) => ({ u: step * FLOW, v: -step * FLOW, z: lift });

const PLATES = [
  { id: "corpus", n: "01", ...at(0, 0), w: 14, grid: 9, cap: "above",
    label: "Your documents",
    term: c => c.total.toLocaleString() + " passages, " + c.docs + " files" },
  { id: "dense", n: "02", ...at(1, 16), w: 7, grid: 4, cap: "above",
    label: "Search by meaning", term: () => "finds similar wording" },
  { id: "sparse", n: "02", ...at(1, -16), w: 7, grid: 4, cap: "below",
    label: "Search by keyword", term: () => "finds your exact words" },
  { id: "fused", n: "03", ...at(2, 0), w: 8, grid: 4, cap: "above",
    label: "Both lists merged", term: () => "ranked together" },
  { id: "reranked", n: "04", ...at(3, 0), w: 8, grid: 4, cap: "above",
    label: "Read again, properly", term: () => "beside your question" },
  { id: "selected", n: "05", ...at(4, 0), w: 6, grid: 3, cap: "above",
    label: "Trimmed", term: () => "max 2 from one file" },
  { id: "answer", n: "06", ...at(5, 0), w: 5, grid: 2, cap: "above",
    label: "Your answer", term: () => "with its sources" },
];

const THICK = 1.6;      // plate edge, so a sheet reads as a solid object
const SLOTS = 5;        // items per row on a stage plate

function palette(sources, ink) {
  const slots = [ink.s1, ink.s2, ink.s3];
  const m = new Map();
  sources.forEach((s, i) => m.set(s, i < 3 ? slots[i] : ink.other));
  return src => m.get(src) || ink.other;
}

/* Where item n sits on a plate: a row-major grid in plate-local u/v, rank 1
   nearest the viewer so "climbing" reads as moving toward the front. */
function slot(plate, i, n) {
  const cols = Math.min(SLOTS, Math.max(1, n));
  const rows = Math.max(1, Math.ceil(n / cols));
  const cx = i % cols, cy = Math.floor(i / cols);
  const stepU = (plate.w * 1.5) / (cols + 1);
  const stepV = (plate.w * 1.5) / (rows + 1);
  return {
    u: plate.u - plate.w * 0.75 + stepU * (cx + 1),
    v: plate.v - plate.w * 0.75 + stepV * (cy + 1),
  };
}

/* ------------------------------------------------------------ the corpus */
export function buildCity(chunks) {
  const sources = chunks.sources || [];
  const doc = chunks.doc || [];
  const total = chunks.n || doc.length;

  // One mark per chunk is 2,768 marks on a plate a few hundred pixels wide --
  // an unreadable smear. Sampling to a fixed budget keeps the density legible
  // while the count beside it stays exact.
  const BUDGET = 400;
  const step = Math.max(1, Math.floor(total / BUDGET));
  const marks = [];
  const ids = [];
  for (let i = 0; i < total; i += step) ids.push(i);

  // A jittered lattice, not a random scatter. Sine-hash scatter correlated and
  // clumped in the middle, which read as noise; an index is an ordered thing
  // and should look like one. The jitter is deterministic, so the drawing is
  // identical between renders.
  const side = Math.ceil(Math.sqrt(ids.length));
  // The plate spans -30..30 in u and v, so a 25-unit lattice packed every mark
  // into the middle third and read as a solid slab rather than as passages.
  const span = 47;
  ids.forEach((id, n) => {
    const gx = n % side, gy = Math.floor(n / side);
    const j = k => ((Math.sin(k * 127.1 + n * 311.7) * 43758.5453) % 1 + 1) % 1 - 0.5;
    marks.push({
      id, doc: doc[id] ?? 0,
      u: -span / 2 + (span * (gx + 0.5)) / side + j(1) * (span / side) * 0.55,
      v: -span / 2 + (span * (gy + 0.5)) / side + j(2) * (span / side) * 0.55,
    });
  });
  const at = new Map(marks.map(m => [m.id, m]));
  return { total, docs: sources.length, sources, marks, chunkAt: at, step };
}

/* --------------------------------------------------------------- a run */
export function buildRun(trace, city, ink) {
  if (!trace || !city) return null;
  const byName = new Map(trace.stages.map(s => [s.name, s]));
  const final = byName.get("selected")?.items || [];
  const order = [...new Set(final.map(i => i.source))];
  const colour = palette(order, ink);
  const survivors = new Set(final.map(i => i.chunk_id));

  // Positions per plate, and the identity needed to join them up.
  const placed = new Map();
  for (const plate of PLATES) {
    const st = byName.get(plate.id);
    if (!st) continue;
    const items = st.items.slice(0, SLOTS * 4);
    placed.set(plate.id, items.map((it, i) => ({
      ...it,
      ...slot(plate, i, items.length),
      z: plate.z,
      lives: survivors.has(it.chunk_id),
      colour: survivors.has(it.chunk_id) ? colour(it.source) : ink.other,
    })));
  }

  // The answer plate has no stage of its own: it is the selection, arrived.
  const sel = placed.get("selected") || [];
  const answerPlate = PLATES.find(p => p.id === "answer");
  placed.set("answer", sel.map((m, i) => ({
    ...m, ...slot(answerPlate, i, sel.length), z: answerPlate.z,
  })));

  // Which corpus marks lit, and for which retriever.
  const lit = new Map();
  for (const [name, kind] of [["dense", "dense"], ["sparse", "sparse"]]) {
    for (const it of byName.get(name)?.items || []) {
      const near = city.chunkAt.get(Math.round(it.chunk_id / city.step) * city.step);
      if (!near) continue;
      const prev = lit.get(near.id);
      if (prev) { prev.both = true; continue; }
      lit.set(near.id, { mark: near, kind, both: false,
                         colour: survivors.has(it.chunk_id) ? colour(it.source) : null });
    }
  }

  // Edges between consecutive plates, by chunk identity.
  const links = [];
  const chain = [["corpus", "dense"], ["corpus", "sparse"], ["dense", "fused"],
                 ["sparse", "fused"], ["fused", "reranked"],
                 ["reranked", "selected"], ["selected", "answer"]];
  for (const [a, b] of chain) {
    const from = placed.get(a) || [];
    const to = placed.get(b) || [];
    if (a === "corpus") continue;
    const idx = new Map(from.map(m => [m.chunk_id, m]));
    for (const m of to) {
      const src = idx.get(m.chunk_id);
      if (src) links.push({ a: src, b: m, lives: m.lives, colour: m.colour });
    }
  }

  const rer = placed.get("reranked") || [];
  const biggest = rer.reduce((best, m) =>
    (Math.abs(m.delta ?? 0) > Math.abs(best?.delta ?? 0) ? m : best), null);

  return {
    colour, order, placed, links, lit, city,
    verdict: trace.verdict, query: trace.query,
    confident: trace.verdict?.confident !== false,
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

/* ------------------------------------------------------------- drawing */
function rhombus(w, d) {
  return [{ u: -w, v: -d }, { u: w, v: -d }, { u: w, v: d }, { u: -w, v: d }];
}

function drawPlate(ctx, T, plate, ink, alpha, dimmed) {
  const w = plate.w, d = plate.w;
  const top = rhombus(w, d).map(p => T(plate.u + p.u, plate.v + p.v, plate.z));

  ctx.save();
  ctx.globalAlpha = alpha;

  // the edge, so a sheet reads as an object with thickness
  const under = rhombus(w, d).map(p => T(plate.u + p.u, plate.v + p.v, plate.z - THICK));
  ctx.beginPath();
  ctx.moveTo(top[1].x, top[1].y); ctx.lineTo(top[2].x, top[2].y);
  ctx.lineTo(under[2].x, under[2].y); ctx.lineTo(under[1].x, under[1].y);
  ctx.closePath();
  ctx.fillStyle = ink.panel; ctx.fill();
  ctx.strokeStyle = dimmed ? ink.other : ink.line; ctx.lineWidth = 1; ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(top[2].x, top[2].y); ctx.lineTo(top[3].x, top[3].y);
  ctx.lineTo(under[3].x, under[3].y); ctx.lineTo(under[2].x, under[2].y);
  ctx.closePath();
  ctx.fillStyle = ink.panel; ctx.fill(); ctx.stroke();

  // the face
  ctx.beginPath();
  ctx.moveTo(top[0].x, top[0].y);
  for (let i = 1; i < 4; i++) ctx.lineTo(top[i].x, top[i].y);
  ctx.closePath();
  ctx.fillStyle = ink.panel;
  ctx.globalAlpha = alpha * 0.82; ctx.fill();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = dimmed ? ink.other : ink.faint;
  ctx.lineWidth = 1.1; ctx.stroke();

  // the grid, which is what makes it read as drawn rather than rendered
  ctx.globalAlpha = alpha * 0.4;
  ctx.strokeStyle = ink.other; ctx.lineWidth = 0.6;
  for (let i = 1; i < plate.grid; i++) {
    const t = -w + (2 * w * i) / plate.grid;
    const a1 = T(plate.u + t, plate.v - d, plate.z);
    const a2 = T(plate.u + t, plate.v + d, plate.z);
    const b1 = T(plate.u - w, plate.v + t, plate.z);
    const b2 = T(plate.u + w, plate.v + t, plate.z);
    ctx.beginPath(); ctx.moveTo(a1.x, a1.y); ctx.lineTo(a2.x, a2.y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(b1.x, b1.y); ctx.lineTo(b2.x, b2.y); ctx.stroke();
  }

  // registration ticks at the corners -- the drafting mark that says this is a
  // measured object, borrowed from the reference's vocabulary
  ctx.globalAlpha = alpha * 0.9;
  ctx.strokeStyle = ink.faint; ctx.lineWidth = 1.2;
  const TICK = 5;
  for (const c of top) {
    ctx.beginPath();
    ctx.moveTo(c.x - TICK, c.y); ctx.lineTo(c.x + TICK, c.y);
    ctx.moveTo(c.x, c.y - TICK); ctx.lineTo(c.x, c.y + TICK);
    ctx.stroke();
  }
  ctx.restore();
  return top;
}

function label(ctx, T, plate, city, run, ink, alpha, dim) {
  const above = plate.cap !== "below";
  // A plate's topmost point sits w above its centre, so clearance has to scale
  // with the plate, not be a fixed nudge -- at w=14 a 0.55 factor put the label
  // straight through the corpus.
  const p = T(plate.u, plate.v,
              plate.z + (above ? plate.w + 11 : -plate.w - 19));
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  ctx.fillStyle = ink.other;
  ctx.font = "500 9px 'DM Mono', ui-monospace, monospace";
  ctx.fillText(plate.n, p.x, p.y - 18);

  ctx.fillStyle = dim ? ink.faint : ink.ink;
  ctx.font = "600 12.5px Inter, system-ui, sans-serif";
  ctx.fillText(plate.label, p.x, p.y - 4);

  ctx.fillStyle = ink.muted;
  ctx.font = "500 9.5px 'DM Mono', ui-monospace, monospace";
  const sub = dim ? "not used" : (run ? runSub(plate.id, run) : plate.term(city));
  ctx.fillText(String(sub).toUpperCase(), p.x, p.y + 10);
  ctx.restore();
}

function diamond(ctx, p, r, fill, stroke, alpha) {
  ctx.save(); ctx.globalAlpha = alpha;
  ctx.beginPath();
  ctx.moveTo(p.x, p.y - r); ctx.lineTo(p.x + r * KX * 2, p.y);
  ctx.lineTo(p.x, p.y + r); ctx.lineTo(p.x - r * KX * 2, p.y);
  ctx.closePath();
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.stroke(); }
  ctx.restore();
}

export function drawScene(ctx, city, run, ink, W, H, progress = 1) {
  ctx.clearRect(0, 0, W, H);
  if (!city) return;

  // Fit: project everything once, measure, then scale. Hard-coding a scale
  // made the drawing overflow at some widths and float in a void at others.
  const probe = [];
  for (const p of PLATES) {
    for (const c of rhombus(p.w, p.w)) probe.push(project(p.u + c.u, p.v + c.v, p.z));
  }
  for (const m of city.marks) probe.push(project(m.u, m.v, 0));
  const xs = probe.map(p => p.x), ys = probe.map(p => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);

  // Labels sit to the right of most plates, so the room they need is reserved
  // on that side only. Padding both sides shrank the drawing twice and left a
  // dead column on the left.
  // Labels sit above and below the plates now, so the room they need is
  // vertical and the horizontal padding can go back to being even.
  const padX = 26, padY = 40;
  const s = Math.min((W - padX * 2) / (maxX - minX), (H - padY * 2) / (maxY - minY));
  const ox = W / 2 - ((minX + maxX) / 2) * s;
  const oy = H / 2 - ((minY + maxY) / 2) * s;
  const T = (u, v, z) => { const p = project(u, v, z); return { x: ox + p.x * s, y: oy + p.y * s }; };

  // Reveal bottom-up as the replay runs: the corpus first, then each stage.
  const reveal = i => {
    if (progress >= 1) return 1;
    const span = 1 / PLATES.length;
    return Math.max(0, Math.min(1, (progress - i * span) / span));
  };

  ctx.lineJoin = "round"; ctx.lineCap = "round";

  // ---- the corpus plate and its marks -----------------------------------
  const corpus = PLATES[0];
  const a0 = reveal(0);
  if (a0 > 0) {
    drawPlate(ctx, T, corpus, ink, a0, false);
    for (const m of city.marks) {
      const hit = run?.lit.get(m.id);
      const p = T(m.u, m.v, 0.4);
      if (hit) {
        diamond(ctx, p, 2.6 * s * 0.06 + 1.6, hit.colour || ink.faint, null, a0);
      } else {
        diamond(ctx, p, 1.5, null, ink.other, a0 * 0.55);
      }
    }
  }

  // ---- links, drawn under the plates so plates occlude them --------------
  if (run) {
    for (const l of run.links) {
      const i = PLATES.findIndex(p => p.z === l.b.z);
      const a = reveal(Math.max(0, i)) * (l.lives ? 0.85 : 0.3);
      if (a <= 0) continue;
      const p1 = T(l.a.u, l.a.v, l.a.z + 0.6);
      const p2 = T(l.b.u, l.b.v, l.b.z + 0.6);
      ctx.save();
      ctx.globalAlpha = a;
      ctx.strokeStyle = l.lives ? l.colour : ink.other;
      ctx.lineWidth = l.lives ? 1.5 : 0.8;
      if (!l.lives) ctx.setLineDash([2, 3]);
      ctx.beginPath(); ctx.moveTo(p1.x, p1.y);
      // a slight bow, so parallel links stay distinguishable
      const mx = (p1.x + p2.x) / 2 + (p2.y - p1.y) * 0.045;
      const my = (p1.y + p2.y) / 2;
      ctx.quadraticCurveTo(mx, my, p2.x, p2.y);
      ctx.stroke();
      ctx.restore();
    }
  }

  // ---- the stage plates, far to near -------------------------------------
  PLATES.slice(1).forEach((plate, n) => {
    const i = n + 1;
    const a = reveal(i);
    if (a <= 0) return;
    const dim = run?.skipped.has(plate.id);
    const top = drawPlate(ctx, T, plate, ink, a, dim);

    for (const m of run?.placed.get(plate.id) || []) {
      const p = T(m.u, m.v, plate.z + 0.6);
      diamond(ctx, p, m.lives ? 4.2 : 3, m.lives ? m.colour : null,
              m.lives ? null : ink.other, a * (m.lives ? 1 : 0.5));
    }

    // Above or below, never beside: with the flow running left to right a
    // side label lands on top of the next plate.
    label(ctx, T, plate, city, run, ink, a, dim);
  });

  if (a0 > 0) label(ctx, T, corpus, city, run, ink, a0, false);

  // ---- the verdict, at the top -------------------------------------------
  if (run && reveal(PLATES.length - 1) > 0.4) {
    const top = PLATES[PLATES.length - 1];
    const p = T(top.u, top.v, top.z + 9);
    ctx.save();
    ctx.globalAlpha = reveal(PLATES.length - 1);
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillStyle = run.confident ? ink.good : ink.critical;
    ctx.font = "600 12.5px Inter, system-ui, sans-serif";
    ctx.fillText(run.confident ? "Answering" : "Declined", p.x, p.y);
    ctx.restore();
  }
}

function runSub(id, run) {
  const c = run.counts;
  switch (id) {
    case "corpus": return run.city.total.toLocaleString() + " passages";
    case "dense": return c.dense + " found";
    case "sparse": return c.sparse ? c.sparse + " found" : "nothing matched";
    case "fused": return c.both + " in both lists";
    case "reranked": return c.biggest && c.biggest.delta
      ? "one moved up " + Math.abs(c.biggest.delta)
      : c.reranked + " re-read";
    case "selected": return c.selected + " kept";
    case "answer": return c.selected + " passages";
    default: return "";
  }
}

/* ------------------------------------------------------------- captions */
export function captions(run) {
  if (!run) return [];
  const c = run.counts;
  const big = c.biggest && c.biggest.delta ? c.biggest : null;
  return [
    { title: "Your documents",
      text: "All " + run.city.total.toLocaleString() + " passages across " +
            run.city.docs + " files. They were indexed before you asked, so "
            + "nothing is being read from scratch now." },
    { title: "Two searches at once",
      text: "One looks for similar wording and found " + c.dense +
            ". The other looks for your exact words and found " + c.sparse +
            ". Each misses what the other catches, so both run." },
    { title: "The lists get merged",
      text: c.both + " passages showed up in both searches. Turning up twice "
            + "counts for more than topping one list alone." },
    { title: "Everything gets read again",
      text: big
        ? "A slower, more careful model reads each passage beside your question. "
          + "One jumped up " + Math.abs(big.delta) + " places: " + shortDoc(big.source) + "."
        : "A slower, more careful model reads each passage beside your question. "
          + "This time it barely changed the order." },
    { title: "Trimmed to a spread",
      text: "At most two passages from any one file, so a single document cannot "
            + "take every slot. " + c.selected + " kept." },
    { title: run.confident ? "Answered" : "Nothing to answer with",
      text: run.confident
        ? "The best passage scored well clear of the cut-off, so it answers."
        : "Everything came in under the cut-off, so it says nothing rather than "
          + "handing you the closest thing it found." },
  ];
}
