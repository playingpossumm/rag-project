/* The pipeline map: one isometric figure that both explains and proves.
 *
 * This replaces the bump chart rather than sitting beside it. The bump chart's
 * logic is not thrown away -- it is the skeleton here. Rank is still vertical
 * position, passages are still ribbons travelling between stages, and every
 * value still appears in the tables below. What changes is that the stages
 * stop being bare columns and become places on a ground plane, which lets the
 * figure say two things a column chart could not:
 *
 *   Dense and BM25 run AT THE SAME TIME. The old chart drew them as column 1
 *   and column 2, which reads as "first this, then that" and is simply wrong
 *   about the mechanism. Here the index forks into two streams that arrive at
 *   two platforms on the same step, braced and labelled as simultaneous.
 *
 *   Score is a magnitude, not an ordering. Rank tells you who won; it does not
 *   say by how much. Height carries that, so a reranker that promotes a
 *   passage ten places AND scores it far above the rest looks different from
 *   one that barely separates the top two.
 *
 * Drawn as a blueprint: hairline edges, no solid masses. A technical drawing
 * rather than a diorama -- it has to sit above dense tables without shouting,
 * and an outline reads at a glance where a shaded solid competes.
 *
 * The projection trick worth knowing: stages advance along (+gx, -gy), which
 * in isometric is pure horizontal movement -- so the pipeline reads left to
 * right like a figure in a paper while every platform still draws as a diamond
 * and the scene still reads as a space. Advancing along +gx alone would send
 * the whole pipeline drifting down the screen.
 *
 * Nothing here invents data. Marks are real chunks, heights are real scores,
 * the pool holds one mark per chunk in the index, and on a refusal the gate
 * is drawn shut because the gate was shut.
 */
"use strict";

/* ---------------------------------------------------------- projection */
const TW = 9.2;   // px per grid step along (gx - gy) -- horizontal on screen
const TH = 5.4;   // px per grid step along (gx + gy) -- depth, vertical-ish
const SZ = 30;    // px per unit of height

const project = (gx, gy, z = 0) => ({
  x: (gx - gy) * TW,
  y: (gx + gy) * TH - z * SZ,
});

/* A point given as `u` along the pipeline and `v` across it. Both retrievers
   share a `u` because they genuinely run at the same moment. */
const at = (u, v) => ({ gx: u + v, gy: -u + v });
const uv = (u, v, z = 0) => project(u + v, -u + v, z);

/* ------------------------------------------------------------- helpers */
const rng = seed => {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

export const shortDoc = s => String(s).replace(/\.(pdf|docx|pptx|xlsx)$/i, "");

/* Only documents that reach the final answer are coloured, and only the first
   three: past that no ordering clears the palette's all-pairs CVD floor, so a
   fourth folds to grey and carries its name as a direct label. Colour follows
   the document, never its rank. */
function palette(finalSources, ink) {
  const slots = [ink.s1, ink.s2, ink.s3];
  const m = new Map();
  finalSources.forEach((s, i) => m.set(s, i < 3 ? slots[i] : ink.other));
  return src => m.get(src) || ink.other;
}

/* --------------------------------------------------------------- plan
   Depths are set so a ten-deep queue cannot reach the neighbouring branch:
   the queue spans 4*D*TH px and the two retrievers sit 4*BRANCH*TH apart.
   Deriving the queue step from platform depth instead is what collapsed the
   two retrievers into one column in the first draft. */
const QUEUE = 10;
const STEP = 1.55;                     // grid units between adjacent ranks
const HALF = ((QUEUE - 1) / 2) * STEP; // half the queue's depth
const BRANCH = 23;                     // how far each retriever sits off-spine

const PLAN = [
  { id: "pool",     u: 0,  v: 0,       w: 12, d: 12,      label: "The index" },
  { id: "dense",    u: 28, v: -BRANCH, w: 5,  d: HALF + 2, label: "Dense" },
  { id: "sparse",   u: 28, v: BRANCH,  w: 5,  d: HALF + 2, label: "BM25" },
  { id: "fused",    u: 50, v: 0,       w: 5,  d: HALF + 2, label: "Fusion" },
  { id: "reranked", u: 66, v: 0,       w: 5,  d: HALF + 2, label: "Reranking" },
  { id: "selected", u: 80, v: 0,       w: 5,  d: HALF + 2, label: "Diversity cap" },
  { id: "gate",     u: 93, v: 0,       w: 5,  d: 6,        label: "The gate" },
];

const STAGE_OF = {
  dense: "dense", sparse: "sparse", fused: "fused",
  reranked: "reranked", selected: "selected",
};

/* ---------------------------------------------------------------- build
   Turns a trace into positioned geometry once, so drawing a frame is pure
   lookup. Recomputing per frame would measure the same layout sixty times a
   second to get the same answer. */
export function buildScene(trace, corpus, ink) {
  const stages = new Map(trace.stages.map(s => [s.name, s]));
  const final = stages.get("selected")?.items || [];
  const order = [...new Set(final.map(i => i.source))];
  const colour = palette(order, ink);
  const finalIds = new Set(final.map(i => i.chunk_id));
  const confident = !!trace.verdict?.confident;

  const platforms = PLAN.map(p => ({ ...p, ...at(p.u, p.v) }));
  const byId = new Map(platforms.map(p => [p.id, p]));

  /* The pool: one mark per chunk in the index, clustered by document. Twenty
     clusters because the corpus holds twenty documents. Seeded, so the pool
     looks the same on every reload of the same corpus. */
  const pool = [];
  const clusters = new Map();          // source -> where its marks live in `pool`
  const sources = corpus?.sources || [];
  const total = corpus?.chunks || 0;
  if (total && sources.length) {
    const r = rng(20250819);
    const P = byId.get("pool");
    const cols = Math.ceil(Math.sqrt(sources.length));
    const rows = Math.ceil(sources.length / cols);
    const per = Math.floor(total / sources.length);
    sources.forEach((src, i) => {
      const cx = ((i % cols) / Math.max(cols - 1, 1) - 0.5) * 1.55;
      const cy = (Math.floor(i / cols) / Math.max(rows - 1, 1) - 0.5) * 1.55;
      const n = i === sources.length - 1 ? total - per * (sources.length - 1) : per;
      const lit = order.includes(src);
      const start = pool.length;
      for (let k = 0; k < n; k++) {
        const a = r() * Math.PI * 2, rad = (r() + r() + r()) / 3;
        pool.push({
          gx: P.gx + cx * P.w + Math.cos(a) * rad * 3.1,
          gy: P.gy + cy * P.d + Math.sin(a) * rad * 3.1,
          lit, src, drawn: false,
        });
      }
      clusters.set(src, { start, n });
    });
  }

  /* Every stage's queue, height normalised inside the stage. Scores are not
     comparable across stages -- a cosine similarity of 0.65, an RRF score of
     0.03 and a cross-encoder logit of +9.27 are three different units -- so
     each platform is its own scale. */
  const queues = new Map();
  for (const [pid, sname] of Object.entries(STAGE_OF)) {
    const st = stages.get(sname);
    if (!st) continue;
    const items = st.items.slice(0, QUEUE);
    const vals = items.map(i => (typeof i.score === "number" ? i.score : 0));
    const lo = Math.min(...vals), hi = Math.max(...vals);
    const span = hi - lo || 1;
    const P = byId.get(pid);
    queues.set(pid, items.map((it, k) => {
      // Rank 1 sits at the top of the screen, as it does in a ranking table.
      const d = -HALF + k * STEP;
      return {
        ...it, stage: pid,
        gx: P.gx + d, gy: P.gy + d,
        z: 0.16 + ((it.score ?? lo) - lo) / span * 1.1,
        colour: finalIds.has(it.chunk_id) ? colour(it.source) : ink.other,
        survives: finalIds.has(it.chunk_id),
      };
    }));
  }

  /* Ribbons: the bump chart's lines, one per passage, between consecutive
     platforms it appears on. Both retrievers feed fusion, which is the whole
     point of drawing them as branches. */
  const links = [];
  const hop = (fromId, toId) => {
    const a = queues.get(fromId) || [], b = queues.get(toId) || [];
    const bi = new Map(b.map(m => [m.chunk_id, m]));
    for (const m of a) {
      const t = bi.get(m.chunk_id);
      if (t) links.push({ from: m, to: t, survives: m.survives && t.survives, colour: t.colour });
    }
  };
  hop("dense", "fused"); hop("sparse", "fused");
  hop("fused", "reranked"); hop("reranked", "selected");

  // The surviving passages converging on the gate. Without these the gate read
  // as an unrelated object parked to the right of the pipeline.
  const G = byId.get("gate");
  for (const m of (queues.get("selected") || [])) {
    links.push({
      from: m,
      to: { gx: G.gx, gy: G.gy, z: 0.6, stage: "gate" },
      survives: true, colour: m.colour,
    });
  }

  /* What the cap threw out, so the trade is visible rather than asserted. */
  const rQ = queues.get("reranked") || [], sQ = queues.get("selected") || [];
  const kept = new Set(sQ.map(m => m.chunk_id));
  const displaced = rQ.filter(m => !kept.has(m.chunk_id) && m.rank <= 6);

  /* The fork: one path per retrieved chunk, from its dot in the index to the
     retriever that found it. A chunk found by both retrievers is drawn twice,
     which is the point -- that is what fusion is about to reward. */
  const forks = [];
  const pickDot = (chunkId, src) => {
    const c = clusters.get(src);
    if (!c || !c.n) return null;
    // Stable, so the same chunk leaves from the same dot on every replay.
    const h = (Math.imul(chunkId ^ 0x9e3779b9, 0x85ebca6b) >>> 0) % c.n;
    return pool[c.start + h];
  };
  for (const pid of ["dense", "sparse"]) {
    for (const m of (queues.get(pid) || [])) {
      const dot = pickDot(m.chunk_id, m.source);
      if (!dot) continue;
      dot.drawn = true;
      forks.push({ id: pid, from: dot, to: m, colour: m.colour, survives: m.survives });
    }
  }

  return {
    platforms, byId, pool, queues, links, displaced, forks,
    colour, order, confident,
    verdict: trace.verdict, stages, query: trace.query,
    counts: {
      pool: total,
      dense: stages.get("dense")?.items.length || 0,
      sparse: stages.get("sparse")?.items.length || 0,
      fused: stages.get("fused")?.items.length || 0,
      reranked: stages.get("reranked")?.items.length || 0,
      selected: final.length,
      both: (stages.get("fused")?.items || []).filter(i => i.agreement).length,
      displaced: displaced.length,
    },
  };
}

/* ----------------------------------------------------------------- draw */

function poly(ctx, pts, close = true) {
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
  if (close) ctx.closePath();
}

/** A platform: a hairline plate with corner ticks. No fill -- blueprint. */
function drawPlatform(ctx, p, ink, opts = {}) {
  const alpha = opts.alpha ?? 1;
  const c = [
    project(p.gx - p.w, p.gy - p.d), project(p.gx + p.w, p.gy - p.d),
    project(p.gx + p.w, p.gy + p.d), project(p.gx - p.w, p.gy + p.d),
  ];
  ctx.globalAlpha = alpha;
  // A very light wash so a plate reads as a surface, not a floating outline.
  poly(ctx, c);
  ctx.fillStyle = ink.panel; ctx.globalAlpha = alpha * 0.55; ctx.fill();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = opts.active ? ink.s1 : ink.line;
  ctx.lineWidth = opts.active ? 1.5 : 1;
  ctx.stroke();
  // Corner ticks: the detail that makes it read as a drawing rather than a box.
  ctx.strokeStyle = opts.active ? ink.s1 : ink.faint;
  ctx.lineWidth = 1;
  for (const q of c) {
    ctx.beginPath();
    ctx.moveTo(q.x - 3, q.y); ctx.lineTo(q.x + 3, q.y);
    ctx.moveTo(q.x, q.y - 2); ctx.lineTo(q.x, q.y + 2);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

/** A mark: a wireframe box standing on the platform, its height the score. */
function drawMark(ctx, m, ink, opts = {}) {
  const s = opts.size ?? 1.75;
  const alpha = opts.alpha ?? 1;
  const strong = m.survives;

  const top = [
    project(m.gx - s, m.gy - s, m.z), project(m.gx + s, m.gy - s, m.z),
    project(m.gx + s, m.gy + s, m.z), project(m.gx - s, m.gy + s, m.z),
  ];
  const base = [
    project(m.gx - s, m.gy - s, 0), project(m.gx + s, m.gy - s, 0),
    project(m.gx + s, m.gy + s, 0), project(m.gx - s, m.gy + s, 0),
  ];

  ctx.globalAlpha = alpha * (strong ? 1 : 0.42);
  ctx.strokeStyle = m.colour;
  ctx.lineWidth = strong ? 1.4 : 1;

  // The three vertical edges a viewer can actually see, plus the near base
  // edges: enough to read as a box without drawing hidden geometry.
  for (const i of [1, 2, 3]) {
    ctx.beginPath();
    ctx.moveTo(base[i].x, base[i].y); ctx.lineTo(top[i].x, top[i].y);
    ctx.stroke();
  }
  ctx.beginPath();
  ctx.moveTo(base[1].x, base[1].y); ctx.lineTo(base[2].x, base[2].y);
  ctx.lineTo(base[3].x, base[3].y);
  ctx.stroke();

  poly(ctx, top);
  if (strong) {
    ctx.globalAlpha = alpha * 0.30; ctx.fillStyle = m.colour; ctx.fill();
    ctx.globalAlpha = alpha;
  }
  ctx.stroke();
  ctx.globalAlpha = 1;
}

/** The gate: two posts and a bar. Raised when answering, down when declined. */
function drawGate(ctx, p, ink, confident, alpha = 1) {
  const H = 1.5;
  const L = project(p.gx - p.w, p.gy + p.d);
  const R = project(p.gx + p.w, p.gy - p.d);
  const Lt = project(p.gx - p.w, p.gy + p.d, H);
  const Rt = project(p.gx + p.w, p.gy - p.d, H);
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = ink.faint; ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(L.x, L.y); ctx.lineTo(Lt.x, Lt.y);
  ctx.moveTo(R.x, R.y); ctx.lineTo(Rt.x, Rt.y);
  ctx.stroke();

  ctx.strokeStyle = confident ? ink.good : ink.critical;
  ctx.lineWidth = 2.6;
  if (confident) {
    // Raised: the bar sits at the top of the posts, out of the way.
    ctx.beginPath(); ctx.moveTo(Lt.x, Lt.y); ctx.lineTo(Rt.x, Rt.y); ctx.stroke();
  } else {
    // Down across the opening, with a second bar so it reads as barred rather
    // than as a line that happens to be red.
    const l1 = project(p.gx - p.w, p.gy + p.d, 0.55);
    const r1 = project(p.gx + p.w, p.gy - p.d, 0.55);
    const l2 = project(p.gx - p.w, p.gy + p.d, 0.95);
    const r2 = project(p.gx + p.w, p.gy - p.d, 0.95);
    ctx.beginPath();
    ctx.moveTo(l1.x, l1.y); ctx.lineTo(r1.x, r1.y);
    ctx.moveTo(l2.x, l2.y); ctx.lineTo(r2.x, r2.y);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

/** The brace that says the two retrievers happen at once. */
function drawBrace(ctx, scene, ink, alpha) {
  const D = scene.byId.get("dense"), S = scene.byId.get("sparse");
  const a = project(D.gx, D.gy, 1.55);
  const b = project(S.gx, S.gy, 1.55);
  const midX = (a.x + b.x) / 2 - 34;
  ctx.globalAlpha = alpha * 0.9;
  ctx.strokeStyle = ink.faint; ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(a.x - 26, a.y); ctx.lineTo(midX, a.y);
  ctx.lineTo(midX, b.y); ctx.lineTo(b.x - 26, b.y);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.globalAlpha = 1;
  return { x: midX, y: (a.y + b.y) / 2 };
}

/**
 * Render the whole scene.
 *
 * `progress` is 0..1 across the pipeline, so the caller owns the clock: the
 * same function draws a settled figure, an autoplayed replay and a scrubbed
 * frame, and none of them can disagree about what stage three looks like.
 */
export function drawScene(ctx, scene, ink, W, H, progress = 1) {
  ctx.clearRect(0, 0, W, H);

  const xs = [], ys = [];
  for (const p of scene.platforms) {
    for (const [dx, dy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      for (const z of [0, 1.7]) {
        const q = project(p.gx + dx * p.w, p.gy + dy * p.d, z);
        xs.push(q.x); ys.push(q.y);
      }
    }
  }
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  // Room reserved for type, which is drawn outside the geometry's bounds:
  // stage names above, BM25's caption below, the verdict off the right end.
  const PAD = { l: 24, r: 118, t: 52, b: 56 };
  const scale = Math.min(
    (W - PAD.l - PAD.r) / (maxX - minX),
    (H - PAD.t - PAD.b) / (maxY - minY),
  );

  ctx.save();
  ctx.translate(W / 2 + (PAD.l - PAD.r) / 2, H / 2 + (PAD.t - PAD.b) / 2);
  ctx.scale(scale, scale);
  ctx.translate(-(minX + maxX) / 2, -(minY + maxY) / 2);
  ctx.lineJoin = "round";

  // Each platform gets a slice of the timeline; 0 = nothing, 1 = fully drawn.
  const SEQ = ["pool", "dense", "fused", "reranked", "selected", "gate"];
  const phase = id => {
    const i = id === "sparse" ? 1 : SEQ.indexOf(id);
    return Math.max(0, Math.min(1, progress * SEQ.length - i));
  };

  // 1 — the index and its pool of chunks
  const pp = phase("pool");
  if (pp > 0) {
    drawPlatform(ctx, scene.byId.get("pool"), ink, { alpha: pp });
    for (const m of scene.pool) {
      const q = project(m.gx, m.gy, 0.22);
      if (m.drawn) continue;                       // drawn again, louder, below
      ctx.globalAlpha = (m.lit ? 0.42 : 0.20) * pp;
      ctx.fillStyle = m.lit ? scene.colour(m.src) : ink.faint;
      ctx.fillRect(q.x, q.y, 1.5, 1.5);
    }
    for (const m of scene.pool) {
      if (!m.drawn) continue;
      const q = project(m.gx, m.gy, 0.22);
      ctx.globalAlpha = pp;
      ctx.fillStyle = scene.colour(m.src);
      ctx.fillRect(q.x - 1, q.y - 1, 3.4, 3.4);
    }
    ctx.globalAlpha = 1;
  }

  // 2 — the fork: every retrieved chunk, drawn leaving its own dot in the index
  //     for the retriever that found it. Both branches at once.
  const fp = phase("dense");
  if (fp > 0) {
    for (const f of scene.forks) {
      const a = project(f.from.gx, f.from.gy, 0.22);
      const b = project(f.to.gx, f.to.gy, f.to.z);
      const x = a.x + (b.x - a.x) * fp, y = a.y + (b.y - a.y) * fp;
      ctx.strokeStyle = f.survives ? f.colour : ink.faint;
      ctx.globalAlpha = f.survives ? 0.55 : 0.20;
      ctx.lineWidth = f.survives ? 1.2 : 0.9;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.bezierCurveTo(a.x + (b.x - a.x) * 0.5, a.y, x - (b.x - a.x) * 0.42, y, x, y);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  // 3 — platforms
  for (const id of ["dense", "sparse", "fused", "reranked", "selected"]) {
    const a = phase(id);
    if (a > 0) drawPlatform(ctx, scene.byId.get(id), ink, { alpha: a });
  }
  let braceAt = null;
  if (fp > 0.6) braceAt = drawBrace(ctx, scene, ink, Math.min(1, (fp - 0.6) / 0.4));

  // 4 — ribbons under the marks, so a passage arriving at a platform reads as
  //     landing on it rather than crossing in front of it
  for (const l of scene.links) {
    const a = phase(l.to.stage);
    if (a <= 0) continue;
    const p0 = project(l.from.gx, l.from.gy, l.from.z);
    const p1 = project(l.to.gx, l.to.gy, l.to.z);
    const x = p0.x + (p1.x - p0.x) * a, y = p0.y + (p1.y - p0.y) * a;
    ctx.strokeStyle = l.colour;
    ctx.globalAlpha = l.survives ? 0.9 : 0.14;
    ctx.lineWidth = l.survives ? 1.7 : 1;
    ctx.beginPath();
    ctx.moveTo(p0.x, p0.y);
    ctx.bezierCurveTo(p0.x + (p1.x - p0.x) * 0.45, p0.y, x - (p1.x - p0.x) * 0.45, y, x, y);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  // 5 — marks
  for (const [pid, q] of scene.queues) {
    const a = phase(pid);
    if (a <= 0) continue;
    for (const m of q) drawMark(ctx, m, ink, { alpha: a });
  }

  // 6 — the gate, last, because it is the answer
  const gp = phase("gate");
  if (gp > 0) {
    const G = scene.byId.get("gate");
    drawPlatform(ctx, G, ink, { alpha: gp });
    drawGate(ctx, G, ink, scene.confident, gp);
  }

  ctx.restore();

  /* ---- type, in screen space -------------------------------------------
     Positions come from the scene, sizes do not: a label that scales with the
     figure is unreadable the moment the figure has to fit a narrow column. */
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  const toScreen = q => ({
    x: W / 2 + (PAD.l - PAD.r) / 2 + (q.x - cx) * scale,
    y: H / 2 + (PAD.t - PAD.b) / 2 + (q.y - cy) * scale,
  });

  const c = scene.counts;
  const LABELS = [
    ["pool", "The index", `${c.pool.toLocaleString()} chunks`, "above"],
    ["dense", "Dense", `${c.dense} candidates · meaning`, "above"],
    ["sparse", "BM25", `${c.sparse} candidates · exact words`, "below"],
    ["fused", "Fusion", `${c.both} of ${c.fused} found by both`, "above"],
    ["reranked", "Reranking", `${c.reranked} rescored`, "above"],
    ["selected", "Diversity cap",
      c.displaced ? `${c.displaced} displaced · ${c.selected} kept` : `${c.selected} kept`, "above"],
    ["gate", scene.confident ? "Answering" : "Declined",
      scene.verdict?.confidence != null
        ? `${scene.verdict.confidence > 0 ? "+" : ""}${scene.verdict.confidence} against ${scene.verdict.threshold.toFixed(1)}`
        : "", "above"],
  ];

  ctx.textAlign = "center";
  for (const [id, title, sub, side] of LABELS) {
    const a = phase(id);
    if (a <= 0.35) continue;
    const p = scene.byId.get(id);
    // Anchor clear of the tallest thing standing on the platform.
    const anchor = toScreen(project(p.gx, p.gy + (side === "below" ? p.d : -p.d),
                                    side === "below" ? 0 : 2.15));
    const y = side === "below" ? anchor.y + 34 : anchor.y - 24;
    ctx.globalAlpha = Math.min(1, (a - 0.35) / 0.4);
    ctx.textBaseline = "alphabetic";
    ctx.fillStyle = id === "gate" ? (scene.confident ? ink.good : ink.critical) : ink.ink;
    ctx.font = '600 13px "Iowan Old Style", Charter, Cambria, Georgia, serif';
    ctx.fillText(title, anchor.x, y);
    ctx.fillStyle = ink.faint;
    ctx.font = '10.5px ui-monospace, "Cascadia Mono", Consolas, monospace';
    ctx.fillText(sub, anchor.x, y + 13);
  }
  if (braceAt) {
    const b = toScreen(braceAt);
    ctx.save();
    ctx.translate(b.x - 9, b.y);
    ctx.rotate(-Math.PI / 2);
    ctx.globalAlpha = Math.min(1, (fp - 0.6) / 0.4);
    ctx.fillStyle = ink.muted;
    ctx.font = '600 10px ui-monospace, "Cascadia Mono", Consolas, monospace';
    ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    ctx.fillText("AT THE SAME MOMENT", 0, 0);
    ctx.restore();
  }
  ctx.globalAlpha = 1;
  return { scale, toScreen };
}

/* Plain-language captions, using this query's real numbers. One per stage,
   because a non-technical viewer needs the mechanism named as it happens and
   nobody needs a paragraph. */
export function captions(scene) {
  const c = scene.counts;
  return [
    { id: "pool", title: "The index",
      text: `Your ${scene.pool.length ? "documents" : "corpus"} cut into ${c.pool.toLocaleString()} searchable passages.` },
    { id: "dense", title: "Two searches, at once",
      text: `One matches meaning, one matches exact words. Each returns its own ${c.dense} best guesses.` },
    { id: "fused", title: "Fusion",
      text: `The two rankings are merged by position, never by score. ${c.both} of ${c.fused} were found by both — agreement is what fusion rewards.` },
    { id: "reranked", title: "Reranking",
      text: `Now a second model reads your question and each passage together, and rescores all ${c.reranked}.` },
    { id: "selected", title: "Diversity cap",
      text: c.displaced
        ? `At most 2 passages per document, so one paper cannot take every slot. ${c.displaced} were displaced.`
        : `At most 2 passages per document. The cap was not reached here.` },
    { id: "gate", title: scene.confident ? "The gate opens" : "The gate stays shut",
      text: scene.confident
        ? `The top passage scores above the threshold, so the system answers.`
        : `Nothing scored above the threshold, so it declines rather than returning the closest topical match.` },
  ];
}
