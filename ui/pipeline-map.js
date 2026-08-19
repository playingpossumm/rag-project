/* The pipeline map: one isometric figure that both explains and proves.
 *
 * It replaces the bump chart rather than sitting beside it. The chart's logic
 * is the skeleton here -- rank is vertical position, passages are ribbons, and
 * every value still lands in the tables below.
 *
 * THE INDEX IS DRAWN AS A FIELD AROUND YOUR QUESTION, and that is the part
 * worth reading carefully, because it is the only claim in the figure that
 * could easily have been a lie.
 *
 *   Distance from the centre is the chunk's similarity RANK against this
 *   question -- the exact quantity dense retrieval orders by, taken from
 *   /api/space. Radius grows as sqrt(rank/n) so marks spread evenly over the
 *   disc instead of piling into a ring.
 *
 *   The angle means nothing. It is a stable hash of the chunk id, and the
 *   figure says so on its face. Nothing is ever read off it.
 *
 *   A 2-D PCA of the 384-dimension embedding space was the obvious way to do
 *   this and is the wrong one: the first two components carry 17% of the
 *   variance on this corpus, so "close on screen" would routinely disagree
 *   with "close to the model". One true dimension beats two misleading ones.
 *
 * What that buys is the point of hybrid retrieval, visible rather than
 * asserted: dense takes the innermost ring, while BM25 lights chunks scattered
 * far out -- passages dense ranked hundreds deep and would never have reached.
 * That is why the pipeline has two retrievers, and you can see it.
 *
 * The wiring is drawn before it is used. Every chunk already has a line to the
 * candidate pool; retrieval lights forty of them. Lines that GROW would say the
 * index is constructing a result. Lines that were always there and light up say
 * every chunk was reachable and the query selected these, which is what
 * retrieval actually is.
 *
 * Elsewhere: stages advance along (+gx, -gy), which in isometric is pure
 * horizontal movement, so the figure reads left to right while every platform
 * still draws as a diamond. Score is height, because rank says who won and not
 * by how much. Drawn as a blueprint -- hairlines, no solid masses -- because it
 * sits above dense tables and must not shout.
 */
"use strict";

/* ---------------------------------------------------------- projection */
const TW = 9.2;   // px per grid step along (gx - gy) -- horizontal on screen
const TH = 6.4;   // px per grid step along (gx + gy) -- depth, vertical-ish
const SZ = 30;    // px per unit of height

const project = (gx, gy, z = 0) => ({
  x: (gx - gy) * TW,
  y: (gx + gy) * TH - z * SZ,
});

/* A point given as `u` along the pipeline and `v` across it. */
const at = (u, v) => ({ gx: u + v, gy: -u + v });

/* ------------------------------------------------------------- helpers */
/* A stable angle per chunk. Deliberately meaningless -- see the header. */
const angleOf = id =>
  ((Math.imul(id ^ 0x9e3779b9, 0x85ebca6b) >>> 8) % 100000) / 100000 * Math.PI * 2;

export const shortDoc = s => String(s).replace(/\.(pdf|docx|pptx|xlsx)$/i, "");

/* Only documents reaching the final answer are coloured, and only the first
   three: past that no ordering clears the palette's all-pairs CVD floor, so a
   fourth folds to grey and carries its name as a direct label. */
function palette(finalSources, ink) {
  const slots = [ink.s1, ink.s2, ink.s3];
  const m = new Map();
  finalSources.forEach((s, i) => m.set(s, i < 3 ? slots[i] : ink.other));
  return src => m.get(src) || ink.other;
}

/* --------------------------------------------------------------- plan */
const QUEUE = 10;
const STEP = 1.55;
const HALF = ((QUEUE - 1) / 2) * STEP;
const FIELD = 27;                      // radius of the index field, grid units

const PLAN = [
  { id: "query",    u: -22, v: 0, w: 4,     d: 4 },
  { id: "index",    u: 8,   v: 0, w: FIELD, d: FIELD },
  { id: "fused",    u: 52,  v: 0, w: 5,     d: HALF + 2 },
  { id: "reranked", u: 66,  v: 0, w: 5,     d: HALF + 2 },
  { id: "selected", u: 79,  v: 0, w: 5,     d: HALF + 2 },
  { id: "gate",     u: 91,  v: 0, w: 5,     d: 6 },
  { id: "answer",   u: 104, v: 0, w: 6,     d: 7 },
];

const STAGE_OF = { fused: "fused", reranked: "reranked", selected: "selected" };

/* The order stages light in. Dense and BM25 are not steps: they are two
   readings of the same index, and they happen together inside one beat. */
const SEQ = ["query", "index", "fused", "reranked", "selected", "gate", "answer"];

/* ---------------------------------------------------------------- build */
export function buildScene(trace, space, ink) {
  const stages = new Map(trace.stages.map(s => [s.name, s]));
  const final = stages.get("selected")?.items || [];
  const order = [...new Set(final.map(i => i.source))];
  const colour = palette(order, ink);
  const finalIds = new Set(final.map(i => i.chunk_id));
  const confident = !!trace.verdict?.confident;

  const platforms = PLAN.map(p => ({ ...p, ...at(p.u, p.v) }));
  const byId = new Map(platforms.map(p => [p.id, p]));
  const I = byId.get("index");

  /* ---- the index as a field around the question ---------------------- */
  const denseIds = new Set((stages.get("dense")?.items || []).map(i => i.chunk_id));
  const sparseIds = new Set((stages.get("sparse")?.items || []).map(i => i.chunk_id));

  const field = [];
  if (space?.sims?.length) {
    const n = space.sims.length;
    const byRank = Array.from({ length: n }, (_, i) => i)
      .sort((a, b) => space.sims[b] - space.sims[a]);
    byRank.forEach((id, rank) => {
      // Log, not sqrt. sqrt spreads marks evenly over the disc's area, which
      // is honest and useless: dense retrieval's top 20 of 2,768 then land
      // inside 8% of the radius and cannot be seen at all. A log radius gives
      // the head of the ranking room -- the top 20 occupy the inner third --
      // while the long tail compresses to the rim, where it belongs. Still
      // strictly monotone in rank, so "closer is more similar" stays exactly
      // true; only the spacing is chosen.
      const rad = FIELD * Math.log1p(rank) / Math.log1p(n);
      const a = angleOf(id);
      field.push({
        chunk_id: id, rank, sim: space.sims[id], src: space.sources[space.doc[id]],
        gx: I.gx + Math.cos(a) * rad, gy: I.gy + Math.sin(a) * rad,
        dense: denseIds.has(id), sparse: sparseIds.has(id),
        lit: denseIds.has(id) || sparseIds.has(id),
      });
    });
  }
  const fieldById = new Map(field.map(f => [f.chunk_id, f]));

  /* ---- queues on the later platforms --------------------------------- */
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
      const d = -HALF + k * STEP;                 // rank 1 at the top
      return {
        ...it, stage: pid,
        gx: P.gx + d, gy: P.gy + d,
        z: 0.16 + ((it.score ?? lo) - lo) / span * 1.1,
        colour: finalIds.has(it.chunk_id) ? colour(it.source) : ink.other,
        survives: finalIds.has(it.chunk_id),
      };
    }));
  }

  /* ---- the wiring, built once ---------------------------------------- */
  const F = byId.get("fused");
  let latent = null;
  if (typeof Path2D !== "undefined" && field.length) {
    latent = new Path2D();
    for (const f of field) {
      const a = project(f.gx, f.gy, 0);
      // Spread the arrival along the platform's leading edge rather than
      // spiking every line into one pixel.
      const t = (f.rank % 97) / 97 - 0.5;
      const e = project(F.gx - F.w * 1.4 + t * (HALF + 2), F.gy + t * (HALF + 2), 0.35);
      latent.moveTo(a.x, a.y);
      latent.lineTo(e.x, e.y);
    }
  }

  // The ones that light: each retrieved chunk, from where it sits in the field
  // to the candidate it became.
  const lit = [];
  for (const m of (queues.get("fused") || [])) {
    const f = fieldById.get(m.chunk_id);
    if (f) lit.push({ from: f, to: m, colour: m.colour, survives: m.survives, both: !!m.agreement });
  }

  /* ---- ribbons between the later platforms --------------------------- */
  const links = [];
  const hop = (a, b) => {
    const A = queues.get(a) || [], B = queues.get(b) || [];
    const bi = new Map(B.map(m => [m.chunk_id, m]));
    for (const m of A) {
      const t = bi.get(m.chunk_id);
      if (t) links.push({ from: m, to: t, survives: m.survives && t.survives, colour: t.colour });
    }
  };
  hop("fused", "reranked"); hop("reranked", "selected");

  const G = byId.get("gate");
  for (const m of (queues.get("selected") || [])) {
    links.push({ from: m, to: { gx: G.gx, gy: G.gy, z: 0.6, stage: "gate" },
                 survives: true, colour: m.colour });
  }

  const rQ = queues.get("reranked") || [], sQ = queues.get("selected") || [];
  const kept = new Set(sQ.map(m => m.chunk_id));
  const displaced = rQ.filter(m => !kept.has(m.chunk_id) && m.rank <= 6);
  const sparseOutside = [...sparseIds].filter(id => !denseIds.has(id)).length;

  return {
    platforms, byId, field, fieldById, queues, links, lit, latent, displaced,
    colour, order, confident, verdict: trace.verdict, query: trace.query, stages,
    counts: {
      pool: space?.n || 0,
      dense: stages.get("dense")?.items.length || 0,
      sparse: stages.get("sparse")?.items.length || 0,
      fused: stages.get("fused")?.items.length || 0,
      reranked: stages.get("reranked")?.items.length || 0,
      selected: final.length,
      both: (stages.get("fused")?.items || []).filter(i => i.agreement).length,
      displaced: displaced.length,
      sparseOutside,
    },
  };
}

/* ----------------------------------------------------------------- draw */
function poly(ctx, pts) {
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
  ctx.closePath();
}

function drawPlatform(ctx, p, ink, alpha = 1) {
  const c = [
    project(p.gx - p.w, p.gy - p.d), project(p.gx + p.w, p.gy - p.d),
    project(p.gx + p.w, p.gy + p.d), project(p.gx - p.w, p.gy + p.d),
  ];
  poly(ctx, c);
  ctx.fillStyle = ink.panel; ctx.globalAlpha = alpha * 0.55; ctx.fill();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = ink.line; ctx.lineWidth = 1; ctx.stroke();
  ctx.strokeStyle = ink.faint;
  for (const q of c) {
    ctx.beginPath();
    ctx.moveTo(q.x - 3, q.y); ctx.lineTo(q.x + 3, q.y);
    ctx.moveTo(q.x, q.y - 2); ctx.lineTo(q.x, q.y + 2);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

/** Concentric rings on the index floor, so radius reads as a scale. */
function drawFieldFloor(ctx, I, ink, alpha) {
  ctx.globalAlpha = alpha * 0.5;
  ctx.strokeStyle = ink.line; ctx.lineWidth = 1;
  for (const frac of [0.33, 0.66, 1]) {
    ctx.beginPath();
    for (let k = 0; k <= 56; k++) {
      const a = (k / 56) * Math.PI * 2;
      const q = project(I.gx + Math.cos(a) * FIELD * frac, I.gy + Math.sin(a) * FIELD * frac, 0);
      k ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y);
    }
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

function drawMark(ctx, m, ink, alpha = 1) {
  const s = 1.75, strong = m.survives;
  const top = [
    project(m.gx - s, m.gy - s, m.z), project(m.gx + s, m.gy - s, m.z),
    project(m.gx + s, m.gy + s, m.z), project(m.gx - s, m.gy + s, m.z),
  ];
  const base = [
    project(m.gx - s, m.gy - s, 0), project(m.gx + s, m.gy - s, 0),
    project(m.gx + s, m.gy + s, 0), project(m.gx - s, m.gy + s, 0),
  ];
  ctx.globalAlpha = alpha * (strong ? 1 : 0.42);
  ctx.strokeStyle = m.colour; ctx.lineWidth = strong ? 1.4 : 1;
  for (const i of [1, 2, 3]) {
    ctx.beginPath();
    ctx.moveTo(base[i].x, base[i].y); ctx.lineTo(top[i].x, top[i].y); ctx.stroke();
  }
  ctx.beginPath();
  ctx.moveTo(base[1].x, base[1].y); ctx.lineTo(base[2].x, base[2].y);
  ctx.lineTo(base[3].x, base[3].y); ctx.stroke();
  poly(ctx, top);
  if (strong) { ctx.globalAlpha = alpha * 0.30; ctx.fillStyle = m.colour; ctx.fill(); ctx.globalAlpha = alpha; }
  ctx.stroke();
  ctx.globalAlpha = 1;
}

function drawGate(ctx, p, ink, confident, alpha) {
  const H = 1.5;
  const L = project(p.gx - p.w, p.gy + p.d), R = project(p.gx + p.w, p.gy - p.d);
  const Lt = project(p.gx - p.w, p.gy + p.d, H), Rt = project(p.gx + p.w, p.gy - p.d, H);
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = ink.faint; ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(L.x, L.y); ctx.lineTo(Lt.x, Lt.y);
  ctx.moveTo(R.x, R.y); ctx.lineTo(Rt.x, Rt.y);
  ctx.stroke();
  ctx.strokeStyle = confident ? ink.good : ink.critical;
  ctx.lineWidth = 2.6;
  ctx.beginPath();
  if (confident) { ctx.moveTo(Lt.x, Lt.y); ctx.lineTo(Rt.x, Rt.y); }
  else {
    for (const z of [0.55, 0.95]) {
      const l = project(p.gx - p.w, p.gy + p.d, z), r = project(p.gx + p.w, p.gy - p.d, z);
      ctx.moveTo(l.x, l.y); ctx.lineTo(r.x, r.y);
    }
  }
  ctx.stroke();
  ctx.globalAlpha = 1;
}

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
  const PAD = { l: 28, r: 116, t: 54, b: 62 };
  const scale = Math.min((W - PAD.l - PAD.r) / (maxX - minX), (H - PAD.t - PAD.b) / (maxY - minY));
  const ox = W / 2 + (PAD.l - PAD.r) / 2, oy = H / 2 + (PAD.t - PAD.b) / 2;

  ctx.save();
  ctx.translate(ox, oy);
  ctx.scale(scale, scale);
  ctx.translate(-(minX + maxX) / 2, -(minY + maxY) / 2);
  ctx.lineJoin = "round";

  const phase = id => Math.max(0, Math.min(1, progress * SEQ.length - SEQ.indexOf(id)));
  const I = scene.byId.get("index");

  // 1 — the question, encoded
  const qp = phase("query");
  if (qp > 0) {
    const Q = scene.byId.get("query");
    const a = project(Q.gx, Q.gy, 0.5), b = project(I.gx, I.gy, 0.5);
    ctx.globalAlpha = qp; ctx.strokeStyle = ink.faint; ctx.lineWidth = 1.4;
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(a.x + (b.x - a.x) * qp, a.y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath(); ctx.arc(a.x, a.y, 5, 0, 6.284); ctx.fillStyle = ink.ink; ctx.fill();
    ctx.globalAlpha = 1;
  }

  // 2 — the index: floor rings, the latent wiring, then the marks
  const ip = phase("index");
  if (ip > 0) {
    drawFieldFloor(ctx, I, ink, ip);
    if (scene.latent) {
      ctx.globalAlpha = 0.018 * ip;
      ctx.strokeStyle = ink.faint; ctx.lineWidth = 0.5;
      ctx.stroke(scene.latent);          // one call for all 2,768
      ctx.globalAlpha = 1;
    }
    for (const f of scene.field) {
      if (f.lit) continue;
      const q = project(f.gx, f.gy, 0);
      ctx.globalAlpha = 0.38 * ip;
      ctx.fillStyle = ink.faint;
      ctx.fillRect(q.x, q.y, 1.5, 1.5);
    }
    // Dense hits are filled, BM25-only hits are rings: the two readings of the
    // index are told apart by shape, never by colour alone.
    for (const f of scene.field) {
      if (!f.lit) continue;
      const q = project(f.gx, f.gy, 0);
      ctx.globalAlpha = ip;
      const col = scene.colour(f.src);
      if (f.dense) { ctx.fillStyle = col; ctx.fillRect(q.x - 2, q.y - 2, 5.2, 5.2); }
      if (f.sparse) {
        ctx.strokeStyle = col; ctx.lineWidth = 1.3;
        ctx.beginPath(); ctx.arc(q.x + 0.5, q.y + 0.5, 5.2, 0, 6.284); ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
    const c = project(I.gx, I.gy, 0);
    ctx.globalAlpha = ip; ctx.strokeStyle = ink.ink; ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.arc(c.x, c.y, 6, 0, 6.284); ctx.stroke();
    ctx.beginPath(); ctx.arc(c.x, c.y, 1.6, 0, 6.284); ctx.fillStyle = ink.ink; ctx.fill();
    ctx.globalAlpha = 1;
  }

  // 3 — the lines that light
  const fp = phase("fused");
  if (fp > 0) {
    for (const l of scene.lit) {
      const a = project(l.from.gx, l.from.gy, 0);
      const b = project(l.to.gx, l.to.gy, l.to.z);
      const x = a.x + (b.x - a.x) * fp, y = a.y + (b.y - a.y) * fp;
      ctx.strokeStyle = l.survives ? l.colour : ink.faint;
      ctx.globalAlpha = l.survives ? 0.75 : 0.3;
      ctx.lineWidth = l.both ? 1.5 : 1;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.bezierCurveTo(a.x + (b.x - a.x) * 0.5, a.y, x - (b.x - a.x) * 0.4, y, x, y);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  // 4 — platforms
  for (const id of ["fused", "reranked", "selected"]) {
    const a = phase(id);
    if (a > 0) drawPlatform(ctx, scene.byId.get(id), ink, a);
  }

  // 5 — ribbons, then marks
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
  for (const [pid, q] of scene.queues) {
    const a = phase(pid);
    if (a > 0) for (const m of q) drawMark(ctx, m, ink, a);
  }

  // 6 — the gate, then the answer it permits
  const gp = phase("gate");
  if (gp > 0) {
    const G = scene.byId.get("gate");
    drawPlatform(ctx, G, ink, gp);
    drawGate(ctx, G, ink, scene.confident, gp);
  }
  const ap = phase("answer");
  if (ap > 0 && scene.confident) {
    const A = scene.byId.get("answer");
    drawPlatform(ctx, A, ink, ap);
    // The surviving passages, expanded to their surrounding context.
    (scene.queues.get("selected") || []).forEach((m, k, arr) => {
      const d = -HALF + k * (2 * HALF / Math.max(arr.length - 1, 1));
      const q = project(A.gx + d * 0.45, A.gy + d * 0.45, 0.3);
      ctx.globalAlpha = ap; ctx.fillStyle = m.colour;
      ctx.fillRect(q.x - 8, q.y - 1.6, 16, 3.2);
    });
    ctx.globalAlpha = 1;
  }

  ctx.restore();

  /* ---- type, in screen space ------------------------------------------ */
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  const toScreen = q => ({ x: ox + (q.x - cx) * scale, y: oy + (q.y - cy) * scale });
  const c = scene.counts;
  const LABELS = [
    ["query", "Your question", "encoded as a vector"],
    ["index", "The index", `${c.pool.toLocaleString()} chunks`],
    ["fused", "Candidates", `${c.both} of ${c.fused} found by both`],
    ["reranked", "Reranking", `${c.reranked} rescored`],
    ["selected", "Diversity cap",
      c.displaced ? `${c.displaced} displaced · ${c.selected} kept` : `${c.selected} kept`],
    ["gate", scene.confident ? "Answering" : "Declined",
      scene.verdict?.confidence != null
        ? `${scene.verdict.confidence > 0 ? "+" : ""}${scene.verdict.confidence} against ${scene.verdict.threshold.toFixed(1)}`
        : ""],
  ];
  if (scene.confident) LABELS.push(["answer", "The answer", `${c.selected} cited passages`]);

  ctx.textAlign = "center";
  for (const [id, title, sub] of LABELS) {
    const a = phase(id);
    if (a <= 0.35) continue;
    const p = scene.byId.get(id);
    const anchor = toScreen(project(p.gx, p.gy - p.d, id === "index" ? 0.3 : 2.15));
    ctx.globalAlpha = Math.min(1, (a - 0.35) / 0.4);
    ctx.textBaseline = "alphabetic";
    ctx.fillStyle = id === "gate" ? (scene.confident ? ink.good : ink.critical) : ink.ink;
    ctx.font = '600 13px "Iowan Old Style", Charter, Cambria, Georgia, serif';
    ctx.fillText(title, anchor.x, anchor.y - 24);
    ctx.fillStyle = ink.faint;
    ctx.font = '10.5px ui-monospace, "Cascadia Mono", Consolas, monospace';
    ctx.fillText(sub, anchor.x, anchor.y - 11);
  }

  // The caveat the field must always carry.
  if (phase("index") > 0.5) {
    const p = toScreen(project(I.gx, I.gy + FIELD, 0));
    ctx.globalAlpha = 0.95;
    ctx.fillStyle = ink.faint;
    ctx.font = '10px ui-monospace, "Cascadia Mono", Consolas, monospace';
    ctx.fillText("distance from centre = similarity rank, log scale · angle means nothing", p.x, p.y + 26);
  }
  ctx.globalAlpha = 1;
  return { scale, toScreen };
}

/* Plain-language captions, using this query's real numbers. */
export function captions(scene) {
  const c = scene.counts;
  return [
    { id: "query", title: "Your question becomes a vector",
      text: "Before anything can be searched, the question is encoded into the same 384-dimension space the passages live in." },
    { id: "index", title: "One index, two readings",
      text: `All ${c.pool.toLocaleString()} chunks, placed by how close each is to your question. Dense retrieval takes the ${c.dense} nearest the centre. BM25 lights ${c.sparseOutside} more scattered further out — passages that share rare words but not meaning.` },
    { id: "fused", title: "Fusion",
      text: `The two rankings are merged by position, never by score. ${c.both} of ${c.fused} were found by both — agreement is what fusion rewards.` },
    { id: "reranked", title: "Reranking",
      text: `A second model now reads your question and each passage together, and rescores all ${c.reranked}.` },
    { id: "selected", title: "Diversity cap",
      text: c.displaced
        ? `At most 2 passages per document, so one paper cannot take every slot. ${c.displaced} were displaced.`
        : "At most 2 passages per document. The cap was not reached here." },
    { id: "gate", title: scene.confident ? "The gate opens" : "The gate stays shut",
      text: scene.confident
        ? "The top passage scores above the threshold, so the system answers."
        : "Nothing scored above the threshold, so it declines rather than returning the closest topical match." },
    { id: "answer", title: scene.confident ? "The answer" : "No answer",
      text: scene.confident
        ? "The surviving passages expand to their surrounding context and are returned verbatim, each with a citation."
        : "The closest matches are shown as rejected candidates, not as an answer." },
  ];
}
