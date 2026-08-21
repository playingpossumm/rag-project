/* The pipeline as a city.
 *
 * The governing rule, and the one earlier drafts broke: THE CITY IS ALWAYS
 * THERE. Every block, every tower and every road is drawn before you ask
 * anything and is still drawn after. A query changes lighting and nothing else.
 *
 * That is not decoration, it is the honest shape of the thing. Roads that grow
 * as a result arrives say the system is building a path; roads that were always
 * there and light up say every chunk was reachable and this query took that
 * route. The second is what retrieval does. An earlier draft animated lines
 * into existence and it read as popping -- the structure appeared to be a
 * consequence of the question, which is backwards.
 *
 * The layout, left to right:
 *
 *   THE INDEX      one block per document, sized by its real chunk count,
 *                  every chunk a window. Nothing is placed meaningfully within
 *                  a block; it is a city block, not a scatter plot.
 *   DENSE / BM25   two towers fed by two roads off the index, because the two
 *                  retrievers run at the same moment. The old chart drew them
 *                  as column 1 and column 2, which reads as "first this, then
 *                  that" and is wrong about the mechanism.
 *   FUSION         where the two roads meet.
 *   RERANKING      the expensive reader.
 *   CAP            at most two passages per document.
 *   GATE           open, or barred.
 *   ANSWER         the passages that survived, expanded to context.
 *
 * Every tower is a stack of ten floors and a floor is a rank: floor one at the
 * top, as in any ranking table. A floor lights in its document's colour when a
 * passage occupies it, so the reranker visibly reshuffles the building. That is
 * the bump chart's logic, kept, with the stages given somewhere to stand.
 *
 * Stages advance along (+gx, -gy), which in isometric is pure horizontal
 * movement -- so the city reads left to right like a figure in a paper while
 * every plate still draws as a diamond. Blueprint throughout: hairlines, no
 * solid masses, because this sits above dense tables and must not shout.
 */
"use strict";

/* ---------------------------------------------------------- projection */
const TW = 9.4;
const TH = 5.6;
const SZ = 26;                       // px per floor of height

const project = (gx, gy, z = 0) => ({
  x: (gx - gy) * TW,
  y: (gx + gy) * TH - z * SZ,
});

/** `u` runs along the pipeline (horizontal on screen), `v` across it. */
const at = (u, v) => ({ gx: u + v, gy: -u + v });

export const shortDoc = s => String(s).replace(/\.(pdf|docx|pptx|xlsx)$/i, "");

function palette(finalSources, ink) {
  const slots = [ink.s1, ink.s2, ink.s3];
  const m = new Map();
  finalSources.forEach((s, i) => m.set(s, i < 3 ? slots[i] : ink.other));
  return src => m.get(src) || ink.other;
}

/* ----------------------------------------------------------- the plan */
const FLOORS = 10;                   // ranks a tower can show
const TOWERS = [
  { id: "dense",    u: 34, v: -19, label: "Dense",         sub: "meaning" },
  { id: "sparse",   u: 34, v: 19,  label: "BM25",          sub: "exact words" },
  { id: "fused",    u: 50, v: 0,   label: "Fusion",        sub: "merged by rank" },
  { id: "reranked", u: 63, v: 0,   label: "Reranking",     sub: "read together" },
  { id: "selected", u: 75, v: 0,   label: "Diversity cap", sub: "2 per document" },
];
const GATE = { id: "gate", u: 86, v: 0, label: "The gate" };
const ANSWER = { id: "answer", u: 97, v: 0, label: "The answer" };
const INDEX = { id: "index", u: 0, v: 0, label: "The index" };

const TOWER_W = 3;                 // half-width of a tower's footprint
const ROADS = [
  ["index", "dense"], ["index", "sparse"],
  ["dense", "fused"], ["sparse", "fused"],
  ["fused", "reranked"], ["reranked", "selected"],
  ["selected", "gate"], ["gate", "answer"],
];

/* ------------------------------------------------------------ the city
   Built from the corpus alone. It does not know what you asked, and does not
   change when you ask it -- which is the whole point. */
export function buildCity(chunks) {
  const sources = chunks?.sources || [];
  const doc = chunks?.doc || [];

  // Real counts per document. An even split looked right and was not.
  const counts = sources.map(() => 0);
  for (const d of doc) counts[d] += 1;

  const cols = 4;
  const rows = Math.max(1, Math.ceil(sources.length / cols));
  const CELL = 12, GAP = 2.6;

  // One block per document; every chunk a window, laid out in rows inside it.
  const blocks = sources.map((src, i) => {
    const cx = (i % cols) - (cols - 1) / 2;
    const cy = Math.floor(i / cols) - (rows - 1) / 2;
    const centre = at(INDEX.u + cx * (CELL + GAP) * 0.5, cy * (CELL + GAP) * 0.5);
    const n = counts[i] || 0;
    const per = Math.ceil(Math.sqrt(n)) || 1;
    const windows = [];
    for (let k = 0; k < n; k++) {
      const r = Math.floor(k / per), c = k % per;
      windows.push({
        gx: centre.gx + (c / Math.max(per - 1, 1) - 0.5) * CELL * 0.82,
        gy: centre.gy + (r / Math.max(per - 1, 1) - 0.5) * CELL * 0.82,
      });
    }
    return { src, i, n, centre, half: CELL / 2, windows };
  });

  // A chunk id is a row position, so the id order walks documents in order.
  const chunkAt = [];
  {
    const cursor = sources.map(() => 0);
    for (let id = 0; id < doc.length; id++) {
      const b = blocks[doc[id]];
      chunkAt[id] = b?.windows[cursor[doc[id]]++] || null;
    }
  }

  const towers = TOWERS.map(t => ({ ...t, ...at(t.u, t.v) }));
  const gate = { ...GATE, ...at(GATE.u, GATE.v) };
  const answer = { ...ANSWER, ...at(ANSWER.u, ANSWER.v) };
  const index = { ...INDEX, ...at(INDEX.u, INDEX.v) };
  const byId = new Map([...towers, gate, answer, index].map(t => [t.id, t]));

  // Road endpoints, resolved once. The index end of a road leaves from its
  // near edge rather than its centre, so a road never crosses the blocks.
  const roads = ROADS.map(([a, b]) => {
    const A = byId.get(a), B = byId.get(b);
    const from = a === "index"
      ? at(INDEX.u + (cols * (CELL + GAP)) / 4 + 3, 0)
      : { gx: A.gx, gy: A.gy };
    return { a, b, from, to: { gx: B.gx, gy: B.gy } };
  });

  return { sources, counts, blocks, chunkAt, towers, gate, answer, index, byId, roads,
           total: doc.length };
}

/* -------------------------------------------------------------- a run
   What one query lights. Positions come from the city; nothing here moves a
   building. */
export function buildRun(trace, city, ink) {
  if (!trace || !city) return null;
  const stages = new Map(trace.stages.map(s => [s.name, s]));
  const final = stages.get("selected")?.items || [];
  const order = [...new Set(final.map(i => i.source))];
  const colour = palette(order, ink);
  const finalIds = new Set(final.map(i => i.chunk_id));
  /* Tri-state, not boolean. With reranking off there is no cross-encoder score
     to test against a threshold calibrated on cross-encoder scores, so the
     server returns null rather than a verdict it cannot justify -- and `!!null`
     would have drawn that as a refusal, inventing a decision nobody made. */
  const confident = trace.verdict?.confident ?? null;
  const opts = trace.options || {};

  // Which windows in the index light, and for which reader.
  const denseIds = (stages.get("dense")?.items || []).map(i => i.chunk_id);
  const sparseIds = (stages.get("sparse")?.items || []).map(i => i.chunk_id);
  const litWindows = [];
  const seen = new Map();
  for (const [ids, kind] of [[denseIds, "dense"], [sparseIds, "sparse"]]) {
    for (const id of ids) {
      const w = city.chunkAt[id];
      if (!w) continue;
      const prev = seen.get(id);
      if (prev) { prev.both = true; continue; }
      const rec = { id, w, kind, both: false,
                    colour: finalIds.has(id) ? colour(trace.stages[0].items
                      .concat(trace.stages[1].items).find(x => x.chunk_id === id)?.source) : ink.other };
      seen.set(id, rec);
      litWindows.push(rec);
    }
  }

  // Floors: rank k of a stage occupies floor k of that tower.
  const floors = new Map();
  for (const t of city.towers) {
    const st = stages.get(t.id);
    if (!st) continue;
    floors.set(t.id, st.items.slice(0, FLOORS).map((it, k) => ({
      ...it, floor: k,
      colour: finalIds.has(it.chunk_id) ? colour(it.source) : ink.other,
      survives: finalIds.has(it.chunk_id),
    })));
  }

  // Traffic: which passages travel which road, so the payload follows the
  // network rather than flying between arbitrary points.
  const traffic = city.roads.map(r => {
    const to = floors.get(r.b) || [];
    const ids = new Set(to.filter(m => m.survives).map(m => m.chunk_id));
    const colours = to.filter(m => m.survives).map(m => m.colour);
    return { ...r, n: Math.max(ids.size, r.b === "gate" || r.b === "answer" ? final.length : 0), colours };
  });

  // Taken from the trace, not inferred. The old inference counted candidates in
  // the reranked top 6 that did not survive, which at k=5 is 1 no matter what
  // the cap did -- so the map read "1 cut" beside a stage note reading "nothing
  // displaced", and one of the two had to be wrong.
  const displaced = trace.selection?.displaced ?? 0;

  return {
    colour, order, confident, floors, litWindows, traffic, opts,
    verdict: trace.verdict, query: trace.query,
    counts: {
      total: city.total,
      dense: denseIds.length, sparse: sparseIds.length,
      fused: stages.get("fused")?.items.length || 0,
      reranked: stages.get("reranked")?.items.length || 0,
      selected: final.length,
      both: (stages.get("fused")?.items || []).filter(i => i.agreement).length,
      onlySparse: sparseIds.filter(id => !denseIds.includes(id)).length,
      displaced,
    },
  };
}

/* ---------------------------------------------------------------- draw */
function diamond(ctx, gx, gy, w, d, z = 0) {
  const c = [
    project(gx - w, gy - d, z), project(gx + w, gy - d, z),
    project(gx + w, gy + d, z), project(gx - w, gy + d, z),
  ];
  ctx.beginPath();
  ctx.moveTo(c[0].x, c[0].y);
  for (let i = 1; i < 4; i++) ctx.lineTo(c[i].x, c[i].y);
  ctx.closePath();
  return c;
}

/** A tower: FLOORS storeys of hairline, always drawn. */
function drawTower(ctx, t, ink, lit) {
  for (let f = 0; f < FLOORS; f++) {
    const z = (FLOORS - 1 - f);              // floor 0 (rank 1) on top
    const m = lit && lit[f];
    diamond(ctx, t.gx, t.gy, TOWER_W, TOWER_W, z);
    if (m) {
      ctx.globalAlpha = m.survives ? 0.34 : 0.13;
      ctx.fillStyle = m.colour; ctx.fill();
      ctx.globalAlpha = 1;
      ctx.strokeStyle = m.colour; ctx.lineWidth = m.survives ? 1.4 : 1;
    } else {
      ctx.strokeStyle = ink.rule || ink.line; ctx.lineWidth = 1;
    }
    ctx.stroke();
  }
  // Corner posts, so the storeys read as one building.
  const c0 = [
    project(t.gx - TOWER_W, t.gy - TOWER_W, 0), project(t.gx + TOWER_W, t.gy - TOWER_W, 0),
    project(t.gx + TOWER_W, t.gy + TOWER_W, 0), project(t.gx - TOWER_W, t.gy + TOWER_W, 0),
  ];
  const cT = [
    project(t.gx - TOWER_W, t.gy - TOWER_W, FLOORS - 1), project(t.gx + TOWER_W, t.gy - TOWER_W, FLOORS - 1),
    project(t.gx + TOWER_W, t.gy + TOWER_W, FLOORS - 1), project(t.gx - TOWER_W, t.gy + TOWER_W, FLOORS - 1),
  ];
  ctx.strokeStyle = ink.muted; ctx.globalAlpha = .5; ctx.lineWidth = 1;
  for (const i of [1, 2, 3]) {
    ctx.beginPath();
    ctx.moveTo(c0[i].x, c0[i].y); ctx.lineTo(cT[i].x, cT[i].y); ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

function drawGate(ctx, g, ink, run) {
  const W = 4.6;
  const L = project(g.gx - W, g.gy + W), R = project(g.gx + W, g.gy - W);
  const Lt = project(g.gx - W, g.gy + W, 4), Rt = project(g.gx + W, g.gy - W, 4);
  ctx.strokeStyle = ink.line; ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(L.x, L.y); ctx.lineTo(Lt.x, Lt.y);
  ctx.moveTo(R.x, R.y); ctx.lineTo(Rt.x, Rt.y);
  ctx.stroke();
  if (!run) {                                  // at rest: the bar simply waits
    ctx.strokeStyle = ink.line; ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.moveTo(Lt.x, Lt.y); ctx.lineTo(Rt.x, Rt.y); ctx.stroke();
    return;
  }
  // Open in green, barred in red, and open in grey when the gate did not run:
  // an ungated pass is not an approval and must not be coloured like one.
  ctx.strokeStyle = run.confident === true ? ink.good
                  : run.confident === false ? ink.critical : ink.faint;
  ctx.lineWidth = run.confident === null ? 1.6 : 2.6;
  ctx.beginPath();
  if (run.confident !== false) { ctx.moveTo(Lt.x, Lt.y); ctx.lineTo(Rt.x, Rt.y); }
  else for (const z of [1.4, 2.4]) {
    const l = project(g.gx - W, g.gy + W, z), r = project(g.gx + W, g.gy - W, z);
    ctx.moveTo(l.x, l.y); ctx.lineTo(r.x, r.y);
  }
  ctx.stroke();
}

/**
 * Draw the city, and optionally light a run through it.
 *
 * `progress` is 0..1 across the route. At 0 the city is simply itself.
 */
export function drawScene(ctx, city, run, ink, W, H, progress = 1) {
  ctx.clearRect(0, 0, W, H);
  if (!city) return null;

  const xs = [], ys = [];
  const note = (gx, gy, z) => { const q = project(gx, gy, z); xs.push(q.x); ys.push(q.y); };
  // All four corners, not two. In this projection x = (gx - gy) * TW, so the
  // corners (gx-w, gy-w) and (gx+w, gy+w) share the block's centre x -- they
  // are its top and bottom points. Sampling only those made every block appear
  // to have zero width, and the index district clipped off the left edge.
  for (const b of city.blocks) {
    for (const [sx, sy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]])
      note(b.centre.gx + sx * b.half, b.centre.gy + sy * b.half, 0);
  }
  for (const t of [...city.towers, city.gate, city.answer]) {
    note(t.gx - 5, t.gy - 5, 0); note(t.gx + 5, t.gy + 5, FLOORS);
  }
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const PAD = { l: 16, r: 104, t: 38, b: 22 };
  const scale = Math.min((W - PAD.l - PAD.r) / (maxX - minX), (H - PAD.t - PAD.b) / (maxY - minY));
  const ox = W / 2 + (PAD.l - PAD.r) / 2, oy = H / 2 + (PAD.t - PAD.b) / 2;

  ctx.save();
  ctx.translate(ox, oy);
  ctx.scale(scale, scale);
  ctx.translate(-(minX + maxX) / 2, -(minY + maxY) / 2);
  ctx.lineJoin = "round";

  const LEGS = ["dense", "fused", "reranked", "selected", "gate", "answer"];
  const legPhase = id => Math.max(0, Math.min(1, progress * LEGS.length - LEGS.indexOf(id)));

  /* 1 — roads. Always drawn, at rest and in use. */
  for (const r of city.roads) {
    const a = project(r.from.gx, r.from.gy, 0.2), b = project(r.to.gx, r.to.gy, 0.2);
    const len = Math.hypot(b.x - a.x, b.y - a.y);
    const n = Math.max(8, Math.round(len / 4));
    ctx.fillStyle = ink.muted; ctx.globalAlpha = 0.55;
    for (let i = 0; i <= n; i++) {
      const t = i / n;
      ctx.fillRect(a.x + (b.x - a.x) * t - 0.6, a.y + (b.y - a.y) * t - 0.6, 1.2, 1.2);
    }
  }
  ctx.globalAlpha = 1;

  /* 2 — the index: one block per document, every chunk a window. */
  for (const b of city.blocks) {
    diamond(ctx, b.centre.gx, b.centre.gy, b.half, b.half, 0);
    ctx.fillStyle = ink.panel; ctx.globalAlpha = 1; ctx.fill();
    ctx.strokeStyle = ink.rule || ink.line; ctx.lineWidth = 1.1; ctx.stroke();
    ctx.fillStyle = ink.muted; ctx.globalAlpha = 0.85;
    for (const w of b.windows) {
      const q = project(w.gx, w.gy, 0.12);
      ctx.fillRect(q.x, q.y, 1.8, 1.8);
    }
    ctx.globalAlpha = 1;
  }

  /* 3 — lighting: the windows this query woke up. */
  if (run) {
    const lp = legPhase("dense");
    for (const l of run.litWindows) {
      const q = project(l.w.gx, l.w.gy, 0.12);
      ctx.globalAlpha = lp;
      if (l.kind === "dense" || l.both) {
        ctx.fillStyle = l.colour; ctx.fillRect(q.x - 1.6, q.y - 1.6, 4.4, 4.4);
      }
      if (l.kind === "sparse" || l.both) {
        ctx.strokeStyle = l.colour; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.arc(q.x + 0.7, q.y + 0.7, 4.4, 0, 6.284); ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
  }

  /* 4 — traffic on the roads. */
  if (run) {
    for (const r of run.traffic) {
      const ph = legPhase(r.b === "index" ? "dense" : r.b);
      if (ph <= 0 || !r.colours.length) continue;
      const a = project(r.from.gx, r.from.gy, 0.2), b = project(r.to.gx, r.to.gy, 0.2);
      // The lit carriageway: the same dotted road, brighter, as far as the
      // traffic has travelled.
      const len = Math.hypot(b.x - a.x, b.y - a.y);
      const n = Math.max(8, Math.round(len / 4));
      ctx.fillStyle = r.colours[0]; ctx.globalAlpha = 0.55;
      for (let i = 0; i <= n * ph; i++) {
        const t = i / n;
        ctx.fillRect(a.x + (b.x - a.x) * t - 0.8, a.y + (b.y - a.y) * t - 0.8, 1.6, 1.6);
      }
      ctx.globalAlpha = 1;
      r.colours.forEach((col, i) => {
        const t = Math.max(0, Math.min(1, ph - i * 0.05));
        const q = { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
        ctx.fillStyle = col;
        ctx.beginPath(); ctx.arc(q.x, q.y, 2.6, 0, 6.284); ctx.fill();
      });
    }
  }

  /* 5 — towers, back to front. */
  for (const t of city.towers) {
    const ph = run ? legPhase(t.id === "sparse" ? "dense" : t.id) : 0;
    drawTower(ctx, t, ink, run && ph > 0.35 ? run.floors.get(t.id) : null);
  }

  /* 6 — the gate, and the answer it permits. */
  drawGate(ctx, city.gate, ink, run && legPhase("gate") > 0.4 ? run : null);
  const A = city.answer;
  diamond(ctx, A.gx, A.gy, 5, 6, 0);
  ctx.fillStyle = ink.panel; ctx.globalAlpha = 0.5; ctx.fill(); ctx.globalAlpha = 1;
  ctx.strokeStyle = ink.line; ctx.lineWidth = 1; ctx.stroke();
  if (run && run.confident !== false && legPhase("answer") > 0.4) {
    const sel = run.floors.get("selected") || [];
    sel.forEach((m, k) => {
      const q = project(A.gx, A.gy - 4 + k * 1.7, 0.3);
      ctx.fillStyle = m.colour; ctx.globalAlpha = legPhase("answer");
      ctx.fillRect(q.x - 9, q.y - 1.5, 18, 3);
      ctx.globalAlpha = 1;
    });
  }

  ctx.restore();

  /* ---- type, in screen space ------------------------------------------ */
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  const toScreen = q => ({ x: ox + (q.x - cx) * scale, y: oy + (q.y - cy) * scale });
  const c = run?.counts;
  /* Canvas has no letter-spacing, so a tracked micro-label is drawn one glyph
     at a time. Worth the loop: the tracking IS the device. */
  const tracked = (text, x, y, px) => {
    const gap = px * 0.16;
    let w = 0;
    for (const ch of text) w += ctx.measureText(ch).width + gap;
    let cx = x - (w - gap) / 2;
    for (const ch of text) {
      ctx.fillText(ch, cx + ctx.measureText(ch).width / 2, y);
      cx += ctx.measureText(ch).width + gap;
    }
  };

  const label = (anchor, title, sub, colourOverride, alpha = 1) => {
    ctx.globalAlpha = alpha;
    ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
    ctx.fillStyle = colourOverride || ink.ink;
    ctx.font = '600 12px Inter, system-ui, sans-serif';
    ctx.fillText(title, anchor.x, anchor.y);
    if (sub) {
      ctx.fillStyle = ink.faint;
      ctx.font = '500 9px "DM Mono", ui-monospace, Consolas, monospace';
      tracked(sub.toUpperCase(), anchor.x, anchor.y + 13, 9);
    }
    ctx.globalAlpha = 1;
  };

  const topOf = t => toScreen(project(t.gx, t.gy - TOWER_W, FLOORS + 0.6));
  label(toScreen(project(city.index.gx, city.index.gy - 20, 0)), "The index",
        `${city.total.toLocaleString()} chunks · ${city.sources.length} documents`);
  /* Names come from the run once there is one. A tower still titled "Diversity
     cap" over a run that applied no cap is the map asserting something that did
     not happen -- the same failure as a stale README, drawn instead of typed. */
  const o = run?.opts || {};
  city.towers.forEach((t, ti) => {
    const sub = !c ? t.sub
      : t.id === "dense" ? `${c.dense} found`
      : t.id === "sparse" ? (o.fusion === "none" ? "not run" : `${c.sparse} found`)
      : t.id === "fused" ? (o.fusion === "none" ? "dense only" : `${c.both} by both`)
      : t.id === "reranked" ? (o.use_reranker === false ? "off" : `${c.reranked} rescored`)
      : c.displaced ? `${c.selected} kept, ${c.displaced} cut` : `${c.selected} kept`;
    const title = !run ? t.label
      : t.id === "fused" && o.fusion === "none" ? "No fusion"
      : t.id === "fused" && o.fusion === "weighted" ? "Weighted"
      : t.id === "selected" && !(o.max_per_source && o.use_reranker !== false) ? "Selection"
      : t.label;
    const faded = run && ((t.id === "sparse" && o.fusion === "none")
                       || (t.id === "reranked" && o.use_reranker === false));
    const a = topOf(t);
    // Spine towers alternate height; the two retrievers sit off-spine already.
    const lift = (t.id === "dense" || t.id === "sparse") ? 0 : (ti % 2 ? 0 : 22);
    label({ x: a.x, y: a.y - 12 - lift }, title, sub, null, faded ? 0.45 : 1);
  });
  const g = toScreen(project(city.gate.gx, city.gate.gy - 4.6, 5));
  const gateName = !run ? "The gate"
                 : run.confident === true ? "Answering"
                 : run.confident === false ? "Declined" : "No gate";
  label({ x: g.x, y: g.y - 12 }, gateName,
        !run ? "answer or decline"
        : run.confident === null ? "no calibrated score"
        : run.verdict?.confidence != null
          ? `${run.verdict.confidence > 0 ? "+" : ""}${run.verdict.confidence} against ${run.verdict.threshold.toFixed(1)}`
          : "answer or decline",
        !run ? null : run.confident === true ? ink.good
             : run.confident === false ? ink.critical : null);
  const an = toScreen(project(A.gx, A.gy - 6, 1));
  label({ x: an.x, y: an.y - 12 }, "The answer",
        run ? (run.confident === false ? "nothing returned"
                                       : `${c.selected} cited passages`)
            : "cited passages");

  return { scale, toScreen };
}

/* Plain-language captions, using this query's real numbers.

   Every one of these is written from the run rather than from the defaults.
   That matters now that the settings re-run the pipeline: a caption reading
   "at most 2 passages per document" under a run with the cap off would be the
   page confidently narrating something that did not happen. */
export function captions(run) {
  if (!run) return [];
  const c = run.counts, o = run.opts || {};
  const cap = o.max_per_source;
  const fused = o.fusion === "none"
    ? { title: "No fusion",
        text: `With the word search off there is nothing to fuse: the candidate pool is the meaning search's ${c.dense} as they stand.` }
    : { title: o.fusion === "weighted" ? "Weighted fusion" : "Fusion",
        text: o.fusion === "weighted"
          ? `Each retriever's scores are normalised and added, weighted by alpha. ${c.both} of ${c.fused} were found by both.`
          : `The two rankings merge by position, never by score. ${c.both} of ${c.fused} were found by both — agreement is what fusion rewards.` };

  return [
    { id: "dense", title: o.fusion === "none" ? "One index, one reading" : "One index, two readings",
      text: o.fusion === "none"
        ? `All ${c.total.toLocaleString()} passages are already here. Only the search on meaning runs, and it lights ${c.dense}.`
        : `All ${c.total.toLocaleString()} passages are already here. Two searches run at the same moment — one on meaning, one on exact words — and each lights its own ${c.dense}. ${c.onlySparse} were found only by the word search.` },
    { id: "fused", ...fused },
    { id: "reranked", title: o.use_reranker === false ? "No reranking" : "Reranking",
      text: o.use_reranker === false
        ? "The cross-encoder is off, so the first stage's ranking goes straight to selection. This is the naive-RAG baseline the rest of the pipeline is measured against."
        : `A second model reads your question and each passage together, and rescores all ${c.reranked}. Watch the floors reshuffle.` },
    { id: "selected", title: cap && o.use_reranker !== false ? "Diversity cap" : "Selection",
      text: o.use_reranker === false
        ? `No cap on this path — the cap trims a reranked pool, so the top ${c.selected} of the shortlist are the result.`
        : !cap
          ? `No cap, so the top ${c.selected} are taken as ranked. One strong document can hold every slot.`
          : c.displaced
            ? `At most ${cap} passage${cap === 1 ? "" : "s"} per document, so one paper cannot take every slot. ${c.displaced} were displaced.`
            : `At most ${cap} passage${cap === 1 ? "" : "s"} per document. The cap was not reached here.` },
    { id: "gate", title: run.confident === true ? "The gate opens"
                       : run.confident === false ? "The gate stays shut" : "The gate is not applied",
      text: run.confident === true
        ? "The top passage scores above the threshold, so the system answers."
        : run.confident === false
          ? "Nothing scored above the threshold, so it declines rather than returning the closest topical match."
          : "The threshold is calibrated on cross-encoder scores, and reranking is off, so there is no comparable number to test. It passes ungated rather than pretending to a judgement." },
    { id: "answer", title: run.confident === false ? "No answer" : "The answer",
      text: run.confident === false
        ? "The closest matches are shown as rejected candidates, not as an answer."
        : o.expansion && o.expansion !== "none"
          ? `The surviving passages expand to ${o.expansion === "page" ? "their whole page" : "their neighbouring chunks"} and are returned verbatim, each with a citation.`
          : "The surviving passages are returned verbatim, as retrieved, each with a citation." },
  ];
}
