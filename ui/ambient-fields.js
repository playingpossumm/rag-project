/* Ambient generative fields for the Retrieval Inspector.
 *
 * Four candidates, kept in one file so the chooser page (/ambient) and the
 * inspector itself cannot drift apart -- there is one implementation of each
 * field, not a copy in each page.
 *
 * House rules, applied to all four:
 *
 *   Seeded and deterministic.  Same seed, same field, every reload. A visual
 *   you cannot reproduce is a visual you cannot compare, and the whole point
 *   of the chooser page is comparison.
 *
 *   Bound to something true.  Each field takes its structure from the actual
 *   pipeline -- 20 basins because the corpus has 20 documents, five columns
 *   because retrieval has five stages, two wave sources because fusion
 *   reconciles two retrievers. None of them is a particle system wearing a
 *   lab coat, and none claims a precision it does not have (the drifting
 *   marks are marks, not chunks).
 *
 *   Themed, never coloured.  Every field reads its ink from CSS custom
 *   properties at start, so light/dark and the validated series palette stay
 *   the single source of colour. Nothing here introduces a hue.
 *
 *   Quiet by construction.  These sit behind content. Opacity ceilings live
 *   in the field, not in the caller, so a field cannot be dropped somewhere
 *   and shout.
 *
 *   Cheap to stop.  Each returns a handle with stop(); fields pause when
 *   scrolled out of view and render a single static frame under
 *   prefers-reduced-motion.
 */
"use strict";

/* ---------------------------------------------------------------- random
   Mulberry32: small, fast, and good enough that clusters look organic rather
   than gridded. Math.random would make every reload a different artwork. */
function rng(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* Value noise on a seeded lattice, smoothstep-interpolated. Perlin's gradient
   noise would be smoother; value noise is fewer lines and the difference is
   invisible at the opacities these fields run at. */
function noiseField(seed) {
  const r = rng(seed);
  const SIZE = 256;
  const g = new Float32Array(SIZE * SIZE);
  for (let i = 0; i < g.length; i++) g[i] = r();
  const at = (x, y) => g[(y & 255) * SIZE + (x & 255)];
  const fade = t => t * t * (3 - 2 * t);
  return function (x, y) {
    const xi = Math.floor(x), yi = Math.floor(y);
    const xf = fade(x - xi), yf = fade(y - yi);
    const a = at(xi, yi), b = at(xi + 1, yi);
    const c = at(xi, yi + 1), d = at(xi + 1, yi + 1);
    return (a + (b - a) * xf) + ((c + (d - c) * xf) - (a + (b - a) * xf)) * yf;
  };
}

/* ------------------------------------------------------------- plumbing */
const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

/* How loud a field is allowed to be, as a multiplier on every alpha it sets.
   0.6 rather than 1 because that is what was chosen after seeing all four at
   full strength side by side: present enough to notice, quiet enough to read
   dense tables over. It is the default rather than a caller's argument so the
   chooser page and the Inspector cannot show different things. */
const DEFAULT_INTENSITY = 0.6;
const dial = opts => opts.intensity ?? DEFAULT_INTENSITY;

function readInk(el) {
  const cs = getComputedStyle(el);
  const v = n => cs.getPropertyValue(n).trim();
  return {
    ink: v("--ink"), faint: v("--faint"), line: v("--line"),
    s1: v("--series-1"), s2: v("--series-2"), s3: v("--series-3"),
    panel: v("--panel"), ground: v("--ground"),
  };
}

/* One canvas lifecycle for all four fields: size to the element at device
   resolution, run a fixed-step loop, stop when off-screen or detached. */
function mount(canvas, build) {
  const ctx = canvas.getContext("2d", { alpha: true });
  let raf = 0, running = false, t = 0, field = null, W = 0, H = 0;

  function resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const r = canvas.getBoundingClientRect();
    W = Math.max(1, Math.round(r.width));
    H = Math.max(1, Math.round(r.height));
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    field = build(W, H, readInk(canvas));
    t = 0;
    // Reduced motion gets the settled frame, not a frozen first frame: the
    // interesting state of every one of these fields is the late one.
    if (reduced()) { for (let i = 0; i < 240; i++) field.step(1 / 60, i / 60); }
    ctx.clearRect(0, 0, W, H);
    field.draw(ctx, W, H, t);
  }

  function frame() {
    if (!running) return;
    t += 1 / 60;
    field.step(1 / 60, t);
    field.draw(ctx, W, H, t);
    raf = requestAnimationFrame(frame);
  }

  function play() {
    if (running || reduced() || !field) return;
    running = true; raf = requestAnimationFrame(frame);
  }
  function pause() { running = false; cancelAnimationFrame(raf); }

  const ro = new ResizeObserver(resize);
  ro.observe(canvas);
  const io = new IntersectionObserver(es => { es[0].isIntersecting ? play() : pause(); }, { threshold: 0 });
  io.observe(canvas);
  resize(); play();

  return {
    stop() { pause(); ro.disconnect(); io.disconnect(); },
    restart: resize,
  };
}

/* ============================================================== A. BASIN
   Marks drift on a noise field and are drawn toward twenty basins -- one per
   document in the corpus. Read: an embedding space settling into the
   documents it is made of. The marks are marks; they do not stand for
   individual chunks and are not counted as such. */
function basin(canvas, opts = {}) {
  const K = dial(opts);
  const seed = opts.seed || 7;
  const BASINS = opts.basins || 20;   // documents in the corpus
  return mount(canvas, (W, H, ink) => {
    // Density, not count, is what the eye reads. A fixed 520 marks looked
    // right in a 210px panel and collapsed to a scribble in a 104px band.
    const N = opts.marks || Math.max(260, Math.min(900, Math.round((W * H) / 210)));
    const r = rng(seed), n = noiseField(seed);
    const wells = Array.from({ length: BASINS }, () => ({
      x: r() * W, y: r() * H, pull: 5 + r() * 11,
    }));
    const p = Array.from({ length: N }, () => ({
      x: r() * W, y: r() * H, vx: 0, vy: 0, w: r(),
    }));
    return {
      step(dt, time) {
        for (const q of p) {
          // Curl-ish flow: sample the scalar field either side and rotate the
          // gradient, so marks stream along contours instead of piling into
          // one sink and stopping.
          const s = 0.006;
          const a = n(q.x * s, q.y * s + time * 0.03) * Math.PI * 4;
          let fx = Math.cos(a) * 14, fy = Math.sin(a) * 14;
          for (const w of wells) {
            const dx = w.x - q.x, dy = w.y - q.y;
            const d2 = dx * dx + dy * dy + 900;
            fx += (dx / d2) * w.pull * 260;
            fy += (dy / d2) * w.pull * 260;
          }
          q.vx = q.vx * 0.94 + fx * dt;
          q.vy = q.vy * 0.94 + fy * dt;
          q.x += q.vx * dt; q.y += q.vy * dt;
          if (q.x < -20) q.x = W + 20; if (q.x > W + 20) q.x = -20;
          if (q.y < -20) q.y = H + 20; if (q.y > H + 20) q.y = -20;
        }
      },
      draw(ctx, W, H) {
        // Fade rather than clear: the trail IS the drawing, and a hard clear
        // would leave 520 unrelated dots.
        ctx.globalCompositeOperation = "destination-out";
        ctx.fillStyle = "rgba(0,0,0,0.055)";
        ctx.fillRect(0, 0, W, H);
        ctx.globalCompositeOperation = "source-over";
        for (const q of p) {
          ctx.fillStyle = q.w > 0.94 ? ink.s1 : ink.faint;
          ctx.globalAlpha = K * (q.w > 0.94 ? 0.30 : 0.10 + q.w * 0.09);
          ctx.fillRect(q.x, q.y, 1.4, 1.4);
        }
        ctx.globalAlpha = 1;
      },
    };
  });
}

/* ============================================================ B. LATTICE
   Rank rows and stage columns -- the bump chart with the data taken out.
   Marks advance stage by stage and change rank as they go. The quietest of
   the four, and the only one that reads as the same instrument as the page
   it sits on. */
function lattice(canvas, opts = {}) {
  const K = dial(opts);
  const seed = opts.seed || 7;
  const COLS = opts.cols || 5;        // stages in the pipeline
  const ROWS = opts.rows || 8;

  return mount(canvas, (W, H, ink) => {
    const r = rng(seed);
    const rowY = i => (H * (i + 0.5)) / ROWS;
    const colX = i => (W * (i + 0.5)) / COLS;
    const N = Math.max(5, Math.round(W / 110));
    const travellers = Array.from({ length: N }, (_, i) => ({
      t: r() * COLS, speed: 0.10 + r() * 0.12,
      row: Math.floor(r() * ROWS), next: Math.floor(r() * ROWS),
      lit: i === 0,
    }));
    return {
      step(dt) {
        for (const m of travellers) {
          const was = Math.floor(m.t);
          m.t += m.speed * dt;
          if (m.t >= COLS) { m.t -= COLS; m.row = m.next; }
          if (Math.floor(m.t) !== was) {
            m.row = m.next;
            m.next = Math.max(0, Math.min(ROWS - 1, m.row + (r() < 0.5 ? -1 : 1) * (1 + Math.floor(r() * 2))));
          }
        }
      },
      draw(ctx, W, H) {
        ctx.clearRect(0, 0, W, H);
        ctx.strokeStyle = ink.line; ctx.lineWidth = 1; ctx.globalAlpha = K * (0.5);
        for (let i = 0; i < ROWS; i++) {
          const y = Math.round(rowY(i)) + 0.5;
          ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
        }
        ctx.globalAlpha = K * (0.30);
        for (let i = 0; i < COLS; i++) {
          const x = Math.round(colX(i)) + 0.5;
          ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
        }
        for (const m of travellers) {
          const i = Math.floor(m.t), f = m.t - i;
          const x0 = colX(i), x1 = colX((i + 1) % COLS) + (i + 1 >= COLS ? W : 0);
          const y0 = rowY(m.row), y1 = rowY(m.next);
          const e = f * f * (3 - 2 * f);
          const x = x0 + (x1 - x0) * f, y = y0 + (y1 - y0) * e;
          ctx.strokeStyle = m.lit ? ink.s1 : ink.faint;
          ctx.globalAlpha = K * (m.lit ? 0.55 : 0.24);
          ctx.lineWidth = m.lit ? 1.6 : 1.1;
          ctx.beginPath();
          ctx.moveTo(x0, y0);
          ctx.bezierCurveTo(x0 + (x1 - x0) * 0.42, y0, x - (x1 - x0) * 0.42, y, x, y);
          ctx.stroke();
          ctx.fillStyle = m.lit ? ink.s1 : ink.faint;
          ctx.globalAlpha = K * (m.lit ? 0.8 : 0.35);
          ctx.beginPath(); ctx.arc(x, y, m.lit ? 2.4 : 1.7, 0, 6.284); ctx.fill();
        }
        ctx.globalAlpha = 1;
      },
    };
  });
}

/* ============================================================= C. FRINGE
   Two wave sources, one per retriever, and the interference between them.
   Where the crests coincide the field brightens -- which is exactly what
   reciprocal rank fusion rewards: the passages both retrievers found. The
   only field of the four that depicts the idea the pipeline is built on.

   Rendered into a coarse ImageData grid and scaled up: a per-pixel field at
   full resolution costs an order of magnitude more for a blur nobody sees. */
function fringe(canvas, opts = {}) {
  const K = dial(opts);
  const seed = opts.seed || 7;
  const CELL = opts.cell || 5;

  return mount(canvas, (W, H, ink) => {
    const r = rng(seed);
    const gw = Math.max(2, Math.ceil(W / CELL)), gh = Math.max(2, Math.ceil(H / CELL));
    const img = new ImageData(gw, gh);
    const rgb = hex => {
      const h = hex.replace("#", "");
      const n = h.length === 3 ? h.split("").map(c => c + c).join("") : h;
      return [parseInt(n.slice(0, 2), 16), parseInt(n.slice(2, 4), 16), parseInt(n.slice(4, 6), 16)];
    };
    const A = rgb(ink.s1 || "#2a78d6"), B = rgb(ink.s2 || "#eb6834");
    // Sources drift slowly on their own ellipses so the fringe pattern never
    // repeats within a session, but starts identically for a given seed.
    const src = [
      { ax: 0.22 + r() * 0.06, ay: 0.34, rx: 0.06, ry: 0.10, ph: r() * 6.28, sp: 0.09 },
      { ax: 0.74 - r() * 0.06, ay: 0.62, rx: 0.07, ry: 0.09, ph: r() * 6.28, sp: 0.07 },
    ];
    let time = 0;
    const buf = new Float32Array(gw * gh);
    return {
      step(dt) { time += dt; },
      draw(ctx, W, H) {
        const s0x = (src[0].ax + Math.cos(time * src[0].sp + src[0].ph) * src[0].rx) * gw;
        const s0y = (src[0].ay + Math.sin(time * src[0].sp + src[0].ph) * src[0].ry) * gh;
        const s1x = (src[1].ax + Math.cos(time * src[1].sp + src[1].ph) * src[1].rx) * gw;
        const s1y = (src[1].ay + Math.sin(time * src[1].sp + src[1].ph) * src[1].ry) * gh;
        const k = 0.34, w = 1.5;
        const d = img.data;
        for (let y = 0; y < gh; y++) {
          for (let x = 0; x < gw; x++) {
            const i = y * gw + x;
            const d0 = Math.hypot(x - s0x, y - s0y);
            const d1 = Math.hypot(x - s1x, y - s1y);
            const a = Math.sin(d0 * k - time * w) / (1 + d0 * 0.045);
            const b = Math.sin(d1 * k - time * w) / (1 + d1 * 0.045);
            buf[i] = a + b;
            // Amplitude of the sum, not of either wave: agreement is the
            // signal. Squared so the bright fringes stay narrow.
            const amp = Math.min(1, Math.abs(buf[i]) * 0.62);
            const lean = d1 / (d0 + d1 + 1e-6);          // nearer source wins the hue
            const e = amp * amp;
            const o = i * 4;
            d[o]     = A[0] * lean + B[0] * (1 - lean);
            d[o + 1] = A[1] * lean + B[1] * (1 - lean);
            d[o + 2] = A[2] * lean + B[2] * (1 - lean);
            d[o + 3] = Math.round(e * 150 * K);
          }
        }
        ctx.clearRect(0, 0, W, H);
        ctx.putImageData(img, 0, 0);
        // Scale the coarse grid up through the smoothing filter, which is the
        // whole reason the grid can be coarse.
        ctx.imageSmoothingEnabled = true;
        ctx.globalAlpha = 1;
        ctx.drawImage(canvas, 0, 0, gw, gh, 0, 0, W, H);
      },
    };
  });
}

/* ============================================================= D. SETTLE
   Marks fall and stack into ranked rows, then stop. The only field with an
   end state, which is its argument: it performs the sort once on load and
   then gets out of the way, rather than moving forever behind text. */
function settle(canvas, opts = {}) {
  const K = dial(opts);
  const seed = opts.seed || 7;
  const ROWS = opts.rows || 9;

  return mount(canvas, (W, H, ink) => {
    const r = rng(seed);
    const N = Math.max(40, Math.round(W / 6));
    const rowY = i => (H * (i + 0.6)) / ROWS;
    const p = Array.from({ length: N }, (_, i) => ({
      x: (i / N) * W + r() * 6,
      y: -r() * H * 1.6,
      target: rowY(Math.floor(Math.pow(r(), 1.7) * ROWS)),
      delay: r() * 2.2,
      v: 0,
      lit: r() > 0.93,
      rest: false,
    }));
    return {
      step(dt, time) {
        for (const q of p) {
          if (q.rest || time < q.delay) continue;
          q.v += 520 * dt;
          q.y += q.v * dt;
          if (q.y >= q.target) {
            q.y = q.target;
            q.v *= -0.28;                 // one small bounce, then done
            if (Math.abs(q.v) < 18) { q.v = 0; q.rest = true; }
          }
        }
      },
      draw(ctx, W, H) {
        ctx.clearRect(0, 0, W, H);
        ctx.strokeStyle = ink.line; ctx.globalAlpha = K * (0.55); ctx.lineWidth = 1;
        for (let i = 0; i < ROWS; i++) {
          const y = Math.round(rowY(i)) + 0.5;
          ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
        }
        for (const q of p) {
          if (q.y < -4) continue;
          ctx.fillStyle = q.lit ? ink.s1 : ink.faint;
          ctx.globalAlpha = K * (q.lit ? 0.75 : (q.rest ? 0.38 : 0.22));
          ctx.fillRect(q.x, q.y - 1.25, 4, 2.5);
        }
        ctx.globalAlpha = 1;
      },
    };
  });
}

const AMBIENT = { basin, lattice, fringe, settle };
if (typeof window !== "undefined") window.AMBIENT = AMBIENT;
