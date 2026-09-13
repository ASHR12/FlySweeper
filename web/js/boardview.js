// Minesweeper board rendering (dark theme, same look as the Python spectator page) plus optional
// mouse input for the human player: left click reveals, right click flags.
// The drawing functions (drawTile, drawCursor, makeEndFx, renderBoard) are copied verbatim into
// flysweeper/ui/index.html — keep the two in sync.

import { HIDDEN, FLAG, MINE } from './minesweeper.js';

// classic Minesweeper hues, muted for the dark theme
export const NUM_COLORS = ['', '#6f9bff', '#63b96a', '#e46a7e', '#a985d6', '#d0745f', '#4db8ad', '#c9d1d9', '#8b95a3'];
const FONT = '-apple-system, BlinkMacSystemFont, "SF Pro Text", Inter, "Segoe UI", sans-serif';
const MONO = '"SF Mono", ui-monospace, "JetBrains Mono", Menlo, monospace';

/** Draw one Minesweeper cell as a soft-bevelled tile. Shared look with flysweeper/ui/index.html. */
export function drawTile(ctx, x, y, size, kind, hover = false) {
  const r = Math.max(2, size * 0.14), inset = Math.max(1, size * 0.035);
  const x0 = x + inset, y0 = y + inset, s = size - 2 * inset;
  ctx.beginPath(); ctx.roundRect(x0, y0, s, s, r);
  if (kind === 'hidden' || kind === 'flag') {
    // raised slate: vertical gradient, light top edge, dark bottom edge
    const g = ctx.createLinearGradient(0, y0, 0, y0 + s);
    g.addColorStop(0, hover ? '#3a4758' : kind === 'flag' ? '#4a3a25' : '#2b3543');
    g.addColorStop(1, hover ? '#2e394a' : kind === 'flag' ? '#3b2d1c' : '#1f2732');
    ctx.fillStyle = g; ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.07)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.roundRect(x0 + 0.5, y0 + 0.5, s - 1, s - 1, r); ctx.stroke();
    ctx.strokeStyle = 'rgba(255,255,255,0.10)'; ctx.beginPath(); ctx.moveTo(x0 + r, y0 + 0.5); ctx.lineTo(x0 + s - r, y0 + 0.5); ctx.stroke();
    ctx.strokeStyle = 'rgba(0,0,0,0.35)'; ctx.beginPath(); ctx.moveTo(x0 + r, y0 + s - 0.5); ctx.lineTo(x0 + s - r, y0 + s - 0.5); ctx.stroke();
  } else {
    // recessed: darker, with a soft inner shadow at the top
    ctx.fillStyle = kind === 'hit' ? '#7a2434' : kind === 'mine' ? '#3a1720' : '#0c1015';
    ctx.fill();
    const g = ctx.createLinearGradient(0, y0, 0, y0 + s * 0.35);
    g.addColorStop(0, 'rgba(0,0,0,0.45)'); g.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = g; ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.035)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.roundRect(x0 + 0.5, y0 + 0.5, s - 1, s - 1, r); ctx.stroke();
  }
}

/** Gold cursor ring with a faint outer glow (red when the looming channel is active or the game is lost). */
export function drawCursor(ctx, x, y, size, red) {
  const inset = Math.max(1, size * 0.035) + 1.5, r = Math.max(2, size * 0.14);
  ctx.save();
  ctx.shadowColor = red ? 'rgba(240,100,124,0.85)' : 'rgba(242,193,92,0.75)';
  ctx.shadowBlur = Math.max(6, size * 0.28);
  ctx.strokeStyle = red ? '#f0647c' : '#f2c15c'; ctx.lineWidth = Math.max(2, size * 0.055);
  ctx.beginPath(); ctx.roundRect(x + inset, y + inset, size - 2 * inset, size - 2 * inset, r); ctx.stroke();
  ctx.restore();
}

/**
 * End-of-game effect schedule, built once when a board reports `outcome` 'lost' or 'won'.
 * lost: the hit mine bursts (~600 ms), then the other mines pop into view nearest-first (70 ms apart);
 * won:  the remaining hidden cells flip to flags (60 ms apart) under a gold border-glow pulse (~1 s).
 * Progress is time-based so the final frame is static once everything has played.
 */
export function makeEndFx(kind, visible, rows, cols, origin, now) {
  const [orow, ocol] = origin || [Math.floor(rows / 2), Math.floor(cols / 2)];
  const dist = (i) => Math.hypot(Math.floor(i / cols) - orow, (i % cols) - ocol);
  const cells = new Map();
  let total;
  if (kind === 'lost') {
    const mines = [];
    for (let i = 0; i < visible.length; i++) if (visible[i] === MINE && !(Math.floor(i / cols) === orow && i % cols === ocol)) mines.push(i);
    mines.sort((a, b) => dist(a) - dist(b));
    mines.forEach((i, k) => cells.set(i, { t0: now + 380 + k * 70, dur: 240 }));
    total = 380 + mines.length * 70 + 240 + 300;
  } else {
    const hidden = [];
    for (let i = 0; i < visible.length; i++) if (visible[i] === HIDDEN) hidden.push(i);
    hidden.sort((a, b) => dist(a) - dist(b));
    hidden.forEach((i, k) => cells.set(i, { t0: now + 120 + k * 60, dur: 280 }));
    total = Math.max(1400, 120 + hidden.length * 60 + 280 + 300);
  }
  // deterministic burst particles (seeded from the origin cell)
  let seed = (orow * 31 + ocol * 17 + 7) >>> 0;
  const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
  const particles = [];
  for (let k = 0; k < 14; k++) {
    const a = (k / 14) * Math.PI * 2 + (rnd() - 0.5) * 0.5;
    particles.push({ dx: Math.cos(a), dy: Math.sin(a), v: 1.3 + rnd() * 1.4, r: 1.5 + rnd() * 1.5, life: 0.7 + rnd() * 0.3 });
  }
  return { kind, t0: now, origin: [orow, ocol], cells, particles, total, tagAt: kind === 'lost' ? 520 : 320 };
}

const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);
const easeOut = (x) => 1 - (1 - x) * (1 - x) * (1 - x);

/** Small rounded tag in the top-right corner of the board (never covers more than one cell). */
function drawCornerTag(ctx, W, cell, pad, text, color, alpha) {
  if (alpha <= 0) return;
  const fs = Math.max(10, Math.round(cell * 0.24));
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.font = `600 ${fs}px ${MONO}`; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  const tw = ctx.measureText(text).width, ph = fs * 0.55, pv = fs * 0.42, h = fs + 2 * pv, w = tw + 2 * ph + fs * 0.25;
  const x = W - pad - 4 - w, y = pad + 4;
  ctx.fillStyle = 'rgba(7,9,12,0.88)'; ctx.beginPath(); ctx.roundRect(x, y, w, h, 6); ctx.fill();
  ctx.strokeStyle = color; ctx.globalAlpha = alpha * 0.55; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.roundRect(x + 0.5, y + 0.5, w - 1, h - 1, 6); ctx.stroke();
  ctx.globalAlpha = alpha; ctx.fillStyle = color;
  ctx.fillText(text, x + w - ph, y + h / 2 + 0.5);
  ctx.restore();
}

/**
 * Draw a whole board. `s`: { visible, cursor, dangerous, mineHit, safeStart, dim, banner, bannerColor,
 * outcome ('lost'|'won'|null), hover } ; `fx` is the schedule from makeEndFx (or null). Returns true
 * while an animation is still running (the caller should redraw on the next frame).
 */
export function renderBoard(ctx, W, H, rows, cols, s, fx, eyeSplit, now) {
  const cell = Math.floor(W / cols), pad = Math.floor((W - cell * cols) / 2);
  ctx.fillStyle = '#07090c'; ctx.fillRect(0, 0, W, H);
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  const symbol = (ch, color, x, y, scale = 1, sizeK = 0.46) => {
    ctx.fillStyle = color; ctx.font = `${Math.floor(cell * sizeK)}px sans-serif`;
    if (scale === 1) { ctx.fillText(ch, x + cell / 2, y + cell / 2 + 1); return; }
    ctx.save(); ctx.translate(x + cell / 2, y + cell / 2 + 1); ctx.scale(scale, scale); ctx.fillText(ch, 0, 0); ctx.restore();
  };
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const i = r * cols + c, v = s.visible[i], x = pad + c * cell, y = pad + r * cell;
    const hit = s.mineHit && s.mineHit[0] === r && s.mineHit[1] === c;
    const hover = v === HIDDEN && s.hover && s.hover[0] === r && s.hover[1] === c && !s.outcome;
    const cf = fx && fx.cells.get(i);
    if (cf) {
      const p = clamp01((now - cf.t0) / cf.dur);
      if (fx.kind === 'lost') {
        // a mine popping into view: hidden tile until its time, then the mine scales in with a little overshoot
        if (p <= 0) { drawTile(ctx, x, y, cell, 'hidden'); continue; }
        drawTile(ctx, x, y, cell, 'mine');
        const sc = p < 0.6 ? 0.25 + 0.95 * easeOut(p / 0.6) : 1.2 - 0.2 * ((p - 0.6) / 0.4);
        symbol('✸', '#f0647c', x, y, sc);
        continue;
      }
      // a hidden cell flipping to a flag (vertical flip, swap at the midpoint)
      const sy = Math.max(0.06, Math.abs(Math.cos(p * Math.PI)));
      const kind = p < 0.5 ? 'hidden' : 'flag';
      ctx.save(); ctx.translate(0, y + cell / 2); ctx.scale(1, sy); ctx.translate(0, -(y + cell / 2));
      drawTile(ctx, x, y, cell, kind);
      if (kind === 'flag') symbol('⚑', '#f2c15c', x, y, 1, 0.42);
      ctx.restore();
      continue;
    }
    drawTile(ctx, x, y, cell, hit ? 'hit' : v === HIDDEN ? 'hidden' : v === FLAG ? 'flag' : v === MINE ? 'mine' : 'open', hover);
    if (v >= 1 && v <= 8) { ctx.fillStyle = NUM_COLORS[v]; ctx.font = `600 ${Math.floor(cell * 0.46)}px ${FONT}`; ctx.fillText(String(v), x + cell / 2, y + cell / 2 + 1); }
    if (v === FLAG) symbol('⚑', '#f2c15c', x, y, 1, 0.42);
    if (v === MINE) symbol('✸', hit ? '#fff' : '#f0647c', x, y);
  }
  if (s.safeStart) {
    const [r, c] = s.safeStart;
    ctx.fillStyle = 'rgba(242,193,92,0.55)';
    ctx.beginPath(); ctx.arc(pad + c * cell + cell / 2, pad + r * cell + cell / 2, Math.max(2, cell * 0.08), 0, Math.PI * 2); ctx.fill();
  }
  if (eyeSplit) {
    ctx.strokeStyle = 'rgba(255,255,255,0.09)'; ctx.lineWidth = 1; ctx.setLineDash([1.5, 5]);
    ctx.beginPath(); ctx.moveTo(W / 2, pad + 4); ctx.lineTo(W / 2, pad + cell * rows - 4); ctx.stroke(); ctx.setLineDash([]);
  }
  if (s.cursor) {
    const [cr, cc] = s.cursor;
    drawCursor(ctx, pad + cc * cell, pad + cr * cell, cell, !!s.dangerous || s.outcome === 'lost');
  }
  let animating = false;
  if (fx) {
    const t = now - fx.t0;
    animating = t < fx.total;
    const [orow, ocol] = fx.origin, cx = pad + ocol * cell + cell / 2, cy = pad + orow * cell + cell / 2;
    if (fx.kind === 'lost') {
      // burst: a flash, three expanding rings and a handful of particles, ~600 ms
      const e = clamp01(t / 600);
      if (e < 1) {
        ctx.save();
        const flash = ctx.createRadialGradient(cx, cy, 0, cx, cy, cell * 1.6);
        flash.addColorStop(0, `rgba(255,120,140,${0.55 * (1 - e) * (1 - e)})`); flash.addColorStop(1, 'rgba(255,120,140,0)');
        ctx.fillStyle = flash; ctx.fillRect(cx - cell * 1.6, cy - cell * 1.6, cell * 3.2, cell * 3.2);
        for (let k = 0; k < 3; k++) {
          const ek = clamp01((e - k * 0.14) / (1 - k * 0.14));
          if (ek <= 0) continue;
          ctx.strokeStyle = `rgba(240,100,124,${0.85 * (1 - ek)})`; ctx.lineWidth = Math.max(1, cell * 0.06 * (1 - ek) + 1);
          ctx.beginPath(); ctx.arc(cx, cy, cell * (0.25 + 2.1 * easeOut(ek)), 0, Math.PI * 2); ctx.stroke();
        }
        for (const p of fx.particles) {
          const ep = clamp01(e / p.life);
          if (ep >= 1) continue;
          const d = cell * p.v * easeOut(ep);
          ctx.fillStyle = `rgba(255,${Math.round(150 + 60 * (1 - ep))},120,${0.9 * (1 - ep)})`;
          ctx.beginPath(); ctx.arc(cx + p.dx * d, cy + p.dy * d + 0.6 * cell * ep * ep, Math.max(0.8, p.r * (1 - ep * 0.6) * cell / 40), 0, Math.PI * 2); ctx.fill();
        }
        ctx.restore();
      }
    } else {
      // gold glow pulse along the board border, fading out over ~1.4 s
      const g = t / 1000;
      if (g < 1.6) {
        const a = Math.max(0, 0.95 * Math.exp(-g * 1.4) * (0.55 + 0.45 * Math.sin(g * Math.PI * 3.5)));
        ctx.save();
        ctx.shadowColor = `rgba(242,193,92,${a})`; ctx.shadowBlur = Math.max(14, cell * 0.6);
        ctx.strokeStyle = `rgba(242,193,92,${a * 0.9})`; ctx.lineWidth = Math.max(2, cell * 0.05);
        ctx.beginPath(); ctx.roundRect(pad + 1.5, pad + 1.5, cell * cols - 3, cell * rows - 3, 8); ctx.stroke();
        ctx.restore();
      }
    }
    const tagA = clamp01((t - fx.tagAt) / 250);
    drawCornerTag(ctx, W, cell, pad, fx.kind === 'lost' ? 'LOST' : 'WON', fx.kind === 'lost' ? '#f0647c' : '#f2c15c', tagA);
  }
  if (s.dim) { ctx.fillStyle = 'rgba(7,9,12,0.5)'; ctx.fillRect(0, 0, W, H); }
  if (s.banner) drawCornerTag(ctx, W, cell, pad, s.banner, s.bannerColor || '#8b95a3', 1);
  return animating;
}

export class BoardView {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {number} rows
   * @param {number} cols
   * @param {object} [opts]
   * @param {boolean} [opts.eyeSplit]        draw the left/right eye split line (fly board)
   * @param {(r:number,c:number)=>void} [opts.onReveal]
   * @param {(r:number,c:number)=>void} [opts.onFlag]
   */
  constructor(canvas, rows, cols, opts = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.rows = rows; this.cols = cols;
    this.eyeSplit = !!opts.eyeSplit;
    this.hover = null;
    this.state = null;
    this.fx = null;          // end-of-game effect schedule (see makeEndFx)
    this.animating = false;  // true while an end-of-game animation still needs frames
    if (opts.onReveal || opts.onFlag) {
      canvas.style.cursor = 'pointer';
      const cellAt = (ev) => {
        const rect = canvas.getBoundingClientRect();
        const x = (ev.clientX - rect.left) * (canvas.width / rect.width);
        const y = (ev.clientY - rect.top) * (canvas.height / rect.height);
        const cell = Math.floor(canvas.width / this.cols), pad = Math.floor((canvas.width - cell * this.cols) / 2);
        const c = Math.floor((x - pad) / cell), r = Math.floor((y - pad) / cell);
        return r >= 0 && r < this.rows && c >= 0 && c < this.cols ? [r, c] : null;
      };
      canvas.addEventListener('contextmenu', (ev) => ev.preventDefault());
      canvas.addEventListener('mousedown', (ev) => {
        const rc = cellAt(ev);
        if (!rc) return;
        ev.preventDefault();
        if (ev.button === 2 || ev.ctrlKey || ev.altKey || ev.metaKey) opts.onFlag && opts.onFlag(rc[0], rc[1]);
        else if (ev.button === 0) opts.onReveal && opts.onReveal(rc[0], rc[1]);
      });
      canvas.addEventListener('mousemove', (ev) => { this.hover = cellAt(ev); if (this.state) this.draw(this.state); });
      canvas.addEventListener('mouseleave', () => { this.hover = null; if (this.state) this.draw(this.state); });
      // long-press = flag on touch devices
      let pressTimer = null;
      canvas.addEventListener('touchstart', (ev) => {
        const t = ev.touches[0]; const rc = cellAt(t); if (!rc) return;
        pressTimer = setTimeout(() => { pressTimer = null; opts.onFlag && opts.onFlag(rc[0], rc[1]); }, 450);
      }, { passive: true });
      canvas.addEventListener('touchend', (ev) => {
        if (pressTimer) { clearTimeout(pressTimer); pressTimer = null; const t = ev.changedTouches[0]; const rc = cellAt(t); if (rc) opts.onReveal && opts.onReveal(rc[0], rc[1]); }
        ev.preventDefault();
      });
    }
  }

  /**
   * @param {object} s
   * @param {Int8Array} s.visible
   * @param {[number,number]|null} [s.cursor]
   * @param {boolean} [s.dangerous]      cursor outline red (looming channel active)
   * @param {[number,number]|null} [s.mineHit]
   * @param {[number,number]|null} [s.safeStart]   mark a guaranteed-safe cell before the first reveal
   * @param {'lost'|'won'|null} [s.outcome]   starts the end-of-game animation (once per s.gameId)
   * @param {string|number} [s.gameId]
   * @param {string|null} [s.banner]     small corner tag (e.g. "settling…", "TIMEOUT")
   * @param {boolean} [s.dim]            grey out (waiting)
   */
  draw(s) {
    this.state = s;
    const now = performance.now();
    if (s.outcome === 'lost' || s.outcome === 'won') {
      if (!this.fx || this.fx.gameId !== s.gameId || this.fx.kind !== s.outcome) {
        this.fx = makeEndFx(s.outcome, s.visible, this.rows, this.cols, s.outcome === 'lost' ? (s.mineHit || s.cursor) : s.cursor, now);
        this.fx.gameId = s.gameId;
      }
    } else this.fx = null;
    this.animating = renderBoard(this.ctx, this.canvas.width, this.canvas.height, this.rows, this.cols, { ...s, hover: this.hover }, this.fx, this.eyeSplit, now);
  }
}
