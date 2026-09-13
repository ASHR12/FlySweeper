// Minesweeper board rendering (dark theme, same look as the Python spectator page) plus optional
// mouse input for the human player: left click reveals, right click flags.

import { HIDDEN, FLAG, MINE } from './minesweeper.js';

// classic Minesweeper hues, muted for the dark theme
export const NUM_COLORS = ['', '#6f9bff', '#63b96a', '#e46a7e', '#a985d6', '#d0745f', '#4db8ad', '#c9d1d9', '#8b95a3'];
const FONT = '-apple-system, BlinkMacSystemFont, "SF Pro Text", Inter, "Segoe UI", sans-serif';

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

/** Gold cursor ring with a faint outer glow (red when the looming channel is active). */
export function drawCursor(ctx, x, y, size, dangerous) {
  const inset = Math.max(1, size * 0.035) + 1.5, r = Math.max(2, size * 0.14);
  ctx.save();
  ctx.shadowColor = dangerous ? 'rgba(240,100,124,0.85)' : 'rgba(242,193,92,0.75)';
  ctx.shadowBlur = Math.max(6, size * 0.28);
  ctx.strokeStyle = dangerous ? '#f0647c' : '#f2c15c'; ctx.lineWidth = Math.max(2, size * 0.055);
  ctx.beginPath(); ctx.roundRect(x + inset, y + inset, size - 2 * inset, size - 2 * inset, r); ctx.stroke();
  ctx.restore();
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
   * @param {string|null} [s.banner]     overlay text (won / lost)
   * @param {boolean} [s.dim]            grey out (waiting)
   */
  draw(s) {
    this.state = s;
    const { ctx } = this, W = this.canvas.width, H = this.canvas.height;
    const cell = Math.floor(W / this.cols), pad = Math.floor((W - cell * this.cols) / 2);
    ctx.fillStyle = '#07090c'; ctx.fillRect(0, 0, W, H);
    for (let r = 0; r < this.rows; r++) for (let c = 0; c < this.cols; c++) {
      const v = s.visible[r * this.cols + c], x = pad + c * cell, y = pad + r * cell;
      const hit = s.mineHit && s.mineHit[0] === r && s.mineHit[1] === c;
      const hover = v === HIDDEN && this.hover && this.hover[0] === r && this.hover[1] === c && !s.banner;
      drawTile(ctx, x, y, cell, hit ? 'hit' : v === HIDDEN ? 'hidden' : v === FLAG ? 'flag' : v === MINE ? 'mine' : 'open', hover);
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      if (v >= 1 && v <= 8) { ctx.fillStyle = NUM_COLORS[v]; ctx.font = `600 ${Math.floor(cell * 0.46)}px ${FONT}`; ctx.fillText(String(v), x + cell / 2, y + cell / 2 + 1); }
      if (v === FLAG) { ctx.fillStyle = '#f2c15c'; ctx.font = `${Math.floor(cell * 0.42)}px sans-serif`; ctx.fillText('⚑', x + cell / 2, y + cell / 2 + 1); }
      if (v === MINE) { ctx.fillStyle = hit ? '#fff' : '#f0647c'; ctx.font = `${Math.floor(cell * 0.46)}px sans-serif`; ctx.fillText('✸', x + cell / 2, y + cell / 2 + 1); }
    }
    if (s.safeStart) {
      const [r, c] = s.safeStart;
      ctx.fillStyle = 'rgba(242,193,92,0.55)';
      ctx.beginPath(); ctx.arc(pad + c * cell + cell / 2, pad + r * cell + cell / 2, Math.max(2, cell * 0.08), 0, Math.PI * 2); ctx.fill();
    }
    if (this.eyeSplit) {
      ctx.strokeStyle = 'rgba(255,255,255,0.09)'; ctx.lineWidth = 1; ctx.setLineDash([1.5, 5]);
      ctx.beginPath(); ctx.moveTo(W / 2, pad + 4); ctx.lineTo(W / 2, pad + cell * this.rows - 4); ctx.stroke(); ctx.setLineDash([]);
    }
    if (s.cursor) {
      const [cr, cc] = s.cursor;
      drawCursor(ctx, pad + cc * cell, pad + cr * cell, cell, !!s.dangerous);
    }
    if (s.dim) { ctx.fillStyle = 'rgba(7,9,12,0.5)'; ctx.fillRect(0, 0, W, H); }
    if (s.banner) {
      const bh = Math.max(40, cell * 0.95);
      ctx.fillStyle = 'rgba(7,9,12,0.72)'; ctx.fillRect(0, H / 2 - bh / 2, W, bh);
      ctx.fillStyle = 'rgba(255,255,255,0.08)'; ctx.fillRect(0, H / 2 - bh / 2, W, 1); ctx.fillRect(0, H / 2 + bh / 2 - 1, W, 1);
      ctx.fillStyle = s.bannerColor || '#e7ebf0'; ctx.font = `600 ${Math.floor(cell * 0.55)}px ${FONT}`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(s.banner, W / 2, H / 2);
    }
  }
}
