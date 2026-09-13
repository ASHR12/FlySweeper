// Minesweeper board rendering (dark theme, same look as the Python spectator page) plus optional
// mouse input for the human player: left click reveals, right click flags.

import { HIDDEN, FLAG, MINE } from './minesweeper.js';

const NUM_COLORS = ['', '#4f7dff', '#4caf50', '#ff5a78', '#8e44ad', '#c0392b', '#16a085', '#111', '#777'];

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
        const cell = Math.floor(canvas.width / this.cols);
        const c = Math.floor(x / cell), r = Math.floor(y / cell);
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
    const cell = Math.floor(W / this.cols);
    ctx.fillStyle = '#0a0d11'; ctx.fillRect(0, 0, W, H);
    for (let r = 0; r < this.rows; r++) for (let c = 0; c < this.cols; c++) {
      const v = s.visible[r * this.cols + c], x = c * cell, y = r * cell;
      const hit = s.mineHit && s.mineHit[0] === r && s.mineHit[1] === c;
      ctx.fillStyle = hit ? '#c0392b' : v === HIDDEN ? '#26313f' : v === FLAG ? '#5a3d1a' : v === MINE ? '#7a1b2a' : '#0f151c';
      if (v === HIDDEN && this.hover && this.hover[0] === r && this.hover[1] === c && !s.banner) ctx.fillStyle = '#34435a';
      ctx.fillRect(x + 1, y + 1, cell - 2, cell - 2);
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      if (v >= 1 && v <= 8) { ctx.fillStyle = NUM_COLORS[v]; ctx.font = `bold ${Math.floor(cell * 0.5)}px -apple-system, sans-serif`; ctx.fillText(String(v), x + cell / 2, y + cell / 2 + 1); }
      if (v === FLAG) { ctx.fillStyle = '#ffb347'; ctx.font = `${Math.floor(cell * 0.45)}px sans-serif`; ctx.fillText('⚑', x + cell / 2, y + cell / 2 + 1); }
      if (v === MINE) { ctx.fillStyle = hit ? '#fff' : '#ff5a78'; ctx.font = `${Math.floor(cell * 0.5)}px sans-serif`; ctx.fillText('✸', x + cell / 2, y + cell / 2 + 1); }
    }
    if (s.safeStart) {
      const [r, c] = s.safeStart;
      ctx.fillStyle = 'rgba(255,196,90,0.55)';
      ctx.beginPath(); ctx.arc(c * cell + cell / 2, r * cell + cell / 2, Math.max(2, cell * 0.08), 0, Math.PI * 2); ctx.fill();
    }
    if (s.cursor) {
      const [cr, cc] = s.cursor;
      ctx.strokeStyle = s.dangerous ? '#ff5a78' : '#ffc45a'; ctx.lineWidth = 3;
      ctx.strokeRect(cc * cell + 2, cr * cell + 2, cell - 4, cell - 4);
    }
    if (this.eyeSplit) {
      ctx.strokeStyle = 'rgba(255,255,255,0.12)'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(W / 2, 0); ctx.lineTo(W / 2, H); ctx.stroke(); ctx.setLineDash([]);
    }
    if (s.dim) { ctx.fillStyle = 'rgba(10,13,17,0.45)'; ctx.fillRect(0, 0, W, H); }
    if (s.banner) {
      ctx.fillStyle = 'rgba(10,13,17,0.55)'; ctx.fillRect(0, H / 2 - 24, W, 48);
      ctx.fillStyle = s.bannerColor || '#e6edf3'; ctx.font = `600 ${Math.floor(cell * 0.6)}px -apple-system, sans-serif`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(s.banner, W / 2, H / 2);
    }
  }
}
