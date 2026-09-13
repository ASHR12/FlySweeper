// Plain Minesweeper, a port of flysweeper/minesweeper.py. Nothing here knows about flies.
//
// Visible codes: -1 hidden, 0-8 revealed number, 9 flag, 10 mine (shown after losing).
// Mines are either placed on the first reveal (classic safe first click), or supplied as a
// pre-generated layout so two boards can share the exact same mines.

import { RNG } from './rng.js';

export const HIDDEN = -1;
export const FLAG = 9;
export const MINE = 10;

export const NEIGHBORS = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]];

export class Minesweeper {
  /**
   * @param {number} rows
   * @param {number} cols
   * @param {number} nMines
   * @param {object} [opts]
   * @param {number} [opts.seed]          seed for mine placement (classic mode)
   * @param {Uint8Array} [opts.layout]    pre-generated mines (rows*cols, 1 = mine); board starts "placed"
   */
  constructor(rows = 9, cols = 9, nMines = 10, opts = {}) {
    if (nMines >= rows * cols - 9) throw new Error('too many mines for a safe first click');
    this.rows = rows; this.cols = cols; this.nMines = nMines;
    this.seed = opts.seed ?? 0;
    this.rng = new RNG(this.seed);
    this.mines = new Uint8Array(rows * cols);
    this.numbers = new Int8Array(rows * cols);
    this.revealed = new Uint8Array(rows * cols);
    this.flagged = new Uint8Array(rows * cols);
    this.placed = false;
    this.lost = false;
    this.moves = 0;
    this.mineHit = null;
    if (opts.layout) {
      this.mines.set(opts.layout);
      this._computeNumbers();
      this.placed = true;
    }
  }

  /** A mine layout with a guaranteed mine-free 3x3 zone around (safeR, safeC), like _place_mines. */
  static makeLayout(rows, cols, nMines, safeR, safeC, rng) {
    const forbidden = new Set([safeR * cols + safeC]);
    for (const [dr, dc] of NEIGHBORS) {
      const r = safeR + dr, c = safeC + dc;
      if (r >= 0 && r < rows && c >= 0 && c < cols) forbidden.add(r * cols + c);
    }
    const candidates = [];
    for (let i = 0; i < rows * cols; i++) if (!forbidden.has(i)) candidates.push(i);
    const layout = new Uint8Array(rows * cols);
    for (const k of rng.sample(candidates.length, nMines)) layout[candidates[k]] = 1;
    return layout;
  }

  get totalSafe() { return this.rows * this.cols - this.nMines; }
  get safeRevealed() { let s = 0; for (let i = 0; i < this.revealed.length; i++) s += this.revealed[i]; return s; }
  get won() { return !this.lost && this.placed && this.safeRevealed === this.totalSafe; }
  get over() { return this.lost || this.won; }
  get clearedFraction() { return this.safeRevealed / this.totalSafe; }
  get flagsPlaced() { let s = 0; for (let i = 0; i < this.flagged.length; i++) s += this.flagged[i] && !this.revealed[i] ? 1 : 0; return s; }

  inBounds(r, c) { return r >= 0 && r < this.rows && c >= 0 && c < this.cols; }

  *neighbors(r, c) {
    for (const [dr, dc] of NEIGHBORS) {
      const rr = r + dr, cc = c + dc;
      if (this.inBounds(rr, cc)) yield [rr, cc];
    }
  }

  /** Int8Array(rows*cols): -1 hidden, 0-8 number, 9 flag, 10 mine (after losing). */
  visible() {
    const v = new Int8Array(this.rows * this.cols).fill(HIDDEN);
    for (let i = 0; i < v.length; i++) {
      if (this.revealed[i]) v[i] = this.numbers[i];
      else if (this.flagged[i]) v[i] = FLAG;
    }
    if (this.lost) for (let i = 0; i < v.length; i++) if (this.mines[i]) v[i] = MINE;
    return v;
  }

  _computeNumbers() {
    for (let r = 0; r < this.rows; r++) for (let c = 0; c < this.cols; c++) {
      let n = 0;
      for (const [rr, cc] of this.neighbors(r, c)) n += this.mines[rr * this.cols + cc];
      this.numbers[r * this.cols + c] = n;
    }
  }

  _placeMines(r0, c0) {
    this.mines.set(Minesweeper.makeLayout(this.rows, this.cols, this.nMines, r0, c0, this.rng));
    this._computeNumbers();
    this.placed = true;
  }

  /** Returns {result: 'mine'|'safe'|'noop', newly: number}. */
  reveal(r, c) {
    const i = r * this.cols + c;
    if (this.over || !this.inBounds(r, c) || this.revealed[i] || this.flagged[i]) return { result: 'noop', newly: 0 };
    this.moves += 1;
    if (!this.placed) this._placeMines(r, c);
    if (this.mines[i]) {
      this.lost = true;
      this.mineHit = [r, c];
      return { result: 'mine', newly: 0 };
    }
    const before = this.safeRevealed;
    const q = [[r, c]];
    this.revealed[i] = 1;
    while (q.length) {
      const [rr, cc] = q.shift();
      if (this.numbers[rr * this.cols + cc] === 0) {
        for (const [nr, nc] of this.neighbors(rr, cc)) {
          const k = nr * this.cols + nc;
          if (!this.revealed[k] && !this.mines[k]) {
            this.revealed[k] = 1;
            this.flagged[k] = 0;
            q.push([nr, nc]);
          }
        }
      }
    }
    return { result: 'safe', newly: this.safeRevealed - before };
  }

  toggleFlag(r, c) {
    const i = r * this.cols + c;
    if (this.over || !this.inBounds(r, c) || this.revealed[i]) return false;
    this.moves += 1;
    this.flagged[i] = this.flagged[i] ? 0 : 1;
    return true;
  }

  /** True if any neighbour of cell i is revealed (i is on the revealed frontier). */
  onFrontier(i) {
    const r = Math.floor(i / this.cols), c = i % this.cols;
    for (const [rr, cc] of this.neighbors(r, c)) if (this.revealed[rr * this.cols + cc]) return true;
    return false;
  }

  /** Move the mine at cell `from` to cell `to` and recompute every number. */
  moveMine(from, to) {
    if (!this.mines[from] || this.mines[to]) throw new Error('moveMine: no mine at source or mine at target');
    this.mines[from] = 0;
    this.mines[to] = 1;
    this._computeNumbers();
  }

  /** [r, c] of every cell that is neither revealed nor flagged. */
  hiddenCells() {
    const out = [];
    for (let r = 0; r < this.rows; r++) for (let c = 0; c < this.cols; c++) {
      const i = r * this.cols + c;
      if (!this.revealed[i] && !this.flagged[i]) out.push([r, c]);
    }
    return out;
  }
}
