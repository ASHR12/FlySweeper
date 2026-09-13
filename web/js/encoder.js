// Board -> visual-neuron drive, a port of flysweeper/encoder.py (route "lamina"). Everything here
// is engineered: which optic column sees which board cell, the luminance table, the 25 Hz cursor
// flicker, Fly64's retinal formula, and the looming channel into LC4/LPLC2.
//
//     drive = clip(0.45*lum + 1.6*|lum - prev_lum| + 0.25*colour, 0, 1) * 0.62
//
// The per-target lists (neuron index, channel, board cell) come from model.json; they were built
// with the same equal-count column partition as the Python encoder (cross-checked at export).

import { FLAG } from './minesweeper.js';

export const LUM = 0, FLAG_CH = 1, CURSOR_CH = 2;

export class Encoder {
  constructor(model) {
    const enc = model.encoder;
    this.p = enc.params;
    this.rows = model.game.rows;
    this.cols = model.game.cols;
    this.idx = Uint32Array.from(enc.idx);
    this.chan = Int8Array.from(enc.chan);
    this.cell = Int32Array.from(enc.cell);
    this.lumTable = Float32Array.from(enc.lum_table);   // index = visible code + 1
    this.loom = { L: Uint32Array.from(enc.loom.L), R: Uint32Array.from(enc.loom.R) };
    this.nTargets = this.idx.length;
    this.prevLum = new Float32Array(this.nTargets);
    this.cellLum = new Float32Array(this.rows * this.cols);
    this.amt = new Float32Array(this.nTargets);
    this.outIdx = new Uint32Array(this.nTargets);
    this.outAmt = new Float32Array(this.nTargets);
    this.lastDanger = 0;
    this.stats = {
      targets: this.nTargets,
      lamina: this.chan.filter((c) => c === LUM).length,
      r7: this.chan.filter((c) => c === FLAG_CH).length,
      r8: this.chan.filter((c) => c === CURSOR_CH).length,
    };
    // the scatter kernel adds; make sure no neuron is listed twice within a step
    const all = new Set(this.idx);
    if (all.size !== this.idx.length) throw new Error('encoder targets contain duplicates');
    for (const s of ['L', 'R']) for (const j of this.loom[s]) if (all.has(j)) throw new Error('looming cells overlap encoder targets');
  }

  reset() { this.prevLum.fill(0); }

  /** Largest revealed number on or around the cursor (0-8). */
  dangerAt(visible, cursor) {
    const [r, c] = cursor;
    let best = 0;
    for (let rr = Math.max(0, r - 1); rr < Math.min(this.rows, r + 2); rr++) {
      for (let cc = Math.max(0, c - 1); cc < Math.min(this.cols, c + 2); cc++) {
        const v = visible[rr * this.cols + cc];
        if (v >= 1 && v <= 8 && v > best) best = v;
      }
    }
    return best;
  }

  /** Looming detectors of the cursor's eye, pulsed every other step with amplitude ~ danger. */
  loomDrive(visible, cursor, step) {
    this.lastDanger = this.dangerAt(visible, cursor);
    if (!this.p.danger_loom || this.lastDanger < this.p.loom_threshold || step % 2) return null;
    const side = cursor[1] < this.cols / 2 ? 'L' : 'R';
    const amp = this.p.loom_gain * Math.min(1.0, (this.lastDanger - 1) / 4.0);
    return { idx: this.loom[side], amp, side };
  }

  /**
   * One 20 ms step of retinal drive. Returns {idx, amt} (views into reusable buffers; copy before
   * the next call) containing only the targets with non-zero drive, in target order.
   */
  drive(visible, cursor, step) {
    const p = this.p, cellLum = this.cellLum, table = this.lumTable;
    for (let i = 0; i < cellLum.length; i++) cellLum[i] = table[visible[i] + 1];
    const cur = cursor[0] * this.cols + cursor[1];
    if (step % 2 === 0) cellLum[cur] = Math.min(1.0, cellLum[cur] + p.cursor_flicker);
    let m = 0;
    for (let k = 0; k < this.nTargets; k++) {
      const cell = this.cell[k];
      const lum = cellLum[cell];
      const temporal = Math.abs(lum - this.prevLum[k]);
      this.prevLum[k] = lum;
      const ch = this.chan[k];
      let d;
      if (ch === LUM) d = p.lum_weight * lum + p.temporal_weight * temporal;
      else if (ch === FLAG_CH) d = visible[cell] === FLAG ? p.color_weight : 0;
      else d = cell === cur ? p.color_weight : 0;
      if (d <= 0) continue;
      d = Math.min(d, 1.0) * p.retina_gain;
      this.outIdx[m] = this.idx[k];
      this.outAmt[m] = d;
      m++;
    }
    return { idx: this.outIdx.subarray(0, m), amt: this.outAmt.subarray(0, m) };
  }

  /**
   * Retinal + looming drive for one step, concatenated into fresh arrays (safe to keep).
   * `blind` = no input at all (the black screen used for idle calibration).
   */
  stepDrive(visible, cursor, step, blind = false) {
    if (blind) { this.lastDanger = 0; return null; }
    const ret = this.drive(visible, cursor, step);
    const loom = this.loomDrive(visible, cursor, step);
    const extra = loom && loom.amp > 0 ? loom.idx.length : 0;
    const idx = new Uint32Array(ret.idx.length + extra);
    const amt = new Float32Array(ret.idx.length + extra);
    idx.set(ret.idx); amt.set(ret.amt);
    if (extra) { idx.set(loom.idx, ret.idx.length); amt.fill(loom.amp, ret.idx.length); }
    return { idx, amt, loom: loom ? { side: loom.side, amp: loom.amp } : null };
  }
}
