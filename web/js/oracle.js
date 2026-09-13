// The helper for the trained fly (condition "fly-mb"): a faithful port of flysweeper/oracle.py plus
// the bits of flysweeper/teacher.py it uses (single-point inference iterated to a fixpoint, the crude
// mine-probability ranking, nearest-cell search). It reads the visible board into 31 facts about the
// cell under the cursor, each a value in [0, 1]; mbpolicy.js turns every channel into current on one
// group of olfactory receptor neurons. Nothing in this file is the fly: it is ordinary Minesweeper
// logic (and the same "cheating" the Python condition is honest about).
//
// The board is the Int8Array from Minesweeper.visible(): -1 hidden, 0-8 number, 9 flag, 10 mine.

import { HIDDEN, FLAG, NEIGHBORS } from './minesweeper.js';

export const CHANNELS = [
  'cursor_hidden', 'cursor_revealed', 'cursor_flagged', 'cursor_safe', 'cursor_mine', 'cursor_frontier', 'cursor_free',
  'adj_numbers_0', 'adj_numbers_1_2', 'adj_numbers_3p',
  'adj_hidden_0', 'adj_hidden_1_3', 'adj_hidden_4p',
  'adj_max_0', 'adj_max_1_2', 'adj_max_3p',
  'safe_up', 'safe_down', 'safe_left', 'safe_right', 'no_safe_known',
  'guess_up', 'guess_down', 'guess_left', 'guess_right', 'guess_here', 'guess_far',
  'cleared_lt_third', 'cleared_mid', 'cleared_gt_two_thirds', 'untouched',
];
export const CH = Object.fromEntries(CHANNELS.map((n, i) => [n, i]));
export const N_CHANNELS = CHANNELS.length;
export const ACTIONS = ['up', 'down', 'left', 'right', 'reveal', 'jump'];

function neighbors(r, c, rows, cols) {
  const out = [];
  for (const [dr, dc] of NEIGHBORS) {
    const rr = r + dr, cc = c + dc;
    if (rr >= 0 && rr < rows && cc >= 0 && cc < cols) out.push(rr * cols + cc);
  }
  return out;
}

/**
 * teacher.infer: single-point inference on the visible grid, iterated to a fixpoint. Returns
 * {safe, mines} as Uint8 masks over cells (flags count as unknown hidden cells).
 */
export function infer(visible, rows, cols) {
  const n = rows * cols;
  const unknown = new Uint8Array(n), safe = new Uint8Array(n), mines = new Uint8Array(n);
  const numbers = [];
  for (let i = 0; i < n; i++) {
    const v = visible[i];
    unknown[i] = v === HIDDEN || v === FLAG ? 1 : 0;
    if (v >= 0 && v <= 8) numbers.push(i);
  }
  const nbrCache = numbers.map((i) => neighbors(Math.floor(i / cols), i % cols, rows, cols).filter((p) => unknown[p]));
  let changed = true;
  while (changed) {
    changed = false;
    for (let q = 0; q < numbers.length; q++) {
      const hidden = nbrCache[q];
      if (!hidden.length) continue;
      const num = visible[numbers[q]];
      let known = 0, rest = 0;
      for (const p of hidden) { if (mines[p]) known++; else rest++; }
      if (!rest) continue;
      if (num - known === rest) {
        for (const p of hidden) if (!mines[p]) { mines[p] = 1; changed = true; }
      } else if (num === known) {
        for (const p of hidden) if (!mines[p] && !safe[p]) { safe[p] = 1; changed = true; }
      }
    }
  }
  return { safe, mines, unknown };
}

/** teacher.mine_probability: max over adjacent numbers of remaining mines / remaining unknown neighbours. */
export function mineProbability(visible, rows, cols, mines, unknown) {
  const prob = new Map();
  for (let i = 0; i < rows * cols; i++) {
    const num = visible[i];
    if (!(num >= 0 && num <= 8)) continue;
    const hidden = neighbors(Math.floor(i / cols), i % cols, rows, cols).filter((p) => unknown[p]);
    let known = 0;
    const rest = [];
    for (const p of hidden) { if (mines[p]) known++; else rest.push(p); }
    if (!rest.length) continue;
    const pMine = Math.max(0, (num - known) / rest.length);
    for (const p of rest) prob.set(p, Math.max(prob.get(p) || 0, pMine));
  }
  return prob;
}

/** teacher._nearest: min over cells by (Manhattan distance, row, col). */
function nearest(cursor, cells, cols) {
  let best = -1, bestKey = null;
  for (const p of cells) {
    const r = Math.floor(p / cols), c = p % cols;
    const key = [Math.abs(r - cursor[0]) + Math.abs(c - cursor[1]), r, c];
    if (best < 0 || key[0] < bestKey[0] || (key[0] === bestKey[0] && (key[1] < bestKey[1] || (key[1] === bestKey[1] && key[2] < bestKey[2])))) {
      best = p; bestKey = key;
    }
  }
  return best;
}

function directionChannels(vec, base, cursor, target, cols) {
  const dr = Math.floor(target / cols) - cursor[0], dc = (target % cols) - cursor[1];
  const dist = Math.abs(dr) + Math.abs(dc);
  if (dist === 0) return;
  const g = 1.0 / dist;
  if (dr < 0) vec[base + 0] = g; else if (dr > 0) vec[base + 1] = g;
  if (dc < 0) vec[base + 2] = g; else if (dc > 0) vec[base + 3] = g;
}

export class Oracle {
  constructor({ jumpDistance = 5 } = {}) {
    this.jumpDistance = jumpDistance;
    this.last = {};
  }

  /** oracle.Oracle.features: Float32Array(31) for the visible board, cursor [r, c] and mine count. */
  features(visible, rows, cols, cursor, nMines = 10) {
    const [r, c] = cursor;
    const n = rows * cols;
    const v = new Float32Array(N_CHANNELS);
    const revealed = new Uint8Array(n);
    let anyRevealed = false, nRevealed = 0;
    for (let i = 0; i < n; i++) if (visible[i] >= 0 && visible[i] <= 8) { revealed[i] = 1; anyRevealed = true; nRevealed++; }
    const here = r * cols + c;
    const cell = visible[here];
    v[CH.cursor_hidden] = cell === HIDDEN ? 1 : 0;
    v[CH.cursor_revealed] = cell >= 0 && cell <= 8 ? 1 : 0;
    v[CH.cursor_flagged] = cell === FLAG ? 1 : 0;
    const untouched = !anyRevealed;
    v[CH.untouched] = untouched ? 1 : 0;

    const { safe, mines, unknown } = untouched ? { safe: null, mines: null, unknown: null } : infer(visible, rows, cols);
    const unk = unknown || (() => { const u = new Uint8Array(n); for (let i = 0; i < n; i++) u[i] = visible[i] === HIDDEN || visible[i] === FLAG ? 1 : 0; return u; })();
    const nbrs = neighbors(r, c, rows, cols);
    let nNumbers = 0, nHidden = 0, maxNum = 0;
    for (const p of nbrs) {
      if (revealed[p]) { nNumbers++; if (visible[p] > maxNum) maxNum = visible[p]; }
      if (unk[p]) nHidden++;
    }
    v[CH.adj_numbers_0 + (nNumbers === 0 ? 0 : nNumbers <= 2 ? 1 : 2)] = 1;
    v[CH.adj_hidden_0 + (nHidden === 0 ? 0 : nHidden <= 3 ? 1 : 2)] = 1;
    v[CH.adj_max_0 + (maxNum === 0 ? 0 : maxNum <= 2 ? 1 : 2)] = 1;

    // provably safe / mine sets and the frontier (hidden cells touching a revealed number)
    let safeSet, mineSet, frontier;
    if (untouched) {
      safeSet = new Uint8Array(n); safeSet[Math.floor(rows / 2) * cols + Math.floor(cols / 2)] = 1;
      mineSet = new Uint8Array(n);
      frontier = new Uint8Array(n);
    } else {
      safeSet = safe; mineSet = mines;
      frontier = new Uint8Array(n);
      for (let i = 0; i < n; i++) if (revealed[i]) for (const p of neighbors(Math.floor(i / cols), i % cols, rows, cols)) frontier[p] = 1;
    }
    const cursorUnknown = unk[here] === 1;
    v[CH.cursor_safe] = safeSet[here] || (untouched && cursorUnknown) ? 1 : 0;
    v[CH.cursor_mine] = mineSet[here] ? 1 : 0;
    v[CH.cursor_frontier] = cursorUnknown && frontier[here] && !safeSet[here] && !mineSet[here] ? 1 : 0;
    v[CH.cursor_free] = cursorUnknown && !frontier[here] && !untouched ? 1 : 0;

    let target = -1, mode = 'safe', nSafe = 0, nMinesKnown = 0;
    const safeCells = [];
    for (let i = 0; i < n; i++) { if (safeSet[i]) { safeCells.push(i); nSafe++; } if (mineSet[i]) nMinesKnown++; }
    if (safeCells.length) {
      target = nearest(cursor, safeCells, cols);
      directionChannels(v, CH.safe_up, cursor, target, cols);
    } else {
      v[CH.no_safe_known] = 1;
      const cand = [];
      for (let i = 0; i < n; i++) if (unk[i] && !mineSet[i]) cand.push(i);
      if (cand.length) {
        const free = cand.filter((p) => !frontier[p]);
        if (free.length) {
          target = nearest(cursor, free, cols);
          mode = 'guess-free';
        } else {
          const prob = mineProbability(visible, rows, cols, mineSet, unk);
          let bestKey = null;
          for (const p of cand) {
            const pr = Math.round((prob.has(p) ? prob.get(p) : 1.0) * 1e6) / 1e6;
            const rr = Math.floor(p / cols), cc = p % cols;
            const key = [pr, Math.abs(rr - r) + Math.abs(cc - c), rr, cc];
            if (bestKey === null || lexLess(key, bestKey)) { bestKey = key; target = p; }
          }
          mode = 'guess-frontier';
        }
        const dist = Math.abs(Math.floor(target / cols) - r) + Math.abs((target % cols) - c);
        if (dist === 0) v[CH.guess_here] = 1;
        else if (dist > this.jumpDistance) v[CH.guess_far] = 1;
        else directionChannels(v, CH.guess_up, cursor, target, cols);
      } else {
        mode = 'none';
      }
    }
    const frac = nRevealed / Math.max(1, n - nMines);
    v[CH.cleared_lt_third + (frac < 1 / 3 ? 0 : frac < 2 / 3 ? 1 : 2)] = 1;
    this.last = { target: target < 0 ? null : [Math.floor(target / cols), target % cols], mode, safe: nSafe, mines: nMinesKnown };
    return v;
  }

  /** oracle.Oracle.rule_action: the fixed reader of the same channels (the "oracle-only" ceiling). */
  static ruleAction(v) {
    if (v[CH.cursor_safe] > 0 || v[CH.guess_here] > 0) return 'reveal';
    if (v[CH.guess_far] > 0) return 'jump';
    let best = -1, max = 0;
    for (let k = 0; k < 4; k++) {
      const d = v[CH.safe_up + k] + v[CH.guess_up + k];
      if (d > max) { max = d; best = k; }
    }
    if (best >= 0) return ACTIONS[best];
    return 'jump';
  }
}

function lexLess(a, b) {
  for (let i = 0; i < a.length; i++) { if (a[i] < b[i]) return true; if (a[i] > b[i]) return false; }
  return false;
}
