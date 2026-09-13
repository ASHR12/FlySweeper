// The trained fly (condition "fly-mb"), a port of the inference path of flysweeper/mb_policy.py:
//
//   oracle.js        board + cursor -> 31 facts in [0, 1]                          (helper, not the fly)
//   odor             fact k -> current amp*value into every ORN cell of channel k    (real ORN cells)
//   the connectome   ORN -> antennal-lobe PN -> Kenyon cells -> MBONs, simulated on the GPU
//   decide           6 MBON pools' spike counts over the 15-step turn -> centred scores -> margin -> argmax
//
// The documented fly-mb model changes are applied to the resident GPU graph once, when the weight
// set is loaded (see `apply`): KC->KC edges scaled by kc_kc_gain (0 = silenced), PN->KC edges scaled
// by pn_kc_gain, the trained KC->MBON weights written over the frozen ones, and a per-neuron bias
// (kc_bias on Kenyon cells, pn_bias on antennal-lobe PNs) added to the external drive every step by
// the step kernel. Learning is off (weights are what the file says); temperature 0 (argmax).
//
// A weight set is `web/weights/<id>.json` + `<id>.bin` written by flysweeper/export_web.py --weights.

import { CH, N_CHANNELS, Oracle } from './oracle.js';

/** Fetch and parse one weight set; `base` is the URL prefix of web/weights/. */
export async function loadWeightSet(base, id, { log = () => {}, version = '' } = {}) {
  const q = version ? `?v=${encodeURIComponent(version)}` : '';
  const t0 = performance.now();
  const specResp = await fetch(`${base}${id}.json${q}`, { cache: 'no-store' });
  if (!specResp.ok) throw new Error(`weight set "${id}" not found (HTTP ${specResp.status} for ${base}${id}.json)`);
  const spec = await specResp.json();
  const binResp = await fetch(`${base}${spec.edges.file}${q}`, { cache: 'no-store' });
  if (!binResp.ok) throw new Error(`HTTP ${binResp.status} fetching ${spec.edges.file}`);
  const buf = await binResp.arrayBuffer();
  const m = spec.edges.count;
  if (buf.byteLength !== 3 * m + 4 * m) throw new Error(`${spec.edges.file}: ${buf.byteLength} bytes, expected ${7 * m}`);
  if (crypto.subtle) {
    const digest = await crypto.subtle.digest('SHA-256', buf);
    const hex = [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
    if (hex !== spec.edges.sha256) throw new Error(`${spec.edges.file}: sha256 mismatch; re-run the export`);
  }
  // layout: u16 kc_index[m] | u8 mbon_index[m] | f32 w[m]  (f32 block is 4-byte aligned only if 3m % 4 == 0, so copy)
  const kcIndex = new Uint16Array(buf, 0, m);
  const mbonIndex = new Uint8Array(buf, 2 * m, m);
  const w = new Float32Array(buf.slice(3 * m, 7 * m));
  log(`weights "${id}": ${m.toLocaleString()} plastic KC→MBON edges (${(buf.byteLength / 1e3).toFixed(0)} KB) in ${(performance.now() - t0).toFixed(0)} ms`);
  return { spec, kcIndex, mbonIndex, w };
}

export async function loadWeightManifest(base) {
  const r = await fetch(`${base}manifest.json`, { cache: 'no-store' });
  if (!r.ok) throw new Error(`weights/manifest.json not found (HTTP ${r.status})`);
  return r.json();
}

export class MBPolicy {
  /** @param {{spec:object,kcIndex:Uint16Array,mbonIndex:Uint8Array,w:Float32Array}} set */
  constructor(set) {
    const spec = set.spec;
    this.set = set;
    this.spec = spec;
    this.id = spec.id;
    this.actions = spec.actions.slice();
    this.pools = this.actions.map((a) => Uint32Array.from(spec.pools[a].idx));
    this.poolCells = this.actions.map((a) => spec.pools[a].cells);
    this.sizes = Float32Array.from(this.pools.map((p) => p.length));
    this.oracle = new Oracle({ jumpDistance: spec.decision.jump_distance ?? 5 });
    // odor map: channel -> ORN cells, each driven with amp * value while the turn lasts
    this.odorAmp = spec.odor.amp;
    this.odorCells = spec.odor.cells.map((c) => Uint32Array.from(c));
    if (this.odorCells.length !== N_CHANNELS) throw new Error(`weight set has ${this.odorCells.length} odor channels, oracle.js has ${N_CHANNELS}`);
    this.features = new Float32Array(N_CHANNELS);
    this.odor = null;                      // {idx, amt} for the current turn, or null when every channel is 0
    // decision (mb_policy.MBPolicy.decide with temperature 0, epsilon 0)
    const d = spec.decision;
    this.centerScores = !!d.center_scores;
    this.scoreAlpha = d.score_mean_alpha;
    this.revealMargin = d.reveal_margin || 0;
    this.maskReveal = !!d.mask_reveal;
    this.scoreMean = new Float32Array(this.actions.length);
    this.scoreMeanReady = false;
    if (this.centerScores && d.score_mean) { this.scoreMean.set(d.score_mean); this.scoreMeanReady = true; }
    // UI-compatible fields (same names as decoder.Decoder)
    this.lastRates = new Float32Array(this.actions.length);
    this.lastScores = new Float32Array(this.actions.length);
    this.running = new Float32Array(this.actions.length);   // score mean expressed in Hz per cell
    this.idle = new Float32Array(this.actions.length);
    this.lastProbs = new Float32Array(this.actions.length).fill(1 / this.actions.length);
    this.turnsSeen = 0;
    this.applied = null;
  }

  // ---------------------------------------------------------------- model changes on the GPU graph
  /**
   * Apply the fly-mb model changes to the simulator: patch the resident weight buffer row by row
   * (in-edge CSR by postsynaptic neuron, so every change is a contiguous segment) and upload the
   * per-neuron bias. `arrays` are the CPU copies the sim was built from (in_indptr, in_indices,
   * in_weights); in_weights is patched in place so later reads see the same graph as the GPU.
   */
  async apply(sim, arrays) {
    const t0 = performance.now();
    const { spec, kcIndex, mbonIndex, w } = this.set;
    const n = sim.n, indptr = arrays.in_indptr, indices = arrays.in_indices, weights = arrays.in_weights;
    const isKC = new Uint8Array(n), isPN = new Uint8Array(n);
    for (const j of spec.kc) isKC[j] = 1;
    for (const j of spec.pn) isPN[j] = 1;
    const km = spec.kc_model;
    const rows = [];
    let kc2kc = 0, pn2kc = 0;
    // (1) KC->KC and PN->KC edge gains on every Kenyon cell's in-edge row
    for (const j of spec.kc) {
      const a = indptr[j], b = indptr[j + 1];
      for (let p = a; p < b; p++) {
        const i = indices[p];
        if (isKC[i]) { weights[p] *= km.kc_kc_gain; kc2kc++; }
        else if (isPN[i]) { weights[p] *= km.pn_kc_gain; pn2kc++; }
      }
      rows.push(j);
    }
    // (2) trained KC->MBON weights: for each pool MBON, a map pre KC -> trained weight
    const mbon = spec.mbon, kc = spec.kc;
    const perPost = mbon.map(() => new Map());
    for (let e = 0; e < w.length; e++) perPost[mbonIndex[e]].set(kc[kcIndex[e]], w[e]);
    let replaced = 0, changed = 0, missing = 0, silenced = 0;
    for (let q = 0; q < mbon.length; q++) {
      const j = mbon[q], a = indptr[j], b = indptr[j + 1], map = perPost[q];
      let seen = 0;
      for (let p = a; p < b; p++) {
        const i = indices[p];
        if (!isKC[i]) continue;
        const nw = map.get(i);
        if (nw === undefined) { missing++; continue; }
        if (Math.abs(Math.abs(nw) - Math.abs(weights[p])) > 1e-7) changed++;
        if (nw === 0) silenced++;                        // w_min_ratio 0: the synapse was trained to silence
        weights[p] = nw; replaced++; seen++;
      }
      if (seen !== map.size) missing += map.size - seen;
      rows.push(j);
    }
    if (replaced !== w.length || missing) {
      throw new Error(`weight set "${this.id}" does not match this graph export: ${replaced} of ${w.length} plastic edges found, ${missing} missing (re-run both exports)`);
    }
    await sim.patchWeightRows(rows, arrays);
    // (3) per-neuron bias (added to the external drive every step by step.wgsl)
    const bias = new Float32Array(n);
    for (const j of spec.kc) bias[j] += km.kc_bias;
    for (const j of spec.pn) bias[j] += km.pn_bias;
    await sim.setBias(bias);
    await sim.setPools(this.pools);
    this.applied = { kc2kc, pn2kc, replaced, changed, silenced, rows: rows.length, ms: performance.now() - t0 };
    return this.applied;
  }

  // ---------------------------------------------------------------- per-turn interface
  /** Start of a turn: the helper's facts for this board -> odor current for every step of the turn. */
  beginTurn(visible, rows, cols, cursor, nMines) {
    const f = this.oracle.features(visible, rows, cols, cursor, nMines);
    this.features = f;
    this.turnsSeen++;
    let m = 0;
    for (let k = 0; k < N_CHANNELS; k++) if (f[k] > 0) m += this.odorCells[k].length;
    if (!m) { this.odor = null; return f; }
    const idx = new Uint32Array(m), amt = new Float32Array(m);
    let o = 0;
    for (let k = 0; k < N_CHANNELS; k++) {
      if (f[k] <= 0) continue;
      const cells = this.odorCells[k], v = this.odorAmp * f[k];
      idx.set(cells, o); amt.fill(v, o, o + cells.length); o += cells.length;
    }
    this.odor = { idx, amt };
    return f;
  }

  /** End of a game (mb_policy.end_game): no odor while the screen is blank. */
  endGame() { this.odor = null; this.features = new Float32Array(N_CHANNELS); }

  /** Merge the retina drive for one step with the odor current (both index lists are disjoint). */
  stepDrive(retina) {
    const od = this.odor;
    if (!od) return retina;
    if (!retina) return { idx: od.idx, amt: od.amt };
    const idx = new Uint32Array(retina.idx.length + od.idx.length), amt = new Float32Array(idx.length);
    idx.set(retina.idx); amt.set(retina.amt);
    idx.set(od.idx, retina.idx.length); amt.set(od.amt, retina.idx.length);
    return { idx, amt, loom: retina.loom };
  }

  rates(counts, steps, dt) {
    const r = new Float32Array(this.actions.length);
    for (let k = 0; k < r.length; k++) r[k] = counts[k] / this.sizes[k] / Math.max(steps * dt, 1e-9);
    return r;
  }

  setIdle(idle) { this.idle = Float32Array.from(idle); }

  /** mb_policy.MBPolicy.decide (temperature 0): centred per-cell pool counts, reveal margin, stable argmax. */
  decide(rng, counts, steps, dt) {
    const K = this.actions.length;
    const raw = new Float32Array(K);
    for (let k = 0; k < K; k++) raw[k] = counts[k] / this.sizes[k];
    let scores;
    if (this.centerScores) {
      if (!this.scoreMeanReady) { this.scoreMean.set(raw); this.scoreMeanReady = true; }
      scores = new Float32Array(K);
      for (let k = 0; k < K; k++) { scores[k] = raw[k] - this.scoreMean[k]; this.scoreMean[k] += this.scoreAlpha * (raw[k] - this.scoreMean[k]); }
    } else {
      scores = raw;
    }
    this.lastRates = this.rates(counts, steps, dt);
    this.lastScores = Float32Array.from(scores);
    for (let k = 0; k < K; k++) this.running[k] = this.scoreMean[k] / Math.max(steps * dt, 1e-9);
    const kRev = this.actions.indexOf('reveal');
    const valid = new Array(K).fill(true);
    if (this.maskReveal && (this.features[CH.cursor_revealed] > 0 || this.features[CH.cursor_mine] > 0)) valid[kRev] = false;
    const sc = Array.from(scores, (s, k) => (valid[k] ? s : -Infinity));
    let order = [...Array(K).keys()].sort((a, b) => sc[b] - sc[a] || a - b);   // stable argsort of -scores
    let dropped = false;
    if (this.revealMargin > 0 && order[0] === kRev && sc[kRev] - sc[order[1]] < this.revealMargin) { order = order.slice(1); dropped = true; }
    let best = [];
    for (let k = 0; k < K; k++) if (sc[k] === sc[order[0]]) best.push(k);
    if (dropped) best = best.filter((k) => k !== kRev);
    this.lastProbs.fill(0);
    for (const k of best) this.lastProbs[k] = 1 / best.length;
    return this.actions[rng.choice(best)];
  }

  /** What the Python /state reports as "plasticity" (weight statistics of the loaded set). */
  summary() {
    const s = this.spec.stats;
    return {
      kc_mbon_edges: s.plastic_edges, edges_changed: s.edges_changed, mean_weight_ratio: s.mean_weight_ratio,
      min_weight_ratio: s.min_weight_ratio, max_weight_ratio: s.max_weight_ratio, pool_ratio: s.pool_ratio,
      games_trained: this.spec.games_trained, attached: !!this.applied, learning: false,
    };
  }
}
