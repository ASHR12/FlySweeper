// Descending-neuron pool spike counts -> a Minesweeper action, a port of flysweeper/decoder.py
// (mode argmax, baseline "running"). Pool membership comes from model.json; the counts themselves
// are accumulated on the GPU (stats[0..6)) over the turn window.
//
//     rate_k  = counts_k / |pool_k| / (steps * dt)                (Hz per cell)
//     score_k = max(rate_k - running_k, 0);  running <- (1-a) running + a rate
//     action  = argmax score (seeded random tie-break), or "hold" if every score is 0

export class Decoder {
  constructor(model) {
    const d = model.decoder;
    this.actions = d.actions.slice();
    this.pools = this.actions.map((a) => Uint32Array.from(d.pools[a].idx));
    this.poolCells = this.actions.map((a) => d.pools[a].cells);
    this.sizes = Float32Array.from(this.pools.map((p) => p.length));
    this.alpha = d.running_alpha;
    this.idle = new Float32Array(this.actions.length);
    this.running = new Float32Array(this.actions.length);
    this.lastRates = new Float32Array(this.actions.length);
    this.lastScores = new Float32Array(this.actions.length);
  }

  /** Spikes per cell per second for each pool over a window of `steps` steps. */
  rates(counts, steps, dt) {
    const r = new Float32Array(this.actions.length);
    for (let k = 0; k < r.length; k++) r[k] = counts[k] / this.sizes[k] / Math.max(steps * dt, 1e-9);
    return r;
  }

  setIdle(idle) {
    this.idle = Float32Array.from(idle);
    this.running = Float32Array.from(idle);
  }

  scores(counts, steps, dt) {
    const r = this.rates(counts, steps, dt);
    const s = new Float32Array(r.length);
    for (let k = 0; k < r.length; k++) {
      s[k] = Math.max(r[k] - this.running[k], 0);
      this.running[k] = (1 - this.alpha) * this.running[k] + this.alpha * r[k];
    }
    this.lastRates = r;
    this.lastScores = s;
    return s;
  }

  decide(rng, counts, steps, dt) {
    const s = this.scores(counts, steps, dt);
    let sum = 0, max = -Infinity;
    for (const x of s) { sum += x; if (x > max) max = x; }
    if (sum <= 0) return 'hold';
    const best = [];
    for (let k = 0; k < s.length; k++) if (s[k] === max) best.push(k);
    return this.actions[rng.choice(best)];
  }
}
