// Population-statistics checks against the Python simulator (flysweeper.sim / flysweeper.probe).
// The RNG streams differ, so we compare rates, not spike trains:
//
//   idle:  500 blind steps after 100 warm-up steps -> mean firing rate (Python: ~2.3 Hz, ~7,600 spikes/step)
//   loom:  pulse left LC4+LPLC2 at amplitude 0.8 every other step -> left DNp01 ~25 Hz, right DNp01 ~1 Hz
//
//   board: a mid-game board with the cursor on the far left column -> L1-L3 lamina rate (Python probe:
//          0.34 Hz blind -> ~3.0 Hz with the board; whole-brain mean 2.28 -> ~2.7 Hz)
//
// All run on the live simulator (state is continued, not reset) at full speed.

import { Minesweeper } from './minesweeper.js';
import { LUM } from './encoder.js';

export async function idleTest(sim, decoder, { warmup = 100, steps = 500, onProgress = () => {} } = {}) {
  const dt = sim.params.dt, n = sim.n;
  let poolCounts = null, total = 0, done = 0;
  const perStep = [];
  for (const [count, measured] of [[warmup, false], [steps, true]]) {
    let left = count, first = true;
    while (left > 0) {
      const k = Math.min(left, sim.maxBatch);
      const res = await sim.runSteps(new Array(k).fill(null), { resetPools: measured && first });
      first = false; left -= k; done += k;
      if (measured) { for (const s of res.spikesPerStep) { total += s; perStep.push(s); } poolCounts = res.poolCounts; }
      onProgress(done / (warmup + steps));
    }
  }
  const poolIdleHz = decoder.rates(poolCounts, steps, dt);
  return {
    steps, warmup,
    mean_rate_hz: total / (steps * n * dt),
    spikes_per_step: total / steps,
    spikes_per_step_min: Math.min(...perStep), spikes_per_step_max: Math.max(...perStep),
    pool_idle_hz: Object.fromEntries(decoder.actions.map((a, i) => [a, poolIdleHz[i]])),
  };
}

export async function loomTest(sim, model, encoder, decoder, { warmup = 100, steps = 500, amp = 0.8, onProgress = () => {} } = {}) {
  const dt = sim.params.dt;
  const jump = model.decoder.pools.jump;
  const dnp01 = { L: [], R: [] };
  jump.idx.forEach((j, i) => { const side = jump.cells[i].split('/')[1]; if (dnp01[side]) dnp01[side].push(j); });
  const probes = [...decoder.pools, Uint32Array.from(dnp01.L), Uint32Array.from(dnp01.R), encoder.loom.L, encoder.loom.R];
  const base = decoder.pools.length;
  await sim.setPools(probes);
  const results = {};
  const conds = ['silent', 'loom-left'];
  let done = 0;
  try {
    for (const cond of conds) {
      let poolCounts = null;
      for (const [count, measured] of [[warmup, false], [steps, true]]) {
        let left = count, first = true;
        while (left > 0) {
          const k = Math.min(left, sim.maxBatch);
          const drives = [];
          for (let s = 0; s < k; s++) {
            const step = sim.stepCount + s;
            drives.push(cond === 'loom-left' && step % 2 === 0 ? { idx: encoder.loom.L, amt: new Float32Array(encoder.loom.L.length).fill(amp) } : null);
          }
          const res = await sim.runSteps(drives, { resetPools: measured && first });
          first = false; left -= k; done += k;
          if (measured) poolCounts = res.poolCounts;
          onProgress(done / (conds.length * (warmup + steps)));
        }
      }
      const hz = (slot, size) => poolCounts[slot] / size / (steps * dt);
      results[cond] = {
        DNp01_L_hz: hz(base, dnp01.L.length), DNp01_R_hz: hz(base + 1, dnp01.R.length),
        LC4_LPLC2_L_hz: hz(base + 2, encoder.loom.L.length), LC4_LPLC2_R_hz: hz(base + 3, encoder.loom.R.length),
        jump_pool_hz: hz(5, decoder.sizes[5]),
      };
    }
  } finally {
    await sim.setPools(decoder.pools);
  }
  return { amp, steps, warmup, ...results };
}

export async function boardTest(sim, model, encoder, decoder, { warmup = 100, steps = 500, cursor = [4, 0], onProgress = () => {} } = {}) {
  const dt = sim.params.dt, n = sim.n, g = model.game;
  // same recipe as flysweeper.probe.sample_board (the RNG differs, so the exact board differs)
  const game = new Minesweeper(g.rows, g.cols, g.mines, { seed: 7 });
  game.reveal(Math.floor(g.rows / 2), Math.floor(g.cols / 2));
  for (const [r, c] of [[0, 0], [g.rows - 1, g.cols - 1], [0, g.cols - 1]]) if (!game.over) game.reveal(r, c);
  const visible = game.visible();
  const lamina = [], r7r8 = [];
  for (let k = 0; k < encoder.nTargets; k++) (encoder.chan[k] === LUM ? lamina : r7r8).push(encoder.idx[k]);
  const base = decoder.pools.length;
  await sim.setPools([...decoder.pools, Uint32Array.from(lamina), Uint32Array.from(r7r8)]);
  let poolCounts = null, total = 0, done = 0;
  encoder.reset();
  try {
    for (const [count, measured] of [[warmup, false], [steps, true]]) {
      let left = count, first = true;
      while (left > 0) {
        const k = Math.min(left, sim.maxBatch);
        const drives = [];
        for (let s = 0; s < k; s++) {
          const d = encoder.drive(visible, cursor, sim.stepCount + s);   // retina only, no looming (like probe.run)
          drives.push({ idx: Uint32Array.from(d.idx), amt: Float32Array.from(d.amt) });
        }
        const res = await sim.runSteps(drives, { resetPools: measured && first });
        first = false; left -= k; done += k;
        if (measured) { for (const s of res.spikesPerStep) total += s; poolCounts = res.poolCounts; }
        onProgress(done / (warmup + steps));
      }
    }
  } finally {
    await sim.setPools(decoder.pools);
  }
  const hz = (slot, size) => poolCounts[slot] / size / (steps * dt);
  return {
    steps, warmup, cursor, revealed: game.safeRevealed,
    mean_rate_hz: total / (steps * n * dt), spikes_per_step: total / steps,
    L1_L3_hz: hz(base, lamina.length), R7_R8_hz: hz(base + 1, r7r8.length),
    pools_hz: Object.fromEntries(decoder.actions.map((a, i) => [a, hz(i, decoder.sizes[i])])),
  };
}
