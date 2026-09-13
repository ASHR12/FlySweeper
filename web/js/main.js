// FlySweeper in the browser: load the exported MaleCNS graph, simulate it on the GPU, let the fly
// play Minesweeper on the left board while the human plays the same mines on the right board.

import { loadData, loadShaders, clearDataCache } from './data.js';
import { FlySim, WebGPUUnavailable } from './gpu.js';
import { Minesweeper, HIDDEN } from './minesweeper.js';
import { Encoder } from './encoder.js';
import { Decoder } from './decoder.js';
import { RNG } from './rng.js';
import { BrainMap, harmonise } from './brainmap.js';
import { BoardView } from './boardview.js';
import { idleTest, loomTest, boardTest } from './tests.js';

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const GAME_OVER_HOLD_MS = 2500;   // finished board stays on screen this long (same as server.py --game-over-hold)
const query = new URLSearchParams(location.search);

// FEATURE FLAG: the human-playable "YOU" board (same mines, click to reveal / right-click to flag,
// safe first click, own scoreboard row). Hidden by default so the whole UI fits one screen; open the
// page with `?human=1` to show it. All the human game logic below stays in place and keeps working;
// when hidden, the human board is simply never drawn and its DOM (class "human-only") is display:none.
const SHOW_HUMAN_BOARD = query.get('human') === '1';
document.body.classList.toggle('no-human', !SHOW_HUMAN_BOARD);

const app = {
  model: null, sim: null, encoder: null, decoder: null, brainMap: null, flyView: null, youView: null,
  rng: null,
  seed0: Number(query.get('seed') ?? 1000),
  gameNumber: 0,
  layout: null,
  pace: { speed: Number(query.get('speed') ?? 1), paused: false },
  fly: { game: null, cursor: [4, 4], lastAction: 'start', lastResult: '', turn: 0, phase: 'loading', danger: 0, actions: {}, abort: false },
  human: { game: null, started: false, t0: null, t1: null },
  totals: { fly: { games: 0, wins: 0, safe: 0, abandoned: 0, last: null }, human: { games: 0, wins: 0, safe: 0, abandoned: 0, last: null } },
  stats: { msPerStep: null, msSource: '', rtf: 0, spikesLastStep: 0, lastBatchEnd: null, batches: 0, samples: [] },
  idleRates: null,
  events: [],
  history: [],          // finished fly games: { game, outcome, safe, turns } (feeds the sparkline)
  suspendRequested: false, parked: false, newGameRequested: false,
};

// ------------------------------------------------------------------------------------------
// loading
// ------------------------------------------------------------------------------------------
function overlayLog(msg) {
  const el = $('load-log');
  el.textContent += msg + '\n';
  el.scrollTop = el.scrollHeight;
  console.log('[flysweeper] ' + msg);
}

function showError(title, detail, hints = []) {
  $('load-progress-wrap').style.display = 'none';
  const box = $('load-error');
  box.style.display = 'block';
  box.innerHTML = `<b>${title}</b><div>${escapeHtml(detail)}</div>` + (hints.length ? `<ul>${hints.map((h) => `<li>${h}</li>`).join('')}</ul>` : '');
}

function escapeHtml(s) { return String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c])); }

async function boot() {
  if (!('gpu' in navigator)) {
    showError('WebGPU is not available in this browser.', 'navigator.gpu is undefined.', [
      'Use Chrome 113+ or Edge 113+ on macOS (WebGPU is on by default), or Safari 18+.',
      'Serve the page from http://localhost or http://127.0.0.1 (WebGPU needs a secure context).',
      'Check chrome://gpu for "WebGPU: Hardware accelerated".',
    ]);
    return;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new WebGPUUnavailable('navigator.gpu.requestAdapter() returned null (no compatible GPU adapter).');
    if (query.has('clearcache')) { await clearDataCache(); overlayLog('cache cleared'); }
    $('load-title').textContent = 'Loading the connectome';
    const shaders = await loadShaders('shaders/');
    const t0 = performance.now();
    const { manifest, arrays, fromCache } = await loadData('data/', {
      log: overlayLog,
      useCache: !query.has('nocache'),
      onProgress: ({ done, total, name }) => {
        $('load-bar').style.width = `${(100 * done / total).toFixed(1)}%`;
        $('load-text').textContent = `${(done / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB · ${name}`;
      },
    });
    overlayLog(`data ready in ${((performance.now() - t0) / 1000).toFixed(1)} s (${fromCache}/${Object.keys(manifest.files).length} files from cache)`);
    const model = arrays['model.json'];
    app.model = model;
    $('load-title').textContent = 'Uploading to the GPU';
    $('load-text').textContent = `${model.n_neurons.toLocaleString()} neurons · ${model.n_edges_active.toLocaleString()} active synaptic edges`;
    app.sim = await FlySim.create({
      model,
      arrays: {
        in_indptr: arrays['in_indptr.u32'], in_indices: arrays['in_indices.u32'], in_weights: arrays['in_weights.f32'],
        order: arrays['order.u32'], pool_id: arrays['pool_id.u32'], plot_idx: arrays['plot_idx.u32'],
      },
      shaders,
      seed: Number(query.get('simseed') ?? 0),
      log: overlayLog,
    });
    app.encoder = new Encoder(model);
    app.decoder = new Decoder(model);
    app.rng = new RNG(app.seed0 * 7919 + 17);
    const plotIdx = arrays['plot_idx.u32'], typeId = arrays['type_id.u16'];
    const typeOfPlotted = new Uint16Array(plotIdx.length);
    for (let i = 0; i < plotIdx.length; i++) typeOfPlotted[i] = typeId[plotIdx[i]];
    app.brainMap = new BrainMap($('brain'), arrays['soma_uv.f32'], arrays['region.u8'], model.regions, typeOfPlotted, model.type_names);
    setupUi(model);
    $('overlay').style.display = 'none';
    requestAnimationFrame(frame);
    flyLoop().catch((e) => { console.error(e); addEvent(`fly loop crashed: ${e.message}`); });
  } catch (e) {
    console.error(e);
    if (e instanceof WebGPUUnavailable) {
      showError('WebGPU could not be initialised.', e.message, [
        'Chrome on macOS: chrome://gpu should list WebGPU as hardware accelerated; try chrome://flags/#enable-unsafe-webgpu if it is not.',
        'The graph needs a storage buffer of ~100 MB; most desktop GPUs allow this.',
      ]);
    } else {
      showError('Failed to start.', e.message || String(e), [
        'Is web/data/ populated? Run <code>./.venv/bin/python -m flysweeper.export_web</code> in the repo.',
        'Reload with <code>?clearcache=1</code> if a partial download is cached.',
      ]);
    }
  }
}

// ------------------------------------------------------------------------------------------
// pacing
// ------------------------------------------------------------------------------------------
const pacer = {
  next: 0,
  reset() { this.next = performance.now(); app.stats.lastBatchEnd = null; },
  batchSize(remaining) {
    const s = app.pace.speed;
    const b = s === 0 ? app.sim.maxBatch : Math.max(1, Math.round(s * 50 * 0.04));
    return Math.max(1, Math.min(b, remaining));
  },
  async before(k) {
    const s = app.pace.speed;
    const now = performance.now();
    if (s === 0) { this.next = now; return; }
    if (this.next < now - 250) this.next = now;
    const wait = this.next - now;
    if (wait > 1) await sleep(wait);
    this.next += (k * app.model.sim.dt * 1000) / s;
  },
};

async function waitIfBlocked({ allowSuspend = true } = {}) {
  const blocked = () => app.pace.paused || (allowSuspend && app.suspendRequested);
  if (!blocked()) return;
  app.parked = allowSuspend;
  while (blocked()) await sleep(20);
  app.parked = false;
  pacer.reset();
}

/** Run a batch of steps through the pacer; drives[s] = {idx, amt} | null. */
async function runBatch(drives, opts = {}) {
  await waitIfBlocked({ allowSuspend: false });
  await pacer.before(drives.length);
  const res = await app.sim.runSteps(drives, opts);
  noteBatch(res);
  return res;
}

function noteBatch(res) {
  const st = app.stats, dt = app.model.sim.dt, now = performance.now();
  const ms = res.gpuMs != null ? res.gpuMs / res.k : res.wallMs / res.k;
  st.msSource = res.gpuMs != null ? 'gpu' : 'wall';
  st.msPerStep = st.msPerStep == null ? ms : 0.9 * st.msPerStep + 0.1 * ms;
  // realtime factor = simulated time / wall time over the last ~1 s of batches
  if (st.lastBatchEnd == null) st.samples = [];
  st.samples.push({ t: now, step: app.sim.stepCount });
  while (st.samples.length > 2 && now - st.samples[1].t > 1000) st.samples.shift();
  const first = st.samples[0];
  if (st.samples.length > 1 && now > first.t) st.rtf = ((app.sim.stepCount - first.step) * dt * 1000) / (now - first.t);
  st.lastBatchEnd = now;
  st.spikesLastStep = res.spikesPerStep[res.k - 1];
  st.batches++;
}

/** Run `fn` with exclusive use of the simulator (the fly loop parks at its next turn boundary). */
async function exclusive(fn) {
  app.suspendRequested = true;
  const t0 = performance.now();
  while (!app.parked) {
    if (performance.now() - t0 > 15000) { app.suspendRequested = false; throw new Error('fly loop did not reach a turn boundary (is it paused mid-turn? unpause first)'); }
    await sleep(10);
  }
  try { return await fn(); } finally { app.suspendRequested = false; }
}

// ------------------------------------------------------------------------------------------
// the fly
// ------------------------------------------------------------------------------------------
function blankBoard() { return new Int8Array(app.model.game.rows * app.model.game.cols).fill(HIDDEN); }

async function calibrateIdle() {
  const g = app.model.game, dt = app.model.sim.dt;
  app.fly.phase = 'calibrating';
  const total = g.idle_warmup_steps + g.idle_steps;
  let done = 0, counts = null;
  for (const [count, measured] of [[g.idle_warmup_steps, false], [g.idle_steps, true]]) {
    let left = count, first = true;
    while (left > 0) {
      const k = Math.min(left, app.sim.maxBatch);
      const res = await app.sim.runSteps(new Array(k).fill(null), { resetPools: measured && first });
      noteBatch(res);
      first = false; left -= k; done += k;
      if (measured) counts = res.poolCounts;
      app.fly.lastResult = `calibrating idle baseline ${done}/${total}`;
    }
  }
  const idle = app.decoder.rates(counts, g.idle_steps, dt);
  app.decoder.setIdle(idle);
  app.idleRates = idle;
  addEvent('idle baseline (Hz/cell): ' + app.decoder.actions.map((a, i) => `${a} ${idle[i].toFixed(1)}`).join(', '));
}

function newSharedGame() {
  const g = app.model.game;
  app.gameNumber += 1;
  const seed = app.seed0 + app.gameNumber - 1;
  const center = [Math.floor(g.rows / 2), Math.floor(g.cols / 2)];
  app.layout = Minesweeper.makeLayout(g.rows, g.cols, g.mines, center[0], center[1], new RNG(seed));
  app.fly.game = new Minesweeper(g.rows, g.cols, g.mines, { seed, layout: app.layout });
  app.human.game = new Minesweeper(g.rows, g.cols, g.mines, { seed, layout: app.layout });
  app.human.started = false; app.human.t0 = null; app.human.t1 = null; app.human.recorded = false; app.human.relocation = null;
  app.fly.cursor = center; app.fly.turn = 0; app.fly.lastAction = 'start'; app.fly.lastResult = ''; app.fly.actions = {}; app.fly.abort = false;
  app.fly.danger = 0; app.fly.outcome = null;
  app.encoder.reset();                                   // prev_lum -> 0 (playFlyGame resets again before settling)
  app.decoder.lastRates.fill(0); app.decoder.lastScores.fill(0);   // decision window display; GPU pool counters are cleared at the first turn
  app.newGameRequested = false;
  addEvent(`game ${app.gameNumber} · seed ${seed} · same ${g.mines} mines on both boards`);
}

async function runBlankBoard(steps, { allowSuspend = true } = {}) {
  const blank = blankBoard();
  let left = steps;
  while (left > 0) {
    await waitIfBlocked({ allowSuspend });
    const k = pacer.batchSize(left);
    const drives = [];
    for (let s = 0; s < k; s++) drives.push(app.encoder.stepDrive(blank, app.fly.cursor, app.sim.stepCount + s));
    await runBatch(drives);
    left -= k;
  }
}

async function playFlyGame() {
  const g = app.model.game, dt = app.model.sim.dt, fly = app.fly, game = fly.game;
  app.encoder.reset();
  fly.phase = 'settle';
  await runBlankBoard(g.settle_steps);
  fly.phase = 'playing';
  let outcome = 'timeout';
  for (let turn = 0; turn < g.max_turns; turn++) {
    await waitIfBlocked({ allowSuspend: true });
    if (fly.abort) { outcome = 'aborted'; break; }
    fly.turn = turn;
    const visible = game.visible();
    const base = app.sim.stepCount;
    const drives = [];
    for (let s = 0; s < g.turn_steps; s++) drives.push(app.encoder.stepDrive(visible, fly.cursor, base + s));
    fly.danger = app.encoder.lastDanger;
    let left = g.turn_steps, first = true, res = null;
    while (left > 0) {
      await waitIfBlocked({ allowSuspend: false });
      const k = pacer.batchSize(left);
      res = await runBatch(drives.slice(g.turn_steps - left, g.turn_steps - left + k), { resetPools: first });
      first = false; left -= k;
    }
    const action = app.decoder.decide(app.rng, res.poolCounts, g.turn_steps, dt);
    fly.actions[action] = (fly.actions[action] || 0) + 1;
    applyFlyAction(action);
    if (game.over) { outcome = game.won ? 'won' : 'lost'; break; }
  }
  fly.outcome = outcome;
  fly.phase = 'over';
  const t = app.totals.fly;
  if (outcome !== 'aborted') {
    t.games += 1; t.wins += outcome === 'won' ? 1 : 0; t.safe += game.safeRevealed;
    t.last = { outcome, safe: game.safeRevealed, turns: fly.turn + 1 };
    app.history.push({ game: app.gameNumber, outcome, safe: game.safeRevealed, turns: fly.turn + 1 });
    if (app.history.length > 200) app.history.splice(0, app.history.length - 200);
    addEvent(`fly ${outcome}: ${game.safeRevealed}/${game.totalSafe} safe cells in ${fly.turn + 1} turns`);
  } else {
    // "new game" pressed mid-game: counted separately, excluded from games/wins/mean
    t.abandoned += 1;
    t.last = { outcome: 'abandoned', safe: game.safeRevealed, turns: fly.turn };
    addEvent(`fly game abandoned after ${fly.turn} turns with ${game.safeRevealed}/${game.totalSafe} safe cells (new game pressed; not counted in the mean)`);
  }
}

function applyFlyAction(action) {
  const fly = app.fly, game = fly.game, g = app.model.game;
  const [r, c] = fly.cursor;
  let text = action;
  if (action === 'up') fly.cursor = [Math.max(0, r - 1), c];
  else if (action === 'down') fly.cursor = [Math.min(g.rows - 1, r + 1), c];
  else if (action === 'left') fly.cursor = [r, Math.max(0, c - 1)];
  else if (action === 'right') fly.cursor = [r, Math.min(g.cols - 1, c + 1)];
  else if (action === 'jump') {
    const hidden = game.hiddenCells();
    if (hidden.length) fly.cursor = hidden[app.rng.int(hidden.length)];
    text += ` from (${r},${c})`;
  } else if (action === 'reveal') {
    const { result, newly } = game.reveal(r, c);
    text += ` → ${result}` + (newly ? ` (+${newly})` : '');
    fly.lastResult = result;
  }
  fly.lastAction = action;
  if (action !== 'hold' || fly.turn % 10 === 0) addEvent(text, true);
}

/**
 * Keep the finished board on screen (phase 'over', brain ticking on a blank screen) so the end-of-game
 * animation can be seen. Like the Python server's --game-over-hold, only honoured when paced (not at max).
 */
async function holdGameOver() {
  const fly = app.fly;
  if (fly.outcome === 'aborted') return;
  const t0 = performance.now();
  while (performance.now() - t0 < GAME_OVER_HOLD_MS) {
    if (app.newGameRequested) break;
    if (app.pace.speed === 0 && !app.pace.paused) break;
    await runBlankBoard(pacer.batchSize(5));
  }
}

async function waitForNextGame() {
  const fly = app.fly, human = app.human;
  fly.phase = 'waiting';
  for (;;) {
    const humanDone = !human.started || human.game.over;
    if (app.newGameRequested || humanDone) break;
    await runBlankBoard(pacer.batchSize(5));
  }
}

async function flyLoop() {
  await calibrateIdle();
  for (;;) {
    newSharedGame();
    await playFlyGame();
    await holdGameOver();
    await waitForNextGame();
    finishHumanBoard();
  }
}

// ------------------------------------------------------------------------------------------
// the human
// ------------------------------------------------------------------------------------------
/**
 * Safe first click for the human on a shared layout: if the first reveal lands on a mine, move that
 * mine to a cell that is hidden on BOTH boards and not adjacent to the click (preferring cells off
 * the fly's revealed frontier so its displayed numbers do not change), on both boards. If the fly
 * has already finished its game, only the human board is changed (the fly's record stays intact).
 */
function relocateMineForFirstClick(r, c) {
  const h = app.human, human = h.game, fly = app.fly.game, g = app.model.game;
  const from = r * g.cols + c;
  if (!human.mines[from]) return null;
  const flyLive = fly && !fly.over;
  const near = new Set([from]);
  for (const [rr, cc] of human.neighbors(r, c)) near.add(rr * g.cols + cc);
  const ok = (i) => !human.mines[i] && !near.has(i) && !human.revealed[i] && !human.flagged[i] && (!flyLive || (!fly.revealed[i] && !fly.flagged[i]));
  let candidates = [];
  for (let i = 0; i < g.rows * g.cols; i++) if (ok(i)) candidates.push(i);
  if (!candidates.length) return null;
  const quiet = candidates.filter((i) => !(flyLive && fly.onFrontier(i)) && !human.onFrontier(i));
  if (quiet.length) candidates = quiet;
  const to = candidates[new RNG(human.seed * 31 + from).int(candidates.length)];
  human.moveMine(from, to);
  if (flyLive) fly.moveMine(from, to);
  const rc = (i) => `(${Math.floor(i / g.cols)},${i % g.cols})`;
  const note = `first click on a mine: moved it ${rc(from)} → ${rc(to)}${flyLive ? ' on both boards' : ' (your board only; the fly had already finished)'}${quiet.length ? '' : ', next to revealed numbers'}`;
  h.relocation = note;
  addEvent('you: ' + note);
  return { from, to, both: flyLive };
}

function humanReveal(r, c) {
  const h = app.human, game = h.game;
  if (!game || game.over) return;
  if (!h.started) { h.started = true; h.t0 = performance.now(); }
  if (game.safeRevealed === 0 && !game.revealed[r * game.cols + c] && !game.flagged[r * game.cols + c]) relocateMineForFirstClick(r, c);
  const { result, newly } = game.reveal(r, c);
  if (result !== 'noop') addEvent(`you: reveal (${r},${c}) → ${result}` + (newly ? ` (+${newly})` : ''), true);
  if (game.over) finishHumanBoard();
}

function humanFlag(r, c) {
  const h = app.human, game = h.game;
  if (!game || game.over) return;
  if (!h.started) { h.started = true; h.t0 = performance.now(); }
  game.toggleFlag(r, c);
}

function finishHumanBoard() {
  const h = app.human, game = h.game;
  if (!game || h.recorded) return;
  const t = app.totals.human;
  if (!game.over) {
    // an unfinished board with at least one reveal is recorded as abandoned (excluded from the mean)
    if (game.safeRevealed > 0) { h.recorded = true; t.abandoned += 1; addEvent(`you abandoned the board with ${game.safeRevealed}/${game.totalSafe} safe cells (not counted in the mean)`); }
    return;
  }
  h.recorded = true;
  h.t1 = performance.now();
  const outcome = game.won ? 'won' : 'lost';
  t.games += 1; t.wins += game.won ? 1 : 0; t.safe += game.safeRevealed;
  t.last = { outcome, safe: game.safeRevealed, seconds: (h.t1 - h.t0) / 1000 };
  addEvent(`you ${outcome}: ${game.safeRevealed}/${game.totalSafe} safe cells in ${((h.t1 - h.t0) / 1000).toFixed(1)} s`);
}

// ------------------------------------------------------------------------------------------
// UI
// ------------------------------------------------------------------------------------------
function addEvent(text, quiet = false) {
  const t = app.sim ? app.sim.time.toFixed(1) : '0.0';
  app.events.push({ t, text });
  if (app.events.length > 60) app.events.splice(0, app.events.length - 60);
  if (!quiet) console.log(`[flysweeper] ${t}s ${text}`);
}

function setupUi(model) {
  const g = model.game;
  app.flyView = new BoardView($('fly-board'), g.rows, g.cols, { eyeSplit: true });
  app.youView = new BoardView($('you-board'), g.rows, g.cols, { onReveal: humanReveal, onFlag: humanFlag });
  $('boards-title').textContent = SHOW_HUMAN_BOARD ? 'Same mines, two players' : 'Fly';
  $('cond-chip').textContent = `fly · frozen connectome`;
  $('legend').innerHTML = model.regions.map((r) => `<span><i style="background:rgb(${harmonise(r.color).join(',')})"></i>${r.name}</span>`).join('');
  $('label').textContent = model.label;
  $('subtitle').textContent = `${model.dataset} · ${model.n_neurons.toLocaleString()} neurons · ${model.n_edges_total.toLocaleString()} connections (${model.n_edges_silenced.toLocaleString()} onto sensory cells silenced) · simulated on your GPU`;
  const s = model.sim;
  $('preset').textContent = `dt ${s.dt}s τ ${s.tau}s gain ${s.gain} tonic ${s.tonic} noise ${s.noise_rate}Hz×${s.noise_amp}`;
  $('route').textContent = `${model.encoder.params.route}${model.encoder.params.danger_loom ? ' + looming→DNp01' : ''}`;
  $('gpu').textContent = [app.sim.adapterInfo.vendor, app.sim.adapterInfo.architecture, app.sim.adapterInfo.description].filter(Boolean).join(' ') || 'WebGPU';
  $('you-hint').textContent = `Left click reveals, right click (or ⌥/⌃-click) flags. Same mines as the fly. Your first click is always safe: if it lands on a mine, that mine is moved (on both boards) to a cell hidden on both and away from your click. The dot marks the fly's start cell, where a 3×3 zone is mine-free.`;
  document.querySelectorAll('button[data-speed]').forEach((b) => {
    b.classList.toggle('on', Number(b.dataset.speed) === app.pace.speed);
    b.onclick = () => {
      app.pace.speed = Number(b.dataset.speed); app.pace.paused = false; pacer.reset();
      document.querySelectorAll('button[data-speed]').forEach((x) => x.classList.toggle('on', x === b));
      $('pause').classList.remove('on');
    };
  });
  $('pause').onclick = () => { app.pace.paused = !app.pace.paused; $('pause').classList.toggle('on', app.pace.paused); if (!app.pace.paused) pacer.reset(); };
  $('new-game').onclick = () => { app.newGameRequested = true; app.fly.abort = true; };
  const brain = $('brain'), tip = $('brain-tip'), brainBox = $('brain-box');
  brain.addEventListener('mousemove', (ev) => {
    const rect = brain.getBoundingClientRect(), box = brainBox.getBoundingClientRect();
    const x = Math.round((ev.clientX - rect.left) * (brain.width / rect.width));
    const y = Math.round((ev.clientY - rect.top) * (brain.height / rect.height));
    const hit = app.brainMap.pick(x, y);
    if (hit) { tip.style.display = 'block'; tip.style.left = `${ev.clientX - box.left + 12}px`; tip.style.top = `${ev.clientY - box.top + 12}px`; tip.textContent = `${hit.type || ''} · ${hit.region}`; }
    else tip.style.display = 'none';
  });
  brain.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
  window.addEventListener('keydown', (ev) => { if (ev.key === ' ' && ev.target === document.body) { ev.preventDefault(); $('pause').click(); } });
  setupResize();
}

// ------------------------------------------------------------------------------------------
// canvas sizing: every canvas sits absolutely inside a flex ".box" and is fitted to the box by a
// ResizeObserver (so the layout is driven by the viewport, never by the canvas)
// ------------------------------------------------------------------------------------------
/** Fit `canvas` into its parent box at `aspect` (w/h; 0 = fill the box); returns true if the backing resolution changed. */
function fitCanvas(canvas, aspect, backingScale) {
  const box = canvas.parentElement;
  const bw = box.clientWidth, bh = box.clientHeight;
  if (bw < 8 || bh < 8) return false;
  let cw = bw, ch = aspect ? bw / aspect : bh;
  if (ch > bh) { ch = bh; cw = bh * aspect; }
  cw = Math.floor(cw); ch = Math.floor(ch);
  canvas.style.width = `${cw}px`; canvas.style.height = `${ch}px`;
  const W = Math.round(cw * backingScale), H = Math.round(ch * backingScale);
  if (canvas.width === W && canvas.height === H) return false;
  canvas.width = W; canvas.height = H;
  return true;
}

function setupResize() {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const fitBoards = () => {
    for (const view of [app.flyView, SHOW_HUMAN_BOARD ? app.youView : null]) {
      if (view && fitCanvas(view.canvas, 1, dpr) && view.state) view.draw(view.state);
    }
  };
  const fitBrain = () => {
    const brain = $('brain');
    // the canvas fills its box (1 backing pixel per CSS pixel); the soma cloud is fitted inside it
    // at its true proportions by BrainMap._layout
    if (fitCanvas(brain, 0, 1)) app.brainMap.resize(brain.width, brain.height);
  };
  const ro = new ResizeObserver((entries) => {
    for (const e of entries) {
      if (e.target === $('brain-box')) fitBrain(); else fitBoards();
    }
  });
  ro.observe($('fly-board').parentElement);
  if (SHOW_HUMAN_BOARD) ro.observe($('you-board').parentElement);
  ro.observe($('brain-box'));
  fitBoards(); fitBrain();
}

let lastDom = 0, lastBrain = 0;
function frame(now) {
  requestAnimationFrame(frame);
  if (now - lastDom > 66) { lastDom = now; updateDom(); }
  else for (const v of [app.flyView, app.youView]) if (v && v.animating && v.state) v.draw(v.state);   // smooth end-of-game animation
  if (now - lastBrain > 40 && app.sim && !app.sim.lost) {
    lastBrain = now;
    app.sim.readActivity().then((a) => { if (a) app.brainMap.draw(a); }).catch(() => {});
  }
}

function fmtOutcome(o) { return o === 'won' ? 'WON' : o === 'lost' ? 'LOST' : o === 'timeout' ? 'TIMEOUT' : o; }

function updateDom() {
  const fly = app.fly, human = app.human, g = app.model.game, dec = app.decoder;
  // boards
  if (fly.game) {
    const ended = fly.outcome === 'won' || fly.outcome === 'lost';
    app.flyView.draw({
      visible: fly.game.visible(), cursor: fly.cursor, dangerous: fly.danger >= app.model.encoder.params.loom_threshold,
      mineHit: fly.game.mineHit, dim: fly.phase === 'waiting',
      outcome: ended ? fly.outcome : null, gameId: fly.game.seed,
      banner: !ended && fly.outcome ? fmtOutcome(fly.outcome) : fly.phase === 'settle' ? 'settling…' : null,
      bannerColor: '#8b95a3',
    });
  }
  if (human.game && SHOW_HUMAN_BOARD) {
    app.youView.draw({
      visible: human.game.visible(), mineHit: human.game.mineHit,
      safeStart: human.game.safeRevealed === 0 && !human.game.over ? [Math.floor(g.rows / 2), Math.floor(g.cols / 2)] : null,
      outcome: human.game.over ? (human.game.won ? 'won' : 'lost') : null, gameId: `${human.game.seed}h`,
      cursor: human.game.over && !human.game.won ? human.game.mineHit : null,
    });
  }
  // fly stats
  const flyStatus = fly.phase === 'calibrating' ? 'calibrating' : fly.phase === 'waiting' ? (human.started && !human.game.over ? 'waiting for you' : 'next game soon') : fly.phase;
  $('fly-status').textContent = flyStatus;
  $('fly-status-top').textContent = fly.phase === 'over' && (fly.outcome === 'won' || fly.outcome === 'lost') ? '' : ` · ${flyStatus}`;
  $('last-action').textContent = fly.lastAction + (fly.lastResult && fly.lastAction === 'reveal' ? ` (${fly.lastResult})` : '');
  $('danger').textContent = fly.danger; $('danger-kv').className = 'kv' + (fly.danger >= app.model.encoder.params.loom_threshold ? ' danger' : '');
  $('fly-turn').textContent = fly.game ? `${fly.phase === 'settle' || fly.phase === 'calibrating' ? 0 : fly.turn + 1} / ${g.max_turns}` : '–';
  // human stats
  if (human.game) {
    const t = human.t0 ? ((human.t1 || performance.now()) - human.t0) / 1000 : 0;
    $('you-time').textContent = `${t.toFixed(1)} s`;
    $('you-flags').textContent = `${human.game.flagsPlaced} / ${g.mines}`;
    $('you-status').textContent = human.game.over ? (human.game.won ? 'won' : 'lost') : human.started ? 'playing' : 'your move';
  }
  // stat tiles (fly) + sparkline of safe cells per finished game
  const tf = app.totals.fly, fg = fly.game;
  $('fly-safe').innerHTML = fg ? `${fg.safeRevealed}<small> / ${fg.totalSafe}</small>` : '–';
  $('fly-safe-sub').textContent = fg ? (fly.phase === 'playing' ? `turn ${fly.turn + 1}` : fly.phase) : 'this board';
  $('fly-games').innerHTML = `${tf.games}<small> (${tf.wins})</small>`;
  $('fly-games-sub').textContent = tf.abandoned ? `${tf.abandoned} abandoned` : 'finished';
  $('fly-mean').textContent = tf.games ? (tf.safe / tf.games).toFixed(1) : '–';
  $('fly-mean-sub').textContent = fg ? `of ${fg.totalSafe} safe` : 'of 71 safe';
  $('fly-last').textContent = tf.last ? tf.last.outcome : '–';
  $('fly-last').className = 'v ' + (tf.last ? tf.last.outcome : '');
  $('fly-last-sub').textContent = tf.last ? `${tf.last.safe} safe · ${tf.last.turns} turns` : '\u00a0';
  drawSparkline($('spark'), app.history.map((h) => h.safe), fg ? fg.totalSafe : 71, $('spark-s'));
  // human scoreline (only when the human board is shown)
  if (SHOW_HUMAN_BOARD) {
    const th = app.totals.human, hg = human.game;
    $('you-safe').textContent = hg ? `${hg.safeRevealed} / ${hg.totalSafe}` : '–';
    $('you-games').textContent = `${th.games} (${th.wins})`;
    $('you-mean').textContent = th.games ? (th.safe / th.games).toFixed(1) : '–';
    $('you-last').textContent = th.last ? `${th.last.outcome}, ${th.last.safe} safe, ${th.last.seconds != null ? `${th.last.seconds.toFixed(1)} s` : ''}` : '–';
    $('you-abandoned').textContent = String(th.abandoned);
    $('you-note').textContent = human.relocation || '';
  }
  $('game-title').innerHTML = fly.game ? `fly · game ${app.gameNumber} · seed ${fly.game.seed}` +
    (fly.outcome === 'won' || fly.outcome === 'lost' ? ` <span class="tag ${fly.outcome}">${fly.outcome.toUpperCase()}</span>` : '') : '';
  // pools: bar = rate this turn, tick = running baseline, gold = the action taken
  const rates = dec.lastRates, base = dec.running, max = Math.max(1, ...rates, ...base) * 1.08;
  let html = '';
  dec.actions.forEach((a, i) => {
    const win = a === fly.lastAction && fly.phase === 'playing';
    const title = `${a}: ${rates[i].toFixed(2)} Hz this turn · running baseline ${base[i].toFixed(2)} Hz · ${dec.sizes[i]} cells`;
    html += `<div class="name" title="${title}">${a}<small>${dec.sizes[i]}</small></div>` +
      `<div class="bar" title="${title}"><i class="${win ? 'win' : ''}" style="width:${Math.min(100, (100 * rates[i]) / max).toFixed(1)}%"></i><s style="left:${Math.min(100, (100 * base[i]) / max).toFixed(1)}%"></s></div>` +
      `<div class="rate" title="${title}"><b>${rates[i].toFixed(1)}</b> Hz</div>`;
  });
  $('pools').innerHTML = html;
  // sim stats
  const st = app.stats, sim = app.sim;
  $('sim-step').textContent = sim.stepCount.toLocaleString();
  $('sim-time').textContent = `${sim.time.toFixed(1)} s`;
  $('spikes').textContent = `${st.spikesLastStep.toLocaleString()} (${(st.spikesLastStep / sim.n / app.model.sim.dt).toFixed(2)} Hz)`;
  $('ms').textContent = st.msPerStep == null ? '–' : `${st.msPerStep.toFixed(2)} ms/step (${st.msSource === 'gpu' ? 'GPU' : 'wall'})`;
  $('ms').title = st.msSource === 'gpu' ? 'GPU timestamp queries' : 'wall time including readback';
  $('rtf').textContent = app.pace.paused ? 'paused' : `${st.rtf.toFixed(1)}× realtime`;
  // live indicator: green + pulsing while the step counter advances
  const advancing = sim.stepCount !== live.lastStep;
  if (advancing) live.lastChange = performance.now();
  live.lastStep = sim.stepCount;
  const isLive = !app.pace.paused && performance.now() - live.lastChange < 600;
  $('live').classList.toggle('on', isLive);
  $('live-text').textContent = app.pace.paused ? 'paused' : isLive ? 'live' : 'idle';
  $('events').innerHTML = app.events.slice(-12).reverse().map((e) => `<div class="${eventClass(e.text)}"><b>${e.t}s</b>${escapeHtml(e.text)}</div>`).join('');
}

const live = { lastStep: -1, lastChange: 0 };
function eventClass(text) {
  if (/→ mine|\blost\b|BOOM|crashed/.test(text)) return 'bad';
  if (/\bwon\b/.test(text)) return 'good';
  return '';
}

/** Tiny line + area chart of safe cells per finished game (last 50), with the mean as a dotted line. */
function drawSparkline(canvas, values, total, labelEl) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const cw = canvas.clientWidth || 200, ch = canvas.clientHeight || 26;
  if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) { canvas.width = Math.round(cw * dpr); canvas.height = Math.round(ch * dpr); }
  const ctx = canvas.getContext('2d'), W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const v = values.slice(-50);
  if (labelEl) labelEl.textContent = v.length ? `${v.length} game${v.length === 1 ? '' : 's'}` : 'no games yet';
  if (!v.length) { ctx.strokeStyle = 'rgba(255,255,255,0.08)'; ctx.setLineDash([2 * dpr, 4 * dpr]); ctx.beginPath(); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke(); ctx.setLineDash([]); return; }
  const pad = 2 * dpr, x = (i) => v.length === 1 ? W / 2 : pad + (i / (v.length - 1)) * (W - 2 * pad), y = (s) => H - pad - (s / total) * (H - 2 * pad);
  const mean = v.reduce((a, b) => a + b, 0) / v.length;
  ctx.strokeStyle = 'rgba(255,255,255,0.10)'; ctx.lineWidth = dpr; ctx.setLineDash([2 * dpr, 4 * dpr]);
  ctx.beginPath(); ctx.moveTo(0, y(mean)); ctx.lineTo(W, y(mean)); ctx.stroke(); ctx.setLineDash([]);
  ctx.beginPath(); ctx.moveTo(x(0), H); v.forEach((s, i) => ctx.lineTo(x(i), y(s))); ctx.lineTo(x(v.length - 1), H); ctx.closePath();
  const g = ctx.createLinearGradient(0, 0, 0, H); g.addColorStop(0, 'rgba(242,193,92,0.28)'); g.addColorStop(1, 'rgba(242,193,92,0.0)');
  ctx.fillStyle = g; ctx.fill();
  ctx.beginPath(); v.forEach((s, i) => (i ? ctx.lineTo(x(i), y(s)) : ctx.moveTo(x(i), y(s))));
  ctx.strokeStyle = '#f2c15c'; ctx.lineWidth = 1.5 * dpr; ctx.lineJoin = 'round'; ctx.stroke();
  const lx = x(v.length - 1), ly = y(v[v.length - 1]);
  ctx.fillStyle = '#f2c15c'; ctx.beginPath(); ctx.arc(lx, ly, 2 * dpr, 0, Math.PI * 2); ctx.fill();
}

// ------------------------------------------------------------------------------------------
// debugging / verification API (used by the README and the browser checks)
// ------------------------------------------------------------------------------------------
window.flysweeper = {
  app,
  humanReveal, humanFlag, relocateMineForFirstClick,
  /** Debug: reveal every safe cell of the fly's current board so the win animation plays at the next turn boundary. */
  forceWin: () => { const g = app.fly.game; if (!g || g.over) return false; for (let r = 0; r < g.rows; r++) for (let c = 0; c < g.cols; c++) if (!g.mines[r * g.cols + c]) g.reveal(r, c); return g.won; },
  /** Debug: reveal a mine on the fly's current board so the loss animation plays at the next turn boundary. */
  forceLoss: () => { const g = app.fly.game; if (!g || g.over) return false; for (let i = 0; i < g.rows * g.cols; i++) if (g.mines[i]) { g.reveal(Math.floor(i / g.cols), i % g.cols); app.fly.cursor = [Math.floor(i / g.cols), i % g.cols]; break; } return g.lost; },
  /** 100 warm-up + 500 blind steps -> mean firing rate; pauses the fly at its next turn boundary. */
  idleTest: (opts) => exclusive(() => idleTest(app.sim, app.decoder, opts)),
  /** Pulse left LC4+LPLC2 (amp 0.8, every other step) -> DNp01 L/R rates vs silent. */
  loomTest: (opts) => exclusive(() => loomTest(app.sim, app.model, app.encoder, app.decoder, opts)),
  /** A mid-game board with the cursor at the far left -> lamina (L1-L3) and whole-brain rates. */
  boardTest: (opts) => exclusive(() => boardTest(app.sim, app.model, app.encoder, app.decoder, opts)),
  /** Time `steps` blind steps at full speed; returns ms/step (GPU timestamps when available). */
  benchmark: (steps = 200) => exclusive(async () => {
    let gpu = 0, wall = 0, n = 0;
    while (n < steps) {
      const k = Math.min(app.sim.maxBatch, steps - n);
      const res = await app.sim.runSteps(new Array(k).fill(null));
      gpu += res.gpuMs ?? 0; wall += res.wallMs; n += k;
    }
    return { steps: n, ms_per_step_gpu: app.sim.hasTimestamps ? gpu / n : null, ms_per_step_wall: wall / n, realtime_factor_wall: (app.model.sim.dt * 1000) / (wall / n) };
  }),
  state: () => ({
    step: app.sim.stepCount, time: app.sim.time, phase: app.fly.phase, cursor: app.fly.cursor, lastAction: app.fly.lastAction,
    rates: Array.from(app.decoder.lastRates), scores: Array.from(app.decoder.lastScores), idle: app.idleRates && Array.from(app.idleRates),
    stats: { ...app.stats }, totals: app.totals, flySafe: app.fly.game?.safeRevealed, gameNumber: app.gameNumber, events: app.events.slice(-10),
  }),
};

boot();
