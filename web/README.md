# FlySweeper · WebGPU

The whole MaleCNS v1.0 fruit-fly connectome (166,700 neurons, 25,582,938 signed synaptic edges,
Berg et al., Cell 2026, CC BY 4.0) simulated as a leaky integrate-and-fire network **entirely in the
browser on your GPU**, wired to Minesweeper. The fly plays the left board; you play the right board
with the same mines.

> MaleCNS v1.0 wiring (Berg et al., Cell 2026, CC BY 4.0) with engineered dynamics, sensors and
> buttons. Not a validated fly. Connectome frozen except 59,334 KC→MBON synapses trained (round 2).
> A helper reads the board into 31 facts, injected as odours.

By default the page runs the **trained fly** (the Python `fly-mb` condition: a helper turns the board
into 31 facts that are injected as odours into olfactory receptor neurons, and six pools of
mushroom-body output neurons are the buttons, with the Kenyon-cell → MBON synapses onto them trained
by `flysweeper/train_mb.py`). A selector in the header switches between exported weight sets and the
original **frozen connectome** (descending-neuron pools decode the actions, nothing trained).

Everything is vanilla JS modules + WGSL; there is no build step. The dynamics, encoder, decoders,
helper and mushroom-body policy are line-by-line ports of the Python package in `flysweeper/`
(`sim.py`, `encoder.py`, `decoder.py`, `oracle.py`, `mb_policy.py`, `agent.py`, `minesweeper.py`),
so the two implementations can be compared.

## Run it

Requirements: the compiled graph in `data/compiled/` (see the repo README), Python 3.11 venv at
`./.venv`, and Chrome 113+ (or Edge) on macOS — WebGPU is on by default there. Safari 18+ and Firefox
Nightly with WebGPU should also work but were not tested.

```bash
# 1. export the compiled graph into browser-friendly binaries (web/data/, ~205 MB, gitignored; ~8 s)
./.venv/bin/python -m flysweeper.export_web

# 2. export the trained weight sets (web/weights/, ~0.6 MB total, committed; already done for rounds 1-2)
./.venv/bin/python -m flysweeper.export_web --weights outputs/mb/warm/weights_000200.npz \
    --name round1 --round 1 --winrate 0.0 --eval-games 30 --notes "..."
./.venv/bin/python -m flysweeper.export_web --weights data/compiled/mb_weights.npz \
    --name round2 --round 2 --winrate 0.48 --eval-games 100 --default --notes "..."

# 3. serve the web/ folder (any static server works; WebGPU needs localhost or https)
./.venv/bin/python -m http.server 8780 --directory web --bind 127.0.0.1

# 4. open in Chrome
open http://127.0.0.1:8780/
```

The first load downloads ~205 MB from the local server (a second or two), verifies each file's
SHA-256 against `data/manifest.json`, and stores the files in the browser Cache API; later loads read
from the cache in ~100 ms. Re-running the export changes the hashes, which invalidates the cache
automatically. `?clearcache=1` wipes it, `?nocache=1` bypasses it.

URL parameters: `?weights=round2|round1|frozen` (weight set; default = `web/weights/manifest.json`
`"default"`), `?speed=0.5|1|4|0` (0 = as fast as the GPU allows), `?seed=1000` (first board seed),
`?simseed=0` (GPU noise seed), `?human=1` (show the human-playable board, see below), `?data=<url>`
(connectome base URL, see *Deploying*), `?v=<token>` (cache-busting version, set automatically when
you switch weight sets).

### Trained weight sets (`web/weights/`)

`flysweeper.export_web --weights <npz> --name <id> [--round k --winrate 0.48 --eval-games 100
--games-trained n --label "..." --notes "..." --default]` converts one `MBPolicy.save()` file (the
self-describing npz `train_mb.py` writes, e.g. `data/compiled/mb_weights.npz`) into

* `web/weights/<id>.json` — the six MBON pools (cells per action, types), the odour map (channel →
  ORN cells, taken from the file's own `orn_types`, and the amplitude), the Kenyon-cell and
  antennal-lobe-PN index lists with the fly-mb model settings (`kc_kc_gain`, `kc_bias`, `pn_kc_gain`,
  `pn_bias`), the decision settings (centred scores + the saved running means, `reveal_margin`,
  `mask_reveal`, temperature 0), metadata (label, round, win rate, games trained, notes, training
  args) and weight statistics;
* `web/weights/<id>.bin` — the plastic edge list: `u16` KC index (into `kc`), `u8` MBON index (into
  `mbon`), `f32` trained weight, in that order (415 KB for the 59,334 round-2 edges). The frozen
  weight of each edge is the one already in the graph export;
* `web/weights/manifest.json` — every set with `label`, `win_rate`, `eval_games`, `games_trained`,
  `notes`, file names and hashes, plus `"default"`. Re-running with the same `--name` replaces that
  entry; `--default` (or the first export) sets the default. When round 3 lands:
  `--weights <round3.npz> --name round3 --round 3 --winrate 0.52 --eval-games 100 --default`.

The export recomputes the plastic edge set from the graph exactly as `mb_policy.MBPolicy` does
(every KC → pool-MBON edge) and refuses files whose `edge_pos`, `w0`, `edge_kc`, `edge_pool` or
channel list disagree; it reads the binary back and checks the round trip. The label defaults to
`"Round k - N teacher games - P% wins (M held-out boards)"`.

### Using the page

The page is laid out to fit one screen without scrolling (a `100vh` CSS grid: one-line header,
three columns, one-line honesty label; the canvases are fitted to their grid cells by a
`ResizeObserver`; the board is square and never below 300 px). Left column (≤ 38 vw): the fly's
board with its action/danger/turn line, four stat tiles and a sparkline directly under it; centre: the
brain map (largest), legend, one stats strip with the pace buttons; right column (240 px): pool bars
on top, the log (last 12 entries, newest first, scrolling inside its box) below. The look is shared
with the Python spectator page (`flysweeper/ui/index.html`): the `<style>` blocks of the two files
are identical, and the board/brain drawing code is the same (kept in sync by hand).
Verified with no vertical or horizontal scroll at 1440×760, 1512×870, 1728×1000, 1280×680, 1512×982
and 1728×1117 (a Chrome window on a 14"/16" MacBook Pro loses ~110 px to browser chrome). The "?"
markers hold the longer explanations (pool anatomy; anatomy vs. engineered; console commands).

* **FLY board** (left): the connectome's cursor is the gold square (red when the looming channel is
  active, i.e. a revealed number ≥ 2 is on or next to it). The dashed line is the split between the
  left and right eye. A turn is 15 brain steps = 300 ms of simulated time; after each turn the
  descending-neuron pools are read out and one action is taken (up/down/left/right/reveal/jump/hold).
* **YOU board** — *hidden by default*, open `http://127.0.0.1:8780/?human=1` to show it (the flag is
  `SHOW_HUMAN_BOARD` at the top of `js/main.js`; the panel is then titled "Same mines, two players"
  and the scoreboard gains a YOU row; the game code is unchanged): left click reveals, right click (or ⌥/⌃-click) flags; long press flags on
  touch screens. Both boards have the *same* mines, and your first click is always safe anywhere: if
  it lands on a mine, that mine is moved to a random cell that is hidden (and unflagged) on both
  boards and not adjacent to your click, on *both* boards, and the numbers are recomputed. Cells off
  the fly's revealed frontier are preferred, so the numbers the fly already sees normally do not
  change (if no such cell exists, a frontier cell is used and the fly's numbers may shift by ±1 —
  the log says so). If the fly has already finished its game when you click, only your board changes,
  so the fly's recorded result stays honest. The small gold dot marks the fly's start cell, whose 3×3
  neighbourhood is mine-free (a hint, not the only safe start). The timer starts at your first click.
* **Game end**: every game ends visibly. On a loss the mine the fly revealed bursts (red rings and
  particles, ~0.6 s), the other mines pop into view nearest-first, the cursor ring turns red and a
  small LOST tag appears in the board's corner and in the panel title. On a win the remaining hidden
  cells flip to flags one after another under a gold glow pulse on the board border, with a WON tag.
  The finished board then stays on screen for 2.5 s (`GAME_OVER_HOLD_MS` in `js/main.js`; skipped at
  `max` pace, like the Python server's `--game-over-hold`) before the next board; the stat tiles
  update the moment the game ends, not after the hold. Console: `flysweeper.forceWin()` /
  `flysweeper.forceLoss()` end the current game programmatically (debug only). The same effects run on
  the Python spectator page (`renderBoard`/`makeEndFx` are copied verbatim there).
* **Stat tiles**: safe cells revealed on the current board, finished games (wins) with the number of
  *abandoned* games underneath, mean safe cells per finished game, and the last result; the sparkline
  under them is safe cells per finished game over the last 50 games (dotted line = their mean),
  tracked client-side. With `?human=1` a YOU scoreline with the same figures appears below. The fly wins by revealing
  all 71 safe cells, loses on a mine, or times out after 400 turns. A new shared board starts 3 s
  after the fly finishes if your board is finished or untouched; if you are mid-game the fly waits for
  you (its brain keeps running on a blank screen). **new game** forces a fresh board at the fly's next
  turn boundary: an unfinished fly game is recorded as *abandoned* (separate counter, not in games,
  wins or the mean), and so is your board if you had revealed at least one cell. The fresh game resets
  the fly's cursor to the centre, its turn counter, the encoder's previous-luminance memory, the
  decision-window display (the GPU pool counters are cleared at the start of the first turn) and your
  timer and flags; the decoder's running baseline persists across games, as in the Python agent.
* **Pools → buttons**: bar = per-cell spike rate of each pool this turn (Hz), the tick on each bar is
  the pool's running EMA baseline; gold is the action taken; the small number after the name is the
  pool size in cells.
* **Whole nervous system**: 139,662 somas (every neuron with a soma position), drawn at the soma
  cloud's true x/z proportions (0.733 wide : 1 tall, brain on top, nerve cord below — the constant
  `CNS_ASPECT` in `js/brainmap.js`) fitted to the panel with a 3 % margin. Quiet somas are a faint
  region-tinted dust; a spike adds a soft 3×3 glow in the region colour that decays over a few frames
  (additive `ImageData`, ~20 fps). Hover to see a neuron's type and region.
* **Header**: the chip after the product name is the condition — `fly-mb · round 2` for a trained
  weight set, `fly · frozen connectome` otherwise — followed by the **weights** selector (entries
  from `web/weights/manifest.json`, e.g. "Round 2 - 1,500 teacher games - 48% wins (100 held-out
  boards)", plus "Frozen connectome (no training)"). Choosing another entry does a *forced full
  reload*: `location.assign(?weights=<id>&v=<timestamp>)`; an import map written by `index.html`
  gives every JS module `?v=<token>`, and the shaders, `config.json` and weight files are fetched with
  the same token, so no stale code or brain state survives the switch (the 205 MB connectome stays in
  the Cache API, keyed by content hash). The honesty line at the bottom changes to "Connectome frozen
  except N KC→MBON synapses trained (round k). A helper reads the board into 31 facts, injected as
  odours." with the full description in its tooltip. The dot on the right pulses green while the step
  counter advances and turns grey when paused.
* **Trained fly (fly-mb)**: each turn `js/oracle.js` computes the 31 facts about the cell under the
  cursor (hidden / revealed / provably safe / provably mine / frontier / free, neighbour counts,
  direction to the nearest provably-safe cell or guess target, cleared fraction, untouched — the same
  single-point inference as the Python helper), `js/mbpolicy.js` turns them into odour current
  (`odor_amp × value` into every ORN cell of that channel, 2,362 cells over 31 channels for round 2,
  added to the retinal drive on every step of the turn), the GPU counts the six MBON pools over the
  15-step window, and the decision is the Python one: per-cell counts minus each pool's running mean
  (α 0.02, initialised from the means saved in the file), reveal only if it beats the runner-up by
  `reveal_margin` (0.25 for round 2, otherwise the runner-up is taken), stable argmax with a random
  tie-break, learning off. The pools panel then shows the MBON pools (the tick is the running mean)
  and a gold chip with the weight statistics; `flysweeper.facts()` prints the active facts.
  The fly's first reveal is made safe wherever it lands (the Python game places its mines on the first
  click): if it is on a mine, that mine is moved, as for the human's first click.
* **Pace**: ½×, 1×, 4× real time, `max`, `pause` (space bar). Speed is the simulated time per wall
  second; at 1× a brain step happens every 20 ms.
* **Stats**: step count, simulated time, spikes in the last step and the corresponding mean firing
  rate, GPU compute per step (from WebGPU timestamp queries when available, otherwise wall time
  including the readback), realtime factor, the GPU adapter, the dynamics constants and input route.

### Browser console

```js
await flysweeper.idleTest()      // 100 warm-up + 500 blind steps -> mean firing rate, spikes/step, pool idle rates
await flysweeper.loomTest()      // pulse left LC4+LPLC2 (amp 0.8, every other step): DNp01 L/R rates vs. silent
await flysweeper.boardTest()     // a mid-game board, cursor far left: L1-L3 lamina rate, whole-brain rate
await flysweeper.benchmark(300)  // ms/step at full speed
flysweeper.state()               // current cursor, phase, rates, timing, totals, weight set, MB state
flysweeper.facts()               // fly-mb: the helper's active facts for the current board and cursor
```

The tests take over the simulator at the fly's next turn boundary (unpause first if paused) and
continue from the current brain state; they are the same population checks as
`python -m flysweeper.sim --no-sensory-input` and `python -m flysweeper.probe`.

## What is anatomy and what is engineered

Anatomy (from the MaleCNS v1.0 release, via `flysweeper/compile_graph.py`):

* neurons, cell types, sides, superclasses, soma positions;
* every synaptic connection between them with its synapse count;
* transmitter prediction per presynaptic cell, turned into a sign (GABA/glutamate/histamine −1, the
  rest +1) and normalised so that each cell's total absolute input is 1;
* the optic-lobe hex column of every lamina and photoreceptor cell, and the LC4/LPLC2 → DNp01 giant
  fiber wiring the "jump" reflex runs through.

Engineered (ours, ported from the Python package, all constants in `data/model.json`):

* leaky integrate-and-fire dynamics: `v ← v·e^(−dt/τ) + gain·Σ w·spike(t−1) + tonic + noise + ext`,
  dt 20 ms, τ 100 ms, gain 3.0, tonic 0.14, Bernoulli noise 1.2 Hz × 0.22, threshold 1, reset 0, floor −5
  (the fly.ai "flyai" preset);
* synapses onto sensory neurons are removed (`sensory_input=False`, the Shiu et al. convention);
  494,831 of the 25,582,938 edges, baked into the export;
* the board → eye mapping (equal-count partition of hex columns into 9 × 9 cells, left eye sees the
  left half), the luminance table, the 25 Hz cursor flicker and Fly64's retinal formula driving the
  L1/L2/L3 lamina cells, R7 (flags) and R8 (cursor);
* the looming channel: the largest revealed number around the cursor pulses LC4/LPLC2 of that eye;
* the decoder: which descending neurons are which button (up DNp09+DNg100+DNg97, down MDN, left/right
  DNa02+DNa11+DNg13 by side, reveal DNpe017+DNp10, jump DNp01), the 15-step turn window, the running
  baseline (EMA α 0.1, initialised from a 500-step blank-screen idle measurement), argmax with random
  tie-break, hold when nothing rises above baseline;
* the game rules, the shared-mines layout with a safe centre cell;
* for the trained fly (`fly-mb`, same as the Python condition): the helper's 31 facts and which ORN
  type is which fact, the Kenyon-cell sparsening (KC→KC synapses × 0, PN→KC × 3, KC bias −0.3, PN
  bias −0.3 for round 2 — the constants travel with the weight file), which MBON types are which
  button, the centred-score decision with the reveal margin, and the trained values of the KC→MBON
  synapses themselves (the only thing that was learned; 55,832 of 59,334 differ from the wiring).

## How it works

`flysweeper/export_web.py` converts `data/compiled/*` into `web/data/`:

| file | content |
| --- | --- |
| `in_indptr.u32`, `in_indices.u32`, `in_weights.f32` | in-edge CSR by **post**synaptic neuron (25,088,107 active edges after silencing sensory targets); the GPU kernel is a pure gather, no atomics |
| `order.u32` | neurons sorted by in-degree (heaviest first) for load balance |
| `pool_id.u32` | decoder pool of each neuron |
| `plot_idx.u32`, `soma_uv.f32`, `region.u8`, `type_id.u16` | brain-map atlas (fly's left drawn on the left, brain on top) |
| `model.json` | constants, encoder target lists for 9 × 9, looming lists, pools, palette, type names |
| `manifest.json` | byte size and SHA-256 of every file |

The export re-reads its output and checks: edge count, monotone `indptr`, `Σ|w| = 1` for every
non-sensory neuron with input (max deviation 5.6e-8), no synapses onto sensory neurons, and that the
transpose preserves every neuron's out-degree; the encoder target lists are cross-checked against
`flysweeper.encoder.RetinaEncoder` when it imports.

`web/shaders/step.wgsl` is one fused kernel per step: a 256-thread workgroup handles 8 neurons, 32
lanes per neuron stream the neuron's in-edges (coalesced reads of index + weight, gather of the
previous step's spike mask), reduce in workgroup memory in a fixed order (deterministic), and lane 0
integrates, spikes, resets, updates the brain-map activity counter and adds to the pool counters.
Noise is a stateless hash (pcg3d of neuron, step, seed), so a run is reproducible for a given seed.
`scatter.wgsl` adds the sparse external drive (≈5–7 k (index, amount) pairs per step) into a
self-clearing `ext` vector; `pack.wgsl` packs one brightness byte per plotted neuron for the map.

Up to 32 steps are encoded into one command buffer (a whole 15-step turn at `max` speed); the only
per-batch readback is a 256-byte stats block (pool counts + spikes per step), plus 140 KB of packed
activity at ≤ 25 Hz for the brain map. All neuron and synapse state stays resident on the GPU
(≈209 MB).

### How the mushroom-body model changes are implemented

The kernel stays a plain gather; the fly-mb condition is a one-off **edit of the resident graph** plus
one extra per-neuron term, applied by `MBPolicy.apply()` after the idle calibration (Python:
`MBPolicy.attach()`):

* *Weights, patched in place.* The in-edge CSR is indexed by postsynaptic neuron, so every change is a
  contiguous row: for each of the 4,064 Kenyon cells, in-edges whose presynaptic cell is a KC are
  multiplied by `kc_kc_gain` (0 → the 642,933 KC→KC synapses are silenced) and those from the 686
  antennal-lobe PNs by `pn_kc_gain` (22,586 edges); for each of the 91 pool MBONs, the in-edges from
  KCs (exactly the 59,334 plastic edges) are overwritten with the trained weights. The CPU copy of
  `in_weights` is edited and the affected rows (4,155 rows, coalesced into contiguous spans) are
  re-uploaded with `queue.writeBuffer` — 23 ms, no kernel change, no per-edge mask to read on every
  step. Switching back to the frozen graph is a page reload (which the selector forces anyway).
* *Bias.* `step.wgsl` adds `bitcast<f32>(nmeta[2n + j])` to the external drive of neuron `j` on every
  step — a third segment of the existing metadata buffer that is zero unless a weight set sets
  `kc_bias` on the KCs and `pn_bias` on the PNs (`FlySim.setBias`). One extra 4-byte read per neuron
  per step, i.e. 0.67 MB against the ~200 MB the gather reads; ms/step is unchanged (1.7–1.9 ms live
  at `max` with the odour drive, vs. 1.6–1.9 ms frozen).
* *Pools.* `FlySim.setPools` rewrites the pool-id segment so the atomic counters count the MBON pools
  instead of the descending neurons; the odour current joins the retinal (index, amount) list that
  `scatter.wgsl` adds before each step (≈7 k pairs per step at most; `maxDrive` is 16,384).

## Performance (Apple M5 Max, Chrome 152, macOS)

| | |
| --- | --- |
| GPU time per 20 ms brain step (timestamp queries, GPU busy) | **0.7–0.9 ms** (`flysweeper.benchmark(300)`: 0.88 ms GPU, 0.91 ms wall incl. readback = 22× realtime) |
| live play at `max` pace, measured over 8 s | 789 steps/s = **15.8× realtime** (includes decisions, readbacks, the brain-map readback and DOM updates) |
| GPU time per step at 1× pace | ≈ 2 ms — the GPU idles between 2-step bursts and clocks down; irrelevant at 50 steps/s |
| data load | ~0.5 s first load from a local server (+ ~0.6 s SHA-256), ~0.1 s from the Cache API |
| GPU memory | 209 MB of buffers |
| spikes per step | ≈ 7,600 (mean firing rate ≈ 2.3 Hz), same as the Python simulator |

For comparison, the Python/numba simulator (`flysweeper/sim.py`, event-driven: it only visits the
out-edges of neurons that spiked) takes 1.6 ms per step on the same machine with 18 threads. The GPU
kernel does a full dense gather over all 25.1 M active edges every step (≈ 200 MB of index + weight
reads), so its cost does not depend on activity.

## Verification against the Python simulator

Different RNG streams, so population statistics are compared (numbers from `flysweeper.idleTest()`
and `flysweeper.loomTest()` in Chrome vs. `python -m flysweeper.sim --no-sensory-input` and the looming
check of `python -m flysweeper.probe`):

| check | Python (numba, seed 0) | WebGPU (Chrome 152, Apple M5 Max) |
| --- | --- | --- |
| idle mean rate, 500 blank steps after 100 warm-up | 2.28 Hz, 7,593 spikes/step | 2.27–2.28 Hz, 7,577–7,589 spikes/step (per-step range 6,734–8,979) |
| looming: left LC4+LPLC2 pulsed at 0.8 every other step → DNp01 | left 25.0 Hz, right 1.5 Hz (silent 0.1 / 1.1) | left 25.0 Hz, right 0.8–1.2 Hz (silent 0.3–0.4 / 0.5–0.6) |
| looming: LC4+LPLC2 left while pulsing | 25.0 Hz | 25.0 Hz (right eye 0.5 Hz) |
| board on, cursor far left: L1–L3 lamina rate | 0.34 Hz blind → 3.03 Hz | 5.1 Hz (different sample board: 62 cells revealed, brighter) |
| board on: whole-brain mean rate | 2.28 → 2.70 Hz, 9,007 spikes/step | 2.84 Hz, 9,472 spikes/step |

Gameplay (seeds 1000–1003, `max` pace): the fly lost all four games after revealing 56, 64, … safe
cells (mean 41.5 / 71), using the same mix of moves, reveals and looming-triggered jumps as the
Python agent: a first reveal at the centre opens most of the board, then jumps (DNp01, driven by the
looming channel when the cursor sits next to a number ≥ 2) carry the cursor to random hidden cells
until one is a mine.

Headless-Chrome UI checks (puppeteer-core against `http://127.0.0.1:8780/`, zero console errors or
warnings): pause freezes the step counter; ½×/1×/4× pace give 25/50/202 steps/s with the display
reading 0.5×/1.0×/4.0×; right click flags and unflags; a forced first click on a known mine while the
fly was mid-game (2 safe cells revealed) relocated that mine on both boards to a non-adjacent cell off
the fly's frontier — both boards still identical with 10 mines, numbers consistent, none of the fly's
displayed numbers changed, the click revealed safely; **new game** mid-game recorded one abandoned game
for the fly (turns/safe cells logged) and one for the human (1 cell revealed, 1 flag), games/wins/mean
stayed at 0, and the fresh board started with cursor at the centre, turn 0, 0 flags, timer 0.0 s;
a normal safe first click (a 0-cell) flood-filled 56 cells with no relocation.

Single-screen layout check after the visual polish pass (headless Chrome, viewports 1440×760,
1512×870, 1728×1000, 1280×680, 1512×982, 1728×1117): `scrollHeight == innerHeight` and
`scrollWidth == innerWidth` at all six (no scrolling either way); board 501 / 549 / 631 / 421 / 549 /
631 px and brain canvas 591×501 / 635×629 / 769×779 / 492×421 / 635×741 / 769×896 (the canvas fills
its box; the soma cloud is fitted inside at aspect 0.733, so it spans the full panel height), both
fully inside the viewport; the human board absent by default; zero console errors; the fly plays
(8 turns: left, right, down, jump×2, reveal×3, hold); pause freezes the step counter, 4× gives
~200 steps/s; the brain-map hover tooltip works. Screenshot at 1440×760:
`outputs/flysweeper-web-classy.png` (Python page, same look: `outputs/flysweeper-py-classy.png`).
With `?human=1` the two boards share the left column and are smaller at these sizes.

## Known limitations / open points

* The two boards share one mine layout; the human's safe first click is implemented by moving the
  offending mine on both boards (see above), so in rare cases (no hidden cell off the fly's frontier)
  a number the fly has already revealed can change by ±1 mid-game.
* Pressing **new game** takes effect at the fly's next turn boundary (≤ 15 steps); the unfinished
  game is counted as abandoned rather than as a loss.
* Timestamp queries are quantised to 100 µs in Chrome unless the browser runs with
  `--enable-dawn-features=allow_unsafe_apis`; without the feature at all the page falls back to wall
  time including the readback.
* Firefox/Safari WebGPU were not tested. The kernel needs 8 storage buffers per stage and a ~100 MB
  storage binding, both within WebGPU's default limits.
* No KC→MBON plasticity (the Python `fly-learning` condition) in the browser; the connectome is frozen.
