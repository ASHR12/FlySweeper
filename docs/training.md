# Teaching the fly: what was tried, what happened

FlySweeper simulates the MaleCNS v1.0 connectome (166,700 neurons, 25.6 M signed edges) as a
leaky integrate-and-fire network wired to Minesweeper. Left alone, the fly is a random player
(about 51 of 71 safe cells per 9x9 / 10-mine game, same as a random walk). This document records
the two ways we tried to teach it, with all numbers, seeds and commands. Plain language, no hype.

Honest framing for the second (main) approach:

> a helper reads the board into a few facts; the fly's mushroom body learns what to do about them.

## 1. What is anatomy and what is invented

| piece | anatomy (from the connectome) | invented (ours) |
|---|---|---|
| neurons, synapse counts, signs, cell types, sides | yes (Berg et al., Cell 2026) | – |
| LIF dynamics (`flyai` preset: dt 20 ms, tau 100 ms, gain 3, tonic 0.14, Bernoulli noise) | – | yes; synapses onto sensory neurons removed so sensory cells are driven only by us |
| board -> lamina L1/L2/L3 by optic column, R8 cursor, R7 flags | cell identities yes | the mapping and the "looming" channel onto LC4/LPLC2 |
| descending-neuron action pools (`fly`) | cell types yes | which type means which action |
| **helper facts -> ORN types (`fly-mb`)** | ORN cells and their glomerular wiring yes | which fact goes to which ORN type; the amplitude |
| **MBON action pools (`fly-mb`)** | MBON cells and their KC inputs yes | which MBON type means which action |
| **KC->MBON plasticity, PAM/PPL1 dopamine** | the plastic site and the dopamine cells are the real ones | the learning rule, rewards, learning rate, clip range |
| **KC sparsening for `fly-mb`** | – | KC->KC synapses silenced, a -0.20 bias on KCs (see 3.2); applied only in `fly-mb` and reverted for every other condition |

Nothing else changes. Every other synapse stays at its connectome weight in every condition.

## 2. Approach A (first attempt): frozen brain as a reservoir + trained linear readout

Files: `flysweeper/teacher.py`, `flysweeper/features.py`, `flysweeper/train_readout.py`,
`flysweeper/decoder.py` (`ReadoutDecoder`, `DecoderParams(mode="readout")`), condition `fly-readout`.

- Teacher: single-point Minesweeper logic expressed in the fly's six actions (walk to the nearest
  provably-safe cell and reveal; otherwise walk to / jump to an unconstrained hidden cell). Alone,
  with the fly's cursor mechanics: 86% wins on seeds 0-29.
- Data: the teacher drives the cursor while the brain watches the board; spike counts of chosen
  populations over each 15-step turn are recorded with the teacher's action. run1: 32,008 turns,
  seeds 20000-20471 (fixed camera). run2_ego: 32,044 turns, seeds 30000-30484 (egocentric camera).
- Readouts (multinomial logistic regression / ridge, standardized, grouped CV), held-out games:

| readout | populations | camera | turn accuracy | majority | shuffled labels |
|---|---|---|---|---|---|
| A | DN + visual projection + CX + MBON (13,562 cells) | fixed | 0.279 | 0.267 | 0.267 |
| B | A + T4/T5 | fixed | 0.300 | 0.267 | – |
| C | A, two windows | fixed | 0.287 | 0.267 | – |
| D | A | egocentric | 0.384 | 0.252 | – |
| E | A + T4/T5 (33k cells) | egocentric | 0.446 (balanced 0.348) | 0.252 | – |
| board-only control: 3x3 neighbourhood one-hot, same classifier | – | – | 0.51 | | |

- Diagnosis: CX, MBON and KC activity are at chance for every board quantity (the board never
  reaches the central brain in this LIF); T5 alone carries as much as all 33k cells; the readout is
  reading a retinotopic image and cannot do the safety inference linearly (reveal-score AUC 0.62).
- Held-out games, seeds 5000-5029 (`outputs/validation/readout/`): fixed camera: fly 48.6+-2.6,
  fly-readout(A) 48.1+-2.3, fly-blind 38.4+-3.0, random-walk 50.4+-2.5, random-click 39.3+-2.8,
  solver 68.8+-2.0 (win 0.90). Egocentric camera: fly 52.6+-2.2, fly-readout(E) 45.9+-2.2,
  fly-blind 46.3+-2.3. **No readout beat random-walk; 0 wins.** This is a real negative result.
- `fly-learning` (the original dopamine-gated KC->MBON rule in `plasticity.py`) vs `fly` on seeds
  6000-6029: 50.2+-2.4 vs 54.4+-1.5. After game 1 all 61,210 KC->MBON edges sit at the 0.1x floor
  and never leave (unnormalized eligibility makes each dopamine event a global switch). No learning.

## 3. Approach B (main): in-brain RL on KC->MBON synapses, facts delivered as smells

Files: `flysweeper/oracle.py` (the helper), `flysweeper/mb_policy.py` (odor encoder, MBON pools,
learning rule, calibration probe), `flysweeper/train_mb.py` (training), conditions `fly-mb` and
`fly-mb-oracle-only` in `agent.py` / `validate.py`. The spectator runs it with
`python -m flysweeper.server --condition fly-mb` (weights are loaded from
`data/compiled/mb_weights.npz` when present; `server.py` was not edited).

### 3.1 The helper ("what the fly smells")

Each turn `Oracle.features(board, cursor)` computes 31 channels in [0, 1] about the cell under the
cursor, using the same single-point inference as the teacher (full list with meanings at the top
of `oracle.py`): cursor cell hidden / revealed / flagged / provably safe / provably mine /
frontier-unknown / unconstrained; adjacent revealed numbers (0, 1-2, 3+); adjacent hidden cells
(0, 1-3, 4+); largest adjacent number (0, 1-2, 3+); direction of the nearest provably-safe cell
(up/down/left/right, graded 1/distance) or "none known"; direction of the nearest guess target
(unconstrained hidden cell, else least-risky frontier cell; only active when no safe cell is known),
"guess target is here", "guess target is far" (> 5 steps, the teacher's jump rule); fraction cleared
(<1/3, 1/3-2/3, >2/3); "nothing revealed yet".

Ceiling of these facts: `fly-mb-oracle-only` reads them with a fixed hand-written rule and no brain
(reveal if safe-here or guess-here; jump if guess-far; else step along the strongest direction
channel). It wins 87% of games with 70.4/71 safe cells on seeds 5000-5029.

### 3.2 Odor encoding and the calibration probe (ORN -> PN -> KC -> MBON)

Channel k drives every cell of one ORN type (the 31 ORN types with the most cells, both
hemispheres: ORN_DA1 204 cells, ORN_VA1d 132, ..., ORN_VC3 34) with current `1.0 x value` on every
step of the 15-step turn window. Synapses onto sensory neurons are removed in the sim, so ORNs are
driven only by the helper. Probe: `python -m flysweeper.mb_policy --probe` (results in
`outputs/mb/probe.json`).

| | PN Hz | KC Hz | KCs active in a window | MBON Hz |
|---|---|---|---|---|
| raw wiring, no odor | 14.9 | 50.0 | 100% | 46.0 |
| raw wiring, ORN_DM1 driven | DM1 PN 14 -> 50 | 50.0 | 100% | 46.1 |
| raw wiring, 16 channels on | 26.3 | 50.0 | 100% | 46.1 (every pool 15.0 spikes/cell = ceiling) |
| `fly-mb` modifications, no odor | 14.7 | 1.5 | 50% | 3.4 |
| `fly-mb`, ORN_DM1 driven | DM1 PN 50 | 2.3 | | 4.6 |
| `fly-mb`, 16 channels on | 26.2 | 7.6 | 81% | 12.5 (pools 4.1-7.6 spikes/cell) |

(a) ORN drive reaches the antennal-lobe PNs (DM1 PN 14 -> 50 Hz) in both configurations.
(b) With the raw wiring **every Kenyon cell fires on every step, odor or not**: 56% of a KC's input
weight is KC->KC recurrence (642,933 edges) and the PNs have a dense 14 Hz spontaneous rate, so the
mushroom body is saturated and carries no odor code (this also explains why KC/MBON/CX were at
chance in approach A). The Minecraft-mod-style fix, documented and applied only in `fly-mb`:
KC->KC gain 0.0, KC bias -0.20 (PN->KC gain kept at 1.0). Then two disjoint channel sets give
KC codes with d' = 4.9 between them vs 0.35 between repeats of the same set (raw wiring: all cells
at ceiling, no discrimination). The code is not biologically sparse (81% of KCs spike at least once
per window) but it is discriminative, which is what learning needs.

Offline ceiling of the KC code (`/tmp` diagnostic, teacher-driven boards, 2,401 turns, held-out
games): a standardized logistic regression on the 4,064 KC window counts predicts the teacher's
action with **0.849 accuracy** (majority 0.305); variants with the PN spontaneous rate suppressed
(PN bias -0.3, PN->KC gain 3-4) reach 0.87-0.88. So the smell -> Kenyon-cell code carries the
helper's facts almost losslessly; what remains is whether the MBON readout can learn them in-brain.

### 3.3 Plastic synapses and the decision

- Action MBONs (both hemispheres, large KC in-degree; KC->MBON edge counts in brackets):
  up MBON01 (2,109), down MBON05 (1,999), left MBON06 (1,944), right MBON03 (1,249),
  reveal MBON11 (4,184; the pool Ramp's Fly Review trained), jump MBON09 (4 cells, 4,682).
  16,167 plastic edges in total; all other 45,043 KC->MBON edges and everything else are frozen.
- Decision: per-cell spike count of each pool over the window -> softmax with temperature T
  (annealed 1.0 -> 0.2 during training; T = 0 = argmax at evaluation).
- Rule (per turn, chosen action a): eligibility e_ij = (KC_i count - KC_i running mean) x (+1 if
  MBON_j in pool(a), -0.2 otherwise); trace z <- 0.5 z + e; reward r = +0.1 per newly revealed safe
  cell (cap 1), -1 mine, -0.05 no-op (reveal on a revealed cell, move into the wall), +1 on a win;
  dopamine da = r - running mean(r); PAM cells stimulated for 2 steps if da > 0, PPL1 cells if
  da < 0 (0.8 x |da|); w_ij <- clip(w_ij + eta da z_ij |w0_ij|, 0.1 |w0|, 5 |w0|), sign kept.
  eta = 0.03. The mean-subtracted eligibility was added after a first run (v1, raw counts) learned
  only pool priors: the shared component of KC activity (present in every state) otherwise dominates
  every update.
- Supervised warm start (Ramp-style fallback): the cursor follows the teacher for the first 300
  games; when the fly's own softmax choice differs from the teacher's, the teacher's pool is credited
  (+1 x e) and the chosen pool debited (-1 x e), with a PAM pulse marking the teaching event.

### 3.4 Training runs (`outputs/mb/`)

All runs: 9x9, 10 mines, 15 steps per turn, lamina visual input on, NUMBA_NUM_THREADS 4-8.
Per-game logs in `games.jsonl`, curves in `curve.md`, weight checkpoints every 200 games.

Three iterations of the rule were needed; every run is kept:

| run | rule | games | result |
|---|---|---|---|
| `rl_v1` | raw KC counts as eligibility, raw pool counts for the decision | 600 (stopped) | safe cells 51.4 -> 50.5 per 100 games, 0 wins; only pool priors moved (reveal 1.5x, right 1.7x) |
| `warm_v1` | same, 300 teacher-driven warm-start games then RL | 800 (stopped) | teacher agreement (sampled) 0.18 -> 0.33 during warm start, 0.25 after; 0 wins in the RL phase |
| `rl_v2`, `warm_v2`, `shuffled_v2` | mean-subtracted KC eligibility, raw pool counts | 500 / 250 / 350 (stopped) | argmax agreement of `warm_v2` at 200 games 0.393: reveal recall 1.00 but "right" (MBON03, 1.4 spikes/cell) was never chosen against pools at 4-7 spikes/cell |
| **`warm`** (v3) | + pool counts centred on their running mean | 300 warm + 100 RL (stopped at 400) | **argmax agreement with the teacher 0.484 at 200 games** (chance 0.17, linear ceiling 0.85), all six actions used; fell to 0.428 after 100 RL games |
| **`rl`** (v3) | same, no warm start | 800 (running when this was written) | safe cells 53.1 / 53.4 / 52.6 / 53.0 per 100 games, 0 wins |
| **`shuffled`** (v3) | same, reward sign randomised | 500 (complete) | 48.6 / 48.8 / 49.1 / 52.0 / 52.1 per 100 games, 0 wins (1 win in the first block: 1 of 500) |

Learning curve of `warm` (v3), 100-game blocks (`outputs/mb/warm/curve.md`; games 0-299 are the
warm start where the cursor follows the teacher, so their wins are the teacher's, not the fly's):

| games | win rate | mean safe | mean turns | teacher agreement (sampled, T) | mean weight ratio |
|---|---|---|---|---|---|
| 0-99 (warm) | 0.86 | 69.3 | 51.3 | 0.265 (T 0.93) | 1.12 |
| 100-199 (warm) | 0.87 | 70.2 | 54.0 | 0.272 (T 0.80) | 1.32 |
| 200-299 (warm) | 0.79 | 69.1 | 51.4 | 0.296 (T 0.67) | 1.43 |
| 300-399 (RL, fly's own cursor) | 0.00 | 50.0 | 20.9 | 0.268 (T 0.53) | 1.48 |

Argmax teacher agreement of checkpoints on fresh teacher-driven states (533 turns, seeds 80000+):
`warm` 200 games 0.484 (recall up 0.30, down 0.51, left 0.31, right 0.26, reveal 0.82);
`warm` 400 games 0.428; `warm_v2` 200 games 0.393; `rl_v2` 200 games 0.351.

Weight changes are real and logged: after 200 warm-start games 15,060 of the 16,167 plastic edges
differ from the connectome values, mean ratio 1.38 (pool means up 1.28, down 1.74, left 1.36,
right 0.53, reveal 1.80, jump 1.14); nothing else in the brain changed. `pure RL` and
`shuffled reward` produce the same per-game |dw| (0.19-0.29) and the same performance, i.e. the
RL phase so far learns nothing that the shuffled control does not.

### 3.5 Held-out evaluation (seeds 5000-5029, exploration off)

`outputs/validation/mb_warm200/` (weights `outputs/mb/warm/weights_000200.npz`, also installed as
`data/compiled/mb_weights.npz`; softmax temperature 0 = argmax; same 30 boards for every row).
`fly` and `fly-blind` rows are from the earlier run on the identical seeds
(`outputs/validation/readout/`, fixed camera).

| condition | games | win rate | mean safe cells (of 71) | SEM | mean turns | reveals |
|---|---|---|---|---|---|---|
| fly-mb-oracle-only (helper facts, fixed rule, no brain) | 30 | **0.87** | 70.4 | 0.4 | 55.9 | 18.0 |
| solver | 30 | 0.90 | 68.8 | 2.0 | 25.7 | 16.9 |
| random-walk | 30 | 0.00 | 50.4 | 2.5 | 47.1 | 4.3 |
| fly (frozen, descending-neuron pools) | 30 | 0.00 | 48.6 | 2.6 | 400 | – |
| **fly-mb (200 warm-start games)** | 30 | **0.00** | **46.9** | 2.0 | 15.8 | 3.5 |
| random-click | 30 | 0.00 | 39.3 | 2.8 | 3.4 | 3.4 |
| fly-blind | 30 | 0.00 | 38.4 | 3.0 | 400 | – |

**The fly's win rate is 0.00.** `fly-mb` reveals the safe centre cell first (the learned
"reveal when safe-here" association works: 30/30 games open correctly and 82% of the teacher's
reveal states are recognised), then on hidden-but-unproven cells it still reveals about 12% of the
time instead of moving, which on a 10-mine board ends the game within ~16 turns. It is not better
than random-walk on safe cells. No learning claim is made beyond: the KC->MBON weights changed in
the direction the teacher signal pushed them, and the resulting in-brain readout agrees with the
teacher on 48% of states (chance 17%), which is the first non-chance behavioural readout we have
obtained from inside this connectome.

What limits it, in order of evidence:
1. The in-brain readout is far below the information in the KC code (0.48 vs 0.85 for an
   unconstrained linear readout of the same KC counts). The MBON pools (2-4 cells, 15 steps) are
   noisy integrators; positive-only weights in [0.1, 5] x w0 and a Hebbian update are a weak
   optimiser. More warm-start games (agreement was still rising at 200), a longer window or more
   cells per pool would help.
2. The RL phase with the specified immediate rewards did not improve on the warm start and matched
   the shuffled-reward control; the eligibility trace (decay 0.5 per turn) carries too little
   credit to the moves that precede a good reveal.
3. Spontaneous MB activity had to be tamed (KC->KC silenced, KC bias) before any odor code existed;
   the resulting code is dense (80% of KCs active), not the sparse code of a real mushroom body.

## 4. Reproduce

```bash
# approach A
./.venv/bin/python -m flysweeper.train_readout collect --turns 32000 --seed0 20000 --out outputs/training/run1
./.venv/bin/python -m flysweeper.train_readout train --data outputs/training/run1 --out outputs/training/run1/readout_A.npz
./.venv/bin/python -m flysweeper.validate --games 30 --seed0 5000 --conditions fly fly-readout fly-blind random-walk random-click solver --out outputs/validation/readout

# approach B
NUMBA_NUM_THREADS=10 ./.venv/bin/python -m flysweeper.mb_policy --probe --out outputs/mb/probe.json
NUMBA_NUM_THREADS=10 ./.venv/bin/python -m flysweeper.train_mb --games 1200 --seed0 12000 --eta 0.03 --warmstart-games 300 --warmstart-follow --anneal-games 900 --out outputs/mb/warm
NUMBA_NUM_THREADS=10 ./.venv/bin/python -m flysweeper.train_mb --games 1200 --seed0 10000 --eta 0.03 --anneal-games 900 --no-install --out outputs/mb/rl
NUMBA_NUM_THREADS=10 ./.venv/bin/python -m flysweeper.train_mb --games 500  --seed0 10000 --eta 0.03 --shuffle-reward --anneal-games 375 --out outputs/mb/shuffled
NUMBA_NUM_THREADS=10 ./.venv/bin/python -m flysweeper.validate --games 30 --seed0 5000 --conditions fly-mb fly-mb-oracle-only fly fly-blind random-walk random-click solver --mb-weights outputs/mb/warm/weights_final.npz --out outputs/validation/mb
./.venv/bin/python -m flysweeper.server --condition fly-mb      # spectator on the trained weights
```
