# FlySweeper validation

Generated 2026-09-13T06:06:45+00:00. Board 9x9, 10 mines, 15 brain steps per turn, max 400 turns. Seeds 5000..5029 shared by all conditions.

Brain: MaleCNS v1.0, 166,700 neurons, 25,582,938 edges. Dynamics preset `flyai`, sensory_input=False, input route `lamina`.

| condition | games | win_rate | mean_safe_revealed | sem_safe_revealed | mean_cleared_fraction | mean_turns | mean_reveals | mean_noop_reveals | mean_holds |
|---|---|---|---|---|---|---|---|---|---|
| fly-mb | 30 | 0.0 | 46.87 | 2.04 | 0.66 | 15.8 | 3.5 | 0.6 | 0.0 |
| fly-mb-oracle-only | 30 | 0.867 | 70.4 | 0.39 | 0.992 | 55.9 | 18.0 | 0.0 | 0.0 |
| random-walk | 30 | 0.0 | 50.4 | 2.49 | 0.71 | 47.1 | 4.3 | 3.2 | 0.0 |
| random-click | 30 | 0.0 | 39.3 | 2.75 | 0.554 | 3.4 | 3.4 | 0.0 | 0.0 |
| solver | 30 | 0.9 | 68.8 | 1.97 | 0.969 | 25.7 | 16.9 | 0.0 | 0.0 |

Idle pool rates (Hz per cell, blank screen): up 0.33, down 0.15, left 0.67, right 0.53, reveal 1.15, jump 0.45

Mushroom-body policy (`fly-mb`): a helper reads the board into 31 facts injected as odors (one ORN type each); actions are read from MBON pools {'up': 'MBON01', 'down': 'MBON05', 'left': 'MBON06', 'right': 'MBON03', 'reveal': 'MBON11', 'jump': 'MBON09'}; the only trained synapses are the 16,167 KC->MBON edges onto those pools. Weights: `outputs/mb/warm/weights_000200.npz` ({'seed0': 12000, 'games': 800}, 200 games); softmax temperature 0.0 (0 = argmax). Mean weight ratio per pool: {'up': 1.2781, 'down': 1.7424, 'left': 1.3633, 'right': 0.5279, 'reveal': 1.7995, 'jump': 1.1388}; 15,055 edges differ from the frozen wiring (min ratio 0.1, max 5.0, 1,274 at the floor, 714 at the ceiling).
Documented KC changes for this condition only: KC->KC gain 0.0 (642,933 edges), KC bias -0.2, PN->KC gain 1.0. `fly-mb-oracle-only` reads the same facts with a fixed rule and no brain.

Reading the table: `fly` vs `fly-blind` isolates the effect of the board on the brain's output; `random-walk` is the same action set with the brain replaced by coin flips; `random-click` and `solver` bracket the task; `fly-readout` is the same frozen brain with the trained readout. `sem_safe_revealed` is the standard error of the mean over games. No learning claim is made unless `fly-learning` beats `fly`, or `fly-readout` beats `random-walk` and `random-click`, on seeds never used for training.
