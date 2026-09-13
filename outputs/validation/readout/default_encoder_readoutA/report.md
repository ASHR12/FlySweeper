# FlySweeper validation

Generated 2026-09-13T05:04:36+00:00. Board 9x9, 10 mines, 15 brain steps per turn, max 400 turns. Seeds 5000..5029 shared by all conditions.

Brain: MaleCNS v1.0, 166,700 neurons, 25,582,938 edges. Dynamics preset `flyai`, sensory_input=False, input route `lamina`.

| condition | games | win_rate | mean_safe_revealed | sem_safe_revealed | mean_cleared_fraction | mean_turns | mean_reveals | mean_noop_reveals | mean_holds |
|---|---|---|---|---|---|---|---|---|---|
| fly | 30 | 0.0 | 48.63 | 2.64 | 0.685 | 42.3 | 5.5 | 1.2 | 1.7 |
| fly-readout | 30 | 0.0 | 48.13 | 2.34 | 0.678 | 107.0 | 4.4 | 80.1 | 0.0 |
| fly-blind | 30 | 0.0 | 38.43 | 3.03 | 0.541 | 30.7 | 3.8 | 2.4 | 1.1 |
| random-walk | 30 | 0.0 | 50.4 | 2.49 | 0.71 | 47.1 | 4.3 | 3.2 | 0.0 |
| random-click | 30 | 0.0 | 39.3 | 2.75 | 0.554 | 3.4 | 3.4 | 0.0 | 0.0 |
| solver | 30 | 0.9 | 68.8 | 1.97 | 0.969 | 25.7 | 16.9 | 0.0 | 0.0 |

Idle pool rates (Hz per cell, blank screen): up 0.33, down 0.15, left 0.67, right 0.53, reveal 1.15, jump 0.45

Readout (`fly-readout`): `outputs/training/run1/readout_A.npz`, 13,562 cells x 1 window(s) from groups ['dn', 'vp', 'cx', 'mbon'], ridge; trained on seeds [20000, 20471] (32,008 turns), held-out turn accuracy 0.27903489640130863.

Reading the table: `fly` vs `fly-blind` isolates the effect of the board on the brain's output; `random-walk` is the same action set with the brain replaced by coin flips; `random-click` and `solver` bracket the task; `fly-readout` is the same frozen brain with the trained readout. `sem_safe_revealed` is the standard error of the mean over games. No learning claim is made unless `fly-learning` beats `fly`, or `fly-readout` beats `random-walk` and `random-click`, on seeds never used for training.
