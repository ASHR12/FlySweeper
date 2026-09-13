# FlySweeper validation

Generated 2026-09-13T08:21:37+00:00. Board 9x9, 10 mines, 15 brain steps per turn, max 400 turns. Seeds 5000..5099 shared by all conditions.

Brain: MaleCNS v1.0, 0 neurons, 0 edges. Dynamics preset `flyai`, sensory_input=False, input route `lamina`.

| condition | games | win_rate | mean_safe_revealed | sem_safe_revealed | mean_cleared_fraction | mean_turns | mean_reveals | mean_noop_reveals | mean_holds |
|---|---|---|---|---|---|---|---|---|---|
| fly-mb-alt | 100 | 0.76 | 67.58 | 0.9 | 0.952 | 65.9 | 15.2 | 0.0 | 0.0 |

Idle pool rates (Hz per cell, blank screen): 

Reading the table: `fly` vs `fly-blind` isolates the effect of the board on the brain's output; `random-walk` is the same action set with the brain replaced by coin flips; `random-click` and `solver` bracket the task; `fly-readout` is the same frozen brain with the trained readout. `sem_safe_revealed` is the standard error of the mean over games. No learning claim is made unless `fly-learning` beats `fly`, or `fly-readout` beats `random-walk` and `random-click`, on seeds never used for training.
