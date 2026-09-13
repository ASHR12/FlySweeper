# FlySweeper validation

Generated 2026-09-13T05:11:23+00:00. Board 9x9, 10 mines, 15 brain steps per turn, max 400 turns. Seeds 6000..6029 shared by all conditions.

Brain: MaleCNS v1.0, 166,700 neurons, 25,582,938 edges. Dynamics preset `flyai`, sensory_input=False, input route `lamina`.

| condition | games | win_rate | mean_safe_revealed | sem_safe_revealed | mean_cleared_fraction | mean_turns | mean_reveals | mean_noop_reveals | mean_holds |
|---|---|---|---|---|---|---|---|---|---|
| fly | 30 | 0.0 | 54.43 | 1.53 | 0.767 | 39.3 | 5.0 | 1.7 | 1.2 |
| fly-learning | 30 | 0.0 | 50.23 | 2.44 | 0.708 | 37.4 | 4.8 | 1.2 | 0.9 |

Idle pool rates (Hz per cell, blank screen): up 0.27, down 0.22, left 0.70, right 0.63, reveal 0.98, jump 0.55

Plasticity (`fly-learning`, 30 games, cumulative): 145 dopamine events, 61,210 of 61,210 KC->MBON edges differ from the frozen wiring, total |dw| 9720.731, mean weight ratio 0.1 (min 0.1, max 0.1; 61,210 at the floor, 0 at the ceiling). Mean ratio after each game: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1].

Reading the table: `fly` vs `fly-blind` isolates the effect of the board on the brain's output; `random-walk` is the same action set with the brain replaced by coin flips; `random-click` and `solver` bracket the task; `fly-readout` is the same frozen brain with the trained readout. `sem_safe_revealed` is the standard error of the mean over games. No learning claim is made unless `fly-learning` beats `fly`, or `fly-readout` beats `random-walk` and `random-click`, on seeds never used for training.
