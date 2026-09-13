# Readout training report

Generated 2026-09-13T04:19:29+00:00. Data `outputs/training/smoke`: 454 teacher-labelled turns from 8 games, seeds 90000..90007, explore fraction 0.25 (pool decoder acted on 106 turns).

Features: 13,562 cells from groups ['dn', 'vp', 'cx', 'mbon'] x 1 window(s) = 13,562 spike counts. Split by game: 7 training games (428 turns), 1 held-out games (26 turns). Regularization by 2-fold grouped CV on the training games.

| model | features | held-out accuracy | balanced accuracy |
|---|---|---|---|
| **brain readout** (ridge {'alpha': 100000.0}) | 13,562 | **0.346** | 0.200 |
| majority class (`reveal`) | 0 | 0.346 | 0.167 |
| brain readout, shuffled labels | 13,562 | 0.346 | 0.200 |
| board-only control: 3x3 around cursor + cursor pos (logreg) | 135 | 0.462 | 0.373 |
| board-only control: whole board + cursor (logreg) | 1053 | 0.269 | 0.209 |

Held-out label mix: {'up': 0.192, 'down': 0.115, 'left': 0.154, 'right': 0.192, 'reveal': 0.346, 'jump': 0.0}

Brain readout per-class recall: {'up': 0.0, 'down': 0.0, 'left': 0.0, 'right': 0.0, 'reveal': 1.0, 'jump': None}
Brain readout predicted mix: {'up': 1, 'down': 0, 'left': 0, 'right': 0, 'reveal': 25, 'jump': 0}

Accuracy by teacher mode: {'first': {'n': 1, 'accuracy': 1.0}, 'safe': {'n': 25, 'accuracy': 0.32}}
Accuracy on explored (off-teacher) turns: {'n': 5, 'accuracy': 0.4}

Confusion (rows = teacher action, cols = predicted; order up, down, left, right, reveal, jump):

    up           0      0      0      0      5      0
    down         0      0      0      0      3      0
    left         1      0      0      0      3      0
    right        0      0      0      0      5      0
    reveal       0      0      0      0      9      0
    jump         0      0      0      0      0      0

Saved readout: `outputs/training/smoke/readout_smoke.npz`
