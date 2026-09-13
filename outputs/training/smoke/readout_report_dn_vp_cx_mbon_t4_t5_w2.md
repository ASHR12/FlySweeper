# Readout training report

Generated 2026-09-13T04:22:15+00:00. Data `outputs/training/smoke, outputs/training/smoke`: 908 teacher-labelled turns from 16 games, seeds 90000..90007, explore fraction 0.25 (pool decoder acted on 212 turns).

Features: 27,147 cells from groups ['dn', 'vp', 'cx', 'mbon', 't4', 't5'] x 2 window(s) = 54,294 spike counts. Split by game: 7 training games (856 turns), 1 held-out games (52 turns). Regularization by 2-fold grouped CV on the training games.

| model | features | held-out accuracy | balanced accuracy |
|---|---|---|---|
| **brain readout** (ridge {'alpha': 1000000.0}) | 54,294 | **0.346** | 0.200 |
| majority class (`reveal`) | 0 | 0.346 | 0.167 |

Held-out label mix: {'up': 0.192, 'down': 0.115, 'left': 0.154, 'right': 0.192, 'reveal': 0.346, 'jump': 0.0}

Brain readout per-class recall: {'up': 0.0, 'down': 0.0, 'left': 0.0, 'right': 0.0, 'reveal': 1.0, 'jump': None}
Brain readout predicted mix: {'up': 0, 'down': 0, 'left': 0, 'right': 0, 'reveal': 52, 'jump': 0}


Confusion (rows = teacher action, cols = predicted; order up, down, left, right, reveal, jump):

    up           0      0      0      0     10      0
    down         0      0      0      0      6      0
    left         0      0      0      0      8      0
    right        0      0      0      0     10      0
    reveal       0      0      0      0     18      0
    jump         0      0      0      0      0      0

What else the same brain features decode linearly (held-out accuracy vs majority):

- cursor_row (9 classes): 0.308 vs 0.000
- cursor_col (9 classes): 0.385 vs 0.038
- cell_under_cursor_hidden (2 classes): 0.423 vs 0.462
- cursor_left_half (2 classes): 0.462 vs 0.462

Saved readout: `outputs/training/smoke/readout_smoke.npz`
