# Readout training report

Generated 2026-09-13T04:45:11+00:00. Data `outputs/training/run1`: 32,008 teacher-labelled turns from 472 games, seeds 20000..20471, explore fraction 0.25 (pool decoder acted on 8,081 turns).

Features: 13,562 cells from groups ['dn', 'vp', 'cx', 'mbon'] x 1 window(s) = 13,562 spike counts. Split by game: 378 training games (24,672 turns), 94 held-out games (7,336 turns). Regularization by 3-fold grouped CV on the training games.

| model | features | held-out accuracy | balanced accuracy |
|---|---|---|---|
| **brain readout** (ridge {'alpha': 100000.0}) | 13,562 | **0.279** | 0.215 |
| majority class (`reveal`) | 0 | 0.267 | 0.167 |
| brain readout, shuffled labels | 13,562 | 0.267 | 0.167 |
| board-only control: 3x3 around cursor + cursor pos (logreg) | 135 | 0.510 | 0.393 |
| board-only control: whole board + cursor (logreg) | 1053 | 0.322 | 0.261 |

Held-out label mix: {'up': 0.185, 'down': 0.188, 'left': 0.182, 'right': 0.173, 'reveal': 0.267, 'jump': 0.004}

Brain readout per-class recall: {'up': 0.208, 'down': 0.192, 'left': 0.185, 'right': 0.196, 'reveal': 0.511, 'jump': 0.0}
Brain readout predicted mix: {'up': 1090, 'down': 976, 'left': 936, 'right': 971, 'reveal': 3363, 'jump': 0}

Accuracy by teacher mode: {'first': {'n': 138, 'accuracy': 0.819}, 'guess-free': {'n': 266, 'accuracy': 0.32}, 'guess-frontier': {'n': 139, 'accuracy': 0.237}, 'safe': {'n': 6793, 'accuracy': 0.267}}
Accuracy on explored (off-teacher) turns: {'n': 1838, 'accuracy': 0.297}

Confusion (rows = teacher action, cols = predicted; order up, down, left, right, reveal, jump):

    up         282    152    159    165    601      0
    down       194    266    154    169    599      0
    left       177    166    248    140    607      0
    right      184    151    141    249    543      0
    reveal     248    239    230    243   1002      0
    jump         5      2      4      5     11      0

What else the same brain features decode linearly (held-out accuracy vs majority):

- cursor_row (9 classes): 0.532 vs 0.115
- cursor_col (9 classes): 0.594 vs 0.117
- cell_under_cursor_hidden (2 classes): 0.540 vs 0.497
- cursor_left_half (2 classes): 0.821 vs 0.583

Saved readout: `/Users/ashutosh/development-work/fruit-fly/data/compiled/readout.npz`
