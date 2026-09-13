# Readout training report

Generated 2026-09-13T04:51:56+00:00. Data `outputs/training/run1`: 32,008 teacher-labelled turns from 472 games, seeds 20000..20471, explore fraction 0.25 (pool decoder acted on 8,081 turns).

Features: 27,147 cells from groups ['dn', 'vp', 'cx', 'mbon', 't4', 't5'] x 1 window(s) = 27,147 spike counts. Split by game: 378 training games (24,672 turns), 94 held-out games (7,336 turns). Regularization by 3-fold grouped CV on the training games.

| model | features | held-out accuracy | balanced accuracy |
|---|---|---|---|
| **brain readout** (ridge {'alpha': 1000000.0}) | 27,147 | **0.300** | 0.229 |
| majority class (`reveal`) | 0 | 0.267 | 0.167 |
| brain readout, shuffled labels | 27,147 | 0.268 | 0.168 |
| board-only control: 3x3 around cursor + cursor pos (logreg) | 135 | 0.510 | 0.393 |
| board-only control: whole board + cursor (logreg) | 1053 | 0.322 | 0.261 |

Held-out label mix: {'up': 0.185, 'down': 0.188, 'left': 0.182, 'right': 0.173, 'reveal': 0.267, 'jump': 0.004}

Brain readout per-class recall: {'up': 0.183, 'down': 0.214, 'left': 0.183, 'right': 0.207, 'reveal': 0.587, 'jump': 0.0}
Brain readout predicted mix: {'up': 903, 'down': 926, 'left': 761, 'right': 882, 'reveal': 3864, 'jump': 0}

Accuracy by teacher mode: {'first': {'n': 138, 'accuracy': 0.819}, 'guess-free': {'n': 266, 'accuracy': 0.305}, 'guess-frontier': {'n': 139, 'accuracy': 0.266}, 'safe': {'n': 6793, 'accuracy': 0.29}}
Accuracy on explored (off-teacher) turns: {'n': 1838, 'accuracy': 0.324}

Confusion (rows = teacher action, cols = predicted; order up, down, left, right, reveal, jump):

    up         249    124    113    142    731      0
    down       147    296    125    164    650      0
    left       148    152    245     91    702      0
    right      149    139     99    263    618      0
    reveal     205    213    175    218   1151      0
    jump         5      2      4      4     12      0

What else the same brain features decode linearly (held-out accuracy vs majority):

- cursor_row (9 classes): 0.677 vs 0.115
- cursor_col (9 classes): 0.706 vs 0.117
- cell_under_cursor_hidden (2 classes): 0.549 vs 0.497
- cursor_left_half (2 classes): 0.849 vs 0.583

Saved readout: `outputs/training/run1/readout_B.npz`
