# Readout training report

Generated 2026-09-13T05:08:45+00:00. Data `outputs/training/run2_ego`: 32,044 teacher-labelled turns from 485 games, seeds 30000..30484, explore fraction 0.25 (pool decoder acted on 8,159 turns).

Features: 13,562 cells from groups ['dn', 'vp', 'cx', 'mbon'] x 1 window(s) = 13,562 spike counts. Split by game: 388 training games (25,770 turns), 97 held-out games (6,274 turns). Regularization by 3-fold grouped CV on the training games.

| model | features | held-out accuracy | balanced accuracy |
|---|---|---|---|
| **brain readout** (ridge {'alpha': 100000.0}) | 13,562 | **0.384** | 0.296 |
| majority class (`reveal`) | 0 | 0.252 | 0.167 |
| brain readout, shuffled labels | 13,562 | 0.253 | 0.167 |
| board-only control: 3x3 around cursor + cursor pos (logreg) | 135 | 0.505 | 0.397 |
| board-only control: whole board + cursor (logreg) | 1053 | 0.337 | 0.279 |

Held-out label mix: {'up': 0.182, 'down': 0.19, 'left': 0.197, 'right': 0.172, 'reveal': 0.252, 'jump': 0.006}

Brain readout per-class recall: {'up': 0.249, 'down': 0.293, 'left': 0.281, 'right': 0.161, 'reveal': 0.793, 'jump': 0.0}
Brain readout predicted mix: {'up': 773, 'down': 1072, 'left': 1051, 'right': 468, 'reveal': 2910, 'jump': 0}

Accuracy by teacher mode: {'first': {'n': 130, 'accuracy': 0.869}, 'guess-free': {'n': 111, 'accuracy': 0.369}, 'guess-frontier': {'n': 139, 'accuracy': 0.324}, 'safe': {'n': 5894, 'accuracy': 0.375}}
Accuracy on explored (off-teacher) turns: {'n': 1570, 'accuracy': 0.376}

Confusion (rows = teacher action, cols = predicted; order up, down, left, right, reveal, jump):

    up         285    209    177     81    393      0
    down       128    350    237     86    393      0
    left       137    236    347     71    442      0
    right      124    184    188    174    412      0
    reveal      93     85    100     50   1256      0
    jump         6      8      2      6     14      0

What else the same brain features decode linearly (held-out accuracy vs majority):

- cursor_row (9 classes): 0.630 vs 0.130
- cursor_col (9 classes): 0.637 vs 0.135
- cell_under_cursor_hidden (2 classes): 0.840 vs 0.513
- cursor_left_half (2 classes): 0.931 vs 0.559

Saved readout: `outputs/training/run2_ego/readout_D.npz`
