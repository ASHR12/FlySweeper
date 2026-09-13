# Readout training report

Generated 2026-09-13T05:11:05+00:00. Data `outputs/training/run2_ego`: 32,044 teacher-labelled turns from 485 games, seeds 30000..30484, explore fraction 0.25 (pool decoder acted on 8,159 turns).

Features: 27,147 cells from groups ['dn', 'vp', 'cx', 'mbon', 't4', 't5'] x 1 window(s) = 27,147 spike counts. Split by game: 388 training games (25,770 turns), 97 held-out games (6,274 turns). Regularization by 3-fold grouped CV on the training games.

| model | features | held-out accuracy | balanced accuracy |
|---|---|---|---|
| **brain readout** (ridge {'alpha': 100000.0}) | 27,147 | **0.446** | 0.348 |
| majority class (`reveal`) | 0 | 0.252 | 0.167 |
| brain readout, shuffled labels | 27,147 | 0.253 | 0.168 |
| board-only control: 3x3 around cursor + cursor pos (logreg) | 135 | 0.505 | 0.397 |
| board-only control: whole board + cursor (logreg) | 1053 | 0.337 | 0.279 |

Held-out label mix: {'up': 0.182, 'down': 0.19, 'left': 0.197, 'right': 0.172, 'reveal': 0.252, 'jump': 0.006}

Brain readout per-class recall: {'up': 0.301, 'down': 0.328, 'left': 0.342, 'right': 0.25, 'reveal': 0.863, 'jump': 0.0}
Brain readout predicted mix: {'up': 806, 'down': 1080, 'left': 1114, 'right': 668, 'reveal': 2606, 'jump': 0}

Accuracy by teacher mode: {'first': {'n': 130, 'accuracy': 0.862}, 'guess-free': {'n': 111, 'accuracy': 0.414}, 'guess-frontier': {'n': 139, 'accuracy': 0.36}, 'safe': {'n': 5894, 'accuracy': 0.439}}
Accuracy on explored (off-teacher) turns: {'n': 1570, 'accuracy': 0.438}

Confusion (rows = teacher action, cols = predicted; order up, down, left, right, reveal, jump):

    up         345    199    206    104    291      0
    down       143    392    221    140    298      0
    left       128    247    422    104    332      0
    right      125    183    198    271    305      0
    reveal      59     49     65     44   1367      0
    jump         6     10      2      5     13      0

What else the same brain features decode linearly (held-out accuracy vs majority):

- cursor_row (9 classes): 0.718 vs 0.130
- cursor_col (9 classes): 0.708 vs 0.135
- cell_under_cursor_hidden (2 classes): 0.955 vs 0.513
- cursor_left_half (2 classes): 0.953 vs 0.559

Saved readout: `outputs/training/run2_ego/readout_E.npz`
