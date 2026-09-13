# Readout training report

Generated 2026-09-13T04:59:07+00:00. Data `outputs/training/run1`: 32,008 teacher-labelled turns from 472 games, seeds 20000..20471, explore fraction 0.25 (pool decoder acted on 8,081 turns).

Features: 13,562 cells from groups ['dn', 'vp', 'cx', 'mbon'] x 2 window(s) = 27,124 spike counts. Split by game: 378 training games (24,672 turns), 94 held-out games (7,336 turns). Regularization by 3-fold grouped CV on the training games.

| model | features | held-out accuracy | balanced accuracy |
|---|---|---|---|
| **brain readout** (ridge {'alpha': 100000.0}) | 27,124 | **0.287** | 0.225 |
| majority class (`reveal`) | 0 | 0.267 | 0.167 |

Held-out label mix: {'up': 0.185, 'down': 0.188, 'left': 0.182, 'right': 0.173, 'reveal': 0.267, 'jump': 0.004}

Brain readout per-class recall: {'up': 0.223, 'down': 0.221, 'left': 0.212, 'right': 0.227, 'reveal': 0.47, 'jump': 0.0}
Brain readout predicted mix: {'up': 1185, 'down': 1165, 'left': 1039, 'right': 1066, 'reveal': 2881, 'jump': 0}


Confusion (rows = teacher action, cols = predicted; order up, down, left, right, reveal, jump):

    up         303    202    186    187    481      0
    down       204    306    176    177    519      0
    left       230    211    283    141    473      0
    right      184    169    151    288    476      0
    reveal     260    274    240    266    922      0
    jump         4      3      3      7     10      0

Saved readout: `outputs/training/run1/readout_C.npz`
