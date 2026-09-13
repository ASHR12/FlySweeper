# Trained weight sets (KC→MBON synapses only)

Each file here is a complete `fly-mb` policy: the trained Kenyon-cell → mushroom-body-output-neuron
weights **plus** every setting needed to replay them (MBON pools, odor map, KC/PN biases and gains,
reveal margin). Nothing else in the connectome is ever changed; every other one of the
25,582,938 edges stays at its MaleCNS v1.0 value. Only these files are redistributed, never the
source connectome tables (see [`DATA-LICENSE.md`](../DATA-LICENSE.md)).

The training story behind each round, with all numbers and commands, is in
[`docs/training.md`](../docs/training.md); the raw evaluation reports are in
[`docs/results/`](../docs/results/README.md).

## Versions

| round | file | sha256 | games trained | plastic edges | held-out wins, 30 seeds (5000–5029) | held-out wins, 100 seeds (5000–5099) | notes |
|---|---|---|---|---|---|---|---|
| 1 | `mb_weights_round1.npz` | `75eb11eb21eec2b0c3252d224b622f59e9072d0db8d8e992d92ec37c27496fb3` | 200 (teacher-driven warm start; checkpoint of run `warm`) | 16,167 (6 single-type MBON pools: MBON01/05/06/03/11/09) | **0/30** (0.00), 46.9 ± 2.0 safe cells | not run | dense KC code (KC→KC gain 0, KC bias −0.2, PN bias 0, PN→KC gain 1); weight range [0.1, 5]×w0; teacher agreement 0.484; unproven-reveal rate ≈12 %; 15,055 edges changed. Source: `outputs/mb/warm/weights_000200.npz` |
| 2 | `mb_weights_round2.npz` | `4bd34de4ed38ae1d037013d8f5c3b09b49c1c3865e6d5997587f4ed3f6af69d9` | 1,500 (teacher-driven, perceptron rule; run `r2_error_sparse`) | 59,334 (34 MBON types dealt into 6 pools, 91 cells) | **18/30** (0.60), 67.1 ± 1.8 safe cells | **48/100** (0.48), 66.7 ± 0.8 safe cells | sparse KC code (PN bias −0.3, PN→KC gain 3, KC bias −0.3, KC→KC gain 0); floor 0×w0, ceiling 10×w0 (none reached); reveal margin 0.25 stored in the file; agreement 0.735 (0.722 with the margin); reveal false-positive rate 0.7 % (0.2 % with the margin); 55,832 edges changed, 5,152 silenced, mean ratio 1.45. Source: `outputs/mb/r2_error_sparse/weights_001500.npz`, installed with the margin by `mb_eval --install` |
| **3** | `mb_weights_round3.npz` | `993582d7cb66172043a61ae7ba3c1585e0531b41ae388bc8b1659b14e233ffc8` | 2,000 (the round-2 weights continued for 500 more teacher-driven games; checkpoint 500 of run `r3_teacher_l2`) | 59,334 (same pools as round 2) | **25/30** (0.83), 69.0 ± 1.3 safe cells — measured with reveal margin 0.25; the installed file uses margin 0 | **76/100** (0.76), 67.6 ± 0.9 safe cells, 66 turns/game (margin 0, mask on, as installed) | level-2 odor map (`extra_orn` 2: a second ORN type for each of the 8 graded direction channels, 49 of 53 ORN types used); perceptron rule with argmax decisions, move-vs-move errors weighted 2×; reveal mask (reveal not selectable on revealed / provably-mine cells); reveal margin 0.0; same sparse KC code as round 2; agreement 0.763, reveal false-positive rate 0.000; 55,907 edges changed, 2,673 silenced, mean ratio 1.68. Source: `outputs/mb/r3_teacher_l2/weights_000500.npz`, installed by `mb_eval --install` |

Round 3 is the currently installed policy (**0.76 ± 0.04**, binomial SE on 100 games; the round-2
weights scored 0.60 in the same run and 0.48 in an earlier run on the same seeds with a different
Numba thread count — the paired 16-point gap is the robust number). An RL fine-tune of the same
checkpoint with the redesigned round-3 rewards (`outputs/mb/r3_rl/weights_000400.npz`) also
reached 76/100 but took 107 instead of 66 turns per game and lost teacher agreement, so the
teacher-only checkpoint was installed; it is not shipped here. Round 1 is kept because it is the
checkpoint the round-1 evaluation in `docs/training.md` §3.5 was run on (0 wins; the first
non-chance in-brain readout, 48 % teacher agreement).

`SHA256SUMS` in this folder lists the same digests in `shasum -c` format.

## How a weight set is used

`fly-mb` (in `flysweeper/agent.py`, `FlyPlayer.enable_mb`) loads **one** file and takes everything
from it: the plastic edge list, the trained values, the pools, the odor→ORN map, the KC/PN
settings and the reveal margin. It then plays with learning off, exploration off (argmax,
temperature 0) and the file's reveal margin, exactly as `validate.py` evaluates it.

Default location: `data/compiled/mb_weights.npz` (gitignored, because `data/` is). Install a round
there with the helper script:

```bash
scripts/install_weights.sh round3          # copies models/mb_weights_round3.npz -> data/compiled/mb_weights.npz (current)
scripts/install_weights.sh round2          # the previous release
scripts/install_weights.sh round1          # the round-1 checkpoint
scripts/install_weights.sh path/to/any.npz # any checkpoint written by train_mb.py / mb_eval.py --install
```

The script verifies the sha256 of a named round against `SHA256SUMS` and keeps a `.bak` of any file
it overwrites. Then:

```bash
python -m flysweeper.server --condition fly-mb                      # live spectator on the installed weights
python -m flysweeper.validate --games 30 --seed0 5000 \
    --conditions fly-mb fly-mb-oracle-only random-walk               # same file, held-out games
python -m flysweeper.validate --games 30 --seed0 5000 --conditions fly-mb \
    --mb-weights models/mb_weights_round1.npz                        # a specific file without installing it
python -m flysweeper.mb_eval models/mb_weights_round3.npz --games 30 --agree-turns 600   # agreement + confusion + games
```

`server.py` has no weights flag; it always reads `data/compiled/mb_weights.npz`, so use the install
script to switch what the spectator plays.

## File format

A NumPy `.npz` written by `flysweeper/mb_policy.py` (`MBPolicy.save`):

| key | content |
|---|---|
| `edge_pos` | int64, position of each plastic edge in the compiled out-edge CSR (`data/compiled/graph.npz`) |
| `w` | float32, trained weight of each plastic edge (normalized units, same as the CSR) |
| `w0` | float32, the connectome's original weight of that edge (so the ratio `w / w0` is the learned change) |
| `edge_pool`, `edge_kc` | int8 / int64, action pool index and presynaptic Kenyon-cell index of each edge |
| `actions` | the six action names in pool order (`up down left right reveal jump`) |
| `score_mean`, `baseline` | running means used to centre the pool scores and the reward |
| `games_trained` | number of games the file has seen |
| `params` | JSON: the full `MBParams` (pools, `extra_orn`, `kc_kc_gain`, `kc_bias`, `pn_bias`, `pn_kc_gain`, `reveal_margin`, `mask_reveal`, rewards, learning rate, clip range, ...) |
| `meta` | JSON: training arguments, seeds, start time, `installed_from`, and the evaluation summary at install time |
| `channels`, `orn_types` | the 31 helper channels and the ORN type(s) each one drives |

Because the edge set is tied to the compiled graph, a weight file only makes sense with a graph
compiled from the same MaleCNS v1.0 release with the same policies (`data/compiled/meta.json`).

## Versioning policy

* One file per training round, never overwritten; a new round adds a row and a file.
* The row states the exact source checkpoint, the seeds it was evaluated on, and every model change
  that applies only when that file is loaded (KC/PN settings, reveal margin, mask).
* Held-out numbers are from seeds 5000–5029 / 5000–5099 with exploration off, the same boards for
  every condition; training seeds are ≥ 10000 and are never used for evaluation.
* A file is "installed" when it is copied to `data/compiled/mb_weights.npz`; the copy in `models/`
  is the record.
