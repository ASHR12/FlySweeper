"""The helper: read the board into a few facts about the cell under the cursor ("what the fly smells").

Honest framing, kept everywhere: a helper reads the board into a few facts; the fly's mushroom
body learns what to do about them.  Everything in this file is ordinary Minesweeper logic
(the same single-point inference as `teacher.infer`); none of it is the fly.  Its output is a
vector of CHANNELS values in [0, 1], one per channel, which `mb_policy.OdorEncoder` turns into
drive on one olfactory receptor neuron type per channel.

Channels (index: name - meaning):
     0 cursor_hidden        the cell under the cursor is hidden
     1 cursor_revealed      ... is a revealed number (a reveal here is a no-op)
     2 cursor_flagged       ... is flagged (the fly cannot flag; kept for human boards)
     3 cursor_safe          ... is provably safe (single-point inference; also 1 before the first
                            click, when every cell is safe and the teacher's target is the centre)
     4 cursor_mine          ... is provably a mine
     5 cursor_frontier      ... is hidden, touches a revealed number, and is not provable either way
     6 cursor_free          ... is hidden and touches no revealed number ("unconstrained")
     7 adj_numbers_0        number of revealed numbers among the 8 neighbours: 0
     8 adj_numbers_1_2      ... 1-2
     9 adj_numbers_3p       ... 3 or more
    10 adj_hidden_0         number of hidden neighbours: 0
    11 adj_hidden_1_3       ... 1-3
    12 adj_hidden_4p        ... 4 or more
    13 adj_max_0            largest revealed number among the neighbours: none / 0
    14 adj_max_1_2          ... 1-2
    15 adj_max_3p           ... 3 or more
    16 safe_up              nearest provably-safe cell lies above the cursor, graded 1/distance
    17 safe_down            ... below
    18 safe_left            ... to the left
    19 safe_right           ... to the right      (16-19 are 0 when none is known)
    20 no_safe_known        the board has been touched and no provably-safe cell exists
    21 guess_up             nearest guess target (unconstrained hidden cell, else the least risky
    22 guess_down           frontier cell) lies above / below / left / right, graded 1/distance.
    23 guess_left           Only active when no safe cell is known (so a linear reader can use
    24 guess_right          them without an AND with channel 20).
    25 guess_here           no safe cell is known and the cursor is on the guess target
    26 guess_far            no safe cell is known and the guess target is > jump_distance away
    27 cleared_lt_third     fraction of safe cells revealed < 1/3
    28 cleared_mid          ... between 1/3 and 2/3
    29 cleared_gt_two_thirds ... > 2/3
    30 untouched            nothing revealed yet (the first click is always safe)

Distances are Manhattan; "graded 1/distance" means 1 for the adjacent cell, 1/2 two away, ...
Both axis components are set when the target is diagonal (either step shortens the path).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .minesweeper import FLAG, HIDDEN, NEIGHBORS
from .teacher import infer, mine_probability, _neighbors, _nearest

CHANNELS = (
    "cursor_hidden", "cursor_revealed", "cursor_flagged", "cursor_safe", "cursor_mine", "cursor_frontier", "cursor_free",
    "adj_numbers_0", "adj_numbers_1_2", "adj_numbers_3p",
    "adj_hidden_0", "adj_hidden_1_3", "adj_hidden_4p",
    "adj_max_0", "adj_max_1_2", "adj_max_3p",
    "safe_up", "safe_down", "safe_left", "safe_right", "no_safe_known",
    "guess_up", "guess_down", "guess_left", "guess_right", "guess_here", "guess_far",
    "cleared_lt_third", "cleared_mid", "cleared_gt_two_thirds", "untouched",
)
CH = {name: i for i, name in enumerate(CHANNELS)}
N_CHANNELS = len(CHANNELS)
ACTIONS = ("up", "down", "left", "right", "reveal", "jump")


@dataclass
class OracleParams:
    jump_distance: int = 5        # same as the teacher: guess targets farther than this -> "jump"


def _direction_channels(vec: np.ndarray, base: int, cursor: tuple[int, int], target: tuple[int, int]) -> None:
    dr, dc = target[0] - cursor[0], target[1] - cursor[1]
    dist = abs(dr) + abs(dc)
    if dist == 0:
        return
    g = 1.0 / dist
    if dr < 0:
        vec[base + 0] = g
    elif dr > 0:
        vec[base + 1] = g
    if dc < 0:
        vec[base + 2] = g
    elif dc > 0:
        vec[base + 3] = g


class Oracle:
    """Computes the channel vector and, for the ceiling condition, a trivial rule-based action."""

    def __init__(self, params: OracleParams | None = None):
        self.p = params or OracleParams()
        self.last: dict = {}

    def features(self, visible: np.ndarray, cursor: tuple[int, int], n_mines: int = 10) -> np.ndarray:
        rows, cols = visible.shape
        r, c = cursor
        v = np.zeros(N_CHANNELS, dtype=np.float32)
        revealed = (visible >= 0) & (visible <= 8)
        unknown = (visible == HIDDEN) | (visible == FLAG)
        cell = int(visible[r, c])
        v[CH["cursor_hidden"]] = cell == HIDDEN
        v[CH["cursor_revealed"]] = 0 <= cell <= 8
        v[CH["cursor_flagged"]] = cell == FLAG
        untouched = not revealed.any()
        v[CH["untouched"]] = untouched

        nbrs = list(_neighbors(r, c, rows, cols))
        n_numbers = sum(1 for p in nbrs if revealed[p])
        n_hidden = sum(1 for p in nbrs if unknown[p])
        max_num = max([int(visible[p]) for p in nbrs if revealed[p]], default=0)
        v[CH["adj_numbers_0"] + (0 if n_numbers == 0 else 1 if n_numbers <= 2 else 2)] = 1.0
        v[CH["adj_hidden_0"] + (0 if n_hidden == 0 else 1 if n_hidden <= 3 else 2)] = 1.0
        v[CH["adj_max_0"] + (0 if max_num == 0 else 1 if max_num <= 2 else 2)] = 1.0

        if untouched:
            safe, mines = {(rows // 2, cols // 2)}, set()
            frontier: set = set()
        else:
            safe, mines = infer(visible)
            frontier = set()
            for rr in range(rows):
                for cc in range(cols):
                    if revealed[rr, cc]:
                        frontier.update(_neighbors(rr, cc, rows, cols))
        here = (r, c)
        v[CH["cursor_safe"]] = here in safe or (untouched and unknown[r, c])
        v[CH["cursor_mine"]] = here in mines
        v[CH["cursor_frontier"]] = unknown[r, c] and here in frontier and here not in safe and here not in mines
        v[CH["cursor_free"]] = unknown[r, c] and here not in frontier and not untouched

        target = None
        mode = "safe"
        if safe:
            target = _nearest(cursor, safe)
            _direction_channels(v, CH["safe_up"], cursor, target)
        else:
            v[CH["no_safe_known"]] = 1.0
            cand = [(rr, cc) for rr in range(rows) for cc in range(cols) if unknown[rr, cc] and (rr, cc) not in mines]
            if cand:
                free = [p for p in cand if p not in frontier]
                if free:
                    target = _nearest(cursor, free)
                    mode = "guess-free"
                else:
                    prob = mine_probability(visible, mines)
                    target = min(cand, key=lambda p: (round(prob.get(p, 1.0), 6), abs(p[0] - r) + abs(p[1] - c), p[0], p[1]))
                    mode = "guess-frontier"
                dist = abs(target[0] - r) + abs(target[1] - c)
                if dist == 0:
                    v[CH["guess_here"]] = 1.0
                elif dist > self.p.jump_distance:
                    v[CH["guess_far"]] = 1.0
                else:
                    _direction_channels(v, CH["guess_up"], cursor, target)
            else:
                mode = "none"

        frac = float(revealed.sum()) / max(1, rows * cols - n_mines)
        v[CH["cleared_lt_third"] + (0 if frac < 1 / 3 else 1 if frac < 2 / 3 else 2)] = 1.0
        self.last = {"target": target, "mode": mode, "safe": len(safe), "mines": len(mines)}
        return v

    @staticmethod
    def rule_action(v: np.ndarray) -> str:
        """The trivial reader of the same channels (the `fly-mb-oracle-only` ceiling): reveal when the
        cursor cell is provably safe or is the guess target; otherwise step along the strongest
        direction channel; jump when the guess target is far; else reveal (nothing better known)."""
        if v[CH["cursor_safe"]] > 0 or v[CH["guess_here"]] > 0:
            return "reveal"
        if v[CH["guess_far"]] > 0:
            return "jump"
        dirs = v[CH["safe_up"]:CH["safe_right"] + 1] + v[CH["guess_up"]:CH["guess_right"] + 1]
        if dirs.max() > 0:
            return ACTIONS[int(np.argmax(dirs))]
        return "jump"


def describe_channels() -> str:
    return "\n".join(f"{i:2d} {name}" for i, name in enumerate(CHANNELS))
