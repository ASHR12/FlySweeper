"""A teacher that plays Minesweeper *in the fly's own action space*.

The fly can only do six things per turn: move the cursor up / down / left / right by one cell,
reveal the cell under the cursor, or "jump" (teleport the cursor to a uniformly random hidden
cell).  The teacher expresses single-point Minesweeper logic (the same inference as
`minesweeper.solver_move`) as a sequence of those six actions:

    1. Nothing revealed yet   -> walk to the board centre and reveal it (the safe first click).
    2. A certainly-safe hidden cell exists (some revealed number already has all of its mines
       accounted for) -> walk toward the nearest one (Manhattan distance; ties broken by row,
       then column; the axis with the larger remaining offset is walked first, vertical on a
       tie) and reveal when standing on it.
    3. No safe cell is known -> pick a guess target: the nearest hidden cell that touches no
       revealed number ("unconstrained": no evidence against it), or, if every hidden cell is
       on the frontier, the one with the lowest single-constraint mine probability.  Walk there
       and reveal; if the target is more than `jump_distance` steps away, jump instead (any
       hidden cell is roughly as good a guess, and a jump costs one turn).
    4. Never reveal a revealed / flagged cell or a cell known to be a mine.

The fly cannot flag, so certain mines are tracked in the teacher's head (from the visible grid
alone, iterated to a fixpoint) rather than on the board.  The policy is a deterministic function
of (visible grid, cursor); `rng` is accepted for interface symmetry but not used.

    python -m flysweeper.teacher --games 20      # plays the teacher alone, reports its score
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np

from .minesweeper import FLAG, HIDDEN, NEIGHBORS, Minesweeper

ACTIONS = ("up", "down", "left", "right", "reveal", "jump")


@dataclass
class TeacherParams:
    jump_distance: int = 5        # guess targets farther than this (Manhattan) -> "jump"


def _neighbors(r: int, c: int, rows: int, cols: int):
    for dr, dc in NEIGHBORS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            yield rr, cc


def infer(visible: np.ndarray) -> tuple[set, set]:
    """Single-point inference on the visible grid, iterated to a fixpoint.

    Returns (safe, mines): hidden cells that are certainly safe / certainly mines.  Flags on
    the board are treated as unknown hidden cells (the fly never places them; a human might).
    """
    rows, cols = visible.shape
    unknown = (visible == HIDDEN) | (visible == FLAG)
    numbers = [(r, c, int(visible[r, c])) for r in range(rows) for c in range(cols) if 0 <= visible[r, c] <= 8]
    mines: set = set()
    safe: set = set()
    changed = True
    while changed:
        changed = False
        for r, c, n in numbers:
            hidden = [(rr, cc) for rr, cc in _neighbors(r, c, rows, cols) if unknown[rr, cc]]
            if not hidden:
                continue
            known = [p for p in hidden if p in mines]
            rest = [p for p in hidden if p not in mines]
            if not rest:
                continue
            if n - len(known) == len(rest):
                mines.update(rest)
                changed = True
            elif n == len(known):
                for p in rest:
                    if p not in safe:
                        safe.add(p)
                        changed = True
    return safe, mines


def mine_probability(visible: np.ndarray, mines: set) -> dict[tuple[int, int], float]:
    """Crude per-cell risk for frontier cells: max over adjacent numbers of remaining mines /
    remaining unknown neighbours.  Only used to rank guesses."""
    rows, cols = visible.shape
    unknown = (visible == HIDDEN) | (visible == FLAG)
    prob: dict[tuple[int, int], float] = {}
    for r in range(rows):
        for c in range(cols):
            n = int(visible[r, c])
            if not 0 <= n <= 8:
                continue
            hidden = [(rr, cc) for rr, cc in _neighbors(r, c, rows, cols) if unknown[rr, cc]]
            rest = [p for p in hidden if p not in mines]
            if not rest:
                continue
            p_mine = max(0.0, (n - sum(1 for p in hidden if p in mines)) / len(rest))
            for p in rest:
                prob[p] = max(prob.get(p, 0.0), p_mine)
    return prob


def _step_toward(cursor: tuple[int, int], target: tuple[int, int]) -> str:
    dr, dc = target[0] - cursor[0], target[1] - cursor[1]
    if abs(dr) >= abs(dc) and dr != 0:
        return "up" if dr < 0 else "down"
    return "left" if dc < 0 else "right"


def _nearest(cursor: tuple[int, int], cells) -> tuple[int, int]:
    return min(cells, key=lambda p: (abs(p[0] - cursor[0]) + abs(p[1] - cursor[1]), p[0], p[1]))


class Teacher:
    def __init__(self, params: TeacherParams | None = None):
        self.p = params or TeacherParams()
        self.last_target: tuple[int, int] | None = None
        self.last_mode = ""

    def plan(self, visible: np.ndarray, cursor: tuple[int, int]) -> tuple[str, tuple[int, int] | None, str]:
        """Return (action, target cell or None, mode) for the visible grid and cursor."""
        rows, cols = visible.shape
        revealed = (visible >= 0) & (visible <= 8)
        if not revealed.any():
            target = (rows // 2, cols // 2)
            mode = "first"
        else:
            safe, mines = infer(visible)
            if safe:
                target = _nearest(cursor, safe)
                mode = "safe"
            else:
                unknown = (visible == HIDDEN) | (visible == FLAG)
                cand = [(r, c) for r in range(rows) for c in range(cols) if unknown[r, c] and (r, c) not in mines]
                if not cand:
                    return "jump", None, "none"
                frontier = set()
                for r in range(rows):
                    for c in range(cols):
                        if revealed[r, c]:
                            frontier.update(_neighbors(r, c, rows, cols))
                free = [p for p in cand if p not in frontier]
                if free:
                    target = _nearest(cursor, free)
                    mode = "guess-free"
                else:
                    prob = mine_probability(visible, mines)
                    target = min(cand, key=lambda p: (round(prob.get(p, 1.0), 6), abs(p[0] - cursor[0]) + abs(p[1] - cursor[1]), p[0], p[1]))
                    mode = "guess-frontier"
                dist = abs(target[0] - cursor[0]) + abs(target[1] - cursor[1])
                if dist > self.p.jump_distance:
                    return "jump", target, mode
        self.last_target, self.last_mode = target, mode
        if tuple(cursor) == tuple(target):
            return "reveal", target, mode
        return _step_toward(cursor, target), target, mode

    def act(self, visible: np.ndarray, cursor: tuple[int, int], rng: np.random.Generator | None = None) -> str:
        return self.plan(visible, cursor)[0]


def teacher_action(visible: np.ndarray, cursor: tuple[int, int], rng: np.random.Generator | None = None,
                   params: TeacherParams | None = None) -> str:
    """Functional form: the teacher's action for one (board, cursor)."""
    return Teacher(params).act(visible, cursor, rng)


def play_teacher(seed: int, rows: int = 9, cols: int = 9, mines: int = 10, max_turns: int = 400,
                 params: TeacherParams | None = None) -> dict:
    """The teacher alone, with the fly's cursor mechanics (the ceiling for imitation)."""
    rng = np.random.default_rng(seed)
    game = Minesweeper(rows, cols, mines, seed=seed)
    cursor = (rows // 2, cols // 2)
    teacher = Teacher(params)
    counts = {a: 0 for a in ACTIONS}
    turns = 0
    for turns in range(1, max_turns + 1):
        action = teacher.act(game.visible(), cursor, rng)
        counts[action] += 1
        r, c = cursor
        if action == "up":
            cursor = (max(0, r - 1), c)
        elif action == "down":
            cursor = (min(rows - 1, r + 1), c)
        elif action == "left":
            cursor = (r, max(0, c - 1))
        elif action == "right":
            cursor = (r, min(cols - 1, c + 1))
        elif action == "jump":
            hidden = game.hidden_cells()
            if hidden:
                cursor = hidden[int(rng.integers(len(hidden)))]
        elif action == "reveal":
            game.reveal(r, c)
        if game.over:
            break
    return {
        "seed": seed, "outcome": "won" if game.won else ("lost" if game.lost else "timeout"),
        "turns": turns, "safe_revealed": game.safe_revealed, "total_safe": game.total_safe, "actions": counts,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--jump-distance", type=int, default=5)
    args = ap.parse_args(argv)
    recs = [play_teacher(s, params=TeacherParams(jump_distance=args.jump_distance)) for s in range(args.seed0, args.seed0 + args.games)]
    won = sum(r["outcome"] == "won" for r in recs)
    safe = np.mean([r["safe_revealed"] for r in recs])
    turns = np.mean([r["turns"] for r in recs])
    tot = {a: sum(r["actions"][a] for r in recs) for a in ACTIONS}
    n_turns = sum(tot.values())
    print(f"[teacher] {args.games} games, seeds {args.seed0}..{args.seed0 + args.games - 1}: win rate {won / args.games:.2f}, "
          f"mean safe revealed {safe:.1f}/{recs[0]['total_safe']}, mean turns {turns:.1f}")
    print("[teacher] action mix: " + ", ".join(f"{a} {tot[a] / n_turns:.2f}" for a in ACTIONS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
