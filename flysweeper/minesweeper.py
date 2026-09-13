"""Plain Minesweeper. Nothing here knows about flies."""
from __future__ import annotations

from collections import deque

import numpy as np

HIDDEN = -1
FLAG = 9
MINE = 10  # shown only after the game is lost

NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class Minesweeper:
    def __init__(self, rows: int = 9, cols: int = 9, mines: int = 10, seed: int = 0):
        if mines >= rows * cols - 9:
            raise ValueError("too many mines for a safe first click")
        self.rows, self.cols, self.n_mines = rows, cols, mines
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.mines = np.zeros((rows, cols), dtype=bool)
        self.numbers = np.zeros((rows, cols), dtype=np.int8)
        self.revealed = np.zeros((rows, cols), dtype=bool)
        self.flagged = np.zeros((rows, cols), dtype=bool)
        self.placed = False
        self.lost = False
        self.moves = 0
        self.mine_hit: tuple[int, int] | None = None

    # ---- state -------------------------------------------------------------------------
    @property
    def total_safe(self) -> int:
        return self.rows * self.cols - self.n_mines

    @property
    def safe_revealed(self) -> int:
        return int(self.revealed.sum())

    @property
    def won(self) -> bool:
        return not self.lost and self.placed and self.safe_revealed == self.total_safe

    @property
    def over(self) -> bool:
        return self.lost or self.won

    @property
    def cleared_fraction(self) -> float:
        return self.safe_revealed / self.total_safe

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def neighbors(self, r: int, c: int):
        for dr, dc in NEIGHBORS:
            rr, cc = r + dr, c + dc
            if self.in_bounds(rr, cc):
                yield rr, cc

    def visible(self) -> np.ndarray:
        """int8 grid: -1 hidden, 0-8 revealed number, 9 flag, 10 exploded/revealed mine."""
        v = np.full((self.rows, self.cols), HIDDEN, dtype=np.int8)
        v[self.revealed] = self.numbers[self.revealed]
        v[self.flagged & ~self.revealed] = FLAG
        if self.lost:
            v[self.mines] = MINE
        return v

    # ---- moves -------------------------------------------------------------------------
    def _place_mines(self, r0: int, c0: int) -> None:
        forbidden = {(r0, c0), *self.neighbors(r0, c0)}
        candidates = [(r, c) for r in range(self.rows) for c in range(self.cols) if (r, c) not in forbidden]
        pick = self.rng.choice(len(candidates), size=self.n_mines, replace=False)
        for k in pick:
            self.mines[candidates[k]] = True
        for r in range(self.rows):
            for c in range(self.cols):
                self.numbers[r, c] = sum(self.mines[rr, cc] for rr, cc in self.neighbors(r, c))
        self.placed = True

    def reveal(self, r: int, c: int) -> tuple[str, int]:
        """Returns (outcome, newly_revealed). outcome in {'mine', 'safe', 'noop'}."""
        if self.over or not self.in_bounds(r, c) or self.revealed[r, c] or self.flagged[r, c]:
            return "noop", 0
        self.moves += 1
        if not self.placed:
            self._place_mines(r, c)
        if self.mines[r, c]:
            self.lost = True
            self.mine_hit = (r, c)
            return "mine", 0
        before = self.safe_revealed
        q = deque([(r, c)])
        self.revealed[r, c] = True
        while q:
            rr, cc = q.popleft()
            if self.numbers[rr, cc] == 0:
                for nr, nc in self.neighbors(rr, cc):
                    if not self.revealed[nr, nc] and not self.mines[nr, nc]:
                        self.revealed[nr, nc] = True
                        self.flagged[nr, nc] = False
                        q.append((nr, nc))
        return "safe", self.safe_revealed - before

    def toggle_flag(self, r: int, c: int) -> bool:
        if self.over or not self.in_bounds(r, c) or self.revealed[r, c]:
            return False
        self.moves += 1
        self.flagged[r, c] = not self.flagged[r, c]
        return True

    def hidden_cells(self) -> list[tuple[int, int]]:
        return [(r, c) for r in range(self.rows) for c in range(self.cols) if not self.revealed[r, c] and not self.flagged[r, c]]


def solver_move(game: Minesweeper, rng: np.random.Generator) -> tuple[str, int, int]:
    """Single-point Minesweeper logic; guesses when stuck. Returns (kind, r, c)."""
    if not game.placed:
        return "reveal", game.rows // 2, game.cols // 2
    vis = game.visible()
    safe, mines = set(), set()
    for r in range(game.rows):
        for c in range(game.cols):
            n = vis[r, c]
            if n < 0 or n > 8:
                continue
            hidden = [(rr, cc) for rr, cc in game.neighbors(r, c) if not game.revealed[rr, cc]]
            flags = [p for p in hidden if game.flagged[p]]
            unknown = [p for p in hidden if not game.flagged[p]]
            if not unknown:
                continue
            if n - len(flags) == len(unknown):
                mines.update(unknown)
            elif n == len(flags):
                safe.update(unknown)
    if safe:
        r, c = sorted(safe)[rng.integers(len(safe))]
        return "reveal", r, c
    if mines:
        r, c = sorted(mines)[rng.integers(len(mines))]
        return "flag", r, c
    hidden = game.hidden_cells()
    if not hidden:
        return "noop", 0, 0
    # Prefer cells not adjacent to any number (no constraint information).
    frontier = {p for r in range(game.rows) for c in range(game.cols) if 0 <= vis[r, c] <= 8 for p in game.neighbors(r, c)}
    unconstrained = [p for p in hidden if p not in frontier]
    pool = unconstrained or hidden
    r, c = pool[rng.integers(len(pool))]
    return "reveal", r, c
