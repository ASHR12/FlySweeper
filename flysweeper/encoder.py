"""Board -> visual-neuron drive.  Everything in this file is invented.

The Minesweeper board is painted onto the fly's two eyes as one wide screen: the left eye sees
the left half of the board, the right eye the right half (the middle column is seen by both).
Every optic-lobe column (the MaleCNS `assignedOlHex1/2` hex grid, ~800 columns per eye) is
assigned to one board cell by an equal-count partition: columns are ranked dorso-ventrally into
`rows` bands and, within each band, antero-posteriorly into board columns.  Which hex axis is
"up" is an engineered registration, not measured optics.

Input routes (which cells of a column receive the drive):

    retina   R1-R6 photoreceptors get luminance + temporal contrast, R7 sees flags, R8 sees
             the cursor.  This is the anatomical route.  Real photoreceptors release histamine
             onto *graded* lamina neurons; in a whole-graph spiking model the image dies at
             that first relay (measured with `python -m flysweeper.probe`).
    lamina   the same drive is injected into the L1/L2/L3 lamina monopolar cells of the column,
             as if the lamina spiked with the image.  This is the fly.ai / Minecraft-style bypass
             of the graded stage.  R7 (flags) and R8 (cursor) still get their channels.

The drive itself is Fly64's retinal formula:
    drive = clip(0.45 * luminance + 1.6 * |luminance change| + 0.25 * colour, 0, 1) * retina_gain
The cursor flickers at 25 Hz so the cells under it get a strong temporal-contrast signal.

Looming channel (hand-built reflex, on by default, `danger_loom`): the largest revealed number on
or next to the cursor is painted as a looming object onto the LC4 / LPLC2 looming detectors of the
eye the cursor is in.  What happens next is the connectome's business: in this graph, LC4/LPLC2
drive the giant fiber DNp01 (the escape take-off pathway, one of the few that every whole-graph
demo found working from the wiring alone), and the decoder reads DNp01 as "jump".  The stimulus
is invented; the escape is anatomy.

Egocentric view (`egocentric=True`, off by default; used by the trained-readout experiments):
instead of a fixed camera over the whole board, the fly "stands on the cursor": the eyes see a
(2*ego_radius+1)^2 window of the board centred on the cursor, as a walking fly sees the ground
around itself.  Cells outside the board are painted at `lum_wall`.  The cursor flicker is off in
this mode (the cursor is always at the centre of the visual field).  Same photoreceptors, same
lamina route; only the registration of board cells to optic columns moves with the cursor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .brain import Brain
from .minesweeper import FLAG, HIDDEN, MINE

LUM = 0
FLAG_CH = 1
CURSOR_CH = 2
WALL = -2   # egocentric only: outside the board


@dataclass
class EncoderParams:
    route: str = "lamina"          # retina | lamina
    retina_gain: float = 0.62      # Fly64: voltage increment per unit drive
    lum_weight: float = 0.45       # Fly64 retinal drive weights
    temporal_weight: float = 1.6
    color_weight: float = 0.25
    cursor_flicker: float = 0.35   # luminance added on even steps under the cursor
    lum_hidden: float = 0.10
    lum_number_base: float = 0.35  # revealed n -> base + step * n
    lum_number_step: float = 0.075
    lum_flag: float = 0.55
    lum_mine: float = 1.00
    binocular: bool = False        # True: both eyes see the whole board
    lamina_types: tuple = ("L1", "L2", "L3")
    danger_loom: bool = True       # adjacent numbers -> LC4/LPLC2 looming detectors
    loom_gain: float = 0.5         # voltage kick per pulse at danger 5+; amp = gain * (danger-1)/4
    loom_threshold: int = 2        # numbers below this are not scary
    loom_types: tuple = ("LC4", "LPLC2")
    egocentric: bool = False       # eyes centred on the cursor (see module docstring)
    ego_radius: int = 4            # egocentric window half-width in board cells
    lum_wall: float = 0.0          # egocentric: luminance of cells outside the board


def _column_cells(brain: Brain, rows: int, cols: int, binocular: bool) -> dict[tuple[int, int, int], int]:
    """Map every optic column (eye, hex1, hex2) to a board cell index by equal-count ranking."""
    nm = brain.neurons
    hx = nm["hex1"].to_numpy(dtype=np.float64)
    hy = nm["hex2"].to_numpy(dtype=np.float64)
    side = nm["side"].to_numpy().astype(str)
    has = ~np.isnan(hx)
    cols_seen = set()
    for e_, s in ((0, "L"), (1, "R")):
        m = has & (side == s)
        for a, b in zip(hx[m].astype(int), hy[m].astype(int)):
            cols_seen.add((e_, a, b))
    eye = brain.eye
    for e_, a, b in zip(eye["eye"], eye["hex1"], eye["hex2"]):
        cols_seen.add((int(e_), int(a), int(b)))

    mapping: dict[tuple[int, int, int], int] = {}
    for e_ in (0, 1):
        cs = sorted(c for c in cols_seen if c[0] == e_)
        if not cs:
            continue
        x = np.array([c[1] + 0.5 * c[2] for c in cs])
        y = np.array([c[2] * np.sqrt(3) / 2 for c in cs])
        order_y = np.argsort(y, kind="stable")
        band = np.empty(len(cs), dtype=np.int32)
        band[order_y] = (np.arange(len(cs)) * rows) // len(cs)
        for b in range(rows):
            members = np.flatnonzero(band == b)
            if len(members) == 0:
                continue
            ox = members[np.argsort(x[members], kind="stable")]
            rank = np.arange(len(ox)) / len(ox)
            if binocular:
                col = np.floor(rank * cols).astype(int)
            else:
                half = cols / 2.0
                col = np.floor(rank * half + (0 if e_ == 0 else half)).astype(int)
            col = np.clip(col, 0, cols - 1)
            r = rows - 1 - b  # band 0 (lowest y) is the bottom row
            for k, ci in zip(ox, col):
                mapping[cs[k]] = int(r * cols + ci)
    return mapping


class RetinaEncoder:
    def __init__(self, brain: Brain, rows: int, cols: int, params: EncoderParams | None = None):
        self.p = params or EncoderParams()
        if self.p.route not in ("retina", "lamina"):
            raise ValueError(self.p.route)
        self.rows, self.cols = rows, cols
        if self.p.egocentric:
            self.grid_rows = self.grid_cols = 2 * self.p.ego_radius + 1
        else:
            self.grid_rows, self.grid_cols = rows, cols
        col2cell = _column_cells(brain, self.grid_rows, self.grid_cols, self.p.binocular)

        idx, chan, cell = [], [], []
        eye = brain.eye
        for i, k, e_, a, b in zip(eye["pr_idx"], eye["kind"], eye["eye"], eye["hex1"], eye["hex2"]):
            key = (int(e_), int(a), int(b))
            if key not in col2cell:
                continue
            if k == 0:
                if self.p.route != "retina":
                    continue
                ch = LUM
            elif k == 1:
                ch = FLAG_CH
            else:
                ch = CURSOR_CH
            idx.append(int(i)); chan.append(ch); cell.append(col2cell[key])
        if self.p.route == "lamina":
            nm = brain.neurons
            lam = brain.cells(types=list(self.p.lamina_types))
            hx = nm["hex1"].to_numpy(); hy = nm["hex2"].to_numpy(); side = nm["side"].to_numpy().astype(str)
            for i in lam:
                if np.isnan(hx[i]):
                    continue
                key = (0 if side[i] == "L" else 1, int(hx[i]), int(hy[i]))
                if key in col2cell:
                    idx.append(int(i)); chan.append(LUM); cell.append(col2cell[key])
        self.idx = np.array(idx, dtype=np.int32)
        self.chan = np.array(chan, dtype=np.int8)
        self.cell = np.array(cell, dtype=np.int32)
        self.lum_cells = self.chan == LUM
        self.cells_per_board_cell = np.bincount(self.cell[self.lum_cells], minlength=self.grid_rows * self.grid_cols)
        self.prev_lum = np.zeros(len(self.idx), dtype=np.float32)
        self.lum_table = self._lum_table()
        self.loom_idx = {
            "L": brain.cells(types=list(self.p.loom_types), side="L"),
            "R": brain.cells(types=list(self.p.loom_types), side="R"),
        }
        self.last_danger = 0

    def danger_at(self, visible: np.ndarray, cursor: tuple[int, int]) -> int:
        """Largest revealed number on or around the cursor (0-8)."""
        r, c = cursor
        r0, r1 = max(0, r - 1), min(self.rows, r + 2)
        c0, c1 = max(0, c - 1), min(self.cols, c + 2)
        patch = visible[r0:r1, c0:c1]
        nums = patch[(patch >= 1) & (patch <= 8)]
        return int(nums.max()) if len(nums) else 0

    def loom_drive(self, visible: np.ndarray, cursor: tuple[int, int], step: int) -> tuple[np.ndarray, float]:
        """Looming detectors of the cursor's eye, pulsed at 25 Hz with amplitude ~ danger."""
        self.last_danger = self.danger_at(visible, cursor)
        if not self.p.danger_loom or self.last_danger < self.p.loom_threshold or step % 2:
            return np.zeros(0, dtype=np.int32), 0.0
        side = "L" if cursor[1] < self.cols / 2 else "R"
        amp = self.p.loom_gain * min(1.0, (self.last_danger - 1) / 4.0)
        return self.loom_idx[side], amp

    def _lum_table(self) -> np.ndarray:
        """Luminance for visible-state codes -2 (wall) .. 10 (index = code + 2)."""
        t = np.zeros(13, dtype=np.float32)
        t[WALL + 2] = self.p.lum_wall
        t[HIDDEN + 2] = self.p.lum_hidden
        for n in range(9):
            t[n + 2] = self.p.lum_number_base + self.p.lum_number_step * n
        t[FLAG + 2] = self.p.lum_flag
        t[MINE + 2] = self.p.lum_mine
        return t

    def reset(self) -> None:
        self.prev_lum[:] = 0

    def grid(self, visible: np.ndarray, cursor: tuple[int, int]) -> tuple[np.ndarray, int]:
        """What the eyes are pointed at: (flattened cell codes, index of the cursor cell)."""
        cr, cc = cursor
        if not self.p.egocentric:
            return visible.ravel().astype(np.int32), cr * self.cols + cc
        R = self.p.ego_radius
        g = np.full((self.grid_rows, self.grid_cols), WALL, dtype=np.int32)
        r0, r1 = max(0, cr - R), min(self.rows, cr + R + 1)
        c0, c1 = max(0, cc - R), min(self.cols, cc + R + 1)
        g[r0 - cr + R:r1 - cr + R, c0 - cc + R:c1 - cc + R] = visible[r0:r1, c0:c1]
        return g.ravel(), R * self.grid_cols + R

    def drive(self, visible: np.ndarray, cursor: tuple[int, int], step: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (target neuron indices, voltage increments) for one 20 ms step."""
        codes, cur = self.grid(visible, cursor)
        cell_lum = self.lum_table[codes + 2]
        if step % 2 == 0 and not self.p.egocentric:
            cell_lum = cell_lum.copy()
            cell_lum[cur] = min(1.0, cell_lum[cur] + self.p.cursor_flicker)
        lum = cell_lum[self.cell]
        temporal = np.abs(lum - self.prev_lum)
        self.prev_lum = lum
        flag_ch = (codes[self.cell] == FLAG).astype(np.float32)
        cursor_ch = (self.cell == cur).astype(np.float32)
        color = np.where(self.chan == FLAG_CH, flag_ch, np.where(self.chan == CURSOR_CH, cursor_ch, 0.0)).astype(np.float32)
        lum_part = np.where(self.lum_cells, self.p.lum_weight * lum + self.p.temporal_weight * temporal, 0.0)
        d = lum_part + self.p.color_weight * color
        d = np.clip(d, 0.0, 1.0) * np.float32(self.p.retina_gain)
        return self.idx, d.astype(np.float32)

    def coverage_summary(self) -> dict:
        cpc = self.cells_per_board_cell
        return {
            "route": self.p.route,
            "egocentric": bool(self.p.egocentric),
            "grid": [int(self.grid_rows), int(self.grid_cols)],
            "target_cells": int(len(self.idx)),
            "luminance_cells": int(self.lum_cells.sum()),
            "flag_cells(R7)": int((self.chan == FLAG_CH).sum()),
            "cursor_cells(R8)": int((self.chan == CURSOR_CH).sum()),
            "board_cells": int(self.rows * self.cols),
            "lum_cells_per_board_cell_min": int(cpc.min()),
            "lum_cells_per_board_cell_median": float(np.median(cpc)),
            "lum_cells_per_board_cell_max": int(cpc.max()),
            "empty_board_cells": int((cpc == 0).sum()),
        }
