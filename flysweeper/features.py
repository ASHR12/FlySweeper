"""Brain activity -> a feature vector for one decision window.

A feature is the spike count of one neuron during the current turn window (15 steps = 300 ms
by default), optionally followed by the same counts for the previous window, as float32.  The
population is a named list of groups resolved through `Brain.cells`, so any superclass / class /
type-prefix / exact-type set can be used:

    default   descending neurons (superclass descending_neuron, 1,314)
            + visual projection neurons (superclass visual_projection, 9,201)
            + central complex (class CX, 2,950)
            + mushroom body output neurons (class MBON, 97)                       = 13,562 cells

    GROUPS below also define T4/T5 (optic-lobe motion detectors), LC/LPLC (lobula columnar,
    already inside visual_projection), Kenyon cells and ascending neurons for experiments.

Nothing here is anatomy beyond the cell-type labels; which cells a readout is allowed to see is
an engineering choice and is recorded in the saved readout.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .brain import Brain

# name -> Brain.cells(**spec)
GROUPS: dict[str, dict] = {
    "dn": {"superclass": "descending_neuron"},
    "vp": {"superclass": "visual_projection"},
    "cx": {"cls": "CX"},
    "mbon": {"cls": "MBON"},
    "an": {"superclass": "ascending_neuron"},
    "kc": {"type_prefix": "KC"},
    "t4": {"type_prefix": "T4"},
    "t5": {"type_prefix": "T5"},
    "lc": {"type_prefix": "LC"},
    "lplc": {"type_prefix": "LPLC"},
    "lamina": {"types": ["L1", "L2", "L3", "L4", "L5"]},
    "medulla_columnar": {"types": ["Mi1", "Mi4", "Mi9", "Tm1", "Tm2", "Tm3", "Tm4", "Tm9", "C2", "C3"]},
}

DEFAULT_GROUPS = ("dn", "vp", "cx", "mbon")


@dataclass
class FeatureSpec:
    groups: tuple = DEFAULT_GROUPS
    windows: int = 1                     # 1 = current turn only, 2 = current + previous turn
    extra: dict = field(default_factory=dict)   # name -> Brain.cells kwargs, merged into GROUPS

    def resolve(self, brain: Brain) -> tuple[np.ndarray, np.ndarray]:
        """(sorted unique neuron indices, group id per index using the first matching group)."""
        table = dict(GROUPS)
        table.update(self.extra)
        idx_all, gid_all = [], []
        for k, name in enumerate(self.groups):
            if name not in table:
                raise KeyError(f"unknown feature group {name!r}; known: {sorted(table)}")
            idx = brain.cells(**table[name])
            idx_all.append(idx)
            gid_all.append(np.full(len(idx), k, dtype=np.int16))
        idx = np.concatenate(idx_all) if idx_all else np.zeros(0, dtype=np.int32)
        gid = np.concatenate(gid_all) if gid_all else np.zeros(0, dtype=np.int16)
        uniq, first = np.unique(idx, return_index=True)
        return uniq.astype(np.int32), gid[first]


class FeatureExtractor:
    """Accumulates spike counts of the chosen population per turn window."""

    def __init__(self, brain: Brain, spec: FeatureSpec | None = None, index: np.ndarray | None = None):
        self.spec = spec or FeatureSpec()
        if index is not None:
            self.index = np.asarray(index, dtype=np.int32)
            self.group_id = np.zeros(len(self.index), dtype=np.int16)
        else:
            self.index, self.group_id = self.spec.resolve(brain)
        self.m = len(self.index)
        self.pos = np.full(brain.n, -1, dtype=np.int32)
        self.pos[self.index] = np.arange(self.m, dtype=np.int32)
        self.cur = np.zeros(self.m, dtype=np.int32)
        self.prev = np.zeros(self.m, dtype=np.int32)
        self.windows_seen = 0

    @property
    def dim(self) -> int:
        return self.m * self.spec.windows

    def reset(self) -> None:
        self.cur[:] = 0
        self.prev[:] = 0
        self.windows_seen = 0

    def new_window(self) -> None:
        """Close the current window (it becomes 'previous') and start counting a new one."""
        self.prev, self.cur = self.cur, self.prev
        self.cur[:] = 0
        self.windows_seen += 1

    def observe(self, fired: np.ndarray) -> None:
        p = self.pos[fired]
        p = p[p >= 0]
        if len(p):
            self.cur += np.bincount(p, minlength=self.m).astype(np.int32)

    def counts(self) -> np.ndarray:
        """Current-window counts as uint8 (a 15-step window cannot exceed 15 spikes)."""
        return np.minimum(self.cur, 255).astype(np.uint8)

    def vector(self) -> np.ndarray:
        if self.spec.windows == 2:
            return np.concatenate([self.cur, self.prev]).astype(np.float32)
        return self.cur.astype(np.float32)

    def describe(self, brain: Brain) -> dict:
        out = {"n_cells": int(self.m), "dim": int(self.dim), "windows": int(self.spec.windows), "groups": {}}
        for k, name in enumerate(self.spec.groups):
            out["groups"][name] = int((self.group_id == k).sum())
        return out
