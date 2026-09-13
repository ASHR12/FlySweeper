"""Load the compiled MaleCNS graph and look up cell populations by name."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as ft

from .paths import COMPILED


@dataclass
class Graph:
    indptr: np.ndarray   # int64, len n+1, out-edges by presynaptic neuron
    indices: np.ndarray  # int32, postsynaptic index
    data: np.ndarray     # float32, signed (normalized) weight
    counts: np.ndarray   # int32, raw synapse count

    @property
    def n(self) -> int:
        return len(self.indptr) - 1

    @property
    def n_edges(self) -> int:
        return len(self.indices)


class Brain:
    """Neuron table + out-edge graph + eye atlas."""

    def __init__(self, compiled: Path = COMPILED):
        self.dir = Path(compiled)
        if not (self.dir / "graph.npz").exists():
            raise FileNotFoundError(f"compiled graph missing in {self.dir}; run `python -m flysweeper.compile_graph`")
        self.neurons: pd.DataFrame = ft.read_feather(self.dir / "neurons.feather")
        g = np.load(self.dir / "graph.npz")
        self.graph = Graph(g["indptr"], g["indices"], g["data"], g["counts"])
        self.eye = dict(np.load(self.dir / "eye.npz"))
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self._types = self.neurons["type"].fillna("").to_numpy().astype(str)
        self._sides = self.neurons["side"].to_numpy().astype(str)
        self._superclass = self.neurons["superclass"].fillna("").to_numpy().astype(str)
        self._class = self.neurons["class"].fillna("").to_numpy().astype(str)

    @property
    def n(self) -> int:
        return self.graph.n

    def cells(
        self,
        types: list[str] | str | None = None,
        side: str | None = None,
        superclass: str | None = None,
        cls: str | None = None,
        type_prefix: str | None = None,
    ) -> np.ndarray:
        """Indices of neurons matching all given filters (types match exactly)."""
        m = np.ones(self.n, dtype=bool)
        if types is not None:
            if isinstance(types, str):
                types = [types]
            m &= np.isin(self._types, np.asarray(types, dtype=str))
        if type_prefix is not None:
            m &= np.char.startswith(self._types, type_prefix)
        if side is not None:
            m &= self._sides == side
        if superclass is not None:
            m &= self._superclass == superclass
        if cls is not None:
            m &= self._class == cls
        return np.flatnonzero(m).astype(np.int32)

    def describe(self, idx: np.ndarray, limit: int = 8) -> str:
        sub = self.neurons.iloc[idx]
        parts = [f"{t}/{s}" for t, s in zip(sub["type"].head(limit), sub["side"].head(limit))]
        more = "" if len(idx) <= limit else f" ... (+{len(idx) - limit})"
        return f"{len(idx)} cells: " + ", ".join(parts) + more
