"""Descending-neuron spike counts -> a Minesweeper action.  The mapping is engineered.

Pools (all real MaleCNS cell types, the same readout sets Xenova's Neural Canvas and Fly64 use;
none of them "means" a Minesweeper move in a fly):

    up      DNp09 + DNg100 + DNg97   forward-walking descending neurons
    down    MDN                      moonwalker descending neurons, backward walking
    left    DNa02 + DNa11 + DNg13    steering, left side
    right   DNa02 + DNa11 + DNg13    steering, right side
    reveal  DNpe017 + DNp10          DOOMFLY's "fire" cell and Fly64's jump pool
    jump    DNp01                    the giant fiber: escape take-off.  The cursor jumps to a
                                     random hidden cell.  Driven by the looming channel (encoder).
    flag    DNb02                    only when flags are enabled

Decision each turn: per-cell spike rate of every pool over the turn window, minus a baseline
(that pool's idle rate on a blank screen, or a running mean of its own recent rate so a pool that
is driven by overall board brightness does not win every turn), then argmax with seeded random
tie-breaking (or a softmax with temperature).  If no pool rises above baseline the fly holds.

ReadoutDecoder (mode="readout"): the trained alternative.  A linear readout fitted by
`flysweeper.train_readout` (imitation of the solver-derived teacher in teacher.py) scores the six
actions from the spike counts of a configured population (features.py) over the turn window and
takes the argmax (or samples a softmax with `readout_temperature` > 0).  It still counts the
pools above so the spectator UI keeps showing them.  The connectome is not touched: what was
trained is the readout, i.e. which brain cells count as which button.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .brain import Brain
from .features import FeatureExtractor, FeatureSpec
from .paths import COMPILED

READOUT_PATH = COMPILED / "readout.npz"

DEFAULT_POOLS = {
    "up": {"types": ["DNp09", "DNg100", "DNg97"]},
    "down": {"types": ["MDN"]},
    "left": {"types": ["DNa02", "DNa11", "DNg13"], "side": "L"},
    "right": {"types": ["DNa02", "DNa11", "DNg13"], "side": "R"},
    "reveal": {"types": ["DNpe017", "DNp10"]},
    "jump": {"types": ["DNp01"]},
    "flag": {"types": ["DNb02"]},
}


@dataclass
class DecoderParams:
    mode: str = "argmax"            # argmax | softmax | readout
    temperature: float = 0.5
    flags: bool = False
    baseline: str = "running"       # none | blank | running
    running_alpha: float = 0.1
    pools: dict = field(default_factory=lambda: dict(DEFAULT_POOLS))
    readout_path: str | None = None          # default data/compiled/readout.npz
    readout_temperature: float = 0.0         # 0 = argmax over readout scores, >0 = softmax sampling


class PoolDecoder:
    def __init__(self, brain: Brain, params: DecoderParams | None = None):
        self.p = params or DecoderParams()
        self.actions = ["up", "down", "left", "right", "reveal", "jump"] + (["flag"] if self.p.flags else [])
        self.pool_idx: dict[str, np.ndarray] = {}
        self.pool_id = np.full(brain.n, -1, dtype=np.int16)
        for k, name in enumerate(self.actions):
            spec = self.p.pools[name]
            idx = brain.cells(types=spec["types"], side=spec.get("side"))
            if len(idx) == 0:
                raise ValueError(f"pool {name!r} matched no cells: {spec}")
            if (self.pool_id[idx] >= 0).any():
                raise ValueError(f"pool {name!r} overlaps another pool")
            self.pool_id[idx] = k
            self.pool_idx[name] = idx
        self.sizes = np.array([len(self.pool_idx[a]) for a in self.actions], dtype=np.float32)
        self.counts = np.zeros(len(self.actions), dtype=np.int64)
        self.idle_rates = np.zeros(len(self.actions), dtype=np.float32)   # Hz per cell, set by set_idle
        self.running = np.zeros(len(self.actions), dtype=np.float32)      # EMA of recent per-cell rates
        self.last_scores = np.zeros(len(self.actions), dtype=np.float32)

    def reset_window(self) -> None:
        self.counts[:] = 0

    def observe(self, fired: np.ndarray) -> None:
        ids = self.pool_id[fired]
        ids = ids[ids >= 0]
        if len(ids):
            self.counts += np.bincount(ids, minlength=len(self.actions))

    def rates(self, steps: int, dt: float) -> np.ndarray:
        """Spikes per cell per second for each pool over the window."""
        return (self.counts / self.sizes / max(steps * dt, 1e-9)).astype(np.float32)

    def set_idle(self, idle_rates: np.ndarray) -> None:
        self.idle_rates = np.asarray(idle_rates, dtype=np.float32)
        self.running = self.idle_rates.copy()

    def scores(self, steps: int, dt: float) -> np.ndarray:
        r = self.rates(steps, dt)
        if self.p.baseline == "blank":
            s = np.maximum(r - self.idle_rates, 0.0)
        elif self.p.baseline == "running":
            s = np.maximum(r - self.running, 0.0)
            self.running = (1 - self.p.running_alpha) * self.running + self.p.running_alpha * r
        else:
            s = r
        self.last_scores = s.astype(np.float32)
        return s

    def decide(self, rng: np.random.Generator, steps: int, dt: float) -> str:
        s = self.scores(steps, dt)
        if s.sum() <= 0:
            return "hold"
        if self.p.mode == "softmax":
            z = s / max(s.max(), 1e-9) / max(self.p.temperature, 1e-6)
            pz = np.exp(z - z.max())
            pz /= pz.sum()
            return self.actions[int(rng.choice(len(self.actions), p=pz))]
        best = np.flatnonzero(s == s.max())
        return self.actions[int(rng.choice(best))]

    def describe(self, brain: Brain) -> dict:
        return {a: brain.describe(self.pool_idx[a]) for a in self.actions}


class ReadoutDecoder(PoolDecoder):
    """Trained linear readout over a population's turn-window spike counts (see train_readout.py).

    File format (readout.npz): W (n_actions, dim) float32, b (n_actions,), mean/std (dim,),
    index (neuron indices, int32), windows (1 or 2), actions (str), meta (JSON string).
    """

    def __init__(self, brain: Brain, params: DecoderParams | None = None, path: str | Path | None = None):
        params = params or DecoderParams()
        if params.flags:
            raise ValueError("the readout is trained on the six flag-less actions")
        super().__init__(brain, params)
        self.path = Path(path or params.readout_path or READOUT_PATH)
        if not self.path.exists():
            raise FileNotFoundError(
                f"no trained readout at {self.path}; run `python -m flysweeper.train_readout` first")
        z = np.load(self.path, allow_pickle=False)
        actions = [str(a) for a in z["actions"]]
        if actions != self.actions:
            raise ValueError(f"readout actions {actions} != decoder actions {self.actions}")
        self.W = z["W"].astype(np.float32)
        self.b = z["b"].astype(np.float32)
        self.mean = z["mean"].astype(np.float32)
        self.std = z["std"].astype(np.float32)
        self.std[self.std <= 0] = 1.0
        windows = int(z["windows"])
        self.features = FeatureExtractor(brain, FeatureSpec(groups=(), windows=windows), index=z["index"])
        if self.W.shape != (len(self.actions), self.features.dim):
            raise ValueError(f"readout W {self.W.shape} does not match feature dim {self.features.dim}")
        self.readout_meta = json.loads(str(z["meta"])) if "meta" in z else {}
        self.last_logits = np.zeros(len(self.actions), dtype=np.float32)
        self.last_pool_scores = np.zeros(len(self.actions), dtype=np.float32)

    def reset_window(self) -> None:
        super().reset_window()
        self.features.new_window()

    def observe(self, fired: np.ndarray) -> None:
        super().observe(fired)
        self.features.observe(fired)

    def logits(self) -> np.ndarray:
        x = (self.features.vector() - self.mean) / self.std
        return (self.W @ x + self.b).astype(np.float32)

    def decide(self, rng: np.random.Generator, steps: int, dt: float) -> str:
        self.last_pool_scores = super().scores(steps, dt)     # keeps the pool baseline / UI numbers alive
        z = self.logits()
        self.last_logits = z
        pz = np.exp(z - z.max())
        pz /= pz.sum()
        self.last_scores = pz.astype(np.float32)
        t = self.p.readout_temperature
        if t > 0:
            q = np.exp((z - z.max()) / t)
            q /= q.sum()
            return self.actions[int(rng.choice(len(self.actions), p=q))]
        best = np.flatnonzero(z == z.max())
        return self.actions[int(rng.choice(best))]

    def describe(self, brain: Brain) -> dict:
        d = super().describe(brain)
        d["readout"] = {"path": str(self.path), "dim": int(self.features.dim), "cells": int(self.features.m),
                        "windows": int(self.features.spec.windows), **{k: self.readout_meta[k] for k in ("groups", "classifier", "held_out_accuracy", "train_seeds", "n_turns") if k in self.readout_meta}}
        return d
