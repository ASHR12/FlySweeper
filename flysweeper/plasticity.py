"""Optional dopamine-gated plasticity on Kenyon cell -> MBON synapses.  OFF by default.

This is the one place the connectome is allowed to change, and only at the synapses where real
flies store associative memories (mushroom body output).  Rule (an eligibility-trace version of
the community "reward-modulated Hebbian" cartoon; not a fitted fly learning rule):

    each step:  post_trace <- post_trace * exp(-dt/0.2) + MBON spikes
                for every KC that spiked:  e[KC->MBON edges] += post_trace[MBON]
                e <- e * exp(-dt/1.0)
    on dopamine (reward +1 via PAM cells, punishment -1 via PPL1 cells):
                w <- clip(w + eta * sign * e * |w0|, 0.1*w0, 3*w0)   (KCs are cholinergic, w0 > 0)

Every dopamine event is also delivered as a real stimulus to the PAM / PPL1 cells, and every
weight change is logged, so "the fly learned" claims can be checked against the log and against
the frozen baseline in validate.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .brain import Brain
from .sim import LIF


@dataclass
class PlasticityParams:
    eta: float = 0.05
    post_tau: float = 0.2
    elig_tau: float = 1.0
    w_min_ratio: float = 0.1
    w_max_ratio: float = 3.0
    dopamine_amount: float = 0.8   # voltage kick into PAM / PPL1 cells per step
    dopamine_steps: int = 2


class KCMBONPlasticity:
    def __init__(self, brain: Brain, sim: LIF, params: PlasticityParams | None = None):
        self.p = params or PlasticityParams()
        self.sim = sim
        g = brain.graph
        kc = brain.cells(type_prefix="KC")
        mbon = np.zeros(brain.n, dtype=bool)
        mbon[brain.cells(cls="MBON")] = True
        self.is_kc = np.zeros(brain.n, dtype=bool)
        self.is_kc[kc] = True
        # positions of KC->MBON edges in the out-edge CSR, grouped per KC
        starts, ends = g.indptr[kc], g.indptr[kc + 1]
        edge_pos = []
        for s, e in zip(starts, ends):
            seg = np.arange(s, e)
            edge_pos.append(seg[mbon[g.indices[s:e]]])
        counts = np.array([len(x) for x in edge_pos])
        self.kc_indptr = np.zeros(len(kc) + 1, dtype=np.int64)
        self.kc_indptr[1:] = np.cumsum(counts)
        self.edge_pos = np.concatenate(edge_pos).astype(np.int64) if len(edge_pos) else np.zeros(0, dtype=np.int64)
        self.kc_row = np.full(brain.n, -1, dtype=np.int64)
        self.kc_row[kc] = np.arange(len(kc))
        self.w0 = np.abs(sim.data[self.edge_pos]).astype(np.float32)
        self.w_frozen = sim.data[self.edge_pos].copy()   # signed originals, restored by detach()
        self.w_learned = self.w_frozen.copy()            # the learned weights while detached
        self.attached = True
        self.e = np.zeros(len(self.edge_pos), dtype=np.float32)
        self.post_trace = np.zeros(brain.n, dtype=np.float32)
        self.post_decay = np.float32(np.exp(-sim.p.dt / self.p.post_tau))
        self.elig_decay = np.float32(np.exp(-sim.p.dt / self.p.elig_tau))
        self.pam = brain.cells(type_prefix="PAM")
        self.ppl1 = brain.cells(type_prefix="PPL1")
        self.pending: list[tuple[int, float]] = []   # (steps left, sign)
        self.log: list[dict] = []
        self.total_abs_delta = 0.0
        self.n_edges = len(self.edge_pos)

    def detach(self) -> None:
        """Put the frozen KC->MBON weights back into the graph (learned ones are kept aside)."""
        if self.attached:
            self.w_learned = self.sim.data[self.edge_pos].copy()
            self.sim.data[self.edge_pos] = self.w_frozen
            self.attached = False

    def attach(self) -> None:
        """Install the learned KC->MBON weights (continues learning where it left off)."""
        if not self.attached:
            self.sim.data[self.edge_pos] = self.w_learned
            self.attached = True

    def observe(self, fired: np.ndarray) -> None:
        self.post_trace *= self.post_decay
        self.post_trace[fired] += 1.0
        self.e *= self.elig_decay
        kcs = fired[self.is_kc[fired]]
        if len(kcs):
            rows = self.kc_row[kcs]
            for r in rows:
                s, t = self.kc_indptr[r], self.kc_indptr[r + 1]
                if t > s:
                    seg = self.edge_pos[s:t]
                    self.e[s:t] += self.post_trace[self.sim.g.indices[seg]]
        # deliver scheduled dopamine stimulus
        if self.pending:
            keep = []
            for steps_left, sign in self.pending:
                cells = self.pam if sign > 0 else self.ppl1
                self.sim.inject(cells, self.p.dopamine_amount * abs(sign))
                if steps_left - 1 > 0:
                    keep.append((steps_left - 1, sign))
            self.pending = keep

    def dopamine(self, sign: float, note: str = "") -> None:
        """Reward (sign>0, PAM) or punishment (sign<0, PPL1): stimulate the cells and update weights."""
        self.pending.append((self.p.dopamine_steps, sign))
        w = self.sim.data[self.edge_pos]
        delta = self.p.eta * sign * self.e * self.w0
        new = np.clip(w + delta, self.p.w_min_ratio * self.w0, self.p.w_max_ratio * self.w0)
        real_delta = new - w
        self.sim.data[self.edge_pos] = new
        abs_delta = float(np.abs(real_delta).sum())
        self.total_abs_delta += abs_delta
        self.log.append({
            "step": int(self.sim.step_count), "sign": float(sign), "note": note,
            "edges_changed": int((real_delta != 0).sum()), "abs_delta": abs_delta,
            "mean_ratio": float((new / self.w0).mean()),
        })

    def summary(self) -> dict:
        w = self.sim.data[self.edge_pos] if self.attached else self.w_learned
        ratio = w / self.w0
        return {
            "kc_mbon_edges": int(self.n_edges),
            "events": len(self.log),
            "total_abs_delta": self.total_abs_delta,
            "edges_changed": int((w != self.w_frozen).sum()),
            "attached": bool(self.attached),
            "mean_weight_ratio": float(ratio.mean()),
            "min_weight_ratio": float(ratio.min()),
            "max_weight_ratio": float(ratio.max()),
            "edges_at_floor": int((ratio <= self.p.w_min_ratio + 1e-6).sum()),
            "edges_at_ceiling": int((ratio >= self.p.w_max_ratio - 1e-6).sum()),
        }
