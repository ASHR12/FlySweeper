"""Leaky integrate-and-fire simulation of the whole MaleCNS graph.

The model is the community "Fly64" cartoon (Paquette 2026; reused by fly.ai and the ack3
fuzzer), not a biophysical fly:

    every dt:  v <- v * exp(-dt/tau) + gain * W @ spikes(t-1) + tonic + noise + external
               spike if v >= threshold, then v <- reset (optional refractory steps)

W is the compiled out-edge CSR: synapse count x transmitter sign, normalized so each cell's
total absolute input is 1.  Propagation is event driven: only the outgoing edges of neurons
that spiked are visited, so the cost per step scales with activity, not with the 25.6M edges.
Zero-spike columns contribute exactly zero; nothing is pruned.

Presets:
    fly64   tonic 0.18, gain 1.5   (Fly64 defaults: every cell rests at threshold, ~4 Hz idle)
    flyai   tonic 0.14, gain 3.0   (fly.ai's inject.py settings, cells rest below threshold)

Usage (calibration / benchmark):
    python -m flysweeper.sim --preset flyai --steps 500
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, asdict

import numba as nb
import numpy as np

from .brain import Brain

PRESETS = {
    "fly64": dict(dt=0.020, tau=0.100, gain=1.5, tonic=0.180, noise_rate=1.2, noise_amp=0.22, threshold=1.0, reset=0.0, refractory_steps=0),
    "flyai": dict(dt=0.020, tau=0.100, gain=3.0, tonic=0.140, noise_rate=1.2, noise_amp=0.22, threshold=1.0, reset=0.0, refractory_steps=0),
}


@dataclass
class Params:
    dt: float = 0.020
    tau: float = 0.100
    gain: float = 3.0
    tonic: float = 0.140
    noise_rate: float = 1.2      # Hz, Bernoulli background kicks per cell
    noise_amp: float = 0.22
    threshold: float = 1.0
    reset: float = 0.0
    refractory_steps: int = 0
    floor: float = -5.0          # clamp for runaway inhibition; Fly64 has none

    @classmethod
    def preset(cls, name: str, **overrides) -> "Params":
        d = dict(PRESETS[name])
        d.update(overrides)
        return cls(**d)


@nb.njit(parallel=True, cache=True, fastmath=True)
def _propagate(fired, indptr, indices, data, acc):
    nthreads = acc.shape[0]
    n = acc.shape[1]
    for t in nb.prange(nthreads):
        row = acc[t]
        for i in range(n):
            row[i] = 0.0
        for k in range(t, fired.shape[0], nthreads):
            j = fired[k]
            for p in range(indptr[j], indptr[j + 1]):
                row[indices[p]] += data[p]


@nb.njit(parallel=True, cache=True, fastmath=True)
def _integrate(v, acc, decay, gain, tonic, noise, ext, threshold, reset, floor, refr, refr_steps, fired_mask):
    n = v.shape[0]
    nthreads = acc.shape[0]
    for i in nb.prange(n):
        s = np.float32(0.0)
        for t in range(nthreads):
            s += acc[t, i]
        if refr[i] > 0:
            refr[i] -= 1
            v[i] = reset
            fired_mask[i] = False
            continue
        vi = v[i] * decay + gain * s + tonic + noise[i] + ext[i]
        if vi >= threshold:
            fired_mask[i] = True
            v[i] = reset
            refr[i] = refr_steps
        else:
            if vi < floor:
                vi = floor
            fired_mask[i] = False
            v[i] = vi


class LIF:
    def __init__(
        self,
        brain: Brain,
        params: Params | None = None,
        seed: int = 0,
        threads: int | None = None,
        sensory_input: bool = True,
    ):
        self.brain = brain
        self.g = brain.graph
        self.p = params or Params()
        self.n = self.g.n
        self.sensory_input = sensory_input
        self.data = self.g.data
        if not sensory_input:
            # Whole-brain models (Shiu et al. 2024; fly.ai's fix for the ORN->ORN runaway loop)
            # remove every synapse onto sensory neurons so they are driven only by the stimulus.
            sc = brain.neurons["superclass"].fillna("").to_numpy().astype(str)
            is_sensory = np.char.find(sc, "sensory") >= 0
            self.data = self.g.data.copy()
            self.data[is_sensory[self.g.indices]] = 0.0
            self.n_silenced_edges = int(is_sensory[self.g.indices].sum())
        self.threads = threads or nb.get_num_threads()
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.v = np.zeros(self.n, dtype=np.float32)
        self.ext = np.zeros(self.n, dtype=np.float32)
        self.bias = np.zeros(self.n, dtype=np.float32)   # per-cell tonic drive added every step
        self.refr = np.zeros(self.n, dtype=np.int32)
        self.fired_mask = np.zeros(self.n, dtype=np.bool_)
        self.fired = np.zeros(0, dtype=np.int32)
        self.acc = np.zeros((self.threads, self.n), dtype=np.float32)
        self.step_count = 0
        self.decay = np.float32(np.exp(-self.p.dt / self.p.tau))
        self.noise_p = self.p.noise_rate * self.p.dt
        self.last_step_seconds = 0.0

    @property
    def time(self) -> float:
        return self.step_count * self.p.dt

    def inject(self, idx: np.ndarray, amount) -> None:
        """Add external drive for the next step (amount scalar or per-index array)."""
        np.add.at(self.ext, idx, np.asarray(amount, dtype=np.float32))

    def reset_state(self) -> None:
        self.v[:] = 0
        self.ext[:] = 0
        self.refr[:] = 0
        self.fired_mask[:] = False
        self.fired = np.zeros(0, dtype=np.int32)
        self.step_count = 0

    def step(self) -> np.ndarray:
        t0 = time.perf_counter()
        _propagate(self.fired, self.g.indptr, self.g.indices, self.data, self.acc)
        noise = (self.rng.random(self.n, dtype=np.float32) < self.noise_p).astype(np.float32)
        noise *= np.float32(self.p.noise_amp)
        self.ext += self.bias
        _integrate(
            self.v, self.acc, self.decay, np.float32(self.p.gain), np.float32(self.p.tonic), noise, self.ext,
            np.float32(self.p.threshold), np.float32(self.p.reset), np.float32(self.p.floor),
            self.refr, np.int32(self.p.refractory_steps), self.fired_mask,
        )
        self.fired = np.flatnonzero(self.fired_mask).astype(np.int32)
        self.ext[:] = 0
        self.step_count += 1
        self.last_step_seconds = time.perf_counter() - t0
        return self.fired

    def run(self, steps: int) -> np.ndarray:
        """Run steps, return per-neuron spike counts."""
        counts = np.zeros(self.n, dtype=np.int32)
        for _ in range(steps):
            counts[self.step()] += 1
        return counts


def calibrate(preset: str, steps: int, seed: int, warmup: int, overrides: dict, sensory_input: bool = True) -> dict:
    brain = Brain()
    params = Params.preset(preset, **overrides)
    sim = LIF(brain, params, seed=seed, sensory_input=sensory_input)
    print(f"[sim] {brain.n:,} neurons, {brain.graph.n_edges:,} edges, {sim.threads} threads, sensory_input={sensory_input}, params {asdict(params)}")
    sim.step()  # JIT compile
    t0 = time.perf_counter()
    sim.run(warmup)
    counts = sim.run(steps)
    elapsed = time.perf_counter() - t0
    per_step_ms = 1000 * elapsed / (warmup + steps)
    rate = counts / (steps * params.dt)
    sc = brain.neurons["superclass"].fillna("").to_numpy()
    by_sc = {}
    for name in sorted(set(sc)):
        m = sc == name
        by_sc[name] = round(float(rate[m].mean()), 2)
    result = {
        "preset": preset,
        "sensory_input": sensory_input,
        "steps": steps,
        "ms_per_step": round(per_step_ms, 2),
        "realtime_factor": round(1000 * params.dt / per_step_ms, 1),
        "mean_rate_hz": round(float(rate.mean()), 2),
        "median_rate_hz": round(float(np.median(rate)), 2),
        "silent_fraction": round(float((counts == 0).mean()), 3),
        "spikes_per_step": round(float(counts.sum() / steps), 0),
        "max_rate_hz": round(float(rate.max()), 1),
        "rate_by_superclass": by_sc,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="flyai", choices=sorted(PRESETS))
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gain", type=float)
    ap.add_argument("--tonic", type=float)
    ap.add_argument("--refractory-steps", type=int)
    ap.add_argument("--no-sensory-input", action="store_true", help="remove synapses onto sensory neurons (Shiu et al. convention)")
    args = ap.parse_args(argv)
    overrides = {k: v for k, v in dict(gain=args.gain, tonic=args.tonic, refractory_steps=args.refractory_steps).items() if v is not None}
    res = calibrate(args.preset, args.steps, args.seed, args.warmup, overrides, sensory_input=not args.no_sensory_input)
    import json
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
