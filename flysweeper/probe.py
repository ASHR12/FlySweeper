"""Does the board reach the descending neurons?  Vision-causality probe.

Runs the same seeded brain under three inputs and reports firing rates along the visual pathway
(photoreceptors -> lamina -> medulla -> T4/T5 -> visual projection neurons -> descending neurons):

    blind         no board (black screen)
    board-left    a mid-game board with the flickering cursor on the far left column
    board-right   the same board, cursor on the far right column

Usage:
    python -m flysweeper.probe [--preset flyai] [--steps 500] [--sensory-input]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .brain import Brain
from .decoder import PoolDecoder
from .encoder import RetinaEncoder
from .minesweeper import Minesweeper
from .sim import LIF, Params, PRESETS


def sample_board(rows: int, cols: int, seed: int = 7) -> np.ndarray:
    g = Minesweeper(rows, cols, 10, seed=seed)
    g.reveal(rows // 2, cols // 2)
    for r, c in [(0, 0), (rows - 1, cols - 1), (0, cols - 1)]:
        if not g.over:
            g.reveal(r, c)
    return g.visible()


def run(brain: Brain, params: Params, encoder: RetinaEncoder, board, cursor, steps: int, seed: int, blind: bool,
        sensory_input: bool, lamina_tonic: float = 0.0):
    sim = LIF(brain, params, seed=seed, sensory_input=sensory_input)
    if lamina_tonic:
        sim.bias[brain.cells(types=["L1", "L2", "L3", "L4", "L5"])] = lamina_tonic
    counts = np.zeros(brain.n, dtype=np.int64)
    encoder.reset()
    for _ in range(100):  # warm-up with the same input
        if not blind:
            idx, d = encoder.drive(board, cursor, sim.step_count)
            sim.inject(idx, d)
        sim.step()
    for _ in range(steps):
        if not blind:
            idx, d = encoder.drive(board, cursor, sim.step_count)
            sim.inject(idx, d)
        counts[sim.step()] += 1
    return counts / (steps * params.dt)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="flyai", choices=sorted(PRESETS))
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sensory-input", action="store_true", help="keep synapses onto sensory neurons")
    ap.add_argument("--retina-gain", type=float)
    ap.add_argument("--route", default="lamina", choices=["retina", "lamina"])
    ap.add_argument("--lamina-tonic", type=float, default=0.0, help="extra tonic drive into L1-L5 (DOOMFLY-style; image encoded by suppression)")
    args = ap.parse_args(argv)

    brain = Brain()
    params = Params.preset(args.preset)
    rows = cols = 9
    from .encoder import EncoderParams
    kw = {"route": args.route}
    if args.retina_gain:
        kw["retina_gain"] = args.retina_gain
    encoder = RetinaEncoder(brain, rows, cols, EncoderParams(**kw))
    decoder = PoolDecoder(brain)
    print("[probe] retina coverage:", json.dumps(encoder.coverage_summary()))
    board = sample_board(rows, cols)

    groups = {
        "R1-R6": brain.cells(types="R1-R6"),
        "R7/R8": np.concatenate([brain.cells(type_prefix="R7"), brain.cells(type_prefix="R8")]),
        "L1-L3": brain.cells(types=["L1", "L2", "L3"]),
        "L4/L5": brain.cells(types=["L4", "L5"]),
        "Mi1/Tm3/Tm1/Tm2": brain.cells(types=["Mi1", "Tm3", "Tm1", "Tm2"]),
        "T4": brain.cells(type_prefix="T4"),
        "T5": brain.cells(type_prefix="T5"),
        "LC/LPLC (loom+object)": brain.cells(types=["LC4", "LPLC2", "LPLC1", "LC6", "LC10a", "LC11", "LC16", "LC17", "LC9"]),
        "visual_projection (all)": brain.cells(superclass="visual_projection"),
        "descending (all)": brain.cells(superclass="descending_neuron"),
    }
    for a in decoder.actions:
        groups[f"pool:{a}"] = decoder.pool_idx[a]
    vp_l = brain.cells(superclass="visual_projection", side="L")
    vp_r = brain.cells(superclass="visual_projection", side="R")
    dn_l = brain.cells(superclass="descending_neuron", side="L")
    dn_r = brain.cells(superclass="descending_neuron", side="R")

    conds = {
        "blind": (True, (4, 4)),
        "board-left": (False, (4, 0)),
        "board-right": (False, (4, cols - 1)),
    }
    rates = {}
    for name, (blind, cursor) in conds.items():
        rates[name] = run(brain, params, encoder, board, cursor, args.steps, args.seed, blind, args.sensory_input, args.lamina_tonic)
        print(f"[probe] {name:12s} mean {rates[name].mean():.2f} Hz, spikes/step {rates[name].sum() * params.dt:.0f}")

    print(f"\n{'population':28s} {'n':>6s} {'blind':>8s} {'left':>8s} {'right':>8s}   (Hz per cell)")
    for g, idx in groups.items():
        print(f"{g:28s} {len(idx):6d} " + " ".join(f"{rates[c][idx].mean():8.2f}" for c in conds))
    print("\nlateralization (left-cursor vs right-cursor), Hz per cell:")
    for label, L, R in [("visual_projection", vp_l, vp_r), ("descending", dn_l, dn_r)]:
        print(f"  {label:18s} L side: {rates['board-left'][L].mean():.3f} vs {rates['board-right'][L].mean():.3f}   "
              f"R side: {rates['board-left'][R].mean():.3f} vs {rates['board-right'][R].mean():.3f}")
    d = rates["board-left"] - rates["blind"]
    print(f"\ncells whose rate changed > 1 Hz with the board on: {(np.abs(d) > 1).sum():,} of {brain.n:,}")
    top = np.argsort(-np.abs(d))[:12]
    nm = brain.neurons
    print("largest changes:", ", ".join(f"{nm['type'].iloc[i]}/{nm['side'].iloc[i]} {d[i]:+.1f}" for i in top))

    # Looming -> giant fiber pathway check (the reflex the decoder's "jump" relies on).
    print("\nlooming check: pulse left LC4+LPLC2 at 25 Hz, amplitude 0.8, no board")
    loom_l = brain.cells(types=["LC4", "LPLC2"], side="L")
    gf = {s: brain.cells(types="DNp01", side=s) for s in ("L", "R")}
    dnp10 = {s: brain.cells(types="DNp10", side=s) for s in ("L", "R")}
    for label, amp in (("silent", 0.0), ("loom-left", 0.8)):
        sim = LIF(brain, params, seed=args.seed, sensory_input=args.sensory_input)
        counts = np.zeros(brain.n, dtype=np.int64)
        for k in range(100 + args.steps):
            if amp and k % 2 == 0:
                sim.inject(loom_l, amp)
            f = sim.step()
            if k >= 100:
                counts[f] += 1
        r = counts / (args.steps * params.dt)
        print(f"  {label:10s} DNp01 L {r[gf['L']].mean():5.1f} Hz  R {r[gf['R']].mean():5.1f} Hz | DNp10 L {r[dnp10['L']].mean():5.1f}  R {r[dnp10['R']].mean():5.1f} | LC4+LPLC2 L {r[loom_l].mean():5.1f} Hz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
