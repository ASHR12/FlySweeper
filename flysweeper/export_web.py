"""Export the compiled MaleCNS graph for the in-browser WebGPU simulator (web/).

Reads data/compiled/{neurons.feather, graph.npz, eye.npz, meta.json} and writes browser-friendly
little-endian binaries plus two JSON files into web/data/:

    in_indptr.u32     n+1        in-edge CSR by POSTsynaptic neuron (GPU gather, no atomics)
    in_indices.u32    E          presynaptic neuron index of each in-edge
    in_weights.f32    E          signed, normalized weight (sum |w| into each post cell = 1)
    order.u32         n          neuron ids sorted by in-degree (descending) for GPU load balance
    pool_id.u32       n          decoder pool of each neuron, 0xFFFFFFFF if none
    type_id.u16       n          index into model.json "type_names" (hover labels)
    plot_idx.u32      n_plot     neurons that have a soma position (brain map)
    soma_uv.f32       2*n_plot   (u, v) in [0,1]: fly's left (high x) on the left, brain top (low z) up
    region.u8         n_plot     colour region of each plotted neuron (palette in model.json)
    model.json        sim params, encoder target lists (9x9 board), looming lists, decoder pools,
                      luminance table, region palette, counts, citation
    manifest.json     byte size, sha256, dtype and element count of every file

Synapses onto sensory neurons (superclass containing "sensory") are DROPPED, which is what
`flysweeper.sim.LIF(..., sensory_input=False)` does by zeroing them; the browser therefore
reproduces the `sensory_input=False` behaviour that the Python agent uses.

The encoder target lists are recomputed here with the same equal-count column partition as
flysweeper/encoder.py (route "lamina"), and cross-checked against RetinaEncoder when importable.

Usage:
    python -m flysweeper.export_web [--out web/data] [--rows 9 --cols 9]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as ft

from .paths import COMPILED, ROOT

SIM_PARAMS = dict(dt=0.020, tau=0.100, gain=3.0, tonic=0.140, noise_rate=1.2, noise_amp=0.22,
                  threshold=1.0, reset=0.0, floor=-5.0, refractory_steps=0, preset="flyai")

ENCODER_PARAMS = dict(
    route="lamina", retina_gain=0.62, lum_weight=0.45, temporal_weight=1.6, color_weight=0.25,
    cursor_flicker=0.35, lum_hidden=0.10, lum_number_base=0.35, lum_number_step=0.075, lum_flag=0.55,
    lum_mine=1.00, binocular=False, lamina_types=["L1", "L2", "L3"], danger_loom=True, loom_gain=0.5,
    loom_threshold=2, loom_types=["LC4", "LPLC2"],
)

POOLS = [  # (action, types, side)  -- same sets as flysweeper/decoder.py DEFAULT_POOLS (flags off)
    ("up", ["DNp09", "DNg100", "DNg97"], None),
    ("down", ["MDN"], None),
    ("left", ["DNa02", "DNa11", "DNg13"], "L"),
    ("right", ["DNa02", "DNa11", "DNg13"], "R"),
    ("reveal", ["DNpe017", "DNp10"], None),
    ("jump", ["DNp01"], None),
]

GAME = dict(rows=9, cols=9, mines=10, turn_steps=15, max_turns=400, settle_steps=25, idle_steps=500,
            idle_warmup_steps=100, running_alpha=0.1)

REGIONS = [  # same palette as flysweeper/server.py
    ("optic lobe", "ol_", (86, 156, 255)),
    ("visual projection", "visual_", (120, 220, 255)),
    ("central brain", "cb_", (255, 196, 90)),
    ("descending / ascending", "descending", (255, 90, 120)),
    ("nerve cord", "vnc_", (140, 255, 150)),
    ("sensory", "sensory", (220, 140, 255)),
    ("other", "", (170, 170, 170)),
]

LABEL = ("MaleCNS v1.0 wiring (Berg et al., Cell 2026, CC BY 4.0) with engineered dynamics, sensors and "
         "buttons. Not a validated fly. Not a trained Minesweeper player. Connectome frozen.")

HIDDEN, FLAG, MINE = -1, 9, 10
LUM, FLAG_CH, CURSOR_CH = 0, 1, 2


def log(msg: str) -> None:
    print(f"[export] {msg}", flush=True)


def region_of(superclass: str) -> int:
    for i, (_, prefix, _) in enumerate(REGIONS):
        if prefix and (superclass.startswith(prefix) or (prefix in ("descending", "sensory") and prefix in superclass)):
            return i
    if "ascending" in superclass:
        return 3
    return len(REGIONS) - 1


# ---------------------------------------------------------------------------------------------
# encoder geometry (verbatim logic of flysweeper/encoder.py::_column_cells, kept local so the
# export does not depend on the encoder module being importable while it is being edited)
# ---------------------------------------------------------------------------------------------
def column_cells(neurons: pd.DataFrame, eye: dict, rows: int, cols: int, binocular: bool) -> dict:
    hx = neurons["hex1"].to_numpy(dtype=np.float64)
    hy = neurons["hex2"].to_numpy(dtype=np.float64)
    side = neurons["side"].to_numpy().astype(str)
    has = ~np.isnan(hx)
    cols_seen = set()
    for e_, s in ((0, "L"), (1, "R")):
        m = has & (side == s)
        for a, b in zip(hx[m].astype(int), hy[m].astype(int)):
            cols_seen.add((e_, a, b))
    for e_, a, b in zip(eye["eye"], eye["hex1"], eye["hex2"]):
        cols_seen.add((int(e_), int(a), int(b)))

    mapping: dict = {}
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
            r = rows - 1 - b
            for k, ci in zip(ox, col):
                mapping[cs[k]] = int(r * cols + ci)
    return mapping


def encoder_targets(neurons: pd.DataFrame, eye: dict, rows: int, cols: int, p: dict):
    """(idx, chan, cell) target lists exactly as RetinaEncoder builds them for route 'lamina'."""
    col2cell = column_cells(neurons, eye, rows, cols, p["binocular"])
    idx, chan, cell = [], [], []
    for i, k, e_, a, b in zip(eye["pr_idx"], eye["kind"], eye["eye"], eye["hex1"], eye["hex2"]):
        key = (int(e_), int(a), int(b))
        if key not in col2cell:
            continue
        if k == 0:
            if p["route"] != "retina":
                continue
            ch = LUM
        elif k == 1:
            ch = FLAG_CH
        else:
            ch = CURSOR_CH
        idx.append(int(i)); chan.append(ch); cell.append(col2cell[key])
    if p["route"] == "lamina":
        types = neurons["type"].fillna("").to_numpy().astype(str)
        lam = np.flatnonzero(np.isin(types, np.asarray(p["lamina_types"], dtype=str)))
        hx = neurons["hex1"].to_numpy(); hy = neurons["hex2"].to_numpy(); side = neurons["side"].to_numpy().astype(str)
        for i in lam:
            if np.isnan(hx[i]):
                continue
            key = (0 if side[i] == "L" else 1, int(hx[i]), int(hy[i]))
            if key in col2cell:
                idx.append(int(i)); chan.append(LUM); cell.append(col2cell[key])
    return np.array(idx, dtype=np.int32), np.array(chan, dtype=np.int8), np.array(cell, dtype=np.int32)


def lum_table(p: dict) -> list:
    t = np.zeros(12, dtype=np.float32)
    t[HIDDEN + 1] = p["lum_hidden"]
    for n in range(9):
        t[n + 1] = p["lum_number_base"] + p["lum_number_step"] * n
    t[FLAG + 1] = p["lum_flag"]
    t[MINE + 1] = p["lum_mine"]
    return [round(float(x), 6) for x in t]


# ---------------------------------------------------------------------------------------------
def cells(types_arr, side_arr, types, side=None) -> np.ndarray:
    m = np.isin(types_arr, np.asarray(types, dtype=str))
    if side is not None:
        m &= side_arr == side
    return np.flatnonzero(m).astype(np.int32)


def write_bin(path: Path, arr: np.ndarray, dtype) -> dict:
    a = np.ascontiguousarray(arr.astype(dtype, copy=False))
    if a.dtype.byteorder == ">":
        a = a.byteswap().newbyteorder()
    data = a.tobytes()
    path.write_bytes(data)
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "dtype": np.dtype(dtype).name, "count": int(a.size)}


def write_json(path: Path, obj) -> dict:
    data = json.dumps(obj, separators=(",", ":")).encode()
    path.write_bytes(data)
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "dtype": "json", "count": 1}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "web" / "data"))
    ap.add_argument("--rows", type=int, default=GAME["rows"])
    ap.add_argument("--cols", type=int, default=GAME["cols"])
    ap.add_argument("--keep-sensory-input", action="store_true", help="keep synapses onto sensory neurons (Python default is to silence them)")
    args = ap.parse_args(argv)
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    neurons = ft.read_feather(COMPILED / "neurons.feather")
    g = np.load(COMPILED / "graph.npz")
    eye = dict(np.load(COMPILED / "eye.npz"))
    meta = json.loads((COMPILED / "meta.json").read_text())
    indptr, post, data = g["indptr"], g["indices"], g["data"]
    n = len(indptr) - 1
    e_total = len(post)
    log(f"{n:,} neurons, {e_total:,} edges")

    # --- in-edge CSR by postsynaptic neuron -------------------------------------------------
    pre = np.repeat(np.arange(n, dtype=np.int32), np.diff(indptr).astype(np.int64))
    sc = neurons["superclass"].fillna("").to_numpy().astype(str)
    is_sensory = np.char.find(sc, "sensory") >= 0
    if args.keep_sensory_input:
        keep = np.ones(e_total, dtype=bool)
    else:
        keep = ~is_sensory[post]
    e_silenced = int((~keep).sum())
    pre, post_k, w = pre[keep], post[keep], data[keep]
    e_kept = len(w)
    log(f"dropping {e_silenced:,} synapses onto {int(is_sensory.sum()):,} sensory neurons -> {e_kept:,} active edges")
    # stable sort by post keeps pre ascending inside each row (pre was ascending in the COO)
    order = np.argsort(post_k, kind="stable")
    in_indices = pre[order].astype(np.uint32)
    in_weights = w[order].astype(np.float32)
    in_deg = np.bincount(post_k, minlength=n).astype(np.int64)
    in_indptr = np.zeros(n + 1, dtype=np.uint32)
    in_indptr[1:] = np.cumsum(in_deg)
    assert in_indptr[-1] == e_kept
    del pre, post_k, w, order

    # GPU work order: heaviest rows first so the tail of the dispatch is made of cheap rows
    gpu_order = np.argsort(-in_deg, kind="stable").astype(np.uint32)

    # --- decoder pools ---------------------------------------------------------------------
    types = neurons["type"].fillna("").to_numpy().astype(str)
    sides = neurons["side"].to_numpy().astype(str)
    pool_id = np.full(n, 0xFFFFFFFF, dtype=np.uint32)
    pools = {}
    for k, (name, tps, side) in enumerate(POOLS):
        idx = cells(types, sides, tps, side)
        if len(idx) == 0:
            raise RuntimeError(f"pool {name} matched no cells")
        if (pool_id[idx] != 0xFFFFFFFF).any():
            raise RuntimeError(f"pool {name} overlaps another pool")
        pool_id[idx] = k
        pools[name] = {"types": tps, "side": side, "idx": idx.tolist(),
                       "cells": [f"{types[i]}/{sides[i]}" for i in idx]}
    log("pools: " + ", ".join(f"{a} {len(p['idx'])}" for a, p in pools.items()))

    # --- encoder targets (9x9 board) --------------------------------------------------------
    p = dict(ENCODER_PARAMS)
    t_idx, t_chan, t_cell = encoder_targets(neurons, eye, args.rows, args.cols, p)
    loom = {s: cells(types, sides, p["loom_types"], s).tolist() for s in ("L", "R")}
    if len(np.unique(t_idx)) != len(t_idx):
        raise RuntimeError("encoder targets contain duplicates")
    if np.intersect1d(t_idx, np.concatenate([loom["L"], loom["R"]])).size:
        raise RuntimeError("looming cells overlap encoder targets")
    per_cell = np.bincount(t_cell[t_chan == LUM], minlength=args.rows * args.cols)
    log(f"encoder: {len(t_idx):,} targets ({int((t_chan == LUM).sum()):,} lamina, {int((t_chan == FLAG_CH).sum()):,} R7, "
        f"{int((t_chan == CURSOR_CH).sum()):,} R8); lamina cells per board cell {per_cell.min()}..{per_cell.max()}; "
        f"loom L {len(loom['L'])} R {len(loom['R'])}")
    try:  # cross-check against the live encoder module when it imports cleanly
        from .brain import Brain
        from .encoder import EncoderParams, RetinaEncoder
        enc = RetinaEncoder(Brain(), args.rows, args.cols, EncoderParams(
            route=p["route"], binocular=p["binocular"], lamina_types=tuple(p["lamina_types"])))
        same = (np.array_equal(enc.idx, t_idx) and np.array_equal(enc.chan, t_chan) and np.array_equal(enc.cell, t_cell))
        log(f"cross-check vs flysweeper.encoder.RetinaEncoder: {'identical' if same else 'MISMATCH'}")
        if not same:
            return 1
    except Exception as ex:  # pragma: no cover - depends on concurrent edits
        log(f"cross-check skipped ({type(ex).__name__}: {ex})")

    # --- brain map atlas -------------------------------------------------------------------
    x = neurons["x"].to_numpy(dtype=np.float32)
    z = neurons["z"].to_numpy(dtype=np.float32)
    plot_idx = np.flatnonzero(~np.isnan(x)).astype(np.uint32)
    xs, zs = x[plot_idx], z[plot_idx]
    u = 1.0 - (xs - xs.min()) / (xs.max() - xs.min())
    v = (zs - zs.min()) / (zs.max() - zs.min())
    soma_uv = np.stack([u, v], axis=1).astype(np.float32).ravel()
    region = np.array([region_of(s) for s in sc[plot_idx]], dtype=np.uint8)

    type_names, type_id = np.unique(types, return_inverse=True)
    if len(type_names) >= 65535:
        raise RuntimeError("too many types for u16")

    # --- write ------------------------------------------------------------------------------
    files = {}
    files["in_indptr.u32"] = write_bin(out / "in_indptr.u32", in_indptr, np.uint32)
    files["in_indices.u32"] = write_bin(out / "in_indices.u32", in_indices, np.uint32)
    files["in_weights.f32"] = write_bin(out / "in_weights.f32", in_weights, np.float32)
    files["order.u32"] = write_bin(out / "order.u32", gpu_order, np.uint32)
    files["pool_id.u32"] = write_bin(out / "pool_id.u32", pool_id, np.uint32)
    files["type_id.u16"] = write_bin(out / "type_id.u16", type_id, np.uint16)
    files["plot_idx.u32"] = write_bin(out / "plot_idx.u32", plot_idx, np.uint32)
    files["soma_uv.f32"] = write_bin(out / "soma_uv.f32", soma_uv, np.float32)
    files["region.u8"] = write_bin(out / "region.u8", region, np.uint8)

    model = {
        "dataset": meta.get("dataset", "MaleCNS v1.0"),
        "citation": meta.get("citation"),
        "label": LABEL,
        "n_neurons": int(n),
        "n_edges_total": int(e_total),
        "n_edges_silenced": e_silenced,
        "n_edges_active": int(e_kept),
        "n_sensory_neurons": int(is_sensory.sum()),
        "sensory_input": bool(args.keep_sensory_input),
        "n_plot": int(len(plot_idx)),
        "max_in_degree": int(in_deg.max()),
        "sim": SIM_PARAMS,
        "game": {**GAME, "rows": args.rows, "cols": args.cols},
        "encoder": {
            "params": p,
            "lum_table": lum_table(p),
            "codes": {"HIDDEN": HIDDEN, "FLAG": FLAG, "MINE": MINE},
            "channels": {"LUM": LUM, "FLAG_CH": FLAG_CH, "CURSOR_CH": CURSOR_CH},
            "idx": t_idx.tolist(),
            "chan": t_chan.tolist(),
            "cell": t_cell.tolist(),
            "lum_cells_per_board_cell": per_cell.tolist(),
            "loom": loom,
        },
        "decoder": {"actions": [a for a, _, _ in POOLS], "pools": pools, "running_alpha": GAME["running_alpha"]},
        "regions": [{"name": name, "color": list(color)} for name, _, color in REGIONS],
        "type_names": type_names.tolist(),
        "superclass_by_type": None,
    }
    files["model.json"] = write_json(out / "model.json", model)

    manifest = {
        "format": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": {"compiled_meta": {k: meta[k] for k in ("dataset", "n_neurons", "n_edges", "n_synapses") if k in meta}},
        "n_neurons": int(n),
        "n_edges_total": int(e_total),
        "n_edges_silenced": e_silenced,
        "n_edges_active": int(e_kept),
        "n_plot": int(len(plot_idx)),
        "endianness": "little",
        "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    total_mb = sum(f["bytes"] for f in files.values()) / 1e6
    log(f"wrote {len(files) + 1} files, {total_mb:.1f} MB -> {out}")

    # --- verify by reading back -------------------------------------------------------------
    ip = np.fromfile(out / "in_indptr.u32", dtype="<u4")
    ii = np.fromfile(out / "in_indices.u32", dtype="<u4")
    iw = np.fromfile(out / "in_weights.f32", dtype="<f4")
    assert len(ip) == n + 1 and len(ii) == e_kept and len(iw) == e_kept, "readback length mismatch"
    assert np.all(np.diff(ip.astype(np.int64)) >= 0) and ip[-1] == e_kept, "indptr not monotonic"
    assert ii.max() < n, "presynaptic index out of range"
    post_rb = np.repeat(np.arange(n, dtype=np.int64), np.diff(ip.astype(np.int64)))
    abs_sum = np.bincount(post_rb, weights=np.abs(iw).astype(np.float64), minlength=n)
    active = (~is_sensory) & (in_deg > 0)
    dev = np.abs(abs_sum[active] - 1.0)
    log(f"readback: {len(ii):,} edges; sum|w| into non-sensory cells with input: mean {abs_sum[active].mean():.6f}, "
        f"max |dev| {dev.max():.2e}, {(dev < 1e-3).mean() * 100:.2f}% within 1e-3 ({int(active.sum()):,} cells)")
    if not args.keep_sensory_input:
        assert abs_sum[is_sensory].sum() == 0.0, "sensory neurons still receive synapses"
        log(f"sensory neurons receive no synapses: OK ({int(is_sensory.sum()):,} cells)")
    # every kept edge appears exactly once (compare with out-CSR total per pre neuron)
    out_deg_kept = np.bincount(ii, minlength=n)
    out_deg_expected = np.bincount(np.repeat(np.arange(n), np.diff(indptr).astype(np.int64))[keep], minlength=n)
    assert np.array_equal(out_deg_kept, out_deg_expected), "out-degree mismatch after transpose"
    log(f"edge count {len(ii):,} == {e_total:,} - {e_silenced:,} silenced: OK; transpose preserves out-degrees: OK")
    log(f"done in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
