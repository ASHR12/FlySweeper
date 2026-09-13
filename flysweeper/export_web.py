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

Trained mushroom-body weight sets (condition "fly-mb", see mb_policy.py) are exported separately,
one small JSON + binary pair per set plus a manifest, into web/weights/ (shipped with the repo):

    python -m flysweeper.export_web --weights data/compiled/mb_weights.npz --name round2 --round 2 \
        --winrate 0.48 --eval-games 100 --games-trained 1500 --default

    web/weights/<name>.json   pools (MBON cells per action), odor map (channel -> ORN cells + amp),
                              KC / PN cell lists and the fly-mb model settings (kc_kc_gain, kc_bias,
                              pn_kc_gain, pn_bias), decision settings (centred scores + saved running
                              means, reveal margin, mask_reveal), metadata and weight statistics
    web/weights/<name>.bin    plastic KC->MBON edges: u16 KC index (into "kc"), u8 MBON index (into
                              "mbon"), f32 trained weight; the frozen weight is the one in the graph
    web/weights/manifest.json every exported set (label "Checkpoint k · N episodes", win rate,
                              games trained, notes) + default
Re-running with the same --name replaces that entry (idempotent).
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


# ---------------------------------------------------------------------------------------------
# trained mushroom-body weight sets (fly-mb) -> web/weights/<name>.{json,bin} + manifest.json
# ---------------------------------------------------------------------------------------------
MB_CHANNELS = (  # flysweeper/oracle.py CHANNELS, cross-checked against the file's own list
    "cursor_hidden", "cursor_revealed", "cursor_flagged", "cursor_safe", "cursor_mine", "cursor_frontier", "cursor_free",
    "adj_numbers_0", "adj_numbers_1_2", "adj_numbers_3p",
    "adj_hidden_0", "adj_hidden_1_3", "adj_hidden_4p",
    "adj_max_0", "adj_max_1_2", "adj_max_3p",
    "safe_up", "safe_down", "safe_left", "safe_right", "no_safe_known",
    "guess_up", "guess_down", "guess_left", "guess_right", "guess_here", "guess_far",
    "cleared_lt_third", "cleared_mid", "cleared_gt_two_thirds", "untouched",
)


def export_weights(args) -> int:
    """Convert one MBPolicy.save() npz into web/weights/<name>.json + .bin and update manifest.json.

    Everything the browser needs to rebuild the fly-mb condition is derived here from the compiled
    graph + the file, exactly as mb_policy.MBPolicy does it: pools = MBON cells of the saved
    action_mbons types (concatenated per type, in order); plastic edges = every KC -> pool-MBON edge
    (recomputed and checked against the file's edge_pos); odor cells = the file's own orn_types per
    channel; KC / PN cell lists for the documented model changes."""
    src = Path(args.weights)
    out = Path(args.weights_out)
    out.mkdir(parents=True, exist_ok=True)
    name = args.name
    if not name or any(ch in name for ch in "/\\ ."):
        raise SystemExit("--name must be a simple identifier (used as the file name), e.g. round2")
    neurons = ft.read_feather(COMPILED / "neurons.feather")
    types = neurons["type"].fillna("").to_numpy().astype(str)
    sides = neurons["side"].to_numpy().astype(str)
    cls = neurons["class"].fillna("").to_numpy().astype(str)
    n = len(types)
    g = np.load(COMPILED / "graph.npz")
    indptr, indices, data = g["indptr"], g["indices"], g["data"]
    z = np.load(src, allow_pickle=False)
    saved = json.loads(str(z["params"])) if "params" in z else {}
    meta = json.loads(str(z["meta"])) if "meta" in z else {}
    actions = [str(a) for a in z["actions"]]
    channels = [str(c) for c in z["channels"]] if "channels" in z else list(MB_CHANNELS)
    if tuple(channels) != MB_CHANNELS:
        raise SystemExit(f"{src}: channel list differs from the one this exporter (and web/js/oracle.js) knows: {channels}")
    log(f"weights {src}: {len(z['edge_pos']):,} plastic edges, {int(z['games_trained'][0])} games trained")

    # ---- pools: MBON cells per action, concatenated per type in the saved order (MBPolicy.__init__)
    mb = {a: ((t,) if isinstance(t, str) else tuple(t)) for a, t in saved["action_mbons"]}
    pool_idx = {}
    for a in actions:
        parts = [np.flatnonzero(types == t).astype(np.int64) for t in mb[a]]
        pool_idx[a] = np.concatenate(parts) if parts else np.zeros(0, np.int64)
        if len(pool_idx[a]) == 0:
            raise SystemExit(f"pool {a} ({mb[a]}) matched no cells")
    pool_of = np.full(n, -1, dtype=np.int64)
    for k, a in enumerate(actions):
        if (pool_of[pool_idx[a]] >= 0).any():
            raise SystemExit(f"pool {a} overlaps another pool")
        pool_of[pool_idx[a]] = k
    mbon = np.concatenate([pool_idx[a] for a in actions])            # "mbon" list, u8 index space
    if len(mbon) > 255:
        raise SystemExit("more than 255 pool MBON cells; widen the mbon index")
    mbon_pos = {int(j): i for i, j in enumerate(mbon)}

    # ---- plastic edges, recomputed like MBPolicy and checked against the file
    kc = np.flatnonzero(np.char.startswith(types, "KC")).astype(np.int64)
    is_kc = np.zeros(n, dtype=bool); is_kc[kc] = True
    pn = np.flatnonzero(cls == "ALPN").astype(np.int64)
    is_pn = np.zeros(n, dtype=bool); is_pn[pn] = True
    pre = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr).astype(np.int64))
    sel = is_kc[pre] & (pool_of[indices] >= 0)
    edge_pos = np.flatnonzero(sel).astype(np.int64)
    if not np.array_equal(edge_pos, z["edge_pos"]):
        raise SystemExit(f"{src}: plastic edge set does not match this graph / pool assignment "
                         f"({len(edge_pos):,} recomputed vs {len(z['edge_pos']):,} in the file)")
    e_pre, e_post = pre[edge_pos], indices[edge_pos].astype(np.int64)
    w = z["w"].astype(np.float32)
    w_frozen = data[edge_pos].astype(np.float32)
    if not np.allclose(np.abs(w_frozen), z["w0"], atol=1e-7):
        raise SystemExit(f"{src}: w0 differs from the compiled graph weights")
    if "edge_kc" in z and not np.array_equal(z["edge_kc"], e_pre):
        raise SystemExit(f"{src}: edge_kc mismatch")
    kc_index = np.searchsorted(kc, e_pre)
    assert np.array_equal(kc[kc_index], e_pre)
    mbon_index = np.array([mbon_pos[int(j)] for j in e_post], dtype=np.uint8)
    edge_pool = pool_of[e_post]
    if "edge_pool" in z and not np.array_equal(z["edge_pool"].astype(np.int64), edge_pool):
        raise SystemExit(f"{src}: edge_pool mismatch")
    onto_kc = is_kc[indices]
    n_kc2kc = int((onto_kc & is_kc[pre]).sum())
    n_pn2kc = int((onto_kc & is_pn[pre]).sum())
    del pre

    # ---- odor map: the file's own channel -> ORN types ("+"-joined), cells concatenated per type
    if "orn_types" in z:
        orn_types = [str(t).split("+") for t in z["orn_types"]]
    else:  # very old files: fall back to the module table
        from .mb_policy import ORN_TYPES
        orn_types = [[t] for t in ORN_TYPES[:len(channels)]]
    odor_cells = []
    for ts in orn_types:
        parts = [np.flatnonzero(types == t).astype(np.int64) for t in ts]
        c = np.concatenate(parts) if parts else np.zeros(0, np.int64)
        if len(c) == 0:
            raise SystemExit(f"ORN types {ts} matched no cells")
        odor_cells.append(c)
    all_odor = np.concatenate(odor_cells)
    if len(np.unique(all_odor)) != len(all_odor):
        raise SystemExit("an ORN cell belongs to two channels; the browser would double-drive it")

    # ---- settings that travel with the weights (mb_policy.MBPolicy.load / params_from_file)
    center_scores = bool(saved.get("center_scores", False)) if saved else True
    ratio = np.abs(w) / np.maximum(np.abs(w_frozen), 1e-12)
    changed = int((np.abs(np.abs(w) - np.abs(w_frozen)) > 1e-7).sum())
    games_trained = int(args.games_trained if args.games_trained is not None else z["games_trained"][0])
    round_no = int(args.round) if args.round is not None else None
    label = args.label
    if not label:
        # standard ML naming for the selector: "Checkpoint k · N episodes"; the win rate is kept in the
        # manifest (win_rate / eval_games) for the README and tooltips but not shown in the menu
        label = f"{f'Checkpoint {round_no}' if round_no is not None else name} · {games_trained:,} episodes"

    # ---- binary: u16 kc index | u8 mbon index | f32 trained weight (little-endian, in that order)
    bin_bytes = (kc_index.astype("<u2").tobytes() + mbon_index.astype("u1").tobytes() + w.astype("<f4").tobytes())
    bin_path = out / f"{name}.bin"
    bin_path.write_bytes(bin_bytes)
    bin_info = {"bytes": len(bin_bytes), "sha256": hashlib.sha256(bin_bytes).hexdigest(),
                "layout": ["kc_index:u16", "mbon_index:u8", "w:f32"], "count": int(len(w))}

    spec = {
        "format": 1,
        "id": name,
        "label": label,
        "round": round_no,
        "win_rate": args.winrate,
        "eval_games": args.eval_games,
        "games_trained": games_trained,
        "notes": args.notes or "",
        "source": str(src),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "condition": "fly-mb",
        "actions": actions,
        "channels": channels,
        "n_neurons": int(n),
        "edges": {"file": f"{name}.bin", **bin_info},
        "kc": kc.tolist(),
        "pn": pn.tolist(),
        "mbon": mbon.tolist(),
        "pools": {a: {"types": list(mb[a]), "idx": pool_idx[a].tolist(),
                      "cells": [f"{types[i]}/{sides[i]}" for i in pool_idx[a]]} for a in actions},
        "odor": {"amp": float(saved.get("odor_amp", 1.0)), "extra_orn": int(saved.get("extra_orn", 0)),
                 "types": orn_types, "cells": [c.tolist() for c in odor_cells]},
        "kc_model": {"kc_kc_gain": float(saved.get("kc_kc_gain", 0.0)), "kc_bias": float(saved.get("kc_bias", -0.2)),
                     "pn_kc_gain": float(saved.get("pn_kc_gain", 1.0)), "pn_bias": float(saved.get("pn_bias", 0.0)),
                     "kc_cells": int(len(kc)), "pn_cells": int(len(pn)), "kc_kc_edges": n_kc2kc, "pn_kc_edges": n_pn2kc},
        "decision": {"temperature": 0.0, "center_scores": center_scores,
                     "score_mean_alpha": float(saved.get("score_mean_alpha", 0.02)),
                     "score_mean": [float(x) for x in z["score_mean"]] if "score_mean" in z else None,
                     "reveal_margin": float(saved.get("reveal_margin", 0.0)),
                     "mask_reveal": bool(saved.get("mask_reveal", False)),
                     "jump_distance": int(saved.get("jump_distance", 5)), "epsilon": 0.0},
        "stats": {"plastic_edges": int(len(w)), "edges_changed": changed, "mean_weight_ratio": float(ratio.mean()),
                  "min_weight_ratio": float(ratio.min()), "max_weight_ratio": float(ratio.max()),
                  "pool_ratio": {a: float(ratio[edge_pool == k].mean()) for k, a in enumerate(actions)},
                  "plastic_edges_per_pool": {a: int((edge_pool == k).sum()) for k, a in enumerate(actions)},
                  "pool_cells": {a: int(len(pool_idx[a])) for a in actions},
                  "baseline": float(z["baseline"][0]) if "baseline" in z else None},
        "train_meta": meta,
        "train_params": saved,
    }
    json_info = write_json(out / f"{name}.json", spec)
    log(f"pools: " + ", ".join(f"{a} {len(pool_idx[a])} cells" for a in actions)
        + f"; {len(w):,} plastic edges ({changed:,} changed, mean ratio {ratio.mean():.3f}); "
        f"odor {len(all_odor)} ORN cells over {len(channels)} channels (extra_orn {spec['odor']['extra_orn']}); "
        f"KC {len(kc)} (kc->kc {n_kc2kc:,} edges x {spec['kc_model']['kc_kc_gain']}, bias {spec['kc_model']['kc_bias']}), "
        f"PN {len(pn)} (pn->kc {n_pn2kc:,} edges x {spec['kc_model']['pn_kc_gain']}, bias {spec['kc_model']['pn_bias']}); "
        f"reveal margin {spec['decision']['reveal_margin']}, centred {center_scores}")
    log(f"wrote {name}.json ({json_info['bytes'] / 1e3:.0f} KB) + {name}.bin ({bin_info['bytes'] / 1e3:.0f} KB) -> {out}")

    # ---- manifest (idempotent: replace the entry with the same id)
    man_path = out / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {"format": 1, "default": None, "sets": []}
    entry = {"id": name, "label": label, "round": round_no, "win_rate": args.winrate, "eval_games": args.eval_games,
             "games_trained": games_trained, "notes": args.notes or "", "json": f"{name}.json", "bin": f"{name}.bin",
             "plastic_edges": int(len(w)), "edges_changed": changed, "bytes": json_info["bytes"] + bin_info["bytes"],
             "sha256": {"json": json_info["sha256"], "bin": bin_info["sha256"]}, "created": spec["created"]}
    manifest["sets"] = [s for s in manifest.get("sets", []) if s.get("id") != name] + [entry]
    manifest["sets"].sort(key=lambda s: ((s.get("round") is None), s.get("round") or 0, s["id"]))
    if args.default or manifest.get("default") not in {s["id"] for s in manifest["sets"]}:
        manifest["default"] = name
    manifest["updated"] = spec["created"]
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"manifest: {len(manifest['sets'])} set(s), default '{manifest['default']}'")

    # ---- read back and verify the binary
    raw = bin_path.read_bytes()
    m = len(w)
    ki = np.frombuffer(raw[: 2 * m], dtype="<u2"); mi = np.frombuffer(raw[2 * m: 3 * m], dtype="u1"); ww = np.frombuffer(raw[3 * m:], dtype="<f4")
    assert np.array_equal(kc[ki], e_pre) and np.array_equal(mbon[mi], e_post) and np.array_equal(ww, w), "binary readback mismatch"
    log("readback: plastic edge list round-trips (pre KC, post MBON, weight): OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "web" / "data"))
    ap.add_argument("--rows", type=int, default=GAME["rows"])
    ap.add_argument("--cols", type=int, default=GAME["cols"])
    ap.add_argument("--keep-sensory-input", action="store_true", help="keep synapses onto sensory neurons (Python default is to silence them)")
    wg = ap.add_argument_group("trained weight sets (fly-mb)", "with --weights only the weight set is exported")
    wg.add_argument("--weights", default=None, help="MBPolicy.save() npz, e.g. data/compiled/mb_weights.npz")
    wg.add_argument("--name", default=None, help="set id / file stem, e.g. round2")
    wg.add_argument("--label", default=None, help='display label; default "Checkpoint k · N episodes" (win rate stays in the manifest, not in the label)')
    wg.add_argument("--round", type=int, default=None)
    wg.add_argument("--winrate", type=float, default=None, help="held-out win rate, e.g. 0.48")
    wg.add_argument("--eval-games", type=int, default=None, help="number of held-out boards behind --winrate")
    wg.add_argument("--games-trained", type=int, default=None, help="override the file's games_trained")
    wg.add_argument("--notes", default=None)
    wg.add_argument("--default", action="store_true", help="make this set the manifest default")
    wg.add_argument("--weights-out", default=str(ROOT / "web" / "weights"))
    args = ap.parse_args(argv)
    if args.weights:
        if not args.name:
            ap.error("--weights needs --name")
        return export_weights(args)
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
