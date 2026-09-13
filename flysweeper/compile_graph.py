"""Compile the MaleCNS v1.0 flat tables into a simulation-ready graph.

What comes out (in data/compiled/):
  neurons.feather   one row per retained neuron: idx, bodyId, type, class, superclass, side,
                    transmitter, sign, soma xyz, optic-lobe hex column, dimorphism labels ...
  graph.npz         out-edge CSR by presynaptic neuron: indptr, indices(post), data(signed,
                    normalized float32), counts(raw synapse count int32)
  eye.npz           photoreceptor -> eye column -> normalized (u, v) screen coordinate
  meta.json         every policy and count used, plus the SHA-256 of the inputs

Policies (all of them are engineering choices, documented in README.md):
  nodes     every annotation row with a superclass and statusLabel != Glia  -> 166,700 neurons
  edges     every released connection between retained neurons            -> 25,582,938 edges
            (optionally --min-weight N to drop weak connections)
  sign      GABA / glutamate / histamine presynaptic cells are inhibitory (-1); acetylcholine,
            dopamine, octopamine, serotonin and unknown are excitatory (+1). Receptor-specific
            effects are ignored.  This is the community convention (Fly64, fly.ai, DOOMFLY).
  weight    synapse count x sign, divided by the postsynaptic cell's total absolute input
            (Fly64 / fly.ai normalization) unless --no-normalize.
  eye       each photoreceptor is assigned to the optic-lobe column (assignedOlHex1/2) of the
            hex-annotated cell it makes the most synapses onto (DOOMFLY's method).

Usage:
    python -m flysweeper.compile_graph [--min-weight 1] [--no-normalize]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.feather as ft
from scipy import sparse

from .paths import COMPILED, RAW, ensure_dirs
from .prepare import FILES, raw_path

INHIBITORY = {"gaba", "glutamate", "histamine"}
EXCITATORY = {"acetylcholine", "dopamine", "octopamine", "serotonin"}


def log(msg: str) -> None:
    print(f"[compile] {msg}", flush=True)


def resolve_nt(row_consensus, row_celltype, row_predicted) -> str:
    for v in (row_consensus, row_celltype, row_predicted):
        if isinstance(v, str) and v and v != "unclear":
            return v
    return "unknown"


def load_neurons() -> pd.DataFrame:
    ann = ft.read_feather(raw_path("annotations"))
    keep = ann["superclass"].notna() & (ann["statusLabel"].astype(str) != "Glia")
    n = ann.loc[keep].copy()
    n = n.sort_values("bodyId").reset_index(drop=True)
    n["idx"] = np.arange(len(n), dtype=np.int32)

    side = n["somaSide"].where(n["somaSide"].notna(), n["rootSide"])
    n["side"] = side.fillna("").astype(str)

    loc = n["somaLocation"]
    xyz = np.full((len(n), 3), np.nan, dtype=np.float32)
    has = loc.notna().to_numpy()
    if has.any():
        arr = np.array([list(v) for v in loc[has]], dtype=np.float32)
        xyz[has] = arr
    n["x"], n["y"], n["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    nt = ft.read_feather(
        raw_path("neurotransmitters"),
        columns=["body", "consensus_nt", "celltype_predicted_nt", "predicted_nt", "predicted_nt_confidence"],
    )
    nt = nt.drop_duplicates("body").set_index("body")
    j = n.join(nt, on="bodyId")
    n["nt"] = [
        resolve_nt(a, b, c)
        for a, b, c in zip(j["consensus_nt"], j["celltype_predicted_nt"], j["predicted_nt"])
    ]
    n["nt_confidence"] = j["predicted_nt_confidence"].astype("float32").to_numpy()
    n["sign"] = np.where(n["nt"].isin(INHIBITORY), -1, 1).astype(np.int8)

    cols = [
        "idx", "bodyId", "type", "class", "superclass", "subclass", "side", "nt", "nt_confidence", "sign",
        "x", "y", "z", "assignedOlHex1", "assignedOlHex2", "dimorphism", "fruDsx", "entryNerve",
        "exitNerve", "receptorType", "statusLabel",
    ]
    out = n[cols].rename(columns={"assignedOlHex1": "hex1", "assignedOlHex2": "hex2"})
    out["statusLabel"] = out["statusLabel"].astype(str)
    for c in ("type", "class", "superclass", "subclass", "dimorphism", "fruDsx", "entryNerve", "exitNerve", "receptorType"):
        out[c] = out[c].astype(object).where(out[c].notna(), None)
    return out


def load_edges(ids: np.ndarray, min_weight: int):
    t = ft.read_table(raw_path("weights"))
    pre = t.column("body_pre").to_numpy()
    post = t.column("body_post").to_numpy()
    w = t.column("weight").to_numpy()
    log(f"raw edge rows: {len(w):,}")
    del t

    def to_idx(bodies):
        pos = np.searchsorted(ids, bodies)
        pos_c = np.minimum(pos, len(ids) - 1)
        ok = ids[pos_c] == bodies
        return pos_c.astype(np.int32), ok

    pi, ok_pre = to_idx(pre)
    qi, ok_post = to_idx(post)
    mask = ok_pre & ok_post
    if min_weight > 1:
        mask &= w >= min_weight
    pi, qi, w = pi[mask], qi[mask], w[mask].astype(np.int32)
    log(f"retained edges: {len(w):,}  (min_weight={min_weight})")
    return pi, qi, w


def build_graph(pi, qi, w, sign, n, normalize: bool):
    signed = (w.astype(np.float32) * sign[pi].astype(np.float32))
    if normalize:
        denom = np.bincount(qi, weights=np.abs(w).astype(np.float64), minlength=n)
        denom[denom == 0] = 1.0
        signed = (signed / denom[qi]).astype(np.float32)

    # Order edges as out-edge CSR (rows = presynaptic). Use a position payload so the
    # raw counts stay aligned with the signed weights.
    e = len(w)
    pos = sparse.csr_matrix((np.arange(1, e + 1, dtype=np.float64), (pi, qi)), shape=(n, n))
    pos.sort_indices()
    if pos.nnz != e:
        raise RuntimeError(f"duplicate pre/post pairs collapsed: nnz {pos.nnz} != {e}")
    order = (pos.data.astype(np.int64) - 1)
    return {
        "indptr": pos.indptr.astype(np.int64),
        "indices": pos.indices.astype(np.int32),
        "data": signed[order],
        "counts": w[order],
    }


def build_eye(neurons: pd.DataFrame, pi, qi, w):
    """Photoreceptor -> optic-lobe column via strongest hex-annotated postsynaptic partner."""
    types = neurons["type"].fillna("").to_numpy()
    kind = np.full(len(neurons), -1, dtype=np.int8)
    kind[types == "R1-R6"] = 0
    kind[np.char.startswith(types.astype(str), "R7")] = 1
    kind[np.char.startswith(types.astype(str), "R8")] = 2
    pr = np.flatnonzero(kind >= 0)
    hex1 = neurons["hex1"].to_numpy(dtype=np.float32)
    hex2 = neurons["hex2"].to_numpy(dtype=np.float32)
    side = neurons["side"].to_numpy()
    has_hex = ~np.isnan(hex1)

    is_pr = np.zeros(len(neurons), dtype=bool)
    is_pr[pr] = True
    sel = is_pr[pi] & has_hex[qi]
    df = pd.DataFrame({
        "pre": pi[sel], "w": w[sel],
        "eye": np.where(side[qi[sel]] == "L", 0, np.where(side[qi[sel]] == "R", 1, -1)).astype(np.int8),
        "h1": hex1[qi[sel]].astype(np.int16), "h2": hex2[qi[sel]].astype(np.int16),
    })
    df = df[df["eye"] >= 0]
    g = df.groupby(["pre", "eye", "h1", "h2"], as_index=False)["w"].sum()
    best = g.sort_values("w", ascending=False).drop_duplicates("pre").set_index("pre")

    mapped = best.index.to_numpy()
    unmapped = np.setdiff1d(pr, mapped)
    log(f"photoreceptors: {len(pr):,} total, {len(mapped):,} mapped to a column, {len(unmapped):,} unmapped")

    eye = best.loc[mapped, "eye"].to_numpy(dtype=np.int8)
    h1 = best.loc[mapped, "h1"].to_numpy(dtype=np.int16)
    h2 = best.loc[mapped, "h2"].to_numpy(dtype=np.int16)

    # Axial hex -> Cartesian, then rank-normalize per eye over the distinct columns so a
    # regular partition of (u, v) gives roughly equal numbers of columns per board cell.
    # The orientation (which hex axis is anterior / dorsal) is an engineered registration.
    x = h1 + 0.5 * h2
    y = h2 * (np.sqrt(3) / 2)
    u = np.zeros(len(mapped), dtype=np.float32)
    v = np.zeros(len(mapped), dtype=np.float32)
    for e_ in (0, 1):
        m = eye == e_
        cols = np.unique(np.stack([h1[m], h2[m]], axis=1), axis=0)
        cx = cols[:, 0] + 0.5 * cols[:, 1]
        cy = cols[:, 1] * (np.sqrt(3) / 2)
        ux = np.unique(cx)
        uy = np.unique(cy)
        u[m] = (np.searchsorted(ux, x[m]) + 0.5) / len(ux)
        v[m] = (np.searchsorted(uy, y[m]) + 0.5) / len(uy)
    return {
        "pr_idx": mapped.astype(np.int32),
        "kind": kind[mapped],
        "eye": eye,
        "hex1": h1,
        "hex2": h2,
        "u": u,
        "v": v,
        "unmapped_idx": unmapped.astype(np.int32),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-weight", type=int, default=1, help="drop connections with fewer synapses (default 1 = keep all)")
    ap.add_argument("--no-normalize", action="store_true", help="keep raw signed synapse counts as weights")
    args = ap.parse_args(argv)
    ensure_dirs()
    t0 = time.time()

    lock_path = RAW / "source.lock.json"
    if not lock_path.exists():
        log("inputs not verified; run `python -m flysweeper.prepare` first")
        return 1
    lock = json.loads(lock_path.read_text())

    neurons = load_neurons()
    n = len(neurons)
    log(f"retained neurons: {n:,}")
    ids = neurons["bodyId"].to_numpy()
    sign = neurons["sign"].to_numpy()

    pi, qi, w = load_edges(ids, args.min_weight)
    graph = build_graph(pi, qi, w, sign, n, normalize=not args.no_normalize)
    eye = build_eye(neurons, pi, qi, w)

    in_deg = np.bincount(qi, minlength=n).astype(np.int32)
    out_deg = np.bincount(pi, minlength=n).astype(np.int32)
    in_syn = np.bincount(qi, weights=w, minlength=n).astype(np.int64)
    neurons["in_degree"] = in_deg
    neurons["out_degree"] = out_deg
    neurons["in_synapses"] = in_syn

    ft.write_feather(neurons, COMPILED / "neurons.feather")
    np.savez(COMPILED / "graph.npz", **graph)
    np.savez(COMPILED / "eye.npz", **eye)

    nt_counts = neurons["nt"].value_counts().to_dict()
    meta = {
        "dataset": "MaleCNS v1.0",
        "citation": "Berg et al., Sexual dimorphism in the complete connectome of the Drosophila male central nervous system, Cell 2026. doi:10.1016/j.cell.2026.08.015. CC BY 4.0.",
        "inputs": lock,
        "node_policy": "superclass not null and statusLabel != Glia",
        "edge_policy": f"all released edges between retained neurons, min_weight={args.min_weight}",
        "sign_policy": "presynaptic GABA/glutamate/histamine -> -1; acetylcholine/dopamine/octopamine/serotonin/unknown -> +1",
        "weight_policy": "synapse count x sign" + ("" if args.no_normalize else ", divided by postsynaptic total absolute synapse input"),
        "eye_policy": "photoreceptor assigned to the hex column of its strongest hex-annotated postsynaptic partner; (u,v) = per-eye rank of axial-hex Cartesian coordinates",
        "n_neurons": int(n),
        "n_edges": int(len(w)),
        "n_synapses": int(w.sum()),
        "nt_counts": {k: int(v) for k, v in nt_counts.items()},
        "n_sign_uncertain": int((neurons["nt"] == "unknown").sum()),
        "photoreceptors_total": int(len(eye["pr_idx"]) + len(eye["unmapped_idx"])),
        "photoreceptors_mapped": int(len(eye["pr_idx"])),
        "photoreceptors_by_kind": {k: int((eye["kind"] == i).sum()) for i, k in enumerate(["R1-R6", "R7", "R8"])},
        "compile_seconds": round(time.time() - t0, 1),
    }
    (COMPILED / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log(f"neurons {n:,}  edges {len(w):,}  synapses {int(w.sum()):,}  uncertain signs {meta['n_sign_uncertain']:,}")
    log(f"done in {meta['compile_seconds']} s -> {COMPILED}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
