"""Teach the frozen fly by training a linear readout to imitate the teacher.

The connectome and its dynamics stay exactly as in the `fly` condition.  What is trained is the
*readout*: which brain cells count as which button.  Protocol:

  collect   Play the full brain loop on training seeds.  Each turn the brain sees the real board
            (retina -> LIF -> ...), the teacher (teacher.py) names the correct action for the
            visible board and cursor, and we store (spike counts of the recorded population over
            the turn window, teacher action).  The cursor follows the teacher on most turns, so
            the brain sees realistic mid-game boards; on a fraction `--explore` of turns the
            fly's own pool decoder acts instead, so off-teacher states are represented too.
  train     Fit a multinomial logistic regression / ridge readout (standardized features,
            regularization chosen by grouped cross-validation over games), evaluate on games the
            readout never saw, and compare with: the majority class, the SAME classifier trained on
            a trivial board encoding (3x3 around the cursor, and the whole board) and on shuffled
            labels.  Save the readout to data/compiled/readout.npz for decoder.ReadoutDecoder.

Usage:
    python -m flysweeper.train_readout collect --turns 30000 --seed0 20000 --out outputs/training/run1
    python -m flysweeper.train_readout train --data outputs/training/run1 --groups dn vp cx mbon
    python -m flysweeper.train_readout all --turns 30000            # both steps
Then:
    python -m flysweeper.validate --games 30 --seed0 5000 --conditions fly fly-readout fly-blind random-walk random-click solver
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .agent import FlyPlayer, GameConfig
from .brain import Brain
from .decoder import READOUT_PATH, DecoderParams
from .encoder import EncoderParams
from .features import DEFAULT_GROUPS, FeatureExtractor, FeatureSpec
from .minesweeper import FLAG, HIDDEN, MINE
from .paths import OUTPUTS
from .sim import PRESETS, Params
from .teacher import ACTIONS, Teacher, TeacherParams

RECORD_GROUPS = ("dn", "vp", "cx", "mbon", "t4", "t5", "kc", "an")   # superset stored on disk
ACTION_ID = {a: i for i, a in enumerate(ACTIONS)}


def log(msg: str) -> None:
    print(f"[train_readout] {msg}", flush=True)


# ----------------------------------------------------------------------------- collection
def collect(args) -> Path:
    out = Path(args.out or (OUTPUTS / "training" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
    out.mkdir(parents=True, exist_ok=True)
    brain = Brain()
    cfg = GameConfig(rows=args.rows, cols=args.cols, mines=args.mines, turn_steps=args.turn_steps, max_turns=args.max_turns)
    spec = FeatureSpec(groups=tuple(args.record_groups))
    fx = FeatureExtractor(brain, spec)
    dec = DecoderParams(mode="readout", readout_path=args.explore_readout) if args.explore_readout else DecoderParams()
    enc = EncoderParams(route=args.route, egocentric=args.ego, ego_radius=args.ego_radius)
    player = FlyPlayer(brain, cfg, Params.preset(args.preset), enc, dec, sensory_input=args.sensory_input,
                       seed=args.seed0, features=fx)
    teacher = Teacher(TeacherParams(jump_distance=args.jump_distance))
    log(f"recording {fx.m:,} cells {fx.describe(brain)['groups']} per {cfg.turn_steps}-step turn; "
        f"explore {args.explore} ({'readout' if args.explore_readout else 'pool decoder'} acts); seeds from {args.seed0}")

    cap = args.turns + args.max_turns + 2 * (args.turns // 20 + 10)   # rows incl. one settle row per game
    X = np.zeros((cap, fx.m), dtype=np.uint8)
    y = np.full(cap, -1, dtype=np.int8)            # teacher action id, -1 for the settle row
    decided = np.full(cap, -1, dtype=np.int8)      # the brain's own decision (-1 hold/other)
    taken = np.full(cap, -1, dtype=np.int8)        # what actually happened
    seed_of = np.zeros(cap, dtype=np.int32)
    turn_of = np.full(cap, -1, dtype=np.int16)
    cursor_of = np.zeros((cap, 2), dtype=np.int8)
    board_of = np.zeros((cap, cfg.rows * cfg.cols), dtype=np.int8)
    explored = np.zeros(cap, dtype=bool)
    mode_of = np.zeros(cap, dtype="U16")
    state = {"n": 0, "seed": 0, "erng": None, "turns": 0}

    def act(player_, game, decided_action, rng):
        i = state["n"]
        turn = state["turns"]
        if turn == 0 and i < cap:   # the settle window (blank screen) as row "turn -1"
            X[i] = np.minimum(player_.features.prev, 255).astype(np.uint8)
            seed_of[i] = state["seed"]
            turn_of[i] = -1
            i += 1
        if i >= cap:
            state["n"] = i
            return decided_action
        visible = game.visible()
        action, target, mode = teacher.plan(visible, player_.cursor)
        X[i] = player_.features.counts()
        y[i] = ACTION_ID[action]
        decided[i] = ACTION_ID.get(decided_action, -1)
        seed_of[i] = state["seed"]
        turn_of[i] = turn
        cursor_of[i] = player_.cursor
        board_of[i] = visible.ravel()
        mode_of[i] = mode
        explore = state["erng"].random() < args.explore
        chosen = decided_action if explore else action
        explored[i] = explore
        taken[i] = ACTION_ID.get(chosen, -1)
        state["n"] = i + 1
        state["turns"] = turn + 1
        return chosen

    t0 = time.time()
    games = 0
    seed = args.seed0
    while (state["n"] - games) < args.turns and state["n"] < cap - args.max_turns - 1:
        state.update(seed=seed, erng=np.random.default_rng([seed, 7]), turns=0)
        rec = player.play("fly", seed, act=act)
        games += 1
        seed += 1
        done = state["n"] - games
        if games % 10 == 0 or done >= args.turns:
            rate = done / max(time.time() - t0, 1e-9)
            log(f"game {games:4d} seed {rec.seed}  {rec.outcome:7s} safe {rec.safe_revealed:2d}/{rec.total_safe} turns {rec.turns:3d} "
                f"overrides {rec.overrides:3d} | turns so far {done:,} ({rate:.1f}/s)")
    n = state["n"]
    sample = y[:n] >= 0
    counts = np.bincount(y[:n][sample].astype(np.int64), minlength=len(ACTIONS))
    meta = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(cfg), "preset": args.preset, "route": args.route, "sensory_input": args.sensory_input,
        "sim_params": asdict(player.sim.p), "encoder": asdict(enc), "decoder_mode": dec.mode,
        "seeds": [args.seed0, seed - 1], "games": games, "turns": int(sample.sum()), "rows": int(n),
        "explore": args.explore, "explore_policy": "readout" if args.explore_readout else "pool",
        "explored_turns": int(explored[:n].sum()), "jump_distance": args.jump_distance,
        "record_groups": list(args.record_groups), "n_cells": int(fx.m),
        "teacher_action_counts": {a: int(c) for a, c in zip(ACTIONS, counts)},
        "seconds": round(time.time() - t0, 1),
    }
    np.save(out / "X.npy", X[:n])
    np.savez(out / "meta.npz", y=y[:n], decided=decided[:n], taken=taken[:n], seed=seed_of[:n], turn=turn_of[:n],
             cursor=cursor_of[:n], board=board_of[:n], explored=explored[:n], mode=mode_of[:n],
             index=fx.index, group_id=fx.group_id, groups=np.array(args.record_groups), actions=np.array(ACTIONS))
    (out / "collect.json").write_text(json.dumps(meta, indent=1))
    log(f"saved {sample.sum():,} turns from {games} games (seeds {args.seed0}..{seed - 1}) to {out} in {meta['seconds']}s; "
        f"teacher actions {meta['teacher_action_counts']}")
    return out


# ----------------------------------------------------------------------------- features / controls
def board_features(boards: np.ndarray, cursors: np.ndarray, rows: int, cols: int, local: bool) -> np.ndarray:
    """Trivial board encodings for the controls.  local: one-hot 3x3 neighbourhood of the cursor
    (13 states incl. out-of-bounds) + cursor row/col one-hot.  full: one-hot whole board (12 states)
    + cursor cell one-hot."""
    n = len(boards)
    codes = boards.astype(np.int32) + 1          # -1..10 -> 0..11
    cursors = cursors.astype(np.int32)
    if local:
        k = 13
        out = np.zeros((n, 9 * k + rows + cols), dtype=np.float32)
        for j, (dr, dc) in enumerate([(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]):
            rr = cursors[:, 0].astype(np.int32) + dr
            cc = cursors[:, 1].astype(np.int32) + dc
            inb = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
            val = np.full(n, 12, dtype=np.int32)
            val[inb] = codes[inb, rr[inb] * cols + cc[inb]]
            out[np.arange(n), j * k + val] = 1.0
        out[np.arange(n), 9 * k + cursors[:, 0].astype(np.int32)] = 1.0
        out[np.arange(n), 9 * k + rows + cursors[:, 1].astype(np.int32)] = 1.0
        return out
    k = 12
    out = np.zeros((n, rows * cols * k + rows * cols), dtype=np.float32)
    cell = np.arange(rows * cols)
    for i in range(n):
        out[i, cell * k + codes[i]] = 1.0
        out[i, rows * cols * k + cursors[i, 0] * cols + cursors[i, 1]] = 1.0
    return out


class Standardizer:
    def __init__(self, X: np.ndarray):
        self.mean = X.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.std = X.std(axis=0, dtype=np.float64).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)


class RidgeSolver:
    """Closed-form ridge on standardized X (zero column means, so the intercept is the target mean).
    One Gram matrix per data set, one Cholesky per alpha, any number of target labelings."""

    def __init__(self, Xs: np.ndarray):
        from scipy import linalg
        self.linalg = linalg
        self.Xs = Xs.astype(np.float64)      # float32 Gram is not reliably PSD at this size
        n, d = Xs.shape
        self.primal = d <= n
        self.M = (self.Xs.T @ self.Xs) if self.primal else (self.Xs @ self.Xs.T)
        self.chol = None

    def factor(self, alpha: float) -> None:
        A = self.M.copy()
        A.flat[:: A.shape[0] + 1] += alpha
        self.chol = self.linalg.cho_factor(A, lower=True, overwrite_a=True, check_finite=False)

    def solve(self, y: np.ndarray, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
        n = len(y)
        Y = np.zeros((n, n_classes), dtype=np.float64)
        Y[np.arange(n), y] = 1.0
        ym = Y.mean(axis=0)
        Yc = Y - ym
        if self.primal:
            W = self.linalg.cho_solve(self.chol, self.Xs.T @ Yc, check_finite=False)   # (d, k)
        else:
            A = self.linalg.cho_solve(self.chol, Yc, check_finite=False)               # (n, k)
            W = self.Xs.T @ A
        return W.T.astype(np.float32), ym.astype(np.float32)


def ridge_multi(X: np.ndarray, targets: dict, seeds: np.ndarray, alphas: list[float], k: int, label: str) -> dict:
    """Grouped k-fold CV ridge for several labelings of the same rows (targets: name -> (y, n_classes)).
    Returns name -> model dict (W, b, mean, std, kind, hyper, cv_accuracy, cv)."""
    t0 = time.time()
    cv = {name: {a: [] for a in alphas} for name in targets}
    for f, (tr, va) in enumerate(group_folds(seeds, k)):
        sc = Standardizer(X[tr])
        Xt, Xv = sc(X[tr]), sc(X[va])
        solver = RidgeSolver(Xt)
        for a in alphas:
            solver.factor(a)
            for name, (y, nc) in targets.items():
                W, b = solver.solve(y[tr], nc)
                cv[name][a].append(accuracy(W, b, Xv, y[va]))
        log(f"{label}: fold {f + 1}/{k} done ({time.time() - t0:.0f}s)")
    best = {name: max(alphas, key=lambda a: np.mean(cv[name][a])) for name in targets}
    sc = Standardizer(X)
    solver = RidgeSolver(sc(X))
    out = {}
    for a in sorted(set(best.values())):
        solver.factor(a)
        for name in (n_ for n_ in targets if best[n_] == a):
            W, b = solver.solve(targets[name][0], targets[name][1])
            out[name] = {"W": W, "b": b, "mean": sc.mean, "std": sc.std, "kind": "ridge", "hyper": {"alpha": a},
                         "cv_accuracy": float(np.mean(cv[name][a])),
                         "cv": {"ridge": {"hyper": {"alpha": a}, "cv_accuracy": float(np.mean(cv[name][a])),
                                          "cv_all": {str(x): round(float(np.mean(v)), 4) for x, v in cv[name].items()}}}}
            log(f"{label}/{name}: ridge CV accuracy by alpha " + ", ".join(f"{x:g}: {np.mean(v):.3f}" for x, v in cv[name].items())
                + f" -> alpha {a:g}")
    log(f"{label}: ridge for {len(targets)} target(s) in {time.time() - t0:.0f}s")
    return out


def accuracy(W, b, Xs, y) -> float:
    pred = np.argmax(Xs @ W.T + b, axis=1)
    return float((pred == y).mean())


def balanced_accuracy(pred, y, n_classes) -> float:
    rec = []
    for k in range(n_classes):
        m = y == k
        if m.any():
            rec.append(float((pred[m] == k).mean()))
    return float(np.mean(rec))


def group_folds(seeds: np.ndarray, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    uniq = np.unique(seeds)
    folds = []
    for f in range(k):
        val_seeds = uniq[f::k]
        val = np.isin(seeds, val_seeds)
        folds.append((np.flatnonzero(~val), np.flatnonzero(val)))
    return folds


def logreg_cv(X: np.ndarray, y: np.ndarray, seeds: np.ndarray, n_classes: int, Cs: list[float], k: int, label: str) -> dict:
    """Multinomial logistic regression (sklearn lbfgs) on standardized X; C chosen by grouped k-fold
    CV on the given (training) rows, then refit on all of them."""
    from sklearn.linear_model import LogisticRegression
    t0 = time.time()
    cv = {C: [] for C in Cs}
    for tr, va in group_folds(seeds, k):
        sc = Standardizer(X[tr])
        Xt, Xv = sc(X[tr]), sc(X[va])
        for C in Cs:
            clf = LogisticRegression(C=C, max_iter=300, tol=1e-3)
            clf.fit(Xt, y[tr])
            cv[C].append(float((clf.predict(Xv) == y[va]).mean()))
    best_C = max(Cs, key=lambda C: np.mean(cv[C]))
    log(f"{label}: logreg CV accuracy by C " + ", ".join(f"{C:g}: {np.mean(v):.3f}" for C, v in cv.items()) + f" -> C {best_C:g}")
    sc = Standardizer(X)
    clf = LogisticRegression(C=best_C, max_iter=500, tol=1e-4)
    clf.fit(sc(X), y)
    W = np.zeros((n_classes, X.shape[1]), dtype=np.float32)
    b = np.zeros(n_classes, dtype=np.float32)
    W[clf.classes_] = clf.coef_.astype(np.float32)
    b[clf.classes_] = clf.intercept_.astype(np.float32)
    log(f"{label}: logreg done in {time.time() - t0:.0f}s")
    return {"W": W, "b": b, "mean": sc.mean, "std": sc.std, "kind": "logreg", "hyper": {"C": best_C},
            "cv_accuracy": float(np.mean(cv[best_C])),
            "cv": {"logreg": {"hyper": {"C": best_C}, "cv_accuracy": float(np.mean(cv[best_C])),
                              "cv_all": {str(C): round(float(np.mean(v)), 4) for C, v in cv.items()}}}}


def train_linear(X: np.ndarray, y: np.ndarray, seeds: np.ndarray, classifier: str, n_classes: int,
                 alphas: list[float], Cs: list[float], k: int, label: str) -> dict:
    """One labeling: ridge and/or logreg with grouped CV; `auto` keeps the better CV accuracy."""
    models = []
    if classifier in ("ridge", "auto"):
        models.append(ridge_multi(X, {"y": (y, n_classes)}, seeds, alphas, k, label)["y"])
    if classifier in ("logreg", "auto"):
        models.append(logreg_cv(X, y, seeds, n_classes, Cs, k, label))
    best = max(models, key=lambda m: m["cv_accuracy"])
    best["cv"] = {kk: vv for m in models for kk, vv in m["cv"].items()}
    log(f"{label}: chose {best['kind']} {best['hyper']} (CV {best['cv_accuracy']:.3f})")
    return best


def evaluate(model: dict, X: np.ndarray, y: np.ndarray, n_classes: int) -> dict:
    Xs = ((X - model["mean"]) / model["std"]).astype(np.float32)
    pred = np.argmax(Xs @ model["W"].T + model["b"], axis=1)
    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(conf, (y, pred), 1)
    return {
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": balanced_accuracy(pred, y, n_classes),
        "per_class_recall": {ACTIONS[k]: (round(float(conf[k, k] / conf[k].sum()), 3) if conf[k].sum() else None) for k in range(n_classes)},
        "predicted_mix": {ACTIONS[k]: int((pred == k).sum()) for k in range(n_classes)},
        "confusion": conf.tolist(),
        "n": int(len(y)),
    }


# ----------------------------------------------------------------------------- training
def load_data(data_dirs: list[Path]):
    """Load one or more `collect` outputs (same recorded population) and concatenate them."""
    Xs, metas, infos = [], [], []
    for d in data_dirs:
        Xs.append(np.load(d / "X.npy", mmap_mode="r"))
        m = np.load(d / "meta.npz", allow_pickle=False)
        metas.append({k: m[k] for k in m.files})
        infos.append(json.loads((d / "collect.json").read_text()))
    for m in metas[1:]:
        if not np.array_equal(m["index"], metas[0]["index"]):
            raise ValueError("data sets record different populations; collect them with the same --record-groups")
    if len(Xs) == 1:
        X, meta, info = Xs[0], metas[0], infos[0]
    else:
        X = np.concatenate([np.asarray(x) for x in Xs], axis=0)
        per_row = ("y", "decided", "taken", "seed", "turn", "cursor", "board", "explored", "mode")
        meta = {k: np.concatenate([m[k] for m in metas]) for k in per_row}
        meta.update({k: metas[0][k] for k in metas[0] if k not in per_row})
        info = dict(infos[0])
        info["turns"] = int(sum(i["turns"] for i in infos))
        info["games"] = int(sum(i["games"] for i in infos))
        info["explored_turns"] = int(sum(i["explored_turns"] for i in infos))
        info["seeds"] = [min(i["seeds"][0] for i in infos), max(i["seeds"][1] for i in infos)]
        info["seed_ranges"] = [i["seeds"] for i in infos]
        info["explore_policy"] = "+".join(sorted({i["explore_policy"] for i in infos}))
        info["sources"] = [str(d) for d in data_dirs]
    return X, meta, info


def diagnostic_targets(boards: np.ndarray, cursors: np.ndarray, cols_b: int) -> dict:
    """Simpler labelings of the same turns: is the information there at all?"""
    cursors = cursors.astype(np.int64)
    under = boards[np.arange(len(boards)), cursors[:, 0] * cols_b + cursors[:, 1]]
    return {
        "cursor_row": (cursors[:, 0], int(cursors[:, 0].max()) + 1),
        "cursor_col": (cursors[:, 1], int(cursors[:, 1].max()) + 1),
        "cell_under_cursor_hidden": ((under == HIDDEN).astype(np.int64), 2),
        "cursor_left_half": ((cursors[:, 1] < cols_b // 2).astype(np.int64), 2),
    }


def summarize_diagnostic(model: dict, X_te: np.ndarray, t_te: np.ndarray, t_tr: np.ndarray, k: int) -> dict:
    Xs = ((X_te - model["mean"]) / model["std"]).astype(np.float32)
    pred = np.argmax(Xs @ model["W"].T + model["b"], axis=1)
    maj = int(np.bincount(t_tr, minlength=k).argmax())
    return {"classes": k, "held_out_accuracy": round(float((pred == t_te).mean()), 3),
            "majority": round(float((t_te == maj).mean()), 3), "alpha": model["hyper"]["alpha"]}


def select_columns(meta: dict, groups: list[str]) -> np.ndarray:
    names = [str(g) for g in meta["groups"]]
    ids = []
    for g in groups:
        if g not in names:
            raise KeyError(f"group {g!r} was not recorded; recorded groups: {names}")
        ids.append(names.index(g))
    return np.flatnonzero(np.isin(meta["group_id"], ids))


def build_matrix(X, meta, cols: np.ndarray, windows: int):
    """Rows = teacher-labelled turns; features = counts of selected cells for the current window
    (and the previous window: the preceding row of the same game, the settle row for turn 0)."""
    y = meta["y"]
    rows = np.flatnonzero(y >= 0)
    Xc = np.asarray(X[:, cols], dtype=np.uint8) if len(cols) < X.shape[1] else np.asarray(X)
    cur = Xc[rows].astype(np.float32)
    if windows == 2:
        prev_rows = rows - 1
        ok = (prev_rows >= 0) & (meta["seed"][np.maximum(prev_rows, 0)] == meta["seed"][rows])
        prev = np.zeros_like(cur)
        prev[ok] = Xc[prev_rows[ok]].astype(np.float32)
        cur = np.concatenate([cur, prev], axis=1)
    return cur, y[rows].astype(np.int64), rows


def train(args) -> Path:
    data_dirs = [Path(d) for d in (args.data if isinstance(args.data, list) else [args.data])]
    data_dir = data_dirs[0]
    X, meta, info = load_data(data_dirs)
    cfg = info["config"]
    rows_b, cols_b = cfg["rows"], cfg["cols"]
    groups = list(args.groups)
    cols = select_columns(meta, groups)
    n_classes = len(ACTIONS)
    Xf, y, rows = build_matrix(X, meta, cols, args.windows)
    seeds = meta["seed"][rows]
    boards = meta["board"][rows]
    cursors = meta["cursor"][rows]
    uniq = np.unique(seeds)
    test_seeds = uniq[args.test_every - 1::args.test_every]
    is_test = np.isin(seeds, test_seeds)
    tr, te = np.flatnonzero(~is_test), np.flatnonzero(is_test)
    log(f"data {data_dir}: {len(y):,} turns from {len(uniq)} games (seeds {uniq.min()}..{uniq.max()}); "
        f"features {Xf.shape[1]:,} = {len(cols):,} cells x {args.windows} window(s) from groups {groups}; "
        f"train {len(tr):,} turns / {len(uniq) - len(test_seeds)} games, held-out {len(te):,} turns / {len(test_seeds)} games")
    majority = int(np.bincount(y[tr], minlength=n_classes).argmax())
    report = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": ", ".join(str(d) for d in data_dirs), "collect": info, "groups": groups, "n_cells": int(len(cols)), "windows": args.windows,
        "dim": int(Xf.shape[1]), "train_turns": int(len(tr)), "test_turns": int(len(te)),
        "train_games": int(len(uniq) - len(test_seeds)), "test_games": int(len(test_seeds)),
        "test_seeds": test_seeds.tolist(), "cv_folds": args.folds,
        "label_mix_test": {a: round(float((y[te] == k).mean()), 3) for k, a in enumerate(ACTIONS)},
        "majority_class": ACTIONS[majority],
        "controls": {"majority": {"accuracy": float((y[te] == majority).mean()),
                                  "balanced_accuracy": float(1.0 / n_classes)}},
    }
    alphas = [float(a) for a in args.alphas]
    Cs = [float(c) for c in args.Cs]

    # Ridge: one Gram / Cholesky per fold shared by every labeling (teacher action, shuffled
    # labels, diagnostics).  Logreg (if requested) only for the teacher action.
    targets = {}
    ridge_models = {}
    if args.classifier in ("ridge", "auto"):
        targets["brain"] = (y[tr], n_classes)
        if not args.skip_controls:
            y_shuf = np.random.default_rng(0).permutation(y[tr])
            targets["shuffled"] = (y_shuf, n_classes)
        diag = diagnostic_targets(boards, cursors, cols_b) if args.diagnostics else {}
        for name, (t, k) in diag.items():
            targets["diag:" + name] = (t[tr], k)
        ridge_models = ridge_multi(Xf[tr], targets, seeds[tr], alphas, args.folds, "ridge")
    candidates = []
    if "brain" in ridge_models:
        candidates.append(ridge_models["brain"])
    if args.classifier in ("logreg", "auto"):
        candidates.append(logreg_cv(Xf[tr], y[tr], seeds[tr], n_classes, Cs, args.folds, "brain"))
    brain_model = max(candidates, key=lambda m: m["cv_accuracy"])
    brain_model["cv"] = {kk: vv for m in candidates for kk, vv in m["cv"].items()}
    log(f"brain readout: chose {brain_model['kind']} {brain_model['hyper']} (CV {brain_model['cv_accuracy']:.3f})")
    report["brain"] = {k: brain_model[k] for k in ("kind", "hyper", "cv_accuracy", "cv")}
    report["brain"]["held_out"] = evaluate(brain_model, Xf[te], y[te], n_classes)
    log(f"brain readout held-out accuracy {report['brain']['held_out']['accuracy']:.3f} "
        f"(balanced {report['brain']['held_out']['balanced_accuracy']:.3f}); majority {report['controls']['majority']['accuracy']:.3f}")
    if args.diagnostics and ridge_models:
        report["diagnostics"] = {}
        for name, (t, k) in diag.items():
            report["diagnostics"][name] = summarize_diagnostic(ridge_models["diag:" + name], Xf[te], t[te], t[tr], k)
            log(f"diagnostic {name}: held-out accuracy {report['diagnostics'][name]['held_out_accuracy']:.3f} vs majority {report['diagnostics'][name]['majority']:.3f}")
    if not args.skip_controls:
        if "shuffled" in ridge_models:
            shuf = ridge_models["shuffled"]
        else:
            y_shuf = np.random.default_rng(0).permutation(y[tr])
            shuf = logreg_cv(Xf[tr], y_shuf, seeds[tr], n_classes, Cs, args.folds, "shuffled-labels")
        report["controls"]["shuffled_labels"] = evaluate(shuf, Xf[te], y[te], n_classes)
        report["controls"]["shuffled_labels"]["cv_accuracy_on_shuffled"] = shuf["cv_accuracy"]
        report["controls"]["shuffled_labels"]["kind"] = shuf["kind"]
        log(f"shuffled-label control held-out accuracy {report['controls']['shuffled_labels']['accuracy']:.3f}")
        for name, local in (("board_3x3", True), ("board_full", False)):
            B = board_features(boards, cursors, rows_b, cols_b, local)
            bm = train_linear(B[tr], y[tr], seeds[tr], "logreg", n_classes, alphas, [0.01, 0.1, 1.0, 10.0], args.folds, name)
            report["controls"][name] = evaluate(bm, B[te], y[te], n_classes)
            report["controls"][name]["dim"] = int(B.shape[1])
            report["controls"][name]["kind"] = bm["kind"]
            report["controls"][name]["hyper"] = bm["hyper"]
            log(f"{name} control ({B.shape[1]} features) held-out accuracy {report['controls'][name]['accuracy']:.3f}")
        # what does the brain readout know that the local board does not, and vice versa?
        report["brain"]["held_out_by_teacher_mode"] = {}
        modes = meta["mode"][rows]
        Xs_te = ((Xf[te] - brain_model["mean"]) / brain_model["std"]).astype(np.float32)
        pred = np.argmax(Xs_te @ brain_model["W"].T + brain_model["b"], axis=1)
        for mo in np.unique(modes[te]):
            mm = modes[te] == mo
            report["brain"]["held_out_by_teacher_mode"][str(mo)] = {"n": int(mm.sum()), "accuracy": round(float((pred[mm] == y[te][mm]).mean()), 3)}
        expl = meta["explored"][rows][te]
        report["brain"]["held_out_on_explored_turns"] = {"n": int(expl.sum()), "accuracy": round(float((pred[expl] == y[te][expl]).mean()), 3) if expl.any() else None}

    out_path = Path(args.out or READOUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_meta = {
        "generated": report["generated"], "groups": groups, "n_cells": int(len(cols)), "windows": args.windows,
        "classifier": brain_model["kind"], "hyper": brain_model["hyper"], "cv_accuracy": brain_model["cv_accuracy"],
        "held_out_accuracy": report["brain"]["held_out"]["accuracy"],
        "held_out_balanced_accuracy": report["brain"]["held_out"]["balanced_accuracy"],
        "controls": {k: {kk: v[kk] for kk in ("accuracy", "balanced_accuracy") if kk in v} for k, v in report["controls"].items()},
        "train_seeds": info["seeds"], "test_seeds_within_training_range": test_seeds.tolist(), "n_turns": int(len(y)),
        "data": report["data"], "preset": info["preset"], "route": info["route"], "config": cfg,
        "encoder": info.get("encoder", {}), "explore": info.get("explore"), "explore_policy": info.get("explore_policy"),
    }
    if args.refit_all:
        # refit on train + held-out games with the chosen hyperparameter (more data for deployment;
        # the held-out numbers above are still the honest ones)
        final = train_linear(Xf, y, seeds, brain_model["kind"], n_classes, [brain_model["hyper"].get("alpha", alphas[0])],
                             [brain_model["hyper"].get("C", Cs[0])], args.folds, "refit-all")
        save_meta["refit_on_all_training_data"] = True
    else:
        final = brain_model
        save_meta["refit_on_all_training_data"] = False
    np.savez(out_path, W=final["W"].astype(np.float32), b=final["b"].astype(np.float32),
             mean=final["mean"].astype(np.float32), std=final["std"].astype(np.float32),
             index=meta["index"][cols].astype(np.int32), windows=np.int64(args.windows),
             actions=np.array(ACTIONS), meta=np.array(json.dumps(save_meta)))
    report["saved"] = str(out_path)
    rep_dir = Path(args.report_dir or data_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or ("_".join(groups) + f"_w{args.windows}")
    (rep_dir / f"readout_report_{tag}.json").write_text(json.dumps(report, indent=1, default=str))
    (rep_dir / f"readout_report_{tag}.md").write_text(report_markdown(report))
    log(f"saved readout to {out_path}; report {rep_dir / f'readout_report_{tag}.md'}")
    return out_path


def report_markdown(rep: dict) -> str:
    c = rep["controls"]
    lines = [
        "# Readout training report",
        "",
        f"Generated {rep['generated']}. Data `{rep['data']}`: {rep['collect']['turns']:,} teacher-labelled turns from "
        f"{rep['collect']['games']} games, seeds {rep['collect']['seeds'][0]}..{rep['collect']['seeds'][1]}, "
        f"explore fraction {rep['collect']['explore']} ({rep['collect']['explore_policy']} decoder acted on {rep['collect']['explored_turns']:,} turns).",
        "",
        f"Features: {rep['n_cells']:,} cells from groups {rep['groups']} x {rep['windows']} window(s) = {rep['dim']:,} spike counts. "
        f"Split by game: {rep['train_games']} training games ({rep['train_turns']:,} turns), {rep['test_games']} held-out games ({rep['test_turns']:,} turns). "
        f"Regularization by {rep['cv_folds']}-fold grouped CV on the training games.",
        "",
        "| model | features | held-out accuracy | balanced accuracy |",
        "|---|---|---|---|",
        f"| **brain readout** ({rep['brain']['kind']} {rep['brain']['hyper']}) | {rep['dim']:,} | **{rep['brain']['held_out']['accuracy']:.3f}** | {rep['brain']['held_out']['balanced_accuracy']:.3f} |",
        f"| majority class (`{rep['majority_class']}`) | 0 | {c['majority']['accuracy']:.3f} | {c['majority']['balanced_accuracy']:.3f} |",
    ]
    if "shuffled_labels" in c:
        lines.append(f"| brain readout, shuffled labels | {rep['dim']:,} | {c['shuffled_labels']['accuracy']:.3f} | {c['shuffled_labels']['balanced_accuracy']:.3f} |")
    if "board_3x3" in c:
        lines.append(f"| board-only control: 3x3 around cursor + cursor pos ({c['board_3x3']['kind']}) | {c['board_3x3']['dim']} | {c['board_3x3']['accuracy']:.3f} | {c['board_3x3']['balanced_accuracy']:.3f} |")
    if "board_full" in c:
        lines.append(f"| board-only control: whole board + cursor ({c['board_full']['kind']}) | {c['board_full']['dim']} | {c['board_full']['accuracy']:.3f} | {c['board_full']['balanced_accuracy']:.3f} |")
    lines += ["", f"Held-out label mix: {rep['label_mix_test']}", "",
              f"Brain readout per-class recall: {rep['brain']['held_out']['per_class_recall']}",
              f"Brain readout predicted mix: {rep['brain']['held_out']['predicted_mix']}", ""]
    if "held_out_by_teacher_mode" in rep["brain"]:
        lines.append(f"Accuracy by teacher mode: {rep['brain']['held_out_by_teacher_mode']}")
        lines.append(f"Accuracy on explored (off-teacher) turns: {rep['brain']['held_out_on_explored_turns']}")
    lines += ["", "Confusion (rows = teacher action, cols = predicted; order " + ", ".join(ACTIONS) + "):", ""]
    for a, row in zip(ACTIONS, rep["brain"]["held_out"]["confusion"]):
        lines.append(f"    {a:7s} " + " ".join(f"{v:6d}" for v in row))
    if rep.get("diagnostics"):
        lines += ["", "What else the same brain features decode linearly (held-out accuracy vs majority):", ""]
        for k, v in rep["diagnostics"].items():
            lines.append(f"- {k} ({v['classes']} classes): {v['held_out_accuracy']:.3f} vs {v['majority']:.3f}")
    lines += ["", f"Saved readout: `{rep.get('saved', '')}`", ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- CLI
def add_collect_args(ap):
    ap.add_argument("--turns", type=int, default=30000, help="stop after this many teacher-labelled turns (games are finished)")
    ap.add_argument("--seed0", type=int, default=20000, help="first training game seed (validation uses 5000+ / 6000+)")
    ap.add_argument("--explore", type=float, default=0.25, help="fraction of turns where the fly's own decoder acts")
    ap.add_argument("--explore-readout", default=None, help="path of a trained readout to act on explore turns (DAgger-style) instead of the pool decoder")
    ap.add_argument("--record-groups", nargs="+", default=list(RECORD_GROUPS))
    ap.add_argument("--jump-distance", type=int, default=5)
    ap.add_argument("--preset", default="flyai", choices=sorted(PRESETS))
    ap.add_argument("--route", default="lamina", choices=["retina", "lamina"])
    ap.add_argument("--ego", action="store_true", help="egocentric encoder: the eyes are centred on the cursor (validate must use --ego too)")
    ap.add_argument("--ego-radius", type=int, default=4)
    ap.add_argument("--sensory-input", action="store_true")
    ap.add_argument("--turn-steps", type=int, default=15)
    ap.add_argument("--max-turns", type=int, default=400)
    ap.add_argument("--rows", type=int, default=9)
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--mines", type=int, default=10)


def add_train_args(ap, need_data: bool):
    if need_data:
        ap.add_argument("--data", required=True, nargs="+", help="director(ies) written by `collect`")
    ap.add_argument("--groups", nargs="+", default=list(DEFAULT_GROUPS))
    ap.add_argument("--diagnostics", action="store_true", help="also decode cursor position / cell state from the same features")
    ap.add_argument("--windows", type=int, default=1, choices=[1, 2])
    ap.add_argument("--classifier", default="auto", choices=["auto", "ridge", "logreg"])
    ap.add_argument("--alphas", nargs="+", default=[1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
    ap.add_argument("--Cs", nargs="+", default=[1e-4, 1e-3, 1e-2])
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--test-every", type=int, default=5, help="every k-th game (by seed order) is held out")
    ap.add_argument("--skip-controls", action="store_true")
    ap.add_argument("--refit-all", action="store_true", help="refit the saved readout on all training games (held-out numbers stay from the split)")
    ap.add_argument("--report-dir", default=None)
    ap.add_argument("--tag", default=None)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="play teacher-driven games and record (features, teacher action)")
    add_collect_args(c)
    c.add_argument("--out", default=None)
    t = sub.add_parser("train", help="fit the readout and the controls from a collected data set")
    add_train_args(t, need_data=True)
    t.add_argument("--out", default=None, help=f"readout file (default {READOUT_PATH})")
    a = sub.add_parser("all", help="collect then train")
    add_collect_args(a)
    add_train_args(a, need_data=False)
    a.add_argument("--out", default=None, help="data directory for collect")
    a.add_argument("--readout-out", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "collect":
        collect(args)
    elif args.cmd == "train":
        train(args)
    else:
        data_dir = collect(args)
        args.data = [str(data_dir)]
        args.out = args.readout_out
        train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
