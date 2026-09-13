"""Evaluate KC->MBON weight checkpoints: per-action agreement with the teacher and held-out games.

    python -m flysweeper.mb_eval outputs/mb/warm2/weights_*.npz --games 30 --seed0 5000 --out outputs/mb/warm2/eval.md
    python -m flysweeper.mb_eval best.npz --margin-sweep 0 0.1 0.2 0.3 --seed0 15000 --games 20   # tune the reveal margin on TRAINING seeds

For every file: (1) argmax agreement with the teacher on teacher-driven boards (the teacher moves
the cursor, the fly only votes; seeds --agree-seed0.., --agree-turns turns) with the full confusion
matrix, and (2) `fly-mb` games with exploration off (argmax) on --seed0..+games, reporting win rate
and mean safe cells.  Everything else (visual input, 15-step turns, KC changes) is as in validate.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .agent import FlyPlayer, GameConfig
from .brain import Brain
from .decoder import DecoderParams
from .encoder import EncoderParams
from .mb_policy import MBPolicy
from .oracle import ACTIONS
from .sim import Params
from .teacher import Teacher


def agreement(player: FlyPlayer, mb: MBPolicy, seed0: int, turns: int) -> dict:
    teacher = Teacher()
    conf = np.zeros((6, 6), dtype=int)

    def act(p, game, decided, rng):
        t = teacher.act(game.visible(), p.cursor, rng)
        conf[ACTIONS.index(t), ACTIONS.index(decided)] += 1
        return t

    seed = seed0
    while conf.sum() < turns:
        player.play("fly-mb", seed, act=act)
        seed += 1
    recall = {a: round(float(conf[i, i] / max(1, conf[i].sum())), 3) for i, a in enumerate(ACTIONS)}
    precision = {a: round(float(conf[i, i] / max(1, conf[:, i].sum())), 3) for i, a in enumerate(ACTIONS)}
    k = ACTIONS.index("reveal")
    return {
        "turns": int(conf.sum()), "accuracy": round(float(np.trace(conf) / conf.sum()), 3),
        "recall": recall, "precision": precision, "confusion": conf.tolist(),
        "reveal_false_positive_rate": round(float(conf[:, k].sum() - conf[k, k]) / max(1, conf.sum() - conf[k].sum()), 3),
    }


def games(player: FlyPlayer, seed0: int, n: int) -> dict:
    recs = [player.play("fly-mb", s) for s in range(seed0, seed0 + n)]
    safe = np.array([r.safe_revealed for r in recs], dtype=float)
    return {
        "games": n, "win_rate": round(sum(r.outcome == "won" for r in recs) / n, 3), "wins": int(sum(r.outcome == "won" for r in recs)),
        "mean_safe": round(float(safe.mean()), 2), "sem_safe": round(float(safe.std(ddof=1) / np.sqrt(n)), 2) if n > 1 else 0.0,
        "mean_turns": round(float(np.mean([r.turns for r in recs])), 1),
        "mean_reveals": round(float(np.mean([r.reveals for r in recs])), 1),
        "timeouts": int(sum(r.outcome == "timeout" for r in recs)),
    }


def evaluate_file(brain: Brain, path: Path, args, margin: float) -> dict:
    cfg = GameConfig(rows=args.rows, cols=args.cols, mines=args.mines, turn_steps=args.turn_steps, max_turns=args.max_turns)
    player = FlyPlayer(brain, cfg, Params.preset("flyai"), EncoderParams(route="lamina"), DecoderParams(), seed=args.seed0,
                       mb_weights_path=str(path))
    mb = player.enable_mb()
    mb.learning = False
    mb.p.temperature = 0.0
    mb.p.reveal_margin = margin
    out = {"file": str(path), "games_trained": mb.games_trained, "margin": margin, "pool_ratio": mb.pool_ratios()}
    w = np.abs(mb.w_learned) / mb.w0
    out["weights"] = {"edges": int(len(w)), "changed": int((np.abs(np.abs(mb.w_learned) - mb.w0) > 1e-7).sum()),
                      "mean_ratio": round(float(w.mean()), 3), "at_floor": int((w <= mb.p.w_min_ratio + 1e-6).sum()),
                      "at_ceiling": int((w >= mb.p.w_max_ratio - 1e-6).sum())}
    if args.agree_turns > 0:
        out["agreement"] = agreement(player, mb, args.agree_seed0, args.agree_turns)
    if args.games > 0:
        out["heldout"] = games(player, args.seed0, args.games)
    if args.install:
        mb.save(args.install, {"installed_from": str(path), "reveal_margin": margin, "eval": {k: v for k, v in out.items() if k != "agreement"}})
        out["installed"] = args.install
    return out


def table(results: list[dict]) -> str:
    cols = ["file", "games_trained", "margin", "agree", "reveal_recall", "reveal_FPR", "win_rate", "mean_safe", "sem", "turns", "changed", "mean_ratio"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in results:
        a, h, w = r.get("agreement", {}), r.get("heldout", {}), r["weights"]
        lines.append("| " + " | ".join(str(x) for x in [
            Path(r["file"]).parent.name + "/" + Path(r["file"]).name, r["games_trained"], r["margin"], a.get("accuracy", ""),
            a.get("recall", {}).get("reveal", ""), a.get("reveal_false_positive_rate", ""), h.get("win_rate", ""), h.get("mean_safe", ""),
            h.get("sem_safe", ""), h.get("mean_turns", ""), w["changed"], w["mean_ratio"]]) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=5000)
    ap.add_argument("--agree-turns", type=int, default=600)
    ap.add_argument("--agree-seed0", type=int, default=80000)
    ap.add_argument("--margin-sweep", type=float, nargs="*", default=None, help="evaluate these reveal margins (default: the file's)")
    ap.add_argument("--turn-steps", type=int, default=15)
    ap.add_argument("--rows", type=int, default=9)
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--mines", type=int, default=10)
    ap.add_argument("--max-turns", type=int, default=400)
    ap.add_argument("--out", default=None, help="markdown table (json alongside)")
    ap.add_argument("--install", default=None, help="save the (single) evaluated policy with its reveal margin to this path, e.g. data/compiled/mb_weights.npz")
    args = ap.parse_args(argv)
    brain = Brain()
    results = []
    t0 = time.time()
    for f in args.files:
        margins = args.margin_sweep if args.margin_sweep is not None else [MBPolicy.params_from_file(f).reveal_margin]
        for m in margins:
            r = evaluate_file(brain, Path(f), args, m)
            results.append(r)
            a, h = r.get("agreement", {}), r.get("heldout", {})
            print(f"[mb_eval] {f} margin {m}: agree {a.get('accuracy')} recall {a.get('recall')} reveal-FPR {a.get('reveal_false_positive_rate')} | "
                  f"held-out win {h.get('win_rate')} ({h.get('wins')}) safe {h.get('mean_safe')}+-{h.get('sem_safe')} turns {h.get('mean_turns')} | "
                  f"{r['weights']} [{time.time() - t0:.0f}s]", flush=True)
    md = table(results)
    print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md + "\n")
        Path(args.out).with_suffix(".json").write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
