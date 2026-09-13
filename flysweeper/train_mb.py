"""Train the fly's mushroom body to play Minesweeper (KC->MBON synapses only; see mb_policy.py).

    NUMBA_NUM_THREADS=10 python -m flysweeper.train_mb --games 2000 --seed0 10000 --out outputs/mb/rl
    NUMBA_NUM_THREADS=10 python -m flysweeper.train_mb --games 2000 --warmstart-games 300 --out outputs/mb/warm
    NUMBA_NUM_THREADS=10 python -m flysweeper.train_mb --games 500 --shuffle-reward --out outputs/mb/shuffled

Per game the trainer writes one JSON line to <out>/games.jsonl (outcome, safe cells, turns, reward,
|dw|, edges changed, per-pool mean weight ratio, temperature), a learning curve per 100 games to
<out>/curve.json + curve.md, and a KC->MBON weight checkpoint every --checkpoint-every games to
<out>/weights_<game>.npz; the final weights go to <out>/weights_final.npz and, unless
--no-install, to data/compiled/mb_weights.npz (what `fly-mb` and the spectator load).

Exploration: softmax over the 6 MBON pools' per-cell spike counts with temperature annealed
linearly from --temperature to --temperature-min over --anneal-games games.  The visual board input
(lamina route) stays on, as in every other fly condition.
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
from .decoder import DecoderParams
from .encoder import EncoderParams
from .mb_policy import DEFAULT_WEIGHTS, GROUP_POOLS, SINGLE_POOLS, MBParams
from .paths import OUTPUTS
from .sim import Params
from .teacher import Teacher


def curve_rows(games: list[dict], block: int = 100) -> list[dict]:
    rows = []
    for i in range(0, len(games), block):
        g = games[i:i + block]
        rows.append({
            "games": f"{g[0]['game']}-{g[-1]['game']}",
            "n": len(g),
            "win_rate": round(sum(x["outcome"] == "won" for x in g) / len(g), 3),
            "mean_safe": round(float(np.mean([x["safe_revealed"] for x in g])), 2),
            "mean_turns": round(float(np.mean([x["turns"] for x in g])), 1),
            "mean_reward": round(float(np.mean([x["reward"] for x in g])), 3),
            "mean_abs_dw": round(float(np.mean([x["abs_delta"] for x in g])), 4),
            "temperature": round(float(np.mean([x["temperature"] for x in g])), 3),
            "mean_ratio": round(float(np.mean([x["mean_weight_ratio"] for x in g])), 4),
            "teacher_agreement": round(float(np.mean([x.get("teacher_agreement", np.nan) for x in g])), 3),
        })
    return rows


def curve_markdown(rows: list[dict], title: str) -> str:
    cols = ["games", "n", "win_rate", "mean_safe", "mean_turns", "mean_reward", "teacher_agreement", "mean_abs_dw", "mean_ratio", "temperature"]
    out = [f"# {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--seed0", type=int, default=10000)
    ap.add_argument("--out", type=Path, default=OUTPUTS / "mb" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    ap.add_argument("--checkpoint-every", type=int, default=250)
    ap.add_argument("--warmstart-games", type=int, default=0, help="first N games: teacher-labelled supervised updates (Ramp-style)")
    ap.add_argument("--warmstart-follow", action="store_true", help="during warm start the cursor follows the teacher (else the fly's own choice)")
    ap.add_argument("--shuffle-reward", action="store_true", help="control: reward sign randomised per event")
    ap.add_argument("--init", default=None, help="start from these KC->MBON weights (npz)")
    ap.add_argument("--no-install", action="store_true", help="do not copy the final weights to data/compiled/mb_weights.npz")
    ap.add_argument("--eta", type=float, default=0.02)
    ap.add_argument("--other-credit", type=float, default=0.2)
    ap.add_argument("--trace-decay", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--temperature-min", type=float, default=0.2)
    ap.add_argument("--anneal-games", type=int, default=1500)
    ap.add_argument("--kc-bias", type=float, default=-0.20)
    ap.add_argument("--pn-bias", type=float, default=0.0)
    ap.add_argument("--pn-kc-gain", type=float, default=1.0)
    ap.add_argument("--raw-eligibility", action="store_true", help="use raw KC counts instead of mean-subtracted ones")
    ap.add_argument("--raw-scores", action="store_true", help="decide on raw pool counts instead of mean-centred ones")
    ap.add_argument("--pools", default="groups", choices=["groups", "single"], help="MBON pools: groups of types (round 2) or one type per action (round 1)")
    ap.add_argument("--no-extra-orn", action="store_true", help="do not give the safety channels a second ORN type")
    ap.add_argument("--sup-mode", default="error", choices=["error", "teacher"], help="supervised rule (see MBParams)")
    ap.add_argument("--reveal-penalty", type=float, default=3.0)
    ap.add_argument("--w-min-ratio", type=float, default=0.0)
    # round 3
    ap.add_argument("--reward-scheme", default="r1", choices=["r1", "r3"], help="RL-phase rewards (see MBPolicy.game_reward)")
    ap.add_argument("--mask-reveal", action="store_true", help="decoder engineering: mask impossible reveals at the MBON readout")
    ap.add_argument("--epsilon", type=float, default=0.0, help="epsilon-greedy over valid actions (sets temperature 0); annealed to --epsilon-min")
    ap.add_argument("--epsilon-min", type=float, default=0.02)
    ap.add_argument("--odor-level", type=int, default=None, help="extra_orn level (0/1/2); overrides the level stored in --init")
    ap.add_argument("--sup-move-weight", type=float, default=1.0, help="supervised: scale of move-vs-move corrections")
    ap.add_argument("--reveal-margin", type=float, default=0.0, help="decoder reveal margin during training (RL phase should match evaluation)")
    ap.add_argument("--kc-kc-gain", type=float, default=0.0)
    ap.add_argument("--w-max-ratio", type=float, default=5.0)
    ap.add_argument("--turn-steps", type=int, default=15)
    ap.add_argument("--max-turns", type=int, default=400)
    ap.add_argument("--blind", action="store_true", help="no visual board input (odor channels only)")
    ap.add_argument("--report-every", type=int, default=25)
    args = ap.parse_args(argv)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = GameConfig(turn_steps=args.turn_steps, max_turns=args.max_turns)
    mbp = MBParams(eta=args.eta, other_credit=args.other_credit, trace_decay=args.trace_decay, temperature=args.temperature,
                   kc_bias=args.kc_bias, kc_kc_gain=args.kc_kc_gain, w_max_ratio=args.w_max_ratio,
                   pn_bias=args.pn_bias, pn_kc_gain=args.pn_kc_gain, centered=not args.raw_eligibility,
                   center_scores=not args.raw_scores, action_mbons=GROUP_POOLS if args.pools == "groups" else SINGLE_POOLS,
                   extra_orn=(0 if args.no_extra_orn else (args.odor_level if args.odor_level is not None else 1)),
                   sup_mode=args.sup_mode, reveal_penalty=args.reveal_penalty, w_min_ratio=args.w_min_ratio,
                   reward_scheme=args.reward_scheme, mask_reveal=args.mask_reveal, epsilon=args.epsilon,
                   sup_move_weight=args.sup_move_weight, reveal_margin=args.reveal_margin)
    if args.epsilon > 0:
        mbp.temperature = args.temperature = args.temperature_min = 0.0
    brain = Brain()
    player = FlyPlayer(brain, cfg, Params.preset("flyai"), EncoderParams(route="lamina"), DecoderParams(), seed=args.seed0,
                       mb_params=mbp, mb_weights_path=args.init)
    mb = player.enable_mb(load=args.init is not None)
    mb.learning = True
    if args.init is not None:                 # the file restores its decision-side settings; training settings win here
        for k in ("reward_scheme", "mask_reveal", "epsilon", "sup_move_weight", "sup_mode", "reveal_penalty", "eta",
                  "other_credit", "trace_decay", "w_min_ratio", "w_max_ratio", "reveal_margin"):
            setattr(mb.p, k, getattr(mbp, k))
        if args.odor_level is not None and int(args.odor_level) != int(mb.p.extra_orn):
            mb.set_odor_level(args.odor_level)
        print(f"[train_mb] continuing from {args.init}: extra_orn {mb.p.extra_orn}, reward {mb.p.reward_scheme}, mask_reveal {mb.p.mask_reveal}, "
              f"epsilon {mb.p.epsilon}, margin {mb.p.reveal_margin}, eta {mb.p.eta}, trace_decay {mb.p.trace_decay}")
    epsilon0 = args.epsilon
    mb.shuffle_reward = args.shuffle_reward
    mb.shuffle_rng = np.random.default_rng(args.seed0 + 7)
    mb.meta = {"trained_on": {"seed0": args.seed0, "games": args.games}, "args": vars(args) | {"out": str(out)},
               "started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "init": args.init}
    teacher = Teacher()
    print("[train_mb] design:", json.dumps({k: v for k, v in mb.describe().items() if k != "odor"}))
    (out / "design.json").write_text(json.dumps(mb.describe(), indent=1))
    (out / "params.json").write_text(json.dumps({"mb": asdict(mbp), "game": asdict(cfg), "args": vars(args)}, indent=1, default=str))

    games: list[dict] = []
    log = (out / "games.jsonl").open("a")
    t0 = time.time()
    for i in range(args.games):
        seed = args.seed0 + i
        frac = min(1.0, i / max(1, args.anneal_games))
        mb.p.temperature = args.temperature + frac * (args.temperature_min - args.temperature)
        if epsilon0 > 0:
            mb.p.epsilon = epsilon0 + frac * (args.epsilon_min - epsilon0)
        warm = i < args.warmstart_games
        player.teacher_act = teacher.act if warm else None
        act = None
        if warm and args.warmstart_follow:
            act = lambda p, game, decided, rng: teacher.act(game.visible(), p.cursor, rng)
        # teacher agreement: fraction of the fly's own decisions that match the teacher (measured, not used)
        agree = {"n": 0, "same": 0}

        def act_measure(p, game, decided, rng, _agree=agree, _act=act):
            t = teacher.act(game.visible(), p.cursor, rng)
            _agree["n"] += 1
            _agree["same"] += decided == t
            return _act(p, game, decided, rng) if _act is not None else decided

        rec = player.play("fly-mb", seed, act=act_measure)
        pl = rec.plasticity or {}
        row = {
            "game": i, "seed": seed, "outcome": rec.outcome, "safe_revealed": rec.safe_revealed, "total_safe": rec.total_safe,
            "turns": rec.turns, "reveals": rec.reveals, "noop_reveals": rec.noop_reveals, "actions": rec.actions,
            "reward": round(pl.get("game", {}).get("reward", 0.0), 3), "events": pl.get("game", {}).get("events", 0),
            "supervised": pl.get("game", {}).get("supervised", 0), "abs_delta": round(pl.get("game", {}).get("abs_delta", 0.0), 5),
            "edges_changed": pl.get("game", {}).get("edges_changed", 0), "mean_weight_ratio": round(pl.get("mean_weight_ratio", 1.0), 4),
            "pool_ratio": pl.get("pool_ratio"), "baseline": round(pl.get("baseline", 0.0), 4), "temperature": round(mb.p.temperature, 3),
            "epsilon": round(mb.p.epsilon, 3),
            "warmstart": warm, "teacher_agreement": round(agree["same"] / max(1, agree["n"]), 3), "seconds": rec.seconds,
        }
        games.append(row)
        log.write(json.dumps(row) + "\n")
        log.flush()
        if (i + 1) % args.report_every == 0 or i == 0:
            last = games[-args.report_every:]
            wins = sum(g["outcome"] == "won" for g in last)
            print(f"[train_mb] game {i + 1:5d}/{args.games} seed {seed}  last {len(last)}: wins {wins}, safe {np.mean([g['safe_revealed'] for g in last]):5.1f}, "
                  f"turns {np.mean([g['turns'] for g in last]):5.1f}, reward {np.mean([g['reward'] for g in last]):+.3f}, agree {np.mean([g['teacher_agreement'] for g in last]):.2f}, "
                  f"|dw| {np.mean([g['abs_delta'] for g in last]):.4f}, ratio {row['mean_weight_ratio']:.3f} pools {row['pool_ratio']}, T {row['temperature']:.2f}, "
                  f"{(time.time() - t0) / (i + 1):.2f}s/game", flush=True)
        if (i + 1) % args.checkpoint_every == 0 or i + 1 == args.games:
            mb.save(out / f"weights_{i + 1:06d}.npz", {"games_done": i + 1})
            rows = curve_rows(games)
            (out / "curve.json").write_text(json.dumps(rows, indent=1))
            (out / "curve.md").write_text(curve_markdown(rows, f"Learning curve ({out.name}); seeds {args.seed0}.. ; shuffle_reward={args.shuffle_reward}"))
    log.close()
    mb.save(out / "weights_final.npz", {"games_done": args.games, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    if not args.no_install and not args.shuffle_reward:
        mb.save(DEFAULT_WEIGHTS, {"games_done": args.games, "run": str(out)})
        print(f"[train_mb] installed weights -> {DEFAULT_WEIGHTS}")
    rows = curve_rows(games)
    print(curve_markdown(rows, out.name))
    print(f"[train_mb] done: {args.games} games in {time.time() - t0:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
