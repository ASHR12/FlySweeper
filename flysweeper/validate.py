"""Play N seeded games per condition and write a comparison report.

The same seeds (hence the same mine layouts) are used for every condition, so differences are
attributable to the player, not the boards.  "fly" beating "fly-blind" would show the board
matters to the brain's output; "fly-learning" beating "fly" on held-out seeds would be the only
admissible evidence of in-brain learning; "fly-readout" (trained linear readout, see
train_readout.py) beating "random-walk" / "random-click" on seeds it was never trained on is the
only admissible evidence that the readout learned anything.  Nothing is claimed until this report
says so.

Usage:
    python -m flysweeper.validate --games 20 --conditions fly fly-blind random-walk random-click solver
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

from .agent import ALL_CONDITIONS, BRAIN_CONDITIONS, FlyPlayer, GameConfig, play_scripted
from .brain import Brain
from .decoder import DecoderParams
from .encoder import EncoderParams
from .paths import OUTPUTS
from .sim import PRESETS, Params


def summarize(records: list[dict]) -> dict:
    n = len(records)
    won = sum(r["outcome"] == "won" for r in records)
    lost = sum(r["outcome"] == "lost" for r in records)
    return {
        "games": n,
        "win_rate": round(won / n, 3),
        "loss_rate": round(lost / n, 3),
        "timeout_rate": round((n - won - lost) / n, 3),
        "mean_safe_revealed": round(float(np.mean([r["safe_revealed"] for r in records])), 2),
        "mean_cleared_fraction": round(float(np.mean([r["cleared_fraction"] for r in records])), 3),
        "mean_turns": round(float(np.mean([r["turns"] for r in records])), 1),
        "mean_reveals": round(float(np.mean([r["reveals"] for r in records])), 1),
        "mean_noop_reveals": round(float(np.mean([r["noop_reveals"] for r in records])), 1),
        "mean_holds": round(float(np.mean([r["holds"] for r in records])), 1),
        "sem_safe_revealed": round(float(np.std([r["safe_revealed"] for r in records], ddof=1) / np.sqrt(n)), 2) if n > 1 else 0.0,
        "seconds": round(float(np.sum([r["seconds"] for r in records])), 1),
    }


def plasticity_summary(records: list[dict]) -> dict | None:
    """Cumulative KC->MBON weight state after the last learning game, plus per-game event counts."""
    with_p = [r for r in records if r.get("plasticity")]
    if not with_p:
        return None
    last = with_p[-1]["plasticity"]
    return {
        "games": len(with_p),
        "dopamine_events_total": int(last["events"]),
        "kc_mbon_edges": int(last["kc_mbon_edges"]),
        "edges_changed_vs_frozen": int(last.get("edges_changed", -1)),
        "total_abs_delta": round(float(last["total_abs_delta"]), 3),
        "mean_weight_ratio": round(float(last["mean_weight_ratio"]), 4),
        "min_weight_ratio": round(float(last["min_weight_ratio"]), 4),
        "max_weight_ratio": round(float(last["max_weight_ratio"]), 4),
        "edges_at_floor": int(last["edges_at_floor"]),
        "edges_at_ceiling": int(last["edges_at_ceiling"]),
        "mean_weight_ratio_by_game": [round(float(r["plasticity"]["mean_weight_ratio"]), 3) for r in with_p],
    }


def report_markdown(summary: dict, meta: dict) -> str:
    cols = ["games", "win_rate", "mean_safe_revealed", "sem_safe_revealed", "mean_cleared_fraction", "mean_turns", "mean_reveals", "mean_noop_reveals", "mean_holds"]
    lines = [
        "# FlySweeper validation",
        "",
        f"Generated {meta['generated']}. Board {meta['config']['rows']}x{meta['config']['cols']}, {meta['config']['mines']} mines, "
        f"{meta['config']['turn_steps']} brain steps per turn, max {meta['config']['max_turns']} turns. "
        f"Seeds {meta['seeds'][0]}..{meta['seeds'][-1]} shared by all conditions.",
        "",
        f"Brain: MaleCNS v1.0, {meta['brain']['n_neurons']:,} neurons, {meta['brain']['n_edges']:,} edges. "
        f"Dynamics preset `{meta['preset']}`, sensory_input={meta['sensory_input']}, input route `{meta['route']}`"
        + (", egocentric view (eyes centred on the cursor)" if meta.get("egocentric") else "") + ".",
        "",
        "| condition | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for cond, s in summary.items():
        lines.append(f"| {cond} | " + " | ".join(str(s[c]) for c in cols) + " |")
    lines += [
        "",
        "Idle pool rates (Hz per cell, blank screen): " + ", ".join(f"{k} {v:.2f}" for k, v in meta.get("idle_rates", {}).items()),
    ]
    if meta.get("readout"):
        r = meta["readout"]
        lines += ["", f"Readout (`fly-readout`): `{r.get('path')}`, {r.get('cells'):,} cells x {r.get('windows')} window(s) from groups "
                  f"{r.get('groups')}, {r.get('classifier')}; trained on seeds {r.get('train_seeds')} ({r.get('n_turns'):,} turns), "
                  f"held-out turn accuracy {r.get('held_out_accuracy')}."]
    if meta.get("plasticity"):
        p = meta["plasticity"]
        lines += ["", f"Plasticity (`fly-learning`, {p['games']} games, cumulative): {p['dopamine_events_total']} dopamine events, "
                  f"{p['edges_changed_vs_frozen']:,} of {p['kc_mbon_edges']:,} KC->MBON edges differ from the frozen wiring, "
                  f"total |dw| {p['total_abs_delta']}, mean weight ratio {p['mean_weight_ratio']} "
                  f"(min {p['min_weight_ratio']}, max {p['max_weight_ratio']}; {p['edges_at_floor']:,} at the floor, {p['edges_at_ceiling']:,} at the ceiling). "
                  f"Mean ratio after each game: {p['mean_weight_ratio_by_game']}."]
    if meta.get("mb"):
        m = meta["mb"]
        ws = meta.get("mb_weights_state") or {}
        lines += ["", f"Mushroom-body policy (`fly-mb`): a helper reads the board into {m.get('channels')} facts injected as odors "
                  f"(one ORN type each); actions are read from MBON pools {m.get('action_mbons')}; the only trained synapses are the "
                  f"{m.get('plastic_edges'):,} KC->MBON edges onto those pools. Weights: `{m.get('weights')}` "
                  f"({(m.get('trained') or {}).get('trained_on')}, {m.get('games_trained')} games); softmax temperature {m.get('temperature')} (0 = argmax). "
                  f"Mean weight ratio per pool: {m.get('pool_ratio')}"
                  + (f"; {ws.get('edges_changed_vs_frozen', 0):,} edges differ from the frozen wiring (min ratio {ws.get('min_weight_ratio')}, max {ws.get('max_weight_ratio')}, "
                     f"{ws.get('edges_at_floor', 0):,} at the floor, {ws.get('edges_at_ceiling', 0):,} at the ceiling)." if ws else "."),
                  f"Documented KC changes for this condition only: KC->KC gain {m.get('kc_kc_gain')} ({m.get('kc_kc_edges_silenced'):,} edges), "
                  f"KC bias {m.get('kc_bias')}, PN->KC gain {m.get('pn_kc_gain')}. `fly-mb-oracle-only` reads the same facts with a fixed rule and no brain."]
    lines += [
        "",
        "Reading the table: `fly` vs `fly-blind` isolates the effect of the board on the brain's output; "
        "`random-walk` is the same action set with the brain replaced by coin flips; `random-click` and "
        "`solver` bracket the task; `fly-readout` is the same frozen brain with the trained readout. "
        "`sem_safe_revealed` is the standard error of the mean over games. No learning claim is made unless "
        "`fly-learning` beats `fly`, or `fly-readout` beats `random-walk` and `random-click`, on seeds never used for training.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--conditions", nargs="+", default=["fly", "fly-blind", "random-walk", "random-click", "solver"],
                    choices=list(ALL_CONDITIONS) + ["fly-mb-alt"], help="fly-mb-alt = fly-mb played by a second brain loaded from --mb-weights-alt")
    ap.add_argument("--mb-weights-alt", default=None, help="weights for the fly-mb-alt condition (a second FlyPlayer, same seeds)")
    ap.add_argument("--preset", default="flyai", choices=sorted(PRESETS))
    ap.add_argument("--route", default="lamina", choices=["retina", "lamina"])
    ap.add_argument("--ego", action="store_true", help="egocentric encoder (eyes centred on the cursor) for all brain conditions")
    ap.add_argument("--ego-radius", type=int, default=4)
    ap.add_argument("--sensory-input", action="store_true")
    ap.add_argument("--turn-steps", type=int, default=15)
    ap.add_argument("--max-turns", type=int, default=400)
    ap.add_argument("--rows", type=int, default=9)
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--mines", type=int, default=10)
    ap.add_argument("--decoder", default="argmax", choices=["argmax", "softmax", "readout"],
                    help="decoder for the seeing conditions; `readout` makes even `fly` use the trained readout")
    ap.add_argument("--readout", default=None, help="readout file for fly-readout (default data/compiled/readout.npz)")
    ap.add_argument("--mb-weights", default=None, help="trained KC->MBON weights for fly-mb (default data/compiled/mb_weights.npz)")
    ap.add_argument("--mb-temperature", type=float, default=0.0, help="fly-mb softmax temperature (0 = argmax, exploration off)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = GameConfig(rows=args.rows, cols=args.cols, mines=args.mines, turn_steps=args.turn_steps, max_turns=args.max_turns)
    seeds = list(range(args.seed0, args.seed0 + args.games))
    out = args.out or (OUTPUTS / "validation" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(parents=True, exist_ok=True)

    records: dict[str, list[dict]] = {c: [] for c in args.conditions}
    meta = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(cfg), "seeds": seeds, "preset": args.preset, "route": args.route,
        "sensory_input": args.sensory_input, "decoder": args.decoder,
    }
    player = player_alt = None
    if "fly-mb-alt" in args.conditions:
        assert args.mb_weights_alt, "--mb-weights-alt is required for fly-mb-alt"
        brain = Brain()
        player_alt = FlyPlayer(brain, cfg, Params.preset(args.preset), EncoderParams(route=args.route, egocentric=args.ego, ego_radius=args.ego_radius),
                               DecoderParams(), sensory_input=args.sensory_input, seed=args.seed0, mb_weights_path=args.mb_weights_alt)
        mb_alt = player_alt.enable_mb()
        meta["mb_alt"] = {"weights": mb_alt.loaded_from, "trained": mb_alt.meta, "pool_ratio": mb_alt.pool_ratios(),
                          "reveal_margin": mb_alt.p.reveal_margin, "mask_reveal": mb_alt.p.mask_reveal, "extra_orn": mb_alt.p.extra_orn}
    if any(c in BRAIN_CONDITIONS for c in args.conditions):
        brain = brain if player_alt is not None else Brain()
        meta["brain"] = {"n_neurons": brain.n, "n_edges": brain.graph.n_edges}
        player = FlyPlayer(
            brain, cfg, Params.preset(args.preset), EncoderParams(route=args.route, egocentric=args.ego, ego_radius=args.ego_radius),
            DecoderParams(mode=args.decoder, readout_path=args.readout), sensory_input=args.sensory_input, seed=args.seed0,
            readout_path=args.readout, mb_weights_path=args.mb_weights,
        )
        if "fly-mb" in args.conditions:
            mb = player.enable_mb()
            mb.p.temperature = args.mb_temperature
            mb.learning = False
            meta["mb"] = {k: v for k, v in mb.describe().items() if k != "odor"} | {
                "weights": mb.loaded_from, "trained": mb.meta, "temperature": args.mb_temperature,
                "pool_ratio": mb.pool_ratios(), "channels": len(mb.odor.types)}
            if mb.loaded_from is None:
                print("[validate] WARNING: fly-mb is running on UNTRAINED (frozen) KC->MBON weights; train with train_mb.py first")
            print("[validate] fly-mb:", json.dumps(meta["mb"], default=str))
        meta["idle_rates"] = {a: float(r) for a, r in zip(player.decoder.actions, player.idle_rates)}
        meta["pools"] = player.pool_decoder.describe(brain)
        meta["encoder"] = player.encoder.coverage_summary()
        meta["egocentric"] = args.ego
        if "fly-readout" in args.conditions or args.decoder == "readout":
            ro = player.enable_readout()
            meta["readout"] = ro.describe(brain)["readout"]
            trained_enc = ro.readout_meta.get("encoder", {})
            if trained_enc and (bool(trained_enc.get("egocentric", False)) != args.ego or trained_enc.get("route", args.route) != args.route):
                print(f"[validate] WARNING: readout was trained with encoder {trained_enc}, but this run uses "
                      f"route={args.route} egocentric={args.ego}; the readout will be evaluated on inputs it was not trained on")
            print("[validate] readout:", meta["readout"])
        print("[validate] idle rates:", meta["idle_rates"])
    else:
        meta["brain"] = {"n_neurons": 0, "n_edges": 0}

    t0 = time.time()
    for cond in args.conditions:
        for seed in seeds:
            if cond == "fly-mb-alt":
                rec = asdict(player_alt.play("fly-mb", seed))
                rec["condition"] = cond
            elif cond in BRAIN_CONDITIONS:
                rec = asdict(player.play(cond, seed))
            else:
                rec = asdict(play_scripted(cond, seed, cfg))
            records[cond].append(rec)
            print(f"[validate] {cond:12s} seed {seed:3d}  {rec['outcome']:7s} safe {rec['safe_revealed']:2d}/{rec['total_safe']}  "
                  f"turns {rec['turns']:3d}  reveals {rec['reveals']:3d} noop {rec['noop_reveals']:3d} holds {rec['holds']:3d}  {rec['seconds']:.1f}s", flush=True)
    summary = {c: summarize(r) for c, r in records.items()}
    if "fly-learning" in records:
        meta["plasticity"] = plasticity_summary(records["fly-learning"])
        print("[validate] plasticity:", json.dumps(meta["plasticity"]))
    if "fly-mb" in records:
        meta["mb_weights_state"] = plasticity_summary(records["fly-mb"])
        print("[validate] fly-mb weight state:", json.dumps(meta["mb_weights_state"]))
    (out / "results.json").write_text(json.dumps({"meta": meta, "summary": summary, "records": records}, indent=1, default=str))
    (out / "report.md").write_text(report_markdown(summary, meta))
    print(json.dumps(summary, indent=2))
    print(f"[validate] wrote {out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
