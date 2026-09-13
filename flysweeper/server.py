"""Live spectator: the fly plays Minesweeper forever; a browser page watches.

    python -m flysweeper.server [--port 8765] [--speed 1.0] [--condition fly]

Binds 127.0.0.1 only.  GET /            the page
                       GET /state       JSON snapshot (board, cursor, pools, stats, events)
                       GET /atlas       binary soma positions for the brain map (once)
                       GET /spikes      binary uint8 activity per plotted neuron
                       GET /control?speed=4&pause=0
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import threading
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from .agent import BRAIN_CONDITIONS, FlyPlayer, GameConfig
from .brain import Brain
from .decoder import DecoderParams
from .encoder import EncoderParams
from .sim import PRESETS, Params

UI_DIR = Path(__file__).resolve().parent / "ui"

REGIONS = [
    ("optic lobe", "ol_", (86, 156, 255)),
    ("visual projection", "visual_", (120, 220, 255)),
    ("central brain", "cb_", (255, 196, 90)),
    ("descending / ascending", "descending", (255, 90, 120)),
    ("nerve cord", "vnc_", (140, 255, 150)),
    ("sensory", "sensory", (220, 140, 255)),
    ("other", "", (170, 170, 170)),
]


def region_of(superclass: str) -> int:
    for i, (_, prefix, _) in enumerate(REGIONS):
        if prefix and (superclass.startswith(prefix) or (prefix in ("descending", "sensory") and prefix in superclass)):
            return i
    if "ascending" in superclass:
        return 3
    return len(REGIONS) - 1


class Spectator:
    def __init__(self, player: FlyPlayer, condition: str, speed: float, seed0: int, game_over_hold: float = 2.5):
        self.player = player
        self.condition = condition
        self.speed = speed
        self.game_over_hold = game_over_hold   # seconds the finished board stays on screen (paced runs only)
        self.paused = False
        self.seed = seed0
        self.lock = threading.Lock()
        self.events: list[dict] = []
        self.games = 0
        self.wins = 0
        self.total_safe = 0
        self.last_record = None
        self.step_wall = time.perf_counter()
        self.rtf = 0.0
        self.ms_per_step = 0.0
        self._ema_step = None
        self.running = True

        nm = player.brain.neurons
        has = nm["x"].notna().to_numpy()
        self.plot_idx = np.flatnonzero(has).astype(np.int32)
        x = nm["x"].to_numpy(dtype=np.float32)[self.plot_idx]
        z = nm["z"].to_numpy(dtype=np.float32)[self.plot_idx]
        # Fly's left (high x) drawn on the left, brain on top (low z), nerve cord below.
        u = 1.0 - (x - x.min()) / (x.max() - x.min())
        v = (z - z.min()) / (z.max() - z.min())
        sc = nm["superclass"].fillna("").to_numpy().astype(str)[self.plot_idx]
        region = np.array([region_of(s) for s in sc], dtype=np.uint8)
        self.atlas = struct.pack("<I", len(self.plot_idx)) + np.stack([u, v], 1).astype("<f4").tobytes() + region.tobytes()

        player.listeners.append(self._on_event)
        player.on_step = self._on_step

    def _on_event(self, player: FlyPlayer, ev: dict) -> None:
        with self.lock:
            if "record" in ev:
                rec = ev["record"]
                self.games += 1
                self.wins += rec["outcome"] == "won"
                self.total_safe += rec["safe_revealed"]
                self.last_record = rec
                self.events.append({"t": round(player.sim.time, 1), "text": f"game {self.games} {rec['outcome']}: {rec['safe_revealed']}/{rec['total_safe']} safe cells in {rec['turns']} turns"})
            else:
                text = ev["action"]
                if ev["action"] == "reveal":
                    text += f" -> {ev.get('result')}" + (f" (+{ev['newly']})" if ev.get("newly") else "")
                if ev["action"] == "jump":
                    text += f" from {ev.get('from')}"
                self.events.append({"t": round(player.sim.time, 1), "text": text})
            del self.events[:-40]

    def _on_step(self, player: FlyPlayer) -> None:
        now = time.perf_counter()
        dt_wall = now - self.step_wall
        self._ema_step = dt_wall if self._ema_step is None else 0.95 * self._ema_step + 0.05 * dt_wall
        self.ms_per_step = 1000 * player.sim.last_step_seconds
        self.rtf = player.sim.p.dt / max(self._ema_step, 1e-6)
        while self.paused and self.running:
            time.sleep(0.05)
            self.step_wall = time.perf_counter()
        if self.speed > 0:
            target = self.step_wall + player.sim.p.dt / self.speed
            remaining = target - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
        self.step_wall = time.perf_counter()

    def loop(self) -> None:
        while self.running:
            self.player.play(self.condition, self.seed)
            self.seed += 1
            # hold the finished board (player.game stays the finished object, so /state keeps reporting
            # it with the mines shown) long enough for the end-of-game animation to be seen
            if self.speed > 0:
                until = time.perf_counter() + self.game_over_hold
                while self.running and time.perf_counter() < until:
                    time.sleep(0.05)

    def spikes(self) -> bytes:
        c = self.player.spike_counts[self.plot_idx]
        return np.minimum(c * 3, 255).astype(np.uint8).tobytes()

    def state(self) -> dict:
        p = self.player
        game = p.game
        with self.lock:
            events = list(self.events)
            games, wins, total_safe, last = self.games, self.wins, self.total_safe, self.last_record
        dec = p.decoder
        return {
            "label": "MaleCNS v1.0 wiring (Berg et al., Cell 2026, CC BY 4.0) with engineered dynamics, sensors and buttons. Not a validated fly. Not a trained Minesweeper player. Connectome frozen." + (" KC->MBON plasticity ON." if p.plasticity else ""),
            "condition": p.condition,
            "board": game.visible().tolist() if game is not None else None,
            "rows": p.cfg.rows, "cols": p.cfg.cols, "mines": p.cfg.mines,
            "cursor": list(p.cursor),
            "danger": int(p.encoder.last_danger),
            "last_action": p.last_action,
            "actions": dec.actions,
            "rates": [round(float(x), 2) for x in p.last_rates],
            "scores": [round(float(x), 2) for x in dec.last_scores],
            "idle": [round(float(x), 2) for x in dec.idle_rates],
            "pools": {a: int(len(dec.pool_idx[a])) for a in dec.actions},
            "game": {
                "number": games + 1,
                "safe_revealed": game.safe_revealed if game else 0,
                "total_safe": game.total_safe if game else 0,
                "moves": game.moves if game else 0,
                "over": game.over if game else False,
                "won": game.won if game else False,
                "seed": game.seed if game else None,
            },
            "totals": {"games": games, "wins": wins, "mean_safe": round(total_safe / games, 1) if games else None, "last": last and {k: last[k] for k in ("outcome", "safe_revealed", "turns")}},
            "sim": {
                "step": int(p.sim.step_count), "time_s": round(p.sim.time, 1), "ms_per_step": round(self.ms_per_step, 2),
                "rtf": round(self.rtf, 1), "speed": self.speed, "paused": self.paused,
                "spikes_last_step": int(len(p.sim.fired)), "neurons": int(p.brain.n), "edges": int(p.brain.graph.n_edges),
                "preset": {k: v for k, v in asdict(p.sim.p).items()}, "sensory_input": p.sim.sensory_input,
            },
            "encoder": {"route": p.encoder.p.route, "danger_loom": p.encoder.p.danger_loom, "loom_threshold": p.encoder.p.loom_threshold},
            "plasticity": p.plasticity.summary() if p.plasticity else None,
            "regions": [{"name": n, "color": c} for n, _, c in REGIONS],
            "events": events[::-1],
        }


def make_handler(spec: Spectator):
    index_path = UI_DIR / "index.html"   # read per request so page edits show up on reload

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                self._send(200, index_path.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/state":
                self._send(200, json.dumps(spec.state()).encode(), "application/json")
            elif url.path == "/atlas":
                self._send(200, spec.atlas, "application/octet-stream")
            elif url.path == "/spikes":
                self._send(200, spec.spikes(), "application/octet-stream")
            elif url.path == "/control":
                q = parse_qs(url.query)
                if "speed" in q:
                    spec.speed = float(q["speed"][0])
                if "pause" in q:
                    spec.paused = q["pause"][0] in ("1", "true")
                self._send(200, json.dumps({"speed": spec.speed, "paused": spec.paused}).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

    return Handler


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--speed", type=float, default=1.0, help="simulated seconds per wall second; 0 = as fast as possible")
    ap.add_argument("--condition", default="fly", choices=BRAIN_CONDITIONS)
    ap.add_argument("--preset", default="flyai", choices=sorted(PRESETS))
    ap.add_argument("--route", default="lamina", choices=["retina", "lamina"])
    ap.add_argument("--no-loom", action="store_true", help="disable the looming/escape reflex channel")
    ap.add_argument("--sensory-input", action="store_true")
    ap.add_argument("--turn-steps", type=int, default=15)
    ap.add_argument("--rows", type=int, default=9)
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--mines", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--game-over-hold", type=float, default=2.5, help="seconds to keep a finished board on screen before the next game (ignored at --speed 0)")
    args = ap.parse_args(argv)

    brain = Brain()
    cfg = GameConfig(rows=args.rows, cols=args.cols, mines=args.mines, turn_steps=args.turn_steps)
    player = FlyPlayer(
        brain, cfg, Params.preset(args.preset), EncoderParams(route=args.route, danger_loom=not args.no_loom),
        DecoderParams(), sensory_input=args.sensory_input, seed=args.seed0,
    )
    spec = Spectator(player, args.condition, args.speed, args.seed0, game_over_hold=args.game_over_hold)
    threading.Thread(target=spec.loop, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(spec))
    print(f"[server] FlySweeper live at http://127.0.0.1:{args.port}/  (condition {args.condition}, speed {args.speed}x)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        spec.running = False
    return 0


if __name__ == "__main__":
    sys.exit(main())
