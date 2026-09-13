"""The loop: board -> retina -> whole-graph LIF -> descending neurons -> cursor / reveal.

Conditions (same boards, same seeds, so they are directly comparable):
    fly           the full loop, connectome frozen, hand-picked descending-neuron pools decode
    fly-readout   the same frozen brain and input, but the actions are read out by the linear
                  readout trained in train_readout.py (imitation of the teacher; decoder.py)
    fly-blind     identical to fly, but the photoreceptors receive no board (black screen)
    fly-learning  fly + dopamine stimulus + KC->MBON plasticity (see plasticity.py)
    fly-mb        a helper reads the board into 31 facts that are injected as odors (ORN cells); the
                  6 actions are read from 6 MBON pools; KC->MBON weights trained by dopamine-gated
                  RL (mb_policy.py, train_mb.py); weights loaded from data/compiled/mb_weights.npz
    random-walk   the brain replaced by a uniform random choice over the same action set
    random-click  reveal a uniformly random hidden cell each turn
    solver        single-point Minesweeper logic with random guesses when stuck
    fly-mb-oracle-only  the same 31 helper facts read by a fixed hand-written rule (no brain): the
                  ceiling the helper's facts allow
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from .brain import Brain
from .decoder import DecoderParams, PoolDecoder, ReadoutDecoder
from .encoder import EncoderParams, RetinaEncoder
from .features import FeatureExtractor
from .mb_policy import DEFAULT_WEIGHTS, MBParams, MBPolicy
from .minesweeper import Minesweeper, solver_move
from .oracle import Oracle
from .plasticity import KCMBONPlasticity, PlasticityParams
from .sim import LIF, Params

BRAIN_CONDITIONS = ("fly", "fly-readout", "fly-blind", "fly-learning", "fly-mb")
ALL_CONDITIONS = BRAIN_CONDITIONS + ("random-walk", "random-click", "solver", "fly-mb-oracle-only")


@dataclass
class GameConfig:
    rows: int = 9
    cols: int = 9
    mines: int = 10
    turn_steps: int = 15          # brain steps per decision (15 x 20 ms = 300 ms)
    max_turns: int = 400
    settle_steps: int = 25        # steps between games with the board blank
    idle_steps: int = 500         # blank-screen steps used to measure each pool's idle rate
    reward_scale: float = 0.25    # dopamine per newly revealed cell (learning condition)


@dataclass
class GameRecord:
    condition: str
    seed: int
    outcome: str                  # won | lost | timeout
    turns: int
    safe_revealed: int
    total_safe: int
    cleared_fraction: float
    reveals: int
    noop_reveals: int
    holds: int
    actions: dict = field(default_factory=dict)
    seconds: float = 0.0
    plasticity: dict | None = None
    action_trace: list[str] = field(default_factory=list)
    overrides: int = 0            # turns where an `act` callback replaced the brain's decision


class MBStatsHook:
    """Stands in for `player.plasticity` while fly-mb plays, so anything that reads
    `player.plasticity.summary()` (the spectator's /state) sees the MB policy's weight stats.
    It never learns: attach/detach/observe/dopamine are no-ops and `attached` is False."""

    attached = False

    def __init__(self, mb: MBPolicy):
        self.mb = mb

    def summary(self) -> dict:
        return self.mb.summary()

    def attach(self) -> None: ...
    def detach(self) -> None: ...
    def observe(self, fired) -> None: ...
    def dopamine(self, *a, **k) -> None: ...


class FlyPlayer:
    """Holds the brain, encoder and decoder; plays games under any brain condition."""

    def __init__(
        self,
        brain: Brain,
        config: GameConfig | None = None,
        sim_params: Params | None = None,
        encoder_params: EncoderParams | None = None,
        decoder_params: DecoderParams | None = None,
        plasticity_params: PlasticityParams | None = None,
        sensory_input: bool = False,
        seed: int = 0,
        readout_path: str | None = None,
        features: FeatureExtractor | None = None,
        mb_params: MBParams | None = None,
        mb_weights_path: str | None = None,
    ):
        self.brain = brain
        self.cfg = config or GameConfig()
        self.sim = LIF(brain, sim_params or Params.preset("flyai"), seed=seed, sensory_input=sensory_input)
        self.encoder = RetinaEncoder(brain, self.cfg.rows, self.cfg.cols, encoder_params)
        self.decoder_params = decoder_params or DecoderParams()
        self.readout_path = readout_path
        # The pool decoder is always built (the UI shows its pools); mode="readout" makes the
        # trained readout the decision maker for every seeing condition.
        self.pool_decoder = PoolDecoder(brain, self.decoder_params)
        self.readout: ReadoutDecoder | None = None
        if self.decoder_params.mode == "readout":
            self.readout = ReadoutDecoder(brain, self.decoder_params, readout_path)
        self.decoder = self.readout or self.pool_decoder
        self.features = features        # optional extra extractor, fed every step (training data)
        self.plasticity_params = plasticity_params or PlasticityParams()
        self.plasticity: KCMBONPlasticity | None = None
        self.mb_params = mb_params
        self.mb_weights_path = mb_weights_path      # None -> data/compiled/mb_weights.npz if it exists
        self.mb: MBPolicy | None = None
        self.mb_active = False
        self.teacher_act = None        # set by train_mb.py for the supervised warm start
        self.listeners: list = []      # callables (player, event: dict) for the spectator
        self.on_step = None            # callable(player) after every brain step (pacing / publishing)
        self.game: Minesweeper | None = None
        self.cursor = (self.cfg.rows // 2, self.cfg.cols // 2)
        self.last_action = "start"
        self.last_rates = np.zeros(len(self.decoder.actions), dtype=np.float32)
        self.condition = "fly"
        self.spike_counts = np.zeros(brain.n, dtype=np.int32)   # rolling window for the UI
        self.spike_decay_every = 5
        self.idle_rates = self.calibrate_idle()

    def _emit(self, event: dict) -> None:
        for fn in self.listeners:
            fn(self, event)

    def calibrate_idle(self) -> np.ndarray:
        """Measure each pool's per-cell rate on a blank screen; the decoder scores rates above it."""
        blank = np.full((self.cfg.rows, self.cfg.cols), -1, dtype=np.int8)
        self._new_window()
        for _ in range(100):
            self._brain_step(blank, blind=True)
        self._new_window()
        for _ in range(self.cfg.idle_steps):
            self._brain_step(blank, blind=True)
        idle = self.decoder.rates(self.cfg.idle_steps, self.sim.p.dt)
        self.decoder.set_idle(idle)
        if self.decoder is not self.pool_decoder:
            self.pool_decoder.set_idle(idle)
        self._new_window()
        return idle

    def enable_learning(self) -> None:
        if self.plasticity is None:
            self.plasticity = KCMBONPlasticity(self.brain, self.sim, self.plasticity_params)

    def enable_readout(self) -> ReadoutDecoder:
        if self.readout is None:
            self.readout = ReadoutDecoder(self.brain, self.decoder_params, self.readout_path)
            self.readout.set_idle(self.pool_decoder.idle_rates)
        return self.readout

    def enable_mb(self, load: bool = True) -> MBPolicy:
        """Build the mushroom-body policy; load trained KC->MBON weights if a file exists.

        A loaded policy plays with exploration OFF (argmax + the file's reveal margin) and learning
        off, exactly as validate.py evaluates it; train_mb.py re-enables both explicitly."""
        if self.mb is None:
            path = self.mb_weights_path or (DEFAULT_WEIGHTS if DEFAULT_WEIGHTS.exists() else None)
            if load and path is not None:
                # pools / odor mapping / KC changes come from the file so the edge set matches
                path = str(Path(path).resolve())
                self.mb = MBPolicy.from_file(self.brain, self.sim, path, self.mb_params)
                self.mb.loaded_from = path
                self.mb.learning = False
                self.mb.p.temperature = 0.0
                p = self.mb.p
                print(f"[fly-mb] weights {path}: {len(self.mb.edge_pos)} plastic KC->MBON edges, "
                      f"{self.mb.games_trained} games trained; pools "
                      + ", ".join(f"{a}={len(self.mb.pool_idx[a])} cells" for a in self.mb.actions)
                      + f"; KC settings kc_kc_gain {p.kc_kc_gain} kc_bias {p.kc_bias} pn_bias {p.pn_bias} "
                      f"pn_kc_gain {p.pn_kc_gain}; decision argmax (temperature {p.temperature}), "
                      f"reveal margin {p.reveal_margin}, centred scores {p.center_scores}", flush=True)
            else:
                self.mb = MBPolicy(self.brain, self.sim, self.mb_params)
                self.mb.loaded_from = None
        return self.mb

    @property
    def mb_summary(self) -> dict | None:
        return self.mb.summary() if self.mb is not None else None

    @property
    def learning_active(self) -> bool:
        return self.plasticity is not None and self.plasticity.attached

    def _brain_step(self, visible, blind: bool) -> np.ndarray:
        if not blind:
            idx, drive = self.encoder.drive(visible, self.cursor, self.sim.step_count)
            self.sim.inject(idx, drive)
            loom_idx, amp = self.encoder.loom_drive(visible, self.cursor, self.sim.step_count)
            if amp > 0:
                self.sim.inject(loom_idx, amp)
        if self.mb_active:
            self.mb.before_step()      # odor channels + scheduled dopamine
        fired = self.sim.step()
        self.decoder.observe(fired)
        if self.features is not None:
            self.features.observe(fired)
        if self.learning_active:
            self.plasticity.observe(fired)
        if self.sim.step_count % self.spike_decay_every == 0:
            self.spike_counts -= self.spike_counts >> 1
        self.spike_counts[fired] += 8
        if self.on_step is not None:
            self.on_step(self)
        return fired

    def _new_window(self) -> None:
        self.decoder.reset_window()
        if self.features is not None:
            self.features.new_window()

    def settle(self, blind: bool) -> None:
        blank = np.full((self.cfg.rows, self.cfg.cols), -1, dtype=np.int8)
        for _ in range(self.cfg.settle_steps):
            self._brain_step(blank, blind)
        self._new_window()

    def apply_action(self, game: Minesweeper, action: str, rng: np.random.Generator, event: dict) -> tuple[str, int]:
        """Move the cursor / reveal / flag; returns (reveal result or '', newly revealed)."""
        r, c = self.cursor
        result, newly = "", 0
        if action == "up":
            self.cursor = (max(0, r - 1), c)
        elif action == "down":
            self.cursor = (min(self.cfg.rows - 1, r + 1), c)
        elif action == "left":
            self.cursor = (r, max(0, c - 1))
        elif action == "right":
            self.cursor = (r, min(self.cfg.cols - 1, c + 1))
        elif action == "flag":
            game.toggle_flag(r, c)
        elif action == "jump":
            hidden = game.hidden_cells()
            if hidden:
                self.cursor = hidden[int(rng.integers(len(hidden)))]
            event["from"] = (r, c)
        elif action == "reveal":
            result, newly = game.reveal(r, c)
            event["result"], event["newly"] = result, newly
        return result, newly

    def play(self, condition: str, seed: int, rng: np.random.Generator | None = None, act=None) -> GameRecord:
        """Play one game.  `act(player, game, decoded_action, rng) -> action` may override the
        brain's decision each turn (used by train_readout.py to let the teacher drive the cursor);
        the brain still sees every board and every step is still simulated."""
        assert condition in BRAIN_CONDITIONS, condition
        t0 = time.perf_counter()
        self.condition = condition
        blind = condition == "fly-blind"
        learning = condition == "fly-learning"
        if isinstance(self.plasticity, MBStatsHook) and condition != "fly-mb":
            self.plasticity = None            # the hook only stands in while fly-mb is playing
        if learning:
            self.enable_learning()
            self.plasticity.attach()
        elif self.plasticity is not None:
            self.plasticity.detach()          # frozen conditions never run on learned weights
        decoder_before = self.decoder
        if condition == "fly-readout":
            self.decoder = self.enable_readout()
        if condition == "fly-mb":
            self.enable_mb()
            self.mb.attach()                  # learned KC->MBON weights + documented KC sparsening
            self.mb_active = True
            self.decoder = self.mb
            if self.plasticity is None:       # lets server.py's state() report the MB weight stats
                self.plasticity = MBStatsHook(self.mb)
        elif self.mb is not None:
            self.mb.detach()                  # every other condition runs on the frozen wiring
        try:
            return self._play(condition, seed, rng, act, blind, learning, t0)
        finally:
            self.decoder = decoder_before
            if self.mb_active:
                self.mb.end_game()
                self.mb_active = False

    def _play(self, condition, seed, rng, act, blind, learning, t0) -> GameRecord:
        rng = rng or np.random.default_rng(seed)
        game = Minesweeper(self.cfg.rows, self.cfg.cols, self.cfg.mines, seed=seed)
        self.game = game
        self.cursor = (self.cfg.rows // 2, self.cfg.cols // 2)
        self.encoder.reset()
        self.settle(blind)
        actions = {a: 0 for a in self.decoder.actions + ["hold"]}
        noop_reveals = reveals = holds = overrides = 0
        trace = []
        outcome = "timeout"
        mb = self.mb if self.mb_active else None
        for turn in range(self.cfg.max_turns):
            visible = game.visible()
            if mb is not None:
                mb.begin_turn(visible, self.cursor, self.cfg.mines)   # the helper's facts -> odor channels
            self._new_window()
            for _ in range(self.cfg.turn_steps):
                self._brain_step(visible, blind)
            self.last_rates = self.decoder.rates(self.cfg.turn_steps, self.sim.p.dt)
            decided = self.decoder.decide(rng, self.cfg.turn_steps, self.sim.p.dt)
            action = decided
            if act is not None:
                action = act(self, game, decided, rng)
                overrides += action != decided
            actions[action] = actions.get(action, 0) + 1
            trace.append(action)
            event = {"turn": turn, "action": action, "decided": decided}
            if action == "hold":
                holds += 1
            cursor_before = self.cursor
            result, newly = self.apply_action(game, action, rng, event)
            if action == "reveal":
                if result == "noop":
                    noop_reveals += 1
                else:
                    reveals += 1
                if learning:
                    if result == "mine":
                        self.plasticity.dopamine(-1.0, "mine")
                    elif result == "safe":
                        self.plasticity.dopamine(min(1.0, self.cfg.reward_scale * newly), f"safe+{newly}")
            if mb is not None and mb.learning:
                if self.teacher_act is not None:
                    mb.supervised(self.teacher_act(visible, cursor_before), decided)
                else:
                    r = mb.game_reward(action, result, newly, self.cursor != cursor_before, game.won)
                    event["reward"] = r
                    event["dopamine"] = mb.reward(action, r)
            if mb is not None:
                event["odor"] = mb.features.tolist()
                event["mb_probs"] = mb.last_probs.tolist()
            self.last_action = action
            self._emit(event)
            if game.over:
                outcome = "won" if game.won else "lost"
                break
        rec = GameRecord(
            condition=condition, seed=seed, outcome=outcome, turns=len(trace),
            safe_revealed=game.safe_revealed, total_safe=game.total_safe,
            cleared_fraction=round(game.cleared_fraction, 4), reveals=reveals,
            noop_reveals=noop_reveals, holds=holds, actions=actions,
            seconds=round(time.perf_counter() - t0, 2),
            plasticity=self.plasticity.summary() if learning else (mb.summary() if mb is not None else None),
            action_trace=trace, overrides=overrides,
        )
        if mb is not None and mb.learning:
            mb.games_trained += 1
        self._emit({"game_over": True, "record": asdict(rec)})
        return rec


def play_scripted(condition: str, seed: int, cfg: GameConfig) -> GameRecord:
    """Brain-free baselines with the same board, cadence and action bookkeeping."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    game = Minesweeper(cfg.rows, cfg.cols, cfg.mines, seed=seed)
    cursor = (cfg.rows // 2, cfg.cols // 2)
    acts = ["up", "down", "left", "right", "reveal", "jump"]
    actions = {a: 0 for a in acts + ["flag", "hold"]}
    noop_reveals = reveals = 0
    trace = []
    outcome = "timeout"
    oracle = Oracle() if condition == "fly-mb-oracle-only" else None
    for turn in range(cfg.max_turns):
        if condition in ("random-walk", "fly-mb-oracle-only"):
            if oracle is not None:
                action = oracle.rule_action(oracle.features(game.visible(), cursor, cfg.mines))
            else:
                action = acts[rng.integers(len(acts))]
            r, c = cursor
            if action == "up":
                cursor = (max(0, r - 1), c)
            elif action == "down":
                cursor = (min(cfg.rows - 1, r + 1), c)
            elif action == "left":
                cursor = (r, max(0, c - 1))
            elif action == "right":
                cursor = (r, min(cfg.cols - 1, c + 1))
            elif action == "jump":
                hidden = game.hidden_cells()
                if hidden:
                    cursor = hidden[int(rng.integers(len(hidden)))]
            else:
                result, _ = game.reveal(r, c)
                if result == "noop":
                    noop_reveals += 1
                else:
                    reveals += 1
        elif condition == "random-click":
            hidden = game.hidden_cells()
            r, c = hidden[rng.integers(len(hidden))]
            game.reveal(r, c)
            reveals += 1
            action = "reveal"
        elif condition == "solver":
            kind, r, c = solver_move(game, rng)
            if kind == "reveal":
                game.reveal(r, c)
                reveals += 1
            elif kind == "flag":
                game.toggle_flag(r, c)
            action = kind
        else:
            raise ValueError(condition)
        actions[action] = actions.get(action, 0) + 1
        trace.append(action)
        if game.over:
            outcome = "won" if game.won else "lost"
            break
    return GameRecord(
        condition=condition, seed=seed, outcome=outcome, turns=len(trace),
        safe_revealed=game.safe_revealed, total_safe=game.total_safe,
        cleared_fraction=round(game.cleared_fraction, 4), reveals=reveals,
        noop_reveals=noop_reveals, holds=0, actions=actions,
        seconds=round(time.perf_counter() - t0, 3), action_trace=trace,
    )
