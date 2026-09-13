"""In-brain reinforcement learning restricted to Kenyon cell -> MBON synapses.

    a helper reads the board into a few facts; the fly's mushroom body learns what to do about them.

Pipeline (all inside the simulated MaleCNS brain except the helper):
    oracle.Oracle      board + cursor -> 31 channel values in [0, 1]          (helper, not the fly)
    OdorEncoder        channel k -> current into every cell of ORN type k     (real ORN cells)
    the connectome     ORN -> antennal-lobe PN -> Kenyon cells -> MBONs       (real wiring, LIF)
    MBPolicy.decide    6 MBON pools' spike counts over the turn -> softmax -> action
    MBPolicy.reward    game reward -> dopamine = r - baseline -> PAM (r>0) / PPL1 (r<0) cells are
                       stimulated, and the KC->MBON weights onto the 6 action pools move by a
                       reward-modulated three-factor rule.  Nothing else in the brain changes.

Documented departures from the raw wiring, needed for the mushroom body to carry an odor code at
all in this LIF preset (measured in `python -m flysweeper.mb_policy --probe`): with the raw
weights every Kenyon cell fires on every step (50 Hz) with or without odor, because 56% of a KC's
input weight is KC->KC recurrence and the antennal-lobe PNs have a dense 14 Hz spontaneous rate.
So, for the `fly-mb` condition only (restored for every other condition):
    kc_kc_gain   = 0.0    KC->KC synapses silenced (642,933 edges)
    kc_bias      = -0.20  a constant hyperpolarising bias on the 4,064 KCs (raises the effective threshold)
    pn_kc_gain   = 1.0    PN->KC weights unchanged
With these, KCs fire ~1.5 Hz without odor and ~5 Hz with odor, different channel sets produce
different KC populations (d' ~ 3-4 between disjoint channel sets, ~0.5 between repeats of the same
set), and MBONs rise from ~3 to ~9 Hz with odor.

Decision: per-cell spike count of each pool over the window, minus that pool's running mean
(center_scores; the pools differ 5x in intrinsic excitability, which otherwise fixes the argmax),
softmax with temperature (0 = argmax).

Learning rule (chosen action a, pool P(a); e is per KC->MBON edge (i -> j)):
    eligibility this turn   e_ij = (KC_i spikes in window - KC_i running mean)  x  (+1 if j in P(a), -other_credit otherwise)
                            (centered; with raw counts the component shared by all states dominates every update)
    trace                   z <- trace_decay * z + e           (credits the last few turns' actions)
    reward                  r = +0.1 per newly revealed safe cell (cap 1), -1 mine, -0.05 no-op
                            (reveal on a revealed cell or a move into the wall), +1 on a win
    dopamine                da = r - running mean of r;  PAM cells stimulated for 2 steps if da>0,
                            PPL1 cells if da<0 (amplitude 0.8 x |da|)
    weight update           w_ij <- clip(w_ij + eta * da * z_ij * |w0_ij|, 0.1 |w0|, 5 |w0|), sign kept
Supervised warm start (Ramp-style fallback): when the fly's choice differs from the teacher's, the
teacher's pool is credited (+1 x e) and the chosen pool debited (-1 x e), with a PAM pulse marking
the teaching event; agreement changes nothing (error-driven, so the weights do not drift).

Everything that changes is logged per game (total |dw|, edges changed, mean weight ratio per pool).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

from .brain import Brain
from .oracle import ACTIONS, CH, CHANNELS, N_CHANNELS, Oracle, OracleParams
from .paths import COMPILED
from .sim import LIF

DEFAULT_WEIGHTS = COMPILED / "mb_weights.npz"

# 31 ORN types with the most cells (both hemispheres), one per channel, in channel order
ORN_TYPES = (
    "ORN_DA1", "ORN_VA1d", "ORN_VA1v", "ORN_DL3", "ORN_VL2a", "ORN_VM5d", "ORN_VA2", "ORN_DL1", "ORN_VL1",
    "ORN_VM4", "ORN_DM1", "ORN_VA6", "ORN_DM3", "ORN_DL4", "ORN_DM6", "ORN_V", "ORN_DM2", "ORN_DA2",
    "ORN_VL2p", "ORN_DL5", "ORN_VM3", "ORN_VM2", "ORN_VC4", "ORN_VM7d", "ORN_DA4m", "ORN_DC3", "ORN_DM5",
    "ORN_VM6v", "ORN_VM5v", "ORN_VA4", "ORN_VC3",
)
assert len(ORN_TYPES) >= N_CHANNELS

# Round 2: the safety facts get a second ORN type each (more PNs -> a larger Kenyon-cell footprint),
# using types not in ORN_TYPES.  channel name -> extra ORN types.
EXTRA_ORN_TYPES = {
    "cursor_safe": ("ORN_DA3", "ORN_DP1l"), "cursor_mine": ("ORN_DA4l", "ORN_VM1"),
    "cursor_frontier": ("ORN_DC1", "ORN_DM4"), "cursor_free": ("ORN_VC2", "ORN_DP1m"),
    "guess_here": ("ORN_VC5",), "no_safe_known": ("ORN_VA3",),
}
# Round 3 (extra_orn level 2): the eight direction channels get a second ORN type each as well.
EXTRA_ORN_TYPES_L2 = {
    "safe_up": ("ORN_VC1",), "safe_down": ("ORN_VA7l",), "safe_left": ("ORN_D",), "safe_right": ("ORN_VM7v",),
    "guess_up": ("ORN_VM6m",), "guess_down": ("ORN_VA7m",), "guess_left": ("ORN_DL2v",), "guess_right": ("ORN_DC4",),
}

# Round 1 pools: one MBON type per action.
SINGLE_POOLS = (("up", ("MBON01",)), ("down", ("MBON05",)), ("left", ("MBON06",)),
                ("right", ("MBON03",)), ("reveal", ("MBON11",)), ("jump", ("MBON09",)))
# Round 2 pools: every MBON type with >= 20% of its input from KCs (34 types, 91 cells), dealt
# greedily by KC edge count into six groups of ~10k KC->MBON edges (10-23 cells each).
GROUP_POOLS = (
    ("up", ("MBON09", "MBON32", "MBON30", "MBON31", "MBON25-like", "MBON28")),
    ("down", ("MBON02", "MBON06", "MBON04", "MBON15-like", "MBON25", "MBON17-like")),
    ("left", ("MBON11", "MBON05", "MBON21", "MBON23", "MBON27")),
    ("right", ("MBON12", "MBON29", "MBON24", "MBON03", "MBON16")),
    ("reveal", ("MBON07", "MBON01", "MBON18", "MBON10", "MBON15", "MBON26")),
    ("jump", ("MBON14", "MBON22", "MBON20", "MBON19", "MBON13", "MBON17")),
)


@dataclass
class MBParams:
    # action -> MBON types (both hemispheres; chosen for large KC in-degree, see --probe)
    action_mbons: tuple = GROUP_POOLS
    extra_orn: int = 1             # 0: one ORN type per channel; 1: safety channels get extra types
                                   # (EXTRA_ORN_TYPES); 2: direction channels too (EXTRA_ORN_TYPES_L2)
    odor_amp: float = 1.0          # ORN current per step at channel value 1 (saturates ORNs at 50 Hz)
    kc_kc_gain: float = 0.0
    kc_bias: float = -0.20
    pn_kc_gain: float = 1.0
    pn_bias: float = 0.0           # optional: suppress the antennal-lobe PNs' spontaneous rate (documented)
    eta: float = 0.02
    other_credit: float = 0.2      # non-chosen pools receive -eta*da*e*other_credit
    w_min_ratio: float = 0.0       # silent synapse allowed
    w_max_ratio: float = 10.0
    sup_mode: str = "error"        # "error": on a mistake +1 teacher pool / -1 chosen pool (perceptron)
                                   # "teacher": every turn +1 teacher pool / -other_credit others
    reveal_penalty: float = 3.0    # extra debit when the fly chose "reveal" and the teacher did not
    reveal_margin: float = 0.0     # decoder engineering: reveal only if it beats the runner-up by this
                                   # (in centred spikes/cell); otherwise take the runner-up. Argmax mode only.
    mask_reveal: bool = False      # decoder engineering (round 3): "reveal" is not selectable when the
                                   # cursor cell is already revealed or provably a mine (greedy + exploration)
    epsilon: float = 0.0           # round 3 exploration: with prob epsilon pick uniformly among VALID actions
                                   # (used when temperature == 0)
    sup_move_weight: float = 1.0   # supervised: scale of the update when a move was confused with another move
    reward_scheme: str = "r1"      # "r1": round-1 rewards (r_safe per cell capped); "r3": see game_reward
    r_progress: float = 0.3        # r3: per successful reveal decision on a provably-safe cell or a forced guess
    r_guess: float = -0.3          # r3: unproven reveal while a provably-safe cell exists (even if it succeeds)
    r_wasted: float = -0.1         # r3: hold, no-op reveal, move into a wall
    trace_decay: float = 0.5       # per-turn eligibility decay
    baseline_alpha: float = 0.05   # running-mean reward
    centered: bool = True          # eligibility uses (KC count - KC running mean): the shared component
    kc_mean_alpha: float = 0.05    # of KC activity cancels, only odor-specific deviations carry credit
    temperature: float = 1.0       # softmax temperature over per-cell pool counts; 0 = argmax
    center_scores: bool = True     # decision uses (pool count - pool running mean): pools differ a lot in
    score_mean_alpha: float = 0.02 # intrinsic excitability (1.4 vs 6.8 spikes/cell), which otherwise fixes the argmax
    dopamine_amount: float = 0.8
    dopamine_steps: int = 2
    r_safe: float = 0.1
    r_cap: float = 1.0
    r_mine: float = -1.0
    r_noop: float = -0.05
    r_win: float = 1.0
    jump_distance: int = 5


class OdorEncoder:
    """Channel k of the helper's vector drives every cell of ORN type k with current amp * value."""

    def __init__(self, brain: Brain, amp: float, extra: int = 1):
        self.amp = amp
        extra = int(extra)
        self.types = [(t,) + (EXTRA_ORN_TYPES.get(name, ()) if extra >= 1 else ())
                      + (EXTRA_ORN_TYPES_L2.get(name, ()) if extra >= 2 else ()) for name, t in zip(CHANNELS, ORN_TYPES)]
        self.cells = [np.concatenate([brain.cells(types=t) for t in ts]) for ts in self.types]
        self.sizes = np.array([len(c) for c in self.cells])
        self.all_idx = np.concatenate(self.cells)
        self.owner = np.repeat(np.arange(N_CHANNELS), self.sizes)
        self.current = np.zeros(len(self.all_idx), dtype=np.float32)

    def set(self, values: np.ndarray) -> None:
        self.current = (self.amp * np.asarray(values, dtype=np.float32))[self.owner]

    def inject(self, sim: LIF) -> None:
        if self.current.any():
            sim.inject(self.all_idx, self.current)

    def describe(self) -> dict:
        return {name: {"orn_types": list(t), "cells": int(n)} for name, t, n in zip(CHANNELS, self.types, self.sizes)}


class MBPolicy:
    """Decoder-compatible (observe / reset_window / rates / decide / set_idle / actions / pool_idx /
    idle_rates) so FlyPlayer can use it in place of the descending-neuron PoolDecoder."""

    def __init__(self, brain: Brain, sim: LIF, params: MBParams | None = None):
        self.p = params or MBParams()
        self.brain, self.sim = brain, sim
        if sim.data is brain.graph.data:          # never write into the shared compiled graph
            sim.data = sim.data.copy()
        self.actions = list(ACTIONS)
        self.oracle = Oracle(OracleParams(jump_distance=self.p.jump_distance))
        self.odor = OdorEncoder(brain, self.p.odor_amp, self.p.extra_orn)
        g = brain.graph
        n = brain.n
        # ---- pools (each action: one or more MBON types)
        mb = {a: ((t,) if isinstance(t, str) else tuple(t)) for a, t in self.p.action_mbons}
        self.p.action_mbons = tuple((a, mb[a]) for a in self.actions)
        self.pool_idx = {a: np.concatenate([brain.cells(types=t) for t in mb[a]]) for a in self.actions}
        self.sizes = np.array([len(self.pool_idx[a]) for a in self.actions], dtype=np.float32)
        self.pool_of = np.full(n, -1, dtype=np.int64)
        for k, a in enumerate(self.actions):
            self.pool_of[self.pool_idx[a]] = k
        # ---- plastic edges: KC -> action-pool MBON
        kc = brain.cells(type_prefix="KC")
        self.kc = kc
        self.is_kc = np.zeros(n, dtype=bool)
        self.is_kc[kc] = True
        pre = np.repeat(np.arange(n), np.diff(g.indptr))
        sel = self.is_kc[pre] & (self.pool_of[g.indices] >= 0)
        self.edge_pos = np.flatnonzero(sel).astype(np.int64)
        self.edge_kc = pre[self.edge_pos]                       # presynaptic KC (neuron id)
        self.edge_pool = self.pool_of[g.indices[self.edge_pos]]  # 0..5
        self.w_frozen = sim.data[self.edge_pos].copy()
        self.w0 = np.abs(self.w_frozen).astype(np.float32)
        self.sign = np.sign(self.w_frozen).astype(np.float32)
        self.sign[self.sign == 0] = 1.0
        self.w_learned = self.w_frozen.copy()
        # ---- documented sim modifications (applied by attach, reverted by detach)
        onto_kc = self.is_kc[g.indices]
        self.kc2kc_pos = np.flatnonzero(onto_kc & self.is_kc[pre]).astype(np.int64)
        self.pn = brain.cells(cls="ALPN")
        ispn = np.zeros(n, dtype=bool)
        ispn[self.pn] = True
        self.pn2kc_pos = np.flatnonzero(onto_kc & ispn[pre]).astype(np.int64)
        self.kc2kc_frozen = sim.data[self.kc2kc_pos].copy()
        self.pn2kc_frozen = sim.data[self.pn2kc_pos].copy()
        self.kc_bias_frozen = sim.bias[kc].copy()
        self.pn_bias_frozen = sim.bias[self.pn].copy()
        self.attached = False
        # ---- dopamine cells
        self.pam = brain.cells(type_prefix="PAM")
        self.ppl1 = brain.cells(type_prefix="PPL1")
        self.pending: list[tuple[int, float]] = []
        # ---- state
        self.kc_counts = np.zeros(n, dtype=np.int32)
        self.kc_mean = np.zeros(n, dtype=np.float32)          # running mean window count per KC
        self.kc_mean_ready = False
        self.counts = np.zeros(len(self.actions), dtype=np.int64)
        self.idle_rates = np.zeros(len(self.actions), dtype=np.float32)
        self.last_scores = np.zeros(len(self.actions), dtype=np.float32)
        self.score_mean = np.zeros(len(self.actions), dtype=np.float32)
        self.score_mean_ready = False
        self.last_probs = np.full(len(self.actions), 1 / len(self.actions), dtype=np.float32)
        self.trace = np.zeros(len(self.edge_pos), dtype=np.float32)
        self.baseline = 0.0
        self.learning = False
        self.shuffle_reward = False
        self.shuffle_rng = np.random.default_rng(0)
        self.features = np.zeros(N_CHANNELS, dtype=np.float32)
        self.turns_seen = 0
        self.games_trained = 0
        self.game_log: dict = self._fresh_game_log()
        self.total_abs_delta = 0.0
        self.n_events = 0
        self.meta: dict = {}

    # ------------------------------------------------------------------ attach / detach
    def attach(self) -> None:
        if self.attached:
            return
        s = self.sim
        s.data[self.kc2kc_pos] = self.kc2kc_frozen * self.p.kc_kc_gain
        s.data[self.pn2kc_pos] = self.pn2kc_frozen * self.p.pn_kc_gain
        s.bias[self.kc] = self.kc_bias_frozen + self.p.kc_bias
        s.bias[self.pn] = self.pn_bias_frozen + self.p.pn_bias
        s.data[self.edge_pos] = self.w_learned
        self.attached = True

    def detach(self) -> None:
        if not self.attached:
            return
        s = self.sim
        self.w_learned = s.data[self.edge_pos].copy()
        s.data[self.edge_pos] = self.w_frozen
        s.data[self.kc2kc_pos] = self.kc2kc_frozen
        s.data[self.pn2kc_pos] = self.pn2kc_frozen
        s.bias[self.kc] = self.kc_bias_frozen
        s.bias[self.pn] = self.pn_bias_frozen
        self.attached = False

    # ------------------------------------------------------------------ per-turn interface
    def begin_turn(self, visible: np.ndarray, cursor: tuple[int, int], n_mines: int) -> None:
        self.features = self.oracle.features(visible, cursor, n_mines)
        self.odor.set(self.features)
        self.turns_seen += 1

    def end_game(self) -> None:
        self.odor.set(np.zeros(N_CHANNELS, dtype=np.float32))
        self.trace[:] = 0.0

    def before_step(self) -> None:
        """Odor current for this step plus any scheduled dopamine stimulus."""
        self.odor.inject(self.sim)
        if self.pending:
            keep = []
            for steps_left, da in self.pending:
                cells = self.pam if da > 0 else self.ppl1
                self.sim.inject(cells, self.p.dopamine_amount * min(1.0, abs(da)))
                if steps_left - 1 > 0:
                    keep.append((steps_left - 1, da))
            self.pending = keep

    def reset_window(self) -> None:
        self.counts[:] = 0
        self.kc_counts[:] = 0

    def observe(self, fired: np.ndarray) -> None:
        kcs = fired[self.is_kc[fired]]
        self.kc_counts[kcs] += 1
        pools = self.pool_of[fired]
        pools = pools[pools >= 0]
        if len(pools):
            self.counts += np.bincount(pools, minlength=len(self.actions))

    def rates(self, steps: int, dt: float) -> np.ndarray:
        return self.counts / self.sizes / (steps * dt)

    def set_idle(self, idle_rates: np.ndarray) -> None:
        self.idle_rates = np.asarray(idle_rates, dtype=np.float32)

    def decide(self, rng: np.random.Generator, steps: int, dt: float) -> str:
        raw = (self.counts / self.sizes).astype(np.float32)         # spikes per cell this window
        if self.p.center_scores:
            if not self.score_mean_ready:
                self.score_mean[:] = raw
                self.score_mean_ready = True
            scores = raw - self.score_mean
            self.score_mean += self.p.score_mean_alpha * (raw - self.score_mean)
        else:
            scores = raw
        self.last_scores = scores
        k_rev = self.actions.index("reveal")
        valid = np.ones(len(self.actions), dtype=bool)
        if self.p.mask_reveal and (self.features[CH["cursor_revealed"]] > 0 or self.features[CH["cursor_mine"]] > 0):
            valid[k_rev] = False                        # decoder engineering: impossible / suicidal reveal masked
        scores = np.where(valid, scores, -np.inf).astype(np.float32)
        T = self.p.temperature
        if T <= 0:
            if self.p.epsilon > 0 and rng.random() < self.p.epsilon:
                cand = np.flatnonzero(valid)
                self.last_probs = np.zeros(len(self.actions), dtype=np.float32)
                self.last_probs[cand] = 1 / len(cand)
                return self.actions[int(rng.choice(cand))]
            order = np.argsort(-scores, kind="stable")
            dropped = False
            if self.p.reveal_margin > 0 and order[0] == k_rev and scores[k_rev] - scores[order[1]] < self.p.reveal_margin:
                order, dropped = order[1:], True        # decoder engineering: not confident enough to reveal
            best = np.flatnonzero(scores == scores[order[0]])
            if dropped:
                best = best[best != k_rev]
            self.last_probs = np.zeros_like(scores)
            self.last_probs[best] = 1 / len(best)
            return self.actions[int(rng.choice(best))]
        z = (scores - scores.max()) / T
        q = np.exp(z)
        q /= q.sum()
        self.last_probs = q
        return self.actions[int(rng.choice(len(self.actions), p=q))]

    # ------------------------------------------------------------------ learning
    def game_reward(self, action: str, result: str, newly: int, moved: bool, won: bool) -> float:
        p = self.p
        if p.reward_scheme == "r3":
            return self._game_reward_r3(action, result, moved, won)
        if action == "reveal":
            if result == "mine":
                r = p.r_mine
            elif result == "noop":
                r = p.r_noop
            else:
                r = min(p.r_cap, p.r_safe * newly)
        elif action in ("up", "down", "left", "right"):
            r = 0.0 if moved else p.r_noop
        else:
            r = 0.0
        if won:
            r += p.r_win
        return float(r)

    def _game_reward_r3(self, action: str, result: str, moved: bool, won: bool) -> float:
        """Round-3 rewards (docs/rl_references.md): +r_progress per successful reveal *decision* on a
        provably-safe cell or a forced guess (no provable move anywhere); r_guess for an unproven
        reveal while a provably-safe cell exists, even if it succeeds; r_mine; r_wasted for hold /
        no-op reveal / move into a wall; r_win on a win.  Uses the helper's facts for this turn."""
        p = self.p
        f = self.features
        if action == "reveal":
            if result == "mine":
                r = p.r_mine
            elif result == "noop":
                r = p.r_wasted
            elif f[CH["cursor_safe"]] > 0 or f[CH["no_safe_known"]] > 0 or f[CH["untouched"]] > 0:
                r = p.r_progress
            else:
                r = p.r_guess
        elif action in ("up", "down", "left", "right"):
            r = 0.0 if moved else p.r_wasted
        elif action == "hold":
            r = p.r_wasted
        else:
            r = 0.0
        if won:
            r += p.r_win
        return float(r)

    def _kc_activity(self) -> np.ndarray:
        """Per-edge presynaptic activity this window: raw KC counts, or (default) counts minus each
        KC's running mean, then the running mean is updated."""
        x = self.kc_counts.astype(np.float32)
        if not self.p.centered:
            return x[self.edge_kc]
        if not self.kc_mean_ready:
            self.kc_mean[self.kc] = x[self.kc]
            self.kc_mean_ready = True
        dev = x - self.kc_mean
        self.kc_mean[self.kc] += self.p.kc_mean_alpha * dev[self.kc]
        return dev[self.edge_kc]

    def _eligibility(self, action: str) -> np.ndarray:
        k = self.actions.index(action)
        credit = np.where(self.edge_pool == k, 1.0, -self.p.other_credit).astype(np.float32)
        return self._kc_activity() * credit

    def _apply(self, da: float, z: np.ndarray, note: str) -> None:
        w = self.sim.data[self.edge_pos]
        new = np.clip(np.abs(w) + self.p.eta * da * z * self.w0, self.p.w_min_ratio * self.w0, self.p.w_max_ratio * self.w0) * self.sign
        real = new - w
        self.sim.data[self.edge_pos] = new
        a = float(np.abs(real).sum())
        self.total_abs_delta += a
        self.n_events += 1
        gl = self.game_log
        gl["events"] += 1
        gl["abs_delta"] += a
        gl["edges_changed"] += int((real != 0).sum())
        gl["sum_da"] += da
        gl["sum_abs_da"] += abs(da)

    def reward(self, action: str, r: float) -> float:
        """Deliver the reward physically (PAM / PPL1) and update the plastic weights.  Returns da."""
        if self.shuffle_reward and r != 0.0:
            r = float(r * self.shuffle_rng.choice([-1.0, 1.0]))
        da = r - self.baseline
        self.baseline += self.p.baseline_alpha * (r - self.baseline)
        self.trace = self.p.trace_decay * self.trace + self._eligibility(action)
        self.game_log["reward"] += r
        if abs(da) > 1e-9:
            self.pending.append((self.p.dopamine_steps, da))
            self._apply(da, self.trace, "rl")
        return da

    def supervised(self, teacher_action: str, chosen_action: str) -> None:
        """Warm start (error-driven, so the weights do not drift): when the fly's choice differs from
        the teacher's, the teacher's pool is credited (+1 x KC counts) and the chosen pool debited
        (-1 x KC counts); a PAM dopamine pulse marks the teaching event.  Agreement -> no change."""
        act = self._kc_activity()          # always consume the window (keeps the running mean current)
        kt, kc = self.actions.index(teacher_action), self.actions.index(chosen_action)
        k_rev = self.actions.index("reveal")
        if self.p.sup_mode == "teacher":
            # every turn: credit the teacher's pool, debit the others a little; a wrong "reveal" a lot
            credit = np.where(self.edge_pool == kt, 1.0, -self.p.other_credit).astype(np.float32)
            if kt != k_rev:
                credit[self.edge_pool == k_rev] = -self.p.reveal_penalty * self.p.other_credit
        else:
            if teacher_action == chosen_action:
                return
            debit = self.p.reveal_penalty if kc == k_rev else 1.0
            credit = np.where(self.edge_pool == kt, 1.0, np.where(self.edge_pool == kc, -debit, 0.0)).astype(np.float32)
            moves = ("up", "down", "left", "right")
            if teacher_action in moves and chosen_action in moves:
                credit *= self.p.sup_move_weight
        self.pending.append((self.p.dopamine_steps, 1.0))
        self.game_log["supervised"] += 1
        self._apply(1.0, act * credit, "teacher")

    # ------------------------------------------------------------------ bookkeeping
    def _fresh_game_log(self) -> dict:
        return {"events": 0, "supervised": 0, "abs_delta": 0.0, "edges_changed": 0, "sum_da": 0.0, "sum_abs_da": 0.0, "reward": 0.0}

    def pool_ratios(self) -> dict:
        w = np.abs(self.sim.data[self.edge_pos] if self.attached else self.w_learned) / self.w0
        return {a: round(float(w[self.edge_pool == k].mean()), 4) for k, a in enumerate(self.actions)}

    def summary(self) -> dict:
        w = np.abs(self.sim.data[self.edge_pos] if self.attached else self.w_learned)
        ratio = w / self.w0
        out = {
            "kc_mbon_edges": int(len(self.edge_pos)),
            "events": int(self.n_events),
            "total_abs_delta": float(self.total_abs_delta),
            "edges_changed": int((np.abs(w - self.w0) > 1e-7).sum()),
            "attached": bool(self.attached),
            "mean_weight_ratio": float(ratio.mean()),
            "min_weight_ratio": float(ratio.min()),
            "max_weight_ratio": float(ratio.max()),
            "edges_at_floor": int((ratio <= self.p.w_min_ratio + 1e-6).sum()),
            "edges_at_ceiling": int((ratio >= self.p.w_max_ratio - 1e-6).sum()),
            "pool_ratio": self.pool_ratios(),
            "baseline": float(self.baseline),
            "games_trained": int(self.games_trained),
            "game": dict(self.game_log),
        }
        self.game_log = self._fresh_game_log()
        return out

    def save(self, path: str | Path, meta: dict | None = None) -> None:
        w = self.sim.data[self.edge_pos].copy() if self.attached else self.w_learned
        np.savez_compressed(
            path, edge_pos=self.edge_pos, w=w.astype(np.float32), w0=self.w0, edge_pool=self.edge_pool.astype(np.int8),
            edge_kc=self.edge_kc, actions=np.array(self.actions), baseline=np.array([self.baseline]), score_mean=self.score_mean,
            games_trained=np.array([self.games_trained]),
            params=json.dumps(asdict(self.p)), meta=json.dumps({**self.meta, **(meta or {})}),
            channels=np.array(CHANNELS), orn_types=np.array(["+".join(t) for t in self.odor.types]),
        )

    @staticmethod
    def params_from_file(path: str | Path, override: MBParams | None = None) -> MBParams:
        """The structural settings (pools, odor mapping, KC changes) a weights file was trained with,
        so a policy can be built that matches it; other settings come from `override` / defaults."""
        z = np.load(path, allow_pickle=False)
        saved = json.loads(str(z["params"])) if "params" in z else {}
        p = MBParams(**asdict(override)) if override is not None else MBParams()
        if "action_mbons" in saved:
            p.action_mbons = tuple((a, tuple(t) if not isinstance(t, str) else (t,)) for a, t in saved["action_mbons"])
        p.extra_orn = int(saved.get("extra_orn", 0))
        for k in ("kc_kc_gain", "kc_bias", "pn_kc_gain", "pn_bias", "odor_amp", "center_scores", "score_mean_alpha", "reveal_margin", "mask_reveal"):
            if k in saved:
                setattr(p, k, saved[k])
        if saved and "center_scores" not in saved:
            p.center_scores = False
        return p

    @classmethod
    def from_file(cls, brain: Brain, sim: LIF, path: str | Path, override: MBParams | None = None) -> "MBPolicy":
        pol = cls(brain, sim, cls.params_from_file(path, override))
        pol.load(path)
        return pol

    def load(self, path: str | Path) -> dict:
        z = np.load(path, allow_pickle=False)
        if len(z["edge_pos"]) != len(self.edge_pos) or not np.array_equal(z["edge_pos"], self.edge_pos):
            raise ValueError(f"{path}: plastic edge set does not match this brain / action-MBON assignment "
                             f"(build the policy with MBPolicy.from_file)")
        self.w_learned = z["w"].astype(self.sim.data.dtype)
        if self.attached:
            self.sim.data[self.edge_pos] = self.w_learned
        self.baseline = float(z["baseline"][0])
        self.games_trained = int(z["games_trained"][0])
        self.meta = json.loads(str(z["meta"])) if "meta" in z else {}
        saved = json.loads(str(z["params"])) if "params" in z else {}
        # the decision-side settings the weights were trained under travel with the weights
        for k in ("center_scores", "score_mean_alpha", "kc_kc_gain", "kc_bias", "pn_kc_gain", "pn_bias", "odor_amp", "reveal_margin", "mask_reveal"):
            if k in saved:
                setattr(self.p, k, saved[k])
        if saved and int(saved.get("extra_orn", 0)) != int(self.p.extra_orn):
            self.p.extra_orn = int(saved["extra_orn"])
            self.odor = OdorEncoder(self.brain, self.p.odor_amp, self.p.extra_orn)
        if saved and "center_scores" not in saved:
            self.p.center_scores = False      # older files were trained on raw pool counts
        if "score_mean" in z and self.p.center_scores:
            self.score_mean[:] = z["score_mean"]
            self.score_mean_ready = True
        return self.meta

    def set_odor_level(self, level: int) -> None:
        """Change the channel -> ORN-type mapping (extra_orn level); the plastic edge set is unchanged."""
        self.p.extra_orn = int(level)
        self.odor = OdorEncoder(self.brain, self.p.odor_amp, self.p.extra_orn)

    def describe(self) -> dict:
        return {
            "action_mbons": dict(self.p.action_mbons),
            "pool_cells": {a: int(len(self.pool_idx[a])) for a in self.actions},
            "plastic_edges": int(len(self.edge_pos)),
            "plastic_edges_per_pool": {a: int((self.edge_pool == k).sum()) for k, a in enumerate(self.actions)},
            "kc_cells": int(len(self.kc)), "kc_kc_edges_silenced": int(len(self.kc2kc_pos)),
            "pn_kc_edges": int(len(self.pn2kc_pos)), "kc_bias": self.p.kc_bias, "kc_kc_gain": self.p.kc_kc_gain,
            "pn_kc_gain": self.p.pn_kc_gain, "pn_bias": self.p.pn_bias, "pn_cells": int(len(self.pn)), "odor": self.odor.describe(), "pam_cells": int(len(self.pam)), "ppl1_cells": int(len(self.ppl1)),
            "games_trained": int(self.games_trained),
        }


# ---------------------------------------------------------------------------- calibration probe
def probe(brain: Brain | None = None, params: MBParams | None = None, seed: int = 0) -> dict:
    """ORN -> PN -> KC -> MBON pathway check, raw wiring vs the fly-mb modifications."""
    from .sim import Params
    brain = brain or Brain()
    p = params or MBParams()
    types = brain.neurons["type"].fillna("").to_numpy().astype(str)
    kc = brain.cells(type_prefix="KC")
    pn = brain.cells(cls="ALPN")
    mbon = brain.cells(cls="MBON")
    dm1_pn = np.flatnonzero(np.char.startswith(types, "DM1_"))
    dm1 = brain.cells(types="ORN_DM1")
    setA = np.zeros(N_CHANNELS, np.float32); setA[0::2] = 1.0
    setB = np.zeros(N_CHANNELS, np.float32); setB[1::2] = 1.0

    def run(modified: bool, values: np.ndarray | None, single: bool = False, steps: int = 15, reps: int = 8, warm: int = 40, s: int = seed):
        sim = LIF(brain, Params.preset("flyai"), seed=s, sensory_input=False)
        pol = MBPolicy(brain, sim, p)
        if modified:
            pol.attach()
        if values is not None:
            pol.odor.set(values)
        wins = []
        for k in range(warm + reps * steps):
            if single:
                sim.inject(dm1, p.odor_amp)
            pol.before_step()
            f = sim.step()
            if k >= warm:
                if (k - warm) % steps == 0:
                    wins.append(np.zeros(brain.n, np.int32))
                wins[-1][f] += 1
        return np.array(wins), pol

    def hz(W, idx, steps=15):
        return float(W[:, idx].mean() / (steps * 0.02))

    out = {}
    for modified in (False, True):
        tag = "fly-mb" if modified else "raw"
        Wb, pol = run(modified, None)
        Wd, _ = run(modified, None, single=True)
        WA, _ = run(modified, setA)
        WB, _ = run(modified, setB)
        WA2, _ = run(modified, setA, s=seed + 1)
        mA, mB, mA2 = WA[:, kc].mean(0), WB[:, kc].mean(0), WA2[:, kc].mean(0)
        within = 0.5 * (np.linalg.norm(WA[:, kc] - mA, axis=1).mean() + np.linalg.norm(WB[:, kc] - mB, axis=1).mean())
        out[tag] = {
            "blind": {"PN_Hz": hz(Wb, pn), "KC_Hz": hz(Wb, kc), "KC_active_frac": float((Wb[:, kc].sum(0) > 0).mean()), "MBON_Hz": hz(Wb, mbon)},
            "ORN_DM1_only": {"ORN_DM1_Hz": hz(Wd, dm1), "DM1_PN_Hz": hz(Wd, dm1_pn), "all_PN_Hz": hz(Wd, pn), "KC_Hz": hz(Wd, kc), "MBON_Hz": hz(Wd, mbon)},
            "odor_setA": {"PN_Hz": hz(WA, pn), "KC_Hz": hz(WA, kc), "KC_active_frac": float((WA[:, kc].sum(0) > 0).mean()), "MBON_Hz": hz(WA, mbon),
                          "pool_spikes_per_cell": {a: float(WA[:, pol.pool_idx[a]].sum(1).mean() / len(pol.pool_idx[a])) for a in pol.actions}},
            "kc_code": {"dprime_A_vs_B": float(np.linalg.norm(mA - mB) / max(within, 1e-9)),
                        "dprime_A_vs_A_other_seed": float(np.linalg.norm(mA - mA2) / max(within, 1e-9)),
                        "corr_A_B": float(np.corrcoef(mA, mB)[0, 1])},
        }
    out["design"] = MBPolicy(brain, LIF(brain, Params.preset("flyai"), seed=0, sensory_input=False), p).describe()
    return out


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", action="store_true", help="run the ORN->PN->KC->MBON calibration probe")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.probe:
        res = probe()
        txt = json.dumps(res, indent=1)
        print(txt)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
