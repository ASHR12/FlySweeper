# Minesweeper-RL references → what transfers to the fly-connectome agent

**Sources.** sdlee94: page, repo code and reward/TensorBoard figures. MDPI (Wang & Lei 2025): full text with tables. Science Buddies: page blocked by a Cloudflare bot check (no Wayback/Common Crawl copy); details from its official companion repo `science-buddies/reinforcement_learning_minesweeper`; anything only on the page (e.g., results) is missing.

## (a) Comparison

| Field | sdlee94 (2020) | Science Buddies repo (2026) | MDPI Wang & Lei 2025 |
|---|---|---|---|
| State | Full 9×9 grid, 1 channel: 0–8, hidden −1, mine −2, ÷8 | Full grid ints (hidden −3, 0–8, flag −2, mine −1) → 13-ch one-hot | Full 6×6; 10-ch one-hot (0–8 + hidden) beat 1-ch/2-ch |
| Actions | Any of 81 cells; revealed cells masked (Q→min) in greedy and random branches | Any cell × {reveal, flag} = 72; hidden-only mask in greedy and random branches | Click any of 36; no mask |
| Rewards | win +1, lose −1, progress +0.3 (≥1 revealed neighbour), guess −0.3 (all 8 neighbours hidden, even if safe), no_progress −0.3 (unreachable). No solver-based penalty | Raw: win +100 (+efficiency bonus), mine −20, safe reveal +1, correct flag +2, wrong flag −1; ÷50, clipped [−1,1] → win +1, mine −0.4, reveal +0.02. No guess penalty | win +36 (m×n), loss −36, progress +1, guess −0.5 (all 8 neighbours hidden, even if safe), YOLO (click revealed) −0.5. No solver-based penalty |
| Algorithm | Double DQN; replay 50k; batch 64; γ=0.1 (Hansen 2017: γ=0 > 0.99) | DQN; replay 250k; batch 64; Huber; soft target; γ=0.95 | DQN (replay 300k, batch 128, γ=0.9; γ=0 also worked) + supervised CNN predicting per-cell mine probability (play = argmin) |
| Network | Conv128×4 + FC512×2 → 81 | Conv64 + 3 residual conv64 + global pool + FC256×2 → 72 | Conv64×3 + FC 2304→256→256→36; 676,804 params |
| Training | "~500k games" (demo); sweeps 60k–115k games | Default `--episodes 100`; no results in repo | DQN ≈1.5M steps (~8 h RTX 3090); SL ≈0.5M steps |
| Board | 9×9, 10 mines | 6×6, 3 mines | 6×6, 4 mines |
| Win rate | Not stated; TensorBoard figure ≈10–11 % after 60k–115k games | Not reported | DQN 93.3 ± 0.8 %, SL 91.2 ± 0.9 %; SP 78 %, CSP 84–90 %, humans 87 % |
| Tricks | Mask revealed; first-click mine relocated; ε 0.95→0.01 (×0.99975/step); no curriculum | Mask revealed/flagged; lazy mines leave a 3×3 mine-free opening at first click; ε 1.0→0.05 (×0.9999/step); `curriculum` flag unimplemented | No mask (YOLO penalty); first-click safety, ε schedule not given; curriculum and "SL first, RL fine-tune" only proposed |

## (b) Prioritized changes for our setup

1. **Penalize an unproven reveal even when it succeeds: −0.3** whenever the cursor cell is not provably safe and a provably-safe cell exists (sdlee94 guess −0.3 = |progress|; MDPI −0.5 vs +1). Under our current scheme a random isolated reveal on 9×9/10 has positive expectation (≈−0.12 from mines vs ≈+0.3 from cascades capped at 1). Refs use isolation as a solver-free proxy for "guess". Leave forced guesses (no provable move anywhere) unpenalized.
2. **Progress = +0.3 per reveal decision, not +0.1 per cell capped at 1** (sdlee94 0.3 vs win 1; MDPI +1 vs ±36; SB +0.02 vs +1). No non-winning move should be worth a win.
3. **Mask invalid `reveal`** (cell revealed or provably mine) at the MBON readout, in greedy and exploratory branches alike (sdlee94, SB; sdlee94 reports faster training). Fallback: MDPI's YOLO −0.5; our −0.05 is too weak against +0.1/cell.
4. **Guaranteed opening on first reveal**: lazy mine placement excluding a 3×3 block around the first click (SB) or relocating a first-click mine (sdlee94). Removes the 10/81 first-move loss and guarantees a frontier from move 1.
5. **Short credit horizon**: γ=0.1 (sdlee94), γ=0 viable (MDPI, Hansen 2017). Let the eligibility trace span one reveal cycle (~3–5 cursor moves), not the game.
6. **Teacher first, reward second** (MDPI: same net, SL took 3× fewer steps and was more stable; recommends SL-led training, then RL fine-tuning). Apply reward learning only to the teacher-initialised policy, with 1–3 in place.
7. **Curriculum** (MDPI future work; SB `curriculum` flag): 6×6/3 → 6×6/4 → 9×9/10, promoting at ≥50 % wins.
8. **Exploration only over valid actions, annealed fast** (sdlee94 ε→0.01 in ≈18k steps ≈1–2k games; SB →0.05 in ≈30k steps). Exploring `reveal` on unproven cells is just losing.

## (c) What does not transfer
- Full-board CNNs with one output per cell (36–81 actions, 0.7M+ parameters): our plastic capacity is a linear map from 31 symbolic facts to 6 actions; spatial inference lives in the helper.
- Budgets of 60k–500k games (sdlee94) or 1.5M steps (MDPI) spent *discovering* Minesweeper logic; we have ~10³ games.
- DQN stabilisers (replay, target nets, Huber, clipping) fix bootstrapped TD in deep nets; our rule has no value function. Only the γ→trace lesson carries over.
- Flag actions (SB): redundant with our "provably mine" fact. MDPI's per-cell mine-probability head: its usable content (act = argmin mine probability) is already our teacher.

## (d) Realistic expectation
- Reward-only from scratch: ~0–5 % on 9×9/10 within thousands of games even after fixes (sdlee94's CNN was at ≈10 % after 60k games).
- Teacher-initialised + fixes: ceiling = the solver behind "provably safe" (MDPI's cited baselines: single-point ≈75 % on beginner, CSP ≈91 %) minus forced guesses. With ~25 reveal decisions per game and unproven-reveal rate p, the chance of a guess-free game is (1−p)²⁵: p=12 % → 4 %, 5 % → 28 %, 2 % → 60 % (lucky guesses add a little; 0 wins at p=12 % hints at further failure modes, e.g., wasted-move loops, worth measuring). So 20–40 % on 9×9/10 is plausible in a few thousand games only if item 1 drives p below ~3 %; 50 %+ on 6×6/3–4 is the realistic first milestone.
