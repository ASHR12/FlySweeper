# Contributing

Thanks for looking. A few house rules keep this project honest and reproducible.

## Ground rules

* **Anatomy vs. invention.** Anything that is not in the MaleCNS v1.0 release — a stimulus, a
  readout, a bias, a learning rule, a reward — is an engineering choice and must be labelled as
  one in the module docstring, in `docs/training.md` and, if user-visible, in the README table
  "What is anatomy and what we invented". Do not describe the model as a fly.
* **No claim without a held-out number.** Performance claims come from
  `python -m flysweeper.validate` on seeds that were never used for training (training seeds are
  ≥ 10000; held-out seeds are 5000–5029 and 5000–5099), with exploration off and every condition on
  the same boards. Put the report under `docs/results/` and link it.
* **Only KC→MBON synapses may be trained.** Every other edge keeps its connectome value in every
  condition. Any model change that applies to one condition only (like the `fly-mb` KC settings)
  must be reverted for the others in the same process and stated in the docs.
* **Negative results are kept.** Runs that did not work stay in `docs/training.md` with their
  numbers; do not delete them.

## Practicalities

* Python 3.11, `pip install -r requirements.txt`; the data pipeline is
  `prepare → compile_graph → (export_web)`; see the README quickstart.
* The browser app is vanilla JS + WGSL with no build step; keep `web/index.html` and
  `flysweeper/ui/index.html` visually in sync (their `<style>` blocks are identical by design).
* Weight files: one per training round in `models/`, never overwritten; add a row to
  `models/README.md` and a line to `models/SHA256SUMS`.
* Do not commit anything from `data/`, `outputs/` or `web/data/`, or any file larger than 5 MB.
  Do not commit absolute local paths (the validation reports print one; strip it when copying).
* Credit borrowed ideas in the README "Credits and citations" section; the ledger there is the
  source of truth.

## Reporting problems

Open an issue with the command you ran, the `NUMBA_NUM_THREADS` value, the seeds, and the
relevant `report.md` or console output. For the browser app include the Chrome version and the GPU
adapter shown in the stats strip.
