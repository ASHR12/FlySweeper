# Data licence and attribution

The code in this repository is MIT-licensed (see [`LICENSE`](LICENSE)). The neural wiring it
simulates is not ours. This file says where it comes from, what we redistribute, and what you must
do if you reuse it.

## Source dataset: MaleCNS v1.0

* **Dataset:** MaleCNS v1.0, the complete connectome of the *Drosophila melanogaster* male central
  nervous system (166,700 neurons, 25,582,938 directed connections, 124,177,617 synapses after the
  retention policies in `flysweeper/compile_graph.py`).
* **Publication:** Berg, S., Beckett, I. R., Costa, M., Schlegel, P., Januszewski, M., Marin, E. C.,
  Nern, A., Preibisch, S., Qiu, W., Takemura, S., … Jefferis, G. S. X. E. (2026). *Sexual dimorphism
  in the complete connectome of the Drosophila male central nervous system.* **Cell.**
  <https://doi.org/10.1016/j.cell.2026.08.015>
* **Produced by:** the Janelia FlyEM Project Team (HHMI Janelia Research Campus) with the Drosophila
  Connectomics Group (University of Cambridge) and collaborators; data portal
  <https://male-cns.janelia.org/>, downloads at <https://male-cns.janelia.org/download/>
  (bucket `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`).
* **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0),
  <https://creativecommons.org/licenses/by/4.0/>.

The three flat tables we consume, with the sizes and SHA-256 digests we verified against the bucket
on 2026-09-13 (also recorded in `flysweeper/prepare.py` and `data/compiled/meta.json`):

| file | bytes | sha256 |
|---|---|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 14,483,314 | `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2` |
| `body-neurotransmitters-male-cns-v1.0.feather` | 43,282,834 | `95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621` |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 1,051,241,946 | `e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1` |

## What this repository redistributes

**We do not redistribute the source tables.** `python -m flysweeper.prepare` downloads them from
the Janelia bucket onto your machine and verifies the digests above; `data/` is gitignored.

We redistribute only **derived artifacts**, which are adaptations of the dataset under CC BY 4.0:

| artifact | where | what it contains |
|---|---|---|
| trained KC→MBON weight sets | `models/*.npz` | for 16,167 (round 1) or 59,334 (rounds 2 and 3) Kenyon-cell → MBON edges: the edge's position in our compiled graph, its original normalized MaleCNS weight (`w0`) and its trained value (`w`), plus our training settings. No other connectome values. |
| the browser export | `web/data/` (gitignored; hosted separately, see `docs/deploy.md`) | the full signed, normalized adjacency (25,088,107 active edges after silencing synapses onto sensory neurons), cell-type ids, soma positions and region labels — a re-encoding of the released tables under our sign and normalization policies, not the original synapse tables |
| the browser copies of the weight sets | `web/weights/` | the same trained edges as `models/`, converted to JSON + binary for the page |
| counts, digests and policies | `data/compiled/meta.json` (regenerated locally), `docs/`, this file | numbers derived from the tables |

Everything derived keeps the dataset's neuron identities and cell-type names. The transmitter sign
convention, the per-cell normalization, the removal of synapses onto sensory neurons, the
input/output mappings and the trained weights are ours and are **not** part of the MaleCNS release;
do not attribute them to the dataset's authors.

## If you reuse the data or our derived files (CC BY 4.0 obligations)

1. **Attribution.** Credit the dataset and its authors: *"MaleCNS v1.0, Berg et al., Cell 2026,
   doi:10.1016/j.cell.2026.08.015, CC BY 4.0"*, with a link to the licence, in any copy,
   redistribution or derived work — including hosted copies of `web/data/` and of the files in
   `models/`.
2. **Indicate changes.** State that the data were modified (signs, normalization, silenced
   synapses, trained synapses), as this file does, and do not imply endorsement by the authors.
3. **Keep the notice.** Keep this file (or an equivalent statement) with any redistributed derived
   file.
4. **No additional restrictions.** Do not apply legal or technical measures that restrict others
   from doing what the licence permits.

If you also reuse our code or trained weights, cite this repository (see
[`CITATION.cff`](CITATION.cff)); that is a courtesy, not a licence condition.

## Not a fly

Nothing in this repository is a validated model of a fly. The wiring is the authors'; the
dynamics, sensors, buttons, learning rule and the game are engineered by us and are documented as
such in the [README](README.md) and [`docs/training.md`](docs/training.md).
