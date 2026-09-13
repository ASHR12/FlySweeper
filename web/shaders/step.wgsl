// One fused simulation step: synaptic gather + leaky integrate-and-fire, mirroring flysweeper/sim.py:
//
//     I_j  = sum over in-edges (i -> j) of w_ij * spike_i(t-1)                 (gather, no atomics)
//     v_j <- v_j * exp(-dt/tau) + gain * I_j + tonic + noise_j + ext_j
//     spike_j(t) = v_j >= threshold;  if spike: v_j <- reset  else v_j <- max(v_j, floor)
//
// Work decomposition (CSR-vector): one 256-thread workgroup handles 8 neurons, 32 lanes per neuron.
// The 32 lanes stream the neuron's in-edge segment with coalesced reads, write partial sums to
// workgroup memory, and lane 0 finishes the sum in a fixed order (deterministic) and integrates.
// Neurons are visited in `order` (heaviest in-degree first) so the dispatch tail is cheap rows.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> indptr: array<u32>;          // n+1, in-edge CSR by post neuron
@group(0) @binding(2) var<storage, read> indices: array<u32>;         // E, presynaptic index
@group(0) @binding(3) var<storage, read> weights: array<f32>;         // E, signed normalized weight
@group(0) @binding(4) var<storage, read> nmeta: array<u32>;            // [0,n): visit order, [n,2n): pool id, [2n,3n): bias (f32 bits)
@group(0) @binding(5) var<storage, read_write> state: array<f32>;     // [0,n): v, [n,2n): external drive (self-clearing)
@group(0) @binding(6) var<storage, read_write> spikes: array<u32>;    // 2n, ping-pong 0/1 masks
@group(0) @binding(7) var<storage, read_write> activity: array<u32>;  // n, decayed spike counter for the brain map
@group(0) @binding(8) var<storage, read_write> stats: array<atomic<u32>>; // [0,16) pool counts, [16,48) spikes per batch slot

const LANES: u32 = 32u;
const ROWS: u32 = 8u;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn step(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let lane = lid.x & (LANES - 1u);
  let row = lid.x / LANES;
  let slot = wid.x * ROWS + row;
  let valid = slot < P.n;
  var j: u32 = 0u;
  var acc: f32 = 0.0;
  if (valid) {
    j = nmeta[slot];
    let a = indptr[j];
    let b = indptr[j + 1u];
    let prev = P.parity * P.n;
    for (var p = a + lane; p < b; p += LANES) {
      let i = indices[p];
      acc += weights[p] * f32(spikes[prev + i]);
    }
  }
  partial[lid.x] = acc;
  workgroupBarrier();
  if (lane == 0u && valid) {
    var s: f32 = 0.0;
    for (var k = 0u; k < LANES; k++) {
      s += partial[row * LANES + k];
    }
    // background kicks: Bernoulli(noise_rate*dt) * noise_amp, per neuron per step
    let u = hash_unit(j, P.step, P.seed);
    let noise = select(0.0, P.noise_amp, u < P.noise_p);
    // per-neuron bias (sim.py: ext += bias every step); zero for every cell unless a trained
    // weight set (fly-mb) sets kc_bias / pn_bias on Kenyon cells / antennal-lobe PNs
    let ext = state[P.n + j] + bitcast<f32>(nmeta[2u * P.n + j]);
    state[P.n + j] = 0.0;
    var vi = state[j] * P.decay + P.gain * s + P.tonic + noise + ext;
    var spk: u32 = 0u;
    if (vi >= P.threshold) {
      spk = 1u;
      vi = P.reset;
    } else {
      vi = max(vi, P.floor);
    }
    state[j] = vi;
    spikes[(1u - P.parity) * P.n + j] = spk;
    // brain-map activity, same bookkeeping as flysweeper/agent.py: every 5th step c -= c >> 1; fired += 8
    var c = activity[j];
    if (((P.step + 1u) % 5u) == 0u) {
      c -= c >> 1u;
    }
    c += spk * 8u;
    activity[j] = c;
    if (spk == 1u) {
      atomicAdd(&stats[POOL_SLOTS + P.slot], 1u);
      let pid = nmeta[P.n + j];
      if (pid != NO_POOL) {
        atomicAdd(&stats[pid], 1u);
      }
    }
  }
}
