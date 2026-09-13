// Shared declarations, prepended to every kernel by gpu.js.
//
// One Params block per simulation step lives in a uniform buffer (256-byte stride, bound with a
// dynamic offset), so a whole batch of steps can be encoded into one command buffer.

struct Params {
  n: u32,            // number of neurons
  step: u32,         // global 0-based step index of this step
  parity: u32,       // spikes(t-1) live at spikes[parity*n ..], spikes(t) are written to the other half
  seed: u32,         // RNG seed (mixed into the per-neuron, per-step hash)
  decay: f32,        // exp(-dt/tau)
  gain: f32,         // synaptic gain
  tonic: f32,        // tonic drive added every step
  noise_p: f32,      // Bernoulli probability of a background kick this step (= noise_rate*dt)
  noise_amp: f32,    // size of a kick (voltage)
  threshold: f32,
  reset: f32,
  floor: f32,        // clamp for runaway inhibition
  drive_offset: u32, // offset (in u32 units) of this step's external-drive pairs in the drive buffer
  drive_count: u32,  // number of (index, amount) pairs
  slot: u32,         // where this step's total spike count is accumulated (stats[16+slot])
  n_plot: u32,       // number of plotted neurons (pack kernel)
};

// Number of pools whose spike counts are accumulated in stats[0 .. POOL_SLOTS).
const POOL_SLOTS: u32 = 16u;
const NO_POOL: u32 = 0xFFFFFFFFu;

// pcg3d (Jarzynski & Olano 2020): a good, cheap, stateless hash -> per-neuron, per-step RNG.
fn pcg3d(p: vec3<u32>) -> vec3<u32> {
  var v = p * 1664525u + 1013904223u;
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  v ^= v >> vec3<u32>(16u);
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  return v;
}

// Uniform in [0, 1) from the top 24 bits of a hash word.
fn hash_unit(neuron: u32, step: u32, seed: u32) -> f32 {
  let h = pcg3d(vec3<u32>(neuron, step, seed ^ 0x9E3779B9u));
  return f32(h.x >> 8u) * (1.0 / 16777216.0);
}
