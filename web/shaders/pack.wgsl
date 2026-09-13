// Pack the brain-map brightness of every plotted neuron into bytes (min(activity*3, 255)),
// four neurons per u32, so the per-frame readback is n_plot bytes (~140 KB).

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> activity: array<u32>;
@group(0) @binding(2) var<storage, read> plot_idx: array<u32>;
@group(0) @binding(3) var<storage, read_write> packed: array<u32>;

@compute @workgroup_size(64)
fn pack(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  if (k * 4u >= P.n_plot) {
    return;
  }
  var w: u32 = 0u;
  for (var b = 0u; b < 4u; b++) {
    let i = k * 4u + b;
    if (i < P.n_plot) {
      let c = min(activity[plot_idx[i]] * 3u, 255u);
      w |= c << (8u * b);
    }
  }
  packed[k] = w;
}
