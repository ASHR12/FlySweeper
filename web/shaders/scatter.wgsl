// External drive: add (index, amount) pairs into the ext half of `state` before a step.
// The step kernel zeroes ext after reading it, so this is `np.add.at(ext, idx, amount)` on a
// fresh zero vector. Indices within one step must be unique (the encoder guarantees this).

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> drive: array<u32>;          // pairs: [index, bitcast<u32>(amount)]
@group(0) @binding(2) var<storage, read_write> state: array<f32>;    // [n,2n) = ext

@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  if (k >= P.drive_count) {
    return;
  }
  let idx = drive[P.drive_offset + 2u * k];
  let amt = bitcast<f32>(drive[P.drive_offset + 2u * k + 1u]);
  state[P.n + idx] += amt;
}
