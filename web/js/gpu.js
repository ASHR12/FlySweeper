// WebGPU leaky integrate-and-fire simulator for the whole MaleCNS graph.
//
// All neuron/synapse state stays resident on the GPU. A batch of up to `maxBatch` steps is encoded
// into one command buffer (scatter external drive -> fused gather+integrate, per step); only a
// 256-byte stats block (pool spike counts + per-step spike totals) is read back per batch, and a
// packed byte per plotted neuron whenever the UI asks for the brain map.

const UNIFORM_STRIDE = 256;   // bytes per per-step Params block (minUniformBufferOffsetAlignment)
const STATS_WORDS = 64;       // [0,16) pool counts, [16,48) per-slot spike totals
const POOL_SLOTS = 16;
const NO_POOL = 0xFFFFFFFF;
const ROWS_PER_WG = 8;        // must match step.wgsl

export class WebGPUUnavailable extends Error {}

export class FlySim {
  /**
   * @param {object} opts
   * @param {object} opts.model        parsed model.json
   * @param {object} opts.arrays       {in_indptr, in_indices, in_weights, order, pool_id, plot_idx} typed arrays
   * @param {object} opts.shaders      {common, step, scatter, pack} WGSL sources
   * @param {number} [opts.seed]
   * @param {number} [opts.maxBatch]   max steps per submit (<= 32)
   * @param {number} [opts.maxDrive]   max (index, amount) pairs per step
   * @param {(msg:string)=>void} [opts.log]
   */
  static async create(opts) {
    if (!('gpu' in navigator)) {
      throw new WebGPUUnavailable('navigator.gpu is missing: this browser has no WebGPU. Use Chrome 113+ (macOS) or Edge.');
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new WebGPUUnavailable('navigator.gpu.requestAdapter() returned null (no compatible GPU adapter).');
    const need = {
      maxStorageBufferBindingSize: opts.arrays.in_indices.byteLength,
      maxBufferSize: opts.arrays.in_indices.byteLength,
    };
    for (const [k, v] of Object.entries(need)) {
      if (adapter.limits[k] < v) throw new WebGPUUnavailable(`adapter limit ${k}=${adapter.limits[k]} is below the ${v} bytes this graph needs.`);
    }
    const requiredLimits = {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    };
    const requiredFeatures = [];
    if (adapter.features.has('timestamp-query')) requiredFeatures.push('timestamp-query');
    const device = await adapter.requestDevice({ requiredLimits, requiredFeatures });
    const info = adapter.info || {};
    const sim = new FlySim(device, opts);
    sim.adapterInfo = { vendor: info.vendor || '', architecture: info.architecture || '', description: info.description || '' };
    await sim._build();
    return sim;
  }

  constructor(device, opts) {
    this.device = device;
    this.model = opts.model;
    this.arrays = opts.arrays;
    this.shaders = opts.shaders;
    this.seed = opts.seed ?? 0;
    this.maxBatch = Math.min(opts.maxBatch ?? 32, 32);
    this.maxDrive = opts.maxDrive ?? 8192;
    this.log = opts.log || (() => {});
    this.n = opts.model.n_neurons;
    this.nPlot = opts.arrays.plot_idx.length;
    this.stepCount = 0;
    this.lost = null;
    this.hasTimestamps = device.features.has('timestamp-query');
    this._lock = Promise.resolve();
    this._packBusy = false;
    const p = opts.model.sim;
    this.params = {
      dt: p.dt, tau: p.tau, gain: p.gain, tonic: p.tonic, noise_rate: p.noise_rate, noise_amp: p.noise_amp,
      threshold: p.threshold, reset: p.reset, floor: p.floor,
    };
    device.lost.then((info) => { this.lost = info; this.log(`WebGPU device lost: ${info.message}`); });
    this.errors = [];
    device.addEventListener('uncapturederror', (ev) => {
      const msg = `WebGPU ${ev.error.constructor.name}: ${ev.error.message}`;
      this.errors.push(msg);
      if (this.errors.length <= 5) { console.error(msg); this.log(msg); }
    });
  }

  /** Debug: read `count` u32/f32 words of a GPU buffer (must have COPY_SRC). */
  async debugRead(name, offsetWords = 0, count = 16, asFloat = false) {
    const buf = this[name];
    if (!buf) throw new Error(`no buffer ${name}`);
    return this._serial(async () => {
      const staging = this.device.createBuffer({ size: count * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const enc = this.device.createCommandEncoder();
      enc.copyBufferToBuffer(buf, offsetWords * 4, staging, 0, count * 4);
      this.device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const out = Array.from(new (asFloat ? Float32Array : Uint32Array)(staging.getMappedRange().slice(0)));
      staging.unmap(); staging.destroy();
      return out;
    });
  }

  get time() { return this.stepCount * this.params.dt; }

  async _build() {
    const { device, n, arrays } = this;
    const t0 = performance.now();
    const storage = (arr, extraUsage = 0) => {
      const buf = device.createBuffer({ size: Math.max(16, align4(arr.byteLength)), usage: GPUBufferUsage.STORAGE | extraUsage, mappedAtCreation: true });
      new Uint8Array(buf.getMappedRange()).set(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength));
      buf.unmap();
      return buf;
    };
    const zeros = (bytes, extraUsage = 0) => device.createBuffer({ size: Math.max(16, align4(bytes)), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | extraUsage });

    this.bufIndptr = storage(arrays.in_indptr);
    this.bufIndices = storage(arrays.in_indices);
    this.bufWeights = storage(arrays.in_weights);
    const meta = new Uint32Array(2 * n);
    meta.set(arrays.order, 0);
    meta.set(arrays.pool_id, n);
    this.bufMeta = storage(meta, GPUBufferUsage.COPY_DST);
    this.bufState = zeros(2 * n * 4);
    this.bufSpikes = zeros(2 * n * 4);
    this.bufActivity = zeros(n * 4);
    this.bufStats = zeros(STATS_WORDS * 4);
    this.bufUniform = device.createBuffer({ size: this.maxBatch * UNIFORM_STRIDE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.bufDrive = device.createBuffer({ size: this.maxBatch * this.maxDrive * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.bufPlotIdx = storage(arrays.plot_idx);
    this.packedWords = Math.ceil(this.nPlot / 4);
    this.bufPacked = device.createBuffer({ size: this.packedWords * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.stagingStats = device.createBuffer({ size: STATS_WORDS * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.stagingPacked = device.createBuffer({ size: this.packedWords * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.uniformCPU = new ArrayBuffer(this.maxBatch * UNIFORM_STRIDE);
    this.uniformU32 = new Uint32Array(this.uniformCPU);
    this.uniformF32 = new Float32Array(this.uniformCPU);
    this.driveCPU = new ArrayBuffer(this.maxDrive * 8);
    this.driveU32 = new Uint32Array(this.driveCPU);
    this.driveF32 = new Float32Array(this.driveCPU);
    this.activityBytes = new Uint8Array(this.nPlot);

    if (this.hasTimestamps) {
      this.querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
      this.bufQuery = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      this.stagingQuery = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    }

    // pipelines --------------------------------------------------------------------------
    const mkModule = async (name, src) => {
      const mod = device.createShaderModule({ code: this.shaders.common + '\n' + src, label: name });
      const info = await mod.getCompilationInfo();
      const errors = info.messages.filter((m) => m.type === 'error');
      if (errors.length) throw new Error(`WGSL ${name}: ` + errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('\n'));
      return mod;
    };
    const U = { buffer: { type: 'uniform', hasDynamicOffset: true } };
    const RO = { buffer: { type: 'read-only-storage' } };
    const RW = { buffer: { type: 'storage' } };
    const C = GPUShaderStage.COMPUTE;
    const layout = (entries) => device.createBindGroupLayout({ entries: entries.map((e, i) => ({ binding: i, visibility: C, ...e })) });
    this.layoutStep = layout([U, RO, RO, RO, RO, RW, RW, RW, RW]);
    this.layoutScatter = layout([U, RO, RW]);
    this.layoutPack = layout([U, RO, RO, RW]);

    device.pushErrorScope('validation');
    const [modStep, modScatter, modPack] = await Promise.all([
      mkModule('step', this.shaders.step), mkModule('scatter', this.shaders.scatter), mkModule('pack', this.shaders.pack),
    ]);
    const mkPipeline = (mod, entry, bgl) => device.createComputePipelineAsync({
      label: entry, layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }), compute: { module: mod, entryPoint: entry },
    });
    [this.pipeStep, this.pipeScatter, this.pipePack] = await Promise.all([
      mkPipeline(modStep, 'step', this.layoutStep), mkPipeline(modScatter, 'scatter', this.layoutScatter), mkPipeline(modPack, 'pack', this.layoutPack),
    ]);
    const err = await device.popErrorScope();
    if (err) throw new Error('WebGPU pipeline creation failed: ' + err.message);

    const bg = (bgl, buffers) => device.createBindGroup({
      layout: bgl,
      entries: buffers.map((b, i) => ({ binding: i, resource: i === 0 ? { buffer: this.bufUniform, size: UNIFORM_STRIDE } : { buffer: b } })),
    });
    this.bgStep = bg(this.layoutStep, [null, this.bufIndptr, this.bufIndices, this.bufWeights, this.bufMeta, this.bufState, this.bufSpikes, this.bufActivity, this.bufStats]);
    this.bgScatter = bg(this.layoutScatter, [null, this.bufDrive, this.bufState]);
    this.bgPack = bg(this.layoutPack, [null, this.bufActivity, this.bufPlotIdx, this.bufPacked]);
    this.wgStep = Math.ceil(n / ROWS_PER_WG);
    this.gpuBytes = [this.bufIndptr, this.bufIndices, this.bufWeights, this.bufMeta, this.bufState, this.bufSpikes, this.bufActivity, this.bufDrive, this.bufPlotIdx, this.bufPacked]
      .reduce((s, b) => s + b.size, 0);
    this.log(`GPU buffers ready: ${(this.gpuBytes / 1e6).toFixed(1)} MB in ${(performance.now() - t0).toFixed(0)} ms, timestamps ${this.hasTimestamps ? 'on' : 'off'}`);
  }

  _writeParams(slot, step, driveCount) {
    const o = (slot * UNIFORM_STRIDE) / 4;
    const u = this.uniformU32, f = this.uniformF32, p = this.params;
    u[o + 0] = this.n;
    u[o + 1] = step;
    u[o + 2] = step & 1;                       // spikes(t-1) live in half (t & 1); step t writes the other half
    u[o + 3] = this.seed >>> 0;
    f[o + 4] = Math.exp(-p.dt / p.tau);
    f[o + 5] = p.gain;
    f[o + 6] = p.tonic;
    f[o + 7] = p.noise_rate * p.dt;
    f[o + 8] = p.noise_amp;
    f[o + 9] = p.threshold;
    f[o + 10] = p.reset;
    f[o + 11] = p.floor;
    u[o + 12] = slot * this.maxDrive * 2;
    u[o + 13] = driveCount;
    u[o + 14] = slot;
    u[o + 15] = this.nPlot;
  }

  /** Serialize GPU-touching operations (a batch, a reset and a pool update must not interleave). */
  _serial(fn) {
    const run = this._lock.then(fn, fn);
    this._lock = run.then(() => {}, () => {});
    return run;
  }

  /**
   * Run `drives.length` steps (<= maxBatch). drives[s] = {idx: Uint32Array|Int32Array, amt: Float32Array}
   * or null for no external input. Returns pool counts accumulated since the last reset, the number
   * of spikes in each step of the batch, and timing.
   */
  runSteps(drives, { resetPools = false } = {}) {
    return this._serial(async () => {
      if (this.lost) throw new Error('WebGPU device lost');
      const k = drives.length;
      if (k < 1 || k > this.maxBatch) throw new Error(`batch of ${k} steps (max ${this.maxBatch})`);
      const { device } = this;
      const counts = new Array(k);
      for (let s = 0; s < k; s++) {
        const d = drives[s];
        const count = d ? d.idx.length : 0;
        if (count > this.maxDrive) throw new Error(`drive list of ${count} exceeds maxDrive ${this.maxDrive}`);
        counts[s] = count;
        this._writeParams(s, this.stepCount + s, count);
        if (count) {
          const u = this.driveU32, f = this.driveF32;
          for (let i = 0; i < count; i++) { u[2 * i] = d.idx[i]; f[2 * i + 1] = d.amt[i]; }
          device.queue.writeBuffer(this.bufDrive, s * this.maxDrive * 8, this.driveCPU, 0, count * 8);
        }
      }
      device.queue.writeBuffer(this.bufUniform, 0, this.uniformCPU, 0, k * UNIFORM_STRIDE);

      const enc = device.createCommandEncoder();
      if (resetPools) enc.clearBuffer(this.bufStats, 0, POOL_SLOTS * 4);
      enc.clearBuffer(this.bufStats, POOL_SLOTS * 4, 32 * 4);
      const passDesc = this.hasTimestamps ? { timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {};
      const pass = enc.beginComputePass(passDesc);
      for (let s = 0; s < k; s++) {
        if (counts[s]) {
          pass.setPipeline(this.pipeScatter);
          pass.setBindGroup(0, this.bgScatter, [s * UNIFORM_STRIDE]);
          pass.dispatchWorkgroups(Math.ceil(counts[s] / 64));
        }
        pass.setPipeline(this.pipeStep);
        pass.setBindGroup(0, this.bgStep, [s * UNIFORM_STRIDE]);
        pass.dispatchWorkgroups(this.wgStep);
      }
      pass.end();
      if (this.hasTimestamps) {
        enc.resolveQuerySet(this.querySet, 0, 2, this.bufQuery, 0);
        enc.copyBufferToBuffer(this.bufQuery, 0, this.stagingQuery, 0, 16);
      }
      enc.copyBufferToBuffer(this.bufStats, 0, this.stagingStats, 0, STATS_WORDS * 4);
      const t0 = performance.now();
      device.queue.submit([enc.finish()]);
      const maps = [this.stagingStats.mapAsync(GPUMapMode.READ)];
      if (this.hasTimestamps) maps.push(this.stagingQuery.mapAsync(GPUMapMode.READ));
      await Promise.all(maps);
      const wallMs = performance.now() - t0;
      const stats = new Uint32Array(this.stagingStats.getMappedRange().slice(0));
      this.stagingStats.unmap();
      let gpuMs = null;
      if (this.hasTimestamps) {
        const ts = new BigUint64Array(this.stagingQuery.getMappedRange().slice(0));
        this.stagingQuery.unmap();
        const ns = Number(ts[1] - ts[0]);
        if (ns > 0 && ns < 1e12) gpuMs = ns / 1e6;
      }
      this.stepCount += k;
      return {
        k,
        poolCounts: stats.subarray(0, POOL_SLOTS),
        spikesPerStep: stats.subarray(POOL_SLOTS, POOL_SLOTS + k),
        wallMs,
        gpuMs,
      };
    });
  }

  /** Brightness byte (min(activity*3,255)) for every plotted neuron, or null if a readback is in flight. */
  async readActivity() {
    if (this._packBusy || this.lost) return null;
    this._packBusy = true;
    try {
      await this._serial(async () => {
        const { device } = this;
        this._writeParams(0, this.stepCount, 0);
        device.queue.writeBuffer(this.bufUniform, 0, this.uniformCPU, 0, UNIFORM_STRIDE);
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(this.pipePack);
        pass.setBindGroup(0, this.bgPack, [0]);
        pass.dispatchWorkgroups(Math.ceil(this.packedWords / 64));
        pass.end();
        enc.copyBufferToBuffer(this.bufPacked, 0, this.stagingPacked, 0, this.packedWords * 4);
        device.queue.submit([enc.finish()]);
      });
      await this.stagingPacked.mapAsync(GPUMapMode.READ);
      this.activityBytes.set(new Uint8Array(this.stagingPacked.getMappedRange(), 0, this.nPlot));
      this.stagingPacked.unmap();
      return this.activityBytes;
    } finally {
      this._packBusy = false;
    }
  }

  /** Replace the pool membership: pools[k] is an index list for stats slot k (max 16 pools). */
  setPools(pools) {
    if (pools.length > POOL_SLOTS) throw new Error(`at most ${POOL_SLOTS} pools`);
    const ids = new Uint32Array(this.n).fill(NO_POOL);
    pools.forEach((idx, k) => { for (const j of idx) ids[j] = k; });
    return this._serial(async () => { this.device.queue.writeBuffer(this.bufMeta, this.n * 4, ids); });
  }

  /** Zero all dynamic state (v, ext, spikes, activity, stats) and the step counter. */
  reset() {
    return this._serial(async () => {
      const enc = this.device.createCommandEncoder();
      for (const b of [this.bufState, this.bufSpikes, this.bufActivity, this.bufStats]) enc.clearBuffer(b);
      this.device.queue.submit([enc.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      this.stepCount = 0;
    });
  }

  destroy() {
    this.device.destroy();
  }
}

function align4(x) { return (x + 3) & ~3; }
