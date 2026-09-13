// Whole-CNS soma map: every neuron with a soma position is a point, drawn as faint region-tinted
// dust when quiet and as a soft additive glow when it has spiked recently.
// The (u, v) atlas coordinates are the soma x/z positions each normalised to [0, 1]; the true
// proportions of the MaleCNS soma cloud are x-range / z-range = 0.733 (91.2 mm-units wide by
// 124.4 tall, brain on top, nerve cord below), so the unit square is drawn into a frame of that
// aspect, fitted to the canvas with a small margin.

export const CNS_ASPECT = 0.733;

/** Slightly desaturate and lift a region colour so the palette reads as one family. */
export function harmonise([r, g, b]) {
  const l = 0.3 * r + 0.59 * g + 0.11 * b;
  const k = 0.78;   // 0 = grey, 1 = original saturation
  return [Math.round(l + (r - l) * k), Math.round(l + (g - l) * k), Math.round(l + (b - l) * k)];
}

export class BrainMap {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {Float32Array} uv       2*n, (u, v) in [0,1]
   * @param {Uint8Array} region     n, palette index
   * @param {Array<{name:string,color:number[]}>} regions
   * @param {Uint16Array} [typeIdOfPlotted]  n, type id per plotted neuron (hover labels)
   * @param {string[]} [typeNames]
   */
  constructor(canvas, uv, region, regions, typeIdOfPlotted = null, typeNames = null) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d', { willReadFrequently: false });
    this.n = region.length;
    this.region = region;
    this.regions = regions;
    this.typeId = typeIdOfPlotted;
    this.typeNames = typeNames;
    this.uv = uv;
    this.pix = new Int32Array(this.n);
    this.palette = regions.map((r) => harmonise(r.color));
    this.blank = new Uint8Array(this.n);
    this.lastAct = this.blank;
    this._layout();
    this.draw(this.blank);
  }

  /** Map every soma to a pixel: the unit square is fitted at CNS_ASPECT into the canvas with a 3% margin. */
  _layout() {
    const W = this.canvas.width, H = this.canvas.height, uv = this.uv;
    this.layoutW = W; this.layoutH = H;
    const m = 0.03;
    let fh = H * (1 - 2 * m), fw = fh * CNS_ASPECT;
    if (fw > W * (1 - 2 * m)) { fw = W * (1 - 2 * m); fh = fw / CNS_ASPECT; }
    const ox = (W - fw) / 2, oy = (H - fh) / 2;
    this.frame = { ox, oy, fw, fh };
    // glow gain falls with point density so a small frame does not saturate to white
    this.gain = Math.min(1, Math.max(0.3, (fw * fh) / (this.n * 2.4)));
    this.owner = new Int32Array(W * H).fill(-1);
    for (let i = 0; i < this.n; i++) {
      const x = Math.min(W - 2, Math.max(1, Math.round(ox + uv[2 * i] * fw)));
      const y = Math.min(H - 2, Math.max(1, Math.round(oy + uv[2 * i + 1] * fh)));
      const p = y * W + x;
      this.pix[i] = p;
      this.owner[p] = i;
    }
    this.img = this.ctx.createImageData(W, H);
    // the resting "dust" layer is constant: precompute it once per layout
    this.dust = new Uint8ClampedArray(this.img.data.length);
    const d = this.dust, pal = this.palette, region = this.region, pix = this.pix;
    for (let i = 3; i < d.length; i += 4) d[i] = 255;
    for (let i = 0; i < this.n; i++) {
      const p = pix[i] * 4, c = pal[region[i]];
      d[p] += 6 + c[0] * 0.13; d[p + 1] += 6 + c[1] * 0.13; d[p + 2] += 6 + c[2] * 0.13;
    }
  }

  /** Change the canvas resolution and redraw. */
  resize(W, H) {
    if (W < 8 || H < 8 || (W === this.layoutW && H === this.layoutH)) return;
    if (W !== this.canvas.width || H !== this.canvas.height) { this.canvas.width = W; this.canvas.height = H; }
    this._layout();
    this.draw(this.lastAct);
  }

  /** @param {Uint8Array} act  recent-spike activity per plotted neuron, 0..255 */
  draw(act) {
    this.lastAct = act;
    const W = this.canvas.width, d = this.img.data;
    d.set(this.dust);
    const pal = this.palette, region = this.region, pix = this.pix, W4 = W * 4, gain = this.gain;
    for (let i = 0; i < this.n; i++) {
      const a = act[i];
      if (a < 6) continue;
      const p = pix[i] * 4, c = pal[region[i]], k = (a / 255) * gain;
      // soft 3x3 additive glow: centre full, edges 40%, corners 15% (Uint8ClampedArray clamps for us)
      const r = c[0] * k, g = c[1] * k, b = c[2] * k, w = 30 * k;
      d[p] += r + w; d[p + 1] += g + w; d[p + 2] += b + w;
      const r1 = r * 0.4, g1 = g * 0.4, b1 = b * 0.4, r2 = r * 0.15, g2 = g * 0.15, b2 = b * 0.15;
      let q = p - 4; d[q] += r1; d[q + 1] += g1; d[q + 2] += b1;
      q = p + 4; d[q] += r1; d[q + 1] += g1; d[q + 2] += b1;
      q = p - W4; d[q] += r1; d[q + 1] += g1; d[q + 2] += b1;
      q = p + W4; d[q] += r1; d[q + 1] += g1; d[q + 2] += b1;
      q = p - W4 - 4; d[q] += r2; d[q + 1] += g2; d[q + 2] += b2;
      q = p - W4 + 4; d[q] += r2; d[q + 1] += g2; d[q + 2] += b2;
      q = p + W4 - 4; d[q] += r2; d[q + 1] += g2; d[q + 2] += b2;
      q = p + W4 + 4; d[q] += r2; d[q + 1] += g2; d[q + 2] += b2;
    }
    this.ctx.putImageData(this.img, 0, 0);
  }

  /** Nearest plotted neuron (within `radius` px) to a canvas-space point, or null. */
  pick(x, y, radius = 3) {
    const W = this.canvas.width, H = this.canvas.height;
    let best = null, bestD = Infinity;
    for (let dy = -radius; dy <= radius; dy++) for (let dx = -radius; dx <= radius; dx++) {
      const px = x + dx, py = y + dy;
      if (px < 0 || py < 0 || px >= W || py >= H) continue;
      const i = this.owner[py * W + px];
      if (i >= 0) { const dd = dx * dx + dy * dy; if (dd < bestD) { bestD = dd; best = i; } }
    }
    if (best === null) return null;
    return { plotIndex: best, region: this.regions[this.region[best]].name, type: this.typeNames && this.typeId ? this.typeNames[this.typeId[best]] || '(untyped)' : null };
  }
}
