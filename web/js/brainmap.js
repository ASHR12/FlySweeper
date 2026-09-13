// Whole-CNS soma map: every neuron with a soma position is one pixel, coloured by region and
// brightened by its recent spikes (same drawing rule as flysweeper/ui/index.html).

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
    this.palette = regions.map((r) => r.color);
    this.blank = new Uint8Array(this.n);
    this.lastAct = this.blank;
    this._layout();
    this.draw(this.blank);
  }

  /** Map every soma to a pixel of the current canvas resolution. */
  _layout() {
    const W = this.canvas.width, H = this.canvas.height, pad = Math.max(4, Math.round(Math.min(W, H) * 0.025)), uv = this.uv;
    this.owner = new Int32Array(W * H).fill(-1);
    for (let i = 0; i < this.n; i++) {
      const x = Math.round(pad + uv[2 * i] * (W - 2 * pad));
      const y = Math.round(pad + uv[2 * i + 1] * (H - 2 * pad));
      const p = y * W + x;
      this.pix[i] = p;
      this.owner[p] = i;
    }
    this.img = this.ctx.createImageData(W, H);
  }

  /** Change the canvas resolution (keeps the same aspect ratio by convention) and redraw. */
  resize(W, H) {
    if (W < 8 || H < 8 || (W === this.canvas.width && H === this.canvas.height)) return;
    this.canvas.width = W; this.canvas.height = H;
    this._layout();
    this.draw(this.lastAct);
  }

  draw(act) {
    this.lastAct = act;
    const d = this.img.data, W = this.canvas.width;
    d.fill(0);
    for (let i = 3; i < d.length; i += 4) d[i] = 255;
    const base = 22 * 0.35, pal = this.palette, region = this.region, pix = this.pix;
    for (let i = 0; i < this.n; i++) {
      const p = pix[i] * 4;
      const c = pal[region[i]];
      const a = act[i] / 255;
      d[p] = Math.min(255, d[p] + base + c[0] * a);
      d[p + 1] = Math.min(255, d[p + 1] + base + c[1] * a);
      d[p + 2] = Math.min(255, d[p + 2] + base + c[2] * a);
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
