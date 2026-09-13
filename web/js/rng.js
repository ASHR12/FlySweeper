// Small seeded PRNG (sfc32) for everything that is random on the CPU side: mine layout, decoder
// tie-breaks, jump targets. Not numpy's Generator, so sequences differ from the Python runs; the
// statistics are what matter.

export class RNG {
  constructor(seed = 0) {
    this.seed = seed;
    // splitmix-style seeding of the four sfc32 words
    let s = (seed >>> 0) || 0x9E3779B9;
    const next = () => { s = (s + 0x9E3779B9) >>> 0; let z = s; z = Math.imul(z ^ (z >>> 16), 0x85EBCA6B) >>> 0; z = Math.imul(z ^ (z >>> 13), 0xC2B2AE35) >>> 0; return (z ^ (z >>> 16)) >>> 0; };
    this.a = next(); this.b = next(); this.c = next(); this.d = 1;
    for (let i = 0; i < 12; i++) this.next();
  }
  next() {
    let { a, b, c, d } = this;
    const t = (((a + b) >>> 0) + d) >>> 0;
    d = (d + 1) >>> 0;
    a = b ^ (b >>> 9);
    b = (c + (c << 3)) >>> 0;
    c = ((c << 21) | (c >>> 11)) >>> 0;
    c = (c + t) >>> 0;
    this.a = a; this.b = b; this.c = c; this.d = d;
    return t;
  }
  /** Uniform float in [0, 1). */
  random() { return this.next() / 4294967296; }
  /** Uniform integer in [0, n). */
  int(n) { return Math.floor(this.random() * n); }
  choice(arr) { return arr[this.int(arr.length)]; }
  /** k distinct indices from [0, n) (partial Fisher-Yates). */
  sample(n, k) {
    const pool = Array.from({ length: n }, (_, i) => i);
    for (let i = 0; i < k; i++) {
      const j = i + this.int(n - i);
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    return pool.slice(0, k);
  }
}
