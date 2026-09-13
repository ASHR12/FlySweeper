// Download the exported connectome (manifest.json + ~205 MB of binaries) with progress reporting,
// keep a copy in the browser Cache API so reloads are instant, and turn the bytes into typed arrays.
//
// Cache entries are keyed by file name + sha256 from the manifest, so a re-export automatically
// invalidates stale copies (which are deleted). `?nocache=1` bypasses the cache; `?clearcache=1`
// wipes it first.

const CACHE_NAME = 'flysweeper-data-v1';

const DTYPES = {
  uint32: Uint32Array, float32: Float32Array, uint16: Uint16Array, uint8: Uint8Array, int32: Int32Array, int8: Int8Array,
};

export async function loadData(baseUrl, { onProgress = () => {}, log = () => {}, useCache = true, verifyHash = true } = {}) {
  const manifestResp = await fetch(baseUrl + 'manifest.json', { cache: 'no-store' });
  if (!manifestResp.ok) throw new Error(`manifest.json not found at ${baseUrl} (HTTP ${manifestResp.status}). Run: ./.venv/bin/python -m flysweeper.export_web`);
  const manifest = await manifestResp.json();
  const files = Object.entries(manifest.files);
  const total = files.reduce((s, [, f]) => s + f.bytes, 0);

  let cache = null;
  if (useCache && 'caches' in window) {
    try {
      cache = await caches.open(CACHE_NAME);
      const wanted = new Set(files.map(([name, f]) => cacheKey(baseUrl, name, f)));
      for (const req of await cache.keys()) if (!wanted.has(req.url)) { await cache.delete(req); log(`cache: dropped stale ${req.url.split('/').pop()}`); }
    } catch (e) {
      log(`cache unavailable (${e.message}); downloading every time`);
      cache = null;
    }
  }

  let done = 0;
  const arrays = {};
  let fromCache = 0;
  for (const [name, info] of files) {
    const t0 = performance.now();
    const { buffer, cached } = await fetchFile(baseUrl, name, info, cache, verifyHash, (delta) => {
      done += delta;
      onProgress({ done, total, name, fraction: done / total });
    });
    fromCache += cached ? 1 : 0;
    const ctor = DTYPES[info.dtype];
    if (info.dtype === 'json') arrays[name] = JSON.parse(new TextDecoder().decode(buffer));
    else if (ctor) arrays[name] = new ctor(buffer);
    else throw new Error(`unknown dtype ${info.dtype} for ${name}`);
    log(`${cached ? 'cache' : 'fetch'} ${name}: ${(info.bytes / 1e6).toFixed(1)} MB in ${(performance.now() - t0).toFixed(0)} ms`);
  }
  return { manifest, arrays, fromCache, total };
}

function cacheKey(baseUrl, name, info) {
  return new URL(`${baseUrl}${name}?sha256=${info.sha256}`, location.href).href;
}

async function fetchFile(baseUrl, name, info, cache, verifyHash, onDelta) {
  const key = cacheKey(baseUrl, name, info);
  let resp = null, cached = false;
  if (cache) {
    try { resp = await cache.match(key); cached = !!resp; } catch { resp = null; }
  }
  if (!resp) {
    resp = await fetch(baseUrl + name, { cache: 'no-store' });
    if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching ${name}`);
  }
  const buffer = await readWithProgress(resp, info.bytes, onDelta);
  if (buffer.byteLength !== info.bytes) {
    if (cached) { await cache.delete(key); }
    throw new Error(`${name}: got ${buffer.byteLength} bytes, manifest says ${info.bytes}. Re-run the export or reload.`);
  }
  if (!cached) {
    if (verifyHash && crypto.subtle) {
      const digest = await crypto.subtle.digest('SHA-256', buffer);
      const hex = [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
      if (hex !== info.sha256) throw new Error(`${name}: sha256 mismatch (${hex.slice(0, 12)}… vs manifest ${info.sha256.slice(0, 12)}…). Re-run the export.`);
    }
    if (cache) {
      try {
        await cache.put(key, new Response(buffer.slice(0), { headers: { 'Content-Type': 'application/octet-stream', 'Content-Length': String(buffer.byteLength) } }));
      } catch (e) { /* quota or private mode: fine, we just download again next time */ }
    }
  }
  return { buffer, cached };
}

async function readWithProgress(resp, expected, onDelta) {
  if (!resp.body) {
    const buf = await resp.arrayBuffer();
    onDelta(buf.byteLength);
    return buf;
  }
  const reader = resp.body.getReader();
  let out = new Uint8Array(expected);
  let off = 0;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    if (off + value.byteLength > out.byteLength) {
      const bigger = new Uint8Array(Math.max(out.byteLength * 2, off + value.byteLength));
      bigger.set(out.subarray(0, off));
      out = bigger;
    }
    out.set(value, off);
    off += value.byteLength;
    onDelta(value.byteLength);
  }
  return off === out.byteLength ? out.buffer : out.buffer.slice(0, off);
}

export async function clearDataCache() {
  if ('caches' in window) await caches.delete(CACHE_NAME);
}

export async function loadShaders(baseUrl) {
  const names = ['common', 'step', 'scatter', 'pack'];
  const srcs = await Promise.all(names.map(async (n) => {
    const r = await fetch(`${baseUrl}${n}.wgsl`, { cache: 'no-store' });
    if (!r.ok) throw new Error(`HTTP ${r.status} fetching ${n}.wgsl`);
    return r.text();
  }));
  return Object.fromEntries(names.map((n, i) => [n, srcs[i]]));
}
