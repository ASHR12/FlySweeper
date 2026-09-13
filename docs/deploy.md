# Deploying the browser app

`web/` is a static site: `index.html`, ES modules in `web/js/`, WGSL shaders in `web/shaders/`, no
build step. It needs two things a static host has to provide:

1. **HTTPS** (or `localhost`) — WebGPU only runs in a secure context. Vercel, GitHub Pages,
   Cloudflare Pages and Netlify all give you that.
2. **The exported connectome, ~205 MB**, produced by `python -m flysweeper.export_web` into
   `web/data/` (gitignored). The two big files are `in_indices.u32` and `in_weights.f32`
   (100,352,428 bytes each); the other eight add ~4.4 MB. The page downloads all of it on first
   visit, verifies every file's SHA-256 against `manifest.json`, and keeps it in the browser
   Cache API so later visits load in ~0.1 s.

The plan below keeps the code repo small (no 100 MB binaries in git), puts the data on a host that
serves it with CORS headers, and points the page at it through `web/config.json`.

## 1. Point the page at a data host: `web/config.json`

The loader reads a base URL for the data files from `web/config.json`:

```json
{
  "data_base": "https://huggingface.co/datasets/<user>/flysweeper-data/resolve/main/",
  "weights_base": "./weights/"
}
```

* `data_base` is where the 205 MB export lives: the page fetches `<data_base>manifest.json` and
  then `<data_base><file>` for each entry in the manifest (a missing trailing `/` is added).
  Leave it as `./data/` (the default) to serve the files from the same origin as the page. For a
  one-off test, `?data=<url>` on the page URL overrides the file, and a host page can set
  `window.FLYSWEEPER_DATA_BASE` before the modules load.
* `weights_base` is where the small trained KC→MBON weight sets and their `manifest.json` live
  (`web/weights/`, tracked in git, ~0.6 MB in total); normally it stays relative.
* The host must answer with `Access-Control-Allow-Origin` (`*` or your page's origin) — the
  files are read with `fetch()` from another origin, and a response without that header is
  unreadable to the page. Hosts that do and do not are listed below.
* Because every file is checked against the SHA-256 in `manifest.json`, a stale or half-uploaded
  mirror fails loudly instead of running a wrong brain. Re-running the export changes the hashes,
  so **upload the whole of `web/data/` together**, `manifest.json` included, after every export.
* Cache keys include the file's hash, so visitors pick up a new export automatically;
  `?nocache=1` bypasses the cache and `?clearcache=1` wipes it.

Check a host from the command line before wiring it in — you want a `2xx`/`3xx` and the CORS
header on **both** the first response and anything it redirects to:

```bash
curl -sIL -H "Origin: https://ashr12.github.io" "<data_base>manifest.json" | grep -iE "^(HTTP|access-control-allow-origin|location)"
```

## 2. Where to put the 205 MB

| host | CORS for browser `fetch()` | per-file limit | cost / bandwidth | verdict |
|---|---|---|---|---|
| **Hugging Face dataset repo** | yes (`Access-Control-Allow-Origin` on `resolve/` URLs and on the CDN they redirect to; this is what in-browser ML demos rely on) | 50 GB | free, generous bandwidth | **recommended** |
| **GitHub Pages** (a separate data repo) | yes (`access-control-allow-origin: *` on `*.github.io`) | 100 MiB hard limit per file in git; the two big files are 95.7 MiB, so they fit (with a >50 MiB warning); LFS objects are *not* served by Pages | free; site ≤ 1 GB, soft limit 100 GB/month | works; re-exports bloat the repo history |
| **GitHub Release assets** | **no** — the download redirects to a blob host that sends no CORS header (as of 2026), so the page cannot read the bytes | 2 GB | free | fine for manual `curl` downloads, not for the app unless you put a CORS proxy (e.g. a Cloudflare Worker) in front |
| own bucket: Cloudflare R2, S3, GCS | yes, once you set a CORS rule on the bucket | none that matters | R2: free egress, 10 GB free storage | good if you already have one |
| same origin as the page (Vercel / Pages / Cloudflare Pages) | not needed | Vercel: CLI source upload capped at 100 MB (Hobby) / 1 GB (Pro); Cloudflare Pages: 25 MiB per file — too small | Vercel Hobby fast data transfer is ~100 GB/month ≈ 500 cold visits | avoid for the data; fine for the page |

### 2a. Hugging Face (recommended)

```bash
pip install -U huggingface_hub
hf auth login                                           # a write token from https://huggingface.co/settings/tokens
hf repo create flysweeper-data --repo-type dataset       # once
./.venv/bin/python -m flysweeper.export_web              # regenerates web/data/ (~8 s)
hf upload flysweeper-data web/data . --repo-type dataset \
   --commit-message "MaleCNS v1.0 export $(date -u +%F)"
```

Then `data_base` is `https://huggingface.co/datasets/<user>/flysweeper-data/resolve/main/`.
Add a dataset card (`README.md` in the dataset repo) that carries the attribution from
[`DATA-LICENSE.md`](../DATA-LICENSE.md): MaleCNS v1.0, Berg et al., Cell 2026,
doi:10.1016/j.cell.2026.08.015, CC BY 4.0, "modified: signs, normalization, silenced synapses onto
sensory neurons" — the export is a derived work and the licence requires the credit to travel with
it. Set the card's `license: cc-by-4.0`.

### 2b. GitHub Pages from a separate data repository

```bash
# a throw-away repo that only holds the export; do NOT use Git LFS (Pages serves LFS pointers, not files)
mkdir flysweeper-data && cd flysweeper-data && git init -b main
cp -r ../FlySweeper/web/data/* .
cp ../FlySweeper/DATA-LICENSE.md .
git add . && git commit -m "MaleCNS v1.0 export (derived, CC BY 4.0)"
gh repo create ASHR12/FlySweeper-data --public --source . --push
gh api -X POST repos/ASHR12/FlySweeper-data/pages -f build_type=legacy -f "source[branch]=main" -f "source[path]=/"
```

`data_base` becomes `https://ashr12.github.io/FlySweeper-data/`. Each re-export adds ~200 MB to
the history; if you re-export often, recreate the branch (`git checkout --orphan main`) instead of
committing on top.

### 2c. GitHub Release assets (manual downloads only)

Attach the ten files (or a `.tar` of `web/data/`) to a release so people without Python can get
the export:

```bash
gh release create data-v1 web/data/* --title "MaleCNS v1.0 browser export" \
   --notes "Derived from MaleCNS v1.0 (Berg et al., Cell 2026, CC BY 4.0); see DATA-LICENSE.md. Not fetchable from the browser directly (no CORS)."
```

Do not point `data_base` at `https://github.com/.../releases/download/...`: the browser will
report a CORS error. If you must use Releases, front them with a tiny proxy that streams the
asset and adds `Access-Control-Allow-Origin` (a Cloudflare Worker is ~30 lines; restrict it to
your own repo's assets so it is not an open proxy).

### 2d. Your own bucket

Any object store works once it has a CORS rule. For S3 / R2 the rule is:

```json
[{"AllowedOrigins": ["*"], "AllowedMethods": ["GET", "HEAD"], "AllowedHeaders": ["*"], "ExposeHeaders": ["Content-Length"], "MaxAgeSeconds": 86400}]
```

Set `Content-Type: application/octet-stream` for the `.u32`/`.f32`/`.u16`/`.u8` files and
`application/json` for the two JSON files; the loader does not care, but some CDNs refuse to
cache untyped objects.

## 3. Deploying `web/` to Vercel

No framework, no build. Either connect the GitHub repo in the Vercel dashboard or use the CLI:

```bash
npm i -g vercel
cd FlySweeper
vercel --prod            # Framework preset: Other · Build command: (empty) · Output directory: web
```

With a Git-connected project set **Root Directory** to `web` (Settings → General) so only the app is
deployed. `web/data/` is gitignored and must stay out of the deployment — with the data on another
host the site is ~150 KB. A `vercel.json` in `web/` is optional; this one makes the shader and
binary types explicit and lets the (hash-keyed) data files cache for a year if you *do* serve
them from Vercel:

```json
{
  "headers": [
    { "source": "/shaders/(.*)\\.wgsl", "headers": [{ "key": "Content-Type", "value": "text/plain; charset=utf-8" }] },
    { "source": "/data/(.*)", "headers": [{ "key": "Cache-Control", "value": "public, max-age=31536000, immutable" }] }
  ]
}
```

Vercel caveats:

* CLI deployments cap the uploaded source at **100 MB on Hobby, 1 GB on Pro**; a `web/` that
  contains `web/data/` (205 MB) fails to upload on Hobby and is a bad idea on Pro.
* Hobby includes on the order of 100 GB/month of fast data transfer. Every first-time visitor
  pulls 205 MB, so serving the data from Vercel buys roughly 500 cold loads a month before the
  project is paused; keep the data on Hugging Face or a bucket.
* Preview deployments get a new URL each time; the Cache API is per origin, so previews re-download
  the data. Production stays cached.
* No WebGPU on the server side is involved — everything runs in the visitor's browser (Chrome 113+
  / Edge; Safari and Firefox with WebGPU enabled were not tested).

## 4. Deploying `web/` to GitHub Pages

Pages publishes a branch root or `/docs`, not an arbitrary subfolder, so use the Actions
deployment and upload only `web/` as the artifact. Save this as
`.github/workflows/pages.yml` (not included in the repo by default — it only makes sense once
Pages is enabled for the repository under Settings → Pages → Source: *GitHub Actions*):

```yaml
name: pages
on:
  push:
    branches: [main]
    paths: ["web/**"]
  workflow_dispatch:
permissions:
  contents: read
  pages: write
  id-token: write
concurrency: { group: pages, cancel-in-progress: true }
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: { name: github-pages, url: "${{ steps.deployment.outputs.page_url }}" }
    steps:
      - uses: actions/checkout@v4
      - run: test -f web/config.json || echo '{ "data_base": "https://huggingface.co/datasets/<user>/flysweeper-data/resolve/main/" }' > web/config.json
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with: { path: web }
      - id: deployment
        uses: actions/deploy-pages@v4
```

The site appears at `https://ashr12.github.io/FlySweeper/`. Pages serves unknown extensions
(`.u32`, `.f32`, `.wgsl`) as `application/octet-stream`, which is fine — the page reads shaders
with `response.text()` and binaries with `arrayBuffer()`. Pages also sends
`access-control-allow-origin: *`, so a Pages-hosted data repo (2b) and a Pages-hosted app can live
on different repos without any further configuration.

## 5. After deploying, check

1. Open the site in Chrome, DevTools → Console: no red lines; the log panel shows
   `fetch in_indices.u32: 100.4 MB in …` on the first load and `cache …` on a reload.
2. `flysweeper.idleTest()` in the console: mean rate ≈ 2.3 Hz, ≈ 7,600 spikes/step (the same
   population numbers as `python -m flysweeper.sim --no-sensory-input`).
3. `flysweeper.benchmark(300)`: ~0.9 ms/step on an M-series Mac; anything under 20 ms/step is
   realtime.
4. The honesty label at the bottom of the page is visible: it is part of the attribution.

## 6. Trained weights in the browser

The browser build ships the `fly-mb` policies in `web/weights/`: a `manifest.json` listing the
sets (`round1`, `round2`, `round3`, with their held-out numbers and SHA-256s; `round3` is the
default) and one `roundN.json` + `roundN.bin` pair per round (~160 KB and ~470 KB), converted from
`models/*.npz`. The **weights** selector in the page header or `?weights=<id>` on the URL selects a
set; `?weights=frozen` runs the untrained connectome. They deploy with
the page (no CORS question) and are covered by the same CC BY 4.0 attribution as `models/` — they
contain original MaleCNS weights for the trained edges. The deployed page already carries the
dataset credit in its bottom label; keep it, and link `DATA-LICENSE.md` from the page or the
hosting repo.
