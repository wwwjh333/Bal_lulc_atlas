# Bal-DC LULC Atlas

One-page academic project site for the Bal-DC LULC Atlas paper: UrbanMIM framework, Bal-DC LULC Benchmark, and interactive atlas (coming soon).

## Local preview

From the repository root:

```bash
npx serve docs
```

Or open `docs/index.html` with VS Code Live Server / any static file server.

## Assets

Copy your figures into `docs/assets/`:

| File | Description |
|------|-------------|
| `framework.png` | Simplified UrbanMIM pipeline (Fig. 1 style) |
| `data_examples.png` | Four-region RGB/GT examples (Fig. 2 style) |

See [docs/assets/README.md](docs/assets/README.md).

## GitHub Pages

1. Push this repository to GitHub.
2. **Settings → Pages**
3. **Build and deployment**: Deploy from a branch
4. **Branch**: `main` (or `master`) · **Folder**: `/docs`
5. Save and wait for the site URL (e.g. `https://<user>.github.io/<repo>/`).

## Structure

```
docs/
├── index.html
├── css/style.css
├── assets/
│   ├── framework.png
│   └── data_examples.png
└── js/          # reserved for map.js (Leaflet, later)
```

## Code

https://github.com/wwwjh333/Bal_lulc_atlas

## Interactive map

Inspired by the simpler static version of `webmap-demo` (ArcGIS MapServer URLs + layer toggles). This site uses **Leaflet + esri-leaflet** (no Next.js build).

Edit [`docs/config/map-config.json`](docs/config/map-config.json):

- `layers.naipUrl` — NAIP RGB MapServer
- `layers.predictionUrl` — UrbanMIM prediction MapServer
- `layers.groundTruthUrl` / `tileBoundaryUrl` — optional; hidden when empty
- `map.center` / `map.zoom` — initial view (lat, lng)
- `prediction.opacity` — default overlay opacity

Same demo endpoints as `D:\atlas_web\webmap-demo\public\config\app-config.json` are pre-filled for Baltimore tiles.
