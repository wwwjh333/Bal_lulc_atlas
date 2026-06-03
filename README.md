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

## Later: interactive map

Replace the `.map-placeholder` block in `index.html` with a Leaflet `#map` div, add `docs/js/map.js`, and tile directories under `docs/atlas_tiles/` (naip, urbanmim, gt). Class colors in `style.css` match the paper legend.
