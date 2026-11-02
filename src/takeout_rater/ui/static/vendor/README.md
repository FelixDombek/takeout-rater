# Vendored third-party assets

Bundled so the UI works fully offline (no CDN requests).

| Path | Source | License |
|------|--------|---------|
| `leaflet/` | [Leaflet](https://leafletjs.com/) 1.9.4 (`dist/` from the npm package) | BSD-2-Clause, see `leaflet/LICENSE` |
| `world/countries-110m.geojson` | [world-atlas](https://github.com/topojson/world-atlas) 2.0.2 `countries-110m.json` (Natural Earth, public domain), converted from TopoJSON to GeoJSON with coordinates rounded to 0.01° and longitudes unwrapped across the antimeridian | ISC, see `world/LICENSE` |
