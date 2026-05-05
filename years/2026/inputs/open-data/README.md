# 2026 Open Data Inputs

Pulled: 2026-05-04

These datasets are supplemental planning inputs. They do not define challenge completion; the official Boise Trails Challenge segment dataset remains authoritative.

## Pulled Datasets

- `r2r-trails-2026-05-04/` - Boise Parks / Ridge to Rivers trail feature layer from ArcGIS REST.
- `osm-2026-05-04/` - Geofabrik current Idaho OSM PBF plus a clipped Boise planning bbox extract.
- `dem-2026-05-04/` - USGS 1/3 arc-second DEM clipped to the 2026 planning bbox.

## Planning Bounds

Combined official 2026 foot segments plus refreshed Boise Parks / Ridge to Rivers trails, padded by 0.05 degrees:

`[-116.432708, 43.475698, -115.907040, 43.852187]`

## Large Files

The local `.osm.pbf` and `.tif` files are ignored by git. Keep their README and summary files committed so the pulls can be reproduced.
