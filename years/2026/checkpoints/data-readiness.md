# 2026 Data Readiness Checkpoint

Before generating a 2026 plan, confirm:

- [x] Official 2026 challenge dates are known.
- [x] Official 2026 segment file is saved under `years/2026/inputs/official/`.
- [ ] Official 2026 segment count, trail count, distance, and elevation totals are recorded.
- [x] Directional requirements are parsed and validated.
- [x] Current Strava activity export is available or explicitly marked unavailable.
- [x] Strava API `activity:read_all` access is verified.
- [x] Current Strava API activity snapshot is saved.
- [x] Current challenge progress export is available or explicitly marked unavailable.
- [x] Connector trail data source is selected.
- [x] Road data source is selected.
- [x] Elevation/DEM source is selected.
- [ ] Personal time and mileage constraints are filled in.
- [x] Baseline 2025 metrics are linked for comparison.

## Current Official 2026 Pull

Source: `years/2026/inputs/official/api-pull-2026-05-04/`

- Challenge window: June 18, 2026 at 12:00:01 a.m. through July 18, 2026 at 11:59:59 p.m.
- Official on-foot trails: 101
- Official on-foot segments: 251
- Official on-foot distance: 164.43 miles
- Direction rules: 228 bidirectional foot segments, 23 ascent-only foot segments
- Trail data last changed: 2026-05-01 19:14:44
- Current account progress from dashboard API: 0.00%, 0 miles, 0 completed segment ids

Elevation totals are still not present in the official site API pull, so the count/distance/elevation checklist item remains open until we derive elevation from GPX/DEM or find an official elevation source.

Do not mark a generated 2026 plan as ready until segment coverage and direction rules have been checked against the official 2026 dataset.

## Current 2026 Open Data Pull

Source root: `years/2026/inputs/open-data/`

- Connector trails: `r2r-trails-2026-05-04/boise_parks_trails_open_data.geojson`
  - Source: Boise Parks / Ridge to Rivers ArcGIS `Boise_Parks_Trails_Open_Data` FeatureServer.
  - Features: 340.
  - Useful fields include `AllWeather`, `Condition`, `SpecialManagement`, access fields, surface, and trail subsystem.
- Roads and paths: `osm-2026-05-04/boise_planning_bbox.osm.pbf`
  - Source: Geofabrik Idaho OSM extract, redirect target `idaho-260503.osm.pbf`.
  - Source timestamp from PBF: 2026-05-03T20:21:30Z.
  - Clipped to padded planning bbox `[-116.432708, 43.475698, -115.907040, 43.852187]`.
- DEM: `dem-2026-05-04/usgs_13arcsec_boise_planning_bbox.tif`
  - Source: USGS 1/3 arc-second DEM tiles `n44w116` and `n44w117`, dated 2026-04-07.
  - Clipped raster size: 5,677 x 4,066 pixels.
  - Elevation range in clipped raster: 761.06 m to 2,311.25 m.

Open follow-up: derive official-segment elevation gain/loss from the DEM and record the method before treating elevation totals as current ground truth.
