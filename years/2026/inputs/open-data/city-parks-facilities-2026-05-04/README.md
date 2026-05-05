# City Parks And Recreation Facilities

Pulled: 2026-05-04T04:23:05+00:00

Source:

- City of Boise Open Data: Parks and Recreation Public and Administrative Facilities
- Item id: `f3f869a1a23648219560176e785d0c06`
- Page: https://opendata.cityofboise.org/datasets/f3f869a1a23648219560176e785d0c06_0/data
- Download: https://opendata.cityofboise.org/api/download/v1/items/f3f869a1a23648219560176e785d0c06/geojson?layers=0

Outputs:

- `parks_recreation_facilities.geojson` - WGS84 normalized facility points.
- `trailhead_candidates.geojson` - WGS84 subset where `facility_type == Trailhead`.

Counts:

- Facilities: 107
- Trailhead candidates: 36

Notes:

- The source GeoJSON downloads as EPSG:3857, so this pull converts coordinates to WGS84.
- `has_parking` is inferred from trailhead/parking naming only; confirm before relying on it.
- `has_restroom` and `has_water` are intentionally null until a source proves them.
