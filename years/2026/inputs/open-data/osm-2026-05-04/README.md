# OpenStreetMap Road And Path Network

Pulled: 2026-05-04

Source:

- `https://download.geofabrik.de/north-america/us/idaho-latest.osm.pbf`

Files:

- `idaho-latest.osm.pbf` - full Idaho extract from Geofabrik, ignored by git.
- `boise_planning_bbox.osm.pbf` - clipped extract for 2026 planning bounds, ignored by git.
- `geofabrik_headers.txt` - HTTP headers and redirect provenance.

Source metadata from `osmium fileinfo`:

- Geofabrik redirect target: `idaho-260503.osm.pbf`
- Source timestamp: 2026-05-03T20:21:30Z
- Source size: 124,732,485 bytes

Clip command:

```bash
osmium extract -b -116.432708,43.475698,-115.907040,43.852187 \
  -o years/2026/inputs/open-data/osm-2026-05-04/boise_planning_bbox.osm.pbf \
  --overwrite \
  years/2026/inputs/open-data/osm-2026-05-04/idaho-latest.osm.pbf
```

Use the clipped extract for road connectors, Greenbelt connectors, neighborhood links, access roads, and bailout routing.
