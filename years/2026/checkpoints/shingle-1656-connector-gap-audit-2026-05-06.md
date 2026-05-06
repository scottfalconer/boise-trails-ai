# Shingle Creek 1656 Connector Gap Audit

Date: 2026-05-06

Objective: determine whether Shingle Creek `1656` is failing the p90 proof
because the routing graph is missing a short legal connector from nearby
parking to the official lower endpoint.

## Verdict

No promotable short connector found.

The closer OSM parking areas are closer by straight-line distance, but the
source graph does not show a legal direct connector from those parking areas to
the Shingle lower endpoint. The graph routes them through service/Hawkins-side
links and then around to Dry Creek / Shingle, which is why they are worse than
the Dry Creek / Sweet Connie roadside start.

## Key Points

Official Shingle lower endpoint:

```text
[-116.153952, 43.702616]
```

Nearby tested starts:

| Start | Straight-line to lower endpoint | Graph access to lower endpoint | Result |
|---|---:|---:|---|
| Dry Creek / Sweet Connie roadside parking | 1.59 mi | 1.73 mi | best current legal graph start |
| Lower Shingle / Dry Creek OSM parking west | 0.70 mi | 3.91 mi | worse |
| Lower Shingle / Dry Creek OSM parking east | 0.77 mi | 4.06 mi | worse |

## OSM / Graph Evidence

Overpass around the Shingle lower endpoint returned relevant ways:

- `#79 Shingle Creek`: `highway=path`, `foot=designated`, `bicycle=designated`,
  `motor_vehicle=no`.
- `#78 Dry Creek`: `highway=path`, `foot=designated`, `bicycle=designated`.
- `Lower Shingle / Dry Creek OSM parking west`: OSM way `1173130562`,
  `amenity=parking`.
- `Lower Shingle / Dry Creek OSM parking east`: OSM way `1173130563`,
  `amenity=parking`.
- Multiple Hawkins Reserve Loop paths, including some `foot=designated` and
  some `foot=no` directional split ways.
- Several nearby private service/driveway ways around the broader lower
  Bogus/Dry Creek area.

Connector graph inspection:

- Parking west nearest graph node: `(-116.1679, 43.7044)`.
  Immediate edges are OSM service connector edges.
- Parking east nearest graph node: `(-116.1689, 43.7053)`.
  Immediate edge is toward Hawkins.
- Shingle lower endpoint nearest graph node: `(-116.154, 43.7026)`.
  Immediate edges are `#79 Shingle Creek`, `#78 Dry Creek`, and official
  repeats for Dry Creek / Shingle.

This means the current graph is not merely failing to snap the parking lot to a
nearby trail endpoint. The available legal graph edges connect those parking
lots to a different side of the local trail/road network than the Shingle lower
endpoint.

## Decision

Do not add a synthetic shortcut from the closer OSM parking areas to the
Shingle lower endpoint. That would create exactly the kind of fake connector the
project is trying to avoid.

The only acceptable ways to improve Shingle `1656` are:

- verify a real signed/legal connector between one of those parking areas and
  the Shingle lower endpoint,
- find a different legal parking/start with better graph access,
- manually design and validate a shorter GPX using real public trails/roads,
- allow a p90 exception or larger bound,
- or explicitly allow a non-default transport variant.

## Validation

The connector-path checks used the current fused connector graph:

```text
years/2026/inputs/open-data/routing-connectors-2026-05-04/combined_r2r_osm_connectors.geojson
```

No private/no-foot shortcut was promoted.
