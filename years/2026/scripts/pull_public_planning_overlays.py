#!/usr/bin/env python3
"""Pull small public overlays that improve route logistics.

This intentionally avoids adding another trail-line authority. The outputs are
trailhead/facility anchors and source notes for dynamic day-of overlays.
"""

from __future__ import annotations

import json
import math
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from personal_route_planner import (  # noqa: E402
    DEFAULT_CONNECTOR_GEOJSON,
    DEFAULT_OFFICIAL_GEOJSON,
    flatten_coordinates,
    haversine_miles,
    load_official_segments,
    point_to_polyline_distance_miles,
    read_json,
    write_json,
)


CITY_FACILITIES_ITEM_ID = "f3f869a1a23648219560176e785d0c06"
CITY_FACILITIES_URL = (
    "https://opendata.cityofboise.org/api/download/v1/items/"
    f"{CITY_FACILITIES_ITEM_ID}/geojson?layers=0"
)
CITY_FACILITIES_PAGE = (
    "https://opendata.cityofboise.org/datasets/"
    f"{CITY_FACILITIES_ITEM_ID}_0/data"
)
OUTPUT_DIR = YEAR_DIR / "inputs" / "open-data" / "city-parks-facilities-2026-05-04"
DYNAMIC_OVERLAYS_DIR = YEAR_DIR / "inputs" / "open-data" / "dynamic-overlays-2026-05-04"
EARTH_RADIUS_WEB_MERCATOR_M = 6378137.0


def web_mercator_to_wgs84(x: float, y: float) -> tuple[float, float]:
    lon = (x / EARTH_RADIUS_WEB_MERCATOR_M) * 180 / math.pi
    lat = (
        2 * math.atan(math.exp(y / EARTH_RADIUS_WEB_MERCATOR_M)) - math.pi / 2
    ) * 180 / math.pi
    return lon, lat


def fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "boise-trails-ai/2026-planner"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def load_connector_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = read_json(path)
    lines = []
    for feature in data.get("features", []):
        coords = flatten_coordinates(feature.get("geometry") or {})
        if len(coords) < 2:
            continue
        props = feature.get("properties") or {}
        lines.append(
            {
                "id": props.get("TrailID") or props.get("OBJECTID"),
                "name": props.get("TrailName") or props.get("Name"),
                "coords": coords,
            }
        )
    return lines


def nearest_official_segment(
    point: tuple[float, float],
    official_segments: list[dict[str, Any]],
) -> dict[str, Any] | None:
    best = None
    for segment in official_segments:
        distance = point_to_polyline_distance_miles(point, segment["coordinates"])
        if best is None or distance < best["distance_miles"]:
            best = {
                "segment_id": segment["seg_id"],
                "trail_name": segment["trail_name"],
                "distance_miles": distance,
            }
    return best


def nearest_connector_line(
    point: tuple[float, float],
    connector_lines: list[dict[str, Any]],
) -> dict[str, Any] | None:
    best = None
    for line in connector_lines:
        distance = point_to_polyline_distance_miles(point, line["coords"])
        if best is None or distance < best["distance_miles"]:
            best = {
                "line_id": line["id"],
                "line_name": line["name"],
                "distance_miles": distance,
            }
    return best


def normalize_facility_feature(
    feature: dict[str, Any],
    official_segments: list[dict[str, Any]],
    connector_lines: list[dict[str, Any]],
) -> dict[str, Any]:
    props = feature.get("properties") or {}
    coords = feature.get("geometry", {}).get("coordinates") or []
    lon, lat = web_mercator_to_wgs84(float(coords[0]), float(coords[1]))
    point = (lon, lat)
    facil_type = str(props.get("FacilType") or "")
    facility_name = str(props.get("FacilityName") or "")
    is_trailhead = facil_type.lower() == "trailhead"
    official = nearest_official_segment(point, official_segments)
    connector = nearest_connector_line(point, connector_lines)
    has_named_parking = "parking" in facility_name.lower()

    normalized = {
        "facility_id": props.get("FacilityID") or props.get("OBJECTID"),
        "facility_name": facility_name,
        "facility_type": facil_type,
        "facility_status": props.get("FacilityStatus"),
        "is_trailhead": is_trailhead,
        "has_parking": True if is_trailhead or has_named_parking else None,
        "has_restroom": None,
        "has_water": None,
        "parking_confidence": "inferred_from_trailhead_layer" if is_trailhead else "unknown",
        "address": props.get("Address"),
        "lat": round(lat, 7),
        "lon": round(lon, 7),
        "nearest_official_segment_id": official["segment_id"] if official else None,
        "nearest_official_trail_name": official["trail_name"] if official else None,
        "nearest_official_distance_miles": round(official["distance_miles"], 3)
        if official
        else None,
        "nearest_open_trail_id": connector["line_id"] if connector else None,
        "nearest_open_trail_name": connector["line_name"] if connector else None,
        "nearest_open_trail_distance_miles": round(connector["distance_miles"], 3)
        if connector
        else None,
        "source_objectid": props.get("OBJECTID"),
    }
    return {
        "type": "Feature",
        "id": feature.get("id"),
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": normalized,
    }


def write_readme(raw_count: int, trailhead_count: int) -> None:
    pulled = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    text = f"""# City Parks And Recreation Facilities

Pulled: {pulled}

Source:

- City of Boise Open Data: Parks and Recreation Public and Administrative Facilities
- Item id: `{CITY_FACILITIES_ITEM_ID}`
- Page: {CITY_FACILITIES_PAGE}
- Download: {CITY_FACILITIES_URL}

Outputs:

- `parks_recreation_facilities.geojson` - WGS84 normalized facility points.
- `trailhead_candidates.geojson` - WGS84 subset where `facility_type == Trailhead`.

Counts:

- Facilities: {raw_count}
- Trailhead candidates: {trailhead_count}

Notes:

- The source GeoJSON downloads as EPSG:3857, so this pull converts coordinates to WGS84.
- `has_parking` is inferred from trailhead/parking naming only; confirm before relying on it.
- `has_restroom` and `has_water` are intentionally null until a source proves them.
"""
    (OUTPUT_DIR / "README.md").write_text(text)


def write_dynamic_overlay_notes() -> None:
    DYNAMIC_OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)
    text = """# Dynamic Planning Overlays

Created: 2026-05-04

These sources are not static route geometry inputs. Use them during scheduling
and pre-run validation.

## Ridge To Rivers Conditions

- Interactive map: https://gisprod.adacounty.id.gov/apps/r2r/
- City of Boise explainer: https://www.cityofboise.org/news/parks-and-recreation/2022/november/new-interactive-map-feature-allows-ridge-to-rivers-users-to-check-real-time-trail-conditions/

Use as `day_of_trail_status.geojson` or equivalent when an extract is available.
Fields to normalize: `trail_id_or_name`, `condition_status`, `last_updated`,
`closure_flag`, `avoid_flag`, `all_weather_flag`.

## Ridge To Rivers Closures And Advisories

- Trail news / closure example: https://www.ridgetorivers.org/trail-news/seasonal-ridge-to-rivers-trail-closures-start-in-december-to-prevent-damage-protect-wildlife-habitat/

Use as a route-validity warning layer. Do not mark a generated route field-ready
when it crosses a currently closed trail or relies on closed vehicle access.

## ACHD Roadwork / RITA

- Roadwork in the Area: https://www.achdidaho.org/my-commute/roadwork-in-the-area
- Traffic advisories: https://www.achdidaho.org/my-commute/traffic/news-alerts

Use for drive-time reliability and alternate trailhead selection. Do not replace
OSM/R2R trail routing with this data.
"""
    (DYNAMIC_OVERLAYS_DIR / "README.md").write_text(text)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = fetch_json(CITY_FACILITIES_URL)
    official_segments, _ = load_official_segments(DEFAULT_OFFICIAL_GEOJSON)
    connector_lines = load_connector_lines(DEFAULT_CONNECTOR_GEOJSON)
    normalized_features = [
        normalize_facility_feature(feature, official_segments, connector_lines)
        for feature in raw.get("features", [])
        if (feature.get("geometry") or {}).get("type") == "Point"
    ]
    all_geojson = {
        "type": "FeatureCollection",
        "source": CITY_FACILITIES_PAGE,
        "features": normalized_features,
    }
    trailheads = [
        feature
        for feature in normalized_features
        if feature["properties"].get("is_trailhead")
    ]
    trailhead_geojson = {
        "type": "FeatureCollection",
        "source": CITY_FACILITIES_PAGE,
        "features": trailheads,
    }
    write_json(OUTPUT_DIR / "parks_recreation_facilities.geojson", all_geojson)
    write_json(OUTPUT_DIR / "trailhead_candidates.geojson", trailhead_geojson)
    write_readme(len(normalized_features), len(trailheads))
    write_dynamic_overlay_notes()
    print(f"Wrote {OUTPUT_DIR / 'parks_recreation_facilities.geojson'}")
    print(f"Wrote {OUTPUT_DIR / 'trailhead_candidates.geojson'}")
    print(f"Wrote {DYNAMIC_OVERLAYS_DIR / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
