#!/usr/bin/env python3
"""Build a rerunnable personal route menu for the 2026 Boise Trails Challenge.

The planner is intentionally conservative: it treats the official challenge
GeoJSON as truth, removes completed and blocked segments from the remaining
set, estimates outing logistics, and ranks candidates inside time buckets. It
does not mutate any external service.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402

DEFAULT_OFFICIAL_GEOJSON = (
    YEAR_DIR
    / "inputs"
    / "official"
    / "api-pull-2026-06-13"
    / "official_foot_segments.geojson"
)
DEFAULT_STATE_PATH = YEAR_DIR / "inputs" / "personal" / "2026-planner-state.example.json"
DEFAULT_STRAVA_DETAILS_DIR = (
    YEAR_DIR / "inputs" / "strava" / "api-pulls" / "2026-05-03" / "activity_details"
)
DEFAULT_ACTIVITY_SUMMARY_CSV = (
    YEAR_DIR / "inputs" / "strava" / "api-pulls" / "2026-05-03" / "activities_summary.csv"
)
DEFAULT_ACTIVITY_DETAIL_SUMMARY_CSV = (
    YEAR_DIR
    / "inputs"
    / "strava"
    / "api-pulls"
    / "2026-05-03"
    / "activity_detail_summaries.csv"
)
DEFAULT_SEGMENT_PERF_CSV = (
    REPO_ROOT
    / "archive"
    / "legacy-root-2025"
    / "projects"
    / "research-20260503-boise-trails-performance"
    / "derived"
    / "segment_perf_runs.csv"
)
DEFAULT_CONNECTOR_GEOJSON = (
    YEAR_DIR
    / "inputs"
    / "open-data"
    / "routing-connectors-2026-05-04"
    / "combined_r2r_osm_connectors.geojson"
)
DEFAULT_R2R_CONNECTOR_GEOJSON = (
    YEAR_DIR
    / "inputs"
    / "open-data"
    / "r2r-trails-2026-05-04"
    / "boise_parks_trails_open_data.geojson"
)
DEFAULT_SPECIAL_MANAGEMENT_RULES_JSON = (
    YEAR_DIR / "inputs" / "open-data" / "special-management-rules-2026.json"
)
DEFAULT_TRAILHEAD_CANDIDATES_GEOJSON = (
    YEAR_DIR
    / "inputs"
    / "open-data"
    / "city-parks-facilities-2026-05-04"
    / "trailhead_candidates.geojson"
)
DEFAULT_PRIVATE_PARKING_ANCHORS_GEOJSON = (
    YEAR_DIR / "inputs" / "personal" / "private" / "strava-parking-anchors-v1.geojson"
)
DEFAULT_DEM_TIF = (
    YEAR_DIR
    / "inputs"
    / "open-data"
    / "dem-2026-05-04"
    / "usgs_13arcsec_boise_planning_bbox.tif"
)
DEFAULT_DEM_SUMMARY_JSON = (
    YEAR_DIR / "inputs" / "open-data" / "dem-2026-05-04" / "dem_summary.json"
)

MILES_PER_FOOT = 1 / 5280
METERS_PER_MILE = 1609.344
TIME_BUCKETS = [
    ("under_1_hour", 0, 60),
    ("one_to_two_hours", 61, 119),
    ("two_to_three_hours", 120, 180),
    ("three_to_four_hours", 181, 240),
    ("four_plus_hours", 241, None),
]
ON_FOOT_SPORTS = {"Run", "TrailRun", "Hike", "Walk"}
ElevationSampler = Callable[[tuple[float, float]], float | None]
GRAPH_VALIDATED_LINK_SOURCES = {"mapped_graph", "mapped_graph_target_snap"}
TARGET_SNAP_OFFICIAL_TOLERANCE_MILES = 0.03
NAME_STOPWORDS = {
    "trail",
    "trails",
    "the",
    "to",
    "of",
    "from",
    "and",
    "up",
    "down",
    "climb",
    "lap",
    "loop",
    "cw",
    "ccw",
    "counter",
    "clockwise",
    "sign",
}


def read_json(path: Path) -> Any:
    with path.open() as fh:
        return json.load(fh)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")


def round_miles(value: float) -> float:
    return round(value + 1e-9, 2)


def ceil_minutes(value: float) -> int:
    return int(math.ceil(max(0, value) - 1e-9))


def haversine_miles(a: tuple[float, float], b: tuple[float, float]) -> float:
    lon1, lat1 = a
    lon2, lat2 = b
    radius_miles = 3958.7613
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    h = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * radius_miles * math.atan2(math.sqrt(h), math.sqrt(1 - h))


def flatten_coordinates(geometry: dict[str, Any]) -> list[tuple[float, float]]:
    coords = geometry.get("coordinates") or []
    if not coords:
        return []
    if geometry.get("type") == "MultiLineString":
        flat: list[tuple[float, float]] = []
        for part in coords:
            flat.extend((float(lon), float(lat)) for lon, lat, *_ in part)
        return flat
    return [(float(lon), float(lat)) for lon, lat, *_ in coords]


def iter_line_parts(geometry: dict[str, Any]) -> list[list[tuple[float, float]]]:
    coords = geometry.get("coordinates") or []
    if not coords:
        return []
    if geometry.get("type") == "MultiLineString":
        return [
            [(float(lon), float(lat)) for lon, lat, *_ in part]
            for part in coords
            if len(part) >= 2
        ]
    if geometry.get("type") == "LineString":
        return [[(float(lon), float(lat)) for lon, lat, *_ in coords]]
    return []


OSM_FOOT_CONNECTOR_HIGHWAYS = {"cycleway", "footway", "path", "pedestrian", "steps"}
OSM_PUBLIC_ROAD_HIGHWAYS = {
    "living_street",
    "primary",
    "residential",
    "secondary",
    "service",
    "tertiary",
    "track",
    "unclassified",
}
UNSAFE_ACCESS_VALUES = {"no", "private"}
R2R_BIKE_ONLY_USE_VALUES = {"bike only", "bikes only", "bicycle only", "bicycles only"}


def connector_access_properties(props: dict[str, Any]) -> dict[str, Any]:
    return {
        "access": props.get("access"),
        "foot": props.get("foot"),
        "highway": props.get("highway"),
        "source": props.get("source"),
    }


def unsafe_connector_access_reasons(props: dict[str, Any]) -> list[str]:
    reasons = []
    for key in ("access", "foot"):
        raw = props.get(key)
        if raw is None:
            continue
        value = str(raw).strip().lower()
        if value in UNSAFE_ACCESS_VALUES:
            reasons.append(f"{key}={value}")
    source = str(props.get("source") or "").strip().lower()
    if "ridge_to_rivers" in source or source == "r2r":
        r2r_use = str(props.get("R2R_Use") or "").strip().lower()
        if r2r_use in R2R_BIKE_ONLY_USE_VALUES:
            reasons.append("r2r_use=bike_only")
    return reasons


def connector_class_for_properties(props: dict[str, Any], edge_type: str = "connector") -> str:
    if edge_type == "official_repeat":
        return "official_repeat"
    explicit = props.get("connector_class")
    if explicit:
        return str(explicit)
    source = str(props.get("source") or "").lower()
    highway = str(props.get("highway") or "").lower()
    if "ridge_to_rivers" in source or source == "r2r":
        return "r2r_trail"
    if source == "openstreetmap":
        if highway in OSM_FOOT_CONNECTOR_HIGHWAYS:
            return "osm_path_footway"
        if highway in OSM_PUBLIC_ROAD_HIGHWAYS:
            return "osm_public_road"
    return "unknown_connector"


def line_length_miles(coords: list[tuple[float, float]]) -> float:
    return sum(haversine_miles(a, b) for a, b in zip(coords, coords[1:]))


def decode_polyline(polyline: str) -> list[tuple[float, float]]:
    """Decode an encoded Google/Strava polyline into lon/lat coordinates."""
    coords: list[tuple[float, float]] = []
    index = 0
    lat = 0
    lon = 0
    while index < len(polyline):
        for coord_index in range(2):
            shift = 0
            result = 0
            while True:
                byte = ord(polyline[index]) - 63
                index += 1
                result |= (byte & 0x1F) << shift
                shift += 5
                if byte < 0x20:
                    break
            delta = ~(result >> 1) if result & 1 else result >> 1
            if coord_index == 0:
                lat += delta
            else:
                lon += delta
        coords.append((lon / 1e5, lat / 1e5))
    return coords


def activity_geometry(activity: dict[str, Any]) -> list[tuple[float, float]]:
    activity_map = activity.get("map") or {}
    polyline = activity_map.get("polyline") or activity_map.get("summary_polyline")
    if not polyline:
        return []
    return decode_polyline(polyline)


def local_xy_miles(point: tuple[float, float], origin_lat: float) -> tuple[float, float]:
    lon, lat = point
    x = lon * 69.172 * math.cos(math.radians(origin_lat))
    y = lat * 69.0
    return x, y


def point_to_segment_distance_miles(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
    origin_lat: float,
) -> float:
    lon_scale = 69.172 * math.cos(math.radians(origin_lat))
    lat_scale = 69.0
    px, py = point[0] * lon_scale, point[1] * lat_scale
    ax, ay = start[0] * lon_scale, start[1] * lat_scale
    bx, by = end[0] * lon_scale, end[1] * lat_scale
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    nearest_x = ax + t * dx
    nearest_y = ay + t * dy
    return math.hypot(px - nearest_x, py - nearest_y)


def point_to_polyline_distance_miles(
    point: tuple[float, float],
    polyline: list[tuple[float, float]],
) -> float:
    if not polyline:
        return float("inf")
    if len(polyline) == 1:
        return haversine_miles(point, polyline[0])
    origin_lat = point[1]
    return min(
        point_to_segment_distance_miles(point, start, end, origin_lat)
        for start, end in zip(polyline, polyline[1:])
    )


def interpolate_point(
    start: tuple[float, float], end: tuple[float, float], fraction: float
) -> tuple[float, float]:
    return (
        start[0] + (end[0] - start[0]) * fraction,
        start[1] + (end[1] - start[1]) * fraction,
    )


def sample_line(coords: list[tuple[float, float]], spacing_miles: float = 0.03) -> list[tuple[float, float]]:
    if len(coords) <= 2:
        return coords
    samples = [coords[0]]
    for start, end in zip(coords, coords[1:]):
        distance = haversine_miles(start, end)
        steps = max(1, int(math.ceil(distance / spacing_miles)))
        for step in range(1, steps + 1):
            samples.append(interpolate_point(start, end, step / steps))
    return samples


def downsample_coords(
    coords: list[tuple[float, float]], max_points: int = 600
) -> list[tuple[float, float]]:
    if len(coords) <= max_points:
        return coords
    step = math.ceil(len(coords) / max_points)
    sampled = coords[::step]
    if sampled[-1] != coords[-1]:
        sampled.append(coords[-1])
    return sampled


def coordinate_bbox(coords: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    lons = [point[0] for point in coords]
    lats = [point[1] for point in coords]
    return min(lons), min(lats), max(lons), max(lats)


def bbox_overlaps(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    lon_buffer: float,
    lat_buffer: float,
) -> bool:
    return not (
        left[2] + lon_buffer < right[0]
        or right[2] + lon_buffer < left[0]
        or left[3] + lat_buffer < right[1]
        or right[3] + lat_buffer < left[1]
    )


def match_activity_geometry_to_segments(
    activity_coords: list[tuple[float, float]],
    official_segments: list[dict[str, Any]],
    threshold_miles: float = 0.045,
    min_fraction: float = 0.55,
) -> list[dict[str, Any]]:
    """Match a Strava activity path to official segments by geometry proximity."""
    if len(activity_coords) < 2:
        return []
    activity_bbox = coordinate_bbox(activity_coords)
    activity_match_coords = downsample_coords(activity_coords)
    origin_lat = sum(point[1] for point in activity_coords) / len(activity_coords)
    lat_buffer = threshold_miles / 69.0
    lon_buffer = threshold_miles / max(1e-6, 69.172 * math.cos(math.radians(origin_lat)))
    matches: list[dict[str, Any]] = []
    for segment in official_segments:
        segment_bbox = segment.get("bbox") or coordinate_bbox(segment["coordinates"])
        if not bbox_overlaps(activity_bbox, segment_bbox, lon_buffer, lat_buffer):
            continue
        samples = sample_line(segment["coordinates"])
        if not samples:
            continue
        close_count = sum(
            1
            for point in samples
            if point_to_polyline_distance_miles(point, activity_match_coords) <= threshold_miles
        )
        fraction = close_count / len(samples)
        if fraction >= min_fraction:
            matches.append(
                {
                    "seg_id": segment["seg_id"],
                    "seg_name": segment["seg_name"],
                    "trail_name": segment["trail_name"],
                    "official_miles": round_miles(segment["official_miles"]),
                    "direction": segment["direction"],
                    "match_fraction": round(fraction, 3),
                    "threshold_miles": threshold_miles,
                    "match_source": "activity_geometry",
                    "direction_verified": segment["direction"] == "both",
                }
            )
    return sorted(matches, key=lambda item: item["seg_id"])


def clean_trail_name(segment_name: str) -> str:
    return re.sub(r"\s+\d+$", "", segment_name.strip())


def normalize_name(name: str) -> str:
    cleaned = re.sub(r"['`]", "", name.lower())
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    cleaned = re.sub(r"\b\d+\b", " ", cleaned)
    return " ".join(cleaned.split())


def name_tokens(name: str) -> set[str]:
    return {token for token in normalize_name(name).split() if token not in NAME_STOPWORDS}


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * pct)
    return float(ordered[index])


def load_official_segments(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = read_json(path)
    segments: list[dict[str, Any]] = []
    for feature in data.get("features", []):
        props = feature.get("properties") or {}
        geometry = feature.get("geometry") or {}
        parts = iter_line_parts(geometry)
        if geometry.get("type") == "MultiLineString" and len(parts) > 1:
            raise ValueError(
                f"Official segment {props.get('segId')} is a MultiLineString; "
                "normalize multipart official geometry into explicit parts before routing"
            )
        coords = parts[0] if parts else []
        if not coords:
            continue
        seg_id = int(props["segId"])
        length_ft = float(props.get("LengthFt") or 0)
        segment_name = str(props.get("segName") or f"Segment {seg_id}")
        segments.append(
            {
                "seg_id": seg_id,
                "seg_name": segment_name,
                "trail_name": clean_trail_name(segment_name),
                "length_ft": length_ft,
                "official_miles": length_ft * MILES_PER_FOOT,
                "direction": props.get("direction") or "both",
                "spec_inst": props.get("specInst") or "",
                "activity_type": props.get("activity_type") or "",
                "coordinates": coords,
                "start": coords[0],
                "end": coords[-1],
                "center": center_point(coords),
                "bbox": coordinate_bbox(coords),
            }
        )
    return segments, {
        "path": str(path),
        "lastUpdatedUTC": data.get("lastUpdatedUTC"),
        "feature_count": len(data.get("features", [])),
    }


def center_point(coords: list[tuple[float, float]]) -> tuple[float, float]:
    lon = sum(point[0] for point in coords) / len(coords)
    lat = sum(point[1] for point in coords) / len(coords)
    return (lon, lat)


def load_dem_context(
    dem_tif: Path | None,
    dem_summary_json: Path | None,
) -> dict[str, Any]:
    if not dem_tif or not dem_tif.exists():
        return {
            "sampler": None,
            "metadata": {
                "path": str(dem_tif) if dem_tif else None,
                "loaded": False,
                "reason": "dem_tif_missing",
            },
        }
    if not dem_summary_json or not dem_summary_json.exists():
        return {
            "sampler": None,
            "metadata": {
                "path": str(dem_tif),
                "summary_path": str(dem_summary_json) if dem_summary_json else None,
                "loaded": False,
                "reason": "dem_summary_missing",
            },
        }
    try:
        from PIL import Image
    except ImportError:
        return {
            "sampler": None,
            "metadata": {
                "path": str(dem_tif),
                "summary_path": str(dem_summary_json),
                "loaded": False,
                "reason": "pillow_unavailable",
            },
        }

    summary = read_json(dem_summary_json)
    bounds = summary.get("requested_bounds_wgs84") or []
    if len(bounds) != 4:
        return {
            "sampler": None,
            "metadata": {
                "path": str(dem_tif),
                "summary_path": str(dem_summary_json),
                "loaded": False,
                "reason": "dem_bounds_missing",
            },
        }

    image = Image.open(dem_tif)
    pixels = image.load()
    width, height = image.size
    min_lon, min_lat, max_lon, max_lat = [float(value) for value in bounds]
    nodata = float(summary.get("nodata", -999999.0))

    def sample(point: tuple[float, float]) -> float | None:
        lon, lat = point
        if lon < min_lon or lon > max_lon or lat < min_lat or lat > max_lat:
            return None
        col = int(round((lon - min_lon) / (max_lon - min_lon) * (width - 1)))
        row = int(round((max_lat - lat) / (max_lat - min_lat) * (height - 1)))
        col = max(0, min(width - 1, col))
        row = max(0, min(height - 1, row))
        value = float(pixels[col, row])
        if value == nodata or value <= -10000:
            return None
        return value * 3.280839895

    return {
        "sampler": sample,
        "metadata": {
            "path": str(dem_tif),
            "summary_path": str(dem_summary_json),
            "loaded": True,
            "source": summary.get("source"),
            "crs": summary.get("crs"),
            "bounds_wgs84": bounds,
            "units": "feet",
        },
    }


def load_state(path: Path | None) -> dict[str, Any]:
    if path and path.exists():
        return read_json(path)
    return {}


def load_trailheads_from_geojson(path: Path | None) -> list[dict[str, Any]]:
    if not path or not path.exists():
        return []
    data = read_json(path)
    trailheads = []
    for feature in data.get("features", []):
        props = feature.get("properties") or {}
        geometry = feature.get("geometry") or {}
        coords = geometry.get("coordinates") or []
        if geometry.get("type") != "Point" or len(coords) < 2:
            continue
        name = props.get("facility_name") or props.get("FacilityName") or props.get("name")
        if not name:
            continue
        trailheads.append(
            {
                "name": str(name),
                "lat": float(props.get("lat") or coords[1]),
                "lon": float(props.get("lon") or coords[0]),
                "parking_minutes": 8,
                "facility_id": props.get("facility_id"),
                "has_parking": props.get("has_parking"),
                "has_restroom": props.get("has_restroom"),
                "has_water": props.get("has_water"),
                "source": props.get("source") or "city_parks_facilities",
                "parking_confidence": props.get("parking_confidence"),
                "privacy": props.get("privacy"),
            }
        )
    return trailheads


def merge_planning_trailheads(
    state: dict[str, Any],
    public_trailheads: list[dict[str, Any]],
) -> dict[str, Any]:
    if not public_trailheads:
        return state
    merged = dict(state)
    trailheads = []
    seen = set()
    for trailhead in [*public_trailheads, *(state.get("trailheads") or [])]:
        key = normalize_name(str(trailhead.get("name") or ""))
        if not key or key in seen:
            continue
        seen.add(key)
        trailheads.append(trailhead)
    merged["trailheads"] = trailheads
    return merged


def as_int_set(values: list[Any] | None) -> set[int]:
    result: set[int] = set()
    for value in values or []:
        if value == "":
            continue
        result.add(int(value))
    return result


def default_drive_model() -> dict[str, Any]:
    return {
        "origin_label": "North End planning origin",
        "origin_lat": 43.63,
        "origin_lon": -116.21,
        "straight_line_factor": 1.25,
        "minutes_per_mile": 2.2,
        "minimum_one_way_minutes": 5,
    }


def default_outing_model() -> dict[str, Any]:
    return {
        "connector_miles_per_official_mile": 0.08,
        "repeat_miles_per_official_mile": 0.0,
        "minimum_connector_miles": 0.0,
        "road_miles_per_outing": 0.0,
        "connector_return_factor": 1.25,
        "prefer_connector_if_shorter_than_repeat": True,
        "allow_official_out_and_back_fallback": True,
        "mapped_connector_snap_tolerance_miles": 0.02,
        "mapped_trailhead_access_max_miles": 1.25,
        "closed_loop_gap_tolerance_miles": 0.05,
    }


def get_drive_model(state: dict[str, Any]) -> dict[str, Any]:
    model = default_drive_model()
    model.update(state.get("drive_model") or {})
    return model


def get_outing_model(state: dict[str, Any]) -> dict[str, Any]:
    model = default_outing_model()
    model.update(state.get("outing_model") or {})
    return model


def load_paces_from_activity_csv(path: Path) -> list[float]:
    if not path.exists():
        return []
    paces: list[float] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("sport_type") not in ON_FOOT_SPORTS:
                continue
            try:
                distance_miles = float(row.get("distance_m") or 0) / METERS_PER_MILE
                moving_minutes = float(row.get("moving_time_s") or 0) / 60
            except ValueError:
                continue
            if distance_miles >= 0.5 and moving_minutes > 0:
                paces.append(moving_minutes / distance_miles)
    return paces


def load_paces_from_segment_perf(path: Path) -> list[float]:
    if not path.exists():
        return []
    paces: list[float] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                pace = float(row.get("pace_min_per_mi") or 0)
            except ValueError:
                continue
            if 5 <= pace <= 40:
                paces.append(pace)
    return paces


def load_strava_effort_paces(activity_details_dir: Path | None) -> dict[str, list[dict[str, Any]]]:
    if not activity_details_dir or not activity_details_dir.exists():
        return {}
    effort_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(activity_details_dir.glob("*.json")):
        try:
            activity = read_json(path)
        except json.JSONDecodeError:
            continue
        for effort in activity.get("segment_efforts") or []:
            segment = effort.get("segment") or {}
            effort_name = effort.get("name") or segment.get("name")
            if not effort_name:
                continue
            try:
                distance_m = float(effort.get("distance") or segment.get("distance") or 0)
                moving_time_s = float(effort.get("moving_time") or 0)
            except (TypeError, ValueError):
                continue
            distance_miles = distance_m / METERS_PER_MILE
            if distance_miles < 0.05 or moving_time_s <= 0:
                continue
            normalized = normalize_name(str(effort_name))
            effort_index[normalized].append(
                {
                    "pace_min_per_mile": (moving_time_s / 60) / distance_miles,
                    "moving_time_seconds": moving_time_s,
                    "distance_miles": distance_miles,
                    "matched_name": normalized,
                    "raw_name": effort_name,
                    "activity_id": activity.get("id"),
                    "activity_name": activity.get("name"),
                    "activity_date": str(activity.get("start_date_local") or "")[:10],
                    "strava_segment_id": segment.get("id"),
                    "tokens": name_tokens(str(effort_name)),
                }
            )
    return dict(effort_index)


def choose_fallback_pace(
    state: dict[str, Any],
    segment_perf_paces: list[float],
    detail_paces: list[float],
    activity_paces: list[float],
) -> tuple[float, str]:
    manual = state.get("pace_min_per_mile")
    if manual:
        return float(manual), "manual_state_pace"
    segment_perf_median = median(segment_perf_paces)
    if segment_perf_median:
        return segment_perf_median, "personal_history_segment_perf_runs"
    detail_median = median(detail_paces)
    if detail_median:
        return detail_median, "strava_detail_activity_median"
    activity_median = median(activity_paces)
    if activity_median:
        return activity_median, "strava_activity_summary_median"
    return 16.0, "default_16_min_per_mile"


def build_performance_profile(
    state: dict[str, Any],
    strava_activity_details_dir: Path | None,
    activity_summary_csv: Path | None,
    activity_detail_summary_csv: Path | None,
    segment_perf_csv: Path | None,
) -> dict[str, Any]:
    segment_perf_paces = load_paces_from_segment_perf(segment_perf_csv) if segment_perf_csv else []
    detail_paces = (
        load_paces_from_activity_csv(activity_detail_summary_csv)
        if activity_detail_summary_csv
        else []
    )
    activity_paces = load_paces_from_activity_csv(activity_summary_csv) if activity_summary_csv else []
    fallback_pace, fallback_source = choose_fallback_pace(
        state, segment_perf_paces, detail_paces, activity_paces
    )
    effort_index = load_strava_effort_paces(strava_activity_details_dir)
    return {
        "fallback_pace_min_per_mile": round(fallback_pace, 2),
        "fallback_pace_source": fallback_source,
        "matched_strava_segment_count": 0,
        "sources": {
            "strava_activity_details_dir": str(strava_activity_details_dir)
            if strava_activity_details_dir
            else None,
            "activity_summary_csv": str(activity_summary_csv) if activity_summary_csv else None,
            "activity_detail_summary_csv": str(activity_detail_summary_csv)
            if activity_detail_summary_csv
            else None,
            "segment_perf_csv": str(segment_perf_csv) if segment_perf_csv else None,
        },
        "pace_samples": {
            "segment_perf_run_count": len(segment_perf_paces),
            "segment_perf_median_min_per_mile": round(median(segment_perf_paces), 2)
            if segment_perf_paces
            else None,
            "segment_perf_75th_percentile_min_per_mile": round(percentile(segment_perf_paces, 0.75), 2)
            if segment_perf_paces
            else None,
            "strava_detail_activity_count": len(detail_paces),
            "strava_detail_median_min_per_mile": round(median(detail_paces), 2)
            if detail_paces
            else None,
            "strava_activity_summary_count": len(activity_paces),
            "strava_activity_summary_median_min_per_mile": round(median(activity_paces), 2)
            if activity_paces
            else None,
            "strava_segment_effort_name_count": len(effort_index),
        },
        "_effort_index": effort_index,
    }


def match_effort_for_segment(
    segment: dict[str, Any], effort_index: dict[str, list[dict[str, Any]]]
) -> dict[str, Any] | None:
    if not effort_index:
        return None
    candidates = [
        normalize_name(segment["trail_name"]),
        normalize_name(segment["seg_name"]),
    ]
    for candidate in candidates:
        if candidate in effort_index:
            return summarize_effort_match(candidate, effort_index[candidate])

    segment_tokens = name_tokens(segment["trail_name"])
    if not segment_tokens:
        return None
    best: tuple[float, str, list[dict[str, Any]]] | None = None
    for effort_name, records in effort_index.items():
        effort_tokens = records[0].get("tokens") or set()
        if not effort_tokens:
            continue
        overlap = len(segment_tokens & effort_tokens)
        if overlap < 1:
            continue
        score = overlap / max(len(segment_tokens), 1)
        if normalize_name(segment["trail_name"]) in effort_name or effort_name in normalize_name(segment["trail_name"]):
            score += 0.25
        if score >= 0.6 and (best is None or score > best[0]):
            best = (score, effort_name, records)
    if best:
        return summarize_effort_match(best[1], best[2])
    return None


def summarize_effort_match(matched_name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    paces = [record["pace_min_per_mile"] for record in records]
    # Never emit raw Strava activity/segment identifiers, titles, or dates into
    # shareable plan data (AGENTS.md privacy). Only the pace summary is consumed
    # downstream; the example_* identifiers were dead-weight private data.
    return {
        "source_type": "matched_strava_segment_effort",
        "matched_name": matched_name,
        "sample_count": len(records),
        "pace_min_per_mile": round(float(statistics.median(paces)), 2),
    }


def estimate_segment_time(
    segment: dict[str, Any],
    performance_profile: dict[str, Any],
) -> dict[str, Any]:
    match = match_effort_for_segment(segment, performance_profile.get("_effort_index") or {})
    if match:
        pace = float(match["pace_min_per_mile"])
        source = match
    else:
        pace = float(performance_profile["fallback_pace_min_per_mile"])
        source = {
            "source_type": performance_profile["fallback_pace_source"],
            "pace_min_per_mile": round(pace, 2),
        }
    moving = ceil_minutes(segment["official_miles"] * pace)
    return {
        "seg_id": segment["seg_id"],
        "seg_name": segment["seg_name"],
        "trail_name": segment["trail_name"],
        "official_miles": round_miles(segment["official_miles"]),
        "direction": segment["direction"],
        "start": segment["start"],
        "end": segment["end"],
        "coordinates": segment["coordinates"],
        "estimated_moving_minutes": moving,
        "time_source": source,
    }


def add_graph_edge(
    graph: dict[tuple[float, float], list[dict[str, Any]]],
    start: tuple[float, float],
    end: tuple[float, float],
    distance: float,
    name: str,
    edge_type: str,
    seg_id: int | None = None,
    connector_class: str = "unknown_connector",
    source: str | None = None,
    highway: str | None = None,
    access_properties: dict[str, Any] | None = None,
    coordinates: list[tuple[float, float]] | None = None,
    official_traversal_direction: str | None = None,
    special_management_allowed_directions: list[str] | None = None,
) -> None:
    if distance <= 0:
        return
    graph[start].append(
        {
            "to": end,
            "distance": distance,
            "name": name,
            "edge_type": edge_type,
            "seg_id": seg_id,
            "connector_class": connector_class,
            "source": source,
            "highway": highway,
            "access_properties": access_properties or {},
            "coordinates": [[point[0], point[1]] for point in coordinates or [start, end]],
            "official_traversal_direction": official_traversal_direction,
            "special_management_allowed_directions": special_management_allowed_directions or [],
        }
    )


def add_polyline_to_graph(
    graph: dict[tuple[float, float], list[dict[str, Any]]],
    nodes: list[tuple[float, float]],
    coords: list[tuple[float, float]],
    name: str,
    edge_type: str,
    bidirectional: bool = True,
    seg_id: int | None = None,
    distance_scale: float = 1.0,
    connector_class: str = "unknown_connector",
    source: str | None = None,
    highway: str | None = None,
    access_properties: dict[str, Any] | None = None,
    official_edge_overrides: dict[tuple[tuple[float, float], tuple[float, float]], dict[str, Any]] | None = None,
    official_named_overlaps: dict[str, list[dict[str, Any]]] | None = None,
    special_management_direction_overrides: dict[int, list[str]] | None = None,
    special_management_allowed_directions: list[str] | None = None,
) -> None:
    for start_raw, end_raw in zip(coords, coords[1:]):
        start = round_node(start_raw)
        end = round_node(end_raw)
        distance = haversine_miles(start_raw, end_raw) * distance_scale
        override = (official_edge_overrides or {}).get((start, end))
        if not override and edge_type != "official_repeat":
            override = official_overlap_override_for_connector_edge(
                start_raw,
                end_raw,
                name,
                official_named_overlaps,
                special_management_direction_overrides=special_management_direction_overrides,
            )
        edge_name = str(override.get("name") or name) if override else name
        edge_type_value = str(override.get("edge_type") or edge_type) if override else edge_type
        edge_seg_id = override.get("seg_id") if override else seg_id
        edge_connector_class = (
            str(override.get("connector_class") or connector_class) if override else connector_class
        )
        edge_source = str(override.get("source") or source) if override else source
        edge_access_properties = (
            dict(override.get("access_properties") or {}) if override else access_properties
        )
        edge_traversal_direction = (
            str(override.get("official_traversal_direction") or "")
            if override and override.get("official_traversal_direction")
            else ("forward" if edge_type == "official_repeat" else None)
        )
        edge_special_allowed = (
            list(override.get("special_management_allowed_directions") or [])
            if override
            else list(special_management_allowed_directions or [])
        )
        add_graph_edge(
            graph,
            start,
            end,
            distance,
            edge_name,
            edge_type_value,
            edge_seg_id,
            edge_connector_class,
            edge_source,
            highway,
            edge_access_properties,
            [start_raw, end_raw],
            edge_traversal_direction,
            edge_special_allowed,
        )
        if bidirectional:
            reverse_override = (official_edge_overrides or {}).get((end, start)) or override
            reverse_name = str(reverse_override.get("name") or name) if reverse_override else name
            reverse_edge_type = str(reverse_override.get("edge_type") or edge_type) if reverse_override else edge_type
            reverse_seg_id = reverse_override.get("seg_id") if reverse_override else seg_id
            reverse_connector_class = (
                str(reverse_override.get("connector_class") or connector_class)
                if reverse_override
                else connector_class
            )
            reverse_source = str(reverse_override.get("source") or source) if reverse_override else source
            reverse_access_properties = (
                dict(reverse_override.get("access_properties") or {})
                if reverse_override
                else access_properties
            )
            reverse_traversal_direction = (
                str(reverse_override.get("official_traversal_direction") or "")
                if reverse_override and reverse_override.get("official_traversal_direction")
                else ("reverse" if edge_type == "official_repeat" else None)
            )
            reverse_special_allowed = (
                list(reverse_override.get("special_management_allowed_directions") or [])
                if reverse_override
                else list(special_management_allowed_directions or [])
            )
            add_graph_edge(
                graph,
                end,
                start,
                distance,
                reverse_name,
                reverse_edge_type,
                reverse_seg_id,
                reverse_connector_class,
                reverse_source,
                highway,
                reverse_access_properties,
                [end_raw, start_raw],
                reverse_traversal_direction,
                reverse_special_allowed,
            )
        nodes.extend([start, end])


def official_edge_override_index(
    official_segments: list[dict[str, Any]] | None,
    special_management_direction_overrides: dict[int, list[str]] | None = None,
) -> dict[tuple[tuple[float, float], tuple[float, float]], dict[str, Any]]:
    index: dict[tuple[tuple[float, float], tuple[float, float]], dict[str, Any]] = {}
    for segment in official_segments or []:
        for start_raw, end_raw in zip(segment.get("coordinates") or [], (segment.get("coordinates") or [])[1:]):
            start = round_node(start_raw)
            end = round_node(end_raw)
            allowed_directions = (special_management_direction_overrides or {}).get(
                int(segment.get("seg_id"))
            )
            index[(start, end)] = official_repeat_edge_override(
                segment,
                official_traversal_direction="forward",
                special_management_allowed_directions=allowed_directions,
            )
            index[(end, start)] = official_repeat_edge_override(
                segment,
                official_traversal_direction="reverse",
                special_management_allowed_directions=allowed_directions,
            )
    return index


def official_repeat_edge_override(
    segment: dict[str, Any],
    official_traversal_direction: str | None = None,
    special_management_allowed_directions: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "edge_type": "official_repeat",
        "seg_id": segment.get("seg_id"),
        "connector_class": "official_repeat",
        "name": segment.get("trail_name") or segment.get("seg_name") or "official segment",
        "source": "official_challenge_connector_overlap",
        "official_traversal_direction": official_traversal_direction,
        "special_management_allowed_directions": special_management_allowed_directions or [],
        "access_properties": {
            "access": None,
            "foot": None,
            "highway": None,
            "source": "official_challenge_connector_overlap",
        },
    }


def official_named_overlap_index(
    official_segments: list[dict[str, Any]] | None,
) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for segment in official_segments or []:
        trail_key = normalize_name(str(segment.get("trail_name") or segment.get("seg_name") or ""))
        if trail_key:
            index[trail_key].append(segment)
    return dict(index)


def fraction_along_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
    reference_lat: float,
) -> float:
    projected_point = projected_node(point, reference_lat)
    projected_start = projected_node(start, reference_lat)
    projected_end = projected_node(end, reference_lat)
    dx = projected_end[0] - projected_start[0]
    dy = projected_end[1] - projected_start[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 0:
        return 0.0
    return max(
        0.0,
        min(
            1.0,
            (
                (projected_point[0] - projected_start[0]) * dx
                + (projected_point[1] - projected_start[1]) * dy
            )
            / length_sq,
        ),
    )


def fraction_along_polyline(point: tuple[float, float], coords: list[tuple[float, float]]) -> float:
    if len(coords) < 2:
        return 0.0
    total_miles = line_length_miles(coords)
    if total_miles <= 0:
        return 0.0
    reference_lat = sum(coord[1] for coord in coords) / len(coords)
    best_distance = math.inf
    best_fraction = 0.0
    distance_before = 0.0
    for start, end in zip(coords, coords[1:]):
        segment_miles = haversine_miles(start, end)
        local_fraction = fraction_along_segment(point, start, end, reference_lat)
        projected = (
            start[0] + (end[0] - start[0]) * local_fraction,
            start[1] + (end[1] - start[1]) * local_fraction,
        )
        distance = haversine_miles(point, projected)
        if distance < best_distance:
            best_distance = distance
            best_fraction = (distance_before + (segment_miles * local_fraction)) / total_miles
        distance_before += segment_miles
    return max(0.0, min(1.0, best_fraction))


def connector_edge_direction_for_official_segment(
    start_raw: tuple[float, float],
    end_raw: tuple[float, float],
    segment: dict[str, Any],
) -> str:
    coords = segment.get("coordinates") or []
    start_fraction = fraction_along_polyline(start_raw, coords)
    end_fraction = fraction_along_polyline(end_raw, coords)
    return "forward" if end_fraction >= start_fraction else "reverse"


def official_overlap_override_for_connector_edge(
    start_raw: tuple[float, float],
    end_raw: tuple[float, float],
    connector_name: str,
    official_named_overlaps: dict[str, list[dict[str, Any]]] | None,
    special_management_direction_overrides: dict[int, list[str]] | None = None,
    threshold_miles: float = 0.025,
) -> dict[str, Any] | None:
    if not official_named_overlaps:
        return None
    candidates = official_named_overlaps.get(normalize_name(str(connector_name or ""))) or []
    if not candidates:
        return None
    midpoint = (
        (float(start_raw[0]) + float(end_raw[0])) / 2,
        (float(start_raw[1]) + float(end_raw[1])) / 2,
    )
    edge_bbox = coordinate_bbox([start_raw, end_raw])
    origin_lat = (float(start_raw[1]) + float(end_raw[1])) / 2
    lat_buffer = threshold_miles / 69.0
    lon_buffer = threshold_miles / max(1e-6, 69.172 * math.cos(math.radians(origin_lat)))
    for segment in candidates:
        coords = segment.get("coordinates") or []
        if len(coords) < 2:
            continue
        segment_bbox = segment.get("bbox") or coordinate_bbox(coords)
        if not bbox_overlaps(edge_bbox, segment_bbox, lon_buffer, lat_buffer):
            continue
        distances = [
            point_to_polyline_distance_miles(point, coords)
            for point in (start_raw, midpoint, end_raw)
        ]
        if max(distances) <= threshold_miles:
            seg_id = int(segment.get("seg_id"))
            return official_repeat_edge_override(
                segment,
                official_traversal_direction=connector_edge_direction_for_official_segment(
                    start_raw,
                    end_raw,
                    segment,
                ),
                special_management_allowed_directions=(
                    special_management_direction_overrides or {}
                ).get(seg_id),
            )
    return None


def load_special_management_direction_overrides(
    rules_json: Path | None = DEFAULT_SPECIAL_MANAGEMENT_RULES_JSON,
) -> dict[int, list[str]]:
    if not rules_json or not rules_json.exists():
        return {}
    try:
        payload = read_json(rules_json)
    except json.JSONDecodeError:
        return {}
    overrides: dict[int, list[str]] = {}
    for rule in payload.get("rules") or []:
        if str(rule.get("rule_type") or "") != "directional_segment_traversal":
            continue
        if not bool(rule.get("blocking", True)):
            continue
        for segment_id, directions in (rule.get("segment_direction_overrides") or {}).items():
            allowed = sorted({str(direction) for direction in directions or []})
            if allowed and allowed != ["forward", "reverse"]:
                overrides[int(segment_id)] = allowed
    return overrides


def load_connector_graph(
    path: Path | None,
    official_segments: list[dict[str, Any]] | None = None,
    special_management_rules_json: Path | None = DEFAULT_SPECIAL_MANAGEMENT_RULES_JSON,
) -> dict[str, Any] | None:
    if not path or not path.exists():
        return None
    try:
        data = read_json(path)
    except json.JSONDecodeError:
        return None
    graph: dict[tuple[float, float], list[dict[str, Any]]] = defaultdict(list)
    nodes: list[tuple[float, float]] = []
    connector_class_counts: Counter[str] = Counter()
    skipped_access_reasons: Counter[str] = Counter()
    skipped_connector_names: Counter[str] = Counter()
    skipped_connector_feature_count = 0
    special_management_overrides = load_special_management_direction_overrides(
        special_management_rules_json
    )
    official_overrides = official_edge_override_index(
        official_segments,
        special_management_direction_overrides=special_management_overrides,
    )
    official_named_overlaps = official_named_overlap_index(official_segments)
    for feature in data.get("features", []):
        props = feature.get("properties") or {}
        name = (
            props.get("TrailName")
            or props.get("Name")
            or props.get("SystemName")
            or f"connector_{props.get('OBJECTID')}"
        )
        unsafe_reasons = unsafe_connector_access_reasons(props)
        if unsafe_reasons:
            skipped_connector_feature_count += 1
            skipped_connector_names[str(name)] += 1
            for reason in unsafe_reasons:
                skipped_access_reasons[reason] += 1
            continue
        connector_class = connector_class_for_properties(props, "connector")
        for coords in iter_line_parts(feature.get("geometry") or {}):
            if len(coords) < 2:
                continue
            distance = line_length_miles(coords)
            if distance <= 0:
                continue
            add_polyline_to_graph(
                graph,
                nodes,
                coords,
                name=str(name),
                edge_type="connector",
                bidirectional=True,
                connector_class=connector_class,
                source=props.get("source"),
                highway=props.get("highway"),
                access_properties=connector_access_properties(props),
                official_edge_overrides=official_overrides,
                official_named_overlaps=official_named_overlaps,
                special_management_direction_overrides=special_management_overrides,
                special_management_allowed_directions=[],
            )
            connector_class_counts[connector_class] += 1
    for segment in official_segments or []:
        bidirectional = segment["direction"] == "both"
        geometry_miles = line_length_miles(segment["coordinates"])
        distance_scale = segment["official_miles"] / geometry_miles if geometry_miles else 1.0
        add_polyline_to_graph(
            graph,
            nodes,
            segment["coordinates"],
            name=segment["trail_name"],
            edge_type="official_repeat",
            bidirectional=bidirectional,
            seg_id=segment["seg_id"],
            distance_scale=distance_scale,
            connector_class="official_repeat",
            source="official_challenge",
            access_properties={
                "access": None,
                "foot": None,
                "highway": None,
                "source": "official_challenge",
            },
            special_management_allowed_directions=special_management_overrides.get(
                int(segment["seg_id"])
            ),
        )
        connector_class_counts["official_repeat"] += 1
    return {
        "path": str(path),
        "graph": dict(graph),
        "nodes": sorted(set(nodes)),
        "feature_count": len(data.get("features", [])),
        "official_segment_count": len(official_segments or []),
        "connector_class_counts": dict(sorted(connector_class_counts.items())),
        "skipped_connector_feature_count": skipped_connector_feature_count,
        "skipped_connector_access_reasons": dict(sorted(skipped_access_reasons.items())),
        "skipped_connector_names": dict(sorted(skipped_connector_names.items())),
    }


def round_node(point: tuple[float, float]) -> tuple[float, float]:
    return (round(point[0], 4), round(point[1], 4))


def nearest_connector_node(
    point: tuple[float, float], nodes: list[tuple[float, float]], tolerance_miles: float
) -> tuple[tuple[float, float], float] | None:
    best: tuple[tuple[float, float], float] | None = None
    for node in nodes:
        distance = haversine_miles(point, node)
        if distance <= tolerance_miles and (best is None or distance < best[1]):
            best = (node, distance)
    return best


def projected_node(point: tuple[float, float], reference_lat: float) -> tuple[float, float]:
    lon, lat = point
    return (
        lon * math.cos(math.radians(reference_lat)) * 69.0,
        lat * 69.0,
    )


def connector_node_index(connector_graph: dict[str, Any]) -> dict[str, Any] | None:
    if "_node_index" in connector_graph:
        return connector_graph["_node_index"]
    nodes = connector_graph.get("nodes") or []
    if not nodes:
        connector_graph["_node_index"] = None
        return None
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        connector_graph["_node_index"] = None
        return None
    reference_lat = statistics.median(point[1] for point in nodes)
    projected = [projected_node(point, reference_lat) for point in nodes]
    connector_graph["_node_index"] = {
        "tree": cKDTree(projected),
        "nodes": nodes,
        "reference_lat": reference_lat,
    }
    return connector_graph["_node_index"]


def nearest_connector_node_for_graph(
    point: tuple[float, float],
    connector_graph: dict[str, Any],
    tolerance_miles: float,
) -> tuple[tuple[float, float], float] | None:
    cache_key = (round_node(point), round(tolerance_miles, 3))
    cache = connector_graph.setdefault("_nearest_node_cache", {})
    if cache_key in cache:
        return cache[cache_key]

    index = connector_node_index(connector_graph)
    if not index:
        result = nearest_connector_node(point, connector_graph.get("nodes") or [], tolerance_miles)
        cache[cache_key] = result
        return result

    nodes = index["nodes"]
    query_k = min(16, len(nodes))
    distances, indexes = index["tree"].query(
        projected_node(point, float(index["reference_lat"])),
        k=query_k,
    )
    if query_k == 1:
        indexes = [int(indexes)]
    best: tuple[tuple[float, float], float] | None = None
    for node_index in indexes:
        if int(node_index) >= len(nodes):
            continue
        node = nodes[int(node_index)]
        distance = haversine_miles(point, node)
        if distance <= tolerance_miles and (best is None or distance < best[1]):
            best = (node, distance)
    cache[cache_key] = best
    return best


def shortest_connector_path(
    start: tuple[float, float],
    end: tuple[float, float],
    connector_graph: dict[str, Any] | None,
    snap_tolerance_miles: float,
    avoid_official_segment_ids: set[int] | None = None,
    max_distance_miles: float | None = None,
) -> dict[str, Any] | None:
    if not connector_graph:
        return None
    avoided_ids = frozenset(int(seg_id) for seg_id in avoid_official_segment_ids or set())
    cache_key = (
        round_node(start),
        round_node(end),
        round(snap_tolerance_miles, 3),
        tuple(sorted(avoided_ids)),
        round(float(max_distance_miles), 3) if max_distance_miles is not None else None,
    )
    cache = connector_graph.setdefault("_shortest_path_cache", {})
    if cache_key in cache:
        return cache[cache_key]
    start_node = nearest_connector_node_for_graph(start, connector_graph, snap_tolerance_miles)
    end_node = nearest_connector_node_for_graph(end, connector_graph, snap_tolerance_miles)
    if not start_node or not end_node:
        cache[cache_key] = None
        return None

    graph = connector_graph["graph"]
    queue: list[tuple[float, int, tuple[float, float]]] = [(0.0, 0, start_node[0])]
    push_count = 1
    distances: dict[tuple[float, float], float] = {start_node[0]: 0.0}
    previous: dict[tuple[float, float], tuple[tuple[float, float], dict[str, Any]]] = {}
    while queue:
        distance, _, node = heapq.heappop(queue)
        if distance > distances.get(node, math.inf):
            continue
        if max_distance_miles is not None and distance > max_distance_miles:
            continue
        if node == end_node[0]:
            path_nodes = [node]
            path_edges: list[dict[str, Any]] = []
            while path_nodes[-1] in previous:
                prev_node, edge = previous[path_nodes[-1]]
                path_edges.append(edge)
                path_nodes.append(prev_node)
            path_nodes.reverse()
            path_edges.reverse()

            connector_names = []
            official_repeat_segment_ids = []
            connector_miles = start_node[1] + end_node[1]
            official_repeat_miles = 0.0
            connector_edges = []
            connector_classes = []
            path_coordinates: list[list[float]] = []
            for from_node, edge in zip(path_nodes, path_edges):
                edge_class = str(edge.get("connector_class") or "unknown_connector")
                edge_coordinates = edge.get("coordinates") or [
                    [from_node[0], from_node[1]],
                    [edge["to"][0], edge["to"][1]],
                ]
                for coord in edge_coordinates:
                    point = [float(coord[0]), float(coord[1])]
                    if path_coordinates and haversine_miles(
                        (path_coordinates[-1][0], path_coordinates[-1][1]),
                        (point[0], point[1]),
                    ) <= 0.000001:
                        continue
                    path_coordinates.append(point)
                connector_classes.append(edge_class)
                connector_edges.append(
                    {
                        "from": [from_node[0], from_node[1]],
                        "to": [edge["to"][0], edge["to"][1]],
                        "distance_miles": round_miles(edge["distance"]),
                        "name": edge.get("name"),
                        "edge_type": edge.get("edge_type"),
                        "connector_class": edge_class,
                        "source": edge.get("source"),
                        "highway": edge.get("highway"),
                        "access_properties": edge.get("access_properties") or {},
                        "seg_id": edge.get("seg_id"),
                        "official_traversal_direction": edge.get("official_traversal_direction"),
                        "special_management_allowed_directions": edge.get(
                            "special_management_allowed_directions"
                        )
                        or [],
                        "coordinates": edge_coordinates,
                    }
                )
                if edge["edge_type"] == "official_repeat":
                    official_repeat_miles += edge["distance"]
                    if edge.get("seg_id") is not None:
                        official_repeat_segment_ids.append(edge["seg_id"])
                else:
                    connector_miles += edge["distance"]
                    connector_names.append(edge["name"])
            result = {
                "distance_miles": distance + start_node[1] + end_node[1],
                "connector_miles": round_miles(connector_miles),
                "official_repeat_miles": round_miles(official_repeat_miles),
                "connector_names": sorted(set(connector_names))[:12],
                "connector_classes": sorted(set(connector_classes)),
                "connector_edges": connector_edges,
                "official_repeat_segment_ids": sorted(set(official_repeat_segment_ids)),
                "path_coordinates": path_coordinates or [[point[0], point[1]] for point in path_nodes],
                "snap_start_miles": start_node[1],
                "snap_end_miles": end_node[1],
            }
            cache[cache_key] = result
            return result
        for edge in graph.get(node, []):
            if edge.get("edge_type") == "official_repeat" and edge.get("seg_id") in avoided_ids:
                continue
            allowed_directions = edge.get("special_management_allowed_directions") or []
            traversal_direction = edge.get("official_traversal_direction")
            if (
                edge.get("edge_type") == "official_repeat"
                and allowed_directions
                and traversal_direction not in allowed_directions
            ):
                continue
            next_distance = distance + edge["distance"]
            if max_distance_miles is not None and next_distance > max_distance_miles:
                continue
            next_node = edge["to"]
            if next_distance >= distances.get(next_node, math.inf):
                continue
            distances[next_node] = next_distance
            previous[next_node] = (node, edge)
            heapq.heappush(queue, (next_distance, push_count, next_node))
            push_count += 1
    cache[cache_key] = None
    return None


def choose_trailhead(
    point: tuple[float, float],
    state: dict[str, Any],
) -> dict[str, Any]:
    trailheads = state.get("trailheads") or []
    if not trailheads:
        drive = get_drive_model(state)
        return {
            "name": "Planning origin",
            "lat": drive["origin_lat"],
            "lon": drive["origin_lon"],
            "parking_minutes": state.get("parking_minutes", 8),
        }
    best = min(
        trailheads,
        key=lambda trailhead: haversine_miles(
            point, (float(trailhead["lon"]), float(trailhead["lat"]))
        ),
    )
    return best


def serialize_trailhead(trailhead: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "name",
        "lat",
        "lon",
        "facility_id",
        "facility_status",
        "has_parking",
        "has_restroom",
        "has_water",
        "parking_confidence",
        "address",
        "source",
        "parking_minutes",
    ]
    return {field: trailhead.get(field) for field in fields if field in trailhead}


def reverse_segment_orientation(segment: dict[str, Any]) -> dict[str, Any]:
    reversed_segment = dict(segment)
    reversed_segment["start"] = segment["end"]
    reversed_segment["end"] = segment["start"]
    reversed_segment["coordinates"] = list(reversed(segment["coordinates"]))
    return reversed_segment


def segment_orientation_options(segment: dict[str, Any]) -> list[dict[str, Any]]:
    if segment.get("direction") == "both":
        return [dict(segment), reverse_segment_orientation(segment)]
    return [dict(segment)]


def orient_segments_for_continuity(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(segments) < 2:
        return [dict(segment) for segment in segments]

    options_by_segment = [segment_orientation_options(segment) for segment in segments]
    states: dict[int, tuple[float, list[dict[str, Any]]]] = {
        index: (0.0, [option])
        for index, option in enumerate(options_by_segment[0])
    }
    for options in options_by_segment[1:]:
        next_states: dict[int, tuple[float, list[dict[str, Any]]]] = {}
        for _, (cost, path) in states.items():
            previous = path[-1]
            for index, option in enumerate(options):
                gap = haversine_miles(previous["end"], option["start"])
                next_cost = cost + gap
                current = next_states.get(index)
                if current is None or next_cost < current[0]:
                    next_states[index] = (next_cost, path + [option])
        states = next_states

    _, best_path = min(states.values(), key=lambda item: item[0])
    return best_path


def reverse_trail_orientation(trail: dict[str, Any]) -> dict[str, Any]:
    reversed_trail = dict(trail)
    reversed_segments = []
    for segment in reversed(trail["segments"]):
        reversed_segments.append(reverse_segment_orientation(segment))
    reversed_trail["segments"] = reversed_segments
    reversed_trail["remaining_segment_ids"] = list(reversed(trail["remaining_segment_ids"]))
    reversed_trail["start"] = trail["end"]
    reversed_trail["end"] = trail["start"]
    return reversed_trail


def trails_are_bidirectional(trails: list[dict[str, Any]]) -> bool:
    return all(
        segment["direction"] == "both"
        for trail in trails
        for segment in trail["segments"]
    )


def snap_confidence_score(confidence: str) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(confidence, 0)


def orient_trails_for_access(
    trails: list[dict[str, Any]],
    state: dict[str, Any],
    connector_graph: dict[str, Any] | None,
    outing_model: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    orientation_options = [
        ("forward", trails),
    ]
    if trails_are_bidirectional(trails):
        orientation_options.append(
            ("reversed", [reverse_trail_orientation(trail) for trail in reversed(trails)])
        )

    scored_options = []
    for direction, option_trails in orientation_options:
        option = combine_trails(option_trails)
        avoid_segment_ids = {int(seg_id) for seg_id in option.get("remaining_segment_ids") or []}
        trailhead = choose_trailhead(option["start"], state)
        snap = trailhead_snap_confidence(
            trailhead,
            option["start"],
            connector_graph,
            outing_model,
            avoid_official_segment_ids=avoid_segment_ids,
        )
        scored_options.append(
            {
                "direction": direction,
                "trails": option_trails,
                "trailhead": trailhead,
                "snap": snap,
                "score": (
                    snap_confidence_score(snap["confidence"]),
                    -float(snap["mapped_access_miles"] if snap["mapped_access_miles"] is not None else snap["direct_gap_miles"]),
                    1 if direction == "forward" else 0,
                ),
            }
        )

    best = max(scored_options, key=lambda item: item["score"])
    return best["trails"], {
        "direction": best["direction"],
        "start_trailhead": best["trailhead"]["name"],
        "trailhead_snap_confidence": best["snap"]["confidence"],
        "direct_gap_miles": best["snap"]["direct_gap_miles"],
        "mapped_access_miles": best["snap"]["mapped_access_miles"],
        "selection_reason": "best_trailhead_access_endpoint",
    }


def drive_minutes_to_trailhead(trailhead: dict[str, Any], drive_model: dict[str, Any]) -> int:
    origin = (float(drive_model["origin_lon"]), float(drive_model["origin_lat"]))
    destination = (float(trailhead["lon"]), float(trailhead["lat"]))
    drive_miles = haversine_miles(origin, destination) * float(
        drive_model["straight_line_factor"]
    )
    minutes = drive_miles * float(drive_model["minutes_per_mile"])
    return max(ceil_minutes(minutes), int(drive_model["minimum_one_way_minutes"]))


def group_remaining_by_trail(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for segment in segments:
        grouped[segment["trail_name"]].append(segment)

    trails = []
    for trail_name, trail_segments in grouped.items():
        ordered = orient_segments_for_continuity(
            sorted(trail_segments, key=lambda segment: natural_key(segment["seg_name"]))
        )
        coords = [point for segment in ordered for point in segment["coordinates"]]
        direction_counts = Counter(segment["direction"] for segment in ordered)
        trails.append(
            {
                "trail_name": trail_name,
                "segments": ordered,
                "remaining_segment_ids": [segment["seg_id"] for segment in ordered],
                "official_miles": sum(segment["official_miles"] for segment in ordered),
                "direction_counts": dict(sorted(direction_counts.items())),
                "start": ordered[0]["start"],
                "end": ordered[-1]["end"],
                "center": center_point(coords),
            }
        )
    return sorted(trails, key=lambda item: (-item["official_miles"], item["trail_name"]))


def natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def classify_bucket(total_minutes: int) -> str:
    for name, low, high in TIME_BUCKETS:
        if total_minutes >= low and (high is None or total_minutes <= high):
            return name
    return "four_plus_hours"


def build_return_to_car(
    trail: dict[str, Any],
    outing_model: dict[str, Any],
    connector_graph: dict[str, Any] | None,
    avoid_official_segment_ids: set[int] | None = None,
) -> dict[str, Any]:
    official_miles = trail["official_miles"]
    endpoint_gap = haversine_miles(trail["end"], trail["start"])
    closed_loop_tolerance = float(outing_model.get("closed_loop_gap_tolerance_miles", 0.05))
    if endpoint_gap <= closed_loop_tolerance:
        return {
            "strategy": "closed_loop",
            "description": "Route endpoint is already at the start trailhead within geometry tolerance.",
            "official_repeat_miles": 0,
            "connector_miles": 0,
            "road_miles": 0,
            "endpoint_gap_miles": round_miles(endpoint_gap),
            "connector_names": [],
            "official_repeat_segment_ids": [],
            "path_coordinates": [[trail["end"][0], trail["end"][1]], [trail["start"][0], trail["start"][1]]],
            "needs_map_validation": False,
            "graph_validated": True,
        }
    snap_tolerance = float(outing_model.get("mapped_connector_snap_tolerance_miles", 0.02))
    mapped = shortest_connector_path(
        trail["end"],
        trail["start"],
        connector_graph,
        snap_tolerance,
        avoid_official_segment_ids=avoid_official_segment_ids,
    )
    if mapped and mapped["distance_miles"] <= max(official_miles * 1.5, endpoint_gap * 1.25):
        if mapped["official_repeat_miles"] and mapped["connector_miles"]:
            strategy = "mapped_mixed_loop"
        elif mapped["official_repeat_miles"]:
            strategy = "mapped_official_repeat_return"
        else:
            strategy = "mapped_connector_loop"
        return {
            "strategy": strategy,
            "description": "Return by graph-routed trail/path network back to the start trailhead.",
            "official_repeat_miles": round_miles(mapped["official_repeat_miles"]),
            "connector_miles": round_miles(mapped["connector_miles"]),
            "road_miles": round_miles(float(outing_model.get("road_miles_per_outing") or 0)),
            "endpoint_gap_miles": round_miles(endpoint_gap),
            "connector_names": mapped["connector_names"],
            "connector_classes": mapped.get("connector_classes", []),
            "connector_edges": mapped.get("connector_edges", []),
            "official_repeat_segment_ids": mapped["official_repeat_segment_ids"],
            "path_coordinates": mapped["path_coordinates"],
            "needs_map_validation": False,
            "graph_validated": True,
            "snap_start_miles": round_miles(mapped["snap_start_miles"]),
            "snap_end_miles": round_miles(mapped["snap_end_miles"]),
        }

    if bool(outing_model.get("allow_official_out_and_back_fallback", True)):
        return {
            "strategy": "out_and_back",
            "description": "Return to the car by reversing the completed official segment set.",
            "official_repeat_miles": round_miles(official_miles),
            "connector_miles": 0,
            "road_miles": 0,
            "endpoint_gap_miles": round_miles(endpoint_gap),
            "connector_names": [],
            "official_repeat_segment_ids": trail["remaining_segment_ids"],
            "path_coordinates": out_and_back_return_coordinates(trail),
            "needs_map_validation": False,
            "graph_validated": True,
        }

    connector_miles = max(
        endpoint_gap * float(outing_model["connector_return_factor"]),
        float(outing_model.get("minimum_connector_miles") or 0),
    )
    if bool(outing_model.get("prefer_connector_if_shorter_than_repeat", True)) and connector_miles <= official_miles:
        return {
            "strategy": "connector_or_road_loop",
            "description": "Estimated non-official connector, R2R trail, road, or path back to the start trailhead.",
            "official_repeat_miles": 0,
            "connector_miles": round_miles(connector_miles),
            "road_miles": round_miles(float(outing_model.get("road_miles_per_outing") or 0)),
            "endpoint_gap_miles": round_miles(endpoint_gap),
            "connector_names": [],
            "official_repeat_segment_ids": [],
            "needs_map_validation": True,
            "graph_validated": False,
        }

    return {
        "strategy": "out_and_back",
        "description": "Return to the car by reversing the completed official segment set.",
        "official_repeat_miles": round_miles(official_miles),
        "connector_miles": 0,
        "road_miles": 0,
        "endpoint_gap_miles": round_miles(endpoint_gap),
        "connector_names": [],
        "official_repeat_segment_ids": trail["remaining_segment_ids"],
        "needs_map_validation": False,
        "graph_validated": True,
    }


def out_and_back_return_coordinates(trail: dict[str, Any]) -> list[list[float]]:
    coords: list[list[float]] = []
    for segment in reversed(trail.get("segments") or []):
        for lon, lat in reversed(segment.get("coordinates") or []):
            point = [float(lon), float(lat)]
            if coords and coords[-1] == point:
                continue
            coords.append(point)
    if coords:
        return coords
    return [
        [float(trail["end"][0]), float(trail["end"][1])],
        [float(trail["start"][0]), float(trail["start"][1])],
    ]


def combine_trails(trails: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = list(trails)
    segments = [segment for trail in ordered for segment in trail["segments"]]
    coords = [point for segment in segments for point in segment["coordinates"]]
    return {
        "trail_names": [trail["trail_name"] for trail in ordered],
        "segments": segments,
        "remaining_segment_ids": [segment["seg_id"] for segment in segments],
        "official_miles": sum(trail["official_miles"] for trail in ordered),
        "direction_counts": dict(sorted(Counter(segment["direction"] for segment in segments).items())),
        "start": ordered[0]["start"],
        "end": ordered[-1]["end"],
        "center": center_point(coords),
    }


def build_between_trail_links(
    trails: list[dict[str, Any]],
    outing_model: dict[str, Any],
    connector_graph: dict[str, Any] | None,
    avoid_official_segment_ids: set[int] | None = None,
) -> dict[str, Any]:
    snap_tolerance = float(outing_model.get("mapped_connector_snap_tolerance_miles", 0.02))
    connector_factor = float(outing_model.get("connector_return_factor") or 1.0)
    required_ids = {int(seg_id) for seg_id in avoid_official_segment_ids or set()}
    earned_ids: set[int] = set()
    links = []
    connector_miles = 0.0
    official_repeat_miles = 0.0
    estimated_miles = 0.0
    all_graph_validated = True
    for left, right in zip(trails, trails[1:]):
        earned_ids.update(
            int(seg_id)
            for seg_id in left.get("remaining_segment_ids") or []
            if str(seg_id).isdigit()
        )
        avoid_ids = required_ids - earned_ids if avoid_official_segment_ids is not None else None
        endpoint_gap = haversine_miles(left["end"], right["start"])
        mapped = shortest_connector_path(
            left["end"],
            right["start"],
            connector_graph,
            snap_tolerance,
            avoid_official_segment_ids=avoid_ids,
        )
        if mapped:
            link_connector = float(mapped["connector_miles"])
            link_repeat = float(mapped["official_repeat_miles"])
            connector_miles += link_connector
            official_repeat_miles += link_repeat
            links.append(
                {
                    "from_trail": left["trail_name"],
                    "to_trail": right["trail_name"],
                    "source": "mapped_graph",
                    "distance_miles": round_miles(mapped["distance_miles"]),
                    "connector_miles": round_miles(link_connector),
                    "official_repeat_miles": round_miles(link_repeat),
                    "connector_names": mapped["connector_names"],
                    "connector_classes": mapped.get("connector_classes", []),
                    "connector_edges": mapped.get("connector_edges", []),
                    "official_repeat_segment_ids": mapped["official_repeat_segment_ids"],
                    "earned_segment_ids_before_link": sorted(earned_ids),
                    "avoided_unearned_segment_ids": sorted(avoid_ids or []),
                    "path_coordinates": mapped["path_coordinates"],
                    "endpoint_gap_miles": round_miles(endpoint_gap),
                }
            )
        else:
            estimate = endpoint_gap * connector_factor
            estimated_miles += estimate
            all_graph_validated = False
            links.append(
                {
                    "from_trail": left["trail_name"],
                    "to_trail": right["trail_name"],
                    "source": "estimated_gap",
                    "distance_miles": round_miles(estimate),
                    "connector_miles": round_miles(estimate),
                    "official_repeat_miles": 0,
                    "connector_names": [],
                    "connector_classes": [],
                    "connector_edges": [],
                    "official_repeat_segment_ids": [],
                    "earned_segment_ids_before_link": sorted(earned_ids),
                    "avoided_unearned_segment_ids": sorted(avoid_ids or []),
                    "endpoint_gap_miles": round_miles(endpoint_gap),
                }
            )
    return {
        "links": links,
        "connector_miles": round_miles(connector_miles + estimated_miles),
        "official_repeat_miles": round_miles(official_repeat_miles),
        "estimated_connector_miles": round_miles(estimated_miles),
        "all_graph_validated": all_graph_validated,
    }


def allow_small_target_snap_on_path(
    mapped: dict[str, Any] | None,
    target_segment_id: int | None,
    tolerance_miles: float = TARGET_SNAP_OFFICIAL_TOLERANCE_MILES,
) -> dict[str, Any] | None:
    if not mapped or target_segment_id is None:
        return None
    repeat_ids = {
        int(seg_id)
        for seg_id in (mapped.get("official_repeat_segment_ids") or [])
        if str(seg_id).isdigit()
    }
    repeat_miles = float(mapped.get("official_repeat_miles") or 0.0)
    if not repeat_ids or repeat_ids - {target_segment_id} or repeat_miles > tolerance_miles:
        return None

    adjusted = dict(mapped)
    adjusted["source"] = "mapped_graph_target_snap"
    adjusted["target_snap_segment_id"] = target_segment_id
    adjusted["target_snap_official_miles"] = round_miles(repeat_miles)
    adjusted["connector_miles"] = round_miles(
        float(mapped.get("connector_miles") or 0.0) + repeat_miles
    )
    adjusted["official_repeat_miles"] = 0.0
    adjusted["official_repeat_segment_ids"] = []
    adjusted["connector_classes"] = sorted(
        set(mapped.get("connector_classes") or []) | {"target_official_snap"}
    )
    return adjusted


def build_inter_segment_links(
    segments: list[dict[str, Any]],
    outing_model: dict[str, Any],
    connector_graph: dict[str, Any] | None,
    avoid_official_segment_ids: set[int] | None = None,
) -> dict[str, Any]:
    snap_tolerance = float(outing_model.get("mapped_connector_snap_tolerance_miles", 0.02))
    connector_factor = float(outing_model.get("connector_return_factor") or 1.0)
    gap_tolerance = float(outing_model.get("closed_loop_gap_tolerance_miles", 0.05))
    required_ids = {int(seg_id) for seg_id in avoid_official_segment_ids or set()}
    earned_ids: set[int] = set()
    links = []
    connector_miles = 0.0
    official_repeat_miles = 0.0
    estimated_miles = 0.0
    all_graph_validated = True

    for index, (left, right) in enumerate(zip(segments, segments[1:]), start=1):
        if str(left.get("seg_id") or "").isdigit():
            earned_ids.add(int(left["seg_id"]))
        avoid_ids = required_ids - earned_ids if avoid_official_segment_ids is not None else None
        endpoint_gap = haversine_miles(left["end"], right["start"])
        if endpoint_gap <= gap_tolerance:
            continue
        mapped = shortest_connector_path(
            left["end"],
            right["start"],
            connector_graph,
            snap_tolerance,
            avoid_official_segment_ids=avoid_ids,
        )
        target_segment_id = (
            int(right["seg_id"]) if str(right.get("seg_id") or "").isdigit() else None
        )
        if mapped is None and avoid_ids and target_segment_id in avoid_ids:
            relaxed = shortest_connector_path(
                left["end"],
                right["start"],
                connector_graph,
                snap_tolerance,
                avoid_official_segment_ids=set(avoid_ids) - {target_segment_id},
            )
            mapped = allow_small_target_snap_on_path(relaxed, target_segment_id)
        base = {
            "from_segment_id": left.get("seg_id"),
            "to_segment_id": right.get("seg_id"),
            "from_trail": left.get("trail_name"),
            "to_trail": right.get("trail_name"),
            "transition_index": index,
            "earned_segment_ids_before_link": sorted(earned_ids),
            "avoided_unearned_segment_ids": sorted(avoid_ids or []),
            "endpoint_gap_miles": round_miles(endpoint_gap),
        }
        if mapped:
            link_connector = float(mapped["connector_miles"])
            link_repeat = float(mapped["official_repeat_miles"])
            connector_miles += link_connector
            official_repeat_miles += link_repeat
            links.append(
                {
                    **base,
                    "source": mapped.get("source") or "mapped_graph",
                    "distance_miles": round_miles(mapped["distance_miles"]),
                    "connector_miles": round_miles(link_connector),
                    "official_repeat_miles": round_miles(link_repeat),
                    "target_snap_segment_id": mapped.get("target_snap_segment_id"),
                    "target_snap_official_miles": mapped.get("target_snap_official_miles"),
                    "connector_names": mapped["connector_names"],
                    "connector_classes": mapped.get("connector_classes", []),
                    "connector_edges": mapped.get("connector_edges", []),
                    "official_repeat_segment_ids": mapped["official_repeat_segment_ids"],
                    "path_coordinates": mapped["path_coordinates"],
                }
            )
        else:
            estimate = endpoint_gap * connector_factor
            estimated_miles += estimate
            all_graph_validated = False
            links.append(
                {
                    **base,
                    "source": "estimated_gap",
                    "distance_miles": round_miles(estimate),
                    "connector_miles": round_miles(estimate),
                    "official_repeat_miles": 0,
                    "connector_names": [],
                    "connector_classes": [],
                    "connector_edges": [],
                    "official_repeat_segment_ids": [],
                    "path_coordinates": [
                        [float(left["end"][0]), float(left["end"][1])],
                        [float(right["start"][0]), float(right["start"][1])],
                    ],
                }
            )

    return {
        "links": links,
        "connector_miles": round_miles(connector_miles + estimated_miles),
        "official_repeat_miles": round_miles(official_repeat_miles),
        "estimated_connector_miles": round_miles(estimated_miles),
        "all_graph_validated": all_graph_validated,
    }


def summarize_between_trail_links(inter_segment_links: dict[str, Any]) -> dict[str, Any]:
    links = [
        link
        for link in inter_segment_links.get("links") or []
        if str(link.get("from_trail") or "") != str(link.get("to_trail") or "")
    ]
    return {
        "links": links,
        "connector_miles": round_miles(sum(float(link.get("connector_miles") or 0) for link in links)),
        "official_repeat_miles": round_miles(sum(float(link.get("official_repeat_miles") or 0) for link in links)),
        "estimated_connector_miles": round_miles(
            sum(float(link.get("distance_miles") or 0) for link in links if link.get("source") == "estimated_gap")
        ),
        "all_graph_validated": all(link.get("source") in GRAPH_VALIDATED_LINK_SOURCES for link in links),
    }


def elevation_gain_loss_for_line(
    coords: list[tuple[float, float]],
    elevation_sampler: ElevationSampler,
) -> tuple[float, float, bool]:
    sampled = []
    for point in sample_line(coords, spacing_miles=0.05):
        elevation = elevation_sampler(point)
        if elevation is not None:
            sampled.append(elevation)
    if len(sampled) < 2:
        return 0.0, 0.0, False

    ascent = 0.0
    descent = 0.0
    for left, right in zip(sampled, sampled[1:]):
        delta = right - left
        if delta > 0:
            ascent += delta
        elif delta < 0:
            descent += abs(delta)
    return ascent, descent, True


def build_elevation_effort(
    segments: list[dict[str, Any]],
    elevation_sampler: ElevationSampler | None,
    official_miles: float,
    estimated_moving_minutes: int,
) -> dict[str, Any]:
    base = {
        "ascent_ft": None,
        "descent_ft": None,
        "grade_adjusted_miles": None,
        "estimated_moving_minutes_p50": estimated_moving_minutes,
        "estimated_moving_minutes_p75": ceil_minutes(estimated_moving_minutes * 1.12),
        "heat_risk": "unknown",
        "effort_score": estimated_moving_minutes,
        "elevation_source": "unavailable",
    }
    if not elevation_sampler:
        return base

    ascent = 0.0
    descent = 0.0
    sampled_line_count = 0
    for segment in segments:
        segment_ascent, segment_descent, sampled = elevation_gain_loss_for_line(
            segment["coordinates"],
            elevation_sampler,
        )
        if not sampled:
            continue
        sampled_line_count += 1
        ascent += segment_ascent
        descent += segment_descent
    if not sampled_line_count:
        return base | {"elevation_source": "dem_no_valid_samples"}

    climb_penalty_minutes = ascent / 100
    effort_score = estimated_moving_minutes + climb_penalty_minutes
    return {
        "ascent_ft": round(ascent),
        "descent_ft": round(descent),
        "grade_adjusted_miles": round_miles(official_miles + ascent / 1000),
        "estimated_moving_minutes_p50": estimated_moving_minutes,
        "estimated_moving_minutes_p75": ceil_minutes(effort_score * 1.12),
        "heat_risk": "unknown",
        "effort_score": ceil_minutes(effort_score),
        "elevation_source": "dem",
    }


def enrich_segment_estimates_with_elevation(
    segment_estimates: list[dict[str, Any]],
    source_segments: list[dict[str, Any]],
    elevation_sampler: ElevationSampler | None,
) -> list[dict[str, Any]]:
    enriched = [dict(segment) for segment in segment_estimates]
    if not elevation_sampler:
        for segment in enriched:
            segment.setdefault("ascent_ft", None)
            segment.setdefault("descent_ft", None)
            segment.setdefault("grade_adjusted_miles", None)
            segment.setdefault("estimated_moving_minutes_p75", ceil_minutes(segment["estimated_moving_minutes"] * 1.12))
            segment.setdefault("elevation_source", "unavailable")
        return enriched

    for estimate, source in zip(enriched, source_segments):
        ascent, descent, sampled = elevation_gain_loss_for_line(
            source["coordinates"],
            elevation_sampler,
        )
        if not sampled:
            estimate["ascent_ft"] = None
            estimate["descent_ft"] = None
            estimate["grade_adjusted_miles"] = None
            estimate["estimated_moving_minutes_p75"] = ceil_minutes(
                estimate["estimated_moving_minutes"] * 1.12
            )
            estimate["elevation_source"] = "dem_no_valid_samples"
            continue
        effort_minutes = estimate["estimated_moving_minutes"] + ascent / 100
        estimate["ascent_ft"] = round(ascent)
        estimate["descent_ft"] = round(descent)
        estimate["grade_adjusted_miles"] = round_miles(
            float(estimate.get("official_miles") or 0.0) + ascent / 1000
        )
        estimate["estimated_moving_minutes_p75"] = ceil_minutes(effort_minutes * 1.12)
        estimate["elevation_source"] = "dem"
    return enriched


def build_route_finding_penalty_minutes(
    *,
    trail_names: list[str],
    between_links: dict[str, Any],
    trailhead_snap_confidence: str,
    official_repeat_miles: float,
    connector_miles: float,
    road_miles: float,
) -> int:
    penalty = 0
    if trailhead_snap_confidence == "medium":
        penalty += 4
    elif trailhead_snap_confidence == "low":
        penalty += 10

    link_count = len(between_links.get("links") or [])
    if link_count > 1:
        penalty += min(8, (link_count - 1) * 2)

    non_credit_miles = connector_miles + road_miles
    if len(trail_names) > 1:
        non_credit_miles += official_repeat_miles
    if non_credit_miles >= 1.5:
        penalty += 8
    elif non_credit_miles >= 0.5:
        penalty += 4

    if len(trail_names) >= 5:
        penalty += 4
    elif len(trail_names) >= 3:
        penalty += 2

    return min(penalty, 25)


def build_time_estimates_minutes(
    *,
    drive_to: int,
    parking_minutes: int,
    raw_moving_minutes: int,
    effort: dict[str, Any],
    route_finding_penalty_minutes: int,
) -> dict[str, int]:
    raw_total = drive_to + parking_minutes + raw_moving_minutes + drive_to
    effort_moving = int(effort.get("effort_score") or raw_moving_minutes)
    p75_moving = int(effort.get("estimated_moving_minutes_p75") or ceil_minutes(effort_moving * 1.12))
    p50_total = drive_to + parking_minutes + effort_moving + drive_to
    p75_total = drive_to + parking_minutes + p75_moving + route_finding_penalty_minutes + drive_to
    p90_total = ceil_minutes(p75_total * 1.12)
    return {
        "door_to_door_raw": raw_total,
        "door_to_door_p50": p50_total,
        "door_to_door_p75": p75_total,
        "door_to_door_p90": p90_total,
        "recommended_door_to_door": p75_total,
        "moving_raw": raw_moving_minutes,
        "moving_effort_p50": effort_moving,
        "moving_effort_p75": p75_moving,
        "route_finding_penalty": route_finding_penalty_minutes,
    }


def build_direction_validation(
    segments: list[dict[str, Any]],
    elevation_sampler: ElevationSampler | None = None,
) -> dict[str, Any]:
    ascent_segments = [segment for segment in segments if segment["direction"] == "ascent"]
    if not ascent_segments:
        return {
            "passed": True,
            "reason": "no_ascent_only_segments",
            "ascent_segment_ids_checked": [],
            "ascent_segment_checks": [],
            "planned_traversal_direction": {},
            "note": "No ascent-only official segments in this candidate.",
        }
    if not elevation_sampler:
        return {
            "passed": False,
            "reason": "ascent_segments_need_elevation_validation",
            "ascent_segment_ids_checked": [segment["seg_id"] for segment in ascent_segments],
            "ascent_segment_checks": [
                {
                    "seg_id": segment["seg_id"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "start_elevation_ft": None,
                    "end_elevation_ft": None,
                    "elevation_delta_ft": None,
                    "passed": False,
                }
                for segment in ascent_segments
            ],
            "planned_traversal_direction": {
                str(segment["seg_id"]): "official_geometry_start_to_end"
                for segment in ascent_segments
            },
            "note": "Ascent-only segments require a pre-run DEM/signage direction check before the candidate can be promoted.",
        }

    checks = []
    for segment in ascent_segments:
        start_elevation = elevation_sampler(segment["start"])
        end_elevation = elevation_sampler(segment["end"])
        delta = (
            None
            if start_elevation is None or end_elevation is None
            else end_elevation - start_elevation
        )
        if delta is None or delta == 0:
            planned_direction = "unknown"
            planned_gain = None
        elif delta > 0:
            planned_direction = "official_geometry_start_to_end"
            planned_gain = delta
        else:
            planned_direction = "official_geometry_end_to_start"
            planned_gain = abs(delta)
        checks.append(
            {
                "seg_id": segment["seg_id"],
                "start": segment["start"],
                "end": segment["end"],
                "start_elevation_ft": round(start_elevation, 1)
                if start_elevation is not None
                else None,
                "end_elevation_ft": round(end_elevation, 1)
                if end_elevation is not None
                else None,
                "elevation_delta_ft": round(delta, 1) if delta is not None else None,
                "planned_traversal_direction": planned_direction,
                "planned_elevation_gain_ft": round(planned_gain, 1)
                if planned_gain is not None
                else None,
                "passed": planned_gain is not None and planned_gain > 0,
            }
        )
    passed = all(check["passed"] for check in checks)
    return {
        "passed": passed,
        "reason": "ascent_segments_checked" if passed else "ascent_segment_direction_failed",
        "ascent_segment_ids_checked": [segment["seg_id"] for segment in ascent_segments],
        "ascent_segment_checks": checks,
        "planned_traversal_direction": {
            str(check["seg_id"]): check["planned_traversal_direction"]
            for check in checks
        },
        "note": "Candidate generation plans ascent-only segments in the uphill DEM direction; current signage/map confirmation remains required for field use.",
    }


def trailhead_snap_confidence(
    trailhead: dict[str, Any],
    candidate_start: tuple[float, float],
    connector_graph: dict[str, Any] | None,
    outing_model: dict[str, Any],
    avoid_official_segment_ids: set[int] | None = None,
) -> dict[str, Any]:
    access = build_trailhead_access(
        trailhead,
        candidate_start,
        connector_graph,
        outing_model,
        avoid_official_segment_ids=avoid_official_segment_ids,
    )
    direct_gap = float(access["direct_gap_miles"])
    mapped_limit = float(outing_model.get("mapped_trailhead_access_max_miles", 0.75))
    access_class = "direct"
    if direct_gap <= 0.1:
        confidence = "high"
    elif access["source"] == "mapped_graph" and bool(access["graph_validated"]):
        confidence = "medium"
        if float(access["one_way_miles"]) > mapped_limit:
            access_class = "long_mapped"
        else:
            access_class = "mapped"
    else:
        confidence = "low"
    return {
        "confidence": confidence,
        "access_class": access_class,
        "direct_gap_miles": access["direct_gap_miles"],
        "mapped_access_miles": access["one_way_miles"] if access["source"] == "mapped_graph" else None,
        "graph_validated": bool(access["graph_validated"]),
    }


def build_trailhead_access(
    trailhead: dict[str, Any],
    candidate_start: tuple[float, float],
    connector_graph: dict[str, Any] | None,
    outing_model: dict[str, Any],
    avoid_official_segment_ids: set[int] | None = None,
) -> dict[str, Any]:
    trailhead_point = (float(trailhead["lon"]), float(trailhead["lat"]))
    direct_gap = haversine_miles(trailhead_point, candidate_start)
    mapped = shortest_connector_path(
        trailhead_point,
        candidate_start,
        connector_graph,
        float(outing_model.get("mapped_connector_snap_tolerance_miles", 0.02)),
        avoid_official_segment_ids=avoid_official_segment_ids,
    )
    if mapped:
        outbound_path = mapped["path_coordinates"]
        one_way_connector = float(mapped["connector_miles"])
        one_way_official_repeat = float(mapped["official_repeat_miles"])
        one_way_miles = float(mapped["distance_miles"])
        source = "mapped_graph"
        graph_validated = True
        connector_names = mapped["connector_names"]
        connector_classes = mapped.get("connector_classes", [])
        connector_edges = mapped.get("connector_edges", [])
        official_repeat_segment_ids = mapped["official_repeat_segment_ids"]
    else:
        outbound_path = [
            [float(trailhead["lon"]), float(trailhead["lat"])],
            [float(candidate_start[0]), float(candidate_start[1])],
        ]
        one_way_connector = direct_gap
        one_way_official_repeat = 0.0
        one_way_miles = direct_gap
        source = "direct_gap_estimate"
        graph_validated = direct_gap <= 0.1
        connector_names = []
        connector_classes = []
        connector_edges = []
        official_repeat_segment_ids = []

    return_path = list(reversed(outbound_path))
    return {
        "source": source,
        "one_way_miles": round_miles(one_way_miles),
        "round_trip_miles": round_miles(one_way_miles * 2),
        "one_way_connector_miles": round_miles(one_way_connector),
        "round_trip_connector_miles": round_miles(one_way_connector * 2),
        "one_way_official_repeat_miles": round_miles(one_way_official_repeat),
        "round_trip_official_repeat_miles": round_miles(one_way_official_repeat * 2),
        "direct_gap_miles": round_miles(direct_gap),
        "graph_validated": graph_validated,
        "connector_names": connector_names,
        "connector_classes": connector_classes,
        "connector_edges": connector_edges,
        "official_repeat_segment_ids": official_repeat_segment_ids,
        "outbound_path_coordinates": outbound_path,
        "return_path_coordinates": return_path,
    }


def candidate_route_status(
    validation: dict[str, Any],
    return_to_car: dict[str, Any],
    between_links: dict[str, Any],
) -> tuple[str, int]:
    graph_validated = (
        validation["return_path_graph_validated"]
        and validation["ascent_direction_passed"]
        and validation["trailhead_snap_confidence"] != "low"
        and validation["connector_overlap_checked"]
        and between_links["all_graph_validated"]
    )
    if graph_validated:
        return "graph_validated", 2
    if return_to_car["strategy"] == "out_and_back" and validation["trailhead_snap_confidence"] != "low":
        return "draft", 1
    return "draft", 0


def candidate_from_trail_group(
    trails: list[dict[str, Any]],
    state: dict[str, Any],
    performance_profile: dict[str, Any],
    connector_graph: dict[str, Any] | None,
    candidate_type: str,
    elevation_sampler: ElevationSampler | None = None,
) -> dict[str, Any]:
    outing_model = get_outing_model(state)
    trails, route_orientation = orient_trails_for_access(
        trails,
        state,
        connector_graph,
        outing_model,
    )
    trail = combine_trails(trails)
    required_segment_ids = {int(seg_id) for seg_id in trail.get("remaining_segment_ids") or []}
    drive_model = get_drive_model(state)
    trailhead = choose_trailhead(trail["start"], state)
    drive_to = drive_minutes_to_trailhead(trailhead, drive_model)
    parking_minutes = int(trailhead.get("parking_minutes") or state.get("parking_minutes") or 8)
    trailhead_access = build_trailhead_access(
        trailhead,
        trail["start"],
        connector_graph,
        outing_model,
        avoid_official_segment_ids=required_segment_ids,
    )
    snap = trailhead_snap_confidence(
        trailhead,
        trail["start"],
        connector_graph,
        outing_model,
        avoid_official_segment_ids=required_segment_ids,
    )
    trailhead_access["snap_confidence"] = snap["confidence"]
    trailhead_access["validated"] = (
        snap["confidence"] in {"high", "medium"} and bool(snap["graph_validated"])
    )
    segment_estimates = [
        estimate_segment_time(segment, performance_profile) for segment in trail["segments"]
    ]
    segment_estimates = enrich_segment_estimates_with_elevation(
        segment_estimates,
        trail["segments"],
        elevation_sampler,
    )
    inter_segment_links = build_inter_segment_links(
        trail["segments"],
        outing_model,
        connector_graph,
        avoid_official_segment_ids=required_segment_ids,
    )
    link_by_to_segment_id = {
        str(link.get("to_segment_id")): link
        for link in inter_segment_links.get("links") or []
        if link.get("to_segment_id") is not None
    }
    for segment in segment_estimates:
        link = link_by_to_segment_id.get(str(segment.get("seg_id")))
        if link:
            segment["pre_connector_link"] = link
    return_to_car = build_return_to_car(
        trail,
        outing_model,
        connector_graph,
        avoid_official_segment_ids=set(),
    )
    fallback_pace = float(performance_profile["fallback_pace_min_per_mile"])
    access_connector_miles = float(trailhead_access["round_trip_connector_miles"])
    access_official_repeat_miles = float(trailhead_access["round_trip_official_repeat_miles"])
    access_on_foot_miles = access_connector_miles + access_official_repeat_miles
    access_minutes = ceil_minutes(access_on_foot_miles * fallback_pace)
    between_links = summarize_between_trail_links(inter_segment_links)
    return_on_foot_miles = (
        float(return_to_car["official_repeat_miles"])
        + float(return_to_car["connector_miles"])
        + float(return_to_car["road_miles"])
        + float(inter_segment_links["connector_miles"])
        + float(inter_segment_links["official_repeat_miles"])
        + access_connector_miles
        + access_official_repeat_miles
    )
    moving_time = sum(segment["estimated_moving_minutes"] for segment in segment_estimates)
    moving_time += ceil_minutes(return_on_foot_miles * fallback_pace)
    raw_total_minutes = drive_to + parking_minutes + moving_time + drive_to
    official_miles = trail["official_miles"]
    total_on_foot = official_miles + return_on_foot_miles
    official_to_total_ratio = official_miles / total_on_foot if total_on_foot else 0
    effort = build_elevation_effort(
        trail["segments"],
        elevation_sampler,
        official_miles,
        moving_time,
    )
    official_repeat_miles = (
        float(return_to_car["official_repeat_miles"])
        + float(inter_segment_links["official_repeat_miles"])
        + access_official_repeat_miles
    )
    connector_miles = (
        float(return_to_car["connector_miles"]) + float(inter_segment_links["connector_miles"])
        + access_connector_miles
    )
    road_miles = float(return_to_car["road_miles"])
    route_finding_penalty_minutes = build_route_finding_penalty_minutes(
        trail_names=trail["trail_names"],
        between_links=between_links,
        trailhead_snap_confidence=snap["confidence"],
        official_repeat_miles=official_repeat_miles,
        connector_miles=connector_miles,
        road_miles=road_miles,
    )
    time_estimates = build_time_estimates_minutes(
        drive_to=drive_to,
        parking_minutes=parking_minutes,
        raw_moving_minutes=moving_time,
        effort=effort,
        route_finding_penalty_minutes=route_finding_penalty_minutes,
    )
    total_minutes = int(time_estimates["recommended_door_to_door"])
    efficiency = official_miles / total_minutes if total_minutes else 0
    effort_total_minutes = (
        drive_to + parking_minutes + int(effort["effort_score"]) + drive_to
    )
    effort_adjusted_efficiency = official_miles / effort_total_minutes if effort_total_minutes else 0
    flags = []
    if return_to_car["strategy"] == "out_and_back" or return_to_car.get("official_repeat_segment_ids"):
        flags.append("requires_official_repeat_to_get_back_to_car")
    repeat_segment_ids = set()
    repeat_segment_ids.update(
        int(seg_id)
        for seg_id in (trailhead_access.get("official_repeat_segment_ids") or [])
        if str(seg_id).isdigit()
    )
    for link in inter_segment_links.get("links") or []:
        earned_before_link = {
            int(seg_id)
            for seg_id in (link.get("earned_segment_ids_before_link") or [])
            if str(seg_id).isdigit()
        }
        repeat_segment_ids.update(
            int(seg_id)
            for seg_id in (link.get("official_repeat_segment_ids") or [])
            if str(seg_id).isdigit() and int(seg_id) not in earned_before_link
        )
    self_repeat_segment_ids = sorted(required_segment_ids & repeat_segment_ids)
    if self_repeat_segment_ids:
        flags.append("self_official_repeat_connector_requires_review")
    if return_to_car["needs_map_validation"]:
        flags.append("return_connector_needs_map_validation")
    if not inter_segment_links["all_graph_validated"]:
        flags.append("between_trail_connector_needs_map_validation")
    if official_to_total_ratio < 0.7:
        flags.append("low_official_to_total_mileage_ratio")
    direction_validation = build_direction_validation(
        trail["segments"],
        elevation_sampler=elevation_sampler,
    )
    validation = {
        "segment_coverage_passed": True,
        "ascent_direction_passed": direction_validation["passed"],
        "return_path_graph_validated": bool(return_to_car.get("graph_validated")),
        "trailhead_snap_confidence": snap["confidence"],
        "trailhead_snap": snap,
        "connector_overlap_checked": bool(connector_graph),
        "special_management_checked": False,
    }
    if not validation["ascent_direction_passed"]:
        flags.append("ascent_direction_needs_pre_run_validation")
    if validation["trailhead_snap_confidence"] == "low":
        flags.append("low_trailhead_snap_confidence")
    if snap.get("access_class") == "long_mapped":
        flags.append("long_mapped_trailhead_access")
    if not validation["connector_overlap_checked"]:
        flags.append("connector_overlap_not_checked")
    route_status, route_quality_score = candidate_route_status(
        validation,
        return_to_car,
        inter_segment_links,
    )
    if self_repeat_segment_ids and route_status == "graph_validated":
        route_status = "draft"
        route_quality_score = min(route_quality_score, 1)

    return {
        "candidate_id": slugify("-".join(trail["trail_names"])),
        "candidate_type": candidate_type,
        "trail_names": trail["trail_names"],
        "segment_ids": trail["remaining_segment_ids"],
        "segments": segment_estimates,
        "custom_traversal_order": True,
        "direction_counts": trail["direction_counts"],
        "route_status": route_status,
        "validation": validation,
        "direction_validation": direction_validation,
        "route_orientation": route_orientation,
        "trailhead": serialize_trailhead(trailhead),
        "return_to_car": return_to_car,
        "trailhead_access": trailhead_access,
        "official_new_miles": round_miles(official_miles),
        "official_repeat_miles": round_miles(official_repeat_miles),
        "connector_miles": round_miles(connector_miles),
        "between_trail_links": between_links,
        "inter_segment_links": inter_segment_links,
        "self_official_repeat_segment_ids": self_repeat_segment_ids,
        "between_trail_connector_miles": round_miles(float(between_links["connector_miles"])),
        "inter_segment_connector_miles": round_miles(float(inter_segment_links["connector_miles"])),
        "road_miles": round_miles(road_miles),
        "estimated_total_on_foot_miles": round_miles(total_on_foot),
        "time_breakdown_minutes": {
            "drive_to_trailhead": drive_to,
            "parking_and_prep": parking_minutes,
            "trailhead_access": access_minutes,
            "moving_time": moving_time,
            "return_drive": drive_to,
        },
        "time_estimates_minutes": time_estimates,
        "raw_total_minutes": raw_total_minutes,
        "total_minutes": total_minutes,
        "time_bucket": classify_bucket(total_minutes),
        "efficiency_score": round(efficiency, 4),
        "route_quality_score": route_quality_score,
        "official_to_total_on_foot_ratio": round(official_to_total_ratio, 3),
        "effort": effort,
        "effort_adjusted_efficiency_score": round(effort_adjusted_efficiency, 4),
        "less_optimal_flags": flags,
        "optimality_rank_in_bucket": None,
    }


def candidate_from_trail(
    trail: dict[str, Any],
    state: dict[str, Any],
    performance_profile: dict[str, Any],
    connector_graph: dict[str, Any] | None,
    elevation_sampler: ElevationSampler | None = None,
) -> dict[str, Any]:
    return candidate_from_trail_group(
        [trail],
        state,
        performance_profile,
        connector_graph,
        candidate_type="single_trail",
        elevation_sampler=elevation_sampler,
    )


def build_bundle_candidates(
    trails: list[dict[str, Any]],
    state: dict[str, Any],
    performance_profile: dict[str, Any],
    connector_graph: dict[str, Any] | None,
    elevation_sampler: ElevationSampler | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trail in trails:
        trailhead = choose_trailhead(trail["start"], state)
        grouped[str(trailhead["name"])].append(trail)

    bundle_candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for trailhead_name, group in grouped.items():
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda item: (-item["official_miles"], item["trail_name"]))
        selected: list[dict[str, Any]] = []
        buckets_seen_for_group: set[str] = set()
        for trail in ordered:
            selected.append(trail)
            if len(selected) < 2:
                continue
            candidate = candidate_from_trail_group(
                selected,
                state,
                performance_profile,
                connector_graph,
                candidate_type="trailhead_bundle",
                elevation_sampler=elevation_sampler,
            )
            if candidate["total_minutes"] < 55:
                continue
            key = (trailhead_name, tuple(sorted(candidate["segment_ids"])))
            if key in seen:
                continue
            seen.add(key)
            # Keep the progressive bundle that first enters each time bucket,
            # plus every four-plus bundle up to roughly a long summer outing.
            if candidate["time_bucket"] not in buckets_seen_for_group or (
                candidate["time_bucket"] == "four_plus_hours" and candidate["total_minutes"] <= 390
            ):
                candidate["bundle_trailhead"] = trailhead_name
                bundle_candidates.append(candidate)
                buckets_seen_for_group.add(candidate["time_bucket"])
            if candidate["total_minutes"] > 390:
                break
    return bundle_candidates


def trail_is_reversible(trail: dict[str, Any]) -> bool:
    return all(segment["direction"] == "both" for segment in trail["segments"])


def order_trails_nearest_neighbor(
    trails: list[dict[str, Any]],
    start_point: tuple[float, float],
) -> list[dict[str, Any]]:
    remaining = list(trails)
    ordered: list[dict[str, Any]] = []
    current = start_point
    while remaining:
        best_index = 0
        best_trail = remaining[0]
        best_distance = haversine_miles(current, best_trail["start"])
        for index, trail in enumerate(remaining):
            options = [(haversine_miles(current, trail["start"]), trail)]
            if trail_is_reversible(trail):
                reversed_trail = reverse_trail_orientation(trail)
                options.append((haversine_miles(current, reversed_trail["start"]), reversed_trail))
            distance, oriented = min(options, key=lambda item: item[0])
            if distance < best_distance:
                best_index = index
                best_trail = oriented
                best_distance = distance
        remaining.pop(best_index)
        ordered.append(best_trail)
        current = best_trail["end"]
    return ordered


def connector_distances_to_targets(
    start: tuple[float, float],
    target_points: list[tuple[float, float]],
    connector_graph: dict[str, Any] | None,
    snap_tolerance_miles: float,
    avoid_official_segment_ids: set[int] | None = None,
) -> dict[tuple[float, float], float]:
    if not connector_graph or not target_points:
        return {}
    avoided_ids = frozenset(int(seg_id) for seg_id in avoid_official_segment_ids or set())
    cache_key = (
        round_node(start),
        tuple(sorted(round_node(point) for point in target_points)),
        round(snap_tolerance_miles, 3),
        tuple(sorted(avoided_ids)),
    )
    cache = connector_graph.setdefault("_target_distance_cache", {})
    if cache_key in cache:
        return cache[cache_key]

    start_node = nearest_connector_node_for_graph(start, connector_graph, snap_tolerance_miles)
    if not start_node:
        cache[cache_key] = {}
        return {}

    target_node_points: dict[tuple[float, float], list[tuple[float, float]]] = defaultdict(list)
    target_snap_miles: dict[tuple[float, float], float] = {}
    for point in target_points:
        target_node = nearest_connector_node_for_graph(point, connector_graph, snap_tolerance_miles)
        if not target_node:
            continue
        target_node_points[target_node[0]].append(point)
        target_snap_miles[point] = target_node[1]
    pending_targets = set(target_node_points)
    if not pending_targets:
        cache[cache_key] = {}
        return {}

    graph = connector_graph["graph"]
    queue: list[tuple[float, int, tuple[float, float]]] = [(0.0, 0, start_node[0])]
    push_count = 1
    distances: dict[tuple[float, float], float] = {start_node[0]: 0.0}
    found: dict[tuple[float, float], float] = {}
    while queue and pending_targets:
        distance, _, node = heapq.heappop(queue)
        if distance > distances.get(node, math.inf):
            continue
        if node in pending_targets:
            pending_targets.remove(node)
            for point in target_node_points[node]:
                found[point] = distance + start_node[1] + target_snap_miles[point]
            if not pending_targets:
                break
        for edge in graph.get(node, []):
            if edge.get("edge_type") == "official_repeat" and edge.get("seg_id") in avoided_ids:
                continue
            next_distance = distance + edge["distance"]
            next_node = edge["to"]
            if next_distance >= distances.get(next_node, math.inf):
                continue
            distances[next_node] = next_distance
            heapq.heappush(queue, (next_distance, push_count, next_node))
            push_count += 1

    cache[cache_key] = found
    return found


def connector_distance_index(
    start: tuple[float, float],
    connector_graph: dict[str, Any] | None,
    snap_tolerance_miles: float,
    avoid_official_segment_ids: set[int] | None = None,
) -> dict[str, Any] | None:
    if not connector_graph:
        return None
    avoided_ids = frozenset(int(seg_id) for seg_id in avoid_official_segment_ids or set())
    cache_key = (
        round_node(start),
        round(snap_tolerance_miles, 3),
        tuple(sorted(avoided_ids)),
    )
    cache = connector_graph.setdefault("_distance_index_cache", {})
    if cache_key in cache:
        return cache[cache_key]

    start_node = nearest_connector_node_for_graph(start, connector_graph, snap_tolerance_miles)
    if not start_node:
        cache[cache_key] = None
        return None

    graph = connector_graph["graph"]
    queue: list[tuple[float, int, tuple[float, float]]] = [(0.0, 0, start_node[0])]
    push_count = 1
    distances: dict[tuple[float, float], float] = {start_node[0]: 0.0}
    while queue:
        distance, _, node = heapq.heappop(queue)
        if distance > distances.get(node, math.inf):
            continue
        for edge in graph.get(node, []):
            if edge.get("edge_type") == "official_repeat" and edge.get("seg_id") in avoided_ids:
                continue
            next_distance = distance + edge["distance"]
            next_node = edge["to"]
            if next_distance >= distances.get(next_node, math.inf):
                continue
            distances[next_node] = next_distance
            heapq.heappush(queue, (next_distance, push_count, next_node))
            push_count += 1

    result = {
        "snap_start_miles": start_node[1],
        "distances": distances,
    }
    cache[cache_key] = result
    return result


def connector_index_distance(
    start: tuple[float, float],
    end: tuple[float, float],
    connector_graph: dict[str, Any] | None,
    snap_tolerance_miles: float,
    avoid_official_segment_ids: set[int] | None = None,
) -> float | None:
    index = connector_distance_index(
        start,
        connector_graph,
        snap_tolerance_miles,
        avoid_official_segment_ids=avoid_official_segment_ids,
    )
    if not index or not connector_graph:
        return None
    end_node = nearest_connector_node_for_graph(end, connector_graph, snap_tolerance_miles)
    if not end_node:
        return None
    distance = index["distances"].get(end_node[0])
    if distance is None:
        return None
    return float(distance) + float(index["snap_start_miles"]) + end_node[1]


def connector_or_gap_distance(
    start: tuple[float, float],
    end: tuple[float, float],
    connector_graph: dict[str, Any] | None,
    snap_tolerance_miles: float,
    avoid_official_segment_ids: set[int] | None = None,
) -> float:
    indexed_distance = connector_index_distance(
        start,
        end,
        connector_graph,
        snap_tolerance_miles,
        avoid_official_segment_ids=avoid_official_segment_ids,
    )
    if indexed_distance is not None:
        return indexed_distance
    return haversine_miles(start, end) * 1.25


def order_trails_by_legal_connector_cost(
    trails: list[dict[str, Any]],
    start_point: tuple[float, float],
    connector_graph: dict[str, Any] | None,
    snap_tolerance_miles: float,
    avoid_official_segment_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Nearest-neighbor ordering using legal graph cost instead of straight-line distance."""

    remaining = list(trails)
    ordered: list[dict[str, Any]] = []
    current = start_point
    required_ids = {int(seg_id) for seg_id in avoid_official_segment_ids or set()}
    earned_ids: set[int] = set()
    while remaining:
        avoid_ids = required_ids - earned_ids if avoid_official_segment_ids is not None else None
        best_index = 0
        best_trail = remaining[0]
        best_distance = float("inf")
        option_records: list[tuple[int, dict[str, Any]]] = []
        for index, trail in enumerate(remaining):
            option_records.append((index, trail))
            if trail_is_reversible(trail):
                option_records.append((index, reverse_trail_orientation(trail)))
        graph_distances = connector_distances_to_targets(
            current,
            [option["start"] for _, option in option_records],
            connector_graph,
            snap_tolerance_miles,
            avoid_official_segment_ids=avoid_ids,
        )
        for index, option in option_records:
            distance = graph_distances.get(option["start"])
            if distance is None:
                distance = haversine_miles(current, option["start"]) * 1.25
            if distance < best_distance:
                best_index = index
                best_trail = option
                best_distance = distance
        remaining.pop(best_index)
        ordered.append(best_trail)
        earned_ids.update(
            int(seg_id)
            for seg_id in best_trail.get("remaining_segment_ids") or []
            if str(seg_id).isdigit()
        )
        current = best_trail["end"]
    return ordered


def build_long_access_bundle_candidates(
    trails: list[dict[str, Any]],
    single_candidates: list[dict[str, Any]],
    state: dict[str, Any],
    performance_profile: dict[str, Any],
    connector_graph: dict[str, Any] | None,
    elevation_sampler: ElevationSampler | None = None,
) -> list[dict[str, Any]]:
    trail_by_name = {trail["trail_name"]: trail for trail in trails}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in single_candidates:
        if candidate.get("candidate_type") != "single_trail":
            continue
        if candidate.get("route_status") != "graph_validated":
            continue
        if "long_mapped_trailhead_access" not in (candidate.get("less_optimal_flags") or []):
            continue
        trail_names = candidate.get("trail_names") or []
        if len(trail_names) != 1 or trail_names[0] not in trail_by_name:
            continue
        grouped[str(candidate["trailhead"]["name"])].append(
            {
                "trail": trail_by_name[trail_names[0]],
                "candidate": candidate,
            }
        )

    long_access_candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for trailhead_name, records in grouped.items():
        if len(records) < 2:
            continue
        records = sorted(
            records,
            key=lambda record: (
                float(record["candidate"].get("official_new_miles") or 0),
                record["trail"]["trail_name"],
            ),
        )
        selected: list[dict[str, Any]] = []
        for record in records[:8]:
            selected.append(record["trail"])
            if len(selected) < 2:
                continue
            start_trailhead = choose_trailhead(selected[0]["start"], state)
            outing_model = get_outing_model(state)
            selected_required_ids = {
                int(seg_id)
                for trail in selected
                for seg_id in trail.get("remaining_segment_ids") or []
                if str(seg_id).isdigit()
            }
            ordered = order_trails_by_legal_connector_cost(
                selected,
                (float(start_trailhead["lon"]), float(start_trailhead["lat"])),
                connector_graph,
                float(outing_model.get("mapped_connector_snap_tolerance_miles", 0.02)),
                avoid_official_segment_ids=selected_required_ids,
            )
            candidate = candidate_from_trail_group(
                ordered,
                state,
                performance_profile,
                connector_graph,
                candidate_type="long_access_bundle",
                elevation_sampler=elevation_sampler,
            )
            key = (trailhead_name, tuple(sorted(candidate["segment_ids"])))
            if key in seen:
                continue
            seen.add(key)
            if candidate["route_status"] != "graph_validated":
                continue
            if candidate["total_minutes"] > 540:
                continue
            candidate["bundle_trailhead"] = trailhead_name
            candidate["bundle_reason"] = "amortize_long_mapped_trailhead_access"
            long_access_candidates.append(candidate)
    return long_access_candidates


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", normalize_name(value)).strip("-")
    return slug or "candidate"


def build_route_menu(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _ in TIME_BUCKETS}
    for candidate in candidates:
        buckets.setdefault(candidate["time_bucket"], []).append(candidate)
    for bucket_candidates in buckets.values():
        bucket_candidates.sort(
            key=lambda item: (
                -(1 if item.get("route_status") == "graph_validated" else 0),
                -item.get("effort_adjusted_efficiency_score", item["efficiency_score"]),
                -item["efficiency_score"],
                -item["official_new_miles"],
                item["estimated_total_on_foot_miles"],
                item["total_minutes"],
                item["trail_names"],
            )
        )
        for rank, candidate in enumerate(bucket_candidates, start=1):
            candidate["optimality_rank_in_bucket"] = rank

    bucket_order = {name: index for index, (name, _, _) in enumerate(TIME_BUCKETS)}
    all_candidates = sorted(
        candidates,
        key=lambda item: (
            bucket_order.get(item["time_bucket"], 999),
            item["optimality_rank_in_bucket"] or 999,
        ),
    )
    primary = {
        bucket_name: bucket_candidates[0]
        for bucket_name, bucket_candidates in buckets.items()
        if bucket_candidates
    }
    primary_validated = {}
    primary_draft = {}
    for bucket_name, bucket_candidates in buckets.items():
        for candidate in bucket_candidates:
            if candidate.get("route_status") == "graph_validated":
                primary_validated[bucket_name] = candidate
                break
        for candidate in bucket_candidates:
            if candidate.get("route_status") != "graph_validated":
                primary_draft[bucket_name] = candidate
                break
    return {
        "bucket_definitions": [
            {"name": name, "min_minutes": low, "max_minutes": high}
            for name, low, high in TIME_BUCKETS
        ],
        "primary_candidates_by_bucket": primary,
        "primary_validated_candidates_by_bucket": primary_validated,
        "primary_draft_candidates_by_bucket": primary_draft,
        "buckets": buckets,
        "all_candidates": all_candidates,
    }


def build_coverage_validation(
    official_segments: list[dict[str, Any]],
    completed_ids: set[int],
    blocked_ids: set[int],
    remaining_segments: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    official_ids = {segment["seg_id"] for segment in official_segments}
    remaining_ids = {segment["seg_id"] for segment in remaining_segments}
    candidate_ids = {
        seg_id for candidate in candidates for seg_id in candidate.get("segment_ids", [])
    }
    unknown_completed = sorted(completed_ids - official_ids)
    unknown_blocked = sorted(blocked_ids - official_ids)
    missing_candidate_ids = sorted(remaining_ids - candidate_ids)
    duplicate_counts = Counter(
        seg_id for candidate in candidates for seg_id in candidate.get("segment_ids", [])
    )
    duplicate_candidate_ids = sorted(seg_id for seg_id, count in duplicate_counts.items() if count > 1)
    return {
        "valid": not (
            unknown_completed
            or unknown_blocked
            or missing_candidate_ids
        ),
        "official_segment_count": len(official_ids),
        "remaining_available_segment_count": len(remaining_ids),
        "candidate_covered_segment_count": len(candidate_ids),
        "unknown_completed_segment_ids": unknown_completed,
        "unknown_blocked_segment_ids": unknown_blocked,
        "remaining_segments_missing_from_candidates": missing_candidate_ids,
        "duplicate_candidate_segment_ids": duplicate_candidate_ids,
        "direction_counts_remaining": dict(
            sorted(Counter(segment["direction"] for segment in remaining_segments).items())
        ),
    }


def build_plan(
    official_geojson: str | Path,
    state: dict[str, Any] | None = None,
    generated_at: str | None = None,
    strava_activity_details_dir: str | Path | None = None,
    activity_summary_csv: str | Path | None = None,
    activity_detail_summary_csv: str | Path | None = None,
    segment_perf_csv: str | Path | None = None,
    connector_geojson: str | Path | None = None,
    trailheads_geojson: str | Path | None = None,
    private_parking_anchors_geojson: str | Path | None = None,
    dem_tif: str | Path | None = None,
    dem_summary_json: str | Path | None = None,
) -> dict[str, Any]:
    state = state or {}
    public_trailheads = load_trailheads_from_geojson(
        Path(trailheads_geojson) if trailheads_geojson else None
    )
    private_parking_anchors = load_trailheads_from_geojson(
        Path(private_parking_anchors_geojson) if private_parking_anchors_geojson else None
    )
    state = merge_planning_trailheads(state, [*public_trailheads, *private_parking_anchors])
    official_path = Path(official_geojson)
    official_segments, official_meta = load_official_segments(official_path)
    completed_ids = as_int_set(state.get("completed_segment_ids"))
    blocked_ids = as_int_set(state.get("blocked_segment_ids"))
    blocked_trails = {normalize_name(name) for name in state.get("blocked_trail_names") or []}
    strava_dir = Path(strava_activity_details_dir) if strava_activity_details_dir else None
    performance_profile = build_performance_profile(
        state=state,
        strava_activity_details_dir=strava_dir,
        activity_summary_csv=Path(activity_summary_csv) if activity_summary_csv else None,
        activity_detail_summary_csv=Path(activity_detail_summary_csv)
        if activity_detail_summary_csv
        else None,
        segment_perf_csv=Path(segment_perf_csv) if segment_perf_csv else None,
    )
    connector_graph = (
        load_connector_graph(Path(connector_geojson), official_segments=official_segments)
        if connector_geojson
        else None
    )
    dem_context = load_dem_context(
        Path(dem_tif) if dem_tif else None,
        Path(dem_summary_json) if dem_summary_json else None,
    )
    elevation_sampler = dem_context["sampler"]

    remaining_segments = [
        segment
        for segment in official_segments
        if segment["seg_id"] not in completed_ids
        and segment["seg_id"] not in blocked_ids
        and normalize_name(segment["trail_name"]) not in blocked_trails
    ]
    blocked_by_trail_ids = {
        segment["seg_id"]
        for segment in official_segments
        if normalize_name(segment["trail_name"]) in blocked_trails
    }
    remaining_trails_raw = group_remaining_by_trail(remaining_segments)
    single_candidates = [
        candidate_from_trail(
            trail,
            state,
            performance_profile,
            connector_graph,
            elevation_sampler=elevation_sampler,
        )
        for trail in remaining_trails_raw
    ]
    bundle_candidates = build_bundle_candidates(
        remaining_trails_raw,
        state,
        performance_profile,
        connector_graph,
        elevation_sampler=elevation_sampler,
    )
    long_access_bundle_candidates = build_long_access_bundle_candidates(
        remaining_trails_raw,
        single_candidates,
        state,
        performance_profile,
        connector_graph,
        elevation_sampler=elevation_sampler,
    )
    candidates = single_candidates + bundle_candidates + long_access_bundle_candidates
    matched_segment_ids = {
        segment["seg_id"]
        for candidate in candidates
        for segment in candidate["segments"]
        if segment["time_source"]["source_type"] == "matched_strava_segment_effort"
    }
    performance_profile["matched_strava_segment_count"] = len(matched_segment_ids)
    performance_profile.pop("_effort_index", None)
    route_menu = build_route_menu(candidates)
    remaining_trails = [
        {
            "trail_name": trail["trail_name"],
            "remaining_segment_ids": trail["remaining_segment_ids"],
            "remaining_segment_count": len(trail["remaining_segment_ids"]),
            "official_miles": round_miles(trail["official_miles"]),
            "direction_counts": trail["direction_counts"],
        }
        for trail in remaining_trails_raw
    ]
    direction_counts = Counter(segment["direction"] for segment in official_segments)
    remaining_direction_counts = Counter(segment["direction"] for segment in remaining_segments)
    run_id = generated_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    plan = {
        "run_id": run_id,
        "generated_at": run_id,
        "planner": {
            "name": "personal_route_planner",
            "version": 1,
            "description": "Rerunnable route-menu planner for completed segments, closures, logistics, and time buckets.",
        },
        "source_datasets": {
            "official_geojson": official_meta,
            "connector_geojson": {
                "path": str(connector_geojson) if connector_geojson else None,
                "loaded": bool(connector_graph),
                "feature_count": connector_graph.get("feature_count") if connector_graph else None,
                "official_segment_edges_loaded": connector_graph.get("official_segment_count")
                if connector_graph
                else None,
                "connector_class_counts": connector_graph.get("connector_class_counts")
                if connector_graph
                else None,
            },
            "trailheads_geojson": {
                "path": str(trailheads_geojson) if trailheads_geojson else None,
                "loaded": bool(public_trailheads),
                "feature_count": len(public_trailheads),
            },
            "dem": dem_context["metadata"],
        },
        "state_inputs": {
            "completed_segment_ids": sorted(completed_ids),
            "blocked_segment_ids": sorted(blocked_ids),
            "blocked_trail_names": sorted(state.get("blocked_trail_names") or []),
            "drive_origin_label": get_drive_model(state).get("origin_label"),
            "parking_minutes_default": state.get("parking_minutes", 8),
            "trailhead_count": len(state.get("trailheads") or []),
            "public_trailheads_geojson": str(trailheads_geojson)
            if trailheads_geojson
            else None,
            "public_trailheads_loaded": len(public_trailheads),
        },
        "summary": {
            "official_segments": len(official_segments),
            "official_trails": len({segment["trail_name"] for segment in official_segments}),
            "official_miles": round_miles(sum(segment["official_miles"] for segment in official_segments)),
            "official_direction_counts": dict(sorted(direction_counts.items())),
            "completed_segments": len(completed_ids & {segment["seg_id"] for segment in official_segments}),
            "completed_official_miles": round_miles(
                sum(
                    segment["official_miles"]
                    for segment in official_segments
                    if segment["seg_id"] in completed_ids
                )
            ),
            "blocked_segments": len(
                (blocked_ids | blocked_by_trail_ids)
                & {segment["seg_id"] for segment in official_segments}
            ),
            "remaining_available_segments": len(remaining_segments),
            "remaining_available_trails": len(remaining_trails),
            "remaining_available_official_miles": round_miles(
                sum(segment["official_miles"] for segment in remaining_segments)
            ),
            "remaining_direction_counts": dict(sorted(remaining_direction_counts.items())),
        },
        "performance_profile": performance_profile,
        "general_plan": {
            "primary_rule": "Prefer graph_validated candidates in the matching time bucket; use draft candidates only as planning ideas when no validated option exists.",
            "remaining_trails_sorted_by_official_miles": remaining_trails,
        },
        "remaining_trails": remaining_trails,
        "route_menu": route_menu,
        "coverage_validation": build_coverage_validation(
            official_segments,
            completed_ids,
            blocked_ids | blocked_by_trail_ids,
            remaining_segments,
            candidates,
        ),
        "caveats": [
            "Draft candidates with estimated connectors, low trailhead confidence, or failed ascent validation are planning ideas only.",
            "Graph-validated candidates still need current Ridge to Rivers/signage/condition validation before field use.",
            "This tool ranks route candidates; it does not yet produce turn-by-turn GPX routes.",
            "Ascent-only official segments now carry pre-run DEM direction checks when the DEM is loaded; current signage remains authoritative.",
        ],
    }
    return plan


def render_markdown(plan: dict[str, Any], max_candidates_per_bucket: int = 8) -> str:
    lines = [
        "# 2026 Personal Route Menu",
        "",
        f"Generated: {plan['generated_at']}",
        "",
        "## Summary",
        "",
        f"- Remaining official miles: {plan['summary']['remaining_available_official_miles']}",
        f"- Remaining official segments: {plan['summary']['remaining_available_segments']}",
        f"- Completed official segments: {plan['summary']['completed_segments']}",
        f"- Blocked official segments: {plan['summary']['blocked_segments']}",
        f"- Timing fallback: {plan['performance_profile']['fallback_pace_min_per_mile']} min/mi ({plan['performance_profile']['fallback_pace_source']})",
        f"- Official segments with matched Strava segment-effort estimates: {plan['performance_profile']['matched_strava_segment_count']}",
        f"- Coverage validation: {'valid' if plan['coverage_validation']['valid'] else 'needs attention'}",
        "",
        "## Primary Picks By Time Window",
        "",
    ]
    primary = plan["route_menu"]["primary_candidates_by_bucket"]
    primary_validated = plan["route_menu"].get("primary_validated_candidates_by_bucket") or {}
    primary_draft = plan["route_menu"].get("primary_draft_candidates_by_bucket") or {}
    for bucket_name, _, _ in TIME_BUCKETS:
        candidate = primary.get(bucket_name)
        label = bucket_name.replace("_", " ")
        if not candidate:
            lines.append(f"- {label}: no candidate")
            continue
        status = candidate.get("route_status") or "unclassified"
        lines.append(
            "- "
            + f"{label}: [{status}] {', '.join(candidate['trail_names'])} "
            + f"({candidate['total_minutes']} min, "
            + f"{candidate['official_new_miles']} official mi, "
            + f"{candidate['estimated_total_on_foot_miles']} total on-foot mi, "
            + f"return: {candidate['return_to_car']['strategy']})"
        )
    lines.extend(["", "## Validated Vs Draft Primaries", ""])
    for bucket_name, _, _ in TIME_BUCKETS:
        label = bucket_name.replace("_", " ")
        validated = primary_validated.get(bucket_name)
        draft = primary_draft.get(bucket_name)
        validated_label = ", ".join(validated["trail_names"]) if validated else "none"
        draft_label = ", ".join(draft["trail_names"]) if draft else "none"
        lines.append(f"- {label}: validated={validated_label}; draft={draft_label}")
    lines.extend(["", "## Route Menu", ""])
    for bucket_name, _, _ in TIME_BUCKETS:
        bucket = plan["route_menu"]["buckets"].get(bucket_name) or []
        lines.append(f"### {bucket_name.replace('_', ' ').title()}")
        lines.append("")
        if not bucket:
            lines.extend(["No candidates.", ""])
            continue
        lines.append(
            "| Rank | Status | Trail | Total min | Official mi | Total foot mi | Return | Flags |"
        )
        lines.append("|---:|---|---|---:|---:|---:|---|---|")
        for candidate in bucket[:max_candidates_per_bucket]:
            flags = ", ".join(candidate["less_optimal_flags"]) or "primary-fit"
            lines.append(
                f"| {candidate['optimality_rank_in_bucket']} "
                f"| {candidate.get('route_status') or 'unclassified'} "
                f"| {', '.join(candidate['trail_names'])} "
                f"| {candidate['total_minutes']} "
                f"| {candidate['official_new_miles']} "
                f"| {candidate['estimated_total_on_foot_miles']} "
                f"| {candidate['return_to_car']['strategy']} "
                f"| {flags} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Caveats",
            "",
            *[f"- {caveat}" for caveat in plan["caveats"]],
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--output-json", type=Path, default=YEAR_DIR / "outputs" / "personal-route-menu.json")
    parser.add_argument(
        "--output-md", type=Path, default=YEAR_DIR / "outputs" / "personal-route-menu.md"
    )
    parser.add_argument("--strava-details-dir", type=Path, default=DEFAULT_STRAVA_DETAILS_DIR)
    parser.add_argument("--activity-summary-csv", type=Path, default=DEFAULT_ACTIVITY_SUMMARY_CSV)
    parser.add_argument(
        "--activity-detail-summary-csv", type=Path, default=DEFAULT_ACTIVITY_DETAIL_SUMMARY_CSV
    )
    parser.add_argument("--segment-perf-csv", type=Path, default=DEFAULT_SEGMENT_PERF_CSV)
    parser.add_argument("--connector-geojson", type=Path, default=DEFAULT_CONNECTOR_GEOJSON)
    parser.add_argument(
        "--trailheads-geojson",
        type=Path,
        default=DEFAULT_TRAILHEAD_CANDIDATES_GEOJSON,
    )
    parser.add_argument(
        "--private-parking-anchors-geojson",
        type=Path,
        default=DEFAULT_PRIVATE_PARKING_ANCHORS_GEOJSON,
    )
    parser.add_argument("--dem-tif", type=Path, default=DEFAULT_DEM_TIF)
    parser.add_argument("--dem-summary-json", type=Path, default=DEFAULT_DEM_SUMMARY_JSON)
    parser.add_argument("--max-candidates-per-bucket", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state = load_state(args.state)
    plan = build_plan(
        official_geojson=args.official,
        state=state,
        strava_activity_details_dir=args.strava_details_dir,
        activity_summary_csv=args.activity_summary_csv,
        activity_detail_summary_csv=args.activity_detail_summary_csv,
        segment_perf_csv=args.segment_perf_csv,
        connector_geojson=args.connector_geojson,
        trailheads_geojson=args.trailheads_geojson,
        private_parking_anchors_geojson=args.private_parking_anchors_geojson,
        dem_tif=args.dem_tif,
        dem_summary_json=args.dem_summary_json,
    )
    write_json(args.output_json, plan)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(
        render_markdown(plan, max_candidates_per_bucket=args.max_candidates_per_bucket)
    )
    manifest_path = args.output_json.parent / f"{args.output_json.stem}-artifact-manifest.json"
    inputs = [
        args.official,
        args.state,
        args.connector_geojson,
        args.trailheads_geojson,
        args.dem_tif,
        args.dem_summary_json,
    ]
    optional_inputs = [
        args.activity_summary_csv,
        args.activity_detail_summary_csv,
        args.segment_perf_csv,
    ]
    inputs.extend(path for path in optional_inputs if path.exists())
    write_manifest(
        manifest_path,
        build_artifact_manifest(
            run_id=plan["run_id"],
            inputs=inputs,
            outputs=[args.output_json, args.output_md],
            command="personal_route_planner.py",
            metadata={"planner": plan["planner"]},
        ),
    )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(f"Wrote {manifest_path}")
    print(
        "Remaining: "
        f"{plan['summary']['remaining_available_segments']} segments, "
        f"{plan['summary']['remaining_available_official_miles']} official miles"
    )
    print(
        "Coverage validation: "
        + ("valid" if plan["coverage_validation"]["valid"] else "needs attention")
    )
    return 0 if plan["coverage_validation"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
