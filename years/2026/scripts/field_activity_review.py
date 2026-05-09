#!/usr/bin/env python3
"""Review one field activity against all official 2026 BTC foot segments."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from personal_route_planner import (  # noqa: E402
    DEFAULT_DEM_SUMMARY_JSON,
    DEFAULT_DEM_TIF,
    DEFAULT_OFFICIAL_GEOJSON,
    activity_geometry,
    haversine_miles,
    load_dem_context,
    load_official_segments,
    point_to_polyline_distance_miles,
    read_json,
    sample_line,
    write_json,
)


DEFAULT_OUTPUT_JSON = YEAR_DIR / "outputs" / "private" / "progress" / "activity-review-latest.json"
ElevationSampler = Callable[[tuple[float, float]], float | None]


def normalized_ids(values: list[Any] | tuple[Any, ...] | set[Any] | None) -> list[str]:
    return sorted({str(value) for value in values or [] if value is not None}, key=lambda item: (len(item), item))


def parse_id_list(value: str | None) -> list[str]:
    if not value:
        return []
    return normalized_ids([item.strip() for item in value.replace(";", ",").split(",") if item.strip()])


def gpx_coordinates(path: Path) -> list[tuple[float, float]]:
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    coords = []
    for elem in root.iter():
        if elem.tag.endswith("trkpt") or elem.tag.endswith("rtept"):
            lat = elem.attrib.get("lat")
            lon = elem.attrib.get("lon")
            if lat is not None and lon is not None:
                coords.append((float(lon), float(lat)))
    return coords


def activity_coordinates(path: Path) -> list[tuple[float, float]]:
    if path.suffix.lower() == ".gpx":
        return gpx_coordinates(path)
    data = read_json(path)
    if isinstance(data, dict):
        if isinstance(data.get("coordinates"), list):
            return [(float(lon), float(lat)) for lon, lat in data["coordinates"]]
        geometry = data.get("geometry") or {}
        if geometry.get("type") == "LineString" and isinstance(geometry.get("coordinates"), list):
            return [(float(lon), float(lat)) for lon, lat in geometry["coordinates"]]
        coords = activity_geometry(data)
        if coords:
            return coords
    return []


def nearest_activity_index(point: tuple[float, float], activity_coords: list[tuple[float, float]]) -> int | None:
    if not activity_coords:
        return None
    best_index = None
    best_distance = None
    for index, coord in enumerate(activity_coords):
        distance = haversine_miles(point, coord)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_index = index
    return best_index


def ascent_direction_ok(
    segment: dict[str, Any],
    activity_coords: list[tuple[float, float]],
    elevation_sampler: ElevationSampler | None,
) -> tuple[bool | None, str]:
    if segment.get("direction") != "ascent":
        return True, "not_ascent_only"
    if elevation_sampler is None:
        return None, "ascent_direction_needs_elevation_validation"
    start = tuple(segment["start"])
    end = tuple(segment["end"])
    start_elevation = elevation_sampler(start)
    end_elevation = elevation_sampler(end)
    if start_elevation is None or end_elevation is None or start_elevation == end_elevation:
        return None, "ascent_direction_unknown"
    start_index = nearest_activity_index(start, activity_coords)
    end_index = nearest_activity_index(end, activity_coords)
    if start_index is None or end_index is None:
        return None, "ascent_direction_activity_endpoint_missing"
    if end_elevation > start_elevation:
        return start_index < end_index, "official_geometry_start_to_end"
    return end_index < start_index, "official_geometry_end_to_start"


def review_segment(
    segment: dict[str, Any],
    activity_coords: list[tuple[float, float]],
    *,
    threshold_miles: float,
    endpoint_threshold_miles: float,
    min_fraction: float,
    partial_min_fraction: float,
    elevation_sampler: ElevationSampler | None = None,
) -> dict[str, Any]:
    samples = sample_line(segment["coordinates"])
    close_count = sum(
        1 for point in samples if point_to_polyline_distance_miles(point, activity_coords) <= threshold_miles
    )
    match_fraction = close_count / len(samples) if samples else 0.0
    start_distance = point_to_polyline_distance_miles(tuple(segment["start"]), activity_coords)
    end_distance = point_to_polyline_distance_miles(tuple(segment["end"]), activity_coords)
    endpoints_ok = start_distance <= endpoint_threshold_miles and end_distance <= endpoint_threshold_miles
    coverage_ok = match_fraction >= min_fraction and endpoints_ok
    direction_ok, direction_basis = ascent_direction_ok(segment, activity_coords, elevation_sampler)
    if coverage_ok and direction_ok is True:
        status = "completed"
    elif coverage_ok and direction_ok is False:
        status = "wrong_ascent_direction"
    elif coverage_ok:
        status = "direction_unverified"
    elif match_fraction >= partial_min_fraction or start_distance <= threshold_miles or end_distance <= threshold_miles:
        status = "partial"
    else:
        status = "not_matched"
    return {
        "seg_id": str(segment["seg_id"]),
        "seg_name": segment.get("seg_name"),
        "trail_name": segment.get("trail_name"),
        "direction": segment.get("direction"),
        "match_fraction": round(match_fraction, 3),
        "start_distance_miles": round(start_distance, 4),
        "end_distance_miles": round(end_distance, 4),
        "endpoints_ok": endpoints_ok,
        "direction_ok": direction_ok,
        "direction_basis": direction_basis,
        "completion_status": status,
    }


def review_activity_against_segments(
    activity_coords: list[tuple[float, float]],
    official_segments: list[dict[str, Any]],
    *,
    planned_segment_ids: list[Any] | tuple[Any, ...] | set[Any] | None = None,
    planned_outing_id: str | None = None,
    threshold_miles: float = 0.045,
    endpoint_threshold_miles: float | None = None,
    min_fraction: float = 0.85,
    partial_min_fraction: float = 0.2,
    elevation_sampler: ElevationSampler | None = None,
    evidence_refs: list[str] | None = None,
) -> dict[str, Any]:
    endpoint_threshold = endpoint_threshold_miles if endpoint_threshold_miles is not None else threshold_miles
    planned_ids = set(normalized_ids(planned_segment_ids))
    segment_reviews = [
        review_segment(
            segment,
            activity_coords,
            threshold_miles=threshold_miles,
            endpoint_threshold_miles=endpoint_threshold,
            min_fraction=min_fraction,
            partial_min_fraction=partial_min_fraction,
            elevation_sampler=elevation_sampler,
        )
        for segment in official_segments
    ]
    completed = {row["seg_id"] for row in segment_reviews if row["completion_status"] == "completed"}
    partial = {
        row["seg_id"]
        for row in segment_reviews
        if row["completion_status"] in {"partial", "wrong_ascent_direction", "direction_unverified"}
        and row["seg_id"] not in completed
    }
    missed = planned_ids - completed
    extra = completed - planned_ids if planned_ids else set()
    return {
        "schema": "boise_trails_activity_segment_review_v1",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "planned_outing_id": planned_outing_id,
        "planned_segment_ids": normalized_ids(planned_ids),
        "completed_segment_ids": normalized_ids(completed),
        "planned_completed_segment_ids": normalized_ids(completed & planned_ids),
        "extra_completed_segment_ids": normalized_ids(extra),
        "missed_segment_ids": normalized_ids(missed),
        "partial_segment_ids": normalized_ids(partial),
        "blocked_segment_ids": [],
        "evidence_refs": evidence_refs or [],
        "parameters": {
            "threshold_miles": threshold_miles,
            "endpoint_threshold_miles": endpoint_threshold,
            "min_fraction": min_fraction,
            "partial_min_fraction": partial_min_fraction,
        },
        "segment_reviews": [
            row for row in segment_reviews if row["completion_status"] != "not_matched" or row["seg_id"] in planned_ids
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activity", type=Path, required=True, help="Strava/BTC JSON or GPX activity file.")
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--planned-outing-id")
    parser.add_argument("--planned-segment-ids", help="Comma-separated planned official segment ids.")
    parser.add_argument("--threshold-miles", type=float, default=0.045)
    parser.add_argument("--endpoint-threshold-miles", type=float)
    parser.add_argument("--min-fraction", type=float, default=0.85)
    parser.add_argument("--partial-min-fraction", type=float, default=0.2)
    parser.add_argument("--dem-tif", type=Path, default=DEFAULT_DEM_TIF)
    parser.add_argument("--dem-summary-json", type=Path, default=DEFAULT_DEM_SUMMARY_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    coords = activity_coordinates(args.activity)
    official_segments, _ = load_official_segments(args.official_geojson)
    dem_context = load_dem_context(args.dem_tif, args.dem_summary_json)
    review = review_activity_against_segments(
        coords,
        official_segments,
        planned_segment_ids=parse_id_list(args.planned_segment_ids),
        planned_outing_id=args.planned_outing_id,
        threshold_miles=args.threshold_miles,
        endpoint_threshold_miles=args.endpoint_threshold_miles,
        min_fraction=args.min_fraction,
        partial_min_fraction=args.partial_min_fraction,
        elevation_sampler=dem_context.get("sampler"),
        evidence_refs=[str(args.activity)],
    )
    write_json(args.output_json, review)
    print(f"Wrote {args.output_json}")
    print(
        json.dumps(
            {
                "completed": len(review["completed_segment_ids"]),
                "extra": len(review["extra_completed_segment_ids"]),
                "missed": len(review["missed_segment_ids"]),
                "partial": len(review["partial_segment_ids"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
