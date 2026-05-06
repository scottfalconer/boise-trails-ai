#!/usr/bin/env python3
"""Build a durable official-segment crosswalk to R2R and connector metadata."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from personal_route_planner import (  # noqa: E402
    DEFAULT_CONNECTOR_GEOJSON,
    DEFAULT_OFFICIAL_GEOJSON,
    DEFAULT_R2R_CONNECTOR_GEOJSON,
    MILES_PER_FOOT,
    center_point,
    clean_trail_name,
    connector_class_for_properties,
    flatten_coordinates,
    haversine_miles,
    iter_line_parts,
    normalize_name,
    read_json,
)


DEFAULT_OUTPUT_JSON = YEAR_DIR / "derived" / "segment-crosswalk" / "segment-crosswalk-2026-05-06.json"
DEFAULT_OUTPUT_CSV = YEAR_DIR / "derived" / "segment-crosswalk" / "segment-crosswalk-2026-05-06.csv"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def line_length_miles(coords: list[tuple[float, float]]) -> float:
    return sum(haversine_miles(left, right) for left, right in zip(coords, coords[1:]))


def official_segments(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    rows = []
    for feature in data.get("features", []):
        props = feature.get("properties") or {}
        coords = flatten_coordinates(feature.get("geometry") or {})
        if len(coords) < 2:
            continue
        seg_name = str(props.get("segName") or props["segId"])
        rows.append(
            {
                "seg_id": int(props["segId"]),
                "seg_name": seg_name,
                "trail_name": clean_trail_name(seg_name),
                "normalized_trail_name": normalize_name(clean_trail_name(seg_name)),
                "official_miles": float(props.get("LengthFt") or 0) * MILES_PER_FOOT,
                "direction": props.get("direction") or "both",
                "coords": coords,
                "center": center_point(coords),
            }
        )
    return rows


def load_line_features(path: Path, source_label: str) -> list[dict[str, Any]]:
    data = read_json(path)
    rows = []
    for feature in data.get("features", []):
        props = feature.get("properties") or {}
        raw_name = (
            props.get("TrailName")
            or props.get("Name")
            or props.get("SystemName")
            or props.get("OBJECTID")
            or "unnamed"
        )
        parts = [coords for coords in iter_line_parts(feature.get("geometry") or {}) if len(coords) >= 2]
        if not parts:
            continue
        rows.append(
            {
                "source_label": source_label,
                "name": str(raw_name),
                "normalized_name": normalize_name(str(raw_name)),
                "parts": parts,
                "properties": props,
                "center": center_point([point for part in parts for point in part]),
            }
        )
    return rows


def sample_coords(coords: list[tuple[float, float]], max_points: int = 18) -> list[tuple[float, float]]:
    if len(coords) <= max_points:
        return coords
    step = (len(coords) - 1) / (max_points - 1)
    return [coords[round(index * step)] for index in range(max_points)]


def min_sample_distance_miles(left: list[tuple[float, float]], right_parts: list[list[tuple[float, float]]]) -> float:
    left_samples = sample_coords(left)
    right_samples = [point for part in right_parts for point in sample_coords(part)]
    return min(haversine_miles(left_point, right_point) for left_point in left_samples for right_point in right_samples)


def name_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if left in right or right in left:
        return 0.9
    return SequenceMatcher(None, left, right).ratio()


def best_match(segment: dict[str, Any], features: list[dict[str, Any]], max_center_miles: float) -> dict[str, Any] | None:
    best = None
    for feature in features:
        center_distance = haversine_miles(segment["center"], feature["center"])
        if center_distance > max_center_miles:
            continue
        distance = min_sample_distance_miles(segment["coords"], feature["parts"])
        similarity = name_similarity(segment["normalized_trail_name"], feature["normalized_name"])
        score = distance + max(0.0, 0.08 - similarity * 0.08)
        if best is None or score < best["score"]:
            best = {
                "feature": feature,
                "distance_miles": distance,
                "center_distance_miles": center_distance,
                "name_similarity": similarity,
                "score": score,
            }
    return best


def match_confidence(distance: float, similarity: float) -> str:
    if distance <= 0.02 and similarity >= 0.75:
        return "high"
    if distance <= 0.05 and similarity >= 0.45:
        return "medium"
    if distance <= 0.1:
        return "low"
    return "review"


def normalize_r2r_match(match: dict[str, Any] | None) -> dict[str, Any] | None:
    if not match:
        return None
    props = match["feature"]["properties"]
    return {
        "trail_id": props.get("TrailID"),
        "trail_name": props.get("TrailName") or props.get("Name") or match["feature"]["name"],
        "system_name": props.get("SystemName"),
        "trail_subsystem": props.get("TrailSubSystem"),
        "trail_status": props.get("TrailStatus"),
        "condition": props.get("Condition"),
        "condition_date": props.get("ConditionDate"),
        "condition_notes": props.get("ConditionNotes"),
        "surface": props.get("TrlSurface"),
        "rating": props.get("Rating"),
        "motorized": props.get("Motorized"),
        "special_management": props.get("SpecialManagement"),
        "special_management_comment": props.get("SpecialManagementComment"),
        "all_weather": props.get("AllWeather"),
        "distance_miles": round(match["distance_miles"], 4),
        "name_similarity": round(match["name_similarity"], 3),
        "confidence": match_confidence(match["distance_miles"], match["name_similarity"]),
    }


def normalize_connector_match(match: dict[str, Any] | None) -> dict[str, Any] | None:
    if not match:
        return None
    props = match["feature"]["properties"]
    connector_class = props.get("connector_class") or connector_class_for_properties(props, "connector")
    return {
        "name": props.get("TrailName") or props.get("Name") or match["feature"]["name"],
        "source": props.get("source"),
        "highway": props.get("highway"),
        "surface": props.get("surface"),
        "access": props.get("access"),
        "foot": props.get("foot"),
        "connector_class": connector_class,
        "distance_miles": round(match["distance_miles"], 4),
        "name_similarity": round(match["name_similarity"], 3),
        "confidence": match_confidence(match["distance_miles"], match["name_similarity"]),
    }


def build_crosswalk(
    official_geojson: Path,
    r2r_geojson: Path,
    connector_geojson: Path,
) -> dict[str, Any]:
    official_rows = official_segments(official_geojson)
    r2r_features = load_line_features(r2r_geojson, "r2r")
    connector_features = load_line_features(connector_geojson, "connector")
    rows = []
    for segment in official_rows:
        r2r_match = normalize_r2r_match(best_match(segment, r2r_features, max_center_miles=2.0))
        connector_match = normalize_connector_match(best_match(segment, connector_features, max_center_miles=2.0))
        rows.append(
            {
                "seg_id": segment["seg_id"],
                "seg_name": segment["seg_name"],
                "trail_name": segment["trail_name"],
                "official_miles": round(segment["official_miles"], 4),
                "direction": segment["direction"],
                "r2r": r2r_match,
                "connector": connector_match,
                "review_required": not r2r_match or r2r_match["confidence"] in {"low", "review"},
            }
        )
    confidence_counts = Counter((row.get("r2r") or {}).get("confidence") or "missing" for row in rows)
    connector_class_counts = Counter((row.get("connector") or {}).get("connector_class") or "missing" for row in rows)
    return {
        "dataset": "segment-crosswalk-2026-05-06",
        "source_datasets": {
            "official_geojson": display_path(official_geojson),
            "r2r_geojson": display_path(r2r_geojson),
            "connector_geojson": display_path(connector_geojson),
        },
        "summary": {
            "segment_count": len(rows),
            "r2r_confidence_counts": dict(sorted(confidence_counts.items())),
            "connector_class_counts": dict(sorted(connector_class_counts.items())),
            "review_required_count": sum(1 for row in rows if row["review_required"]),
        },
        "rows": rows,
    }


def write_csv(path: Path, crosswalk: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "seg_id",
        "seg_name",
        "trail_name",
        "official_miles",
        "direction",
        "r2r_trail_id",
        "r2r_trail_name",
        "r2r_confidence",
        "r2r_distance_miles",
        "r2r_condition",
        "r2r_surface",
        "r2r_rating",
        "r2r_special_management",
        "connector_name",
        "connector_class",
        "connector_source",
        "connector_highway",
        "connector_distance_miles",
        "review_required",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in crosswalk["rows"]:
            r2r = row.get("r2r") or {}
            connector = row.get("connector") or {}
            writer.writerow(
                {
                    "seg_id": row["seg_id"],
                    "seg_name": row["seg_name"],
                    "trail_name": row["trail_name"],
                    "official_miles": row["official_miles"],
                    "direction": row["direction"],
                    "r2r_trail_id": r2r.get("trail_id"),
                    "r2r_trail_name": r2r.get("trail_name"),
                    "r2r_confidence": r2r.get("confidence"),
                    "r2r_distance_miles": r2r.get("distance_miles"),
                    "r2r_condition": r2r.get("condition"),
                    "r2r_surface": r2r.get("surface"),
                    "r2r_rating": r2r.get("rating"),
                    "r2r_special_management": r2r.get("special_management"),
                    "connector_name": connector.get("name"),
                    "connector_class": connector.get("connector_class"),
                    "connector_source": connector.get("source"),
                    "connector_highway": connector.get("highway"),
                    "connector_distance_miles": connector.get("distance_miles"),
                    "review_required": row["review_required"],
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--r2r-geojson", type=Path, default=DEFAULT_R2R_CONNECTOR_GEOJSON)
    parser.add_argument("--connector-geojson", type=Path, default=DEFAULT_CONNECTOR_GEOJSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    crosswalk = build_crosswalk(args.official_geojson, args.r2r_geojson, args.connector_geojson)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(crosswalk, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_csv, crosswalk)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")
    print(json.dumps(crosswalk["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
