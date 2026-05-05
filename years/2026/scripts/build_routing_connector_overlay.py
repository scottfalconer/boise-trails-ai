#!/usr/bin/env python3
"""Build a combined R2R + OSM connector layer for route validation."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from personal_route_planner import (
    DEFAULT_OFFICIAL_GEOJSON,
    DEFAULT_R2R_CONNECTOR_GEOJSON,
    YEAR_DIR,
    connector_class_for_properties,
    flatten_coordinates,
    iter_line_parts,
    read_json,
    write_json,
)


DEFAULT_OSM_PBF = YEAR_DIR / "inputs" / "open-data" / "osm-2026-05-04" / "boise_planning_bbox.osm.pbf"
DEFAULT_OUTPUT = (
    YEAR_DIR
    / "inputs"
    / "open-data"
    / "routing-connectors-2026-05-04"
    / "combined_r2r_osm_connectors.geojson"
)

ALLOWED_HIGHWAYS = {
    "cycleway",
    "footway",
    "living_street",
    "path",
    "pedestrian",
    "primary",
    "residential",
    "secondary",
    "service",
    "steps",
    "tertiary",
    "track",
    "unclassified",
}

BLOCKED_ACCESS_VALUES = {"no", "private"}


def coordinate_bbox(coords: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    lons = [point[0] for point in coords]
    lats = [point[1] for point in coords]
    return min(lons), min(lats), max(lons), max(lats)


def expanded_bbox(
    coords: list[tuple[float, float]],
    pad_degrees: float,
) -> tuple[float, float, float, float]:
    left, bottom, right, top = coordinate_bbox(coords)
    return (
        left - pad_degrees,
        bottom - pad_degrees,
        right + pad_degrees,
        top + pad_degrees,
    )


def bboxes_overlap(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    return not (
        left[2] < right[0]
        or right[2] < left[0]
        or left[3] < right[1]
        or right[3] < left[1]
    )


def official_search_bboxes(path: Path, pad_degrees: float) -> list[tuple[float, float, float, float]]:
    data = read_json(path)
    bboxes = []
    for feature in data.get("features", []):
        coords = flatten_coordinates(feature.get("geometry") or {})
        if len(coords) >= 2:
            bboxes.append(expanded_bbox(coords, pad_degrees))
    return bboxes


def osm_feature_is_usable(feature: dict[str, Any]) -> bool:
    props = feature.get("properties") or {}
    highway = props.get("highway")
    if highway not in ALLOWED_HIGHWAYS:
        return False
    access = str(props.get("access") or "").lower()
    foot = str(props.get("foot") or "").lower()
    if access in BLOCKED_ACCESS_VALUES or foot in BLOCKED_ACCESS_VALUES:
        return False
    return any(len(coords) >= 2 for coords in iter_line_parts(feature.get("geometry") or {}))


def osm_feature_near_official(
    feature: dict[str, Any],
    bboxes: list[tuple[float, float, float, float]],
) -> bool:
    coords = flatten_coordinates(feature.get("geometry") or {})
    if len(coords) < 2:
        return False
    bbox = coordinate_bbox(coords)
    return any(bboxes_overlap(bbox, official_bbox) for official_bbox in bboxes)


def export_osm_highways(osm_pbf: Path, output_geojson: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        highways_pbf = Path(tmp) / "highways.osm.pbf"
        subprocess.run(
            [
                "osmium",
                "tags-filter",
                str(osm_pbf),
                "w/highway",
                "-o",
                str(highways_pbf),
                "--overwrite",
            ],
            check=True,
        )
        subprocess.run(
            [
                "osmium",
                "export",
                str(highways_pbf),
                "--geometry-types=linestring",
                "-o",
                str(output_geojson),
                "--overwrite",
            ],
            check=True,
        )


def normalize_osm_feature(feature: dict[str, Any], index: int) -> dict[str, Any]:
    props = feature.get("properties") or {}
    highway = props.get("highway")
    name = props.get("name") or f"OSM {highway} connector {index}"
    return {
        "type": "Feature",
        "geometry": feature.get("geometry"),
        "properties": {
            "TrailName": str(name),
            "Name": str(name),
            "source": "openstreetmap",
            "highway": highway,
            "connector_class": connector_class_for_properties(
                {"source": "openstreetmap", "highway": highway},
                "connector",
            ),
            "surface": props.get("surface"),
            "access": props.get("access"),
            "foot": props.get("foot"),
            "bicycle": props.get("bicycle"),
        },
    }


def build_combined_connector_overlay(
    r2r_geojson: Path,
    osm_pbf: Path,
    official_geojson: Path,
    output_geojson: Path,
    pad_degrees: float = 0.01,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        osm_export = Path(tmp) / "osm_highways.geojson"
        export_osm_highways(osm_pbf, osm_export)
        osm_data = read_json(osm_export)

    official_bboxes = official_search_bboxes(official_geojson, pad_degrees)
    osm_features = []
    for index, feature in enumerate(osm_data.get("features", []), start=1):
        if not osm_feature_is_usable(feature):
            continue
        if not osm_feature_near_official(feature, official_bboxes):
            continue
        osm_features.append(normalize_osm_feature(feature, index))

    r2r_data = read_json(r2r_geojson)
    r2r_features = []
    for feature in r2r_data.get("features", []):
        copied = dict(feature)
        props = dict(copied.get("properties") or {})
        props.setdefault("source", "ridge_to_rivers_open_data")
        props.setdefault("connector_class", connector_class_for_properties(props, "connector"))
        copied["properties"] = props
        r2r_features.append(copied)

    combined = {
        "type": "FeatureCollection",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_datasets": {
            "r2r_geojson": str(r2r_geojson),
            "osm_pbf": str(osm_pbf),
            "official_geojson": str(official_geojson),
            "official_bbox_pad_degrees": pad_degrees,
        },
        "summary": {
            "r2r_features": len(r2r_features),
            "osm_features_kept": len(osm_features),
            "features": len(r2r_features) + len(osm_features),
            "allowed_highways": sorted(ALLOWED_HIGHWAYS),
        },
        "features": r2r_features + osm_features,
    }
    write_json(output_geojson, combined)
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r2r-geojson", type=Path, default=DEFAULT_R2R_CONNECTOR_GEOJSON)
    parser.add_argument("--osm-pbf", type=Path, default=DEFAULT_OSM_PBF)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--output-geojson", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pad-degrees", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    combined = build_combined_connector_overlay(
        r2r_geojson=args.r2r_geojson,
        osm_pbf=args.osm_pbf,
        official_geojson=args.official_geojson,
        output_geojson=args.output_geojson,
        pad_degrees=args.pad_degrees,
    )
    print(f"Wrote {args.output_geojson}")
    print(
        "Combined connectors: "
        f"{combined['summary']['r2r_features']} R2R + "
        f"{combined['summary']['osm_features_kept']} OSM = "
        f"{combined['summary']['features']} features"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
