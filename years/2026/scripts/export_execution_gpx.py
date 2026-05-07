#!/usr/bin/env python3
"""Export simulated executable on-foot recommendations to GPX tracks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402
from personal_route_planner import (  # noqa: E402
    DEFAULT_OFFICIAL_GEOJSON,
    haversine_miles,
    iter_line_parts,
    load_connector_graph,
    load_official_segments,
    read_json,
    shortest_connector_path,
)


DEFAULT_PLAN_JSON = YEAR_DIR / "outputs" / "personal-route-menu.json"
DEFAULT_EXECUTION_JSON = (
    YEAR_DIR
    / "experiments"
    / "2026-05-04-outing-execution-simulation"
    / "outing_execution.json"
)
DEFAULT_OUTPUT_DIR = YEAR_DIR / "outputs" / "gpx" / "executable-routes-2026-05-04"


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "route"


def load_official_segment_index(path: Path) -> dict[int, dict[str, Any]]:
    data = read_json(path)
    index = {}
    for feature in data.get("features", []):
        props = feature.get("properties") or {}
        seg_id = int(props["segId"])
        geometry = feature.get("geometry") or {}
        parts = iter_line_parts(geometry)
        if geometry.get("type") == "MultiLineString" and len(parts) > 1:
            raise ValueError(
                f"Official segment {seg_id} is a MultiLineString; "
                "normalize multipart official geometry before GPX export"
            )
        index[seg_id] = {
            "seg_id": seg_id,
            "seg_name": props.get("segName"),
            "trail_name": re.sub(r"\s+\d+$", "", str(props.get("segName") or "")),
            "direction": props.get("direction") or "both",
            "coordinates": parts[0] if parts else [],
        }
    return index


def load_candidate_index(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(candidate["candidate_id"]): candidate
        for candidate in plan["route_menu"]["all_candidates"]
    }


def dedupe_append(
    target: list[tuple[float, float]],
    coords: list[tuple[float, float]],
) -> None:
    for coord in coords:
        point = (float(coord[0]), float(coord[1]))
        if target and target[-1] == point:
            continue
        target.append(point)


def densify_coordinates(
    coords: list[tuple[float, float]],
    max_gap_miles: float = 0.03,
) -> list[tuple[float, float]]:
    if len(coords) < 2:
        return list(coords)
    dense = [coords[0]]
    for left, right in zip(coords, coords[1:]):
        gap = haversine_miles(left, right)
        steps = max(1, int((gap / max_gap_miles) + 0.999999))
        for step in range(1, steps + 1):
            fraction = step / steps
            dense.append(
                (
                    left[0] + (right[0] - left[0]) * fraction,
                    left[1] + (right[1] - left[1]) * fraction,
                )
            )
    return dense


def candidate_segment_coordinates(
    candidate: dict[str, Any],
    segment: dict[str, Any],
    official_index: dict[int, dict[str, Any]],
    densify_source_lines: bool = False,
    densify_max_gap_miles: float = 0.03,
) -> list[tuple[float, float]]:
    seg_id = int(segment["seg_id"])
    official = official_index[seg_id]
    coords = list(official["coordinates"])
    planned_directions = (
        (candidate.get("direction_validation") or {}).get("planned_traversal_direction")
        or {}
    )
    if planned_directions.get(str(seg_id)) == "official_geometry_end_to_start":
        coords = list(reversed(coords))
    elif (
        (candidate.get("route_orientation") or {}).get("direction") == "reversed"
        and official.get("direction") == "both"
    ):
        coords = list(reversed(coords))
    if densify_source_lines:
        return densify_coordinates(coords, max_gap_miles=densify_max_gap_miles)
    return coords


def path_coordinate_tuples(
    raw_coords: list[Any],
    densify_source_lines: bool = False,
    densify_max_gap_miles: float = 0.03,
) -> list[tuple[float, float]]:
    coords = [(float(coord[0]), float(coord[1])) for coord in raw_coords or []]
    if densify_source_lines:
        return densify_coordinates(coords, max_gap_miles=densify_max_gap_miles)
    return coords


def candidate_segments_for_track(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    segments = list(candidate.get("segments") or [])
    if (
        (candidate.get("route_orientation") or {}).get("direction") == "reversed"
        and not candidate.get("custom_traversal_order")
    ):
        return list(reversed(segments))
    return segments


def raise_unstitched_gap(candidate: dict[str, Any], from_point: tuple[float, float], to_point: tuple[float, float], gap: float) -> None:
    raise ValueError(
        f"Candidate {candidate.get('candidate_id')} has an unstitched source gap "
        f"of {gap:.4f} mi from {from_point} to {to_point}"
    )


def candidate_track_coordinates(
    candidate: dict[str, Any],
    official_index: dict[int, dict[str, Any]],
    connector_graph: dict[str, Any] | None = None,
    stitch_gap_threshold_miles: float = 0.05,
    stitch_snap_tolerance_miles: float = 0.02,
    densify_source_lines: bool = False,
    densify_max_gap_miles: float = 0.03,
    fail_on_unstitched_gap: bool = False,
) -> list[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    trailhead_access = candidate.get("trailhead_access") or {}
    dedupe_append(
        coords,
        path_coordinate_tuples(
            trailhead_access.get("outbound_path_coordinates") or [],
            densify_source_lines=densify_source_lines,
            densify_max_gap_miles=densify_max_gap_miles,
        ),
    )
    between_links = list(((candidate.get("between_trail_links") or {}).get("links")) or [])
    next_link_index = 0
    previous_trail = None
    for segment in candidate_segments_for_track(candidate):
        trail_name = segment.get("trail_name")
        if previous_trail is not None and trail_name != previous_trail:
            if next_link_index < len(between_links):
                dedupe_append(
                    coords,
                    path_coordinate_tuples(
                        between_links[next_link_index].get("path_coordinates") or [],
                        densify_source_lines=densify_source_lines,
                        densify_max_gap_miles=densify_max_gap_miles,
                    ),
                )
                next_link_index += 1
        segment_coords = candidate_segment_coordinates(
            candidate,
            segment,
            official_index,
            densify_source_lines=densify_source_lines,
            densify_max_gap_miles=densify_max_gap_miles,
        )
        if coords and segment_coords and haversine_miles(coords[-1], segment_coords[0]) > stitch_gap_threshold_miles:
            gap = haversine_miles(coords[-1], segment_coords[0])
            stitch = shortest_connector_path(
                coords[-1],
                segment_coords[0],
                connector_graph,
                stitch_snap_tolerance_miles,
            )
            if stitch:
                dedupe_append(
                    coords,
                    path_coordinate_tuples(
                        stitch.get("path_coordinates") or [],
                        densify_source_lines=densify_source_lines,
                        densify_max_gap_miles=densify_max_gap_miles,
                    ),
                )
            elif fail_on_unstitched_gap:
                raise_unstitched_gap(candidate, coords[-1], segment_coords[0], gap)
        dedupe_append(coords, segment_coords)
        previous_trail = trail_name

    return_path = path_coordinate_tuples(
        (candidate.get("return_to_car") or {}).get("path_coordinates") or [],
        densify_source_lines=densify_source_lines,
        densify_max_gap_miles=densify_max_gap_miles,
    )
    trailhead_return_path = path_coordinate_tuples(
        trailhead_access.get("return_path_coordinates") or [],
        densify_source_lines=densify_source_lines,
        densify_max_gap_miles=densify_max_gap_miles,
    )
    if coords and return_path:
        current = coords[-1]
        already_at_trailhead_return = (
            bool(trailhead_return_path)
            and haversine_miles(current, trailhead_return_path[0]) <= stitch_gap_threshold_miles
        )
        if already_at_trailhead_return:
            return_path = []
        elif haversine_miles(current, return_path[0]) > stitch_gap_threshold_miles:
            if haversine_miles(current, return_path[-1]) <= stitch_gap_threshold_miles:
                return_path = list(reversed(return_path))
            else:
                gap = haversine_miles(current, return_path[0])
                stitch = shortest_connector_path(
                    current,
                    return_path[0],
                    connector_graph,
                    stitch_snap_tolerance_miles,
                )
                if stitch:
                    dedupe_append(
                        coords,
                        path_coordinate_tuples(
                            stitch.get("path_coordinates") or [],
                            densify_source_lines=densify_source_lines,
                            densify_max_gap_miles=densify_max_gap_miles,
                        ),
                    )
                elif fail_on_unstitched_gap:
                    raise_unstitched_gap(candidate, current, return_path[0], gap)
    dedupe_append(coords, return_path)
    if coords and trailhead_return_path and haversine_miles(coords[-1], trailhead_return_path[0]) > stitch_gap_threshold_miles:
        if haversine_miles(coords[-1], trailhead_return_path[-1]) <= stitch_gap_threshold_miles:
            trailhead_return_path = list(reversed(trailhead_return_path))
        else:
            gap = haversine_miles(coords[-1], trailhead_return_path[0])
            stitch = shortest_connector_path(
                coords[-1],
                trailhead_return_path[0],
                connector_graph,
                stitch_snap_tolerance_miles,
            )
            if stitch:
                dedupe_append(
                    coords,
                    path_coordinate_tuples(
                        stitch.get("path_coordinates") or [],
                        densify_source_lines=densify_source_lines,
                        densify_max_gap_miles=densify_max_gap_miles,
                    ),
                )
            elif fail_on_unstitched_gap:
                raise_unstitched_gap(candidate, coords[-1], trailhead_return_path[0], gap)
    dedupe_append(coords, trailhead_return_path)
    return coords


def recommendation_candidate_ids(recommendation: dict[str, Any]) -> list[str]:
    source = recommendation.get("source") or recommendation
    if source.get("outing_ids"):
        return [str(candidate_id) for candidate_id in source["outing_ids"]]
    if recommendation.get("id"):
        return [str(recommendation["id"])]
    if source.get("candidate_id"):
        return [str(source["candidate_id"])]
    return []


def recommendation_track_coordinates(
    recommendation: dict[str, Any],
    candidate_index: dict[str, dict[str, Any]],
    official_index: dict[int, dict[str, Any]],
    connector_graph: dict[str, Any] | None = None,
    densify_source_lines: bool = False,
) -> list[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    for candidate_id in recommendation_candidate_ids(recommendation):
        candidate = candidate_index[candidate_id]
        dedupe_append(
            coords,
            candidate_track_coordinates(
                candidate,
                official_index,
                connector_graph=connector_graph,
                densify_source_lines=densify_source_lines,
            ),
        )
    return coords


def recommendation_track_segments(
    recommendation: dict[str, Any],
    candidate_index: dict[str, dict[str, Any]],
    official_index: dict[int, dict[str, Any]],
    connector_graph: dict[str, Any] | None = None,
    densify_source_lines: bool = False,
) -> list[list[tuple[float, float]]]:
    return [
        candidate_track_coordinates(
            candidate_index[candidate_id],
            official_index,
            connector_graph=connector_graph,
            densify_source_lines=densify_source_lines,
        )
        for candidate_id in recommendation_candidate_ids(recommendation)
    ]


def render_gpx_segments(name: str, track_segments: list[list[tuple[float, float]]]) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="boise-trails-ai" xmlns="http://www.topografix.com/GPX/1/1">',
        "  <trk>",
        f"    <name>{escape(name)}</name>",
    ]
    for coords in track_segments:
        lines.append("    <trkseg>")
        for lon, lat in coords:
            lines.append(f'      <trkpt lat="{lat:.6f}" lon="{lon:.6f}" />')
        lines.append("    </trkseg>")
    lines.extend(
        [
            "  </trk>",
            "</gpx>",
            "",
        ]
    )
    return "\n".join(lines)


def render_gpx(name: str, coords: list[tuple[float, float]]) -> str:
    return render_gpx_segments(name, [coords])


def validate_track_segments(
    track_segments: list[list[tuple[float, float]]],
    max_gap_miles: float = 0.05,
) -> dict[str, Any]:
    failures = []
    max_gap = 0.0
    point_count = 0
    for segment_index, coords in enumerate(track_segments):
        point_count += len(coords)
        if len(coords) < 2:
            failures.append(
                {
                    "code": "track_segment_too_short",
                    "track_segment_index": segment_index,
                    "point_count": len(coords),
                }
            )
            continue
        for point_index, (left, right) in enumerate(zip(coords, coords[1:])):
            gap = haversine_miles(left, right)
            max_gap = max(max_gap, gap)
            if gap > max_gap_miles:
                failures.append(
                    {
                        "code": "max_trackpoint_gap_exceeded",
                        "track_segment_index": segment_index,
                        "point_index": point_index,
                        "gap_miles": round(gap, 4),
                        "max_allowed_gap_miles": max_gap_miles,
                    }
                )
    if not track_segments:
        failures.append({"code": "no_track_segments"})
    return {
        "passed": not failures,
        "track_segment_count": len(track_segments),
        "point_count": point_count,
        "max_trackpoint_gap_miles": round(max_gap, 4),
        "max_allowed_gap_miles": max_gap_miles,
        "failures": failures,
    }


def selected_menu(execution: dict[str, Any], menu_name: str) -> dict[str, Any]:
    if menu_name == "single-car":
        return execution["single_car_menu"]["recommended_by_bucket"]
    if menu_name == "best-executable":
        return execution["best_executable_menu"]["recommended_by_bucket"]
    raise ValueError(f"Unsupported menu: {menu_name}")


def export_recommendation_gpx(
    plan: dict[str, Any],
    execution: dict[str, Any],
    official_index: dict[int, dict[str, Any]],
    output_dir: Path,
    menu_name: str,
    max_gap_miles: float = 0.05,
    connector_graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_index = load_candidate_index(plan)
    routes = []
    for bucket, recommendation in selected_menu(execution, menu_name).items():
        track_segments = recommendation_track_segments(
            recommendation,
            candidate_index,
            official_index,
            connector_graph=connector_graph,
            densify_source_lines=True,
        )
        point_count = sum(len(coords) for coords in track_segments)
        validation = validate_track_segments(track_segments, max_gap_miles=max_gap_miles)
        route_name = f"{bucket}: {', '.join(recommendation.get('trail_names') or [])}"
        filename = f"{bucket}-{slugify(str(recommendation.get('id') or route_name))}.gpx"
        output_path = output_dir / filename
        output_path.write_text(render_gpx_segments(route_name, track_segments), encoding="utf-8")
        routes.append(
            {
                "bucket": bucket,
                "recommendation_type": recommendation.get("recommendation_type"),
                "id": recommendation.get("id"),
                "trail_names": recommendation.get("trail_names"),
                "official_new_miles": recommendation.get("official_new_miles"),
                "simulated_total_minutes": recommendation.get("simulated_total_minutes"),
                "track_segment_count": len(track_segments),
                "point_count": point_count,
                "path": str(output_path),
                "gpx_validation": validation,
            }
        )
    return {
        "menu": menu_name,
        "summary": {
            "route_count": len(routes),
            "gpx_validation_passed": all(route["gpx_validation"]["passed"] for route in routes),
            "failed_route_count": len([route for route in routes if not route["gpx_validation"]["passed"]]),
            "max_allowed_gap_miles": max_gap_miles,
        },
        "routes": routes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", type=Path, default=DEFAULT_PLAN_JSON)
    parser.add_argument("--execution-json", type=Path, default=DEFAULT_EXECUTION_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--menu", choices=["single-car", "best-executable"], default="single-car")
    parser.add_argument("--max-gap-miles", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = read_json(args.plan_json)
    execution = read_json(args.execution_json)
    official_index = load_official_segment_index(args.official_geojson)
    official_segments, _official_meta = load_official_segments(args.official_geojson)
    connector_meta = ((plan.get("source_datasets") or {}).get("connector_geojson") or {})
    connector_path = Path(str(connector_meta.get("path"))) if connector_meta.get("path") else None
    connector_graph = (
        load_connector_graph(connector_path, official_segments=official_segments)
        if connector_path and connector_path.exists()
        else None
    )
    manifest = export_recommendation_gpx(
        plan,
        execution,
        official_index,
        args.output_dir,
        args.menu,
        max_gap_miles=args.max_gap_miles,
        connector_graph=connector_graph,
    )
    manifest_path = args.output_dir / f"{args.menu}-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    artifact_manifest_path = args.output_dir / f"{args.menu}-artifact-manifest.json"
    write_manifest(
        artifact_manifest_path,
        build_artifact_manifest(
            run_id=str(plan.get("run_id") or execution.get("run_id") or args.menu),
            inputs=[args.plan_json, args.execution_json, args.official_geojson],
            outputs=[manifest_path] + [Path(route["path"]) for route in manifest["routes"]],
            command="export_execution_gpx.py",
            metadata={"menu": args.menu},
        ),
    )
    print(f"Wrote {manifest['summary']['route_count']} GPX files to {args.output_dir}")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {artifact_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
