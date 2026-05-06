#!/usr/bin/env python3
"""Export day-level GPX files for the relaxed-drive draft calendar plan."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from export_execution_gpx import (  # noqa: E402
    candidate_track_coordinates,
    load_official_segment_index,
    render_gpx_segments,
    slugify,
    validate_track_segments,
)
from personal_route_planner import (  # noqa: E402
    DEFAULT_CONNECTOR_GEOJSON,
    DEFAULT_OFFICIAL_GEOJSON,
    haversine_miles,
    load_connector_graph,
    load_official_segments,
    read_json,
    shortest_connector_path,
)
from p90_relaxed_drive_gpx_readiness_audit import (  # noqa: E402
    field_packet_gpx_index,
    forced_anchor_gpx_index,
    hybrid_index,
    personal_index,
)


DEFAULT_CALENDAR_JSON = YEAR_DIR / "checkpoints" / "p90-relaxed-drive-calendar-assignment-2026-05-06.json"
DEFAULT_PERSONAL_ROUTE_MENU_JSON = YEAR_DIR / "outputs" / "private" / "personal-route-menu.json"
DEFAULT_HYBRID_ROUTE_PASS_JSON = YEAR_DIR / "outputs" / "private" / "route-blocks" / "block-hybrid-route-pass-v1.json"
DEFAULT_FIELD_PACKET_MANIFEST_JSON = REPO_ROOT / "docs" / "field-packet" / "manifest.json"
DEFAULT_FORCED_ANCHOR_GPX_MANIFEST_JSON = YEAR_DIR / "checkpoints" / "p90-forced-anchor-gpx-export-2026-05-06.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "outputs" / "private" / "relaxed-drive-gpx" / "day-navigation"
DEFAULT_MANIFEST = YEAR_DIR / "checkpoints" / "p90-relaxed-drive-day-gpx-export-2026-05-06.json"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def parse_gpx_track_segments(path: Path) -> list[list[tuple[float, float]]]:
    root = ET.parse(path).getroot()
    segments = []
    for trkseg in root.findall(".//{*}trkseg"):
        coords = []
        for trkpt in trkseg.findall("{*}trkpt"):
            coords.append((float(trkpt.attrib["lon"]), float(trkpt.attrib["lat"])))
        if coords:
            segments.append(coords)
    return segments


def candidate_track(
    candidate: dict[str, Any],
    official_index: dict[int, dict[str, Any]],
    connector_graph: dict[str, Any],
    max_gap_miles: float = 0.05,
) -> list[list[tuple[float, float]]]:
    track = candidate_track_coordinates(
        candidate,
        official_index,
        connector_graph=connector_graph,
        densify_source_lines=True,
    )
    return [
        stitch_remaining_track_gaps(
            track,
            connector_graph=connector_graph,
            max_gap_miles=max_gap_miles,
        )
    ]


def append_deduped_coordinate(
    target: list[tuple[float, float]],
    coord: Any,
) -> None:
    point = (float(coord[0]), float(coord[1]))
    if target and target[-1] == point:
        return
    target.append(point)


def stitch_remaining_track_gaps(
    track: list[tuple[float, float]],
    *,
    connector_graph: dict[str, Any],
    max_gap_miles: float,
    stitch_snap_tolerance_miles: float = 0.2,
) -> list[tuple[float, float]]:
    if not track:
        return []
    stitched = [track[0]]
    for point in track[1:]:
        previous = stitched[-1]
        if haversine_miles(previous, point) > max_gap_miles:
            stitch = shortest_connector_path(
                previous,
                point,
                connector_graph,
                stitch_snap_tolerance_miles,
            )
            if stitch:
                for coord in stitch.get("path_coordinates") or []:
                    append_deduped_coordinate(stitched, coord)
        append_deduped_coordinate(stitched, point)
    return stitched


def component_fallback_tracks(
    candidate: dict[str, Any],
    *,
    personal_candidates: dict[str, dict[str, Any]],
    hybrid_candidates: dict[str, dict[str, Any]],
    official_index: dict[int, dict[str, Any]],
    connector_graph: dict[str, Any],
) -> list[list[tuple[float, float]]]:
    tracks = []
    for component_id in candidate.get("source_component_candidate_ids") or []:
        component = personal_candidates.get(str(component_id)) or hybrid_candidates.get(str(component_id))
        if not component:
            raise ValueError(f"Missing source component candidate: {component_id}")
        tracks.extend(candidate_track(component, official_index, connector_graph))
    if not tracks:
        raise ValueError(f"No source component candidates for {candidate.get('candidate_id')}")
    return tracks


def loop_track_segments(
    loop: dict[str, Any],
    *,
    personal_candidates: dict[str, dict[str, Any]],
    hybrid_candidates: dict[str, dict[str, Any]],
    field_packet_gpx: dict[str, dict[str, Any]],
    forced_anchor_gpx: dict[str, dict[str, Any]],
    official_index: dict[int, dict[str, Any]],
    connector_graph: dict[str, Any],
) -> tuple[list[list[tuple[float, float]]], str]:
    source = loop.get("source")
    candidate_id = str(loop.get("candidate_id") or "")
    if source == "personal_route_menu":
        return candidate_track(personal_candidates[candidate_id], official_index, connector_graph), "stored_personal_candidate"
    if source == "hybrid_candidate_index":
        candidate = hybrid_candidates[candidate_id]
        tracks = candidate_track(candidate, official_index, connector_graph)
        if validate_track_segments(tracks)["passed"]:
            return tracks, "stored_hybrid_candidate"
        return component_fallback_tracks(
            candidate,
            personal_candidates=personal_candidates,
            hybrid_candidates=hybrid_candidates,
            official_index=official_index,
            connector_graph=connector_graph,
        ), "hybrid_component_fallback"
    if source == "canonical_field_menu":
        path = REPO_ROOT / str((field_packet_gpx[candidate_id]["gpx_path"]))
        return parse_gpx_track_segments(path), "field_packet_navigation_gpx"
    if source == "forced_anchor_probe":
        path = REPO_ROOT / str((forced_anchor_gpx[str(loop["loop_id"])]["path"]))
        return parse_gpx_track_segments(path), "generated_forced_anchor_gpx"
    raise ValueError(f"Unsupported loop source: {source}")


def loop_endpoint_gap_miles(track_segments: list[list[tuple[float, float]]]) -> float | None:
    if not track_segments or not track_segments[0]:
        return None
    first = track_segments[0][0]
    last = track_segments[-1][-1]
    return round(haversine_miles(first, last), 4)


def validate_loop_track(
    track_segments: list[list[tuple[float, float]]],
    *,
    max_gap_miles: float,
    max_endpoint_gap_miles: float,
) -> dict[str, Any]:
    validation = validate_track_segments(track_segments, max_gap_miles=max_gap_miles)
    endpoint_gap = loop_endpoint_gap_miles(track_segments)
    failures = list(validation["failures"])
    if endpoint_gap is None:
        failures.append({"code": "missing_endpoint_gap"})
    elif endpoint_gap > max_endpoint_gap_miles:
        failures.append(
            {
                "code": "loop_endpoint_gap_exceeded",
                "endpoint_gap_miles": endpoint_gap,
                "max_allowed_endpoint_gap_miles": max_endpoint_gap_miles,
            }
        )
    return {
        **validation,
        "endpoint_gap_miles": endpoint_gap,
        "max_allowed_endpoint_gap_miles": max_endpoint_gap_miles,
        "passed": not failures,
        "failures": failures,
    }


def build_export_report(
    *,
    calendar: dict[str, Any],
    output_dir: Path,
    max_gap_miles: float,
    max_endpoint_gap_miles: float,
) -> dict[str, Any]:
    personal_candidates = personal_index(DEFAULT_PERSONAL_ROUTE_MENU_JSON)
    hybrid_candidates = hybrid_index(DEFAULT_HYBRID_ROUTE_PASS_JSON)
    field_packet_gpx = field_packet_gpx_index(DEFAULT_FIELD_PACKET_MANIFEST_JSON)
    forced_anchor_gpx = forced_anchor_gpx_index(DEFAULT_FORCED_ANCHOR_GPX_MANIFEST_JSON)
    official_index = load_official_segment_index(DEFAULT_OFFICIAL_GEOJSON)
    official_segments, _meta = load_official_segments(DEFAULT_OFFICIAL_GEOJSON)
    connector_graph = load_connector_graph(DEFAULT_CONNECTOR_GEOJSON, official_segments=official_segments)
    output_dir.mkdir(parents=True, exist_ok=True)
    day_rows = []
    for assignment in calendar.get("assignments") or []:
        day = assignment["field_day"]
        day_segments: list[list[tuple[float, float]]] = []
        loop_rows = []
        for loop in day.get("loops") or []:
            segments, source = loop_track_segments(
                loop,
                personal_candidates=personal_candidates,
                hybrid_candidates=hybrid_candidates,
                field_packet_gpx=field_packet_gpx,
                forced_anchor_gpx=forced_anchor_gpx,
                official_index=official_index,
                connector_graph=connector_graph,
            )
            validation = validate_loop_track(
                segments,
                max_gap_miles=max_gap_miles,
                max_endpoint_gap_miles=max_endpoint_gap_miles,
            )
            day_segments.extend(segments)
            loop_rows.append(
                {
                    "loop_id": loop["loop_id"],
                    "label": loop.get("label"),
                    "source": source,
                    "track_segment_count": len(segments),
                    "validation": validation,
                }
            )
        filename = f"{assignment['date']}-day-{day['draft_day_number']:02d}-{slugify(str(day['field_day_id']))[:80]}.gpx"
        path = output_dir / filename
        route_name = f"{assignment['date']} draft day {day['draft_day_number']}"
        path.write_text(render_gpx_segments(route_name, day_segments), encoding="utf-8")
        day_validation = validate_track_segments(day_segments, max_gap_miles=max_gap_miles)
        day_rows.append(
            {
                "date": assignment["date"],
                "draft_day_number": day["draft_day_number"],
                "field_day_id": day["field_day_id"],
                "path": display_path(path),
                "loop_count": len(loop_rows),
                "track_segment_count": len(day_segments),
                "loop_validations_passed": all(row["validation"]["passed"] for row in loop_rows),
                "day_track_validation": day_validation,
                "loops": loop_rows,
            }
        )
    return {
        "objective": "export day-level GPX files for the relaxed-drive calendar assignment",
        "source_files": {
            "calendar_assignment_json": display_path(DEFAULT_CALENDAR_JSON),
            "personal_route_menu_json": display_path(DEFAULT_PERSONAL_ROUTE_MENU_JSON),
            "hybrid_route_pass_json": display_path(DEFAULT_HYBRID_ROUTE_PASS_JSON),
            "field_packet_manifest_json": display_path(DEFAULT_FIELD_PACKET_MANIFEST_JSON),
            "forced_anchor_gpx_manifest_json": display_path(DEFAULT_FORCED_ANCHOR_GPX_MANIFEST_JSON),
        },
        "summary": {
            "day_gpx_count": len(day_rows),
            "loop_validation_passed": all(row["loop_validations_passed"] for row in day_rows),
            "day_track_validation_passed": all(row["day_track_validation"]["passed"] for row in day_rows),
            "failed_day_count": sum(1 for row in day_rows if not row["loop_validations_passed"] or not row["day_track_validation"]["passed"]),
            "max_gap_miles": max_gap_miles,
            "max_endpoint_gap_miles": max_endpoint_gap_miles,
        },
        "days": day_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calendar-json", type=Path, default=DEFAULT_CALENDAR_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--max-gap-miles", type=float, default=0.05)
    parser.add_argument("--max-endpoint-gap-miles", type=float, default=0.35)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_export_report(
        calendar=read_json(args.calendar_json),
        output_dir=args.output_dir,
        max_gap_miles=args.max_gap_miles,
        max_endpoint_gap_miles=args.max_endpoint_gap_miles,
    )
    write_json(args.manifest, report)
    print(f"Wrote {args.manifest}")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
