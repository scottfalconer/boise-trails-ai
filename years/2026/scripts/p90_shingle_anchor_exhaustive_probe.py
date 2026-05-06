#!/usr/bin/env python3
"""Exhaustively probe Shingle Creek 1656 against every known parking anchor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from export_execution_gpx import load_official_segment_index  # noqa: E402
from p90_forced_anchor_probe import (  # noqa: E402
    DEFAULT_MANUAL_DESIGN_JSONS,
    anchor_distance_to_segment,
    anchor_field_ready,
    forced_anchor_state,
    load_all_anchors,
    parking_risk_score,
)
from p90_segment_split_probe import probe_candidate_row, segment_trail  # noqa: E402
from personal_route_planner import (  # noqa: E402
    DEFAULT_ACTIVITY_DETAIL_SUMMARY_CSV,
    DEFAULT_ACTIVITY_SUMMARY_CSV,
    DEFAULT_CONNECTOR_GEOJSON,
    DEFAULT_DEM_SUMMARY_JSON,
    DEFAULT_DEM_TIF,
    DEFAULT_OFFICIAL_GEOJSON,
    DEFAULT_PRIVATE_PARKING_ANCHORS_GEOJSON,
    DEFAULT_SEGMENT_PERF_CSV,
    DEFAULT_STRAVA_DETAILS_DIR,
    DEFAULT_TRAILHEAD_CANDIDATES_GEOJSON,
    build_performance_profile,
    candidate_from_trail_group,
    load_connector_graph,
    load_dem_context,
    load_official_segments,
    read_json,
)


DEFAULT_STATE_JSON = YEAR_DIR / "inputs" / "personal" / "2026-planner-state.private.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "checkpoints"
DEFAULT_BASENAME = "p90-shingle-1656-anchor-exhaustive-probe-2026-05-06"
DEFAULT_SEGMENT_ID = 1656


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def row_is_graph_track_valid(row: dict[str, Any]) -> bool:
    return row.get("route_status") == "graph_validated" and row.get("track_validation_passed") is True


def row_is_strict_success(row: dict[str, Any], p90_bound_minutes: int) -> bool:
    return (
        row_is_graph_track_valid(row)
        and row.get("field_ready") is True
        and int(row.get("door_to_door_p90_minutes") or 999999) <= p90_bound_minutes
    )


def row_rank(row: dict[str, Any]) -> tuple[int, int, float, int, str]:
    return (
        int(row.get("door_to_door_p90_minutes") or 999999),
        int(row.get("door_to_door_p75_minutes") or 999999),
        float(row.get("on_foot_miles") or 999999),
        int(row.get("parking_risk") or 999),
        str(row.get("anchor_name") or row.get("trailhead") or ""),
    )


def best_row(rows: list[dict[str, Any]], *, require_field_ready: bool = False) -> dict[str, Any] | None:
    eligible = [
        row
        for row in rows
        if row_is_graph_track_valid(row) and (not require_field_ready or row.get("field_ready") is True)
    ]
    if not eligible:
        return None
    return min(eligible, key=row_rank)


def summarize_rows(rows: list[dict[str, Any]], *, p90_bound_minutes: int) -> dict[str, Any]:
    best_any = best_row(rows)
    best_ready = best_row(rows, require_field_ready=True)
    strict_successes = [row for row in rows if row_is_strict_success(row, p90_bound_minutes)]
    under_bound_graph_track = [
        row
        for row in rows
        if row_is_graph_track_valid(row) and int(row.get("door_to_door_p90_minutes") or 999999) <= p90_bound_minutes
    ]
    return {
        "anchor_count": len(rows),
        "graph_validated_count": sum(1 for row in rows if row.get("route_status") == "graph_validated"),
        "track_validation_passed_count": sum(1 for row in rows if row.get("track_validation_passed") is True),
        "field_ready_count": sum(1 for row in rows if row.get("field_ready") is True),
        "p90_bound_minutes": p90_bound_minutes,
        "under_bound_graph_track_count": len(under_bound_graph_track),
        "strict_success_count": len(strict_successes),
        "strict_success": bool(strict_successes),
        "best_any_anchor": best_any.get("anchor_name") if best_any else None,
        "best_any_p90_minutes": best_any.get("door_to_door_p90_minutes") if best_any else None,
        "best_any_p75_minutes": best_any.get("door_to_door_p75_minutes") if best_any else None,
        "best_any_on_foot_miles": best_any.get("on_foot_miles") if best_any else None,
        "best_field_ready_anchor": best_ready.get("anchor_name") if best_ready else None,
        "best_field_ready_p90_minutes": best_ready.get("door_to_door_p90_minutes") if best_ready else None,
        "best_field_ready_p75_minutes": best_ready.get("door_to_door_p75_minutes") if best_ready else None,
        "best_field_ready_on_foot_miles": best_ready.get("on_foot_miles") if best_ready else None,
        "minutes_over_bound_best_field_ready": (
            int(best_ready.get("door_to_door_p90_minutes")) - p90_bound_minutes if best_ready else None
        ),
    }


def build_probe_row(
    *,
    segment: dict[str, Any],
    anchor: dict[str, Any],
    state: dict[str, Any],
    performance_profile: dict[str, Any],
    connector_graph: dict[str, Any],
    elevation_sampler: Any,
    official_index: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    distance, basis = anchor_distance_to_segment(segment, anchor)
    candidate = candidate_from_trail_group(
        [segment_trail(segment)],
        forced_anchor_state(state, anchor),
        performance_profile,
        connector_graph,
        candidate_type="shingle_anchor_exhaustive_probe",
        elevation_sampler=elevation_sampler,
    )
    row = probe_candidate_row(candidate, segment, official_index, connector_graph)
    row["anchor_name"] = anchor.get("name")
    row["anchor_source"] = anchor.get("source")
    row["parking_confidence"] = anchor.get("parking_confidence")
    row["field_ready"] = anchor_field_ready(anchor)
    row["parking_risk"] = parking_risk_score(anchor)
    row["anchor_distance_miles"] = round(distance, 4)
    row["anchor_distance_basis"] = basis
    return row


def build_report(
    *,
    state: dict[str, Any],
    official_geojson_path: Path,
    connector_geojson: Path,
    dem_tif: Path,
    dem_summary_json: Path,
    public_trailheads_geojson: Path,
    private_parking_anchors_geojson: Path,
    manual_design_jsons: list[Path],
    segment_id: int,
) -> dict[str, Any]:
    official_segments, _meta = load_official_segments(official_geojson_path)
    segments_by_id = {int(segment["seg_id"]): segment for segment in official_segments}
    if segment_id not in segments_by_id:
        raise ValueError(f"Official segment not found: {segment_id}")
    segment = segments_by_id[segment_id]
    official_index = load_official_segment_index(official_geojson_path)
    connector_graph = load_connector_graph(connector_geojson, official_segments=official_segments)
    dem_context = load_dem_context(dem_tif, dem_summary_json)
    performance_profile = build_performance_profile(
        state=state,
        strava_activity_details_dir=DEFAULT_STRAVA_DETAILS_DIR,
        activity_summary_csv=DEFAULT_ACTIVITY_SUMMARY_CSV,
        activity_detail_summary_csv=DEFAULT_ACTIVITY_DETAIL_SUMMARY_CSV,
        segment_perf_csv=DEFAULT_SEGMENT_PERF_CSV,
    )
    anchors = load_all_anchors(
        public_trailheads_geojson=public_trailheads_geojson,
        private_parking_anchors_geojson=private_parking_anchors_geojson,
        manual_design_jsons=manual_design_jsons,
    )
    rows = [
        build_probe_row(
            segment=segment,
            anchor=anchor,
            state=state,
            performance_profile=performance_profile,
            connector_graph=connector_graph,
            elevation_sampler=dem_context["sampler"],
            official_index=official_index,
        )
        for anchor in anchors
    ]
    availability = state.get("availability_model") or {}
    p90_bound = max(
        int(availability.get("weekday_max_minutes") or 0),
        int(availability.get("weekend_max_minutes") or 0),
    )
    sorted_rows = sorted(rows, key=row_rank)
    summary = summarize_rows(rows, p90_bound_minutes=p90_bound)
    return {
        "objective": "exhaustively test Shingle Creek 1656 against every known parking anchor",
        "source_files": {
            "state_json": display_path(DEFAULT_STATE_JSON),
            "official_geojson": display_path(official_geojson_path),
            "connector_geojson": display_path(connector_geojson),
            "public_trailheads_geojson": display_path(public_trailheads_geojson),
            "private_parking_anchors_geojson": display_path(private_parking_anchors_geojson),
            "manual_design_jsons": [display_path(path) for path in manual_design_jsons],
        },
        "segment": {
            "seg_id": segment["seg_id"],
            "seg_name": segment["seg_name"],
            "trail_name": segment["trail_name"],
            "direction": segment["direction"],
            "official_miles": segment["official_miles"],
        },
        "summary": summary,
        "best_rows": sorted_rows[:15],
        "field_ready_best_rows": [row for row in sorted_rows if row.get("field_ready") is True][:15],
        "all_rows": sorted_rows,
        "interpretation": [
            "This closes the straight-line-nearest-anchor loophole for Shingle 1656 by testing every known public trailhead, manual anchor, and private Strava-derived parking anchor.",
            "A strict success requires graph validation, continuous track validation, field-ready parking, and p90 at or below the active max bound.",
            "The report intentionally records anchor names and metrics, not exact private anchor coordinates.",
        ],
    }


def render_md(report: dict[str, Any]) -> str:
    summary = report["summary"]
    segment = report["segment"]
    lines = [
        "# Shingle 1656 Anchor Exhaustive Probe",
        "",
        f"Objective: {report['objective']}",
        "",
        "## Verdict",
        "",
        f"- Segment: {segment['seg_name']} (`{segment['seg_id']}`)",
        f"- P90 bound: {summary['p90_bound_minutes']} min",
        f"- Strict success found: {summary['strict_success']}",
        f"- Best field-ready anchor: {summary['best_field_ready_anchor']}",
        f"- Best field-ready p90/p75: {summary['best_field_ready_p90_minutes']} / {summary['best_field_ready_p75_minutes']} min",
        f"- Minutes over bound: {summary['minutes_over_bound_best_field_ready']}",
        "",
        "## Summary",
        "",
        f"- Anchors tested: {summary['anchor_count']}",
        f"- Graph-validated rows: {summary['graph_validated_count']}",
        f"- Track-valid rows: {summary['track_validation_passed_count']}",
        f"- Field-ready anchors: {summary['field_ready_count']}",
        f"- Under-bound graph/track-valid rows: {summary['under_bound_graph_track_count']}",
        "",
        "## Best Rows",
        "",
        "| Rank | Anchor | P90 | P75 | On foot | Field ready | Parking | Source |",
        "|---:|---|---:|---:|---:|---|---|---|",
    ]
    for rank, row in enumerate(report["best_rows"], start=1):
        lines.append(
            f"| {rank} | {row['anchor_name']} | {row['door_to_door_p90_minutes']} | "
            f"{row['door_to_door_p75_minutes']} | {row['on_foot_miles']} | "
            f"{row['field_ready']} | {row['parking_confidence']} | {row['anchor_source']} |"
        )
    lines.extend(
        [
            "",
            "## Best Field-Ready Rows",
            "",
            "| Rank | Anchor | P90 | P75 | On foot | Distance basis | Distance mi | Parking |",
            "|---:|---|---:|---:|---:|---|---:|---|",
        ]
    )
    for rank, row in enumerate(report["field_ready_best_rows"], start=1):
        lines.append(
            f"| {rank} | {row['anchor_name']} | {row['door_to_door_p90_minutes']} | "
            f"{row['door_to_door_p75_minutes']} | {row['on_foot_miles']} | "
            f"{row['anchor_distance_basis']} | {row['anchor_distance_miles']} | "
            f"{row['parking_confidence']} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {item}" for item in report["interpretation"])
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-json", type=Path, default=DEFAULT_STATE_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--connector-geojson", type=Path, default=DEFAULT_CONNECTOR_GEOJSON)
    parser.add_argument("--dem-tif", type=Path, default=DEFAULT_DEM_TIF)
    parser.add_argument("--dem-summary-json", type=Path, default=DEFAULT_DEM_SUMMARY_JSON)
    parser.add_argument("--public-trailheads-geojson", type=Path, default=DEFAULT_TRAILHEAD_CANDIDATES_GEOJSON)
    parser.add_argument("--private-parking-anchors-geojson", type=Path, default=DEFAULT_PRIVATE_PARKING_ANCHORS_GEOJSON)
    parser.add_argument("--manual-design-json", type=Path, action="append", dest="manual_design_jsons")
    parser.add_argument("--segment-id", type=int, default=DEFAULT_SEGMENT_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manual_design_jsons = args.manual_design_jsons or DEFAULT_MANUAL_DESIGN_JSONS
    report = build_report(
        state=read_json(args.state_json),
        official_geojson_path=args.official_geojson,
        connector_geojson=args.connector_geojson,
        dem_tif=args.dem_tif,
        dem_summary_json=args.dem_summary_json,
        public_trailheads_geojson=args.public_trailheads_geojson,
        private_parking_anchors_geojson=args.private_parking_anchors_geojson,
        manual_design_jsons=manual_design_jsons,
        segment_id=args.segment_id,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.basename}.json"
    md_path = args.output_dir / f"{args.basename}.md"
    write_json(json_path, report)
    md_path.write_text(render_md(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
