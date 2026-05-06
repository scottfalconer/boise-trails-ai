#!/usr/bin/env python3
"""Regenerate GPX tracks for selected forced-anchor probe loops."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import p90_forced_anchor_probe as forced_probe  # noqa: E402
from export_execution_gpx import (  # noqa: E402
    candidate_track_coordinates,
    load_official_segment_index,
    render_gpx,
    validate_track_segments,
)
from p90_segment_split_probe import segment_trail  # noqa: E402
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
    slugify,
)


DEFAULT_DRAFT_JSON = YEAR_DIR / "checkpoints" / "p90-relaxed-drive-draft-field-day-plan-2026-05-06.json"
DEFAULT_STATE_JSON = YEAR_DIR / "inputs" / "personal" / "2026-planner-state.private.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "outputs" / "private" / "relaxed-drive-gpx" / "forced-anchor"
DEFAULT_MANIFEST = YEAR_DIR / "checkpoints" / "p90-forced-anchor-gpx-export-2026-05-06.json"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def forced_anchor_loops(draft: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        loop | {"draft_day_number": day["draft_day_number"]}
        for day in draft.get("field_days") or []
        for loop in day.get("loops") or []
        if loop.get("source") == "forced_anchor_probe"
    ]


def segment_id_from_candidate_id(candidate_id: str) -> int:
    match = re.search(r"single-segment-(\d+)-", candidate_id)
    if not match:
        raise ValueError(f"Could not parse segment id from candidate id: {candidate_id}")
    return int(match.group(1))


def anchor_by_name(anchors: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for anchor in anchors:
        if str(anchor.get("name") or "") == name:
            return anchor
    raise ValueError(f"Anchor not found: {name}")


def build_export_report(
    *,
    draft: dict[str, Any],
    state: dict[str, Any],
    output_dir: Path,
    max_gap_miles: float,
) -> dict[str, Any]:
    official_segments, _meta = load_official_segments(DEFAULT_OFFICIAL_GEOJSON)
    segments_by_id = {int(segment["seg_id"]): segment for segment in official_segments}
    official_index = load_official_segment_index(DEFAULT_OFFICIAL_GEOJSON)
    connector_graph = load_connector_graph(DEFAULT_CONNECTOR_GEOJSON, official_segments=official_segments)
    dem_context = load_dem_context(DEFAULT_DEM_TIF, DEFAULT_DEM_SUMMARY_JSON)
    performance_profile = build_performance_profile(
        state=state,
        strava_activity_details_dir=DEFAULT_STRAVA_DETAILS_DIR,
        activity_summary_csv=DEFAULT_ACTIVITY_SUMMARY_CSV,
        activity_detail_summary_csv=DEFAULT_ACTIVITY_DETAIL_SUMMARY_CSV,
        segment_perf_csv=DEFAULT_SEGMENT_PERF_CSV,
    )
    anchors = forced_probe.load_all_anchors(
        public_trailheads_geojson=DEFAULT_TRAILHEAD_CANDIDATES_GEOJSON,
        private_parking_anchors_geojson=DEFAULT_PRIVATE_PARKING_ANCHORS_GEOJSON,
        manual_design_jsons=forced_probe.DEFAULT_MANUAL_DESIGN_JSONS,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for loop in forced_anchor_loops(draft):
        seg_id = segment_id_from_candidate_id(str(loop["candidate_id"]))
        segment = segments_by_id[seg_id]
        anchor = anchor_by_name(anchors, str(loop["trailhead"]))
        candidate = candidate_from_trail_group(
            [segment_trail(segment)],
            forced_probe.forced_anchor_state(state, anchor),
            performance_profile,
            connector_graph,
            candidate_type="forced_anchor_gpx_export",
            elevation_sampler=dem_context["sampler"],
        )
        track = candidate_track_coordinates(
            candidate,
            official_index,
            connector_graph=connector_graph,
            densify_source_lines=True,
        )
        validation = validate_track_segments([track], max_gap_miles=max_gap_miles)
        filename = f"{loop['draft_day_number']:02d}-{slugify(str(loop['label']))}-{seg_id}.gpx"
        path = output_dir / filename
        path.write_text(render_gpx(str(loop["label"]), track), encoding="utf-8")
        rows.append(
            {
                "draft_day_number": loop["draft_day_number"],
                "loop_id": loop["loop_id"],
                "candidate_id": loop["candidate_id"],
                "seg_id": seg_id,
                "label": loop["label"],
                "trailhead": loop["trailhead"],
                "path": display_path(path),
                "point_count": len(track),
                "validation": validation,
            }
        )
    return {
        "objective": "regenerate GPX tracks for selected forced-anchor probe loops",
        "source_files": {
            "draft_json": display_path(DEFAULT_DRAFT_JSON),
            "official_geojson": display_path(DEFAULT_OFFICIAL_GEOJSON),
            "state_json": display_path(DEFAULT_STATE_JSON),
            "connector_geojson": display_path(DEFAULT_CONNECTOR_GEOJSON),
        },
        "summary": {
            "forced_anchor_loop_count": len(rows),
            "gpx_validation_passed": all(row["validation"]["passed"] for row in rows),
            "failed_count": sum(1 for row in rows if not row["validation"]["passed"]),
            "max_gap_miles": max_gap_miles,
        },
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-json", type=Path, default=DEFAULT_DRAFT_JSON)
    parser.add_argument("--state-json", type=Path, default=DEFAULT_STATE_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--max-gap-miles", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_export_report(
        draft=read_json(args.draft_json),
        state=read_json(args.state_json),
        output_dir=args.output_dir,
        max_gap_miles=args.max_gap_miles,
    )
    write_json(args.manifest, report)
    print(f"Wrote {args.manifest}")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
