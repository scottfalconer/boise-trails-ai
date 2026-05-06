#!/usr/bin/env python3
"""Probe p90 feasibility by forcing candidate routes from nearby parking anchors."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from export_execution_gpx import load_official_segment_index  # noqa: E402
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
    haversine_miles,
    load_connector_graph,
    load_dem_context,
    load_official_segments,
    load_trailheads_from_geojson,
    read_json,
)
from p90_segment_split_probe import probe_candidate_row, segment_trail  # noqa: E402


DEFAULT_SPLIT_PROBE_JSON = YEAR_DIR / "checkpoints" / "p90-segment-split-probe-2026-05-06.json"
DEFAULT_STATE_JSON = YEAR_DIR / "inputs" / "personal" / "2026-planner-state.private.json"
DEFAULT_MANUAL_DESIGN_JSONS = [
    YEAR_DIR / "inputs" / "personal" / "2026-manual-route-designs-v1.json",
    YEAR_DIR / "inputs" / "personal" / "2026-harlow-spring-manual-route-design-v1.json",
]
DEFAULT_OUTPUT_DIR = YEAR_DIR / "checkpoints"
DEFAULT_BASENAME = "p90-forced-anchor-probe-2026-05-06"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def manual_anchor_trailheads(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = read_json(path)
    anchors = []
    for area in data.get("areas") or []:
        for anchor in area.get("anchors") or []:
            anchors.append(
                {
                    "name": anchor["name"],
                    "lat": float(anchor["lat"]),
                    "lon": float(anchor["lon"]),
                    "parking_minutes": int(anchor.get("parking_minutes") or 8),
                    "has_parking": anchor.get("has_parking") is True,
                    "source": anchor.get("source") or "manual_route_design",
                    "parking_confidence": anchor.get("parking_confidence"),
                    "field_ready": anchor.get("field_ready") is True,
                    "manual_anchor_id": anchor.get("anchor_id"),
                    "manual_area_id": area.get("area_id"),
                }
            )
    return anchors


def dedupe_anchors(anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[str, float, float], dict[str, Any]] = {}
    for anchor in anchors:
        key = (
            str(anchor.get("name") or ""),
            round(float(anchor["lon"]), 6),
            round(float(anchor["lat"]), 6),
        )
        current = deduped.get(key)
        if current is None or anchor_field_ready(anchor) and not anchor_field_ready(current):
            deduped[key] = anchor
    return list(deduped.values())


def anchor_field_ready(anchor: dict[str, Any]) -> bool:
    if anchor.get("field_ready") is False:
        return False
    if anchor.get("field_ready") is True:
        return True
    confidence = str(anchor.get("parking_confidence") or "").lower()
    if anchor.get("has_parking") is not True:
        return False
    return any(
        token in confidence
        for token in [
            "strava_reused",
            "strava_seen",
            "validated",
            "inferred_from_trailhead_layer",
            "osm_amenity",
        ]
    )


def anchor_distance_to_segment(
    segment: dict[str, Any],
    anchor: dict[str, Any],
) -> tuple[float, str]:
    point = (float(anchor["lon"]), float(anchor["lat"]))
    candidates = [
        ("start", segment["start"]),
        ("end", segment["end"]),
        ("center", segment.get("center") or segment["start"]),
    ]
    return min(
        ((haversine_miles(point, coord), label) for label, coord in candidates),
        key=lambda item: item[0],
    )


def rank_anchors_for_segment(
    segment: dict[str, Any],
    anchors: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    ranked = []
    for anchor in anchors:
        distance, basis = anchor_distance_to_segment(segment, anchor)
        ranked.append(
            {
                "anchor": anchor,
                "distance_miles": round(distance, 4),
                "distance_basis": basis,
            }
        )
    return sorted(
        ranked,
        key=lambda item: (
            item["distance_miles"],
            parking_risk_score(item["anchor"]),
            str(item["anchor"].get("name") or ""),
        ),
    )[:limit]


def parking_risk_score(anchor: dict[str, Any]) -> int:
    if anchor_field_ready(anchor):
        confidence = str(anchor.get("parking_confidence") or "").lower()
        if "strava_reused" in confidence or "validated" in confidence:
            return 0
        return 1
    if anchor.get("has_parking") is True:
        return 2
    return 3


def target_segment_ids_from_split_probe(split_probe: dict[str, Any]) -> list[int]:
    summary = split_probe.get("summary") or {}
    ids = summary.get("still_missing_after_single_segment_probe_ids")
    if ids:
        return [int(seg_id) for seg_id in ids]
    return sorted(
        {
            int(row["seg_id"])
            for row in split_probe.get("probe_rows") or []
            if int(row.get("door_to_door_p90_minutes") or 0)
            > int(summary.get("max_bound_minutes") or 0)
        }
    )


def forced_anchor_state(state: dict[str, Any], anchor: dict[str, Any]) -> dict[str, Any]:
    copied = copy.deepcopy(state)
    copied["trailheads"] = [anchor]
    return copied


def forced_probe_row(
    *,
    segment: dict[str, Any],
    anchor: dict[str, Any],
    state: dict[str, Any],
    performance_profile: dict[str, Any],
    connector_graph: dict[str, Any],
    elevation_sampler: Any,
    official_index: dict[int, dict[str, Any]],
    distance_basis: str,
    anchor_distance_miles: float,
) -> dict[str, Any]:
    candidate = candidate_from_trail_group(
        [segment_trail(segment)],
        forced_anchor_state(state, anchor),
        performance_profile,
        connector_graph,
        candidate_type="forced_anchor_p90_probe",
        elevation_sampler=elevation_sampler,
    )
    row = probe_candidate_row(candidate, segment, official_index, connector_graph)
    row["anchor_name"] = anchor.get("name")
    row["anchor_source"] = anchor.get("source")
    row["parking_confidence"] = anchor.get("parking_confidence")
    row["field_ready"] = anchor_field_ready(anchor)
    row["parking_risk"] = parking_risk_score(anchor)
    row["anchor_distance_miles"] = anchor_distance_miles
    row["anchor_distance_basis"] = distance_basis
    return row


def best_rows_by_segment(
    rows: list[dict[str, Any]],
    *,
    p90_bound_minutes: int,
    require_field_ready: bool,
) -> dict[int, dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if int(row["door_to_door_p90_minutes"]) <= p90_bound_minutes
        and row["route_status"] == "graph_validated"
        and row["track_validation_passed"] is True
        and (not require_field_ready or row["field_ready"] is True)
    ]
    best: dict[int, dict[str, Any]] = {}
    for row in eligible:
        seg_id = int(row["seg_id"])
        current = best.get(seg_id)
        key = (
            int(row.get("door_to_door_p75_minutes") or row["door_to_door_p90_minutes"]),
            int(row["door_to_door_p90_minutes"]),
            float(row.get("on_foot_miles") or 0),
            int(row.get("parking_risk") or 0),
        )
        current_key = (
            int(current.get("door_to_door_p75_minutes") or current["door_to_door_p90_minutes"]),
            int(current["door_to_door_p90_minutes"]),
            float(current.get("on_foot_miles") or 0),
            int(current.get("parking_risk") or 0),
        ) if current else None
        if current is None or key < current_key:
            best[seg_id] = row
    return best


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    target_segment_ids: list[int],
    p90_bound_minutes: int,
) -> dict[str, Any]:
    target_set = set(int(seg_id) for seg_id in target_segment_ids)
    strict_best = best_rows_by_segment(rows, p90_bound_minutes=p90_bound_minutes, require_field_ready=True)
    conditional_best = best_rows_by_segment(rows, p90_bound_minutes=p90_bound_minutes, require_field_ready=False)
    conditional_only = set(conditional_best) - set(strict_best)
    return {
        "target_segment_count": len(target_set),
        "probe_row_count": len(rows),
        "p90_bound_minutes": p90_bound_minutes,
        "under_p90_bound_count": sum(1 for row in rows if int(row["door_to_door_p90_minutes"]) <= p90_bound_minutes),
        "track_validation_passed_count": sum(1 for row in rows if row["track_validation_passed"] is True),
        "graph_validated_count": sum(1 for row in rows if row["route_status"] == "graph_validated"),
        "strict_field_ready_segment_count": len(set(strict_best) & target_set),
        "conditional_segment_count": len(conditional_only & target_set),
        "strict_missing_segment_ids": sorted(target_set - set(strict_best)),
        "conditional_missing_segment_ids": sorted(target_set - set(conditional_best)),
    }


def load_all_anchors(
    *,
    public_trailheads_geojson: Path,
    private_parking_anchors_geojson: Path,
    manual_design_jsons: list[Path],
) -> list[dict[str, Any]]:
    anchors = []
    anchors.extend(load_trailheads_from_geojson(public_trailheads_geojson))
    anchors.extend(load_trailheads_from_geojson(private_parking_anchors_geojson))
    for path in manual_design_jsons:
        anchors.extend(manual_anchor_trailheads(path))
    return dedupe_anchors(anchors)


def build_report(
    *,
    split_probe: dict[str, Any],
    state: dict[str, Any],
    official_geojson_path: Path,
    connector_geojson: Path,
    dem_tif: Path,
    dem_summary_json: Path,
    public_trailheads_geojson: Path,
    private_parking_anchors_geojson: Path,
    manual_design_jsons: list[Path],
    nearest_anchors_per_segment: int,
) -> dict[str, Any]:
    official_segments, _meta = load_official_segments(official_geojson_path)
    segments_by_id = {int(segment["seg_id"]): segment for segment in official_segments}
    target_segment_ids = [
        seg_id for seg_id in target_segment_ids_from_split_probe(split_probe) if seg_id in segments_by_id
    ]
    anchors = load_all_anchors(
        public_trailheads_geojson=public_trailheads_geojson,
        private_parking_anchors_geojson=private_parking_anchors_geojson,
        manual_design_jsons=manual_design_jsons,
    )
    connector_graph = load_connector_graph(connector_geojson, official_segments=official_segments)
    dem_context = load_dem_context(dem_tif, dem_summary_json)
    official_index = load_official_segment_index(official_geojson_path)
    performance_profile = build_performance_profile(
        state=state,
        strava_activity_details_dir=DEFAULT_STRAVA_DETAILS_DIR,
        activity_summary_csv=DEFAULT_ACTIVITY_SUMMARY_CSV,
        activity_detail_summary_csv=DEFAULT_ACTIVITY_DETAIL_SUMMARY_CSV,
        segment_perf_csv=DEFAULT_SEGMENT_PERF_CSV,
    )
    rows = []
    for seg_id in target_segment_ids:
        segment = segments_by_id[seg_id]
        for ranked in rank_anchors_for_segment(segment, anchors, limit=nearest_anchors_per_segment):
            rows.append(
                forced_probe_row(
                    segment=segment,
                    anchor=ranked["anchor"],
                    state=state,
                    performance_profile=performance_profile,
                    connector_graph=connector_graph,
                    elevation_sampler=dem_context["sampler"],
                    official_index=official_index,
                    distance_basis=ranked["distance_basis"],
                    anchor_distance_miles=ranked["distance_miles"],
                )
            )
    availability = state.get("availability_model") or {}
    p90_bound = max(
        int(availability.get("weekday_max_minutes") or 0),
        int(availability.get("weekend_max_minutes") or 0),
    )
    strict_best = best_rows_by_segment(rows, p90_bound_minutes=p90_bound, require_field_ready=True)
    conditional_best = best_rows_by_segment(rows, p90_bound_minutes=p90_bound, require_field_ready=False)
    return {
        "objective": "test p90-missing official segments against nearby forced parking anchors",
        "source_files": {
            "split_probe_json": display_path(DEFAULT_SPLIT_PROBE_JSON),
            "state_json": display_path(DEFAULT_STATE_JSON),
            "official_geojson": display_path(official_geojson_path),
            "connector_geojson": display_path(connector_geojson),
            "public_trailheads_geojson": display_path(public_trailheads_geojson),
            "private_parking_anchors_geojson": display_path(private_parking_anchors_geojson),
            "manual_design_jsons": [display_path(path) for path in manual_design_jsons],
        },
        "config": {
            "nearest_anchors_per_segment": nearest_anchors_per_segment,
            "anchor_count": len(anchors),
        },
        "summary": summarize_rows(rows, target_segment_ids=target_segment_ids, p90_bound_minutes=p90_bound),
        "strict_best_rows": [strict_best[seg_id] for seg_id in sorted(strict_best)],
        "conditional_best_rows": [conditional_best[seg_id] for seg_id in sorted(conditional_best)],
        "probe_rows": sorted(
            rows,
            key=lambda row: (
                int(row["seg_id"]),
                int(row["door_to_door_p90_minutes"]),
                int(row["parking_risk"]),
                str(row.get("anchor_name") or ""),
            ),
        ),
        "caveats": [
            "This is still a probe, not a promoted field-menu route set.",
            "Strict rows require field-ready parking. Conditional rows may depend on manual parking/access verification.",
            "Private Strava-derived anchor coordinates stay in the ignored private source file; this report records names and metrics only.",
        ],
    }


def render_md(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# P90 Forced Anchor Probe",
        "",
        f"Objective: {report['objective']}",
        "",
        "## Summary",
        "",
        f"- Target segments: {summary['target_segment_count']}",
        f"- Probe rows: {summary['probe_row_count']}",
        f"- P90 bound: {summary['p90_bound_minutes']} min",
        f"- Rows under p90 bound: {summary['under_p90_bound_count']}",
        f"- Strict field-ready segments resolved: {summary['strict_field_ready_segment_count']}",
        f"- Conditional-only segments resolved: {summary['conditional_segment_count']}",
        f"- Strict missing segment ids: {', '.join(str(seg_id) for seg_id in summary['strict_missing_segment_ids'])}",
        f"- Conditional missing segment ids: {', '.join(str(seg_id) for seg_id in summary['conditional_missing_segment_ids'])}",
        "",
        "## Strict Best Rows",
        "",
        "| Segment | Trail | Anchor | P90 | P75 | On foot | Parking |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for row in report.get("strict_best_rows") or []:
        lines.append(
            f"| {row['seg_id']} | {row['trail_name']} | {row['anchor_name']} | "
            f"{row['door_to_door_p90_minutes']} | {row['door_to_door_p75_minutes']} | "
            f"{row['on_foot_miles']} | {row['parking_confidence']} |"
        )
    lines.extend(
        [
            "",
            "## Conditional Best Rows",
            "",
            "| Segment | Trail | Anchor | P90 | P75 | On foot | Field ready | Parking |",
            "|---:|---|---|---:|---:|---:|---|---|",
        ]
    )
    for row in report.get("conditional_best_rows") or []:
        lines.append(
            f"| {row['seg_id']} | {row['trail_name']} | {row['anchor_name']} | "
            f"{row['door_to_door_p90_minutes']} | {row['door_to_door_p75_minutes']} | "
            f"{row['on_foot_miles']} | {row['field_ready']} | {row['parking_confidence']} |"
        )
    lines.extend(["", "## Caveats", ""])
    lines.extend(f"- {caveat}" for caveat in report.get("caveats") or [])
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-probe-json", type=Path, default=DEFAULT_SPLIT_PROBE_JSON)
    parser.add_argument("--state-json", type=Path, default=DEFAULT_STATE_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--connector-geojson", type=Path, default=DEFAULT_CONNECTOR_GEOJSON)
    parser.add_argument("--dem-tif", type=Path, default=DEFAULT_DEM_TIF)
    parser.add_argument("--dem-summary-json", type=Path, default=DEFAULT_DEM_SUMMARY_JSON)
    parser.add_argument("--public-trailheads-geojson", type=Path, default=DEFAULT_TRAILHEAD_CANDIDATES_GEOJSON)
    parser.add_argument("--private-parking-anchors-geojson", type=Path, default=DEFAULT_PRIVATE_PARKING_ANCHORS_GEOJSON)
    parser.add_argument("--manual-design-json", type=Path, action="append", dest="manual_design_jsons")
    parser.add_argument("--nearest-anchors-per-segment", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manual_design_jsons = args.manual_design_jsons or DEFAULT_MANUAL_DESIGN_JSONS
    report = build_report(
        split_probe=read_json(args.split_probe_json),
        state=read_json(args.state_json),
        official_geojson_path=args.official_geojson,
        connector_geojson=args.connector_geojson,
        dem_tif=args.dem_tif,
        dem_summary_json=args.dem_summary_json,
        public_trailheads_geojson=args.public_trailheads_geojson,
        private_parking_anchors_geojson=args.private_parking_anchors_geojson,
        manual_design_jsons=manual_design_jsons,
        nearest_anchors_per_segment=args.nearest_anchors_per_segment,
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
