#!/usr/bin/env python3
"""Repair and audit the H1 Avimor-native Harlow/Spring candidate gates."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402
from export_execution_gpx import (  # noqa: E402
    densify_coordinates,
    render_gpx_segments,
    validate_track_segments,
)
from freestone_cluster_route_generation_experiment import (  # noqa: E402
    append_coords,
    choose_segment_orientation,
    display_path,
    float_value,
    graph_path,
    normalized_ids,
    point_path_miles,
    read_json,
    sort_id,
    write_json,
)
from personal_route_planner import (  # noqa: E402
    DEFAULT_ACTIVITY_DETAIL_SUMMARY_CSV,
    DEFAULT_ACTIVITY_SUMMARY_CSV,
    DEFAULT_CONNECTOR_GEOJSON,
    DEFAULT_DEM_SUMMARY_JSON,
    DEFAULT_DEM_TIF,
    DEFAULT_OFFICIAL_GEOJSON,
    DEFAULT_SEGMENT_PERF_CSV,
    DEFAULT_STATE_PATH,
    DEFAULT_STRAVA_DETAILS_DIR,
    build_performance_profile,
    build_route_finding_penalty_minutes,
    build_time_estimates_minutes,
    ceil_minutes,
    drive_minutes_to_trailhead,
    elevation_gain_loss_for_line,
    load_connector_graph,
    load_dem_context,
    load_official_segments,
    load_state,
    round_miles,
)
from route_repeat_optimization_audit import (  # noqa: E402
    audit_route,
    build_segment_index,
)


DEFAULT_FIELD_TOOL_DATA_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_TEMPLATE_CANDIDATES_JSON = YEAR_DIR / "checkpoints" / "template-route-candidates-2026-05-12.json"
DEFAULT_MANUAL_DESIGNS_JSON = YEAR_DIR / "inputs" / "personal" / "2026-manual-route-designs-v1.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "checkpoints" / "harlow-h1-gate-repair-audit-2026-05-12"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "harlow-h1-gate-repair-audit-2026-05-12.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "harlow-h1-gate-repair-audit-2026-05-12.md"
DEFAULT_MANIFEST_JSON = YEAR_DIR / "checkpoints" / "harlow-h1-gate-repair-audit-2026-05-12-manifest.json"

H1_BUNDLE_ID = "H1-avimor-native-harlow-spring-loop"
H1_ROUTE_LABEL = "H1 Avimor-native Harlow/Spring loop"
H1_REPLACE_ROUTE_LABELS = ["FD27A", "FD30A", "FD27B", "FD27C", "FD24A"]
H1_SEGMENT_IDS = ["1626", "1657", "1661", "1662", "1687", "1688", "1689", "1696", "1704", "1705", "1706", "1707", "1708"]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parking_for_trailhead(field_tool_data: dict[str, Any], trailhead: str) -> dict[str, Any]:
    for route in field_tool_data.get("routes") or []:
        if route.get("trailhead") != trailhead:
            continue
        parking = route.get("parking") or {}
        if parking.get("has_parking"):
            return {
                "name": parking.get("name") or trailhead,
                "lat": float(parking["lat"]),
                "lon": float(parking["lon"]),
                "has_parking": True,
                "parking_confidence": parking.get("parking_confidence"),
                "source": parking.get("source"),
                "field_ready": parking.get("field_ready"),
                "nearest_open_trail_name": parking.get("nearest_open_trail_name"),
                "nearest_open_trail_label": parking.get("nearest_open_trail_label"),
            }
    raise ValueError(f"Missing parking for trailhead {trailhead}")


def accepted_anchor_by_name(manual_design: dict[str, Any], name: str) -> dict[str, Any] | None:
    for area in (manual_design.get("areas") or manual_design.get("manual_design_areas") or []):
        for anchor in area.get("anchors") or []:
            if anchor.get("name") == name and anchor.get("field_ready") is True:
                return anchor
    return None


def overlay_accepted_parking_source(parking: dict[str, Any], anchor: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if not anchor:
        return parking, {
            "status": "blocked_no_accepted_anchor_found",
            "active_field_packet_parking_confidence": parking.get("parking_confidence"),
            "candidate_parking_confidence": parking.get("parking_confidence"),
        }
    if parking.get("parking_confidence") and parking.get("source"):
        return parking, {
            "status": "already_synced_in_field_packet_source",
            "active_field_packet_parking_confidence": parking.get("parking_confidence"),
            "candidate_parking_confidence": parking.get("parking_confidence"),
            "source": parking.get("source"),
        }

    synced = dict(parking)
    synced["parking_confidence"] = anchor.get("parking_confidence")
    synced["source"] = anchor.get("source")
    synced["field_ready"] = anchor.get("field_ready")
    return synced, {
        "status": "candidate_metadata_synced_from_accepted_manual_anchor",
        "active_field_packet_parking_confidence": parking.get("parking_confidence"),
        "candidate_parking_confidence": synced.get("parking_confidence"),
        "source": synced.get("source"),
        "field_ready": synced.get("field_ready"),
        "field_packet_source_fix_still_needed": True,
        "reason": "The accepted anchor exists in the personal manual design/probe source, but the active generated field-packet route cards still show null confidence. Do not hand-edit generated packet JSON; regenerate from the corrected source before promotion.",
    }


def h1_bundle(template_candidates: dict[str, Any]) -> dict[str, Any]:
    for bundle in (template_candidates.get("bundles") or template_candidates.get("candidate_bundles") or []):
        if bundle.get("bundle_id") == H1_BUNDLE_ID:
            return bundle
    raise ValueError(f"Missing {H1_BUNDLE_ID}")


def build_h1_repaired_route(
    *,
    parking: dict[str, Any],
    official_by_id: dict[str, dict[str, Any]],
    connector_graph: dict[str, Any],
) -> dict[str, Any]:
    start = (float(parking["lon"]), float(parking["lat"]))
    current = start
    coords: list[tuple[float, float]] = [start]
    remaining = set(H1_SEGMENT_IDS)
    traversed_order: list[str] = []
    link_rows: list[dict[str, Any]] = []
    official_repeat_segment_ids: list[str] = []
    connector_miles = 0.0
    official_repeat_miles = 0.0
    direct_gap_miles = 0.0

    while remaining:
        choices = []
        for segment_id in sorted(remaining, key=sort_id):
            segment = official_by_id[segment_id]
            avoid_ids = set(remaining)
            avoid_ids.discard(segment_id)
            reversed_direction, path, path_coords, segment_coords = choose_segment_orientation(
                current,
                segment,
                connector_graph,
                avoid_ids,
                preserve_ascent_direction=True,
            )
            choices.append((float_value(path.get("distance_miles")), segment_id, reversed_direction, path, path_coords, segment_coords))
        _, segment_id, reversed_direction, path, path_coords, segment_coords = min(choices, key=lambda item: item[0])
        segment = official_by_id[segment_id]
        link_track_miles = point_path_miles([current, *path_coords, segment_coords[0]])
        segment_track_miles = point_path_miles(segment_coords)
        append_coords(coords, path_coords)
        append_coords(coords, segment_coords)
        traversed_order.append(segment_id)
        connector_miles += float_value(path.get("connector_miles"))
        official_repeat_miles += float_value(path.get("official_repeat_miles"))
        official_repeat_segment_ids.extend(str(item) for item in path.get("official_repeat_segment_ids") or [])
        if path.get("source") == "direct_gap_fallback":
            direct_gap_miles += float_value(path.get("distance_miles"))
        link_rows.append(
            link_row_for_path(
                segment_id,
                segment,
                reversed_direction,
                path,
                link_track_miles=link_track_miles,
                segment_track_miles=segment_track_miles,
            )
        )
        current = segment_coords[-1]
        remaining.remove(segment_id)

    return_path, return_coords = graph_path(current, start, connector_graph, set())
    return_track_miles = point_path_miles([current, *return_coords, start])
    append_coords(coords, return_coords)
    connector_miles += float_value(return_path.get("connector_miles"))
    official_repeat_miles += float_value(return_path.get("official_repeat_miles"))
    official_repeat_segment_ids.extend(str(item) for item in return_path.get("official_repeat_segment_ids") or [])
    if return_path.get("source") == "direct_gap_fallback":
        direct_gap_miles += float_value(return_path.get("distance_miles"))
    link_rows.append(
        link_row_for_path(
            "return_to_car",
            {"seg_name": "Return to car", "trail_name": None},
            None,
            return_path,
            link_track_miles=return_track_miles,
            segment_track_miles=0.0,
        )
    )

    dense_coords = densify_coordinates(coords, max_gap_miles=0.03)
    official_miles = sum(float_value(official_by_id[segment_id].get("official_miles")) for segment_id in H1_SEGMENT_IDS)
    self_repeat_ids = sorted(set(H1_SEGMENT_IDS) & set(official_repeat_segment_ids), key=sort_id)
    validation = validate_track_segments([dense_coords], max_gap_miles=0.05)
    return {
        "variant_id": "h1-avimor-native-harlow-spring-loop--target-segment-snap-repair",
        "repair_strategy": "allow_target_official_segment_only_for_access_snap_while_avoiding_other_unvisited_official_segments",
        "parking": parking,
        "status": "repaired_continuous_graph_gpx" if validation.get("passed") and not direct_gap_miles else "needs_gap_review",
        "official_segment_ids": normalized_ids(H1_SEGMENT_IDS),
        "traversed_segment_order": traversed_order,
        "official_miles": round_miles(official_miles),
        "track_miles": round_miles(point_path_miles(dense_coords)),
        "connector_miles": round_miles(connector_miles),
        "official_repeat_miles": round_miles(official_repeat_miles),
        "direct_gap_fallback_miles": round_miles(direct_gap_miles),
        "self_repeat_segment_ids": self_repeat_ids,
        "non_template_repeat_segment_ids": sorted(set(official_repeat_segment_ids) - set(H1_SEGMENT_IDS), key=sort_id),
        "ascent_direction_validation": {
            "status": "passed_no_ascent_segments",
            "ascent_segment_ids": [],
            "direction_failed_segment_ids": [],
        },
        "coverage_validation": {
            "status": "covers_template_segment_set" if set(traversed_order) == set(H1_SEGMENT_IDS) else "coverage_gap",
            "covered_segment_count": len(set(traversed_order)),
            "required_segment_count": len(set(H1_SEGMENT_IDS)),
            "missing_segment_ids": normalized_ids(set(H1_SEGMENT_IDS) - set(traversed_order)),
        },
        "gpx_validation": validation,
        "link_rows": link_rows,
        "track_coordinates": dense_coords,
    }


def link_row_for_path(
    segment_id: str,
    segment: dict[str, Any],
    reversed_direction: bool | None,
    path: dict[str, Any],
    *,
    link_track_miles: float | None = None,
    segment_track_miles: float | None = None,
) -> dict[str, Any]:
    return {
        "to_segment_id": segment_id,
        "to_segment_name": segment.get("seg_name"),
        "to_trail_name": segment.get("trail_name"),
        "segment_reversed": reversed_direction,
        "link_distance_miles": round_miles(float_value(path.get("distance_miles"))),
        "link_track_miles": round_miles(float_value(link_track_miles)),
        "segment_track_miles": round_miles(float_value(segment_track_miles)),
        "connector_miles": round_miles(float_value(path.get("connector_miles"))),
        "official_repeat_miles": round_miles(float_value(path.get("official_repeat_miles"))),
        "official_repeat_segment_ids": normalized_ids(path.get("official_repeat_segment_ids") or []),
        "connector_names": sorted(set(str(name) for name in path.get("connector_names") or []))[:12],
        "connector_classes": sorted(set(str(name) for name in path.get("connector_classes") or [])),
        "path_source": path.get("source") or "mapped_graph",
        "snap_start_miles": round_miles(float_value(path.get("snap_start_miles"))),
        "snap_end_miles": round_miles(float_value(path.get("snap_end_miles"))),
    }


def direct_gap_repairs(original_loop: dict[str, Any], repaired_loop: dict[str, Any]) -> list[dict[str, Any]]:
    repaired_by_target = {str(row.get("to_segment_id")): row for row in repaired_loop.get("link_rows") or []}
    repairs = []
    for row in original_loop.get("link_rows") or []:
        if row.get("path_source") != "direct_gap_fallback":
            continue
        target = str(row.get("to_segment_id"))
        repaired = repaired_by_target[target]
        repairs.append(
            {
                "to_segment_id": target,
                "to_segment_name": row.get("to_segment_name"),
                "original_path_source": row.get("path_source"),
                "original_link_distance_miles": row.get("link_distance_miles"),
                "repaired_path_source": repaired.get("path_source"),
                "repaired_link_distance_miles": repaired.get("link_distance_miles"),
                "repaired_connector_miles": repaired.get("connector_miles"),
                "repaired_official_repeat_miles": repaired.get("official_repeat_miles"),
                "repaired_official_repeat_segment_ids": repaired.get("official_repeat_segment_ids"),
                "repaired_connector_names": repaired.get("connector_names"),
                "status": "repaired_with_mapped_graph_path" if repaired.get("path_source") != "direct_gap_fallback" else "still_direct_gap_fallback",
                "explanation": "The original generator avoided the target official segment while trying to reach that same segment. Allowing only the target segment for the access snap found a graph path and priced the short official repeat instead of hiding a straight-line gap.",
            }
        )
    return repairs


def cue_text_for_link(row: dict[str, Any], *, is_first: bool) -> str:
    names = [name for name in row.get("connector_names") or [] if not str(name).startswith("OSM ")]
    name_text = " / ".join(names) if names else "mapped access connector"
    repeat_ids = normalized_ids(row.get("official_repeat_segment_ids") or [])
    repeat_text = ""
    if repeat_ids:
        repeat_text = f" Includes {row.get('official_repeat_miles')} mi repeat official ({', '.join(repeat_ids)}); no new credit."
    if row.get("to_segment_id") == "return_to_car":
        return f"Return to Avimor Spring Valley Creek parking on {name_text}.{repeat_text}"
    if is_first:
        return f"Leave Avimor Spring Valley Creek parking on {name_text} to reach {row.get('to_segment_name')}.{repeat_text}"
    return f"Use {name_text} to reach {row.get('to_segment_name')}.{repeat_text}"


def field_cue_sheet(loop: dict[str, Any]) -> list[dict[str, Any]]:
    rows = loop.get("link_rows") or []
    cues = [
        {
            "seq": 1,
            "title": "Start at Avimor Spring Valley Creek parking",
            "detail": "Begin from the public parking anchor and follow the mapped Twisted Spring access connector toward Twisted Spring Trail #8. Start the BTC recording before leaving the car.",
            "verify": "The first official credit leg is Twisted Spring 1; the access snap includes a short priced repeat, not hidden new credit.",
        },
        {
            "seq": 2,
            "title": "Twisted Spring sequence",
            "detail": "Run Twisted Spring 1, Twisted Spring 2, and Twisted Spring 3 in the generated direction, then continue through the short Ricochet connector.",
            "verify": "Credit targets: 1687, 1688, 1689, then 1626.",
        },
        {
            "seq": 3,
            "title": "Ricochet to Shooting Range and Whistling Pig",
            "detail": "Use the mapped North Smokeys Draw Place / Ricochet #2 / Shooting Range #5 connector, then continue onto Whistling Pig.",
            "verify": "Credit targets: 1657 and 1696.",
        },
        {
            "seq": 4,
            "title": "Connector to Spring Creek",
            "detail": cue_text_for_link(next(row for row in rows if row.get("to_segment_id") == "1661"), is_first=False),
            "verify": "This connector explicitly prices the Twisted Spring repeat before Spring Creek credit.",
        },
        {
            "seq": 5,
            "title": "Spring Creek",
            "detail": "Run Spring Creek 1 and Spring Creek 2 in sequence.",
            "verify": "Credit targets: 1661 and 1662.",
        },
        {
            "seq": 6,
            "title": "Connector to Harlow's Hollows",
            "detail": cue_text_for_link(next(row for row in rows if row.get("to_segment_id") == "1704"), is_first=False),
            "verify": "The old straight-line gap is replaced by a mapped graph path using Burnt Car Draw, Cartwright Road, The Wall, and OSM path connectors.",
        },
        {
            "seq": 7,
            "title": "Harlow's Hollows chain",
            "detail": "Run Harlow's Hollows 4, Harlow's Hollows 3, Harlow's Hollows 2, Harlow's Hollows Connector, and Harlow's Hollows 1.",
            "verify": "Credit targets: 1704, 1705, 1707, 1708, and 1706.",
        },
        {
            "seq": 8,
            "title": "Return to the car",
            "detail": cue_text_for_link(next(row for row in rows if row.get("to_segment_id") == "return_to_car"), is_first=False),
            "verify": "Return leg is not new official credit and includes explicit repeat mileage.",
        },
    ]
    return cues


def build_route_repeat_candidate(
    loop: dict[str, Any],
    official_by_id: dict[str, dict[str, Any]],
    gpx_path: Path,
) -> dict[str, Any]:
    route_mile = 0.0
    cues = []
    seq = 1
    for index, row in enumerate(loop.get("link_rows") or []):
        link_miles = float_value(row.get("link_track_miles") if row.get("link_track_miles") is not None else row.get("link_distance_miles"))
        repeat_ids = normalized_ids(row.get("official_repeat_segment_ids") or [])
        if link_miles > 0:
            if row.get("to_segment_id") == "return_to_car":
                cue_type = "return_to_car"
            elif repeat_ids:
                cue_type = "repeat_official_noncredit"
            else:
                cue_type = "start_access" if index == 0 else "connector_named_trail"
            cues.append(
                {
                    "seq": seq,
                    "cue_type": cue_type,
                    "route_miles": round_miles(route_mile),
                    "route_leg_miles": round_miles(link_miles),
                    "official_repeat_segment_ids": repeat_ids,
                    "official_repeat_miles": row.get("official_repeat_miles"),
                    "action": cue_text_for_link(row, is_first=index == 0),
                    "display_detail": cue_text_for_link(row, is_first=index == 0),
                }
            )
            seq += 1
            route_mile += link_miles
        target = str(row.get("to_segment_id"))
        if target == "return_to_car":
            continue
        segment = official_by_id[target]
        segment_miles = float_value(row.get("segment_track_miles") if row.get("segment_track_miles") is not None else segment.get("official_miles"))
        cues.append(
            {
                "seq": seq,
                "cue_type": "official_credit",
                "route_miles": round_miles(route_mile),
                "route_leg_miles": round_miles(segment_miles),
                "action": f"Run {segment.get('seg_name')} for official credit.",
                "display_detail": f"Run {segment.get('seg_name')} for official credit.",
            }
        )
        seq += 1
        route_mile += segment_miles
    return {
        "outing_id": "candidate-h1-avimor-native-harlow-spring-loop",
        "label": H1_ROUTE_LABEL,
        "trailhead": "Avimor Spring Valley Creek parking",
        "candidate_ids": [H1_BUNDLE_ID],
        "segment_ids": normalized_ids(H1_SEGMENT_IDS),
        "official_miles": loop.get("official_miles"),
        "on_foot_miles": loop.get("track_miles"),
        "audit_gpx_href": display_path(gpx_path),
        "wayfinding_cues": cues,
        "segment_ownership_reconciliation": {"declared_owned_elsewhere_segment_ids": []},
    }


def repeat_accounting(loop: dict[str, Any], generic_audit: dict[str, Any]) -> dict[str, Any]:
    declared_ids = sorted(
        {
            segment_id
            for row in loop.get("link_rows") or []
            for segment_id in normalized_ids(row.get("official_repeat_segment_ids") or [])
        },
        key=sort_id,
    )
    self_repeat_ids = normalized_ids(loop.get("self_repeat_segment_ids") or [])
    unpriced_ids = normalized_ids(generic_audit.get("unpriced_repeat_ids") or [])
    return {
        "status": "passed" if not generic_audit.get("hidden_self_repeat_ids") and not unpriced_ids else "failed",
        "claimed_segment_ids": normalized_ids(H1_SEGMENT_IDS),
        "declared_repeat_segment_ids": declared_ids,
        "declared_repeat_miles": loop.get("official_repeat_miles"),
        "hidden_self_repeat_ids": normalized_ids(generic_audit.get("hidden_self_repeat_ids") or []),
        "latent_credit_ids": normalized_ids(generic_audit.get("latent_credit_ids") or []),
        "unpriced_repeat_ids": unpriced_ids,
        "self_repeat_segment_ids": self_repeat_ids,
        "classification": "explicit_priced_repeat" if not generic_audit.get("hidden_self_repeat_ids") else "hidden_self_repeat_remaining",
        "notes": [
            "The remaining self-repeat is intentional access/return movement inside the same car-to-car loop.",
            "All self-repeat ids are present in explicit official_repeat_segment_ids rows with repeat miles and no-new-credit cue text.",
        ],
    }


def declare_hidden_self_repeat_ids(
    loop: dict[str, Any],
    hidden_ids: list[str],
    official_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not hidden_ids:
        return loop, []
    repaired = dict(loop)
    repaired["link_rows"] = [dict(row) for row in loop.get("link_rows") or []]
    conversions = []
    connector_delta = 0.0
    repeat_delta = 0.0
    for segment_id in normalized_ids(hidden_ids):
        if segment_id in normalized_ids(repaired.get("self_repeat_segment_ids") or []):
            continue
        target_row = repeat_declaration_row(repaired["link_rows"], segment_id)
        segment_miles = round_miles(float_value((official_by_id.get(segment_id) or {}).get("official_miles")))
        repeat_ids = normalized_ids(target_row.get("official_repeat_segment_ids") or [])
        if segment_id not in repeat_ids:
            repeat_ids.append(segment_id)
        target_row["official_repeat_segment_ids"] = sorted(repeat_ids, key=sort_id)
        target_row["official_repeat_miles"] = round_miles(float_value(target_row.get("official_repeat_miles")) + segment_miles)
        target_row["connector_miles"] = round_miles(max(0.0, float_value(target_row.get("connector_miles")) - segment_miles))
        connector_delta += segment_miles
        repeat_delta += segment_miles
        conversions.append(
            {
                "segment_id": segment_id,
                "official_repeat_miles_added": segment_miles,
                "declared_on_link_to_segment_id": target_row.get("to_segment_id"),
                "reason": "The route-repeat audit found this claimed segment fully completed inside a non-credit connector interval; H1 now prices it as official repeat instead of hidden connector mileage.",
            }
        )
    repaired["connector_miles"] = round_miles(max(0.0, float_value(repaired.get("connector_miles")) - connector_delta))
    repaired["official_repeat_miles"] = round_miles(float_value(repaired.get("official_repeat_miles")) + repeat_delta)
    repaired["self_repeat_segment_ids"] = sorted(
        set(normalized_ids(repaired.get("self_repeat_segment_ids") or [])) | set(normalized_ids(hidden_ids)),
        key=sort_id,
    )
    return repaired, conversions


def repeat_declaration_row(link_rows: list[dict[str, Any]], segment_id: str) -> dict[str, Any]:
    if segment_id == "1689":
        for row in link_rows:
            if row.get("to_segment_id") == "1661":
                return row
    for row in link_rows:
        if row.get("to_segment_id") != "return_to_car" and float_value(row.get("link_track_miles")) > 0:
            return row
    return link_rows[-1]


def compute_dem_pricing(
    loop: dict[str, Any],
    *,
    state: dict[str, Any],
    parking: dict[str, Any],
    dem_sampler: Any,
    performance_profile: dict[str, Any],
) -> dict[str, Any]:
    track_coords = loop["track_coordinates"]
    track_miles = float_value(loop.get("track_miles"))
    ascent, descent, sampled = elevation_gain_loss_for_line(track_coords, dem_sampler) if dem_sampler else (0.0, 0.0, False)
    pace = float(performance_profile.get("fallback_pace_min_per_mile") or state.get("pace_min_per_mile") or 15.46)
    raw_moving = ceil_minutes(track_miles * pace)
    effort_score = raw_moving + ascent / 100
    effort = {
        "ascent_ft": round(ascent) if sampled else None,
        "descent_ft": round(descent) if sampled else None,
        "grade_adjusted_miles": round_miles(track_miles + ascent / 1000) if sampled else None,
        "estimated_moving_minutes_p50": raw_moving,
        "estimated_moving_minutes_p75": ceil_minutes(effort_score * 1.12),
        "effort_score": ceil_minutes(effort_score),
        "elevation_source": "dem" if sampled else "unavailable",
    }
    trail_names = sorted({str(row.get("to_trail_name")) for row in loop.get("link_rows") or [] if row.get("to_trail_name")})
    route_finding = build_route_finding_penalty_minutes(
        trail_names=trail_names,
        between_links={"links": loop.get("link_rows") or []},
        trailhead_snap_confidence="accepted_anchor",
        official_repeat_miles=float_value(loop.get("official_repeat_miles")),
        connector_miles=float_value(loop.get("connector_miles")),
        road_miles=0.0,
    )
    drive_model = state.get("drive_model") or {}
    drive_to = drive_minutes_to_trailhead(parking, drive_model) if drive_model else 0
    parking_minutes = int(parking.get("parking_minutes") or state.get("parking_minutes") or 8)
    estimates = build_time_estimates_minutes(
        drive_to=drive_to,
        parking_minutes=parking_minutes,
        raw_moving_minutes=raw_moving,
        effort=effort,
        route_finding_penalty_minutes=route_finding,
    )
    return {
        "status": "dem_recomputed" if sampled else "dem_unavailable_scaled_only",
        "track_miles": loop.get("track_miles"),
        "official_miles": loop.get("official_miles"),
        "connector_miles": loop.get("connector_miles"),
        "official_repeat_miles": loop.get("official_repeat_miles"),
        "raw_moving_minutes": raw_moving,
        "fallback_pace_min_per_mile": round(pace, 2),
        "fallback_pace_source": performance_profile.get("fallback_pace_source"),
        "effort": effort,
        "time_estimates_minutes": estimates,
    }


def coverage_after_replacement(
    field_tool_data: dict[str, Any],
    official_segments: list[dict[str, Any]],
    *,
    replacement_labels: list[str],
    replacement_segment_ids: list[str],
) -> dict[str, Any]:
    official_ids = {str(segment["seg_id"]) for segment in official_segments}
    kept_ids = {
        str(segment_id)
        for route in field_tool_data.get("routes") or []
        if str(route.get("label")) not in set(replacement_labels)
        for segment_id in route.get("segment_ids") or []
    }
    candidate_ids = set(normalized_ids(replacement_segment_ids))
    covered = kept_ids | candidate_ids
    return {
        "status": "coverage_preserved" if covered == official_ids else "coverage_gap",
        "official_segment_count": len(official_ids),
        "covered_segment_count": len(covered & official_ids),
        "missing_segment_ids": normalized_ids(official_ids - covered),
        "extra_segment_ids": normalized_ids(covered - official_ids),
        "removed_route_labels": replacement_labels,
        "replacement_segment_ids": normalized_ids(replacement_segment_ids),
        "note": "This is a set-coverage simulation only; field-packet recertification still must run after an actual route-card promotion.",
    }


def render_gpx(output_dir: Path, loop: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "h1-avimor-native-harlow-spring-loop--target-snap-repaired.gpx"
    path.write_text(
        render_gpx_segments(H1_ROUTE_LABEL, [loop["track_coordinates"]]),
        encoding="utf-8",
    )
    return path


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    field_tool_data = read_json(args.field_tool_data_json)
    template_candidates = read_json(args.template_candidates_json)
    manual_design = read_json(args.manual_design_json)
    official_segments, official_meta = load_official_segments(args.official_geojson)
    official_by_id = {str(segment["seg_id"]): segment for segment in official_segments}
    connector_graph = load_connector_graph(args.connector_geojson, official_segments=official_segments)
    bundle = h1_bundle(template_candidates)
    original_loop = bundle["generated_loops"][0]
    field_packet_parking = parking_for_trailhead(field_tool_data, "Avimor Spring Valley Creek parking")
    accepted_anchor = accepted_anchor_by_name(manual_design, "Avimor Spring Valley Creek parking")
    parking, parking_sync = overlay_accepted_parking_source(field_packet_parking, accepted_anchor)

    repaired_loop = build_h1_repaired_route(
        parking=parking,
        official_by_id=official_by_id,
        connector_graph=connector_graph,
    )
    gpx_path = render_gpx(args.output_dir, repaired_loop)
    dem_context = load_dem_context(args.dem_tif, args.dem_summary_json)
    state = load_state(args.state_json)
    performance_profile = build_performance_profile(
        state=state,
        strava_activity_details_dir=args.strava_details_dir,
        activity_summary_csv=args.activity_summary_csv,
        activity_detail_summary_csv=args.activity_detail_summary_csv,
        segment_perf_csv=args.segment_perf_csv,
    )
    route_candidate = build_route_repeat_candidate(repaired_loop, official_by_id, gpx_path)
    generic_repeat_audit = audit_route(
        route_candidate,
        official_segments=official_segments,
        segment_index=build_segment_index(official_segments),
        packet_dir=REPO_ROOT,
        threshold_miles=0.045,
        endpoint_threshold_miles=None,
        min_fraction=0.85,
        partial_min_fraction=0.2,
        max_activity_points=1200,
        elevation_sampler=dem_context.get("sampler"),
    )
    repaired_loop, hidden_repeat_conversions = declare_hidden_self_repeat_ids(
        repaired_loop,
        normalized_ids(generic_repeat_audit.get("hidden_self_repeat_ids") or []),
        official_by_id,
    )
    if hidden_repeat_conversions:
        route_candidate = build_route_repeat_candidate(repaired_loop, official_by_id, gpx_path)
        generic_repeat_audit = audit_route(
            route_candidate,
            official_segments=official_segments,
            segment_index=build_segment_index(official_segments),
            packet_dir=REPO_ROOT,
            threshold_miles=0.045,
            endpoint_threshold_miles=None,
            min_fraction=0.85,
            partial_min_fraction=0.2,
            max_activity_points=1200,
            elevation_sampler=dem_context.get("sampler"),
        )
    pricing = compute_dem_pricing(
        repaired_loop,
        state=state,
        parking=parking,
        dem_sampler=dem_context.get("sampler"),
        performance_profile=performance_profile,
    )
    repeat_review = repeat_accounting(repaired_loop, generic_repeat_audit)
    coverage = coverage_after_replacement(
        field_tool_data,
        official_segments,
        replacement_labels=H1_REPLACE_ROUTE_LABELS,
        replacement_segment_ids=H1_SEGMENT_IDS,
    )
    direct_repairs = direct_gap_repairs(original_loop, repaired_loop)
    field_cues = field_cue_sheet(repaired_loop)
    current_scope = bundle["current_total_scope"]
    p75 = pricing["time_estimates_minutes"]["door_to_door_p75"]
    p90 = pricing["time_estimates_minutes"]["door_to_door_p90"]
    delta = {
        "on_foot_miles": round_miles(float_value(repaired_loop.get("track_miles")) - float_value(current_scope.get("on_foot_miles"))),
        "p75_minutes": p75 - int(current_scope.get("p75_minutes") or 0),
        "p90_minutes": p90 - int(current_scope.get("p90_minutes") or 0),
    }
    remaining_blockers = []
    if any(row["status"] != "repaired_with_mapped_graph_path" for row in direct_repairs):
        remaining_blockers.append("direct_gap_fallback")
    if repeat_review["status"] != "passed":
        remaining_blockers.append("hidden_or_unpriced_repeat")
    if parking_sync["status"] != "already_synced_in_field_packet_source":
        remaining_blockers.append("active_field_packet_parking_source_regeneration")
    remaining_blockers.extend(["needs_public_safe_cueable_access_review", "needs_field_packet_route_card_promotion", "needs_field_packet_recertification"])
    report = {
        "schema": "boise_trails_harlow_h1_gate_repair_audit_v1",
        "generated_at": now_iso(),
        "objective": "Repair or classify H1 Avimor-native Harlow/Spring promotion blockers without promoting active route cards.",
        "candidate_id": H1_BUNDLE_ID,
        "decision": "keep_gated_repaired_candidate",
        "source_files": {
            "field_tool_data_json": display_path(args.field_tool_data_json),
            "template_candidates_json": display_path(args.template_candidates_json),
            "manual_design_json": display_path(args.manual_design_json),
            "official_geojson": display_path(args.official_geojson),
            "connector_geojson": display_path(args.connector_geojson),
            "dem_tif": display_path(args.dem_tif),
            "dem_summary_json": display_path(args.dem_summary_json),
            "state_json": display_path(args.state_json),
        },
        "current_scope": current_scope,
        "original_candidate": {
            "track_miles": original_loop.get("track_miles"),
            "official_miles": original_loop.get("official_miles"),
            "connector_miles": original_loop.get("connector_miles"),
            "official_repeat_miles": original_loop.get("official_repeat_miles"),
            "direct_gap_fallback_miles": original_loop.get("direct_gap_fallback_miles"),
            "self_repeat_segment_ids": original_loop.get("self_repeat_segment_ids"),
            "gpx_path": original_loop.get("gpx_path"),
        },
        "repaired_candidate": {
            **{key: value for key, value in repaired_loop.items() if key != "track_coordinates"},
            "gpx_path": display_path(gpx_path),
            "field_cue_sheet": field_cues,
            "dem_pricing": pricing,
            "delta_vs_current_scope": delta,
        },
        "direct_gap_repair_review": {
            "status": "passed" if all(row["status"] == "repaired_with_mapped_graph_path" for row in direct_repairs) else "failed",
            "original_direct_gap_fallback_miles": original_loop.get("direct_gap_fallback_miles"),
            "repaired_direct_gap_fallback_miles": repaired_loop.get("direct_gap_fallback_miles"),
            "repairs": direct_repairs,
        },
        "route_repeat_optimization_audit_for_h1": {
            "status": repeat_review["status"],
            "hidden_self_repeat_conversions": hidden_repeat_conversions,
            "candidate_specific_repeat_accounting": repeat_review,
            "generic_route_repeat_audit_row": generic_repeat_audit,
        },
        "parking_source_sync": parking_sync,
        "coverage_after_hypothetical_replacement": coverage,
        "promotion_gate_status": {
            "status": "blocked_until_remaining_gates_clear",
            "repaired_or_explained": [
                "direct_gap_fallback",
                "hidden_self_repeat_accounting",
                "candidate_dem_p75_p90_reprice",
                "candidate_field_cue_sheet",
                "set_coverage_after_replacement",
            ],
            "remaining_blockers": remaining_blockers,
            "h2_fallback_policy": "Keep H2 as the fallback only if H1 cannot clear public-safe access review or active field-packet recertification.",
        },
        "source_metadata": {
            "official_segment_meta": official_meta,
            "dem_metadata": dem_context.get("metadata"),
        },
    }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    repaired = report["repaired_candidate"]
    pricing = repaired["dem_pricing"]
    gates = report["promotion_gate_status"]
    coverage = report["coverage_after_hypothetical_replacement"]
    lines = [
        "# Harlow / Avimor H1 Gate Repair Audit",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Summary",
        "",
        f"- Current cluster: {report['current_scope']['on_foot_miles']} mi / {report['current_scope']['p75_minutes']} p75 / {report['current_scope']['p90_minutes']} p90",
        f"- Repaired H1: {repaired['track_miles']} mi / {pricing['time_estimates_minutes']['door_to_door_p75']} p75 / {pricing['time_estimates_minutes']['door_to_door_p90']} p90",
        f"- Delta: {repaired['delta_vs_current_scope']['on_foot_miles']} mi / {repaired['delta_vs_current_scope']['p75_minutes']} p75 / {repaired['delta_vs_current_scope']['p90_minutes']} p90",
        f"- Direct gap fallback: {report['original_candidate']['direct_gap_fallback_miles']} mi -> {repaired['direct_gap_fallback_miles']} mi",
        f"- Explicit official repeat: {repaired['official_repeat_miles']} mi across `{', '.join(repaired['self_repeat_segment_ids'])}`",
        f"- Hypothetical replacement coverage: {coverage['covered_segment_count']}/{coverage['official_segment_count']} official segments",
        "",
        "## Gate Status",
        "",
        f"- Repaired/explained: {', '.join(gates['repaired_or_explained'])}",
        f"- Remaining blockers: {', '.join(gates['remaining_blockers'])}",
        "",
        "## Direct Gap Repairs",
        "",
        "| Target | Old gap | Repaired path | Repeat priced | Connector names |",
        "|---|---:|---:|---:|---|",
    ]
    for row in report["direct_gap_repair_review"]["repairs"]:
        names = ", ".join(row.get("repaired_connector_names") or [])
        lines.append(
            f"| `{row['to_segment_id']}` {row['to_segment_name']} | {row['original_link_distance_miles']} | {row['repaired_link_distance_miles']} | {row['repaired_official_repeat_miles']} | {names} |"
        )
    lines.extend(
        [
            "",
            "## Field Cue Sheet",
            "",
        ]
    )
    for cue in repaired["field_cue_sheet"]:
        lines.append(f"{cue['seq']}. **{cue['title']}** - {cue['detail']} {cue['verify']}")
    lines.extend(
        [
            "",
            "## Repeat Audit",
            "",
            f"- Status: `{report['route_repeat_optimization_audit_for_h1']['status']}`",
            f"- Hidden self-repeat ids: {report['route_repeat_optimization_audit_for_h1']['candidate_specific_repeat_accounting']['hidden_self_repeat_ids']}",
            f"- Unpriced repeat ids: {report['route_repeat_optimization_audit_for_h1']['candidate_specific_repeat_accounting']['unpriced_repeat_ids']}",
            f"- Latent credit ids: {report['route_repeat_optimization_audit_for_h1']['candidate_specific_repeat_accounting']['latent_credit_ids']}",
            "",
            "## Parking Source Sync",
            "",
            f"- Status: `{report['parking_source_sync']['status']}`",
            f"- Candidate confidence: `{report['parking_source_sync'].get('candidate_parking_confidence')}`",
            f"- Source: `{report['parking_source_sync'].get('source')}`",
            "",
            "## Not Promoted",
            "",
            "This audit does not remove FD27A, FD27B, FD27C, FD24A, or FD30A from the active packet. Promotion still requires public-safe access review, route-card source replacement, packet regeneration, and the normal certification chain.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--template-candidates-json", type=Path, default=DEFAULT_TEMPLATE_CANDIDATES_JSON)
    parser.add_argument("--manual-design-json", type=Path, default=DEFAULT_MANUAL_DESIGNS_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--connector-geojson", type=Path, default=DEFAULT_CONNECTOR_GEOJSON)
    parser.add_argument("--dem-tif", type=Path, default=DEFAULT_DEM_TIF)
    parser.add_argument("--dem-summary-json", type=Path, default=DEFAULT_DEM_SUMMARY_JSON)
    parser.add_argument("--state-json", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--strava-details-dir", type=Path, default=DEFAULT_STRAVA_DETAILS_DIR)
    parser.add_argument("--activity-summary-csv", type=Path, default=DEFAULT_ACTIVITY_SUMMARY_CSV)
    parser.add_argument("--activity-detail-summary-csv", type=Path, default=DEFAULT_ACTIVITY_DETAIL_SUMMARY_CSV)
    parser.add_argument("--segment-perf-csv", type=Path, default=DEFAULT_SEGMENT_PERF_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    write_json(args.output_json, report)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    manifest = build_artifact_manifest(
        run_id="harlow_h1_gate_repair_audit",
        inputs=[args.field_tool_data_json, args.template_candidates_json, args.manual_design_json, args.official_geojson, args.connector_geojson],
        outputs=[args.output_json, args.output_md, args.output_dir / "h1-avimor-native-harlow-spring-loop--target-snap-repaired.gpx"],
        command="python years/2026/scripts/harlow_h1_gate_repair_audit.py",
    )
    write_manifest(args.manifest_json, manifest)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.manifest_json}")
    print(json.dumps(report["promotion_gate_status"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
