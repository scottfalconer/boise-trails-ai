#!/usr/bin/env python3
"""Generate Freestone/Shane's/Three Bears cluster route candidates."""

from __future__ import annotations

import argparse
import json
import re
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
from personal_route_planner import (  # noqa: E402
    DEFAULT_CONNECTOR_GEOJSON,
    DEFAULT_OFFICIAL_GEOJSON,
    haversine_miles,
    load_connector_graph,
    load_official_segments,
    round_miles,
    shortest_connector_path,
)


DEFAULT_FIELD_TOOL_DATA_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_TEMPLATE_CANDIDATES_JSON = YEAR_DIR / "checkpoints" / "common-route-template-candidates-2026-05-12.json"
DEFAULT_CLUSTER_AUDIT_JSON = YEAR_DIR / "checkpoints" / "cluster-route-optimization-audit-2026-05-12.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "checkpoints" / "freestone-route-generation-experiment-2026-05-12"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "freestone-route-generation-experiment-2026-05-12.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "freestone-route-generation-experiment-2026-05-12.md"
DEFAULT_MANIFEST_JSON = YEAR_DIR / "checkpoints" / "freestone-route-generation-experiment-2026-05-12-manifest.json"

TEMPLATE_ID = "freestone-shanes-three-bears-loop"
MATCHED_ROUTE_LABELS = ["FD19C", "FD04A", "3", "FD20A"]
CONTAINED_ROUTE_IDS = {"119-3", "120-1"}
TOUCHED_ROUTE_IDS = {"104-1", "115-1"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def sort_id(value: Any) -> tuple[int, str]:
    text = str(value)
    return (0, f"{int(text):08d}") if text.isdigit() else (1, text)


def normalized_ids(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, int, float)):
        values = [values]
    return sorted({str(value) for value in values if value is not None}, key=sort_id)


def float_value(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def int_value(value: Any) -> int:
    return int(round(float_value(value)))


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "route"


def route_key(route: dict[str, Any]) -> str:
    return str(route.get("outing_id") or route.get("route_key") or route.get("label") or "")


def route_label(route: dict[str, Any]) -> str:
    label = str(route.get("label") or route_key(route))
    outing_id = route.get("outing_id")
    return f"{outing_id}: {label}" if outing_id and str(outing_id) not in label else label


def route_index(routes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for route in routes:
        keys = [route_key(route), str(route.get("outing_id") or ""), str(route.get("label") or ""), route_label(route)]
        for candidate_id in route.get("candidate_ids") or []:
            keys.append(str(candidate_id))
        for key in keys:
            if key:
                index[key] = route
    return index


def current_route_metrics(routes: list[dict[str, Any]]) -> dict[str, Any]:
    segment_ids = sorted(
        {segment_id for route in routes for segment_id in normalized_ids(route.get("segment_ids") or [])},
        key=sort_id,
    )
    return {
        "route_count": len(routes),
        "labels": [route.get("label") for route in routes],
        "route_ids": [route_key(route) for route in routes],
        "on_foot_miles": round_miles(sum(float_value(route.get("on_foot_miles")) for route in routes)),
        "official_miles": round_miles(sum(float_value(route.get("official_miles")) for route in routes)),
        "p75_minutes": sum(int_value(route.get("door_to_door_minutes_p75")) for route in routes),
        "p90_minutes": sum(int_value(route.get("door_to_door_minutes_p90")) for route in routes),
        "segment_ids": segment_ids,
        "segment_count": len(segment_ids),
    }


def natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def template_row(template_candidates: dict[str, Any]) -> dict[str, Any]:
    for template in template_candidates.get("templates_ranked") or []:
        if template.get("template_id") == TEMPLATE_ID:
            return template
    raise ValueError(f"Missing template {TEMPLATE_ID}")


def cluster_bundle_row(cluster_audit: dict[str, Any]) -> dict[str, Any]:
    for row in cluster_audit.get("cluster_bundle_replacements") or []:
        if row.get("template_id") == TEMPLATE_ID:
            return row
    raise ValueError(f"Missing cluster bundle {TEMPLATE_ID}")


def freestone_parking(field_tool_data: dict[str, Any]) -> dict[str, Any]:
    for route in field_tool_data.get("routes") or []:
        if route.get("trailhead") == "Freestone Creek" and (route.get("parking") or {}).get("has_parking"):
            parking = dict(route.get("parking") or {})
            return {
                "name": parking.get("name") or "Freestone Creek Trailhead",
                "lat": float(parking["lat"]),
                "lon": float(parking["lon"]),
                "has_parking": bool(parking.get("has_parking")),
                "parking_confidence": parking.get("parking_confidence"),
                "source": parking.get("source"),
                "nearest_open_trail_name": parking.get("nearest_open_trail_name"),
                "nearest_open_trail_label": parking.get("nearest_open_trail_label"),
            }
    raise ValueError("Missing Freestone Creek parking in field-tool data")


def template_segment_order(template: dict[str, Any], official_by_id: dict[str, dict[str, Any]]) -> list[str]:
    wanted = set(normalized_ids(template.get("candidate_segment_ids") or []))
    by_trail: dict[str, list[dict[str, Any]]] = {}
    for segment_id in wanted:
        segment = official_by_id[segment_id]
        by_trail.setdefault(str(segment["trail_name"]), []).append(segment)
    ordered: list[str] = []
    for trail_name in template.get("trail_sequence") or []:
        trail_segments = sorted(by_trail.get(str(trail_name), []), key=lambda row: natural_key(str(row.get("seg_name"))))
        ordered.extend(str(segment["seg_id"]) for segment in trail_segments)
    missing = wanted - set(ordered)
    if missing:
        ordered.extend(sorted(missing, key=sort_id))
    return ordered


def point_path_miles(coords: list[tuple[float, float]]) -> float:
    return sum(haversine_miles(left, right) for left, right in zip(coords, coords[1:]))


def append_coords(target: list[tuple[float, float]], coords: list[Any]) -> None:
    for coord in coords:
        point = (float(coord[0]), float(coord[1]))
        if target and target[-1] == point:
            continue
        target.append(point)


def graph_path(
    start: tuple[float, float],
    end: tuple[float, float],
    connector_graph: dict[str, Any],
    avoid_official_segment_ids: set[str],
) -> tuple[dict[str, Any], list[tuple[float, float]]]:
    mapped = shortest_connector_path(
        start,
        end,
        connector_graph,
        snap_tolerance_miles=0.02,
        avoid_official_segment_ids={int(segment_id) for segment_id in avoid_official_segment_ids if str(segment_id).isdigit()},
    )
    if mapped:
        return mapped, [(float(lon), float(lat)) for lon, lat in mapped.get("path_coordinates") or []]
    gap = haversine_miles(start, end)
    return (
        {
            "distance_miles": gap,
            "connector_miles": gap,
            "official_repeat_miles": 0.0,
            "official_repeat_segment_ids": [],
            "connector_names": [],
            "connector_classes": [],
            "source": "direct_gap_fallback",
        },
        [start, end],
    )


def generated_route_status(validation: dict[str, Any], direct_gap_miles: float) -> str:
    if not validation.get("passed"):
        return "generated_gpx_has_gaps"
    if direct_gap_miles > 0:
        return "generated_continuous_gpx_with_direct_gap_fallback"
    return "generated_continuous_graph_gpx"


def segment_orientation_options(segment: dict[str, Any], preserve_ascent_direction: bool = True) -> list[bool]:
    if preserve_ascent_direction and segment.get("direction") == "ascent":
        return [False]
    return [False, True]


def choose_segment_orientation(
    current: tuple[float, float],
    segment: dict[str, Any],
    connector_graph: dict[str, Any],
    avoid_official_segment_ids: set[str],
    preserve_ascent_direction: bool = True,
) -> tuple[bool, dict[str, Any], list[tuple[float, float]], list[tuple[float, float]]]:
    options = []
    for reversed_direction in segment_orientation_options(segment, preserve_ascent_direction=preserve_ascent_direction):
        segment_coords = list(reversed(segment["coordinates"])) if reversed_direction else list(segment["coordinates"])
        path, path_coords = graph_path(current, segment_coords[0], connector_graph, avoid_official_segment_ids)
        options.append((float_value(path.get("distance_miles")), reversed_direction, path, path_coords, segment_coords))
    _, reversed_direction, path, path_coords, segment_coords = min(options, key=lambda item: item[0])
    return reversed_direction, path, path_coords, segment_coords


def build_generated_route(
    *,
    variant_id: str,
    strategy: str,
    ordered_segment_ids: list[str],
    parking: dict[str, Any],
    official_by_id: dict[str, dict[str, Any]],
    connector_graph: dict[str, Any],
    preserve_ascent_direction: bool = True,
) -> dict[str, Any]:
    start = (float(parking["lon"]), float(parking["lat"]))
    current = start
    coords: list[tuple[float, float]] = [start]
    remaining = set(ordered_segment_ids)
    traversed_order: list[str] = []
    link_rows: list[dict[str, Any]] = []
    official_repeat_segment_ids: list[str] = []
    connector_miles = 0.0
    official_repeat_miles = 0.0
    direct_gap_miles = 0.0
    while remaining:
        if strategy == "nearest_segment_greedy":
            choices = []
            for segment_id in sorted(remaining, key=sort_id):
                segment = official_by_id[segment_id]
                reversed_direction, path, path_coords, segment_coords = choose_segment_orientation(
                    current,
                    segment,
                    connector_graph,
                    remaining,
                    preserve_ascent_direction=preserve_ascent_direction,
                )
                choices.append((float_value(path.get("distance_miles")), segment_id, reversed_direction, path, path_coords, segment_coords))
            _, segment_id, reversed_direction, path, path_coords, segment_coords = min(choices, key=lambda item: item[0])
        else:
            segment_id = next(segment_id for segment_id in ordered_segment_ids if segment_id in remaining)
            segment = official_by_id[segment_id]
            reversed_direction, path, path_coords, segment_coords = choose_segment_orientation(
                current,
                segment,
                connector_graph,
                remaining,
                preserve_ascent_direction=preserve_ascent_direction,
            )
        segment = official_by_id[segment_id]
        append_coords(coords, path_coords)
        append_coords(coords, segment_coords)
        traversed_order.append(segment_id)
        connector_miles += float_value(path.get("connector_miles"))
        official_repeat_miles += float_value(path.get("official_repeat_miles"))
        official_repeat_segment_ids.extend(str(item) for item in path.get("official_repeat_segment_ids") or [])
        if path.get("source") == "direct_gap_fallback":
            direct_gap_miles += float_value(path.get("distance_miles"))
        link_rows.append(
            {
                "to_segment_id": segment_id,
                "to_segment_name": segment.get("seg_name"),
                "to_trail_name": segment.get("trail_name"),
                "segment_reversed": reversed_direction,
                "link_distance_miles": round_miles(float_value(path.get("distance_miles"))),
                "connector_miles": round_miles(float_value(path.get("connector_miles"))),
                "official_repeat_miles": round_miles(float_value(path.get("official_repeat_miles"))),
                "official_repeat_segment_ids": normalized_ids(path.get("official_repeat_segment_ids") or []),
                "connector_names": sorted(set(str(name) for name in path.get("connector_names") or []))[:12],
                "connector_classes": sorted(set(str(name) for name in path.get("connector_classes") or [])),
                "path_source": path.get("source") or "mapped_graph",
            }
        )
        current = segment_coords[-1]
        remaining.remove(segment_id)
    return_path, return_coords = graph_path(current, start, connector_graph, set())
    append_coords(coords, return_coords)
    connector_miles += float_value(return_path.get("connector_miles"))
    official_repeat_miles += float_value(return_path.get("official_repeat_miles"))
    official_repeat_segment_ids.extend(str(item) for item in return_path.get("official_repeat_segment_ids") or [])
    if return_path.get("source") == "direct_gap_fallback":
        direct_gap_miles += float_value(return_path.get("distance_miles"))
    link_rows.append(
        {
            "to_segment_id": "return_to_car",
            "to_segment_name": "Return to Freestone Creek Trailhead",
            "to_trail_name": None,
            "segment_reversed": None,
            "link_distance_miles": round_miles(float_value(return_path.get("distance_miles"))),
            "connector_miles": round_miles(float_value(return_path.get("connector_miles"))),
            "official_repeat_miles": round_miles(float_value(return_path.get("official_repeat_miles"))),
            "official_repeat_segment_ids": normalized_ids(return_path.get("official_repeat_segment_ids") or []),
            "connector_names": sorted(set(str(name) for name in return_path.get("connector_names") or []))[:12],
            "connector_classes": sorted(set(str(name) for name in return_path.get("connector_classes") or [])),
            "path_source": return_path.get("source") or "mapped_graph",
        }
    )
    dense_coords = densify_coordinates(coords, max_gap_miles=0.03)
    official_miles = sum(float_value(official_by_id[segment_id].get("official_miles")) for segment_id in ordered_segment_ids)
    self_repeat_ids = sorted(set(ordered_segment_ids) & set(str(item) for item in official_repeat_segment_ids), key=sort_id)
    validation = validate_track_segments([dense_coords], max_gap_miles=0.05)
    ascent_segment_ids = [
        segment_id for segment_id in ordered_segment_ids if official_by_id[segment_id].get("direction") == "ascent"
    ]
    ascent_direction_failed_ids = [
        str(row["to_segment_id"])
        for row in link_rows
        if row.get("segment_reversed")
        and official_by_id.get(str(row.get("to_segment_id")), {}).get("direction") == "ascent"
    ]
    cue_complexity = {
        "official_cue_count": len({official_by_id[segment_id]["trail_name"] for segment_id in traversed_order}),
        "connector_or_return_cue_count": len([row for row in link_rows if float_value(row.get("link_distance_miles")) > 0.05]),
        "link_count": len(link_rows),
        "connector_name_count": len({name for row in link_rows for name in row.get("connector_names") or []}),
        "self_repeat_warning_count": len(self_repeat_ids),
    }
    return {
        "variant_id": variant_id,
        "strategy": strategy,
        "parking": parking,
        "status": generated_route_status(validation, direct_gap_miles),
        "official_segment_ids": normalized_ids(ordered_segment_ids),
        "traversed_segment_order": traversed_order,
        "official_miles": round_miles(official_miles),
        "track_miles": round_miles(point_path_miles(dense_coords)),
        "connector_miles": round_miles(connector_miles),
        "official_repeat_miles": round_miles(official_repeat_miles),
        "direct_gap_fallback_miles": round_miles(direct_gap_miles),
        "self_repeat_segment_ids": self_repeat_ids,
        "non_template_repeat_segment_ids": sorted(set(str(item) for item in official_repeat_segment_ids) - set(ordered_segment_ids), key=sort_id),
        "ascent_direction_validation": {
            "status": "passed_no_ascent_segments"
            if not ascent_segment_ids
            else ("passed_ascent_direction_preserved" if not ascent_direction_failed_ids else "failed_ascent_direction"),
            "ascent_segment_ids": ascent_segment_ids,
            "direction_failed_segment_ids": ascent_direction_failed_ids,
        },
        "coverage_validation": {
            "status": "covers_template_segment_set" if set(traversed_order) == set(ordered_segment_ids) else "coverage_gap",
            "covered_segment_count": len(set(traversed_order)),
            "required_segment_count": len(set(ordered_segment_ids)),
            "missing_segment_ids": normalized_ids(set(ordered_segment_ids) - set(traversed_order)),
        },
        "gpx_validation": validation,
        "cue_complexity": cue_complexity,
        "link_rows": link_rows,
        "track_coordinates": dense_coords,
    }


def scaled_minutes(candidate_miles: float, current_miles: float, current_minutes: int) -> int:
    if current_miles <= 0:
        return 0
    return int(round(candidate_miles * (current_minutes / current_miles)))


def candidate_summary(
    candidate: dict[str, Any],
    current_all: dict[str, Any],
    contained_current: dict[str, Any],
    touched_current: dict[str, Any],
    uncovered_ids: list[str],
) -> dict[str, Any]:
    p75 = scaled_minutes(candidate["track_miles"], current_all["on_foot_miles"], current_all["p75_minutes"])
    p90 = scaled_minutes(candidate["track_miles"], current_all["on_foot_miles"], current_all["p90_minutes"])
    direct_replace_contained_delta = round_miles(candidate["track_miles"] - contained_current["on_foot_miles"])
    keep_touched_delta = round_miles(candidate["track_miles"] + touched_current["on_foot_miles"] - current_all["on_foot_miles"])
    direct_gap_miles = float_value(candidate.get("direct_gap_fallback_miles"))
    gpx_gate = "passes_graph_validated_continuity" if candidate["gpx_validation"]["passed"] and direct_gap_miles == 0 else (
        "needs_direct_gap_review" if candidate["gpx_validation"]["passed"] else "failed_continuity"
    )
    repeat_gate = "needs_repeat_credit_review" if (
        candidate["self_repeat_segment_ids"] or candidate["non_template_repeat_segment_ids"]
    ) else "passes_no_hidden_repeat"
    return {
        "variant_id": candidate["variant_id"],
        "strategy": candidate["strategy"],
        "status": candidate["status"],
        "track_miles": candidate["track_miles"],
        "official_miles": candidate["official_miles"],
        "connector_miles": candidate["connector_miles"],
        "official_repeat_miles": candidate["official_repeat_miles"],
        "direct_gap_fallback_miles": candidate["direct_gap_fallback_miles"],
        "p75_minutes_scaled": p75,
        "p90_minutes_scaled": p90,
        "pricing_status": "scaled_from_current_cluster_route_cards_needs_dem_and_field_calibration",
        "coverage_status": candidate["coverage_validation"]["status"],
        "gpx_status": "passed" if candidate["gpx_validation"]["passed"] else "failed",
        "hidden_repeat_or_latent_credit_status": {
            "self_repeat_segment_ids": candidate["self_repeat_segment_ids"],
            "non_template_repeat_segment_ids": candidate["non_template_repeat_segment_ids"],
            "status": "has_self_repeat_review_needed" if candidate["self_repeat_segment_ids"] else "no_template_self_repeat_in_connector_links",
        },
        "ascent_direction_validation": candidate["ascent_direction_validation"],
        "cue_complexity": candidate["cue_complexity"],
        "promotion_gates": {
            "overall": "blocked_not_active_route_card",
            "coverage": "covers_template_only_current_bundle_still_uncovered" if uncovered_ids else "covers_current_bundle_segments",
            "continuous_gpx": gpx_gate,
            "timing": "scaled_estimate_needs_dem_and_field_calibration",
            "access": "uses_existing_field_packet_parking_anchor_needs_current_condition_recheck",
            "cues": "needs_human_cue_rewrite",
            "repeat_credit": repeat_gate,
            "recertification": "not_run_against_active_field_packet",
        },
        "comparison": {
            "current_all_four_on_foot_miles": current_all["on_foot_miles"],
            "current_all_four_p75_minutes": current_all["p75_minutes"],
            "current_contained_only_on_foot_miles": contained_current["on_foot_miles"],
            "direct_replace_contained_routes_delta_miles": direct_replace_contained_delta,
            "candidate_plus_unshrunk_touched_routes_delta_miles": keep_touched_delta,
            "uncovered_current_segment_ids_if_replacing_all_four": uncovered_ids,
            "uncovered_current_segment_count": len(uncovered_ids),
            "replacement_readiness": "not_direct_replacement_needs_additional_loops_or_shrunk_cards"
            if uncovered_ids
            else "candidate_covers_current_claimed_segments",
        },
    }


def render_route_gpx(output_dir: Path, candidate: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{slugify(candidate['variant_id'])}.gpx"
    path.write_text(
        render_gpx_segments(
            f"Freestone cluster experiment: {candidate['variant_id']}",
            [candidate["track_coordinates"]],
        ),
        encoding="utf-8",
    )
    return path


def build_report(
    field_tool_data: dict[str, Any],
    template_candidates: dict[str, Any],
    cluster_audit: dict[str, Any],
    official_segments: list[dict[str, Any]],
    connector_graph: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    template = template_row(template_candidates)
    bundle = cluster_bundle_row(cluster_audit)
    routes_by_key = route_index(field_tool_data.get("routes") or [])
    matched_routes = [routes_by_key[label] for label in MATCHED_ROUTE_LABELS]
    contained_routes = [route for route in matched_routes if route_key(route) in CONTAINED_ROUTE_IDS]
    touched_routes = [route for route in matched_routes if route_key(route) in TOUCHED_ROUTE_IDS]
    current_all = current_route_metrics(matched_routes)
    contained_current = current_route_metrics(contained_routes)
    touched_current = current_route_metrics(touched_routes)
    official_by_id = {str(segment["seg_id"]): segment for segment in official_segments}
    parking = freestone_parking(field_tool_data)
    ordered_segment_ids = template_segment_order(template, official_by_id)
    generated = [
        build_generated_route(
            variant_id="template-sequence-greedy",
            strategy="template_sequence_greedy",
            ordered_segment_ids=ordered_segment_ids,
            parking=parking,
            official_by_id=official_by_id,
            connector_graph=connector_graph,
        ),
        build_generated_route(
            variant_id="nearest-segment-greedy",
            strategy="nearest_segment_greedy",
            ordered_segment_ids=ordered_segment_ids,
            parking=parking,
            official_by_id=official_by_id,
            connector_graph=connector_graph,
        ),
    ]
    gpx_outputs = []
    summaries = []
    uncovered_ids = normalized_ids(bundle.get("uncovered_current_segment_ids") or [])
    for candidate in generated:
        gpx_path = render_route_gpx(output_dir, candidate)
        gpx_outputs.append(gpx_path)
        summary = candidate_summary(candidate, current_all, contained_current, touched_current, uncovered_ids)
        summary["gpx_path"] = display_path(gpx_path)
        summaries.append(summary)
        candidate["gpx_path"] = display_path(gpx_path)
        candidate.pop("track_coordinates", None)
    best = min(summaries, key=lambda row: float_value(row["track_miles"]))
    return {
        "schema": "boise_trails_freestone_cluster_route_generation_experiment_v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "generated_candidates_not_direct_replacements",
        "source_files": {
            "field_tool_data_json": display_path(DEFAULT_FIELD_TOOL_DATA_JSON),
            "template_candidates_json": display_path(DEFAULT_TEMPLATE_CANDIDATES_JSON),
            "cluster_route_optimization_audit_json": display_path(DEFAULT_CLUSTER_AUDIT_JSON),
            "official_segments_geojson": display_path(DEFAULT_OFFICIAL_GEOJSON),
            "connector_geojson": display_path(DEFAULT_CONNECTOR_GEOJSON),
        },
        "scope": {
            "purpose": "First true route-generation experiment for the Freestone/Shane's/Three Bears top archetype mismatch.",
            "promotion_policy": "Generated GPX candidates are experiment outputs only. They are not active route cards until p75/DEM, access, cues, coverage, ascent direction, and future-day recertification pass.",
            "template_id": TEMPLATE_ID,
        },
        "template": {
            "template_id": template.get("template_id"),
            "target_area": template.get("target_area"),
            "normal_start": template.get("normal_start"),
            "normal_direction": template.get("normal_direction"),
            "trail_sequence": template.get("trail_sequence"),
            "candidate_segment_ids": normalized_ids(template.get("candidate_segment_ids") or []),
            "candidate_official_miles": template.get("candidate_official_miles"),
            "status": template.get("status"),
            "optimizer_attention_score": template.get("optimizer_attention_score"),
            "current_route_pressure": template.get("current_route_pressure"),
        },
        "current": {
            "matched_cards": current_all,
            "contained_replacement_cards": contained_current,
            "touched_partial_cards": touched_current,
            "bundle_status": bundle.get("replacement_status"),
            "uncovered_current_segment_ids": uncovered_ids,
            "uncovered_current_segment_count": len(uncovered_ids),
        },
        "summary": {
            "variant_count": len(summaries),
            "best_variant_id": best["variant_id"],
            "best_variant_track_miles": best["track_miles"],
            "best_variant_scaled_p75_minutes": best["p75_minutes_scaled"],
            "current_all_four_on_foot_miles": current_all["on_foot_miles"],
            "current_all_four_p75_minutes": current_all["p75_minutes"],
            "uncovered_current_segment_count": len(uncovered_ids),
            "recommendation": "use_generated_gpx_as_shrink_seed_not_active_replacement",
        },
        "candidate_summaries": summaries,
        "generated_candidates": generated,
        "outputs": {
            "gpx_paths": [display_path(path) for path in gpx_outputs],
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    current = report.get("current") or {}
    lines = [
        "# Freestone Cluster Route Generation Experiment",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        f"Status: `{report.get('status')}`",
        "",
        "## Summary",
        "",
        f"- Best generated variant: `{summary.get('best_variant_id')}`",
        f"- Best variant on-foot: {summary.get('best_variant_track_miles')} mi",
        f"- Best variant scaled p75: {summary.get('best_variant_scaled_p75_minutes')} min",
        f"- Current four matched cards: {summary.get('current_all_four_on_foot_miles')} mi / {summary.get('current_all_four_p75_minutes')} p75",
        f"- Uncovered current segment IDs if replacing all four: {summary.get('uncovered_current_segment_count')}",
        "",
        "This is a route-generation experiment, not a route-card promotion. The generated GPX covers the template segment set, but the cluster bundle still has uncovered current IDs, so it is not a direct replacement.",
        "",
        "## Generated Variants",
        "",
        "| Variant | Status | GPX | On-foot | P75/P90 scaled | Connector | Repeat | Coverage | Cue load | Replacement readout |",
        "|---|---|---|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in report.get("candidate_summaries") or []:
        cue = row.get("cue_complexity") or {}
        comparison = row.get("comparison") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('variant_id')}`",
                    f"`{row.get('status')}`",
                    str(row.get("gpx_path")),
                    str(row.get("track_miles")),
                    f"{row.get('p75_minutes_scaled')}/{row.get('p90_minutes_scaled')}",
                    str(row.get("connector_miles")),
                    str(row.get("official_repeat_miles")),
                    str(row.get("coverage_status")),
                    str(cue.get("official_cue_count", 0) + cue.get("connector_or_return_cue_count", 0)),
                    comparison.get("replacement_readiness"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Current Bundle Gap",
            "",
            f"- Bundle status: `{current.get('bundle_status')}`",
            f"- Contained cards fully covered by template: {(current.get('contained_replacement_cards') or {}).get('labels')}",
            f"- Partial cards touched but not replaced: {(current.get('touched_partial_cards') or {}).get('labels')}",
            f"- Uncovered IDs: {', '.join(current.get('uncovered_current_segment_ids') or [])}",
            "",
            "## Gate Notes",
            "",
            "- The nearest-segment variant is shorter, but less like the public/common route archetype.",
            "- The template-sequence variant preserves the intended common-route order but is much longer because graph connector links are expensive.",
            "- Next useful work is generating shrink/leftover loops for the 19 uncovered IDs, not promoting this template GPX by itself.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--template-candidates-json", type=Path, default=DEFAULT_TEMPLATE_CANDIDATES_JSON)
    parser.add_argument("--cluster-audit-json", type=Path, default=DEFAULT_CLUSTER_AUDIT_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--connector-geojson", type=Path, default=DEFAULT_CONNECTOR_GEOJSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    field_tool_data = read_json(args.field_tool_data_json)
    template_candidates = read_json(args.template_candidates_json)
    cluster_audit = read_json(args.cluster_audit_json)
    official_segments, _official_meta = load_official_segments(args.official_geojson)
    connector_graph = load_connector_graph(args.connector_geojson, official_segments=official_segments)
    report = build_report(
        field_tool_data,
        template_candidates,
        cluster_audit,
        official_segments,
        connector_graph,
        args.output_dir,
    )
    report["source_files"] = {
        "field_tool_data_json": display_path(args.field_tool_data_json),
        "template_candidates_json": display_path(args.template_candidates_json),
        "cluster_route_optimization_audit_json": display_path(args.cluster_audit_json),
        "official_segments_geojson": display_path(args.official_geojson),
        "connector_geojson": display_path(args.connector_geojson),
    }
    write_json(args.output_json, report)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    output_paths = [args.output_json, args.output_md] + [Path(path) for path in report["outputs"]["gpx_paths"]]
    manifest = build_artifact_manifest(
        run_id="freestone_cluster_route_generation_experiment",
        command="python years/2026/scripts/freestone_cluster_route_generation_experiment.py",
        inputs=[
            args.field_tool_data_json,
            args.template_candidates_json,
            args.cluster_audit_json,
            args.official_geojson,
            args.connector_geojson,
        ],
        outputs=output_paths,
        metadata={"status": report.get("status"), "summary": report.get("summary")},
    )
    write_manifest(args.manifest_json, manifest)
    print(f"Wrote {display_path(args.output_json)}")
    print(f"Wrote {display_path(args.output_md)}")
    print(f"Wrote {display_path(args.manifest_json)}")
    for path in report["outputs"]["gpx_paths"]:
        print(f"Wrote {path}")
    print(json.dumps({"status": report["status"], "summary": report["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
