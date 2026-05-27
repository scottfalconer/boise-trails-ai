#!/usr/bin/env python3
"""Audit post-credit exits against the shortest legal connector graph path."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402
from field_activity_review import activity_coordinates, normalized_ids  # noqa: E402
from personal_route_planner import (  # noqa: E402
    DEFAULT_CONNECTOR_GEOJSON,
    DEFAULT_OFFICIAL_GEOJSON,
    line_length_miles,
    load_connector_graph,
    load_official_segments,
    shortest_connector_path,
)
from route_repeat_optimization_audit import interval_coordinates  # noqa: E402


DEFAULT_FIELD_TOOL_DATA_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_PACKET_DIR = REPO_ROOT / "docs" / "field-packet"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "post-credit-connector-audit-2026-05-27.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "post-credit-connector-audit-2026-05-27.md"
DEFAULT_MANIFEST_JSON = YEAR_DIR / "checkpoints" / "post-credit-connector-audit-2026-05-27-manifest.json"
DEFAULT_DISTANCE_TOLERANCE_MILES = 0.005
DEFAULT_SNAP_TOLERANCE_MILES = 0.045
DEFAULT_ROUTE_MILEAGE_TOLERANCE_MILES = 0.25

POST_CREDIT_CONNECTOR_TYPES = {
    "connector_named_trail",
    "connector_road",
    "repeat_official_noncredit",
    "overlap_repeat",
    "exit_access",
    "return_to_car",
    "car_pass_connector",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def float_value(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def sort_id(value: Any) -> tuple[int, str]:
    text = str(value)
    return (0, f"{int(text):08d}") if text.isdigit() else (1, text)


def sorted_ids(values: Any) -> list[str]:
    return sorted(set(normalized_ids(values or [])), key=sort_id)


def route_key(route: dict[str, Any]) -> str:
    return str(route.get("outing_id") or route.get("label") or "unknown-route")


def route_label(route: dict[str, Any]) -> str:
    label = str(route.get("label") or route_key(route))
    outing_id = route.get("outing_id")
    if outing_id and str(outing_id) not in label:
        return f"{outing_id}: {label}"
    return label


def route_gpx_path(route: dict[str, Any], packet_dir: Path) -> Path | None:
    href = route.get("gpx_href")
    if not href:
        return None
    path = Path(str(href))
    return path if path.is_absolute() else packet_dir / path


def route_card_miles(route: dict[str, Any]) -> float:
    return float_value(route.get("on_foot_miles") or route.get("distance_miles") or route.get("total_miles"))


def coordinate_pair(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None


def cue_source_path_coords(cue: dict[str, Any]) -> list[tuple[float, float]]:
    coords = [point for point in (coordinate_pair(value) for value in cue.get("source_path_coordinates") or []) if point]
    return coords if len(coords) >= 2 else []


def cue_planned_leg_miles(cue: dict[str, Any]) -> float:
    for key in ("route_leg_miles", "source_leg_miles", "leg_miles"):
        value = cue.get(key)
        if value is not None:
            return float_value(value)
    return 0.0


def cue_card_leg_miles(cue: dict[str, Any]) -> float:
    for key in ("leg_miles", "source_leg_miles", "route_leg_miles"):
        value = cue.get(key)
        if value is not None:
            return float_value(value)
    return 0.0


def cue_source_leg_miles(cue: dict[str, Any]) -> float:
    for key in ("source_leg_miles", "leg_miles", "route_leg_miles"):
        value = cue.get(key)
        if value is not None:
            return float_value(value)
    return 0.0


def cue_official_credit_miles(cue: dict[str, Any]) -> float:
    for key in ("official_miles", "segment_official_miles"):
        value = cue.get(key)
        if value is not None:
            return float_value(value)
    return float_value(cue.get("source_leg_miles"))


def cue_interval_coords(
    route_coords: list[tuple[float, float]],
    cue: dict[str, Any],
) -> list[tuple[float, float]]:
    source_coords = cue_source_path_coords(cue)
    if source_coords:
        return source_coords
    start = cue.get("route_miles")
    length = cue.get("route_leg_miles")
    if length is not None and float_value(length) <= 0 and float_value(cue.get("leg_miles")) > 0:
        start = cue.get("cum_miles")
        length = cue.get("leg_miles")
    if start is None or length is None:
        start = cue.get("cum_miles")
        length = cue.get("leg_miles")
    if start is None or length is None:
        return []
    return interval_coordinates(
        route_coords,
        start_mile=float_value(start),
        end_mile=float_value(start) + float_value(length),
    )


def shortest_exit_path(
    coords: list[tuple[float, float]],
    connector_graph: dict[str, Any] | None,
    *,
    snap_tolerance_miles: float,
    avoid_official_segment_ids: set[int],
) -> dict[str, Any] | None:
    if len(coords) < 2:
        return None
    return shortest_connector_path(
        coords[0],
        coords[-1],
        connector_graph,
        snap_tolerance_miles,
        avoid_official_segment_ids=avoid_official_segment_ids,
    )


def repeat_ids_for_cue(cue: dict[str, Any]) -> set[str]:
    return set(sorted_ids(cue.get("official_repeat_segment_ids") or []))


def official_repeat_miles_for_cue(cue: dict[str, Any]) -> float:
    return float_value(cue.get("official_repeat_miles"))


def shortest_result_payload(shortest: dict[str, Any], actual_miles: float) -> dict[str, Any]:
    shortest_miles = float_value(shortest.get("distance_miles"))
    savings = actual_miles - shortest_miles
    return {
        "shortest_miles": round(shortest_miles, 4),
        "savings_miles": round(savings, 4),
        "savings_feet": round(savings * 5280),
        "shortest_connector_names": shortest.get("connector_names") or [],
        "shortest_connector_classes": shortest.get("connector_classes") or [],
        "shortest_official_repeat_segment_ids": sorted_ids(shortest.get("official_repeat_segment_ids") or []),
        "snap_start_miles": round(float_value(shortest.get("snap_start_miles")), 4),
        "snap_end_miles": round(float_value(shortest.get("snap_end_miles")), 4),
    }


def connector_finding(
    *,
    route: dict[str, Any],
    cue: dict[str, Any],
    coords: list[tuple[float, float]],
    connector_graph: dict[str, Any] | None,
    credited_ids: set[str],
    required_ids: set[str],
    distance_tolerance_miles: float,
    snap_tolerance_miles: float,
) -> dict[str, Any] | None:
    planned_miles = cue_planned_leg_miles(cue)
    source_coords = cue_source_path_coords(cue)
    if source_coords:
        source_declared_miles = cue_source_leg_miles(cue)
        actual_miles = source_declared_miles if source_declared_miles > 0 else line_length_miles(source_coords)
    else:
        actual_miles = planned_miles if planned_miles > 0 else line_length_miles(coords)
    if (
        cue.get("source_leg_miles") is not None
        and float_value(cue.get("source_leg_miles")) <= distance_tolerance_miles
        and float_value(cue.get("route_leg_miles")) <= distance_tolerance_miles
    ):
        return None
    if actual_miles <= distance_tolerance_miles and len(coords) < 2:
        return None
    remaining_required_ids = required_ids - credited_ids
    avoid_ids = {int(seg_id) for seg_id in remaining_required_ids if str(seg_id).isdigit()}
    shortest = shortest_exit_path(
        coords,
        connector_graph,
        snap_tolerance_miles=snap_tolerance_miles,
        avoid_official_segment_ids=avoid_ids,
    )
    base = {
        "code": "post_credit_exit_shortest_connector",
        "route_key": route_key(route),
        "label": route_label(route),
        "outing_id": route.get("outing_id"),
        "candidate_ids": [str(value) for value in route.get("candidate_ids") or []],
        "seq": cue.get("seq"),
        "cue_type": cue.get("cue_type"),
        "signed_as": cue.get("signed_as") or [],
        "target": cue.get("target"),
        "actual_miles": round(actual_miles, 4),
        "credited_segment_ids_before_cue": sorted_ids(credited_ids),
        "avoided_unearned_route_segment_ids": sorted_ids(remaining_required_ids),
    }
    if len(coords) < 2:
        return {
            **base,
            "status": "failed",
            "failure_code": "missing_cue_interval_geometry",
            "message": "Post-credit connector/return cue has no route interval geometry to prove.",
        }
    if shortest is None:
        if source_coords and actual_miles <= snap_tolerance_miles:
            return {**base, "status": "passed", "short_connector_within_snap_tolerance": True}
        repeat_ids = repeat_ids_for_cue(cue)
        if repeat_ids and repeat_ids <= set(credited_ids):
            repeat_miles = official_repeat_miles_for_cue(cue)
            if repeat_miles <= 0 or abs(actual_miles - repeat_miles) <= max(snap_tolerance_miles, actual_miles * 0.02):
                return {
                    **base,
                    "status": "passed",
                    "credited_official_repeat_return_proven": True,
                    "official_repeat_segment_ids": sorted_ids(repeat_ids),
                    "official_repeat_miles": round(repeat_miles, 4),
                }
        shortest_with_target_slivers = shortest_exit_path(
            coords,
            connector_graph,
            snap_tolerance_miles=snap_tolerance_miles,
            avoid_official_segment_ids=set(),
        )
        if shortest_with_target_slivers:
            repeat_miles = float_value(shortest_with_target_slivers.get("official_repeat_miles"))
            repeat_ids = set(sorted_ids(shortest_with_target_slivers.get("official_repeat_segment_ids") or []))
            if repeat_ids <= set(sorted_ids(remaining_required_ids)) and repeat_miles <= snap_tolerance_miles:
                return {
                    **base,
                    **shortest_result_payload(shortest_with_target_slivers, actual_miles),
                    "status": "passed",
                    "target_official_sliver_within_snap_tolerance": True,
                }
        return {
            **base,
            "status": "failed",
            "failure_code": "no_legal_connector_path_proven",
            "message": "No legal graph path was found for this post-credit exit while avoiding unearned route segments.",
        }
    result = {
        **base,
        **shortest_result_payload(shortest, actual_miles),
    }
    if result["savings_miles"] > distance_tolerance_miles:
        return {
            **result,
            "status": "failed",
            "failure_code": "shorter_legal_connector_found",
            "message": "A shorter legal connector graph path exists for this post-credit exit.",
        }
    return {**result, "status": "passed"}


def hidden_exit_finding(
    *,
    route: dict[str, Any],
    cue: dict[str, Any],
    distance_tolerance_miles: float,
) -> dict[str, Any] | None:
    official_ids = sorted_ids(cue.get("official_segment_ids") or [])
    if not official_ids:
        return None
    official_basis = cue_official_credit_miles(cue)
    if official_basis <= 0:
        return None
    actual_leg = cue_source_leg_miles(cue)
    hidden_miles = actual_leg - official_basis
    if hidden_miles <= distance_tolerance_miles:
        return None
    return {
        "code": "official_credit_cue_hides_post_credit_exit",
        "status": "failed",
        "route_key": route_key(route),
        "label": route_label(route),
        "outing_id": route.get("outing_id"),
        "candidate_ids": [str(value) for value in route.get("candidate_ids") or []],
        "seq": cue.get("seq"),
        "cue_type": cue.get("cue_type"),
        "official_segment_ids": official_ids,
        "official_basis_miles": round(float_value(official_basis), 4),
        "actual_route_leg_miles": round(actual_leg, 4),
        "hidden_exit_miles": round(hidden_miles, 4),
        "hidden_exit_feet": round(hidden_miles * 5280),
        "message": "Official-credit cue contains extra movement that must be split into an explicit post-credit connector/return cue before shortest-path proof is possible.",
    }


def audit_route(
    route: dict[str, Any],
    *,
    packet_dir: Path,
    connector_graph: dict[str, Any] | None,
    distance_tolerance_miles: float,
    snap_tolerance_miles: float,
    route_mileage_tolerance_miles: float,
) -> dict[str, Any]:
    required_ids = set(sorted_ids(route.get("segment_ids") or []))
    gpx_path = route_gpx_path(route, packet_dir)
    route_coords = activity_coordinates(gpx_path) if gpx_path and gpx_path.exists() else []
    credited_ids: set[str] = set()
    findings = []
    warnings = []
    connector_proofs = []

    if not route_coords:
        findings.append(
            {
                "code": "missing_route_gpx",
                "status": "failed",
                "route_key": route_key(route),
                "label": route_label(route),
                "outing_id": route.get("outing_id"),
                "gpx_path": display_path(gpx_path),
                "message": "No navigation GPX was available to prove post-credit connector exits.",
            }
        )

    if route_coords:
        gpx_miles = line_length_miles(route_coords)
        card_miles = route_card_miles(route)
        mileage_delta = gpx_miles - card_miles
        tolerance = max(route_mileage_tolerance_miles, card_miles * 0.08)
        if card_miles > 0 and abs(mileage_delta) > tolerance:
            warnings.append(
                {
                    "code": "route_card_gpx_mileage_mismatch_warning",
                    "status": "warning",
                    "route_key": route_key(route),
                    "label": route_label(route),
                    "outing_id": route.get("outing_id"),
                    "candidate_ids": [str(value) for value in route.get("candidate_ids") or []],
                    "card_on_foot_miles": round(card_miles, 4),
                    "gpx_track_miles": round(gpx_miles, 4),
                    "delta_miles": round(mileage_delta, 4),
                    "delta_feet": round(mileage_delta * 5280),
                    "tolerance_miles": round(tolerance, 4),
                    "message": "Route card mileage and exported GPX mileage disagree; route cards remain mileage authority, while cue intervals and graph paths remain connector-proof authority.",
                }
            )

    source_gap_repair = route.get("source_gap_repair") or {}
    if (
        int(source_gap_repair.get("raw_inter_segment_gap_count") or 0) > 0
        or int(source_gap_repair.get("repaired_inter_segment_gap_count") or 0) > 0
        or int(source_gap_repair.get("remaining_inter_segment_gap_count") or 0) > 0
    ):
        findings.append(
            {
                "code": "route_source_gap_repair_prevents_post_credit_proof",
                "status": "failed",
                "route_key": route_key(route),
                "label": route_label(route),
                "outing_id": route.get("outing_id"),
                "candidate_ids": [str(value) for value in route.get("candidate_ids") or []],
                "source_gap_repair": source_gap_repair,
                "message": "Route GPX depends on source gap repair, so post-credit connector intervals are not a continuous route-source proof.",
            }
        )
        return {
            "route_key": route_key(route),
            "outing_id": route.get("outing_id"),
            "label": route_label(route),
            "candidate_ids": [str(value) for value in route.get("candidate_ids") or []],
            "audit_status": "failed",
            "gpx_path": display_path(gpx_path),
            "segment_ids": sorted_ids(required_ids),
            "post_credit_connector_proofs": connector_proofs,
            "findings": findings,
            "warnings": warnings,
        }

    for cue in route.get("wayfinding_cues") or []:
        hidden = hidden_exit_finding(route=route, cue=cue, distance_tolerance_miles=distance_tolerance_miles)
        if hidden:
            findings.append(hidden)

        cue_type = str(cue.get("cue_type") or "")
        if credited_ids and cue_type in POST_CREDIT_CONNECTOR_TYPES:
            coords = cue_interval_coords(route_coords, cue)
            proof = connector_finding(
                route=route,
                cue=cue,
                coords=coords,
                connector_graph=connector_graph,
                credited_ids=credited_ids,
                required_ids=required_ids,
                distance_tolerance_miles=distance_tolerance_miles,
                snap_tolerance_miles=snap_tolerance_miles,
            )
            if proof:
                connector_proofs.append(proof)
                if proof["status"] != "passed":
                    findings.append(proof)

        credited_ids.update(sorted_ids(cue.get("official_segment_ids") or []))

    return {
        "route_key": route_key(route),
        "outing_id": route.get("outing_id"),
        "label": route_label(route),
        "candidate_ids": [str(value) for value in route.get("candidate_ids") or []],
        "audit_status": "failed" if findings else "passed",
        "gpx_path": display_path(gpx_path),
        "segment_ids": sorted_ids(required_ids),
        "post_credit_connector_proofs": connector_proofs,
        "findings": findings,
        "warnings": warnings,
    }


def build_post_credit_connector_audit(
    field_tool_data: dict[str, Any],
    *,
    packet_dir: Path,
    connector_graph: dict[str, Any] | None,
    connector_graph_path: Path | None = None,
    distance_tolerance_miles: float = DEFAULT_DISTANCE_TOLERANCE_MILES,
    snap_tolerance_miles: float = DEFAULT_SNAP_TOLERANCE_MILES,
    route_mileage_tolerance_miles: float = DEFAULT_ROUTE_MILEAGE_TOLERANCE_MILES,
    source_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    routes = [
        audit_route(
            route,
            packet_dir=packet_dir,
            connector_graph=connector_graph,
            distance_tolerance_miles=distance_tolerance_miles,
            snap_tolerance_miles=snap_tolerance_miles,
            route_mileage_tolerance_miles=route_mileage_tolerance_miles,
        )
        for route in field_tool_data.get("routes") or []
    ]
    failed_routes = [route for route in routes if route["audit_status"] != "passed"]
    findings = [finding for route in routes for finding in route.get("findings") or []]
    warnings = [warning for route in routes for warning in route.get("warnings") or []]
    connector_proofs = [proof for route in routes for proof in route.get("post_credit_connector_proofs") or []]
    return {
        "schema": "boise_trails_post_credit_connector_audit_v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "objective": "prove each post-credit exit uses the shortest legal connector graph path",
        "status": "failed" if failed_routes else "passed",
        "summary": {
            "route_count": len(routes),
            "failed_route_count": len(failed_routes),
            "finding_count": len(findings),
            "warning_count": len(warnings),
            "post_credit_connector_proof_count": len(connector_proofs),
            "hidden_exit_finding_count": sum(1 for finding in findings if finding.get("code") == "official_credit_cue_hides_post_credit_exit"),
            "shorter_connector_finding_count": sum(1 for finding in findings if finding.get("failure_code") == "shorter_legal_connector_found"),
            "unproved_connector_finding_count": sum(1 for finding in findings if finding.get("failure_code") in {"missing_cue_interval_geometry", "no_legal_connector_path_proven"}),
            "source_gap_proof_blocker_count": sum(
                1 for finding in findings if finding.get("code") == "route_source_gap_repair_prevents_post_credit_proof"
            ),
            "route_card_gpx_mismatch_count": sum(
                1 for warning in warnings if warning.get("code") == "route_card_gpx_mileage_mismatch_warning"
            ),
        },
        "parameters": {
            "distance_tolerance_miles": distance_tolerance_miles,
            "snap_tolerance_miles": snap_tolerance_miles,
            "route_mileage_tolerance_miles": route_mileage_tolerance_miles,
        },
        "source_files": source_files or {},
        "connector_graph": {
            "path": display_path(connector_graph_path),
            "loaded": bool(connector_graph),
        },
        "findings": findings,
        "warnings": warnings,
        "failed_routes": failed_routes,
        "routes": routes,
    }


def render_markdown(audit: dict[str, Any]) -> str:
    summary = audit["summary"]
    lines = [
        "# Post-Credit Connector Audit",
        "",
        f"Generated: {audit['generated_at']}",
        f"Status: `{audit['status']}`",
        "",
        "## Summary",
        "",
        f"- Routes audited: {summary['route_count']}",
        f"- Failed routes: {summary['failed_route_count']}",
        f"- Findings: {summary['finding_count']}",
        f"- Warnings: {summary.get('warning_count', 0)}",
        f"- Explicit post-credit connector proofs: {summary['post_credit_connector_proof_count']}",
        f"- Hidden official-cue exit findings: {summary['hidden_exit_finding_count']}",
        f"- Shorter connector findings: {summary['shorter_connector_finding_count']}",
        f"- Unproved connector findings: {summary['unproved_connector_finding_count']}",
        f"- Source-gap proof blockers: {summary.get('source_gap_proof_blocker_count', 0)}",
        f"- Route-card/GPX mileage warnings: {summary.get('route_card_gpx_mismatch_count', 0)}",
        "",
        "## Findings",
        "",
    ]
    if not audit.get("findings"):
        lines.append("No findings.")
    else:
        lines.extend(
            [
                "| Route | Cue | Code | Miles | Feet | Message |",
                "|---|---:|---|---:|---:|---|",
            ]
        )
        for finding in audit.get("findings") or []:
            code = finding.get("failure_code") or finding.get("code")
            miles = finding.get("savings_miles")
            if miles is None:
                miles = finding.get("hidden_exit_miles")
            if miles is None:
                miles = finding.get("delta_miles")
            feet = finding.get("savings_feet")
            if feet is None:
                feet = finding.get("hidden_exit_feet")
            if feet is None:
                feet = finding.get("delta_feet")
            lines.append(
                "| {route} | {seq} | {code} | {miles} | {feet} | {message} |".format(
                    route=finding.get("label"),
                    seq=finding.get("seq"),
                    code=code,
                    miles="" if miles is None else miles,
                    feet="" if feet is None else feet,
                    message=finding.get("message"),
                )
            )
    if audit.get("warnings"):
        lines.extend(
            [
                "",
                "## Warnings",
                "",
                "| Route | Code | Miles | Feet | Message |",
                "|---|---|---:|---:|---|",
            ]
        )
        for warning in audit.get("warnings") or []:
            lines.append(
                "| {route} | {code} | {miles} | {feet} | {message} |".format(
                    route=warning.get("label"),
                    code=warning.get("code"),
                    miles=warning.get("delta_miles", ""),
                    feet=warning.get("delta_feet", ""),
                    message=warning.get("message"),
                )
            )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- This audit proves generated field-packet cue intervals, not abstract segment lists.",
            "- It excludes still-unearned route segments from connector alternatives so a shortcut cannot silently consume future official credit.",
            "- If an official-credit cue contains extra movement, the route must be regenerated with that movement represented as an explicit connector or return cue before shortest-path proof is possible.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--packet-dir", type=Path, default=DEFAULT_PACKET_DIR)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--connector-geojson", type=Path, default=DEFAULT_CONNECTOR_GEOJSON)
    parser.add_argument("--distance-tolerance-miles", type=float, default=DEFAULT_DISTANCE_TOLERANCE_MILES)
    parser.add_argument("--snap-tolerance-miles", type=float, default=DEFAULT_SNAP_TOLERANCE_MILES)
    parser.add_argument("--route-mileage-tolerance-miles", type=float, default=DEFAULT_ROUTE_MILEAGE_TOLERANCE_MILES)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    field_tool_data = read_json(args.field_tool_data_json)
    official_segments, official_metadata = load_official_segments(args.official_geojson)
    connector_graph = load_connector_graph(args.connector_geojson, official_segments=official_segments)
    source_files = {
        "field_tool_data_json": display_path(args.field_tool_data_json),
        "packet_dir": display_path(args.packet_dir),
        "official_geojson": display_path(args.official_geojson),
        "official_last_updated_utc": str(official_metadata.get("lastUpdatedUTC")),
        "connector_geojson": display_path(args.connector_geojson),
    }
    audit = build_post_credit_connector_audit(
        field_tool_data,
        packet_dir=args.packet_dir,
        connector_graph=connector_graph,
        connector_graph_path=args.connector_geojson,
        distance_tolerance_miles=args.distance_tolerance_miles,
        snap_tolerance_miles=args.snap_tolerance_miles,
        route_mileage_tolerance_miles=args.route_mileage_tolerance_miles,
        source_files=source_files,
    )
    write_json(args.output_json, audit)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(audit), encoding="utf-8")
    manifest = build_artifact_manifest(
        run_id=args.output_json.stem,
        inputs=[
            display_path(args.field_tool_data_json),
            display_path(args.official_geojson),
            display_path(args.connector_geojson),
        ],
        outputs=[display_path(args.output_json), display_path(args.output_md)],
        command="python years/2026/scripts/post_credit_connector_audit.py",
        metadata={"schema": audit["schema"], "status": audit["status"]},
    )
    write_manifest(args.manifest_json, manifest)
    print(f"Wrote {display_path(args.output_json)}")
    print(f"Wrote {display_path(args.output_md)}")
    print(f"Wrote {display_path(args.manifest_json)}")
    print(json.dumps({"status": audit["status"], **audit["summary"]}, indent=2))
    if audit["status"] != "passed" and not args.report_only:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
