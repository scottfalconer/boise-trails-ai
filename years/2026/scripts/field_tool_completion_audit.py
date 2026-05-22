#!/usr/bin/env python3
"""Audit the generated field tool against the field-use definition of done."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from special_management_rule_audit import build_special_management_audit  # noqa: E402

DEFAULT_FIELD_PACKET_DIR = REPO_ROOT / "docs" / "field-packet"
DEFAULT_FIELD_TOOL_DATA_JSON = DEFAULT_FIELD_PACKET_DIR / "field-tool-data.json"
DEFAULT_MANIFEST_JSON = DEFAULT_FIELD_PACKET_DIR / "manifest.json"
DEFAULT_INDEX_HTML = DEFAULT_FIELD_PACKET_DIR / "index.html"
DEFAULT_CANONICAL_MAP_DATA_JSON = YEAR_DIR / "outputs" / "private" / "2026-outing-menu-map-data.json"
DEFAULT_OFFICIAL_GEOJSON = YEAR_DIR / "inputs" / "official" / "api-pull-2026-05-04" / "official_foot_segments.geojson"
DEFAULT_RECERTIFICATION_JSON = YEAR_DIR / "outputs" / "private" / "progress" / "field-recertification-latest.json"
DEFAULT_OFFICIAL_REPEAT_AUDIT_JSON = YEAR_DIR / "checkpoints" / "field-official-repeat-audit-2026-05-11.json"
DEFAULT_ROUTE_REPEAT_AUDIT_JSON = YEAR_DIR / "checkpoints" / "route-repeat-optimization-audit-2026-05-12.json"
DEFAULT_LATENT_REPRICING_AUDIT_JSON = YEAR_DIR / "checkpoints" / "latent-credit-delta-repricing-audit-2026-05-12.json"
DEFAULT_OWNERSHIP_AUDIT_JSON = YEAR_DIR / "checkpoints" / "ownership-reassignment-optimization-audit-2026-05-12.json"
DEFAULT_SIMULATED_SWEEP_AUDIT_JSON = YEAR_DIR / "checkpoints" / "simulated-progress-sweep-audit-2026-05-12.json"
DEFAULT_SPECIAL_MANAGEMENT_RULES_JSON = YEAR_DIR / "inputs" / "open-data" / "special-management-rules-2026.json"
DEFAULT_R2R_TRAILS_GEOJSON = (
    YEAR_DIR / "inputs" / "open-data" / "r2r-trails-2026-05-04" / "boise_parks_trails_open_data.geojson"
)
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "field-tool-completion-audit-2026-05-06.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "field-tool-completion-audit-2026-05-06.md"

PRIVATE_LITERAL_PATTERNS = (
    "/Users/scott",
    "outputs/private",
    "GETAthleteDashboard",
    "access_token",
    "refresh_token",
    "client_secret",
)
PRIVATE_REGEX_PATTERNS = (
    re.compile(r"\b911\s+n\.?\s+18th\b", re.IGNORECASE),
    re.compile(r"\b911\s+north\s+18th\b", re.IGNORECASE),
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_json_sha256(data: dict[str, Any]) -> str:
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalized_ids(values: list[Any] | tuple[Any, ...] | set[Any] | None) -> list[str]:
    return sorted({str(value) for value in values or [] if value is not None}, key=lambda item: (len(item), item))


def official_segment_ids(official_geojson: dict[str, Any]) -> list[str]:
    return normalized_ids(
        (feature.get("properties") or {}).get("segId")
        for feature in official_geojson.get("features") or []
        if (feature.get("properties") or {}).get("segId") is not None
    )


def requirement(name: str, passed: bool, evidence: str) -> dict[str, Any]:
    return {"requirement": name, "passed": bool(passed), "evidence": evidence}


def advisory_check(name: str, status: str, evidence: str, action_count: int = 0) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "action_count": int(action_count),
        "evidence": evidence,
    }


def audit_summary(audit: dict[str, Any] | None) -> dict[str, Any]:
    return (audit or {}).get("summary") or {}


def audit_status(audit: dict[str, Any] | None) -> str | None:
    return (audit or {}).get("status") if audit else None


def official_repeat_gate(official_repeat_audit: dict[str, Any] | None) -> dict[str, Any]:
    summary = audit_summary(official_repeat_audit)
    failure_counts = {
        "bucket_a_bad_hidden_self_repeat_count": int(summary.get("bucket_a_bad_hidden_self_repeat_count") or 0),
        "repeat_legs_missing_segment_ids": int(summary.get("repeat_legs_missing_segment_ids") or 0),
        "repeat_cues_missing_text": int(summary.get("repeat_cues_missing_text") or 0),
        "unreconciled_extra_credit_segment_count": int(summary.get("unreconciled_extra_credit_segment_count") or 0),
    }
    passed = audit_status(official_repeat_audit) == "passed" and not any(failure_counts.values())
    return requirement(
        "Official repeat audit hard gate has no hidden repeat-accounting failures",
        passed,
        json.dumps({"status": audit_status(official_repeat_audit), **failure_counts}, sort_keys=True),
    )


def route_repeat_gate(route_repeat_audit: dict[str, Any] | None) -> dict[str, Any]:
    summary = audit_summary(route_repeat_audit)
    failure_counts = {
        "failed_route_count": int(summary.get("failed_route_count") or 0),
        "missing_gpx_route_count": int(summary.get("missing_gpx_route_count") or 0),
        "hidden_self_repeat_segment_count": int(summary.get("hidden_self_repeat_segment_count") or 0),
        "latent_credit_segment_count": int(summary.get("latent_credit_segment_count") or 0),
        "unpriced_repeat_segment_count": int(summary.get("unpriced_repeat_segment_count") or 0),
    }
    passed = audit_status(route_repeat_audit) == "passed" and not any(failure_counts.values())
    return requirement(
        "Route repeat optimization hard gate has no hidden self-repeat, latent credit, or unpriced repeat failures",
        passed,
        json.dumps({"status": audit_status(route_repeat_audit), **failure_counts}, sort_keys=True),
    )


def special_management_gate(special_management_audit: dict[str, Any] | None) -> dict[str, Any]:
    summary = audit_summary(special_management_audit)
    failed_routes = [
        route for route in (special_management_audit or {}).get("routes") or []
        if route.get("passed") is not True
    ]
    evidence_rows = []
    for route in failed_routes:
        label = route.get("label") or route.get("outing_id") or "unknown-route"
        for item in route.get("failures") or []:
            target = (
                f"segment {item.get('segment_id')}"
                if item.get("segment_id") is not None
                else f"matched {float(item.get('matched_miles') or 0):.2f} mi"
            )
            evidence_rows.append(
                f"{label}: {item.get('code')} {item.get('rule_id')} {target}"
            )
    passed = audit_status(special_management_audit) == "passed" and not failed_routes
    evidence = (
        "; ".join(evidence_rows[:12])
        if evidence_rows
        else json.dumps(
            {
                "status": audit_status(special_management_audit),
                "failed_route_count": int(summary.get("failed_route_count") or 0),
                "failure_counts": summary.get("failure_counts") or {},
            },
            sort_keys=True,
        )
    )
    return requirement(
        "Land-manager special-management rules pass for every published route",
        passed,
        evidence,
    )


def optimization_advisories(
    *,
    latent_repricing_audit: dict[str, Any] | None,
    ownership_audit: dict[str, Any] | None,
    simulated_sweep_audit: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    latent_summary = audit_summary(latent_repricing_audit)
    latent_removed = int(latent_summary.get("current_calendar_removed_route_count") or 0)
    latent_saved = float(latent_summary.get("current_calendar_saved_on_foot_miles") or 0)
    ownership_summary = audit_summary(ownership_audit)
    skip_ready_removed = int(ownership_summary.get("current_calendar_skip_ready_removed_route_count") or 0)
    order_free_saved = float(ownership_summary.get("order_free_saved_on_foot_miles") or 0)
    sweep_summary = audit_summary(simulated_sweep_audit)
    future_removed = int(sweep_summary.get("sweeps_with_future_removed_route_count") or 0)
    future_shrunk = int(sweep_summary.get("sweeps_with_future_shrunk_route_count") or 0)
    return [
        advisory_check(
            "Latent-credit delta repricing advisory",
            "actionable" if latent_removed else "informational",
            json.dumps(
                {
                    "status": audit_status(latent_repricing_audit),
                    "current_calendar_removed_route_count": latent_removed,
                    "current_calendar_saved_on_foot_miles": round(latent_saved, 2),
                    "current_calendar_saved_p75_minutes": int(
                        latent_summary.get("current_calendar_saved_p75_minutes") or 0
                    ),
                },
                sort_keys=True,
            ),
            action_count=latent_removed,
        ),
        advisory_check(
            "Ownership reassignment optimization advisory",
            "actionable" if skip_ready_removed else ("informational" if order_free_saved else "clear"),
            json.dumps(
                {
                    "status": audit_status(ownership_audit),
                    "current_calendar_skip_ready_removed_route_count": skip_ready_removed,
                    "current_calendar_skip_ready_saved_on_foot_miles": round(
                        float(ownership_summary.get("current_calendar_skip_ready_saved_on_foot_miles") or 0),
                        2,
                    ),
                    "order_free_saved_on_foot_miles": round(order_free_saved, 2),
                },
                sort_keys=True,
            ),
            action_count=skip_ready_removed,
        ),
        advisory_check(
            "Simulated-progress priority advisory",
            "actionable" if future_removed or future_shrunk else "informational",
            json.dumps(
                {
                    "status": audit_status(simulated_sweep_audit),
                    "sweeps_with_future_removed_route_count": future_removed,
                    "sweeps_with_future_shrunk_route_count": future_shrunk,
                },
                sort_keys=True,
            ),
            action_count=future_removed + future_shrunk,
        ),
    ]


def haversine_miles(a: tuple[float, float], b: tuple[float, float]) -> float:
    lon1, lat1 = a
    lon2, lat2 = b
    radius_miles = 3958.7613
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    h = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * radius_miles * math.atan2(math.sqrt(h), math.sqrt(1 - h))


def href_exists(packet_dir: Path, href: str | None) -> bool:
    if not href:
        return False
    return (packet_dir / href).exists()


def normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("’", "'").strip().lower())


def unique_text(values: list[Any]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def cue_names_for_types(
    wayfinding_cues: list[dict[str, Any]],
    cue_types: set[str],
    *,
    reverse: bool = False,
) -> list[str]:
    cues = list(reversed(wayfinding_cues)) if reverse else wayfinding_cues
    for cue in cues:
        if str(cue.get("cue_type") or "") in cue_types:
            names = unique_text(cue.get("signed_as") or [])
            if names:
                return names
    return []


def names_missing_from_text(names: list[str], text: str) -> list[str]:
    haystack = normalized_text(text)
    return [name for name in names if normalized_text(name) not in haystack]


def primary_required_cue_names(names: list[str]) -> list[str]:
    non_osm = [
        name
        for name in names
        if not normalized_text(name).startswith("osm ")
    ]
    return (non_osm or names)[:1]


def scan_public_safety(paths: list[Path]) -> list[str]:
    failures = []
    for path in paths:
        if not path.exists() or not path.is_file():
            failures.append(f"{path}: missing")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in PRIVATE_LITERAL_PATTERNS:
            if token in text:
                failures.append(f"{path}: contains private token {token!r}")
        for pattern in PRIVATE_REGEX_PATTERNS:
            if pattern.search(text):
                failures.append(f"{path}: contains exact home address pattern")
    return failures


def line_parts(geometry: dict[str, Any] | None) -> list[list[tuple[float, float]]]:
    if not geometry:
        return []
    coords = geometry.get("coordinates") or []
    if geometry.get("type") == "LineString":
        raw_parts = [coords]
    elif geometry.get("type") == "MultiLineString":
        raw_parts = coords
    else:
        return []
    parts = []
    for raw_part in raw_parts:
        part = []
        for coord in raw_part or []:
            if len(coord) >= 2:
                part.append((float(coord[0]), float(coord[1])))
        if len(part) >= 2:
            parts.append(part)
    return parts


def official_segment_geometry_index(official_geojson: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index = {}
    for feature in official_geojson.get("features") or []:
        props = feature.get("properties") or {}
        seg_id = props.get("segId")
        if seg_id is None:
            continue
        index[str(seg_id)] = {
            "direction": props.get("direction") or "both",
            "parts": line_parts(feature.get("geometry") or {}),
        }
    return index


def parse_gpx_track_segments(path: Path) -> list[list[tuple[float, float]]]:
    try:
        root = ET.fromstring(path.read_text(encoding="utf-8"))
    except (ET.ParseError, FileNotFoundError):
        return []
    segments = []
    for trkseg in root.findall(".//{*}trkseg"):
        points = []
        for trkpt in trkseg.findall("{*}trkpt"):
            lat = trkpt.get("lat")
            lon = trkpt.get("lon")
            if lat is None or lon is None:
                continue
            points.append((float(lon), float(lat)))
        if points:
            segments.append(points)
    return segments


def cue_total_miles(wayfinding_cues: list[dict[str, Any]]) -> float:
    return max(
        (
            float(cue.get("cum_miles") or 0) + float(cue.get("leg_miles") or 0)
            for cue in wayfinding_cues
        ),
        default=0.0,
    )


def mileage_tolerance_miles(card_miles: float) -> float:
    return max(0.25, float(card_miles or 0) * 0.08)


def local_xy_miles(point: tuple[float, float], origin_lat: float) -> tuple[float, float]:
    lon, lat = point
    return (lon * 69.172 * math.cos(math.radians(origin_lat)), lat * 69.0)


def point_to_segment_distance_miles(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
    origin_lat: float,
) -> float:
    px, py = local_xy_miles(point, origin_lat)
    ax, ay = local_xy_miles(start, origin_lat)
    bx, by = local_xy_miles(end, origin_lat)
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def point_to_track_distance_miles(point: tuple[float, float], track_segments: list[list[tuple[float, float]]]) -> float:
    best = float("inf")
    for segment in track_segments:
        if len(segment) == 1:
            best = min(best, haversine_miles(point, segment[0]))
            continue
        for start, end in zip(segment, segment[1:]):
            best = min(best, point_to_segment_distance_miles(point, start, end, point[1]))
    return best


def link_declares_field_gap(link: dict[str, Any]) -> bool:
    if not link:
        return False
    if link.get("intentional_repark") or link.get("manual_day_of_access_hold"):
        return True
    if link.get("connector_names") or link.get("signpost_labels"):
        return True
    if link.get("connector_miles") or link.get("road_miles") or link.get("official_repeat_miles"):
        return True
    classes = {str(item) for item in link.get("connector_classes") or []}
    return bool(classes & {"r2r_trail", "osm_path_footway", "osm_public_road", "official_repeat"})


def link_declares_source_gap_exception(link: dict[str, Any]) -> bool:
    if not link:
        return False
    return bool(
        link.get("intentional_repark")
        or link.get("multi_start_boundary")
        or link.get("manual_day_of_access_hold")
    )


def candidate_has_declared_gap(canonical_map_data: dict[str, Any], candidate_id: str) -> bool:
    cue = (canonical_map_data.get("route_cues") or {}).get(candidate_id) or {}
    for link in cue.get("between_links") or []:
        if link_declares_source_gap_exception(link):
            return True
    return link_declares_source_gap_exception(cue.get("return_to_car") or {})


def source_gap_failures(canonical_map_data: dict[str, Any] | None, routes: list[dict[str, Any]]) -> list[str]:
    return source_gap_analysis(canonical_map_data, routes)["failures"]


def source_gap_analysis(canonical_map_data: dict[str, Any] | None, routes: list[dict[str, Any]]) -> dict[str, Any]:
    if not canonical_map_data:
        return {"failures": [], "declared_count": 0, "repaired_count": 0, "warning_count": 0}
    route_candidate_ids = {
        str(candidate_id)
        for route in routes
        for candidate_id in route.get("candidate_ids") or []
    }
    routes_by_candidate_id = {
        str(candidate_id): route
        for route in routes
        for candidate_id in route.get("candidate_ids") or []
    }
    failures = []
    declared_count = 0
    repaired_count = 0
    warning_count = 0
    for validation in (canonical_map_data.get("map_validation") or {}).get("route_validations") or []:
        candidate_id = str(validation.get("candidate_id") or "")
        if route_candidate_ids and candidate_id and candidate_id not in route_candidate_ids:
            continue
        if validation.get("source_gap_warning") is True:
            warning_count += 1
            if candidate_id and candidate_has_declared_gap(canonical_map_data, candidate_id):
                declared_count += 1
                continue
            route = routes_by_candidate_id.get(candidate_id) or {}
            repair = route.get("source_gap_repair") or {}
            if (
                (route.get("validation") or {}).get("passed") is True
                and int(repair.get("repaired_inter_segment_gap_count") or 0) > 0
                and int(repair.get("remaining_inter_segment_gap_count") or 0) == 0
            ):
                repaired_count += 1
                continue
            failures.append(
                f"{candidate_id or 'unknown'} source gap {validation.get('source_max_gap_miles')} mi"
            )
    return {
        "failures": failures,
        "declared_count": declared_count,
        "repaired_count": repaired_count,
        "warning_count": warning_count,
    }


def route_geometry_coverage_failures(
    routes: list[dict[str, Any]],
    packet_dir: Path,
    official_geometry_index: dict[str, dict[str, Any]],
    endpoint_tolerance_miles: float = 0.04,
) -> list[str]:
    failures = []
    for route in routes:
        label = f"{route.get('label') or route.get('outing_id')} {route.get('trailhead') or ''}".strip()
        gpx_href = route.get("gpx_href")
        gpx_path = packet_dir / gpx_href if gpx_href else None
        track_segments = parse_gpx_track_segments(gpx_path) if gpx_path else []
        if not track_segments:
            failures.append(f"{label}: Nav GPX has no track")
            continue
        for segment_id in normalized_ids(route.get("segment_ids")):
            official = official_geometry_index.get(segment_id)
            if not official or not official.get("parts"):
                continue
            endpoint_failures = []
            for part in official["parts"]:
                for point in (part[0], part[-1]):
                    distance = point_to_track_distance_miles(point, track_segments)
                    if distance > endpoint_tolerance_miles:
                        endpoint_failures.append(round(distance, 4))
            if endpoint_failures:
                failures.append(
                    f"{label}: does not cover official segment {segment_id} endpoints within "
                    f"{endpoint_tolerance_miles} mi; nearest misses {endpoint_failures[:4]}"
                )
    return failures


def route_field_failures(routes: list[dict[str, Any]], packet_dir: Path, official_ids: set[str]) -> list[str]:
    failures = []
    movement_cue_types = {
        "start_access",
        "official_segment_start",
        "follow_official_segment",
        "junction_turn",
        "connector_named_trail",
        "connector_road",
        "repeat_official_noncredit",
        "exit_access",
        "return_to_car",
    }
    for route in routes:
        label = f"{route.get('label') or route.get('outing_id')} {route.get('trailhead') or ''}".strip()
        parking = route.get("parking") or {}
        effort = route.get("effort") or {}
        validation = route.get("validation") or {}
        navigation_quality = route.get("navigation_quality") or {}
        completion_safety = route.get("completion_safety") or {}
        segment_ids = set(normalized_ids(route.get("segment_ids")))
        steps = route.get("turn_by_turn_steps") or []
        if parking.get("lat") is None or parking.get("lon") is None or parking.get("has_parking") is not True:
            failures.append(f"{label}: missing verified parked start")
        if validation.get("passed") is not True:
            failures.append(f"{label}: GPX validation did not pass")
        if not href_exists(packet_dir, route.get("gpx_href")):
            failures.append(f"{label}: missing Nav GPX file")
        if not segment_ids:
            failures.append(f"{label}: no official segment ids")
        if not segment_ids <= official_ids:
            failures.append(f"{label}: segment ids outside official target")
        if not route.get("door_to_door_minutes_p75") or not route.get("door_to_door_minutes_p90"):
            failures.append(f"{label}: missing p75/p90 door-to-door time")
        if not route.get("on_foot_miles") or not route.get("official_miles"):
            failures.append(f"{label}: missing mileage")
        if effort.get("ascent_ft") is None or effort.get("descent_ft") is None or effort.get("grade_adjusted_miles") is None:
            failures.append(f"{label}: missing DEM effort")
        if effort.get("estimated_moving_minutes_p75") is None:
            failures.append(f"{label}: missing p75 moving effort")
        if completion_safety.get("normal_completion_preserves_remaining_menu_coverage") is not True:
            failures.append(f"{label}: normal completion does not preserve remaining menu coverage")
        step_kinds = {str(step.get("kind")) for step in steps}
        if not {"park", "navigate", "return"} <= step_kinds:
            failures.append(f"{label}: turn cues must include park, navigate, and return steps")
        wayfinding_cues = route.get("wayfinding_cues") or []
        if not wayfinding_cues:
            failures.append(f"{label}: missing wayfinding cue sheet")
        card_miles = float(route.get("on_foot_miles") or 0)
        if card_miles > 0:
            tolerance = mileage_tolerance_miles(card_miles)
            cue_miles = cue_total_miles(wayfinding_cues)
            if cue_miles and abs(cue_miles - card_miles) > tolerance:
                failures.append(
                    f"{label}: wayfinding cue mileage {cue_miles:.2f} mi does not match "
                    f"card on-foot mileage {card_miles:.2f} mi within {tolerance:.2f} mi"
                )
        prior_seq = 0
        for cue in wayfinding_cues:
            seq = int(cue.get("seq") or 0)
            cue_type = str(cue.get("cue_type") or "")
            if seq <= prior_seq:
                failures.append(f"{label}: wayfinding cue {seq} is not in increasing order")
            prior_seq = seq
            if cue_type in movement_cue_types:
                if not cue.get("until"):
                    failures.append(f"{label}: wayfinding cue {seq} {cue_type} missing until")
                if not cue.get("target"):
                    failures.append(f"{label}: wayfinding cue {seq} {cue_type} missing target")
                if not cue.get("signed_as") and not cue.get("landmarks") and not cue.get("road_name"):
                    failures.append(f"{label}: wayfinding cue {seq} {cue_type} missing signed_as/landmark")
        start_access_gap = float(navigation_quality.get("start_access_gap_miles") or 0)
        return_access_gap = float(navigation_quality.get("return_access_gap_miles") or 0)
        access_text = " ".join(
            f"{step.get('title') or ''} {step.get('detail') or ''}"
            for step in steps
            if step.get("kind") == "access"
        )
        return_text = " ".join(
            f"{step.get('title') or ''} {step.get('detail') or ''}"
            for step in steps
            if step.get("kind") == "return"
        )
        if start_access_gap > 0.05 and not re.search(r"\b(access|connector|road|gpx|start on)\b", access_text, re.I):
            failures.append(f"{label}: missing explicit non-credit start-access cue")
        start_access_names = cue_names_for_types(wayfinding_cues, {"start_access"})
        if start_access_gap > 0.05 and start_access_names:
            missing_names = names_missing_from_text(primary_required_cue_names(start_access_names), access_text)
            if missing_names:
                failures.append(f"{label}: missing named start-access cue {', '.join(missing_names)}")
        if return_access_gap > 0.05:
            if "you should be back at the parking point" in return_text.lower():
                failures.append(f"{label}: return cue implies done at car despite non-credit return leg")
            elif not re.search(r"\b(access|connector|road|gpx|return via)\b", return_text, re.I):
                failures.append(f"{label}: missing explicit non-credit return-access cue")
            return_names = cue_names_for_types(wayfinding_cues, {"exit_access", "return_to_car"}, reverse=True)
            if return_names:
                missing_names = names_missing_from_text(primary_required_cue_names(return_names), return_text)
                if missing_names:
                    failures.append(f"{label}: missing named return-access cue {', '.join(missing_names)}")
    return failures


def build_completion_audit(
    *,
    field_tool_data: dict[str, Any],
    manifest: dict[str, Any],
    official_geojson: dict[str, Any],
    index_html: str,
    packet_dir: Path,
    canonical_map_data: dict[str, Any] | None = None,
    recertification_report: dict[str, Any] | None = None,
    official_repeat_audit: dict[str, Any] | None = None,
    route_repeat_audit: dict[str, Any] | None = None,
    latent_repricing_audit: dict[str, Any] | None = None,
    ownership_audit: dict[str, Any] | None = None,
    simulated_sweep_audit: dict[str, Any] | None = None,
    special_management_rules_config: dict[str, Any] | None = None,
    open_trails_geojson: dict[str, Any] | None = None,
    special_management_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    official_ids = set(official_segment_ids(official_geojson))
    routes = field_tool_data.get("routes") or []

    def is_held_route(route: dict[str, Any]) -> bool:
        return (
            route.get("field_ready") is False
            or str(route.get("field_readiness_status") or "") == "blocked_special_management"
        )

    held_routes = [route for route in routes if is_held_route(route)]
    field_ready_routes = [route for route in routes if not is_held_route(route)]
    route_segment_ids = {
        segment_id
        for route in routes
        for segment_id in normalized_ids(route.get("segment_ids"))
    }
    progress = field_tool_data.get("progress") or {}
    completed_segment_ids = set(normalized_ids(progress.get("completed_segment_ids_at_export")))
    blocked_segment_ids = set(normalized_ids(progress.get("blocked_segment_ids_at_export")))
    accounted_segment_ids = route_segment_ids | completed_segment_ids | blocked_segment_ids
    source = field_tool_data.get("source") or {}
    source_hash = source.get("map_data_sha256")
    canonical_hash = stable_json_sha256(canonical_map_data) if canonical_map_data is not None else None
    baseline = field_tool_data.get("certified_baseline") or {}
    summary = field_tool_data.get("summary") or {}
    manifest_summary = manifest.get("summary") or {}
    recert_summary = (recertification_report or {}).get("summary") or {}
    if special_management_audit is None:
        rules_config = special_management_rules_config or read_json(DEFAULT_SPECIAL_MANAGEMENT_RULES_JSON)
        special_management_audit = build_special_management_audit(
            field_tool_data=field_tool_data,
            official_geojson=official_geojson,
            rules_config=rules_config,
            packet_dir=packet_dir,
            open_trails_geojson=open_trails_geojson or read_json(DEFAULT_R2R_TRAILS_GEOJSON),
        )
    route_failures = route_field_failures(routes, packet_dir, official_ids)
    source_gap_status = source_gap_analysis(canonical_map_data, routes)
    source_failures = source_gap_status["failures"]
    if source_failures:
        source_gap_evidence = "; ".join(source_failures[:12])
    elif source_gap_status["warning_count"]:
        source_gap_evidence = (
            f"{source_gap_status['declared_count']} source-gap warnings are represented by explicit "
            "re-park/manual metadata and "
            f"{source_gap_status['repaired_count']} are repaired by exported Nav GPX connector geometry; "
            "0 hidden source gaps"
        )
    else:
        source_gap_evidence = "canonical map source has no source_gap_warning routes"
    geometry_failures = route_geometry_coverage_failures(
        routes,
        packet_dir,
        official_segment_geometry_index(official_geojson),
    )
    safety_failures = scan_public_safety(
        [
            packet_dir / "index.html",
            packet_dir / "field-tool-data.json",
            packet_dir / "manifest.json",
            packet_dir / "service-worker.js",
        ]
    )
    required_filters = [60, 90, 120, 180, 240, 360]
    checks = [
        requirement(
            "Phone page and map share the canonical field-menu source",
            bool(source_hash and canonical_hash and source_hash == canonical_hash),
            f"field source hash {source_hash}; canonical map hash {canonical_hash}",
        ),
        requirement(
            "Certified completion baseline covers 251 official segments",
            baseline.get("status") == "passed"
            and int(baseline.get("official_segment_count") or 0) == 251
            and int(baseline.get("covered_segment_count") or 0) == 251
            and int(baseline.get("missing_segment_count") or 0) == 0,
            json.dumps(
                {
                    "status": baseline.get("status"),
                    "official": baseline.get("official_segment_count"),
                    "covered": baseline.get("covered_segment_count"),
                    "missing": baseline.get("missing_segment_count"),
                },
                sort_keys=True,
            ),
        ),
        requirement(
            "Daily filtering supports the required door-to-door windows",
            field_tool_data.get("time_filters_minutes") == required_filters
            and all(f'data-filter="{value}"' in index_html for value in required_filters),
            f"filters {field_tool_data.get('time_filters_minutes')}",
        ),
        requirement(
            "Listed outings have parking, car-to-car Nav GPX, turn cues, segment ids, time, mileage, and DEM effort",
            not route_failures and bool(routes),
            (
                "; ".join(route_failures[:12])
                if route_failures
                else (
                    f"{len(routes)} route cards passed field-structure checks; "
                    f"{len(held_routes)} held by legality/certification gates"
                )
            ),
        ),
        requirement(
            "Source routes have no hidden unstitched gaps",
            not source_failures,
            source_gap_evidence,
        ),
        requirement(
            "Nav GPX covers claimed official segment endpoints",
            not geometry_failures,
            "; ".join(geometry_failures[:12]) if geometry_failures else "each route Nav GPX reaches listed official segment endpoints",
        ),
        requirement(
            "Active field packet accounts for every official segment geometry id",
            accounted_segment_ids == official_ids
            and int(summary.get("segment_count_in_field_menu") or 0) == len(route_segment_ids),
            (
                f"field menu {len(route_segment_ids)} ids; "
                f"completed {len(completed_segment_ids)} ids; "
                f"blocked {len(blocked_segment_ids)} ids; "
                f"accounted {len(accounted_segment_ids)} ids; "
                f"official target {len(official_ids)} ids"
            ),
        ),
        requirement(
            "GPX validation passed for every exported route card",
            manifest_summary.get("gpx_validation_passed") is True
            and int(manifest_summary.get("navigation_gpx_count") or 0) == len(routes)
            and int(manifest_summary.get("failed_gpx_count") or 0) == 0,
            json.dumps(
                {
                    "navigation": manifest_summary.get("navigation_gpx_count"),
                    "failed": manifest_summary.get("failed_gpx_count"),
                    "passed": manifest_summary.get("gpx_validation_passed"),
                },
                sort_keys=True,
            ),
        ),
        requirement(
            "Phone progress can hide completed outings and export reviewed progress",
            "fieldPacketCompletedOutings" in index_html
            and "Mark done" in index_html
            and "Hide completed" in index_html
            and "Export progress" in index_html
            and "missed_segment_ids" in index_html,
            "localStorage completion, hide completed, export progress, and missed segment review fields are present",
        ),
        requirement(
            "Phone page presents field decisions as tappable cue cards",
            "Field Cue Sheet" in index_html
            and "Tap the cue you are working on" in index_html
            and "decision-cards" in index_html
            and "current-step" in index_html
            and "Turn-by-turn from car" not in index_html,
            "expected Field Cue Sheet heading, tappable decision card class, current-step highlighting, and no legacy turn-by-turn heading",
        ),
        requirement(
            "Best-today recommendation uses the active time window and remaining segment ids",
            "Best today for" in index_html
            and "new official segment(s)" in index_html
            and "completion-safe in the current menu" in index_html
            and "completedSegmentSet" in index_html,
            "phone JavaScript ranks visible incomplete cards by completion-safety and new remaining segment count inside the active filter",
        ),
        requirement(
            "Adaptive recertification reports whether selected-profile completion remains feasible",
            (recertification_report or {}).get("status") == "passed"
            and recert_summary.get("remaining_full_completion_feasible") is True
            and recert_summary.get("remaining_coverage_preserved") is True,
            json.dumps(
                {
                    "status": (recertification_report or {}).get("status"),
                    "remaining_full_completion_feasible": recert_summary.get("remaining_full_completion_feasible"),
                    "remaining_coverage_preserved": recert_summary.get("remaining_coverage_preserved"),
                },
                sort_keys=True,
            ),
        ),
        requirement(
            "Public field outputs do not expose private origin, tokens, dashboard data, or private paths",
            not safety_failures,
            "; ".join(safety_failures[:8]) if safety_failures else "public packet files passed private-token scan",
        ),
        official_repeat_gate(official_repeat_audit),
        route_repeat_gate(route_repeat_audit),
        special_management_gate(special_management_audit),
    ]
    advisory_checks = optimization_advisories(
        latent_repricing_audit=latent_repricing_audit,
        ownership_audit=ownership_audit,
        simulated_sweep_audit=simulated_sweep_audit,
    )
    status = "passed" if all(check["passed"] for check in checks) else "failed"
    return {
        "schema": "boise_trails_field_tool_completion_audit_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "objective": "field-usable daily decision tool with certified full-completion baseline",
        "status": status,
        "summary": {
            "passed_requirement_count": len([check for check in checks if check["passed"]]),
            "requirement_count": len(checks),
            "route_count": len(routes),
            "field_ready_route_count": len(field_ready_routes),
            "held_route_count": len(held_routes),
            "official_segment_count": len(official_ids),
            "field_menu_segment_count": len(route_segment_ids),
            "completed_segment_count_at_export": len(completed_segment_ids),
            "blocked_segment_count_at_export": len(blocked_segment_ids),
            "accounted_segment_count": len(accounted_segment_ids),
            "advisory_check_count": len(advisory_checks),
            "advisory_action_count": sum(int(check.get("action_count") or 0) for check in advisory_checks),
            "special_management_status": special_management_audit.get("status"),
            "special_management_failed_route_count": int(
                (special_management_audit.get("summary") or {}).get("failed_route_count") or 0
            ),
        },
        "checks": checks,
        "advisory_checks": advisory_checks,
        "special_management_audit": special_management_audit,
    }


def render_md(audit: dict[str, Any]) -> str:
    lines = [
        "# Field Tool Completion Audit - 2026-05-06",
        "",
        f"- Status: `{audit['status']}`",
        f"- Requirements: {audit['summary']['passed_requirement_count']} / {audit['summary']['requirement_count']} passed",
        f"- Advisory optimization actions surfaced: {audit['summary'].get('advisory_action_count', 0)}",
        f"- Field-ready route cards: {audit['summary'].get('field_ready_route_count', audit['summary']['route_count'])}",
        f"- Held route cards: {audit['summary'].get('held_route_count', 0)}",
        f"- Total route cards: {audit['summary']['route_count']}",
        (
            f"- Official segment accounting: {audit['summary']['accounted_segment_count']} / "
            f"{audit['summary']['official_segment_count']} "
            f"({audit['summary']['field_menu_segment_count']} active field-menu ids, "
            f"{audit['summary']['completed_segment_count_at_export']} completed, "
            f"{audit['summary']['blocked_segment_count_at_export']} blocked)"
        ),
        "",
        "## Requirement Checklist",
        "",
        "| Requirement | Status | Evidence |",
        "|---|---|---|",
    ]
    for check in audit["checks"]:
        status = "Pass" if check["passed"] else "Fail"
        evidence = str(check["evidence"]).replace("|", "\\|")
        lines.append(f"| {check['requirement']} | {status} | {evidence} |")
    lines.extend(
        [
            "",
            "## Optimization Advisories",
            "",
            "| Advisory | Status | Actions | Evidence |",
            "|---|---|---:|---|",
        ]
    )
    for check in audit.get("advisory_checks") or []:
        evidence = str(check.get("evidence")).replace("|", "\\|")
        lines.append(
            f"| {check.get('name')} | {check.get('status')} | {int(check.get('action_count') or 0)} | {evidence} |"
        )
    lines.extend(
        [
            "",
            "## Validation Commands",
            "",
            "- `python years/2026/scripts/export_mobile_field_packet.py`",
            "- `python years/2026/scripts/field_official_repeat_audit.py`",
            "- `python years/2026/scripts/field_progress_report.py`",
            "- `python years/2026/scripts/field_recertification_report.py`",
            "- `python years/2026/scripts/field_tool_completion_audit.py`",
            "- `python years/2026/scripts/route_repeat_optimization_audit.py`",
            "- `python years/2026/scripts/latent_credit_delta_repricing_audit.py`",
            "- `python years/2026/scripts/ownership_reassignment_optimization_audit.py`",
            "- `python years/2026/scripts/simulated_progress_sweep_audit.py`",
            "- `pytest -q years/2026/tests/test_export_mobile_field_packet.py years/2026/tests/test_field_progress_report.py years/2026/tests/test_field_recertification_report.py years/2026/tests/test_field_tool_completion_audit.py`",
            "",
            "## Remaining Risk",
            "",
            "This audit verifies the generated field tool and selected-profile recertification gates. It is not a global proof of optimality over every possible real-world route or a substitute for day-of trail signage and condition checks.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    parser.add_argument("--index-html", type=Path, default=DEFAULT_INDEX_HTML)
    parser.add_argument("--canonical-map-data-json", type=Path, default=DEFAULT_CANONICAL_MAP_DATA_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--recertification-json", type=Path, default=DEFAULT_RECERTIFICATION_JSON)
    parser.add_argument("--official-repeat-audit-json", type=Path, default=DEFAULT_OFFICIAL_REPEAT_AUDIT_JSON)
    parser.add_argument("--route-repeat-audit-json", type=Path, default=DEFAULT_ROUTE_REPEAT_AUDIT_JSON)
    parser.add_argument("--latent-repricing-audit-json", type=Path, default=DEFAULT_LATENT_REPRICING_AUDIT_JSON)
    parser.add_argument("--ownership-audit-json", type=Path, default=DEFAULT_OWNERSHIP_AUDIT_JSON)
    parser.add_argument("--simulated-sweep-audit-json", type=Path, default=DEFAULT_SIMULATED_SWEEP_AUDIT_JSON)
    parser.add_argument("--special-management-rules-json", type=Path, default=DEFAULT_SPECIAL_MANAGEMENT_RULES_JSON)
    parser.add_argument("--open-trails-geojson", type=Path, default=DEFAULT_R2R_TRAILS_GEOJSON)
    parser.add_argument("--packet-dir", type=Path, default=DEFAULT_FIELD_PACKET_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audit = build_completion_audit(
        field_tool_data=read_json(args.field_tool_data_json),
        manifest=read_json(args.manifest_json),
        official_geojson=read_json(args.official_geojson),
        index_html=args.index_html.read_text(encoding="utf-8"),
        packet_dir=args.packet_dir,
        canonical_map_data=read_json(args.canonical_map_data_json),
        recertification_report=read_json(args.recertification_json) if args.recertification_json.exists() else None,
        official_repeat_audit=read_json(args.official_repeat_audit_json) if args.official_repeat_audit_json.exists() else None,
        route_repeat_audit=read_json(args.route_repeat_audit_json) if args.route_repeat_audit_json.exists() else None,
        latent_repricing_audit=read_json(args.latent_repricing_audit_json) if args.latent_repricing_audit_json.exists() else None,
        ownership_audit=read_json(args.ownership_audit_json) if args.ownership_audit_json.exists() else None,
        simulated_sweep_audit=read_json(args.simulated_sweep_audit_json) if args.simulated_sweep_audit_json.exists() else None,
        special_management_rules_config=read_json(args.special_management_rules_json),
        open_trails_geojson=read_json(args.open_trails_geojson) if args.open_trails_geojson.exists() else None,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_md(audit), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(json.dumps({"status": audit["status"], "summary": audit["summary"]}, indent=2))
    return 0 if audit["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
