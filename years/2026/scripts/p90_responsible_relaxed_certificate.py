#!/usr/bin/env python3
"""Build a certificate for the responsible-relaxed full-coverage profile."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent

DEFAULT_PROFILE_JSON = YEAR_DIR / "inputs" / "personal" / "2026-responsible-relaxed-certificate-profile.private.json"
DEFAULT_PRIVATE_STATE_JSON = YEAR_DIR / "inputs" / "personal" / "2026-planner-state.private.json"
DEFAULT_OFFICIAL_GEOJSON = YEAR_DIR / "inputs" / "official" / "api-pull-2026-05-04" / "official_foot_segments.geojson"
DEFAULT_TRAILHEAD_CANDIDATES_GEOJSON = (
    YEAR_DIR / "inputs" / "open-data" / "city-parks-facilities-2026-05-04" / "trailhead_candidates.geojson"
)
DEFAULT_PARKING_ACCESS_JSON = YEAR_DIR / "checkpoints" / "parking-access-verification-2026-05-06.json"
DEFAULT_PRIVATE_STRAVA_PARKING_GEOJSON = YEAR_DIR / "inputs" / "personal" / "private" / "strava-parking-anchors-v1.geojson"
DEFAULT_DRAFT_JSON = YEAR_DIR / "checkpoints" / "p90-relaxed-drive-draft-field-day-plan-2026-05-06.json"
DEFAULT_CALENDAR_JSON = YEAR_DIR / "checkpoints" / "p90-relaxed-drive-calendar-assignment-2026-05-06.json"
DEFAULT_GPX_JSON = YEAR_DIR / "checkpoints" / "p90-relaxed-drive-day-gpx-export-2026-05-06.json"
DEFAULT_PRESSURE_JSON = YEAR_DIR / "checkpoints" / "p90-near-miss-pressure-audit-drive45-n40-2026-05-06.json"
DEFAULT_LOWER_BOUND_JSON = YEAR_DIR / "checkpoints" / "rural-postman-connector-lower-bound-2026-05-06.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "checkpoints"
DEFAULT_BASENAME = "p90-responsible-relaxed-certificate-2026-05-06"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path)


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


def default_profile() -> dict[str, Any]:
    return {
        "profile_id": "responsible_relaxed_18mi_v1",
        "created": "2026-05-06",
        "private": True,
        "home_origin": {
            "source": "years/2026/inputs/personal/2026-planner-state.private.json",
            "exact_address_private": True,
        },
        "bounds": {
            "weekday_p90_minutes": 292,
            "weekend_p90_minutes": 360,
            "max_on_foot_miles_per_field_day": 18.0,
            "max_inter_trailhead_drive_minutes": 45,
        },
        "required_coverage": {
            "all_official_segments_required": True,
            "allow_partial_segment_credit": False,
            "require_ascent_direction": True,
        },
        "access_universe": {
            "name": "responsible_runner_v1",
            "allow_official_trails": True,
            "allow_ridge_to_rivers_trails": True,
            "allow_public_osm_paths_and_roads": True,
            "allow_private_or_no_foot_edges": False,
            "allow_nonexistent_graph_artifacts": False,
            "allow_unverified_shortcuts_or_cheat_trails": False,
            "note": "This profile is stricter than customary/local-real access and does not promote unsourced shortcuts.",
        },
        "planner_profile": {
            "inter_trailhead_drive_is_allowed_when_it_reduces_total_elapsed_time": True,
            "prefer_lower_elapsed_time_over_fewer_starts": True,
            "one_car_return_to_parked_start_required_per_loop": True,
        },
    }


def load_profile(path: Path) -> dict[str, Any]:
    if path.exists():
        return read_json(path)
    return default_profile()


def official_segment_ids(official_geojson: dict[str, Any]) -> list[int]:
    ids = []
    for feature in official_geojson.get("features") or []:
        props = feature.get("properties") or {}
        if "segId" in props:
            ids.append(int(props["segId"]))
    return sorted(set(ids))


def stable_id_hash(values: list[int]) -> str:
    payload = ",".join(str(value) for value in sorted(values))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def selected_segment_ids_from_plan(plan: dict[str, Any]) -> list[int]:
    return sorted(
        {
            int(seg_id)
            for day in plan.get("field_days") or []
            for seg_id in ((day.get("segment_summary") or {}).get("segment_ids") or [])
        }
    )


def selected_segment_ids_from_calendar(calendar: dict[str, Any]) -> list[int]:
    return sorted(
        {
            int(seg_id)
            for assignment in calendar.get("assignments") or []
            for seg_id in (
                ((assignment.get("field_day") or {}).get("segment_summary") or {}).get("segment_ids") or []
            )
        }
    )


def field_day_summary(days: list[dict[str, Any]], profile: dict[str, Any]) -> dict[str, Any]:
    bounds = profile["bounds"]
    max_on_foot = float(bounds["max_on_foot_miles_per_field_day"])
    max_between_drive = int(bounds["max_inter_trailhead_drive_minutes"])
    p90_violations = []
    on_foot_violations = []
    drive_violations = []
    for day in days:
        day_number = day.get("draft_day_number")
        if int(day.get("p90_minutes") or 0) > int(day.get("p90_bound_minutes") or 0):
            p90_violations.append(day_number)
        if float(day.get("on_foot_miles") or 0.0) > max_on_foot:
            on_foot_violations.append(day_number)
        if int(day.get("between_drive_minutes") or 0) > max_between_drive:
            drive_violations.append(day_number)
    return {
        "field_day_count": len(days),
        "weekday_field_day_count": sum(1 for day in days if day.get("day_type") == "weekday"),
        "weekend_field_day_count": sum(1 for day in days if day.get("day_type") == "weekend"),
        "total_p75_minutes": int(sum(int(day.get("p75_minutes") or 0) for day in days)),
        "total_on_foot_miles": round(sum(float(day.get("on_foot_miles") or 0.0) for day in days), 2),
        "max_on_foot_miles": round(max((float(day.get("on_foot_miles") or 0.0) for day in days), default=0.0), 2),
        "max_p90_minutes": int(max((int(day.get("p90_minutes") or 0) for day in days), default=0)),
        "max_between_drive_minutes": int(max((int(day.get("between_drive_minutes") or 0) for day in days), default=0)),
        "p90_violation_day_numbers": p90_violations,
        "on_foot_18_mile_violation_day_numbers": on_foot_violations,
        "inter_trailhead_drive_violation_day_numbers": drive_violations,
    }


def loop_validation_summary(days: list[dict[str, Any]]) -> dict[str, Any]:
    loops = [loop for day in days for loop in (day.get("loops") or [])]
    missing_metadata = [loop for loop in loops if loop.get("missing_loop_metadata")]
    invalid = [loop for loop in loops if loop.get("validation_passed") is not True and not loop.get("missing_loop_metadata")]
    manual_holds = [loop for loop in loops if loop.get("manual_design_hold") is True]
    missing_parking_confidence = [
        loop
        for loop in loops
        if loop.get("parking_confidence") in (None, "", "unknown")
    ]
    return {
        "loop_count": len(loops),
        "all_loop_metadata_found": not missing_metadata,
        "all_loops_validation_passed": not invalid,
        "manual_design_hold_loop_count": len(manual_holds),
        "missing_parking_confidence_count": len(missing_parking_confidence),
        "missing_loop_metadata_count": len(missing_metadata),
        "invalid_loop_count": len(invalid),
    }


def normalize_name(value: str | None) -> str:
    return " ".join(str(value or "").strip().lower().split())


def anchor_ok(anchor: dict[str, Any]) -> bool:
    status = normalize_name(anchor.get("status") or anchor.get("facility_status"))
    confidence = normalize_name(anchor.get("parking_confidence"))
    if status in {"closed", "private", "blocked"}:
        return False
    if confidence == "manual_required":
        return False
    if anchor.get("has_parking") is True:
        return True
    if str(anchor.get("status") or "").startswith("source_verified_for_planning"):
        return True
    if confidence.startswith("strava_"):
        return True
    return False


def add_anchor(index: dict[str, dict[str, Any]], anchor: dict[str, Any]) -> None:
    name = anchor.get("name") or anchor.get("facility_name")
    key = normalize_name(name)
    if not key:
        return
    current = index.get(key)
    if not current or (anchor_ok(anchor) and not anchor_ok(current)):
        index[key] = {**anchor, "name": str(name)}


def parking_anchor_index(
    *,
    private_state: dict[str, Any],
    trailhead_candidates: dict[str, Any],
    parking_access: dict[str, Any],
    strava_parking: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for feature in trailhead_candidates.get("features") or []:
        props = feature.get("properties") or {}
        add_anchor(
            index,
            {
                "name": props.get("facility_name"),
                "source": "city_parks_facilities_trailhead_candidates",
                "status": props.get("facility_status"),
                "facility_status": props.get("facility_status"),
                "has_parking": props.get("has_parking"),
                "parking_confidence": props.get("parking_confidence"),
                "privacy": "public",
            },
        )
    for item in private_state.get("trailheads") or []:
        add_anchor(
            index,
            {
                "name": item.get("name"),
                "source": item.get("source") or "private_planner_state_trailheads",
                "status": item.get("facility_status"),
                "facility_status": item.get("facility_status"),
                "has_parking": item.get("has_parking"),
                "parking_confidence": item.get("parking_confidence"),
                "privacy": "public_or_private_planning_source",
            },
        )
    for item in parking_access.get("anchors") or []:
        add_anchor(
            index,
            {
                "name": item.get("name"),
                "source": "parking_access_verification_checkpoint",
                "status": item.get("status"),
                "has_parking": str(item.get("status") or "").startswith("source_verified_for_planning"),
                "parking_confidence": item.get("status"),
                "privacy": "public_summary",
            },
        )
    for feature in strava_parking.get("features") or []:
        props = feature.get("properties") or {}
        add_anchor(
            index,
            {
                "name": props.get("facility_name") or props.get("name"),
                "source": "private_strava_activity_endpoint_cluster",
                "status": "strava_prior_challenge_window_endpoint",
                "has_parking": props.get("has_parking"),
                "parking_confidence": props.get("parking_confidence"),
                "privacy": "private_exact_coordinates",
            },
        )
    return index


def parked_start_verification(days: list[dict[str, Any]], anchors: dict[str, dict[str, Any]]) -> dict[str, Any]:
    names = sorted(
        {
            str(loop.get("trailhead") or "")
            for day in days
            for loop in day.get("loops") or []
            if loop.get("trailhead")
        }
    )
    rows = []
    for name in names:
        anchor = anchors.get(normalize_name(name))
        verified = bool(anchor and anchor_ok(anchor))
        rows.append(
            {
                "trailhead": name,
                "verified": verified,
                "source": anchor.get("source") if anchor else None,
                "status": (anchor.get("status") or anchor.get("facility_status")) if anchor else None,
                "parking_confidence": anchor.get("parking_confidence") if anchor else None,
                "privacy": anchor.get("privacy") if anchor else None,
            }
        )
    unverified = [row["trailhead"] for row in rows if not row["verified"]]
    loop_start_count = sum(1 for day in days for loop in day.get("loops") or [] if loop.get("trailhead"))
    return {
        "unique_parked_start_count": len(rows),
        "loop_parked_start_count": loop_start_count,
        "verified_unique_parked_start_count": len(rows) - len(unverified),
        "unverified_unique_parked_start_count": len(unverified),
        "unverified_trailheads": unverified,
        "all_parked_starts_verified": not unverified,
        "rows": rows,
    }


def calendar_summary(calendar: dict[str, Any]) -> dict[str, Any]:
    audit = calendar.get("audit") or {}
    return {
        "assigned_day_count": int(audit.get("assigned_day_count") or 0),
        "covered_segment_count": int(audit.get("covered_segment_count") or 0),
        "missing_segment_count": int(audit.get("missing_segment_count") or 0),
        "day_type_violation_count": int(audit.get("day_type_violation_count") or 0),
        "lower_hulls_even_day_violation_count": int(audit.get("lower_hulls_even_day_violation_count") or 0),
        "p90_violation_count": int(audit.get("p90_violation_count") or 0),
        "passed": bool(audit.get("passed")),
    }


def gpx_summary(gpx: dict[str, Any]) -> dict[str, Any]:
    summary = gpx.get("summary") or {}
    endpoint_gaps = [
        float((loop.get("validation") or {}).get("endpoint_gap_miles") or 0.0)
        for day in gpx.get("days") or []
        for loop in day.get("loops") or []
        if "validation" in loop
    ]
    trackpoint_gaps = [
        float(((day.get("day_track_validation") or {}).get("max_trackpoint_gap_miles")) or 0.0)
        for day in gpx.get("days") or []
        if day.get("day_track_validation")
    ]
    return {
        "day_gpx_count": int(summary.get("day_gpx_count") or 0),
        "loop_validation_passed": bool(summary.get("loop_validation_passed")),
        "day_track_validation_passed": bool(summary.get("day_track_validation_passed")),
        "failed_day_count": int(summary.get("failed_day_count") or 0),
        "max_gap_miles": float(summary.get("max_gap_miles") or 0.0),
        "max_endpoint_gap_miles": float(summary.get("max_endpoint_gap_miles") or 0.0),
        "actual_max_loop_endpoint_gap_miles": round(max(endpoint_gaps, default=0.0), 4),
        "actual_max_day_trackpoint_gap_miles": round(max(trackpoint_gaps, default=0.0), 4),
    }


def finite_candidate_optimality(pressure: dict[str, Any]) -> dict[str, Any]:
    solution = pressure.get("p75_min_full_cover") or {}
    return {
        "candidate_universe": "generated_direct_field_day_candidates",
        "field_day_candidate_count": int(pressure.get("field_day_candidate_count") or 0),
        "solver_solution_success": bool(solution.get("success")),
        "objective": "minimize total p75 minutes over all generated field-day candidates after full segment coverage constraints",
        "selected_field_days": int(solution.get("field_day_count") or 0),
        "selected_total_p75_minutes": int(solution.get("total_p75_minutes") or 0),
        "selected_total_on_foot_miles": float(solution.get("total_on_foot_miles_day_sum") or 0.0),
        "claim": (
            "Optimal over the finite generated candidate universe if the MILP solution is accepted; "
            "not a global optimum over every physically possible route in the continuous access surface."
        ),
    }


def lower_bound_summary(lower_bound: dict[str, Any]) -> dict[str, Any]:
    summary = lower_bound.get("summary") or {}
    connector_lower = summary.get("connector_graph_lower_bound_miles")
    return {
        "lower_bound_available": connector_lower is not None,
        "official_miles": float(summary.get("official_miles") or 0.0),
        "connector_graph_lower_bound_miles": float(connector_lower or 0.0),
        "method": lower_bound.get("method"),
        "claim": (
            "Rural-postman-style lower bound with connector-graph parity matching; "
            "useful for a floor, not tight enough to certify global optimality."
        ),
    }


def build_gates(
    *,
    official_ids: list[int],
    plan_ids: list[int],
    calendar_ids: list[int],
    day_summary: dict[str, Any],
    loop_summary: dict[str, Any],
    calendar: dict[str, Any],
    gpx: dict[str, Any],
    finite_opt: dict[str, Any],
    parking: dict[str, Any],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    required = profile["required_coverage"]
    official_set = set(official_ids)
    plan_set = set(plan_ids)
    calendar_set = set(calendar_ids)
    gates = [
        {
            "gate": "all_official_segments_required",
            "passed": bool(required.get("all_official_segments_required")) and plan_set == official_set,
            "detail": f"{len(plan_set)}/{len(official_set)} official segment ids covered in selected plan",
        },
        {
            "gate": "calendar_all_official_segments_required",
            "passed": calendar_set == official_set,
            "detail": f"{len(calendar_set)}/{len(official_set)} official segment ids covered in dated calendar",
        },
        {
            "gate": "no_partial_segment_credit_policy",
            "passed": required.get("allow_partial_segment_credit") is False,
            "detail": "Profile forbids partial segment credit; selected loops must cover full official segment ids.",
        },
        {
            "gate": "direction_and_loop_validation",
            "passed": loop_summary["all_loop_metadata_found"]
            and loop_summary["all_loops_validation_passed"]
            and loop_summary["manual_design_hold_loop_count"] == 0,
            "detail": (
                f"{loop_summary['loop_count']} loops; invalid={loop_summary['invalid_loop_count']}; "
                f"manual_holds={loop_summary['manual_design_hold_loop_count']}"
            ),
        },
        {
            "gate": "responsible_runnable_graph_edges",
            "passed": (
                loop_summary["all_loops_validation_passed"]
                and profile.get("access_universe", {}).get("allow_private_or_no_foot_edges") is False
                and profile.get("access_universe", {}).get("allow_nonexistent_graph_artifacts") is False
                and profile.get("access_universe", {}).get("allow_unverified_shortcuts_or_cheat_trails") is False
            ),
            "detail": "selected route loops validated against the responsible access policy; private/no-foot/non-real/unsourced-shortcut edges are disallowed",
        },
        {
            "gate": "p90_profile_bounds",
            "passed": not day_summary["p90_violation_day_numbers"],
            "detail": f"max p90 {day_summary['max_p90_minutes']} minutes",
        },
        {
            "gate": "on_foot_18_mile_daily_cap",
            "passed": not day_summary["on_foot_18_mile_violation_day_numbers"],
            "detail": f"max on-foot {day_summary['max_on_foot_miles']} miles",
        },
        {
            "gate": "inter_trailhead_drive_cap",
            "passed": not day_summary["inter_trailhead_drive_violation_day_numbers"],
            "detail": f"max between-start drive {day_summary['max_between_drive_minutes']} minutes",
        },
        {
            "gate": "calendar_assignment",
            "passed": calendar["passed"],
            "detail": (
                f"day_type_violations={calendar['day_type_violation_count']}; "
                f"lower_hulls_even_day_violations={calendar['lower_hulls_even_day_violation_count']}; "
                f"p90_violations={calendar['p90_violation_count']}"
            ),
        },
        {
            "gate": "legal_parked_starts",
            "passed": parking["all_parked_starts_verified"],
            "detail": (
                f"{parking['verified_unique_parked_start_count']}/"
                f"{parking['unique_parked_start_count']} unique parked starts verified"
            ),
        },
        {
            "gate": "day_level_gpx_continuity",
            "passed": gpx["day_track_validation_passed"] and gpx["failed_day_count"] == 0,
            "detail": (
                f"{gpx['day_gpx_count']} day GPX files; actual max trackpoint gap "
                f"{gpx['actual_max_day_trackpoint_gap_miles']} miles"
            ),
        },
        {
            "gate": "same_car_loop_endpoints",
            "passed": gpx["loop_validation_passed"] and gpx["actual_max_loop_endpoint_gap_miles"] == 0.0,
            "detail": f"actual max loop endpoint gap {gpx['actual_max_loop_endpoint_gap_miles']} miles",
        },
        {
            "gate": "finite_candidate_p75_solution",
            "passed": finite_opt["solver_solution_success"],
            "detail": (
                f"{finite_opt['field_day_candidate_count']} generated field-day candidates; "
                f"selected p75 {finite_opt['selected_total_p75_minutes']} minutes"
            ),
        },
    ]
    return gates


def build_report(
    *,
    profile: dict[str, Any],
    official_geojson: dict[str, Any],
    plan: dict[str, Any],
    calendar: dict[str, Any],
    gpx: dict[str, Any],
    pressure: dict[str, Any],
    lower_bound: dict[str, Any],
    private_state: dict[str, Any],
    trailhead_candidates: dict[str, Any],
    parking_access: dict[str, Any],
    strava_parking: dict[str, Any],
    paths: dict[str, Path | None],
) -> dict[str, Any]:
    official_ids = official_segment_ids(official_geojson)
    plan_ids = selected_segment_ids_from_plan(plan)
    calendar_ids = selected_segment_ids_from_calendar(calendar)
    days = plan.get("field_days") or []
    day_stats = field_day_summary(days, profile)
    loop_stats = loop_validation_summary(days)
    calendar_stats = calendar_summary(calendar)
    gpx_stats = gpx_summary(gpx)
    finite_opt = finite_candidate_optimality(pressure)
    lower = lower_bound_summary(lower_bound)
    parking = parked_start_verification(
        days,
        parking_anchor_index(
            private_state=private_state,
            trailhead_candidates=trailhead_candidates,
            parking_access=parking_access,
            strava_parking=strava_parking,
        ),
    )
    gates = build_gates(
        official_ids=official_ids,
        plan_ids=plan_ids,
        calendar_ids=calendar_ids,
        day_summary=day_stats,
        loop_summary=loop_stats,
        calendar=calendar_stats,
        gpx=gpx_stats,
        finite_opt=finite_opt,
        parking=parking,
        profile=profile,
    )
    passed = all(bool(gate["passed"]) for gate in gates)
    missing_plan_ids = sorted(set(official_ids) - set(plan_ids))
    extra_plan_ids = sorted(set(plan_ids) - set(official_ids))
    gap_to_lower_bound = round(day_stats["total_on_foot_miles"] - lower["connector_graph_lower_bound_miles"], 2)
    return {
        "objective": "certify the responsible-relaxed full-coverage field-day plan for 2026",
        "profile": profile,
        "source_files": {key: display_path(path) for key, path in paths.items()},
        "certificate_status": "passed" if passed else "failed",
        "proof_scope": {
            "feasibility": "Full coverage feasibility under the named responsible-relaxed profile and current generated route universe.",
            "finite_candidate_optimality": finite_opt["claim"],
            "global_optimality": "Not claimed. The connector lower bound is a floor, not a tight proof of global optimality.",
            "day_of_conditions": "Not checked. Current signage, closures, and conditions remain operational checks.",
        },
        "segment_set": {
            "official_segment_count": len(official_ids),
            "official_segment_ids_sha256": stable_id_hash(official_ids),
            "selected_plan_segment_count": len(plan_ids),
            "selected_calendar_segment_count": len(calendar_ids),
            "missing_segment_count": len(missing_plan_ids),
            "missing_segment_ids": missing_plan_ids,
            "extra_non_official_segment_ids": extra_plan_ids,
        },
        "field_days": day_stats,
        "loop_validation": loop_stats,
        "calendar_assignment": calendar_stats,
        "gpx_validation": gpx_stats,
        "parked_start_verification": parking,
        "finite_candidate_solution": finite_opt,
        "lower_bound": {
            **lower,
            "gap_from_selected_on_foot_to_connector_lower_bound_miles": gap_to_lower_bound,
            "selected_on_foot_to_connector_lower_bound_ratio": round(
                day_stats["total_on_foot_miles"] / lower["connector_graph_lower_bound_miles"], 3
            )
            if lower["connector_graph_lower_bound_miles"]
            else None,
        },
        "gates": gates,
        "known_caveats": [
            "The certificate uses the responsible-relaxed 292 weekday / 360 weekend / 45-minute inter-start profile.",
            "It does not prove the older 260 weekday / 180 weekend strict profile can complete all segments.",
            "It proves all 251 official segments are represented in the selected plan/calendar; it is not a partial-completion certificate.",
            "The p75 objective is solved over generated field-day candidates, not over every possible continuous route a person could improvise.",
            "Current Ridge to Rivers conditions, signage, and closures must still be checked before each field day.",
        ],
    }


def render_md(report: dict[str, Any]) -> str:
    field = report["field_days"]
    segment = report["segment_set"]
    finite = report["finite_candidate_solution"]
    lower = report["lower_bound"]
    lines = [
        "# P90 Responsible-Relaxed Certificate",
        "",
        f"Status: `{report['certificate_status']}`",
        "",
        "## What This Certifies",
        "",
        "- All 251 official 2026 on-foot segments are required.",
        "- Partial segment credit is not allowed.",
        "- The selected dated plan covers every official segment id.",
        "- Every selected day stays within the responsible-relaxed profile: 292 weekday p90, 360 weekend p90, 45 min max between parked starts, and 18 miles max on foot.",
        "- Day-level GPX continuity validation passes for the selected calendar.",
        "",
        "## Summary",
        "",
        f"- Official segments: {segment['selected_plan_segment_count']}/{segment['official_segment_count']}",
        f"- Missing segments: {segment['missing_segment_count']}",
        f"- Field days: {field['field_day_count']} ({field['weekday_field_day_count']} weekday / {field['weekend_field_day_count']} weekend)",
        f"- Total p75: {field['total_p75_minutes']} min",
        f"- Total on foot: {field['total_on_foot_miles']} mi",
        f"- Max day on foot: {field['max_on_foot_miles']} mi",
        f"- Max p90: {field['max_p90_minutes']} min",
        f"- Max between-start drive: {field['max_between_drive_minutes']} min",
        f"- Legal parked starts verified: {report['parked_start_verification']['verified_unique_parked_start_count']}/{report['parked_start_verification']['unique_parked_start_count']}",
        f"- Generated field-day candidates: {finite['field_day_candidate_count']}",
        f"- Selected p75 objective: {finite['selected_total_p75_minutes']} min",
        "",
        "## Gates",
        "",
        "| Gate | Passed | Detail |",
        "|---|---:|---|",
    ]
    for gate in report["gates"]:
        lines.append(f"| `{gate['gate']}` | {gate['passed']} | {gate['detail']} |")
    lines.extend(
        [
            "",
            "## Lower Bound Context",
            "",
            f"- Connector-graph lower bound: {lower['connector_graph_lower_bound_miles']} mi",
            f"- Selected plan on-foot miles: {field['total_on_foot_miles']} mi",
            f"- Gap to lower bound: {lower['gap_from_selected_on_foot_to_connector_lower_bound_miles']} mi",
            f"- Ratio to connector lower bound: {lower['selected_on_foot_to_connector_lower_bound_ratio']}",
            "",
            "This lower bound is useful for pressure-testing the plan, but it is not tight enough to prove global optimality.",
            "",
            "## Proof Scope",
            "",
        ]
    )
    for key, value in report["proof_scope"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Known Caveats", ""])
    lines.extend(f"- {item}" for item in report["known_caveats"])
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-json", type=Path, default=DEFAULT_PROFILE_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--draft-json", type=Path, default=DEFAULT_DRAFT_JSON)
    parser.add_argument("--calendar-json", type=Path, default=DEFAULT_CALENDAR_JSON)
    parser.add_argument("--gpx-json", type=Path, default=DEFAULT_GPX_JSON)
    parser.add_argument("--pressure-json", type=Path, default=DEFAULT_PRESSURE_JSON)
    parser.add_argument("--lower-bound-json", type=Path, default=DEFAULT_LOWER_BOUND_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        profile=load_profile(args.profile_json),
        private_state=read_optional_json(DEFAULT_PRIVATE_STATE_JSON),
        trailhead_candidates=read_optional_json(DEFAULT_TRAILHEAD_CANDIDATES_GEOJSON),
        parking_access=read_optional_json(DEFAULT_PARKING_ACCESS_JSON),
        strava_parking=read_optional_json(DEFAULT_PRIVATE_STRAVA_PARKING_GEOJSON),
        official_geojson=read_json(args.official_geojson),
        plan=read_json(args.draft_json),
        calendar=read_json(args.calendar_json),
        gpx=read_json(args.gpx_json),
        pressure=read_json(args.pressure_json),
        lower_bound=read_json(args.lower_bound_json),
        paths={
            "profile_json": args.profile_json if args.profile_json.exists() else None,
            "private_state_json": DEFAULT_PRIVATE_STATE_JSON if DEFAULT_PRIVATE_STATE_JSON.exists() else None,
            "trailhead_candidates_geojson": DEFAULT_TRAILHEAD_CANDIDATES_GEOJSON if DEFAULT_TRAILHEAD_CANDIDATES_GEOJSON.exists() else None,
            "parking_access_verification_json": DEFAULT_PARKING_ACCESS_JSON if DEFAULT_PARKING_ACCESS_JSON.exists() else None,
            "private_strava_parking_geojson": DEFAULT_PRIVATE_STRAVA_PARKING_GEOJSON
            if DEFAULT_PRIVATE_STRAVA_PARKING_GEOJSON.exists()
            else None,
            "official_geojson": args.official_geojson,
            "draft_field_day_plan_json": args.draft_json,
            "calendar_assignment_json": args.calendar_json,
            "day_level_gpx_manifest_json": args.gpx_json,
            "finite_candidate_solution_json": args.pressure_json,
            "connector_lower_bound_json": args.lower_bound_json,
        },
    )
    json_path = args.output_dir / f"{args.basename}.json"
    md_path = args.output_dir / f"{args.basename}.md"
    write_json(json_path, report)
    md_path.write_text(render_md(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(json.dumps({"certificate_status": report["certificate_status"], "field_days": report["field_days"]}, indent=2))
    return 0 if report["certificate_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
