#!/usr/bin/env python3
"""Prepare and verify the focused FD04A -> FD19C route-card promotion path."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from human_loop_plan import recompute_package_summary  # noqa: E402
from multi_start_field_menu_replacements import (  # noqa: E402
    add_segment_to_component,
    segment_row_from_source,
    update_cue_for_segment_promotion,
)

RUN_ID = "fd04a-fd19c-route-card-promotion-path-2026-05-13"
EXPERIMENT_DIR = YEAR_DIR / "outputs" / "private" / RUN_ID
AUDIT_DIR = EXPERIMENT_DIR / "audits"

DEFAULT_SEGMENT_PROMOTIONS_JSON = YEAR_DIR / "inputs" / "personal" / "2026-cross-package-segment-promotions-v1.json"
DEFAULT_CALENDAR_JSON = YEAR_DIR / "checkpoints" / "p90-relaxed-drive-calendar-assignment-2026-05-06.json"
DEFAULT_PROOF_JSON = YEAR_DIR / "checkpoints" / "fd04a-fd19c-credit-promotion-experiment-2026-05-12.json"
DEFAULT_CALENDAR_REORDER_JSON = YEAR_DIR / "checkpoints" / "calendar-reorder-latent-credit-experiment-2026-05-12.json"
DEFAULT_OFFICIAL_GEOJSON = YEAR_DIR / "inputs" / "official" / "api-pull-2026-06-13" / "official_foot_segments.geojson"
DEFAULT_BASE_MAP_DATA_JSON = YEAR_DIR / "outputs" / "private" / "2026-outing-menu-map-data.json"
DEFAULT_BASE_PROMOTION_REPORT_JSON = YEAR_DIR / "checkpoints" / "harlow-h1-route-card-promotion-2026-05-12.json"

DEFAULT_PREP_PROMOTIONS_JSON = EXPERIMENT_DIR / "fd04a-fd19c-segment-promotions.promoted.json"
DEFAULT_PREP_CALENDAR_JSON = EXPERIMENT_DIR / "fd04a-fd19c-calendar-reordered.json"
DEFAULT_PREP_REPORT_JSON = YEAR_DIR / "checkpoints" / f"{RUN_ID}-prep.json"
DEFAULT_PREP_REPORT_MD = YEAR_DIR / "checkpoints" / f"{RUN_ID}-prep.md"

DEFAULT_PROMOTED_MAP_DATA_JSON = EXPERIMENT_DIR / "promoted-map-data.json"
DEFAULT_FOCUSED_MAP_DATA_JSON = EXPERIMENT_DIR / "focused-promoted-map-data.json"
DEFAULT_PROMOTED_FIELD_DAY_REPORT_JSON = EXPERIMENT_DIR / "promote-field-day-loops-report.json"
DEFAULT_COMBINED_PROMOTION_REPORT_JSON = EXPERIMENT_DIR / "combined-route-card-promotion-report.json"
DEFAULT_FINAL_PACKET_DIR = EXPERIMENT_DIR / "field-packet-final"
DEFAULT_FINAL_FIELD_TOOL_DATA_JSON = DEFAULT_FINAL_PACKET_DIR / "field-tool-data.json"
DEFAULT_FINAL_FIELD_DAY_LAYER_JSON = EXPERIMENT_DIR / "field-day-layer.json"
DEFAULT_SAFE_FIELD_DAY_LAYER_JSON = EXPERIMENT_DIR / "field-day-layer.public-safe.json"
DEFAULT_ROUTE_REPEAT_JSON = AUDIT_DIR / "route-repeat-optimization-audit.json"
DEFAULT_LATENT_JSON = AUDIT_DIR / "field-latent-credit-audit.json"
DEFAULT_PROGRESS_JSON = AUDIT_DIR / "field-progress-report.json"
DEFAULT_RECERTIFICATION_JSON = AUDIT_DIR / "field-recertification-report.json"
DEFAULT_COMPLETION_JSON = AUDIT_DIR / "field-tool-completion-audit.json"
DEFAULT_WALKTHROUGH_JSON = AUDIT_DIR / "field-route-walkthrough-audit.json"
DEFAULT_VERIFY_REPORT_JSON = YEAR_DIR / "checkpoints" / f"{RUN_ID}.json"
DEFAULT_VERIFY_REPORT_MD = YEAR_DIR / "checkpoints" / f"{RUN_ID}.md"

SOURCE_OUTING_ID = "104-1"
SOURCE_LABEL = "FD04A"
SOURCE_CANDIDATE_ID = "combo-two-point-femrites-patrol-shanes-connector"
SOURCE_PACKAGE_NUMBER = 104
OWNER_OUTING_ID = "119-3"
OWNER_LABEL = "FD19C"
OWNER_CANDIDATE_ID = "shanes-trail"
OWNER_PACKAGE_NUMBER = 119
TARGET_SEGMENT_IDS = ["1649", "1650", "1651"]
SCENARIO_ID = "104-1-before-119-3"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
    if isinstance(values, str | int | float):
        values = [values]
    return sorted({str(value) for value in values if value is not None}, key=sort_id)


def int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def update_assignment_date(assignment: dict[str, Any], date_text: str) -> dict[str, Any]:
    updated = copy.deepcopy(assignment)
    day = date.fromisoformat(date_text)
    updated["date"] = date_text
    updated["day_of_month"] = day.day
    updated["weekday_name"] = day.strftime("%A")
    updated["day_type"] = "weekend" if day.weekday() >= 5 else "weekday"
    updated["is_even_day"] = day.day % 2 == 0
    field_day = updated.setdefault("field_day", {})
    field_day["day_type"] = updated["day_type"]
    return updated


def assignment_for_candidate(assignments: list[dict[str, Any]], candidate_id: str) -> dict[str, Any]:
    for assignment in assignments:
        for loop in ((assignment.get("field_day") or {}).get("loops") or []):
            if str(loop.get("candidate_id")) == candidate_id:
                return assignment
    raise KeyError(f"No calendar assignment includes candidate {candidate_id}")


def scenario_for_id(calendar_reorder: dict[str, Any], scenario_id: str = SCENARIO_ID) -> dict[str, Any]:
    for scenario in calendar_reorder.get("pairwise_scenarios") or []:
        if scenario.get("scenario_id") == scenario_id:
            return scenario
    raise KeyError(f"Scenario {scenario_id} not found")


def materialize_reordered_calendar(
    calendar: dict[str, Any],
    calendar_reorder: dict[str, Any],
) -> dict[str, Any]:
    scenario = scenario_for_id(calendar_reorder)
    assignments = [copy.deepcopy(assignment) for assignment in calendar.get("assignments") or []]
    source_original = assignment_for_candidate(assignments, SOURCE_CANDIDATE_ID)
    owner_original = assignment_for_candidate(assignments, OWNER_CANDIDATE_ID)
    result_assignments = []
    for assignment in assignments:
        if assignment is source_original:
            result_assignments.append(update_assignment_date(assignment, scenario["source_target_date"]))
        elif assignment is owner_original:
            result_assignments.append(update_assignment_date(assignment, scenario["owner_target_date"]))
        else:
            result_assignments.append(copy.deepcopy(assignment))
    result_assignments.sort(key=lambda row: str(row.get("date") or ""))
    result = copy.deepcopy(calendar)
    result["assignments"] = result_assignments
    result.setdefault("audit", {}).update(
        {
            "fd04a_fd19c_reorder_experiment": {
                "scenario_id": scenario["scenario_id"],
                "source_target_date": scenario["source_target_date"],
                "owner_target_date": scenario["owner_target_date"],
                "fd19c_removal_is_applied_by_route_card_promotion_report": True,
            }
        }
    )
    return result


def route_promotion_rows(proof: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in proof.get("proposed_segment_promotion_rows") or []:
        segment_id = str(row.get("segment_id"))
        if segment_id not in TARGET_SEGMENT_IDS:
            continue
        promoted = copy.deepcopy(row)
        promoted["status"] = "promoted"
        promoted["reason"] = (
            f"FD04A's current GPX already covers Shane's Trail segment {segment_id} "
            "end-to-end before FD19C after the calendar reorder. Claiming it on FD04A "
            "lets FD19C be removed while preserving 251/251 official coverage."
        )
        promoted["source_action"] = "remove_route_card"
        promoted["from"] = {
            "package_number": OWNER_PACKAGE_NUMBER,
            "candidate_id": OWNER_CANDIDATE_ID,
            "field_menu_label": OWNER_LABEL,
            "outing_id": OWNER_OUTING_ID,
        }
        promoted["to"] = {
            **(promoted.get("to") or {}),
            "package_number": SOURCE_PACKAGE_NUMBER,
            "candidate_id": SOURCE_CANDIDATE_ID,
            "field_menu_label": SOURCE_LABEL,
            "outing_id": SOURCE_OUTING_ID,
        }
        promoted["promotion_notes"] = [
            "This is segment ownership, not phone-progress credit.",
            "FD19C removal is only valid after regenerated phone cues show the Shane's segments as FD04A credit.",
        ]
        rows.append(promoted)
    return sorted(rows, key=lambda item: sort_id(item.get("segment_id")))


def append_unique_promotions(existing_payload: dict[str, Any], new_rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload = copy.deepcopy(existing_payload)
    payload.setdefault("promotions", [])
    seen = {
        (
            str(row.get("segment_id")),
            str(((row.get("from") or {}).get("candidate_id")) or ""),
            str(((row.get("to") or {}).get("candidate_id")) or ""),
        )
        for row in payload.get("promotions") or []
    }
    for row in new_rows:
        key = (
            str(row.get("segment_id")),
            str(((row.get("from") or {}).get("candidate_id")) or ""),
            str(((row.get("to") or {}).get("candidate_id")) or ""),
        )
        if key not in seen:
            payload["promotions"].append(row)
            seen.add(key)
    payload["experiment_note"] = (
        "Experiment-only augmented promotions payload for FD04A -> FD19C. "
        "The canonical source file is not changed by this prepare step."
    )
    return payload


def render_prep_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# FD04A -> FD19C Route-Card Promotion Path Prep",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Outputs",
        "",
        f"- Segment promotions input: `{report['outputs']['segment_promotions_json']}`",
        f"- Reordered calendar input: `{report['outputs']['calendar_json']}`",
        "",
        "## Promotion Rows",
        "",
        "| Segment | Source | Target |",
        "|---|---|---|",
    ]
    for row in report["new_promotion_rows"]:
        lines.append(
            f"| `{row['segment_id']}` | `{(row.get('from') or {}).get('field_menu_label')}` | `{(row.get('to') or {}).get('field_menu_label')}` |"
        )
    lines.extend(
        [
            "",
            "## Calendar Reorder",
            "",
            f"- Scenario: `{report['calendar_reorder']['scenario_id']}`",
            f"- FD04A target date: `{report['calendar_reorder']['source_target_date']}`",
            f"- FD19 owner day target date: `{report['calendar_reorder']['owner_target_date']}`",
            f"- FD19C stays in the input calendar for the route-card promotion report to remove: `{report['calendar_reorder']['fd19c_loop_present_for_skip']}`",
            "",
        ]
    )
    return "\n".join(lines)


def prepare(args: argparse.Namespace) -> int:
    existing_promotions = read_json(args.segment_promotions_json)
    proof = read_json(args.proof_json)
    calendar = read_json(args.calendar_json)
    calendar_reorder = read_json(args.calendar_reorder_json)
    rows = route_promotion_rows(proof)
    if len(rows) != len(TARGET_SEGMENT_IDS):
        raise ValueError("Did not build all FD04A -> FD19C promotion rows")
    augmented_promotions = append_unique_promotions(existing_promotions, rows)
    reordered_calendar = materialize_reordered_calendar(calendar, calendar_reorder)
    write_json(args.output_promotions_json, augmented_promotions)
    write_json(args.output_calendar_json, reordered_calendar)
    scenario = scenario_for_id(calendar_reorder)
    fd19c_present = bool(
        [
            loop
            for assignment in reordered_calendar.get("assignments") or []
            for loop in ((assignment.get("field_day") or {}).get("loops") or [])
            if str(loop.get("candidate_id")) == OWNER_CANDIDATE_ID
        ]
    )
    report = {
        "schema": "boise_trails_fd04a_fd19c_route_card_promotion_path_prep_v1",
        "generated_at": now_iso(),
        "status": "prepared",
        "source_files": {
            "segment_promotions_json": display_path(args.segment_promotions_json),
            "proof_json": display_path(args.proof_json),
            "calendar_json": display_path(args.calendar_json),
            "calendar_reorder_json": display_path(args.calendar_reorder_json),
        },
        "outputs": {
            "segment_promotions_json": display_path(args.output_promotions_json),
            "calendar_json": display_path(args.output_calendar_json),
        },
        "new_promotion_rows": rows,
        "calendar_reorder": {
            "scenario_id": scenario["scenario_id"],
            "source_target_date": scenario["source_target_date"],
            "owner_target_date": scenario["owner_target_date"],
            "fd19c_loop_present_for_skip": fd19c_present,
        },
    }
    write_json(args.report_json, report)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(render_prep_markdown(report), encoding="utf-8")
    print(f"Wrote {display_path(args.output_promotions_json)}")
    print(f"Wrote {display_path(args.output_calendar_json)}")
    print(f"Wrote {display_path(args.report_json)}")
    print(f"Wrote {display_path(args.report_md)}")
    print(json.dumps({"status": report["status"], "promotion_rows": len(rows)}, indent=2))
    return 0


def package_component_for_candidate(map_data: dict[str, Any], candidate_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    for package in map_data.get("packages") or []:
        for component in package.get("components") or []:
            if str(component.get("candidate_id")) == candidate_id:
                return package, component
    raise KeyError(f"Candidate {candidate_id} not found in map data")


def remove_candidate_from_map_data(map_data: dict[str, Any], candidate_id: str) -> None:
    for package in map_data.get("packages") or []:
        package["components"] = [
            component
            for component in package.get("components") or []
            if str(component.get("candidate_id")) != candidate_id
        ]
    map_data.get("route_cues", {}).pop(candidate_id, None)
    for collection_name, collection in (map_data.get("feature_collections") or {}).items():
        features = collection.get("features") or []
        if collection_name == "official_segments":
            for feature in features:
                props = feature.get("properties") or {}
                segment_id = str(props.get("seg_id") or props.get("segment_id") or "")
                if segment_id in TARGET_SEGMENT_IDS:
                    props["candidate_id"] = SOURCE_CANDIDATE_ID
                    props["field_menu_label"] = SOURCE_LABEL
            continue
        collection["features"] = [
            feature
            for feature in features
            if str((feature.get("properties") or {}).get("candidate_id")) != candidate_id
        ]
    map_validation = map_data.get("map_validation") or {}
    map_validation["route_validations"] = [
        row
        for row in map_validation.get("route_validations") or []
        if str(row.get("candidate_id")) != candidate_id
    ]


def clean_empty_repeat_metadata(cue: dict[str, Any]) -> None:
    repeat_bearing_legs = []
    for key in ("start_access", "return_to_car"):
        leg = cue.get(key)
        if isinstance(leg, dict):
            repeat_bearing_legs.append(leg)
    repeat_bearing_legs.extend(link for link in cue.get("between_links") or [] if isinstance(link, dict))
    for leg in repeat_bearing_legs:
        if leg.get("official_repeat_segment_ids"):
            continue
        if "official_repeat_miles" in leg:
            leg["official_repeat_miles"] = 0.0
        if "connector_classes" in leg:
            leg["connector_classes"] = [
                value for value in leg.get("connector_classes") or [] if str(value) != "official_repeat"
            ]


def apply_fd04a_promotions_to_map_data(map_data: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    updated = copy.deepcopy(map_data)
    _package, component = package_component_for_candidate(updated, SOURCE_CANDIDATE_ID)
    target_cue = (updated.get("route_cues") or {}).get(SOURCE_CANDIDATE_ID)
    if not target_cue:
        raise KeyError(f"Route cue {SOURCE_CANDIDATE_ID} not found")
    for promotion in rows:
        segment_row = segment_row_from_source(
            segment_id=str(promotion.get("segment_id")),
            promotion=promotion,
            current_map=updated,
            official_by_id={},
            target_cue=target_cue,
        )
        add_segment_to_component(component, segment_row)
        update_cue_for_segment_promotion(
            target_cue,
            segment_row=segment_row,
            insert_after_segment_id=(promotion.get("to") or {}).get("insert_after_segment_id"),
            promotion=promotion,
        )
    clean_empty_repeat_metadata(target_cue)
    remove_candidate_from_map_data(updated, OWNER_CANDIDATE_ID)
    recompute_package_summary(updated, updated)
    updated["generated_at"] = now_iso()
    updated.setdefault("source_files", {})["fd04a_fd19c_route_card_promotion_experiment"] = "focused map-data mutation"
    return updated


def owner_loop_from_calendar(calendar: dict[str, Any]) -> dict[str, Any]:
    for assignment in calendar.get("assignments") or []:
        for loop in ((assignment.get("field_day") or {}).get("loops") or []):
            if str(loop.get("candidate_id")) == OWNER_CANDIDATE_ID:
                return {
                    "assignment_date": assignment.get("date"),
                    "draft_day_number": (assignment.get("field_day") or {}).get("draft_day_number"),
                    "loop": loop,
                }
    raise KeyError(f"No loop found for {OWNER_CANDIDATE_ID}")


def combined_promotion_report(
    *,
    base_report: dict[str, Any],
    calendar: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    owner = owner_loop_from_calendar(calendar)
    loop = owner["loop"]
    combined = copy.deepcopy(base_report)
    combined["schema"] = "boise_trails_combined_route_card_promotion_report_v1"
    combined["generated_at"] = now_iso()
    combined["objective"] = "Existing H1 route-card promotion report plus focused FD04A -> FD19C source removal."
    combined.setdefault("source_files", {})["fd04a_fd19c_segment_promotions"] = display_path(DEFAULT_PREP_PROMOTIONS_JSON)
    promotion = {
        "rank": len(combined.get("promotions") or []) + 1,
        "date": owner.get("assignment_date"),
        "draft_day_number": owner.get("draft_day_number"),
        "loop_id": loop.get("loop_id"),
        "source": loop.get("source"),
        "source_candidate_id": OWNER_CANDIDATE_ID,
        "route_card_candidate_id": SOURCE_CANDIDATE_ID,
        "label": OWNER_LABEL,
        "trailhead": loop.get("trailhead"),
        "official_miles": loop.get("official_miles"),
        "on_foot_miles": loop.get("on_foot_miles"),
        "p75_minutes": loop.get("p75_minutes"),
        "p90_minutes": loop.get("p90_minutes"),
        "mode": "removed_source_loop_after_segment_ownership_promotion",
        "skipped_route_card_source": True,
        "reassigned_segment_ids": TARGET_SEGMENT_IDS,
        "reassigned_to_candidate_id": SOURCE_CANDIDATE_ID,
        "reasons": [row.get("reason") for row in rows if row.get("reason")],
    }
    combined.setdefault("promotions", []).append(promotion)
    combined.setdefault("summary", {})["fd04a_fd19c_removed_route_card"] = OWNER_LABEL
    combined["summary"]["fd04a_fd19c_reassigned_segment_ids"] = TARGET_SEGMENT_IDS
    return combined


def materialize(args: argparse.Namespace) -> int:
    map_data = read_json(args.map_data_json)
    proof = read_json(args.proof_json)
    base_report = read_json(args.base_promotion_report_json)
    calendar = read_json(args.calendar_json)
    rows = route_promotion_rows(proof)
    focused_map_data = apply_fd04a_promotions_to_map_data(map_data, rows)
    promotion_report = combined_promotion_report(base_report=base_report, calendar=calendar, rows=rows)
    write_json(args.output_map_data_json, focused_map_data)
    write_json(args.output_promotion_report_json, promotion_report)
    print(f"Wrote {display_path(args.output_map_data_json)}")
    print(f"Wrote {display_path(args.output_promotion_report_json)}")
    print(
        json.dumps(
            {
                "status": "materialized",
                "route_card_count": len([c for p in focused_map_data.get("packages") or [] for c in p.get("components") or []]),
                "combined_promotion_rows": len(promotion_report.get("promotions") or []),
            },
            indent=2,
        )
    )
    return 0


def public_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: public_safe_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [public_safe_value(child) for child in value]
    if isinstance(value, str) and ("outputs/private" in value or "/Users/scott" in value):
        return "private-experiment-source"
    return value


def sanitize_layer(args: argparse.Namespace) -> int:
    layer = read_json(args.input_json)
    if "source_files" in layer:
        layer["source_files"] = public_safe_value(layer.get("source_files") or {})
    write_json(args.output_json, layer)
    print(f"Wrote {display_path(args.output_json)}")
    print(json.dumps({"status": "sanitized"}, indent=2))
    return 0


def route_matches(route: dict[str, Any], label: str | None = None, candidate_id: str | None = None, outing_id: str | None = None) -> bool:
    if label and str(route.get("label")) == label:
        return True
    if outing_id and str(route.get("outing_id")) == outing_id:
        return True
    if candidate_id and candidate_id in [str(value) for value in route.get("candidate_ids") or []]:
        return True
    return False


def find_route(routes: list[dict[str, Any]], **kwargs: str) -> dict[str, Any] | None:
    for route in routes:
        if route_matches(route, **kwargs):
            return route
    return None


def cue_segment_ids(route: dict[str, Any], key: str) -> set[str]:
    ids: set[str] = set()
    for cue in route.get("wayfinding_cues") or []:
        ids.update(normalized_ids(cue.get(key) or []))
    return ids


def target_credit_cue_rows(route: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    target_set = set(TARGET_SEGMENT_IDS)
    for cue in route.get("wayfinding_cues") or []:
        official_ids = set(normalized_ids(cue.get("official_segment_ids") or []))
        repeat_ids = set(normalized_ids(cue.get("official_repeat_segment_ids") or []))
        matched = sorted(target_set & official_ids, key=sort_id)
        if not matched:
            continue
        rows.append(
            {
                "seq": cue.get("seq"),
                "cue_type": cue.get("cue_type"),
                "official_segment_ids": matched,
                "also_repeat_target_ids": sorted(target_set & repeat_ids, key=sort_id),
                "signed_as": cue.get("signed_as") or [],
                "target": cue.get("target"),
                "until": cue.get("until"),
                "display_detail": cue.get("display_detail"),
            }
        )
    return rows


def field_day_contains_candidate(layer: dict[str, Any], candidate_id: str) -> bool:
    for day in layer.get("field_days") or []:
        for loop in day.get("loops") or []:
            candidates = set(str(value) for value in ((loop.get("route_card_ref") or {}).get("candidate_ids") or []))
            candidates.add(str(loop.get("candidate_id") or ""))
            if candidate_id in candidates:
                return True
    return False


def route_audit_row(audit: dict[str, Any], outing_id: str) -> dict[str, Any] | None:
    for key in ("routes", "route_reviews"):
        for row in audit.get(key) or []:
            if str(row.get("outing_id")) == outing_id or str(row.get("route_key")) == outing_id:
                return row
    return None


def official_segment_count(official_geojson: dict[str, Any]) -> int:
    return len(official_geojson.get("features") or [])


def build_verification_report(args: argparse.Namespace) -> dict[str, Any]:
    field_tool_data = read_json(args.field_tool_data_json)
    field_day_layer = read_json(args.field_day_layer_json)
    promote_report = read_json(args.promote_report_json)
    route_repeat = read_json(args.route_repeat_json)
    latent = read_json(args.latent_json)
    progress = read_json(args.progress_json)
    recertification = read_json(args.recertification_json)
    completion = read_json(args.completion_json)
    walkthrough = read_json(args.walkthrough_json)
    official_geojson = read_json(args.official_geojson)

    routes = field_tool_data.get("routes") or []
    fd04a = find_route(routes, label=SOURCE_LABEL, candidate_id=SOURCE_CANDIDATE_ID, outing_id=SOURCE_OUTING_ID)
    fd19c = find_route(routes, label=OWNER_LABEL, candidate_id=OWNER_CANDIDATE_ID, outing_id=OWNER_OUTING_ID)
    fd04a_claimed_ids = set(normalized_ids((fd04a or {}).get("segment_ids") or []))
    fd04a_credit_cue_ids = cue_segment_ids(fd04a or {}, "official_segment_ids")
    fd04a_repeat_cue_ids = cue_segment_ids(fd04a or {}, "official_repeat_segment_ids")
    target_set = set(TARGET_SEGMENT_IDS)
    fd04a_repeat = route_audit_row(route_repeat, SOURCE_OUTING_ID) or {}
    fd04a_latent = route_audit_row(latent, SOURCE_OUTING_ID) or {}
    field_day_summary = field_day_layer.get("summary") or {}
    completion_summary = completion.get("summary") or {}
    route_repeat_summary = route_repeat.get("summary") or {}
    latent_summary = latent.get("summary") or {}
    progress_summary = progress.get("summary") or {}
    recert_summary = recertification.get("summary") or {}
    official_count = official_segment_count(official_geojson)
    promote_summary = promote_report.get("summary") or {}
    skipped_fd19c_rows = [
        row
        for row in promote_report.get("promotions") or []
        if row.get("mode") == "removed_source_loop_after_segment_ownership_promotion"
        and str(row.get("source_candidate_id")) == OWNER_CANDIDATE_ID
    ]
    freestone_terms = ("freestone", "shane", "three-bears", "mountain-cove", "military")
    new_freestone_promotions = [
        row
        for row in promote_report.get("promotions") or []
        if row.get("mode") == "promoted_candidate_to_route_card_source"
        and any(
            term
            in " ".join(
                str(row.get(key) or "")
                for key in ("label", "source_candidate_id", "route_card_candidate_id", "loop_id")
            ).lower()
            for term in freestone_terms
        )
    ]

    gates = {
        "fd04a_route_present": "passed" if fd04a else "blocked",
        "fd19c_removed_from_routes": "passed" if fd19c is None else "blocked",
        "fd19c_removed_from_field_day_layer": "passed"
        if not field_day_contains_candidate(field_day_layer, OWNER_CANDIDATE_ID)
        else "blocked",
        "promotion_report_skipped_fd19c": "passed" if skipped_fd19c_rows else "blocked",
        "fd04a_claims_target_segments": "passed" if target_set <= fd04a_claimed_ids else "blocked",
        "fd04a_phone_cues_claim_target_segments": "passed" if target_set <= fd04a_credit_cue_ids else "blocked",
        "fd04a_target_segments_not_left_as_repeat_only": "passed" if not (target_set & fd04a_repeat_cue_ids - fd04a_credit_cue_ids) else "blocked",
        "coverage_251_preserved": "passed"
        if int_value((field_tool_data.get("summary") or {}).get("segment_count_in_field_menu")) == official_count == 251
        and int_value(completion_summary.get("field_menu_segment_count")) == 251
        else "blocked",
        "field_day_reassignment_p90_date_route_audit": "passed"
        if field_day_layer.get("publication_status") == "field_day_certified"
        and int_value(field_day_summary.get("schedule_p90_violation_day_count")) == 0
        and int_value(field_day_summary.get("needs_route_card_promotion_loop_count")) == 0
        and int_value(field_day_summary.get("needs_route_card_audit_fix_loop_count")) == 0
        else "blocked",
        "route_repeat_audit_clean": "passed"
        if int_value(route_repeat_summary.get("hidden_self_repeat_segment_count")) == 0
        and int_value(route_repeat_summary.get("unpriced_repeat_segment_count")) == 0
        and int_value(route_repeat_summary.get("latent_credit_segment_count")) == 0
        and not fd04a_repeat.get("hidden_self_repeat_ids")
        and not fd04a_repeat.get("unpriced_repeat_ids")
        and not fd04a_repeat.get("latent_credit_ids")
        else "blocked",
        "latent_credit_reconciled": "passed"
        if int_value(latent_summary.get("unclaimed_uncompleted_segment_count")) == 0
        and not (target_set & set(normalized_ids(fd04a_latent.get("latent_completed_segment_ids") or [])))
        else "blocked",
        "progress_and_recertification_pass": "passed"
        if progress_summary.get("remaining_coverage_preserved") is True
        and recertification.get("status") == "passed"
        and recert_summary.get("remaining_full_completion_feasible") is True
        else "blocked",
        "field_tool_completion_pass": "passed" if completion.get("status") == "passed" else "blocked",
        "field_route_walkthrough_pass": "passed" if walkthrough.get("status") == "passed" else "blocked",
        "no_new_freestone_mega_routes": "passed" if not new_freestone_promotions else "blocked",
    }
    status = "passed" if all(value == "passed" for value in gates.values()) else "blocked"
    return {
        "schema": "boise_trails_fd04a_fd19c_route_card_promotion_path_verification_v1",
        "generated_at": now_iso(),
        "status": status,
        "source_files": {
            "field_tool_data_json": display_path(args.field_tool_data_json),
            "field_day_layer_json": display_path(args.field_day_layer_json),
            "promote_report_json": display_path(args.promote_report_json),
            "route_repeat_json": display_path(args.route_repeat_json),
            "latent_json": display_path(args.latent_json),
            "progress_json": display_path(args.progress_json),
            "recertification_json": display_path(args.recertification_json),
            "completion_json": display_path(args.completion_json),
            "walkthrough_json": display_path(args.walkthrough_json),
        },
        "summary": {
            "route_count": len(routes),
            "official_segment_count": official_count,
            "field_menu_segment_count": int_value((field_tool_data.get("summary") or {}).get("segment_count_in_field_menu")),
            "field_day_count": field_day_summary.get("field_day_count"),
            "field_day_loop_count": field_day_summary.get("loop_count"),
            "total_p75_minutes": field_day_summary.get("total_p75_minutes"),
            "max_p90_minutes": field_day_summary.get("max_p90_minutes"),
            "skipped_source_loop_count": field_day_summary.get("skipped_source_loop_count"),
            "promotion_report_skipped_fd19c_count": len(skipped_fd19c_rows),
            "new_freestone_route_card_promotion_count": len(new_freestone_promotions),
        },
        "fd04a": {
            "outing_id": (fd04a or {}).get("outing_id"),
            "label": (fd04a or {}).get("label"),
            "candidate_ids": (fd04a or {}).get("candidate_ids") or [],
            "segment_ids": normalized_ids((fd04a or {}).get("segment_ids") or []),
            "target_credit_cue_rows": target_credit_cue_rows(fd04a or {}),
        },
        "fd19c": {
            "present_in_routes": fd19c is not None,
            "present_in_field_day_layer": field_day_contains_candidate(field_day_layer, OWNER_CANDIDATE_ID),
            "skipped_rows": skipped_fd19c_rows,
        },
        "new_freestone_route_card_promotions": new_freestone_promotions,
        "audit_summaries": {
            "route_repeat": route_repeat_summary,
            "latent": latent_summary,
            "progress": progress_summary,
            "recertification_status": recertification.get("status"),
            "completion": completion_summary,
            "walkthrough": walkthrough.get("summary") or {},
        },
        "gates": gates,
    }


def render_verify_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# FD04A -> FD19C Route-Card Promotion Path",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Result",
        "",
        f"- Route count: {summary['route_count']}",
        f"- Coverage: {summary['field_menu_segment_count']}/{summary['official_segment_count']}",
        f"- Field-day loops: {summary['field_day_loop_count']} across {summary['field_day_count']} days",
        f"- Total field-day p75: {summary['total_p75_minutes']}",
        f"- Max field-day p90: {summary['max_p90_minutes']}",
        f"- FD19C skipped by promotion report: {summary['promotion_report_skipped_fd19c_count']}",
        "",
        "## Gates",
        "",
        "| Gate | Status |",
        "|---|---|",
    ]
    for gate, status in report["gates"].items():
        lines.append(f"| `{gate}` | `{status}` |")
    lines.extend(["", "## FD04A Credit Cues", "", "| Cue | Segments | Target | Until |", "|---:|---|---|---|"])
    for cue in report["fd04a"]["target_credit_cue_rows"]:
        lines.append(
            f"| {cue.get('seq')} | `{', '.join(cue.get('official_segment_ids') or [])}` | {cue.get('target')} | {cue.get('until')} |"
        )
    lines.append("")
    return "\n".join(lines)


def verify(args: argparse.Namespace) -> int:
    report = build_verification_report(args)
    write_json(args.verify_report_json, report)
    args.verify_report_md.parent.mkdir(parents=True, exist_ok=True)
    args.verify_report_md.write_text(render_verify_markdown(report), encoding="utf-8")
    print(f"Wrote {display_path(args.verify_report_json)}")
    print(f"Wrote {display_path(args.verify_report_md)}")
    print(json.dumps({"status": report["status"], "summary": report["summary"]}, indent=2))
    return 0 if report["status"] == "passed" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare")
    prep.add_argument("--segment-promotions-json", type=Path, default=DEFAULT_SEGMENT_PROMOTIONS_JSON)
    prep.add_argument("--calendar-json", type=Path, default=DEFAULT_CALENDAR_JSON)
    prep.add_argument("--proof-json", type=Path, default=DEFAULT_PROOF_JSON)
    prep.add_argument("--calendar-reorder-json", type=Path, default=DEFAULT_CALENDAR_REORDER_JSON)
    prep.add_argument("--output-promotions-json", type=Path, default=DEFAULT_PREP_PROMOTIONS_JSON)
    prep.add_argument("--output-calendar-json", type=Path, default=DEFAULT_PREP_CALENDAR_JSON)
    prep.add_argument("--report-json", type=Path, default=DEFAULT_PREP_REPORT_JSON)
    prep.add_argument("--report-md", type=Path, default=DEFAULT_PREP_REPORT_MD)

    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--map-data-json", type=Path, default=DEFAULT_BASE_MAP_DATA_JSON)
    materialize_parser.add_argument("--proof-json", type=Path, default=DEFAULT_PROOF_JSON)
    materialize_parser.add_argument("--base-promotion-report-json", type=Path, default=DEFAULT_BASE_PROMOTION_REPORT_JSON)
    materialize_parser.add_argument("--calendar-json", type=Path, default=DEFAULT_PREP_CALENDAR_JSON)
    materialize_parser.add_argument("--output-map-data-json", type=Path, default=DEFAULT_FOCUSED_MAP_DATA_JSON)
    materialize_parser.add_argument("--output-promotion-report-json", type=Path, default=DEFAULT_COMBINED_PROMOTION_REPORT_JSON)

    sanitize_parser = subparsers.add_parser("sanitize-layer")
    sanitize_parser.add_argument("--input-json", type=Path, default=DEFAULT_FINAL_FIELD_DAY_LAYER_JSON)
    sanitize_parser.add_argument("--output-json", type=Path, default=DEFAULT_SAFE_FIELD_DAY_LAYER_JSON)

    check = subparsers.add_parser("verify")
    check.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FINAL_FIELD_TOOL_DATA_JSON)
    check.add_argument("--field-day-layer-json", type=Path, default=DEFAULT_FINAL_FIELD_DAY_LAYER_JSON)
    check.add_argument("--promote-report-json", type=Path, default=DEFAULT_PROMOTED_FIELD_DAY_REPORT_JSON)
    check.add_argument("--route-repeat-json", type=Path, default=DEFAULT_ROUTE_REPEAT_JSON)
    check.add_argument("--latent-json", type=Path, default=DEFAULT_LATENT_JSON)
    check.add_argument("--progress-json", type=Path, default=DEFAULT_PROGRESS_JSON)
    check.add_argument("--recertification-json", type=Path, default=DEFAULT_RECERTIFICATION_JSON)
    check.add_argument("--completion-json", type=Path, default=DEFAULT_COMPLETION_JSON)
    check.add_argument("--walkthrough-json", type=Path, default=DEFAULT_WALKTHROUGH_JSON)
    check.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    check.add_argument("--verify-report-json", type=Path, default=DEFAULT_VERIFY_REPORT_JSON)
    check.add_argument("--verify-report-md", type=Path, default=DEFAULT_VERIFY_REPORT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        return prepare(args)
    if args.command == "materialize":
        return materialize(args)
    if args.command == "sanitize-layer":
        return sanitize_layer(args)
    if args.command == "verify":
        return verify(args)
    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
