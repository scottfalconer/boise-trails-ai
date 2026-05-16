#!/usr/bin/env python3
"""Test the focused FD04A -> FD19C segment-ownership promotion path."""

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
from latent_credit_delta_repricing_audit import (  # noqa: E402
    display_path,
    float_value,
    normalized_ids,
    rounded,
    sort_id,
    write_json,
)


DEFAULT_FIELD_TOOL_DATA_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_ROUTE_REPEAT_AUDIT_JSON = YEAR_DIR / "checkpoints" / "route-repeat-optimization-audit-2026-05-12.json"
DEFAULT_FIELD_LATENT_AUDIT_JSON = YEAR_DIR / "checkpoints" / "field-latent-credit-audit-2026-05-11.json"
DEFAULT_CALENDAR_REORDER_JSON = YEAR_DIR / "checkpoints" / "calendar-reorder-latent-credit-experiment-2026-05-12.json"
DEFAULT_OFFICIAL_GEOJSON = YEAR_DIR / "inputs" / "official" / "api-pull-2026-05-04" / "official_foot_segments.geojson"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "fd04a-fd19c-credit-promotion-experiment-2026-05-12.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "fd04a-fd19c-credit-promotion-experiment-2026-05-12.md"
DEFAULT_MANIFEST_JSON = YEAR_DIR / "checkpoints" / "fd04a-fd19c-credit-promotion-experiment-2026-05-12-manifest.json"

SOURCE_OUTING_ID = "104-1"
SOURCE_LABEL = "FD04A"
SOURCE_CANDIDATE_ID = "combo-two-point-femrites-patrol-shanes-connector"
OWNER_OUTING_ID = "119-3"
OWNER_LABEL = "FD19C"
OWNER_CANDIDATE_ID = "shanes-trail"
TARGET_SEGMENT_IDS = ["1649", "1650", "1651"]
SCENARIO_ID = "104-1-before-119-3"

INSERT_AFTER_BY_SEGMENT = {
    "1650": "1748",
    "1651": "1652",
    "1649": "1558",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def route_key(route: dict[str, Any]) -> str:
    return str(route.get("outing_id") or route.get("id") or route.get("label") or "")


def route_index(field_tool_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for route in field_tool_data.get("routes") or []:
        keys = [route_key(route), str(route.get("label") or "")]
        keys.extend(str(candidate_id) for candidate_id in route.get("candidate_ids") or [])
        for key in keys:
            if key:
                rows[key] = route
    return rows


def route_repeat_index(audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for route in audit.get("routes") or []:
        keys = [str(route.get("route_key") or ""), str(route.get("outing_id") or ""), str(route.get("label") or "")]
        for key in keys:
            if key:
                rows[key] = route
    return rows


def latent_route_index(audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for route in audit.get("route_reviews") or []:
        keys = [str(route.get("route_key") or ""), str(route.get("outing_id") or ""), str(route.get("label") or "")]
        for key in keys:
            if key:
                rows[key] = route
    return rows


def official_index(official_geojson: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for feature in official_geojson.get("features") or []:
        props = feature.get("properties") or {}
        segment_id = str(props.get("seg_id") or props.get("segId") or props.get("segment_id") or "")
        if not segment_id:
            continue
        normalized = dict(props)
        normalized.setdefault("seg_id", segment_id)
        normalized.setdefault("seg_name", props.get("segName"))
        normalized.setdefault("trail_name", props.get("trailName") or props.get("trail_name"))
        rows[segment_id] = normalized
    return rows


def all_claimed_segment_ids(routes: list[dict[str, Any]]) -> set[str]:
    claimed: set[str] = set()
    for route in routes:
        claimed.update(normalized_ids(route.get("segment_ids") or []))
    return claimed


def segment_ids_in_wayfinding(route: dict[str, Any], key: str) -> set[str]:
    ids: set[str] = set()
    for cue in route.get("wayfinding_cues") or []:
        ids.update(normalized_ids(cue.get(key) or []))
    return ids


def repeat_cue_rows(route: dict[str, Any], target_ids: set[str]) -> list[dict[str, Any]]:
    rows = []
    for cue in route.get("wayfinding_cues") or []:
        repeat_ids = set(normalized_ids(cue.get("official_repeat_segment_ids") or []))
        official_ids = set(normalized_ids(cue.get("official_segment_ids") or []))
        for segment_id in sorted(target_ids & repeat_ids, key=sort_id):
            rows.append(
                {
                    "segment_id": segment_id,
                    "current_cue_seq": cue.get("seq"),
                    "current_cue_type": cue.get("cue_type"),
                    "currently_repeat": True,
                    "currently_official": segment_id in official_ids,
                    "signed_as": cue.get("signed_as") or [],
                    "target": cue.get("target"),
                    "until": cue.get("until"),
                    "display_detail": cue.get("display_detail"),
                    "official_repeat_miles": cue.get("official_repeat_miles"),
                }
            )
    return sorted(rows, key=lambda row: (int(row.get("current_cue_seq") or 0), sort_id(row.get("segment_id"))))


def source_credit_support(source_route: dict[str, Any], target_ids: set[str]) -> dict[str, Any]:
    claimed = set(normalized_ids(source_route.get("segment_ids") or []))
    repeat_cued = segment_ids_in_wayfinding(source_route, "official_repeat_segment_ids")
    official_cued = segment_ids_in_wayfinding(source_route, "official_segment_ids")
    declared_elsewhere = set(
        normalized_ids(
            ((source_route.get("segment_ownership_reconciliation") or {}).get("declared_owned_elsewhere_segment_ids") or [])
        )
    )
    repeat_rows = repeat_cue_rows(source_route, target_ids)
    return {
        "source_claims_all_target_ids_now": target_ids <= claimed,
        "source_cues_all_target_ids_as_repeat_now": target_ids <= repeat_cued,
        "source_cues_all_target_ids_as_official_now": target_ids <= official_cued,
        "source_declares_all_target_ids_owned_elsewhere_now": target_ids <= declared_elsewhere,
        "target_repeat_cue_rows": repeat_rows,
        "status": "requires_route_card_claim_and_cue_promotion"
        if not target_ids <= claimed
        else "already_claimed_by_source",
    }


def direction_and_coverage_proof(
    *,
    source_repeat_row: dict[str, Any],
    source_latent_row: dict[str, Any],
    official_by_id: dict[str, dict[str, Any]],
    target_ids: set[str],
) -> dict[str, Any]:
    actual_full = set(normalized_ids(source_repeat_row.get("actual_full_segment_ids") or []))
    latent_ids = set(normalized_ids(source_latent_row.get("latent_completed_segment_ids") or []))
    segment_rows = {
        str(row.get("seg_id")): row
        for row in source_latent_row.get("segments") or []
        if str(row.get("seg_id")) in target_ids
    }
    direction_rows = []
    for segment_id in sorted(target_ids, key=sort_id):
        official = official_by_id.get(segment_id) or {}
        latent_segment = segment_rows.get(segment_id) or {}
        direction = str(official.get("direction") or latent_segment.get("direction") or "both")
        direction_rows.append(
            {
                "segment_id": segment_id,
                "segment_name": official.get("seg_name") or latent_segment.get("seg_name"),
                "trail_name": official.get("trail_name") or latent_segment.get("trail_name"),
                "direction_rule": direction,
                "coverage_status": "actual_full_and_reconciled_owned_elsewhere"
                if segment_id in actual_full and segment_id in latent_ids and latent_segment.get("status") == "reconciled_owned_elsewhere"
                else "not_proven",
                "direction_status": "passed_no_ascent_direction_requirement"
                if direction == "both"
                else "requires_ascent_direction_review",
            }
        )
    return {
        "source_repeat_audit_status": source_repeat_row.get("audit_status"),
        "source_latent_audit_status": source_latent_row.get("audit_status"),
        "target_ids_actual_full_in_source_gpx": sorted(target_ids & actual_full, key=sort_id),
        "target_ids_latent_reconciled_in_source_audit": sorted(target_ids & latent_ids, key=sort_id),
        "missing_actual_full_ids": sorted(target_ids - actual_full, key=sort_id),
        "missing_latent_audit_ids": sorted(target_ids - latent_ids, key=sort_id),
        "segments": direction_rows,
        "status": "passed"
        if all(
            row["coverage_status"] == "actual_full_and_reconciled_owned_elsewhere"
            and row["direction_status"] == "passed_no_ascent_direction_requirement"
            for row in direction_rows
        )
        else "blocked",
    }


def coverage_after_hypothetical_promotion(
    *,
    routes: list[dict[str, Any]],
    official_segment_count: int,
    source_route: dict[str, Any],
    owner_route: dict[str, Any],
    target_ids: set[str],
) -> dict[str, Any]:
    before = all_claimed_segment_ids(routes)
    source_ids_after = set(normalized_ids(source_route.get("segment_ids") or [])) | target_ids
    owner_ids = set(normalized_ids(owner_route.get("segment_ids") or []))
    after: set[str] = set()
    for route in routes:
        label = str(route.get("label") or "")
        if label == OWNER_LABEL:
            continue
        if label == SOURCE_LABEL:
            after.update(source_ids_after)
        else:
            after.update(normalized_ids(route.get("segment_ids") or []))
    return {
        "route_count_before": len(routes),
        "route_count_after": len(routes) - 1,
        "removed_route_label": OWNER_LABEL,
        "promoted_source_label": SOURCE_LABEL,
        "source_claimed_ids_before": normalized_ids(source_route.get("segment_ids") or []),
        "source_claimed_ids_after": sorted(source_ids_after, key=sort_id),
        "owner_claimed_ids_removed": sorted(owner_ids, key=sort_id),
        "covered_segment_count_before": len(before),
        "covered_segment_count_after": len(after),
        "official_segment_count": official_segment_count,
        "missing_after_ids": sorted(before - after, key=sort_id),
        "new_extra_ids": sorted(after - before, key=sort_id),
        "status": "passed" if len(after) == official_segment_count and not before - after else "blocked",
    }


def calendar_scenario(calendar_reorder: dict[str, Any]) -> dict[str, Any]:
    for scenario in calendar_reorder.get("pairwise_scenarios") or []:
        if scenario.get("scenario_id") == SCENARIO_ID:
            return scenario
    return {}


def proposed_promotion_rows(
    *,
    source_route: dict[str, Any],
    target_ids: set[str],
    official_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    repeat_rows = {row["segment_id"]: row for row in repeat_cue_rows(source_route, target_ids)}
    rows = []
    for segment_id in sorted(target_ids, key=lambda value: (int(repeat_rows.get(value, {}).get("current_cue_seq") or 99), sort_id(value))):
        official = official_by_id.get(segment_id) or {}
        cue = repeat_rows.get(segment_id) or {}
        signed_as = " / ".join(str(value) for value in cue.get("signed_as") or []) or str(official.get("trail_name") or "")
        rows.append(
            {
                "status": "promoted_candidate_row_not_applied",
                "segment_id": int(segment_id) if segment_id.isdigit() else segment_id,
                "segment_name": official.get("seg_name"),
                "from": {
                    "candidate_id": OWNER_CANDIDATE_ID,
                    "field_menu_label": OWNER_LABEL,
                    "outing_id": OWNER_OUTING_ID,
                },
                "to": {
                    "candidate_id": SOURCE_CANDIDATE_ID,
                    "field_menu_label": SOURCE_LABEL,
                    "outing_id": SOURCE_OUTING_ID,
                    "insert_after_segment_id": int(INSERT_AFTER_BY_SEGMENT.get(segment_id, "")) if INSERT_AFTER_BY_SEGMENT.get(segment_id, "").isdigit() else INSERT_AFTER_BY_SEGMENT.get(segment_id),
                },
                "current_repeat_cue_seq": cue.get("current_cue_seq"),
                "current_repeat_cue_signed_as": cue.get("signed_as") or [],
                "runner_facing_claim_text": f"Claim {official.get('seg_name') or segment_id} while following {signed_as}; keep following the blue route line through the Shane's/Freestone connector chain.",
                "source_action": "remove_route_card",
                "evidence": {
                    "field_latent_credit_audit_json": display_path(DEFAULT_FIELD_LATENT_AUDIT_JSON),
                    "route_key": SOURCE_OUTING_ID,
                    "required_status": "reconciled_owned_elsewhere",
                    "requires_audit_status_passed": True,
                    "net_effort_proof_json": display_path(DEFAULT_CALENDAR_REORDER_JSON),
                },
            }
        )
    return rows


def build_experiment(
    *,
    field_tool_data: dict[str, Any],
    route_repeat_audit: dict[str, Any],
    field_latent_audit: dict[str, Any],
    calendar_reorder: dict[str, Any],
    official_geojson: dict[str, Any],
) -> dict[str, Any]:
    routes = field_tool_data.get("routes") or []
    routes_by_key = route_index(field_tool_data)
    repeat_by_key = route_repeat_index(route_repeat_audit)
    latent_by_key = latent_route_index(field_latent_audit)
    official_by_id = official_index(official_geojson)
    source_route = routes_by_key[SOURCE_OUTING_ID]
    owner_route = routes_by_key.get(OWNER_OUTING_ID)
    source_repeat_row = repeat_by_key[SOURCE_OUTING_ID]
    source_latent_row = latent_by_key[SOURCE_OUTING_ID]
    target_ids = set(TARGET_SEGMENT_IDS)
    scenario = calendar_scenario(calendar_reorder)
    official_segment_count = len(official_by_id)

    if owner_route is None:
        claimed_ids = all_claimed_segment_ids(routes)
        source_ids = set(normalized_ids(source_route.get("segment_ids") or []))
        coverage_status = (
            "passed"
            if len(claimed_ids) == official_segment_count and target_ids <= source_ids
            else "blocked"
        )
        hard_gates = {
            "fd19c_removed_from_active_routes": "passed",
            "fd04a_claims_fd19c_segments": "passed" if target_ids <= source_ids else "blocked",
            "coverage_251_preserved": coverage_status,
        }
        status = (
            "active_packet_already_promoted"
            if all(value == "passed" for value in hard_gates.values())
            else "blocked_owner_route_missing"
        )
        return {
            "schema": "boise_trails_fd04a_fd19c_credit_promotion_experiment_v1",
            "generated_at": now_iso(),
            "objective": "Test whether FD04A can claim/cue FD19C Shane's Trail segments and remove FD19C without a Freestone mega-route.",
            "status": status,
            "decision": "superseded_by_active_packet_promotion",
            "source_files": {
                "field_tool_data_json": display_path(DEFAULT_FIELD_TOOL_DATA_JSON),
                "route_repeat_audit_json": display_path(DEFAULT_ROUTE_REPEAT_AUDIT_JSON),
                "field_latent_credit_audit_json": display_path(DEFAULT_FIELD_LATENT_AUDIT_JSON),
                "calendar_reorder_json": display_path(DEFAULT_CALENDAR_REORDER_JSON),
                "official_geojson": display_path(DEFAULT_OFFICIAL_GEOJSON),
            },
            "routes": {
                "source": {
                    "outing_id": SOURCE_OUTING_ID,
                    "label": SOURCE_LABEL,
                    "candidate_id": SOURCE_CANDIDATE_ID,
                    "segment_ids_before": normalized_ids(source_route.get("segment_ids") or []),
                    "on_foot_miles": rounded(source_route.get("on_foot_miles")),
                    "p75_minutes": source_route.get("door_to_door_minutes_p75"),
                    "p90_minutes": source_route.get("door_to_door_minutes_p90"),
                },
                "owner_to_remove": None,
            },
            "target_segment_ids": TARGET_SEGMENT_IDS,
            "source_current_credit_support": source_credit_support(source_route, target_ids),
            "fd04a_gpx_segment_proof": {
                "status": "not_applicable_active_packet_already_promoted",
                "segments": [],
            },
            "route_repeat_gate": {
                "status": "not_applicable_active_packet_already_promoted",
            },
            "hypothetical_after_promotion": {
                "status": coverage_status,
                "route_count_after": len(routes),
                "covered_segment_count_after": len(claimed_ids),
                "official_segment_count": official_segment_count,
                "missing_after_ids": sorted(set(official_by_id) - claimed_ids, key=sort_id),
            },
            "calendar_reorder_scenario": {
                "scenario_id": scenario.get("scenario_id") if scenario else None,
                "status": scenario.get("status") if scenario else None,
                "source_target_date": scenario.get("source_target_date") if scenario else None,
                "owner_target_date": scenario.get("owner_target_date") if scenario else None,
                "removed_owner_route_label": scenario.get("removed_owner_route_label") if scenario else None,
                "saved_on_foot_miles": scenario.get("saved_on_foot_miles") if scenario else None,
                "saved_p75_minutes": scenario.get("saved_p75_minutes") if scenario else None,
                "saved_p90_minutes": scenario.get("saved_p90_minutes") if scenario else None,
                "route_count_after": scenario.get("route_count_after") if scenario else len(routes),
                "owner_day_after_removal": scenario.get("owner_day_after_removal") if scenario else None,
                "source_day_after_reorder": scenario.get("source_day_after_reorder") if scenario else None,
            },
            "proposed_segment_promotion_rows": [],
            "hard_gates": hard_gates,
            "summary": {
                "current_active_route_count": len(routes),
                "hypothetical_route_count_after": len(routes),
                "saved_on_foot_miles_if_promoted": rounded((scenario or {}).get("saved_on_foot_miles")),
                "saved_p75_minutes_if_promoted": int((scenario or {}).get("saved_p75_minutes") or 0),
                "saved_p90_minutes_if_promoted": int((scenario or {}).get("saved_p90_minutes") or 0),
                "official_coverage_after_hypothetical_promotion": f"{len(claimed_ids)}/{official_segment_count}",
                "active_packet_mutated": True,
            },
            "remaining_steps_for_real_promotion": [
                "Use fd04a_fd19c_route_card_promotion_path_experiment.py verify for the active-packet certification gate.",
            ],
        }

    source_support = source_credit_support(source_route, target_ids)
    proof = direction_and_coverage_proof(
        source_repeat_row=source_repeat_row,
        source_latent_row=source_latent_row,
        official_by_id=official_by_id,
        target_ids=target_ids,
    )
    coverage = coverage_after_hypothetical_promotion(
        routes=routes,
        official_segment_count=official_segment_count,
        source_route=source_route,
        owner_route=owner_route,
        target_ids=target_ids,
    )
    repeat_gate = {
        "source_route_repeat_status": source_repeat_row.get("audit_status"),
        "hidden_self_repeat_ids_now": normalized_ids(source_repeat_row.get("hidden_self_repeat_ids") or []),
        "unpriced_repeat_ids_now": normalized_ids(source_repeat_row.get("unpriced_repeat_ids") or []),
        "latent_credit_ids_now": normalized_ids(source_repeat_row.get("latent_credit_ids") or []),
        "declared_repeat_segment_ids": normalized_ids(source_repeat_row.get("declared_repeat_segment_ids") or []),
        "target_ids_are_declared_repeat_now": sorted(target_ids & set(normalized_ids(source_repeat_row.get("declared_repeat_segment_ids") or [])), key=sort_id),
        "status": "passed"
        if source_repeat_row.get("audit_status") == "passed"
        and not source_repeat_row.get("hidden_self_repeat_ids")
        and not source_repeat_row.get("unpriced_repeat_ids")
        and target_ids <= set(normalized_ids(source_repeat_row.get("declared_repeat_segment_ids") or []))
        else "blocked",
    }
    proposed_rows = proposed_promotion_rows(source_route=source_route, target_ids=target_ids, official_by_id=official_by_id)
    hard_gates = {
        "fd04a_gpx_full_covers_fd19c_segments": proof["status"],
        "route_repeat_hard_gate": repeat_gate["status"],
        "hypothetical_coverage_after_fd19c_removal": coverage["status"],
        "calendar_reorder_supported": "passed" if scenario else "missing_calendar_reorder_scenario",
        "phone_visible_claim_cues_can_be_generated": "passed"
        if len(proposed_rows) == len(target_ids) and all(row["runner_facing_claim_text"] for row in proposed_rows)
        else "blocked",
    }
    active_mutation_accepted = False
    ready = all(value == "passed" for value in hard_gates.values())
    status = "ready_for_controlled_source_promotion" if ready else "blocked_keep_current_cards"
    return {
        "schema": "boise_trails_fd04a_fd19c_credit_promotion_experiment_v1",
        "generated_at": now_iso(),
        "objective": "Test whether FD04A can claim/cue FD19C Shane's Trail segments and remove FD19C without a Freestone mega-route.",
        "status": status,
        "decision": "experiment_only_no_active_packet_mutation",
        "source_files": {
            "field_tool_data_json": display_path(DEFAULT_FIELD_TOOL_DATA_JSON),
            "route_repeat_audit_json": display_path(DEFAULT_ROUTE_REPEAT_AUDIT_JSON),
            "field_latent_credit_audit_json": display_path(DEFAULT_FIELD_LATENT_AUDIT_JSON),
            "calendar_reorder_json": display_path(DEFAULT_CALENDAR_REORDER_JSON),
            "official_geojson": display_path(DEFAULT_OFFICIAL_GEOJSON),
        },
        "routes": {
            "source": {
                "outing_id": SOURCE_OUTING_ID,
                "label": SOURCE_LABEL,
                "candidate_id": SOURCE_CANDIDATE_ID,
                "segment_ids_before": normalized_ids(source_route.get("segment_ids") or []),
                "on_foot_miles": rounded(source_route.get("on_foot_miles")),
                "p75_minutes": source_route.get("door_to_door_minutes_p75"),
                "p90_minutes": source_route.get("door_to_door_minutes_p90"),
            },
            "owner_to_remove": {
                "outing_id": OWNER_OUTING_ID,
                "label": OWNER_LABEL,
                "candidate_id": OWNER_CANDIDATE_ID,
                "segment_ids": normalized_ids(owner_route.get("segment_ids") or []),
                "on_foot_miles": rounded(owner_route.get("on_foot_miles")),
                "p75_minutes": owner_route.get("door_to_door_minutes_p75"),
                "p90_minutes": owner_route.get("door_to_door_minutes_p90"),
            },
        },
        "target_segment_ids": TARGET_SEGMENT_IDS,
        "source_current_credit_support": source_support,
        "fd04a_gpx_segment_proof": proof,
        "route_repeat_gate": repeat_gate,
        "hypothetical_after_promotion": coverage,
        "calendar_reorder_scenario": {
            "scenario_id": scenario.get("scenario_id"),
            "status": scenario.get("status"),
            "source_target_date": scenario.get("source_target_date"),
            "owner_target_date": scenario.get("owner_target_date"),
            "removed_owner_route_label": scenario.get("removed_owner_route_label"),
            "saved_on_foot_miles": scenario.get("saved_on_foot_miles"),
            "saved_p75_minutes": scenario.get("saved_p75_minutes"),
            "saved_p90_minutes": scenario.get("saved_p90_minutes"),
            "route_count_after": scenario.get("route_count_after"),
            "owner_day_after_removal": scenario.get("owner_day_after_removal"),
            "source_day_after_reorder": scenario.get("source_day_after_reorder"),
        },
        "proposed_segment_promotion_rows": proposed_rows,
        "hard_gates": hard_gates,
        "summary": {
            "current_active_route_count": len(routes),
            "hypothetical_route_count_after": coverage["route_count_after"],
            "saved_on_foot_miles_if_promoted": rounded(owner_route.get("on_foot_miles")),
            "saved_p75_minutes_if_promoted": int(owner_route.get("door_to_door_minutes_p75") or 0),
            "saved_p90_minutes_if_promoted": int(owner_route.get("door_to_door_minutes_p90") or 0),
            "official_coverage_after_hypothetical_promotion": f"{coverage['covered_segment_count_after']}/{coverage['official_segment_count']}",
            "active_packet_mutated": active_mutation_accepted,
        },
        "remaining_steps_for_real_promotion": [
            "Append the proposed segment-promotion rows to years/2026/inputs/personal/2026-cross-package-segment-promotions-v1.json with status promoted.",
            "Regenerate canonical route-card source so FD04A owns 1649/1650/1651 and FD19C is skipped.",
            "Regenerate the mobile field packet and field-day layer.",
            "Rerun latent, official-repeat, route-repeat, progress, recertification, completion, walkthrough, and pytest gates.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    if report["status"] == "active_packet_already_promoted":
        mode_note = "The active packet already reflects this promotion. This historical experiment is superseded by the active route-card promotion-path verifier."
    else:
        mode_note = "This is an experiment-only artifact. It does not mutate the active packet or remove `FD19C` by itself."
    lines = [
        "# FD04A -> FD19C Credit Promotion Experiment",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"Status: `{report['status']}`",
        "",
        mode_note,
        "",
        "## Result",
        "",
        f"- Hypothetical route count: {summary['current_active_route_count']} -> {summary['hypothetical_route_count_after']}",
        f"- Hypothetical saved effort: {summary['saved_on_foot_miles_if_promoted']} mi / {summary['saved_p75_minutes_if_promoted']} p75 / {summary['saved_p90_minutes_if_promoted']} p90",
        f"- Official coverage after hypothetical promotion: {summary['official_coverage_after_hypothetical_promotion']}",
        f"- Active packet mutated: `{summary['active_packet_mutated']}`",
        "",
        "## Hard Gates",
        "",
        "| Gate | Status |",
        "|---|---|",
    ]
    for gate, status in report["hard_gates"].items():
        lines.append(f"| `{gate}` | `{status}` |")
    lines.extend(["", "## Segment Proof", "", "| Segment | Coverage | Direction |", "|---|---|---|"])
    for row in report["fd04a_gpx_segment_proof"]["segments"]:
        lines.append(f"| `{row['segment_id']}` {row.get('segment_name') or ''} | `{row['coverage_status']}` | `{row['direction_status']}` |")
    lines.extend(["", "## Proposed Source Rows", "", "| Segment | Current cue | Insert after | Runner-facing claim |", "|---|---:|---:|---|"])
    for row in report["proposed_segment_promotion_rows"]:
        lines.append(
            f"| `{row['segment_id']}` {row.get('segment_name') or ''} | {row.get('current_repeat_cue_seq')} | `{(row.get('to') or {}).get('insert_after_segment_id')}` | {row['runner_facing_claim_text']} |"
        )
    scenario = report["calendar_reorder_scenario"]
    lines.extend(
        [
            "",
            "## Calendar Reprice",
            "",
            f"- Scenario: `{scenario.get('scenario_id')}` (`{scenario.get('status')}`)",
            f"- Source target date: `{scenario.get('source_target_date')}`",
            f"- Owner target date after removal: `{scenario.get('owner_target_date')}`",
            "",
            "## Remaining Steps",
            "",
        ]
    )
    lines.extend(f"- {step}" for step in report["remaining_steps_for_real_promotion"])
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--route-repeat-audit-json", type=Path, default=DEFAULT_ROUTE_REPEAT_AUDIT_JSON)
    parser.add_argument("--field-latent-audit-json", type=Path, default=DEFAULT_FIELD_LATENT_AUDIT_JSON)
    parser.add_argument("--calendar-reorder-json", type=Path, default=DEFAULT_CALENDAR_REORDER_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_experiment(
        field_tool_data=read_json(args.field_tool_data_json),
        route_repeat_audit=read_json(args.route_repeat_audit_json),
        field_latent_audit=read_json(args.field_latent_audit_json),
        calendar_reorder=read_json(args.calendar_reorder_json),
        official_geojson=read_json(args.official_geojson),
    )
    write_json(args.output_json, report)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    manifest = build_artifact_manifest(
        run_id="fd04a_fd19c_credit_promotion_experiment",
        inputs=[
            args.field_tool_data_json,
            args.route_repeat_audit_json,
            args.field_latent_audit_json,
            args.calendar_reorder_json,
            args.official_geojson,
        ],
        outputs=[args.output_json, args.output_md],
        command="python years/2026/scripts/fd04a_fd19c_credit_promotion_experiment.py",
        metadata={
            "schema": report["schema"],
            "status": report["status"],
            "saved_on_foot_miles_if_promoted": report["summary"]["saved_on_foot_miles_if_promoted"],
            "active_packet_mutated": report["summary"]["active_packet_mutated"],
        },
    )
    write_manifest(args.manifest_json, manifest)
    print(f"Wrote {display_path(args.output_json)}")
    print(f"Wrote {display_path(args.output_md)}")
    print(f"Wrote {display_path(args.manifest_json)}")
    print(json.dumps({"status": report["status"], **report["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
