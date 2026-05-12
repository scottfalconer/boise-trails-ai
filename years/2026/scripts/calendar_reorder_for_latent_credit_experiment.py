#!/usr/bin/env python3
"""Experiment with calendar reorders that turn latent credit into skipped work."""

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

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402
from latent_credit_delta_repricing_audit import (  # noqa: E402
    display_path,
    float_value,
    int_value,
    normalized_ids,
    route_key,
    route_label,
    rounded,
    sort_id,
    write_json,
)


DEFAULT_FIELD_TOOL_DATA_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_LATENT_DELTA_AUDIT_JSON = YEAR_DIR / "checkpoints" / "latent-credit-delta-repricing-audit-2026-05-12.json"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "calendar-reorder-latent-credit-experiment-2026-05-12.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "calendar-reorder-latent-credit-experiment-2026-05-12.md"
DEFAULT_MANIFEST_JSON = YEAR_DIR / "checkpoints" / "calendar-reorder-latent-credit-experiment-2026-05-12-manifest.json"

BOGUS_FIRST_WINDOW_CLOSURE_DATES = {date(2026, 6, 18), date(2026, 6, 19)}
BOGUS_TERMS = {
    "around the mountain",
    "bogus",
    "brewer",
    "deer point",
    "eastside",
    "freddy",
    "lodge",
    "mores",
    "mr. big",
    "pioneer lodge",
    "simplot lodge",
    "stack rock",
    "tempest",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def day_type_for(date_text: str | None) -> str | None:
    if not date_text:
        return None
    day = date.fromisoformat(date_text)
    return "weekend" if day.weekday() >= 5 else "weekday"


def day_number(date_text: str | None) -> int | None:
    if not date_text:
        return None
    return date.fromisoformat(date_text).day


def route_index(routes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for route in routes:
        keys = [route_key(route), str(route.get("outing_id") or ""), str(route.get("label") or ""), route_label(route)]
        for candidate_id in route.get("candidate_ids") or []:
            keys.append(str(candidate_id))
        for key in keys:
            if key:
                rows[key] = route
    return rows


def route_claimed_ids(route: dict[str, Any]) -> set[str]:
    return set(normalized_ids(route.get("segment_ids") or []))


def route_certification_status(route: dict[str, Any] | None) -> str:
    if not route:
        return "missing_route_card"
    blockers = []
    if not bool((route.get("validation") or {}).get("passed")):
        blockers.append("route_validation_failed")
    if not (route.get("parking") or {}).get("has_parking"):
        blockers.append("missing_verified_parked_start")
    if not route.get("gpx_href"):
        blockers.append("missing_nav_gpx")
    if not route.get("wayfinding_cues"):
        blockers.append("missing_wayfinding_cues")
    return "certified" if not blockers else "blocked:" + ",".join(blockers)


def segment_ids_in_wayfinding(route: dict[str, Any], key: str) -> set[str]:
    ids: set[str] = set()
    for cue in route.get("wayfinding_cues") or []:
        ids.update(normalized_ids(cue.get(key) or []))
    return ids


def latent_support_status(source_route: dict[str, Any], latent_ids: set[str]) -> dict[str, Any]:
    claimed = route_claimed_ids(source_route)
    repeat_cued = segment_ids_in_wayfinding(source_route, "official_repeat_segment_ids")
    official_cued = segment_ids_in_wayfinding(source_route, "official_segment_ids")
    reconciliation = source_route.get("segment_ownership_reconciliation") or {}
    declared_elsewhere = set(normalized_ids(reconciliation.get("declared_owned_elsewhere_segment_ids") or []))
    return {
        "source_claims_all_latent_ids": latent_ids <= claimed,
        "source_cues_all_latent_ids_as_repeat": latent_ids <= repeat_cued,
        "source_cues_all_latent_ids_as_official": latent_ids <= official_cued,
        "source_declares_all_latent_ids_owned_elsewhere": latent_ids <= declared_elsewhere,
        "pre_challenge_menu_change_requires_route_card_credit_promotion": not latent_ids <= claimed,
        "status": "claimed_on_source"
        if latent_ids <= claimed
        else "physically_cued_as_repeat_and_declared_owned_elsewhere"
        if latent_ids <= repeat_cued and latent_ids <= declared_elsewhere
        else "needs_source_credit_or_cue_review",
    }


def route_key_for_ref(ref: dict[str, Any], routes_by_key: dict[str, dict[str, Any]]) -> str | None:
    for key in [str(ref.get("outing_id") or ""), str(ref.get("label") or "")]:
        route = routes_by_key.get(key)
        if route:
            return route_key(route)
    return None


def field_day_positions(field_tool_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    routes_by_key = route_index(field_tool_data.get("routes") or [])
    rows: dict[str, dict[str, Any]] = {}
    for day_index, day in enumerate((field_tool_data.get("field_day_layer") or {}).get("field_days") or []):
        for loop_index, loop in enumerate(day.get("loops") or []):
            key = route_key_for_ref(loop.get("route_card_ref") or {}, routes_by_key)
            if not key:
                continue
            rows[key] = {
                "day_index": day_index,
                "loop_index": loop_index,
                "date": day.get("date"),
                "day_type": day.get("day_type"),
                "field_day_id": day.get("field_day_id"),
                "loop_label": loop.get("label"),
            }
    return rows


def all_claimed_segment_ids(routes: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for route in routes:
        ids.update(route_claimed_ids(route))
    return ids


def route_miles(route: dict[str, Any] | None) -> float:
    return rounded((route or {}).get("on_foot_miles"))


def route_p75(route: dict[str, Any] | None) -> int:
    return int_value((route or {}).get("door_to_door_minutes_p75"))


def route_p90(route: dict[str, Any] | None) -> int:
    return int_value((route or {}).get("door_to_door_minutes_p90"))


def clone_day(day: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(day)


def loop_schedule_minutes(loop: dict[str, Any], key: str) -> int:
    return int_value(loop.get(f"field_day_schedule_{key}_minutes") or loop.get(f"route_card_door_to_door_{key}_minutes"))


def reprice_day_after_removals(day: dict[str, Any], removed_labels: set[str]) -> dict[str, Any] | None:
    remaining_loops = [copy.deepcopy(loop) for loop in day.get("loops") or [] if str(loop.get("label")) not in removed_labels]
    removed_loops = [loop for loop in day.get("loops") or [] if str(loop.get("label")) in removed_labels]
    if not remaining_loops:
        return None
    removed_p75 = sum(loop_schedule_minutes(loop, "p75") for loop in removed_loops)
    removed_p90 = sum(loop_schedule_minutes(loop, "p90") for loop in removed_loops)
    p75 = max(0, int_value(day.get("field_day_schedule_p75_minutes") or day.get("p75_minutes")) - removed_p75)
    p90 = max(0, int_value(day.get("field_day_schedule_p90_minutes") or day.get("p90_minutes")) - removed_p90)
    p75 = max(p75, max(loop_schedule_minutes(loop, "p75") for loop in remaining_loops))
    p90 = max(p90, max(loop_schedule_minutes(loop, "p90") for loop in remaining_loops))
    segment_ids = sorted(
        {segment_id for loop in remaining_loops for segment_id in normalized_ids(loop.get("segment_ids") or [])},
        key=sort_id,
    )
    result = clone_day(day)
    result["loops"] = remaining_loops
    result["loop_count"] = len(remaining_loops)
    result["p75_minutes"] = p75
    result["p90_minutes"] = p90
    result["field_day_schedule_p75_minutes"] = p75
    result["field_day_schedule_p90_minutes"] = p90
    result["route_card_door_to_door_p75_sum"] = sum(loop_schedule_minutes(loop, "p75") for loop in remaining_loops)
    result["route_card_door_to_door_p90_sum"] = sum(loop_schedule_minutes(loop, "p90") for loop in remaining_loops)
    result["on_foot_miles"] = rounded(sum(float_value(loop.get("on_foot_miles")) for loop in remaining_loops))
    result["official_miles"] = rounded(sum(float_value(loop.get("official_miles")) for loop in remaining_loops))
    result["segment_ids"] = segment_ids
    result["segment_count"] = len(segment_ids)
    result["transfer_count"] = max(len(remaining_loops) - 1, 0)
    return result


def day_has_lower_hulls(day: dict[str, Any]) -> bool:
    text = " ".join(
        [
            " ".join(str(value) for value in day.get("constraints") or []),
            " ".join(str(trail) for loop in day.get("loops") or [] for trail in loop.get("trail_names") or []),
        ]
    ).lower()
    return "lower_hulls_even_day" in text or "lower hull" in text


def lower_hulls_check(day: dict[str, Any], target_date: str | None) -> dict[str, Any]:
    if not day_has_lower_hulls(day):
        return {"status": "not_applicable"}
    number = day_number(target_date)
    passed = bool(number and number % 2 == 0)
    return {
        "status": "passed" if passed else "blocked_lower_hulls_odd_day",
        "target_date": target_date,
        "requires_even_day": True,
    }


def day_has_bogus_closure_risk(day: dict[str, Any]) -> bool:
    haystack = " ".join(
        str(value)
        for value in [
            day.get("field_day_id"),
            *(loop.get("trailhead") for loop in day.get("loops") or []),
            *(trail for loop in day.get("loops") or [] for trail in loop.get("trail_names") or []),
        ]
    ).lower()
    return any(term in haystack for term in BOGUS_TERMS)


def bogus_closure_check(day: dict[str, Any], target_date: str | None) -> dict[str, Any]:
    if not target_date or not day_has_bogus_closure_risk(day):
        return {"status": "not_applicable"}
    target = date.fromisoformat(target_date)
    blocked = target in BOGUS_FIRST_WINDOW_CLOSURE_DATES and target.weekday() < 5
    return {
        "status": "blocked_bogus_first_window_closure" if blocked else "passed",
        "target_date": target_date,
        "closure_assumption": "Deer Point/Stack Rock/Bogus first-window closure applies Jun 18-19 weekdays unless current sources override.",
    }


def day_type_check(day: dict[str, Any], target_date: str | None) -> dict[str, Any]:
    expected = day_type_for(target_date)
    planned = day.get("day_type")
    return {
        "status": "passed" if not expected or planned == expected else "day_type_changed",
        "source_day_type": planned,
        "target_date": target_date,
        "target_date_day_type": expected,
    }


def p90_check(day: dict[str, Any]) -> dict[str, Any]:
    bound = int_value(day.get("p90_bound_minutes"))
    p90 = int_value(day.get("field_day_schedule_p90_minutes") or day.get("p90_minutes"))
    return {
        "status": "passed" if not bound or p90 <= bound else "p90_bound_violation",
        "p90_minutes": p90,
        "p90_bound_minutes": bound,
        "stress": round(p90 / bound, 3) if bound else None,
    }


def evaluate_day_on_date(day: dict[str, Any], target_date: str | None) -> dict[str, Any]:
    return {
        "day_type": day_type_check(day, target_date),
        "p90": p90_check(day),
        "lower_hulls": lower_hulls_check(day, target_date),
        "bogus_closure": bogus_closure_check(day, target_date),
    }


def check_status(checks: dict[str, Any]) -> str:
    for group in checks.values():
        if isinstance(group, dict):
            for row in group.values():
                if isinstance(row, dict) and str(row.get("status") or "").startswith(("blocked", "p90_bound_violation")):
                    return "blocked"
    return "passed"


def scenario_for_pair(
    pair: dict[str, Any],
    *,
    field_days: list[dict[str, Any]],
    positions: dict[str, dict[str, Any]],
    routes_by_key: dict[str, dict[str, Any]],
    route_count_before: int,
    baseline_segment_ids: set[str],
    official_segment_count: int,
) -> dict[str, Any]:
    source_key = str(pair.get("source_route_key") or "")
    owner_key = str(pair.get("owner_route_key") or "")
    source_route = routes_by_key.get(source_key)
    owner_route = routes_by_key.get(owner_key)
    source_position = positions.get(source_key) or {}
    owner_position = positions.get(owner_key) or {}
    source_day = field_days[int(source_position["day_index"])] if source_position else None
    owner_day = field_days[int(owner_position["day_index"])] if owner_position else None
    latent_ids = set(normalized_ids(pair.get("latent_segment_ids") or []))
    owner_ids = route_claimed_ids(owner_route or {})
    remaining_owner_ids = set(normalized_ids(pair.get("remaining_owner_segment_ids") or []))
    removed_owner_labels = {str((owner_route or {}).get("label") or (pair.get("owner_route") or {}).get("label") or "")}

    scenario_type = (
        "same_day_owner_deletion"
        if source_position.get("day_index") == owner_position.get("day_index")
        else "swap_source_day_before_owner_day"
    )
    owner_day_after = reprice_day_after_removals(owner_day or {}, removed_owner_labels) if owner_day else None
    source_day_after = clone_day(source_day or {}) if source_day else None
    source_target_date = owner_day.get("date") if source_day and owner_day and scenario_type == "swap_source_day_before_owner_day" else (source_day or {}).get("date")
    owner_target_date = source_day.get("date") if source_day and owner_day and scenario_type == "swap_source_day_before_owner_day" else (owner_day or {}).get("date")
    if scenario_type == "same_day_owner_deletion" and owner_day_after:
        source_day_after = clone_day(owner_day_after)

    days_after: list[dict[str, Any]] = []
    for index, day in enumerate(field_days):
        if source_position.get("day_index") == index and scenario_type == "swap_source_day_before_owner_day":
            replacement = clone_day(source_day_after or day)
            replacement["date"] = source_target_date
            days_after.append(replacement)
        elif owner_position.get("day_index") == index:
            if owner_day_after:
                replacement = clone_day(owner_day_after)
                replacement["date"] = owner_target_date
                days_after.append(replacement)
        else:
            days_after.append(clone_day(day))

    if scenario_type == "same_day_owner_deletion":
        days_after = []
        for index, day in enumerate(field_days):
            if owner_position.get("day_index") == index and owner_day_after:
                days_after.append(owner_day_after)
            else:
                days_after.append(clone_day(day))

    claimed_after_credit = (baseline_segment_ids - owner_ids) | latent_ids
    source_support = latent_support_status(source_route or {}, latent_ids)
    source_eval = evaluate_day_on_date(source_day_after or {}, source_target_date)
    owner_eval = evaluate_day_on_date(owner_day_after or {}, owner_target_date) if owner_day_after else {"removed_empty_day": {"status": "passed"}}
    all_days_p90 = [p90_check(day) for day in days_after]
    all_p90_passed = all(row["status"] == "passed" for row in all_days_p90)
    owner_fully_covered = owner_ids <= latent_ids and not remaining_owner_ids
    route_count_delta = -1 if owner_fully_covered else 0
    route_count_after = route_count_before + route_count_delta
    coverage_preserved = len(claimed_after_credit) == official_segment_count
    checks = {
        "source_route_card": {"status": route_certification_status(source_route)},
        "owner_route_card": {"status": route_certification_status(owner_route)},
        "owner_segments_covered_by_source": {
            "status": "passed" if owner_fully_covered else "owner_has_remaining_segments",
            "owner_segment_ids": normalized_ids(owner_ids),
            "latent_segment_ids": normalized_ids(latent_ids),
            "remaining_owner_segment_ids": normalized_ids(remaining_owner_ids),
        },
        "coverage": {
            "status": "passed" if coverage_preserved else "coverage_gap",
            "covered_segment_count_after_credit": len(claimed_after_credit),
            "official_segment_count": official_segment_count,
        },
        "source_day_after_reorder": source_eval,
        "owner_day_after_removal": owner_eval,
        "all_field_days_p90": {
            "status": "passed" if all_p90_passed else "p90_bound_violation",
            "violation_count": sum(1 for row in all_days_p90 if row["status"] != "passed"),
        },
        "field_day_gpx_continuity": {
            "status": "preserved_existing_day_gpx_subset",
            "note": "Experiment swaps whole field-day dates or removes loops from an already validated field-day sequence; no new GPX geometry is generated.",
        },
    }
    hard_passed = check_status(checks) == "passed"
    soft_status = (
        "order_reprice_supported_requires_segment_validation"
        if hard_passed and not source_support["pre_challenge_menu_change_requires_route_card_credit_promotion"]
        else "order_reprice_supported_requires_credit_promotion_or_post_run_validation"
        if hard_passed
        else "blocked"
    )
    return {
        "scenario_id": f"{source_key}-before-{owner_key}",
        "scenario_type": scenario_type,
        "status": soft_status,
        "source_route": pair.get("source_route"),
        "owner_route": pair.get("owner_route"),
        "latent_segment_ids": normalized_ids(latent_ids),
        "source_original_date": (source_day or {}).get("date"),
        "owner_original_date": (owner_day or {}).get("date"),
        "source_target_date": source_target_date,
        "owner_target_date": owner_target_date,
        "removed_owner_route_label": (owner_route or {}).get("label") or (pair.get("owner_route") or {}).get("label"),
        "saved_on_foot_miles": rounded(pair.get("proven_saved_on_foot_miles")),
        "saved_p75_minutes": int_value(pair.get("proven_saved_p75_minutes")),
        "saved_p90_minutes": int_value(pair.get("proven_saved_p90_minutes")),
        "field_day_count_before": len(field_days),
        "field_day_count_after": len(days_after),
        "route_count_before": route_count_before,
        "route_count_delta": route_count_delta,
        "route_count_after": route_count_after,
        "total_field_day_p75_before": sum(int_value(day.get("field_day_schedule_p75_minutes") or day.get("p75_minutes")) for day in field_days),
        "total_field_day_p75_after": sum(int_value(day.get("field_day_schedule_p75_minutes") or day.get("p75_minutes")) for day in days_after),
        "total_field_day_p90_before": sum(int_value(day.get("field_day_schedule_p90_minutes") or day.get("p90_minutes")) for day in field_days),
        "total_field_day_p90_after": sum(int_value(day.get("field_day_schedule_p90_minutes") or day.get("p90_minutes")) for day in days_after),
        "max_field_day_p90_after": max((int_value(day.get("field_day_schedule_p90_minutes") or day.get("p90_minutes")) for day in days_after), default=0),
        "owner_day_after_removal": {
            "date": owner_target_date,
            "loop_count": owner_day_after.get("loop_count") if owner_day_after else 0,
            "labels": [loop.get("label") for loop in (owner_day_after or {}).get("loops") or []],
            "p75_minutes": owner_day_after.get("field_day_schedule_p75_minutes") if owner_day_after else 0,
            "p90_minutes": owner_day_after.get("field_day_schedule_p90_minutes") if owner_day_after else 0,
        },
        "source_day_after_reorder": {
            "date": source_target_date,
            "loop_count": source_day_after.get("loop_count") if source_day_after else 0,
            "labels": [loop.get("label") for loop in (source_day_after or {}).get("loops") or []],
            "p75_minutes": source_day_after.get("field_day_schedule_p75_minutes") if source_day_after else 0,
            "p90_minutes": source_day_after.get("field_day_schedule_p90_minutes") if source_day_after else 0,
        },
        "source_latent_credit_support": source_support,
        "checks": checks,
    }


def group_scenarios(pair_scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for scenario in pair_scenarios:
        source = str((scenario.get("source_route") or {}).get("outing_id") or scenario.get("scenario_id"))
        by_source.setdefault(source, []).append(scenario)
    rows = []
    for source, scenarios in by_source.items():
        if len(scenarios) < 2:
            continue
        unique_owner_labels = []
        for scenario in scenarios:
            label = str(scenario.get("removed_owner_route_label") or "")
            if label and label not in unique_owner_labels:
                unique_owner_labels.append(label)
        rows.append(
            {
                "source_route_id": source,
                "source_route_label": (scenarios[0].get("source_route") or {}).get("label"),
                "owner_route_ids": [
                    str((scenario.get("owner_route") or {}).get("outing_id") or "")
                    for scenario in scenarios
                    if (scenario.get("owner_route") or {}).get("outing_id")
                ],
                "status": "all_pairwise_supported"
                if all(str(scenario.get("status") or "").startswith("order_reprice_supported") for scenario in scenarios)
                else "has_blocked_pair",
                "removable_owner_labels": unique_owner_labels,
                "saved_on_foot_miles": rounded(sum(float_value(scenario.get("saved_on_foot_miles")) for scenario in scenarios)),
                "saved_p75_minutes": sum(int_value(scenario.get("saved_p75_minutes")) for scenario in scenarios),
                "saved_p90_minutes": sum(int_value(scenario.get("saved_p90_minutes")) for scenario in scenarios),
                "requires_credit_promotion_or_post_run_validation": any(
                    (scenario.get("source_latent_credit_support") or {}).get(
                        "pre_challenge_menu_change_requires_route_card_credit_promotion"
                    )
                    for scenario in scenarios
                ),
            }
        )
    return sorted(rows, key=lambda row: (-float_value(row.get("saved_on_foot_miles")), str(row.get("source_route_label"))))


def independent_portfolio(pair_scenarios: list[dict[str, Any]], source_groups: list[dict[str, Any]]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for group in source_groups:
        if group.get("status") != "all_pairwise_supported":
            continue
        candidates.append(
            {
                "candidate_type": "source_group",
                "source_route_label": group.get("source_route_label"),
                "owner_route_ids": list(group.get("owner_route_ids") or []),
                "removes": list(group.get("removable_owner_labels") or []),
                "saved_on_foot_miles": rounded(group.get("saved_on_foot_miles")),
                "saved_p75_minutes": int_value(group.get("saved_p75_minutes")),
                "saved_p90_minutes": int_value(group.get("saved_p90_minutes")),
                "gate": "requires credit promotion or post-run validation"
                if group.get("requires_credit_promotion_or_post_run_validation")
                else "source already claims latent credit",
            }
        )
    for scenario in pair_scenarios:
        if not str(scenario.get("status") or "").startswith("order_reprice_supported"):
            continue
        candidates.append(
            {
                "candidate_type": "pairwise",
                "source_route_label": (scenario.get("source_route") or {}).get("label"),
                "owner_route_ids": [str((scenario.get("owner_route") or {}).get("outing_id") or "")],
                "removes": [str(scenario.get("removed_owner_route_label") or "")],
                "saved_on_foot_miles": rounded(scenario.get("saved_on_foot_miles")),
                "saved_p75_minutes": int_value(scenario.get("saved_p75_minutes")),
                "saved_p90_minutes": int_value(scenario.get("saved_p90_minutes")),
                "gate": "requires credit promotion or post-run validation"
                if (scenario.get("source_latent_credit_support") or {}).get(
                    "pre_challenge_menu_change_requires_route_card_credit_promotion"
                )
                else "source already claims latent credit",
            }
        )
    selected = []
    used_owner_ids: set[str] = set()
    for candidate in sorted(candidates, key=lambda row: (-float_value(row.get("saved_on_foot_miles")), str(row.get("source_route_label")))):
        owner_ids = {owner_id for owner_id in candidate.get("owner_route_ids") or [] if owner_id}
        if owner_ids & used_owner_ids:
            continue
        selected.append(candidate)
        used_owner_ids.update(owner_ids)
    return {
        "selection_policy": "Greedy by on-foot savings after expanding multi-owner source-group candidates; owner routes are unique.",
        "selected": selected,
        "saved_on_foot_miles": rounded(sum(float_value(row.get("saved_on_foot_miles")) for row in selected)),
        "saved_p75_minutes": sum(int_value(row.get("saved_p75_minutes")) for row in selected),
        "saved_p90_minutes": sum(int_value(row.get("saved_p90_minutes")) for row in selected),
        "removed_owner_route_count": len(used_owner_ids),
    }


def build_calendar_reorder_experiment(field_tool_data: dict[str, Any], latent_delta_audit: dict[str, Any]) -> dict[str, Any]:
    field_day_layer = field_tool_data.get("field_day_layer") or {}
    field_days = field_day_layer.get("field_days") or []
    routes = field_tool_data.get("routes") or []
    routes_by_key = route_index(routes)
    positions = field_day_positions(field_tool_data)
    baseline_segment_ids = all_claimed_segment_ids(routes)
    official_segment_count = int_value((field_tool_data.get("summary") or {}).get("segment_count_in_field_menu")) or len(
        baseline_segment_ids
    )
    pair_scenarios = [
        scenario_for_pair(
            pair,
            field_days=field_days,
            positions=positions,
            routes_by_key=routes_by_key,
            route_count_before=len(routes),
            baseline_segment_ids=baseline_segment_ids,
            official_segment_count=official_segment_count,
        )
        for pair in latent_delta_audit.get("pairwise_full_removals") or []
    ]
    supported = [
        scenario
        for scenario in pair_scenarios
        if str(scenario.get("status") or "").startswith("order_reprice_supported")
    ]
    source_groups = group_scenarios(supported)
    portfolio = independent_portfolio(supported, source_groups)
    return {
        "schema": "boise_trails_calendar_reorder_for_latent_credit_experiment_v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "supported_reorders_found" if supported else "no_supported_reorders",
        "source_files": {
            "field_tool_data_json": display_path(DEFAULT_FIELD_TOOL_DATA_JSON),
            "latent_credit_delta_audit_json": display_path(DEFAULT_LATENT_DELTA_AUDIT_JSON),
        },
        "scope": {
            "promotion_policy": "Experiment only. It does not rewrite the active field packet or mark progress complete.",
            "timing_policy": "Preserve calendar-assignment timing authority; subtract removed loop schedule cost and floor at the largest remaining loop schedule time.",
            "credit_policy": "Owner card removal before the challenge still requires either source route-card credit/cue promotion or post-run segment-first validation of the source GPX.",
        },
        "summary": {
            "pairwise_full_removal_candidate_count": len(pair_scenarios),
            "supported_pairwise_reorder_count": len(supported),
            "blocked_pairwise_reorder_count": len(pair_scenarios) - len(supported),
            "pairwise_savings_are_additive": False,
            "pairwise_saved_on_foot_miles_non_additive": rounded(sum(float_value(row.get("saved_on_foot_miles")) for row in supported)),
            "pairwise_saved_p75_minutes_non_additive": sum(int_value(row.get("saved_p75_minutes")) for row in supported),
            "pairwise_saved_p90_minutes_non_additive": sum(int_value(row.get("saved_p90_minutes")) for row in supported),
            "source_group_candidate_count": len(source_groups),
            "recommended_portfolio_saved_on_foot_miles": portfolio["saved_on_foot_miles"],
            "recommended_portfolio_saved_p75_minutes": portfolio["saved_p75_minutes"],
            "recommended_portfolio_saved_p90_minutes": portfolio["saved_p90_minutes"],
            "recommended_portfolio_removed_route_count": portfolio["removed_owner_route_count"],
            "current_route_count": len(routes),
            "current_field_day_count": len(field_days),
            "official_segment_count": official_segment_count,
        },
        "pairwise_scenarios": sorted(
            pair_scenarios,
            key=lambda row: (-float_value(row.get("saved_on_foot_miles")), str((row.get("source_route") or {}).get("label"))),
        ),
        "source_group_scenarios": source_groups,
        "recommended_non_overlapping_portfolio": portfolio,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# Calendar Reorder For Latent Credit Experiment",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        f"Status: `{report.get('status')}`",
        "",
        "## Summary",
        "",
        f"- Pairwise full-removal candidates: {summary.get('pairwise_full_removal_candidate_count')}",
        f"- Supported pairwise reorders: {summary.get('supported_pairwise_reorder_count')}",
        f"- Blocked pairwise reorders: {summary.get('blocked_pairwise_reorder_count')}",
        f"- Pairwise saved on-foot miles, non-additive: {summary.get('pairwise_saved_on_foot_miles_non_additive')}",
        f"- Pairwise saved p75 minutes, non-additive: {summary.get('pairwise_saved_p75_minutes_non_additive')}",
        f"- Pairwise saved p90 minutes, non-additive: {summary.get('pairwise_saved_p90_minutes_non_additive')}",
        f"- Source-group candidates: {summary.get('source_group_candidate_count')}",
        f"- Recommended non-overlapping portfolio saved miles: {summary.get('recommended_portfolio_saved_on_foot_miles')}",
        f"- Recommended non-overlapping portfolio saved p75: {summary.get('recommended_portfolio_saved_p75_minutes')}",
        "",
        "This is an experiment, not an active packet mutation. Pre-challenge deletion still needs source route-card credit/cue promotion; post-run deletion needs segment-first validation of the source GPX.",
        "",
        "## Pairwise Reorders",
        "",
        "| Source before owner | Move | Status | Saved mi | Saved p75 | Dates | Owner day after removal | Key gate |",
        "|---|---|---|---:|---:|---|---|---|",
    ]
    for row in report.get("pairwise_scenarios") or []:
        source_label = (row.get("source_route") or {}).get("label")
        owner_label = (row.get("owner_route") or {}).get("label")
        support = row.get("source_latent_credit_support") or {}
        key_gate = support.get("status")
        owner_day = row.get("owner_day_after_removal") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{source_label} -> {owner_label}",
                    str(row.get("scenario_type")),
                    f"`{row.get('status')}`",
                    str(row.get("saved_on_foot_miles")),
                    str(row.get("saved_p75_minutes")),
                    f"{row.get('source_original_date')} -> {row.get('source_target_date')}; owner day -> {row.get('owner_target_date')}",
                    f"{owner_day.get('loop_count')} loops, {owner_day.get('p75_minutes')}/{owner_day.get('p90_minutes')} p75/p90",
                    str(key_gate),
                ]
            )
            + " |"
        )
    if report.get("source_group_scenarios"):
        lines.extend(["", "## Source-Group Scenarios", "", "| Source | Removes | Status | Saved mi | Saved p75 | Gate |", "|---|---|---|---:|---:|---|"])
        for row in report.get("source_group_scenarios") or []:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("source_route_label")),
                        ", ".join(row.get("removable_owner_labels") or []),
                        f"`{row.get('status')}`",
                        str(row.get("saved_on_foot_miles")),
                        str(row.get("saved_p75_minutes")),
                        "requires credit promotion or post-run validation"
                        if row.get("requires_credit_promotion_or_post_run_validation")
                        else "source already claims latent credit",
                    ]
                )
                + " |"
            )
    portfolio = report.get("recommended_non_overlapping_portfolio") or {}
    if portfolio.get("selected"):
        lines.extend(["", "## Recommended Non-Overlapping Portfolio", "", f"Policy: {portfolio.get('selection_policy')}", "", "| Source | Removes | Type | Saved mi | Saved p75 | Gate |", "|---|---|---|---:|---:|---|"])
        for row in portfolio.get("selected") or []:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("source_route_label")),
                        ", ".join(row.get("removes") or []),
                        str(row.get("candidate_type")),
                        str(row.get("saved_on_foot_miles")),
                        str(row.get("saved_p75_minutes")),
                        str(row.get("gate")),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Gate Notes",
            "",
            "- `FD04A -> FD19C` is the largest direct reorder candidate.",
            "- `FD30A` can remove both Avimor microcards `FD27A` and `FD27C` as a source-group candidate, but it still needs the source route to run earlier or post-run validation before deleting them.",
            "- `FD14B -> FD14A` is low mileage but has cue evidence as declared repeat credit; pre-challenge deletion still needs claim/cue promotion.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--latent-delta-audit-json", type=Path, default=DEFAULT_LATENT_DELTA_AUDIT_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    field_tool_data = read_json(args.field_tool_data_json)
    latent_delta_audit = read_json(args.latent_delta_audit_json)
    report = build_calendar_reorder_experiment(field_tool_data, latent_delta_audit)
    report["source_files"] = {
        "field_tool_data_json": display_path(args.field_tool_data_json),
        "latent_credit_delta_audit_json": display_path(args.latent_delta_audit_json),
    }
    write_json(args.output_json, report)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    manifest = build_artifact_manifest(
        run_id="calendar_reorder_for_latent_credit_experiment",
        command="python years/2026/scripts/calendar_reorder_for_latent_credit_experiment.py",
        inputs=[args.field_tool_data_json, args.latent_delta_audit_json],
        outputs=[args.output_json, args.output_md],
        metadata={"status": report.get("status"), "summary": report.get("summary")},
    )
    write_manifest(args.manifest_json, manifest)
    print(f"Wrote {display_path(args.output_json)}")
    print(f"Wrote {display_path(args.output_md)}")
    print(f"Wrote {display_path(args.manifest_json)}")
    print(json.dumps({"status": report["status"], "summary": report["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
