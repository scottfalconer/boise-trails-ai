#!/usr/bin/env python3
"""Create an actionable day-by-day runbook from schedule and execution artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable


YEAR_DIR = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, utc_now_id, write_manifest  # noqa: E402
from day_of_preflight import evaluate_route_preflight  # noqa: E402
from simulate_outing_execution import (  # noqa: E402
    DEFAULT_STATE_PATH,
    build_route_provider,
    get_drive_model,
    load_state,
)

DEFAULT_EXECUTION_JSON = (
    YEAR_DIR
    / "experiments"
    / "2026-05-04-outing-execution-simulation"
    / "outing_execution.json"
)
DEFAULT_SCHEDULE_JSON = (
    YEAR_DIR
    / "outputs"
    / "calendar"
    / "full-ready-sensitivity-120s"
    / "2026-06-18-to-2026-07-18-draft.json"
)
DEFAULT_OUTPUT_DIR = YEAR_DIR / "outputs" / "runbooks"

EXECUTION_CHAIN = [
    "drive_to_trailhead",
    "park",
    "access_to_trail",
    "run_official_route",
    "return_to_car",
    "drive_home",
]
RouteProvider = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def index_execution_outings(execution: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(outing.get("candidate_id")): outing
        for outing in execution.get("outings") or []
        if outing.get("candidate_id")
    }


def legs_by_type(outing: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(leg.get("leg_type")): leg for leg in outing.get("legs") or []}


def validation_passed(outing: dict[str, Any]) -> bool:
    if outing.get("execution_status") != "simulated_ready":
        return False
    if outing.get("blocking_reasons"):
        return False
    validation = outing.get("validation") or {}
    required_checks = [
        "drive_to_trailhead_validated",
        "parking_validated",
        "access_to_trail_validated",
        "official_segment_traversal_validated",
        "return_to_car_validated",
        "drive_home_validated",
    ]
    return all(bool(validation.get(check)) for check in required_checks)


def route_readiness_label(
    scheduled_item: dict[str, Any],
    execution_outing: dict[str, Any],
    legs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    official = float(scheduled_item.get("new_official_miles") or 0)
    total = float(scheduled_item.get("estimated_total_on_foot_miles") or 0)
    ascent = int(scheduled_item.get("ascent_ft") or 0)
    ratio = round(official / total, 3) if total > 0 else 0
    return_to_car = legs.get("return_to_car") or {}
    park = legs.get("park") or {}
    endpoint_gap = float(return_to_car.get("endpoint_gap_miles") or 0)
    flags = []
    if not validation_passed(execution_outing):
        flags.append("execution_validation_not_passed")
    if endpoint_gap >= 1.5:
        flags.append("large_endpoint_distance_before_return_path")
    if total > 0 and ratio < 0.35:
        flags.append("low_official_to_total_ratio")
    if official < 1 and total >= 2:
        flags.append("low_credit_mop_up")
    if str(park.get("parking_confidence") or "").startswith("inferred"):
        flags.append("parking_source_inferred")
    if execution_outing.get("day_of_checks_remaining"):
        flags.append("day_of_checks_required")

    if not validation_passed(execution_outing):
        label = "QA-hold"
    elif official < 1 and total >= 2:
        label = "mop_up"
    elif total >= 12 or ratio < 0.35 or endpoint_gap >= 1.5:
        label = "necessary_grinder"
    elif total >= 8 or ascent >= 2000 or ratio < 0.55:
        label = "B-route"
    else:
        label = "A-route"

    return {
        "label": label,
        "official_to_total_ratio": ratio,
        "flags": flags,
    }


def runbook_outing(
    scheduled_item: dict[str, Any],
    execution_outing: dict[str, Any],
    run_date: str,
    start_time: str,
) -> dict[str, Any]:
    legs = legs_by_type(execution_outing)
    trail_names = scheduled_item.get("trail_names") or execution_outing.get("trail_names") or []
    static_preflight = evaluate_route_preflight(
        {
            "candidate_id": scheduled_item.get("outing_id"),
            "recommendation_type": "calendar_runbook_outing",
            "trail_names": trail_names,
        },
        run_date=run_date,
        start_time=start_time,
    )
    readiness = route_readiness_label(scheduled_item, execution_outing, legs)
    return {
        "outing_id": str(scheduled_item.get("outing_id")),
        "trail_names": trail_names,
        "route_label": readiness["label"],
        "official_to_total_ratio": readiness["official_to_total_ratio"],
        "readiness_flags": readiness["flags"],
        "trailhead": scheduled_item.get("trailhead"),
        "simulated_total_minutes": scheduled_item.get("simulated_total_minutes"),
        "new_official_miles": scheduled_item.get("new_official_miles"),
        "raw_official_miles": scheduled_item.get("raw_official_miles"),
        "estimated_total_on_foot_miles": scheduled_item.get("estimated_total_on_foot_miles"),
        "ascent_ft": scheduled_item.get("ascent_ft"),
        "descent_ft": scheduled_item.get("descent_ft"),
        "grade_adjusted_miles": scheduled_item.get("grade_adjusted_miles"),
        "new_segment_ids": scheduled_item.get("new_segment_ids") or [],
        "repeat_segment_ids": scheduled_item.get("repeat_segment_ids") or [],
        "execution_status": execution_outing.get("execution_status"),
        "validation_passed": validation_passed(execution_outing),
        "blocking_reasons": execution_outing.get("blocking_reasons") or [],
        "day_of_checks_remaining": execution_outing.get("day_of_checks_remaining") or [],
        "static_preflight": static_preflight,
        "execution_chain": EXECUTION_CHAIN,
        "drive_to_trailhead": legs.get("drive_to_trailhead") or {},
        "park": legs.get("park") or {},
        "access_to_trail": legs.get("access_to_trail") or {},
        "run_official_route": legs.get("run_official_route") or {},
        "return_to_car": legs.get("return_to_car") or {},
        "drive_home": legs.get("drive_home") or {},
    }


def trailhead_point_from_outing(outing: dict[str, Any]) -> dict[str, Any]:
    park = outing.get("park") or {}
    return {
        "name": str(park.get("trailhead") or outing.get("trailhead") or "Unknown trailhead"),
        "lat": float(park.get("lat")) if park.get("lat") is not None else None,
        "lon": float(park.get("lon")) if park.get("lon") is not None else None,
    }


def drive_minutes_from_leg(outing: dict[str, Any], leg_name: str) -> int:
    return int((outing.get(leg_name) or {}).get("duration_minutes") or 0)


def parse_day(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def analyze_human_load(
    days: list[dict[str, Any]],
    buffer_days: list[dict[str, Any]],
    constraints: dict[str, Any],
) -> dict[str, Any]:
    if not days and not buffer_days:
        return {}

    all_dates = [parse_day(str(day["date"])) for day in days + buffer_days if day.get("date")]
    start = min(all_dates)
    end = max(all_dates)
    scheduled_by_date = {parse_day(str(day["date"])): day for day in days}
    buffer_dates = {parse_day(str(day["date"])) for day in buffer_days if day.get("date")}

    weekday_normal = constraints.get("weekday_normal_max_minutes")
    weekday_normal = int(weekday_normal) if weekday_normal is not None else None
    weekday_exceptions = []
    for day in days:
        day_date = parse_day(str(day["date"]))
        normal_minutes = day.get("normal_available_minutes")
        if normal_minutes is None and weekday_normal is not None and day_date.weekday() < 5:
            normal_minutes = weekday_normal
        exception_minutes = (
            max(0, int(day.get("realistic_total_minutes") or 0) - int(normal_minutes))
            if normal_minutes is not None
            else 0
        )
        day["normal_available_minutes"] = normal_minutes
        day["requires_normal_cap_exception"] = exception_minutes > 0
        day["normal_cap_exception_minutes"] = exception_minutes
        if exception_minutes > 0:
            weekday_exceptions.append(
                {
                    "date": day["date"],
                    "realistic_total_minutes": day.get("realistic_total_minutes"),
                    "normal_available_minutes": normal_minutes,
                    "minutes_over_normal": exception_minutes,
                    "estimated_total_on_foot_miles": day.get("estimated_total_on_foot_miles"),
                    "ascent_ft": day.get("ascent_ft"),
                }
            )

    target_consecutive = constraints.get("max_consecutive_scheduled_days_target")
    if target_consecutive is None:
        target_consecutive = constraints.get("max_consecutive_scheduled_days")
    target_consecutive = int(target_consecutive) if target_consecutive else None
    consecutive_blocks = []
    current_block: list[date] = []
    current = start
    while current <= end:
        if current in scheduled_by_date:
            current_block.append(current)
        elif current_block:
            consecutive_blocks.append(current_block)
            current_block = []
        current += timedelta(days=1)
    if current_block:
        consecutive_blocks.append(current_block)
    consecutive_summary = [
        {
            "start_date": block[0].isoformat(),
            "end_date": block[-1].isoformat(),
            "scheduled_days": len(block),
            "violates_target": bool(target_consecutive and len(block) > target_consecutive),
        }
        for block in consecutive_blocks
    ]

    latest = constraints.get("latest_scheduled_date")
    latest_date = parse_day(str(latest)) if latest else max(scheduled_by_date or {end})
    required_rest = constraints.get("required_rest_days_before_latest")
    required_rest = int(required_rest) if required_rest is not None else None
    rest_before_latest = 0
    current = start
    while current <= latest_date:
        if current not in scheduled_by_date:
            rest_before_latest += 1
        current += timedelta(days=1)
    rest_deficit = (
        max(0, required_rest - rest_before_latest)
        if required_rest is not None
        else 0
    )

    max_weekly = constraints.get("max_weekly_on_foot_miles")
    max_weekly = float(max_weekly) if max_weekly is not None else None
    long_day_miles = float(constraints.get("long_on_foot_day_miles") or 15)
    max_long_days = constraints.get("max_long_on_foot_days_per_week")
    max_long_days = int(max_long_days) if max_long_days is not None else None
    week_count = ((end - start).days // 7) + 1
    weekly_loads = []
    for week_index in range(week_count):
        week_start = start + timedelta(days=week_index * 7)
        week_end = min(end, week_start + timedelta(days=6))
        week_days = [
            day
            for day_date, day in scheduled_by_date.items()
            if week_start <= day_date <= week_end
        ]
        on_foot = round(sum(float(day.get("estimated_total_on_foot_miles") or 0) for day in week_days), 2)
        official = round(sum(float(day.get("official_new_miles") or 0) for day in week_days), 2)
        ascent = sum(int(day.get("ascent_ft") or 0) for day in week_days)
        long_days = [
            day["date"]
            for day in week_days
            if float(day.get("estimated_total_on_foot_miles") or 0) >= long_day_miles
        ]
        weekly_loads.append(
            {
                "week_index": week_index + 1,
                "start_date": week_start.isoformat(),
                "end_date": week_end.isoformat(),
                "scheduled_days": len(week_days),
                "on_foot_miles": on_foot,
                "official_miles": official,
                "ascent_ft": ascent,
                "long_on_foot_day_count": len(long_days),
                "long_on_foot_dates": sorted(long_days),
                "weekly_mileage_violation": bool(max_weekly is not None and on_foot > max_weekly),
                "long_day_count_violation": bool(max_long_days is not None and len(long_days) > max_long_days),
            }
        )

    return {
        "weekday_normal_max_minutes": weekday_normal,
        "weekday_exception_days": weekday_exceptions,
        "weekday_exception_day_count": len(weekday_exceptions),
        "weekday_exception_minutes": sum(int(day["minutes_over_normal"]) for day in weekday_exceptions),
        "max_consecutive_scheduled_days_target": target_consecutive,
        "max_consecutive_scheduled_days": max((len(block) for block in consecutive_blocks), default=0),
        "consecutive_scheduled_blocks": consecutive_summary,
        "consecutive_day_violation_count": len(
            [block for block in consecutive_summary if block["violates_target"]]
        ),
        "required_rest_days_before_latest": required_rest,
        "rest_days_before_latest": rest_before_latest,
        "rest_days_before_latest_deficit": rest_deficit,
        "buffer_days": sorted(day.isoformat() for day in buffer_dates),
        "weekly_loads": weekly_loads,
        "weekly_mileage_violation_count": len(
            [week for week in weekly_loads if week["weekly_mileage_violation"]]
        ),
        "long_day_count_violation_count": len(
            [week for week in weekly_loads if week["long_day_count_violation"]]
        ),
    }


def build_day_transport(
    day_outings: list[dict[str, Any]],
    route_provider: RouteProvider | None,
) -> dict[str, Any]:
    if not day_outings:
        return {
            "legs": [],
            "actual_day_drive_minutes": 0,
            "conservative_scheduled_drive_minutes": 0,
            "drive_minutes_saved_vs_conservative": 0,
            "route_validated": False,
        }

    first = day_outings[0]
    last = day_outings[-1]
    legs = [
        {
            **(first.get("drive_to_trailhead") or {}),
            "leg_type": "drive_to_first_trailhead",
        }
    ]
    actual_drive_minutes = drive_minutes_from_leg(first, "drive_to_trailhead")

    for left, right in zip(day_outings, day_outings[1:]):
        left_point = trailhead_point_from_outing(left)
        right_point = trailhead_point_from_outing(right)
        if left_point["name"] == right_point["name"]:
            interdrive = {
                "provider": "same_trailhead",
                "route_validated": True,
                "distance_miles": 0.0,
                "duration_minutes": 0,
                "from": left_point["name"],
                "to": right_point["name"],
            }
        elif route_provider and left_point["lat"] is not None and right_point["lat"] is not None:
            interdrive = route_provider(left_point, right_point)
            interdrive.setdefault("from", left_point["name"])
            interdrive.setdefault("to", right_point["name"])
        else:
            interdrive = {
                "provider": "not_computed",
                "route_validated": False,
                "distance_miles": None,
                "duration_minutes": 0,
                "from": left_point["name"],
                "to": right_point["name"],
            }
        legs.append({"leg_type": "drive_between_trailheads", **interdrive})
        actual_drive_minutes += int(interdrive.get("duration_minutes") or 0)

    legs.append(
        {
            **(last.get("drive_home") or {}),
            "leg_type": "drive_home_from_last_trailhead",
        }
    )
    actual_drive_minutes += drive_minutes_from_leg(last, "drive_home")
    conservative_drive_minutes = sum(
        drive_minutes_from_leg(outing, "drive_to_trailhead")
        + drive_minutes_from_leg(outing, "drive_home")
        for outing in day_outings
    )
    return {
        "legs": legs,
        "actual_day_drive_minutes": actual_drive_minutes,
        "conservative_scheduled_drive_minutes": conservative_drive_minutes,
        "drive_minutes_saved_vs_conservative": conservative_drive_minutes - actual_drive_minutes,
        "route_validated": all(bool(leg.get("route_validated")) for leg in legs),
    }


def build_runbook(
    schedule: dict[str, Any],
    execution: dict[str, Any],
    profile_name: str,
    route_provider: RouteProvider | None = None,
) -> dict[str, Any]:
    execution_index = index_execution_outings(execution)
    days = []
    missing_ids: list[str] = []
    failed_validation_ids: list[str] = []
    scheduled_segment_ids: set[int] = set()
    start_time = str((schedule.get("constraints") or {}).get("start_time") or "07:00")

    for day in schedule.get("days") or []:
        if day.get("status") != "scheduled":
            continue
        day_outings = []
        for scheduled_item in day.get("outings") or []:
            outing_id = str(scheduled_item.get("outing_id"))
            execution_outing = execution_index.get(outing_id)
            if not execution_outing:
                missing_ids.append(outing_id)
                continue
            entry = runbook_outing(
                scheduled_item,
                execution_outing,
                run_date=str(day.get("date")),
                start_time=start_time,
            )
            if not entry["validation_passed"]:
                failed_validation_ids.append(outing_id)
            scheduled_segment_ids.update(int(seg_id) for seg_id in scheduled_item.get("new_segment_ids") or [])
            day_outings.append(entry)
        day_transport = build_day_transport(day_outings, route_provider)
        conservative_day_minutes = int(day.get("simulated_total_minutes") or 0)
        realistic_total_minutes = (
            conservative_day_minutes
            - int(day_transport["conservative_scheduled_drive_minutes"])
            + int(day_transport["actual_day_drive_minutes"])
        )
        official_new_miles = day.get("official_new_miles")
        if official_new_miles is None:
            official_new_miles = round(
                sum(float(outing.get("new_official_miles") or 0) for outing in day_outings),
                2,
            )
        estimated_total_on_foot_miles = day.get("estimated_total_on_foot_miles")
        if estimated_total_on_foot_miles is None:
            estimated_total_on_foot_miles = round(
                sum(float(outing.get("estimated_total_on_foot_miles") or 0) for outing in day_outings),
                2,
            )
        ascent_ft = day.get("ascent_ft")
        if ascent_ft is None:
            ascent_ft = sum(int(outing.get("ascent_ft") or 0) for outing in day_outings)
        descent_ft = day.get("descent_ft")
        if descent_ft is None:
            descent_ft = sum(int(outing.get("descent_ft") or 0) for outing in day_outings)
        grade_adjusted_miles = day.get("grade_adjusted_miles")
        if grade_adjusted_miles is None:
            grade_adjusted_miles = round(
                sum(float(outing.get("grade_adjusted_miles") or 0) for outing in day_outings),
                2,
            )
        days.append(
            {
                "date": day.get("date"),
                "available_minutes": day.get("available_minutes"),
                "normal_available_minutes": day.get("normal_available_minutes"),
                "simulated_total_minutes": day.get("simulated_total_minutes"),
                "realistic_total_minutes": realistic_total_minutes,
                "official_new_miles": official_new_miles,
                "estimated_total_on_foot_miles": estimated_total_on_foot_miles,
                "ascent_ft": ascent_ft,
                "descent_ft": descent_ft,
                "grade_adjusted_miles": grade_adjusted_miles,
                "trailheads": day.get("trailheads") or [],
                "day_labels": sorted(
                    {
                        str(outing.get("route_label"))
                        for outing in day_outings
                        if outing.get("route_label")
                    }
                ),
                "day_transport": day_transport,
                "outings": day_outings,
            }
        )

    buffer_days = [
        {
            "date": day.get("date"),
            "status": day.get("status"),
            "reason": day.get("reason"),
            "available_minutes": day.get("available_minutes"),
        }
        for day in schedule.get("days") or []
        if day.get("status") != "scheduled"
    ]
    load_analysis = analyze_human_load(days, buffer_days, schedule.get("constraints") or {})

    return {
        "run_id": utc_now_id(),
        "profile_name": profile_name,
        "schedule_type": schedule.get("schedule_type"),
        "source_schedule_run_id": schedule.get("run_id") or schedule.get("generated_at"),
        "source_execution_run_id": execution.get("run_id") or execution.get("generated_at"),
        "constraints": schedule.get("constraints") or {},
        "summary": schedule.get("summary") or {},
        "days": days,
        "buffer_days": buffer_days,
        "load_analysis": load_analysis,
        "audit": {
            "scheduled_days": len(days),
            "buffer_days": len(buffer_days),
            "weekday_exception_days": load_analysis.get("weekday_exception_day_count", 0),
            "consecutive_day_violations": load_analysis.get("consecutive_day_violation_count", 0),
            "weekly_mileage_violations": load_analysis.get("weekly_mileage_violation_count", 0),
            "long_day_count_violations": load_analysis.get("long_day_count_violation_count", 0),
            "rest_days_before_latest_deficit": load_analysis.get("rest_days_before_latest_deficit", 0),
            "scheduled_executable_units": sum(len(day["outings"]) for day in days),
            "scheduled_new_segments_in_runbook": len(scheduled_segment_ids),
            "missing_execution_outing_ids": sorted(set(missing_ids)),
            "failed_execution_validation_outing_ids": sorted(set(failed_validation_ids)),
            "execution_validation_passed": not missing_ids and not failed_validation_ids,
            "day_transport_validation_passed": all(
                bool(day["day_transport"].get("route_validated")) for day in days
            ),
        },
        "caveats": [
            "This runbook uses simulated OSRM drive routes and graph-validated trail/connectors; check live road closures before driving.",
            "Each listed outing is a single-car loop that returns to the parked car.",
            "Field-ready status still requires current Ridge to Rivers conditions, closures, signage, heat, and smoke checks for the actual day.",
            "Multi-outing days include an actual day-level transport chain: drive from home to the first trailhead, drive between trailheads if needed, and drive home from the last trailhead.",
        ],
    }


def fmt_minutes(value: Any) -> str:
    return "" if value is None else f"{value} min"


def fmt_miles(value: Any) -> str:
    return "" if value is None else f"{value} mi"


def transport_leg_label(leg: dict[str, Any]) -> str:
    leg_type = str(leg.get("leg_type") or "")
    if leg_type == "drive_to_first_trailhead":
        action = "Drive from home to first trailhead"
    elif leg_type == "drive_between_trailheads":
        action = "Drive between trailheads"
    elif leg_type == "drive_home_from_last_trailhead":
        action = "Drive home from last trailhead"
    else:
        action = "Drive"
    return (
        f"- {action}: {leg.get('from')} -> {leg.get('to')}, "
        f"{fmt_minutes(leg.get('duration_minutes'))}, "
        f"{fmt_miles(leg.get('distance_miles'))}, provider={leg.get('provider')}, "
        f"validated={leg.get('route_validated')}"
    )


def render_markdown(runbook: dict[str, Any]) -> str:
    lines = [
        "# 2026 Executable Route Runbook",
        "",
        f"- Profile: {runbook['profile_name']}",
        f"- Schedule type: {runbook.get('schedule_type')}",
        f"- Scheduled days: {runbook['audit']['scheduled_days']}",
        f"- Scheduled executable units: {runbook['audit']['scheduled_executable_units']}",
        f"- Scheduled official miles: {runbook['summary'].get('scheduled_official_miles')}",
        f"- Scheduled total on-foot miles: {runbook['summary'].get('scheduled_total_on_foot_miles')}",
        f"- Scheduled ascent: {runbook['summary'].get('scheduled_ascent_ft')} ft",
        f"- Scheduled segments: {runbook['summary'].get('scheduled_segments')}",
        f"- Buffer/open days: {runbook['audit'].get('buffer_days')}",
        f"- Weekday normal-cap exception days: {runbook['audit'].get('weekday_exception_days')}",
        f"- Recovery/load violations: "
        f"consecutive={runbook['audit'].get('consecutive_day_violations')}, "
        f"weekly={runbook['audit'].get('weekly_mileage_violations')}, "
        f"long-day={runbook['audit'].get('long_day_count_violations')}, "
        f"rest-deficit={runbook['audit'].get('rest_days_before_latest_deficit')}",
        f"- Execution validation: {'passed' if runbook['audit']['execution_validation_passed'] else 'needs attention'}",
        "",
    ]
    if runbook["audit"]["missing_execution_outing_ids"]:
        lines.append(
            "- Missing execution evidence: "
            + ", ".join(runbook["audit"]["missing_execution_outing_ids"])
        )
        lines.append("")

    for day in runbook["days"]:
        lines.extend(
            [
                f"## {day['date']}",
                "",
                f"- Day total: {fmt_minutes(day.get('simulated_total_minutes'))}, "
                f"realistic transport-adjusted {fmt_minutes(day.get('realistic_total_minutes'))}, "
                f"{fmt_miles(day.get('official_new_miles'))} official, "
                f"{fmt_miles(day.get('estimated_total_on_foot_miles'))} on foot, "
                f"{day.get('ascent_ft') or 0} ft ascent",
            ]
        )
        transport = day.get("day_transport") or {}
        if transport:
            lines.append(
                f"- Day transport: {fmt_minutes(transport.get('actual_day_drive_minutes'))} driving "
                f"({fmt_minutes(transport.get('drive_minutes_saved_vs_conservative'))} saved vs conservative per-outing drives), "
                f"validated={transport.get('route_validated')}"
            )
            lines.append("- Day drive sequence follows these legs, not each outing's solo drive-home reference:")
            for leg in transport.get("legs") or []:
                lines.append(transport_leg_label(leg))
        if day.get("day_labels"):
            lines.append(f"- Day labels: {', '.join(day['day_labels'])}")
        if day.get("requires_normal_cap_exception"):
            lines.append(
                f"- Normal-cap exception: {day.get('normal_cap_exception_minutes')} min over "
                f"{day.get('normal_available_minutes')} min normal weekday cap"
            )
        for outing in day["outings"]:
            trail_label = " + ".join(outing["trail_names"])
            drive_to = outing["drive_to_trailhead"]
            park = outing["park"]
            access = outing["access_to_trail"]
            run = outing["run_official_route"]
            return_to_car = outing["return_to_car"]
            drive_home = outing["drive_home"]
            preflight = outing["static_preflight"]
            lines.extend(
                [
                    "",
                    f"### Run {trail_label}",
                    "",
                    f"- Route label: {outing.get('route_label')}; "
                    f"official/total ratio={outing.get('official_to_total_ratio')}; "
                    f"flags={', '.join(outing.get('readiness_flags') or []) or 'none'}",
                    f"- Trailhead for this outing: {outing.get('trailhead')}",
                    f"- Solo-drive reference if running this outing alone: "
                    f"to trailhead {fmt_minutes(drive_to.get('duration_minutes'))} / "
                    f"{fmt_miles(drive_to.get('distance_miles'))}; "
                    f"home {fmt_minutes(drive_home.get('duration_minutes'))} / "
                    f"{fmt_miles(drive_home.get('distance_miles'))}",
                    f"- Park at {park.get('trailhead')}: can_park={park.get('can_park')}, "
                    f"confidence={park.get('parking_confidence')}, status={park.get('facility_status')}",
                    f"- Access trail: {fmt_miles(access.get('one_way_miles'))} one-way, "
                    f"{fmt_miles(access.get('round_trip_miles'))} round-trip, "
                    f"source={access.get('source')}, validated={access.get('validated')}",
                    f"- Run {trail_label}: {fmt_miles(run.get('official_new_miles'))} official, "
                    f"{fmt_miles(run.get('estimated_total_on_foot_miles'))} total on-foot, "
                    f"{fmt_minutes(run.get('moving_minutes'))} moving, "
                    f"{run.get('ascent_ft') or 0} ft ascent, validated={run.get('validated')}",
                    f"- Return to car: strategy={return_to_car.get('strategy')}, "
                    f"validated={return_to_car.get('validated')}, "
                    f"endpoint_distance_before_return={fmt_miles(return_to_car.get('endpoint_gap_miles'))}",
                    f"- Static preflight: {preflight.get('field_status')}; "
                    f"manual={', '.join(preflight.get('manual_checks_required') or []) or 'none'}; "
                    f"warnings={', '.join(preflight.get('warnings') or []) or 'none'}",
                    f"- Day-of checks: {', '.join(outing['day_of_checks_remaining']) or 'none recorded'}",
                ]
            )
        lines.append("")

    load = runbook.get("load_analysis") or {}
    if load:
        lines.extend(["## Human Load Flags", ""])
        lines.append(
            f"- Weekday exception days: {load.get('weekday_exception_day_count', 0)} "
            f"({load.get('weekday_exception_minutes', 0)} min over normal cap total)"
        )
        lines.append(
            f"- Max consecutive scheduled days: {load.get('max_consecutive_scheduled_days')} "
            f"against target {load.get('max_consecutive_scheduled_days_target')}"
        )
        lines.append(
            f"- Rest days before latest scheduled date: {load.get('rest_days_before_latest')} "
            f"against required {load.get('required_rest_days_before_latest')}"
        )
        for week in load.get("weekly_loads") or []:
            if week.get("weekly_mileage_violation") or week.get("long_day_count_violation"):
                lines.append(
                    f"- Week {week['week_index']} ({week['start_date']} to {week['end_date']}): "
                    f"{week['on_foot_miles']} mi on foot, "
                    f"{week['long_on_foot_day_count']} long days"
                )
        lines.append("")

    if runbook.get("buffer_days"):
        lines.extend(["## Buffer Days", ""])
        for day in runbook["buffer_days"]:
            lines.append(
                f"- {day.get('date')}: {day.get('status')}, "
                f"{day.get('available_minutes')} min available, "
                f"reason={day.get('reason')}"
            )
        lines.append("")

    lines.extend(["## Caveats", ""])
    for caveat in runbook["caveats"]:
        lines.append(f"- {caveat}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule-json", type=Path, default=DEFAULT_SCHEDULE_JSON)
    parser.add_argument("--execution-json", type=Path, default=DEFAULT_EXECUTION_JSON)
    parser.add_argument("--profile-name", default="full-clear-sensitivity")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--drive-routing", choices=["osrm", "model"], default="osrm")
    parser.add_argument("--osrm-base-url", default="https://router.project-osrm.org")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    schedule = read_json(args.schedule_json)
    execution = read_json(args.execution_json)
    state = load_state(args.state)
    drive_model = get_drive_model(state)
    route_provider = build_route_provider(args.drive_routing, drive_model, args.osrm_base_url)
    runbook = build_runbook(
        schedule,
        execution,
        profile_name=args.profile_name,
        route_provider=route_provider,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.profile_name.replace(" ", "-")
    json_path = args.output_dir / f"{stem}-runbook.json"
    md_path = args.output_dir / f"{stem}-runbook.md"
    write_json(json_path, runbook)
    md_path.write_text(render_markdown(runbook))
    manifest_path = args.output_dir / f"{stem}-runbook-artifact-manifest.json"
    write_manifest(
        manifest_path,
        build_artifact_manifest(
            run_id=runbook["run_id"],
            inputs=[args.schedule_json, args.execution_json, args.state],
            outputs=[json_path, md_path],
            command="calendar_runbook.py",
            metadata={
                "profile_name": args.profile_name,
                "source_schedule_run_id": runbook.get("source_schedule_run_id"),
                "source_execution_run_id": runbook.get("source_execution_run_id"),
            },
        ),
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {manifest_path}")
    print(
        "Runbook: "
        f"{runbook['audit']['scheduled_days']} days, "
        f"{runbook['audit']['scheduled_executable_units']} executable units, "
        f"validation={'passed' if runbook['audit']['execution_validation_passed'] else 'needs attention'}"
    )
    return 0 if runbook["audit"]["execution_validation_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
