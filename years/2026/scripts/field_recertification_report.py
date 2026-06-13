#!/usr/bin/env python3
"""Re-check selected-profile feasibility after exported phone progress."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import field_progress_report  # noqa: E402
import p90_joint_field_day_optimizer as optimizer_module  # noqa: E402
from p90_completion_gap_analyzer import official_segments  # noqa: E402
from personal_route_planner import read_json  # noqa: E402


DEFAULT_FIELD_TOOL_DATA_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_OFFICIAL_GEOJSON = YEAR_DIR / "inputs" / "official" / "api-pull-2026-06-13" / "official_foot_segments.geojson"
DEFAULT_CALENDAR_JSON = YEAR_DIR / "checkpoints" / "post-h1-cleanup-calendar-assignment-2026-05-13.json"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "outputs" / "private" / "progress" / "field-recertification-latest.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "outputs" / "private" / "progress" / "field-recertification-latest.md"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def selected_profile_config(field_tool_data: dict[str, Any]) -> dict[str, Any]:
    baseline = field_tool_data.get("certified_baseline") or {}
    return {
        "profile_id": baseline.get("profile_id"),
        "weekday_bound_minutes": int(baseline.get("weekday_p90_minutes") or 292),
        "weekend_bound_minutes": int(baseline.get("weekend_p90_minutes") or 360),
        "max_combo_size": 4,
        "neighbor_limit": 40,
        "connected_expansion_limit": 40,
        "inter_trailhead_drive_minutes": 45,
    }


def optimize_remaining_segments(remaining_segment_ids: list[int], config: dict[str, Any]) -> dict[str, Any]:
    if not remaining_segment_ids:
        return {
            "success": True,
            "reason": "all_segments_already_completed",
            "target_segment_ids": [],
            "covered_segment_count": 0,
            "missing_segment_ids": [],
            "field_day_count": 0,
        }
    state = read_json(optimizer_module.repaired.DEFAULT_STATE_JSON)
    state.setdefault("availability_model", {})["acceptable_inter_trailhead_drive_minutes"] = int(
        config.get("inter_trailhead_drive_minutes") or 45
    )
    all_target_ids, candidates, field_menu = optimizer_module.all_repaired_candidates()
    official_index = official_segments(read_json(optimizer_module.DEFAULT_OFFICIAL_GEOJSON))
    official_miles_by_segment = {
        seg_id: float(official_index[seg_id]["official_miles"])
        for seg_id in remaining_segment_ids
        if seg_id in official_index
    }
    max_bound = max(int(config["weekday_bound_minutes"]), int(config["weekend_bound_minutes"]))
    bounded = [
        candidate
        for candidate in candidates
        if int(candidate["door_to_door_p90_minutes"]) <= max_bound
        and candidate.get("validation_passed") is True
        and candidate.get("manual_design_hold") is not True
    ]
    loops, missing_coordinates = optimizer_module.candidate_loop_records(bounded, field_menu, state)
    availability = state.get("availability_model") or {}
    counts = optimizer_module.pack_audit.date_counts(
        (availability.get("available_dates") or {})["start"],
        (availability.get("available_dates") or {})["end"],
    )
    field_days = optimizer_module.generate_direct_field_day_candidates(
        loops,
        state,
        weekday_bound=int(config["weekday_bound_minutes"]),
        weekend_bound=int(config["weekend_bound_minutes"]),
        max_combo_size=int(config["max_combo_size"]),
        neighbor_limit=int(config["neighbor_limit"]),
        connected_expansion_limit=int(config["connected_expansion_limit"]),
    )
    solution = optimizer_module.solve_direct_field_day_cover(field_days, remaining_segment_ids, counts)
    max_coverage = optimizer_module.solve_direct_field_day_max_coverage(
        field_days,
        remaining_segment_ids,
        counts,
        official_miles_by_segment=official_miles_by_segment,
    )
    return {
        "success": solution.get("success") is True,
        "target_segment_ids": remaining_segment_ids,
        "candidate_loop_count": len(loops),
        "missing_coordinate_count": len(missing_coordinates),
        "field_day_candidate_count": len(field_days),
        "date_counts": counts,
        "solution": solution,
        "max_coverage_solution": max_coverage,
        "covered_segment_count": solution.get("covered_segment_count") or max_coverage.get("covered_segment_count"),
        "missing_segment_ids": solution.get("missing_segment_ids") or max_coverage.get("missing_segment_ids") or [],
        "field_day_count": solution.get("field_day_count"),
        "total_p75_minutes": solution.get("total_p75_minutes"),
        "caveats": [
            "This reruns the generated field-day set-cover optimizer for remaining segment ids.",
            "It does not assign exact future dates or perform day-of condition checks.",
        ],
    }


def fast_remaining_certificate_check(
    *,
    remaining_segment_ids: list[int],
    config: dict[str, Any],
    baseline: dict[str, Any],
    progress_report: dict[str, Any],
) -> dict[str, Any]:
    missing_ids = [int(seg_id) for seg_id in progress_report["missing_remaining_segment_ids"]]
    success = baseline.get("status") == "passed" and progress_report["summary"]["remaining_coverage_preserved"] is True
    return {
        "success": success,
        "method": "certified_baseline_plus_remaining_menu_coverage",
        "target_segment_ids": remaining_segment_ids,
        "covered_segment_count": len(remaining_segment_ids) - len(missing_ids),
        "missing_segment_ids": missing_ids,
        "field_day_count": baseline.get("field_day_count"),
        "total_p75_minutes": baseline.get("total_p75_minutes"),
        "profile_id": config.get("profile_id"),
        "reason": "baseline_passed_and_remaining_menu_covers_all_remaining_segments"
        if success
        else "baseline_or_remaining_menu_coverage_failed",
        "caveats": [
            "Fast recertification does not recompute the MILP calendar; it verifies that the certified baseline is still present and the remaining field menu covers all remaining segment ids.",
            "Use --run-heavy-optimizer for a deeper generated-candidate set-cover rerun.",
        ],
    }


def progress_requires_heavy_recertification(progress: dict[str, Any]) -> bool:
    nontrivial_keys = (
        "completed_outing_ids",
        "completed_segment_ids",
        "missed_segment_ids",
        "blocked_segment_ids",
        "blocked_trail_names",
    )
    return any(progress.get(key) for key in nontrivial_keys)


def progress_has_validated_segment_state(progress: dict[str, Any]) -> bool:
    segment_state_keys = (
        "completed_segment_ids",
        "extra_completed_segment_ids",
        "missed_segment_ids",
        "blocked_segment_ids",
    )
    return any(progress.get(key) for key in segment_state_keys)


def active_field_menu_matches_progress_export(
    field_tool_data: dict[str, Any],
    progress_report: dict[str, Any],
    progress: dict[str, Any],
) -> bool:
    if not progress_has_validated_segment_state(progress):
        return False
    exported = field_tool_data.get("progress") or {}
    exported_completed = set(field_progress_report.normalized_ids(exported.get("completed_segment_ids_at_export")))
    exported_blocked = set(field_progress_report.normalized_ids(exported.get("blocked_segment_ids_at_export")))
    report_completed = set(field_progress_report.normalized_ids(progress_report["completed_segment_ids"]))
    report_blocked = set(field_progress_report.normalized_ids(progress_report["blocked_segment_ids"]))
    return exported_completed == report_completed and exported_blocked == report_blocked


def active_field_menu_certificate_check(
    *,
    remaining_segment_ids: list[int],
    config: dict[str, Any],
    baseline: dict[str, Any],
    progress_report: dict[str, Any],
    field_tool_data: dict[str, Any],
) -> dict[str, Any]:
    missing_ids = [int(seg_id) for seg_id in progress_report["missing_remaining_segment_ids"]]
    routes = [
        route
        for route in field_tool_data.get("routes") or []
        if (route.get("validation") or {}).get("passed") is True
    ]
    success = baseline.get("status") == "passed" and progress_report["summary"]["remaining_coverage_preserved"] is True
    return {
        "success": success,
        "method": "active_progress_field_menu_certificate",
        "target_segment_ids": remaining_segment_ids,
        "covered_segment_count": len(remaining_segment_ids) - len(missing_ids),
        "missing_segment_ids": missing_ids,
        "field_day_count": len(routes),
        "total_p75_minutes": int(sum(int(route.get("door_to_door_minutes_p75") or 0) for route in routes)),
        "profile_id": config.get("profile_id"),
        "reason": "active_field_packet_export_matches_validated_segment_progress"
        if success
        else "baseline_or_remaining_menu_coverage_failed",
        "caveats": [
            "This recertifies the current active field-packet menu after validated segment progress was applied.",
            "It does not rerun the global generated-candidate MILP calendar.",
            "Phone completed_outing_ids alone remain provisional and cannot trigger this certificate.",
        ],
    }


def challenge_dates(calendar: dict[str, Any]) -> list[str]:
    window = calendar.get("challenge_window") or {}
    start = date.fromisoformat(window.get("start") or "2026-06-18")
    end = date.fromisoformat(window.get("end") or "2026-07-18")
    values = []
    current = start
    while current <= end:
        values.append(current.isoformat())
        current += timedelta(days=1)
    return values


def calendar_reassignment_check(
    calendar: dict[str, Any] | None,
    remaining_segment_ids: list[int],
    progress: dict[str, Any],
) -> dict[str, Any]:
    if not calendar:
        return {
            "passed": None,
            "reason": "calendar_not_loaded",
            "remaining_scheduled_day_count": None,
            "remaining_available_date_count": None,
            "remaining_assignment_dates": [],
        }
    as_of = progress.get("as_of_date")
    cutoff = date.fromisoformat(as_of) if as_of else None
    remaining = set(int(seg_id) for seg_id in remaining_segment_ids)
    remaining_assignments = []
    for assignment in calendar.get("assignments") or []:
        assignment_ids = set(
            int(seg_id)
            for seg_id in (((assignment.get("field_day") or {}).get("segment_summary") or {}).get("segment_ids") or [])
        )
        if assignment_ids & remaining:
            remaining_assignments.append(assignment)
    available_dates = []
    for value in challenge_dates(calendar):
        current = date.fromisoformat(value)
        if cutoff is None or current > cutoff:
            available_dates.append(value)
    return {
        "passed": len(remaining_assignments) <= len(available_dates),
        "reason": "remaining_assignments_fit_available_dates"
        if len(remaining_assignments) <= len(available_dates)
        else "not_enough_future_dates_for_remaining_assignments",
        "as_of_date": as_of,
        "remaining_scheduled_day_count": len(remaining_assignments),
        "remaining_available_date_count": len(available_dates),
        "remaining_assignment_dates": [assignment.get("date") for assignment in remaining_assignments],
    }


def recertification_gates(
    *,
    baseline: dict[str, Any],
    progress_report: dict[str, Any],
    optimizer: dict[str, Any],
    calendar_check: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "gate": "certified_baseline_loaded",
            "passed": baseline.get("status") == "passed",
            "detail": f"baseline status {baseline.get('status')}",
        },
        {
            "gate": "remaining_menu_coverage",
            "passed": progress_report["summary"]["remaining_coverage_preserved"] is True,
            "detail": f"missing remaining segments {progress_report['summary']['missing_remaining_segment_count']}",
        },
        {
            "gate": "remaining_optimizer_solution",
            "passed": optimizer.get("success") is True,
            "detail": (
                f"field days {optimizer.get('field_day_count')}"
                if optimizer.get("success") is True
                else f"missing {optimizer.get('missing_segment_ids')}"
            ),
        },
        {
            "gate": "remaining_calendar_capacity",
            "passed": calendar_check.get("passed") is True,
            "detail": (
                f"{calendar_check.get('remaining_scheduled_day_count')} scheduled remaining day(s) / "
                f"{calendar_check.get('remaining_available_date_count')} available date(s)"
            ),
        },
    ]


def build_recertification_report(
    field_tool_data: dict[str, Any],
    official_geojson: dict[str, Any],
    progress: dict[str, Any] | None = None,
    *,
    calendar: dict[str, Any] | None = None,
    optimizer: Callable[[list[int], dict[str, Any]], dict[str, Any]] | None = None,
    skip_heavy_optimizer: bool = False,
    run_heavy_optimizer: bool = False,
) -> dict[str, Any]:
    effective_progress = (
        field_progress_report.progress_from_field_tool_export(field_tool_data)
        if progress is None
        else progress
    )
    progress_report = field_progress_report.build_progress_report(field_tool_data, official_geojson, progress)
    baseline = field_tool_data.get("certified_baseline") or {}
    config = selected_profile_config(field_tool_data)
    remaining_segment_ids = [int(seg_id) for seg_id in progress_report["remaining_segment_ids"]]
    if skip_heavy_optimizer:
        optimizer_result = {
            "success": None,
            "reason": "heavy_optimizer_skipped",
            "target_segment_ids": remaining_segment_ids,
            "missing_segment_ids": [],
        }
    elif progress_report["summary"]["remaining_coverage_preserved"] is not True:
        optimizer_result = {
            "success": False,
            "reason": "remaining_menu_coverage_failed",
            "target_segment_ids": remaining_segment_ids,
            "missing_segment_ids": [int(seg_id) for seg_id in progress_report["missing_remaining_segment_ids"]],
        }
    elif optimizer is not None:
        optimizer_result = optimizer(remaining_segment_ids, config)
    elif run_heavy_optimizer:
        optimizer_result = optimize_remaining_segments(remaining_segment_ids, config)
    elif active_field_menu_matches_progress_export(field_tool_data, progress_report, effective_progress):
        optimizer_result = active_field_menu_certificate_check(
            remaining_segment_ids=remaining_segment_ids,
            config=config,
            baseline=baseline,
            progress_report=progress_report,
            field_tool_data=field_tool_data,
        )
    elif progress_requires_heavy_recertification(effective_progress):
        optimizer_result = {
            "success": False,
            "reason": "heavy_optimizer_required_after_progress_change",
            "target_segment_ids": remaining_segment_ids,
            "missing_segment_ids": [],
            "caveats": [
                "Fast baseline+menu coverage is not enough after completed, missed, or blocked progress changes.",
                "Run with --run-heavy-optimizer or provide an optimizer result before treating the remaining plan as recertified.",
            ],
        }
    else:
        optimizer_result = fast_remaining_certificate_check(
            remaining_segment_ids=remaining_segment_ids,
            config=config,
            baseline=baseline,
            progress_report=progress_report,
        )
    calendar_check = calendar_reassignment_check(calendar, remaining_segment_ids, effective_progress)
    gates = recertification_gates(
        baseline=baseline,
        progress_report=progress_report,
        optimizer=optimizer_result,
        calendar_check=calendar_check,
    )
    status = "passed" if all(gate["passed"] is True for gate in gates) else "failed"
    if skip_heavy_optimizer:
        status = "not_run"
    return {
        "schema": "boise_trails_field_recertification_report_v1",
        "objective": "rerun selected-profile feasibility after exported phone progress",
        "status": status,
        "selected_profile": config,
        "summary": {
            **progress_report["summary"],
            "remaining_full_completion_feasible": status == "passed",
        },
        "gates": gates,
        "progress": {
            "completed_outing_ids": progress_report["completed_outing_ids"],
            "completed_segment_ids": progress_report["completed_segment_ids"],
            "missed_segment_ids": progress_report["missed_segment_ids"],
            "remaining_segment_ids": progress_report["remaining_segment_ids"],
            "missing_remaining_segment_ids": progress_report["missing_remaining_segment_ids"],
        },
        "optimizer": optimizer_result,
        "calendar_reassignment": calendar_check,
        "private_state_patch": progress_report["private_state_patch"],
        "caveats": [
            "This report reruns feasibility over generated route candidates; it is not a proof of global optimality over every physical route.",
            "Exact date assignment and day-of signage/condition checks remain separate operational gates.",
        ],
    }


def render_md(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Field Recertification Report",
        "",
        f"- Status: `{report['status']}`",
        f"- Profile: `{report['selected_profile'].get('profile_id')}`",
        f"- Completed segments: {summary['completed_segment_count']}",
        f"- Remaining segments: {summary['remaining_segment_count']}",
        f"- Remaining menu coverage preserved: {summary['remaining_coverage_preserved']}",
        f"- Remaining full completion feasible: {summary['remaining_full_completion_feasible']}",
        f"- Remaining scheduled days: {report.get('calendar_reassignment', {}).get('remaining_scheduled_day_count')}",
        f"- Future available dates: {report.get('calendar_reassignment', {}).get('remaining_available_date_count')}",
        "",
        "## Gates",
        "",
        "| Gate | Passed | Detail |",
        "|---|---:|---|",
    ]
    for gate in report["gates"]:
        lines.append(f"| {gate['gate']} | {gate['passed']} | {gate['detail']} |")
    optimizer = report.get("optimizer") or {}
    lines.extend(
        [
            "",
            "## Optimizer",
            "",
            f"- Success: {optimizer.get('success')}",
            f"- Field days: {optimizer.get('field_day_count')}",
            f"- Missing segment ids: {', '.join(str(seg_id) for seg_id in optimizer.get('missing_segment_ids') or [])}",
            "",
            "## Caveats",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report.get("caveats") or [])
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--calendar-json", type=Path, default=DEFAULT_CALENDAR_JSON)
    parser.add_argument("--progress-json", type=Path)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--run-heavy-optimizer", action="store_true")
    parser.add_argument("--skip-heavy-optimizer", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    progress = read_json(args.progress_json) if args.progress_json else None
    report = build_recertification_report(
        read_json(args.field_tool_data_json),
        read_json(args.official_geojson),
        progress,
        calendar=read_json(args.calendar_json) if args.calendar_json and args.calendar_json.exists() else None,
        skip_heavy_optimizer=args.skip_heavy_optimizer,
        run_heavy_optimizer=args.run_heavy_optimizer,
    )
    write_json(args.output_json, report)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_md(report), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(json.dumps({"status": report["status"], "summary": report["summary"]}, indent=2))
    return 0 if report["status"] in {"passed", "not_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
