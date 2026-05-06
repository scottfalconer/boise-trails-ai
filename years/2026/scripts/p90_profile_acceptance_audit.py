#!/usr/bin/env python3
"""Compare a draft completion plan against the active personal p90 profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent

DEFAULT_STATE_JSON = YEAR_DIR / "inputs" / "personal" / "2026-planner-state.private.json"
DEFAULT_DRAFT_JSON = YEAR_DIR / "checkpoints" / "p90-relaxed-drive-draft-field-day-plan-2026-05-06.json"
DEFAULT_CALENDAR_JSON = YEAR_DIR / "checkpoints" / "p90-relaxed-drive-calendar-assignment-2026-05-06.json"
DEFAULT_GPX_READINESS_JSON = YEAR_DIR / "checkpoints" / "p90-relaxed-drive-gpx-readiness-audit-2026-05-06.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "checkpoints"
DEFAULT_BASENAME = "p90-profile-acceptance-audit-2026-05-06"


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


def availability_model(state: dict[str, Any]) -> dict[str, Any]:
    return state.get("availability_model") or {}


def current_bounds(state: dict[str, Any]) -> dict[str, Any]:
    availability = availability_model(state)
    return {
        "profile": availability.get("profile"),
        "source": availability.get("source"),
        "weekday_max_minutes": int(availability.get("weekday_max_minutes") or 0),
        "weekend_max_minutes": int(availability.get("weekend_max_minutes") or 0),
        "acceptable_inter_trailhead_drive_minutes": int(
            availability.get("acceptable_inter_trailhead_drive_minutes") or 0
        ),
        "target_completion_level": availability.get("target_completion_level"),
        "primary_limiting_factor": availability.get("primary_limiting_factor"),
    }


def candidate_bounds(draft: dict[str, Any]) -> dict[str, int]:
    config = draft.get("config") or {}
    return {
        "weekday_bound_minutes": int(config.get("weekday_bound_minutes") or 0),
        "weekend_bound_minutes": int(config.get("weekend_bound_minutes") or 0),
        "inter_trailhead_drive_minutes": int(config.get("inter_trailhead_drive_minutes") or 0),
    }


def profile_matches(current: dict[str, Any], candidate: dict[str, Any]) -> bool:
    return (
        int(current.get("weekday_max_minutes") or 0) == int(candidate.get("weekday_bound_minutes") or 0)
        and int(current.get("weekend_max_minutes") or 0) == int(candidate.get("weekend_bound_minutes") or 0)
        and int(current.get("acceptable_inter_trailhead_drive_minutes") or 0)
        == int(candidate.get("inter_trailhead_drive_minutes") or 0)
    )


def bound_for_day(day: dict[str, Any], *, weekday_bound: int, weekend_bound: int) -> int:
    return weekend_bound if day.get("day_type") == "weekend" else weekday_bound


def bound_violations(
    days: list[dict[str, Any]],
    *,
    weekday_bound: int,
    weekend_bound: int,
) -> list[dict[str, Any]]:
    rows = []
    for day in days:
        bound = bound_for_day(day, weekday_bound=weekday_bound, weekend_bound=weekend_bound)
        p90 = int(day.get("p90_minutes") or 0)
        if p90 <= bound:
            continue
        rows.append(
            {
                "draft_day_number": day.get("draft_day_number"),
                "day_type": day.get("day_type"),
                "p90_minutes": p90,
                "p90_bound_minutes": bound,
                "minutes_over_bound": p90 - bound,
                "field_day_id": day.get("field_day_id"),
            }
        )
    return rows


def inter_trailhead_drive_violations(days: list[dict[str, Any]], *, max_drive_minutes: int) -> list[dict[str, Any]]:
    rows = []
    for day in days:
        drive = int(day.get("between_drive_minutes") or 0)
        if drive <= max_drive_minutes:
            continue
        rows.append(
            {
                "draft_day_number": day.get("draft_day_number"),
                "day_type": day.get("day_type"),
                "between_drive_minutes": drive,
                "max_between_drive_minutes": max_drive_minutes,
                "minutes_over_bound": drive - max_drive_minutes,
                "field_day_id": day.get("field_day_id"),
            }
        )
    return rows


def build_report(
    *,
    state: dict[str, Any],
    draft: dict[str, Any],
    calendar: dict[str, Any],
    gpx_readiness: dict[str, Any],
) -> dict[str, Any]:
    personal = current_bounds(state)
    candidate = candidate_bounds(draft)
    days = list(draft.get("field_days") or [])
    p90_violations = bound_violations(
        days,
        weekday_bound=personal["weekday_max_minutes"],
        weekend_bound=personal["weekend_max_minutes"],
    )
    drive_violations = inter_trailhead_drive_violations(
        days,
        max_drive_minutes=personal["acceptable_inter_trailhead_drive_minutes"],
    )
    coverage = draft.get("coverage") or {}
    calendar_audit = calendar.get("audit") or {}
    gpx_summary = gpx_readiness.get("summary") or {}
    profile_match = profile_matches(personal, candidate)
    coverage_passed = (
        int(coverage.get("covered_segment_count") or 0) == int(coverage.get("official_segment_count") or 0)
        and int(coverage.get("missing_segment_count") or 0) == 0
    )
    schedule_shape_passed = calendar_audit.get("passed") is True
    gpx_passed = gpx_summary.get("day_level_gpx_ready") is True
    accepted = profile_match and coverage_passed and schedule_shape_passed and gpx_passed and not p90_violations and not drive_violations
    return {
        "objective": "compare the relaxed-drive completion draft against the active personal p90 profile",
        "source_files": {
            "personal_state_json": display_path(DEFAULT_STATE_JSON),
            "draft_field_day_plan_json": display_path(DEFAULT_DRAFT_JSON),
            "calendar_assignment_json": display_path(DEFAULT_CALENDAR_JSON),
            "gpx_readiness_json": display_path(DEFAULT_GPX_READINESS_JSON),
        },
        "current_personal_bounds": personal,
        "candidate_plan_bounds": candidate,
        "summary": {
            "profile_matches_current_personal_bounds": profile_match,
            "accepted_as_active_personal_plan": accepted,
            "coverage_passed": coverage_passed,
            "calendar_assignment_passed": schedule_shape_passed,
            "day_level_gpx_ready": gpx_passed,
            "field_day_count": len(days),
            "total_p75_minutes": (draft.get("time_and_logistics") or {}).get("total_p75_minutes"),
            "current_bound_p90_violation_count": len(p90_violations),
            "inter_trailhead_drive_violation_count": len(drive_violations),
            "max_minutes_over_current_p90_bound": max((row["minutes_over_bound"] for row in p90_violations), default=0),
            "max_minutes_over_current_inter_trailhead_drive_bound": max(
                (row["minutes_over_bound"] for row in drive_violations),
                default=0,
            ),
        },
        "current_bound_p90_violations": p90_violations,
        "inter_trailhead_drive_violations": drive_violations,
        "interpretation": [
            "The relaxed-drive draft has useful completion evidence only if the user explicitly accepts its availability profile.",
            "Until then, the active personal-bound proof remains incomplete even if coverage, date assignment, and GPX validation pass for the draft.",
        ],
    }


def render_md(report: dict[str, Any]) -> str:
    summary = report["summary"]
    current = report["current_personal_bounds"]
    candidate = report["candidate_plan_bounds"]
    lines = [
        "# P90 Profile Acceptance Audit",
        "",
        f"Objective: {report['objective']}",
        "",
        "## Verdict",
        "",
        f"- Accepted as active personal plan: {summary['accepted_as_active_personal_plan']}",
        f"- Profile matches current personal bounds: {summary['profile_matches_current_personal_bounds']}",
        f"- Coverage passed: {summary['coverage_passed']}",
        f"- Calendar assignment passed: {summary['calendar_assignment_passed']}",
        f"- Day-level GPX ready: {summary['day_level_gpx_ready']}",
        "",
        "## Bounds",
        "",
        f"- Current personal weekday/weekend p90: {current['weekday_max_minutes']} / {current['weekend_max_minutes']} min",
        f"- Current inter-trailhead drive limit: {current['acceptable_inter_trailhead_drive_minutes']} min",
        f"- Candidate weekday/weekend p90: {candidate['weekday_bound_minutes']} / {candidate['weekend_bound_minutes']} min",
        f"- Candidate inter-trailhead drive limit: {candidate['inter_trailhead_drive_minutes']} min",
        "",
        "## Violations Against Current Personal Bounds",
        "",
        f"- P90 day violations: {summary['current_bound_p90_violation_count']}",
        f"- Max minutes over current p90 bound: {summary['max_minutes_over_current_p90_bound']}",
        f"- Inter-trailhead drive violations: {summary['inter_trailhead_drive_violation_count']}",
        f"- Max minutes over current inter-trailhead drive bound: {summary['max_minutes_over_current_inter_trailhead_drive_bound']}",
        "",
        "## Interpretation",
        "",
    ]
    lines.extend(f"- {item}" for item in report["interpretation"])
    lines.extend(
        [
            "",
            "## P90 Violating Days",
            "",
            "| Draft day | Type | P90 | Current bound | Over |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in report["current_bound_p90_violations"]:
        lines.append(
            f"| {row['draft_day_number']} | {row['day_type']} | {row['p90_minutes']} | "
            f"{row['p90_bound_minutes']} | {row['minutes_over_bound']} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-json", type=Path, default=DEFAULT_STATE_JSON)
    parser.add_argument("--draft-json", type=Path, default=DEFAULT_DRAFT_JSON)
    parser.add_argument("--calendar-json", type=Path, default=DEFAULT_CALENDAR_JSON)
    parser.add_argument("--gpx-readiness-json", type=Path, default=DEFAULT_GPX_READINESS_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        state=read_json(args.state_json),
        draft=read_json(args.draft_json),
        calendar=read_json(args.calendar_json),
        gpx_readiness=read_json(args.gpx_readiness_json),
    )
    json_path = args.output_dir / f"{args.basename}.json"
    md_path = args.output_dir / f"{args.basename}.md"
    write_json(json_path, report)
    md_path.write_text(render_md(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
