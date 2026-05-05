#!/usr/bin/env python3
"""Stamp the current primary calendar/runbook choice from generated variants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, utc_now_id, write_manifest  # noqa: E402


DEFAULT_RUNBOOKS = [
    YEAR_DIR / "outputs" / "runbooks" / "full-clear-sensitivity-runbook.json",
    YEAR_DIR / "outputs" / "runbooks" / "strava-p90-inferred-runbook.json",
    YEAR_DIR / "outputs" / "runbooks" / "default-120s-runbook.json",
]
DEFAULT_OUTPUT_DIR = YEAR_DIR / "outputs" / "selected-plan"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def runbook_summary(path: Path) -> dict[str, Any]:
    runbook = read_json(path)
    summary = runbook.get("summary") or {}
    audit = runbook.get("audit") or {}
    load = runbook.get("load_analysis") or {}
    scheduled_days = runbook.get("days") or []
    return {
        "profile_name": runbook.get("profile_name") or path.stem.replace("-runbook", ""),
        "runbook_path": str(path),
        "run_id": runbook.get("run_id"),
        "scheduled_days": audit.get("scheduled_days"),
        "scheduled_executable_units": audit.get("scheduled_executable_units"),
        "scheduled_official_miles": float(summary.get("scheduled_official_miles") or 0),
        "scheduled_total_on_foot_miles": float(summary.get("scheduled_total_on_foot_miles") or 0),
        "scheduled_ascent_ft": int(summary.get("scheduled_ascent_ft") or 0),
        "scheduled_segments": int(summary.get("scheduled_segments") or 0),
        "last_scheduled_date": scheduled_days[-1].get("date") if scheduled_days else None,
        "open_or_recovery_days": int(summary.get("open_or_recovery_days") or 0),
        "weekday_exception_days": int(audit.get("weekday_exception_days") or 0),
        "consecutive_day_violations": int(audit.get("consecutive_day_violations") or 0),
        "weekly_mileage_violations": int(audit.get("weekly_mileage_violations") or 0),
        "long_day_count_violations": int(audit.get("long_day_count_violations") or 0),
        "rest_days_before_latest_deficit": int(audit.get("rest_days_before_latest_deficit") or 0),
        "max_consecutive_scheduled_days": int(load.get("max_consecutive_scheduled_days") or 0),
        "execution_validation_passed": bool(audit.get("execution_validation_passed")),
    }


def selection_score(summary: dict[str, Any]) -> tuple[int, int, float, int, float]:
    return (
        1 if summary["execution_validation_passed"] else 0,
        int(summary["scheduled_segments"]),
        float(summary["scheduled_official_miles"]),
        -int(summary["scheduled_days"] or 999),
        -float(summary["scheduled_total_on_foot_miles"] or 999999),
    )


def build_selected_plan(
    runbook_paths: list[Path],
    selected_profile: str | None = None,
) -> dict[str, Any]:
    summaries = [runbook_summary(path) for path in runbook_paths if path.exists()]
    if not summaries:
        raise ValueError("No readable runbook paths supplied")
    if selected_profile:
        matching = [summary for summary in summaries if summary["profile_name"] == selected_profile]
        if not matching:
            raise ValueError(f"Requested profile not found: {selected_profile}")
        selected = matching[0]
    else:
        selected = max(summaries, key=selection_score)
    alternatives = [
        summary
        for summary in sorted(summaries, key=selection_score, reverse=True)
        if summary["profile_name"] != selected["profile_name"]
    ]
    return {
        "run_id": utc_now_id(),
        "selection_status": "selected_draft",
        "selected_profile": selected["profile_name"],
        "selected_runbook_path": selected["runbook_path"],
        "selected_runbook_run_id": selected.get("run_id"),
        "coverage_basis": {
            "scheduled_segments": selected["scheduled_segments"],
            "scheduled_official_miles": selected["scheduled_official_miles"],
            "scheduled_total_on_foot_miles": selected["scheduled_total_on_foot_miles"],
            "scheduled_ascent_ft": selected["scheduled_ascent_ft"],
            "scheduled_days": selected["scheduled_days"],
            "scheduled_executable_units": selected["scheduled_executable_units"],
            "last_scheduled_date": selected["last_scheduled_date"],
            "open_or_recovery_days": selected["open_or_recovery_days"],
            "weekday_exception_days": selected["weekday_exception_days"],
            "consecutive_day_violations": selected["consecutive_day_violations"],
            "weekly_mileage_violations": selected["weekly_mileage_violations"],
            "long_day_count_violations": selected["long_day_count_violations"],
            "rest_days_before_latest_deficit": selected["rest_days_before_latest_deficit"],
            "max_consecutive_scheduled_days": selected["max_consecutive_scheduled_days"],
            "execution_validation_passed": selected["execution_validation_passed"],
        },
        "alternatives": alternatives,
        "road_policy": {
            "public_roads_allowed": True,
            "blocked_road_types": ["private", "access_no", "foot_no", "non_real_unmapped"],
        },
        "remaining_hardening_notes": [
            "Day-of Ridge to Rivers signage and trail condition checks are still required.",
            "This selection ignores heat and weather by request.",
            "Personal availability and target completion can still override the selected draft.",
        ],
    }


def render_markdown(selected: dict[str, Any]) -> str:
    basis = selected["coverage_basis"]
    lines = [
        "# 2026 Selected Plan Draft",
        "",
        f"- Selected profile: {selected['selected_profile']}",
        f"- Runbook: {selected['selected_runbook_path']}",
        f"- Scheduled official miles: {basis['scheduled_official_miles']}",
        f"- Scheduled total on-foot miles: {basis['scheduled_total_on_foot_miles']}",
        f"- Scheduled ascent: {basis.get('scheduled_ascent_ft')} ft",
        f"- Scheduled segments: {basis['scheduled_segments']}",
        f"- Scheduled days: {basis['scheduled_days']}",
        f"- Last scheduled day: {basis.get('last_scheduled_date')}",
        f"- Buffer/open days in schedule window: {basis.get('open_or_recovery_days')}",
        f"- Weekday exception days: {basis.get('weekday_exception_days')}",
        f"- Load violations: consecutive={basis.get('consecutive_day_violations')}, "
        f"weekly={basis.get('weekly_mileage_violations')}, "
        f"long-day={basis.get('long_day_count_violations')}, "
        f"rest-deficit={basis.get('rest_days_before_latest_deficit')}",
        f"- Execution validation: {'passed' if basis['execution_validation_passed'] else 'needs attention'}",
        "",
        "## Alternatives",
        "",
        "| Profile | Segments | Official mi | On-foot mi | Ascent ft | Days | Last scheduled | Buffer days | Weekday exceptions | Load violations | Validation |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for alternative in selected["alternatives"]:
        lines.append(
            f"| {alternative['profile_name']} "
            f"| {alternative['scheduled_segments']} "
            f"| {alternative['scheduled_official_miles']} "
            f"| {alternative['scheduled_total_on_foot_miles']} "
            f"| {alternative['scheduled_ascent_ft']} "
            f"| {alternative['scheduled_days']} "
            f"| {alternative.get('last_scheduled_date') or ''} "
            f"| {alternative.get('open_or_recovery_days') or 0} "
            f"| {alternative.get('weekday_exception_days') or 0} "
            f"| "
            f"{(alternative.get('consecutive_day_violations') or 0) + (alternative.get('weekly_mileage_violations') or 0) + (alternative.get('long_day_count_violations') or 0) + (alternative.get('rest_days_before_latest_deficit') or 0)} "
            f"| {'passed' if alternative['execution_validation_passed'] else 'needs attention'} |"
        )
    lines.extend(["", "## Notes", ""])
    for note in selected["remaining_hardening_notes"]:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runbook", type=Path, action="append", dest="runbooks")
    parser.add_argument("--selected-profile")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runbooks = args.runbooks or DEFAULT_RUNBOOKS
    selected = build_selected_plan(runbooks, selected_profile=args.selected_profile)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "selected-plan.json"
    md_path = args.output_dir / "selected-plan.md"
    json_path.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(selected), encoding="utf-8")
    manifest_path = args.output_dir / "selected-plan-artifact-manifest.json"
    write_manifest(
        manifest_path,
        build_artifact_manifest(
            run_id=selected["run_id"],
            inputs=runbooks,
            outputs=[json_path, md_path],
            command="select_calendar_plan.py",
            metadata={"selected_profile": selected["selected_profile"]},
        ),
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
